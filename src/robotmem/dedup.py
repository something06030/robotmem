"""多层去重管道 — exact → jaccard → cosine(可选)

组合 db_cog 原语实现三层去重，供 learn MCP 工具使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- 阈值常量 ---

JACCARD_DUP_THRESHOLD = 0.70
JACCARD_SIMILAR_THRESHOLD = 0.4
COSINE_DUP_THRESHOLD = 0.85


@dataclass
class DedupResult:
    """去重检查结果"""
    is_dup: bool
    similar_facts: list[dict] = field(default_factory=list)
    method: str = "none"        # "exact" | "jaccard" | "cosine" | "session_cosine" | "none"
    similarity: float = 0.0


# 中英文停用词 — 过滤虚词提升中文近重复检测率
_STOPWORDS = frozenset(
    "的 了 在 是 把 被 给 和 与 从 到 也 都 就 对 又 所 而 且 但 或 "
    "a an the is are was were be to of and in for on with "
    "preference constraint decision observation code config pattern "
    "architecture root_cause tradeoff revert".split()
)


def jaccard_similarity(a: str, b: str) -> float:
    """token 级 Jaccard 相似度 — 过滤停用词后 set intersection / union"""
    tokens_a = set(a.lower().split()) - _STOPWORDS
    tokens_b = set(b.lower().split()) - _STOPWORDS
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def check_duplicate(
    assertion: str,
    collection: str,
    session_id: str | None,
    db_cog,
    embedder=None,
    exclude_id: int | None = None,
) -> DedupResult:
    """三层去重检查

    Layer 1: 精确匹配（memory_exists）
    Layer 2: Jaccard token 相似度（fts_search_memories 候选）
    Layer 3: 向量余弦相似度（可选，需 embedder）

    Args:
        assertion: 待检查的记忆文本
        collection: 项目集合名
        session_id: 会话 ID
        db_cog: CogDatabase 实例
        embedder: Embedder 实例（None 则跳过向量层）
        exclude_id: 排除的记忆 ID（update 时排除自身）

    Returns:
        DedupResult 包含 is_dup / similar_facts / method / similarity
    """
    # --- Layer 1: 精确匹配 ---
    try:
        if db_cog.memory_exists(
            assertion=assertion,
            session_id=session_id,
            collection=collection,
        ):
            return DedupResult(is_dup=True, method="exact", similarity=1.0)
    except Exception as e:
        logger.debug("dedup Layer 1 失败: %s", e)

    # --- Layer 2: Jaccard token 相似度 ---
    similar_facts: list[dict] = []
    best_jaccard = 0.0
    try:
        candidates = db_cog.fts_search_memories(
            query=assertion,
            collection=collection,
            limit=5,
        )
        for cand in candidates:
            if exclude_id is not None and cand.get("id") == exclude_id:
                continue
            cand_assertion = cand.get("assertion", "")
            sim = jaccard_similarity(assertion, cand_assertion)
            if sim > best_jaccard:
                best_jaccard = sim
            if sim >= JACCARD_DUP_THRESHOLD:
                return DedupResult(
                    is_dup=True,
                    similar_facts=[{
                        "id": cand.get("id"),
                        "assertion": cand_assertion,
                        "similarity": round(sim, 3),
                    }],
                    method="jaccard",
                    similarity=round(sim, 3),
                )
            if sim >= JACCARD_SIMILAR_THRESHOLD:
                similar_facts.append({
                    "id": cand.get("id"),
                    "assertion": cand_assertion,
                    "similarity": round(sim, 3),
                })
    except Exception as e:
        logger.debug("dedup Layer 2 失败: %s", e)

    # --- Layer 3: 向量余弦（可选） ---
    if embedder is not None and getattr(embedder, "available", False):
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                logger.debug("dedup Layer 3 跳过: 事件循环已运行，无法同步调用 embed_one")
            else:
                embedding = asyncio.run(embedder.embed_one(assertion))
                if embedding:
                    vec_results = db_cog.vec_search_memories(
                        query_embedding=embedding,
                        collection=collection,
                        limit=3,
                    )
                    for vr in vec_results:
                        if exclude_id is not None and vr.get("id") == exclude_id:
                            continue
                        distance = vr.get("distance", 1.0)
                        cosine_sim = 1.0 - distance
                        if cosine_sim >= COSINE_DUP_THRESHOLD:
                            return DedupResult(
                                is_dup=True,
                                similar_facts=[{
                                    "id": vr.get("id"),
                                    "assertion": vr.get("assertion", ""),
                                    "similarity": round(cosine_sim, 3),
                                }],
                                method="cosine",
                                similarity=round(cosine_sim, 3),
                            )
                        if cosine_sim >= JACCARD_SIMILAR_THRESHOLD:
                            vr_id = vr.get("id")
                            if not any(s.get("id") == vr_id for s in similar_facts):
                                similar_facts.append({
                                    "id": vr_id,
                                    "assertion": vr.get("assertion", ""),
                                    "similarity": round(cosine_sim, 3),
                                })
        except Exception as e:
            logger.debug("dedup Layer 3 失败: %s", e)

    return DedupResult(
        is_dup=False,
        similar_facts=similar_facts[:5],
        method="none",
        similarity=round(best_jaccard, 3),
    )


def check_session_cosine_dup(
    assertion: str,
    session_id: str | None,
    collection: str,
    db_cog,
    embedder,
) -> DedupResult:
    """Session-scoped cosine 去重

    在同 session 内检查新记忆是否与已有记忆语义重复。
    使用 embedding + vec_search + session_id 后过滤。
    阈值 COSINE_DUP_THRESHOLD (0.85)。

    Args:
        assertion: 新记忆文本
        session_id: 当前会话 ID（None → 跳过）
        collection: 项目集合名
        db_cog: CogDatabase 实例（需要 _vec_loaded=True）
        embedder: Embedder 实例

    Returns:
        DedupResult: is_dup=True 表示同 session 内有语义重复
    """
    if not session_id or embedder is None:
        return DedupResult(is_dup=False)

    if not getattr(db_cog, '_vec_loaded', False):
        return DedupResult(is_dup=False)

    try:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            logger.debug("session cosine dedup 跳过: 事件循环已运行")
            return DedupResult(is_dup=False)

        embedding = asyncio.run(embedder.embed_one(assertion))
    except Exception as e:
        logger.debug("session cosine dedup embedding 失败: %s", e)
        return DedupResult(is_dup=False)

    if not embedding:
        return DedupResult(is_dup=False)

    try:
        results = db_cog.vec_search_memories(
            query_embedding=embedding,
            collection=collection,
            limit=20,
        )
    except Exception as e:
        logger.debug("session cosine dedup vec_search 失败: %s", e)
        return DedupResult(is_dup=False)

    for r in results:
        if r.get("session_id") != session_id:
            continue
        distance = r.get("distance", 1.0)
        cosine_sim = 1.0 - distance
        if cosine_sim >= COSINE_DUP_THRESHOLD:
            return DedupResult(
                is_dup=True,
                similar_facts=[{
                    "id": r.get("id"),
                    "assertion": r.get("assertion", ""),
                    "similarity": round(cosine_sim, 3),
                }],
                method="session_cosine",
                similarity=round(cosine_sim, 3),
            )

    return DedupResult(is_dup=False)


def cleanup_exact_duplicates(
    db_cog,
    collection: str | None = None,
    dry_run: bool = True,
) -> list[dict]:
    """一次性清理完全重复的 active memories

    找出 content 完全相同的 active memories 分组，
    每组保留 confidence 最高的（平局保留最新），其余 supersede。

    三层防御:
      L1 事前: 只处理 status='active' + content 完全相同 + 至少 2 条
      L2 事中: 每批上限 200 对，超限即停
      L3 事后: 返回每次操作的 (old_id, keep_id, content_preview) 供审计

    Args:
        db_cog: CogDatabase 实例
        collection: 限定集合（None=全部集合）
        dry_run: True=只预览不修改

    Returns:
        list[dict] — 每个元素 {"old_id", "keep_id", "assertion_preview"}
    """
    conn = db_cog.conn
    MAX_OPS = 200  # L2: 单次上限

    if collection:
        groups = conn.execute("""
            SELECT content, collection, GROUP_CONCAT(id) AS ids,
                   COUNT(*) AS cnt
            FROM memories
            WHERE status = 'active' AND collection = ?
            GROUP BY content, collection
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
        """, (collection,)).fetchall()
    else:
        groups = conn.execute("""
            SELECT content, collection, GROUP_CONCAT(id) AS ids,
                   COUNT(*) AS cnt
            FROM memories
            WHERE status = 'active'
            GROUP BY content, collection
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
        """).fetchall()

    ops: list[dict] = []
    for row in groups:
        try:
            assertion = row[0]
            coll = row[1]
            id_strs = row[2].split(",")
            memory_ids = [int(x.strip()) for x in id_strs]

            if len(memory_ids) < 2:
                continue

            placeholders = ",".join("?" * len(memory_ids))
            details = conn.execute(
                f"SELECT id, confidence, created_at FROM memories "
                f"WHERE id IN ({placeholders})",
                tuple(memory_ids),
            ).fetchall()

            if not details:
                continue

            # 排序 confidence DESC, created_at DESC
            sorted_details = sorted(
                details,
                key=lambda r: (float(r[1] or 0), r[2] or ""),
                reverse=True,
            )
            keep_id = sorted_details[0][0]
            to_supersede = [r[0] for r in sorted_details[1:]]

            for old_id in to_supersede:
                if len(ops) >= MAX_OPS:
                    logger.warning(
                        "cleanup_exact_duplicates: 达到上限 %d 对，剩余跳过",
                        MAX_OPS,
                    )
                    break
                ops.append({
                    "old_id": old_id,
                    "keep_id": keep_id,
                    "assertion_preview": (assertion or "")[:80],
                    "collection": coll,
                })
        except Exception as e:
            logger.warning("cleanup_exact_duplicates: 跳过一组: %s", e)
            continue

        if len(ops) >= MAX_OPS:
            break

    if not dry_run:
        for op in ops:
            try:
                db_cog.supersede_memory(
                    old_id=op["old_id"],
                    new_id=op["keep_id"],
                    reason="exact_duplicate_cleanup",
                )
            except Exception as e:
                logger.warning(
                    "cleanup_exact_duplicates: supersede %d→%d failed: %s",
                    op["old_id"], op["keep_id"], e,
                )
        logger.info(
            "cleanup_exact_duplicates: superseded %d duplicates (dry_run=%s)",
            len(ops), dry_run,
        )

    return ops
