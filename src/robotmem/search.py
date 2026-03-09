"""认知搜索 — recall 入口

BM25 + Vec + RRF merge + confidence 过滤。
支持 session_id 过滤、source_weight、perception_type 返回。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .db_cog import CogDatabase
from .embed import Embedder
from .ops.memories import batch_touch_memories
from .ops.search import fts_search_memories, vec_search_memories

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """recall 返回值"""
    memories: list[dict] = field(default_factory=list)
    total: int = 0
    mode: str = "bm25_only"     # "bm25_only" / "hybrid" / "vec_only"
    query_ms: float = 0.0


def extract_context_fields(mem: dict) -> None:
    """从 context JSON 提取 params/spatial/robot 便捷字段（原地修改）"""
    try:
        ctx = json.loads(mem.get("context") or "{}")
        for key in ("params", "spatial", "robot", "task"):
            if key in ctx:
                mem[key] = ctx[key]
    except (json.JSONDecodeError, TypeError):
        pass  # 降级：不提取，不报错


def rrf_merge(
    *ranked_lists: list[dict],
    k: int = 60,
    id_key: str = "id",
) -> list[dict]:
    """Reciprocal Rank Fusion — N-way 合并

    score = sum(1 / (k + rank + 1)) 对每路排名列表。
    返回按 _rrf_score 降序排列的 dict 列表（浅拷贝，不污染上游）。
    """
    scores: dict[int, float] = {}
    items_map: dict[int, dict] = {}

    for ranked in ranked_lists:
        if not ranked:
            continue
        for rank, item in enumerate(ranked):
            item_id = item.get(id_key)
            if item_id is None:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in items_map:
                items_map[item_id] = item

    return [
        {**items_map[item_id], "_rrf_score": score}
        for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def _apply_source_weight(memories: list[dict]) -> list[dict]:
    """sim/real 源权重 — real 数据加权 1.5x（antirez 产品架构 Round 2）

    通过 context JSON 中的 env.sim_or_real 字段判断。
    注意：原地修改 memories 中每个 dict 的 _rrf_score 字段。
    """
    for m in memories:
        ctx = m.get("context")
        if not ctx:
            continue
        try:
            parsed = json.loads(ctx) if isinstance(ctx, str) else ctx
            if parsed.get("env", {}).get("sim_or_real") == "real":
                m["_rrf_score"] = m.get("_rrf_score", 0.0) * 1.5
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return memories


_MISSING = object()


def _resolve_dotpath(d: dict, path: str):
    """点分路径解析 — 'task.success' → d['task']['success']"""
    current = d
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _match_context_filter(mem: dict, spec: dict) -> bool:
    """结构化过滤 — 检查 mem 是否匹配所有条件

    等值: {"task.success": True}
    范围: {"params.final_distance.value": {"$lt": 0.05}}
    组合: {"task.success": True, "params.final_distance.value": {"$lt": 0.1}}

    类型不匹配时返回 False（不崩溃）。
    """
    for path, expected in spec.items():
        actual = _resolve_dotpath(mem, path)
        if actual is _MISSING:
            return False
        try:
            if isinstance(expected, dict):
                for op, val in expected.items():
                    if op == "$lt" and not (actual < val):
                        return False
                    if op == "$lte" and not (actual <= val):
                        return False
                    if op == "$gt" and not (actual > val):
                        return False
                    if op == "$gte" and not (actual >= val):
                        return False
                    if op == "$ne" and not (actual != val):
                        return False
            else:
                if actual != expected:
                    return False
        except TypeError:
            return False
    return True


def _compute_spatial_distance(mem: dict, field: str, target: list[float]) -> float:
    """欧氏距离 — mem[field] vs target（维度不匹配返回 inf）"""
    actual = _resolve_dotpath(mem, field)
    if actual is _MISSING or not isinstance(actual, (list, tuple)):
        return float("inf")
    if len(actual) != len(target):
        return float("inf")
    return sum((a - t) ** 2 for a, t in zip(actual, target)) ** 0.5


async def recall(
    query: str,
    db: CogDatabase,
    embedder: Embedder | None = None,
    collection: str = "default",
    top_k: int = 5,
    min_confidence: float = 0.3,
    session_id: str | None = None,
    context_filter: dict | None = None,
    spatial_sort: dict | None = None,
) -> RecallResult:
    """认知搜索主入口 — BM25 + Vec + RRF

    三层防御：
    - L1 事前：query 非空 + top_k 范围
    - L2 事中：try-except 搜索异常不崩溃
    - L3 事后：touch 更新访问计数 + 日志

    Args:
        session_id: episode 回放过滤（Abbeel 产品架构 Round 2）
        context_filter: 结构化过滤（#17 P1），如 {"task.success": True}
        spatial_sort: 空间近邻排序（#17 P2），如
            {"field": "spatial.object_position", "target": [1.3, 0.7, 0.42]}
    """
    t0 = time.monotonic()

    # L1: 事前校验
    if not query or not query.strip():
        return RecallResult()
    top_k = min(max(1, top_k), 100)
    session_id = session_id or None  # 空字符串统一为 None

    # 有过滤/排序时多取候选，补偿过滤损耗
    fetch_mul = 4 if (context_filter or spatial_sort) else 2
    fetch_limit = top_k * fetch_mul

    # L2: BM25 搜索（每次通过 db.conn 获取连接，避免缓存旧连接）
    try:
        bm25_results = fts_search_memories(db.conn, query, collection, limit=fetch_limit)
    except Exception as e:
        logger.warning("recall BM25 搜索异常: %s", e)
        bm25_results = []

    # L2: Vec 搜索（embedder 可用时）
    vec_results: list[dict] = []
    if embedder and embedder.available:
        try:
            embedding = await embedder.embed_one(query)
            vec_results = vec_search_memories(
                db.conn, embedding, collection, limit=fetch_limit, vec_loaded=db.vec_loaded,
            )
        except Exception as e:
            logger.warning("recall Vec 搜索异常: %s", e)
    elif embedder:
        logger.debug("recall: embedder 不可用 (%s)，降级为 BM25-only", embedder.unavailable_reason)

    if not bm25_results and not vec_results:
        return RecallResult(query_ms=(time.monotonic() - t0) * 1000)

    # RRF 融合 + mode 确定
    if bm25_results and vec_results:
        merged = rrf_merge(bm25_results, vec_results, k=60)
        mode = "hybrid"
    elif bm25_results:
        merged = [{**m, "_rrf_score": 1.0 / (60 + i + 1)} for i, m in enumerate(bm25_results)]
        mode = "bm25_only"
    else:
        merged = [{**m, "_rrf_score": 1.0 / (60 + i + 1)} for i, m in enumerate(vec_results)]
        mode = "vec_only"

    # sim/real 加权（antirez 产品架构 Round 2）+ 重新排序
    _apply_source_weight(merged)
    merged.sort(key=lambda x: x.get("_rrf_score", 0), reverse=True)

    # 基础过滤：confidence + session_id（不限 top_k，后续还有 context/spatial 过滤）
    candidates = []
    for m in merged:
        if m.get("confidence", 0) < min_confidence:
            continue
        if session_id and m.get("session_id") != session_id:
            continue
        candidates.append(m)

    # context JSON 解析（在过滤前提取，使 context_filter 可用）
    for m in candidates:
        extract_context_fields(m)

    # P1: 结构化过滤（#17）
    if context_filter:
        candidates = [m for m in candidates if _match_context_filter(m, context_filter)]

    # P2: 空间近邻排序（#17）
    if spatial_sort:
        try:
            sp_field = spatial_sort["field"]
            sp_target = spatial_sort["target"]
        except (KeyError, TypeError) as e:
            logger.warning("recall spatial_sort 参数无效: %s", e)
            spatial_sort = None
    if spatial_sort:
        sp_max = spatial_sort.get("max_distance")
        for m in candidates:
            m["_spatial_distance"] = _compute_spatial_distance(m, sp_field, sp_target)
        if sp_max is not None:
            candidates = [m for m in candidates if m["_spatial_distance"] <= sp_max]
        candidates.sort(key=lambda m: m.get("_spatial_distance", float("inf")))

    # 截取 top_k
    filtered = candidates[:top_k]

    # MaxScore 归一化（防除零）
    if filtered:
        max_score = filtered[0].get("_rrf_score", 0.0)
        if max_score > 0:
            for m in filtered:
                m["_rrf_score"] = m.get("_rrf_score", 0) / max_score

    # L3: touch 更新访问计数
    hit_ids = [m["id"] for m in filtered if "id" in m]
    if hit_ids:
        try:
            batch_touch_memories(db.conn, hit_ids)
        except Exception as e:
            logger.warning("recall touch 失败: %s", e)

    query_ms = (time.monotonic() - t0) * 1000
    logger.debug("recall: %d 条结果, mode=%s, %.1fms", len(filtered), mode, query_ms)

    return RecallResult(
        memories=filtered,
        total=len(filtered),
        mode=mode,
        query_ms=query_ms,
    )
