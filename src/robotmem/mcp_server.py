"""robotmem MCP Server — 7 个工具

写入: learn, save_perception
读取: recall
修改: forget, update
会话: start_session, end_session
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from mcp.server.fastmcp import Context, FastMCP

from .auto_classify import (
    build_context_json,
    classify_category,
    classify_tags,
    estimate_confidence,
    extract_scope,
    normalize_scope_files,
)
from .config import Config, load_config
from .db_cog import CogDatabase
from .dedup import check_duplicate
from .embed import Embedder, create_embedder
from .ops.memories import (
    apply_time_decay,
    consolidate_session as do_consolidate,
    get_memory,
    insert_memory,
    invalidate_memory,
    update_memory,
    update_memory_embedding,
)
from .ops.sessions import (
    get_or_create_session,
    get_session_summary,
    insert_session_outcome,
    mark_session_ended,
    update_session_context,
)
from .ops.tags import add_tags
from .resilience import mcp_error_boundary
from .search import recall as do_recall
from .validators import (
    EndSessionParams,
    ForgetParams,
    LearnParams,
    RecallParams,
    SavePerceptionParams,
    StartSessionParams,
    UpdateParams,
    parse_params,
)

logger = logging.getLogger(__name__)

# ── AppContext ──

@dataclass
class AppContext:
    """MCP 服务全局上下文"""
    config: Config
    db_cog: CogDatabase
    embedder: Embedder
    default_collection: str = "default"


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """服务生命周期 — 启动: DB + Embedder，关闭: 释放资源"""
    config = load_config()
    db_cog = CogDatabase(config)
    # 触发 lazy connect
    try:
        _ = db_cog.conn
    except Exception as e:
        logger.error("robotmem 启动失败: 数据库连接异常 — %s", e)
        raise

    embedder = create_embedder(config)

    # 检查 embedding 可用性
    try:
        embed_ok = await embedder.check_availability()
        if embed_ok:
            logger.info(
                "robotmem 启动: %s 后端可用，模型 %s (%dd)",
                config.embed_backend, embedder.model, embedder.dim,
            )
        else:
            logger.warning(
                "robotmem 启动: embedding 不可用 — %s。仅 BM25 搜索可用",
                embedder.unavailable_reason,
            )
    except Exception as e:
        logger.warning("robotmem 启动: embedding 检测异常 — %s", e)

    try:
        yield AppContext(
            config=config,
            db_cog=db_cog,
            embedder=embedder,
            default_collection=config.default_collection,
        )
    finally:
        await embedder.close()
        db_cog.close()


mcp = FastMCP("robotmem", lifespan=app_lifespan)


def _get_ctx(ctx: Context) -> AppContext:
    """从 MCP Context 获取 AppContext"""
    return ctx.request_context.lifespan_context


def _resolve_collection(app: AppContext, collection: str | None) -> str:
    """collection 参数解析 — 优先用户传入，否则用默认值"""
    return collection.strip() if collection and collection.strip() else app.default_collection


# ── Tool 1: learn ──

@mcp.tool()
@mcp_error_boundary
async def learn(
    insight: str,
    ctx: Context,
    context: str = "",
    collection: str | None = None,
    session_id: str | None = None,
) -> dict:
    """记录物理经验（declarative memory）

    三层防御：
    - L1 事前：insight 非空 + 截断 300 字
    - L2 事中：auto_classify + dedup + 原子写入
    - L3 事后：返回 memory_id + auto_inferred
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    result = parse_params(
        LearnParams, insight=insight, context=context,
        collection=collection, session_id=session_id,
    )
    if isinstance(result, dict):
        return result
    params = result

    coll = _resolve_collection(app, params.collection)
    insight = params.insight
    context = params.context

    # L2: auto_classify — 每步 try-except 降级
    try:
        category = classify_category(insight)
    except Exception as e:
        logger.debug("learn classify_category 降级: %s", e)
        category = "observation"

    try:
        confidence = estimate_confidence(insight, context)
    except Exception as e:
        logger.debug("learn estimate_confidence 降级: %s", e)
        confidence = 0.9

    try:
        scope = extract_scope(insight)
        scope_files = normalize_scope_files(scope.get("scope_files", []))
        scope_entities = scope.get("scope_entities", [])
    except Exception as e:
        logger.debug("learn extract_scope 降级: %s", e)
        scope_files, scope_entities = [], []

    try:
        inferred_tags = classify_tags(insight, context)
    except Exception as e:
        logger.debug("learn classify_tags 降级: %s", e)
        inferred_tags = []

    try:
        ctx_json = build_context_json(insight, context)
    except Exception as e:
        logger.debug("learn build_context_json 降级: %s", e)
        ctx_json = context

    # L2: 去重
    try:
        dedup_result = check_duplicate(
            insight, coll, session_id,
            app.db_cog, app.embedder if app.embedder.available else None,
        )
        if dedup_result.is_dup:
            existing_id = (
                dedup_result.similar_facts[0].get("id")
                if dedup_result.similar_facts else None
            )
            return {
                "status": "duplicate",
                "method": dedup_result.method,
                "existing_id": existing_id,
                "similarity": dedup_result.similarity,
            }
    except Exception as e:
        logger.warning("learn 去重检查异常: %s", e)

    # L2: embedding（降级为 None）
    embedding = None
    if app.embedder.available:
        try:
            embedding = await app.embedder.embed_one(insight)
        except Exception as e:
            logger.warning("learn embedding 失败: %s", e)

    # L2: 原子写入
    memory_id = insert_memory(app.db_cog.conn, {
        "session_id": params.session_id,
        "collection": coll,
        "type": "fact",
        "content": insight,
        "human_summary": insight[:200],
        "context": ctx_json if isinstance(ctx_json, str) else json.dumps(ctx_json),
        "category": category,
        "confidence": confidence,
        "source": "tool",
        "scope": "project",
        "scope_files": json.dumps(scope_files),
        "scope_entities": json.dumps(scope_entities),
        "embedding": embedding,
        "tags": inferred_tags,
        "tag_source": "auto",
    }, vec_loaded=app.db_cog.vec_loaded)

    if not memory_id:
        return {"error": "写入失败（可能重复）"}

    # L3: 返回
    return {
        "status": "created",
        "memory_id": memory_id,
        "auto_inferred": {
            "category": category,
            "confidence": confidence,
            "tags": inferred_tags,
            "scope_files": scope_files,
        },
    }


# ── Tool 2: recall ──

@mcp.tool()
@mcp_error_boundary
async def recall(
    query: str,
    ctx: Context,
    collection: str | None = None,
    n: int = 5,
    min_confidence: float = 0.3,
    session_id: str | None = None,
    context_filter: str | None = None,
    spatial_sort: str | None = None,
) -> dict:
    """检索经验 — BM25 + Vec 混合搜索

    传 session_id 时进入 episode 回放模式，返回该 session 全部记忆按时间排序。

    context_filter: JSON 字符串，结构化过滤条件。
        等值: '{"task.success": true}'
        范围: '{"params.final_distance.value": {"$lt": 0.05}}'

    spatial_sort: JSON 字符串，空间近邻排序。
        '{"field": "spatial.object_position", "target": [1.3, 0.7, 0.42]}'
        可选 max_distance 截断: '{"field": "...", "target": [...], "max_distance": 0.1}'
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    validated = parse_params(
        RecallParams, query=query, collection=collection,
        n=n, min_confidence=min_confidence, session_id=session_id,
        context_filter=context_filter, spatial_sort=spatial_sort,
    )
    if isinstance(validated, dict):
        return validated
    params = validated

    coll = _resolve_collection(app, params.collection)

    # L2: 解析 context_filter JSON
    cf_dict = None
    if params.context_filter:
        try:
            cf_dict = json.loads(params.context_filter)
            if not isinstance(cf_dict, dict):
                return {"error": "context_filter 必须为 JSON 对象"}
            if len(cf_dict) > 10:
                return {"error": "context_filter 过滤条件不得超过 10 项"}
        except json.JSONDecodeError as e:
            return {"error": f"context_filter JSON 解析失败: {e}"}

    # L2: 解析 spatial_sort JSON
    ss_dict = None
    if params.spatial_sort:
        try:
            ss_dict = json.loads(params.spatial_sort)
            if not isinstance(ss_dict, dict):
                return {"error": "spatial_sort 必须为 JSON 对象"}
            if "field" not in ss_dict or "target" not in ss_dict:
                return {"error": "spatial_sort 必须包含 field 和 target 字段"}
            if not isinstance(ss_dict.get("target"), list):
                return {"error": "spatial_sort.target 必须为数组"}
        except json.JSONDecodeError as e:
            return {"error": f"spatial_sort JSON 解析失败: {e}"}

    result = await do_recall(
        query=params.query,
        db=app.db_cog,
        embedder=app.embedder if app.embedder.available else None,
        collection=coll,
        top_k=params.n,
        min_confidence=params.min_confidence,
        session_id=params.session_id,
        context_filter=cf_dict,
        spatial_sort=ss_dict,
    )

    return {
        "memories": result.memories,
        "total": result.total,
        "mode": result.mode,
        "query_ms": round(result.query_ms, 1),
    }


# ── Tool 3: save_perception ──

@mcp.tool()
@mcp_error_boundary
async def save_perception(
    description: str,
    ctx: Context,
    perception_type: str = "visual",
    data: str | None = None,
    metadata: str | None = None,
    collection: str | None = None,
    session_id: str | None = None,
) -> dict:
    """保存感知/轨迹/力矩（procedural memory）

    三层防御：
    - L1 事前：description 非空 + perception_type 白名单
    - L2 事中：embedding + 原子写入
    - L3 事后：返回 memory_id
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    result = parse_params(
        SavePerceptionParams, description=description,
        perception_type=perception_type, data=data, metadata=metadata,
        collection=collection, session_id=session_id,
    )
    if isinstance(result, dict):
        return result
    params = result

    coll = _resolve_collection(app, params.collection)
    description = params.description

    # L2: embedding（降级为 None）
    embedding = None
    if app.embedder.available:
        try:
            embedding = await app.embedder.embed_one(description)
        except Exception as e:
            logger.warning("save_perception embedding 失败: %s", e)

    # L2: 原子写入
    memory_id = insert_memory(app.db_cog.conn, {
        "session_id": params.session_id,
        "collection": coll,
        "type": "perception",
        "content": description,
        "human_summary": description[:200],
        "perception_type": params.perception_type,
        "perception_data": params.data,
        "perception_metadata": params.metadata,
        "category": "observation",
        "confidence": 0.9,
        "source": "tool",
        "scope": "project",
        "embedding": embedding,
    }, vec_loaded=app.db_cog.vec_loaded)

    if not memory_id:
        return {"error": "写入失败（可能重复）"}

    # L3: 返回
    return {
        "memory_id": memory_id,
        "perception_type": params.perception_type,
        "collection": coll,
        "has_embedding": embedding is not None,
    }


# ── Tool 4: forget ──

@mcp.tool()
@mcp_error_boundary
async def forget(
    memory_id: int,
    reason: str,
    ctx: Context,
) -> dict:
    """删除错误记忆（软删除）

    三层防御：
    - L1 事前：memory_id 正整数 + reason 非空
    - L2 事中：归属校验 + invalidate
    - L3 事后：返回确认
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    result = parse_params(ForgetParams, memory_id=memory_id, reason=reason)
    if isinstance(result, dict):
        return result
    params = result

    # L2: 归属校验
    mem = get_memory(app.db_cog.conn, params.memory_id)
    if not mem:
        return {"error": f"记忆 #{params.memory_id} 不存在"}
    if mem.get("status") != "active":
        return {"error": f"记忆 #{params.memory_id} 状态为 {mem.get('status')}，无法删除"}

    # L2: 软删除
    invalidate_memory(app.db_cog.conn, params.memory_id, params.reason)

    # L3: 返回
    return {
        "status": "forgotten",
        "memory_id": params.memory_id,
        "content": (mem.get("content") or "")[:100],
        "reason": params.reason,
    }


# ── Tool 5: update ──

@mcp.tool()
@mcp_error_boundary
async def update(
    memory_id: int,
    new_content: str,
    ctx: Context,
    context: str = "",
) -> dict:
    """修正记忆内容

    三层防御：
    - L1 事前：memory_id 正整数 + new_content 非空
    - L2 事中：归属校验 + auto_classify + 原子更新
    - L3 事后：返回 old/new 对照
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    result = parse_params(
        UpdateParams, memory_id=memory_id,
        new_content=new_content, context=context,
    )
    if isinstance(result, dict):
        return result
    params = result

    # L2: 归属校验
    mem = get_memory(app.db_cog.conn, params.memory_id)
    if not mem:
        return {"error": f"记忆 #{params.memory_id} 不存在"}
    if mem.get("status") != "active":
        return {"error": f"记忆 #{params.memory_id} 状态为 {mem.get('status')}，无法更新"}

    old_content = mem.get("content", "")

    # L2: 重新分类
    try:
        category = classify_category(params.new_content)
        confidence = estimate_confidence(params.new_content, params.context)
    except Exception:
        category = mem.get("category", "observation")
        confidence = mem.get("confidence", 0.9)

    # L2: 更新
    update_memory(
        app.db_cog.conn, params.memory_id,
        content=params.new_content,
        category=category,
        confidence=confidence,
    )

    # L2: 重建 embedding（降级跳过）
    if app.embedder.available:
        try:
            new_emb = await app.embedder.embed_one(params.new_content)
            update_memory_embedding(
                app.db_cog.conn, params.memory_id, new_emb, vec_loaded=app.db_cog.vec_loaded,
            )
        except Exception as e:
            logger.warning("update embedding 重建失败: %s", e)

    # L2: 重建 tags
    try:
        inferred_tags = classify_tags(params.new_content, params.context)
        if inferred_tags:
            add_tags(app.db_cog.conn, params.memory_id, inferred_tags, source="auto")
    except Exception as e:
        logger.warning("update tags 重建失败: %s", e)

    # L3: 返回
    return {
        "status": "updated",
        "memory_id": params.memory_id,
        "old_content": old_content[:100],
        "new_content": params.new_content[:100],
        "auto_inferred": {
            "category": category,
            "confidence": confidence,
        },
    }


# ── Tool 6: start/end session ──

@mcp.tool()
@mcp_error_boundary
async def start_session(
    ctx: Context,
    collection: str | None = None,
    context: str | None = None,
) -> dict:
    """开始新会话（episode）

    机器人应用推荐的 context 格式::

        start_session(context='{"robot_id": "arm-01", "robot_model": "UR5e",
                                "environment": "kitchen-3F", "task_domain": "pick-and-place"}')

    三层防御：
    - L1 事前：collection 解析
    - L2 事中：创建 session + 写入 context
    - L3 事后：返回 session_id
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    result = parse_params(
        StartSessionParams, collection=collection, context=context,
    )
    if isinstance(result, dict):
        return result
    params = result

    coll = _resolve_collection(app, params.collection)

    # L2: 创建 session
    ext_id = str(uuid.uuid4())
    session = get_or_create_session(app.db_cog.conn, ext_id, coll)
    if not session:
        return {"error": "创建 session 失败"}

    # L2: 写入 context
    if params.context:
        update_session_context(app.db_cog.conn, ext_id, params.context)

    # L3: 统计 active 记忆数
    try:
        active_count = app.db_cog.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE collection=? AND status='active'",
            (coll,),
        ).fetchone()[0]
    except Exception:
        active_count = 0

    logger.info("MCP start_session: session_id=%s, collection=%s", ext_id, coll)

    return {
        "session_id": ext_id,
        "collection": coll,
        "active_memories_count": active_count,
    }


@mcp.tool()
@mcp_error_boundary
async def end_session(
    session_id: str,
    ctx: Context,
    outcome_score: float | None = None,
) -> dict:
    """结束会话 — 标记结束 + 时间衰减 + 评分

    三层防御：
    - L1 事前：session_id 非空
    - L2 事中：mark_session_ended + apply_time_decay + insert_outcome
    - L3 事后：返回 summary
    """
    app = _get_ctx(ctx)

    # L1: Pydantic 校验
    result = parse_params(
        EndSessionParams, session_id=session_id, outcome_score=outcome_score,
    )
    if isinstance(result, dict):
        return result
    params = result

    # 查询 session 关联的 collection
    try:
        row = app.db_cog.conn.execute(
            "SELECT collection FROM sessions WHERE external_id=?",
            (params.session_id,),
        ).fetchone()
        coll = row[0] if row else app.default_collection
    except Exception:
        coll = app.default_collection

    # L2: 标记结束
    mark_session_ended(app.db_cog.conn, params.session_id)

    # L2: 时间衰减（Abbeel 产品架构 Round 2）
    decayed = 0
    try:
        decayed = apply_time_decay(app.db_cog.conn)
    except Exception as e:
        logger.warning("end_session time_decay 失败: %s", e)

    # L2: 记忆巩固 — 独立容错（Levine 约束）
    consolidated = {"merged_groups": 0, "superseded_count": 0}
    try:
        consolidated = do_consolidate(app.db_cog.conn, params.session_id, coll)
    except Exception as e:
        logger.warning("consolidate_session 失败: %s", e)

    # L2: proactive recall — 独立容错（Levine 约束）
    related = []
    try:
        # 取当前 session 最新记忆的 content 作为 query
        top_row = app.db_cog.conn.execute(
            "SELECT content FROM memories "
            "WHERE session_id=? AND collection=? AND status='active' "
            "ORDER BY created_at DESC LIMIT 1",
            [params.session_id, coll],
        ).fetchone()
        top_content = top_row[0] if top_row else ""
        if top_content:
            pr_result = await do_recall(
                query=top_content,
                db=app.db_cog,
                embedder=app.embedder if app.embedder.available else None,
                collection=coll,
                top_k=5,
            )
            related = [
                m for m in pr_result.memories
                if m.get("session_id") != params.session_id
            ][:5]
    except Exception as e:
        logger.warning("proactive recall 失败: %s", e)

    # L2: 记录评分
    if params.outcome_score is not None:
        try:
            insert_session_outcome(app.db_cog.conn, params.session_id, params.outcome_score)
        except Exception as e:
            logger.warning("end_session outcome 写入失败: %s", e)

    # L3: session 摘要
    summary = get_session_summary(app.db_cog.conn, params.session_id, coll)

    logger.info(
        "MCP end_session: session_id=%s, memories=%d, decayed=%d, consolidated=%d",
        params.session_id, summary.get("memory_count", 0), decayed,
        consolidated.get("superseded_count", 0),
    )

    return {
        "status": "ended",
        "session_id": params.session_id,
        "summary": summary,
        "decayed_count": decayed,
        "consolidated": consolidated,
        "related_memories": related,
    }


# ── 入口 ──

def main():
    """CLI 入口 — python -m robotmem"""
    mcp.run()


if __name__ == "__main__":
    main()
