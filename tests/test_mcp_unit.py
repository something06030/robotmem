"""mcp_server.py 单元测试 — 6 个 MCP tool 的核心逻辑"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from robotmem.mcp_server import (
    AppContext,
    _resolve_collection,
    learn,
    recall,
    save_perception,
    forget,
    update,
    start_session,
    end_session,
)


# ── 测试辅助 ──

def _make_app_context(*, default_collection="default", vec_loaded=False):
    """构造 AppContext mock"""
    config = MagicMock()
    config.default_collection = default_collection

    db_cog = MagicMock()
    db_cog.vec_loaded = vec_loaded
    db_cog.conn = MagicMock()

    embedder = MagicMock()
    embedder.available = False
    embedder.unavailable_reason = "test"

    return AppContext(
        config=config,
        db_cog=db_cog,
        embedder=embedder,
        default_collection=default_collection,
    )


def _make_ctx(app):
    """构造 MCP Context mock"""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


# ── _resolve_collection ──


class TestResolveCollection:
    def test_user_value(self):
        app = _make_app_context()
        assert _resolve_collection(app, "my_project") == "my_project"

    def test_empty_string(self):
        app = _make_app_context(default_collection="default")
        assert _resolve_collection(app, "") == "default"

    def test_none(self):
        app = _make_app_context(default_collection="default")
        assert _resolve_collection(app, None) == "default"

    def test_whitespace(self):
        app = _make_app_context(default_collection="default")
        assert _resolve_collection(app, "   ") == "default"

    def test_strip(self):
        app = _make_app_context()
        assert _resolve_collection(app, " trimmed ") == "trimmed"


# ── Tool 1: learn ──


class TestLearn:
    @pytest.mark.asyncio
    async def test_learn_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.check_duplicate") as mock_dedup:
            mock_dedup.return_value = MagicMock(is_dup=False)
            with patch("robotmem.mcp_server.insert_memory", return_value=42):
                result = await learn("重要经验", ctx, collection="test")

        assert result["status"] == "created"
        assert result["memory_id"] == 42

    @pytest.mark.asyncio
    async def test_learn_duplicate(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        mock_result = MagicMock()
        mock_result.is_dup = True
        mock_result.method = "exact"
        mock_result.similarity = 1.0
        mock_result.similar_facts = [{"id": 1}]

        with patch("robotmem.mcp_server.check_duplicate", return_value=mock_result):
            result = await learn("重复的经验", ctx, collection="test")

        assert result["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_learn_empty_insight(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await learn("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_learn_write_failure(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.check_duplicate") as mock_dedup:
            mock_dedup.return_value = MagicMock(is_dup=False)
            with patch("robotmem.mcp_server.insert_memory", return_value=None):
                result = await learn("经验", ctx, collection="test")

        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_learn_with_embedding(self):
        app = _make_app_context()
        app.embedder.available = True
        app.embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.check_duplicate") as mock_dedup:
            mock_dedup.return_value = MagicMock(is_dup=False)
            with patch("robotmem.mcp_server.insert_memory", return_value=1):
                result = await learn("test insight", ctx, collection="test")

        assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_learn_classify_fallback(self):
        """auto_classify 异常 → 降级"""
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.classify_category", side_effect=Exception("fail")):
            with patch("robotmem.mcp_server.check_duplicate") as mock_dedup:
                mock_dedup.return_value = MagicMock(is_dup=False)
                with patch("robotmem.mcp_server.insert_memory", return_value=1):
                    result = await learn("test", ctx, collection="test")

        assert result["status"] == "created"


# ── Tool 2: recall ──


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        mock_result = MagicMock()
        mock_result.memories = [{"id": 1, "content": "test"}]
        mock_result.total = 1
        mock_result.mode = "bm25_only"
        mock_result.query_ms = 5.0

        with patch("robotmem.mcp_server.do_recall", return_value=mock_result):
            result = await recall("test query", ctx, collection="test")

        assert result["total"] == 1
        assert result["mode"] == "bm25_only"

    @pytest.mark.asyncio
    async def test_recall_empty_query(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("", ctx)
        # Pydantic 校验拒绝空 query
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_with_context_filter(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        mock_result = MagicMock()
        mock_result.memories = []
        mock_result.total = 0
        mock_result.mode = "bm25_only"
        mock_result.query_ms = 1.0

        with patch("robotmem.mcp_server.do_recall", return_value=mock_result):
            result = await recall(
                "test", ctx, collection="test",
                context_filter='{"task.success": true}',
            )

        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_recall_invalid_context_filter_json(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("test", ctx, context_filter="not json")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_context_filter_not_dict(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("test", ctx, context_filter="[1,2,3]")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_context_filter_too_many(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        many = {f"key{i}": i for i in range(11)}
        result = await recall("test", ctx, context_filter=json.dumps(many))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_with_spatial_sort(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        mock_result = MagicMock()
        mock_result.memories = []
        mock_result.total = 0
        mock_result.mode = "bm25_only"
        mock_result.query_ms = 1.0

        ss = json.dumps({"field": "spatial.pos", "target": [1.0, 2.0]})
        with patch("robotmem.mcp_server.do_recall", return_value=mock_result):
            result = await recall("test", ctx, collection="test", spatial_sort=ss)

        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_recall_invalid_spatial_sort(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("test", ctx, spatial_sort='{"bad": "format"}')
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_spatial_sort_target_not_list(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        ss = json.dumps({"field": "pos", "target": "not_list"})
        result = await recall("test", ctx, spatial_sort=ss)
        assert "error" in result


# ── Tool 3: save_perception ──


class TestSavePerception:
    @pytest.mark.asyncio
    async def test_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.insert_memory", return_value=10):
            result = await save_perception("触觉反馈数据", ctx, collection="test")

        assert result["memory_id"] == 10
        assert result["perception_type"] == "visual"

    @pytest.mark.asyncio
    async def test_empty_description(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await save_perception("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_custom_perception_type(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.insert_memory", return_value=1):
            result = await save_perception(
                "力矩数据记录", ctx, perception_type="tactile", collection="test",
            )

        assert result["perception_type"] == "tactile"

    @pytest.mark.asyncio
    async def test_write_failure(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.insert_memory", return_value=None):
            result = await save_perception("test", ctx, collection="test")

        assert "error" in result


# ── Tool 4: forget ──


class TestForget:
    @pytest.mark.asyncio
    async def test_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value={
            "id": 1, "content": "old memory", "status": "active",
        }):
            with patch("robotmem.mcp_server.invalidate_memory"):
                result = await forget(1, "错误记忆", ctx)

        assert result["status"] == "forgotten"

    @pytest.mark.asyncio
    async def test_not_found(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value=None):
            result = await forget(999, "test", ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_already_deleted(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value={
            "id": 1, "content": "x", "status": "superseded",
        }):
            result = await forget(1, "test", ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_reason(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await forget(1, "", ctx)
        assert "error" in result


# ── Tool 5: update ──


class TestUpdate:
    @pytest.mark.asyncio
    async def test_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value={
            "id": 1, "content": "old content", "status": "active",
            "category": "observation", "confidence": 0.9,
        }):
            with patch("robotmem.mcp_server.update_memory"):
                result = await update(1, "新内容", ctx)

        assert result["status"] == "updated"
        assert result["old_content"] == "old content"

    @pytest.mark.asyncio
    async def test_not_found(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value=None):
            result = await update(999, "new", ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_not_active(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value={
            "id": 1, "content": "x", "status": "superseded",
        }):
            result = await update(1, "new", ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_content(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await update(1, "", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_embedding_rebuild(self):
        app = _make_app_context()
        app.embedder.available = True
        app.embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_memory", return_value={
            "id": 1, "content": "old", "status": "active",
            "category": "observation", "confidence": 0.9,
        }):
            with patch("robotmem.mcp_server.update_memory"):
                with patch("robotmem.mcp_server.update_memory_embedding"):
                    result = await update(1, "new content", ctx)

        assert result["status"] == "updated"


# ── Tool 6: start/end session ──


class TestStartSession:
    @pytest.mark.asyncio
    async def test_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        app.db_cog.conn.execute.return_value.fetchone.return_value = (5,)

        with patch("robotmem.mcp_server.get_or_create_session", return_value={"id": 1}):
            result = await start_session(ctx, collection="test")

        assert "session_id" in result
        assert result["collection"] == "test"

    @pytest.mark.asyncio
    async def test_create_failure(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        with patch("robotmem.mcp_server.get_or_create_session", return_value=None):
            result = await start_session(ctx, collection="test")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_context(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        app.db_cog.conn.execute.return_value.fetchone.return_value = (0,)

        with patch("robotmem.mcp_server.get_or_create_session", return_value={"id": 1}):
            with patch("robotmem.mcp_server.update_session_context"):
                result = await start_session(
                    ctx, collection="test",
                    context='{"robot_id": "arm-01"}',
                )

        assert "session_id" in result


class TestEndSession:
    @pytest.mark.asyncio
    async def test_success(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        app.db_cog.conn.execute.return_value.fetchone.return_value = ("test",)

        with patch("robotmem.mcp_server.mark_session_ended"):
            with patch("robotmem.mcp_server.apply_time_decay", return_value=5):
                with patch("robotmem.mcp_server.do_consolidate", return_value={
                    "merged_groups": 0, "superseded_count": 0,
                }):
                    with patch("robotmem.mcp_server.get_session_summary", return_value={
                        "memory_count": 3,
                    }):
                        result = await end_session("sess-1", ctx)

        assert result["status"] == "ended"
        assert result["decayed_count"] == 5

    @pytest.mark.asyncio
    async def test_empty_session_id(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await end_session("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_outcome_score(self):
        app = _make_app_context()
        ctx = _make_ctx(app)

        app.db_cog.conn.execute.return_value.fetchone.return_value = ("test",)

        with patch("robotmem.mcp_server.mark_session_ended"):
            with patch("robotmem.mcp_server.apply_time_decay", return_value=0):
                with patch("robotmem.mcp_server.do_consolidate", return_value={
                    "merged_groups": 0, "superseded_count": 0,
                }):
                    with patch("robotmem.mcp_server.insert_session_outcome"):
                        with patch("robotmem.mcp_server.get_session_summary", return_value={}):
                            result = await end_session(
                                "sess-1", ctx, outcome_score=0.85,
                            )

        assert result["status"] == "ended"

    @pytest.mark.asyncio
    async def test_consolidate_failure(self):
        """consolidate 失败不影响返回"""
        app = _make_app_context()
        ctx = _make_ctx(app)

        app.db_cog.conn.execute.return_value.fetchone.return_value = ("test",)

        with patch("robotmem.mcp_server.mark_session_ended"):
            with patch("robotmem.mcp_server.apply_time_decay", return_value=0):
                with patch("robotmem.mcp_server.do_consolidate", side_effect=Exception("fail")):
                    with patch("robotmem.mcp_server.get_session_summary", return_value={}):
                        result = await end_session("sess-1", ctx)

        assert result["status"] == "ended"

    @pytest.mark.asyncio
    async def test_time_decay_failure(self):
        """time_decay 失败不影响返回"""
        app = _make_app_context()
        ctx = _make_ctx(app)

        app.db_cog.conn.execute.return_value.fetchone.return_value = ("test",)

        with patch("robotmem.mcp_server.mark_session_ended"):
            with patch("robotmem.mcp_server.apply_time_decay", side_effect=Exception("fail")):
                with patch("robotmem.mcp_server.do_consolidate", return_value={
                    "merged_groups": 0, "superseded_count": 0,
                }):
                    with patch("robotmem.mcp_server.get_session_summary", return_value={}):
                        result = await end_session("sess-1", ctx)

        assert result["status"] == "ended"
        assert result["decayed_count"] == 0


# ── AppContext ──


class TestAppContext:
    def test_dataclass_fields(self):
        app = _make_app_context()
        assert app.default_collection == "default"
        assert app.config is not None
        assert app.db_cog is not None
        assert app.embedder is not None
