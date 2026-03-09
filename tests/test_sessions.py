"""测试 ops/sessions.py — 会话生命周期"""

import json
import pytest

from robotmem.ops.sessions import (
    get_or_create_session,
    get_session_context,
    get_session_summary,
    insert_session_outcome,
    mark_session_ended,
    update_session_context,
)


class TestGetOrCreateSession:

    def test_create_new(self, conn):
        """首次创建 session"""
        result = get_or_create_session(conn, "ext-1", "test")
        assert result is not None
        assert result["external_id"] == "ext-1"
        assert result["collection"] == "test"
        assert result["session_count"] == 1

    def test_reuse_existing(self, conn):
        """相同 external_id 复用并递增 session_count"""
        s1 = get_or_create_session(conn, "ext-1", "test")
        s2 = get_or_create_session(conn, "ext-1", "test")
        assert s2["session_count"] == 2

    def test_none_external_id(self, conn):
        """external_id=None 每次创建新 session"""
        s1 = get_or_create_session(conn, None, "test")
        s2 = get_or_create_session(conn, None, "test")
        assert s1["id"] != s2["id"]

    def test_empty_collection_rejected(self, conn):
        """空 collection 返回 None"""
        result = get_or_create_session(conn, "ext-1", "")
        assert result is None


class TestSessionContext:

    def test_update_and_get(self, conn):
        """写入 + 读取 context JSON"""
        get_or_create_session(conn, "ext-ctx", "test")
        ctx = json.dumps({"task": "pick_and_place", "env": {"sim_or_real": "sim"}})
        update_session_context(conn, "ext-ctx", ctx)

        result = get_session_context(conn, "ext-ctx")
        assert result is not None
        assert result["task"] == "pick_and_place"
        assert result["env"]["sim_or_real"] == "sim"

    def test_get_nonexistent(self, conn):
        """不存在的 session 返回 None"""
        result = get_session_context(conn, "nonexistent")
        assert result is None

    def test_context_truncation(self, conn):
        """超过 64KB 的 context 被截断"""
        get_or_create_session(conn, "ext-big", "test")
        big_ctx = "x" * 70000
        update_session_context(conn, "ext-big", big_ctx)
        # 应该不崩溃


class TestSessionLifecycle:

    def test_mark_ended(self, conn):
        get_or_create_session(conn, "ext-end", "test")
        result = mark_session_ended(conn, "ext-end")
        assert result is True

        # 再次标记也不崩溃
        result2 = mark_session_ended(conn, "ext-end")

    def test_insert_outcome(self, conn):
        get_or_create_session(conn, "ext-outcome", "test")
        result = insert_session_outcome(conn, "ext-outcome", score=0.85)
        assert result is True

    def test_get_summary(self, conn_with_data):
        summary = get_session_summary(conn_with_data, "sess-1", "test")
        assert summary["memory_count"] == 5  # 3 facts + 2 perceptions
        assert "fact" in summary["by_type"]
        assert "perception" in summary["by_type"]
        assert summary["by_type"]["fact"] == 3
        assert summary["by_type"]["perception"] == 2
