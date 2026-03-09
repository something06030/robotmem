"""schema.py 单元测试 — DDL 初始化 + vec0"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from robotmem.schema import initialize_schema, initialize_vec


class TestInitializeSchema:
    def test_creates_tables(self):
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        # 检查 5 个核心表
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "memories" in table_names
        assert "sessions" in table_names
        assert "session_outcomes" in table_names
        assert "memory_tags" in table_names
        assert "tag_meta" in table_names
        conn.close()

    def test_creates_indexes(self):
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        idx_names = {r[0] for r in indexes}
        assert "idx_mem_collection" in idx_names
        assert "idx_mem_status" in idx_names
        assert "idx_mem_session" in idx_names
        conn.close()

    def test_creates_fts5(self):
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "memories_fts" in table_names
        conn.close()

    def test_idempotent(self):
        """重复调用不报错"""
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        initialize_schema(conn)  # 第二次不报错
        conn.close()

    def test_human_summary_migration(self):
        """已存在 human_summary 列时不报错"""
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        # 第二次调用会尝试 ALTER TABLE ADD COLUMN → 列已存在 → 捕获
        initialize_schema(conn)
        # 验证列存在
        cursor = conn.execute("PRAGMA table_info(memories)")
        cols = {r[1] for r in cursor.fetchall()}
        assert "human_summary" in cols
        conn.close()


class TestInitializeVec:
    def test_invalid_dim_zero(self):
        conn = sqlite3.connect(":memory:")
        assert initialize_vec(conn, dim=0) is False
        conn.close()

    def test_invalid_dim_negative(self):
        conn = sqlite3.connect(":memory:")
        assert initialize_vec(conn, dim=-1) is False
        conn.close()

    def test_invalid_dim_too_large(self):
        conn = sqlite3.connect(":memory:")
        assert initialize_vec(conn, dim=100000) is False
        conn.close()

    def test_invalid_dim_not_int(self):
        conn = sqlite3.connect(":memory:")
        assert initialize_vec(conn, dim="384") is False
        conn.close()

    def test_no_sqlite_vec_module(self):
        """sqlite_vec 未安装 → False"""
        conn = sqlite3.connect(":memory:")
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            with patch("builtins.__import__", side_effect=ImportError("no sqlite_vec")):
                result = initialize_vec(conn, dim=384)
                # ImportError 被捕获 → 返回 False
        conn.close()
        assert result is False

    def test_valid_dim(self):
        """合法 dim 值（如果 sqlite_vec 已安装则成功，否则 False）"""
        conn = sqlite3.connect(":memory:")
        result = initialize_vec(conn, dim=384)
        # 不论成功失败都返回 bool
        assert isinstance(result, bool)
        conn.close()
