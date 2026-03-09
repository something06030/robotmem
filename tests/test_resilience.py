"""resilience.py 测试 — 容错原语"""

import asyncio
import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from robotmem.resilience import (
    ServiceCooldown,
    safe_db_write,
    safe_db_transaction,
    mcp_error_boundary,
)


# ── ServiceCooldown ──


class TestServiceCooldown:
    """指数退避冷却器"""

    def test_initial_not_cooling(self):
        sc = ServiceCooldown("test")
        assert sc.is_cooling is False

    def test_initial_backoff_zero(self):
        sc = ServiceCooldown("test")
        assert sc.current_backoff == 0.0

    def test_one_failure(self):
        sc = ServiceCooldown("test", base_cooldown=60.0)
        sc.record_failure()
        assert sc.is_cooling is True
        assert sc.current_backoff == pytest.approx(60.0)

    def test_exponential_backoff(self):
        sc = ServiceCooldown("test", base_cooldown=10.0, backoff_factor=2.0)
        sc.record_failure()
        assert sc.current_backoff == pytest.approx(10.0)
        sc.record_failure()
        assert sc.current_backoff == pytest.approx(20.0)
        sc.record_failure()
        assert sc.current_backoff == pytest.approx(40.0)

    def test_max_cooldown_cap(self):
        sc = ServiceCooldown("test", base_cooldown=100.0, max_cooldown=200.0)
        for _ in range(10):
            sc.record_failure()
        assert sc.current_backoff <= 200.0

    def test_success_resets(self):
        sc = ServiceCooldown("test")
        sc.record_failure()
        sc.record_failure()
        sc.record_success()
        assert sc.is_cooling is False
        assert sc.current_backoff == 0.0

    def test_reset(self):
        sc = ServiceCooldown("test")
        sc.record_failure()
        sc.reset()
        assert sc.is_cooling is False
        assert sc._consecutive_failures == 0

    def test_cooling_expires(self):
        """冷却时间过后不再 cooling"""
        sc = ServiceCooldown("test", base_cooldown=0.01)
        sc.record_failure()
        time.sleep(0.05)  # 裕量 5x（冷却 10ms, 等 50ms）
        assert sc.is_cooling is False


# ── safe_db_write ──


class TestSafeDbWrite:
    """单条 DB 写保护"""

    @pytest.fixture
    def conn(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        yield c
        c.close()

    def test_success(self, conn):
        row_id = safe_db_write(conn, "INSERT INTO t (val) VALUES (?)", ["hello"])
        assert row_id is not None
        assert row_id > 0

    def test_returns_lastrowid(self, conn):
        row_id = safe_db_write(conn, "INSERT INTO t (val) VALUES (?)", ["a"])
        assert isinstance(row_id, int)

    def test_no_params(self, conn):
        row_id = safe_db_write(conn, "INSERT INTO t (val) VALUES ('x')")
        assert row_id is not None

    def _mock_conn(self, error):
        """构造 mock conn — 模拟 with conn: conn.execute() 抛异常"""
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.execute.side_effect = error
        return mock

    def test_locked_returns_none(self):
        """DB 锁超时 → None"""
        mock = self._mock_conn(sqlite3.OperationalError("database is locked"))
        result = safe_db_write(mock, "INSERT INTO t VALUES (?)", ["x"])
        assert result is None

    def test_disk_error_returns_none(self):
        """磁盘错误 → None"""
        mock = self._mock_conn(sqlite3.OperationalError("disk i/o error"))
        result = safe_db_write(mock, "INSERT INTO t VALUES (?)", ["x"])
        assert result is None

    def test_malformed_db_returns_none(self):
        """DB 损坏 → None"""
        mock = self._mock_conn(sqlite3.DatabaseError("database disk image is malformed"))
        result = safe_db_write(mock, "INSERT INTO t VALUES (?)", ["x"])
        assert result is None

    def test_unknown_error_reraises(self):
        """未知 OperationalError 重新抛出"""
        mock = self._mock_conn(sqlite3.OperationalError("something unexpected"))
        with pytest.raises(sqlite3.OperationalError, match="something unexpected"):
            safe_db_write(mock, "INSERT INTO t VALUES (?)", ["x"])


# ── safe_db_transaction ──


class TestSafeDbTransaction:
    """原子批量写入"""

    @pytest.fixture
    def conn(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        yield c
        c.close()

    def test_success(self, conn):
        def insert(c):
            c.execute("INSERT INTO t (val) VALUES ('a')")
            c.execute("INSERT INTO t (val) VALUES ('b')")
            return 2

        ok, result = safe_db_transaction(conn, insert)
        assert ok is True
        assert result == 2
        rows = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert rows == 2

    def test_locked_returns_false(self, conn):
        """DB 锁超时 → (False, None)"""
        def locked_fn(c):
            raise sqlite3.OperationalError("database is locked")

        ok, result = safe_db_transaction(conn, locked_fn)
        assert ok is False
        assert result is None

    def test_disk_full_returns_false(self, conn):
        def disk_fn(c):
            raise sqlite3.OperationalError("disk is full")

        ok, result = safe_db_transaction(conn, disk_fn)
        assert ok is False
        assert result is None

    def test_malformed_returns_false(self, conn):
        def malformed_fn(c):
            raise sqlite3.DatabaseError("not a database")

        ok, result = safe_db_transaction(conn, malformed_fn)
        assert ok is False
        assert result is None

    def test_unknown_error_reraises(self, conn):
        def bad_fn(c):
            raise sqlite3.OperationalError("unexpected error")

        with pytest.raises(sqlite3.OperationalError, match="unexpected error"):
            safe_db_transaction(conn, bad_fn)

    def test_rollback_on_failure(self, conn):
        """失败时事务回滚"""
        def partial_fn(c):
            c.execute("INSERT INTO t (val) VALUES ('a')")
            raise sqlite3.OperationalError("database is locked")

        ok, _ = safe_db_transaction(conn, partial_fn)
        assert ok is False
        # 事务回滚，不应该有数据
        rows = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert rows == 0


# ── mcp_error_boundary ──


class TestMcpErrorBoundary:
    """MCP 错误边界装饰器"""

    def test_normal_return(self):
        @mcp_error_boundary
        async def good_tool():
            return {"status": "ok"}

        result = asyncio.run(good_tool())
        assert result == {"status": "ok"}

    def test_db_error_caught(self):
        @mcp_error_boundary
        async def db_tool():
            raise sqlite3.DatabaseError("db crashed")

        result = asyncio.run(db_tool())
        assert "error" in result
        assert "数据库" in result["error"]

    def test_generic_error_caught(self):
        @mcp_error_boundary
        async def bad_tool():
            raise ValueError("something wrong")

        result = asyncio.run(bad_tool())
        assert "error" in result
        assert "内部错误" in result["error"]

    def test_preserves_function_name(self):
        @mcp_error_boundary
        async def my_tool():
            return {}

        assert my_tool.__name__ == "my_tool"

    def test_with_args(self):
        @mcp_error_boundary
        async def tool_with_args(a, b, c="default"):
            return {"sum": a + b, "c": c}

        result = asyncio.run(tool_with_args(1, 2, c="custom"))
        assert result == {"sum": 3, "c": "custom"}
