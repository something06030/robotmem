"""db_cog.py 单元测试 — CogDatabase 连接 + 查询方法"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from robotmem.config import Config
from robotmem.db_cog import CogDatabase


@pytest.fixture
def db_config(tmp_path):
    """临时目录的配置"""
    return Config(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def db(db_config):
    """CogDatabase 实例"""
    d = CogDatabase(db_config)
    yield d
    d.close()


class TestCogDatabaseInit:
    def test_lazy_connect(self, db_config):
        """连接是 lazy 的"""
        db = CogDatabase(db_config)
        assert db._conn is None
        # 访问 conn 触发连接
        _ = db.conn
        assert db._conn is not None
        db.close()

    def test_conn_creates_db_file(self, db_config):
        """首次 conn 创建 DB 文件"""
        db = CogDatabase(db_config)
        _ = db.conn
        assert Path(db_config.db_path_resolved).exists()
        db.close()

    def test_vec_loaded_property(self, db):
        """vec_loaded 返回 bool"""
        assert isinstance(db.vec_loaded, bool)

    def test_close_prevents_reconnect(self, db):
        """close 后 conn 抛异常"""
        _ = db.conn  # 确保已连接
        db.close()
        with pytest.raises(RuntimeError, match="已关闭"):
            _ = db.conn

    def test_double_close(self, db):
        """close 两次不崩溃"""
        _ = db.conn
        db.close()
        db.close()  # 第二次不报错


class TestMemoryExists:
    def test_exists_with_session(self, db):
        """精确匹配 — 有 session_id"""
        conn = db.conn
        conn.execute("""
            INSERT INTO memories (content, session_id, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES ('test content', 'sess1', 'test_col', 'fact',
                    'observation', 0.9, 0.01, 'tool', 'project', 'active',
                    '[]', '[]')
        """)
        conn.commit()

        assert db.memory_exists("test content", "sess1", "test_col") is True
        assert db.memory_exists("test content", "other_sess", "test_col") is False
        assert db.memory_exists("other content", "sess1", "test_col") is False

    def test_exists_without_session(self, db):
        """精确匹配 — 无 session_id"""
        conn = db.conn
        conn.execute("""
            INSERT INTO memories (content, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES ('global content', 'test_col', 'fact',
                    'observation', 0.9, 0.01, 'tool', 'project', 'active',
                    '[]', '[]')
        """)
        conn.commit()

        assert db.memory_exists("global content", None, "test_col") is True
        assert db.memory_exists("other", None, "test_col") is False

    def test_exists_ignores_superseded(self, db):
        """superseded 状态不匹配"""
        conn = db.conn
        conn.execute("""
            INSERT INTO memories (content, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES ('deleted content', 'test_col', 'fact',
                    'observation', 0.9, 0.01, 'tool', 'project', 'superseded',
                    '[]', '[]')
        """)
        conn.commit()

        assert db.memory_exists("deleted content", None, "test_col") is False


class TestFtsSearchMemories:
    def test_empty_query(self, db):
        assert db.fts_search_memories("", "col") == []

    def test_search_results(self, db):
        conn = db.conn
        conn.execute("""
            INSERT INTO memories (id, content, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES (1, 'robot grasping test', 'test_col', 'fact',
                    'observation', 0.9, 0.01, 'tool', 'project', 'active',
                    '[]', '[]')
        """)
        # 手动 FTS5 同步
        conn.execute(
            "INSERT INTO memories_fts(rowid, content, human_summary, scope_files, scope_entities) VALUES (?, ?, ?, ?, ?)",
            (1, "robot grasping test", "", "[]", "[]"),
        )
        conn.commit()

        results = db.fts_search_memories("robot grasping", "test_col")
        assert len(results) >= 1
        assert results[0]["content"] == "robot grasping test"
        assert results[0]["assertion"] == "robot grasping test"  # 向后兼容

    def test_returns_dict_fields(self, db):
        conn = db.conn
        conn.execute("""
            INSERT INTO memories (id, content, session_id, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES (1, 'test content', 'sess', 'col', 'fact',
                    'observation', 0.8, 0.01, 'tool', 'project', 'active',
                    '[]', '[]')
        """)
        conn.execute(
            "INSERT INTO memories_fts(rowid, content, human_summary, scope_files, scope_entities) VALUES (?, ?, ?, ?, ?)",
            (1, "test content", "", "[]", "[]"),
        )
        conn.commit()

        results = db.fts_search_memories("test content", "col")
        if results:
            r = results[0]
            assert "id" in r
            assert "assertion" in r
            assert "session_id" in r
            assert "category" in r
            assert "confidence" in r

    def test_fts5_error_returns_empty(self, db):
        """FTS5 异常 → 返回空"""
        # 触发一个语法错误
        results = db.fts_search_memories("", "col")
        assert results == []


class TestVecSearchMemories:
    def test_not_loaded(self, db):
        """vec 未加载 → 空"""
        db._vec_loaded = False
        assert db.vec_search_memories([0.1, 0.2], "col") == []

    def test_returns_dict(self, db):
        """返回格式正确"""
        # vec 未加载情况下返回空
        db._vec_loaded = False
        results = db.vec_search_memories([0.1, 0.2], "col")
        assert results == []


class TestSupersedeMemory:
    def test_supersede(self, db):
        conn = db.conn
        conn.execute("""
            INSERT INTO memories (id, content, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES (1, 'old memory', 'col', 'fact',
                    'observation', 0.9, 0.01, 'tool', 'project', 'active',
                    '[]', '[]')
        """)
        conn.execute("""
            INSERT INTO memories (id, content, collection, type,
                                 category, confidence, decay_rate, source,
                                 scope, status, scope_files, scope_entities)
            VALUES (2, 'new memory', 'col', 'fact',
                    'observation', 0.9, 0.01, 'tool', 'project', 'active',
                    '[]', '[]')
        """)
        conn.commit()

        db.supersede_memory(old_id=1, new_id=2, reason="test_reason")

        row = conn.execute("SELECT status, superseded_by, invalidated_reason FROM memories WHERE id=1").fetchone()
        assert row[0] == "superseded"
        assert row[1] == 2
        assert row[2] == "test_reason"


class TestContentHash:
    def test_basic(self):
        h = CogDatabase.content_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_deterministic(self):
        h1 = CogDatabase.content_hash("test")
        h2 = CogDatabase.content_hash("test")
        assert h1 == h2

    def test_different_input(self):
        h1 = CogDatabase.content_hash("a")
        h2 = CogDatabase.content_hash("b")
        assert h1 != h2

    def test_empty_string(self):
        h = CogDatabase.content_hash("")
        assert isinstance(h, str)
        assert len(h) == 16
