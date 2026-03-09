"""测试共享 fixtures — 内存 DB + 隔离环境"""

import sqlite3
import pytest

from robotmem.config import Config
from robotmem.schema import initialize_schema


@pytest.fixture
def config(tmp_path):
    """临时配置 — DB 在 tmp_path 下"""
    return Config(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def conn(tmp_path):
    """内存中的 SQLite 连接 + schema 初始化"""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    initialize_schema(c)
    yield c
    c.close()


@pytest.fixture
def conn_with_data(conn):
    """预填充测试数据的连接"""
    # 插入 3 条 fact 类型记忆
    for i in range(1, 4):
        conn.execute("""
            INSERT INTO memories (session_id, collection, type, content, category,
                                  confidence, decay_rate, source, scope, status,
                                  content_hash, scope_files, scope_entities)
            VALUES (?, 'test', 'fact', ?, 'observation', 0.9, 0.01,
                    'tool', 'project', 'active', ?, '[]', '[]')
        """, [f"sess-1", f"测试记忆内容 {i}", f"hash{i}"])

    # 插入 2 条 perception 类型记忆
    for i in range(1, 3):
        conn.execute("""
            INSERT INTO memories (session_id, collection, type, content,
                                  perception_type, category, confidence,
                                  decay_rate, source, scope, status,
                                  content_hash, scope_files, scope_entities)
            VALUES (?, 'test', 'perception', ?, 'tactile', 'observation',
                    0.9, 0.01, 'tool', 'project', 'active', ?, '[]', '[]')
        """, [f"sess-1", f"触觉感知数据 {i}", f"phash{i}"])

    # 插入 session
    conn.execute("""
        INSERT INTO sessions (external_id, collection, session_count, status)
        VALUES ('sess-1', 'test', 1, 'active')
    """)

    conn.commit()
    return conn
