"""数据库 Schema — DDL + 初始化

统一记忆表（memories 合并 facts + perceptions），7 个表/索引对象。
"""

from __future__ import annotations

import logging
import sqlite3

from .db import SUPPORTS_CONTENTLESS_DELETE

logger = logging.getLogger(__name__)

# --- 核心表 DDL ---

_MEMORIES_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    collection          TEXT NOT NULL DEFAULT 'default',
    type                TEXT NOT NULL DEFAULT 'fact',
    content             TEXT NOT NULL,
    human_summary       TEXT,
    context             TEXT,
    perception_type     TEXT,
    perception_data     BLOB,
    perception_metadata TEXT,
    concept             TEXT,
    category            TEXT DEFAULT 'observation',
    confidence          REAL NOT NULL DEFAULT 0.9,
    decay_rate          REAL NOT NULL DEFAULT 0.01,
    source              TEXT DEFAULT 'tool',
    scope               TEXT DEFAULT 'project',
    status              TEXT NOT NULL DEFAULT 'active',
    superseded_by       INTEGER,
    invalidated_reason  TEXT,
    invalidated_at      TEXT,
    content_hash        TEXT,
    media_hash          TEXT,
    access_count        INTEGER DEFAULT 0,
    return_count        INTEGER DEFAULT 0,
    last_accessed       TEXT,
    last_validated      TEXT,
    embedding           BLOB,
    scope_files         TEXT DEFAULT '[]',
    scope_entities      TEXT DEFAULT '[]',
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
)
"""

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id     TEXT UNIQUE,
    collection      TEXT NOT NULL DEFAULT 'default',
    context         TEXT,
    session_count   INTEGER DEFAULT 1,
    status          TEXT DEFAULT 'active',
    client_type     TEXT DEFAULT 'mcp_direct',
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
)
"""

_SESSION_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS session_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    score       REAL,
    created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
)
"""

_MEMORY_TAGS_DDL = """
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id   INTEGER NOT NULL,
    tag         TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'auto',
    created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (memory_id, tag)
)
"""

_TAG_META_DDL = """
CREATE TABLE IF NOT EXISTS tag_meta (
    tag         TEXT PRIMARY KEY,
    parent      TEXT,
    display_name TEXT,
    created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
)
"""

# --- 索引 DDL ---

_INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_mem_collection ON memories(collection)",
    "CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)",
    "CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)",
    "CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash) WHERE content_hash IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_mem_no_embed ON memories(collection) WHERE embedding IS NULL AND status='active'",
    "CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)",
]

# --- FTS5 DDL ---

_FTS5_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
    "content, human_summary, scope_files, scope_entities, "
    "content=''"
    + (", contentless_delete=1" if SUPPORTS_CONTENTLESS_DELETE else "")
    + ")"
)

# --- FTS5 触发器 ---
# contentless_delete=1 时，用 DELETE FROM 替代特殊 'delete' 命令
# 无 contentless_delete 时，用 INSERT ... VALUES('delete', ...) 传统语法

# INSERT 触发器已移除 — FTS5 写入由 ops/memories.py 手动同步（需 jieba 分词 CJK）
# 仅保留 DELETE / UPDATE 触发器（删除不需要分词）

if SUPPORTS_CONTENTLESS_DELETE:
    _FTS5_TRIGGERS = [
        """
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            DELETE FROM memories_fts WHERE rowid = old.id;
        END
        """,
    ]
else:
    _FTS5_TRIGGERS = [
        """
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, human_summary, scope_files, scope_entities)
            VALUES ('delete', old.id, old.content, old.human_summary, old.scope_files, old.scope_entities);
        END
        """,
    ]


def initialize_schema(conn: sqlite3.Connection) -> None:
    """创建所有表、索引、触发器

    幂等操作 — 重复调用安全。
    """
    # 核心表
    for ddl in [_MEMORIES_DDL, _SESSIONS_DDL, _SESSION_OUTCOMES_DDL,
                _MEMORY_TAGS_DDL, _TAG_META_DDL]:
        conn.execute(ddl)

    # B-tree 索引
    for idx in _INDEXES_DDL:
        conn.execute(idx)

    # 迁移：已有数据库补 human_summary 列（必须在 FTS5 触发器之前，触发器引用此列）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN human_summary TEXT")
        logger.info("迁移：memories 表已添加 human_summary 列")
    except sqlite3.OperationalError:
        pass  # 列已存在

    # FTS5
    try:
        conn.execute(_FTS5_DDL)
        # 迁移：删除旧的 INSERT/UPDATE 触发器（改为 Python 手动同步以支持 jieba 分词）
        for old_trigger in ("memories_ai", "memories_au"):
            try:
                conn.execute(f"DROP TRIGGER IF EXISTS {old_trigger}")
            except sqlite3.OperationalError:
                pass
        for trigger in _FTS5_TRIGGERS:
            conn.execute(trigger)
    except sqlite3.OperationalError as e:
        logger.warning("FTS5 创建失败（可能已存在或不支持）: %s", e)

    conn.commit()
    logger.info("robotmem schema 初始化完成")


def initialize_vec(conn: sqlite3.Connection, dim: int = 384) -> bool:
    """创建 vec0 向量索引

    Returns:
        True 成功，False 失败（sqlite-vec 未安装）
    """
    if not isinstance(dim, int) or not (1 <= dim <= 65536):
        logger.error("initialize_vec: 非法 dim=%r，拒绝执行", dim)
        return False

    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (ImportError, AttributeError, sqlite3.OperationalError) as e:
        logger.warning("sqlite-vec 加载失败: %s", e)
        return False

    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
            f"USING vec0(embedding float[{dim}])"
        )
        conn.commit()
        return True
    except sqlite3.OperationalError as e:
        logger.warning("vec0 表创建失败: %s", e)
        return False
