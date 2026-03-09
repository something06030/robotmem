"""dedup.py 扩展测试 — 覆盖 cleanup_exact_duplicates + Layer 3 + check_session_cosine_dup"""

import sqlite3
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from robotmem.dedup import (
    check_duplicate,
    check_session_cosine_dup,
    cleanup_exact_duplicates,
    DedupResult,
    COSINE_DUP_THRESHOLD,
)
from robotmem.schema import initialize_schema


# ── cleanup_exact_duplicates ──


class TestCleanupExactDuplicates:
    """一次性清理完全重复的 active memories"""

    def _make_db_with_dups(self):
        """创建含重复记忆的 mock db"""
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)

        # 插入 3 条完全重复的记忆（不同 confidence）
        for conf in [0.7, 0.9, 0.5]:
            conn.execute("""
                INSERT INTO memories (collection, type, content, category,
                                     confidence, decay_rate, source, scope, status,
                                     scope_files, scope_entities)
                VALUES ('test', 'fact', '重复的记忆内容', 'observation', ?, 0.01,
                        'tool', 'project', 'active', '[]', '[]')
            """, (conf,))

        # 插入 1 条不重复的记忆
        conn.execute("""
            INSERT INTO memories (collection, type, content, category,
                                 confidence, decay_rate, source, scope, status,
                                 scope_files, scope_entities)
            VALUES ('test', 'fact', '唯一的记忆', 'observation', 0.8, 0.01,
                    'tool', 'project', 'active', '[]', '[]')
        """)
        conn.commit()

        db = MagicMock()
        db.conn = conn
        db.supersede_memory = MagicMock()
        return db

    def test_dry_run(self):
        """dry_run=True → 只预览不修改"""
        db = self._make_db_with_dups()
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=True)
        assert len(ops) == 2  # 3 条重复，保留 1 条，supersede 2 条
        for op in ops:
            assert "old_id" in op
            assert "keep_id" in op
            assert "assertion_preview" in op
        # supersede_memory 不被调用
        db.supersede_memory.assert_not_called()

    def test_execute(self):
        """dry_run=False → 实际执行 supersede"""
        db = self._make_db_with_dups()
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=False)
        assert len(ops) == 2
        assert db.supersede_memory.call_count == 2

    def test_keep_highest_confidence(self):
        """保留 confidence 最高的记忆"""
        db = self._make_db_with_dups()
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=True)
        # confidence=0.9 的应该被保留
        keep_ids = {op["keep_id"] for op in ops}
        assert len(keep_ids) == 1  # 所有 ops 的 keep_id 相同
        # 验证 keep_id 对应的 confidence 是 0.9
        keep_id = list(keep_ids)[0]
        row = db.conn.execute(
            "SELECT confidence FROM memories WHERE id=?", (keep_id,)
        ).fetchone()
        assert row[0] == 0.9

    def test_no_duplicates(self):
        """无重复 → 空列表"""
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        conn.execute("""
            INSERT INTO memories (collection, type, content, category,
                                 confidence, decay_rate, source, scope, status,
                                 scope_files, scope_entities)
            VALUES ('test', 'fact', '唯一内容', 'observation', 0.9, 0.01,
                    'tool', 'project', 'active', '[]', '[]')
        """)
        conn.commit()
        db = MagicMock()
        db.conn = conn
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=True)
        assert ops == []

    def test_no_collection_filter(self):
        """collection=None → 全局搜索"""
        db = self._make_db_with_dups()
        ops = cleanup_exact_duplicates(db, collection=None, dry_run=True)
        assert len(ops) >= 2

    def test_max_ops_limit(self):
        """超过 200 对上限时停止"""
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        # 插入 250 条相同内容
        for _ in range(250):
            conn.execute("""
                INSERT INTO memories (collection, type, content, category,
                                     confidence, decay_rate, source, scope, status,
                                     scope_files, scope_entities)
                VALUES ('test', 'fact', '大量重复', 'observation', 0.9, 0.01,
                        'tool', 'project', 'active', '[]', '[]')
            """)
        conn.commit()
        db = MagicMock()
        db.conn = conn
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=True)
        assert len(ops) <= 200

    def test_supersede_failure(self):
        """supersede 失败不崩溃"""
        db = self._make_db_with_dups()
        db.supersede_memory.side_effect = Exception("DB error")
        # 不崩溃
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=False)
        assert len(ops) == 2
        # 验证 supersede 确实被调用了（只是失败了）
        assert db.supersede_memory.call_count == 2

    def test_assertion_preview_truncated(self):
        """assertion_preview 最多 80 字"""
        conn = sqlite3.connect(":memory:")
        initialize_schema(conn)
        long_content = "A" * 200
        for _ in range(2):
            conn.execute("""
                INSERT INTO memories (collection, type, content, category,
                                     confidence, decay_rate, source, scope, status,
                                     scope_files, scope_entities)
                VALUES ('test', 'fact', ?, 'observation', 0.9, 0.01,
                        'tool', 'project', 'active', '[]', '[]')
            """, (long_content,))
        conn.commit()
        db = MagicMock()
        db.conn = conn
        ops = cleanup_exact_duplicates(db, collection="test", dry_run=True)
        assert len(ops) == 1
        assert len(ops[0]["assertion_preview"]) <= 80


# ── check_duplicate Layer 3 ──


class TestCheckDuplicateLayer3:
    """check_duplicate 向量余弦层"""

    def _make_db_cog(self):
        db = MagicMock()
        db.memory_exists.return_value = False
        db.fts_search_memories.return_value = []
        return db

    def test_cosine_dup_detected(self):
        """Layer 3: 余弦相似度超阈值"""
        db = self._make_db_cog()
        db.vec_search_memories.return_value = [
            {"id": 10, "assertion": "similar text", "distance": 0.1},  # cosine_sim=0.9
        ]

        embedder = MagicMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # asyncio.run 需要不在事件循环中
        result = check_duplicate("test", "col", "sess", db, embedder=embedder)
        assert result.is_dup is True
        assert result.method == "cosine"

    def test_cosine_similar_not_dup(self):
        """Layer 3: 余弦相似度中等 → 记入 similar_facts"""
        db = self._make_db_cog()
        db.vec_search_memories.return_value = [
            {"id": 10, "assertion": "somewhat similar", "distance": 0.5},  # cosine_sim=0.5
        ]

        embedder = MagicMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])

        result = check_duplicate("test", "col", "sess", db, embedder=embedder)
        assert result.is_dup is False
        assert len(result.similar_facts) >= 1

    def test_embedder_not_available(self):
        """embedder.available=False → 跳过 Layer 3"""
        db = self._make_db_cog()
        embedder = MagicMock()
        embedder.available = False
        result = check_duplicate("test", "col", "sess", db, embedder=embedder)
        assert result.is_dup is False

    def test_embed_exception(self):
        """embedding 异常 → 降级"""
        db = self._make_db_cog()
        embedder = MagicMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(side_effect=Exception("embed fail"))

        result = check_duplicate("test", "col", "sess", db, embedder=embedder)
        assert result.is_dup is False

    def test_exclude_id_in_layer3(self):
        """Layer 3 也应排除指定 ID"""
        db = self._make_db_cog()
        db.vec_search_memories.return_value = [
            {"id": 42, "assertion": "same", "distance": 0.05},  # cosine_sim=0.95
        ]

        embedder = MagicMock()
        embedder.available = True
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])

        result = check_duplicate("test", "col", "sess", db, embedder=embedder, exclude_id=42)
        assert result.is_dup is False


# ── check_session_cosine_dup extended ──


class TestCheckSessionCosineDupExtended:
    """check_session_cosine_dup 完整覆盖"""

    def test_cosine_dup_found(self):
        """同 session 内高余弦相似度"""
        db = MagicMock()
        db._vec_loaded = True
        db.vec_search_memories.return_value = [
            {"id": 5, "session_id": "s1", "assertion": "same meaning",
             "distance": 0.1},  # cosine_sim=0.9
        ]

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])

        result = check_session_cosine_dup("test", "s1", "col", db, embedder)
        assert result.is_dup is True
        assert result.method == "session_cosine"

    def test_different_session_not_dup(self):
        """不同 session 的记忆 → 不算重复"""
        db = MagicMock()
        db._vec_loaded = True
        db.vec_search_memories.return_value = [
            {"id": 5, "session_id": "other_session", "assertion": "same meaning",
             "distance": 0.1},
        ]

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])

        result = check_session_cosine_dup("test", "s1", "col", db, embedder)
        assert result.is_dup is False

    def test_low_similarity_not_dup(self):
        """同 session 但余弦相似度低"""
        db = MagicMock()
        db._vec_loaded = True
        db.vec_search_memories.return_value = [
            {"id": 5, "session_id": "s1", "assertion": "different meaning",
             "distance": 0.5},  # cosine_sim=0.5
        ]

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])

        result = check_session_cosine_dup("test", "s1", "col", db, embedder)
        assert result.is_dup is False

    def test_embed_returns_none(self):
        """embedding 返回 None/空"""
        db = MagicMock()
        db._vec_loaded = True

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(return_value=None)

        result = check_session_cosine_dup("test", "s1", "col", db, embedder)
        assert result.is_dup is False

    def test_vec_search_exception(self):
        """vec_search 异常 → 降级"""
        db = MagicMock()
        db._vec_loaded = True
        db.vec_search_memories.side_effect = Exception("vec error")

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(return_value=[0.1, 0.2])

        result = check_session_cosine_dup("test", "s1", "col", db, embedder)
        assert result.is_dup is False

    def test_embed_exception(self):
        """embedding 异常 → 降级"""
        db = MagicMock()
        db._vec_loaded = True

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(side_effect=Exception("embed fail"))

        result = check_session_cosine_dup("test", "s1", "col", db, embedder)
        assert result.is_dup is False
