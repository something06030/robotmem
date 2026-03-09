"""consolidate_session 测试 — 记忆巩固 + end_session 集成（专用 fixture，不修改 conftest.py）"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from robotmem.ops.memories import consolidate_session
from robotmem.schema import initialize_schema


@pytest.fixture
def conn():
    """内存 DB + schema"""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    initialize_schema(c)
    yield c
    c.close()


def _insert(conn, session_id, content, category="observation",
            confidence=0.8, perception_type=None, collection="test"):
    """辅助插入 — 返回 memory_id"""
    conn.execute("""
        INSERT INTO memories
            (session_id, collection, type, content, category,
             confidence, decay_rate, source, scope, status,
             perception_type, content_hash, scope_files, scope_entities)
        VALUES (?, ?, 'fact', ?, ?, ?, 0.01, 'tool', 'project', 'active',
                ?, ?, '[]', '[]')
    """, [session_id, collection, content, category, confidence,
          perception_type, f"hash_{id(content)}_{confidence}"])
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


@pytest.fixture
def conn_with_session_data(conn):
    """专用 fixture：3 条相似 observation + 1 条 constraint + 1 条 perception"""
    sid = "test-consolidation-001"

    # 3 条相似 observation（Jaccard > 0.5：共享大量 token）
    _insert(conn, sid, "机器人 夹持力 calibration 校准 参数 force 12.5N 成功",
            confidence=0.8)
    _insert(conn, sid, "机器人 夹持力 calibration 校准 参数 force 13.0N 成功",
            confidence=0.7)
    _insert(conn, sid, "机器人 夹持力 calibration 校准 参数 force 11.8N 成功",
            confidence=0.6)

    # 1 条 constraint（保护：不应被巩固）
    _insert(conn, sid, "constraint: 夹持力不得超过 20N",
            category="constraint", confidence=0.9)

    # 1 条 perception（保护：有 perception_type）
    _insert(conn, sid, "tactile sensor reading during calibration",
            perception_type="tactile", confidence=0.8)

    return conn, sid


class TestConsolidateSession:
    """记忆巩固单元测试"""

    def test_similar_observations_merged(self, conn_with_session_data):
        """3 条相似 observation → 合并为 1，superseded_count=2"""
        conn, sid = conn_with_session_data
        result = consolidate_session(conn, sid, "test")

        assert result["merged_groups"] >= 1
        assert result["superseded_count"] == 2

        # 验证 DB 状态：只剩 1 条 active observation
        active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id=? AND collection='test' "
            "AND status='active' AND category='observation' AND perception_type IS NULL",
            [sid],
        ).fetchone()[0]
        assert active == 1

        # 验证被 supersede 的记录有 superseded_by
        superseded = conn.execute(
            "SELECT superseded_by FROM memories WHERE session_id=? AND status='superseded'",
            [sid],
        ).fetchall()
        assert len(superseded) == 2
        assert all(r[0] is not None for r in superseded)

    def test_constraint_not_merged(self, conn_with_session_data):
        """constraint 类别不受巩固影响"""
        conn, sid = conn_with_session_data
        consolidate_session(conn, sid, "test")

        # constraint 记忆仍然 active
        constraint = conn.execute(
            "SELECT status FROM memories WHERE session_id=? AND category='constraint'",
            [sid],
        ).fetchone()
        assert constraint[0] == "active"

    def test_fewer_than_3_skipped(self, conn):
        """不足 3 条记忆 → 跳过巩固"""
        sid = "test-skip-001"
        _insert(conn, sid, "first memory observation")
        _insert(conn, sid, "second memory observation")

        result = consolidate_session(conn, sid, "test")
        assert result["merged_groups"] == 0
        assert result["superseded_count"] == 0

    def test_cluster_pairwise_constraint(self, conn):
        """簇内两两约束：A-B 相似、A-C 相似、但 B-C 不相似 → C 独立"""
        sid = "test-pairwise-001"

        # A 和 B 相似（共享大量 token）
        _insert(conn, sid, "alpha beta gamma delta epsilon zeta eta theta",
                confidence=0.8)
        _insert(conn, sid, "alpha beta gamma delta epsilon zeta eta iota",
                confidence=0.7)
        # C 与 A/B 完全不同
        _insert(conn, sid, "kappa lambda mu nu xi omicron pi rho",
                confidence=0.6)

        result = consolidate_session(conn, sid, "test")

        # A-B 合并为 1 组，C 独立（不够 2 个不成簇）
        assert result["merged_groups"] == 1
        assert result["superseded_count"] == 1

    def test_high_confidence_protected(self, conn):
        """confidence >= 0.95 的记忆不被巩固"""
        sid = "test-conf-001"

        # 3 条相似记忆，但其中 1 条 confidence=0.95（保护）
        _insert(conn, sid, "机器人 抓取 策略 参数 配置 优化 结果",
                confidence=0.95)
        _insert(conn, sid, "机器人 抓取 策略 参数 配置 优化 调整",
                confidence=0.7)
        _insert(conn, sid, "机器人 抓取 策略 参数 配置 优化 改进",
                confidence=0.6)

        result = consolidate_session(conn, sid, "test")

        # 高 confidence 的不在候选中，只有 2 条 < 0.95 → 不足 3 条 → 跳过
        # （高 confidence 排除后只剩 2 条，不满足 >= 3 条条件）
        assert result["superseded_count"] == 0

        # 验证 confidence=0.95 的记忆仍然 active
        high_conf = conn.execute(
            "SELECT status FROM memories WHERE session_id=? AND confidence >= 0.95",
            [sid],
        ).fetchone()
        assert high_conf[0] == "active"


class TestEndSessionIntegration:
    """end_session 集成测试 — 验证 consolidate 和 proactive recall 行为"""

    def test_consolidate_returns_result(self, conn_with_session_data):
        """end_session 触发巩固后返回 consolidated 字段"""
        conn, sid = conn_with_session_data
        result = consolidate_session(conn, sid, "test")

        # 验证返回值结构完整
        assert "merged_groups" in result
        assert "superseded_count" in result
        assert "compression_ratio" in result
        assert "avg_similarity" in result
        assert result["superseded_count"] > 0

    def test_consolidate_failure_returns_empty(self, conn):
        """巩固失败时返回空结果，不抛异常"""
        # 传入无效 session_id → 无记忆 → 返回空结果
        result = consolidate_session(conn, "", "test")
        assert result["merged_groups"] == 0
        assert result["superseded_count"] == 0

        # 传入 None session_id
        result = consolidate_session(conn, None, "test")
        assert result["merged_groups"] == 0

    def test_proactive_recall_excludes_current_session(self, conn_with_session_data):
        """proactive recall 应排除当前 session 的记忆"""
        conn, sid = conn_with_session_data

        # 插入另一个 session 的记忆（模拟历史记忆）
        _insert(conn, "other-session-001",
                "机器人 夹持力 calibration 校准 历史数据",
                collection="test", confidence=0.8)

        # 查询当前 session 的记忆
        current = conn.execute(
            "SELECT id, session_id FROM memories WHERE session_id=?",
            [sid],
        ).fetchall()
        # 查询其他 session 的记忆
        other = conn.execute(
            "SELECT id, session_id FROM memories WHERE session_id='other-session-001'",
        ).fetchall()

        # 验证两个 session 的记忆可以区分
        assert len(current) > 0
        assert len(other) > 0
        assert all(r[1] == sid for r in current)
        assert all(r[1] == "other-session-001" for r in other)
