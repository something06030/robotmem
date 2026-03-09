"""测试 ops/tags.py — 标签管理"""

import pytest

from robotmem.ops.memories import insert_memory
from robotmem.ops.tags import _normalize_tag, add_tags, get_tag_stats, get_tags


class TestNormalizeTag:

    def test_english_lowercase(self):
        assert _normalize_tag("Robot Arm") == "robot_arm"

    def test_snake_case(self):
        assert _normalize_tag("pick-and-place") == "pick_and_place"

    def test_chinese_preserved(self):
        assert _normalize_tag("机器人抓取") == "机器人抓取"

    def test_empty(self):
        assert _normalize_tag("") == ""

    def test_whitespace_strip(self):
        assert _normalize_tag("  hello  ") == "hello"

    def test_consecutive_underscores(self):
        assert _normalize_tag("a__b___c") == "a_b_c"


class TestAddTags:

    def test_add_tags_to_memory(self, conn):
        mid = insert_memory(conn, {
            "collection": "test",
            "content": "test memory for tags",
        })
        count = add_tags(conn, mid, ["robotics", "manipulation", "perception"])
        assert count == 3

        tags = get_tags(conn, mid)
        assert tags == ["manipulation", "perception", "robotics"]

    def test_idempotent(self, conn):
        mid = insert_memory(conn, {
            "collection": "test",
            "content": "test idempotent tags",
        })
        add_tags(conn, mid, ["tag1"])
        add_tags(conn, mid, ["tag1"])  # 重复添加
        tags = get_tags(conn, mid)
        assert tags == ["tag1"]

    def test_invalid_memory_id(self, conn):
        count = add_tags(conn, -1, ["tag"])
        assert count == 0

    def test_empty_tags(self, conn):
        count = add_tags(conn, 1, [])
        assert count == 0


class TestGetTagStats:

    def test_stats(self, conn):
        for i in range(3):
            mid = insert_memory(conn, {
                "collection": "test",
                "content": f"memory for tag stats {i}",
                "content_hash": f"stats_hash_{i}",
            })
            if mid:
                add_tags(conn, mid, ["common_tag"])
                if i == 0:
                    add_tags(conn, mid, ["rare_tag"])

        stats = get_tag_stats(conn, collection="test")
        assert stats["common_tag"] == 3
        assert stats["rare_tag"] == 1

    def test_stats_no_collection(self, conn):
        """全局统计"""
        stats = get_tag_stats(conn)
        assert isinstance(stats, dict)
