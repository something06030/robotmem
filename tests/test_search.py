"""测试 search.py — recall + RRF + source weight + context 解析"""

import json
import pytest

from robotmem.search import (
    RecallResult, _apply_source_weight, _compute_spatial_distance,
    _match_context_filter, _resolve_dotpath, _MISSING,
    extract_context_fields, rrf_merge,
)


class TestRRFMerge:
    """Reciprocal Rank Fusion 算法测试"""

    def test_two_lists(self):
        """双路合并"""
        bm25 = [{"id": 1}, {"id": 2}, {"id": 3}]
        vec = [{"id": 2}, {"id": 4}, {"id": 1}]
        merged = rrf_merge(bm25, vec, k=60)

        ids = [m["id"] for m in merged]
        # id=2 和 id=1 在两路都出现，分数应该最高
        assert ids[0] in (1, 2)
        assert ids[1] in (1, 2)

        # 所有结果都有 _rrf_score
        assert all("_rrf_score" in m for m in merged)

    def test_single_list(self):
        """单路退化"""
        results = [{"id": 10}, {"id": 20}]
        merged = rrf_merge(results, k=60)
        assert len(merged) == 2
        assert merged[0]["_rrf_score"] > merged[1]["_rrf_score"]

    def test_empty_lists(self):
        """全空返回空"""
        merged = rrf_merge([], [], k=60)
        assert merged == []

    def test_no_id_pollution(self):
        """浅拷贝不污染上游"""
        original = [{"id": 1, "content": "abc"}]
        merged = rrf_merge(original, k=60)
        # 原始列表中不应有 _rrf_score
        assert "_rrf_score" not in original[0]

    def test_missing_id_skipped(self):
        """缺少 id 的 item 被跳过"""
        results = [{"id": 1}, {"no_id": True}, {"id": 2}]
        merged = rrf_merge(results, k=60)
        assert len(merged) == 2


class TestApplySourceWeight:
    """sim/real 加权测试"""

    def test_real_gets_boost(self):
        """real 数据 ×1.5"""
        memories = [
            {
                "id": 1,
                "_rrf_score": 1.0,
                "context": json.dumps({"env": {"sim_or_real": "real"}}),
            },
            {
                "id": 2,
                "_rrf_score": 1.0,
                "context": json.dumps({"env": {"sim_or_real": "sim"}}),
            },
        ]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == pytest.approx(1.5)
        assert memories[1]["_rrf_score"] == pytest.approx(1.0)  # sim 不加权

    def test_no_context(self):
        """无 context 不崩溃"""
        memories = [{"id": 1, "_rrf_score": 1.0, "context": None}]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == 1.0

    def test_invalid_json(self):
        """非法 JSON 不崩溃"""
        memories = [{"id": 1, "_rrf_score": 1.0, "context": "not json"}]
        _apply_source_weight(memories)
        assert memories[0]["_rrf_score"] == 1.0


class TestExtractContextFields:
    """context JSON 解析 — 提取 params/spatial/robot 便捷字段"""

    def test_full_partition(self):
        """全分区 — 四个字段都提取"""
        mem = {"context": json.dumps({
            "params": {"grip_force": {"value": 12.5}},
            "spatial": {"position": [1, 2, 3]},
            "robot": {"model": "UR5"},
            "task": {"success": True, "steps": 38},
        })}
        extract_context_fields(mem)
        assert mem["params"] == {"grip_force": {"value": 12.5}}
        assert mem["spatial"] == {"position": [1, 2, 3]}
        assert mem["robot"] == {"model": "UR5"}
        assert mem["task"] == {"success": True, "steps": 38}

    def test_params_only(self):
        """只有 params — 只提取 params"""
        mem = {"context": json.dumps({"params": {"grip_force": {"value": 12.5}}})}
        extract_context_fields(mem)
        assert mem["params"] == {"grip_force": {"value": 12.5}}
        assert "spatial" not in mem
        assert "robot" not in mem

    def test_task_extraction(self):
        """task 字段 — 提取 success/steps/total_reward"""
        mem = {"context": json.dumps({
            "task": {"name": "push_to_target", "success": True, "steps": 38, "total_reward": -12.0},
        })}
        extract_context_fields(mem)
        assert mem["task"]["success"] is True
        assert mem["task"]["steps"] == 38
        assert "params" not in mem

    def test_empty_context(self):
        """空 context — 不报错，无新字段"""
        for ctx in ("", None):
            mem = {"context": ctx}
            extract_context_fields(mem)
            assert "params" not in mem
            assert "spatial" not in mem
            assert "robot" not in mem

    def test_non_json(self):
        """非 JSON — 不报错，无新字段"""
        mem = {"context": "some plain text"}
        extract_context_fields(mem)
        assert "params" not in mem

    def test_nested_complete(self):
        """嵌套完整 — value/unit/type 完整保留"""
        mem = {"context": json.dumps({
            "params": {"force": {"value": 12.5, "unit": "N", "type": "scalar"}},
        })}
        extract_context_fields(mem)
        assert mem["params"]["force"] == {"value": 12.5, "unit": "N", "type": "scalar"}


class TestResolveDotpath:
    """点分路径解析"""

    def test_flat(self):
        assert _resolve_dotpath({"a": 1}, "a") == 1

    def test_nested(self):
        assert _resolve_dotpath({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_missing(self):
        assert _resolve_dotpath({"a": 1}, "b") is _MISSING

    def test_missing_nested(self):
        assert _resolve_dotpath({"a": {"b": 1}}, "a.c") is _MISSING

    def test_non_dict_intermediate(self):
        assert _resolve_dotpath({"a": 42}, "a.b") is _MISSING


class TestMatchContextFilter:
    """结构化过滤（#17 P1）"""

    def test_equality_true(self):
        mem = {"task": {"success": True}}
        assert _match_context_filter(mem, {"task.success": True}) is True

    def test_equality_false(self):
        mem = {"task": {"success": False}}
        assert _match_context_filter(mem, {"task.success": True}) is False

    def test_missing_field(self):
        mem = {"params": {}}
        assert _match_context_filter(mem, {"task.success": True}) is False

    def test_range_lt(self):
        mem = {"params": {"final_distance": {"value": 0.03}}}
        assert _match_context_filter(mem, {"params.final_distance.value": {"$lt": 0.05}}) is True
        assert _match_context_filter(mem, {"params.final_distance.value": {"$lt": 0.01}}) is False

    def test_range_gte(self):
        mem = {"task": {"steps": 38}}
        assert _match_context_filter(mem, {"task.steps": {"$gte": 38}}) is True
        assert _match_context_filter(mem, {"task.steps": {"$gte": 39}}) is False

    def test_combined(self):
        mem = {"task": {"success": True}, "params": {"final_distance": {"value": 0.03}}}
        assert _match_context_filter(mem, {
            "task.success": True,
            "params.final_distance.value": {"$lt": 0.05},
        }) is True

    def test_combined_partial_fail(self):
        mem = {"task": {"success": False}, "params": {"final_distance": {"value": 0.03}}}
        assert _match_context_filter(mem, {
            "task.success": True,
            "params.final_distance.value": {"$lt": 0.05},
        }) is False

    def test_ne_operator(self):
        mem = {"task": {"name": "push"}}
        assert _match_context_filter(mem, {"task.name": {"$ne": "pull"}}) is True
        assert _match_context_filter(mem, {"task.name": {"$ne": "push"}}) is False

    def test_empty_filter(self):
        """空过滤条件 — 全部匹配"""
        assert _match_context_filter({"any": "thing"}, {}) is True

    def test_type_mismatch_no_crash(self):
        """类型不匹配 — 返回 False，不崩溃"""
        mem = {"task": {"steps": "not_a_number"}}
        assert _match_context_filter(mem, {"task.steps": {"$lt": 10}}) is False


class TestComputeSpatialDistance:
    """欧氏距离（#17 P2）"""

    def test_same_point(self):
        mem = {"spatial": {"position": [1.0, 2.0, 3.0]}}
        assert _compute_spatial_distance(mem, "spatial.position", [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_known_distance(self):
        """3-4-5 三角"""
        mem = {"spatial": {"position": [0.0, 0.0, 0.0]}}
        assert _compute_spatial_distance(mem, "spatial.position", [3.0, 4.0, 0.0]) == pytest.approx(5.0)

    def test_missing_field(self):
        mem = {"params": {}}
        assert _compute_spatial_distance(mem, "spatial.position", [0, 0, 0]) == float("inf")

    def test_non_list(self):
        mem = {"spatial": {"position": "not a list"}}
        assert _compute_spatial_distance(mem, "spatial.position", [0, 0, 0]) == float("inf")

    def test_nested_field(self):
        mem = {"spatial": {"object_position": [1.3, 0.7, 0.42]}}
        dist = _compute_spatial_distance(mem, "spatial.object_position", [1.3, 0.7, 0.42])
        assert dist == pytest.approx(0.0)

    def test_dimension_mismatch(self):
        """维度不匹配 — 返回 inf"""
        mem = {"spatial": {"position": [1.0, 2.0]}}
        assert _compute_spatial_distance(mem, "spatial.position", [1.0, 2.0, 3.0]) == float("inf")


class TestRecallResult:
    """RecallResult 数据结构测试"""

    def test_defaults(self):
        r = RecallResult()
        assert r.memories == []
        assert r.total == 0
        assert r.mode == "bm25_only"
        assert r.query_ms == 0.0
