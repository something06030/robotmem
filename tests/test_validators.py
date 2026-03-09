"""L1 校验原语测试 — Pydantic 模型 + @validate_args"""

import pytest
from pydantic import ValidationError

from robotmem.validators import (
    EndSessionParams,
    ForgetParams,
    LearnParams,
    RecallParams,
    SavePerceptionParams,
    StartSessionParams,
    UpdateParams,
    parse_params,
    positive_int,
    non_empty_str,
    validate_args,
)


# ── 校验器函数 ──


class TestPositiveInt:

    def test_valid(self):
        assert positive_int(1) == 1
        assert positive_int(999) == 999

    def test_zero(self):
        with pytest.raises(ValueError):
            positive_int(0)

    def test_negative(self):
        with pytest.raises(ValueError):
            positive_int(-1)

    def test_not_int(self):
        with pytest.raises(ValueError):
            positive_int("1")

    def test_none(self):
        with pytest.raises(ValueError):
            positive_int(None)


class TestNonEmptyStr:

    def test_valid(self):
        assert non_empty_str("hello") == "hello"

    def test_strips(self):
        assert non_empty_str("  hello  ") == "hello"

    def test_empty(self):
        with pytest.raises(ValueError):
            non_empty_str("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError):
            non_empty_str("   ")

    def test_not_str(self):
        with pytest.raises(ValueError):
            non_empty_str(123)


# ── @validate_args 装饰器 ──


class TestValidateArgs:

    def test_positional_arg(self):
        @validate_args(memory_id=positive_int)
        def get_memory(conn, memory_id):
            return memory_id

        assert get_memory("conn", 5) == 5
        assert get_memory("conn", -1) is None
        assert get_memory("conn", "bad") is None

    def test_keyword_arg(self):
        @validate_args(memory_id=positive_int)
        def get_memory(conn, memory_id):
            return memory_id

        assert get_memory("conn", memory_id=5) == 5
        assert get_memory("conn", memory_id=0) is None

    def test_multiple_validators(self):
        @validate_args(x=positive_int, name=non_empty_str)
        def do_thing(x, name):
            return f"{name}:{x}"

        assert do_thing(1, "ok") == "ok:1"
        assert do_thing(-1, "ok") is None
        assert do_thing(1, "") is None


# ── Pydantic 模型 ──


class TestLearnParams:

    def test_valid(self):
        p = LearnParams(insight="test insight")
        assert p.insight == "test insight"
        assert p.context == ""

    def test_strips_whitespace(self):
        p = LearnParams(insight="  hello  ")
        assert p.insight == "hello"

    def test_empty_insight(self):
        with pytest.raises(ValidationError):
            LearnParams(insight="")

    def test_too_long(self):
        with pytest.raises(ValidationError):
            LearnParams(insight="x" * 301)

    def test_max_length(self):
        p = LearnParams(insight="x" * 300)
        assert len(p.insight) == 300


class TestRecallParams:

    def test_valid(self):
        p = RecallParams(query="test")
        assert p.n == 5
        assert p.min_confidence == 0.3

    def test_n_bounds(self):
        with pytest.raises(ValidationError):
            RecallParams(query="test", n=0)
        with pytest.raises(ValidationError):
            RecallParams(query="test", n=101)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            RecallParams(query="test", min_confidence=-0.1)
        with pytest.raises(ValidationError):
            RecallParams(query="test", min_confidence=1.1)

    def test_context_filter_default(self):
        p = RecallParams(query="test")
        assert p.context_filter is None

    def test_context_filter_json_string(self):
        p = RecallParams(query="test", context_filter='{"task.success": true}')
        assert p.context_filter == '{"task.success": true}'

    def test_spatial_sort_default(self):
        p = RecallParams(query="test")
        assert p.spatial_sort is None

    def test_spatial_sort_json_string(self):
        p = RecallParams(
            query="test",
            spatial_sort='{"field": "spatial.position", "target": [1, 2, 3]}',
        )
        assert p.spatial_sort == '{"field": "spatial.position", "target": [1, 2, 3]}'

    def test_context_filter_non_string(self):
        with pytest.raises(ValidationError):
            RecallParams(query="test", context_filter=123)

    def test_spatial_sort_non_string(self):
        with pytest.raises(ValidationError):
            RecallParams(query="test", spatial_sort=[1, 2, 3])


class TestSavePerceptionParams:

    def test_valid(self):
        p = SavePerceptionParams(description="robot arm moved")
        assert p.perception_type == "visual"

    def test_min_length(self):
        with pytest.raises(ValidationError):
            SavePerceptionParams(description="hi")

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            SavePerceptionParams(description="robot arm", perception_type="invalid")

    def test_all_types(self):
        for t in ("visual", "tactile", "auditory", "proprioceptive", "procedural"):
            p = SavePerceptionParams(description="test data", perception_type=t)
            assert p.perception_type == t


class TestForgetParams:

    def test_valid(self):
        p = ForgetParams(memory_id=1, reason="outdated")
        assert p.memory_id == 1

    def test_zero_id(self):
        with pytest.raises(ValidationError):
            ForgetParams(memory_id=0, reason="test")

    def test_empty_reason(self):
        with pytest.raises(ValidationError):
            ForgetParams(memory_id=1, reason="")


class TestUpdateParams:

    def test_valid(self):
        p = UpdateParams(memory_id=1, new_content="updated")
        assert p.new_content == "updated"

    def test_strips(self):
        p = UpdateParams(memory_id=1, new_content="  updated  ")
        assert p.new_content == "updated"

    def test_too_long(self):
        with pytest.raises(ValidationError):
            UpdateParams(memory_id=1, new_content="x" * 301)


class TestEndSessionParams:

    def test_valid(self):
        p = EndSessionParams(session_id="abc-123")
        assert p.outcome_score is None

    def test_with_score(self):
        p = EndSessionParams(session_id="abc", outcome_score=0.8)
        assert p.outcome_score == 0.8

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            EndSessionParams(session_id="abc", outcome_score=1.5)
        with pytest.raises(ValidationError):
            EndSessionParams(session_id="abc", outcome_score=-0.1)

    def test_empty_session_id(self):
        with pytest.raises(ValidationError):
            EndSessionParams(session_id="")

    def test_whitespace_session_id(self):
        with pytest.raises(ValidationError):
            EndSessionParams(session_id="   ")


class TestStartSessionParams:

    def test_valid_defaults(self):
        p = StartSessionParams()
        assert p.collection is None
        assert p.context is None

    def test_with_values(self):
        p = StartSessionParams(collection="robots", context="lab test")
        assert p.collection == "robots"
        assert p.context == "lab test"


# ── 空白字符串防御 ──


class TestWhitespaceDefense:
    """验证所有 strip validator 在 strip 后拦截空白输入"""

    def test_learn_whitespace_insight(self):
        with pytest.raises(ValidationError):
            LearnParams(insight="   ")

    def test_recall_whitespace_query(self):
        with pytest.raises(ValidationError):
            RecallParams(query="   ")

    def test_save_perception_whitespace_description(self):
        with pytest.raises(ValidationError):
            SavePerceptionParams(description="     ")

    def test_forget_whitespace_reason(self):
        with pytest.raises(ValidationError):
            ForgetParams(memory_id=1, reason="   ")

    def test_update_whitespace_content(self):
        with pytest.raises(ValidationError):
            UpdateParams(memory_id=1, new_content="   ")


# ── parse_params 工具函数 ──


class TestParseParams:

    def test_success(self):
        result = parse_params(LearnParams, insight="hello")
        assert isinstance(result, LearnParams)
        assert result.insight == "hello"

    def test_failure_returns_dict(self):
        result = parse_params(LearnParams, insight="")
        assert isinstance(result, dict)
        assert "error" in result
        assert "参数校验失败" in result["error"]
