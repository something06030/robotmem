"""auto_classify.py 测试 — L0 分类引擎"""

import json
import pytest

from robotmem.auto_classify import (
    classify_category,
    classify_tags,
    estimate_confidence,
    extract_scope,
    normalize_scope_files,
    build_context_json,
)


# ── classify_category ──


class TestClassifyCategory:
    """首命中分类"""

    def test_constraint_cn(self):
        assert classify_category("必须使用参数化查询") == "constraint"

    def test_constraint_en(self):
        assert classify_category("must always validate input") == "constraint"

    def test_constraint_never(self):
        assert classify_category("never use eval()") == "constraint"

    def test_preference_cn(self):
        assert classify_category("优先使用 pytest") == "preference"

    def test_preference_en(self):
        assert classify_category("prefer using asyncio over threading") == "preference"

    def test_worldview_cn(self):
        assert classify_category("更好的做法是用类型注解") == "worldview"

    def test_worldview_en(self):
        assert classify_category("the best approach is to use dependency injection") == "worldview"

    def test_tradeoff(self):
        assert classify_category("权衡：内存 vs 速度") == "tradeoff"

    def test_root_cause(self):
        assert classify_category("根因是缺少错误处理") == "root_cause"

    def test_decision(self):
        assert classify_category("决定采用 SQLite 替代 PostgreSQL") == "decision"

    def test_revert(self):
        assert classify_category("回滚了上次的修改") == "revert"

    def test_pattern(self):
        assert classify_category("规律：每次部署都出同样的错") == "pattern"

    def test_architecture(self):
        assert classify_category("架构：三层分离设计") == "architecture"

    def test_config(self):
        assert classify_category("配置端口为 8080") == "config"

    def test_postmortem(self):
        assert classify_category("教训：没有做充分测试") == "postmortem"

    def test_gotcha(self):
        assert classify_category("踩坑：asyncio.run 不能嵌套") == "gotcha"

    def test_self_defect(self):
        assert classify_category("AI缺陷：训练偏好覆盖指令") == "self_defect"

    def test_observation_debug(self):
        assert classify_category("发现了一个 timeout 异常") == "observation_debug"

    def test_observation_code(self):
        assert classify_category("发现 main.py 中有未使用的变量") == "observation_code"

    def test_observation_generic(self):
        assert classify_category("观察到系统负载很高") == "observation"

    def test_fallback_code(self):
        """无匹配时兜底到 code"""
        assert classify_category("这是一段普通文本") == "code"

    def test_empty_string(self):
        assert classify_category("") == "code"

    def test_priority_constraint_over_decision(self):
        """constraint 优先级高于 decision"""
        assert classify_category("必须决定使用哪个框架") == "constraint"


# ── classify_tags ──


class TestClassifyTags:
    """多标签分类"""

    def test_single_tag(self):
        tags = classify_tags("必须检查输入")
        assert "constraint" in tags

    def test_multiple_tags(self):
        """同时命中多个标签"""
        tags = classify_tags("教训：因为没有权衡利弊就决定了")
        assert len(tags) >= 2

    def test_fallback_code(self):
        tags = classify_tags("hello world")
        assert tags == ["code"]

    def test_scenario_tags_from_context(self):
        """context_json 中的 scenario_tags"""
        ctx = json.dumps({"scenario_tags": ["constraint"]})
        tags = classify_tags("普通文本", context_json=ctx)
        assert "constraint" in tags

    def test_invalid_scenario_tag_ignored(self):
        """无效的 scenario_tag 被忽略"""
        ctx = json.dumps({"scenario_tags": ["invalid_nonexistent_tag_xyz"]})
        tags = classify_tags("普通文本", context_json=ctx)
        assert "invalid_nonexistent_tag_xyz" not in tags

    def test_invalid_context_json(self):
        """非法 JSON 不崩溃"""
        tags = classify_tags("必须做", context_json="{broken json")
        assert "constraint" in tags

    def test_context_json_not_dict(self):
        """context 是 list 不崩溃"""
        tags = classify_tags("普通文本", context_json="[1,2,3]")
        assert tags == ["code"]

    def test_empty_text(self):
        tags = classify_tags("")
        assert tags == ["code"]


# ── estimate_confidence ──


class TestEstimateConfidence:
    """置信度评估"""

    def test_base_confidence(self):
        """无信号时基准 0.80"""
        assert estimate_confidence("hello") == 0.80

    def test_file_path_signal(self):
        """有文件路径 +0.05"""
        conf = estimate_confidence("修改了 main.py 的逻辑")
        assert conf >= 0.85

    def test_backtick_signal(self):
        """有反引号 +0.05"""
        conf = estimate_confidence("调用了 `foo()` 函数")
        assert conf >= 0.85

    def test_causal_signal(self):
        """有因果词 +0.05"""
        conf = estimate_confidence("因为没有校验导致崩溃")
        assert conf >= 0.85

    def test_context_signal(self):
        """有足够长的 context +0.05"""
        conf = estimate_confidence("test", context="x" * 30)
        assert conf >= 0.85

    def test_all_signals_capped(self):
        """四个信号全命中，cap 到 0.95"""
        text = "因为修改了 main.py 中 `foo()` 的逻辑"
        conf = estimate_confidence(text, context="x" * 30)
        assert conf == pytest.approx(0.95)

    def test_empty_context_no_bonus(self):
        conf = estimate_confidence("hello", context="")
        assert conf == 0.80

    def test_short_context_no_bonus(self):
        conf = estimate_confidence("hello", context="short")
        assert conf == 0.80


# ── extract_scope ──


class TestExtractScope:
    """scope 提取"""

    def test_file_paths(self):
        scope = extract_scope("修改了 src/robotmem/db.py 和 tests/test_db.py")
        assert "src/robotmem/db.py" in scope["scope_files"]
        assert "tests/test_db.py" in scope["scope_files"]

    def test_backtick_entities(self):
        scope = extract_scope("调用 `safe_db_write` 和 `mcp_error_boundary`")
        assert "safe_db_write" in scope["scope_entities"]
        assert "mcp_error_boundary" in scope["scope_entities"]

    def test_pascal_case_entities(self):
        scope = extract_scope("使用 CogDatabase 和 ServiceCooldown")
        assert "CogDatabase" in scope["scope_entities"]
        assert "ServiceCooldown" in scope["scope_entities"]

    def test_modules_from_paths(self):
        scope = extract_scope("修改了 ops/memories.py")
        assert "ops" in scope["scope_modules"]

    def test_no_matches(self):
        scope = extract_scope("这是普通文本")
        assert scope["scope_files"] == []
        assert scope["scope_entities"] == []
        assert scope["scope_modules"] == []

    def test_dedup_files(self):
        """重复文件路径去重"""
        scope = extract_scope("db.py 和 db.py 出了问题")
        file_count = scope["scope_files"].count("db.py")
        assert file_count <= 1

    def test_src_module_excluded(self):
        """src/lib/tests 不算 module"""
        scope = extract_scope("修改了 src/main.py")
        assert "src" not in scope["scope_modules"]


# ── normalize_scope_files ──


class TestNormalizeScopeFiles:
    """路径归一化"""

    def test_absolute_to_relative(self):
        result = normalize_scope_files(
            ["/tmp/project/src/main.py"],
            project_root="/tmp/project",
        )
        assert result == ["src/main.py"]

    def test_already_relative(self):
        result = normalize_scope_files(["src/main.py"])
        assert result == ["src/main.py"]

    def test_dedup(self):
        result = normalize_scope_files(
            ["/project/a.py", "a.py"],
            project_root="/project",
        )
        assert result == ["a.py"]

    def test_empty_list(self):
        assert normalize_scope_files([]) == []

    def test_empty_strings_filtered(self):
        result = normalize_scope_files(["", "  ", "a.py"])
        assert result == ["a.py"]

    def test_none_root(self):
        result = normalize_scope_files(["/abs/path.py"])
        assert result == ["abs/path.py"]


# ── build_context_json ──


class TestBuildContextJson:
    """context_json 构建"""

    def test_empty_context(self):
        result = json.loads(build_context_json("test", ""))
        assert result == {"source": "learn_tool"}

    def test_valid_json_context(self):
        ctx = json.dumps({"robot": "arm1", "task": "pick"})
        result = json.loads(build_context_json("test", ctx))
        assert result["source"] == "learn_tool"
        assert result["robot"] == "arm1"

    def test_plain_text_context(self):
        result = json.loads(build_context_json("test", "some notes"))
        assert result["source"] == "learn_tool"
        assert result["user_context"] == "some notes"

    def test_invalid_json_context(self):
        result = json.loads(build_context_json("test", "{broken"))
        assert result["source"] == "learn_tool"
        assert result["user_context"] == "{broken"

    def test_list_context(self):
        """JSON list 不是 dict，存为 user_context"""
        result = json.loads(build_context_json("test", "[1,2,3]"))
        assert result["user_context"] == "[1,2,3]"

    def test_whitespace_only_context(self):
        result = json.loads(build_context_json("test", "   "))
        assert result == {"source": "learn_tool"}
