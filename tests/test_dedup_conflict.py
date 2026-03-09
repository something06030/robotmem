"""dedup.py + conflict.py 测试 — 去重管道 + 冲突检测"""

from unittest.mock import MagicMock
import pytest

from robotmem.dedup import (
    DedupResult,
    jaccard_similarity,
    check_duplicate,
    check_session_cosine_dup,
    JACCARD_DUP_THRESHOLD,
    JACCARD_SIMILAR_THRESHOLD,
)
from robotmem.conflict import (
    ConflictResult,
    detect_conflicts,
    _has_negation,
)


# ── jaccard_similarity ──


class TestJaccardSimilarity:
    """token 级 Jaccard 相似度"""

    def test_identical(self):
        assert jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert jaccard_similarity("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        sim = jaccard_similarity("hello world test", "hello world foo")
        assert 0.0 < sim < 1.0

    def test_stopwords_filtered(self):
        """停用词过滤后，两个文本等价 → 相似度 1.0"""
        sim = jaccard_similarity("the hello world", "a hello world")
        # "the"/"a" 是停用词，过滤后都剩 {"hello", "world"}
        assert sim == pytest.approx(1.0)

    def test_empty_after_stopwords(self):
        """全是停用词 → 0.0"""
        assert jaccard_similarity("the a an", "is are was") == pytest.approx(0.0)

    def test_empty_string(self):
        assert jaccard_similarity("", "hello") == pytest.approx(0.0)

    def test_both_empty(self):
        assert jaccard_similarity("", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert jaccard_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_chinese_tokens(self):
        """中文分词（空格分词）"""
        sim = jaccard_similarity("夹持力 15N 光滑", "夹持力 20N 光滑")
        assert sim > 0.3


# ── DedupResult ──


class TestDedupResult:
    """去重结果数据类"""

    def test_defaults(self):
        r = DedupResult(is_dup=False)
        assert r.similar_facts == []
        assert r.method == "none"
        assert r.similarity == 0.0

    def test_exact_dup(self):
        r = DedupResult(is_dup=True, method="exact", similarity=1.0)
        assert r.is_dup is True
        assert r.method == "exact"


# ── check_duplicate ──


class TestCheckDuplicate:
    """三层去重管道"""

    def _make_db_cog(self, *, exists=False, fts_results=None):
        """构造 mock db_cog"""
        db = MagicMock()
        db.memory_exists.return_value = exists
        db.fts_search_memories.return_value = fts_results or []
        db.vec_search_memories.return_value = []
        return db

    def test_exact_match(self):
        db = self._make_db_cog(exists=True)
        result = check_duplicate("test", "col", "sess", db)
        assert result.is_dup is True
        assert result.method == "exact"

    def test_no_duplicate(self):
        db = self._make_db_cog(exists=False, fts_results=[])
        result = check_duplicate("unique content", "col", "sess", db)
        assert result.is_dup is False
        assert result.method == "none"

    def test_jaccard_duplicate(self):
        """Jaccard 相似度超过阈值"""
        db = self._make_db_cog(
            exists=False,
            fts_results=[
                {"id": 1, "assertion": "hello world test content"},
            ],
        )
        # 4/5 token 重叠 → Jaccard=0.80 > 0.70 阈值
        result = check_duplicate("hello world test content extra", "col", "sess", db)
        assert result.is_dup is True
        assert result.method == "jaccard"

    def test_jaccard_similar_not_dup(self):
        """Jaccard 中等相似度 → 不算重复但记录 similar_facts"""
        db = self._make_db_cog(
            exists=False,
            fts_results=[
                {"id": 1, "assertion": "apple banana cherry date elderberry"},
            ],
        )
        result = check_duplicate("apple banana grape fig honeydew", "col", "sess", db)
        assert result.is_dup is False

    def test_exclude_id(self):
        """排除自身 ID"""
        db = self._make_db_cog(
            exists=False,
            fts_results=[
                {"id": 42, "assertion": "same content exactly"},
            ],
        )
        result = check_duplicate("same content exactly", "col", "sess", db, exclude_id=42)
        assert result.is_dup is False

    def test_layer1_exception_fallthrough(self):
        """Layer 1 异常 → 继续 Layer 2"""
        db = self._make_db_cog()
        db.memory_exists.side_effect = Exception("DB error")
        db.fts_search_memories.return_value = []
        result = check_duplicate("test", "col", "sess", db)
        assert result.is_dup is False

    def test_layer2_exception_fallthrough(self):
        """Layer 2 异常 → 继续到结果"""
        db = self._make_db_cog(exists=False)
        db.fts_search_memories.side_effect = Exception("FTS error")
        result = check_duplicate("test", "col", "sess", db)
        assert result.is_dup is False

    def test_no_embedder_skips_layer3(self):
        """无 embedder → 跳过 Layer 3"""
        db = self._make_db_cog(exists=False, fts_results=[])
        result = check_duplicate("test", "col", "sess", db, embedder=None)
        assert result.is_dup is False


# ── check_session_cosine_dup ──


class TestCheckSessionCosineDup:
    """Session 级余弦去重"""

    def test_no_session_id(self):
        result = check_session_cosine_dup("text", None, "col", MagicMock(), MagicMock())
        assert result.is_dup is False

    def test_no_embedder(self):
        result = check_session_cosine_dup("text", "sess", "col", MagicMock(), None)
        assert result.is_dup is False

    def test_no_vec_loaded(self):
        db = MagicMock()
        db._vec_loaded = False
        result = check_session_cosine_dup("text", "sess", "col", db, MagicMock())
        assert result.is_dup is False


# ── conflict._has_negation ──


class TestHasNegation:
    """否定词检测"""

    def test_chinese_negation(self):
        assert _has_negation("不是这样") is True

    def test_chinese_forbidden(self):
        assert _has_negation("禁止使用") is True

    def test_english_not(self):
        assert _has_negation("do not use this") is True

    def test_english_never(self):
        assert _has_negation("never do that") is True

    def test_no_negation(self):
        assert _has_negation("this is fine") is False

    def test_empty(self):
        assert _has_negation("") is False


# ── detect_conflicts ──


class TestDetectConflicts:
    """冲突检测"""

    def test_no_similar_facts(self):
        result = detect_conflicts("new assertion", [])
        assert result.action == "keep_both"

    def test_empty_assertion(self):
        result = detect_conflicts("", [{"id": 1, "assertion": "old", "similarity": 0.9}])
        assert result.action == "keep_both"

    def test_low_similarity_keep_both(self):
        """低相似度 → keep_both"""
        result = detect_conflicts(
            "new assertion",
            [{"id": 1, "assertion": "completely different text", "similarity": 0.3}],
        )
        assert result.action == "keep_both"

    def test_high_sim_with_negation_mismatch(self):
        """高相似度 + 否定词不匹配 → keep_new"""
        result = detect_conflicts(
            "不应该使用 eval 函数处理用户输入",
            [{"id": 1, "assertion": "应该使用 eval 函数处理用户输入", "similarity": 0.9}],
        )
        assert result.action == "keep_new"
        assert result.superseded_id == 1

    def test_high_sim_same_negation(self):
        """高相似度但否定词一致 → 通常 keep_both"""
        result = detect_conflicts(
            "不要使用 eval",
            [{"id": 1, "assertion": "不要使用 exec", "similarity": 0.85}],
        )
        # 都有否定词，不算 mismatch → posterior 较低
        assert result.action == "keep_both"

    def test_medium_similarity_negation_mismatch(self):
        """中等相似度 + 否定 mismatch → 可能 keep_new"""
        result = detect_conflicts(
            "这个方法不可行",
            [{"id": 1, "assertion": "这个方法可行", "similarity": 0.7}],
        )
        # similarity=0.7 < 0.8 → high_sim=False, 但有 neg_mismatch
        assert result.action in ("keep_new", "keep_both")

    def test_conflict_result_frozen(self):
        """ConflictResult 是 frozen dataclass"""
        r = ConflictResult(action="keep_both", superseded_id=None, reason="test")
        with pytest.raises(AttributeError):
            r.action = "keep_new"

    def test_similarity_clamped(self):
        """similarity 超范围被 clamp"""
        result = detect_conflicts(
            "不要这样做",
            [{"id": 1, "assertion": "要这样做", "similarity": 1.5}],
        )
        # 不崩溃，正常返回
        assert result.action in ("keep_new", "keep_both")

    def test_none_assertion_input(self):
        """非 str assertion → keep_both"""
        result = detect_conflicts(
            None,
            [{"id": 1, "assertion": "old", "similarity": 0.9}],
        )
        assert result.action == "keep_both"
