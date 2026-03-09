"""tag_tree.py 单元测试"""

from robotmem.tag_tree import (
    TAG_META_TREE,
    VALID_TAGS,
    DISPLAY_NAMES,
    TAG_META_TREE_COMPAT,
    dimension_prefix,
)


class TestTagMetaTree:
    def test_non_empty(self):
        assert len(TAG_META_TREE) > 50

    def test_tuple_format(self):
        for item in TAG_META_TREE:
            assert len(item) == 3
            tag, parent, display = item
            assert isinstance(tag, str)
            assert parent is None or isinstance(parent, str)
            assert isinstance(display, str)

    def test_parent_exists(self):
        """每个非根节点的 parent 必须在树中"""
        tags = {tag for tag, _, _ in TAG_META_TREE}
        for tag, parent, _ in TAG_META_TREE:
            if parent is not None:
                assert parent in tags, f"{tag} 的 parent {parent} 不在树中"


class TestValidTags:
    def test_frozenset(self):
        assert isinstance(VALID_TAGS, frozenset)

    def test_contains_root(self):
        for root in ("metacognition", "capability", "domain", "technique",
                      "timing", "boundary", "experience", "self_defect", "reflection"):
            assert root in VALID_TAGS

    def test_contains_leaf(self):
        assert "gotcha" in VALID_TAGS
        assert "postmortem" in VALID_TAGS


class TestDisplayNames:
    def test_dict(self):
        assert isinstance(DISPLAY_NAMES, dict)

    def test_root(self):
        assert DISPLAY_NAMES["metacognition"] == "元认知"

    def test_leaf(self):
        assert DISPLAY_NAMES["gotcha"] == "踩坑"


class TestTagMetaTreeCompat:
    def test_format(self):
        for item in TAG_META_TREE_COMPAT:
            assert len(item) == 2
            tag, parent = item
            assert isinstance(tag, str)

    def test_same_length(self):
        assert len(TAG_META_TREE_COMPAT) == len(TAG_META_TREE)


class TestDimensionPrefix:
    def test_root_node(self):
        assert dimension_prefix("experience") == "[经验]"

    def test_child_node(self):
        assert dimension_prefix("gotcha") == "[经验/踩坑]"

    def test_none(self):
        assert dimension_prefix(None) == ""

    def test_empty(self):
        assert dimension_prefix("") == ""

    def test_invalid(self):
        assert dimension_prefix("nonexistent_tag") == ""

    def test_not_string(self):
        assert dimension_prefix(123) == ""

    def test_metacognition(self):
        assert dimension_prefix("metacognition") == "[元认知]"

    def test_deep_child(self):
        # compiler_lang → cs_fundamentals → domain
        assert dimension_prefix("compiler_lang") == "[计算机基础/编译/语言]"

    def test_all_roots(self):
        roots = [tag for tag, parent, _ in TAG_META_TREE if parent is None]
        for root in roots:
            prefix = dimension_prefix(root)
            assert prefix.startswith("[")
            assert prefix.endswith("]")
