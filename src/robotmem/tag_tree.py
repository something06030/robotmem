"""记忆树标签定义 — 共享受控词表

所有标签相关模块从此处导入，确保单一真相源。
auto_classify / db_cog 共用。

扩展方式：在 TAG_META_TREE 中加一行即可，_ensure_tag_meta_tree() 启动时自动同步 DB。
"""

from __future__ import annotations

# (tag, parent, display_name)
# parent=None 表示根节点（9 维认知维度）
TAG_META_TREE: list[tuple[str, str | None, str]] = [
    # ── 根节点（9 维）──
    ("metacognition", None, "元认知"),
    ("capability", None, "能力"),
    ("domain", None, "领域"),
    ("technique", None, "技法"),
    ("timing", None, "时机"),
    ("boundary", None, "边界"),
    ("experience", None, "经验"),
    ("self_defect", None, "AI缺陷"),
    ("reflection", None, "反思"),
    # ── metacognition 子节点 ──
    ("reasoning", "metacognition", "推理"),
    ("cognitive_bias", "metacognition", "认知偏差"),
    ("decision_framework", "metacognition", "决策框架"),
    ("systems_thinking", "metacognition", "系统思维"),
    ("risk_thinking", "metacognition", "风险思维"),
    # ── capability 子节点 ──
    ("build", "capability", "构建"),
    ("debug", "capability", "问题排查"),
    ("design", "capability", "系统设计"),
    ("review", "capability", "代码审查"),
    ("explain", "capability", "解释说明"),
    ("optimize", "capability", "性能优化"),
    ("plan", "capability", "方案规划"),
    # ── domain 子节点 ──
    ("cs_fundamentals", "domain", "计算机基础"),
    ("ai_ml", "domain", "AI/ML"),
    ("finance", "domain", "金融"),
    ("business", "domain", "商业"),
    ("cross_domain", "domain", "跨领域"),
    # ── cs_fundamentals 子节点 ──
    ("compiler_lang", "cs_fundamentals", "编译/语言"),
    ("os", "cs_fundamentals", "操作系统"),
    ("concurrency", "cs_fundamentals", "并发"),
    ("database", "cs_fundamentals", "数据库"),
    ("distributed", "cs_fundamentals", "分布式"),
    ("network", "cs_fundamentals", "网络"),
    ("algo_ds", "cs_fundamentals", "算法/数据结构"),
    ("security", "cs_fundamentals", "安全"),
    # ── finance 子节点 ──
    ("crypto", "finance", "加密货币"),
    ("traditional", "finance", "传统金融"),
    ("behavioral_finance", "finance", "行为金融"),
    # ── technique 子节点 ──
    ("patterns", "technique", "设计模式"),
    ("anti_patterns", "technique", "反模式"),
    ("recipes", "technique", "实用技巧"),
    ("language_specific", "technique", "语言特性"),
    # ── timing 子节点 ──
    ("when_to_start", "timing", "何时开始"),
    ("when_to_stop", "timing", "何时停止"),
    ("when_to_switch", "timing", "何时切换"),
    # ── boundary 子节点 ──
    ("tradeoff", "boundary", "权衡"),
    ("not_applicable", "boundary", "不适用"),
    ("diminishing_returns", "boundary", "收益递减"),
    # ── experience 子节点 ──
    ("war_story", "experience", "实战故事"),
    ("postmortem", "experience", "教训/复盘"),
    ("gotcha", "experience", "踩坑"),
    # ── self_defect 子节点 ──
    ("hallucination", "self_defect", "幻觉"),
    ("sycophancy", "self_defect", "讨好倾向"),
    ("overengineering", "self_defect", "过度设计"),
    ("no_verification", "self_defect", "缺少验证"),
    # ── reflection 子节点 ──
    ("accuracy_calibration", "reflection", "准确度校准"),
    ("behavior_rule", "reflection", "行为规则"),
    ("blind_spot", "reflection", "盲点"),
    # ── auto_classify 专用（不在树中但需要在词表内）──
    ("constraint", "boundary", "约束"),
    ("preference", "reflection", "偏好"),
    ("worldview", "metacognition", "世界观"),
    ("root_cause", "experience", "根因"),
    ("decision", "metacognition", "决策"),
    ("revert", "experience", "回滚"),
    ("pattern", "technique", "规律"),
    ("architecture", "capability", "架构设计"),
    ("config", "domain", "配置"),
    ("observation", "domain", "观察发现"),
    ("observation_code", "domain", "代码观察"),
    ("observation_debug", "domain", "调试观察"),
    ("code", "capability", "代码"),
]

# 所有合法标签集合 — 用于白名单校验
VALID_TAGS: frozenset[str] = frozenset(tag for tag, _, _ in TAG_META_TREE)

# tag → display_name 映射 — 用于展示
DISPLAY_NAMES: dict[str, str] = {tag: dn for tag, _, dn in TAG_META_TREE}

# 旧格式兼容：(tag, parent) 列表 — 供 db_cog 迁移代码使用
TAG_META_TREE_COMPAT: list[tuple[str, str | None]] = [
    (tag, parent) for tag, parent, _ in TAG_META_TREE
]

# tag → parent 映射 — 用于维度前缀 lookup
_TAG_PARENTS: dict[str, str | None] = {tag: parent for tag, parent, _ in TAG_META_TREE}


def dimension_prefix(category: str | None) -> str:
    """category → [维度/子维度] 前缀

    L0 纯 dict lookup，0ms。
    L1 防御：category 无效时返回空字符串。
    """
    if not category or not isinstance(category, str):
        return ""
    if category not in DISPLAY_NAMES:
        return ""
    parent = _TAG_PARENTS.get(category)
    if parent is None:
        # 根节点：[经验]
        return f"[{DISPLAY_NAMES[category]}]"
    # 子节点：[经验/踩坑]
    parent_display = DISPLAY_NAMES.get(parent, "")
    child_display = DISPLAY_NAMES[category]
    if parent_display:
        return f"[{parent_display}/{child_display}]"
    return f"[{child_display}]"
