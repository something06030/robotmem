"""冲突检测 — learn() 入库前矛盾判断

L0 贝叶斯冲突检测（纯规则，无 LLM 依赖）。
新记忆与已有 similar memories 是否语义矛盾。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# action 白名单 — 只接受这三个值
_VALID_ACTIONS = frozenset({"keep_new", "keep_old", "keep_both"})


@dataclass(frozen=True)
class ConflictResult:
    """冲突检测结果（frozen — 防止共享单例被意外修改）"""
    action: str                 # "keep_new" | "keep_old" | "keep_both"
    superseded_id: int | None   # 被淘汰的记忆 ID
    reason: str                 # 判断理由

_KEEP_BOTH = ConflictResult(action="keep_both", superseded_id=None, reason="")

# ── L0 贝叶斯冲突检测 ──

# 否定词模式（中/英）— 用于检测语义反转
_NEGATION_CN = re.compile(r"不[是要能会再]|没有|未|禁止|不应|不可|不需|无法")
_NEGATION_EN = re.compile(r"\b(not|no|never|don'?t|doesn'?t|shouldn'?t|can'?t|won'?t|isn'?t|aren'?t)\b", re.I)

# 贝叶斯先验参数
_PRIOR_CONFLICT = 0.08      # P(conflict) — 8% 的相似 memories 是矛盾的
_P_HIGH_SIM_CONFLICT = 0.75  # P(sim>0.8 | conflict)
_P_HIGH_SIM_NO_CONFLICT = 0.25  # P(sim>0.8 | ¬conflict)
_P_NEG_CONFLICT = 0.60       # P(negation_mismatch | conflict)
_P_NEG_NO_CONFLICT = 0.10    # P(negation_mismatch | ¬conflict)
_POSTERIOR_THRESHOLD = 0.40   # posterior > 0.4 → keep_new


def _has_negation(text: str) -> bool:
    """检测文本是否包含否定词"""
    return bool(_NEGATION_CN.search(text) or _NEGATION_EN.search(text))


def _l0_bayesian_conflict(
    new_assertion: str,
    similar_facts: list[dict],
) -> ConflictResult:
    """L0 贝叶斯冲突检测（无 LLM 兜底）

    贝叶斯模型：
    P(conflict | evidence) ∝ P(evidence | conflict) × P(conflict)

    evidence = {high_similarity, negation_mismatch}

    三层防御：
    - L1: 输入校验（similar_facts 非空，similarity 范围 [0,1]）
    - L2: posterior clamp [0,1]，纯数学不抛异常
    - L3: 记录判决结果（posterior + decision）
    """
    if not similar_facts:
        return _KEEP_BOTH

    top = similar_facts[0]
    sim = top.get("similarity", 0)
    old_assertion = top.get("assertion", "")
    old_id = top.get("id")

    # L1: similarity 范围校验
    sim = max(0.0, min(1.0, float(sim)))

    # 低相似度 → 不可能矛盾
    if sim < 0.6:
        return _KEEP_BOTH

    # 计算证据
    high_sim = sim > 0.8
    neg_mismatch = _has_negation(new_assertion) != _has_negation(old_assertion)

    # 贝叶斯后验
    p_c = _PRIOR_CONFLICT
    p_nc = 1.0 - p_c

    likelihood_c = 1.0
    likelihood_nc = 1.0

    if high_sim:
        likelihood_c *= _P_HIGH_SIM_CONFLICT
        likelihood_nc *= _P_HIGH_SIM_NO_CONFLICT
    else:
        likelihood_c *= (1.0 - _P_HIGH_SIM_CONFLICT)
        likelihood_nc *= (1.0 - _P_HIGH_SIM_NO_CONFLICT)

    if neg_mismatch:
        likelihood_c *= _P_NEG_CONFLICT
        likelihood_nc *= _P_NEG_NO_CONFLICT
    else:
        likelihood_c *= (1.0 - _P_NEG_CONFLICT)
        likelihood_nc *= (1.0 - _P_NEG_NO_CONFLICT)

    numerator = likelihood_c * p_c
    denominator = numerator + likelihood_nc * p_nc

    # L2: 防除零 + clamp
    posterior = max(0.0, min(1.0, numerator / denominator)) if denominator > 0 else 0.0

    # L3: 日志
    logger.info(
        "L0 贝叶斯: sim=%.3f neg_mismatch=%s posterior=%.3f threshold=%.2f",
        sim, neg_mismatch, posterior, _POSTERIOR_THRESHOLD,
    )

    if posterior >= _POSTERIOR_THRESHOLD and old_id is not None:
        return ConflictResult(
            action="keep_new",
            superseded_id=old_id,
            reason=f"L0 Bayesian: posterior={posterior:.3f} (sim={sim:.2f}, neg_mismatch={neg_mismatch})",
        )

    return _KEEP_BOTH


def detect_conflicts(
    new_assertion: str,
    similar_facts: list[dict],
) -> ConflictResult:
    """检测新记忆与相似 memories 是否矛盾 — L0 贝叶斯（纯规则，无 LLM）

    Args:
        new_assertion: 新记忆文本
        similar_facts: dedup 返回的 similar_facts 列表
                       [{"id": int, "assertion": str, "similarity": float}, ...]

    Returns:
        ConflictResult — action + superseded_id + reason
    """
    # --- L1 事前校验 ---
    if not similar_facts:
        return _KEEP_BOTH

    if not isinstance(new_assertion, str) or not new_assertion.strip():
        logger.warning("detect_conflicts: 空 assertion，跳过")
        return _KEEP_BOTH

    result = _l0_bayesian_conflict(new_assertion, similar_facts)
    logger.debug("detect_conflicts: L0 贝叶斯 action=%s", result.action)
    return result
