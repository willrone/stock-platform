"""
拒绝原因分类与汇总工具
将原始 execution_reason 字符串映射为标准类别，供 repository/API/report 共用。
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# 标准拒绝原因类别（与工单#24 要求一致）
REJECTION_CATEGORIES = {
    "no_position": "无持仓",
    "insufficient_buy_quantity": "可买数量不足",
    "insufficient_funds": "资金不足",
    "position_limit": "仓位限制",
    "strength_too_low": "强度过低",
    "matching_failed": "撮合失败",
    "other": "其他",
}

# 关键词到标准类别的映射（不区分大小写）
_REASON_PATTERNS = [
    # 无持仓
    (["无持仓", "持仓数量为0", "no position"], "no_position"),
    # 可买数量不足
    (["可买数量不足", "无法买入100股"], "insufficient_buy_quantity"),
    # 资金不足
    (["资金不足", "可用资金不足", "需要保留5%"], "insufficient_funds"),
    # 仓位限制
    (
        [
            "最大持仓",
            "持仓限制",
            "超过topk",
            "topk持仓上限",
            "topk=",
            "股票代码不在universe",
        ],
        "position_limit",
    ),
    # 强度过低
    (["强度过低", "信号验证", "验证失败", "strength"], "strength_too_low"),
    # 撮合失败
    (["执行失败", "执行异常", "撮合"], "matching_failed"),
]


def classify_rejection_reason(raw_reason: Optional[str]) -> str:
    """
    将原始 execution_reason 映射为标准拒绝原因类别。

    Args:
        raw_reason: 原始未执行原因字符串

    Returns:
        标准类别 key，如 "no_position", "insufficient_funds" 等
    """
    if not raw_reason or not isinstance(raw_reason, str):
        return "other"

    r = raw_reason.strip().lower() if raw_reason else ""
    if not r:
        return "other"

    # 按优先级匹配（第一个匹配到的类别）
    for keywords, category in _REASON_PATTERNS:
        for kw in keywords:
            if kw.lower() in r or kw in raw_reason:
                return category

    return "other"


def get_category_label(category_key: str) -> str:
    """获取标准类别的中文展示标签"""
    return REJECTION_CATEGORIES.get(category_key, category_key)


# 不计入 actionable 的拒绝类别（P0 语义：这些是前置条件不满足，非“真正可执行”）
_NON_ACTIONABLE_CATEGORIES = frozenset(
    {
        "no_position",
        "insufficient_buy_quantity",
        "insufficient_funds",
        "position_limit",
        "strength_too_low",
        "other",
    }
)


def is_actionable_rejection(raw_reason: Optional[str]) -> bool:
    """
    判断该拒绝原因对应的信号是否计入 actionable。

    P0 语义：actionable = 真正“可执行”的信号。
    - 不计入：无持仓、可买数量不足、资金不足、仓位限制、强度过低、其他（前置条件不满足）
    - 计入：matching_failed（撮合失败，执行阶段失败，体现“本可执行但执行时失败”）

    Args:
        raw_reason: 原始 execution_reason 字符串（未执行信号的拒绝原因）

    Returns:
        True 仅当该拒绝原因对应“可执行但执行失败”（如撮合失败）；否则 False
    """
    cat = classify_rejection_reason(raw_reason)
    return cat not in _NON_ACTIONABLE_CATEGORIES  # 即 matching_failed 计入 actionable


def aggregate_rejection_reasons(
    raw_reasons: List[Optional[str]],
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    聚合原始拒绝原因列表，返回分类统计与 Top 拒绝原因。

    Args:
        raw_reasons: 原始 execution_reason 列表（可含 None）

    Returns:
        (rejection_reason_breakdown, top_rejection_reasons)
        - rejection_reason_breakdown: {category_key: count}
        - top_rejection_reasons: [{"reason": label, "count": n}, ...] 按 count 降序
    """
    breakdown: Counter = Counter()
    for r in raw_reasons:
        cat = classify_rejection_reason(r)
        breakdown[cat] += 1

    breakdown_dict = dict(breakdown)
    top_list = [
        {"reason": get_category_label(k), "reason_key": k, "count": v}
        for k, v in breakdown.most_common()
    ]
    return breakdown_dict, top_list
