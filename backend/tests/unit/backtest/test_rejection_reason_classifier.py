"""
拒绝原因分类器单元测试（工单#24）
"""

from app.services.backtest.utils.rejection_reason_classifier import (
    REJECTION_CATEGORIES,
    aggregate_rejection_reasons,
    classify_rejection_reason,
    get_category_label,
    is_actionable_rejection,
)


class TestClassifyRejectionReason:
    """测试 classify_rejection_reason"""

    def test_no_position(self):
        assert classify_rejection_reason("无持仓") == "no_position"
        assert classify_rejection_reason("持仓数量为0") == "no_position"

    def test_insufficient_buy_quantity(self):
        assert classify_rejection_reason("可买数量不足") == "insufficient_buy_quantity"
        assert classify_rejection_reason("无法买入100股") == "insufficient_buy_quantity"

    def test_insufficient_funds(self):
        assert classify_rejection_reason("资金不足") == "insufficient_funds"
        assert classify_rejection_reason("可用资金不足") == "insufficient_funds"
        assert classify_rejection_reason("需要保留5%") == "insufficient_funds"

    def test_position_limit(self):
        assert classify_rejection_reason("已达到最大持仓限制") == "position_limit"
        assert (
            classify_rejection_reason("超过topk持仓上限(topk=10)") == "position_limit"
        )
        assert (
            classify_rejection_reason("股票代码不在universe中: 000001.SZ")
            == "position_limit"
        )

    def test_strength_too_low(self):
        assert classify_rejection_reason("信号验证失败") == "strength_too_low"
        assert classify_rejection_reason("验证失败") == "strength_too_low"

    def test_matching_failed(self):
        assert classify_rejection_reason("执行失败（未知原因）") == "matching_failed"
        assert (
            classify_rejection_reason("执行异常: division by zero") == "matching_failed"
        )

    def test_none_or_empty(self):
        assert classify_rejection_reason(None) == "other"
        assert classify_rejection_reason("") == "other"
        assert classify_rejection_reason("   ") == "other"

    def test_unknown_falls_to_other(self):
        assert classify_rejection_reason("未知原因xyz") == "other"


class TestGetCategoryLabel:
    """测试 get_category_label"""

    def test_known_categories(self):
        assert get_category_label("no_position") == "无持仓"
        assert get_category_label("insufficient_funds") == "资金不足"

    def test_unknown_returns_key(self):
        assert get_category_label("unknown_key") == "unknown_key"


class TestAggregateRejectionReasons:
    """测试 aggregate_rejection_reasons"""

    def test_empty_list(self):
        breakdown, top = aggregate_rejection_reasons([])
        assert breakdown == {}
        assert top == []

    def test_single_reason(self):
        breakdown, top = aggregate_rejection_reasons(["无持仓"])
        assert breakdown == {"no_position": 1}
        assert len(top) == 1
        assert top[0]["reason"] == "无持仓"
        assert top[0]["count"] == 1

    def test_multiple_same_reason(self):
        breakdown, top = aggregate_rejection_reasons(
            ["资金不足", "资金不足", "可用资金不足"]
        )
        assert breakdown["insufficient_funds"] == 3
        assert top[0]["reason"] == "资金不足"
        assert top[0]["count"] == 3

    def test_mixed_reasons_sorted_by_count(self):
        reasons = ["无持仓", "资金不足", "无持仓", "资金不足", "资金不足", "其他"]
        breakdown, top = aggregate_rejection_reasons(reasons)
        assert breakdown["insufficient_funds"] == 3
        assert breakdown["no_position"] == 2
        assert breakdown["other"] == 1
        assert top[0]["count"] >= top[1]["count"] >= top[2]["count"]

    def test_none_in_list(self):
        breakdown, top = aggregate_rejection_reasons([None, "无持仓", None])
        assert breakdown.get("no_position") == 1
        assert breakdown.get("other") == 2  # None -> other
        assert sum(breakdown.values()) == 3


class TestRejectionCategories:
    """测试标准类别覆盖"""

    def test_all_required_categories(self):
        required = [
            "no_position",
            "insufficient_buy_quantity",
            "insufficient_funds",
            "position_limit",
            "strength_too_low",
            "matching_failed",
            "other",
        ]
        for k in required:
            assert k in REJECTION_CATEGORIES
            assert len(REJECTION_CATEGORIES[k]) > 0


class TestIsActionableRejection:
    """测试 is_actionable_rejection：不计入 actionable vs 计入 actionable（工单#24 P0 语义）"""

    def test_no_position_not_actionable(self):
        assert is_actionable_rejection("无持仓") is False
        assert is_actionable_rejection("持仓数量为0") is False

    def test_insufficient_buy_quantity_not_actionable(self):
        assert is_actionable_rejection("可买数量不足") is False
        assert is_actionable_rejection("无法买入100股") is False

    def test_insufficient_cash_not_actionable(self):
        assert is_actionable_rejection("资金不足") is False
        assert is_actionable_rejection("可用资金不足") is False
        assert is_actionable_rejection("需要保留5%") is False

    def test_position_limit_not_actionable(self):
        assert is_actionable_rejection("已达到最大持仓限制") is False
        assert is_actionable_rejection("超过topk持仓上限") is False

    def test_strength_too_low_not_actionable(self):
        assert is_actionable_rejection("信号验证失败") is False
        assert is_actionable_rejection("强度过低") is False

    def test_matching_failed_actionable(self):
        assert is_actionable_rejection("执行失败（未知原因）") is True
        assert is_actionable_rejection("执行异常: division by zero") is True
        assert is_actionable_rejection("撮合失败") is True

    def test_other_not_actionable(self):
        assert is_actionable_rejection("未知原因xyz") is False
        assert is_actionable_rejection(None) is False
        assert is_actionable_rejection("") is False
