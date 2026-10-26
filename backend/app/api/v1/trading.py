"""
可执行交易信号 API（2026-10-25）

对外提供「今天该买什么、买多少、为什么」：
  GET /api/v1/trading/strategies                可用信号策略列表
  GET /api/v1/trading/signal?strategy=xxx      今日可执行信号（含股数/金额/成本）
  GET /api/v1/trading/performance?strategy=xxx 纸面业绩（真实 track record）

这些是付费用户真正买的东西 —— 不是回测报告，是可以照着下单的清单。

**付费墙**：按订阅等级限制可见策略线、信号延迟与执行明细。
限制在**后端强制**（见 app/services/trading/entitlements.py），
前端只负责展示升级入口。
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.v1.dependencies import require_current_user
from app.services.trading.entitlements import (
    allowed_strategies,
    effective_delay_days,
    entitlement_for,
    normalize_tier,
    upgrade_required,
)
from app.services.trading.signal_service import (
    SIGNAL_STRATEGIES,
    TradingSignalService,
)

router = APIRouter(prefix="/trading", tags=["可执行信号"])


def _tier_of(user: Any) -> str:
    """取用户的订阅等级。

    管理员（平台运营）绕过付费墙 —— 否则运营自己看不到完整信号，
    既没法支持付费用户，也没法核对纸面账本。
    """
    if getattr(user, "is_admin", False):
        return "pro"
    return normalize_tier(getattr(user, "subscription_tier", None))


@router.get("/strategies", summary="可用信号策略列表")
async def list_signal_strategies(
    user: Any = Depends(require_current_user),
) -> Dict[str, Any]:
    """列出可产出信号的策略及其纸面盘状态。

    受订阅等级限制：免费用户只看到 1 条线，其余标 locked 并隐藏净值细节。
    """
    svc = TradingSignalService()
    tier = _tier_of(user)
    ent = entitlement_for(tier)
    allowed = set(allowed_strategies(tier))

    out: List[Dict[str, Any]] = []
    for s in svc.list_strategies():
        key = s["key"]
        locked = key not in allowed
        item = dict(s)
        item["locked"] = locked
        item["delay_days"] = ent.delay_days
        if locked:
            # 锁定策略不暴露持仓与净值细节
            item["n_positions"] = 0
            item["last_value"] = None
            item["has_paper"] = False
        out.append(item)

    return {
        "strategies": out,
        "count": len(out),
        "tier": tier,
        "tier_label": ent.label,
        "allowed_strategies": sorted(allowed),
        "delay_days": ent.delay_days,
        "include_execution_detail": ent.include_execution_detail,
    }


@router.get("/signal", summary="今日可执行信号")
async def get_daily_signal(
    strategy: str = Query(
        "honest_42f", description="策略 key，见 /trading/strategies"
    ),
    user: Any = Depends(require_current_user),
) -> Dict[str, Any]:
    """返回某策略今天的买卖清单。

    字段说明：
      buys   买入清单（代码/名称/价格/股数/金额/成本）—— 可直接照着下单
      sells  卖出清单（同上）
      holds  当前持仓（代码/名称/股数/成本/买入日）

    免费用户（tier=free）：
      - 只能访问 1 条线，其余返回 403 并附升级提示
      - 延迟 1 个交易日（返回上一次调仓的信号）
      - 不返回股数 / 金额明细
    """
    tier = _tier_of(user)
    ent = entitlement_for(tier)

    if strategy not in SIGNAL_STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}",
        )

    hint = upgrade_required(tier, strategy)
    if hint:
        raise HTTPException(status_code=403, detail=hint)

    svc = TradingSignalService()
    result = svc.daily_signal(strategy)

    # 免费用户：延迟 1 个交易日 + 隐去执行明细
    if ent.delay_days > 0:
        result = _apply_delay(result, ent.delay_days)
    if not ent.include_execution_detail:
        result = _strip_execution_detail(result)

    result["tier"] = tier
    result["tier_label"] = ent.label
    result["delay_days"] = ent.delay_days
    result["upgrade_hint"] = ent.upgrade_hint
    return result


@router.get("/signal-quality", summary="信号质量（胜率与超额收益）")
async def get_signal_quality(
    strategy: str = Query("honest_42f", description="策略 key"),
    user: Any = Depends(require_current_user),
) -> Dict[str, Any]:
    """信号质量统计：T+1/T+5/T+10/T+20 的胜率与超额收益。

    这是「凭什么信你的信号」的回答 —— 每个历史买入信号发出后的
    实际表现，相对全市场等权基准。
    """
    tier = _tier_of(user)
    ent = entitlement_for(tier)

    if strategy not in SIGNAL_STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}",
        )

    hint = upgrade_required(tier, strategy)
    if hint:
        raise HTTPException(status_code=403, detail=hint)

    result = TradingSignalService().signal_quality(strategy)
    result["tier"] = tier
    result["tier_label"] = ent.label
    if not ent.include_full_ledger:
        # 免费用户：只给汇总指标，不给逐笔信号明细
        result["recent_10"] = []
    return result


@router.get("/performance", summary="纸面业绩（真实 track record）")
async def get_performance(
    strategy: str = Query("honest_42f", description="策略 key"),
    user: Any = Depends(require_current_user),
) -> Dict[str, Any]:
    """纸面组合的业绩、持仓与交易记录 —— 对外展示的可验证记录。"""
    tier = _tier_of(user)
    ent = entitlement_for(tier)

    if strategy not in SIGNAL_STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}",
        )

    hint = upgrade_required(tier, strategy)
    if hint:
        raise HTTPException(status_code=403, detail=hint)

    svc = TradingSignalService()
    result = svc.performance(strategy)

    if not ent.include_full_ledger:
        # 免费用户：只给汇总，不给持仓明细和逐笔交易
        result = {
            "strategy": result["strategy"],
            "display_name": result["display_name"],
            "status": result["status"],
            "date": result.get("date"),
            "capital": result.get("capital"),
            "value": result.get("value"),
            "total_return_pct": result.get("total_return_pct"),
            "n_positions": result.get("n_positions"),
            "n_trades": result.get("n_trades"),
            "positions": [],
            "recent_trades": [],
            "locked_detail": True,
            "upgrade_hint": ent.upgrade_hint,
        }

    result["tier"] = tier
    result["tier_label"] = ent.label
    return result


# ── 内部工具 ──────────────────────────────────────────────────────────

def _apply_delay(payload: Dict[str, Any], delay_days: int) -> Dict[str, Any]:
    """延迟 N 个交易日 —— 返回上一次调仓的信号。

    纸面盘的 portfolio.json 只保留最近一次调仓结果，所以「上一次」需要从
    <strategy>_ledger.csv 回溯，找到倒数第 N+1 个不同的调仓日。
    """
    import csv
    import os

    from app.services.trading.signal_service import _ledger_path

    strategy = payload.get("strategy", "")
    ledger = _ledger_path(strategy)
    if not os.path.exists(ledger):
        return payload

    try:
        with open(ledger) as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return payload

    days: List[str] = []
    for r in rows:
        d = r.get("date", "")
        if d and (not days or days[-1] != d):
            days.append(d)
    # days 升序（最早→最晚），取倒数第 delay_days+1 个
    if len(days) <= delay_days:
        return payload

    target_day = days[-1 - delay_days]
    out = dict(payload)
    out["date"] = target_day
    out["delayed"] = True
    out["delay_days"] = delay_days
    # 延迟版本不暴露当天的买卖明细（那是新信号的价值所在）
    out["buys"] = []
    out["sells"] = []
    return out


def _strip_execution_detail(payload: Dict[str, Any]) -> Dict[str, Any]:
    """隐去股数 / 金额等执行明细（免费用户）。"""
    out = dict(payload)
    for key in ("buys", "sells"):
        rows = []
        for r in out.get(key) or []:
            r = dict(r)
            for f in ("shares", "amount", "commission", "stamp_tax", "slippage", "total_cost"):
                r.pop(f, None)
            rows.append(r)
        out[key] = rows
    holds = []
    for h in out.get("holds") or []:
        h = dict(h)
        h.pop("shares", None)
        h.pop("cost_amount", None)
        holds.append(h)
    out["holds"] = holds
    out["execution_detail_locked"] = True
    return out
