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

from dataclasses import replace
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.v1.dependencies import require_current_user
from app.api.v1.schemas import StandardResponse
from app.services.trading.entitlements import (
    ALL_STRATEGIES,
    SignalEntitlement,
    allowed_strategies,
    entitlement_for,
    normalize_tier,
    upgrade_required,
)
from app.services.trading.signal_service import (
    SIGNAL_STRATEGIES,
    TradingSignalService,
)
from app.services.trading.strategy_race import strategy_race_service

router = APIRouter(prefix="/trading", tags=["可执行信号"])


def _tier_of(user: Any) -> str:
    """取用户的**真实**订阅等级（用于展示、计费与权限判定）。

    【P0 · consistency】旧实现对管理员直接返回 "pro"（想让它绕过付费墙），
    结果同一个用户：
      - /pricing  读 user.subscription_tier → 显示「专业版」
      - /today-signal 读 trading tier      → 显示「信号版」
    两页档位打架，用户以为套餐被降级了。

    正确做法：**付费墙绕过是访问控制，不是档位改写**。
    管理员的真实档位照常返回，访问权限由 _is_admin 单独放宽。
    """
    return normalize_tier(getattr(user, "subscription_tier", None))


def _is_admin(user: Any) -> bool:
    """平台运营 —— 绕过付费墙看全部信号线。

    否则运营自己看不到完整信号，既没法支持付费用户，也没法核对纸面账本。
    """
    return bool(getattr(user, "is_admin", False))


def _allowed_for(user: Any, tier: str) -> set:
    """该用户可访问的信号线集合。管理员拿全部。"""
    if _is_admin(user):
        return set(ALL_STRATEGIES)
    return set(allowed_strategies(tier))


def _ent_for(user: Any, tier: str) -> SignalEntitlement:
    """该用户的权限描述。

    管理员：权限全开（全部信号线 / 零延迟 / 完整执行明细 / 完整账本），
    但 **tier 与 label 保持真实档位** —— 付费墙绕过不该改写用户看到的套餐。
    """
    base = entitlement_for(tier)
    if not _is_admin(user):
        return base
    return replace(
        base,
        strategies=None,
        delay_days=0,
        include_execution_detail=True,
        include_full_ledger=True,
    )


@router.get("/strategies", response_model=StandardResponse, summary="可用信号策略列表")
async def list_signal_strategies(
    user: Any = Depends(require_current_user),
) -> StandardResponse:
    """列出可产出信号的策略及其纸面盘状态。

    受订阅等级限制：免费用户只看到 1 条线，其余标 locked 并隐藏净值细节。
    """
    svc = TradingSignalService()
    tier = _tier_of(user)
    ent = _ent_for(user, tier)
    # 管理员绕过付费墙，但**档位仍显示真实档位**（不改成 pro）
    allowed = _allowed_for(user, tier)

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

    return StandardResponse(
        success=True,
        message="策略列表获取成功",
        data={
            "strategies": out,
            "count": len(out),
            "tier": tier,
            "tier_label": ent.label,
            "allowed_strategies": sorted(allowed),
            "delay_days": ent.delay_days,
            "include_execution_detail": ent.include_execution_detail,
        },
    )


@router.get("/signal", response_model=StandardResponse, summary="今日可执行信号")
async def get_daily_signal(
    strategy: str = Query("honest_42f", description="策略 key，见 /trading/strategies"),
    user: Any = Depends(require_current_user),
) -> StandardResponse:
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
    ent = _ent_for(user, tier)

    if strategy not in SIGNAL_STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}",
        )

    # 管理员绕过付费墙（访问控制），档位展示仍为真实档位
    hint = None if _is_admin(user) else upgrade_required(tier, strategy)
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
    return StandardResponse(
        success=True,
        message="今日可执行信号获取成功",
        data=result,
    )


@router.get(
    "/signal-quality",
    response_model=StandardResponse,
    summary="信号质量（胜率与超额收益）",
)
async def get_signal_quality(
    strategy: str = Query("honest_42f", description="策略 key"),
    user: Any = Depends(require_current_user),
) -> StandardResponse:
    """信号质量统计：T+1/T+5/T+10/T+20 的胜率与超额收益。

    这是「凭什么信你的信号」的回答 —— 每个历史买入信号发出后的
    实际表现，相对全市场等权基准。
    """
    tier = _tier_of(user)
    ent = _ent_for(user, tier)

    if strategy not in SIGNAL_STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}",
        )

    # 管理员绕过付费墙（访问控制），档位展示仍为真实档位
    hint = None if _is_admin(user) else upgrade_required(tier, strategy)
    if hint:
        raise HTTPException(status_code=403, detail=hint)

    result = TradingSignalService().signal_quality(strategy)
    result["tier"] = tier
    result["tier_label"] = ent.label
    if not ent.include_full_ledger:
        # 免费用户：只给汇总指标，不给逐笔信号明细
        result["recent_10"] = []
    return StandardResponse(
        success=True,
        message="信号质量统计获取成功",
        data=result,
    )


@router.get(
    "/strategy-race",
    response_model=StandardResponse,
    summary="策略赛马（多策略 3 年真实盘口径横向对比）",
)
async def get_strategy_race(
    user: Any = Depends(require_current_user),
) -> StandardResponse:
    """多策略赛马：夏普 / 最大回撤 / 换手率 / 净值曲线 + 基准对比。

    【P0-04】这是「策略赛马」页真正的数据源 —— 之前该页只有纸面盘的**区间**
    收益（两次调仓之间几天），无法横向比较。这里给的是 **3 年全市场真实盘
    口径**回测（lag1 + 可成交性检查）的累计表现。

    付费墙：免费用户只返回汇总指标（不含净值曲线与分年明细）；
    信号版/专业版返回完整对比数据。
    """
    tier = _tier_of(user)
    ent = _ent_for(user, tier)
    result = strategy_race_service.race()

    if not ent.include_full_ledger:
        # 免费用户：保留排序后的核心指标，隐去净值曲线与分年明细
        result = {
            **result,
            "strategies": [
                {
                    "key": s["key"],
                    "name": s["name"],
                    "is_mainline": s["is_mainline"],
                    "metrics": s["metrics"],
                    "turnover": s["turnover"],
                    "nav": [],
                }
                for s in result.get("strategies", [])
            ],
            "locked_detail": True,
            "upgrade_hint": ent.upgrade_hint,
        }

    result["tier"] = tier
    result["tier_label"] = ent.label
    return StandardResponse(
        success=True,
        message="策略赛马数据获取成功",
        data=result,
    )


@router.get(
    "/performance",
    response_model=StandardResponse,
    summary="纸面业绩（真实 track record）",
)
async def get_performance(
    strategy: str = Query("honest_42f", description="策略 key"),
    user: Any = Depends(require_current_user),
) -> StandardResponse:
    """纸面组合的业绩、持仓与交易记录 —— 对外展示的可验证记录。"""
    tier = _tier_of(user)
    ent = _ent_for(user, tier)

    if strategy not in SIGNAL_STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}",
        )

    # 管理员绕过付费墙（访问控制），档位展示仍为真实档位
    hint = None if _is_admin(user) else upgrade_required(tier, strategy)
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
    return StandardResponse(
        success=True,
        message="纸面业绩获取成功",
        data=result,
    )


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
            for f in (
                "shares",
                "amount",
                "commission",
                "stamp_tax",
                "slippage",
                "total_cost",
            ):
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
