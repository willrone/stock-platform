"""订阅权限（付费墙）—— 2026-10-25

把「按信号收费」落到代码上：
  free       1 条线（honest_42f），延迟 1 个交易日
  pro        全部 4 条线，当日
  enterprise 全部 4 条线 + 自定义策略线预留

设计原则：
  - **后端强制**，前端只做展示 —— 改前端绕过限制没有意义
  - 免费用户拿到的是**真的旧一天**的信号，不是「藏起来」而是「延迟给出」，
    这样既不破坏体验，也保证付费用户买到的确实是新东西
  - 每次拒绝都带明确的升级路径文案，前端直接展示
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class SignalEntitlement:
    """一个订阅等级能拿到什么。"""

    tier: str
    label: str
    #: 可访问的策略 key；None 表示全部
    strategies: Optional[Set[str]]
    #: 信号延迟（交易日）。0 = 当日
    delay_days: int
    #: 是否包含股数 / 金额明细
    include_execution_detail: bool
    #: 是否可看全部纸面账本
    include_full_ledger: bool
    #: 升级提示
    upgrade_hint: str


# 公开可见的策略集合（与 signal_service.SIGNAL_STRATEGIES 对齐）
ALL_STRATEGIES: Set[str] = {
    "honest_42f",
    "honest_42f_neu",
    "gbm_voltgt",
    "rsi_71421",
    "factor_ranking",
    "22f",
}

ENTITLEMENTS: Dict[str, SignalEntitlement] = {
    "free": SignalEntitlement(
        tier="free",
        label="体验版",
        strategies={"honest_42f"},
        delay_days=1,
        include_execution_detail=False,
        include_full_ledger=False,
        upgrade_hint="升级信号版可解锁全部 4 条信号线、当日信号与完整执行清单",
    ),
    "pro": SignalEntitlement(
        tier="pro",
        label="信号版",
        strategies=None,          # 全部
        delay_days=0,
        include_execution_detail=True,
        include_full_ledger=True,
        upgrade_hint="",
    ),
    "enterprise": SignalEntitlement(
        tier="enterprise",
        label="专业版",
        strategies=None,
        delay_days=0,
        include_execution_detail=True,
        include_full_ledger=True,
        upgrade_hint="",
    ),
}

DEFAULT_TIER = "free"


def normalize_tier(tier: Optional[str]) -> str:
    t = (tier or "").strip().lower()
    if t in ENTITLEMENTS:
        return t
    # 兼容历史数据里可能出现的其它写法
    if t in ("basic", "trial", ""):
        return DEFAULT_TIER
    return DEFAULT_TIER


def entitlement_for(tier: Optional[str]) -> SignalEntitlement:
    return ENTITLEMENTS[normalize_tier(tier)]


def allowed_strategies(tier: Optional[str]) -> List[str]:
    ent = entitlement_for(tier)
    if ent.strategies is None:
        return sorted(ALL_STRATEGIES)
    return sorted(ent.strategies)


def can_access_strategy(tier: Optional[str], strategy: str) -> bool:
    ent = entitlement_for(tier)
    if ent.strategies is None:
        return strategy in ALL_STRATEGIES
    return strategy in ent.strategies


def upgrade_required(tier: Optional[str], strategy: str) -> Optional[str]:
    """返回升级提示；None 表示有权限。"""
    if can_access_strategy(tier, strategy):
        return None
    ent = entitlement_for(tier)
    return (
        f"「{strategy}」是信号版功能。{ent.upgrade_hint}"
    )


def effective_delay_days(tier: Optional[str]) -> int:
    return entitlement_for(tier).delay_days
