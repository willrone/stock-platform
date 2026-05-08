"""Extensible trade-mode executors for cross-sectional portfolio strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager import PortfolioManager
from ..models import SignalType, TradingSignal


@dataclass(frozen=True)
class TradeModeExecutionContext:
    current_date: datetime
    all_signals: List[TradingSignal]
    current_prices: Dict[str, float]
    portfolio_manager: PortfolioManager
    strategy: Optional[BaseStrategy]
    strategy_config: Dict[str, Any]
    stock_universe: List[str]


@dataclass(frozen=True)
class TradeModeExecutionResult:
    executed_trade_signals: List[Dict[str, Any]]
    unexecuted_signals: List[Dict[str, Any]]
    trades_this_day: int
    signal_records: List[TradingSignal]


class TradeModeExecutor(Protocol):
    mode_name: str

    def execute(self, context: TradeModeExecutionContext) -> TradeModeExecutionResult:
        ...


def _build_execution_result(
    executed_trade_signals: List[Dict[str, Any]],
    unexecuted_signals: List[Dict[str, Any]],
    trades_this_day: int,
    signal_records: List[TradingSignal],
) -> TradeModeExecutionResult:
    return TradeModeExecutionResult(
        executed_trade_signals=executed_trade_signals,
        unexecuted_signals=unexecuted_signals,
        trades_this_day=trades_this_day,
        signal_records=signal_records,
    )


def _resolve_signal_score(signal: TradingSignal) -> float:
    metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
    explicit_score = metadata.get("ranking_score")
    if explicit_score is not None:
        return float(explicit_score)

    if signal.signal_type == SignalType.BUY:
        return float(signal.strength or 0.0)
    if signal.signal_type == SignalType.SELL:
        return -float(signal.strength or 0.0)
    return float(signal.strength or 0.0)


def _extract_scores(context: TradeModeExecutionContext) -> Dict[str, float]:
    tradeable_codes = set(context.current_prices.keys())
    scores: Dict[str, float] = {}
    for signal in context.all_signals:
        if signal.stock_code not in tradeable_codes:
            continue
        score = _resolve_signal_score(signal)
        current = scores.get(signal.stock_code, float("-inf"))
        if score > current:
            scores[signal.stock_code] = score
    return scores


def _validate_and_execute(
    *,
    context: TradeModeExecutionContext,
    signal: TradingSignal,
    executed_trade_signals: List[Dict[str, Any]],
    unexecuted_signals: List[Dict[str, Any]],
) -> bool:
    current_price = context.current_prices.get(signal.stock_code, signal.price)
    if current_price is None or float(current_price) <= 0:
        unexecuted_signals.append(
            {
                "stock_code": signal.stock_code,
                "timestamp": signal.timestamp,
                "signal_type": signal.signal_type.name,
                "execution_reason": "缺少当日价格",
            }
        )
        return False

    strategy = context.strategy
    if strategy is not None:
        is_valid, validation_reason = strategy.validate_signal(
            signal,
            context.portfolio_manager.get_portfolio_value(context.current_prices),
            context.portfolio_manager.positions,
        )
        if not is_valid:
            unexecuted_signals.append(
                {
                    "stock_code": signal.stock_code,
                    "timestamp": signal.timestamp,
                    "signal_type": signal.signal_type.name,
                    "execution_reason": validation_reason or "信号验证失败",
                }
            )
            return False

    trade, failure_reason = context.portfolio_manager.execute_signal(
        signal, context.current_prices
    )
    if trade:
        executed_trade_signals.append(
            {
                "stock_code": signal.stock_code,
                "timestamp": signal.timestamp,
                "signal_type": signal.signal_type.name,
            }
        )
        return True

    unexecuted_signals.append(
        {
            "stock_code": signal.stock_code,
            "timestamp": signal.timestamp,
            "signal_type": signal.signal_type.name,
            "execution_reason": failure_reason or "执行失败（未知原因）",
        }
    )
    return False


def _build_rebalance_signal(
    *,
    current_date: datetime,
    stock_code: str,
    current_prices: Dict[str, float],
    signal_type: SignalType,
    reason: str,
    trade_mode: str,
) -> TradingSignal:
    return TradingSignal(
        timestamp=current_date,
        stock_code=stock_code,
        signal_type=signal_type,
        strength=1.0,
        price=float(current_prices.get(stock_code, 0.0) or 0.0),
        reason=reason,
        metadata={"trade_mode": trade_mode},
    )


class TopkDropoutTradeModeExecutor:
    """Official-style TopK/Dropout rotation using daily ranking scores."""

    mode_name = "topk_dropout"

    def execute(self, context: TradeModeExecutionContext) -> TradeModeExecutionResult:
        executed_trade_signals: List[Dict[str, Any]] = []
        unexecuted_signals: List[Dict[str, Any]] = []
        signal_records: List[TradingSignal] = []
        trades_this_day = 0

        topk = int(context.strategy_config.get("topk", 10))
        n_drop = int(context.strategy_config.get("n_drop", 2))
        hold_thresh = max(0, int(context.strategy_config.get("hold_thresh", 0) or 0))
        if topk <= 0 or n_drop <= 0:
            return _build_execution_result(
                executed_trade_signals,
                unexecuted_signals,
                trades_this_day,
                signal_records,
            )

        scores = _extract_scores(context)
        if not scores:
            return _build_execution_result(
                executed_trade_signals,
                unexecuted_signals,
                trades_this_day,
                signal_records,
            )

        ranked = sorted(
            scores.items(), key=lambda item: (item[1], item[0]), reverse=True
        )
        holdings = list(context.portfolio_manager.positions.keys())
        holdings_set = set(holdings)
        tradeable_codes = set(scores.keys())

        # Initial build: directly fill the portfolio with topk ranked names.
        if len(holdings) < topk:
            buy_candidates = [code for code, _ in ranked if code not in holdings_set][
                : topk - len(holdings)
            ]
            for code in buy_candidates:
                signal = _build_rebalance_signal(
                    current_date=context.current_date,
                    stock_code=code,
                    current_prices=context.current_prices,
                    signal_type=SignalType.BUY,
                    reason=f"topk_dropout initial buy (enter top{topk})",
                    trade_mode=self.mode_name,
                )
                signal_records.append(signal)
                if _validate_and_execute(
                    context=context,
                    signal=signal,
                    executed_trade_signals=executed_trade_signals,
                    unexecuted_signals=unexecuted_signals,
                ):
                    trades_this_day += 1
            return _build_execution_result(
                executed_trade_signals,
                unexecuted_signals,
                trades_this_day,
                signal_records,
            )

        rank_index = {code: idx for idx, (code, _) in enumerate(ranked)}
        held_sorted_best_to_worst = sorted(
            holdings,
            key=lambda code: rank_index.get(code, len(ranked) + len(holdings)),
        )
        sell_candidates = [
            code
            for code in reversed(held_sorted_best_to_worst)
            if code in tradeable_codes
            and rank_index.get(code, len(ranked)) >= topk + hold_thresh
        ][:n_drop]
        buy_candidates = [code for code, _ in ranked if code not in holdings_set][
            :n_drop
        ]

        successful_sells = 0
        for code in sell_candidates:
            signal = _build_rebalance_signal(
                current_date=context.current_date,
                stock_code=code,
                current_prices=context.current_prices,
                signal_type=SignalType.SELL,
                reason=f"topk_dropout sell worst-ranked holding (n_drop={n_drop})",
                trade_mode=self.mode_name,
            )
            signal_records.append(signal)
            if _validate_and_execute(
                context=context,
                signal=signal,
                executed_trade_signals=executed_trade_signals,
                unexecuted_signals=unexecuted_signals,
            ):
                successful_sells += 1
                trades_this_day += 1

        for code in buy_candidates[:successful_sells]:
            if len(context.portfolio_manager.positions) >= topk:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": context.current_date,
                        "signal_type": SignalType.BUY.name,
                        "execution_reason": f"超过topk持仓上限(topk={topk})，跳过买入",
                    }
                )
                break

            signal = _build_rebalance_signal(
                current_date=context.current_date,
                stock_code=code,
                current_prices=context.current_prices,
                signal_type=SignalType.BUY,
                reason=f"topk_dropout buy highest-ranked replacement (n_drop={n_drop})",
                trade_mode=self.mode_name,
            )
            signal_records.append(signal)
            if _validate_and_execute(
                context=context,
                signal=signal,
                executed_trade_signals=executed_trade_signals,
                unexecuted_signals=unexecuted_signals,
            ):
                trades_this_day += 1

        return _build_execution_result(
            executed_trade_signals,
            unexecuted_signals,
            trades_this_day,
            signal_records,
        )


_TRADE_MODE_EXECUTORS: Dict[str, TradeModeExecutor] = {
    TopkDropoutTradeModeExecutor.mode_name: TopkDropoutTradeModeExecutor(),
}


def get_trade_mode_executor(trade_mode: Optional[str]) -> Optional[TradeModeExecutor]:
    if not trade_mode:
        return None
    return _TRADE_MODE_EXECUTORS.get(str(trade_mode).lower())
