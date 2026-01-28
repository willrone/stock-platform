"""\
组合策略超参数优化器（Portfolio / Ensemble）

目标：在“收益最大”的前提下，同时对回撤做软惩罚（B 方案），找出更稳的组合。

设计要点：
- 先小样本（随机抽样股票池）跑通闭环，再扩到全市场。
- 用 Optuna 单目标优化：score = annualized_return - lambda_dd * abs(max_drawdown)
- 支持：
  - 选择策略子集（最多 K 个）
  - 子策略参数（先覆盖常用技术策略）
  - 权重（归一化）

注意：当前回测引擎中组合策略名称为 "portfolio"，但 API 层可能尚未完全支持。
本优化器直接调用 BacktestExecutor.run_backtest（内部支持组合检测）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as e:
    logger.error(f"无法导入 optuna 模块: {e}")
    raise

from app.core.config import settings
from app.services.backtest import BacktestConfig, BacktestExecutor


@dataclass
class PortfolioOptConfig:
    # 软惩罚系数：越大越保守
    lambda_drawdown: float = 1.0
    # 组合中最多启用多少个策略（先小一点）
    max_strategies: int = 3
    # 是否允许试验中“禁用”某策略（结构搜索）
    allow_strategy_selection: bool = True


def _normalize_weights(ws: List[float]) -> List[float]:
    s = float(sum(ws))
    if s <= 0:
        return [1.0 / len(ws)] * len(ws)
    return [float(w) / s for w in ws]


class PortfolioHyperparameterOptimizer:
    def __init__(self):
        self._history: Dict[str, Any] = {}

    def _sample_universe(
        self, all_stock_codes: List[str], universe_size: int, seed: int
    ) -> List[str]:
        rng = random.Random(seed)
        if universe_size >= len(all_stock_codes):
            return list(all_stock_codes)
        return rng.sample(list(all_stock_codes), universe_size)

    def _score_soft_balance(
        self,
        annualized_return: float,
        max_drawdown: float,
        cfg: PortfolioOptConfig,
    ) -> float:
        # max_drawdown 通常为负数（如 -0.2），用 abs 做惩罚
        return float(annualized_return) - float(cfg.lambda_drawdown) * abs(float(max_drawdown))

    async def optimize(
        self,
        all_stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        universe_size: int = 200,
        universe_seed: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        cfg: Optional[PortfolioOptConfig] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = cfg or PortfolioOptConfig()

        # 1) 随机抽样股票池（可复现）
        stock_codes = self._sample_universe(all_stock_codes, universe_size, universe_seed)
        logger.info(
            f"组合优化：universe_size={len(stock_codes)} seed={universe_seed} "
            f"date={start_date.date()}~{end_date.date()} trials={n_trials}"
        )

        # 2) 回测执行器
        executor = BacktestExecutor(
            data_dir=str(settings.DATA_ROOT_PATH),
            enable_performance_profiling=False,
        )

        backtest_cfg = BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0003,
            slippage_rate=0.0001,
        )

        # 3) 策略池（先从现有策略列表里挑常用的）
        # NOTE：这里的 key 需与 StrategyFactory 支持的一致
        candidate_strategies = [
            "moving_average",
            "rsi",
            "macd",
            "bollinger",
            "stochastic",
            "cci",
            "mean_reversion",
            "momentum_factor",
            "value_factor",
            "low_volatility",
            "multi_factor",
        ]

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=universe_seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        def objective(trial: optuna.Trial) -> float:
            # 3.1 选择组合结构（最多 K 个）
            chosen: List[str] = []
            if cfg.allow_strategy_selection:
                # 每个策略一个开关，但限制最多 K
                for s in candidate_strategies:
                    use = trial.suggest_categorical(f"use__{s}", [0, 1])
                    if use == 1:
                        chosen.append(s)
                if not chosen:
                    # 至少选一个
                    chosen.append(trial.suggest_categorical("fallback_strategy", candidate_strategies))
                # 裁剪到 max_strategies（保持确定性：按 trial 给的权重排序后取前 K）
                if len(chosen) > cfg.max_strategies:
                    ranked = []
                    for s in chosen:
                        w = trial.suggest_float(f"w_rank__{s}", 0.0, 1.0)
                        ranked.append((w, s))
                    ranked.sort(reverse=True)
                    chosen = [s for _, s in ranked[: cfg.max_strategies]]
            else:
                # 固定取前 K 个（后续可改为配置）
                chosen = candidate_strategies[: cfg.max_strategies]

            # 3.2 为每个选中的策略采样参数 + 权重
            strategies_payload = []
            raw_weights: List[float] = []

            for s in chosen:
                # 权重先采样非负，后归一化
                raw_w = trial.suggest_float(f"weight__{s}", 0.0, 1.0)
                raw_weights.append(raw_w)

                params: Dict[str, Any] = {}
                if s == "moving_average":
                    short = trial.suggest_int(f"{s}__short_window", 3, 20)
                    long = trial.suggest_int(f"{s}__long_window", 10, 120)
                    if long <= short:
                        long = short + 5
                    params = {"short_window": short, "long_window": long}
                elif s == "rsi":
                    params = {
                        "rsi_period": trial.suggest_int(f"{s}__rsi_period", 6, 30),
                        "oversold_threshold": trial.suggest_int(f"{s}__oversold", 10, 40),
                        "overbought_threshold": trial.suggest_int(f"{s}__overbought", 60, 90),
                    }
                elif s == "macd":
                    fast = trial.suggest_int(f"{s}__fast", 5, 20)
                    slow = trial.suggest_int(f"{s}__slow", 15, 60)
                    if slow <= fast:
                        slow = fast + 5
                    signal = trial.suggest_int(f"{s}__signal", 3, 20)
                    params = {"fast_period": fast, "slow_period": slow, "signal_period": signal}
                elif s == "bollinger":
                    params = {
                        "period": trial.suggest_int(f"{s}__period", 10, 40),
                        "std_dev": trial.suggest_float(f"{s}__std_dev", 1.0, 3.0),
                        "entry_threshold": trial.suggest_float(f"{s}__entry_threshold", 0.0, 0.08),
                    }
                elif s == "stochastic":
                    params = {
                        "k_period": trial.suggest_int(f"{s}__k", 5, 21),
                        "d_period": trial.suggest_int(f"{s}__d", 3, 9),
                        "oversold": trial.suggest_int(f"{s}__oversold", 5, 30),
                        "overbought": trial.suggest_int(f"{s}__overbought", 70, 95),
                    }
                elif s == "cci":
                    params = {
                        "period": trial.suggest_int(f"{s}__period", 10, 40),
                        "oversold": trial.suggest_int(f"{s}__oversold", -200, -50),
                        "overbought": trial.suggest_int(f"{s}__overbought", 50, 200),
                    }
                else:
                    # 因子类/高级策略先不展开太多参数：用默认（或很小的参数空间）
                    params = {}

                strategies_payload.append({"name": s, "weight": 0.0, "config": params})

            weights = _normalize_weights(raw_weights)
            for i, w in enumerate(weights):
                strategies_payload[i]["weight"] = w

            # 3.3 组合层参数
            integration_method = trial.suggest_categorical(
                "integration_method", ["weighted_voting"]
            )

            # 3.4 组装 strategy_config
            strategy_config: Dict[str, Any] = {
                "strategies": strategies_payload,
                "integration_method": integration_method,
                # 同时允许优化执行参数（先小范围）
                "commission_rate": trial.suggest_float("commission_rate", 0.0001, 0.001),
                "slippage_rate": trial.suggest_float("slippage_rate", 0.0, 0.001),
            }

            # 3.5 运行回测（在 Optuna 同步 objective 内安全运行 async）
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        executor.run_backtest(
                            strategy_name="portfolio",
                            stock_codes=stock_codes,
                            start_date=start_date,
                            end_date=end_date,
                            strategy_config=strategy_config,
                            backtest_config=BacktestConfig(
                                initial_cash=backtest_cfg.initial_cash,
                                commission_rate=strategy_config.get("commission_rate", backtest_cfg.commission_rate),
                                slippage_rate=strategy_config.get("slippage_rate", backtest_cfg.slippage_rate),
                            ),
                        )
                    )
                finally:
                    new_loop.close()

            try:
                try:
                    asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        backtest_report = pool.submit(run_in_new_loop).result()
                except RuntimeError:
                    backtest_report = run_in_new_loop()
            except Exception as e:
                logger.warning(f"Trial {trial.number} 回测失败: {e}")
                raise

            metrics = backtest_report.get("metrics", {})
            ann = float(metrics.get("annualized_return", 0.0) or 0.0)
            mdd = float(metrics.get("max_drawdown", 0.0) or 0.0)
            score = self._score_soft_balance(ann, mdd, cfg)

            # 记录一些辅助信息（方便你看）
            trial.set_user_attr("annualized_return", ann)
            trial.set_user_attr("max_drawdown", mdd)
            trial.set_user_attr("chosen_strategies", chosen)
            trial.set_user_attr("integration_method", integration_method)

            if progress_callback:
                progress_callback(
                    {
                        "trial": trial.number,
                        "score": score,
                        "annualized_return": ann,
                        "max_drawdown": mdd,
                        "chosen": chosen,
                    }
                )

            return score

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = study.best_trial
        result = {
            "universe_seed": universe_seed,
            "universe_size": len(stock_codes),
            "stock_codes": stock_codes,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "n_trials": n_trials,
            "lambda_drawdown": cfg.lambda_drawdown,
            "best": {
                "score": best.value,
                "params": best.params,
                "attrs": best.user_attrs,
            },
        }
        self._history["last"] = result
        return result
