"""
策略超参数优化器

使用 Optuna 对策略参数进行优化，支持单目标和多目标优化

优化特性（2026-02 增强）：
- SQLite 持久化存储，支持断点续跑
- 多进程并行优化（n_jobs 可配置）
- 激进剪枝策略（HyperbandPruner）
- 数据预加载缓存
"""

import asyncio
import concurrent.futures
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import NSGAIISampler, TPESampler
    from optuna.storages import RDBStorage
except ImportError as e:
    logger.error(f"无法导入 optuna 模块: {e}")
    logger.error("请运行: pip install optuna>=3.4.0")
    raise ImportError(
        "optuna 模块未安装。超参优化功能需要 optuna 库。" "请运行: pip install optuna>=3.4.0"
    ) from e

from app.core.config import settings
from app.services.backtest import BacktestConfig, BacktestExecutor
from app.services.backtest.optimization.data_cache import get_data_cache


class StrategyHyperparameterOptimizer:
    """策略超参数优化器"""

    def __init__(self, n_jobs: int = 4, use_persistent_storage: bool = True):
        """
        初始化优化器
        
        Args:
            n_jobs: 并行进程数（默认 4）
            use_persistent_storage: 是否使用 SQLite 持久化存储（支持断点续跑）
        """
        self.optimization_history = {}
        self.n_jobs = n_jobs
        self.use_persistent_storage = use_persistent_storage
        self._data_cache = get_data_cache()
        
        # 确保 optuna 存储目录存在
        self._storage_dir = os.path.join(settings.DATA_ROOT_PATH, "optuna_studies")
        os.makedirs(self._storage_dir, exist_ok=True)
        
        logger.info(f"StrategyHyperparameterOptimizer 初始化: n_jobs={n_jobs}, persistent={use_persistent_storage}")

    async def optimize_strategy_parameters(
        self,
        strategy_name: str,
        param_space: Dict[str, Any],
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        objective_config: Dict[str, Any],
        backtest_config: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        optimization_method: str = "tpe",
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        优化策略参数

        Args:
            strategy_name: 策略名称
            param_space: 参数空间定义
            stock_codes: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            objective_config: 目标函数配置
                - objective_metric:
                    单目标:
                        "sharpe"             夏普比率
                        "calmar"             卡玛比率（年化收益 / 最大回撤）
                        "ic"                 信息系数近似（当前用胜率近似）
                        "ic_ir"              信息比率（使用无成本组合的 IR）
                        "total_return"       总收益率
                        "annualized_return"  年化收益率
                        "win_rate"           胜率
                        "profit_factor"      盈亏比（Profit Factor）
                        "max_drawdown"       最大回撤（内部转换为“越小越好”的得分）
                        "cost"               交易成本（手续费+滑点，占初始资金比例，越低越好）
                        "custom"             自定义加权组合
                    多目标:
                        由上述字符串组成的列表，例如 ["sharpe", "calmar", "ic"]
                - objective_weights: 自定义权重（custom 时使用）
                - direction: "maximize" | "minimize"
            backtest_config: 回测配置（初始资金、手续费等）
            n_trials: 试验次数
            optimization_method: 优化方法 ("tpe", "random", "grid", "nsga2", "motpe")
            timeout: 超时时间（秒）
            progress_callback: 进度回调函数

        Returns:
            Dict: 优化结果
        """
        logger.info(
            f"开始策略参数优化: {strategy_name}, 方法: {optimization_method}, 试验次数: {n_trials}"
        )

        start_time = datetime.utcnow()

        # 解析目标配置
        objective_metric = objective_config.get("objective_metric", "sharpe")
        is_multi_objective = (
            isinstance(objective_metric, list) and len(objective_metric) > 1
        )

        # 预加载数据到缓存（避免每个 trial 重复加载）
        logger.info(f"预加载股票数据: {len(stock_codes)} 只股票")
        await self._data_cache.preload_async(stock_codes, start_date, end_date)
        logger.info("数据预加载完成")

        # 创建 SQLite 存储（支持断点续跑和并行）
        #
        # ⚠️ 重要：study_name 不能只用 strategy_name + 日期区间。
        # 否则不同 param_space / 不同实验会复用同一个 optuna study，导致“0 trials 就完成/复用旧最优值”。
        # 这里加入 param_space 的稳定签名，确保不同实验隔离，同时仍支持断点续跑。
        storage = None
        import hashlib
        import json

        base_study_name = f"{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        try:
            sig_src = json.dumps(param_space, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            param_sig = hashlib.sha1(sig_src).hexdigest()[:8]
        except Exception:
            param_sig = "nosig"

        study_name = f"{base_study_name}_{param_sig}"

        if self.use_persistent_storage:
            storage_path = os.path.join(self._storage_dir, f"{study_name}.db")
            storage = RDBStorage(
                url=f"sqlite:///{storage_path}",
                engine_kwargs={"connect_args": {"timeout": 30}}
            )
            logger.info(f"使用 SQLite 存储: {storage_path}")
            logger.info(f"Optuna study_name: {study_name} (base={base_study_name}, sig={param_sig})")

        # 创建激进剪枝器（比 MedianPruner 更早终止差的 trial）
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=n_trials,
            reduction_factor=3
        )

        # 创建 Optuna study
        if is_multi_objective:
            # 多目标优化
            directions = [
                "maximize"
                if objective_config.get("direction", "maximize") == "maximize"
                else "minimize"
            ] * len(objective_metric)
            if optimization_method == "nsga2":
                sampler = NSGAIISampler()
            elif optimization_method == "motpe":
                # MOTPESampler 在 optuna 4.6.0 中不存在，使用 NSGA-II 替代
                logger.warning("MOTPESampler 不可用，使用 NSGAIISampler 替代")
                sampler = NSGAIISampler()
            else:
                sampler = NSGAIISampler()  # 默认使用 NSGA-II
            study = optuna.create_study(
                study_name=study_name,
                directions=directions,
                sampler=sampler,
                storage=storage,
                load_if_exists=True  # 支持断点续跑
            )
        else:
            # 单目标优化
            direction = objective_config.get("direction", "maximize")
            if optimization_method == "tpe":
# 启用 multivariate 模式，学习参数间相关性，提升收敛速度
                sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=10)
            elif optimization_method == "random":
                sampler = optuna.samplers.RandomSampler(seed=42)
            else:
                sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=10)

            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True  # 支持断点续跑
            )

        # 注入先验知识：将默认参数作为第一个 trial，加速收敛
        default_params = {}
        for param_name, param_config in param_space.items():
            if param_config.get("enabled", True) and "default" in param_config:
                default_params[param_name] = param_config["default"]
        if default_params and not is_multi_objective:
            try:
                study.enqueue_trial(default_params)
                logger.info(f"注入默认参数作为初始 trial: {list(default_params.keys())}")
            except Exception as e:
                logger.warning(f"注入默认参数失败: {e}")

        # 断点续跑：扣除已完成的 trials，避免重复执行
        existing_completed = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if existing_completed > 0:
            remaining = max(0, n_trials - existing_completed)
            logger.info(
                f"断点续跑: 已有 {existing_completed} 个已完成 trials，"
                f"原计划 {n_trials} 个，剩余需执行 {remaining} 个"
            )
            n_trials = remaining
            if n_trials <= 0:
                logger.info("所有 trials 已完成，无需继续优化")
                # 直接返回已有结果，避免 n_trials=0 导致后续除零错误
                completed_trials = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                best_trial = study.best_trial if not is_multi_objective and completed_trials else None
                return {
                    "success": True,
                    "strategy_name": strategy_name,
                    "best_params": best_trial.params if best_trial else None,
                    "best_score": best_trial.value if best_trial else None,
                    "best_trial_number": best_trial.number if best_trial else None,
                    "objective_metric": objective_metric,
                    "n_trials": existing_completed,
                    "completed_trials": existing_completed,
                    "running_trials": 0,
                    "pruned_trials": len([
                        t for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]),
                    "failed_trials": len([
                        t for t in study.trials
                        if t.state == optuna.trial.TrialState.FAIL
                    ]),
                    "message": f"断点续跑: 所有 {existing_completed} 个 trials 已完成，无需继续",
                }

        # 创建回测执行器
        try:
            enable_perf = os.getenv(
                "ENABLE_BACKTEST_PERFORMANCE_PROFILING", "false"
            ).strip().lower() in {"1", "true", "yes", "y", "on"}
            executor = BacktestExecutor(
                data_dir=str(settings.DATA_ROOT_PATH),
                enable_performance_profiling=enable_perf,
            )
            logger.info(f"回测执行器已创建，数据目录: {settings.DATA_ROOT_PATH}")
        except Exception as e:
            logger.error(f"创建回测执行器失败: {e}", exc_info=True)
            raise

        # 默认回测配置
        if backtest_config is None:
            backtest_config = {
                "initial_cash": 100000.0,
                "commission_rate": 0.0003,
                "slippage_rate": 0.0001,
            }

        backtest_cfg = BacktestConfig(
            initial_cash=backtest_config.get("initial_cash", 100000.0),
            commission_rate=backtest_config.get("commission_rate", 0.0003),
            slippage_rate=backtest_config.get("slippage_rate", 0.0001),
        )

        # 定义目标函数
        def objective(trial: optuna.Trial):
            try:
                # 从参数空间采样参数
                strategy_params = {}
                for param_name, param_config in param_space.items():
                    if not param_config.get("enabled", True):
                        # 使用默认值
                        strategy_params[param_name] = param_config.get("default")
                        continue

                    param_type = param_config.get("type", "float")
                    if param_type == "int":
                        strategy_params[param_name] = trial.suggest_int(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=param_config.get("log", False),
                        )
                    elif param_type == "float":
                        strategy_params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=param_config.get("log", False),
                        )
                    elif param_type == "categorical":
                        strategy_params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )

                # 记录采样到的参数（用于调试）
                logger.info(f"Trial {trial.number}: 采样参数 = {strategy_params}")

                # Build strategy_config for backtest.
                # - single strategy: use sampled params directly
                # - portfolio strategy: expand flattened params into portfolio config
                strategy_config_payload: Dict[str, Any]
                if str(strategy_name) == "portfolio":
                    # portfolio config (trade_mode + ensemble strategies)
                    # Plan A: trade_mode/topk/buffer/max_changes_per_day 固定，不进入 param_space。
                    topk = int((backtest_config or {}).get("topk", 10) or 10)
                    buffer_n = int((backtest_config or {}).get("buffer", 20) or 20)
                    max_changes = int((backtest_config or {}).get("max_changes_per_day", 2) or 2)
                    integration_method = strategy_params.get("integration_method") or "weighted_voting"

                    # gather enabled sub-strategies
                    strategies_list = []
                    for k, v in list(strategy_params.items()):
                        if not str(k).startswith("use__"):
                            continue
                        sk = str(k).split("use__", 1)[1]
                        try:
                            enabled = int(v) == 1
                        except Exception:
                            enabled = bool(v)
                        if not enabled:
                            continue

                        # weight
                        w = strategy_params.get(f"weight__{sk}")
                        try:
                            w = float(w) if w is not None else 0.5
                        except Exception:
                            w = 0.5

                        # nested params: <sk>__<param>
                        sub_cfg: Dict[str, Any] = {}
                        prefix = f"{sk}__"
                        for pk, pv in strategy_params.items():
                            if str(pk).startswith(prefix):
                                sub_cfg[str(pk)[len(prefix) :]] = pv

                        strategies_list.append({"name": sk, "weight": w, "config": sub_cfg})

                    # fallback: ensure at least 1
                    if not strategies_list:
                        # If user didn't include use__* in param_space, treat any prefixed params as enabled.
                        inferred = set()
                        for pk in strategy_params.keys():
                            if "__" in str(pk):
                                inferred.add(str(pk).split("__", 1)[0])
                        for sk in sorted(inferred):
                            sub_cfg: Dict[str, Any] = {}
                            prefix = f"{sk}__"
                            for pk, pv in strategy_params.items():
                                if str(pk).startswith(prefix):
                                    sub_cfg[str(pk)[len(prefix) :]] = pv
                            strategies_list.append({"name": sk, "weight": 1.0 / max(1, len(inferred)), "config": sub_cfg})

                    strategy_config_payload = {
                        "integration_method": integration_method,
                        "trade_mode": "topk_buffer",
                        "topk": topk,
                        "buffer": buffer_n,
                        "max_changes_per_day": max_changes,
                        "strategies": strategies_list,
                    }
                else:
                    strategy_config_payload = strategy_params

                # 运行回测（在同步函数中运行异步代码）
                # 在 Optuna 的 trial 函数中，需要安全地运行异步代码
                # 使用新的事件循环，避免与外部事件循环冲突
                try:
                    # 尝试获取当前运行中的事件循环
                    loop = asyncio.get_running_loop()
                    # 如果已经有运行中的循环，在新线程中运行
                    with concurrent.futures.ThreadPoolExecutor() as executor_pool:

                        def run_in_new_loop():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(
                                    executor.run_backtest(
                                        strategy_name=strategy_name,
                                        stock_codes=stock_codes,
                                        start_date=start_date,
                                        end_date=end_date,
                                        strategy_config=strategy_config_payload,
                                        backtest_config=backtest_cfg,
                                    )
                                )
                            finally:
                                new_loop.close()

                        future = executor_pool.submit(run_in_new_loop)
                        backtest_report = future.result()
                except RuntimeError:
                    # 如果没有运行中的循环，创建新的事件循环
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        backtest_report = new_loop.run_until_complete(
                            executor.run_backtest(
                                strategy_name=strategy_name,
                                stock_codes=stock_codes,
                                start_date=start_date,
                                end_date=end_date,
                                strategy_config=strategy_config_payload,
                                backtest_config=backtest_cfg,
                            )
                        )
                    finally:
                        new_loop.close()

                # 计算目标函数值
                # 记录回测结果的关键指标（用于调试）
                metrics = backtest_report.get("metrics", {})
                logger.info(
                    f"Trial {trial.number}: 回测指标 - sharpe_ratio={metrics.get('sharpe_ratio', 0):.4f}, "
                    f"total_return={metrics.get('total_return', 0):.4f}, "
                    f"annualized_return={metrics.get('annualized_return', 0):.4f}, "
                    f"max_drawdown={metrics.get('max_drawdown', 0):.4f}"
                )

                if is_multi_objective:
                    # 多目标：返回多个值
                    objectives = []
                    for metric in objective_metric:
                        score = self._calculate_objective_score(
                            backtest_report,
                            metric,
                            objective_config.get("objective_weights"),
                        )
                        objectives.append(score)
                    logger.info(f"Trial {trial.number}: 多目标得分 = {objectives}")

                    # 更新进度（多目标优化）
                    if progress_callback:
                        # 计算 trial 统计信息
                        completed_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE
                            ]
                        )
                        running_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.RUNNING
                            ]
                        )
                        pruned_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED
                            ]
                        )
                        failed_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.FAIL
                            ]
                        )
                        trial_num = trial.number + 1  # trial.number 从 0 开始，所以 +1 得到当前编号
                        progress_callback(
                            trial_num,
                            n_trials,
                            strategy_params,
                            None,  # 多目标没有单一得分
                            backtest_report,
                            completed_trials=completed_trials,
                            running_trials=running_trials,
                            pruned_trials=pruned_trials,
                            failed_trials=failed_trials,
                            best_score=None,  # 多目标没有单一最佳得分
                            best_trial_number=None,
                            best_params=None,
                        )

                    return tuple(objectives)
                else:
                    # 单目标：返回单个值
                    score = self._calculate_objective_score(
                        backtest_report,
                        objective_metric,
                        objective_config.get("objective_weights"),
                    )

                    # 记录 stability 的分解细节到 trial.user_attrs，便于前端/接口展示。
                    # 同时确保 score 永远是可比较数值（避免 NaN -> null）。
                    direction = str(objective_config.get("direction", "maximize")).lower()
                    if objective_metric == "stability":
                        try:
                            import math

                            # 复用 stability 计算逻辑（从 backtest_report 里提取 OOS 分量）
                            history = backtest_report.get("portfolio_history") or backtest_report.get(
                                "portfolioHistory"
                            )
                            oos_ratio = float(
                                (backtest_report.get("stability_config") or {}).get("oos_ratio", 0.3)
                            )
                            oos_ratio = max(0.05, min(0.5, oos_ratio))

                            raw_dates = [
                                str(h.get("date") or h.get("snapshot_date") or "") for h in (history or [])
                            ]
                            raw_values = [
                                float(h.get("portfolio_value") or h.get("portfolioValue") or 0.0)
                                for h in (history or [])
                            ]
                            dates = []
                            values = []
                            for d, v in zip(raw_dates, raw_values):
                                if isinstance(v, (int, float)) and math.isfinite(v):
                                    dates.append(d)
                                    values.append(float(v))

                            details = {
                                "oos_ratio": oos_ratio,
                                "oos_total_return": None,
                                "oos_max_drawdown": None,
                                "oos_pos_month_ratio": None,
                                "oos_monthly_std": None,
                                "ret_score": None,
                                "dd_score": None,
                                "pm_score": None,
                                "std_score": None,
                                "blend": None,
                            }

                            if len(values) >= 10:
                                n = len(values)
                                split = max(1, min(n - 1, int(n * (1.0 - oos_ratio))))
                                oos_dates = dates[split:]
                                oos_values = values[split:]

                                # total return
                                total_ret_oos = (
                                    (oos_values[-1] / oos_values[0] - 1.0) if oos_values and oos_values[0] else 0.0
                                )

                                # max drawdown
                                peak = oos_values[0] if oos_values else 0.0
                                mdd = 0.0
                                for v in oos_values:
                                    if v > peak:
                                        peak = v
                                    dd = (v / peak - 1.0) if peak else 0.0
                                    if dd < mdd:
                                        mdd = dd
                                mdd_oos = abs(mdd)

                                # monthly returns
                                first = {}
                                last = {}
                                for d, v in zip(oos_dates, oos_values):
                                    if not (isinstance(v, (int, float)) and math.isfinite(v)):
                                        continue
                                    m = str(d)[:7]
                                    if m not in first:
                                        first[m] = float(v)
                                    last[m] = float(v)
                                mrets = []
                                for m in sorted(last.keys()):
                                    f = first.get(m)
                                    l = last.get(m)
                                    if f and f != 0 and math.isfinite(f) and math.isfinite(l):
                                        mrets.append(l / f - 1.0)

                                if mrets:
                                    pos_month_ratio = sum(1 for r in mrets if r > 0) / len(mrets)
                                    mean = sum(mrets) / len(mrets)
                                    var = sum((r - mean) ** 2 for r in mrets) / len(mrets)
                                    mstd = var ** 0.5
                                else:
                                    pos_month_ratio = 0.0
                                    mstd = 0.0

                                ret_score = max(0.0, min(1.0, (total_ret_oos + 0.3) / 0.9))
                                dd_score = 1.0 - min(1.0, mdd_oos / 0.6)
                                std_score = 1.0 - min(1.0, mstd / 0.10)
                                pm_score = max(0.0, min(1.0, pos_month_ratio))
                                blend = 0.45 * ret_score + 0.30 * dd_score + 0.15 * pm_score + 0.10 * std_score

                                details.update(
                                    {
                                        "oos_total_return": total_ret_oos,
                                        "oos_max_drawdown": mdd_oos,
                                        "oos_pos_month_ratio": pos_month_ratio,
                                        "oos_monthly_std": mstd,
                                        "ret_score": ret_score,
                                        "dd_score": dd_score,
                                        "pm_score": pm_score,
                                        "std_score": std_score,
                                        "blend": blend,
                                    }
                                )

                            trial.set_user_attr("objective_details", details)
                        except Exception:
                            # 细节计算失败不应影响 trial
                            pass

                    # score 兜底：NaN/None -> 最差值，避免 API 中出现 null
                    try:
                        import math

                        if score is None or not (isinstance(score, (int, float)) and math.isfinite(score)):
                            score = float("-inf") if direction == "maximize" else float("inf")
                    except Exception:
                        score = float("-inf") if direction == "maximize" else float("inf")

                    logger.info(
                        f"Trial {trial.number}: 目标得分 = {score:.6f} (原始指标: {objective_metric})"
                    )

                    # 更新进度
                    if progress_callback:
                        # 计算 trial 统计信息
                        completed_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE
                            ]
                        )
                        running_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.RUNNING
                            ]
                        )
                        pruned_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED
                            ]
                        )
                        failed_trials = len(
                            [
                                t
                                for t in study.trials
                                if t.state == optuna.trial.TrialState.FAIL
                            ]
                        )
                        trial_num = trial.number + 1  # trial.number 从 0 开始，所以 +1 得到当前编号
                        # 注意：在 objective 回调里，当前 trial 还未被 Optuna 标记为 COMPLETE，
                        # 直接访问 study.best_trial 可能抛出 "No trials are completed yet"。
                        best_score = None
                        best_trial_number = None
                        best_params = None
                        if not is_multi_objective:
                            try:
                                best_score = study.best_value
                                best_trial_number = study.best_trial.number
                                best_params = study.best_params
                            except Exception:
                                # fallback：至少给出当前 trial 的信息，避免把本 trial 误判为失败
                                best_score = score
                                best_trial_number = trial.number
                                best_params = strategy_params

                        progress_callback(
                            trial_num,
                            n_trials,
                            strategy_params,
                            score,
                            backtest_report,
                            completed_trials=completed_trials,
                            running_trials=running_trials,
                            pruned_trials=pruned_trials,
                            failed_trials=failed_trials,
                            best_score=best_score,
                            best_trial_number=best_trial_number,
                            best_params=best_params,
                        )

                    return score

            except Exception as e:
                logger.error(f"Trial {trial.number} 失败: {e}", exc_info=True)
                logger.error(f"Trial {trial.number} 参数: {strategy_params}")
                # 返回最差分数
                if is_multi_objective:
                    return tuple(
                        [
                            float("-inf")
                            if objective_config.get("direction", "maximize")
                            == "maximize"
                            else float("inf")
                        ]
                        * len(objective_metric)
                    )
                else:
                    return (
                        float("-inf")
                        if objective_config.get("direction", "maximize") == "maximize"
                        else float("inf")
                    )

        # 执行优化
        try:
            logger.info(
                f"开始执行优化，策略: {strategy_name}, 股票: {stock_codes}, 日期范围: {start_date} - {end_date}"
            )
            logger.info(f"参数空间: {list(param_space.keys())}")
            logger.info(
                f"目标函数: {objective_metric}, 方向: {objective_config.get('direction', 'maximize')}"
            )

            # 初始化进度状态（在优化开始前）
            if progress_callback:
                progress_callback(
                    0,
                    n_trials,
                    {},
                    None,
                    {},
                    completed_trials=0,
                    running_trials=0,
                    pruned_trials=0,
                    failed_trials=0,
                    best_score=None,
                    best_trial_number=None,
                    best_params=None,
                )

            # 启用并行优化（n_jobs 控制并发数）
            # 注意：SQLite 存储支持多进程并发访问
            logger.info(f"开始优化: n_trials={n_trials}, n_jobs={self.n_jobs}, timeout={timeout}")
            study.optimize(
                objective, 
                n_trials=n_trials, 
                timeout=timeout, 
                n_jobs=self.n_jobs,  # 并行执行
                show_progress_bar=False
            )

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # 计算参数重要性（仅单目标）
            param_importance = {}
            if not is_multi_objective and len(study.trials) > 0:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    param_importance = importance
                except Exception as e:
                    logger.warning(f"计算参数重要性失败: {e}")

            # 构建优化历史
            optimization_history = []
            for trial in study.trials:
                trial_data = {
                    "trial_number": trial.number,
                    "params": trial.params,
                    "state": trial.state.name.lower(),
                    "duration_seconds": trial.duration.total_seconds()
                    if trial.duration
                    else None,
                    "timestamp": trial.datetime_start.isoformat()
                    if trial.datetime_start
                    else None,
                }

                if trial.state == optuna.trial.TrialState.COMPLETE:
                    if is_multi_objective:
                        trial_data["objectives"] = trial.values
                    else:
                        trial_data["score"] = trial.value
                        # 附加 objective 的分解细节（例如 stability 的各分量）
                        if isinstance(trial.user_attrs, dict) and trial.user_attrs.get(
                            "objective_details"
                        ):
                            trial_data["objective_details"] = trial.user_attrs.get(
                                "objective_details"
                            )

                optimization_history.append(trial_data)

            # 构建结果
            result = {
                "success": True,
                "strategy_name": strategy_name,
                "best_params": study.best_params
                if not is_multi_objective and len(study.trials) > 0
                else None,
                "best_score": study.best_value
                if not is_multi_objective and len(study.trials) > 0
                else None,
                "best_trial_number": study.best_trial.number
                if not is_multi_objective and len(study.trials) > 0
                else None,
                "objective_metric": objective_metric,
                "n_trials": n_trials,
                "completed_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
                "running_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.RUNNING
                    ]
                ),
                "pruned_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
                "failed_trials": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                ),
                "optimization_history": optimization_history,
                "param_importance": param_importance,
                "optimization_metadata": {
                    "method": optimization_method,
                    "direction": objective_config.get("direction", "maximize"),
                    "duration_seconds": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "data_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                },
            }

            # 多目标优化：添加 Pareto front
            if is_multi_objective:
                # 计算 Pareto front（非支配解）
                try:
                    completed_trials = [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                        and t.values is not None
                    ]

                    if len(completed_trials) > 0:
                        # 计算非支配解（Pareto front）
                        pareto_front = []
                        for trial in completed_trials:
                            is_dominated = False
                            for other_trial in completed_trials:
                                if trial.number == other_trial.number:
                                    continue
                                # 检查是否被支配：如果另一个解在所有目标上都更好或相等，且至少有一个目标更好
                                if all(
                                    other_trial.values[i] >= trial.values[i]
                                    for i in range(len(trial.values))
                                ):
                                    if any(
                                        other_trial.values[i] > trial.values[i]
                                        for i in range(len(trial.values))
                                    ):
                                        is_dominated = True
                                        break
                            if not is_dominated:
                                pareto_front.append(
                                    {
                                        "trial_number": trial.number,
                                        "params": trial.params,
                                        "objectives": list(trial.values),
                                    }
                                )
                        result["pareto_front"] = pareto_front
                    else:
                        result["pareto_front"] = []
                except Exception as e:
                    logger.warning(f"计算 Pareto front 失败: {e}")
                    result["pareto_front"] = []

            logger.info(
                f"策略参数优化完成: {strategy_name}, 最佳得分: {result.get('best_score', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.error(f"策略参数优化失败: {e}", exc_info=True)
            return {"success": False, "error": str(e), "strategy_name": strategy_name}

    def _calculate_objective_score(
        self,
        backtest_report: Dict[str, Any],
        objective_metric: str,
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        计算目标函数得分

        Args:
            backtest_report: 回测报告
            objective_metric: 目标指标 ("sharpe", "calmar", "ic", "ic_ir", "total_return",
                                       "annualized_return", "win_rate", "profit_factor",
                                       "max_drawdown", "cost", "custom")
            objective_weights: 自定义权重（custom 时使用）

        Returns:
            float: 目标函数得分（统一归一化到 0-1，越大越好）
        """
        # 回测报告可能直接包含指标，也可能在 metrics 字段中
        metrics = backtest_report.get("metrics", {})
        # 如果 metrics 为空，尝试直接从 report 中获取
        if not metrics:
            metrics = {
                "sharpe_ratio": backtest_report.get("sharpe_ratio", 0.0),
                "total_return": backtest_report.get("total_return", 0.0),
                "annualized_return": backtest_report.get("annualized_return", 0.0),
                "max_drawdown": backtest_report.get("max_drawdown", 0.0),
                "win_rate": backtest_report.get("win_rate", 0.0),
            }

        logger.debug(f"计算目标得分: metric={objective_metric}, metrics={metrics}")

        if objective_metric == "sharpe":
            # 夏普比率
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            # 归一化到 0-1（假设夏普比率范围 -2 到 5）
            normalized = (sharpe_ratio + 2) / 7
            return max(0.0, min(1.0, normalized))

        elif objective_metric == "calmar":
            # 卡玛比率 = 年化收益 / 最大回撤
            annualized_return = metrics.get("annualized_return", 0.0)
            max_drawdown = abs(metrics.get("max_drawdown", 0.0))
            if max_drawdown == 0:
                return 0.0
            calmar_ratio = annualized_return / max_drawdown
            # 归一化到 0-1（假设卡玛比率范围 0 到 10）
            normalized = min(1.0, calmar_ratio / 10)
            return max(0.0, normalized)

        elif objective_metric == "ic":
            # 信息系数（简化版本：使用胜率作为近似）
            win_rate = metrics.get("win_rate", 0.0)
            # IC 通常范围 -1 到 1，这里用胜率作为近似
            ic = (win_rate - 0.5) * 2  # 将 0-1 映射到 -1 到 1
            normalized = (ic + 1) / 2  # 归一化到 0-1
            return max(0.0, min(1.0, normalized))

        elif objective_metric == "ic_ir":
            # 使用无成本组合的 information_ratio 作为信息比率
            ir_info = backtest_report.get("excess_return_without_cost", {})
            information_ratio = ir_info.get("information_ratio", 0.0)
            # 与夏普类似的范围假设 [-2, 5]
            normalized = (information_ratio + 2) / 7
            return max(0.0, min(1.0, normalized))

        elif objective_metric == "total_return":
            # 总收益率，假设范围 [-0.5, 1.0]
            total_return = metrics.get("total_return", 0.0)
            normalized = (total_return + 0.5) / 1.5
            return max(0.0, min(1.0, normalized))

        elif objective_metric == "annualized_return":
            # 年化收益率，假设范围 [-0.5, 1.0]
            annualized_return = metrics.get("annualized_return", 0.0)
            normalized = (annualized_return + 0.5) / 1.5
            return max(0.0, min(1.0, normalized))

        elif objective_metric == "win_rate":
            # 胜率本身已经在 0-1 之间
            win_rate = metrics.get("win_rate", 0.0)
            return max(0.0, min(1.0, win_rate))

        elif objective_metric == "profit_factor":
            # Profit Factor，通常 0-5 之间，>1 才有意义
            profit_factor = metrics.get("profit_factor", 0.0)
            if not isinstance(profit_factor, (int, float)) or profit_factor <= 0:
                return 0.0
            # 将 [0, 3] 映射到 [0, 1]，>3 视为 1
            normalized = min(1.0, profit_factor / 3.0)
            return max(0.0, normalized)

        elif objective_metric == "max_drawdown":
            # 最大回撤（负数或0），越小越好，这里转换为“越大越好”的得分
            max_drawdown = metrics.get("max_drawdown", 0.0)
            dd = abs(max_drawdown)
            # 假设 0-60% 的回撤区间，将 0 回撤映射到 1，60% 回撤映射到 0
            normalized = 1.0 - min(1.0, dd / 0.6)
            return max(0.0, normalized)

        elif objective_metric == "cost":
            # 交易成本：手续费 + 滑点，占初始资金比例，越低越好
            cost_stats = backtest_report.get("cost_statistics", {})
            cost_ratio = cost_stats.get("cost_ratio", 0.0)
            # 0 成本 → 1 分，5% 成本 → 0 分，线性下降
            normalized = 1.0 - min(1.0, max(0.0, cost_ratio) / 0.05)
            return max(0.0, normalized)

        elif objective_metric == "stability":
            # 稳定赚钱：更偏向“样本外（后段）表现 + 低回撤 + 月度稳定”。
            # 默认使用最后 30% 作为近似 out-of-sample 区间。
            #
            # 注意：回测过程中 portfolio_value 可能出现 NaN/inf（数据缺失/除零/溢出），
            # 会导致 stability 的 score 变成 NaN，最终在 API/JSON 层表现为 null。
            # 这里做 finite 过滤 + 兜底，确保返回值永远是可比较的 float。
            import math

            history = backtest_report.get("portfolio_history") or backtest_report.get(
                "portfolioHistory"
            )
            try:
                oos_ratio = float(
                    (backtest_report.get("stability_config") or {}).get("oos_ratio", 0.3)
                )
            except Exception:
                oos_ratio = 0.3
            oos_ratio = max(0.05, min(0.5, oos_ratio))

            def _max_drawdown(values: list[float]) -> float:
                if not values:
                    return 0.0
                peak = values[0]
                mdd = 0.0
                for v in values:
                    if v > peak:
                        peak = v
                    dd = (v / peak - 1.0) if peak != 0 else 0.0
                    if dd < mdd:
                        mdd = dd
                return mdd  # negative

            def _monthly_returns(dates: list[str], values: list[float]) -> list[float]:
                # month -> first/last
                if not dates or not values or len(dates) != len(values):
                    return []
                first: dict[str, float] = {}
                last: dict[str, float] = {}
                for d, v in zip(dates, values):
                    if not (isinstance(v, (int, float)) and math.isfinite(v)):
                        continue
                    m = str(d)[:7]
                    if m not in first:
                        first[m] = float(v)
                    last[m] = float(v)
                rets = []
                for m in sorted(last.keys()):
                    f = first.get(m)
                    l = last.get(m)
                    if f and f != 0 and math.isfinite(f) and math.isfinite(l):
                        rets.append(l / f - 1.0)
                return rets

            if history and isinstance(history, list) and len(history) >= 10:
                raw_dates = [
                    str(h.get("date") or h.get("snapshot_date") or "") for h in history
                ]
                raw_values = [
                    float(h.get("portfolio_value") or h.get("portfolioValue") or 0.0)
                    for h in history
                ]

                # filter non-finite points (keep alignment)
                dates: list[str] = []
                values: list[float] = []
                for d, v in zip(raw_dates, raw_values):
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        dates.append(d)
                        values.append(float(v))

                if len(values) >= 10:
                    n = len(values)
                    split = int(n * (1.0 - oos_ratio))
                    split = max(1, min(n - 1, split))
                    oos_dates = dates[split:]
                    oos_values = values[split:]

                    total_ret_oos = (
                        (oos_values[-1] / oos_values[0] - 1.0) if oos_values[0] else 0.0
                    )
                    mdd_oos = abs(_max_drawdown(oos_values))

                    # 零交易惩罚：portfolio 完全平坦（无收益无回撤）说明没有实际交易
                    if total_ret_oos == 0.0 and mdd_oos == 0.0:
                        return 0.05

                    mrets = _monthly_returns(oos_dates, oos_values)
                    if mrets:
                        pos_month_ratio = sum(1 for r in mrets if r > 0) / len(mrets)
                        mean = sum(mrets) / len(mrets)
                        var = sum((r - mean) ** 2 for r in mrets) / len(mrets)
                        mstd = var ** 0.5
                    else:
                        pos_month_ratio = 0.0
                        mstd = 0.0

                    # normalize components into [0,1]
                    # return: [-0.3, +0.6] -> [0,1]
                    ret_score = max(0.0, min(1.0, (total_ret_oos + 0.3) / 0.9))
                    # drawdown: 0..60% -> 1..0
                    dd_score = 1.0 - min(1.0, mdd_oos / 0.6)
                    # stability: monthly std 0..10% -> 1..0
                    std_score = 1.0 - min(1.0, mstd / 0.10)
                    pm_score = max(0.0, min(1.0, pos_month_ratio))

                    # weighted blend
                    score = (
                        0.45 * ret_score
                        + 0.30 * dd_score
                        + 0.15 * pm_score
                        + 0.10 * std_score
                    )
                    if not (isinstance(score, (int, float)) and math.isfinite(score)):
                        return float("nan")
                    return max(0.0, min(1.0, float(score)))

            # fallback to calmar-like behavior when no usable history
            annualized_return = metrics.get("annualized_return", 0.0)
            max_drawdown = abs(metrics.get("max_drawdown", 0.0))
            if max_drawdown <= 0:
                return 0.0
            calmar_ratio = annualized_return / max_drawdown
            normalized = min(1.0, calmar_ratio / 10)
            return max(0.0, float(normalized))

        elif objective_metric == "custom":
            # 自定义组合
            if not objective_weights:
                objective_weights = {"sharpe_ratio": 0.6, "total_return": 0.4}

            total_score = 0.0
            total_weight = 0.0

            for metric_name, weight in objective_weights.items():
                if metric_name == "sharpe_ratio":
                    sharpe = metrics.get("sharpe_ratio", 0.0)
                    normalized = (sharpe + 2) / 7
                    total_score += weight * max(0.0, min(1.0, normalized))
                elif metric_name == "total_return":
                    total_return = metrics.get("total_return", 0.0)
                    # 归一化总收益率（假设范围 -0.5 到 1.0）
                    normalized = (total_return + 0.5) / 1.5
                    total_score += weight * max(0.0, min(1.0, normalized))
                elif metric_name == "calmar_ratio":
                    annualized_return = metrics.get("annualized_return", 0.0)
                    max_drawdown = abs(metrics.get("max_drawdown", 0.0))
                    if max_drawdown == 0:
                        calmar = 0.0
                    else:
                        calmar = annualized_return / max_drawdown
                    normalized = min(1.0, calmar / 10)
                    total_score += weight * max(0.0, normalized)
                elif metric_name == "win_rate":
                    win_rate = metrics.get("win_rate", 0.0)
                    total_score += weight * max(0.0, min(1.0, win_rate))
                elif metric_name == "profit_factor":
                    profit_factor = metrics.get("profit_factor", 0.0)
                    if isinstance(profit_factor, (int, float)) and profit_factor > 0:
                        normalized = min(1.0, profit_factor / 3.0)
                        total_score += weight * max(0.0, normalized)
                elif metric_name == "information_ratio":
                    ir_info = backtest_report.get("excess_return_without_cost", {})
                    information_ratio = ir_info.get("information_ratio", 0.0)
                    normalized = (information_ratio + 2) / 7
                    total_score += weight * max(0.0, min(1.0, normalized))
                elif metric_name == "cost_ratio":
                    cost_stats = backtest_report.get("cost_statistics", {})
                    cost_ratio = cost_stats.get("cost_ratio", 0.0)
                    normalized = 1.0 - min(1.0, max(0.0, cost_ratio) / 0.05)
                    total_score += weight * max(0.0, normalized)
                elif metric_name == "max_drawdown":
                    max_drawdown_val = abs(metrics.get("max_drawdown", 0.0))
                    normalized = 1.0 - min(1.0, max_drawdown_val / 0.6)
                    total_score += weight * max(0.0, normalized)
                elif metric_name == "annualized_return":
                    annualized_return_val = metrics.get("annualized_return", 0.0)
                    normalized = (annualized_return_val + 0.5) / 1.5
                    total_score += weight * max(0.0, min(1.0, normalized))

                total_weight += weight

            if total_weight > 0:
                return total_score / total_weight
            return 0.0

        else:
            # 默认返回夏普比率
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            normalized = (sharpe_ratio + 2) / 7
            return max(0.0, min(1.0, normalized))

    def get_default_param_space(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略的默认参数空间"""

        if strategy_name.lower() == "cointegration":
            return {
                "lookback_period": {
                    "type": "int",
                    "low": 30,
                    "high": 120,
                    "default": 60,
                    "enabled": True,
                },
                "half_life": {
                    "type": "int",
                    "low": 10,
                    "high": 50,
                    "default": 20,
                    "enabled": True,
                },
                "entry_threshold": {
                    "type": "float",
                    "low": 1.0,
                    "high": 3.0,
                    "default": 2.0,
                    "enabled": True,
                },
                "exit_threshold": {
                    "type": "float",
                    "low": 0.1,
                    "high": 1.0,
                    "default": 0.5,
                    "enabled": True,
                },
            }
        else:
            # 其他策略的默认参数空间
            return {}
