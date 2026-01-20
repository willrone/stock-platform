"""
策略超参数优化器

使用 Optuna 对策略参数进行优化，支持单目标和多目标优化
"""

import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from loguru import logger

try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIISampler
    from optuna.pruners import MedianPruner
except ImportError as e:
    logger.error(f"无法导入 optuna 模块: {e}")
    logger.error("请运行: pip install optuna>=3.4.0")
    raise ImportError(
        "optuna 模块未安装。超参优化功能需要 optuna 库。"
        "请运行: pip install optuna>=3.4.0"
    ) from e

from app.services.backtest import BacktestExecutor, BacktestConfig
from app.core.config import settings


class StrategyHyperparameterOptimizer:
    """策略超参数优化器"""
    
    def __init__(self):
        self.optimization_history = {}
    
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
        progress_callback: Optional[Callable] = None
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
                - objective_metric: "sharpe" | "calmar" | "ic" | "custom" | ["sharpe", "calmar"] (多目标)
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
        logger.info(f"开始策略参数优化: {strategy_name}, 方法: {optimization_method}, 试验次数: {n_trials}")
        
        start_time = datetime.utcnow()
        
        # 解析目标配置
        objective_metric = objective_config.get("objective_metric", "sharpe")
        is_multi_objective = isinstance(objective_metric, list) and len(objective_metric) > 1
        
        # 创建 Optuna study
        if is_multi_objective:
            # 多目标优化
            directions = ["maximize" if objective_config.get("direction", "maximize") == "maximize" else "minimize"] * len(objective_metric)
            if optimization_method == "nsga2":
                sampler = NSGAIISampler()
            elif optimization_method == "motpe":
                # MOTPESampler 在 optuna 4.6.0 中不存在，使用 NSGA-II 替代
                logger.warning("MOTPESampler 不可用，使用 NSGAIISampler 替代")
                sampler = NSGAIISampler()
            else:
                sampler = NSGAIISampler()  # 默认使用 NSGA-II
            study = optuna.create_study(
                directions=directions,
                sampler=sampler
            )
        else:
            # 单目标优化
            direction = objective_config.get("direction", "maximize")
            if optimization_method == "tpe":
                sampler = TPESampler(seed=42)
            elif optimization_method == "random":
                sampler = optuna.samplers.RandomSampler(seed=42)
            else:
                sampler = TPESampler(seed=42)
            
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
        
        # 创建回测执行器
        try:
            import os
            enable_perf = os.getenv("ENABLE_BACKTEST_PERFORMANCE_PROFILING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
            executor = BacktestExecutor(
                data_dir=str(settings.DATA_ROOT_PATH),
                enable_performance_profiling=enable_perf
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
                "slippage_rate": 0.0001
            }
        
        backtest_cfg = BacktestConfig(
            initial_cash=backtest_config.get("initial_cash", 100000.0),
            commission_rate=backtest_config.get("commission_rate", 0.0003),
            slippage_rate=backtest_config.get("slippage_rate", 0.0001)
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
                            log=param_config.get("log", False)
                        )
                    elif param_type == "float":
                        strategy_params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=param_config.get("log", False)
                        )
                    elif param_type == "categorical":
                        strategy_params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config["choices"]
                        )
                
                # 记录采样到的参数（用于调试）
                logger.info(f"Trial {trial.number}: 采样参数 = {strategy_params}")
                
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
                                        strategy_config=strategy_params,
                                        backtest_config=backtest_cfg
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
                                strategy_config=strategy_params,
                                backtest_config=backtest_cfg
                            )
                        )
                    finally:
                        new_loop.close()
                
                # 计算目标函数值
                # 记录回测结果的关键指标（用于调试）
                metrics = backtest_report.get("metrics", {})
                logger.info(f"Trial {trial.number}: 回测指标 - sharpe_ratio={metrics.get('sharpe_ratio', 0):.4f}, "
                           f"total_return={metrics.get('total_return', 0):.4f}, "
                           f"annualized_return={metrics.get('annualized_return', 0):.4f}, "
                           f"max_drawdown={metrics.get('max_drawdown', 0):.4f}")
                
                if is_multi_objective:
                    # 多目标：返回多个值
                    objectives = []
                    for metric in objective_metric:
                        score = self._calculate_objective_score(
                            backtest_report,
                            metric,
                            objective_config.get("objective_weights")
                        )
                        objectives.append(score)
                    logger.info(f"Trial {trial.number}: 多目标得分 = {objectives}")
                    
                    # 更新进度（多目标优化）
                    if progress_callback:
                        # 计算 trial 统计信息
                        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                        running_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
                        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
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
                            best_params=None
                        )
                    
                    return tuple(objectives)
                else:
                    # 单目标：返回单个值
                    score = self._calculate_objective_score(
                        backtest_report,
                        objective_metric,
                        objective_config.get("objective_weights")
                    )
                    logger.info(f"Trial {trial.number}: 目标得分 = {score:.6f} (原始指标: {objective_metric})")
                    
                    # 更新进度
                    if progress_callback:
                        # 计算 trial 统计信息
                        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                        running_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
                        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                        trial_num = trial.number + 1  # trial.number 从 0 开始，所以 +1 得到当前编号
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
                            best_score=study.best_value if not is_multi_objective and len(study.trials) > 0 else None,
                            best_trial_number=study.best_trial.number if not is_multi_objective and len(study.trials) > 0 else None,
                            best_params=study.best_params if not is_multi_objective and len(study.trials) > 0 else None
                        )
                    
                    return score
                    
            except Exception as e:
                logger.error(f"Trial {trial.number} 失败: {e}", exc_info=True)
                logger.error(f"Trial {trial.number} 参数: {strategy_params}")
                # 返回最差分数
                if is_multi_objective:
                    return tuple([float('-inf') if objective_config.get("direction", "maximize") == "maximize" else float('inf')] * len(objective_metric))
                else:
                    return float('-inf') if objective_config.get("direction", "maximize") == "maximize" else float('inf')
        
        # 执行优化
        try:
            logger.info(f"开始执行优化，策略: {strategy_name}, 股票: {stock_codes}, 日期范围: {start_date} - {end_date}")
            logger.info(f"参数空间: {list(param_space.keys())}")
            logger.info(f"目标函数: {objective_metric}, 方向: {objective_config.get('direction', 'maximize')}")
            
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
                    best_params=None
                )
            
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
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
                    "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
                    "timestamp": trial.datetime_start.isoformat() if trial.datetime_start else None
                }
                
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    if is_multi_objective:
                        trial_data["objectives"] = trial.values
                    else:
                        trial_data["score"] = trial.value
                
                optimization_history.append(trial_data)
            
            # 构建结果
            result = {
                "success": True,
                "strategy_name": strategy_name,
                "best_params": study.best_params if not is_multi_objective and len(study.trials) > 0 else None,
                "best_score": study.best_value if not is_multi_objective and len(study.trials) > 0 else None,
                "best_trial_number": study.best_trial.number if not is_multi_objective and len(study.trials) > 0 else None,
                "objective_metric": objective_metric,
                "n_trials": n_trials,
                "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "running_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]),
                "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
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
                        "end_date": end_date.isoformat()
                    }
                }
            }
            
            # 多目标优化：添加 Pareto front
            if is_multi_objective:
                # 计算 Pareto front（非支配解）
                try:
                    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
                    
                    if len(completed_trials) > 0:
                        # 计算非支配解（Pareto front）
                        pareto_front = []
                        for trial in completed_trials:
                            is_dominated = False
                            for other_trial in completed_trials:
                                if trial.number == other_trial.number:
                                    continue
                                # 检查是否被支配：如果另一个解在所有目标上都更好或相等，且至少有一个目标更好
                                if all(other_trial.values[i] >= trial.values[i] for i in range(len(trial.values))):
                                    if any(other_trial.values[i] > trial.values[i] for i in range(len(trial.values))):
                                        is_dominated = True
                                        break
                            if not is_dominated:
                                pareto_front.append({
                                    "trial_number": trial.number,
                                    "params": trial.params,
                                    "objectives": list(trial.values)
                                })
                        result["pareto_front"] = pareto_front
                    else:
                        result["pareto_front"] = []
                except Exception as e:
                    logger.warning(f"计算 Pareto front 失败: {e}")
                    result["pareto_front"] = []
            
            logger.info(f"策略参数优化完成: {strategy_name}, 最佳得分: {result.get('best_score', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"策略参数优化失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "strategy_name": strategy_name
            }
    
    def _calculate_objective_score(
        self,
        backtest_report: Dict[str, Any],
        objective_metric: str,
        objective_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算目标函数得分
        
        Args:
            backtest_report: 回测报告
            objective_metric: 目标指标 ("sharpe", "calmar", "ic", "custom")
            objective_weights: 自定义权重（custom 时使用）
            
        Returns:
            float: 目标函数得分
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
                    "enabled": True
                },
                "half_life": {
                    "type": "int",
                    "low": 10,
                    "high": 50,
                    "default": 20,
                    "enabled": True
                },
                "entry_threshold": {
                    "type": "float",
                    "low": 1.0,
                    "high": 3.0,
                    "default": 2.0,
                    "enabled": True
                },
                "exit_threshold": {
                    "type": "float",
                    "low": 0.1,
                    "high": 1.0,
                    "default": 0.5,
                    "enabled": True
                }
            }
        else:
            # 其他策略的默认参数空间
            return {}

