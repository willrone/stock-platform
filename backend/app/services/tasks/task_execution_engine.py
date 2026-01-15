"""
任务执行引擎 - 具体的任务执行逻辑和进度跟踪
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
from loguru import logger

from app.models.task_models import TaskType, TaskStatus, PredictionTaskConfig, BacktestTaskConfig, TrainingTaskConfig
from .task_queue import QueuedTask, TaskExecutionContext, TaskPriority
from app.services.prediction import PredictionEngine, PredictionConfig
from app.repositories.task_repository import TaskRepository, PredictionResultRepository
from app.core.error_handler import TaskError, ErrorSeverity, ErrorContext
from app.core.logging_config import PerformanceLogger, set_log_context


@dataclass
class TaskProgress:
    """任务进度信息"""
    task_id: str
    current_step: str
    progress_percentage: float
    estimated_remaining_seconds: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, task_id: str, total_steps: int, 
                 progress_callback: Optional[Callable] = None):
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_callback = progress_callback
        self.start_time = datetime.utcnow()
        self.step_start_time = datetime.utcnow()
        self.step_durations: List[float] = []
        
    def update_step(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        """更新当前步骤"""
        if self.current_step > 0:
            # 记录上一步的耗时
            step_duration = (datetime.utcnow() - self.step_start_time).total_seconds()
            self.step_durations.append(step_duration)
        
        self.current_step += 1
        self.step_start_time = datetime.utcnow()
        
        # 计算进度百分比
        progress_percentage = (self.current_step / self.total_steps) * 100
        
        # 估算剩余时间
        estimated_remaining = self._estimate_remaining_time()
        
        progress = TaskProgress(
            task_id=self.task_id,
            current_step=step_name,
            progress_percentage=progress_percentage,
            estimated_remaining_seconds=estimated_remaining,
            details=details
        )
        
        # 调用进度回调
        if self.progress_callback:
            self.progress_callback(progress_percentage, step_name)
        
        logger.info(f"任务进度: {self.task_id}, 步骤: {step_name}, 进度: {progress_percentage:.1f}%")
        return progress
    
    def _estimate_remaining_time(self) -> Optional[int]:
        """估算剩余时间"""
        if len(self.step_durations) < 2:
            return None
        
        # 计算平均步骤耗时
        avg_step_duration = sum(self.step_durations) / len(self.step_durations)
        
        # 估算剩余步骤数
        remaining_steps = self.total_steps - self.current_step
        
        # 估算剩余时间
        estimated_remaining = int(avg_step_duration * remaining_steps)
        return estimated_remaining


class PredictionTaskExecutor:
    """预测任务执行器"""
    
    def __init__(self, prediction_engine: PredictionEngine, 
                 task_repository: TaskRepository,
                 prediction_result_repository: PredictionResultRepository):
        self.prediction_engine = prediction_engine
        self.task_repository = task_repository
        self.prediction_result_repository = prediction_result_repository
    
    def execute(self, queued_task: QueuedTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """执行预测任务"""
        task_id = queued_task.task_id
        
        with set_log_context(task_id=task_id, user_id=queued_task.user_id):
            try:
                # 解析任务配置
                config_dict = queued_task.config
                stock_codes = config_dict.get('stock_codes', [])
                model_id = config_dict.get('model_id', 'default_model')
                horizon = config_dict.get('horizon', 'short_term')
                confidence_level = config_dict.get('confidence_level', 0.95)
                
                # 创建预测配置
                prediction_config = PredictionConfig(
                    model_id=model_id,
                    horizon=horizon,
                    confidence_level=confidence_level,
                    use_ensemble=config_dict.get('use_ensemble', False),
                    risk_assessment=config_dict.get('risk_assessment', True)
                )
                
                # 更新任务状态为运行中
                self.task_repository.update_task_status(task_id, TaskStatus.RUNNING)
                
                # 创建进度跟踪器
                total_steps = len(stock_codes) + 2  # 股票数量 + 初始化 + 完成
                progress_tracker = ProgressTracker(
                    task_id, total_steps, context.progress_callback
                )
                
                # 步骤1: 初始化
                progress_tracker.update_step("初始化预测任务", {"stock_count": len(stock_codes)})
                
                # 检查取消信号
                if context.cancel_event and context.cancel_event.is_set():
                    raise TaskError("任务被用户取消", severity=ErrorSeverity.LOW)
                
                # 执行预测
                prediction_results = []
                failed_stocks = []
                
                for i, stock_code in enumerate(stock_codes):
                    # 检查取消信号
                    if context.cancel_event and context.cancel_event.is_set():
                        raise TaskError("任务被用户取消", severity=ErrorSeverity.LOW)
                    
                    try:
                        # 更新进度
                        progress_tracker.update_step(
                            f"预测股票 {stock_code}", 
                            {"current_stock": stock_code, "completed": i, "total": len(stock_codes)}
                        )
                        
                        # 执行单股预测
                        prediction_result = self.prediction_engine.predict_single_stock(
                            stock_code, prediction_config
                        )
                        
                        # 保存预测结果到数据库
                        db_result = self.prediction_result_repository.save_prediction_result(
                            task_id=task_id,
                            stock_code=stock_code,
                            prediction_date=prediction_result.prediction_date,
                            predicted_price=prediction_result.predicted_price,
                            predicted_direction=prediction_result.predicted_direction,
                            confidence_score=prediction_result.confidence_score,
                            confidence_interval_lower=prediction_result.confidence_interval[0],
                            confidence_interval_upper=prediction_result.confidence_interval[1],
                            model_id=prediction_result.model_id,
                            features_used=prediction_result.features_used,
                            risk_metrics=prediction_result.risk_metrics.to_dict()
                        )
                        
                        prediction_results.append(prediction_result)
                        
                    except Exception as e:
                        logger.error(f"股票预测失败: {stock_code}, 错误: {e}")
                        failed_stocks.append({"stock_code": stock_code, "error": str(e)})
                        continue
                
                # 步骤N: 完成任务
                progress_tracker.update_step("完成预测任务", {
                    "successful_predictions": len(prediction_results),
                    "failed_predictions": len(failed_stocks)
                })
                
                # 准备结果
                task_result = {
                    "total_stocks": len(stock_codes),
                    "successful_predictions": len(prediction_results),
                    "failed_predictions": len(failed_stocks),
                    "failed_stocks": failed_stocks,
                    "model_id": model_id,
                    "horizon": horizon,
                    "confidence_level": confidence_level,
                    "execution_time": (datetime.utcnow() - context.start_time).total_seconds()
                }
                
                # 更新任务状态为完成
                self.task_repository.update_task_status(
                    task_id, TaskStatus.COMPLETED, progress=100.0, result=task_result
                )
                
                logger.info(f"预测任务完成: {task_id}, 成功: {len(prediction_results)}, 失败: {len(failed_stocks)}")
                return task_result
                
            except TaskError:
                # 更新任务状态为失败
                self.task_repository.update_task_status(
                    task_id, TaskStatus.FAILED, error_message="任务被取消"
                )
                raise
            except Exception as e:
                # 更新任务状态为失败
                error_message = f"预测任务执行失败: {str(e)}"
                self.task_repository.update_task_status(
                    task_id, TaskStatus.FAILED, error_message=error_message
                )
                
                raise TaskError(
                    message=error_message,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(task_id=task_id, user_id=queued_task.user_id),
                    original_exception=e
                )


class BacktestTaskExecutor:
    """回测任务执行器"""
    
    def __init__(self, task_repository: TaskRepository):
        self.task_repository = task_repository
    
    def execute(self, queued_task: QueuedTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """执行回测任务"""
        task_id = queued_task.task_id
        
        with set_log_context(task_id=task_id, user_id=queued_task.user_id):
            try:
                # 解析任务配置
                config_dict = queued_task.config
                strategy_name = config_dict.get('strategy_name', 'default_strategy')
                stock_codes = config_dict.get('stock_codes', [])
                start_date = datetime.fromisoformat(config_dict.get('start_date'))
                end_date = datetime.fromisoformat(config_dict.get('end_date'))
                initial_cash = config_dict.get('initial_cash', 100000.0)
                
                # 更新任务状态为运行中
                self.task_repository.update_task_status(task_id, TaskStatus.RUNNING)
                
                # 创建进度跟踪器
                total_steps = 5  # 初始化、数据加载、策略执行、结果计算、完成
                progress_tracker = ProgressTracker(
                    task_id, total_steps, context.progress_callback
                )
                
                # 步骤1: 初始化回测
                progress_tracker.update_step("初始化回测任务", {
                    "strategy": strategy_name,
                    "stocks": stock_codes,
                    "period": f"{start_date.date()} - {end_date.date()}"
                })
                
                # 检查取消信号
                if context.cancel_event and context.cancel_event.is_set():
                    raise TaskError("任务被用户取消", severity=ErrorSeverity.LOW)
                
                # 步骤2: 加载历史数据
                progress_tracker.update_step("加载历史数据")
                
                # 步骤3: 执行回测策略
                progress_tracker.update_step("执行回测策略")
                
                # 使用真实的回测执行器
                from app.services.backtest import BacktestExecutor, BacktestConfig
                from app.core.config import settings
                
                executor = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH))
                
                # 创建回测配置
                strategy_config = config_dict.get('strategy_config', {})
                backtest_config = BacktestConfig(
                    initial_cash=initial_cash,
                    commission_rate=strategy_config.get("commission_rate", 0.0003),
                    slippage_rate=strategy_config.get("slippage_rate", 0.0001)
                )
                
                # 执行回测
                backtest_report = executor.run_backtest(
                    strategy_name=strategy_name,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_config=strategy_config,
                    backtest_config=backtest_config
                )
                
                # 步骤4: 计算回测结果
                progress_tracker.update_step("计算回测结果")
                
                # 转换数据格式
                portfolio_history = backtest_report.get("portfolio_history", [])
                dates = [snapshot["date"] for snapshot in portfolio_history]
                equity_curve = [snapshot["portfolio_value"] for snapshot in portfolio_history]
                
                # 计算回撤曲线
                drawdown_curve = []
                peak = initial_cash
                for value in equity_curve:
                    if value > peak:
                        peak = value
                    drawdown = (value - peak) / peak * 100 if peak > 0 else 0
                    drawdown_curve.append(drawdown)
                
                # 格式化交易记录
                trade_history = []
                for trade in backtest_report.get("trade_history", []):
                    trade_history.append({
                        "date": trade.get("timestamp", ""),
                        "action": "buy" if trade.get("action") == "BUY" else "sell",
                        "price": trade.get("price", 0),
                        "quantity": trade.get("quantity", 0),
                        "pnl": trade.get("pnl", 0)
                    })
                
                task_result = {
                    "strategy_name": backtest_report.get("strategy_name", strategy_name),
                    "stock_codes": stock_codes,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "initial_cash": initial_cash,
                    "final_value": backtest_report.get("final_value", initial_cash),
                    "total_return": backtest_report.get("total_return", 0),
                    "annualized_return": backtest_report.get("annualized_return", 0),
                    "volatility": backtest_report.get("volatility", 0),
                    "sharpe_ratio": backtest_report.get("sharpe_ratio", 0),
                    "max_drawdown": backtest_report.get("max_drawdown", 0),
                    "win_rate": backtest_report.get("win_rate", 0),
                    "profit_factor": backtest_report.get("profit_factor", 0),
                    "total_trades": backtest_report.get("total_trades", 0),
                    "execution_time": (datetime.utcnow() - context.start_time).total_seconds(),
                    # 添加前端需要的图表数据
                    "equity_curve": equity_curve,
                    "drawdown_curve": drawdown_curve,
                    "dates": dates,
                    "trade_history": trade_history,
                    # 保留原始报告数据
                    "portfolio": {
                        "initial_cash": backtest_report.get("initial_cash", initial_cash),
                        "final_value": backtest_report.get("final_value", initial_cash),
                        "total_return": backtest_report.get("total_return", 0),
                        "annualized_return": backtest_report.get("annualized_return", 0)
                    },
                    "risk_metrics": {
                        "volatility": backtest_report.get("volatility", 0),
                        "sharpe_ratio": backtest_report.get("sharpe_ratio", 0),
                        "max_drawdown": backtest_report.get("max_drawdown", 0)
                    },
                    "trading_stats": {
                        "total_trades": backtest_report.get("total_trades", 0),
                        "win_rate": backtest_report.get("win_rate", 0),
                        "profit_factor": backtest_report.get("profit_factor", 0)
                    }
                }
                
                # 步骤5: 完成回测
                total_return = backtest_report.get("total_return", 0)
                final_value = backtest_report.get("final_value", initial_cash)
                progress_tracker.update_step("完成回测任务", {
                    "total_return": f"{total_return:.2%}",
                    "final_value": final_value
                })
                
                # 更新任务状态为完成
                logger.info(f"保存回测结果: task_id={task_id}, result包含字段={list(task_result.keys())[:20]}")
                logger.info(f"回测结果数据: equity_curve长度={len(task_result.get('equity_curve', []))}, "
                           f"drawdown_curve长度={len(task_result.get('drawdown_curve', []))}, "
                           f"dates长度={len(task_result.get('dates', []))}, "
                           f"portfolio存在={task_result.get('portfolio') is not None}, "
                           f"risk_metrics存在={task_result.get('risk_metrics') is not None}")
                
                self.task_repository.update_task_status(
                    task_id, TaskStatus.COMPLETED, progress=100.0, result=task_result
                )
                
                # 验证保存后的数据
                saved_task = self.task_repository.get_task_by_id(task_id)
                if saved_task and saved_task.result:
                    logger.info(f"验证保存结果: task_id={task_id}, result类型={type(saved_task.result)}, "
                               f"result是否为None={saved_task.result is None}")
                    if isinstance(saved_task.result, dict):
                        logger.info(f"保存后的result包含字段={list(saved_task.result.keys())[:20]}")
                
                logger.info(f"回测任务完成: {task_id}, 总收益: {total_return:.2%}")
                return task_result
                
            except TaskError:
                # 更新任务状态为失败
                self.task_repository.update_task_status(
                    task_id, TaskStatus.FAILED, error_message="任务被取消"
                )
                raise
            except Exception as e:
                # 更新任务状态为失败
                error_message = f"回测任务执行失败: {str(e)}"
                self.task_repository.update_task_status(
                    task_id, TaskStatus.FAILED, error_message=error_message
                )
                
                raise TaskError(
                    message=error_message,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(task_id=task_id, user_id=queued_task.user_id),
                    original_exception=e
                )


class TrainingTaskExecutor:
    """模型训练任务执行器"""
    
    def __init__(self, task_repository: TaskRepository):
        self.task_repository = task_repository
    
    def execute(self, queued_task: QueuedTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """执行训练任务"""
        task_id = queued_task.task_id
        
        with set_log_context(task_id=task_id, user_id=queued_task.user_id):
            try:
                # 解析任务配置
                config_dict = queued_task.config
                model_name = config_dict.get('model_name', 'default_model')
                model_type = config_dict.get('model_type', 'xgboost')
                stock_codes = config_dict.get('stock_codes', [])
                start_date = datetime.fromisoformat(config_dict.get('start_date'))
                end_date = datetime.fromisoformat(config_dict.get('end_date'))
                
                # 更新任务状态为运行中
                self.task_repository.update_task_status(task_id, TaskStatus.RUNNING)
                
                # 创建进度跟踪器
                total_steps = 6  # 初始化、数据准备、特征工程、模型训练、验证、保存
                progress_tracker = ProgressTracker(
                    task_id, total_steps, context.progress_callback
                )
                
                # 步骤1: 初始化训练任务
                progress_tracker.update_step("初始化训练任务", {
                    "model_name": model_name,
                    "model_type": model_type,
                    "stocks": len(stock_codes)
                })
                
                # 检查取消信号
                if context.cancel_event and context.cancel_event.is_set():
                    raise TaskError("任务被用户取消", severity=ErrorSeverity.LOW)
                
                # 步骤2: 准备训练数据
                progress_tracker.update_step("准备训练数据")
                time.sleep(3)  # 模拟数据准备时间
                
                # 步骤3: 特征工程
                progress_tracker.update_step("特征工程")
                time.sleep(4)  # 模拟特征工程时间
                
                # 步骤4: 模型训练
                progress_tracker.update_step("模型训练")
                time.sleep(10)  # 模拟模型训练时间
                
                # 步骤5: 模型验证
                progress_tracker.update_step("模型验证")
                time.sleep(2)  # 模拟模型验证时间
                
                # 模拟训练结果
                import random
                random.seed(42)
                
                task_result = {
                    "model_name": model_name,
                    "model_type": model_type,
                    "stock_codes": stock_codes,
                    "training_period": f"{start_date.date()} - {end_date.date()}",
                    "performance_metrics": {
                        "accuracy": random.uniform(0.6, 0.85),
                        "precision": random.uniform(0.55, 0.8),
                        "recall": random.uniform(0.5, 0.75),
                        "f1_score": random.uniform(0.55, 0.78),
                        "mse": random.uniform(0.01, 0.05),
                        "mae": random.uniform(0.008, 0.04)
                    },
                    "hyperparameters": config_dict.get('hyperparameters', {}),
                    "training_samples": random.randint(10000, 50000),
                    "validation_samples": random.randint(2000, 10000),
                    "execution_time": (datetime.utcnow() - context.start_time).total_seconds()
                }
                
                # 步骤6: 保存模型
                progress_tracker.update_step("保存模型", {
                    "accuracy": f"{task_result['performance_metrics']['accuracy']:.3f}"
                })
                
                # 更新任务状态为完成
                self.task_repository.update_task_status(
                    task_id, TaskStatus.COMPLETED, progress=100.0, result=task_result
                )
                
                logger.info(f"训练任务完成: {task_id}, 模型: {model_name}, 准确率: {task_result['performance_metrics']['accuracy']:.3f}")
                return task_result
                
            except TaskError:
                # 更新任务状态为失败
                self.task_repository.update_task_status(
                    task_id, TaskStatus.FAILED, error_message="任务被取消"
                )
                raise
            except Exception as e:
                # 更新任务状态为失败
                error_message = f"训练任务执行失败: {str(e)}"
                self.task_repository.update_task_status(
                    task_id, TaskStatus.FAILED, error_message=error_message
                )
                
                raise TaskError(
                    message=error_message,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(task_id=task_id, user_id=queued_task.user_id),
                    original_exception=e
                )


class TaskExecutionEngine:
    """任务执行引擎 - 统一管理所有类型的任务执行器"""
    
    def __init__(self, prediction_engine: PredictionEngine, task_repository: TaskRepository,
                 prediction_result_repository: PredictionResultRepository):
        self.task_repository = task_repository
        
        # 创建各类型任务执行器
        self.executors = {
            TaskType.PREDICTION: PredictionTaskExecutor(
                prediction_engine, task_repository, prediction_result_repository
            ),
            TaskType.BACKTEST: BacktestTaskExecutor(task_repository),
            TaskType.TRAINING: TrainingTaskExecutor(task_repository),
            TaskType.HYPERPARAMETER_OPTIMIZATION: HyperparameterOptimizationTaskExecutor(task_repository)
        }
    
    def get_task_handler(self, task_type: TaskType) -> Callable:
        """获取任务处理器"""
        executor = self.executors.get(task_type)
        if not executor:
            raise TaskError(
                message=f"不支持的任务类型: {task_type.value}",
                severity=ErrorSeverity.HIGH
            )
        
        return executor.execute
    
    def register_handlers_to_scheduler(self, scheduler):
        """将所有处理器注册到调度器"""
        for task_type, executor in self.executors.items():
            scheduler.register_task_handler(task_type, executor.execute)
        
        logger.info("所有任务处理器已注册到调度器")
    
    def validate_task_config(self, task_type: TaskType, config: Dict[str, Any]) -> bool:
        """验证任务配置"""
        try:
            if task_type == TaskType.PREDICTION:
                required_fields = ['stock_codes', 'model_id']
                for field in required_fields:
                    if field not in config:
                        raise TaskError(f"预测任务缺少必需字段: {field}")
                
                if not isinstance(config['stock_codes'], list) or len(config['stock_codes']) == 0:
                    raise TaskError("股票代码列表不能为空")
            
            elif task_type == TaskType.BACKTEST:
                required_fields = ['strategy_name', 'stock_codes', 'start_date', 'end_date']
                for field in required_fields:
                    if field not in config:
                        raise TaskError(f"回测任务缺少必需字段: {field}")
                
                # 验证日期格式
                try:
                    start_date = datetime.fromisoformat(config['start_date'])
                    end_date = datetime.fromisoformat(config['end_date'])
                    if start_date >= end_date:
                        raise TaskError("开始日期必须早于结束日期")
                except ValueError:
                    raise TaskError("日期格式错误，应为ISO格式")
            
            elif task_type == TaskType.TRAINING:
                required_fields = ['model_name', 'model_type', 'stock_codes', 'start_date', 'end_date']
                for field in required_fields:
                    if field not in config:
                        raise TaskError(f"训练任务缺少必需字段: {field}")
            
            elif task_type == TaskType.HYPERPARAMETER_OPTIMIZATION:
                required_fields = ['stock_codes', 'start_date', 'end_date', 'optimization_config']
                for field in required_fields:
                    if field not in config:
                        raise TaskError(f"超参优化任务缺少必需字段: {field}")
                
                # 验证优化配置
                optimization_config = config.get('optimization_config', {})
                if 'strategy_name' not in optimization_config:
                    raise TaskError("超参优化任务缺少策略名称")
                if 'param_space' not in optimization_config:
                    raise TaskError("超参优化任务缺少参数空间定义")
                if 'objective_config' not in optimization_config:
                    raise TaskError("超参优化任务缺少目标函数配置")
                
                # 验证日期格式
                try:
                    start_date = datetime.fromisoformat(config['start_date'])
                    end_date = datetime.fromisoformat(config['end_date'])
                    if start_date >= end_date:
                        raise TaskError("开始日期必须早于结束日期")
                except ValueError:
                    raise TaskError("日期格式错误，应为ISO格式")
            
            return True
            
        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                message=f"任务配置验证失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e
            )
    
    def estimate_task_duration(self, task_type: TaskType, config: Dict[str, Any]) -> int:
        """估算任务执行时间（秒）"""
        try:
            if task_type == TaskType.PREDICTION:
                stock_count = len(config.get('stock_codes', []))
                # 每只股票预测大约需要5-10秒
                return stock_count * 8 + 30  # 加上初始化时间
            
            elif task_type == TaskType.BACKTEST:
                stock_count = len(config.get('stock_codes', []))
                start_date = datetime.fromisoformat(config.get('start_date'))
                end_date = datetime.fromisoformat(config.get('end_date'))
                days = (end_date - start_date).days
                # 回测时间与股票数量和时间跨度相关
                return min(stock_count * days * 0.1 + 60, 3600)  # 最多1小时
            
            elif task_type == TaskType.TRAINING:
                stock_count = len(config.get('stock_codes', []))
                # 训练时间与股票数量相关，但主要取决于模型复杂度
                return min(stock_count * 20 + 300, 7200)  # 最多2小时
            
            elif task_type == TaskType.HYPERPARAMETER_OPTIMIZATION:
                n_trials = config.get('optimization_config', {}).get('n_trials', 50)
                # 优化时间与试验次数相关，每个试验大约需要1-2分钟
                return min(n_trials * 90, 7200)  # 最多2小时
            
            return 600  # 默认10分钟
            
        except Exception:
            return 600  # 出错时返回默认值


class HyperparameterOptimizationTaskExecutor:
    """超参优化任务执行器"""
    
    def __init__(self, task_repository: TaskRepository):
        self.task_repository = task_repository
    
    def execute(self, queued_task: QueuedTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """执行超参优化任务"""
        task_id = queued_task.task_id
        
        with set_log_context(task_id=task_id, user_id=queued_task.user_id):
            try:
                # 解析任务配置
                config_dict = queued_task.config
                if not config_dict:
                    raise ValueError("任务配置为空")
                
                optimization_config = config_dict.get('optimization_config', {})
                if not optimization_config:
                    raise ValueError("优化配置为空")
                
                strategy_name = optimization_config.get('strategy_name')
                if not strategy_name:
                    raise ValueError("策略名称未指定")
                
                param_space = optimization_config.get('param_space', {})
                if not param_space:
                    raise ValueError("参数空间未定义")
                
                stock_codes = config_dict.get('stock_codes', [])
                if not stock_codes or not isinstance(stock_codes, list):
                    raise ValueError("股票代码列表为空或格式错误")
                
                start_date_str = config_dict.get('start_date')
                if not start_date_str:
                    raise ValueError("开始日期未指定")
                start_date = datetime.fromisoformat(start_date_str)
                
                end_date_str = config_dict.get('end_date')
                if not end_date_str:
                    raise ValueError("结束日期未指定")
                end_date = datetime.fromisoformat(end_date_str)
                
                if start_date >= end_date:
                    raise ValueError("开始日期必须早于结束日期")
                
                objective_config = optimization_config.get('objective_config', {})
                if not objective_config:
                    raise ValueError("目标函数配置为空")
                
                n_trials = optimization_config.get('n_trials', 50)
                if n_trials <= 0:
                    raise ValueError(f"试验次数必须大于0，当前值: {n_trials}")
                
                optimization_method = optimization_config.get('optimization_method', 'tpe')
                timeout = optimization_config.get('timeout')
                
                backtest_config = config_dict.get('backtest_config', {})
                
                logger.info(f"解析超参优化任务配置: 策略={strategy_name}, 股票数={len(stock_codes)}, "
                           f"日期范围={start_date.date()} - {end_date.date()}, 试验次数={n_trials}")
                
                # 更新任务状态为运行中
                self.task_repository.update_task_status(task_id, TaskStatus.RUNNING)
                
                # 创建进度回调
                # 注意：StrategyHyperparameterOptimizer 的 progress_callback 签名已扩展，包含 trial 统计信息
                def progress_callback(trial_num, n_trials, strategy_params, score, backtest_report,
                                     completed_trials=0, running_trials=0, pruned_trials=0, failed_trials=0,
                                     best_score=None, best_trial_number=None, best_params=None):
                    progress = (trial_num / n_trials) * 100
                    message = f"Trial {trial_num}/{n_trials}"
                    if score is not None:
                        message += f", Score: {score:.4f}"
                    
                    if context.progress_callback:
                        context.progress_callback(progress, message)
                    
                    # 构建当前状态数据
                    current_result = {
                        "n_trials": n_trials,
                        "completed_trials": completed_trials,
                        "running_trials": running_trials,
                        "pruned_trials": pruned_trials,
                        "failed_trials": failed_trials,
                    }
                    if best_score is not None:
                        current_result["best_score"] = best_score
                    if best_trial_number is not None:
                        current_result["best_trial_number"] = best_trial_number
                    if best_params is not None:
                        current_result["best_params"] = best_params
                    
                    # 更新任务进度和状态
                    self.task_repository.update_task_status(
                        task_id,
                        TaskStatus.RUNNING,
                        progress=progress,
                        result=current_result
                    )
                
                # 执行优化（在同步函数中运行异步代码）
                # 使用新的事件循环，避免与外部事件循环冲突
                try:
                    from app.services.backtest.strategy_hyperparameter_optimizer import StrategyHyperparameterOptimizer
                except ImportError as e:
                    error_msg = f"无法导入超参优化器: {e}. 请确保已安装 optuna: pip install optuna>=3.4.0"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
                
                optimizer = StrategyHyperparameterOptimizer()
                
                # 创建新的事件循环来运行异步代码
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(
                        optimizer.optimize_strategy_parameters(
                            strategy_name=strategy_name,
                            param_space=param_space,
                            stock_codes=stock_codes,
                            start_date=start_date,
                            end_date=end_date,
                            objective_config=objective_config,
                            backtest_config=backtest_config,
                            n_trials=n_trials,
                            optimization_method=optimization_method,
                            timeout=timeout,
                            progress_callback=progress_callback
                        )
                    )
                finally:
                    new_loop.close()
                
                # 保存结果
                if result.get('success'):
                    self.task_repository.update_task_status(
                        task_id,
                        TaskStatus.COMPLETED,
                        result=result,
                        progress=100.0
                    )
                else:
                    self.task_repository.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error_message=result.get('error', '优化失败'),
                        progress=100.0
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"超参优化任务执行失败: {e}", exc_info=True)
                self.task_repository.update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    error_message=str(e)
                )
                raise TaskError(
                    f"超参优化任务执行失败: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(
                        task_id=task_id,
                        additional_data={
                            "task_type": TaskType.HYPERPARAMETER_OPTIMIZATION.value,
                            "original_exception": str(e)
                        }
                    ),
                    original_exception=e
                )