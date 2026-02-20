"""
API依赖注入和共享函数

注意：任务执行函数（execute_prediction_task_simple, execute_backtest_task_simple）
会在独立进程中执行，不能依赖全局变量或单例。每个进程必须独立创建所需资源。
"""

import os
from typing import Optional

from fastapi import Header
from loguru import logger

from app.core.database import SessionLocal
from app.models.task_models import TaskStatus
from app.repositories.task_repository import (
    ModelInfoRepository,
    PredictionResultRepository,
    TaskRepository,
)
from app.services.tasks import TaskQueueManager


# 用户认证依赖
async def get_current_user(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    获取当前用户ID

    支持多种认证方式：
    1. X-User-ID 请求头（简单模式，用于开发和内部系统）
    2. Authorization 请求头（Bearer token，用于生产环境）
    3. ��认用户（当没有提供认证信息时）

    Args:
        x_user_id: 通过 X-User-ID 请求头传递的用户ID
        authorization: Authorization 请求头（Bearer token）

    Returns:
        用户ID字符串
    """
    # 优先使用 X-User-ID 请求头
    if x_user_id:
        logger.debug(f"使用 X-User-ID 认证: {x_user_id}")
        return x_user_id

    # 尝试从 Authorization 头解析用户信息
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        # 这里可以添加 JWT 解析逻辑
        # 目前简单处理：如果 token 格式为 "user:{user_id}"，则提取用户ID
        if token.startswith("user:"):
            user_id = token[5:]
            logger.debug(f"使用 Bearer token 认证: {user_id}")
            return user_id

    # 默认用户（用于开发环境或未认证请求）
    default_user = os.getenv("DEFAULT_USER_ID", "default_user")
    logger.debug(f"使用默认用户: {default_user}")
    return default_user


# 全局任务队列管理器实例（仅用于主进程的任务调度）
task_queue_manager = TaskQueueManager()

# 启动任务调度器（在模块加载时启动）
try:
    task_queue_manager.start_all_schedulers()
    logger.info("任务队列管理器已启动")
except Exception as e:
    logger.warning(f"任务队列管理器启动失败: {e}")


def get_task_repository():
    """获取任务仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return TaskRepository(session), session
    except Exception:
        session.close()
        raise


def get_prediction_result_repository():
    """获取预测结果仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return PredictionResultRepository(session), session
    except Exception:
        session.close()
        raise


def get_model_info_repository():
    """获取模型信息仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return ModelInfoRepository(session), session
    except Exception:
        session.close()
        raise


def _parse_bool_env(var_name: str, default: bool = False) -> bool:
    """从环境变量解析布尔值（支持 1/true/yes/on）。"""
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


# 简化的任务执行函数（用于进程池执行）
# 注意：此函数在独立进程中执行，不能使用全局变量或单例
def execute_prediction_task_simple(task_id: str):
    """
    简化的预测任务执行函数（进程池执行）

    重要：此函数在独立进程中执行，必须：
    1. 每个进程独立创建数据库连接
    2. 不使用全局缓存、服务容器等单例
    3. 独立创建所需服务实例
    4. 添加进程ID到日志上下文
    """
    # 绑定进程ID到日志上下文
    process_id = os.getpid()
    task_logger = logger.bind(process_id=process_id, task_id=task_id, log_type="task")

    # 每个进程独立创建数据库连接
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        prediction_result_repository = PredictionResultRepository(session)

        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            task_logger.error(f"任务不存在: {task_id}")
            return

        # 更新任务状态为运行中
        try:
            task_repository.update_task_status(
                task_id=task_id, status=TaskStatus.RUNNING, progress=10.0
            )
            task_logger.info(f"任务状态已更新为RUNNING，进程ID: {process_id}")
        except Exception as status_error:
            task_logger.error(f"任务状态更新失败: {status_error}", exc_info=True)
            raise

        # 解析任务配置
        config = task.config or {}
        stock_codes = config.get("stock_codes", [])

        # 自动补全股票代码后缀（防御性处理）
        def _normalize_code(code: str) -> str:
            code = code.strip()
            if code and '.' not in code and code.isdigit():
                return code + ('.SH' if code.startswith('6') else '.SZ')
            return code

        stock_codes = [_normalize_code(c) for c in stock_codes]
        model_id = config.get("model_id", "default_model")

        task_logger.info(
            f"开始执行预测任务: {task_id}, 股票数量: {len(stock_codes)}, 进程ID: {process_id}"
        )

        # 执行真实预测
        # 每个进程独立创建PredictionEngine实例（不使用全局缓存）
        from app.core.config import settings
        from app.services.prediction.prediction_engine import (
            PredictionConfig,
            PredictionEngine,
        )

        # 创建独立的PredictionEngine实例，不使用全局单例
        prediction_engine = PredictionEngine(
            model_dir=str(settings.MODEL_STORAGE_PATH),
            data_dir=str(settings.DATA_ROOT_PATH),
        )

        prediction_config = PredictionConfig(
            model_id=model_id,
            horizon=config.get("horizon", "short_term"),
            confidence_level=config.get("confidence_level", 0.95),
            features=config.get("features"),
            use_ensemble=config.get("use_ensemble", True),
            risk_assessment=config.get("risk_assessment", True),
        )

        total_stocks = len(stock_codes)
        success_count = 0
        failures = []
        for i, stock_code in enumerate(stock_codes):
            try:
                # 更新进度
                progress = 10 + (i + 1) * 80 / total_stocks
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,  # 添加必需的status参数
                    progress=progress,
                )

                prediction_output = prediction_engine.predict_single_stock(
                    stock_code=stock_code, config=prediction_config
                )

                # 保存预测结果
                prediction_result_repository.save_prediction_result(
                    task_id=task_id,
                    stock_code=stock_code,
                    prediction_date=prediction_output.prediction_date,
                    predicted_price=prediction_output.predicted_price,
                    predicted_direction=prediction_output.predicted_direction,
                    confidence_score=prediction_output.confidence_score,
                    confidence_interval_lower=prediction_output.confidence_interval[0],
                    confidence_interval_upper=prediction_output.confidence_interval[1],
                    model_id=model_id,
                    features_used=prediction_output.features_used,
                    risk_metrics=prediction_output.risk_metrics.to_dict(),
                )

                task_logger.info(
                    f"完成股票预测: {stock_code}, 方向: {prediction_output.predicted_direction}, "
                    f"置信度: {prediction_output.confidence_score:.2f}"
                )
                success_count += 1

            except Exception as e:
                task_logger.error(f"预测股票 {stock_code} 失败: {e}", exc_info=True)
                failures.append(f"{stock_code}: {type(e).__name__} {e}")
                continue

        if success_count == 0:
            details = "; ".join(failures[:5])
            raise RuntimeError(f"所有股票预测失败，未生成任何结果: {details}")

        # 更新任务状态为完成
        task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.COMPLETED, progress=100.0
        )

        task_logger.info(f"预测任务完成: {task_id}, 进程ID: {process_id}")

    except Exception as e:
        task_logger.error(f"执行预测任务失败: {task_id}, 错误: {e}", exc_info=True)
        try:
            # 确保有session和task_repository
            if "task_repository" in locals():
                task_repository.update_task_status(
                    task_id=task_id, status=TaskStatus.FAILED, error_message=str(e)
                )
        except Exception as update_error:
            task_logger.error(f"更新任务状态失败: {update_error}", exc_info=True)
    finally:
        # 确保关闭数据库连接
        if "session" in locals():
            session.close()


def execute_backtest_task_simple(task_id: str):
    """
    简化的回测任务执行函数（进程池执行）

    重要：此函数在独立进程中执行，必须：
    1. 每个进程独立创建数据库连接
    2. 不使用全局缓存、服务容器等单例
    3. 独立创建所需服务实例
    4. 添加进程ID到日志上下文
    """
    import asyncio

    # 绑定进程ID到日志上下文
    process_id = os.getpid()
    task_logger = logger.bind(process_id=process_id, task_id=task_id, log_type="task")

    # 每个进程独立创建数据库连接
    session = SessionLocal()
    task_logger.info(f"开始执行回测任务: {task_id}, 进程ID: {process_id}")

    # 添加全局异常捕获
    try:
        from datetime import datetime, timezone

        from app.core.config import settings
        from app.core.error_handler import ErrorSeverity, TaskError
        from app.services.backtest import BacktestConfig, BacktestExecutor

        task_repository = TaskRepository(session)

        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            task_logger.error(f"任务不存在: {task_id}")
            return

        # 更新任务状态为运行中
        task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.RUNNING, progress=10.0
        )

        # 解析任务配置
        config = task.config or {}
        task_logger.info(f"任务配置: {config}")
        task_logger.info(f"配置键: {list(config.keys())}")
        task_logger.info(f"策略配置 (strategy_config): {config.get('strategy_config', {})}")
        task_logger.info(f"策略配置类型: {type(config.get('strategy_config', {}))}")
        task_logger.info(f"策略配置是否为空: {not config.get('strategy_config', {})}")

        # 检查是否有系统字段被意外包含在配置中
        system_fields = [
            "task_id",
            "task_name",
            "task_type",
            "status",
            "progress",
            "created_at",
            "completed_at",
            "error_message",
            "result",
        ]
        found_system_fields = []
        for field in system_fields:
            if field in config:
                found_system_fields.append(field)
                task_logger.error(f"配置中发现系统字段 '{field}': {config[field]}")
                # 抛出异常来立即停止执行
                raise ValueError(f"配置中发现意外的系统字段 '{field}': {config[field]}")

        if found_system_fields:
            task_logger.warning(f"配置中发现系统字段: {found_system_fields}")

        stock_codes = config.get("stock_codes", [])

        # 自动补全股票代码后缀（防御性处理）
        # 纯数字代码根据规则补 .SZ 或 .SH：6开头为上海，其余为深圳
        def _normalize_stock_code(code: str) -> str:
            code = code.strip()
            if code and '.' not in code and code.isdigit():
                suffix = '.SH' if code.startswith('6') else '.SZ'
                return code + suffix
            return code

        stock_codes = [_normalize_stock_code(c) for c in stock_codes]

        # Bug fix: 展开 stock_pool_type 为完整股票列表
        # 当 stock_pool_type 为 "fixed_500" 时，从 parquet 数据目录读取可用股票
        stock_pool_type = config.get("stock_pool_type", "")
        if stock_pool_type == "fixed_500" and len(stock_codes) <= 1:
            try:
                from pathlib import Path

                data_root = Path(settings.DATA_ROOT_PATH).resolve()
                parquet_dir = data_root / "parquet" / "stock_data"
                if not parquet_dir.exists():
                    # 尝试从项目根目录查找
                    alt_dir = (
                        Path(__file__).resolve().parent.parent.parent.parent
                        / "data"
                        / "parquet"
                        / "stock_data"
                    )
                    if alt_dir.exists():
                        parquet_dir = alt_dir
                if parquet_dir.exists():
                    all_codes = sorted(
                        [
                            f.stem.replace("_", ".")
                            for f in parquet_dir.glob("*.parquet")
                        ]
                    )
                    if len(all_codes) >= 500:
                        stock_codes = all_codes[:500]
                    elif all_codes:
                        stock_codes = all_codes
                    task_logger.info(
                        f"stock_pool_type=fixed_500: 展开为 {len(stock_codes)} 只股票"
                    )
                else:
                    task_logger.warning(
                        f"stock_pool_type=fixed_500 但 parquet 目录不存在: {parquet_dir}"
                    )
            except Exception as pool_err:
                task_logger.error(f"展开 stock_pool_type 失败: {pool_err}")

        strategy_name = config.get("strategy_name") or config.get("strategy", "default_strategy")
        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        initial_cash = config.get("initial_cash", 100000.0)

        # 检查是否有意外的配置字段
        unexpected_keys = []
        for key in ["status", "progress", "result", "completed_at", "error_message"]:
            if key in config:
                unexpected_keys.append(key)
                task_logger.warning(f"配置中包含意外的'{key}'字段: {config[key]}")

        if unexpected_keys:
            task_logger.warning(f"配置中包含意外字段: {unexpected_keys}")

        # 记录所有配置字段
        task_logger.info(f"完整配置字段: {list(config.keys())}")

        task_logger.info(
            f"解析配置: stock_codes={len(stock_codes) if stock_codes else 0}, strategy_name={strategy_name}, start_date={start_date_str}, end_date={end_date_str}"
        )

        if not start_date_str or not end_date_str:
            raise ValueError("回测任务需要提供start_date和end_date")

        # 更健壮的日期解析
        try:
            if isinstance(start_date_str, str):
                # 尝试多种日期格式
                try:
                    start_date = datetime.fromisoformat(start_date_str)
                except ValueError:
                    # 如果是其他格式，尝试解析
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            else:
                start_date = start_date_str

            if isinstance(end_date_str, str):
                try:
                    end_date = datetime.fromisoformat(end_date_str)
                except ValueError:
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            else:
                end_date = end_date_str

        except Exception as date_error:
            task_logger.error(
                f"日期解析失败: start_date={start_date_str}, end_date={end_date_str}, 错误: {date_error}"
            )
            raise ValueError(
                f"无效的日期格式: start_date={start_date_str}, end_date={end_date_str}"
            )

        task_logger.info(
            f"开始执行回测任务: {task_id}, 股票数量: {len(stock_codes)}, 期间: {start_date} - {end_date}, 进程ID: {process_id}"
        )

        # 创建回测执行器（每个进程独立创建，不使用全局单例）
        # 使用配置中的并行化设置
        enable_parallel = getattr(settings, "BACKTEST_PARALLEL_ENABLED", True)
        max_workers = getattr(settings, "BACKTEST_MAX_WORKERS", 4)

        # 性能监控开关：任务级配置优先，其次环境变量兜底
        enable_performance_profiling = bool(
            config.get(
                "enable_performance_profiling",
                _parse_bool_env("ENABLE_BACKTEST_PERFORMANCE_PROFILING", default=False),
            )
        )

        # 创建持久化服务并传给执行器
        from app.services.backtest.persistence import BacktestPersistenceService
        persistence = BacktestPersistenceService()

        executor = BacktestExecutor(
            data_dir=str(settings.DATA_ROOT_PATH),
            enable_parallel=enable_parallel,
            max_workers=max_workers,
            enable_performance_profiling=enable_performance_profiling,
            persistence=persistence,
        )

        # 创建回测配置
        strategy_config = config.get("strategy_config", {})
        backtest_config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=config.get("commission_rate", 0.0003),
            slippage_rate=config.get("slippage_rate", 0.0001),
            enable_unlimited_buy=config.get("enable_unlimited_buy", False)
            or strategy_config.get("enable_unlimited_buy", False),
            # 风控参数：从 config 或 strategy_config 中读取
            max_position_size=config.get("max_position_size",
                strategy_config.get("max_position_size", 0.2)),
            stop_loss_pct=config.get("stop_loss_pct",
                strategy_config.get("stop_loss_pct", 0.05)),
            take_profit_pct=config.get("take_profit_pct",
                strategy_config.get("take_profit_pct", 0.15)),
            max_drawdown_pct=config.get("max_drawdown_pct",
                strategy_config.get("max_drawdown_pct", None)),
        )

        # 执行回测
        task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.RUNNING, progress=30.0
        )

        try:
            # 在新的事件循环中运行异步任务
            import nest_asyncio

            nest_asyncio.apply()

            async def run_backtest_and_save():
                """执行回测并通过 persistence 服务保存详细数据"""
                # 1. 执行回测
                backtest_report = await executor.run_backtest(
                    strategy_name=strategy_name,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_config=config.get("strategy_config", {}),
                    backtest_config=backtest_config,
                    task_id=task_id,
                )

                # 更新进度到90%
                task_repository.update_task_status(
                    task_id=task_id, status=TaskStatus.RUNNING, progress=90.0
                )

                task_logger.info(
                    f"回测执行完成: {task_id}, 总收益: {backtest_report.get('total_return', 0):.2%}, 进程ID: {process_id}"
                )

                # 2. 通过 persistence 服务统一保存所有详细数据
                backtest_id = backtest_report.get("backtest_id", "")
                try:
                    task_logger.info(f"开始保存回测详细数据: {task_id}")
                    success = await persistence.save_backtest_results(
                        task_id=task_id,
                        backtest_id=backtest_id,
                        backtest_report=backtest_report,
                    )
                    if success:
                        task_logger.info(f"回测详细数据保存成功: {task_id}")
                    else:
                        task_logger.error(f"回测详细数据保存返回失败: {task_id}")
                        # persistence 内部已处理 tasks.status，这里兜底
                        task_repository.update_task_status(
                            task_id=task_id,
                            status=TaskStatus.COMPLETED,
                            progress=100.0,
                            result=backtest_report,
                        )
                except Exception as save_error:
                    import traceback as tb
                    task_logger.error(
                        f"保存详细数据时出错: {task_id}, 错误: {type(save_error).__name__}: {save_error}\n{tb.format_exc()}"
                    )
                    # 保存失败不影响主流程，兜底标记完成
                    task_repository.update_task_status(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        progress=100.0,
                        result=backtest_report,
                    )

                return backtest_report

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                backtest_report = loop.run_until_complete(run_backtest_and_save())
            finally:
                loop.close()

        except TaskError as task_error:
            # 处理任务错误（如任务被删除）
            if task_error.severity == ErrorSeverity.LOW:
                # 低严重程度错误（如任务被删除），直接退出，不更新任务状态
                task_logger.info("任务被取消或删除: {}, 原因: {}", task_id, task_error.message)
                return
            else:
                # 其他任务错误，标记为失败
                task_logger.error(
                    "回测任务错误: {}, 错误: {}", task_id, task_error.message, exc_info=True
                )
                try:
                    task_repository.update_task_status(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error_message=f"回测执行失败: {task_error.message}",
                    )
                except Exception:
                    # 如果更新失败（可能任务已被删除），忽略
                    pass
                raise task_error
        except Exception as backtest_error:
            task_logger.error(f"回测执行失败: {task_id}, 错误: {backtest_error}", exc_info=True)
            # 如果回测执行失败，尝试标记任务为失败
            try:
                # 先检查任务是否还存在
                existing_task = task_repository.get_task_by_id(task_id)
                if existing_task:
                    task_repository.update_task_status(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error_message=f"回测执行失败: {str(backtest_error)}",
                    )
                else:
                    task_logger.warning(f"任务不存在，无法更新状态: {task_id}")
            except Exception as update_error:
                task_logger.warning(f"更新任务状态失败: {update_error}")
            raise backtest_error

    except KeyError as ke:
        # 专门处理 KeyError，提供更详细的信息
        error_msg = f"KeyError in backtest task {task_id}: {ke}"
        task_logger.error(error_msg, exc_info=True)
        import traceback

        error_details = traceback.format_exc()
        task_logger.error(f"KeyError 详细堆栈: {error_details}")

        if "config" in locals():
            task_logger.error(f"配置内容: {config}")
            task_logger.error(f"配置类型: {type(config)}")
            if isinstance(config, dict):
                task_logger.error(f"配置键: {list(config.keys())}")
        task_logger.error(f"尝试访问的键: {ke}")

        try:
            if "task_repository" in locals():
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"KeyError: {str(ke)}",
                )
        except Exception as update_error:
            task_logger.error(f"更新任务状态失败: {update_error}", exc_info=True)

    except Exception as e:
        task_logger.error(
            f"执行回测任务失败: {task_id}, 错误类型: {type(e).__name__}, 错误: {e}", exc_info=True
        )
        import traceback

        error_details = traceback.format_exc()
        task_logger.error(f"详细错误信息: {error_details}")

        if "config" in locals():
            task_logger.error(f"配置内容: {config}")
        if "task" in locals():
            task_logger.error(
                f"任务对象: task_id={task.task_id}, task_type={task.task_type}, status={task.status}"
            )

        try:
            if "task_repository" in locals():
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"{type(e).__name__}: {str(e)}",
                )
        except Exception as update_error:
            task_logger.error(f"更新任务状态失败: {update_error}", exc_info=True)
    finally:
        # 确保关闭数据库连接
        if "session" in locals():
            session.close()


def execute_qlib_precompute_task_simple(task_id: str):
    """
    简化的Qlib预计算任务执行函数（进程池执行）

    重要：此函数在独立进程中执行，必须：
    1. 每个进程独立创建数据库连接
    2. 不使用全局缓存、服务容器等单例
    3. 独立创建所需服务实例
    4. 添加进程ID到日志上下文
    """

    # 绑定进程ID到日志上下文
    process_id = os.getpid()
    task_logger = logger.bind(process_id=process_id, task_id=task_id, log_type="task")

    # 每个进程独立创建数据库连接
    session = SessionLocal()
    task_logger.info(f"开始执行Qlib预计算任务: {task_id}, 进程ID: {process_id}")

    # 添加全局异常捕获
    try:
        # 在独立进程中，确保所有类型都正确导入
        import threading
        from datetime import datetime, timezone

        from app.services.tasks.task_execution_engine import QlibPrecomputeTaskExecutor
        from app.services.tasks.task_queue import QueuedTask, TaskExecutionContext

        task_repository = TaskRepository(session)

        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            task_logger.error(f"任务不存在: {task_id}")
            return

        # 更新任务状态为运行中
        task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.RUNNING, progress=0.0
        )

        # 创建QueuedTask对象
        from app.models.task_models import TaskType

        queued_task = QueuedTask(
            task_id=task.task_id,
            task_type=TaskType.QLIB_PRECOMPUTE,
            user_id=task.user_id,
            priority=1,  # 默认优先级
            config=task.config or {},
            created_at=task.created_at or datetime.now(timezone.utc),
        )

        # 创建执行上下文
        cancel_event = threading.Event()
        progress_callback = (
            lambda progress, message: task_repository.update_task_progress(
                task_id, progress
            )
        )

        context = TaskExecutionContext(
            task_id=task_id,
            executor_id=f"process_{process_id}",
            start_time=datetime.now(timezone.utc),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        # 创建任务执行器
        executor = QlibPrecomputeTaskExecutor(task_repository)

        # 执行任务（同步执行，因为已经在独立进程中）
        task_logger.info(f"开始执行Qlib预计算任务: {task_id}")
        result = executor.execute(queued_task, context)

        task_logger.info(
            f"Qlib预计算任务执行完成: {task_id}, 结果: {result.get('success', False)}"
        )

    except Exception as e:
        task_logger.error(f"Qlib预计算任务执行失败: {task_id}, 错误: {e}", exc_info=True)
        try:
            if "task_repository" in locals():
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"{type(e).__name__}: {str(e)}",
                )
        except Exception as update_error:
            task_logger.error(f"更新任务状态失败: {update_error}", exc_info=True)
    finally:
        # 确保关闭数据库连接
        if "session" in locals():
            session.close()
