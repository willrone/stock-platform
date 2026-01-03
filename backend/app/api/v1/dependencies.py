"""
API依赖注入和共享函数
"""

from datetime import datetime
from loguru import logger
from app.core.container import (
    get_data_service, 
    get_indicators_service
)
from app.core.database import SessionLocal
from app.repositories.task_repository import TaskRepository, PredictionResultRepository, ModelInfoRepository
from app.models.task_models import TaskStatus
from app.services.tasks import TaskQueueManager

# 全局任务队列管理器实例
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
    except:
        session.close()
        raise


def get_prediction_result_repository():
    """获取预测结果仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return PredictionResultRepository(session), session
    except:
        session.close()
        raise


def get_model_info_repository():
    """获取模型信息仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return ModelInfoRepository(session), session
    except:
        session.close()
        raise


# 简化的任务执行函数（用于后台任务）
def execute_prediction_task_simple(task_id: str):
    """简化的预测任务执行函数（后台任务）"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        prediction_result_repository = PredictionResultRepository(session)
        
        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return
        
        # 更新任务状态为运行中
        print(f"DEBUG: 准备更新任务状态为 RUNNING")
        try:
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                progress=10.0
            )
            print(f"DEBUG: 任务状态更新成功")
        except Exception as status_error:
            print(f"DEBUG: 任务状态更新失败: {status_error}")
            import traceback
            traceback.print_exc()
        
        # 解析任务配置
        config = task.config or {}
        stock_codes = config.get('stock_codes', [])
        model_id = config.get('model_id', 'default_model')
        
        logger.info(f"开始执行预测任务: {task_id}, 股票数量: {len(stock_codes)}")
        
        # 模拟预测执行（这里可以替换为真实的预测逻辑）
        total_stocks = len(stock_codes)
        for i, stock_code in enumerate(stock_codes):
            try:
                # 更新进度
                progress = 10 + (i + 1) * 80 / total_stocks
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,  # 添加必需的status参数
                    progress=progress
                )
                
                # TODO: 这里应该调用真实的预测引擎
                # prediction_result = prediction_engine.predict(stock_code, model_id)
                
                # 模拟预测结果（临时实现）
                import random
                predicted_direction = random.choice([-1, 0, 1])
                confidence_score = random.uniform(0.6, 0.95)
                
                # 保存预测结果
                prediction_result_repository.save_prediction_result(
                    task_id=task_id,
                    stock_code=stock_code,
                    prediction_date=datetime.utcnow(),
                    predicted_price=100.0,  # 临时值
                    predicted_direction=predicted_direction,
                    confidence_score=confidence_score,
                    confidence_interval_lower=confidence_score - 0.1,
                    confidence_interval_upper=confidence_score + 0.1,
                    model_id=model_id,
                    features_used=[],
                    risk_metrics={}
                )
                
                logger.info(f"完成股票预测: {stock_code}, 方向: {predicted_direction}, 置信度: {confidence_score:.2f}")
                
            except Exception as e:
                logger.error(f"预测股票 {stock_code} 失败: {e}", exc_info=True)
                continue
        
        # 更新任务状态为完成
        task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0
        )
        
        logger.info(f"预测任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"执行预测任务失败: {task_id}, 错误: {e}", exc_info=True)
        try:
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
        except:
            pass
    finally:
        session.close()


def execute_backtest_task_simple(task_id: str):
    """简化的回测任务执行函数（后台任务）"""
    import asyncio
    session = SessionLocal()
    print(f"DEBUG: 开始执行回测任务函数: {task_id}")  # 临时调试
    logger.info(f"开始执行回测任务函数: {task_id}")

    # 添加全局异常捕获
    try:
        from app.services.backtest import BacktestExecutor, BacktestConfig
        from app.core.config import settings
        from datetime import datetime
        
        task_repository = TaskRepository(session)
        
        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return
        
        # 更新任务状态为运行中
        task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=10.0
        )
        
        # 解析任务配置
        config = task.config or {}
        print(f"DEBUG: 原始配置: {config}")
        logger.info(f"任务配置: {config}")
        logger.info(f"配置键: {list(config.keys())}")

        # 检查是否有系统字段被意外包含在配置中
        system_fields = ['task_id', 'task_name', 'task_type', 'status', 'progress', 'created_at', 'completed_at', 'error_message', 'result']
        found_system_fields = []
        for field in system_fields:
            if field in config:
                found_system_fields.append(field)
                print(f"DEBUG: 配置中发现系统字段 '{field}': {config[field]}")
                # 抛出异常来立即停止执行
                raise ValueError(f"配置中发现意外的系统字段 '{field}': {config[field]}")

        if found_system_fields:
            print(f"DEBUG: 配置中发现系统字段: {found_system_fields}")
            logger.warning(f"配置中发现系统字段: {found_system_fields}")

        stock_codes = config.get('stock_codes', [])
        strategy_name = config.get('strategy_name', 'default_strategy')
        start_date_str = config.get('start_date')
        end_date_str = config.get('end_date')
        initial_cash = config.get('initial_cash', 100000.0)

        # 检查是否有意外的配置字段
        unexpected_keys = []
        for key in ['status', 'progress', 'result', 'completed_at', 'error_message']:
            if key in config:
                unexpected_keys.append(key)
                logger.warning(f"配置中包含意外的'{key}'字段: {config[key]}")

        if unexpected_keys:
            logger.warning(f"配置中包含意外字段: {unexpected_keys}")

        # 记录所有配置字段
        logger.info(f"完整配置字段: {list(config.keys())}")

        logger.info(f"解析配置: stock_codes={len(stock_codes) if stock_codes else 0}, strategy_name={strategy_name}, start_date={start_date_str}, end_date={end_date_str}")
        
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
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            else:
                start_date = start_date_str

            if isinstance(end_date_str, str):
                try:
                    end_date = datetime.fromisoformat(end_date_str)
                except ValueError:
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            else:
                end_date = end_date_str

        except Exception as date_error:
            logger.error(f"日期解析失败: start_date={start_date_str}, end_date={end_date_str}, 错误: {date_error}")
            raise ValueError(f"无效的日期格式: start_date={start_date_str}, end_date={end_date_str}")
        
        logger.info(f"开始执行回测任务: {task_id}, 股票数量: {len(stock_codes)}, 期间: {start_date} - {end_date}")
        
        # 创建回测执行器
        executor = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH))
        
        # 创建回测配置
        backtest_config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=config.get('commission_rate', 0.0003),
            slippage_rate=config.get('slippage_rate', 0.0001)
        )
        
        # 执行回测
        task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=30.0
        )
        
        try:
            # 创建异步任务并等待完成
            async def run_async_backtest():
                return await executor.run_backtest(
                    strategy_name=strategy_name,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_config=config.get('strategy_config', {}),
                    backtest_config=backtest_config,
                    task_id=task_id
                )

            # 在新的事件循环中运行异步任务
            import nest_asyncio
            nest_asyncio.apply()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                backtest_report = loop.run_until_complete(run_async_backtest())
            finally:
                loop.close()
            
            # 更新进度到90%
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                progress=90.0
            )
            
            # 保存回测结果并完成任务
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                progress=100.0,
                result=backtest_report
            )
            
            logger.info(f"回测任务完成: {task_id}, 总收益: {backtest_report.get('total_return', 0):.2%}")
            
            # 异步保存详细数据到数据库
            try:
                logger.info(f"开始保存回测详细数据: {task_id}")
                
                async def save_detailed_data():
                    """异步保存详细数据"""
                    from app.core.database import get_async_session
                    from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
                    from app.services.backtest.backtest_data_adapter import BacktestDataAdapter
                    
                    adapter = BacktestDataAdapter()
                    
                    # 转换数据
                    enhanced_result = await adapter.adapt_backtest_result(backtest_report)
                    
                    async for session in get_async_session():
                        try:
                            repository = BacktestDetailedRepository(session)
                            
                            # 创建详细结果记录
                            await repository.create_detailed_result(
                                task_id=task_id,
                                backtest_id=f"bt_{task_id[:8]}",
                                sortino_ratio=enhanced_result.extended_risk_metrics.sortino_ratio if enhanced_result.extended_risk_metrics else 0,
                                calmar_ratio=enhanced_result.extended_risk_metrics.calmar_ratio if enhanced_result.extended_risk_metrics else 0,
                                max_drawdown_duration=enhanced_result.extended_risk_metrics.max_drawdown_duration if enhanced_result.extended_risk_metrics else 0,
                                var_95=enhanced_result.extended_risk_metrics.var_95 if enhanced_result.extended_risk_metrics else 0,
                                downside_deviation=enhanced_result.extended_risk_metrics.downside_deviation if enhanced_result.extended_risk_metrics else 0,
                                drawdown_analysis=enhanced_result.drawdown_analysis.to_dict() if enhanced_result.drawdown_analysis else {},
                                monthly_returns=[mr.to_dict() for mr in enhanced_result.monthly_returns] if enhanced_result.monthly_returns else [],
                                position_analysis=[pa.to_dict() for pa in enhanced_result.position_analysis] if enhanced_result.position_analysis else [],
                                benchmark_comparison=enhanced_result.benchmark_data or {},
                                rolling_metrics={}
                            )
                            
                            # 创建组合快照记录
                            if enhanced_result.portfolio_history:
                                for snapshot in enhanced_result.portfolio_history:
                                    await repository.create_portfolio_snapshot(
                                        task_id=task_id,
                                        backtest_id=f"bt_{task_id[:8]}",
                                        snapshot_date=snapshot.get("date"),
                                        portfolio_value=snapshot.get("portfolio_value", 0),
                                        cash=snapshot.get("cash", 0),
                                        positions_count=snapshot.get("positions_count", 0),
                                        total_return=snapshot.get("total_return", 0),
                                        drawdown=0,
                                        positions=snapshot.get("positions", {})
                                    )
                            
                            # 创建交易记录
                            if enhanced_result.trade_history:
                                for trade in enhanced_result.trade_history:
                                    await repository.create_trade_record(
                                        task_id=task_id,
                                        backtest_id=f"bt_{task_id[:8]}",
                                        trade_id=trade.get("trade_id", ""),
                                        stock_code=trade.get("stock_code", ""),
                                        stock_name=trade.get("stock_code", ""),
                                        action=trade.get("action", ""),
                                        quantity=trade.get("quantity", 0),
                                        price=trade.get("price", 0),
                                        timestamp=trade.get("timestamp"),
                                        commission=trade.get("commission", 0),
                                        pnl=trade.get("pnl", 0),
                                        holding_days=trade.get("holding_days", 0),
                                        technical_indicators={}
                                    )
                            
                            await session.commit()
                            logger.info(f"回测详细数据保存成功: {task_id}")
                            
                        except Exception as e:
                            await session.rollback()
                            logger.error(f"保存回测详细数据失败: {task_id}, 错误: {e}", exc_info=True)
                        break
                
                # 在新的事件循环中运行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(save_detailed_data())
                finally:
                    loop.close()
                    
            except Exception as save_error:
                logger.error(f"保存详细数据时出错: {task_id}, 错误: {save_error}", exc_info=True)
                # 不影响主流程，继续执行
            
        except Exception as backtest_error:
            logger.error(f"回测执行失败: {task_id}, 错误: {backtest_error}", exc_info=True)
            # 如果回测执行失败，标记任务为失败
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=f"回测执行失败: {str(backtest_error)}"
            )
            raise backtest_error
        
    except KeyError as ke:
        # 专门处理 KeyError，提供更详细的信息
        error_msg = f"KeyError in backtest task {task_id}: {ke}"
        print(f"DEBUG: {error_msg}")  # 临时输出到控制台

        # 写入调试文件
        try:
            with open('/tmp/backtest_debug.log', 'a') as f:
                f.write(f"\n=== KeyError Debug {datetime.now()} ===\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Error: {ke}\n")
                f.write(f"Config: {config}\n")
                f.write(f"Config type: {type(config)}\n")
                if isinstance(config, dict):
                    f.write(f"Config keys: {list(config.keys())}\n")
                import traceback
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
                f.write("=" * 50 + "\n")
        except Exception as file_error:
            print(f"DEBUG: 无法写入调试文件: {file_error}")

        logger.error(error_msg, exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"KeyError 详细堆栈: {error_details}")
        logger.error(f"配置内容: {config}")
        logger.error(f"配置类型: {type(config)}")
        if isinstance(config, dict):
            logger.error(f"配置键: {list(config.keys())}")
        logger.error(f"尝试访问的键: {ke}")

        # 尝试确定是哪个对象导致了 KeyError
        try:
            # 检查是否是 config 相关的
            if str(ke).strip("'\"") in ['status', 'progress', 'result', 'completed_at']:
                logger.error("KeyError 可能与任务状态或结果访问相关")
            elif str(ke).strip("'\"") in config:
                logger.error(f"键 '{ke}' 存在于配置中，但访问失败")
            else:
                logger.error(f"键 '{ke}' 不存在于配置中")
        except:
            pass

        try:
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=f"KeyError: {str(ke)}"
            )
        except Exception as update_error:
            logger.error(f"更新任务状态失败: {update_error}", exc_info=True)

    except Exception as e:
        logger.error(f"执行回测任务失败: {task_id}, 错误类型: {type(e).__name__}, 错误: {e}", exc_info=True)
        # 记录更详细的错误信息
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"详细错误信息: {error_details}")
        logger.error(f"配置内容: {config}")
        logger.error(f"任务对象: task_id={task.task_id}, task_type={task.task_type}, status={task.status}")

        try:
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=f"{type(e).__name__}: {str(e)}"
            )
        except Exception as update_error:
            logger.error(f"更新任务状态失败: {update_error}", exc_info=True)
            import traceback
            update_details = traceback.format_exc()
            logger.error(f"更新任务状态详细错误: {update_details}")
    finally:
        session.close()
