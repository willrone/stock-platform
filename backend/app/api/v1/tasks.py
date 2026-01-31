"""
任务管理路由
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app.api.v1.dependencies import (
    execute_backtest_task_simple,
    execute_prediction_task_simple,
)
from app.api.v1.schemas import (
    BacktestCompareRequest,
    BacktestExportRequest,
    StandardResponse,
    TaskCreateRequest,
)
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.error_handler import TaskError
from app.models.task_models import TaskStatus, TaskType
from app.repositories.task_repository import PredictionResultRepository, TaskRepository
from app.services.data.stock_data_loader import StockDataLoader
from app.services.prediction.prediction_engine import PredictionConfig, PredictionEngine
from app.services.tasks.process_executor import get_process_executor
from app.services.tasks.task_monitor import task_monitor

router = APIRouter(prefix="/tasks", tags=["任务管理"])


@router.post("", response_model=StandardResponse)
async def create_task(request: TaskCreateRequest):
    """创建任务（支持预测和回测）"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)

        # 确定任务类型
        task_type_str = request.task_type.lower() if request.task_type else "prediction"
        if task_type_str == "backtest":
            task_type = TaskType.BACKTEST
        else:
            task_type = TaskType.PREDICTION

        # 构建任务配置
        if task_type == TaskType.PREDICTION:
            if not request.model_id:
                raise HTTPException(status_code=400, detail="预测任务需要提供model_id")
            config = {
                "stock_codes": request.stock_codes,
                "model_id": request.model_id,
                **(request.prediction_config or {}),
            }
        else:  # BACKTEST
            if not request.backtest_config:
                raise HTTPException(status_code=400, detail="回测任务需要提供backtest_config")

            # 兼容前端/调用方使用 backtest_config 内的 strategy_type / strategy_params 字段。
            # 后端执行器期望字段：
            # - strategy_name: 策略名称（如 multi_factor / momentum_factor / low_volatility 等）
            # - strategy_config: 策略参数字典
            #
            # 这里做一次映射：
            # - strategy_type -> strategy_name
            # - strategy_params -> strategy_config
            backtest_config = dict(request.backtest_config or {})
            if "strategy_type" in backtest_config and "strategy_name" not in backtest_config:
                backtest_config["strategy_name"] = backtest_config["strategy_type"]
            if "strategy_params" in backtest_config and "strategy_config" not in backtest_config:
                backtest_config["strategy_config"] = backtest_config["strategy_params"]

            config = {"stock_codes": request.stock_codes, **backtest_config}

        # 创建任务
        task = task_repository.create_task(
            task_name=request.task_name,
            task_type=task_type,
            user_id="default_user",  # TODO: 从认证中获取真实用户ID
            config=config,
        )

        # 将任务提交到进程池执行（异步，不阻塞）
        try:
            process_executor = get_process_executor()
            loop = asyncio.get_event_loop()

            # 使用run_in_executor将任务提交到进程池
            # 注意：这里只是提交任务，不等待结果，立即返回
            if task_type == TaskType.PREDICTION:
                future = process_executor.submit(
                    execute_prediction_task_simple, task.task_id
                )
            else:  # BACKTEST
                future = process_executor.submit(
                    execute_backtest_task_simple, task.task_id
                )

            logger.info(f"任务已提交到进程池: {task.task_id}, 类型: {task_type.value}")
        except Exception as submit_error:
            logger.error(f"将任务提交到进程池时出错: {submit_error}", exc_info=True)
            # 如果提交失败，标记任务为失败
            try:
                task_repository.update_task_status(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"任务提交失败: {str(submit_error)}",
                )
            except:
                pass

        # 转换为前端期望的格式
        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "stock_codes": request.stock_codes,
            "model_id": config.get("model_id", ""),
            "created_at": task.created_at.isoformat()
            if task.created_at
            else datetime.now().isoformat(),
            "completed_at": task.completed_at.isoformat()
            if task.completed_at
            else None,
            "error_message": task.error_message,
        }

        return StandardResponse(success=True, message="任务创建成功", data=task_data)
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"创建任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")
    finally:
        session.close()


@router.get("", response_model=StandardResponse)
async def list_tasks(status: Optional[str] = None, limit: int = 20, offset: int = 0):
    """获取任务列表"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)

        # 转换状态字符串为TaskStatus枚举
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status)
            except ValueError:
                logger.warning(f"无效的任务状态: {status}")

        # 获取任务列表
        tasks = task_repository.get_tasks_by_user(
            user_id="default_user",
            limit=limit,
            offset=offset,
            status_filter=status_filter,
        )

        # 获取总数
        total_tasks = task_repository.get_tasks_by_user(
            user_id="default_user", limit=10000, offset=0, status_filter=status_filter
        )
        total = len(total_tasks)

        # 转换为前端期望的格式
        task_list = []
        for task in tasks:
            config = task.config or {}
            stock_codes = config.get("stock_codes", [])
            model_id = config.get("model_id", "")

            # 处理 task_type：可能是字符串或枚举对象
            task_type_value = None
            if task.task_type:
                if hasattr(task.task_type, 'value'):
                    # 如果是枚举对象，获取其值
                    task_type_value = task.task_type.value
                else:
                    # 如果已经是字符串，直接使用
                    task_type_value = task.task_type
            
            task_data = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_type": task_type_value,  # 添加任务类型字段
                "status": task.status,
                "progress": task.progress,
                "stock_codes": stock_codes if isinstance(stock_codes, list) else [],
                "model_id": model_id,
                "created_at": task.created_at.isoformat()
                if task.created_at
                else datetime.now().isoformat(),
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
                "error_message": task.error_message,
                "config": config,  # 添加完整配置，方便前端使用
            }
            task_list.append(task_data)

        return StandardResponse(
            success=True,
            message="任务列表获取成功",
            data={"tasks": task_list, "total": total, "limit": limit, "offset": offset},
        )
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")
    finally:
        session.close()


@router.get("/{task_id}/detailed", response_model=StandardResponse)
async def get_task_detailed_result(task_id: str):
    """获取任务的详细回测结果（用于可视化）"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)

        # 获取基础任务信息
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        if task.task_type != "backtest":
            raise HTTPException(status_code=400, detail="只有回测任务支持详细结果查看")

        # 获取原始回测结果
        raw_result = task.result
        if not raw_result:
            raise HTTPException(status_code=404, detail="回测结果不存在")

        # 使用适配器转换数据
        from app.services.backtest.utils import BacktestDataAdapter

        adapter = BacktestDataAdapter()

        # 确保raw_result是字典格式
        if isinstance(raw_result, str):
            import json

            raw_result = json.loads(raw_result)

        enhanced_result = await adapter.adapt_backtest_result(raw_result)

        return StandardResponse(
            success=True, message="获取详细回测结果成功", data=enhanced_result.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取详细回测结果失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取详细回测结果失败: {str(e)}")
    finally:
        session.close()


@router.get("/{task_id}/prediction-series", response_model=StandardResponse)
async def get_prediction_series(
    task_id: str, stock_code: str, lookback_days: Optional[int] = None
):
    """获取预测任务的历史预测与实际价格序列"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.task_type != "prediction":
            raise HTTPException(status_code=400, detail="仅预测任务支持该接口")

        config = task.config or {}
        model_id = config.get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="预测任务缺少model_id")

        loader = StockDataLoader(data_root=str(settings.DATA_ROOT_PATH))
        actual_data = loader.load_stock_data(stock_code)
        if actual_data.empty or "close" not in actual_data.columns:
            raise HTTPException(status_code=404, detail="实际价格数据不存在")

        end_date = actual_data.index.max().to_pydatetime()
        start_date = actual_data.index.min().to_pydatetime()
        if lookback_days is not None and lookback_days > 0:
            start_date = end_date - timedelta(days=lookback_days)

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

        predicted_returns = await prediction_engine.predict_return_series(
            stock_code=stock_code,
            config=prediction_config,
            start_date=start_date,
            end_date=end_date,
        )
        if not predicted_returns.empty:
            abs_returns = predicted_returns.abs()
            logger.info(
                "预测收益率统计: count={}, mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}",
                len(predicted_returns),
                float(predicted_returns.mean()),
                float(predicted_returns.std()),
                float(predicted_returns.min()),
                float(predicted_returns.max()),
            )
            logger.info(
                "预测收益率绝对值分位数: p50={:.6f}, p90={:.6f}, p95={:.6f}",
                float(abs_returns.quantile(0.5)),
                float(abs_returns.quantile(0.9)),
                float(abs_returns.quantile(0.95)),
            )
            unique_values = predicted_returns.round(6).unique()
            logger.info(
                "预测收益率唯一值数量: {}, 前几个值: {}",
                len(unique_values),
                unique_values[:5].tolist(),
            )

        actual_data = actual_data[
            (actual_data.index >= pd.Timestamp(start_date))
            & (actual_data.index <= pd.Timestamp(end_date))
        ]

        actual_close_by_date = {}
        for idx, row in actual_data.iterrows():
            date_key = pd.Timestamp(idx).normalize().strftime("%Y-%m-%d")
            actual_close_by_date[date_key] = float(row["close"])

        actual_date_keys = sorted(actual_close_by_date.keys())
        actual_date_index = {date_key: i for i, date_key in enumerate(actual_date_keys)}

        horizon_map = {
            "intraday": 1,
            "short_term": 5,
            "medium_term": 20,
            "long_term": 60,
        }
        horizon_days = horizon_map.get(config.get("horizon", "short_term"), 5)

        series = []
        for date, predicted_return in predicted_returns.items():
            if isinstance(date, tuple):
                date = date[-1]
            date_key = pd.Timestamp(date).normalize().strftime("%Y-%m-%d")
            origin_price = actual_close_by_date.get(date_key)
            origin_index = actual_date_index.get(date_key)
            if origin_price is None or origin_index is None:
                continue

            target_index = origin_index + horizon_days
            if target_index >= len(actual_date_keys):
                continue

            target_date_key = actual_date_keys[target_index]
            target_actual = actual_close_by_date.get(target_date_key)
            if target_actual is None:
                continue

            predicted_price = origin_price * (1 + float(predicted_return))
            series.append(
                {
                    "date": target_date_key,
                    "actual": target_actual,
                    "predicted": predicted_price,
                }
            )

        return StandardResponse(
            success=True,
            message="预测序列获取成功",
            data={"stock_code": stock_code, "series": series},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取预测序列失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取预测序列失败: {str(e)}")
    finally:
        session.close()


@router.get("/{task_id}/charts/{chart_type}")
async def get_chart_data(task_id: str, chart_type: str):
    """获取特定图表数据"""

    print(f"DEBUG: 请求图表数据 - task_id: {task_id}, chart_type: {chart_type}")

    valid_chart_types = [
        "equity_curve",
        "drawdown_curve",
        "monthly_heatmap",
        "trade_distribution",
        "position_weights",
        "risk_metrics",
    ]

    if chart_type not in valid_chart_types:
        print(f"DEBUG: 不支持的图表类型: {chart_type}")
        raise HTTPException(status_code=400, detail=f"不支持的图表类型: {chart_type}")

    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)

        print(
            f"DEBUG: 任务查询结果 - task: {task is not None}, result: {task.result is not None if task else None}"
        )

        if not task or not task.result:
            print(
                f"DEBUG: 回测数据不存在 - task: {task}, result: {task.result if task else None}"
            )
            raise HTTPException(status_code=404, detail="回测数据不存在")

        if task.task_type != "backtest":
            print(f"DEBUG: 任务类型不是回测 - task_type: {task.task_type}")
            raise HTTPException(status_code=400, detail="只有回测任务支持图表数据")

        # 获取原始回测结果
        raw_result = task.result
        print(f"DEBUG: 原始结果类型: {type(raw_result)}")
        if isinstance(raw_result, str):
            import json

            raw_result = json.loads(raw_result)
            print(f"DEBUG: 解析后的结果类型: {type(raw_result)}")

        # 生成图表数据
        print(f"DEBUG: 开始生成图表数据 - chart_type: {chart_type}")
        from app.services.backtest.reporting import ChartDataGenerator

        chart_generator = ChartDataGenerator()
        chart_data = await chart_generator.generate_chart_data(raw_result, chart_type)

        print(f"DEBUG: 图表数据生成成功")

        return StandardResponse(
            success=True,
            message="获取图表数据成功",
            data={"chart_type": chart_type, "chart_data": chart_data},
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: 图表数据生成失败 - error: {e}")
        logger.error(f"获取图表数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取图表数据失败: {str(e)}")
    finally:
        session.close()


@router.post("/compare", response_model=StandardResponse)
async def compare_backtest_results(request: BacktestCompareRequest):
    """对比多个回测结果"""

    if len(request.task_ids) > 5:
        raise HTTPException(status_code=400, detail="最多支持对比5个回测结果")

    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        comparison_results = []

        for task_id in request.task_ids:
            task = task_repository.get_task_by_id(task_id)
            if not task:
                raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

            if task.task_type != "backtest":
                raise HTTPException(status_code=400, detail=f"任务 {task_id} 不是回测任务")

            if not task.result:
                raise HTTPException(status_code=404, detail=f"任务 {task_id} 没有回测结果")

            # 转换结果数据
            raw_result = task.result
            if isinstance(raw_result, str):
                import json

                raw_result = json.loads(raw_result)

            from app.services.backtest.utils import BacktestDataAdapter

            adapter = BacktestDataAdapter()
            enhanced_result = await adapter.adapt_backtest_result(raw_result)

            comparison_results.append(
                {
                    "task_id": task_id,
                    "task_name": task.task_name,
                    "result": enhanced_result.to_dict(),
                }
            )

        # 计算对比指标
        from app.services.backtest.analysis import BacktestComparisonAnalyzer

        comparison_analyzer = BacktestComparisonAnalyzer()
        comparison_analysis = await comparison_analyzer.analyze_comparison(
            comparison_results, request.comparison_metrics
        )

        return StandardResponse(
            success=True,
            message="回测对比分析完成",
            data={
                "individual_results": comparison_results,
                "comparison_analysis": comparison_analysis,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"回测对比分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"回测对比分析失败: {str(e)}")
    finally:
        session.close()


@router.post("/{task_id}/export", response_model=StandardResponse)
async def export_backtest_report(task_id: str, export_request: BacktestExportRequest):
    """导出回测报告"""

    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)

        if not task or not task.result:
            raise HTTPException(status_code=404, detail="回测数据不存在")

        if task.task_type != "backtest":
            raise HTTPException(status_code=400, detail="只有回测任务支持报告导出")

        # 获取原始回测结果
        raw_result = task.result
        if isinstance(raw_result, str):
            import json

            raw_result = json.loads(raw_result)

        # 生成报告
        from app.services.backtest.reporting import BacktestReportGenerator

        report_generator = BacktestReportGenerator()

        if export_request.format == "pdf":
            report_path = await report_generator.generate_pdf_report(
                raw_result, export_request.include_charts, export_request.include_tables
            )
        elif export_request.format == "excel":
            report_path = await report_generator.generate_excel_report(
                raw_result, export_request.include_raw_data
            )
        else:
            raise HTTPException(status_code=400, detail="不支持的导出格式")

        import os

        return StandardResponse(
            success=True,
            message="报告生成成功",
            data={
                "download_url": f"/api/v1/files/download/{os.path.basename(report_path)}",
                "file_name": os.path.basename(report_path),
                "file_size": os.path.getsize(report_path)
                if os.path.exists(report_path)
                else 0,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出回测报告失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"导出回测报告失败: {str(e)}")
    finally:
        session.close()


@router.get("/{task_id}", response_model=StandardResponse)
async def get_task_detail(task_id: str):
    """获取任务详情"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        prediction_result_repository = PredictionResultRepository(session)

        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        # 获取任务配置
        config = task.config or {}
        stock_codes = config.get("stock_codes", [])
        model_id = config.get("model_id", "")

        # 获取预测结果
        prediction_results = (
            prediction_result_repository.get_prediction_results_by_task(task_id)
        )

        # 构建预测结果列表
        from app.core.config import settings
        from app.services.data.stock_data_loader import StockDataLoader

        stock_loader = StockDataLoader(data_root=str(settings.DATA_ROOT_PATH))
        latest_prices = {}
        predictions = []
        total_confidence = 0.0
        for result in prediction_results:
            if result.stock_code not in latest_prices:
                latest_price = None
                try:
                    data = stock_loader.load_stock_data(
                        result.stock_code, end_date=result.prediction_date
                    )
                    if not data.empty and "close" in data.columns:
                        latest_price = float(data["close"].iloc[-1])
                except Exception:
                    latest_price = None
                latest_prices[result.stock_code] = latest_price

            current_price = latest_prices.get(result.stock_code)
            predicted_return = 0.0
            if current_price:
                predicted_return = (
                    result.predicted_price - current_price
                ) / current_price

            prediction = {
                "stock_code": result.stock_code,
                "predicted_direction": result.predicted_direction,
                "predicted_return": predicted_return,
                "confidence_score": result.confidence_score,
                "confidence_interval": {
                    "lower": result.confidence_interval_lower or 0,
                    "upper": result.confidence_interval_upper or 0,
                },
                "risk_assessment": result.risk_metrics or {},
            }
            predictions.append(prediction)
            total_confidence += result.confidence_score

        # 计算平均置信度
        average_confidence = (
            total_confidence / len(prediction_results) if prediction_results else 0.0
        )

        # 获取回测结果（如果任务类型是回测，或者结果中包含回测数据）
        backtest_results = None
        if task.task_type == "backtest" or (
            task.result and isinstance(task.result, (dict, str))
        ):
            logger.info(
                f"处理任务结果: task_id={task_id}, task_type={task.task_type}, result存在={task.result is not None}, result类型={type(task.result)}"
            )
            if task.result:
                try:
                    import json

                    if isinstance(task.result, str):
                        parsed_result = json.loads(task.result)
                    else:
                        parsed_result = task.result

                    # 检查是否包含回测相关的字段
                    is_backtest_data = False
                    if isinstance(parsed_result, dict):
                        # 检查是否包含回测相关的关键字段
                        backtest_keys = [
                            "equity_curve",
                            "drawdown_curve",
                            "portfolio",
                            "risk_metrics",
                            "trade_history",
                            "dates",
                        ]
                        is_backtest_data = any(
                            key in parsed_result for key in backtest_keys
                        )

                    # 如果是回测任务，或者结果中包含回测数据，则使用该结果
                    if task.task_type == "backtest" or is_backtest_data:
                        backtest_results = parsed_result
                        logger.info(
                            f"回测结果解析成功: task_id={task_id}, 包含字段={list(backtest_results.keys())[:20] if isinstance(backtest_results, dict) else '非字典类型'}"
                        )
                        if isinstance(backtest_results, dict):
                            logger.info(
                                f"回测结果关键字段: equity_curve={len(backtest_results.get('equity_curve', []))}, "
                                f"portfolio={backtest_results.get('portfolio') is not None}, "
                                f"risk_metrics={backtest_results.get('risk_metrics') is not None}"
                            )
                    else:
                        logger.debug(f"任务结果不包含回测数据: task_id={task_id}")
                except Exception as e:
                    logger.warning(f"解析回测结果失败: {e}", exc_info=True)
            else:
                if task.task_type == "backtest":
                    logger.warning(
                        f"回测任务但无结果数据: task_id={task_id}, result={task.result}"
                    )

        # 构建任务详情（含 config，供回测结果页策略配置卡片等使用）
        task_detail = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "stock_codes": stock_codes if isinstance(stock_codes, list) else [],
            "model_id": model_id,
            "created_at": task.created_at.isoformat()
            if task.created_at
            else datetime.now().isoformat(),
            "completed_at": task.completed_at.isoformat()
            if task.completed_at
            else None,
            "error_message": task.error_message,
            "config": config,
            "results": {
                "total_stocks": len(stock_codes)
                if isinstance(stock_codes, list)
                else 0,
                "successful_predictions": len(prediction_results),
                "average_confidence": average_confidence,
                "predictions": predictions,
                "backtest_results": backtest_results,
            },
            "backtest_results": backtest_results
            if backtest_results is not None
            else None,
            "result": backtest_results if backtest_results is not None else None,
        }

        # 添加调试日志
        if backtest_results is not None:
            logger.info(f"回测结果详情返回: task_id={task_id}, task_type={task.task_type}")
            if isinstance(backtest_results, dict):
                logger.info(f"回测结果包含字段: {list(backtest_results.keys())[:20]}")

        return StandardResponse(success=True, message="任务详情获取成功", data=task_detail)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")
    finally:
        session.close()


@router.delete("/{task_id}", response_model=StandardResponse)
async def delete_task(
    task_id: str, force: bool = Query(False, description="是否强制删除运行中的任务")
):
    """删除任务"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)

        try:
            success = task_repository.delete_task(
                task_id=task_id, user_id="default_user", force=force
            )

            if not success:
                raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

            return StandardResponse(
                success=True,
                message="任务删除成功",
                data={"task_id": task_id, "force": force},
            )

        except TaskError as e:
            # TaskError 包含更详细的错误信息
            error_message = e.message
            if "正在运行中" in error_message or "无权限" in error_message:
                raise HTTPException(status_code=400, detail=error_message)
            elif "数据库约束" in error_message or "关联数据" in error_message:
                raise HTTPException(status_code=409, detail=error_message)  # Conflict
            else:
                raise HTTPException(status_code=500, detail=error_message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {e}", exc_info=True)
        error_msg = str(e)
        # 检查是否是数据库约束错误
        if "foreign key" in error_msg.lower() or "constraint" in error_msg.lower():
            raise HTTPException(
                status_code=409, detail=f"删除任务失败：存在关联数据。请先删除相关数据，或使用强制删除（force=true）。"
            )
        else:
            raise HTTPException(status_code=500, detail=f"删除任务失败: {error_msg}")
    finally:
        session.close()


@router.post("/{task_id}/stop", response_model=StandardResponse)
async def stop_task(task_id: str):
    """停止运行中的任务"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        if task.status not in [TaskStatus.RUNNING.value, TaskStatus.QUEUED.value]:
            raise HTTPException(status_code=400, detail=f"任务状态为 {task.status}，无法停止")

        # 更新任务状态为已取消
        task = task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.CANCELLED
        )

        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "progress": task.progress,
        }

        return StandardResponse(success=True, message="任务已停止", data=task_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")
    finally:
        session.close()


@router.post("/{task_id}/retry", response_model=StandardResponse)
async def retry_task(task_id: str):
    """重新运行失败的任务"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        if task.status not in [TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
            raise HTTPException(status_code=400, detail=f"任务状态为 {task.status}，无法重试")

        # 重置任务状态
        task = task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.CREATED, progress=0.0
        )

        # 将任务提交到进程池重新执行
        try:
            process_executor = get_process_executor()

            if task.task_type == "prediction":
                future = process_executor.submit(
                    execute_prediction_task_simple, task_id
                )
            elif task.task_type == "backtest":
                future = process_executor.submit(execute_backtest_task_simple, task_id)
            else:
                raise HTTPException(
                    status_code=400, detail=f"不支持的任务类型: {task.task_type}"
                )

            logger.info(f"任务已重新提交到进程池: {task_id}")
        except Exception as submit_error:
            logger.error(f"重新提交任务到进程池失败: {submit_error}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"重新提交任务失败: {str(submit_error)}"
            )

        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "progress": task.progress,
        }

        return StandardResponse(success=True, message="任务已重新创建", data=task_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重试任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")
    finally:
        session.close()


@router.get("/stats", response_model=StandardResponse)
async def get_task_stats():
    """获取任务统计信息"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        stats = task_repository.get_task_statistics(user_id="default_user", days=30)

        # 转换为前端期望的格式
        status_counts = stats.get("status_counts", {})
        task_stats = {
            "total": stats.get("total_tasks", 0),
            "completed": status_counts.get(TaskStatus.COMPLETED.value, 0),
            "running": status_counts.get(TaskStatus.RUNNING.value, 0)
            + status_counts.get(TaskStatus.QUEUED.value, 0),
            "failed": status_counts.get(TaskStatus.FAILED.value, 0),
            "success_rate": stats.get("success_rate", 0.0),
        }

        return StandardResponse(success=True, message="任务统计获取成功", data=task_stats)

    except Exception as e:
        logger.error(f"获取任务统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务统计失败: {str(e)}")
    finally:
        session.close()


@router.get("/monitor/stuck", response_model=StandardResponse)
async def get_stuck_tasks(timeout_minutes: int = 30):
    """获取卡住的任务"""
    try:
        stuck_tasks = task_monitor.get_stuck_tasks(timeout_minutes)

        return StandardResponse(
            success=True,
            message=f"发现 {len(stuck_tasks)} 个卡住的任务",
            data={"stuck_tasks": stuck_tasks, "timeout_minutes": timeout_minutes},
        )

    except Exception as e:
        logger.error(f"获取卡住任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取卡住任务失败: {str(e)}")


@router.post("/monitor/cleanup", response_model=StandardResponse)
async def cleanup_stuck_tasks(timeout_minutes: int = 30, auto_fix: bool = False):
    """清理卡住的任务"""
    try:
        result = task_monitor.cleanup_stuck_tasks(timeout_minutes, auto_fix)

        message = f"处理完成：发现 {result['total_stuck']} 个卡住任务"
        if auto_fix:
            message += (
                f"，修复 {len(result['fixed_tasks'])} 个，失败 {len(result['failed_tasks'])} 个"
            )

        return StandardResponse(success=True, message=message, data=result)

    except Exception as e:
        logger.error(f"清理卡住任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清理卡住任务失败: {str(e)}")


@router.post("/monitor/force-complete/{task_id}", response_model=StandardResponse)
async def force_complete_task(task_id: str, status: str = "cancelled"):
    """强制完成指定任务"""
    try:
        if status not in ["cancelled", "failed", "completed"]:
            raise HTTPException(
                status_code=400, detail="状态必须是 cancelled、failed 或 completed"
            )

        success = task_monitor.force_complete_task(task_id, status)

        if success:
            return StandardResponse(
                success=True,
                message=f"任务已强制设置为 {status}",
                data={"task_id": task_id, "status": status},
            )
        else:
            raise HTTPException(status_code=404, detail=f"任务不存在或处理失败: {task_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"强制完成任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"强制完成任务失败: {str(e)}")


@router.get("/monitor/statistics", response_model=StandardResponse)
async def get_task_monitor_statistics():
    """获取任务监控统计信息"""
    try:
        stats = task_monitor.get_task_statistics()

        return StandardResponse(success=True, message="获取统计信息成功", data=stats)

    except Exception as e:
        logger.error(f"获取监控统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取监控统计失败: {str(e)}")
