"""
简化的数据管理路由
只提供连接状态检查和数据获取功能
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.api.v1.schemas import (
    QlibPrecomputeRequest,
    RemoteDataSyncRequest,
    StandardResponse,
)
from app.core.config import settings
from app.core.container import get_data_service, get_sftp_sync_service
from app.services.data import SimpleDataService
from app.services.data.parquet_manager import ParquetManager
from app.services.data.sftp_sync_service import SFTPSyncService
from app.services.events.data_sync_events import (
    DataSyncEventType,
    get_data_sync_event_manager,
)

router = APIRouter(prefix="/data", tags=["数据管理"])


# Qlib预计算相关接口
@router.post(
    "/qlib/precompute",
    response_model=StandardResponse,
    summary="触发Qlib指标/因子预计算",
    description="为全市场所有股票（或指定股票）预计算所有指标和因子，存储为Qlib格式",
)
async def trigger_qlib_precompute(request: QlibPrecomputeRequest):
    """
    触发Qlib指标/因子预计算任务

    Args:
        request: 预计算请求参数

    Returns:
        任务创建结果，包含task_id
    """
    from app.api.v1.dependencies import execute_qlib_precompute_task_simple
    from app.core.database import SessionLocal
    from app.models.task_models import TaskStatus, TaskType
    from app.repositories.task_repository import TaskRepository
    from app.services.tasks.process_executor import get_process_executor

    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)

        # 构建任务配置
        config = {
            "batch_size": request.batch_size,
        }

        if request.stock_codes:
            config["stock_codes"] = request.stock_codes
        if request.start_date:
            config["start_date"] = request.start_date
        if request.end_date:
            config["end_date"] = request.end_date
        if request.max_workers:
            config["max_workers"] = request.max_workers

        # 创建任务
        task = task_repository.create_task(
            task_name=f"Qlib预计算任务",
            task_type=TaskType.QLIB_PRECOMPUTE,
            user_id="default_user",  # TODO: 从认证中获取真实用户ID
            config=config,
        )

        # 将任务提交到进程池执行（异步，不阻塞）
        try:
            process_executor = get_process_executor()

            # 提交任务到进程池
            future = process_executor.submit(
                execute_qlib_precompute_task_simple, task.task_id
            )

            logger.info(f"Qlib预计算任务已提交到进程池: {task.task_id}")
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
            "config": config,
            "created_at": task.created_at.isoformat()
            if task.created_at
            else datetime.now().isoformat(),
            "completed_at": task.completed_at.isoformat()
            if task.completed_at
            else None,
            "error_message": task.error_message,
        }

        return StandardResponse(success=True, message="Qlib预计算任务创建成功", data=task_data)

    except Exception as e:
        session.rollback()
        logger.error(f"创建Qlib预计算任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建Qlib预计算任务失败: {str(e)}")
    finally:
        session.close()


@router.get(
    "/status",
    response_model=StandardResponse,
    summary="获取数据服务状态",
    description="获取远端数据服务连接状态和响应时间",
)
async def get_data_service_status(
    data_service: SimpleDataService = Depends(get_data_service),
):
    """获取数据服务状态"""
    logger.info("收到数据服务状态检查请求")
    try:
        status = await data_service.check_remote_service_status()

        response_data = {
            "service_url": status.service_url,
            "is_connected": status.is_available,
            "last_check": status.last_check.isoformat() if status.last_check else None,
            "response_time": status.response_time_ms,
            "error_message": status.error_message,
        }

        logger.info(
            f"数据服务状态检查完成: 连接状态={status.is_available}, URL={status.service_url}, 响应时间={status.response_time_ms}ms"
        )

        return StandardResponse(
            success=status.is_available,
            message="数据服务状态检查完成"
            if status.is_available
            else f"数据服务不可用: {status.error_message}",
            data=response_data,
        )

    except Exception as e:
        logger.error(f"获取数据服务状态失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"获取数据服务状态失败: {str(e)}",
            data={
                "service_url": "unknown",
                "is_connected": False,
                "last_check": datetime.now().isoformat(),
                "response_time": None,
                "error_message": str(e),
            },
        )


@router.get(
    "/remote/stocks",
    response_model=StandardResponse,
    summary="获取远端服务股票列表",
    description="从远端数据服务获取可用的股票列表",
)
async def get_remote_stock_list(
    data_service: SimpleDataService = Depends(get_data_service),
):
    """获取远端服务的股票列表"""
    logger.info("收到获取远端股票列表请求")
    try:
        logger.info("开始从远端服务获取股票列表...")
        stocks = await data_service.get_remote_stock_list()

        if stocks is None:
            logger.warning("无法从远端服务获取股票列表，返回空列表")
            return StandardResponse(
                success=False,
                message="无法从远端服务获取股票列表",
                data={"stocks": [], "total_stocks": 0},
            )

        logger.info(f"成功从远端服务获取股票列表: {len(stocks)} 只股票")
        stock_codes = [
            stock.get("ts_code", "") for stock in stocks if stock.get("ts_code")
        ]

        # 计算响应大小（估算）
        import json

        response_data = {
            "stocks": stocks,
            "stock_codes": stock_codes,
            "total_stocks": len(stocks),
        }
        estimated_size = len(json.dumps(response_data))
        logger.info(
            f"准备返回股票列表，股票数量: {len(stocks)}, 股票代码数量: {len(stock_codes)}, 估算响应大小: {estimated_size/1024:.2f} KB"
        )

        if estimated_size > 5 * 1024 * 1024:  # 5MB
            logger.warning(f"响应数据较大 ({estimated_size/1024/1024:.2f} MB)，可能导致前端处理失败")

        return StandardResponse(
            success=True, message=f"成功获取远端股票列表: {len(stocks)} 只股票", data=response_data
        )

    except Exception as e:
        logger.error(f"获取远端股票列表失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"获取远端股票列表失败: {str(e)}",
            data={"stocks": [], "stock_codes": [], "total_stocks": 0},
        )


@router.get(
    "/local/stocks",
    response_model=StandardResponse,
    summary="获取本地股票列表",
    description="从本地parquet文件获取可用的股票列表",
)
async def get_local_stock_list():
    """获取本地股票列表"""
    try:
        from pathlib import Path

        import pandas as pd

        # 确定搜索路径 - 尝试多个可能的路径
        # 将相对路径转换为绝对路径
        data_root = Path(settings.DATA_ROOT_PATH)
        if not data_root.is_absolute():
            # 如果是相对路径，从backend目录开始解析
            backend_dir = Path(__file__).parent.parent.parent.parent
            data_root = (backend_dir / data_root).resolve()

        possible_paths = [
            data_root / "parquet",
            data_root / "stocks" / "daily",  # 另一个可能的位置
            Path("data") / "parquet",
            Path("./data") / "parquet",
        ]

        search_path = None
        for path in possible_paths:
            path_resolved = path.resolve() if path.exists() else None
            if path_resolved and path_resolved.exists():
                search_path = path_resolved
                logger.info(f"找到parquet目录: {search_path} (原始路径: {path})")
                break

        if search_path is None:
            logger.warning(
                f"未找到parquet目录，尝试的路径: {possible_paths}, DATA_ROOT_PATH={settings.DATA_ROOT_PATH}"
            )
            return StandardResponse(
                success=False,
                message=f"未找到parquet数据目录，尝试的路径: {[str(p) for p in possible_paths]}",
                data={"stocks": [], "stock_codes": [], "total_stocks": 0},
            )

        # 检查是否有daily子目录或stock_data子目录
        daily_dir = search_path / "daily"
        stock_data_dir = search_path / "stock_data"

        if daily_dir.exists() and daily_dir.is_dir():
            search_path = daily_dir
            logger.info(f"使用daily子目录: {search_path}")
        elif stock_data_dir.exists() and stock_data_dir.is_dir():
            search_path = stock_data_dir
            logger.info(f"使用stock_data子目录: {search_path}")

        # 从parquet文件内部读取股票代码，而不是依赖目录名
        stock_data_map: Dict[str, Dict] = {}
        file_count = 0
        error_count = 0

        # 递归查找所有parquet文件
        parquet_files = list(search_path.rglob("*.parquet"))
        logger.info(f"找到 {len(parquet_files)} 个parquet文件在 {search_path}")

        for file_path in parquet_files:
            try:
                file_count += 1
                # 尝试使用 pyarrow 引擎，如果失败则使用 fastparquet
                try:
                    df = pd.read_parquet(file_path, engine="pyarrow")
                except Exception as e:
                    logger.debug(f"使用 pyarrow 引擎读取失败: {e}，尝试使用 fastparquet")
                    try:
                        df = pd.read_parquet(file_path, engine="fastparquet")
                    except Exception as e2:
                        logger.error(f"使用 fastparquet 引擎也失败: {e2}")
                        raise
                if df.empty:
                    logger.debug(f"文件为空: {file_path}")
                    continue

                # 检查是否有stock_code或ts_code列（优先使用ts_code，因为这是实际数据格式）
                stock_code_col = None
                if "ts_code" in df.columns:
                    stock_code_col = "ts_code"
                elif "stock_code" in df.columns:
                    stock_code_col = "stock_code"
                else:
                    logger.warning(f"文件缺少股票代码列: {file_path}, 列名: {df.columns.tolist()}")
                    continue

                # 按股票代码分组统计
                unique_stocks = df[stock_code_col].unique()
                logger.debug(f"文件 {file_path} 包含 {len(unique_stocks)} 个股票代码")

                for stock_code in unique_stocks:
                    if pd.isna(stock_code):
                        continue

                    stock_code = str(stock_code).strip()
                    if not stock_code:
                        continue

                    stock_df = df[df[stock_code_col] == stock_code]

                    if stock_code not in stock_data_map:
                        stock_data_map[stock_code] = {
                            "file_count": 0,
                            "total_size": 0,
                            "record_count": 0,
                            "dates": [],
                        }

                    file_stat = file_path.stat()
                    stock_data_map[stock_code]["file_count"] += 1
                    stock_data_map[stock_code]["total_size"] += file_stat.st_size
                    stock_data_map[stock_code]["record_count"] += len(stock_df)

                    # 收集日期 - 支持多种日期列名
                    date_col = None
                    for col_name in ["date", "trade_date", "datetime", "time", "Date", "TradeDate"]:
                        if col_name in stock_df.columns:
                            date_col = col_name
                            break
                    
                    # 如果列中没有日期，尝试从索引获取
                    if date_col is None and isinstance(stock_df.index, pd.DatetimeIndex):
                        dates = stock_df.index.tolist()
                        stock_data_map[stock_code]["dates"].extend(dates)
                    elif date_col:
                        try:
                            dates = pd.to_datetime(stock_df[date_col]).tolist()
                            stock_data_map[stock_code]["dates"].extend(dates)
                        except Exception as e:
                            logger.debug(f"解析日期列 {date_col} 失败 {stock_code}: {e}")

            except Exception as e:
                error_count += 1
                logger.warning(f"读取parquet文件失败 {file_path}: {e}", exc_info=True)
                continue

        logger.info(
            f"处理完成: 文件数={file_count}, 错误数={error_count}, 股票数={len(stock_data_map)}"
        )

        # 构建股票列表
        stocks = []
        for stock_code, stock_stats in stock_data_map.items():
            dates = stock_stats.get("dates", [])
            stock_info = {
                "ts_code": stock_code,
                "name": stock_code,  # 本地数据可能没有名称，使用代码作为名称
                "file_count": stock_stats.get("file_count", 0),
                "total_size": stock_stats.get("total_size", 0),
                "record_count": stock_stats.get("record_count", 0),
            }

            if dates:
                start_date = min(dates)
                end_date = max(dates)

                # 处理日期格式
                if hasattr(start_date, "strftime"):
                    start_date_str = start_date.strftime("%Y-%m-%d")
                elif hasattr(start_date, "to_pydatetime"):
                    start_date_str = start_date.to_pydatetime().strftime("%Y-%m-%d")
                else:
                    start_date_str = (
                        str(start_date).split()[0]
                        if " " in str(start_date)
                        else str(start_date)
                    )

                if hasattr(end_date, "strftime"):
                    end_date_str = end_date.strftime("%Y-%m-%d")
                elif hasattr(end_date, "to_pydatetime"):
                    end_date_str = end_date.to_pydatetime().strftime("%Y-%m-%d")
                else:
                    end_date_str = (
                        str(end_date).split()[0]
                        if " " in str(end_date)
                        else str(end_date)
                    )

                # 计算总天数
                try:
                    if hasattr(start_date, "to_pydatetime"):
                        start_dt = start_date.to_pydatetime()
                    elif isinstance(start_date, datetime):
                        start_dt = start_date
                    else:
                        start_dt = pd.to_datetime(start_date).to_pydatetime()

                    if hasattr(end_date, "to_pydatetime"):
                        end_dt = end_date.to_pydatetime()
                    elif isinstance(end_date, datetime):
                        end_dt = end_date
                    else:
                        end_dt = pd.to_datetime(end_date).to_pydatetime()

                    total_days = (end_dt - start_dt).days + 1
                except Exception as e:
                    logger.warning(f"计算总天数失败 {stock_code}: {e}")
                    total_days = 0

                stock_info["data_range"] = {
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "total_days": total_days,
                }

            stocks.append(stock_info)

        # 按股票代码排序
        stocks.sort(key=lambda x: x["ts_code"])

        stock_codes = [stock["ts_code"] for stock in stocks]

        return StandardResponse(
            success=True,
            message=f"成功获取本地股票列表: {len(stocks)} 只股票",
            data={
                "stocks": stocks,
                "stock_codes": stock_codes,
                "total_stocks": len(stocks),
            },
        )

    except Exception as e:
        logger.error(f"获取本地股票列表失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"获取本地股票列表失败: {str(e)}",
            data={"stocks": [], "stock_codes": [], "total_stocks": 0},
        )


@router.get(
    "/local/stocks/simple",
    response_model=StandardResponse,
    summary="获取本地股票列表（快速版）",
    description="快速获取本地股票代码列表，仅用于选择股票，不包含详细信息",
)
async def get_local_stock_list_simple():
    """快速获取本地股票列表（仅股票代码和名称）"""
    try:
        from pathlib import Path

        # 固定路径：backend/data/parquet/stock_data/
        backend_dir = Path(__file__).parent.parent.parent.parent
        stock_data_path = backend_dir / "data" / "parquet" / "stock_data"

        # 如果固定路径不存在，尝试其他可能的路径
        if not stock_data_path.exists():
            data_root = Path(settings.DATA_ROOT_PATH)
            if not data_root.is_absolute():
                data_root = (backend_dir / data_root).resolve()

            possible_paths = [
                data_root / "parquet" / "stock_data",
                data_root / "parquet",
                backend_dir / "data" / "parquet" / "stock_data",
                Path("data") / "parquet" / "stock_data",
            ]

            stock_data_path = None
            for path in possible_paths:
                path_resolved = path.resolve() if path.exists() else None
                if path_resolved and path_resolved.exists():
                    stock_data_path = path_resolved
                    break

            if stock_data_path is None:
                logger.warning(f"未找到stock_data目录，尝试的路径: {possible_paths}")
                return StandardResponse(
                    success=False,
                    message="未找到parquet数据目录",
                    data={"stocks": [], "stock_codes": [], "total_stocks": 0},
                )

        logger.info(f"使用股票数据目录: {stock_data_path}")

        # 快速方法：从文件名获取股票代码，不读取文件内容
        stock_codes_set = set()

        # 查找所有parquet文件
        parquet_files = list(stock_data_path.glob("*.parquet"))
        logger.info(f"找到 {len(parquet_files)} 个parquet文件")

        for file_path in parquet_files:
            file_name = file_path.stem  # 不含扩展名的文件名，例如: 000001_SZ

            # 文件名格式: {code}_{market}.parquet，例如: 000001_SZ.parquet, 920000_BJ.parquet
            # 需要转换为标准格式: {code}.{market}，例如: 000001.SZ, 920000.BJ
            if "_" in file_name:
                parts = file_name.split("_")
                if len(parts) >= 2:
                    code = parts[0]
                    market = parts[1].upper()  # SZ, SH, 或 BJ

                    # 验证市场代码（支持所有市场：SZ深圳、SH上海、BJ北京）
                    if market in ["SZ", "SH", "BJ"]:
                        # 转换为标准格式
                        stock_code = f"{code}.{market}"
                        stock_codes_set.add(stock_code)
                    else:
                        logger.debug(f"跳过未知的市场代码: {file_name} (市场: {market})")
                else:
                    logger.debug(f"文件名格式不正确: {file_name}")
            else:
                # 如果文件名已经是标准格式（带点），直接使用
                if "." in file_name and (
                    file_name.endswith(".SZ")
                    or file_name.endswith(".SH")
                    or file_name.endswith(".BJ")
                ):
                    stock_codes_set.add(file_name)

        # 转换为列表并排序
        stock_codes = sorted(list(stock_codes_set))

        logger.info(f"提取到 {len(stock_codes)} 个股票代码")

        # 构建简化的股票列表（只有代码和名称）
        stocks = [
            {"ts_code": code, "name": code}  # 本地数据可能没有名称，使用代码作为名称
            for code in stock_codes
        ]

        logger.info(f"快速获取本地股票列表: {len(stocks)} 只股票")

        return StandardResponse(
            success=True,
            message=f"成功获取本地股票列表: {len(stocks)} 只股票",
            data={
                "stocks": stocks,
                "stock_codes": stock_codes,
                "total_stocks": len(stocks),
            },
        )

    except Exception as e:
        logger.error(f"快速获取本地股票列表失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"获取本地股票列表失败: {str(e)}",
            data={"stocks": [], "stock_codes": [], "total_stocks": 0},
        )


@router.post(
    "/sync/remote",
    response_model=StandardResponse,
    summary="同步远端数据",
    description="通过SFTP从远端服务器同步股票parquet数据",
)
async def sync_remote_data(
    request: RemoteDataSyncRequest,
    sftp_sync_service: SFTPSyncService = Depends(get_sftp_sync_service),
):
    # Friendly message when SFTP sync is disabled (default)
    if not getattr(sftp_sync_service, "enabled", False):
        return StandardResponse(
            success=False,
            message=(
                "SFTP同步未启用（SFTP_SYNC_ENABLED=false）。"
                "如需在分布式部署中使用远端同步，请在backend/.env中开启并配置SFTP参数。"
            ),
            data={
                "success": False,
                "total_files": 0,
                "synced_files": 0,
                "failed_files": [],
                "total_size": 0,
                "total_size_mb": 0,
                "duration_seconds": 0,
            },
        )

    """
    同步远端数据

    通过SFTP从远端服务器下载股票parquet数据到本地
    """
    start_time = datetime.now()

    # 立即输出日志，确保能看到请求到达
    print("=" * 60)
    print("收到同步远端数据请求 - API端点被调用")
    print(f"请求参数: stock_codes={'已提供' if request.stock_codes else '未提供（将同步所有股票）'}")
    if request.stock_codes:
        print(f"要同步的股票数量: {len(request.stock_codes)}")
    print("=" * 60)

    logger.info("=" * 60)
    logger.info("收到同步远端数据请求 - API端点被调用")
    logger.info(f"请求参数: stock_codes={'已提供' if request.stock_codes else '未提供（将同步所有股票）'}")
    if request.stock_codes:
        logger.info(f"要同步的股票数量: {len(request.stock_codes)}")
    logger.info("=" * 60)

    try:
        # 使用线程池执行同步操作（因为SFTP是同步的）
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        # 执行同步
        logger.info("开始执行同步任务...")
        if request.stock_codes:
            logger.info(f"同步选定的 {len(request.stock_codes)} 只股票")
            result = await loop.run_in_executor(
                executor, sftp_sync_service.sync_selected_stocks, request.stock_codes
            )
        else:
            logger.info("同步所有股票数据")
            result = await loop.run_in_executor(
                executor, sftp_sync_service.sync_all_stocks, None
            )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"同步任务执行完成，总耗时: {duration:.1f}秒")

        response_data = {
            "success": result.success,
            "total_files": result.total_files,
            "synced_files": result.synced_files,
            "failed_files": result.failed_files,
            "total_size": result.total_size,
            "total_size_mb": round(result.total_size / (1024 * 1024), 2)
            if result.total_size > 0
            else 0,
            "duration_seconds": round(duration, 2),
        }

        logger.info(f"返回响应: {response_data}")

        return StandardResponse(
            success=result.success, message=result.message, data=response_data
        )

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"同步远端数据失败: {e}", exc_info=True)
        logger.error(f"失败时已耗时: {duration:.1f}秒")
        return StandardResponse(
            success=False,
            message=f"同步远端数据失败: {str(e)}",
            data={
                "success": False,
                "total_files": 0,
                "synced_files": 0,
                "failed_files": [],
                "total_size": 0,
                "total_size_mb": 0,
                "duration_seconds": round(duration, 2),
            },
        )


@router.get(
    "/events/history",
    response_model=StandardResponse,
    summary="获取数据同步事件历史",
    description="获取数据同步事件的历史记录",
)
async def get_sync_event_history(
    stock_code: Optional[str] = None, event_type: Optional[str] = None, limit: int = 50
):
    """获取数据同步事件历史"""
    try:
        event_manager = get_data_sync_event_manager()

        # 转换事件类型
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = DataSyncEventType(event_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的事件类型: {event_type}。有效类型: {[e.value for e in DataSyncEventType]}",
                )

        # 获取事件历史
        events = event_manager.get_event_history(
            stock_code=stock_code,
            event_type=event_type_enum,
            limit=min(limit, 200),  # 限制最大返回数量
        )

        # 转换为字典格式
        events_data = [event.to_dict() for event in events]

        return StandardResponse(
            success=True,
            message=f"成功获取事件历史: {len(events_data)} 条记录",
            data={
                "events": events_data,
                "total_events": len(events_data),
                "filters": {
                    "stock_code": stock_code,
                    "event_type": event_type,
                    "limit": limit,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取事件历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取事件历史失败: {str(e)}")


@router.get(
    "/events/stats",
    response_model=StandardResponse,
    summary="获取数据同步事件统计",
    description="获取数据同步事件的统计信息",
)
async def get_sync_event_stats():
    """获取数据同步事件统计"""
    try:
        event_manager = get_data_sync_event_manager()
        stats = event_manager.get_stats()

        return StandardResponse(success=True, message="成功获取事件统计信息", data=stats)

    except Exception as e:
        logger.error(f"获取事件统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取事件统计失败: {str(e)}")


@router.delete(
    "/events/history",
    response_model=StandardResponse,
    summary="清空事件历史",
    description="清空所有数据同步事件历史记录",
)
async def clear_sync_event_history():
    """清空事件历史"""
    try:
        event_manager = get_data_sync_event_manager()
        event_manager.clear_history()

        return StandardResponse(success=True, message="事件历史已清空", data={})

    except Exception as e:
        logger.error(f"清空事件历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空事件历史失败: {str(e)}")
