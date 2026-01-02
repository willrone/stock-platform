"""
简化的数据管理路由
只提供连接状态检查和数据获取功能
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

from loguru import logger
from app.api.v1.schemas import StandardResponse, RemoteDataSyncRequest
from app.core.container import get_data_service, get_sftp_sync_service
from app.core.config import settings
from app.services.data import SimpleDataService
from app.services.data.sftp_sync_service import SFTPSyncService
from app.services.data.parquet_manager import ParquetManager
from app.services.events.data_sync_events import get_data_sync_event_manager, DataSyncEventType

router = APIRouter(prefix="/data", tags=["数据管理"])


@router.get("/status", response_model=StandardResponse, summary="获取数据服务状态", description="获取远端数据服务连接状态和响应时间")
async def get_data_service_status(data_service: SimpleDataService = Depends(get_data_service)):
    """获取数据服务状态"""
    try:
        status = await data_service.check_remote_service_status()
        
        response_data = {
            "service_url": status.service_url,
            "is_connected": status.is_available,
            "last_check": status.last_check.isoformat() if status.last_check else None,
            "response_time": status.response_time_ms,
            "error_message": status.error_message
        }
        
        return StandardResponse(
            success=status.is_available,
            message="数据服务状态检查完成" if status.is_available else f"数据服务不可用: {status.error_message}",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"获取数据服务状态失败: {e}")
        return StandardResponse(
            success=False,
            message=f"获取数据服务状态失败: {str(e)}",
            data={
                "service_url": "unknown",
                "is_connected": False,
                "last_check": datetime.now().isoformat(),
                "response_time": None,
                "error_message": str(e)
            }
        )


@router.get("/remote/stocks", response_model=StandardResponse, summary="获取远端服务股票列表", description="从远端数据服务获取可用的股票列表")
async def get_remote_stock_list(data_service: SimpleDataService = Depends(get_data_service)):
    """获取远端服务的股票列表"""
    try:
        stocks = await data_service.get_remote_stock_list()
        
        if stocks is None:
            return StandardResponse(
                success=False,
                message="无法从远端服务获取股票列表",
                data={
                    "stocks": [],
                    "total_stocks": 0
                }
            )
        
        stock_codes = [stock.get("ts_code", "") for stock in stocks if stock.get("ts_code")]
        
        return StandardResponse(
            success=True,
            message=f"成功获取远端股票列表: {len(stocks)} 只股票",
            data={
                "stocks": stocks,
                "stock_codes": stock_codes,
                "total_stocks": len(stocks)
            }
        )
    
    except Exception as e:
        logger.error(f"获取远端股票列表失败: {e}")
        return StandardResponse(
            success=False,
            message=f"获取远端股票列表失败: {str(e)}",
            data={
                "stocks": [],
                "stock_codes": [],
                "total_stocks": 0
            }
        )


@router.get("/local/stocks", response_model=StandardResponse, summary="获取本地股票列表", description="从本地parquet文件获取可用的股票列表")
async def get_local_stock_list():
    """获取本地股票列表"""
    try:
        import pandas as pd
        from pathlib import Path
        
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
            logger.warning(f"未找到parquet目录，尝试的路径: {possible_paths}, DATA_ROOT_PATH={settings.DATA_ROOT_PATH}")
            return StandardResponse(
                success=False,
                message=f"未找到parquet数据目录，尝试的路径: {[str(p) for p in possible_paths]}",
                data={
                    "stocks": [],
                    "stock_codes": [],
                    "total_stocks": 0
                }
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
                df = pd.read_parquet(file_path)
                if df.empty:
                    logger.debug(f"文件为空: {file_path}")
                    continue
                
                # 检查是否有stock_code或ts_code列（优先使用ts_code，因为这是实际数据格式）
                stock_code_col = None
                if 'ts_code' in df.columns:
                    stock_code_col = 'ts_code'
                elif 'stock_code' in df.columns:
                    stock_code_col = 'stock_code'
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
                            'file_count': 0,
                            'total_size': 0,
                            'record_count': 0,
                            'dates': []
                        }
                    
                    file_stat = file_path.stat()
                    stock_data_map[stock_code]['file_count'] += 1
                    stock_data_map[stock_code]['total_size'] += file_stat.st_size
                    stock_data_map[stock_code]['record_count'] += len(stock_df)
                    
                    # 收集日期
                    if 'date' in stock_df.columns:
                        dates = pd.to_datetime(stock_df['date']).tolist()
                        stock_data_map[stock_code]['dates'].extend(dates)
                
            except Exception as e:
                error_count += 1
                logger.warning(f"读取parquet文件失败 {file_path}: {e}", exc_info=True)
                continue
        
        logger.info(f"处理完成: 文件数={file_count}, 错误数={error_count}, 股票数={len(stock_data_map)}")
        
        # 构建股票列表
        stocks = []
        for stock_code, stock_stats in stock_data_map.items():
            dates = stock_stats.get('dates', [])
            stock_info = {
                "ts_code": stock_code,
                "name": stock_code,  # 本地数据可能没有名称，使用代码作为名称
                "file_count": stock_stats.get('file_count', 0),
                "total_size": stock_stats.get('total_size', 0),
                "record_count": stock_stats.get('record_count', 0),
            }
            
            if dates:
                start_date = min(dates)
                end_date = max(dates)
                
                # 处理日期格式
                if hasattr(start_date, 'strftime'):
                    start_date_str = start_date.strftime('%Y-%m-%d')
                elif hasattr(start_date, 'to_pydatetime'):
                    start_date_str = start_date.to_pydatetime().strftime('%Y-%m-%d')
                else:
                    start_date_str = str(start_date).split()[0] if ' ' in str(start_date) else str(start_date)
                
                if hasattr(end_date, 'strftime'):
                    end_date_str = end_date.strftime('%Y-%m-%d')
                elif hasattr(end_date, 'to_pydatetime'):
                    end_date_str = end_date.to_pydatetime().strftime('%Y-%m-%d')
                else:
                    end_date_str = str(end_date).split()[0] if ' ' in str(end_date) else str(end_date)
                
                # 计算总天数
                try:
                    if hasattr(start_date, 'to_pydatetime'):
                        start_dt = start_date.to_pydatetime()
                    elif isinstance(start_date, datetime):
                        start_dt = start_date
                    else:
                        start_dt = pd.to_datetime(start_date).to_pydatetime()
                    
                    if hasattr(end_date, 'to_pydatetime'):
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
                    "total_days": total_days
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
                "total_stocks": len(stocks)
            }
        )
    
    except Exception as e:
        logger.error(f"获取本地股票列表失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"获取本地股票列表失败: {str(e)}",
            data={
                "stocks": [],
                "stock_codes": [],
                "total_stocks": 0
            }
        )


@router.post("/sync/remote", response_model=StandardResponse, summary="同步远端数据", description="通过SFTP从远端服务器同步股票parquet数据")
async def sync_remote_data(
    request: RemoteDataSyncRequest,
    sftp_sync_service: SFTPSyncService = Depends(get_sftp_sync_service)
):
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
                executor,
                sftp_sync_service.sync_selected_stocks,
                request.stock_codes
            )
        else:
            logger.info("同步所有股票数据")
            result = await loop.run_in_executor(
                executor,
                sftp_sync_service.sync_all_stocks,
                None
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"同步任务执行完成，总耗时: {duration:.1f}秒")
        
        response_data = {
            "success": result.success,
            "total_files": result.total_files,
            "synced_files": result.synced_files,
            "failed_files": result.failed_files,
            "total_size": result.total_size,
            "total_size_mb": round(result.total_size / (1024 * 1024), 2) if result.total_size > 0 else 0,
            "duration_seconds": round(duration, 2)
        }
        
        logger.info(f"返回响应: {response_data}")
        
        return StandardResponse(
            success=result.success,
            message=result.message,
            data=response_data
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
                "duration_seconds": round(duration, 2)
            }
        )


@router.get("/events/history", response_model=StandardResponse, summary="获取数据同步事件历史", description="获取数据同步事件的历史记录")
async def get_sync_event_history(
    stock_code: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 50
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
                    detail=f"无效的事件类型: {event_type}。有效类型: {[e.value for e in DataSyncEventType]}"
                )
        
        # 获取事件历史
        events = event_manager.get_event_history(
            stock_code=stock_code,
            event_type=event_type_enum,
            limit=min(limit, 200)  # 限制最大返回数量
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
                    "limit": limit
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取事件历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取事件历史失败: {str(e)}")


@router.get("/events/stats", response_model=StandardResponse, summary="获取数据同步事件统计", description="获取数据同步事件的统计信息")
async def get_sync_event_stats():
    """获取数据同步事件统计"""
    try:
        event_manager = get_data_sync_event_manager()
        stats = event_manager.get_stats()
        
        return StandardResponse(
            success=True,
            message="成功获取事件统计信息",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取事件统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取事件统计失败: {str(e)}")


@router.delete("/events/history", response_model=StandardResponse, summary="清空事件历史", description="清空所有数据同步事件历史记录")
async def clear_sync_event_history():
    """清空事件历史"""
    try:
        event_manager = get_data_sync_event_manager()
        event_manager.clear_history()
        
        return StandardResponse(
            success=True,
            message="事件历史已清空",
            data={}
        )
        
    except Exception as e:
        logger.error(f"清空事件历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空事件历史失败: {str(e)}")