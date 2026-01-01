"""
简化的数据管理路由
只提供连接状态检查和数据获取功能
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from datetime import datetime
import logging

from app.api.v1.schemas import StandardResponse
from app.core.container import get_data_service
from app.services.data import SimpleDataService

router = APIRouter(prefix="/data", tags=["数据管理"])
logger = logging.getLogger(__name__)


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
