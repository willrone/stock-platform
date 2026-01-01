"""
股票数据路由
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from datetime import datetime, timedelta
import logging

from app.api.v1.schemas import StandardResponse
from app.core.container import get_data_service, get_indicators_service
from app.services.data import DataService as StockDataService
from app.services.prediction import TechnicalIndicatorCalculator
from app.core.config import settings
from app.services.data import ParquetManager

router = APIRouter(prefix="/stocks", tags=["股票数据"])
logger = logging.getLogger(__name__)


@router.get(
    "/data", 
    response_model=StandardResponse,
    summary="获取股票数据",
    description="根据股票代码和时间范围获取历史价格数据"
)
async def get_stock_data(
    stock_code: str,
    start_date: datetime,
    end_date: datetime,
    data_service: StockDataService = Depends(get_data_service)
):
    """
    获取股票历史数据
    
    根据指定的股票代码和时间范围，从数据服务获取股票的历史价格数据。
    数据包括开盘价、最高价、最低价、收盘价和成交量。
    """
    try:
        stock_data = await data_service.get_stock_data(stock_code, start_date, end_date)
        
        if not stock_data:
            return StandardResponse(
                success=False,
                message=f"未找到股票 {stock_code} 在指定时间范围内的数据",
                data=None
            )
        
        # 转换数据格式
        data_points = []
        for item in stock_data:
            data_points.append({
                "date": item.date.isoformat(),
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume,
                "adj_close": item.adj_close
            })
        
        response_data = {
            "stock_code": stock_code,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data_points": len(data_points),
            "data": data_points
        }
        
        return StandardResponse(
            success=True,
            message="股票数据获取成功",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票数据失败: {str(e)}")


@router.get("/{stock_code}/indicators", response_model=StandardResponse)
async def get_technical_indicators(
    stock_code: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    indicators: Optional[str] = Query(default="MA5,MA10,MA20,RSI,MACD", description="指标列表，逗号分隔"),
    data_service: StockDataService = Depends(get_data_service),
    indicators_service: TechnicalIndicatorCalculator = Depends(get_indicators_service)
):
    """获取技术指标"""
    try:
        if not start_date:
            start_date = datetime.now() - timedelta(days=60)
        if not end_date:
            end_date = datetime.now()
        
        # 解析指标列表
        indicator_list = [ind.strip() for ind in indicators.split(',')]
        
        # 获取股票数据
        stock_data = await data_service.get_stock_data(stock_code, start_date, end_date)
        
        if not stock_data:
            return StandardResponse(
                success=False,
                message=f"未找到股票 {stock_code} 的数据",
                data=None
            )
        
        # 计算技术指标
        indicator_results = indicators_service.calculate_indicators(stock_data, indicator_list)
        
        # 格式化结果
        formatted_results = []
        for result in indicator_results:
            formatted_results.append(result.to_dict())
        
        # 获取最新的指标值
        latest_indicators = {}
        if indicator_results:
            latest_result = indicator_results[-1]
            latest_indicators = latest_result.indicators
        
        response_data = {
            "stock_code": stock_code,
            "calculation_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "indicators": latest_indicators,
            "calculation_date": end_date.isoformat(),
            "total_data_points": len(indicator_results),
            "detailed_results": formatted_results
        }
        
        return StandardResponse(
            success=True,
            message="技术指标计算成功",
            data=response_data
        )
        
    except ValueError as e:
        logger.error(f"技术指标计算参数错误: {e}")
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算技术指标失败: {str(e)}")


@router.get("/popular", response_model=StandardResponse)
async def get_popular_stocks(
    limit: int = 20,
    data_service: StockDataService = Depends(get_data_service)
):
    """获取热门股票列表"""
    try:
        # 从远端数据服务获取股票列表
        stocks = await data_service.get_remote_stock_list()
        
        if not stocks or len(stocks) == 0:
            # 如果远端服务不可用，尝试从本地数据文件获取
            try:
                parquet_manager = ParquetManager(settings.PARQUET_DATA_PATH)
                stats = parquet_manager.get_comprehensive_stats()
                stocks = []
                if hasattr(stats, 'stocks_by_size') and stats.stocks_by_size:
                    for stock_code, _ in stats.stocks_by_size[:limit]:
                        stocks.append({
                            "ts_code": stock_code,
                            "name": stock_code
                        })
            except Exception as e:
                logger.warning(f"无法从本地数据获取股票列表: {e}")
                stocks = []
        
        # 转换为前端期望的格式
        popular_stocks = []
        for stock in stocks[:limit]:
            stock_code = stock.get("ts_code", "")
            stock_name = stock.get("name", "")
            market = "深圳" if ".SZ" in stock_code else "上海" if ".SH" in stock_code else "未知"
            
            popular_stocks.append({
                "code": stock_code,
                "name": stock_name,
                "market": market,
                "change_percent": 0.0,
                "volume": 0
            })
        
        return StandardResponse(
            success=True,
            message="热门股票列表获取成功",
            data={"stocks": popular_stocks, "total": len(popular_stocks)}
        )
        
    except Exception as e:
        logger.error(f"获取热门股票列表失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"获取热门股票列表失败: {str(e)}",
            data={"stocks": [], "total": 0}
        )


@router.get("/search", response_model=StandardResponse)
async def search_stocks(
    keyword: str = Query(..., description="搜索关键词（股票代码或名称）"),
    limit: int = 50,
    data_service: StockDataService = Depends(get_data_service)
):
    """搜索股票"""
    try:
        if not keyword or len(keyword) < 1:
            return StandardResponse(
                success=True,
                message="搜索关键词不能为空",
                data={"stocks": [], "total": 0}
            )
        
        # 从远端数据服务获取股票列表
        all_stocks = await data_service.get_remote_stock_list()
        
        if not all_stocks:
            return StandardResponse(
                success=True,
                message="无法获取股票列表",
                data={"stocks": [], "total": 0}
            )
        
        # 搜索匹配的股票
        keyword_lower = keyword.lower()
        matched_stocks = []
        
        for stock in all_stocks:
            stock_code = stock.get("ts_code", "").lower()
            stock_name = stock.get("name", "").lower()
            
            if keyword_lower in stock_code or keyword_lower in stock_name:
                market = "深圳" if ".sz" in stock_code else "上海" if ".sh" in stock_code else "未知"
                
                matched_stocks.append({
                    "code": stock.get("ts_code", ""),
                    "name": stock.get("name", ""),
                    "market": market
                })
                
                if len(matched_stocks) >= limit:
                    break
        
        return StandardResponse(
            success=True,
            message=f"搜索到 {len(matched_stocks)} 只股票",
            data={"stocks": matched_stocks, "total": len(matched_stocks)}
        )
        
    except Exception as e:
        logger.error(f"搜索股票失败: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            message=f"搜索股票失败: {str(e)}",
            data={"stocks": [], "total": 0}
        )

