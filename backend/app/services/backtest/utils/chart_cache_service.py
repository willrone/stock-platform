"""
回测图表数据缓存服务
用于管理图表数据的缓存和过期机制
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_, or_
from loguru import logger

from app.core.database import get_async_session, retry_db_operation
from app.models.backtest_detailed_models import BacktestChartCache


class ChartCacheService:
    """图表缓存服务"""
    
    # 默认缓存过期时间（小时）
    DEFAULT_CACHE_EXPIRY_HOURS = 24
    
    # 支持的图表类型
    SUPPORTED_CHART_TYPES = [
        "equity_curve",          # 收益曲线
        "drawdown_curve",        # 回撤曲线
        "monthly_heatmap",       # 月度收益热力图
        "trade_distribution",    # 交易分布图
        "position_weights",      # 持仓权重图
        "risk_metrics",          # 风险指标图
        "rolling_metrics",       # 滚动指标图
        "benchmark_comparison",  # 基准对比图
    ]
    
    def __init__(self):
        self.logger = logger.bind(service="chart_cache")
    
    async def get_cached_chart_data(
        self, 
        task_id: str, 
        chart_type: str
    ) -> Optional[Dict[str, Any]]:
        """获取缓存的图表数据"""
        
        if chart_type not in self.SUPPORTED_CHART_TYPES:
            self.logger.warning(f"不支持的图表类型: {chart_type}")
            return None
        
        async for session in get_async_session():
            try:
                async def _get_cache():
                    # 查询缓存记录
                    stmt = select(BacktestChartCache).where(
                        and_(
                            BacktestChartCache.task_id == task_id,
                            BacktestChartCache.chart_type == chart_type
                        )
                    )
                    result = await session.execute(stmt)
                    cache_record = result.scalar_one_or_none()
                    
                    if not cache_record:
                        self.logger.debug(f"未找到缓存数据: task_id={task_id}, chart_type={chart_type}")
                        return None
                    
                    # 检查是否过期
                    if cache_record.is_expired():
                        self.logger.info(f"缓存已过期，删除记录: task_id={task_id}, chart_type={chart_type}")
                        await session.delete(cache_record)
                        await session.commit()
                        return None
                    
                    self.logger.info(f"命中缓存: task_id={task_id}, chart_type={chart_type}")
                    return cache_record.chart_data
                
                return await retry_db_operation(
                    _get_cache,
                    max_retries=3,
                    retry_delay=0.1,
                    operation_name=f"获取缓存数据 (task_id={task_id}, chart_type={chart_type})"
                )
                
            except Exception as e:
                self.logger.error(f"获取缓存数据失败: {e}", exc_info=True)
                await session.rollback()
                return None
            finally:
                break
    
    async def cache_chart_data(
        self,
        task_id: str,
        chart_type: str,
        chart_data: Dict[str, Any],
        expiry_hours: Optional[int] = None
    ) -> bool:
        """缓存图表数据"""
        
        if chart_type not in self.SUPPORTED_CHART_TYPES:
            self.logger.warning(f"不支持的图表类型: {chart_type}")
            return False
        
        async for session in get_async_session():
            try:
                async def _cache_data():
                    # 计算数据哈希值
                    data_hash = self._calculate_data_hash(chart_data)
                    
                    # 计算过期时间
                    expiry_hours = expiry_hours or self.DEFAULT_CACHE_EXPIRY_HOURS
                    expires_at = datetime.utcnow() + timedelta(hours=expiry_hours)
                    
                    # 查找现有记录
                    stmt = select(BacktestChartCache).where(
                        and_(
                            BacktestChartCache.task_id == task_id,
                            BacktestChartCache.chart_type == chart_type
                        )
                    )
                    result = await session.execute(stmt)
                    existing_record = result.scalar_one_or_none()
                    
                    if existing_record:
                        # 更新现有记录
                        existing_record.chart_data = chart_data
                        existing_record.data_hash = data_hash
                        existing_record.expires_at = expires_at
                        existing_record.created_at = datetime.utcnow()
                        self.logger.info(f"更新缓存: task_id={task_id}, chart_type={chart_type}")
                    else:
                        # 创建新记录
                        cache_record = BacktestChartCache(
                            task_id=task_id,
                            chart_type=chart_type,
                            chart_data=chart_data,
                            data_hash=data_hash,
                            expires_at=expires_at
                        )
                        session.add(cache_record)
                        self.logger.info(f"创建缓存: task_id={task_id}, chart_type={chart_type}")
                    
                    await session.commit()
                    return True
                
                return await retry_db_operation(
                    _cache_data,
                    max_retries=3,
                    retry_delay=0.1,
                    operation_name=f"缓存图表数据 (task_id={task_id}, chart_type={chart_type})"
                )
                
            except Exception as e:
                self.logger.error(f"缓存图表数据失败: {e}", exc_info=True)
                await session.rollback()
                return False
            finally:
                break  # 只使用第一个会话
    
    async def invalidate_cache(
        self, 
        task_id: str, 
        chart_type: Optional[str] = None
    ) -> bool:
        """使缓存失效"""
        
        async for session in get_async_session():
            try:
                async def _invalidate():
                    if chart_type:
                        # 删除特定图表类型的缓存
                        stmt = delete(BacktestChartCache).where(
                            and_(
                                BacktestChartCache.task_id == task_id,
                                BacktestChartCache.chart_type == chart_type
                            )
                        )
                        self.logger.info(f"删除特定缓存: task_id={task_id}, chart_type={chart_type}")
                    else:
                        # 删除任务的所有缓存
                        stmt = delete(BacktestChartCache).where(
                            BacktestChartCache.task_id == task_id
                        )
                        self.logger.info(f"删除任务所有缓存: task_id={task_id}")
                    
                    result = await session.execute(stmt)
                    await session.commit()
                    
                    deleted_count = result.rowcount
                    self.logger.info(f"删除了 {deleted_count} 条缓存记录")
                    return True
                
                return await retry_db_operation(
                    _invalidate,
                    max_retries=3,
                    retry_delay=0.1,
                    operation_name=f"删除缓存 (task_id={task_id}, chart_type={chart_type})"
                )
                
            except Exception as e:
                self.logger.error(f"删除缓存失败: {e}", exc_info=True)
                await session.rollback()
                return False
            finally:
                break
    
    async def cleanup_expired_cache(self) -> int:
        """清理过期的缓存记录"""
        
        async for session in get_async_session():
            try:
                async def _cleanup():
                    # 删除过期的缓存记录
                    stmt = delete(BacktestChartCache).where(
                        and_(
                            BacktestChartCache.expires_at.isnot(None),
                            BacktestChartCache.expires_at < datetime.utcnow()
                        )
                    )
                    
                    result = await session.execute(stmt)
                    await session.commit()
                    
                    deleted_count = result.rowcount
                    if deleted_count > 0:
                        self.logger.info(f"清理了 {deleted_count} 条过期缓存记录")
                    
                    return deleted_count
                
                return await retry_db_operation(
                    _cleanup,
                    max_retries=3,
                    retry_delay=0.1,
                    operation_name="清理过期缓存"
                )
                
            except Exception as e:
                self.logger.error(f"清理过期缓存失败: {e}", exc_info=True)
                await session.rollback()
                return 0
            finally:
                break
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        
        async for session in get_async_session():
            try:
                # 总缓存记录数
                total_stmt = select(BacktestChartCache)
                total_result = await session.execute(total_stmt)
                total_count = len(total_result.scalars().all())
                
                # 过期记录数
                expired_stmt = select(BacktestChartCache).where(
                    and_(
                        BacktestChartCache.expires_at.isnot(None),
                        BacktestChartCache.expires_at < datetime.utcnow()
                    )
                )
                expired_result = await session.execute(expired_stmt)
                expired_count = len(expired_result.scalars().all())
                
                # 按图表类型统计
                type_stats = {}
                for chart_type in self.SUPPORTED_CHART_TYPES:
                    type_stmt = select(BacktestChartCache).where(
                        BacktestChartCache.chart_type == chart_type
                    )
                    type_result = await session.execute(type_stmt)
                    type_count = len(type_result.scalars().all())
                    type_stats[chart_type] = type_count
                
                return {
                    "total_cache_records": total_count,
                    "expired_records": expired_count,
                    "active_records": total_count - expired_count,
                    "cache_by_type": type_stats,
                    "supported_chart_types": self.SUPPORTED_CHART_TYPES,
                    "default_expiry_hours": self.DEFAULT_CACHE_EXPIRY_HOURS
                }
                
            except Exception as e:
                self.logger.error(f"获取缓存统计失败: {e}", exc_info=True)
                return {}
            finally:
                break
    
    async def get_task_cache_info(self, task_id: str) -> Dict[str, Any]:
        """获取特定任务的缓存信息"""
        
        async for session in get_async_session():
            try:
                stmt = select(BacktestChartCache).where(
                    BacktestChartCache.task_id == task_id
                )
                result = await session.execute(stmt)
                cache_records = result.scalars().all()
                
                cache_info = []
                for record in cache_records:
                    cache_info.append({
                        "chart_type": record.chart_type,
                        "created_at": record.created_at.isoformat(),
                        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                        "is_expired": record.is_expired(),
                        "data_hash": record.data_hash
                    })
                
                return {
                    "task_id": task_id,
                    "total_cached_charts": len(cache_records),
                    "cache_details": cache_info
                }
                
            except Exception as e:
                self.logger.error(f"获取任务缓存信息失败: {e}", exc_info=True)
                return {"task_id": task_id, "error": str(e)}
            finally:
                break
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """计算数据的哈希值"""
        try:
            # 将数据转换为JSON字符串并计算MD5哈希
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(data_str.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.warning(f"计算数据哈希失败: {e}")
            return ""
    
    async def batch_cache_charts(
        self,
        task_id: str,
        charts_data: Dict[str, Dict[str, Any]],
        expiry_hours: Optional[int] = None
    ) -> Dict[str, bool]:
        """批量缓存多个图表数据"""
        
        results = {}
        
        for chart_type, chart_data in charts_data.items():
            success = await self.cache_chart_data(
                task_id=task_id,
                chart_type=chart_type,
                chart_data=chart_data,
                expiry_hours=expiry_hours
            )
            results[chart_type] = success
        
        return results
    
    async def is_cache_valid(
        self,
        task_id: str,
        chart_type: str,
        data_hash: Optional[str] = None
    ) -> bool:
        """检查缓存是否有效"""
        
        async for session in get_async_session():
            try:
                stmt = select(BacktestChartCache).where(
                    and_(
                        BacktestChartCache.task_id == task_id,
                        BacktestChartCache.chart_type == chart_type
                    )
                )
                result = await session.execute(stmt)
                cache_record = result.scalar_one_or_none()
                
                if not cache_record:
                    return False
                
                # 检查是否过期
                if cache_record.is_expired():
                    return False
                
                # 如果提供了数据哈希，检查数据是否变化
                if data_hash and cache_record.data_hash != data_hash:
                    return False
                
                return True
                
            except Exception as e:
                self.logger.error(f"检查缓存有效性失败: {e}", exc_info=True)
                return False
            finally:
                break


# 全局缓存服务实例
chart_cache_service = ChartCacheService()