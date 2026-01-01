"""
数据同步引擎
协调数据服务和文件管理器，实现批量数据同步
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict

from .data_service import StockDataService
from .parquet_manager import ParquetManager
from app.models.sync_models import (
    BatchSyncRequest, SyncOptions, SyncResult, BatchSyncResult,
    SyncProgress, RetryResult, SyncHistoryEntry, SyncMode, SyncStatus
)


class SyncProgressTracker:
    """同步进度跟踪器"""
    
    def __init__(self):
        self._progress_data: Dict[str, SyncProgress] = {}
        self._lock = asyncio.Lock()
    
    async def create_progress(self, sync_id: str, total_stocks: int) -> SyncProgress:
        """创建进度跟踪"""
        async with self._lock:
            progress = SyncProgress(
                sync_id=sync_id,
                total_stocks=total_stocks,
                completed_stocks=0,
                failed_stocks=0,
                current_stock=None,
                progress_percentage=0.0,
                estimated_remaining_time=None,
                start_time=datetime.now(),
                status=SyncStatus.PENDING,
                last_update=datetime.now()
            )
            self._progress_data[sync_id] = progress
            return progress
    
    async def update_progress(
        self, 
        sync_id: str, 
        completed: int = None,
        failed: int = None,
        current_stock: str = None,
        status: SyncStatus = None
    ):
        """更新进度"""
        async with self._lock:
            if sync_id not in self._progress_data:
                return
            
            progress = self._progress_data[sync_id]
            
            if completed is not None:
                progress.completed_stocks = completed
            if failed is not None:
                progress.failed_stocks = failed
            if current_stock is not None:
                progress.current_stock = current_stock
            if status is not None:
                progress.status = status
            
            # 计算进度百分比
            total_processed = progress.completed_stocks + progress.failed_stocks
            progress.progress_percentage = (total_processed / progress.total_stocks) * 100 if progress.total_stocks > 0 else 0
            
            # 估算剩余时间
            if total_processed > 0 and progress.status == SyncStatus.RUNNING:
                elapsed = datetime.now() - progress.start_time
                avg_time_per_stock = elapsed / total_processed
                remaining_stocks = progress.total_stocks - total_processed
                progress.estimated_remaining_time = avg_time_per_stock * remaining_stocks
            
            progress.last_update = datetime.now()
    
    async def get_progress(self, sync_id: str) -> Optional[SyncProgress]:
        """获取进度"""
        async with self._lock:
            return self._progress_data.get(sync_id)
    
    async def remove_progress(self, sync_id: str):
        """移除进度跟踪"""
        async with self._lock:
            self._progress_data.pop(sync_id, None)


class DataSyncEngine:
    """数据同步引擎"""
    
    def __init__(self, data_service: StockDataService, parquet_manager: ParquetManager):
        self.data_service = data_service
        self.parquet_manager = parquet_manager
        self.progress_tracker = SyncProgressTracker()
        self.sync_history: List[SyncHistoryEntry] = []
        self.logger = logging.getLogger(__name__)
        
        # 同步队列和控制
        self.sync_queue = asyncio.Queue()
        self._active_syncs: Dict[str, asyncio.Task] = {}
        self._shutdown = False
    
    async def sync_stocks_batch(self, request: BatchSyncRequest) -> BatchSyncResult:
        """
        批量同步股票数据
        
        Args:
            request: 批量同步请求
        
        Returns:
            BatchSyncResult: 批量同步结果
        """
        sync_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"开始批量同步 {sync_id}: {len(request.stock_codes)} 只股票")
        
        # 创建进度跟踪
        await self.progress_tracker.create_progress(sync_id, len(request.stock_codes))
        await self.progress_tracker.update_progress(sync_id, status=SyncStatus.RUNNING)
        
        successful_syncs = []
        failed_syncs = []
        
        try:
            # 设置默认日期范围
            end_date = request.end_date or datetime.now()
            start_date = request.start_date or (end_date - timedelta(days=365))
            
            # 创建同步选项
            sync_options = SyncOptions(
                force_update=request.force_update,
                sync_mode=request.sync_mode,
                retry_count=request.retry_count
            )
            
            # 并发控制
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def sync_single_stock_wrapper(stock_code: str):
                async with semaphore:
                    # 更新当前处理的股票
                    await self.progress_tracker.update_progress(sync_id, current_stock=stock_code)
                    
                    # 执行同步
                    result = await self.sync_single_stock(stock_code, sync_options)
                    
                    # 更新进度
                    if result.success:
                        successful_syncs.append(result)
                        await self.progress_tracker.update_progress(
                            sync_id, 
                            completed=len(successful_syncs),
                            failed=len(failed_syncs)
                        )
                    else:
                        failed_syncs.append(result)
                        await self.progress_tracker.update_progress(
                            sync_id,
                            completed=len(successful_syncs), 
                            failed=len(failed_syncs)
                        )
                    
                    return result
            
            # 并发执行同步任务
            tasks = [sync_single_stock_wrapper(code) for code in request.stock_codes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    stock_code = request.stock_codes[i]
                    error_result = SyncResult(
                        stock_code=stock_code,
                        success=False,
                        records_synced=0,
                        start_time=start_time,
                        end_time=datetime.now(),
                        error_message=f"同步异常: {str(result)}"
                    )
                    failed_syncs.append(error_result)
            
            end_time = datetime.now()
            success = len(failed_syncs) == 0
            
            # 更新最终状态
            final_status = SyncStatus.COMPLETED if success else SyncStatus.FAILED
            await self.progress_tracker.update_progress(sync_id, status=final_status)
            
            # 创建批量同步结果
            batch_result = BatchSyncResult(
                sync_id=sync_id,
                success=success,
                total_stocks=len(request.stock_codes),
                successful_syncs=successful_syncs,
                failed_syncs=failed_syncs,
                start_time=start_time,
                end_time=end_time,
                message=f"批量同步完成: 成功 {len(successful_syncs)}, 失败 {len(failed_syncs)}, 总记录 {sum(r.records_synced for r in successful_syncs)}"
            )
            
            # 保存到历史记录
            history_entry = SyncHistoryEntry(
                sync_id=sync_id,
                request=request,
                result=batch_result,
                created_at=start_time
            )
            self.sync_history.append(history_entry)
            
            # 限制历史记录数量
            if len(self.sync_history) > 100:
                self.sync_history = self.sync_history[-100:]
            
            self.logger.info(f"批量同步完成 {sync_id}: {batch_result.message}")
            return batch_result
        
        except Exception as e:
            self.logger.error(f"批量同步失败 {sync_id}: {e}")
            
            # 更新失败状态
            await self.progress_tracker.update_progress(sync_id, status=SyncStatus.FAILED)
            
            # 创建失败结果
            return BatchSyncResult(
                sync_id=sync_id,
                success=False,
                total_stocks=len(request.stock_codes),
                successful_syncs=successful_syncs,
                failed_syncs=failed_syncs,
                start_time=start_time,
                end_time=datetime.now(),
                message=f"批量同步失败: {str(e)}"
            )
        
        finally:
            # 清理进度跟踪（延迟清理，给客户端时间获取最终状态）
            asyncio.create_task(self._cleanup_progress_later(sync_id))
    
    async def sync_single_stock(self, stock_code: str, options: SyncOptions) -> SyncResult:
        """
        同步单只股票数据
        
        Args:
            stock_code: 股票代码
            options: 同步选项
        
        Returns:
            SyncResult: 同步结果
        """
        start_time = datetime.now()
        
        try:
            # 确定同步日期范围
            end_date = datetime.now()
            
            if options.sync_mode == SyncMode.INCREMENTAL:
                # 增量同步：获取本地数据的最新日期
                date_range = self.parquet_manager.get_available_date_range(stock_code)
                if date_range and not options.force_update:
                    start_date = date_range[1] + timedelta(days=1)  # 从最新日期的下一天开始
                else:
                    start_date = end_date - timedelta(days=30)  # 默认获取30天数据
            else:
                # 全量同步
                start_date = end_date - timedelta(days=365)  # 获取一年数据
            
            # 如果开始日期晚于结束日期，说明数据已经是最新的
            if start_date >= end_date:
                return SyncResult(
                    stock_code=stock_code,
                    success=True,
                    records_synced=0,
                    start_time=start_time,
                    end_time=datetime.now(),
                    data_range=(start_date, end_date),
                    error_message=None
                )
            
            # 从数据服务获取数据
            stock_data = await self.data_service.get_stock_data(
                stock_code, 
                start_date, 
                end_date,
                force_remote=options.force_update
            )
            
            if not stock_data:
                return SyncResult(
                    stock_code=stock_code,
                    success=False,
                    records_synced=0,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message="未获取到数据"
                )
            
            # 保存到本地存储
            save_success = self.parquet_manager.save_stock_data(stock_data, merge_with_existing=True)
            
            if not save_success:
                return SyncResult(
                    stock_code=stock_code,
                    success=False,
                    records_synced=len(stock_data),
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message="数据保存失败"
                )
            
            return SyncResult(
                stock_code=stock_code,
                success=True,
                records_synced=len(stock_data),
                start_time=start_time,
                end_time=datetime.now(),
                data_range=(start_date, end_date)
            )
        
        except Exception as e:
            self.logger.error(f"同步股票失败 {stock_code}: {e}")
            return SyncResult(
                stock_code=stock_code,
                success=False,
                records_synced=0,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
    
    async def get_sync_progress(self, sync_id: str) -> Optional[SyncProgress]:
        """
        获取同步进度
        
        Args:
            sync_id: 同步ID
        
        Returns:
            Optional[SyncProgress]: 同步进度
        """
        return await self.progress_tracker.get_progress(sync_id)
    
    async def retry_failed_syncs(self, sync_id: str) -> RetryResult:
        """
        重试失败的同步
        
        Args:
            sync_id: 同步ID
        
        Returns:
            RetryResult: 重试结果
        """
        # 查找历史记录
        history_entry = None
        for entry in self.sync_history:
            if entry.sync_id == sync_id:
                history_entry = entry
                break
        
        if not history_entry:
            return RetryResult(
                sync_id=sync_id,
                retried_stocks=[],
                retry_results=[],
                success=False,
                message="未找到同步记录"
            )
        
        # 获取失败的股票
        failed_stocks = [result.stock_code for result in history_entry.result.failed_syncs]
        
        if not failed_stocks:
            return RetryResult(
                sync_id=sync_id,
                retried_stocks=[],
                retry_results=[],
                success=True,
                message="没有失败的股票需要重试"
            )
        
        # 创建重试请求
        retry_request = BatchSyncRequest(
            stock_codes=failed_stocks,
            start_date=history_entry.request.start_date,
            end_date=history_entry.request.end_date,
            force_update=history_entry.request.force_update,
            sync_mode=history_entry.request.sync_mode,
            max_concurrent=history_entry.request.max_concurrent,
            retry_count=history_entry.request.retry_count
        )
        
        # 执行重试
        retry_result_batch = await self.sync_stocks_batch(retry_request)
        
        return RetryResult(
            sync_id=sync_id,
            retried_stocks=failed_stocks,
            retry_results=retry_result_batch.successful_syncs + retry_result_batch.failed_syncs,
            success=retry_result_batch.success,
            message=f"重试完成: 成功 {retry_result_batch.success_count}, 失败 {retry_result_batch.failure_count}"
        )
    
    def get_sync_history(self, limit: int = 50) -> List[SyncHistoryEntry]:
        """
        获取同步历史
        
        Args:
            limit: 限制数量
        
        Returns:
            List[SyncHistoryEntry]: 同步历史列表
        """
        return self.sync_history[-limit:]
    
    async def cleanup(self):
        """清理资源"""
        self._shutdown = True
        
        # 取消所有活跃的同步任务
        for task in self._active_syncs.values():
            if not task.done():
                task.cancel()
        
        # 等待任务完成
        if self._active_syncs:
            await asyncio.gather(*self._active_syncs.values(), return_exceptions=True)
        
        self._active_syncs.clear()
    
    async def _cleanup_progress_later(self, sync_id: str):
        """延迟清理进度跟踪"""
        await asyncio.sleep(300)  # 5分钟后清理
        await self.progress_tracker.remove_progress(sync_id)