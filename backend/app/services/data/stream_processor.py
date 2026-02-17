"""
流式数据处理服务
实现大数据流式处理和内存优化
"""

import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import pandas as pd
import psutil
import pyarrow.parquet as pq
from loguru import logger


@dataclass
class ProcessingStats:
    """处理统计信息"""

    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    processing_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    throughput_records_per_second: float = 0.0


class MemoryMonitor:
    """内存监控器"""

    def __init__(
        self, warning_threshold_mb: int = 500, critical_threshold_mb: int = 1000
    ):
        self.warning_threshold = warning_threshold_mb * 1024 * 1024
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
        self.peak_usage = 0
        self.process = psutil.Process()

    def get_current_usage(self) -> int:
        """获取当前内存使用量（字节）"""
        return self.process.memory_info().rss

    def check_memory(self) -> Dict[str, Any]:
        """检查内存状态"""
        current_usage = self.get_current_usage()
        self.peak_usage = max(self.peak_usage, current_usage)

        status = "normal"
        if current_usage > self.critical_threshold:
            status = "critical"
        elif current_usage > self.warning_threshold:
            status = "warning"

        return {
            "current_mb": current_usage / (1024 * 1024),
            "peak_mb": self.peak_usage / (1024 * 1024),
            "status": status,
            "warning_threshold_mb": self.warning_threshold / (1024 * 1024),
            "critical_threshold_mb": self.critical_threshold / (1024 * 1024),
        }

    def force_gc(self):
        """强制垃圾回收"""
        gc.collect()


class ChunkedDataReader:
    """分块数据读取器"""

    def __init__(
        self, chunk_size: int = 10000, memory_monitor: Optional[MemoryMonitor] = None
    ):
        self.chunk_size = chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()

    async def read_parquet_chunks(self, file_path: Path) -> AsyncIterator[pd.DataFrame]:
        """异步读取Parquet文件分块"""
        try:
            # 读取文件元数据
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows

            logger.info(f"开始分块读取Parquet文件: {file_path}, 总行数: {total_rows}")

            # 分批读取
            for batch_start in range(0, total_rows, self.chunk_size):
                batch_end = min(batch_start + self.chunk_size, total_rows)

                # 检查内存状态
                memory_status = self.memory_monitor.check_memory()
                if memory_status["status"] == "critical":
                    logger.warning(
                        f"内存使用过高: {memory_status['current_mb']:.1f}MB，强制垃圾回收"
                    )
                    self.memory_monitor.force_gc()
                    await asyncio.sleep(0.1)  # 让出控制权

                # 读取数据块
                table = parquet_file.read_row_group_batch(
                    row_groups=range(batch_start // 10000, (batch_end - 1) // 10000 + 1)
                )
                df_chunk = table.to_pandas()

                # 过滤到实际需要的行
                if len(df_chunk) > self.chunk_size:
                    df_chunk = df_chunk.iloc[: self.chunk_size]

                logger.debug(f"读取数据块: {batch_start}-{batch_end}, 大小: {len(df_chunk)}")
                yield df_chunk

                # 让出控制权，避免阻塞
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"读取Parquet文件失败: {file_path}, 错误: {e}")
            raise

    async def read_csv_chunks(
        self, file_path: Path, **kwargs
    ) -> AsyncIterator[pd.DataFrame]:
        """异步读取CSV文件分块"""
        try:
            logger.info(f"开始分块读取CSV文件: {file_path}")

            # 使用pandas的chunksize参数
            chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size, **kwargs)

            for chunk_idx, df_chunk in enumerate(chunk_reader):
                # 检查内存状态
                memory_status = self.memory_monitor.check_memory()
                if memory_status["status"] == "critical":
                    logger.warning(
                        f"内存使用过高: {memory_status['current_mb']:.1f}MB，强制垃圾回收"
                    )
                    self.memory_monitor.force_gc()
                    await asyncio.sleep(0.1)

                logger.debug(f"读取CSV数据块 {chunk_idx}: 大小 {len(df_chunk)}")
                yield df_chunk

                # 让出控制权
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"读取CSV文件失败: {file_path}, 错误: {e}")
            raise


class StreamProcessor:
    """流式数据处理器"""

    def __init__(
        self, chunk_size: int = 10000, max_workers: int = 4, memory_limit_mb: int = 1000
    ):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.memory_monitor = MemoryMonitor(
            warning_threshold_mb=memory_limit_mb // 2,
            critical_threshold_mb=memory_limit_mb,
        )
        self.reader = ChunkedDataReader(chunk_size, self.memory_monitor)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._processing_lock = threading.Lock()

    async def process_file_stream(
        self,
        file_path: Path,
        processor_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[Path] = None,
        file_type: str = "parquet",
    ) -> ProcessingStats:
        """流式处理文件"""
        start_time = datetime.now()
        stats = ProcessingStats()

        try:
            logger.info(f"开始流式处理文件: {file_path}")

            # 选择读取器
            if file_type.lower() == "parquet":
                chunk_iterator = self.reader.read_parquet_chunks(file_path)
            elif file_type.lower() == "csv":
                chunk_iterator = self.reader.read_csv_chunks(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")

            # 处理结果收集器
            processed_chunks = []

            # 流式处理每个数据块
            async for chunk_df in chunk_iterator:
                stats.total_records += len(chunk_df)

                try:
                    # 在线程池中处理数据块
                    loop = asyncio.get_event_loop()
                    processed_chunk = await loop.run_in_executor(
                        self.executor, processor_func, chunk_df
                    )

                    if processed_chunk is not None and not processed_chunk.empty:
                        processed_chunks.append(processed_chunk)
                        stats.processed_records += len(processed_chunk)

                    # 更新内存峰值
                    memory_status = self.memory_monitor.check_memory()
                    stats.memory_peak_mb = max(
                        stats.memory_peak_mb, memory_status["current_mb"]
                    )

                    # 如果内存压力过大，提前写入部分结果
                    if (
                        memory_status["status"] == "warning"
                        and len(processed_chunks) > 10
                    ):
                        await self._flush_chunks(processed_chunks, output_path)
                        processed_chunks.clear()
                        self.memory_monitor.force_gc()

                except Exception as e:
                    logger.error(f"处理数据块失败: {e}")
                    stats.failed_records += len(chunk_df)
                    continue

            # 处理剩余的数据块
            if processed_chunks and output_path:
                await self._flush_chunks(processed_chunks, output_path)

            # 计算统计信息
            end_time = datetime.now()
            stats.processing_time_seconds = (end_time - start_time).total_seconds()

            if stats.processing_time_seconds > 0:
                stats.throughput_records_per_second = (
                    stats.processed_records / stats.processing_time_seconds
                )

            logger.info(
                f"流式处理完成: 处理 {stats.processed_records}/{stats.total_records} 条记录"
            )

            return stats

        except Exception as e:
            logger.error(f"流式处理失败: {e}")
            raise

    async def _flush_chunks(
        self, chunks: List[pd.DataFrame], output_path: Optional[Path]
    ):
        """刷新数据块到文件"""
        if not chunks or not output_path:
            return

        try:
            # 合并数据块
            combined_df = pd.concat(chunks, ignore_index=True)

            # 根据文件扩展名选择保存格式
            if output_path.suffix.lower() == ".parquet":
                combined_df.to_parquet(output_path, index=False, engine="pyarrow")
            elif output_path.suffix.lower() == ".csv":
                combined_df.to_csv(output_path, index=False)
            else:
                # 默认保存为parquet
                output_path = output_path.with_suffix(".parquet")
                combined_df.to_parquet(output_path, index=False, engine="pyarrow")

            logger.info(f"保存处理结果: {output_path}, {len(combined_df)} 条记录")

        except Exception as e:
            logger.error(f"保存处理结果失败: {e}")
            raise

    async def aggregate_stream(
        self,
        file_path: Path,
        group_by_columns: List[str],
        agg_functions: Dict[str, Union[str, List[str]]],
        file_type: str = "parquet",
    ) -> pd.DataFrame:
        """流式聚合处理"""
        logger.info(f"开始流式聚合: {file_path}")

        # 累积聚合结果
        aggregated_results = {}

        # 选择读取器
        if file_type.lower() == "parquet":
            chunk_iterator = self.reader.read_parquet_chunks(file_path)
        elif file_type.lower() == "csv":
            chunk_iterator = self.reader.read_csv_chunks(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")

        async for chunk_df in chunk_iterator:
            try:
                # 对当前块进行聚合
                chunk_agg = chunk_df.groupby(group_by_columns).agg(agg_functions)

                # 合并到累积结果中
                for group_key, group_data in chunk_agg.iterrows():
                    if group_key not in aggregated_results:
                        aggregated_results[group_key] = group_data.to_dict()
                    else:
                        # 合并聚合结果（这里需要根据具体的聚合函数来处理）
                        for col, value in group_data.items():
                            if col in aggregated_results[group_key]:
                                # 简单的累加处理，实际应用中需要更复杂的逻辑
                                if isinstance(value, (int, float)):
                                    aggregated_results[group_key][col] += value
                            else:
                                aggregated_results[group_key][col] = value

            except Exception as e:
                logger.error(f"聚合数据块失败: {e}")
                continue

        # 转换为DataFrame
        if aggregated_results:
            result_df = pd.DataFrame.from_dict(aggregated_results, orient="index")
            result_df.index.names = group_by_columns
            result_df = result_df.reset_index()

            logger.info(f"流式聚合完成: {len(result_df)} 个分组")
            return result_df
        else:
            logger.warning("聚合结果为空")
            return pd.DataFrame()

    async def filter_stream(
        self,
        file_path: Path,
        filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Path,
        file_type: str = "parquet",
    ) -> ProcessingStats:
        """流式过滤处理"""
        return await self.process_file_stream(
            file_path=file_path,
            processor_func=filter_func,
            output_path=output_path,
            file_type=file_type,
        )

    async def transform_stream(
        self,
        file_path: Path,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Path,
        file_type: str = "parquet",
    ) -> ProcessingStats:
        """流式转换处理"""
        return await self.process_file_stream(
            file_path=file_path,
            processor_func=transform_func,
            output_path=output_path,
            file_type=file_type,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return self.memory_monitor.check_memory()

    async def close(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)


class BatchProcessor:
    """批处理器"""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    async def process_batches(
        self,
        data_iterator: AsyncIterator[Any],
        processor_func: Callable[[List[Any]], List[Any]],
    ) -> AsyncIterator[List[Any]]:
        """批量处理数据"""
        batch = []

        async for item in data_iterator:
            batch.append(item)

            if len(batch) >= self.batch_size:
                # 处理当前批次
                try:
                    processed_batch = processor_func(batch)
                    yield processed_batch
                except Exception as e:
                    logger.error(f"批处理失败: {e}")
                    yield []

                batch.clear()
                await asyncio.sleep(0)  # 让出控制权

        # 处理剩余的数据
        if batch:
            try:
                processed_batch = processor_func(batch)
                yield processed_batch
            except Exception as e:
                logger.error(f"最后批次处理失败: {e}")
                yield []


# 全局流处理器实例
stream_processor = StreamProcessor()
