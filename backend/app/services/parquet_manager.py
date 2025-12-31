"""
Parquet文件管理系统
实现按股票代码和时间范围的目录结构组织，以及Parquet文件的读写和索引
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

from app.models.stock_simple import StockData
from app.models.file_management import (
    DetailedFileInfo, ComprehensiveStats, FilterCriteria, 
    ValidationResult, DeletionResult, FileFilters, IntegrityStatus
)


@dataclass
class ParquetFileInfo:
    """Parquet文件信息"""
    file_path: str
    stock_code: str
    start_date: datetime
    end_date: datetime
    record_count: int
    file_size: int
    created_at: datetime
    modified_at: datetime


class ParquetManager:
    """Parquet文件管理器"""
    
    def __init__(self, base_path: str = "data/parquet"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 文件组织结构: data/parquet/{stock_code}/{year}/{month}/
        # 例如: data/parquet/000001.SZ/2023/01/000001.SZ_2023-01.parquet
    
    def get_file_path(self, stock_code: str, date: datetime) -> Path:
        """获取指定股票和日期的Parquet文件路径"""
        year = date.year
        month = date.month
        filename = f"{stock_code}_{year}-{month:02d}.parquet"
        return self.base_path / stock_code / str(year) / f"{month:02d}" / filename
    
    def get_directory_path(self, stock_code: str, date: datetime) -> Path:
        """获取指定股票和日期的目录路径"""
        year = date.year
        month = date.month
        return self.base_path / stock_code / str(year) / f"{month:02d}"
    
    def ensure_directory(self, file_path: Path):
        """确保目录存在"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save_stock_data(self, stock_data: List[StockData], merge_with_existing: bool = True) -> bool:
        """
        保存股票数据到Parquet文件
        
        Args:
            stock_data: 股票数据列表
            merge_with_existing: 是否与现有数据合并
        
        Returns:
            bool: 保存是否成功
        """
        if not stock_data:
            return True
        
        try:
            # 按股票代码和月份分组
            grouped_data = self._group_data_by_month(stock_data)
            
            for (stock_code, year, month), data_group in grouped_data.items():
                file_path = self.get_file_path(stock_code, datetime(year, month, 1))
                self.ensure_directory(file_path)
                
                # 转换为DataFrame
                df_new = self._stock_data_to_dataframe(data_group)
                
                if merge_with_existing and file_path.exists():
                    # 读取现有数据并合并
                    df_existing = pd.read_parquet(file_path)
                    
                    # 合并数据，去重并排序
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset=['stock_code', 'date'])
                    df_combined = df_combined.sort_values('date')
                    
                    df_to_save = df_combined
                else:
                    df_to_save = df_new.sort_values('date')
                
                # 保存到Parquet文件
                df_to_save.to_parquet(file_path, index=False, engine='pyarrow')
                
                self.logger.info(f"保存股票数据到 {file_path}: {len(df_to_save)} 条记录")
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存股票数据失败: {e}")
            return False
    
    def load_stock_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[StockData]:
        """
        从Parquet文件加载股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            List[StockData]: 股票数据列表
        """
        try:
            # 获取需要读取的文件列表
            file_paths = self._get_file_paths_for_date_range(stock_code, start_date, end_date)
            
            if not file_paths:
                return []
            
            # 读取所有相关文件
            dataframes = []
            for file_path in file_paths:
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    dataframes.append(df)
            
            if not dataframes:
                return []
            
            # 合并所有数据
            df_combined = pd.concat(dataframes, ignore_index=True)
            
            # 过滤日期范围
            df_filtered = df_combined[
                (df_combined['date'] >= start_date) & 
                (df_combined['date'] <= end_date) &
                (df_combined['stock_code'] == stock_code)
            ]
            
            # 排序并去重
            df_filtered = df_filtered.drop_duplicates(subset=['stock_code', 'date'])
            df_filtered = df_filtered.sort_values('date')
            
            # 转换为StockData对象
            return self._dataframe_to_stock_data(df_filtered)
            
        except Exception as e:
            self.logger.error(f"加载股票数据失败 {stock_code}: {e}")
            return []
    
    def get_available_date_range(self, stock_code: str) -> Optional[Tuple[datetime, datetime]]:
        """
        获取指定股票的可用数据日期范围
        
        Args:
            stock_code: 股票代码
        
        Returns:
            Optional[Tuple[datetime, datetime]]: (最早日期, 最晚日期)，如果没有数据则返回None
        """
        try:
            stock_dir = self.base_path / stock_code
            if not stock_dir.exists():
                return None
            
            all_dates = []
            
            # 遍历所有年份和月份目录
            for year_dir in stock_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    # 查找Parquet文件
                    for file_path in month_dir.glob(f"{stock_code}_*.parquet"):
                        try:
                            df = pd.read_parquet(file_path)
                            if not df.empty:
                                all_dates.extend(df['date'].tolist())
                        except Exception as e:
                            self.logger.warning(f"读取文件失败 {file_path}: {e}")
            
            if not all_dates:
                return None
            
            return min(all_dates), max(all_dates)
            
        except Exception as e:
            self.logger.error(f"获取日期范围失败 {stock_code}: {e}")
            return None
    
    def get_file_info(self, stock_code: str) -> List[ParquetFileInfo]:
        """
        获取指定股票的所有Parquet文件信息
        
        Args:
            stock_code: 股票代码
        
        Returns:
            List[ParquetFileInfo]: 文件信息列表
        """
        file_infos = []
        
        try:
            stock_dir = self.base_path / stock_code
            if not stock_dir.exists():
                return file_infos
            
            # 遍历所有Parquet文件
            for file_path in stock_dir.rglob(f"{stock_code}_*.parquet"):
                try:
                    # 获取文件统计信息
                    stat = file_path.stat()
                    
                    # 读取文件获取数据范围
                    df = pd.read_parquet(file_path)
                    
                    if not df.empty:
                        start_date = df['date'].min()
                        end_date = df['date'].max()
                        record_count = len(df)
                        
                        file_info = ParquetFileInfo(
                            file_path=str(file_path),
                            stock_code=stock_code,
                            start_date=start_date,
                            end_date=end_date,
                            record_count=record_count,
                            file_size=stat.st_size,
                            created_at=datetime.fromtimestamp(stat.st_ctime),
                            modified_at=datetime.fromtimestamp(stat.st_mtime)
                        )
                        
                        file_infos.append(file_info)
                
                except Exception as e:
                    self.logger.warning(f"获取文件信息失败 {file_path}: {e}")
            
            # 按开始日期排序
            file_infos.sort(key=lambda x: x.start_date)
            
        except Exception as e:
            self.logger.error(f"获取文件信息失败 {stock_code}: {e}")
        
        return file_infos
    
    def cleanup_old_files(self, days_to_keep: int = 365) -> int:
        """
        清理旧的Parquet文件
        
        Args:
            days_to_keep: 保留天数
        
        Returns:
            int: 删除的文件数量
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        try:
            for file_path in self.base_path.rglob("*.parquet"):
                try:
                    stat = file_path.stat()
                    if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.info(f"删除旧文件: {file_path}")
                
                except Exception as e:
                    self.logger.warning(f"删除文件失败 {file_path}: {e}")
            
            # 清理空目录
            self._cleanup_empty_directories()
            
        except Exception as e:
            self.logger.error(f"清理旧文件失败: {e}")
        
        return deleted_count
    
    def get_storage_stats(self) -> Dict[str, any]:
        """
        获取存储统计信息
        
        Returns:
            Dict: 存储统计信息
        """
        stats = {
            'total_files': 0,
            'total_size': 0,
            'stock_count': 0,
            'date_range': None,
            'stocks': {}
        }
        
        try:
            all_dates = []
            
            for stock_dir in self.base_path.iterdir():
                if not stock_dir.is_dir():
                    continue
                
                stock_code = stock_dir.name
                stock_stats = {
                    'file_count': 0,
                    'total_size': 0,
                    'record_count': 0,
                    'date_range': None
                }
                
                stock_dates = []
                
                for file_path in stock_dir.rglob("*.parquet"):
                    try:
                        stat = file_path.stat()
                        df = pd.read_parquet(file_path)
                        
                        stock_stats['file_count'] += 1
                        stock_stats['total_size'] += stat.st_size
                        stock_stats['record_count'] += len(df)
                        
                        if not df.empty:
                            file_dates = df['date'].tolist()
                            stock_dates.extend(file_dates)
                            all_dates.extend(file_dates)
                        
                        stats['total_files'] += 1
                        stats['total_size'] += stat.st_size
                    
                    except Exception as e:
                        self.logger.warning(f"统计文件失败 {file_path}: {e}")
                
                if stock_dates:
                    stock_stats['date_range'] = (min(stock_dates), max(stock_dates))
                
                if stock_stats['file_count'] > 0:
                    stats['stocks'][stock_code] = stock_stats
                    stats['stock_count'] += 1
            
            if all_dates:
                stats['date_range'] = (min(all_dates), max(all_dates))
        
        except Exception as e:
            self.logger.error(f"获取存储统计失败: {e}")
        
        return stats
    
    def _group_data_by_month(self, stock_data: List[StockData]) -> Dict[Tuple[str, int, int], List[StockData]]:
        """按股票代码和月份分组数据"""
        grouped = {}
        
        for data in stock_data:
            key = (data.stock_code, data.date.year, data.date.month)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(data)
        
        return grouped
    
    def _get_file_paths_for_date_range(self, stock_code: str, start_date: datetime, end_date: datetime) -> List[Path]:
        """获取日期范围内的所有文件路径"""
        file_paths = []
        
        current_date = start_date.replace(day=1)  # 从月初开始
        end_month = end_date.replace(day=1)
        
        while current_date <= end_month:
            file_path = self.get_file_path(stock_code, current_date)
            file_paths.append(file_path)
            
            # 移动到下个月
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return file_paths
    
    def _stock_data_to_dataframe(self, stock_data: List[StockData]) -> pd.DataFrame:
        """将StockData列表转换为DataFrame"""
        data_dicts = []
        for data in stock_data:
            data_dicts.append({
                'stock_code': data.stock_code,
                'date': data.date,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume,
                'adj_close': data.adj_close
            })
        
        return pd.DataFrame(data_dicts)
    
    def _dataframe_to_stock_data(self, df: pd.DataFrame) -> List[StockData]:
        """将DataFrame转换为StockData列表"""
        stock_data = []
        
        for _, row in df.iterrows():
            stock_data.append(StockData(
                stock_code=row['stock_code'],
                date=row['date'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                adj_close=row['adj_close']
            ))
        
        return stock_data
    
    def get_detailed_file_list(self, filters: FileFilters) -> List[DetailedFileInfo]:
        """
        获取详细文件列表
        
        Args:
            filters: 文件过滤条件
        
        Returns:
            List[DetailedFileInfo]: 详细文件信息列表
        """
        detailed_files = []
        
        try:
            # 确定搜索范围
            if filters.stock_code:
                stock_dirs = [self.base_path / filters.stock_code]
            else:
                stock_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
            
            for stock_dir in stock_dirs:
                if not stock_dir.exists():
                    continue
                
                stock_code = stock_dir.name
                
                # 遍历所有Parquet文件
                for file_path in stock_dir.rglob("*.parquet"):
                    try:
                        # 获取文件统计信息
                        stat = file_path.stat()
                        file_size = stat.st_size
                        last_modified = datetime.fromtimestamp(stat.st_mtime)
                        created_at = datetime.fromtimestamp(stat.st_ctime)
                        
                        # 应用大小过滤
                        if filters.min_size and file_size < filters.min_size:
                            continue
                        if filters.max_size and file_size > filters.max_size:
                            continue
                        
                        # 读取文件获取数据信息
                        try:
                            df = pd.read_parquet(file_path)
                            if df.empty:
                                continue
                            
                            record_count = len(df)
                            start_date = df['date'].min()
                            end_date = df['date'].max()
                            
                            # 应用日期过滤
                            if filters.start_date and end_date < filters.start_date:
                                continue
                            if filters.end_date and start_date > filters.end_date:
                                continue
                            
                            # 验证文件完整性
                            integrity_status = self._check_file_integrity(file_path, df)
                            
                            # 应用完整性过滤
                            if filters.integrity_status and integrity_status != filters.integrity_status:
                                continue
                            
                            # 计算压缩比（估算）
                            uncompressed_size = record_count * 8 * 8  # 估算未压缩大小
                            compression_ratio = file_size / uncompressed_size if uncompressed_size > 0 else 0
                            
                            detailed_info = DetailedFileInfo(
                                file_path=str(file_path),
                                stock_code=stock_code,
                                date_range=(start_date, end_date),
                                record_count=record_count,
                                file_size=file_size,
                                last_modified=last_modified,
                                integrity_status=integrity_status,
                                compression_ratio=compression_ratio,
                                created_at=created_at
                            )
                            
                            detailed_files.append(detailed_info)
                        
                        except Exception as e:
                            self.logger.warning(f"读取文件失败 {file_path}: {e}")
                            # 创建基本信息，标记为损坏
                            detailed_info = DetailedFileInfo(
                                file_path=str(file_path),
                                stock_code=stock_code,
                                date_range=(datetime.min, datetime.min),
                                record_count=0,
                                file_size=file_size,
                                last_modified=last_modified,
                                integrity_status=IntegrityStatus.CORRUPTED,
                                compression_ratio=0.0,
                                created_at=created_at
                            )
                            detailed_files.append(detailed_info)
                    
                    except Exception as e:
                        self.logger.warning(f"获取文件信息失败 {file_path}: {e}")
            
            # 排序
            detailed_files.sort(key=lambda x: x.last_modified, reverse=True)
            
            # 应用分页
            start_idx = filters.offset
            end_idx = start_idx + filters.limit
            return detailed_files[start_idx:end_idx]
        
        except Exception as e:
            self.logger.error(f"获取详细文件列表失败: {e}")
            return []
    
    def get_comprehensive_stats(self) -> ComprehensiveStats:
        """
        获取综合统计信息
        
        Returns:
            ComprehensiveStats: 综合统计信息
        """
        try:
            total_files = 0
            total_size = 0
            total_records = 0
            stock_count = 0
            all_dates = []
            stocks_by_size = []
            monthly_distribution = {}
            
            for stock_dir in self.base_path.iterdir():
                if not stock_dir.is_dir():
                    continue
                
                stock_code = stock_dir.name
                stock_size = 0
                stock_records = 0
                stock_files = 0
                stock_dates = []
                
                for file_path in stock_dir.rglob("*.parquet"):
                    try:
                        stat = file_path.stat()
                        file_size = stat.st_size
                        
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            record_count = len(df)
                            file_dates = pd.to_datetime(df['date']).tolist()
                            
                            # 统计月份分布
                            for date in file_dates:
                                month_key = date.strftime("%Y-%m")
                                monthly_distribution[month_key] = monthly_distribution.get(month_key, 0) + 1
                            
                            stock_dates.extend(file_dates)
                            all_dates.extend(file_dates)
                            stock_records += record_count
                            total_records += record_count
                        
                        stock_size += file_size
                        total_size += file_size
                        stock_files += 1
                        total_files += 1
                    
                    except Exception as e:
                        self.logger.warning(f"统计文件失败 {file_path}: {e}")
                
                if stock_files > 0:
                    stocks_by_size.append((stock_code, stock_size))
                    stock_count += 1
            
            # 排序股票按大小
            stocks_by_size.sort(key=lambda x: x[1], reverse=True)
            
            # 计算统计指标
            average_file_size = total_size / total_files if total_files > 0 else 0
            storage_efficiency = total_records / (total_size / 1024 / 1024) if total_size > 0 else 0  # 记录数/MB
            
            date_range = (min(all_dates), max(all_dates)) if all_dates else (datetime.min, datetime.min)
            
            # 获取最后同步时间（最新文件的修改时间）
            last_sync_time = None
            try:
                latest_file = max(
                    self.base_path.rglob("*.parquet"),
                    key=lambda p: p.stat().st_mtime,
                    default=None
                )
                if latest_file:
                    last_sync_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
            except Exception:
                pass
            
            return ComprehensiveStats(
                total_files=total_files,
                total_size_bytes=total_size,
                total_records=total_records,
                stock_count=stock_count,
                date_range=date_range,
                average_file_size=average_file_size,
                storage_efficiency=storage_efficiency,
                last_sync_time=last_sync_time,
                stocks_by_size=stocks_by_size[:10],  # 前10个最大的
                monthly_distribution=monthly_distribution
            )
        
        except Exception as e:
            self.logger.error(f"获取综合统计失败: {e}")
            return ComprehensiveStats(
                total_files=0,
                total_size_bytes=0,
                total_records=0,
                stock_count=0,
                date_range=(datetime.min, datetime.min),
                average_file_size=0.0,
                storage_efficiency=0.0,
                last_sync_time=None,
                stocks_by_size=[],
                monthly_distribution={}
            )
    
    def validate_file_integrity(self, file_path: str) -> ValidationResult:
        """
        验证文件完整性
        
        Args:
            file_path: 文件路径
        
        Returns:
            ValidationResult: 验证结果
        """
        validation_time = datetime.now()
        error_messages = []
        
        try:
            path = Path(file_path)
            
            # 检查文件是否存在
            if not path.exists():
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    integrity_status=IntegrityStatus.CORRUPTED,
                    error_messages=["文件不存在"],
                    validation_time=validation_time
                )
            
            # 获取文件大小
            file_size = path.stat().st_size
            if file_size == 0:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    integrity_status=IntegrityStatus.CORRUPTED,
                    error_messages=["文件为空"],
                    file_size=file_size,
                    validation_time=validation_time
                )
            
            # 尝试读取Parquet文件
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    integrity_status=IntegrityStatus.CORRUPTED,
                    error_messages=[f"无法读取Parquet文件: {str(e)}"],
                    file_size=file_size,
                    validation_time=validation_time
                )
            
            record_count = len(df)
            
            # 检查数据完整性
            integrity_status = self._check_file_integrity(path, df)
            
            if integrity_status == IntegrityStatus.CORRUPTED:
                error_messages.append("数据格式不正确")
            elif integrity_status == IntegrityStatus.INCOMPLETE:
                error_messages.append("数据不完整")
            
            is_valid = integrity_status == IntegrityStatus.VALID
            
            return ValidationResult(
                file_path=file_path,
                is_valid=is_valid,
                integrity_status=integrity_status,
                error_messages=error_messages,
                record_count=record_count,
                file_size=file_size,
                validation_time=validation_time
            )
        
        except Exception as e:
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                integrity_status=IntegrityStatus.UNKNOWN,
                error_messages=[f"验证过程出错: {str(e)}"],
                validation_time=validation_time
            )
    
    def delete_files_safely(self, file_paths: List[str]) -> DeletionResult:
        """
        安全删除文件
        
        Args:
            file_paths: 要删除的文件路径列表
        
        Returns:
            DeletionResult: 删除结果
        """
        deleted_files = []
        failed_files = []
        freed_space = 0
        
        try:
            for file_path in file_paths:
                try:
                    path = Path(file_path)
                    
                    if not path.exists():
                        failed_files.append((file_path, "文件不存在"))
                        continue
                    
                    # 获取文件大小
                    file_size = path.stat().st_size
                    
                    # 创建备份（可选，这里简化处理）
                    # backup_path = path.with_suffix(path.suffix + '.backup')
                    # shutil.copy2(path, backup_path)
                    
                    # 删除文件
                    path.unlink()
                    
                    deleted_files.append(file_path)
                    freed_space += file_size
                    
                    self.logger.info(f"成功删除文件: {file_path}")
                
                except Exception as e:
                    error_msg = f"删除失败: {str(e)}"
                    failed_files.append((file_path, error_msg))
                    self.logger.error(f"删除文件失败 {file_path}: {e}")
            
            # 清理空目录
            self._cleanup_empty_directories()
            
            success = len(failed_files) == 0
            total_deleted = len(deleted_files)
            
            if success:
                message = f"成功删除 {total_deleted} 个文件，释放空间 {freed_space / 1024 / 1024:.2f} MB"
            else:
                message = f"删除完成: 成功 {total_deleted}, 失败 {len(failed_files)}"
            
            return DeletionResult(
                success=success,
                deleted_files=deleted_files,
                failed_files=failed_files,
                total_deleted=total_deleted,
                freed_space_bytes=freed_space,
                message=message
            )
        
        except Exception as e:
            self.logger.error(f"批量删除文件失败: {e}")
            return DeletionResult(
                success=False,
                deleted_files=deleted_files,
                failed_files=failed_files + [(f"批量操作", str(e))],
                total_deleted=len(deleted_files),
                freed_space_bytes=freed_space,
                message=f"批量删除失败: {str(e)}"
            )
    
    def filter_files(self, criteria: FilterCriteria) -> List[DetailedFileInfo]:
        """
        根据条件筛选文件
        
        Args:
            criteria: 筛选条件
        
        Returns:
            List[DetailedFileInfo]: 筛选后的文件列表
        """
        try:
            all_files = []
            
            # 确定搜索范围
            if criteria.stock_codes:
                stock_dirs = [self.base_path / code for code in criteria.stock_codes if (self.base_path / code).exists()]
            else:
                stock_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
            
            for stock_dir in stock_dirs:
                stock_code = stock_dir.name
                
                for file_path in stock_dir.rglob("*.parquet"):
                    try:
                        stat = file_path.stat()
                        file_size = stat.st_size
                        last_modified = datetime.fromtimestamp(stat.st_mtime)
                        
                        # 应用大小过滤
                        if criteria.min_file_size and file_size < criteria.min_file_size:
                            continue
                        if criteria.max_file_size and file_size > criteria.max_file_size:
                            continue
                        
                        # 读取文件信息
                        df = pd.read_parquet(file_path)
                        if df.empty:
                            continue
                        
                        record_count = len(df)
                        start_date = df['date'].min()
                        end_date = df['date'].max()
                        
                        # 应用记录数过滤
                        if criteria.min_records and record_count < criteria.min_records:
                            continue
                        if criteria.max_records and record_count > criteria.max_records:
                            continue
                        
                        # 应用日期过滤
                        if criteria.date_range:
                            range_start, range_end = criteria.date_range
                            if end_date < range_start or start_date > range_end:
                                continue
                        
                        # 检查完整性
                        integrity_status = self._check_file_integrity(file_path, df)
                        if criteria.integrity_status and integrity_status != criteria.integrity_status:
                            continue
                        
                        # 计算压缩比
                        uncompressed_size = record_count * 8 * 8
                        compression_ratio = file_size / uncompressed_size if uncompressed_size > 0 else 0
                        
                        file_info = DetailedFileInfo(
                            file_path=str(file_path),
                            stock_code=stock_code,
                            date_range=(start_date, end_date),
                            record_count=record_count,
                            file_size=file_size,
                            last_modified=last_modified,
                            integrity_status=integrity_status,
                            compression_ratio=compression_ratio,
                            created_at=datetime.fromtimestamp(stat.st_ctime)
                        )
                        
                        all_files.append(file_info)
                    
                    except Exception as e:
                        self.logger.warning(f"处理文件失败 {file_path}: {e}")
            
            # 排序
            reverse = criteria.sort_order == "desc"
            if criteria.sort_by == "file_size":
                all_files.sort(key=lambda x: x.file_size, reverse=reverse)
            elif criteria.sort_by == "record_count":
                all_files.sort(key=lambda x: x.record_count, reverse=reverse)
            elif criteria.sort_by == "stock_code":
                all_files.sort(key=lambda x: x.stock_code, reverse=reverse)
            else:  # last_modified
                all_files.sort(key=lambda x: x.last_modified, reverse=reverse)
            
            return all_files
        
        except Exception as e:
            self.logger.error(f"筛选文件失败: {e}")
            return []
    
    def _check_file_integrity(self, file_path: Path, df: pd.DataFrame) -> IntegrityStatus:
        """
        检查文件完整性
        
        Args:
            file_path: 文件路径
            df: 数据框
        
        Returns:
            IntegrityStatus: 完整性状态
        """
        try:
            # 检查必需的列
            required_columns = ['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return IntegrityStatus.CORRUPTED
            
            # 检查数据类型
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                return IntegrityStatus.CORRUPTED
            
            # 检查数值列
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return IntegrityStatus.CORRUPTED
            
            # 检查数据逻辑
            invalid_rows = df[
                (df['high'] < df['low']) |
                (df['open'] <= 0) |
                (df['high'] <= 0) |
                (df['low'] <= 0) |
                (df['close'] <= 0) |
                (df['volume'] < 0)
            ]
            
            if len(invalid_rows) > 0:
                # 如果无效数据比例超过5%，认为文件损坏
                if len(invalid_rows) / len(df) > 0.05:
                    return IntegrityStatus.CORRUPTED
                else:
                    return IntegrityStatus.INCOMPLETE
            
            # 检查日期连续性（简单检查）
            df_sorted = df.sort_values('date')
            date_diffs = df_sorted['date'].diff().dt.days
            
            # 如果有超过7天的间隔，可能数据不完整
            if date_diffs.max() > 7:
                return IntegrityStatus.INCOMPLETE
            
            return IntegrityStatus.VALID
        
        except Exception as e:
            self.logger.warning(f"检查文件完整性失败 {file_path}: {e}")
            return IntegrityStatus.UNKNOWN
    
    def _cleanup_empty_directories(self):
        """清理空目录"""
        try:
            for root, dirs, files in os.walk(self.base_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            self.logger.info(f"删除空目录: {dir_path}")
                    except OSError:
                        pass  # 目录不为空或其他错误
        except Exception as e:
            self.logger.warning(f"清理空目录失败: {e}")