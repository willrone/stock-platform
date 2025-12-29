"""
Parquet数据访问层
使用Parquet文件格式存储股票数据
"""
import logging
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
import os
from pathlib import Path
from .config import Config

logger = logging.getLogger(__name__)


class ParquetDAO:
    """Parquet数据访问对象"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化Parquet DAO
        
        Args:
            data_dir: 数据存储目录，如果为None则使用配置或默认目录
        """
        if data_dir is None:
            # 优先使用配置中的路径
            if Config.PARQUET_DATA_DIR:
                data_dir = Config.PARQUET_DATA_DIR
            else:
                # 默认数据目录：项目根目录下的data/parquet
                base_dir = Path(__file__).parent.parent.parent
                data_dir = base_dir / "data" / "parquet"
        
        self.data_dir = Path(data_dir)
        self.stock_data_dir = self.data_dir / "stock_data"
        self.stock_list_file = self.data_dir / "stock_list.parquet"
        
        # 确保目录存在
        self.stock_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Parquet DAO初始化成功，数据目录: {self.data_dir}")
    
    def _get_stock_file_path(self, ts_code: str) -> Path:
        """获取股票数据文件路径"""
        # 使用股票代码作为文件名，替换特殊字符
        safe_code = ts_code.replace('.', '_')
        return self.stock_data_dir / f"{safe_code}.parquet"
    
    def save_stock_data(self, ts_code: str, df: pd.DataFrame) -> int:
        """
        保存股票数据到Parquet文件
        
        Args:
            ts_code: 股票代码
            df: 股票数据DataFrame，必须包含date, open, high, low, close, volume列
            
        Returns:
            保存的记录数
        """
        if df is None or df.empty:
            logger.warning(f"股票数据为空: {ts_code}")
            return 0
        
        try:
            # 确保date列是datetime类型
            if 'date' in df.columns:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
            
            # 添加ts_code列
            df['ts_code'] = ts_code
            
            # 确保列顺序一致
            df = df[['ts_code', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
            # 读取现有数据（如果存在）
            file_path = self._get_stock_file_path(ts_code)
            if file_path.exists():
                try:
                    existing_df = pd.read_parquet(file_path)
                    # 合并数据，去重（保留新数据）
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['date'], keep='last')
                    df = df.sort_values('date')
                except Exception as e:
                    logger.warning(f"读取现有数据失败，将覆盖: {e}")
            
            # 保存到Parquet文件
            df.to_parquet(file_path, index=False, engine='pyarrow', compression='snappy')
            
            saved_count = len(df)
            logger.info(f"保存股票数据成功: {ts_code}, 记录数: {saved_count}")
            return saved_count
            
        except Exception as e:
            logger.error(f"保存股票数据失败: {ts_code}, 错误: {e}")
            raise
    
    def get_stock_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从Parquet文件获取股票数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            
        Returns:
            股票数据DataFrame，如果不存在返回None
        """
        try:
            file_path = self._get_stock_file_path(ts_code)
            
            if not file_path.exists():
                logger.debug(f"股票数据文件不存在: {ts_code}")
                return None
            
            # 读取Parquet文件
            df = pd.read_parquet(file_path)
            
            if df.empty:
                logger.debug(f"股票数据为空: {ts_code}")
                return None
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 转换查询日期格式
            start_date_formatted = pd.to_datetime(start_date, format='%Y%m%d')
            end_date_formatted = pd.to_datetime(end_date, format='%Y%m%d')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date_formatted) & (df['date'] <= end_date_formatted)]
            
            if df.empty:
                logger.debug(f"未找到股票数据: {ts_code} ({start_date} - {end_date})")
                return None
            
            # 设置date为索引
            df = df.set_index('date')
            df = df.sort_index()
            
            # 移除ts_code列（如果存在）
            if 'ts_code' in df.columns:
                df = df.drop(columns=['ts_code'])
            
            logger.debug(f"从Parquet获取股票数据成功: {ts_code}, 记录数: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {ts_code}, 错误: {e}")
            return None
    
    def save_stock_list(self, stock_list: List[Dict[str, str]]):
        """
        保存股票列表到Parquet文件
        
        Args:
            stock_list: 股票列表，每个元素包含ts_code和name
        """
        try:
            if not stock_list:
                logger.warning("股票列表为空")
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(stock_list)
            
            # 保存到Parquet文件
            df.to_parquet(self.stock_list_file, index=False, engine='pyarrow', compression='snappy')
            
            logger.info(f"保存股票列表成功: {len(stock_list)} 条记录")
        except Exception as e:
            logger.error(f"保存股票列表失败: {e}")
            raise
    
    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表"""
        try:
            if not self.stock_list_file.exists():
                logger.debug("股票列表文件不存在")
                return []
            
            # 读取Parquet文件
            df = pd.read_parquet(self.stock_list_file)
            
            if df.empty:
                return []
            
            # 转换为字典列表
            result = df.to_dict('records')
            logger.debug(f"获取股票列表成功: {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_latest_date(self, ts_code: str) -> Optional[str]:
        """
        获取股票的最新数据日期

        Args:
            ts_code: 股票代码

        Returns:
            最新日期 (YYYYMMDD格式)，如果不存在返回None
        """
        try:
            file_path = self._get_stock_file_path(ts_code)
            
            if not file_path.exists():
                return None
            
            # 读取Parquet文件
            df = pd.read_parquet(file_path)
            
            if df.empty:
                return None
            
            # 获取最新日期
            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max()
            
            return latest_date.strftime('%Y%m%d')
            
        except Exception as e:
            logger.error(f"获取最新日期失败: {ts_code}, 错误: {e}")
            return None
    
    def fetch_one(self, query: str, params=None):
        """
        执行查询并返回第一行（兼容MySQL接口）
        注意：Parquet不支持SQL查询，此方法仅用于兼容性
        
        Args:
            query: SQL查询语句（不支持）
            params: 查询参数（不支持）
            
        Returns:
            None（Parquet不支持SQL查询）
        """
        logger.warning("Parquet DAO不支持SQL查询，fetch_one方法不可用")
        return None
    
    def get_all_stock_files(self) -> List[str]:
        """获取所有股票代码列表（从文件系统中）"""
        try:
            stock_files = list(self.stock_data_dir.glob("*.parquet"))
            stock_codes = []
            for file_path in stock_files:
                # 从文件名恢复股票代码
                safe_code = file_path.stem
                ts_code = safe_code.replace('_', '.')
                stock_codes.append(ts_code)
            return stock_codes
        except Exception as e:
            logger.error(f"获取股票文件列表失败: {e}")
            return []
    
    def get_stock_data_range(self, ts_code: str) -> Optional[Dict]:
        """
        获取股票的数据范围
        
        Args:
            ts_code: 股票代码
            
        Returns:
            包含start_date, end_date, total_records的字典，如果不存在返回None
        """
        try:
            file_path = self._get_stock_file_path(ts_code)
            
            if not file_path.exists():
                return None
            
            # 读取Parquet文件
            df = pd.read_parquet(file_path)
            
            if df.empty:
                return None
            
            # 获取日期范围
            df['date'] = pd.to_datetime(df['date'])
            start_date = df['date'].min()
            end_date = df['date'].max()
            total_records = len(df)
            
            # 获取最后更新时间（使用文件修改时间）
            last_update = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_records': total_records,
                'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"获取股票数据范围失败: {ts_code}, 错误: {e}")
            return None
    
    def close(self):
        """关闭资源（Parquet DAO无需关闭连接）"""
        pass
    
    def __del__(self):
        """析构函数"""
        pass


def create_dao():
    """创建Parquet DAO实例"""
    return ParquetDAO()

