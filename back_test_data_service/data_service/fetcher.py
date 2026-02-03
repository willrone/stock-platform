"""
数据获取服务
从Tushare获取股票数据
"""
import logging
import tushare as ts
import pandas as pd
import time
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from .config import Config
from .parquet_dao import create_dao

logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取服务"""
    
    def __init__(self):
        """初始化数据获取服务"""
        Config.validate()
        ts.set_token(Config.TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.dao = create_dao()  # 使用parquet_dao替代mysql_dao
        # Tushare API限制：单次最多返回5000条记录
        self.max_limit = 5000
        logger.info("数据获取服务初始化成功")
    
    def _split_date_range(self, start_date: str, end_date: str, days_per_chunk: int = 500) -> List[Tuple[str, str]]:
        """
        将日期范围分段，用于分页获取数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            days_per_chunk: 每段的天数（默认500天，约2年）
            
        Returns:
            日期段列表，每个元素为(start_date, end_date)元组
        """
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        date_ranges = []
        current_start = start_dt
        
        while current_start <= end_dt:
            current_end = min(current_start + timedelta(days=days_per_chunk), end_dt)
            date_ranges.append((
                current_start.strftime('%Y%m%d'),
                current_end.strftime('%Y%m%d')
            ))
            current_start = current_end + timedelta(days=1)
        
        return date_ranges
    
    def _validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        验证日期范围格式和有效性
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            
        Returns:
            是否有效
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            if start_dt > end_dt:
                logger.error(f"日期范围无效: 开始日期 {start_date} 大于结束日期 {end_date}")
                return False
            
            # 检查日期是否在未来
            if end_dt > datetime.now():
                logger.warning(f"结束日期 {end_date} 在未来，将使用当前日期")
                return False
            
            return True
        except ValueError as e:
            logger.error(f"日期格式错误: {e}")
            return False
    
    def _fetch_single_chunk(self, ts_code: str, start_date: str, end_date: str, retry_count: int = 0) -> Optional[pd.DataFrame]:
        """
        获取单个日期段的数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            retry_count: 重试次数
            
        Returns:
            股票数据DataFrame，如果获取失败返回None
        """
        max_retries = 3
        start_time = time.time()
        
        while retry_count < max_retries:
            try:
                # 使用 pro.daily() 接口替代 pro_bar()（兼容新版 pandas）
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 检查数据是否为空
                if df is None or df.empty:
                    # 非交易日或无数据是正常情况，记录为INFO级别
                    logger.info(f"未获取到数据（可能是非交易日）: {ts_code} ({start_date} - {end_date})")
                    return None
                
                # 保留需要的列
                df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol']]
                df.rename(columns={
                    'trade_date': 'date',
                    'vol': 'volume'
                }, inplace=True)
                
                # 日期处理和排序
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()  # 按日期升序排列
                
                logger.debug(f"获取数据段成功: {ts_code} ({start_date} - {end_date}), 记录数: {len(df)}")
                return df
                
            except Exception as e:
                retry_count += 1
                error_type = type(e).__name__
                error_msg = str(e)
                
                # 区分不同类型的异常
                if 'token' in error_msg.lower() or 'auth' in error_msg.lower() or '401' in error_msg:
                    # Token过期或认证失败
                    logger.error(
                        f"Tushare API认证失败: {ts_code} ({start_date} - {end_date}), "
                        f"错误类型: {error_type}, 错误信息: {error_msg}",
                        exc_info=True
                    )
                    return None  # 认证失败不需要重试
                elif 'network' in error_msg.lower() or 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                    # 网络错误，可以重试
                    if retry_count >= max_retries:
                        logger.error(
                            f"网络错误，已达到最大重试次数: {ts_code} ({start_date} - {end_date}), "
                            f"错误类型: {error_type}, 错误信息: {error_msg}",
                            exc_info=True
                        )
                        return None
                    logger.warning(
                        f"网络错误，重试中 ({retry_count}/{max_retries}): {ts_code} ({start_date} - {end_date}), "
                        f"错误: {error_msg}"
                    )
                elif 'limit' in error_msg.lower() or 'rate' in error_msg.lower() or '429' in error_msg:
                    # API调用频率限制
                    if retry_count >= max_retries:
                        logger.error(
                            f"API调用频率限制，已达到最大重试次数: {ts_code} ({start_date} - {end_date}), "
                            f"错误类型: {error_type}, 错误信息: {error_msg}",
                            exc_info=True
                        )
                        return None
                    wait_time = retry_count * 2  # 递增等待时间
                    logger.warning(
                        f"API调用频率限制，等待 {wait_time} 秒后重试 ({retry_count}/{max_retries}): "
                        f"{ts_code} ({start_date} - {end_date}), 错误: {error_msg}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # 其他未知错误
                    if retry_count >= max_retries:
                        logger.error(
                            f"获取数据段失败: {ts_code} ({start_date} - {end_date}), "
                            f"错误类型: {error_type}, 错误信息: {error_msg}",
                            exc_info=True
                        )
                        return None
                    logger.warning(
                        f"获取失败，重试中 ({retry_count}/{max_retries}): {ts_code} ({start_date} - {end_date}), "
                        f"错误类型: {error_type}, 错误信息: {error_msg}"
                    )
                
                time.sleep(1)
        
        return None
    
    def fetch_stock_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从Tushare获取股票数据（支持分页获取）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            
        Returns:
            股票数据DataFrame，如果获取失败返回None
        """
        ts_code = str(ts_code) if ts_code is not None else None
        if not ts_code:
            raise ValueError("股票代码不能为空")
        
        # 验证日期范围
        if not self._validate_date_range(start_date, end_date):
            raise ValueError(f"日期范围无效: {start_date} - {end_date}")
        
        logger.info(f"开始获取股票数据: {ts_code} ({start_date} - {end_date})")
        overall_start_time = time.time()
        
        # 先尝试一次性获取所有数据
        df = self._fetch_single_chunk(ts_code, start_date, end_date)
        
        if df is not None:
            # 检查是否达到了limit限制
            if len(df) >= self.max_limit:
                logger.warning(f"返回数据达到limit限制 ({len(df)} >= {self.max_limit})，需要分页获取")
                # 使用分页逻辑
                date_ranges = self._split_date_range(start_date, end_date, days_per_chunk=500)
                logger.info(f"将日期范围分为 {len(date_ranges)} 段进行获取")
                
                all_dataframes = []
                for i, (chunk_start, chunk_end) in enumerate(date_ranges, 1):
                    logger.info(f"获取第 {i}/{len(date_ranges)} 段: {chunk_start} - {chunk_end}")
                    chunk_df = self._fetch_single_chunk(ts_code, chunk_start, chunk_end)
                    if chunk_df is not None and not chunk_df.empty:
                        all_dataframes.append(chunk_df)
                    time.sleep(0.2)  # 避免API调用过快
                
                if not all_dataframes:
                    # 检查日期范围是否很小（可能是单个非交易日）
                    start_dt = datetime.strptime(start_date, '%Y%m%d')
                    end_dt = datetime.strptime(end_date, '%Y%m%d')
                    days_diff = (end_dt - start_dt).days + 1
                    
                    if days_diff <= 3:
                        # 日期范围很小，可能是非交易日，记录为INFO
                        logger.info(
                            f"日期范围内无数据（可能是非交易日）: {ts_code} "
                            f"({start_date} - {end_date}), 日期范围: {days_diff} 天"
                        )
                    else:
                        # 日期范围较大但无数据，可能是API调用失败
                        logger.error(f"分页获取失败: {ts_code}，所有数据段都获取失败")
                    return None
                
                # 合并所有数据段
                df = pd.concat(all_dataframes, axis=0)
                df = df[~df.index.duplicated(keep='first')]  # 去重
                df = df.sort_index()  # 重新排序
                logger.info(f"分页获取完成: {ts_code}，合并后总记录数: {len(df)}")
            else:
                logger.info(f"单次获取成功: {ts_code}，记录数: {len(df)}")
        else:
            # 单次获取失败，可能是非交易日或API调用失败，尝试分页获取
            logger.info(f"单次获取未返回数据，尝试分页获取: {ts_code}")
            date_ranges = self._split_date_range(start_date, end_date, days_per_chunk=500)
            logger.info(f"将日期范围分为 {len(date_ranges)} 段进行获取")
            
            all_dataframes = []
            failed_chunks = 0
            for i, (chunk_start, chunk_end) in enumerate(date_ranges, 1):
                logger.info(f"获取第 {i}/{len(date_ranges)} 段: {chunk_start} - {chunk_end}")
                chunk_df = self._fetch_single_chunk(ts_code, chunk_start, chunk_end)
                if chunk_df is not None and not chunk_df.empty:
                    all_dataframes.append(chunk_df)
                else:
                    failed_chunks += 1
                time.sleep(0.2)  # 避免API调用过快
            
            if not all_dataframes:
                # 检查日期范围是否很小（可能是单个非交易日）
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                days_diff = (end_dt - start_dt).days + 1
                
                if days_diff <= 3:
                    # 日期范围很小，可能是非交易日，记录为INFO
                    logger.info(
                        f"日期范围内无数据（可能是非交易日）: {ts_code} "
                        f"({start_date} - {end_date}), 日期范围: {days_diff} 天"
                    )
                else:
                    # 日期范围较大但无数据，可能是API调用失败
                    logger.error(
                        f"获取股票数据失败: {ts_code}，所有数据段都获取失败 "
                        f"({start_date} - {end_date}), 失败段数: {failed_chunks}/{len(date_ranges)}"
                    )
                return None
            
            # 合并所有数据段
            df = pd.concat(all_dataframes, axis=0)
            df = df[~df.index.duplicated(keep='first')]  # 去重
            df = df.sort_index()  # 重新排序
            
            # 如果部分数据段失败，记录警告
            if failed_chunks > 0:
                logger.warning(
                    f"分页获取部分成功: {ts_code}，成功 {len(all_dataframes)} 段，"
                    f"失败 {failed_chunks} 段，合并后总记录数: {len(df)}"
                )
            else:
                logger.info(f"分页获取完成: {ts_code}，合并后总记录数: {len(df)}")
        
        # 验证数据量
        if df is not None and not df.empty:
            # 计算预期的交易日数量（粗略估算：约250个交易日/年）
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            years = (end_dt - start_dt).days / 365.25
            expected_records = int(years * 250)  # 粗略估算
            
            actual_records = len(df)
            
            # 如果实际记录数远少于预期，记录警告
            if actual_records < expected_records * 0.3:  # 少于预期的30%
                logger.warning(
                    f"数据量异常少: {ts_code} "
                    f"预期约 {expected_records} 条，实际 {actual_records} 条 "
                    f"({start_date} - {end_date})"
                )
            
            # 如果只有很少的记录（如22条），记录严重警告
            if actual_records < 50:
                logger.error(
                    f"数据量严重不足: {ts_code} 只有 {actual_records} 条记录 "
                    f"({start_date} - {end_date})，可能存在数据获取问题"
                )
            
            # 验证数据连续性
            self._check_data_continuity(df, ts_code)
            
            logger.info(
                f"获取股票数据成功: {ts_code}, "
                f"记录数: {actual_records}, "
                f"日期范围: {df.index.min()} 至 {df.index.max()}, "
                f"耗时: {time.time() - overall_start_time:.2f}秒"
            )
        
        return df
    
    def fetch_and_save_stock_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[bool]:
        """
        获取股票数据并保存到Parquet文件
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            
        Returns:
            True: 成功获取并保存数据
            False: 真正的失败（API调用失败等）
            None: 非交易日或无数据（正常情况，不算失败）
        """
        try:
            # 直接从Tushare获取数据
            df = self.fetch_stock_data(ts_code, start_date, end_date)
            
            if df is None or df.empty:
                # 检查日期范围，判断是否为非交易日
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                days_diff = (end_dt - start_dt).days + 1
                
                # 如果日期范围很小（<=3天），很可能是非交易日，返回None表示正常情况
                if days_diff <= 3:
                    logger.info(f"未获取到数据（可能是非交易日）: {ts_code} ({start_date} - {end_date})")
                    return None  # None表示非交易日，不算失败
                else:
                    # 日期范围较大但无数据，可能是真正的失败
                    logger.warning(f"日期范围内无数据: {ts_code} ({start_date} - {end_date}), 日期范围: {days_diff} 天")
                    return False  # False表示失败
            
            # 保存到Parquet文件
            saved_count = self.dao.save_stock_data(ts_code, df)
            print(f"保存股票数据: {ts_code}，保存了 {saved_count} 条记录")
            logger.info(f"保存股票数据成功: {ts_code}，保存了 {saved_count} 条记录")
            return True
            
        except Exception as e:
            logger.error(
                f"获取并保存股票数据失败: {ts_code}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}",
                exc_info=True
            )
            return False  # 异常情况，返回False表示失败
    
    def fetch_stock_list(self) -> list:
        """
        从Tushare获取股票列表
        
        Returns:
            股票列表，每个元素包含ts_code和name
        """
        try:
            logger.info("开始获取股票列表")
            stock_list = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
            
            if stock_list is None or stock_list.empty:
                logger.warning("获取股票列表为空")
                return []
            
            result = [{
                'ts_code': row['ts_code'],
                'name': row['name']
            } for _, row in stock_list.iterrows()]
            
            logger.info(f"获取股票列表成功: {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(
                f"获取股票列表失败: 错误类型: {type(e).__name__}, 错误信息: {str(e)}",
                exc_info=True
            )
            return []
    
    def fetch_and_save_stock_list(self) -> bool:
        """
        获取股票列表并保存到MySQL
        
        Returns:
            是否成功
        """
        try:
            stock_list = self.fetch_stock_list()
            if not stock_list:
                return False
            
            self.dao.save_stock_list(stock_list)
            return True
            
        except Exception as e:
            logger.error(
                f"获取并保存股票列表失败: 错误类型: {type(e).__name__}, 错误信息: {str(e)}",
                exc_info=True
            )
            return False
    
    def update_stock_data_incremental(self, ts_code: str) -> bool:
        """
        增量更新股票数据（从最新日期到当前）
        
        Args:
            ts_code: 股票代码
            
        Returns:
            是否成功（True=成功，False=失败，None会被视为成功，因为非交易日不算失败）
        """
        try:
            from datetime import datetime, timedelta
            
            # 获取最新日期
            latest_date = self.dao.get_latest_date(ts_code)
            
            if latest_date:
                # 从最新日期的下一天开始
                latest_dt = datetime.strptime(latest_date, '%Y%m%d')
                start_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
            else:
                # 如果没有数据，使用默认开始日期
                start_date = Config.DEFAULT_START_DATE
            
            # 结束日期为今天
            end_date = datetime.now().strftime('%Y%m%d')
            
            if start_date > end_date:
                logger.info(f"股票数据已是最新: {ts_code}")
                return True
            
            logger.info(f"增量更新股票数据: {ts_code} ({start_date} - {end_date})")
            result = self.fetch_and_save_stock_data(ts_code, start_date, end_date)
            
            # result可能是True/False/None
            # True: 成功获取并保存数据
            # False: 真正的失败
            # None: 非交易日或无数据（正常情况，视为成功）
            if result is None:
                # 非交易日或无数据，不算失败
                logger.debug(f"增量更新完成（非交易日或无数据）: {ts_code}")
                return True
            
            return result  # True或False
            
        except Exception as e:
            logger.error(
                f"增量更新股票数据失败: {ts_code}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}",
                exc_info=True
            )
            return False
    
    def _check_data_continuity(self, df: pd.DataFrame, ts_code: str):
        """验证数据的连续性"""
        # 检查是否有缺失值
        if df.isnull().any().any():
            missing_dates = df[df.isnull().any(axis=1)].index
            logger.warning(f"警告：发现缺失值在 {ts_code}: {missing_dates.tolist()}")
        
        # 检查价格跳跃（超过10%的日变化）
        price_changes = df['close'].pct_change().abs()
        large_jumps = price_changes[price_changes > 0.1]
        if not large_jumps.empty:
            logger.warning(f"警告：发现价格跳跃日期 {ts_code}: {large_jumps.index.tolist()}")
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'dao'):
            self.dao.close()
