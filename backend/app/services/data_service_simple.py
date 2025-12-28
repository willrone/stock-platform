"""
简化的股票数据服务（不依赖复杂的第三方库）
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.models.stock_simple import (
    DataServiceStatus,
    DataSyncRequest,
    DataSyncResponse,
    StockData,
)


class ServiceStatusTracker:
    """服务状态跟踪器"""
    
    def __init__(self):
        self.status_history = []
        self.consecutive_failures = 0
        self.last_success_time = None
        self.last_failure_time = None
    
    def record_success(self, response_time_ms: float):
        """记录成功的服务调用"""
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        
        status = DataServiceStatus(
            service_url="",
            is_available=True,
            last_check=self.last_success_time,
            response_time_ms=response_time_ms
        )
        self.status_history.append(status)
        
        # 保留最近100条记录
        if len(self.status_history) > 100:
            self.status_history.pop(0)
    
    def record_failure(self, error_message: str, response_time_ms: float = None):
        """记录失败的服务调用"""
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        
        status = DataServiceStatus(
            service_url="",
            is_available=False,
            last_check=self.last_failure_time,
            response_time_ms=response_time_ms,
            error_message=error_message
        )
        self.status_history.append(status)
        
        # 保留最近100条记录
        if len(self.status_history) > 100:
            self.status_history.pop(0)
    
    def is_service_degraded(self) -> bool:
        """判断服务是否处于降级状态"""
        # 连续失败3次以上认为服务降级
        return self.consecutive_failures >= 3
    
    def get_recent_status(self, count: int = 10) -> List[DataServiceStatus]:
        """获取最近的服务状态记录"""
        return self.status_history[-count:] if self.status_history else []


class SimpleStockDataService:
    """简化的股票数据服务"""
    
    def __init__(self, data_path: str = "./data", remote_url: str = "http://192.168.3.62"):
        self.remote_url = remote_url
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # 创建简单的数据存储目录
        self.stocks_path = self.data_path / "stocks"
        self.stocks_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化服务状态跟踪器
        self.status_tracker = ServiceStatusTracker()
    
    def get_local_data_path(self, stock_code: str) -> Path:
        """获取本地数据文件路径"""
        return self.stocks_path / f"{stock_code}.json"
    
    def check_local_data_exists(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> bool:
        """检查本地数据是否存在且覆盖指定日期范围"""
        data_file = self.get_local_data_path(stock_code)
        
        if not data_file.exists():
            return False
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return False
            
            # 检查日期范围
            dates = []
            for item in data:
                try:
                    # 处理可能的日期格式
                    date_str = item['date']
                    if 'T' in date_str:
                        # ISO格式: 2023-01-01T00:00:00
                        date_obj = datetime.fromisoformat(date_str.replace('T', ' ').split('.')[0])
                    else:
                        # 简单格式: 2023-01-01
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date_obj.date())  # 只比较日期部分
                except (ValueError, KeyError):
                    continue
            
            if not dates:
                return False
            
            data_start = min(dates)
            data_end = max(dates)
            
            # 检查是否覆盖所需范围（只比较日期部分）
            return data_start <= start_date.date() and data_end >= end_date.date()
        
        except Exception as e:
            print(f"检查本地数据时出错: {e}")
            return False
    
    def get_local_data_date_range(self, stock_code: str) -> Optional[tuple[datetime, datetime]]:
        """获取本地数据的日期范围"""
        data_file = self.get_local_data_path(stock_code)
        
        if not data_file.exists():
            return None
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return None
            
            # 获取所有日期
            dates = []
            for item in data:
                try:
                    date_str = item['date']
                    if 'T' in date_str:
                        date_obj = datetime.fromisoformat(date_str.replace('T', ' ').split('.')[0])
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date_obj)
                except (ValueError, KeyError):
                    continue
            
            if not dates:
                return None
            
            return min(dates), max(dates)
        
        except Exception as e:
            print(f"获取本地数据日期范围时出错: {e}")
            return None
    
    def identify_missing_date_ranges(
        self, 
        stock_code: str, 
        requested_start: datetime, 
        requested_end: datetime
    ) -> List[tuple[datetime, datetime]]:
        """识别缺失的数据时间段"""
        local_range = self.get_local_data_date_range(stock_code)
        
        if local_range is None:
            # 没有本地数据，整个范围都需要获取
            return [(requested_start, requested_end)]
        
        local_start, local_end = local_range
        missing_ranges = []
        
        # 检查请求开始日期之前的缺失数据
        if requested_start < local_start:
            missing_ranges.append((requested_start, min(requested_end, local_start - timedelta(days=1))))
        
        # 检查请求结束日期之后的缺失数据
        if requested_end > local_end:
            missing_ranges.append((max(requested_start, local_end + timedelta(days=1)), requested_end))
        
        # 过滤掉无效的范围
        valid_ranges = []
        for start, end in missing_ranges:
            if start <= end:
                valid_ranges.append((start, end))
        
        return valid_ranges
    
    async def check_remote_service_status(self) -> DataServiceStatus:
        """检查远端数据服务状态"""
        start_time = datetime.now()
        
        try:
            # 模拟网络检查（实际应用中使用httpx或aiohttp）
            import socket
            
            # 解析URL获取主机和端口
            host = self.remote_url.replace('http://', '').replace('https://', '')
            if ':' in host:
                host, port = host.split(':')
                port = int(port)
            else:
                port = 80
            
            # 简单的连接测试
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            try:
                # 尝试解析主机名
                socket.gethostbyname(host)
                
                # 尝试连接
                result = sock.connect_ex((host, port))
                sock.close()
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if result == 0:
                    # 记录成功
                    self.status_tracker.record_success(response_time)
                    
                    return DataServiceStatus(
                        service_url=self.remote_url,
                        is_available=True,
                        last_check=start_time,
                        response_time_ms=response_time
                    )
                else:
                    # 记录失败
                    error_msg = f"连接失败，错误代码: {result}"
                    self.status_tracker.record_failure(error_msg, response_time)
                    
                    return DataServiceStatus(
                        service_url=self.remote_url,
                        is_available=False,
                        last_check=start_time,
                        response_time_ms=response_time,
                        error_message=error_msg
                    )
            
            except socket.gaierror as e:
                # DNS解析失败
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                error_msg = f"DNS解析失败: {str(e)}"
                self.status_tracker.record_failure(error_msg, response_time)
                
                return DataServiceStatus(
                    service_url=self.remote_url,
                    is_available=False,
                    last_check=start_time,
                    response_time_ms=response_time,
                    error_message=error_msg
                )
            
            finally:
                try:
                    sock.close()
                except:
                    pass
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = str(e)
            
            # 记录异常
            self.status_tracker.record_failure(error_msg, response_time)
            
            return DataServiceStatus(
                service_url=self.remote_url,
                is_available=False,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=error_msg
            )
    
    def generate_mock_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """生成模拟股票数据"""
        data = []
        current_date = start_date
        base_price = 100.0  # 基础价格
        
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                # 简单的随机价格生成
                import random
                
                change_pct = random.uniform(-0.05, 0.05)  # ±5%的变化
                open_price = base_price * (1 + change_pct)
                
                high_change = random.uniform(0, 0.03)  # 0-3%的上涨
                low_change = random.uniform(-0.03, 0)  # 0-3%的下跌
                
                high_price = open_price * (1 + high_change)
                low_price = open_price * (1 + low_change)
                
                close_change = random.uniform(-0.02, 0.02)  # ±2%的变化
                close_price = open_price * (1 + close_change)
                
                # 确保价格逻辑正确
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                volume = random.randint(1000000, 10000000)  # 100万到1000万
                
                data.append({
                    'stock_code': stock_code,
                    'date': current_date.isoformat(),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'adj_close': round(close_price, 2)
                })
                
                # 更新基础价格
                base_price = close_price
            
            current_date += timedelta(days=1)
        
        return data
    
    async def fetch_remote_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """从远端服务获取股票数据（增强错误处理）"""
        try:
            # 检查服务是否处于降级状态
            if self.status_tracker.is_service_degraded():
                print(f"服务处于降级状态，跳过远端请求: {stock_code}")
                return None
            
            # 检查远端服务状态
            status = await self.check_remote_service_status()
            
            if not status.is_available:
                print(f"远端服务不可用: {status.error_message}")
                return None
            
            # 模拟网络延迟
            await asyncio.sleep(0.1)
            
            # 生成模拟数据（实际应用中这里会调用真实的API）
            data = self.generate_mock_data(stock_code, start_date, end_date)
            
            # 记录成功获取数据
            print(f"从远端获取数据成功: {stock_code}, {len(data)} 条记录")
            return data
        
        except asyncio.TimeoutError:
            error_msg = f"请求超时: {stock_code}"
            print(error_msg)
            self.status_tracker.record_failure(error_msg)
            return None
        
        except ConnectionError as e:
            error_msg = f"连接错误: {stock_code}, {str(e)}"
            print(error_msg)
            self.status_tracker.record_failure(error_msg)
            return None
        
        except Exception as e:
            error_msg = f"从远端获取数据失败 {stock_code}: {e}"
            print(error_msg)
            self.status_tracker.record_failure(error_msg)
            return None
    
    def save_to_local(self, data: List[Dict[str, Any]], stock_code: str, merge_with_existing: bool = True) -> bool:
        """将数据保存到本地文件，支持增量更新"""
        try:
            if not data:
                print(f"尝试保存空数据: {stock_code}")
                return False
            
            data_file = self.get_local_data_path(stock_code)
            
            if merge_with_existing and data_file.exists():
                # 合并现有数据
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    # 创建日期到数据的映射，避免重复
                    data_by_date = {}
                    
                    # 先添加现有数据
                    for item in existing_data:
                        date_key = item['date']
                        data_by_date[date_key] = item
                    
                    # 添加新数据（会覆盖相同日期的数据）
                    for item in data:
                        date_key = item['date']
                        data_by_date[date_key] = item
                    
                    # 按日期排序
                    merged_data = list(data_by_date.values())
                    merged_data.sort(key=lambda x: x['date'])
                    
                    with open(data_file, 'w', encoding='utf-8') as f:
                        json.dump(merged_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"增量保存本地数据成功: {data_file}, 新增 {len(data)} 条记录，总计 {len(merged_data)} 条记录")
                    return True
                
                except Exception as e:
                    print(f"合并数据时出错: {e}，将覆盖现有文件")
                    # 如果合并失败，直接覆盖
            
            # 直接保存新数据
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"保存本地数据成功: {data_file}, {len(data)} 条记录")
            return True
        
        except Exception as e:
            print(f"保存本地数据失败 {stock_code}: {e}")
            return False
    
    def load_from_local(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """从本地文件加载数据"""
        try:
            data_file = self.get_local_data_path(stock_code)
            
            if not data_file.exists():
                return None
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return None
            
            # 过滤日期范围
            filtered_data = []
            for item in data:
                item_date = datetime.fromisoformat(item['date'])
                if start_date <= item_date <= end_date:
                    filtered_data.append(item)
            
            print(f"从本地加载数据成功: {stock_code}, {len(filtered_data)} 条记录")
            return filtered_data
        
        except Exception as e:
            print(f"从本地加载数据失败 {stock_code}: {e}")
            return None
    
    async def get_stock_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime,
        force_remote: bool = False,
        enable_incremental: bool = True
    ) -> Optional[List[StockData]]:
        """
        获取股票数据（支持增量更新）
        """
        print(f"获取股票数据: {stock_code}, {start_date.date()} - {end_date.date()}")
        
        if not force_remote and enable_incremental:
            # 检查是否需要增量更新
            missing_ranges = self.identify_missing_date_ranges(stock_code, start_date, end_date)
            
            if not missing_ranges:
                # 本地数据完整，直接加载
                print(f"本地数据完整，直接加载: {stock_code}")
                data = self.load_from_local(stock_code, start_date, end_date)
                if data:
                    return self._dict_list_to_stock_data(data)
            else:
                print(f"发现缺失数据段: {len(missing_ranges)} 个范围")
                
                # 获取缺失的数据段
                all_new_data = []
                for missing_start, missing_end in missing_ranges:
                    print(f"获取缺失数据: {missing_start.date()} - {missing_end.date()}")
                    
                    new_data = await self.fetch_remote_data(stock_code, missing_start, missing_end)
                    if new_data:
                        all_new_data.extend(new_data)
                
                if all_new_data:
                    # 保存新数据（增量合并）
                    if self.save_to_local(all_new_data, stock_code, merge_with_existing=True):
                        print(f"增量数据保存成功: {stock_code}")
                    
                    # 加载完整数据返回
                    complete_data = self.load_from_local(stock_code, start_date, end_date)
                    if complete_data:
                        return self._dict_list_to_stock_data(complete_data)
                else:
                    print(f"获取缺失数据失败，尝试使用现有本地数据: {stock_code}")
                    # 尝试使用现有本地数据
                    data = self.load_from_local(stock_code, start_date, end_date)
                    if data:
                        return self._dict_list_to_stock_data(data)
        
        # 原有的逻辑：检查本地数据是否存在
        if not force_remote:
            if self.check_local_data_exists(stock_code, start_date, end_date):
                # 本地数据完整，直接加载
                data = self.load_from_local(stock_code, start_date, end_date)
                if data:
                    return self._dict_list_to_stock_data(data)
        
        # 本地数据不完整或强制从远端获取
        print(f"从远端获取完整数据: {stock_code}")
        
        # 从远端获取数据
        data = await self.fetch_remote_data(stock_code, start_date, end_date)
        
        if not data:
            print(f"远端获取数据失败，尝试使用本地数据: {stock_code}")
            
            # 降级到本地数据
            data = self.load_from_local(stock_code, start_date, end_date)
            if data:
                return self._dict_list_to_stock_data(data)
            else:
                return None
        
        # 保存到本地
        if self.save_to_local(data, stock_code, merge_with_existing=False):
            print(f"数据保存成功: {stock_code}")
        else:
            print(f"数据保存失败: {stock_code}")
        
        return self._dict_list_to_stock_data(data)
    
    def _dict_list_to_stock_data(self, data: List[Dict[str, Any]]) -> List[StockData]:
        """将字典列表转换为StockData列表"""
        stock_data_list = []
        
        for item in data:
            stock_data = StockData(
                stock_code=item['stock_code'],
                date=datetime.fromisoformat(item['date']),
                open=float(item['open']),
                high=float(item['high']),
                low=float(item['low']),
                close=float(item['close']),
                volume=int(item['volume']),
                adj_close=float(item.get('adj_close', item['close']))
            )
            stock_data_list.append(stock_data)
        
        return stock_data_list
    
    async def sync_multiple_stocks(
        self, 
        request: DataSyncRequest
    ) -> DataSyncResponse:
        """批量同步多只股票数据"""
        print(f"开始批量同步: {len(request.stock_codes)} 只股票")
        
        # 默认日期范围
        end_date = request.end_date or datetime.now()
        start_date = request.start_date or (end_date - timedelta(days=30))
        
        synced_stocks = []
        failed_stocks = []
        total_records = 0
        
        # 并发控制
        semaphore = asyncio.Semaphore(3)  # 最多3个并发请求
        
        async def sync_single_stock(stock_code: str):
            async with semaphore:
                try:
                    data = await self.get_stock_data(
                        stock_code, 
                        start_date, 
                        end_date,
                        force_remote=request.force_update,
                        enable_incremental=not request.force_update  # 如果强制更新则禁用增量
                    )
                    
                    if data:
                        synced_stocks.append(stock_code)
                        return len(data)
                    else:
                        failed_stocks.append(stock_code)
                        return 0
                
                except Exception as e:
                    print(f"同步股票失败 {stock_code}: {e}")
                    failed_stocks.append(stock_code)
                    return 0
        
        # 并发执行同步任务
        tasks = [sync_single_stock(code) for code in request.stock_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        for result in results:
            if isinstance(result, int):
                total_records += result
        
        success = len(failed_stocks) == 0
        message = f"同步完成: 成功 {len(synced_stocks)}, 失败 {len(failed_stocks)}"
        
        print(message)
        
        return DataSyncResponse(
            success=success,
            synced_stocks=synced_stocks,
            failed_stocks=failed_stocks,
            total_records=total_records,
            message=message
        )
    
    def get_service_status_summary(self) -> Dict[str, Any]:
        """获取服务状态摘要"""
        recent_statuses = self.status_tracker.get_recent_status(10)
        
        if not recent_statuses:
            return {
                "service_url": self.remote_url,
                "current_status": "未知",
                "consecutive_failures": 0,
                "last_success_time": None,
                "last_failure_time": None,
                "is_degraded": False,
                "recent_success_rate": 0.0
            }
        
        # 计算最近的成功率
        success_count = sum(1 for status in recent_statuses if status.is_available)
        success_rate = success_count / len(recent_statuses)
        
        current_status = "正常" if recent_statuses[-1].is_available else "不可用"
        if self.status_tracker.is_service_degraded():
            current_status = "降级"
        
        return {
            "service_url": self.remote_url,
            "current_status": current_status,
            "consecutive_failures": self.status_tracker.consecutive_failures,
            "last_success_time": self.status_tracker.last_success_time,
            "last_failure_time": self.status_tracker.last_failure_time,
            "is_degraded": self.status_tracker.is_service_degraded(),
            "recent_success_rate": success_rate,
            "recent_statuses": [
                {
                    "timestamp": status.last_check,
                    "is_available": status.is_available,
                    "response_time_ms": status.response_time_ms,
                    "error_message": status.error_message
                }
                for status in recent_statuses
            ]
        }
    
    def reset_service_status(self):
        """重置服务状态（用于手动恢复）"""
        self.status_tracker = ServiceStatusTracker()
        print("服务状态已重置")
    
    async def health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        # 检查远端服务
        remote_status = await self.check_remote_service_status()
        
        # 检查本地存储
        local_storage_ok = self.data_path.exists() and self.stocks_path.exists()
        
        # 统计本地文件数量
        local_files_count = len(list(self.stocks_path.glob("*.json")))
        
        return {
            "timestamp": datetime.now(),
            "remote_service": {
                "url": self.remote_url,
                "is_available": remote_status.is_available,
                "response_time_ms": remote_status.response_time_ms,
                "error_message": remote_status.error_message
            },
            "local_storage": {
                "is_available": local_storage_ok,
                "data_path": str(self.data_path),
                "stocks_count": local_files_count
            },
            "service_status": self.get_service_status_summary()
        }