"""
SFTP同步服务
用于从远端服务器通过SFTP下载股票parquet数据
"""

import paramiko
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from loguru import logger
from app.core.config import settings
from ..events.data_sync_events import get_data_sync_event_manager, DataSyncEventType

# 绑定日志类型为数据同步
logger = logger.bind(log_type="data_sync")


@dataclass
class SyncResult:
    """同步结果"""
    success: bool
    total_files: int
    synced_files: int
    failed_files: List[str]
    total_size: int
    message: str


class SFTPSyncService:
    """SFTP同步服务"""
    
    def __init__(
        self,
        host: str = "192.168.3.62",
        username: str = "ronghui",
        password: str = "101618",
        remote_list_path: str = "/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_list.parquet",
        remote_data_dir: str = "/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data",
        local_data_dir: str = None
    ):
        """
        初始化SFTP同步服务
        
        Args:
            host: 远端服务器IP
            username: 用户名
            password: 密码
            remote_list_path: 远端股票列表文件路径
            remote_data_dir: 远端数据文件目录
            local_data_dir: 本地数据存储目录
        """
        self.host = host
        self.username = username
        self.password = password
        self.remote_list_path = remote_list_path
        self.remote_data_dir = remote_data_dir
        
        # 本地数据目录，默认使用配置中的路径
        if local_data_dir is None:
            # 使用ParquetManager的默认路径结构
            self.local_data_dir = Path(settings.DATA_ROOT_PATH) / "parquet"
        else:
            self.local_data_dir = Path(local_data_dir)
        
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取事件管理器
        self.event_manager = get_data_sync_event_manager()
        
        logger.info(f"SFTP同步服务初始化: {host}, 本地目录: {self.local_data_dir}")
    
    def _connect_sftp(self) -> Tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        """
        建立SFTP连接
        
        Returns:
            (SSHClient, SFTPClient) 元组
        """
        logger.info(f"开始连接SFTP服务器: {self.host}, 用户: {self.username}")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            logger.debug(f"正在建立SSH连接...")
            ssh.connect(
                hostname=self.host,
                username=self.username,
                password=self.password,
                timeout=30
            )
            logger.debug(f"SSH连接成功，正在打开SFTP通道...")
            sftp = ssh.open_sftp()
            logger.info(f"成功连接到SFTP服务器: {self.host}")
            return ssh, sftp
        except Exception as e:
            logger.error(f"连接SFTP服务器失败: {e}", exc_info=True)
            raise
    
    def _disconnect_sftp(self, ssh: paramiko.SSHClient, sftp: paramiko.SFTPClient):
        """关闭SFTP连接"""
        try:
            sftp.close()
            ssh.close()
            logger.debug("SFTP连接已关闭")
        except Exception as e:
            logger.warning(f"关闭SFTP连接时出错: {e}")
    
    def get_remote_stock_list(self) -> List[str]:
        """
        从远端服务器获取股票列表
        
        Returns:
            股票代码列表
        """
        ssh = None
        sftp = None
        temp_file = None
        
        try:
            logger.info(f"开始获取远端股票列表，路径: {self.remote_list_path}")
            ssh, sftp = self._connect_sftp()
            
            # 检查远端文件是否存在
            try:
                file_stat = sftp.stat(self.remote_list_path)
                logger.info(f"远端股票列表文件存在，大小: {file_stat.st_size} 字节")
            except FileNotFoundError:
                logger.error(f"远端股票列表文件不存在: {self.remote_list_path}")
                raise
            
            # 创建临时文件
            temp_file = Path("/tmp") / f"stock_list_{datetime.now().timestamp()}.parquet"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"临时文件路径: {temp_file}")
            
            # 下载股票列表文件
            logger.info(f"正在下载股票列表文件...")
            start_time = datetime.now()
            sftp.get(self.remote_list_path, str(temp_file))
            download_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"股票列表文件下载完成，耗时: {download_time:.2f}秒")
            
            # 读取parquet文件获取股票列表
            logger.debug(f"正在读取parquet文件...")
            df = pd.read_parquet(temp_file)
            logger.debug(f"parquet文件读取完成，行数: {len(df)}, 列: {list(df.columns)}")
            
            # 假设股票代码在'ts_code'列中
            if 'ts_code' in df.columns:
                stock_codes = df['ts_code'].unique().tolist()
            else:
                # 如果没有ts_code列，尝试第一列
                logger.warning(f"未找到ts_code列，使用第一列作为股票代码")
                stock_codes = df.iloc[:, 0].unique().tolist()
            
            logger.info(f"成功获取股票列表: {len(stock_codes)} 只股票")
            return stock_codes
            
        except Exception as e:
            logger.error(f"获取远端股票列表失败: {e}", exc_info=True)
            raise
        finally:
            if ssh and sftp:
                self._disconnect_sftp(ssh, sftp)
            if temp_file and temp_file.exists():
                temp_file.unlink()
                logger.debug(f"已清理临时文件: {temp_file}")
    
    def sync_stock_file(self, stock_code: str, sftp: paramiko.SFTPClient) -> Tuple[bool, int]:
        """
        同步单个股票文件
        
        Args:
            stock_code: 股票代码
            sftp: SFTP客户端
            
        Returns:
            (是否成功, 文件大小)
        """
        try:
            # 远端文件路径：尝试多种可能的文件名格式
            safe_code = stock_code.replace('.', '_')
            possible_files = [
                f"{self.remote_data_dir}/{safe_code}.parquet",
                f"{self.remote_data_dir}/{stock_code}.parquet",
                f"{self.remote_data_dir}/{stock_code.replace('.', '_')}.parquet",
            ]
            
            remote_file = None
            for file_path in possible_files:
                try:
                    sftp.stat(file_path)
                    remote_file = file_path
                    break
                except FileNotFoundError:
                    continue
            
            if remote_file is None:
                logger.warning(f"远端文件不存在: {stock_code} (尝试了 {possible_files})")
                return False, 0
            
            # 本地文件路径：使用stock_data子目录，保持与远端结构一致
            local_file = self.local_data_dir / "stock_data" / f"{safe_code}.parquet"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            logger.info(f"正在同步: {stock_code} ({remote_file} -> {local_file})")
            sftp.get(remote_file, str(local_file))
            
            # 获取文件大小
            file_size = local_file.stat().st_size
            
            logger.info(f"成功同步: {stock_code}, 大小: {file_size} 字节")
            return True, file_size
            
        except Exception as e:
            logger.error(f"同步股票文件失败 {stock_code}: {e}")
            return False, 0
    
    def sync_all_stocks(self, stock_codes: Optional[List[str]] = None) -> SyncResult:
        """
        同步所有股票数据
        
        Args:
            stock_codes: 要同步的股票代码列表，如果为None则从远端获取列表
            
        Returns:
            同步结果
        """
        import asyncio
        
        sync_start_time = datetime.now()
        ssh = None
        sftp = None
        
        try:
            logger.info("=" * 60)
            logger.info("开始SFTP数据同步任务")
            logger.info("=" * 60)
            
            # 获取股票列表
            if stock_codes is None:
                logger.info("股票代码列表为空，从远端获取股票列表...")
                stock_codes = self.get_remote_stock_list()
            else:
                logger.info(f"使用提供的股票代码列表，共 {len(stock_codes)} 只股票")
            
            if not stock_codes:
                logger.warning("未找到要同步的股票")
                return SyncResult(
                    success=False,
                    total_files=0,
                    synced_files=0,
                    failed_files=[],
                    total_size=0,
                    message="未找到要同步的股票"
                )
            
            # 建立SFTP连接
            logger.info("建立SFTP连接...")
            ssh, sftp = self._connect_sftp()
            logger.info("SFTP连接建立成功")
            
            # 同步每个股票文件
            total_files = len(stock_codes)
            synced_files = 0
            failed_files = []
            total_size = 0
            
            logger.info(f"开始同步 {total_files} 只股票的数据...")
            logger.info(f"本地存储目录: {self.local_data_dir / 'stock_data'}")
            
            for i, stock_code in enumerate(stock_codes, 1):
                try:
                    # 发出同步开始事件
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.event_manager.emit_sync_started(
                                stock_code=stock_code,
                                date_range=(sync_start_time, datetime.now()),
                                sync_type="sftp_sync",
                                metadata={"batch_index": i, "total_files": total_files}
                            )
                        )
                        loop.close()
                    except Exception as e:
                        logger.warning(f"发出同步开始事件失败 {stock_code}: {e}")
                    
                    file_start_time = datetime.now()
                    success, file_size = self.sync_stock_file(stock_code, sftp)
                    file_duration = (datetime.now() - file_start_time).total_seconds()
                    
                    if success:
                        synced_files += 1
                        total_size += file_size
                        logger.debug(f"[{i}/{total_files}] {stock_code}: 成功, 大小: {file_size} 字节, 耗时: {file_duration:.2f}秒")
                        
                        # 发出同步完成事件
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(
                                self.event_manager.emit_sync_completed(
                                    stock_code=stock_code,
                                    date_range=(file_start_time, datetime.now()),
                                    sync_type="sftp_sync",
                                    metadata={
                                        "file_size": file_size,
                                        "duration_seconds": file_duration,
                                        "batch_index": i,
                                        "total_files": total_files
                                    }
                                )
                            )
                            loop.close()
                        except Exception as e:
                            logger.warning(f"发出同步完成事件失败 {stock_code}: {e}")
                    else:
                        failed_files.append(stock_code)
                        logger.warning(f"[{i}/{total_files}] {stock_code}: 失败")
                        
                        # 发出同步失败事件
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(
                                self.event_manager.emit_sync_failed(
                                    stock_code=stock_code,
                                    date_range=(file_start_time, datetime.now()),
                                    sync_type="sftp_sync",
                                    error_message="文件同步失败",
                                    metadata={
                                        "batch_index": i,
                                        "total_files": total_files
                                    }
                                )
                            )
                            loop.close()
                        except Exception as e:
                            logger.warning(f"发出同步失败事件失败 {stock_code}: {e}")
                    
                    # 每10个文件记录一次进度
                    if i % 10 == 0:
                        elapsed = (datetime.now() - sync_start_time).total_seconds()
                        avg_time = elapsed / i
                        remaining = (total_files - i) * avg_time
                        logger.info(f"同步进度: {i}/{total_files} ({synced_files} 成功, {len(failed_files)} 失败) | "
                                  f"已耗时: {elapsed:.1f}秒 | 预计剩余: {remaining:.1f}秒")
                        
                except Exception as e:
                    logger.error(f"同步股票 {stock_code} 时出错: {e}", exc_info=True)
                    failed_files.append(stock_code)
            
            total_duration = (datetime.now() - sync_start_time).total_seconds()
            success = synced_files > 0
            message = f"同步完成: {synced_files}/{total_files} 成功"
            if failed_files:
                message += f", {len(failed_files)} 失败"
            message += f", 总耗时: {total_duration:.1f}秒"
            
            logger.info("=" * 60)
            logger.info(f"同步任务完成: {message}")
            logger.info(f"总文件数: {total_files}, 成功: {synced_files}, 失败: {len(failed_files)}")
            logger.info(f"总数据大小: {total_size / (1024*1024):.2f} MB")
            logger.info(f"总耗时: {total_duration:.1f}秒")
            if failed_files:
                logger.warning(f"失败的股票代码（前10个）: {failed_files[:10]}")
            logger.info("=" * 60)
            
            return SyncResult(
                success=success,
                total_files=total_files,
                synced_files=synced_files,
                failed_files=failed_files,
                total_size=total_size,
                message=message
            )
            
        except Exception as e:
            total_duration = (datetime.now() - sync_start_time).total_seconds()
            logger.error(f"同步过程出错: {e}", exc_info=True)
            logger.error(f"同步失败，总耗时: {total_duration:.1f}秒")
            return SyncResult(
                success=False,
                total_files=0,
                synced_files=0,
                failed_files=[],
                total_size=0,
                message=f"同步失败: {str(e)}"
            )
        finally:
            if ssh and sftp:
                logger.info("正在关闭SFTP连接...")
                self._disconnect_sftp(ssh, sftp)
                logger.info("SFTP连接已关闭")
    
    def sync_selected_stocks(self, stock_codes: List[str]) -> SyncResult:
        """
        同步选定的股票数据
        
        Args:
            stock_codes: 要同步的股票代码列表
            
        Returns:
            同步结果
        """
        return self.sync_all_stocks(stock_codes=stock_codes)

