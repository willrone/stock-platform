"""
SFTP同步服务
用于从远端服务器通过SFTP下载股票parquet数据
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

paramiko = cast(Any, import_module("paramiko"))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from loguru import logger as _logger
except ImportError:
    _logger = cast(Any, logging.getLogger(__name__))

from app.core.config import settings

# 绑定日志类型为数据同步
logger = _logger.bind(log_type="data_sync") if hasattr(_logger, "bind") else _logger


def _get_data_sync_event_manager() -> Any:
    """延迟加载事件管理器，不吞掉内部导入错误。"""
    events_module = import_module("..events.data_sync_events", package=__package__)
    return events_module.get_data_sync_event_manager()


def _require_pandas() -> None:
    """在实际读取 parquet 前检查 pandas 可用性。"""
    if pd is None:
        raise ImportError("SFTP 股票列表读取依赖 pandas，请先安装 pandas。")


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
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        remote_list_path: Optional[str] = None,
        remote_data_dir: Optional[str] = None,
        local_data_dir: Optional[str] = None,
        port: Optional[int] = None,
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
        # SECURITY: do not hardcode credentials/paths in code. Use env-config via settings.
        self.enabled = bool(settings.SFTP_SYNC_ENABLED)
        self.host = host or settings.SFTP_HOST
        self.port = port or settings.SFTP_PORT
        self.username = username or settings.SFTP_USERNAME
        self.password = password or settings.SFTP_PASSWORD
        self.remote_list_path = remote_list_path or settings.SFTP_REMOTE_LIST_PATH
        self.remote_data_dir = remote_data_dir or settings.SFTP_REMOTE_DATA_DIR

        if not self.enabled:
            logger.warning(
                "SFTP同步未启用（SFTP_SYNC_ENABLED=false），远端同步接口将不可用"
            )

        # Validate required fields only when enabled
        if self.enabled:
            missing = [
                k
                for k, v in {
                    "SFTP_HOST": self.host,
                    "SFTP_USERNAME": self.username,
                    "SFTP_REMOTE_LIST_PATH": self.remote_list_path,
                    "SFTP_REMOTE_DATA_DIR": self.remote_data_dir,
                }.items()
                if not v
            ]
            if missing:
                raise RuntimeError(
                    f"SFTP同步已启用但配置缺失: {', '.join(missing)}。请在backend/.env中设置对应环境变量。"
                )

        # 本地数据目录，默认使用配置中的路径
        if local_data_dir is None:
            # 使用ParquetManager的默认路径结构
            self.local_data_dir = Path(settings.DATA_ROOT_PATH) / "parquet"
        else:
            self.local_data_dir = Path(local_data_dir)

        self.local_data_dir.mkdir(parents=True, exist_ok=True)

        # 获取事件管理器
        self.event_manager = _get_data_sync_event_manager()

        # 缓存远端文件列表，避免重复查询
        self._remote_files_cache: Optional[Dict[str, str]] = (
            None  # {stock_code: actual_file_path}
        )

        logger.info(
            f"SFTP同步服务初始化: {self.host}:{self.port}, 本地目录: {self.local_data_dir}, enabled={self.enabled}"
        )

    def _connect_sftp(self) -> Tuple[Any, Any]:
        """
        建立SFTP连接

        Returns:
            (SSHClient, SFTPClient) 元组
        """
        if not self.enabled:
            raise RuntimeError(
                "SFTP同步未启用（SFTP_SYNC_ENABLED=false）。如需使用远端同步，请在backend/.env开启并配置SFTP参数。"
            )

        logger.info(f"开始连接SFTP服务器: {self.host}, 用户: {self.username}")
        ssh = cast(Any, paramiko.SSHClient())
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            logger.debug("正在建立SSH连接...")
            connect_kwargs = dict(
                hostname=self.host,
                port=self.port,
                username=self.username,
                timeout=30,
            )
            if self.password:
                connect_kwargs["password"] = self.password
            else:
                # 无密码时使用 SSH key 认证（自动查找 ~/.ssh/id_*）
                connect_kwargs["allow_agent"] = True
                connect_kwargs["look_for_keys"] = True
            ssh.connect(**connect_kwargs)
            logger.debug("SSH连接成功，正在打开SFTP通道...")
            sftp = ssh.open_sftp()
            logger.info(f"成功连接到SFTP服务器: {self.host}")
            return ssh, sftp
        except Exception as e:
            logger.error(f"连接SFTP服务器失败: {e}", exc_info=True)
            raise

    def _disconnect_sftp(self, ssh: Any, sftp: Any) -> Any:
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
            temp_file = (
                Path("/tmp") / f"stock_list_{datetime.now().timestamp()}.parquet"
            )
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"临时文件路径: {temp_file}")

            # 下载股票列表文件
            logger.info("正在下载股票列表文件...")
            start_time = datetime.now()
            sftp.get(self.remote_list_path, str(temp_file))
            download_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"股票列表文件下载完成，耗时: {download_time:.2f}秒")

            # 读取parquet文件获取股票列表
            logger.debug("正在读取parquet文件...")
            _require_pandas()
            # 尝试使用 pyarrow 引擎，如果失败则使用 fastparquet
            try:
                df = pd.read_parquet(temp_file, engine="pyarrow")
            except Exception as e:
                logger.warning(f"使用 pyarrow 引擎读取失败: {e}，尝试使用 fastparquet")
                try:
                    df = pd.read_parquet(temp_file, engine="fastparquet")
                except Exception as e2:
                    logger.error(f"使用 fastparquet 引擎也失败: {e2}")
                    raise
            logger.debug(
                f"parquet文件读取完成，行数: {len(df)}, 列: {list(df.columns)}"
            )

            # 假设股票代码在'ts_code'列中
            if "ts_code" in df.columns:
                stock_codes = df["ts_code"].unique().tolist()
            else:
                # 如果没有ts_code列，尝试第一列
                logger.warning("未找到ts_code列，使用第一列作为股票代码")
                stock_codes = df.iloc[:, 0].unique().tolist()

            logger.info(f"成功获取股票列表: {len(stock_codes)} 只股票")
            return cast(List[str], stock_codes)

        except Exception as e:
            logger.error(f"获取远端股票列表失败: {e}", exc_info=True)
            raise
        finally:
            if ssh and sftp:
                self._disconnect_sftp(ssh, sftp)
            if temp_file and temp_file.exists():
                temp_file.unlink()
                logger.debug(f"已清理临时文件: {temp_file}")

    def _build_remote_files_cache(self, sftp: Any) -> Dict[str, str]:
        """
        构建远端文件缓存，列出所有parquet文件并建立股票代码到文件路径的映射

        Returns:
            {stock_code: actual_file_path} 字典
        """
        if self._remote_files_cache is not None:
            return self._remote_files_cache

        logger.info(f"开始构建远端文件缓存，目录: {self.remote_data_dir}")
        cache = {}

        try:
            # 列出远端目录的所有文件
            files = sftp.listdir(self.remote_data_dir)
            logger.info(f"远端目录共有 {len(files)} 个文件")

            # 过滤出parquet文件并建立映射
            parquet_files = [f for f in files if f.endswith(".parquet")]
            logger.info(f"找到 {len(parquet_files)} 个parquet文件")

            for filename in parquet_files:
                # 尝试从文件名提取股票代码
                # 支持多种格式：600868_SH.parquet, 600868.SH.parquet, 600868_sh.parquet 等
                base_name = filename.replace(".parquet", "")

                # 尝试匹配股票代码格式：数字+市场代码（SH/SZ/BJ）
                # 支持大小写：SH/sH/Sh/sh
                match = re.match(r"^(\d{6})[._]([A-Za-z]{2})$", base_name)
                if match:
                    code, market = match.groups()
                    # 统一转换为大写
                    market = market.upper()
                    stock_code = f"{code}.{market}"
                    full_path = f"{self.remote_data_dir}/{filename}"
                    # 只保存股票代码到路径的映射，避免重复
                    if stock_code not in cache:
                        cache[stock_code] = full_path
                    else:
                        logger.debug(
                            f"股票代码 {stock_code} 已存在，跳过重复映射: {filename}"
                        )
                else:
                    # 如果正则匹配失败，尝试其他格式
                    # 例如：600868_SH -> 600868.SH
                    if "_" in base_name:
                        parts = base_name.split("_")
                        if len(parts) == 2 and parts[1].upper() in ["SH", "SZ", "BJ"]:
                            stock_code = f"{parts[0]}.{parts[1].upper()}"
                            cache[stock_code] = f"{self.remote_data_dir}/{filename}"
                    elif "." in base_name:
                        # 已经是 600868.SH 格式
                        parts = base_name.split(".")
                        if len(parts) == 2 and parts[1].upper() in ["SH", "SZ", "BJ"]:
                            stock_code = f"{parts[0]}.{parts[1].upper()}"
                            cache[stock_code] = f"{self.remote_data_dir}/{filename}"

            logger.info(f"成功构建文件缓存，映射了 {len(cache)} 个股票代码")
            self._remote_files_cache = cache
            return cache

        except Exception as e:
            logger.error(f"构建远端文件缓存失败: {e}", exc_info=True)
            return {}

    def _find_remote_file(self, stock_code: str, sftp: Any) -> Optional[str]:
        """
        查找远端文件路径

        Args:
            stock_code: 股票代码
            sftp: SFTP客户端

        Returns:
            文件路径，如果找不到返回None
        """
        # 先尝试使用缓存
        cache = self._build_remote_files_cache(sftp)
        if stock_code in cache:
            return cache[stock_code]

        # 如果缓存中没有，尝试直接匹配
        safe_code = stock_code.replace(".", "_")
        possible_files = [
            f"{self.remote_data_dir}/{safe_code}.parquet",
            f"{self.remote_data_dir}/{stock_code}.parquet",
            f"{self.remote_data_dir}/{stock_code.replace('.', '_')}.parquet",
        ]

        for file_path in possible_files:
            try:
                sftp.stat(file_path)
                return file_path
            except FileNotFoundError:
                continue
            except Exception:
                continue

        # 如果直接匹配失败，尝试模糊匹配（从缓存中查找）
        # 提取股票代码的数字部分
        code_match = re.match(r"^(\d{6})\.([A-Z]{2})$", stock_code)
        if code_match:
            code, market = code_match.groups()
            # 尝试不同的文件名格式
            possible_basenames = [
                f"{code}_{market}",  # 600868_SH
                f"{code}.{market}",  # 600868.SH
                f"{code}_{market.lower()}",  # 600868_sh
            ]

            # 在缓存的文件路径中查找匹配
            for basename in possible_basenames:
                for cached_code, cached_path in cache.items():
                    # 检查文件路径是否包含这个basename
                    if basename in cached_path or basename in cached_code:
                        logger.debug(
                            f"[{stock_code}] 通过模糊匹配找到文件: {cached_path} (匹配: {basename})"
                        )
                        return cached_path

            # 如果还是找不到，尝试反向查找：从文件名提取股票代码
            for cached_path in cache.values():
                filename = Path(cached_path).name.replace(".parquet", "")
                # 尝试从文件名提取股票代码
                file_match = re.match(r"^(\d{6})[._]([A-Z]{2})$", filename)
                if file_match:
                    file_code, file_market = file_match.groups()
                    if file_code == code and file_market == market:
                        logger.debug(
                            f"[{stock_code}] 通过反向匹配找到文件: {cached_path}"
                        )
                        return cached_path

        return None

    def sync_stock_file(
        self, stock_code: str, sftp: Any, max_retries: int = 3
    ) -> Tuple[bool, int, str]:
        """
        同步单个股票文件（带重试机制）

        Args:
            stock_code: 股票代码
            sftp: SFTP客户端
            max_retries: 最大重试次数

        Returns:
            (是否成功, 文件大小, 错误信息)
        """
        safe_code = stock_code.replace(".", "_")

        # 本地文件路径：使用stock_data子目录，保持与远端结构一致
        local_file = self.local_data_dir / "stock_data" / f"{safe_code}.parquet"
        local_file.parent.mkdir(parents=True, exist_ok=True)

        last_error = None
        for attempt in range(max_retries):
            try:
                # 检查SFTP连接是否有效
                try:
                    sftp.stat(".")
                except Exception as conn_error:
                    error_msg = f"SFTP连接已断开: {conn_error}"
                    logger.warning(
                        f"[{stock_code}] 尝试 {attempt + 1}/{max_retries}: {error_msg}"
                    )
                    if attempt < max_retries - 1:
                        # 重新连接（需要调用者处理）
                        raise ConnectionError(error_msg)
                    last_error = error_msg
                    continue

                # 查找远端文件（使用改进的查找方法）
                remote_file = self._find_remote_file(stock_code, sftp)

                if remote_file is None:
                    error_msg = "远端文件不存在，已尝试多种格式和模糊匹配"
                    logger.warning(f"[{stock_code}] {error_msg}")
                    return False, 0, error_msg

                # 下载文件
                if attempt > 0:
                    logger.info(
                        f"[{stock_code}] 重试 {attempt + 1}/{max_retries}: 正在同步 ({remote_file} -> {local_file})"
                    )
                else:
                    logger.debug(
                        f"[{stock_code}] 正在同步: {remote_file} -> {local_file}"
                    )

                # 原子下载：先下到临时文件，再替换，避免覆盖/半文件
                tmp_file = local_file.with_suffix(local_file.suffix + ".tmp")
                if tmp_file.exists():
                    tmp_file.unlink()

                sftp.get(remote_file, str(tmp_file))
                os.replace(tmp_file, local_file)

                # 验证文件是否下载成功
                if not local_file.exists():
                    error_msg = "下载后本地文件不存在"
                    logger.error(f"[{stock_code}] {error_msg}")
                    if attempt < max_retries - 1:
                        continue
                    return False, 0, error_msg

                # 获取文件大小
                file_size = local_file.stat().st_size
                if file_size == 0:
                    error_msg = "下载的文件大小为0"
                    logger.warning(f"[{stock_code}] {error_msg}")
                    if attempt < max_retries - 1:
                        local_file.unlink()  # 删除空文件
                        continue
                    return False, 0, error_msg

                logger.debug(f"[{stock_code}] 成功同步, 大小: {file_size} 字节")
                return True, file_size, ""

            except (paramiko.SSHException, paramiko.socket.error, OSError) as e:
                error_msg = f"连接错误: {type(e).__name__}: {str(e)}"
                logger.warning(
                    f"[{stock_code}] 尝试 {attempt + 1}/{max_retries}: {error_msg}"
                )
                last_error = error_msg
                if attempt < max_retries - 1:
                    import time

                    time.sleep(1)  # 等待1秒后重试
                    continue
            except Exception as e:
                error_msg = f"未知错误: {type(e).__name__}: {str(e)}"
                logger.error(
                    f"[{stock_code}] 尝试 {attempt + 1}/{max_retries}: {error_msg}",
                    exc_info=True,
                )
                last_error = error_msg
                if attempt < max_retries - 1:
                    import time

                    time.sleep(1)
                    continue

        # 所有重试都失败
        return False, 0, last_error or "未知错误"

    def _log_sync_event_task_result(self, task: Any, failure_message: str) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning(f"{failure_message}: 任务已取消")
        except Exception as error:
            logger.warning(f"{failure_message}: {error}")

    def _emit_sync_event(self, coro: Any, failure_message: str) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            try:
                task = running_loop.create_task(coro)
                task.add_done_callback(
                    lambda completed_task: self._log_sync_event_task_result(
                        completed_task,
                        failure_message,
                    )
                )
            except Exception as error:
                coro.close()
                logger.warning(f"{failure_message}: {error}")
            return

        try:
            asyncio.run(coro)
        except Exception as error:
            logger.warning(f"{failure_message}: {error}")

    def _get_target_stock_codes(self, stock_codes: Optional[List[str]]) -> List[str]:
        if stock_codes is None:
            logger.info("股票代码列表为空，从远端获取股票列表...")
            return self.get_remote_stock_list()

        logger.info(f"使用提供的股票代码列表，共 {len(stock_codes)} 只股票")
        return stock_codes

    def _partition_stock_codes(
        self, stock_codes: List[str], remote_files_cache: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        available_stock_codes = [
            code for code in stock_codes if code in remote_files_cache
        ]
        missing_stock_codes = [
            code for code in stock_codes if code not in remote_files_cache
        ]
        return available_stock_codes, missing_stock_codes

    def _sync_stock_with_reconnect(
        self,
        stock_code: str,
        ssh: Any,
        sftp: Any,
    ) -> Tuple[bool, int, str, Any, Any]:
        success, file_size, error_msg = self.sync_stock_file(stock_code, sftp)
        if success or "连接" not in error_msg.lower():
            return success, file_size, error_msg, ssh, sftp

        logger.warning(f"[{stock_code}] 检测到连接问题，尝试重新连接...")
        try:
            self._disconnect_sftp(ssh, sftp)
            ssh, sftp = self._connect_sftp()
            logger.info(f"[{stock_code}] 重新连接成功，重试同步...")
            success, file_size, error_msg = self.sync_stock_file(stock_code, sftp)
        except Exception as reconnect_error:
            logger.error(f"[{stock_code}] 重新连接失败: {reconnect_error}")
            error_msg = f"重新连接失败: {reconnect_error}"

        return success, file_size, error_msg, ssh, sftp

    def _refresh_connection_if_needed(
        self,
        index: int,
        ssh: Any,
        sftp: Any,
    ) -> Tuple[Any, Any]:
        if index % 100 != 0 or index <= 0:
            return ssh, sftp

        try:
            sftp.stat(".")
        except Exception:
            logger.warning("检测到SFTP连接可能断开，尝试重新连接...")
            try:
                self._disconnect_sftp(ssh, sftp)
                ssh, sftp = self._connect_sftp()
                logger.info("SFTP连接已恢复")
            except Exception as reconnect_error:
                logger.error(f"重新连接失败: {reconnect_error}")

        return ssh, sftp

    def _log_sync_progress(
        self,
        index: int,
        total_files: int,
        sync_start_time: datetime,
        synced_files: int,
        failed_files: List[str],
    ) -> None:
        if index % 10 != 0:
            return

        elapsed = (datetime.now() - sync_start_time).total_seconds()
        avg_time = elapsed / index if index > 0 else 0
        remaining = (total_files - index) * avg_time if avg_time > 0 else 0
        success_rate = (synced_files / index * 100) if index > 0 else 0
        logger.info(
            f"同步进度: {index}/{total_files} "
            f"({synced_files} 成功, {len(failed_files)} 失败, 成功率: "
            f"{success_rate:.1f}%) | 已耗时: {elapsed:.1f}秒 | "
            f"预计剩余: {remaining:.1f}秒"
        )

    def _build_sync_message(
        self,
        synced_files: int,
        total_files: int,
        total_original_files: int,
        available_stock_codes: List[str],
        failed_files: List[str],
        missing_stock_codes: List[str],
        total_duration: float,
    ) -> str:
        if total_original_files > 0:
            original_success_rate = synced_files / total_original_files * 100
            available_success_rate = (
                (synced_files / len(available_stock_codes) * 100)
                if available_stock_codes
                else 0
            )
            message = f"同步完成: {synced_files}/{total_original_files} 成功"
            message += (
                f" (相对于原始列表: {original_success_rate:.1f}%, "
                f"相对于可用文件: {available_success_rate:.1f}%)"
            )
        else:
            success_rate = (synced_files / total_files * 100) if total_files > 0 else 0
            message = (
                f"同步完成: {synced_files}/{total_files} 成功 "
                f"(成功率: {success_rate:.1f}%)"
            )

        if failed_files:
            message += f", {len(failed_files)} 失败"

        if missing_stock_codes:
            message += f", {len(missing_stock_codes)} 个文件在远端不存在（已跳过）"

        message += f", 总耗时: {total_duration:.1f}秒"
        return message

    def sync_all_stocks(self, stock_codes: Optional[List[str]] = None) -> SyncResult:
        """
        同步所有股票数据

        Args:
            stock_codes: 要同步的股票代码列表，如果为None则从远端获取列表

        Returns:
            同步结果
        """
        sync_start_time = datetime.now()
        ssh = None
        sftp = None

        try:
            logger.info("=" * 60)
            logger.info("开始SFTP数据同步任务")
            logger.info("=" * 60)

            # 建立SFTP连接
            logger.info("建立SFTP连接...")
            ssh, sftp = self._connect_sftp()
            logger.info("SFTP连接建立成功")

            # 清空文件缓存，重新构建
            self._remote_files_cache = None
            # 预构建文件缓存以提高效率
            logger.info("预构建远端文件缓存...")
            remote_files_cache = self._build_remote_files_cache(sftp)
            logger.info(
                f"远端文件缓存构建完成，找到 {len(remote_files_cache)} 个可用的股票文件"
            )

            stock_codes = self._get_target_stock_codes(stock_codes)

            if not stock_codes:
                logger.warning("未找到要同步的股票")
                return SyncResult(
                    success=False,
                    total_files=0,
                    synced_files=0,
                    failed_files=[],
                    total_size=0,
                    message="未找到要同步的股票",
                )

            available_stock_codes, missing_stock_codes = self._partition_stock_codes(
                stock_codes, remote_files_cache
            )

            if missing_stock_codes:
                logger.info(
                    f"远端服务器上缺少 {len(missing_stock_codes)} 个股票文件，将跳过这些文件"
                )
                logger.debug(
                    f"缺少的股票代码示例（前20个）: {missing_stock_codes[:20]}"
                )

            original_count = len(stock_codes)
            logger.info(
                f"实际可同步的股票数量: {len(available_stock_codes)}/{original_count}"
            )

            if not available_stock_codes:
                logger.warning("远端服务器上没有可用的股票文件")
                return SyncResult(
                    success=False,
                    total_files=original_count,
                    synced_files=0,
                    failed_files=missing_stock_codes,
                    total_size=0,
                    message=f"远端服务器上没有可用的股票文件（远端共有 {len(remote_files_cache)} 个文件，但股票列表有 {original_count} 个股票代码）",
                )

            # 使用过滤后的股票列表
            stock_codes = available_stock_codes
            total_original_files = original_count  # 保存原始总数用于报告

            # 同步每个股票文件
            total_files = len(stock_codes)
            synced_files = 0
            failed_files: List[str] = []
            total_size = 0

            logger.info(f"开始同步 {total_files} 只股票的数据...")
            logger.info(f"本地存储目录: {self.local_data_dir / 'stock_data'}")

            for i, stock_code in enumerate(stock_codes, 1):
                try:
                    self._emit_sync_event(
                        self.event_manager.emit_sync_started(
                            stock_code=stock_code,
                            date_range=(sync_start_time, datetime.now()),
                            sync_type="sftp_sync",
                            metadata={"batch_index": i, "total_files": total_files},
                        ),
                        f"发出同步开始事件失败 {stock_code}",
                    )

                    file_start_time = datetime.now()
                    (
                        success,
                        file_size,
                        error_msg,
                        ssh,
                        sftp,
                    ) = self._sync_stock_with_reconnect(
                        stock_code,
                        ssh,
                        sftp,
                    )
                    file_duration = (datetime.now() - file_start_time).total_seconds()

                    if success:
                        synced_files += 1
                        total_size += file_size
                        logger.debug(
                            f"[{i}/{total_files}] {stock_code}: 成功, 大小: {file_size} 字节, 耗时: {file_duration:.2f}秒"
                        )

                        self._emit_sync_event(
                            self.event_manager.emit_sync_completed(
                                stock_code=stock_code,
                                date_range=(file_start_time, datetime.now()),
                                sync_type="sftp_sync",
                                metadata={
                                    "file_size": file_size,
                                    "duration_seconds": file_duration,
                                    "batch_index": i,
                                    "total_files": total_files,
                                },
                            ),
                            f"发出同步完成事件失败 {stock_code}",
                        )
                    else:
                        failed_files.append(stock_code)
                        logger.warning(
                            f"[{i}/{total_files}] {stock_code}: 失败 - {error_msg}"
                        )

                        self._emit_sync_event(
                            self.event_manager.emit_sync_failed(
                                stock_code=stock_code,
                                date_range=(file_start_time, datetime.now()),
                                sync_type="sftp_sync",
                                error_message=error_msg or "文件同步失败",
                                metadata={
                                    "batch_index": i,
                                    "total_files": total_files,
                                },
                            ),
                            f"发出同步失败事件失败 {stock_code}",
                        )

                    ssh, sftp = self._refresh_connection_if_needed(i, ssh, sftp)
                    self._log_sync_progress(
                        i,
                        total_files,
                        sync_start_time,
                        synced_files,
                        failed_files,
                    )

                except Exception as e:
                    logger.error(f"同步股票 {stock_code} 时出错: {e}", exc_info=True)
                    failed_files.append(stock_code)

            total_duration = (datetime.now() - sync_start_time).total_seconds()
            success = synced_files > 0

            message = self._build_sync_message(
                synced_files=synced_files,
                total_files=total_files,
                total_original_files=total_original_files,
                available_stock_codes=available_stock_codes,
                failed_files=failed_files,
                missing_stock_codes=missing_stock_codes,
                total_duration=total_duration,
            )

            if failed_files:
                # 如果失败文件较多，只记录前20个
                if len(failed_files) > 20:
                    logger.warning(f"失败文件列表（前20个）: {failed_files[:20]}")
                    logger.warning(f"还有 {len(failed_files) - 20} 个失败文件未显示")
                else:
                    logger.warning(f"失败文件列表: {failed_files}")

            if missing_stock_codes:
                logger.info(f"远端不存在的文件数量: {len(missing_stock_codes)}")

            logger.info("=" * 60)
            logger.info(f"同步任务完成: {message}")
            if total_original_files > 0:
                logger.info(
                    f"原始股票列表: {total_original_files} 个, 远端可用文件: "
                    f"{len(available_stock_codes)} 个"
                )
            logger.info(
                f"实际同步文件数: {total_files}, 成功: {synced_files}, 失败: {len(failed_files)}"
            )
            logger.info(f"总数据大小: {total_size / (1024 * 1024):.2f} MB")
            logger.info(f"总耗时: {total_duration:.1f}秒")
            if synced_files > 0:
                logger.info(f"平均速度: {synced_files / total_duration:.2f} 文件/秒")
            if failed_files:
                logger.warning(f"失败的股票代码（前10个）: {failed_files[:10]}")
            if missing_stock_codes:
                logger.info(
                    f"远端不存在的股票代码示例（前10个）: {missing_stock_codes[:10]}"
                )
            logger.info("=" * 60)

            return SyncResult(
                success=success,
                total_files=total_files,
                synced_files=synced_files,
                failed_files=failed_files,
                total_size=total_size,
                message=message,
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
                message=f"同步失败: {str(e)}",
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
