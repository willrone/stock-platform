"""
Qlib兼容性补丁

修复Qlib库中的已知问题：
1. Path对象与字符串拼接问题
2. 日历文件路径格式问题
3. mount_path配置问题
"""

from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from loguru import logger


def clean_path(path_val: Union[str, Path, Any]) -> Union[str, Any]:
    """
    清理路径，移除末尾的异常字符（如 :\\ 或 :/）

    Args:
        path_val: 路径值，可以是字符串、Path对象或其他类型

    Returns:
        清理后的路径字符串，或原始值（如果不是路径类型）
    """
    if isinstance(path_val, Path):
        path_str = path_val.resolve().as_posix()
    elif isinstance(path_val, str):
        path_str = path_val
    else:
        return path_val

    # 清理路径末尾的异常字符
    path_str = path_str.rstrip(":\\").rstrip(":/").rstrip("\\").rstrip("/")
    # 如果路径中有 :/，替换为 /
    path_str = path_str.replace(r":/", "/").replace(r":\\/", "/")
    return path_str


def _create_patched_mount_nfs_uri(original_func):
    """创建修复后的 _mount_nfs_uri 函数"""

    def _patched_mount_nfs_uri(provider_uri, mount_path, auto_mount):
        """修复后的 _mount_nfs_uri，处理 Path 对象和路径格式问题"""

        # 确保 mount_path 是字符串，而不是 Path 对象
        if isinstance(mount_path, Path):
            mount_path = clean_path(mount_path)
        elif isinstance(mount_path, (list, tuple)):
            mount_path = [clean_path(item) for item in mount_path]
        elif isinstance(mount_path, dict):
            mount_path = {k: clean_path(v) for k, v in mount_path.items()}

        # 如果路径不存在，创建它
        if isinstance(mount_path, str):
            Path(mount_path).mkdir(parents=True, exist_ok=True)
        elif isinstance(mount_path, dict):
            for path_val in mount_path.values():
                if isinstance(path_val, str):
                    Path(path_val).mkdir(parents=True, exist_ok=True)

        # 调用原始函数
        result = original_func(provider_uri, mount_path, auto_mount)

        # 修复 C.dpm.data_path
        _fix_dpm_data_path()

        return result

    return _patched_mount_nfs_uri


def _fix_dpm_data_path():
    """修复 C.dpm.data_path 中的路径问题"""
    try:
        from qlib.config import C

        if not (hasattr(C, "dpm") and hasattr(C.dpm, "data_path")):
            return

        data_path = C.dpm.data_path
        if not isinstance(data_path, dict):
            return

        fixed_data_path = {}
        needs_fix = False

        for freq, path_val in data_path.items():
            path_str = str(path_val)
            fixed_path = clean_path(path_val)

            if (
                ":\\" in path_str
                or r":/" in path_str
                or path_str.endswith(":\\")
                or path_str.endswith(":/")
            ):
                needs_fix = True
                logger.info(f"检测到 data_path[{freq}] 路径问题: {path_str} -> {fixed_path}")
            fixed_data_path[freq] = fixed_path

        if needs_fix:
            _set_dpm_data_path(C, fixed_data_path)

    except Exception as fix_error:
        logger.debug(f"修复 data_path 时出错: {fix_error}")


def _set_dpm_data_path(C, fixed_data_path: Dict[str, str]):
    """尝试多种方式设置 C.dpm.data_path"""
    try:
        C.dpm.data_path = fixed_data_path
        logger.info(f"已修复 C.dpm.data_path: {fixed_data_path}")
    except Exception as set_error:
        try:
            if hasattr(C.dpm, "__dict__"):
                C.dpm.__dict__["data_path"] = fixed_data_path
                logger.info("通过 __dict__ 修复 data_path")
        except Exception:
            logger.warning(f"无法修复 data_path: {set_error}")


def _create_patched_load_calendar(original_func):
    """创建修复后的 load_calendar 方法"""

    def _patched_load_calendar(self, freq, future=False):
        r"""修复后的 load_calendar，修复路径拼接问题（:/ 字符）"""
        try:
            return original_func(self, freq, future)
        except Exception as e:
            error_msg = str(e)

            if "calendar not exists" not in error_msg.lower():
                raise

            if not (r":\/" in error_msg or r":/" in error_msg or ":\\\\/" in error_msg):
                raise

            # 尝试修复路径
            import re

            match = re.search(r"calendar not exists:\s*(.+)", error_msg, re.IGNORECASE)

            if not match:
                raise

            wrong_path = match.group(1).strip()
            fixed_path = (
                wrong_path.replace(r":\\/", "/")
                .replace(r":\\/", "/")
                .replace(r":/", "/")
            )
            fixed_path_obj = Path(fixed_path)

            if fixed_path_obj.exists():
                logger.info(f"检测到日历路径问题，已修复: {wrong_path} -> {fixed_path}")
                return _read_calendar_file(fixed_path_obj)

            # 尝试使用配置中的路径
            return _try_config_calendar_path()

    return _patched_load_calendar


def _read_calendar_file(file_path: Path) -> pd.DatetimeIndex:
    """读取日历文件并返回日期索引"""
    try:
        with open(file_path, "r") as f:
            dates = [line.strip() for line in f if line.strip()]
        calendar_dates = pd.to_datetime(dates, format="%Y%m%d")
        logger.info(f"成功从修复后的路径读取日历，包含 {len(calendar_dates)} 个交易日")
        return calendar_dates
    except Exception as read_error:
        logger.warning(f"读取日历文件失败: {read_error}")
        raise


def _try_config_calendar_path() -> pd.DatetimeIndex:
    """尝试从配置路径读取日历"""
    try:
        from app.core.config import settings

        config_path = Path(settings.QLIB_DATA_PATH).resolve() / "calendars" / "day.txt"
        if config_path.exists():
            logger.info(f"使用配置路径读取日历: {config_path}")
            return _read_calendar_file(config_path)
    except Exception as config_error:
        logger.warning(f"使用配置路径也失败: {config_error}")
    raise


def apply_qlib_patches():
    """
    应用所有Qlib兼容性补丁

    应在导入qlib后立即调用此函数
    """
    try:
        import qlib

        # Patch _mount_nfs_uri
        if hasattr(qlib, "_mount_nfs_uri"):
            original_mount = qlib._mount_nfs_uri
            qlib._mount_nfs_uri = _create_patched_mount_nfs_uri(original_mount)
            logger.debug("已 patch qlib._mount_nfs_uri")

        # Patch CalendarProvider.load_calendar
        try:
            from qlib.data import CalendarProvider

            if hasattr(CalendarProvider, "load_calendar"):
                original_load = CalendarProvider.load_calendar
                CalendarProvider.load_calendar = _create_patched_load_calendar(
                    original_load
                )
                logger.debug("已 patch CalendarProvider.load_calendar")
        except Exception as cal_patch_error:
            logger.debug(f"无法 patch CalendarProvider: {cal_patch_error}")

    except ImportError:
        pass  # Qlib未安装


# ============================================================
# Qlib可用性检测和导入
# ============================================================

QLIB_AVAILABLE = False
ALPHA158_AVAILABLE = False
Alpha158DL = None
Alpha158Handler = None
QlibDataLoader = None

# Qlib相关的可选导入（供外部使用）
qlib = None
REG_CN = None
C = None
D = None
DatasetH = None
ExpressionDFilter = None
NameDFilter = None
init_instance_by_config = None

try:
    import qlib as _qlib
    from qlib.config import REG_CN as _REG_CN
    from qlib.config import C as _C
    from qlib.contrib.data.handler import Alpha158 as _Alpha158Handler
    from qlib.contrib.data.loader import Alpha158DL as _Alpha158DL
    from qlib.data import D as _D
    from qlib.data.dataset import DatasetH as _DatasetH
    from qlib.data.dataset.loader import QlibDataLoader as _QlibDataLoader
    from qlib.data.filter import ExpressionDFilter as _ExpressionDFilter
    from qlib.data.filter import NameDFilter as _NameDFilter
    from qlib.utils import init_instance_by_config as _init_instance_by_config

    # 导出
    qlib = _qlib
    REG_CN = _REG_CN
    C = _C
    D = _D
    DatasetH = _DatasetH
    Alpha158DL = _Alpha158DL
    Alpha158Handler = _Alpha158Handler
    QlibDataLoader = _QlibDataLoader
    ExpressionDFilter = _ExpressionDFilter
    NameDFilter = _NameDFilter
    init_instance_by_config = _init_instance_by_config

    QLIB_AVAILABLE = True
    ALPHA158_AVAILABLE = True

    # 应用补丁
    apply_qlib_patches()

except ImportError as e:
    error_msg = str(e)
    missing_module = None

    if "setuptools_scm" in error_msg:
        missing_module = "setuptools_scm"
    elif "ruamel" in error_msg or "ruamel.yaml" in error_msg:
        missing_module = "ruamel.yaml"
    elif "cvxpy" in error_msg:
        missing_module = "cvxpy"

    if missing_module:
        logger.warning(
            f"Qlib缺少依赖 {missing_module}: {e}\n"
            f"解决方法: pip install {missing_module}\n"
            f"或运行修复脚本: ./fix_qlib_dependencies.sh"
        )
    else:
        logger.warning(f"Qlib未安装或Alpha158不可用: {e}")
