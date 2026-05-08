"""数据管理模块。

保持对 ``SimpleDataService`` 的兼容导出，同时避免在包导入阶段
立即加载可选依赖较重的实现模块。
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    """按需加载兼容导出。"""
    if name != "SimpleDataService":
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    service_class = getattr(import_module(".simple_data_service", __name__), name)
    globals()[name] = service_class
    return service_class


__all__ = ["SimpleDataService"]
