"""数据管理路由模块（兼容别名层）。

拆包说明：旧版单文件 ``app/api/v1/data.py`` 的实现整体保留在
``app/api.v1.data_impl.py``。本包把 impl 的名字转发到包命名空间，并把父模块
（``app.api.v1``）上的 ``data`` 属性指向 impl 本身，使存量 contract 测试
``patch("app.api.v1.data.<name>")``（settings / get_data_service /
get_data_sync_event_manager ...）打桩能真正作用于路由函数（详见 models 包 docstring）。

常规导入路径下由 ``app/api/v1/__init__.py`` 别名机制接管，本 ``__init__``
只在"父模块被替换/伪造"的测试场景执行。

后续将按 query / import_export / maintenance 子模块进一步拆分，
届时替换本别名层即可。
"""

import importlib as _importlib
import sys as _sys
from types import ModuleType as _ModuleType

_impl = _importlib.import_module("app.api.v1.data_impl")

_impl_items = list(vars(_impl).items())
for _name, _value in _impl_items:
    if not _name.startswith("__"):
        globals()[_name] = _value
del _impl_items

_parent = _sys.modules.get("app.api.v1")
if _parent is not None and getattr(_parent, "data", None) is not _impl:
    try:
        setattr(_parent, "data", _impl)  # noqa: B010
    except Exception:  # pragma: no cover - 只读父模块时退化
        pass

_sys.modules[__name__] = _impl

__all__ = ["router"] + [
    n
    for n in list(globals())
    if not n.startswith("_") and not isinstance(globals()[n], _ModuleType)
]
