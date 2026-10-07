"""模型管理路由模块（兼容别名层）。

拆包说明：旧版单文件 ``app/api/v1/models.py`` 的实现整体保留在
``app/api/v1/models_impl.py``。本包把实现模块的名字转发到包命名空间，
并把父模块（``app.api.v1``）上的 ``models`` 属性指向 impl 模块本身。

为什么"父属性指向 impl"是必须的：
存量 contract 测试会伪造 ``app.api.v1`` 包（``ModuleType`` + ``__path__``）
再执行 ``import app.api.v1.models as models_module``。CPython 的
``import a.b as c`` 通过 ``getattr(a, "b")`` 绑定，因此 ``models_module``
拿到的是父模块上的 ``models`` 属性。若它指向包模块（而非 impl），
``patch.object(models_module, "SessionLocal")`` 落在包 dict 上，而路由函数
的 ``__globals__`` 是 impl dict → mock 永远落空，路由回落到真实 DB。
把父属性指向 impl 后，``models_module`` 就是 impl 本身，打桩与路由共享
同一命名空间，行为与旧单文件完全一致。

常规（非伪造）导入路径下，``app/api/v1/__init__.py`` 的别名机制会先把
``sys.modules["app.api.v1.models"]`` 指向 impl，本 ``__init__`` 根本不会执行；
这里的逻辑只服务于"父模块被替换/伪造"的测试场景。

后续将按 crud / training / prediction / evaluation / lifecycle / search
子模块进一步拆分，届时替换本别名层即可。
"""

import importlib as _importlib
import sys as _sys
from types import ModuleType as _ModuleType

_impl = _importlib.import_module("app.api.v1.models_impl")

# 转发 impl 的全部名字到包命名空间（等价旧单文件的模块命名空间）。
# 用快照迭代，避免"字典边迭代边变更"。
_impl_items = list(vars(_impl).items())
for _name, _value in _impl_items:
    if not _name.startswith("__"):
        globals()[_name] = _value
del _impl_items

# 关键：让 `import app.api.v1.models as m` 绑定到 impl（见模块 docstring）。
_parent = _sys.modules.get("app.api.v1")
if _parent is not None and getattr(_parent, "models", None) is not _impl:
    try:
        setattr(_parent, "models", _impl)  # noqa: B010
    except Exception:  # pragma: no cover - 只读父模块时退化
        pass

# `from app.api.v1.models import ...` / sys.modules 查找都解析到 impl。
_sys.modules[__name__] = _impl

__all__ = ["router"] + [
    n
    for n in list(globals())
    if not n.startswith("_") and not isinstance(globals()[n], _ModuleType)
]
