"""
API v1 模块。

保持包入口轻量，避免导入单个路由模块时连带加载所有 API 与可选重依赖。
"""

import importlib as _importlib

__all__ = [
    "auth",
    "billing",
    "commerce",  # 按使用量计费模块
    "health",
    "stocks",
    "predictions",
    "tasks",
    "models",
    "backtest",
    "backtest_detailed",
    "backtest_websocket",
    "data",
    "system",
    "qlib",
    "infrastructure",
    "data_versioning",
    "features",
    "training_progress",
    "monitoring",
    "files",
    "strategy_configs",
    "optimization",
    "signals",
    "laya",  # Laya 决策模型
]


def _alias_impl_submodules() -> None:
    """把拆包后"包 + *_impl.py"形式的路由模块暴露为单一实现模块。

    旧版单文件 ``models.py``/``tasks.py``/``data.py`` 被拆成
    ``<name>/``（兼容层）+ ``<name>_impl.py``（实现）。存量 contract 测试用
    ``patch("app.api.v1.<name>.<obj>")`` 打桩并直接 ``import app.api.v1.<name>``
    重建模块实例。若包是普通包（``__path__`` 指向目录），该 import 会重新执行
    包 ``__init__`` 造出新的包命名空间，与被 ``from ... import router`` 绑定的
    impl 模块 ``__dict__`` 不是同一份，mock 永远落空（路由回落到真实 DB）。

    这里在包入口把这三个子模块条目直接指向其 impl 模块对象：
    ``from app.api.v1 import models``（getattr）与 ``import app.api.v1.models``
    （sys.modules 命中）都拿到实现模块本身，mock 与路由共享同一命名空间。
    """
    import sys

    for _name in ("models", "tasks", "data"):
        try:
            _impl = _importlib.import_module(f"app.api.v1.{_name}_impl")
        except Exception:  # pragma: no cover - impl 缺失时退化为普通包
            continue
        sys.modules[f"app.api.v1.{_name}"] = _impl
        globals()[_name] = _impl


_alias_impl_submodules()
del _alias_impl_submodules
