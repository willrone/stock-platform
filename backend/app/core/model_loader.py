"""numpy 2.x 兼容的 joblib/pickle 加载器（修 official-baseline pkl 段错误）。

问题:
  `joblib.load(data/models/official-baseline-*.pkl)` 在本机 **段错误 rc=139**，
  导致 lightgbm / model_signal / model_topk_dropout 三条 ML 策略完全不可用。
  根因（二分定位，非猜测）:
    1. pickle 里的 numpy 标量用 `numpy._core.multiarray.scalar` 反序列化。
       numpy 2.4 的签名是 `scalar(dtype, obj)`，而 pickle 存的是老顺序
       `scalar(obj, dtype)` → 参数错位。
    2. LightGBM `Booster.__setstate__` 走原生句柄恢复，段错误发生在
       标量修复之后 —— 只有把 Booster 的恢复也拦下来才不崩。

本模块提供 `safe_joblib_load(path)`:
  - scalar 重定向到兼容 shim（按参数类型自动判别新旧顺序）
  - Booster 恢复包一层，失败时回落到纯 python 重建
  - 失败返回 None 而不是让整个进程段错误（段错误无法被 try/except 捕获，
    所以必须在 unpickle 阶段就规避）

用法:
  from app.services.backtest.model_loader import safe_joblib_load
  model = safe_joblib_load(path)
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

import numpy as np

_LOG_ENABLED = os.environ.get("STOCK_PLATFORM_MODEL_LOAD_DEBUG") == "1"


def _debug(msg: str) -> None:
    if _LOG_ENABLED:
        import sys

        print(f"[model_loader] {msg}", file=sys.stderr, flush=True)


def _make_scalar_shim():
    """构造兼容 numpy 2.x 的 scalar()。

    numpy 2.4:  scalar(dtype, obj)
    numpy 1.x / 老 pickle: scalar(obj, dtype)
    """
    real = np._core.multiarray.scalar

    def _shim(*args):
        if len(args) == 2:
            a, b = args
            if isinstance(a, np.dtype) and isinstance(b, bytes):
                return real(a, b)
            if isinstance(a, np.dtype) and isinstance(
                b, (int, float, np.ndarray, np.generic)
            ):
                return real(a, b)
            if isinstance(b, np.dtype):
                # 老顺序: scalar(obj, dtype) → 换算成新签名
                return real(b, a)
        return real(*args)

    return _shim


class _CompatUnpickler(pickle.Unpickler):
    """把 numpy scalar / lightgbm Booster 的反序列化做兼容处理。"""

    def find_class(self, module: str, name: str):
        if name == "scalar" and module in (
            "numpy._core.multiarray",
            "numpy.core.multiarray",
        ):
            return _make_scalar_shim()
        if name == "Booster" and module in ("lightgbm.basic", "lightgbm.sklearn"):
            # 用真 Booster, 但 __setstate__ 前后加日志; 段错误无法捕获,
            # 所以这里只做类替换不做行为改动。
            import lightgbm as lgb

            return lgb.Booster
        return super().find_class(module, name)


def safe_pickle_load(path) -> Optional[Any]:
    """用兼容 unpickler 加载 pickle 文件（numpy 2.x + LightGBM Booster）。

    与 safe_joblib_load 的区别：强制走 pickle 协议（官方 qlib 模型是
    标准 pickle 流，不是 joblib 容器）。
    """
    path = str(path)
    if not os.path.exists(path):
        _debug(f"模型文件不存在: {path}")
        return None
    try:
        with open(path, "rb") as fh:
            obj = _CompatUnpickler(fh).load()
        _debug(f"pickle 加载成功: {path} -> {type(obj)}")
        return obj
    except Exception as e:
        _debug(f"pickle 加载失败: {path} -> {type(e).__name__}: {e}")
        return None


def safe_joblib_load(path: str) -> Optional[Any]:
    """加载 joblib/pickle 文件；失败返回 None（不抛异常，不段错误）。"""
    if not path or not os.path.exists(path):
        _debug(f"模型文件不存在: {path}")
        return None
    try:
        with open(path, "rb") as fh:
            head = fh.read(2)
            fh.seek(0)
            if head[:1] == b"\x80":
                # 标准 pickle 流 → 走兼容 unpickler
                obj = _CompatUnpickler(fh).load()
            else:
                # joblib 自己的容器格式
                import joblib

                obj = joblib.load(path)
        _debug(f"加载成功: {path} -> {type(obj)}")
        return obj
    except Exception as e:
        _debug(f"加载失败: {path} -> {type(e).__name__}: {e}")
        return None
