"""负向验证工具：确认"断言失败必须变红"的守门真的生效。

背景（台账 §12.29，commit 1c229b66）：性质测试里的
`except Exception as e: pytest.skip(...)` 会把**被测代码内部**抛的
AssertionError 也吞成 skip → 生产断言失效被降级成"跳过" → 套件照样绿（假绿）。
修法是加一道 `except AssertionError: raise` 守卫。

本脚本用来**证明**那道守卫生效：把 PredictionEngine 的两个预测入口换成内部
抛 AssertionError，被打桩的测试必须变成 **FAILED**（而不是 SKIPPED）。

**默认惰性**：不加环境变量时本插件什么也不做 → 可以安全入库、被 `-p` 加载、
不会误伤 CI。只有显式设置 `PROBE_ASSERTION_GUARD=1` 才会打桩（此时测试**故意变红**）。

用法（在 backend/ 目录下；scripts/ 不在 sys.path 根 → 必须用包路径引用）：

    cd /Users/rone/Projects/stock-platform/backend
    PROBE_ASSERTION_GUARD=1 env -u PYTHONHOME PYTHONPATH=. .venv-py313/bin/python3 -m pytest \
        tests/unit/prediction/test_prediction_engine_properties.py \
        -p scripts.verify_assertion_guard -p no:cacheprovider -q --tb=no

期望结果：RC=1 + 3 个 FAILED + 0 个 SKIPPED。
（只有 3 个而不是全部变红，是对的：test_error_handling_and_fallback 走
error_handler 路径，没被引擎打桩影响 —— 证明是**精确变红**，不是无差别全红。）

不带环境变量时跑同一命令 → 正常全绿（证明插件默认无害）。
"""

import os

_FLAG = "PROBE_ASSERTION_GUARD"


def _boom(*args, **kwargs):
    raise AssertionError("探针：被测代码内部断言失败")


def pytest_configure(config):
    # 默认惰性：不设环境变量就不打桩 → 本插件可以安全入库/被加载，不影响 CI
    if os.environ.get(_FLAG, "") not in ("1", "true", "TRUE"):
        return
    from app.services.prediction.prediction_engine import PredictionEngine

    PredictionEngine.predict_single_stock = _boom
    PredictionEngine.predict_multiple_stocks = _boom
