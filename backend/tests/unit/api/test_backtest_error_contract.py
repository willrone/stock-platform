"""
回测路由错误处理 contract tests
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

api_package = ModuleType("app.api")
api_package.__path__ = [str(BACKEND_ROOT / "app" / "api")]
v1_package = ModuleType("app.api.v1")
v1_package.__path__ = [str(BACKEND_ROOT / "app" / "api" / "v1")]
sys.modules.setdefault("app.api", api_package)
sys.modules.setdefault("app.api.v1", v1_package)

fake_backtest_services_module = ModuleType("app.services.backtest")
fake_backtest_services_module.BacktestConfig = object
fake_backtest_services_module.BacktestExecutor = object

# backtest.py 还会从 utils.official_style_params 导入参数注入函数。
# 预载这两个叶子模块（而非给父 fake 加 __path__ 走真实包），避免拖入
# utils/__init__ 的 data_adapter/signal_integrator 重导入链（见技能陷阱 3）。
fake_utils_module = ModuleType("app.services.backtest.utils")
fake_style_params_module = ModuleType("app.services.backtest.utils.official_style_params")
setattr(
    fake_style_params_module,
    "apply_official_style_topk_dropout_params",
    lambda *args, **kwargs: None,
)

with patch.dict(
    "sys.modules",
    {
        "app.services.backtest": fake_backtest_services_module,
        "app.services.backtest.utils": fake_utils_module,
        "app.services.backtest.utils.official_style_params": fake_style_params_module,
        "vectorbt": MagicMock(),
        "vectorbt.portfolio": MagicMock(),
    },
):
    from app.api.v1.backtest import _coerce_numeric_value
    from app.core.error_handler import ErrorContext


def test_coerce_numeric_value_logs_and_falls_back():
    """非法数值必须记录日志，不能静默吞掉。"""

    context = ErrorContext(
        additional_data={
            "route": "run_backtest",
            "strategy_name": "multi_factor",
            "stock_codes": ["000001.SZ"],
        }
    )

    # 本文件在 patch.dict 块内导入 backtest 模块，块退出时会被从 sys.modules
    # 清除；按字符串 patch 会重新导入出第二实例、mock 落空。直接对函数所在
    # 模块字典 patch，保证命中函数真正读取的全局名。
    mock_log = MagicMock()
    with patch.dict(_coerce_numeric_value.__globals__, {"log_structured_exception": mock_log}):
        value = _coerce_numeric_value(
            "bad-number",
            field_name="total_return",
            default=0.0,
            context=context,
        )

    assert value == 0.0
    mock_log.assert_called_once()
    assert "total_return" in mock_log.call_args.args[0]
