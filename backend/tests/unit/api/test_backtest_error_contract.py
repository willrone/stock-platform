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

with patch.dict(
    "sys.modules",
    {
        "app.services.backtest": fake_backtest_services_module,
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

    with patch("app.api.v1.backtest.log_structured_exception") as mock_log:
        value = _coerce_numeric_value(
            "bad-number",
            field_name="total_return",
            default=0.0,
            context=context,
        )

    assert value == 0.0
    mock_log.assert_called_once()
    assert "total_return" in mock_log.call_args.args[0]
