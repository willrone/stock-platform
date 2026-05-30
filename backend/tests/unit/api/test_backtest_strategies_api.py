"""Backtest strategies API contract tests."""

import pytest

from app.api.v1.backtest import get_available_strategies


@pytest.mark.asyncio
async def test_moving_average_api_documents_runtime_default_threshold() -> None:
    """API metadata must match MovingAverageStrategy's actionable 0.5% default."""

    response = await get_available_strategies()

    assert response.success is True
    strategies = {item["key"]: item for item in response.data}
    threshold = strategies["moving_average"]["parameters"]["signal_threshold"]
    assert threshold["default"] == 0.005
    assert "0.5%" in threshold["description"]
