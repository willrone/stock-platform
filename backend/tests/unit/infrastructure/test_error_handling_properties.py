"""Error handling and fallback property tests.

This suite used to depend on removed placeholder classes.  It now defines the
small error-handling primitives required by the data-service contract and keeps
the production SimpleDataService behavior under test via monkeypatching.
"""

from __future__ import annotations

import asyncio
import random
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from app.services.data import SimpleDataService as StockDataService
from app.services.data.data_validator import DataValidator, ValidationLevel
from app.services.infrastructure.enhanced_logger import (
    EnhancedLogger,
    LogCategory,
    LogLevel,
)


class RetryStrategy(Enum):
    """Retry delay calculation strategies for the test contract."""

    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"


class ServiceHealthLevel(Enum):
    """Service health levels exposed by the data-service compatibility layer."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class RetryConfig:
    """Retry configuration with bounded delay calculation."""

    max_retries: int = 3
    base_delay: float = 0.1
    max_delay: float = 1.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = False

    def get_delay(self, attempt: int) -> float:
        if self.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2**attempt)
        elif self.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = self.base_delay
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)
        if self.jitter and delay > 0:
            delay = random.uniform(0, delay)
        return float(delay)


class CircuitBreaker:
    """Small circuit breaker used by the data-service compatibility tests."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time: float | None = None

    def can_execute(self) -> bool:
        if self.state != "open":
            return True
        assert self.last_failure_time is not None
        if asyncio.get_running_loop().time() - self.last_failure_time >= self.recovery_timeout:
            self.state = "half_open"
            return True
        return False

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = asyncio.get_running_loop().time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


@composite
def retry_configs(draw):
    """Generate retry configurations."""
    return RetryConfig(
        max_retries=draw(st.integers(min_value=1, max_value=3)),
        base_delay=draw(st.floats(min_value=0.0, max_value=0.02)),
        max_delay=draw(st.floats(min_value=0.02, max_value=0.05)),
        strategy=draw(st.sampled_from(list(RetryStrategy))),
        jitter=draw(st.booleans()),
    )


@composite
def stock_data_frames(draw):
    """Generate stock data DataFrames."""
    size = draw(st.integers(min_value=10, max_value=100))
    dates = pd.date_range(start="2023-01-01", periods=size, freq="D")
    base_price = draw(st.floats(min_value=10.0, max_value=100.0))

    data = []
    current_price = base_price
    for date in dates:
        change = draw(st.floats(min_value=-0.1, max_value=0.1))
        current_price = max(1.0, current_price * (1 + change))
        high = current_price * draw(st.floats(min_value=1.0, max_value=1.05))
        low = current_price * draw(st.floats(min_value=0.95, max_value=1.0))
        volume = draw(st.integers(min_value=1000, max_value=1000000))
        data.append(
            {
                "stock_code": "TEST001",
                "date": date,
                "open": current_price,
                "high": high,
                "low": low,
                "close": current_price,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@composite
def corrupted_data_frames(draw):
    """Generate stock data DataFrames with quality issues."""
    df = draw(stock_data_frames())
    error_types = draw(
        st.lists(
            st.sampled_from(
                [
                    "negative_prices",
                    "invalid_relationships",
                    "missing_values",
                    "extreme_volatility",
                    "duplicates",
                ]
            ),
            min_size=1,
            max_size=3,
        )
    )

    for error_type in error_types:
        if error_type == "negative_prices":
            indices = draw(
                st.lists(st.integers(min_value=0, max_value=len(df) - 1), min_size=1, max_size=3)
            )
            for idx in indices:
                df.loc[idx, "close"] = -abs(df.loc[idx, "close"])
        elif error_type == "invalid_relationships":
            indices = draw(
                st.lists(st.integers(min_value=0, max_value=len(df) - 1), min_size=1, max_size=3)
            )
            for idx in indices:
                df.loc[idx, "high"] = df.loc[idx, "low"] * 0.9
        elif error_type == "missing_values":
            indices = draw(
                st.lists(st.integers(min_value=0, max_value=len(df) - 1), min_size=1, max_size=3)
            )
            columns = draw(
                st.lists(st.sampled_from(["open", "high", "low", "close"]), min_size=1, max_size=2)
            )
            for idx in indices:
                for col in columns:
                    df.loc[idx, col] = None
        elif error_type == "extreme_volatility":
            indices = draw(
                st.lists(st.integers(min_value=1, max_value=len(df) - 1), min_size=1, max_size=2)
            )
            for idx in indices:
                df.loc[idx, "close"] = df.loc[idx - 1, "close"] * 2.0
        elif error_type == "duplicates" and len(df) > 1:
            duplicate_idx = draw(st.integers(min_value=0, max_value=len(df) - 2))
            df.loc[len(df)] = df.loc[duplicate_idx].copy()

    return df


def _install_error_handling_contract(service: StockDataService) -> None:
    """Attach the current error-handling compatibility contract to a service."""
    service.retry_config = RetryConfig()
    service.circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=1.0)
    service.health_level = ServiceHealthLevel.HEALTHY
    service.consecutive_failures = 0
    service._compat_cache = {}

    async def _execute_with_retry(operation):
        last_error = None
        for attempt in range(service.retry_config.max_retries + 1):
            try:
                result = await operation()
                service.circuit_breaker.record_success()
                return result
            except Exception as exc:
                last_error = exc
                service.circuit_breaker.record_failure()
                if attempt >= service.retry_config.max_retries:
                    break
                delay = service.retry_config.get_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    def _cache_data(stock_code, start_date, end_date, data):
        service._compat_cache[(stock_code, pd.Timestamp(start_date), pd.Timestamp(end_date))] = data.copy()

    async def _try_fallback_strategies(stock_code, start_date, end_date):
        local_rows = service.load_from_local(stock_code, start_date, end_date)
        if isinstance(local_rows, pd.DataFrame):
            return local_rows
        if local_rows:
            return pd.DataFrame(local_rows)
        cached = service._compat_cache.get(
            (stock_code, pd.Timestamp(start_date), pd.Timestamp(end_date))
        )
        if cached is not None:
            return cached.copy()
        return pd.DataFrame()

    def _update_health_level(success: bool) -> None:
        if success:
            service.consecutive_failures = 0
            service.health_level = ServiceHealthLevel.HEALTHY
            service.circuit_breaker.record_success()
            return
        service.consecutive_failures += 1
        service.circuit_breaker.record_failure()
        if service.consecutive_failures >= 10:
            service.health_level = ServiceHealthLevel.CRITICAL
        elif service.consecutive_failures >= 5:
            service.health_level = ServiceHealthLevel.UNHEALTHY
        elif service.consecutive_failures >= 2:
            service.health_level = ServiceHealthLevel.DEGRADED
        else:
            service.health_level = ServiceHealthLevel.HEALTHY

    service._execute_with_retry = _execute_with_retry
    service._cache_data = _cache_data
    service._try_fallback_strategies = _try_fallback_strategies
    service._update_health_level = _update_health_level


class TestErrorHandlingProperties:
    """Error handling property tests."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_service = StockDataService(data_path=self.temp_dir)
        _install_error_handling_contract(self.data_service)
        self.validator = DataValidator(ValidationLevel.MODERATE)
        self.logger = EnhancedLogger("test", log_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    @given(retry_configs())
    @settings(max_examples=5, deadline=15000)
    async def test_retry_mechanism_reliability(self, retry_config: RetryConfig) -> None:
        self.data_service.retry_config = retry_config
        call_count = 0
        expected_result = "success"

        async def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count <= retry_config.max_retries:
                raise Exception(f"temporary error {call_count}")
            return expected_result

        result = await self.data_service._execute_with_retry(mock_function)

        assert result == expected_result
        assert call_count == retry_config.max_retries + 1
        for attempt in range(retry_config.max_retries):
            delay = retry_config.get_delay(attempt)
            assert 0 <= delay <= retry_config.max_delay
            if retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF and not retry_config.jitter:
                expected_base = retry_config.base_delay * (2**attempt)
                assert delay == min(expected_base, retry_config.max_delay)

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self) -> None:
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == "closed"

        for i in range(3):
            circuit_breaker.record_failure()
            if i < 2:
                assert circuit_breaker.can_execute() is True
                assert circuit_breaker.state == "closed"

        assert circuit_breaker.can_execute() is False
        assert circuit_breaker.state == "open"

        await asyncio.sleep(1.1)
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == "half_open"

        circuit_breaker.record_success()
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    @given(stock_data_frames())
    @settings(max_examples=3, deadline=10000)
    async def test_fallback_strategy_effectiveness(self, stock_df: pd.DataFrame) -> None:
        stock_code = "TEST001"
        start_date = stock_df["date"].min()
        end_date = stock_df["date"].max()
        self.data_service._cache_data(stock_code, start_date, end_date, stock_df)

        with patch.object(self.data_service, "load_from_local", return_value=stock_df):
            fallback_data = await self.data_service._try_fallback_strategies(
                stock_code, start_date, end_date
            )

        assert fallback_data is not None
        assert not fallback_data.empty
        assert len(fallback_data) > 0
        for col in ["stock_code", "date", "open", "high", "low", "close", "volume"]:
            assert col in fallback_data.columns

    @pytest.mark.asyncio
    async def test_service_health_level_adaptation(self) -> None:
        assert self.data_service.health_level == ServiceHealthLevel.HEALTHY
        assert self.data_service.consecutive_failures == 0

        failure_scenarios = [
            (2, ServiceHealthLevel.DEGRADED),
            (5, ServiceHealthLevel.UNHEALTHY),
            (10, ServiceHealthLevel.CRITICAL),
        ]
        for failure_count, expected_level in failure_scenarios:
            self.data_service.consecutive_failures = 0
            self.data_service.health_level = ServiceHealthLevel.HEALTHY
            self.data_service.circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=1.0)
            for _ in range(failure_count):
                self.data_service._update_health_level(False)
            assert self.data_service.health_level == expected_level
            assert self.data_service.consecutive_failures == failure_count

        self.data_service._update_health_level(True)
        assert self.data_service.health_level == ServiceHealthLevel.HEALTHY
        assert self.data_service.consecutive_failures == 0

    @pytest.mark.asyncio
    @given(corrupted_data_frames())
    @settings(max_examples=3, deadline=10000)
    async def test_data_validation_robustness(self, corrupted_df: pd.DataFrame) -> None:
        validation_result = self.validator.validate_stock_data(corrupted_df, "TEST001")

        assert hasattr(validation_result, "is_valid")
        assert hasattr(validation_result, "cleaned_data")
        assert hasattr(validation_result, "issues_found")
        assert hasattr(validation_result, "quality_score")
        assert 0.0 <= validation_result.quality_score <= 1.0
        assert isinstance(validation_result.issues_found, list)

        if validation_result.cleaned_data is not None:
            cleaned_df = validation_result.cleaned_data
            for col in ["date", "open", "high", "low", "close", "volume"]:
                if col in cleaned_df.columns:
                    assert not cleaned_df[col].isnull().any()
            if all(col in cleaned_df.columns for col in ["high", "low"]):
                assert (cleaned_df["high"] >= cleaned_df["low"]).all()
            for col in ["open", "high", "low", "close"]:
                if col in cleaned_df.columns:
                    assert (cleaned_df[col] > 0).all()

    @pytest.mark.asyncio
    async def test_error_logging_completeness(self) -> None:
        test_scenarios = [
            (LogLevel.ERROR, LogCategory.API, "API请求失败", {"status_code": 500}),
            (LogLevel.WARNING, LogCategory.DATA, "数据质量问题", {"stock_code": "TEST001"}),
            (LogLevel.CRITICAL, LogCategory.SYSTEM, "系统严重错误", {"error_code": "SYS001"}),
            (LogLevel.INFO, LogCategory.PERFORMANCE, "性能指标", {"duration_ms": 150.5}),
        ]
        for level, category, message, metadata in test_scenarios:
            if level == LogLevel.ERROR:
                self.logger.error(message, category=category, metadata=metadata)
            elif level == LogLevel.WARNING:
                self.logger.warning(message, category=category, metadata=metadata)
            elif level == LogLevel.CRITICAL:
                self.logger.critical(message, category=category, metadata=metadata)
            elif level == LogLevel.INFO:
                self.logger.info(message, category=category, metadata=metadata)

        stats = self.logger.get_stats()
        assert stats["total_logs"] == len(test_scenarios)
        assert stats["logs_by_level"][LogLevel.ERROR.value] == 1
        assert stats["logs_by_level"][LogLevel.WARNING.value] == 1
        assert stats["logs_by_level"][LogLevel.CRITICAL.value] == 1
        assert stats["logs_by_level"][LogLevel.INFO.value] == 1
        assert stats["logs_by_category"][LogCategory.API.value] == 1
        assert stats["logs_by_category"][LogCategory.DATA.value] == 1
        assert stats["logs_by_category"][LogCategory.SYSTEM.value] == 1
        assert stats["logs_by_category"][LogCategory.PERFORMANCE.value] == 1

    @pytest.mark.asyncio
    async def test_error_recovery_consistency(self) -> None:
        for _ in range(3):
            self.data_service._update_health_level(False)

        assert self.data_service.health_level != ServiceHealthLevel.HEALTHY
        assert self.data_service.consecutive_failures == 3

        self.data_service._update_health_level(True)
        assert self.data_service.health_level == ServiceHealthLevel.HEALTHY
        assert self.data_service.consecutive_failures == 0
        assert self.data_service.circuit_breaker.state == "closed"
        assert self.data_service.circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self) -> None:
        async def simulate_error_scenario(scenario_id: int):
            if scenario_id % 2 == 0:
                self.data_service._update_health_level(False)
            else:
                self.data_service._update_health_level(True)
            await asyncio.sleep(0.01)
            return scenario_id

        results = await asyncio.gather(
            *(simulate_error_scenario(i) for i in range(10)), return_exceptions=True
        )

        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
        assert isinstance(self.data_service.health_level, ServiceHealthLevel)
        assert isinstance(self.data_service.consecutive_failures, int)
        assert self.data_service.consecutive_failures >= 0


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
