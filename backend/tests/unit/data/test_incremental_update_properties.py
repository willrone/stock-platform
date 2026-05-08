"""当前 SimpleDataService 本地缓存契约测试。"""

from datetime import datetime

import pandas as pd

from app.services.data import SimpleDataService


def _build_service(tmp_path) -> SimpleDataService:
    return SimpleDataService(data_path=str(tmp_path), remote_url="http://example.test")


def test_generate_mock_data_returns_business_day_rows(tmp_path) -> None:
    service = _build_service(tmp_path)
    rows = service.generate_mock_data(
        "000001.SZ",
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
    )

    assert rows
    assert all(row["stock_code"] == "000001.SZ" for row in rows)
    assert all("date" in row for row in rows)
    assert len(rows) <= 8  # 仅工作日


def test_get_stock_data_prefers_existing_local_cache(tmp_path) -> None:
    service = _build_service(tmp_path)
    stock_code = "000001.SZ"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)
    local_rows = service.generate_mock_data(stock_code, start_date, end_date)
    assert service.save_to_local(local_rows, stock_code) is True

    result = service.load_from_local(stock_code, start_date, end_date)

    assert result is not None
    assert len(result) == len(local_rows)
    assert service.check_local_data_exists(stock_code, start_date, end_date) is True


def test_get_stock_data_uses_offline_fallback_and_caches_result(tmp_path) -> None:
    service = _build_service(tmp_path)
    stock_code = "000001.SZ"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)

    result = __import__("asyncio").run(
        service.get_stock_data(stock_code, start_date, end_date)
    )

    assert result is not None
    assert len(result) > 0
    assert service.get_local_data_path(stock_code).exists() is True
    assert service.check_local_data_exists(stock_code, start_date, end_date) is True


def test_get_stock_data_accepts_dataframe_remote_results(tmp_path) -> None:
    service = _build_service(tmp_path)
    stock_code = "000001.SZ"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)

    remote_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [1000, 2000],
            "adj_close": [10.2, 11.2],
        }
    )

    async def fake_fetch_remote_data(*args, **kwargs):
        return remote_df

    service.fetch_remote_data = fake_fetch_remote_data

    result = __import__("asyncio").run(
        service.get_stock_data(stock_code, start_date, end_date, force_remote=True)
    )

    assert result is not None
    assert len(result) == 2
    cached_rows = service.load_from_local(stock_code, start_date, end_date)
    assert cached_rows is not None
    assert len(cached_rows) == 2
