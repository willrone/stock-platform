"""当前 IncrementalUpdater 契约测试。"""

import pandas as pd

from app.services.data.incremental_updater import IncrementalUpdater


def _build_updater(tmp_path, monkeypatch) -> IncrementalUpdater:
    monkeypatch.setattr(
        "app.services.data.incremental_updater.settings.DATA_ROOT_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "app.services.data.incremental_updater.settings.QLIB_DATA_PATH",
        str(tmp_path / "qlib"),
    )
    return IncrementalUpdater()


def test_get_all_stock_codes_reads_current_filename_contract(
    tmp_path, monkeypatch
) -> None:
    updater = _build_updater(tmp_path, monkeypatch)
    updater.parquet_data_path.mkdir(parents=True, exist_ok=True)

    (updater.parquet_data_path / "000001_SZ.parquet").touch()
    (updater.parquet_data_path / "600000_SH.parquet").touch()
    (updater.parquet_data_path / "430001_BJ.parquet").touch()
    (updater.parquet_data_path / "invalid_name.parquet").touch()

    stock_codes = updater._get_all_stock_codes()

    assert stock_codes == ["000001.SZ", "430001.BJ", "600000.SH"]


def test_detect_changes_only_returns_new_or_updated_stocks(
    tmp_path, monkeypatch
) -> None:
    updater = _build_updater(tmp_path, monkeypatch)
    monkeypatch.setattr(
        updater,
        "_get_all_stock_codes",
        lambda: ["000001.SZ", "000002.SZ", "000003.SZ"],
    )

    actions = {
        "000001.SZ": {"action": "new"},
        "000002.SZ": {"action": "update"},
        "000003.SZ": {"action": "none"},
    }
    monkeypatch.setattr(
        updater, "_detect_stock_changes", lambda stock_code: actions[stock_code]
    )

    changes = updater.detect_changes()

    assert changes == {
        "000001.SZ": {"action": "new"},
        "000002.SZ": {"action": "update"},
    }


def test_get_stocks_to_update_respects_force_update_and_detected_changes(
    tmp_path, monkeypatch
) -> None:
    updater = _build_updater(tmp_path, monkeypatch)
    monkeypatch.setattr(
        updater,
        "_get_all_stock_codes",
        lambda: ["000001.SZ", "000002.SZ"],
    )
    monkeypatch.setattr(
        updater,
        "detect_changes",
        lambda stock_codes=None: {
            "000001.SZ": {"action": "update"},
            "000002.SZ": {"action": "none"},
        },
    )

    assert updater.get_stocks_to_update(force_update=True) == ["000001.SZ", "000002.SZ"]
    assert updater.get_stocks_to_update(["000003.SZ"], force_update=True) == [
        "000003.SZ"
    ]
    assert updater.get_stocks_to_update(force_update=False) == ["000001.SZ"]


def test_merge_incremental_data_replaces_overlapping_dates_and_preserves_rest(
    tmp_path, monkeypatch
) -> None:
    updater = _build_updater(tmp_path, monkeypatch)
    existing_data = pd.DataFrame(
        {"close": [10.0, 11.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    new_data = pd.DataFrame(
        {"close": [12.5, 13.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    merged = updater.merge_incremental_data(existing_data, new_data, "000001.SZ")

    dates = list(merged.index.get_level_values(1).strftime("%Y-%m-%d"))
    close_values = list(merged["close"])

    assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert close_values == [10.0, 12.5, 13.0]
