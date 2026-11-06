"""`build_benchmark_index.py` 增量追加的门禁测试（2026-11-06 根因修复的回归闸）。

要防的事故：增量追加写成 `base = float(lvl[code].iloc[-1])` + 逐格 `lvl.loc[d, code]=...`
时，列序第一个 code 会先把新日期那一整行插出来（整行 NaN），此后每个 code 的 `iloc[-1]`
都读到那行空值 → base=NaN → **新追加那天只有 1 只票有电平**（实测 2026-11-05：1/5596）。
更糟的是 `last = lvl.index.max()` 已是该日，之后每次增量 `date > last` 恒为空
→ 坏尾巴永不自愈，所有 `benchmark_return_pct` / `excess_pp` 都拿一只平安银行算。

用合成小数据（4 只票、3 个交易日），不碰真实行情、不写真实产物目录。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "build_benchmark_index.py"
CODES = ["000001_SZ", "000002_SZ", "600000_SH", "600001_SH"]
DATES = ["2026-01-05", "2026-01-06", "2026-01-07"]


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bbi():
    return _load(SCRIPT, "build_benchmark_index_test")


def _write(frames: Dict[str, pd.DataFrame], pct_dir: Path, stock_dir: Path) -> None:
    for code, df in frames.items():
        df.to_parquet(pct_dir / f"{code}.parquet")
        # 池子 = stock_data ∩ baostock_daily，所以两边都要有文件
        df[["date", "close"]].to_parquet(stock_dir / f"{code}.parquet")


def _frame(dates: List[str], pct: List[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "close": [10.0] * len(dates),
        "pct_chg": pct,          # 交易所口径，百分数（+1.00 = +1%）
    })


@pytest.fixture()
def env(tmp_path: Path, bbi: Any, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    pct_dir = tmp_path / "baostock_daily"
    stock_dir = tmp_path / "stock_data"
    out_dir = tmp_path / "paper"
    for d in (pct_dir, stock_dir, out_dir):
        d.mkdir(parents=True)
    monkeypatch.setattr(bbi, "PCT_DIR", str(pct_dir))
    monkeypatch.setattr(bbi, "STOCK_DIR", str(stock_dir))
    monkeypatch.setattr(bbi, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(bbi, "LEVELS_FP", str(out_dir / "benchmark_levels.parquet"))
    frames = {c: _frame(DATES[:2], [0.0, 1.0]) for c in CODES}
    _write(frames, pct_dir, stock_dir)
    return {"bbi": bbi, "pct": pct_dir, "stock": stock_dir, "frames": frames}


# ── 全量路径本来就是对的，先钉住它 ────────────────────────────────────
def test_full_build_covers_every_code(env: Dict[str, Any]) -> None:
    lvl = env["bbi"].build_levels("2026-01-01", incremental=False)
    assert lvl.shape == (2, len(CODES))
    assert int(lvl.notna().sum().sum()) == 2 * len(CODES)


# ── 核心回归：增量追加的新日期必须"几乎全员有价" ──────────────────────
def test_incremental_append_fills_every_code(env: Dict[str, Any]) -> None:
    bbi = env["bbi"]
    base = bbi.build_levels("2026-01-01", incremental=False)
    base.to_parquet(bbi.LEVELS_FP)

    # 新交易日 2026-01-07：全部 4 只都有涨跌幅，其中 1 只停牌（涨跌幅缺失）
    for code in CODES:
        pct = np.nan if code == "600001_SH" else 2.0
        env["frames"][code] = _frame(DATES, [0.0, 1.0, pct])
    _write(env["frames"], env["pct"], env["stock"])

    lvl = bbi.build_levels("2026-01-01", incremental=True)
    new_row = lvl.loc[pd.Timestamp("2026-01-07")]

    # 旧 bug 的样子：只有列序第一个 code 有值
    assert int(new_row.notna().sum()) == len(CODES) - 1, (
        "增量追加把新日期的电平全算成 NaN 了 —— "
        "锚点必须取已知完整的那一行，不能用 iloc[-1]"
    )
    # 连乘口径：锚点(1.01) × (1+2%) = 1.0302，不是从 1.0 重算
    for code in CODES[:3]:
        assert float(new_row[code]) == pytest.approx(1.01 * 1.02, rel=1e-5)
    assert np.isnan(new_row["600001_SH"])          # 停牌 → 该日留 NaN，不造假


# ── 追加不得把历史行改写（只增不改）───────────────────────────────────
def test_incremental_append_does_not_rewrite_history(env: Dict[str, Any]) -> None:
    bbi = env["bbi"]
    base = bbi.build_levels("2026-01-01", incremental=False)
    base.to_parquet(bbi.LEVELS_FP)
    for code in CODES:
        env["frames"][code] = _frame(DATES, [0.0, 1.0, 2.0])
    _write(env["frames"], env["pct"], env["stock"])
    lvl = bbi.build_levels("2026-01-01", incremental=True)
    pd.testing.assert_frame_equal(
        lvl.loc[base.index].astype("float32"), base.astype("float32"))
    assert len(lvl.index) == 3


# ── 红场景：尾巴覆盖率塌了必须非零退出（不许静默落盘）──────────────────
def test_tail_coverage_guard_fires(env: Dict[str, Any]) -> None:
    bbi = env["bbi"]
    base = bbi.build_levels("2026-01-01", incremental=False)
    base.to_parquet(bbi.LEVELS_FP)
    # 只有 1 只票拿到了新日期的涨跌幅（正是 11-05 那种"数据层没同步完"的样子）
    for i, code in enumerate(CODES):
        if i == 0:
            env["frames"][code] = _frame(DATES, [0.0, 1.0, 2.0])
        else:
            env["frames"][code] = _frame(DATES[:2], [0.0, 1.0])
    _write(env["frames"], env["pct"], env["stock"])

    with pytest.raises(SystemExit) as exc:
        bbi.build_levels("2026-01-01", incremental=True)
    assert exc.value.code == 1
