"""baostock 长窗口 + 含退市宇宙（§12.83 前置 B）的契约测试。

前 6 条用合成数据，永远该绿：它们锁的是「验收闸门本身有效」。
最后 1 条是真数据闸门：数据没拉之前自动 skip，数据一到就必须绿——
B 这一步算不算做完，就看这条绿不绿（skip 不算做完）。
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from scripts.baostock_full_history_audit import (
    ENDPOINT_WINDOW_DAYS,
    FULL_DIR,
    REGISTRY,
    chain_consistency,
    coverage_ratio,
    endpoint_gap_days,
    endpoint_in_window,
    reg_key,
    run_audit,
)


def _cal(n=200, start="2012-01-02"):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=n))


def test_reg_key_format_bridge_is_exact():
    assert reg_key("sh.600000") == "600000.SH"
    assert reg_key("sz.000018") == "000018.SZ"
    assert reg_key("sh.688981") == "688981.SH"


def test_coverage_ratio_detects_half_missing_bars():
    cal = _cal(200)
    full = pd.Series(cal)
    half = pd.Series(cal[::2])
    assert coverage_ratio(full, cal[0], cal[-1], cal) == 1.0
    assert abs(coverage_ratio(half, cal[0], cal[-1], cal) - 0.5) < 1e-9
    assert coverage_ratio(pd.Series([]), cal[0], cal[-1], cal) == 0.0
    # 只算窗口内：早于 from_d 的 bar 不许 inflate 覆盖率
    early = pd.Series(cal[:80])
    assert coverage_ratio(early, cal[100], cal[-1], cal) == 0.0


def test_endpoint_window_semantics_for_delisting():
    out = pd.Timestamp("2020-01-15")
    assert endpoint_gap_days(pd.Timestamp("2020-01-08"), out) == 7
    assert endpoint_in_window(7)
    assert endpoint_in_window(0) and endpoint_in_window(ENDPOINT_WINDOW_DAYS)
    assert not endpoint_in_window(-5), "末根 bar 晚于摘牌日 = 数据错了，不许通过"
    assert not endpoint_in_window(200), "末根 bar 早于摘牌日 200 天 = 提前消失，不许通过"


def test_chain_consistency_catches_small_caliber_drift():
    cal = _cal(300)
    r = pd.Series([0.001] * 300, index=cal)
    same = chain_consistency(r, r.copy())
    assert same["max_rel_diff"] is not None and same["max_rel_diff"] < 1e-9
    off = pd.Series([0.001 + 0.00002] * 300, index=cal)          # 每天差 0.002pp
    drift = chain_consistency(r, off)
    assert drift["max_rel_diff"] > 1e-3, f"累计漂移必须被抓出来，实测 {drift['max_rel_diff']}"
    short = chain_consistency(r.iloc[:30], r.iloc[:30].copy())
    assert short["max_rel_diff"] is None, "重叠不足 60 天不许给出一致性结论"


def test_fetch_layer_writes_new_dir_not_the_live_layer():
    """新层必须落在 baostock_full，不许覆盖现役 baostock_daily（否则现役回测口径会被静默改掉）。"""
    from scripts.fetch_akshare_full_history import OUT_DIR, START, out_name
    assert os.path.basename(OUT_DIR) == "akshare_full"
    assert "baostock_daily" not in OUT_DIR
    assert START == "20120101"
    assert out_name("600000", "SH") == "600000_SH.parquet"
    assert out_name("300104", "SZ") == "300104_SZ.parquet"
    from scripts.fetch_akshare_full_history import COLS
    assert "close" in COLS and "close_hfq" in COLS and "pct_chg_raw" in COLS


@pytest.mark.skipif(not os.path.exists(REGISTRY), reason="注册表副本还没落盘（数据到位后这条必须绿，skip 不算完工）")
def test_registry_excludes_index_rows():
    """注册表 type=1 才是股票；把指数混进宇宙会让覆盖率与退市计数全是假的。

    baostock 已把本机 IP 拉黑（10001011 黑名单用户）→ 验收只读落盘副本，不许再依赖登录。
    """
    df = pd.read_parquet(REGISTRY)
    stk = df[df["type"] == "1"]
    assert len(stk) > 5000
    assert (stk["status"] != 1).sum() >= 300, "注册表里必须有退市股，否则幸存者偏差没补"
    assert (stk["type"] != "1").sum() == 0


@pytest.mark.skipif(not os.path.exists(REGISTRY), reason="baostock_full 还没拉（数据到位后这条必须绿，skip 不算完工）")
def test_full_history_layer_passes_acceptance_gates():
    res = run_audit()
    g = res["gates"]
    assert g["delisted_out_date_after_start"] >= 250, g
    assert g["delisted_with_bars"] >= g["delisted_out_date_after_start"] * 0.95, g
    assert g["delisted_endpoint_bad_n"] <= max(3, int(g["delisted_with_bars"] * 0.05)), g["delisted_endpoint_bad"]
    assert g["coverage_median_pct"] >= 90.0, g
    assert g["low_coverage_n"] <= int(g["files"] * 0.10), g["low_coverage_worst"]
    c = g["chain_vs_live_layer"]
    # 口径判决后（hfq 配股减法出负价 / pctChg 送转虚增，两侧都不合格），与 baostock 层的链对比只作披露，
    # 口径闸门改由 build_total_return_panel.py 负责
    assert "status" in c, c
    assert g["data_first"] <= "2012-01-05" and g["data_last"] >= "2026-11-01", g
    assert g["all_pass"], g
