#!/usr/bin/env python3
"""从 TDX raw 的 `prev_close` 同步精确涨跌幅到 baostock_daily 目录 —— 2026-10-26

## 为什么做
1. **北交所缺口**：baostock 不支持 `bj.` 前缀，354 只 `920xxx.BJ` 拉不到数据 →
   它们在纸面盘里只能走「涨跌幅不可靠」回退路径 → 涨跌停判断失真。
   而 **TDX 的 `prev_close` 是交易所口径的前收盘**（已做除权除息调整），
   实测 600000.SH 的 close/prev_close-1 = -0.94%，与 baostock 的 -0.9356% 一致。
2. **每日增量**：TDX raw 每天新增一个分区，可以把当日精确涨跌幅补进去
   （比每次用 baostock 重拉 5240 只快两个数量级）。

## 数据流
  TDX raw 分区 (date, ts_code, open, high, low, close, prev_close, volume, amount)
      → pct_chg = close / prev_close - 1
      → 合并进 data/parquet/baostock_daily/<code>_<SH|SZ|BJ>.parquet
         （已有文件：更新对应日期的行；baostock 没有的：新建）

## 边界
- TDX raw 目前只回溯到 2026-05-11（更早的分区已被清理），
  所以 2026-05-11 之前的历史仍依赖 baostock（北交所那段缺失）。
- 本地已有 baostock 行时**只更新 pct_chg**，不覆盖其它字段
  （成交额/换手率/ST 标记只有 baostock 有）。

用法：
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/sync_pct_chg_from_tdx.py            # 全量同步所有分区
  ... --since 2026-10-01                               # 只同步指定日期之后
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")
OUT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
COLS = [
    "date", "code", "open", "high", "low", "close", "volume", "amount",
    "pct_chg", "turn", "is_st", "trade_status", "pe_ttm", "pb_mrq", "ps_ttm",
]


def _normalize_since(since: str) -> str:
    """把 `--since` 归一成 `YYYY-MM-DD`（分区目录名的格式是 `date=YYYY-MM-DD`）。

    ⚠️ 2026-11-05 修的真实 bug（不是防御性代码）：
    排期脚本 `scripts/daily_tdx_update.sh` 传的是 `date -v-7d +%Y%m%d`（如 `20261029`），
    而分区名是 `2026-11-04` → 按**字符串**比较时
    `"2026-11-04" < "20261029"` 恒为 True（第 5 位 `'-'`(45) < `'1'`(49)）
    → **所有分区都被判成"太旧"而跳过** → 脚本打印「没有可用的 TDX raw 分区」
    却**退出码 0** → shell 高高兴兴报 "✓ sync pct_chg done"。
    后果：`baostock_daily`（真正喂后复权估值的涨跌幅层）自 2026-11-02 起没更新
    → `adj_close` 在 11-03/11-04 整列 NaN → 纸面盘所有 A 股腿的市值被**冻在 11-02 的价**
    （实测 `multi_asset_final_a` 11-04 = 11-02 = ¥499,225.89，60 只里 59 只标 stale）。

    规则：能解析就归一；解析不了就**报错退出**，绝不静默什么也不做。
    """
    s = str(since or "").strip()
    if not s:
        return ""
    digits = s.replace("-", "")
    if len(digits) == 8 and digits.isdigit():
        candidate = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        try:
            _dt.strptime(candidate, "%Y-%m-%d")
        except ValueError:
            raise SystemExit(
                f"❌ --since 是 8 位数字但不是合法日期：{since!r} → {candidate!r}")
        return candidate
    try:
        _dt.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(
            f"❌ --since 既不是 YYYYMMDD 也不是 YYYY-MM-DD：{since!r}"
            "（分区名是 date=YYYY-MM-DD，格式错了会一个分区都选不中）")
    return s


def raw_partitions(since: str = "") -> List[Tuple[str, str]]:
    since = _normalize_since(since)
    out = []
    for name in sorted(os.listdir(RAW_DIR)):
        if not name.startswith("date="):
            continue
        day = name.split("=", 1)[1]
        if since and day < since:
            continue
        out.append((day, os.path.join(RAW_DIR, name, "part.parquet")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="",
                    help="只同步该日期之后的分区（YYYY-MM-DD 或 YYYYMMDD 都接受）")
    args = ap.parse_args()

    since = _normalize_since(args.since)
    parts = raw_partitions(since)
    if not parts:
        # ⚠️ 必须非 0 退出：这一步"什么也没做"却报成功，就是后复权层掉队的开始
        print(f"❌ 没有可用的 TDX raw 分区（RAW_DIR={RAW_DIR}，since={since or '不限'}）"
              " → 后复权涨跌幅层不会更新，本次判失败", flush=True)
        sys.exit(1)
    print(f"[sync] TDX 分区 {len(parts)} 个："
          f"{parts[0][0]} → {parts[-1][0]}"
          f"（since={since or '不限'}）", flush=True)

    # 白名单：stock_data 目录里的代码（= 我们真正的股票池）
    stock_dir = os.path.join(ROOT, "data", "parquet", "stock_data")
    whitelist = set()
    for f in os.listdir(stock_dir):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market in ("SH", "SZ", "BJ") and code.isdigit():
            whitelist.add(f"{code}.{market}")
    print(f"[sync] 白名单（stock_data）{len(whitelist)} 只", flush=True)
    if not whitelist:
        print("[sync] 白名单为空，终止")
        return

    # 1) 汇总所有分区的 (code, date) → pct_chg / OHLCV
    rows: Dict[str, Dict[str, dict]] = {}
    t0 = time.time()
    for day, path in parts:
        try:
            df = pd.read_parquet(
                path,
                columns=["ts_code", "date", "open", "high", "low", "close",
                         "prev_close", "volume", "amount", "is_suspended"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  跳过 {day}: {e}")
            continue
        df["ts_code"] = df["ts_code"].astype(str)
        df = df[df["ts_code"].str.contains(r"\.(SH|SZ|BJ)$", regex=True)]
        # 白名单：只要我们自己的股票池。
        # TDX raw 一天有 5 万+ 条记录（指数、基金、可转债…），
        # 不加白名单会误建 5 万个文件。
        df = df[df["ts_code"].isin(whitelist)]
        for r in df.itertuples(index=False):
            code = r.ts_code
            stem = code.replace(".", "_")
            prev = float(r.prev_close) if r.prev_close == r.prev_close else 0.0
            close = float(r.close) if r.close == r.close else float("nan")
            if prev <= 0 or not np.isfinite(close) or close <= 0:
                continue
            pct = close / prev - 1.0
            rec = {
                "date": str(r.date)[:10],
                "open": float(r.open) if r.open == r.open else np.nan,
                "high": float(r.high) if r.high == r.high else np.nan,
                "low": float(r.low) if r.low == r.low else np.nan,
                "close": close,
                "volume": float(r.volume) if r.volume == r.volume else np.nan,
                "amount": float(r.amount) if r.amount == r.amount else np.nan,
                "pct_chg": pct * 100.0,   # 与 baostock 一致用百分数
                "suspended": int(r.is_suspended) if r.is_suspended == r.is_suspended else 0,
            }
            rows.setdefault(stem, {})[rec["date"]] = rec
        print(f"  {day}: 累计 {len(rows)} 只", flush=True)

    print(f"[sync] 汇总 {len(rows)} 只，{time.time()-t0:.0f}s", flush=True)

    # 2) 合并进 baostock_daily
    os.makedirs(OUT_DIR, exist_ok=True)
    updated = created = 0
    for stem, recs in rows.items():
        path = os.path.join(OUT_DIR, f"{stem}.parquet")
        new_rows = []
        for d, rec in sorted(recs.items()):
            new_rows.append({
                "date": pd.Timestamp(d),
                "code": stem.replace("_", "."),
                "open": rec["open"], "high": rec["high"], "low": rec["low"],
                "close": rec["close"], "volume": rec["volume"],
                "amount": rec["amount"], "pct_chg": rec["pct_chg"],
                "turn": np.nan, "is_st": 0, "trade_status": 0 if rec["suspended"] else 1,
                "pe_ttm": np.nan, "pb_mrq": np.nan, "ps_ttm": np.nan,
            })
        new_df = pd.DataFrame(new_rows)

        if os.path.exists(path):
            try:
                old = pd.read_parquet(path)
            except Exception:
                old = pd.DataFrame(columns=COLS)
            old["date"] = pd.to_datetime(old["date"])
            # ⚠️ 2026-11-02 修正：原来用 how="left" —— 只保留 **baostock 已有**的
            #   日期，TDX 有而 baostock（T+1 才更新）还没有的**当天**被整个丢弃
            #   → baostock_daily 里永远缺最近一个交易日
            #   → load_adjusted_close 合成后复权价时该日整行 NaN
            #   → 纸面盘 adj_cost 写成 NaN → 市值被冻结在买入金额（台账 §12.35）。
            #   现在用 outer：TDX 独有的日期整行补进来（close/OHLCV 取 TDX，
            #   turn/PE/PB 无来源置 NaN），两边都有的行以 TDX 的 pct_chg 为准。
            merged = old.merge(
                new_df,
                on="date", how="outer", suffixes=("", "_tdx"),
            )
            mask = merged["pct_chg_tdx"].notna()
            merged.loc[mask, "pct_chg"] = merged.loc[mask, "pct_chg_tdx"]
            merged.loc[mask, "trade_status"] = merged.loc[mask, "trade_status_tdx"]
            # 其余字段：baostock 优先，缺失时用 TDX 补（TDX 独有日期的关键路径）
            for _c in COLS:
                if _c in ("date", "pct_chg", "trade_status"):
                    continue
                _tc = f"{_c}_tdx"
                if _tc in merged.columns:
                    _m = merged[_c].isna() & merged[_tc].notna()
                    merged.loc[_m, _c] = merged.loc[_m, _tc]
            merged = merged.drop(
                columns=[c for c in merged.columns if c.endswith("_tdx")], errors="ignore"
            )
            out = merged[[c for c in COLS if c in merged.columns]]
            updated += 1
        else:
            out = new_df[COLS]
            created += 1
        out = out.sort_values("date").reset_index(drop=True)
        tmp = path + ".tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    print(f"[sync] 更新 {updated} 个文件，新建 {created} 个文件", flush=True)
    print(f"[sync] 输出目录 {OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
