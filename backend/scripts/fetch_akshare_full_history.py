#!/usr/bin/env python3
"""补长窗口 + 补幸存者偏差：2012 起全 A 股（含已退市）日线，价格用 akshare(东财) 后复权。

为什么换源（实测，不是猜）：
- baostock 单只 2012→今 要 14–17 秒，5,565 只串行 ≈ 23 小时，并行被限流（实测 6 进程 4 分钟只落 5 只）。
- akshare `stock_zh_a_hist` 单只 15 年 0.2–0.4 秒，而且**退市股在架**：
  300104 乐视退 2012-01-04→2020-07-20；000018 神城A退 →2020-01-06；002231 *ST奥维 →2026-01-29。
- 宇宙与摘牌日期仍用 baostock 注册表（5,565 只，其中 341 只 status=0）——**退市可见性靠这张表**。

口径（和 §12.81/§12.83 一致）：
- `close` 列 = **未复权真实价**（照此算手数金额）；
- `close_hfq` 列 = **后复权收盘**（东财 hfq，基准锚在上市日，不随新数据重锚 → 可复现，不会像前复权那样除息重排历史）；
- 总回报一律用 `close_hfq` 的比值：r = close_hfq[t]/close_hfq[entry] − 1（含分红、含送转）；
- `pct_chg_raw` 经实测就是**未复权涨跌幅**（与 close 逐日反算的差 ≤0.005pp），所以它等于价格口径、不含分红送转，
  只可用于对账/发现除权日，**不许当总回报用**；
  另注意：**链式累计必须从第 2 根 bar 起**——首行涨跌幅是「相对发行价/上市前收」的变化（301018 首行 +241.62%），
  带着它连乘会把 +134% 的价比虚增成 +700%（本次踩过，已写进闸门）。

输出：data/parquet/akshare_full/<num>_<SH|SZ>.parquet，断点续传（已存在跳过），失败清单 _failures.txt
用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/fetch_akshare_full_history.py --limit 8   # 冒烟
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/fetch_akshare_full_history.py --workers 6  # 全量
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
OUT_DIR = os.path.join(ROOT, "data", "parquet", "akshare_full")
REGISTRY = os.path.join(ROOT, "data", "parquet", "baostock_full", "_registry.parquet")
FAILURES = os.path.join(OUT_DIR, "_failures.txt")
START = "20120101"
END = "20261110"
COLS = ["date", "code", "open", "high", "low", "close", "close_hfq", "volume", "amount",
        "pct_chg_raw", "turnover"]


def load_registry() -> pd.DataFrame:
    if not os.path.exists(REGISTRY):
        raise SystemExit(f"缺注册表 {REGISTRY} → 先跑 scripts/fetch_baostock_full_history.py（它只用来取注册表）")
    reg = pd.read_parquet(REGISTRY)
    reg["status"] = pd.to_numeric(reg["status"], errors="coerce")
    return reg


def out_name(num: str, mkt: str) -> str:
    return f"{num}_{mkt.upper()}.parquet"


def _hist(ak, symbol: str, adjust: str) -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=START,
                                    end_date=END, adjust=adjust)
            if df is None or len(df) == 0:
                return None
            return df
        except Exception:
            time.sleep(1.0 + attempt * 1.5 + random.random())
    return None


def fetch_one(symbol: str) -> tuple[str, int, str]:
    import akshare as ak
    raw = _hist(ak, symbol, "")
    if raw is None:
        return (symbol, 0, "raw empty/fail")
    hfq = _hist(ak, symbol, "hfq")
    if hfq is None:
        return (symbol, 0, "hfq empty/fail")
    mkt = "SH" if symbol.startswith(("6", "9")) else ("BJ" if symbol.startswith(("4", "8")) else "SZ")
    r = raw.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
                            "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg_raw", "换手率": "turnover"})
    h = hfq.rename(columns={"日期": "date", "收盘": "close_hfq"})
    r["date"] = pd.to_datetime(r["date"])
    h["date"] = pd.to_datetime(h["date"])
    df = r.merge(h[["date", "close_hfq"]], on="date", how="left")
    if df["close_hfq"].isna().mean() > 0.02:
        return (symbol, 0, f"hfq 对齐缺口 {df['close_hfq'].isna().mean():.1%}")
    for c in ("open", "high", "low", "close", "close_hfq", "volume", "amount", "pct_chg_raw", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = f"{symbol}.{mkt}"
    df = df[COLS].sort_values("date").drop_duplicates("date")
    path = os.path.join(OUT_DIR, out_name(symbol, mkt))
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return (symbol, len(df), "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-failures", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    reg["num"] = reg["code"].str.split(".").str[1]
    delisted = reg[reg["status"] != 1]
    print(f"注册表 {len(reg)} 只（上市 {(reg['status']==1).sum()}，已退市 {len(delisted)}）")

    todo = list(reg["num"])
    if args.only_failures and os.path.exists(FAILURES):
        bad = {ln.split(" ")[0] for ln in open(FAILURES) if ln.strip()}
        todo = [n for n in reg["num"] if n in bad]
    if not args.force:
        mkt_of = lambda n: "SH" if n.startswith(("6", "9")) else ("BJ" if n.startswith(("4", "8")) else "SZ")  # noqa: E731
        todo = [n for n in todo if not os.path.exists(os.path.join(OUT_DIR, out_name(n, mkt_of(n))))]
    if args.limit:
        todo = todo[: args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"待拉 {len(todo)} 只（raw+hfq 各一次），起点 {START}，{args.workers} 线程")
    if not todo:
        print("全部已存在")
        return 0

    # 摘牌日 < 2012 的老退市股本就没有 2012 后 bar —— 那是正常缺失，不能算拉取失败（会把失败率闸门污染）
    out_of = {r["num"]: str(r["outDate"]) for _, r in reg.iterrows()}
    expected_absent = {n for n, d in out_of.items() if d not in ("", "NaT", "nan") and d < "2012-01-01"}
    t0 = time.time()
    rows = fails = delisted_ok = absent_ok = 0
    delisted_nums = set(delisted["num"])
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, n): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, n, err = fut.result()
            if err:
                if sym in expected_absent:
                    absent_ok += 1
                else:
                    fails += 1
                    failures.append(f"{sym} fail:{err}")
            else:
                rows += n
                if sym in delisted_nums:
                    delisted_ok += 1
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] 行数 {rows} 失败 {fails} 正常缺失 {absent_ok} 退市股已拉 {delisted_ok} "
                      f"用时 {el:.0f}s 速率 {i/max(el,1):.2f}/s", flush=True)
            time.sleep(0.05 + random.random() * 0.1)      #  polite：东财对无节制并发会翻脸
    with open(FAILURES, "w") as fh:
        fh.write("\n".join(failures) + ("\n" if failures else ""))
    print(f"\n完成：成功 {len(todo)-fails-absent_ok}/{len(todo)}，行数 {rows}，失败 {fails}，"
          f"正常缺失(摘牌<2012) {absent_ok}，退市股成功 {delisted_ok}/{len(delisted)}")
    if failures:
        print(f"失败清单 → {FAILURES}")
    if fails > len(todo) * 0.1:
        print("警告：失败率 >10%，先 --only-failures 重试再采信覆盖率数字")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
