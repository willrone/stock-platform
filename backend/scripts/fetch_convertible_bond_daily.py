#!/usr/bin/env python3
"""抓取可转债日线（含已退市），并产出覆盖率闸门报告（转债 POC 第二步，2026-11-10）

## 为什么
东财行情 K 线接口对**已退市转债返回 0 根 bar**（实测 110076/113046/127025/128136，
换 market 前缀 0./1./2. 全部 data:null）。只用东财取价 = 只剩存续 315 只 =
746 只已退市（占全历史 70%）全部消失 = 幸存者偏差。
新浪 `CN_MarketData.getKLineData` 实测**保留已退市转债日线**，
且首日与登记 LISTING_DATE 完全对齐（抽样 20 只 gap=0）。

## 已知语义陷阱（不是数据缺失）
已退市转债的**最后一根 bar 早于 DELIST_DATE 1~47 个交易日**（转债先停止交易再摘牌，
DELIST_DATE 是摘牌日）。所以退市侧必须用「最后可得 bar」作为可交易终点日，
不能假设 bar 能铺到 DELIST_DATE。

## 产出
- `data/cb/cb_daily_bars.parquet`   code,date,open,high,low,close,volume
- `data/cb/cb_daily_coverage.json`  逐只覆盖率闸门（起点/终点/缺口/bar 数）

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv/bin/python3 \\
    backend/scripts/fetch_convertible_bond_daily.py [--status all|alive|dead] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "cb"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn",
}
REQUEST_PAUSE_S = 0.35
MAX_RETRIES = 3
TAKE_MAX = 4000


def sina_symbol(code: str, exchange: str = "") -> str:
    """转债代码 → 新浪 symbol。优先用登记表给的交易所，其次按代码段猜。

    115003 中兴债1 这种 11 开头但挂在深市的行，按代码段猜会取错市场。
    """
    if exchange in ("sh", "sz"):
        return f"{exchange}{code}"
    if code.startswith("11"):
        return f"sh{code}"
    if code.startswith("12"):
        return f"sz{code}"
    raise ValueError(f"无可用市场前缀: {code}")


def fetch_kline(code: str, exchange: str = "", take: int = TAKE_MAX) -> list[dict]:
    """取单只转债日线，失败重试 3 次后返回空列表。"""
    symbol = sina_symbol(code, exchange)
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={take}"
    )
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", "replace")
            return json.loads(body) or []
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def bars_to_rows(code: str, bars: list[dict]) -> list[dict]:
    """K 线对象 → 扁平行。"""
    rows = []
    for bar in bars:
        rows.append(
            {
                "code": code,
                "date": bar.get("day"),
                "open": _f(bar.get("open")),
                "high": _f(bar.get("high")),
                "low": _f(bar.get("low")),
                "close": _f(bar.get("close")),
                "volume": _f(bar.get("volume")),
            }
        )
    return rows


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _days_between(a: str, b: str) -> int:
    left = pd.Timestamp(a)
    right = pd.Timestamp(b)
    return int((right - left).days)


def coverage_row(univ: dict, bars: list[dict], today: str, skip_reason: str = "") -> dict:
    """逐只闸门：起点是否对齐上市日、终点距可交易终点多远、有没有断档。"""
    base = {
        "code": univ["code"],
        "name": univ["name"],
        "status": univ["status"],
        "instrument_class": univ.get("instrument_class"),
        "exchange": univ.get("exchange") or "",
        "skip_reason": skip_reason,
    }
    if not bars:
        return {**base, "bars": 0, "first_bar": None, "last_bar": None,
                "gap_start_days": None, "gap_end_days": None, "is_covered": False}
    first = bars[0].get("day", "")[:10]
    last = bars[-1].get("day", "")[:10]
    listing = univ.get("listing_date") or first
    delist = univ.get("delist_date") or today
    dates = sorted({b.get("day", "")[:10] for b in bars})
    return {
        **base,
        "bars": len(bars),
        "first_bar": first,
        "last_bar": last,
        "gap_start_days": _days_between(listing, first),
        "gap_end_days": _days_between(last, delist),
        "distinct_dates": len(dates),
        "has_duplicate_dates": len(dates) != len(bars),
        "is_covered": len(bars) > 0 and _days_between(listing, first) <= 10,
    }


def select_universe(status: str, limit: int | None) -> list[dict]:
    """从 cb_universe.parquet 读宇宙并按分层筛选。"""
    path = CB_DIR / "cb_universe.parquet"
    if not path.exists():
        raise SystemExit(f"缺少 {path}，先跑 fetch_convertible_bond_universe.py")
    df = pd.read_parquet(path)
    if status != "all":
        wanted = {"alive": "存续", "dead": "已退市/已到期"}[status]
        df = df[df["status"] == wanted]
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def ingest_one(univ: dict, today: str) -> tuple[list[dict], dict]:
    """单只转债：取 K 线 + 出闸门行（供线程池并发调用）。"""
    exchange = univ.get("exchange") or ""
    if not exchange:
        return [], coverage_row(univ, [], today, skip_reason="老三板无日线源")
    bars = fetch_kline(univ["code"], exchange)
    return bars_to_rows(univ["code"], bars), coverage_row(univ, bars, today)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="all", choices=["all", "alive", "dead"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    today = date.today().isoformat()
    CB_DIR.mkdir(parents=True, exist_ok=True)
    universe = select_universe(args.status, args.limit)

    rows: list[dict] = []
    audits: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ingest_one, u, today) for u in universe]
        for future in as_completed(futures):
            bar_rows, audit = future.result()
            rows.extend(bar_rows)
            audits.append(audit)
            done += 1
            if done % 100 == 0:
                print(f"progress {done}/{len(universe)}", flush=True)
            time.sleep(REQUEST_PAUSE_S)

    audits.sort(key=lambda a: a["code"])
    pd.DataFrame(rows).to_parquet(CB_DIR / "cb_daily_bars.parquet", index=False)
    covered = sum(1 for a in audits if a["is_covered"])
    summary = {
        "fetched_date": today,
        "requested": len(universe),
        "covered": covered,
        "coverage_rate": round(covered / max(len(universe), 1), 4),
        "total_bars": len(rows),
        "dead_last_bar_gap_days": _gap_stats(audits, "已退市/已到期"),
        "alive_last_bar_gap_days": _gap_stats(audits, "存续"),
        "coverage_by_instrument_class": _coverage_by_class(audits),
        "rows": audits,
    }
    (CB_DIR / "cb_daily_coverage.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))


def _coverage_by_class(audits: list[dict]) -> dict:
    """按券种类给覆盖率——非转债（分离债/老三板退债）本就不该进策略宇宙。"""
    out: dict[str, dict] = {}
    for a in audits:
        cls = a.get("instrument_class") or "未知"
        slot = out.setdefault(cls, {"requested": 0, "covered": 0})
        slot["requested"] += 1
        slot["covered"] += 1 if a["is_covered"] else 0
    for slot in out.values():
        slot["coverage_rate"] = round(slot["covered"] / max(slot["requested"], 1), 4)
    return out


def _gap_stats(audits: list[dict], status: str) -> dict:
    """已退市侧「最后 bar 距摘牌日」的分布——用于确认终点语义。"""
    gaps = sorted(a["gap_end_days"] for a in audits if a["status"] == status and a["bars"] > 0)
    if not gaps:
        return {}
    return {
        "n": len(gaps),
        "min": gaps[0],
        "p50": gaps[len(gaps) // 2],
        "p90": gaps[int(len(gaps) * 0.9)],
        "max": gaps[-1],
    }


if __name__ == "__main__":
    main()
