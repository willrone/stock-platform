#!/usr/bin/env python3
"""抓取官方市场指数日线，用作**无幸存者偏差**的市场基准。

## 为什么需要

平台自带的 `data/qlib_data/features_full/` 与 `data/parquet/stock_data/`
**都不含已退市股票**（600087/600247/600462/000033/000982 等全部缺失），
所以用它们构造的等权市场基准带幸存者偏差。

官方指数（中证全指 / 国证A指 / 沪深300 …）由交易所计算，
**成分股的历史变动（含退市）全部留在指数序列里** → 无幸存者偏差。

## 数据源

东方财富历史 K 线接口：

    https://push2his.eastmoney.com/api/qt/stock/kline/get
        ?secid={1|0}.{code}
        &fields1=f1,f2,f3,f4,f5
        &fields2=f51,f52,f53,f54,f55,f56,f57,f58
        &klt=101&fqt=1&beg=20130101&end=20261031&lmt=10000

`secid` 前缀：1 = 上交所（SH），0 = 深交所（SZ）。
返回的 `klines` 每行是 `日期,开,收,高,低,成交量,成交额,涨跌幅`。

## 网络受限时的兜底

某些网络环境（如本机）直连该接口会被 `RemoteDisconnected` 掐断。
这种情况下：

1. 用任何能访问该 URL 的通道取回原始 JSON（例如 Hermes 的 `web_extract` 工具）；
2. 把原始 JSON 存成文件，用 `--from-json <file>` 生成 parquet。

仓库里已经带了一份**离线快照**（2026-10-30 收盘）：

    backend/fixtures/market_benchmarks/IDX_399317.csv   # 国证A指
    backend/fixtures/market_benchmarks/IDX_000300.csv   # 沪深300

`multi_asset_p11_alpha_beta.py` 会**先读 `data/benchmarks/*.parquet`，
读不到再回退到这份离线快照**，所以整条链路不需要联网。

## 用法

    # 直接抓（联网可用时）
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \
        scripts/fetch_market_index_benchmarks.py

    # 从已保存的原始 JSON 生成
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \
        scripts/fetch_market_index_benchmarks.py --from-json /path/to/raw.json

    # 从仓库里的离线快照恢复 data/benchmarks/
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \
        scripts/fetch_market_index_benchmarks.py --from-fixtures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND_DIR)
OUT_DIR = os.path.join(ROOT, "data", "benchmarks")
FIXTURES = os.path.join(BACKEND_DIR, "fixtures", "market_benchmarks")

TARGETS = [
    ("1.000985", "中证全指"),
    ("0.399317", "国证A指"),
    ("1.000300", "沪深300"),
    ("1.000905", "中证500"),
    ("0.399303", "国证2000"),
    ("1.000852", "中证1000"),
]
FIELDS = "trade_date,open,close,high,low,volume"
# 历史遗留：早期这里写的是 "date"，而 `_rows_from_klines` 产出的列名是 **trade_date**，
# 于是 `df[[c for c in FIELDS.split(",") if c in df.columns]]` 会**把 trade_date 悄悄丢掉**，
# 写出缺日期列的 parquet → `load_market_proxies()` 的 `_ingest` 因缺 trade_date 直接跳过
# → **新基准被静默忽略**（P11/P12 都可能因此少一个基准口径而不报错）。
# 现在显式保证 trade_date 一定保留。


def _url(secid: str) -> str:
    return ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
            f"?secid={secid}&fields1=f1,f2,f3,f4,f5"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
            "&klt=101&fqt=1&beg=20130101&end=20261031&lmt=10000")


def _rows_from_klines(klines: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for k in klines:
        p = k.split(",")
        rows.append({
            "trade_date": p[0],
            "open": float(p[1]),
            "close": float(p[2]),
            "high": float(p[3]),
            "low": float(p[4]),
            "volume": float(p[5]),
        })
    return rows


def _rows_from_payload(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """接受三种形态：整包 {"data":{...}} / {"code": {...}} / {secid: {"klines":[...]}}"""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(payload.get("data"), dict) and payload["data"].get("klines"):
        d = payload["data"]
        secid = ("1." if str(d.get("code"))[0] == "0" or d.get("market") == 1 else "0.")
        secid = f"{d.get('market', 1)}.{d.get('code')}"
        out[secid] = _rows_from_klines(d["klines"])
        return out
    for secid, val in payload.items():
        if isinstance(val, dict) and val.get("klines"):
            out[secid] = _rows_from_klines(val["klines"])
        elif isinstance(val, dict) and "data" in val and isinstance(val["data"], dict):
            d = val["data"]
            if d.get("klines"):
                out[f"{d.get('market', 1)}.{d.get('code')}"] = _rows_from_klines(d["klines"])
    return out


def _write(secid: str, rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    os.makedirs(OUT_DIR, exist_ok=True)
    code = secid.split(".")[1]
    df = pd.DataFrame(rows)
    df = df[[c for c in FIELDS.split(",") if c in df.columns]]
    path = os.path.join(OUT_DIR, f"IDX_{code}.parquet")
    df.to_parquet(path, index=False)
    return path


def from_network() -> int:
    import time
    ok = 0
    for secid, name in TARGETS:
        path = os.path.join(OUT_DIR, f"IDX_{secid.split('.')[1]}.parquet")
        if os.path.exists(path):
            print(f"  已存在，跳过 {secid} {name}")
            ok += 1
            continue
        try:
            req = urllib.request.Request(_url(secid), headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Referer": "https://quote.eastmoney.com/",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8", "ignore"))
            data = payload.get("data") or {}
            if not data.get("klines"):
                print(f"  ✗ {secid} {name}: 空返回")
                continue
            p = _write(f"{data.get('market', 1)}.{data.get('code')}",
                       _rows_from_klines(data["klines"]))
            print(f"  ✓ {secid} {name}: {len(data['klines'])} 根 → {p}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {secid} {name}: {type(exc).__name__}: {str(exc)[:110]}")
        time.sleep(1.2)
    return ok


def from_json(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    groups = _rows_from_payload(payload)
    if not groups:
        print("  ✗ 未能从 JSON 里解析出 klines")
        return 0
    ok = 0
    for secid, rows in groups.items():
        p = _write(secid, rows)
        print(f"  ✓ {secid}: {len(rows)} 根 → {p}")
        ok += 1
    return ok


def from_fixtures() -> int:
    import pandas as pd
    if not os.path.isdir(FIXTURES):
        print(f"  ✗ 无离线快照目录 {FIXTURES}")
        return 0
    ok = 0
    for fn in sorted(os.listdir(FIXTURES)):
        if not fn.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(FIXTURES, fn))
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, fn[:-4] + ".parquet")
        df.to_parquet(out, index=False)
        print(f"  ✓ {fn}: {len(df)} 根 → {out}")
        ok += 1
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", default="")
    ap.add_argument("--from-fixtures", action="store_true")
    args = ap.parse_args()
    if args.from_json:
        n = from_json(args.from_json)
    elif args.from_fixtures:
        n = from_fixtures()
    else:
        n = from_network()
    print(f"完成：{n} 个指数")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
