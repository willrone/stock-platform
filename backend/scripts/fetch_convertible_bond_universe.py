#!/usr/bin/env python3
"""抓取可转债宇宙登记表 + 存续池实时快照（转债数据入口 POC 第一步，2026-11-10）

## 为什么
转债是台账里唯一「未碰资产类」。任何截面研究的前置条件是**点位在-time 宇宙**。
东财登记报告给的是**全历史发行**（1066 只，其中 746 只已退市），
而不是「今天还在交易的 317 只」——后者会把幸存者偏差直接写进结论。

## 数据源事实（2026-11-10 本机实测，全部免密）
- 东财 datacenter `RPT_BOND_CB_LIST`：分页全量；含 LISTING_DATE / CEASE_DATE /
  DELIST_DATE / ACTUAL_ISSUE_SCALE / RATING / 票息说明 / 赎回与回售条款。
- 同一报告里 `TRANSFER_PRICE` / `REDEEM_TRIG_PRICE` / `RESALE_TRIG_PRICE`
  **1066 行全空**；`CURRENT_BOND_PRICENEW` 仅对 5 只未挂牌有值。
  → 登记报告**不含现价与转股价**，只做宇宙与合约静态。
- 已退市行的 `TRANSFER_PREMIUM_RATIO` 是占位值 100（不可用，必须过滤）。
- 实时快照走东财行情板块 `push2 clist fs=b:MK0354`（317 只存续）。

## 产出
- `data/cb/cb_universe_raw.json`   原始登记（审计用）
- `data/cb/cb_universe.parquet`    规范化全量（1066）
- `data/cb/cb_snapshot.parquet`    存续池实时快照 + 双低/剩余年限
- `data/cb/cb_universe_audit.json` 宇宙分层 + 字段缺失计数

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv/bin/python3 \\
    backend/scripts/fetch_convertible_bond_universe.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
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
    "Referer": "https://data.eastmoney.com/kzz/",
}
PAGE_SIZE = 200
STATUS_ALIVE = "存续"
STATUS_DEAD = "已退市/已到期"
STATUS_PRELIST = "未上市"
MARKET_SH = "CNSESH"
MARKET_SZ = "CNSESZ"
MARKET_THIRD_BOARD = "STAS00"


def classify_instrument(name: str, trade_market: str) -> str:
    """区分真转债 / 老三板退债 / 转退过渡码 / 分离交易债。

    登记报告 RPT_BOND_CB_LIST 里混着 23 行非可转债：
    08宝钢债这类分离交易债（附认股权证，本身不可转股）、
    404xxx 老三板退债、以及普利转退/蓝盾转退这类退市过渡码。
    不做这层闸门，截面里会混进纯债，还会把同一只债按两个代码数两遍。
    """
    if trade_market == MARKET_THIRD_BOARD:
        return "退债(老三板)"
    if "转退" in name:
        return "转债退市过渡码"
    if "转" in name:
        return "可转债"
    return "分离交易债/公司债"


def exchange_of(trade_market: str, code: str) -> str:
    """交易所 → 取价用的市场前缀。老三板暂无可用日线源，返回空串。"""
    if trade_market == MARKET_SH:
        return "sh"
    if trade_market == MARKET_SZ:
        return "sz"
    if trade_market == MARKET_THIRD_BOARD:
        return ""
    if code.startswith("11"):
        return "sh"
    if code.startswith("12"):
        return "sz"
    return ""


def _get_json(url: str, params: dict) -> dict:
    """GET 并解析 JSON。失败抛异常，由调用方决定是否重试。"""
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def fetch_registration() -> list[dict]:
    """分页拉取东财可转债全量登记报告。"""
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get_json(
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            {
                "reportName": "RPT_BOND_CB_LIST",
                "columns": "ALL",
                "pageNumber": page,
                "pageSize": PAGE_SIZE,
                "sortColumns": "PUBLIC_START_DATE",
                "sortTypes": -1,
                "source": "WEB",
                "client": "WEB",
            },
        )
        result = payload.get("result") or {}
        batch = result.get("data") or []
        rows.extend(batch)
        if not batch or page >= int(result.get("pages") or 1):
            return rows
        page += 1
        time.sleep(0.6)


def _as_date(value) -> str | None:
    """把 '2026-11-13 00:00:00' 归一成 'YYYY-MM-DD'，空值返回 None。"""
    if not isinstance(value, str) or not value.strip():
        return None
    return value[:10]


def classify(row: dict, today: str) -> str:
    """按上市/摘牌日给转债分层。"""
    listing = _as_date(row.get("LISTING_DATE"))
    delist = _as_date(row.get("DELIST_DATE"))
    if not listing or listing > today:
        return STATUS_PRELIST
    if delist and delist <= today:
        return STATUS_DEAD
    return STATUS_ALIVE


def normalize(row: dict, today: str) -> dict:
    """把登记行压成研究用的扁平结构（丢弃长文本条款，保留可算的字段）。"""
    code = row.get("SECURITY_CODE") or ""
    name = row.get("SECURITY_NAME_ABBR") or ""
    market = row.get("TRADE_MARKET") or ""
    return {
        "code": code,
        "secucode": row.get("SECUCODE"),
        "name": name,
        "trade_market": market,
        "stock_code": row.get("CONVERT_STOCK_CODE"),
        "listing_date": _as_date(row.get("LISTING_DATE")),
        "cease_date": _as_date(row.get("CEASE_DATE")),
        "expire_date": _as_date(row.get("EXPIRE_DATE")),
        "delist_date": _as_date(row.get("DELIST_DATE")),
        "transfer_start_date": _as_date(row.get("TRANSFER_START_DATE")),
        "transfer_end_date": _as_date(row.get("TRANSFER_END_DATE")),
        "issue_years": row.get("ISSUE_YEAR"),
        "coupon_years": row.get("BOND_EXPIRE"),
        "coupon_rate_text": row.get("INTEREST_RATE_EXPLAIN"),
        "issue_scale_actual": row.get("ACTUAL_ISSUE_SCALE"),
        "par_value": row.get("PAR_VALUE"),
        "initial_transfer_price": row.get("INITIAL_TRANSFER_PRICE"),
        "rating": row.get("RATING"),
        "transfer_value": row.get("TRANSFER_VALUE"),
        "transfer_premium_ratio": row.get("TRANSFER_PREMIUM_RATIO"),
        "status": classify(row, today),
        "instrument_class": classify_instrument(name, market),
        "exchange": exchange_of(market, code),
    }


def fetch_live_quotes() -> list[dict]:
    """存续转债实时快照（东财行情板块 MK0354）。"""
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get_json(
            "https://push2.eastmoney.com/api/qt/clist/get",
            {
                "pn": page,
                "pz": 100,
                "po": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fid": "f12",
                "fs": "b:MK0354",
                "fields": "f12,f14,f2,f3,f8,f10,f152",
            },
        )
        data = payload.get("data") or {}
        batch = data.get("diff") or []
        rows.extend(batch)
        if not batch or len(rows) >= int(data.get("total") or 0):
            return rows
        page += 1
        time.sleep(0.4)


def build_snapshot(universe: pd.DataFrame, quotes: list[dict], today: str) -> pd.DataFrame:
    """存续宇宙 × 实时行情 → 快照；双低 = 现价 + 溢价率。"""
    live = pd.DataFrame(quotes).rename(
        columns={
            "f12": "code",
            "f14": "quote_name",
            "f2": "close",
            "f3": "pct_chg",
            "f8": "turnover_rate",
            "f10": "volume_ratio",
        }
    )
    if live.empty:
        return live
    # 停牌/未开板的转债行情给的是字符串 "-"，不 coerce 会把整列变 object
    for col in ("close", "pct_chg", "turnover_rate", "volume_ratio"):
        live[col] = pd.to_numeric(live[col], errors="coerce")
    alive = universe[universe["status"] == STATUS_ALIVE].copy()
    snap = alive.merge(live[["code", "close", "pct_chg", "turnover_rate"]], on="code", how="left")

    premium = pd.to_numeric(snap["transfer_premium_ratio"], errors="coerce")
    price = pd.to_numeric(snap["close"], errors="coerce")
    snap["premium_rt"] = premium
    snap["double_low"] = price + premium
    snap["years_left"] = _years_left(snap["expire_date"], today)
    return snap


def _years_left(expire_dates, today: str) -> pd.Series:
    """剩余年限（按到期日 CEASE/EXPIRE 计算）。"""
    end = pd.to_datetime(expire_dates, errors="coerce")
    now = pd.Timestamp(today)
    return ((end - now).dt.days / 365.25).round(3)


def audit_table(df: pd.DataFrame) -> dict:
    """宇宙分层与关键字段缺失计数——闸门数据，不是装饰。"""
    total = len(df)
    missing = {
        col: int((df[col].isna() | (df[col].astype(str) == "None")).sum())
        for col in ["transfer_value", "transfer_premium_ratio", "rating", "issue_scale_actual"]
    }
    ratings = df["rating"].fillna("无").value_counts().head(15).to_dict()
    return {
        "total_rows": total,
        "by_status": df["status"].value_counts().to_dict(),
        "by_instrument_class": df["instrument_class"].value_counts().to_dict(),
        "issue_years": df["issue_years"].fillna("无").value_counts().sort_index().to_dict(),
        "rating_counts": ratings,
        "missing_counts": missing,
    }


def main() -> None:
    today = date.today().isoformat()
    CB_DIR.mkdir(parents=True, exist_ok=True)

    raw = fetch_registration()
    (CB_DIR / "cb_universe_raw.json").write_text(
        json.dumps({"fetched_date": today, "row_count": len(raw), "rows": raw},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    universe = pd.DataFrame([normalize(r, today) for r in raw])
    universe.to_parquet(CB_DIR / "cb_universe.parquet", index=False)

    snapshot = build_snapshot(universe, fetch_live_quotes(), today)
    snapshot.to_parquet(CB_DIR / "cb_snapshot.parquet", index=False)

    report = audit_table(universe)
    report["snapshot_rows"] = int(len(snapshot))
    report["fetched_date"] = today
    (CB_DIR / "cb_universe_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
