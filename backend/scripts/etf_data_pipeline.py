#!/usr/bin/env python3
"""ETF 日线数据管道 —— 多资产腿（ETF 轮动/资产配置）的数据底座。

## 背景

`docs/backtest/CONCLUSION_2026-10-27.md` 的「决定性发现」+ 
`docs/backtest/POOL_PARTITION_2026-10-30.md` 共同证明：

**所有 A 股多头选股策略的日收益相关都在 0.90~0.96**（25 因子 / 42 因子 /
RSI / GBM / 行业分池 / 随机分池全部如此）→ 相关来自**共同的市场 beta**，
不是持仓重叠。→ A 股多头策略的夏普天花板在 **beta 层，不在选股层**。

所以下一步唯一未被试过、且机制上能突破 0.88~1.03 框架上限的路是
**引入非 beta 相关的资产**（黄金 / 债券 / 跨境 / 红利低波）。

本脚本建立这条路的 ETF 日线数据底座。

## 数据源（已在本机实测验证）

| 源 | 接口 | 用途 | 状态 |
|---|---|---|---|
| 腾讯 | `web.ifzq.gtimg.cn/appstock/app/fqkline/get` | **主源**，前复权日线 | ✅ 可用 |
| akshare | `fund_etf_hist_em(symbol, adjust='qfq')` | 兜底（单只） | ✅ 可用 |
| akshare | `fund_etf_spot_ths()` | **ETF 清单**（1772 只，带名称） | ✅ 可用 |
| akshare | `fund_etf_hist_sina` | 二级兜底 | 可用 |
| akshare | `fund_etf_spot_em` / `index_zh_a_hist` | — | ❌ 东财 IP 封禁 |

### 腾讯源的坑（已实测）

**`count` 是「结束端的条数上限」（取最近 N 条），不是总量。**
- `param=sh510300,day,2023-01-01,2026-10-30,320,qfq` → 返回**最近** 320 行（2025-07-04 起），
  不是 2023-01-01 起的 320 行
- `count=2000` → 返回 640 行（超过某阈值后行为不同步）
- `count=3000` + `start=2015-01-01` → `{"code":0,"msg":"param error","data":[]}`

→ **拉长历史必须分段倒推**：固定 start，用递减的 end 日期往前翻，每段取 ~700 条，
段间留重叠防漏。

## 产出

`data/etf/parquet/<CODE>.parquet` —— 每只 ETF 一个文件，列：
`date / open / high / low / close / volume`（前复权）

`data/etf/etf_universe.json` —— ETF 清单 + 资产类别标注

## 用法

    # 抓取指定 ETF
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/etf_data_pipeline.py --codes 510300,518880

    # 抓取资产类别池（推荐）
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/etf_data_pipeline.py --pool

    # 只更新最近（增量）
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/etf_data_pipeline.py --pool --incremental
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

ETF_DIR = os.path.join(ROOT, "data", "etf")
PARQUET_DIR = os.path.join(ETF_DIR, "parquet")
UNIVERSE_PATH = os.path.join(ETF_DIR, "etf_universe.json")

TENCENT_URL = (
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    "?param={code},day,{start},{end},{count},qfq"
)

#: 腾讯单段最大条数（实测 640~727 稳定，800 时行为可能不一致）
SEGMENT_SIZE = 700
#: 段间重叠天数（防边界漏行）
SEGMENT_OVERLAP_DAYS = 10

#: 资产类别 → 关键词（用于给 ETF 清单打标签）
ASSET_CLASSES: Dict[str, List[str]] = {
    "gold": ["黄金", "金ETF", "上海金"],
    "bond": ["债ETF", "国债ETF", "国债", "政金债", "信用债", "短融", "同业存单", "城投债"],
    "cross_border": ["纳指", "标普", "恒生", "日经", "德国", "法国", "沙特", "东南亚", "海外", "全球", "亚太", "美国", "日本"],
    "dividend": ["红利", "低波", "低波动", "价值", "央企", "国企"],
    "commodity": ["豆粕", "商品", "能源化工", "有色期货", "有色金属期货", "原油"],
    "a_share": ["沪深300", "中证500", "中证1000", "中证2000", "创业板", "科创", "上证50", "北证", "A50", "等权", "微盘", "中盘", "全指"],
}

#: 名称关键词 → 强制重定向到指定类别（**覆盖**关键词匹配的结果）。
#: 规则按顺序匹配，**只应用第一条命中的**（从具体到宽泛）。
#:
#: 为什么需要：纯关键词匹配会把**资产性质完全不同**的 ETF 混进同一类，
#: 而本项目的全部结论都建立在「类别 = 低相关资产」这个前提上。
#:
#: - 「黄金股 ETF」「黄金产业 ETF」跟踪的是**矿业公司股价**（股票 beta，
#:   与 A 股腿高相关），不是金价。混进 gold 类会让 gold 腿的相关性假设失效。
#: - 「有色金属**期货** ETF」跟踪**工业金属期货**（铜/铝/锌），属于商品。
#: - 「有色金属**主题/矿业** ETF」跟踪的是**A股有色板块股票**，属于股票，
#:   绝不能当商品 —— 它与 A 股腿高相关，混进 commodity 会直接毒化商品腿。
#:
#: 曾实测（2026-10-30）：159980 大成有色金属期货ETF 在两个窗口的成交额
#: 排名都进入 commodity 前 3，却因为关键词 "有色金属期货" 被标成 gold；
#: 而 26 只 "中证有色金属主题/矿业 ETF" 又会被宽泛的 "有色金属" 扫进来。
CLASS_OVERRIDES: List[Tuple[List[str], Optional[str]]] = [
    # 黄金股 / 黄金产业 → 单独一类（非 gold）
    (["黄金股", "黄金产业"], "gold_equity"),
    # 有色金属期货 → commodity
    (["有色金属期货", "有色期货"], "commodity"),
    # 有色金属主题/矿业（股票）→ a_share
    (["有色金属", "有色矿业", "工业有色"], "a_share"),
]


def _to_market_code(code: str) -> str:
    """6 位代码 → 腾讯格式（sh/sz/bj）。"""
    code = str(code).strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code.startswith(("5", "6", "9")):
        return "sh" + code
    if code.startswith(("1", "0", "2", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code


def _fetch_tencent(code: str, start: str, end: str, count: int) -> List[List[str]]:
    """拉一段腾讯日线。返回 [date, open, close, high, low, volume] 列表。"""
    import urllib.request

    url = TENCENT_URL.format(code=_to_market_code(code), start=start, end=end, count=count)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    data = payload.get("data")
    if not isinstance(data, dict) or not data:
        return []
    node = data.get(_to_market_code(code)) or list(data.values())[0]
    rows = node.get("qfqday") or node.get("day") or []
    return rows


def _fetch_tencent_full(
    code: str,
    start_date: date,
    end_date: date,
    max_segments: int = 12,
) -> List[Dict[str, Any]]:
    """分段倒推拉全历史。

    腾讯的 count 是「结束端条数上限」，所以要固定 start、递减 end 往前翻。
    每段取 SEGMENT_SIZE 条，段间留重叠。
    """
    rows_by_date: Dict[str, List[str]] = {}
    end = end_date
    for _ in range(max_segments):
        if end < start_date:
            break
        seg_start = max(start_date, end - timedelta(days=SEGMENT_SIZE * 2 + SEGMENT_OVERLAP_DAYS))
        try:
            rows = _fetch_tencent(code, seg_start.isoformat(), end.isoformat(), SEGMENT_SIZE)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] 段 {seg_start}~{end} 失败: {type(exc).__name__}: {exc}")
            rows = []
        if not rows:
            break
        for r in rows:
            if r and len(r) >= 6:
                rows_by_date[str(r[0])] = r
        oldest = min(str(r[0]) for r in rows if r)
        if oldest <= start_date.isoformat():
            break
        new_end = datetime.strptime(oldest, "%Y-%m-%d").date() - timedelta(days=SEGMENT_OVERLAP_DAYS)
        if new_end >= end:
            break  # 没前进，防死循环
        end = new_end
    return _rows_to_records(list(rows_by_date.values()))


def _rows_to_records(rows: List[List[str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            out.append(
                {
                    "date": str(r[0]),
                    "open": float(r[1]),
                    "close": float(r[2]),
                    "high": float(r[3]),
                    "low": float(r[4]),
                    "volume": float(r[5]) if r[5] not in ("", None) else None,
                }
            )
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda x: x["date"])
    return out


def _fetch_akshare(code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """akshare 兜底源。"""
    import akshare as ak
    import pandas as pd

    df = ak.fund_etf_hist_em(
        symbol=str(code),
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if df is None or df.empty:
        return []
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["date", "open", "close", "high", "low", "volume"]
    return df[[c for c in cols if c in df.columns]].to_dict("records")


def fetch_etf(
    code: str,
    start_date: date,
    end_date: date,
    allow_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """主源腾讯，失败则 akshare 兜底。"""
    recs = _fetch_tencent_full(code, start_date, end_date)
    if recs:
        return recs
    if allow_fallback:
        try:
            return _fetch_akshare(code, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] akshare 兜底失败 {code}: {type(exc).__name__}: {exc}")
    return []


def save_parquet(code: str, records: List[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None
    import pandas as pd

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    path = os.path.join(PARQUET_DIR, f"{code}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_existing(code: str) -> List[Dict[str, Any]]:
    path = os.path.join(PARQUET_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return []
    import pandas as pd

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.to_dict("records")


def tag_asset_class(name: str) -> List[str]:
    """按名称关键词打资产类别标签（一只 ETF 可多标签）。

    `CLASS_OVERRIDES` 里的规则**覆盖**关键词匹配：命中的关键词会把 ETF
    从误判的类别（gold）里踢出，并归入正确类别。
    """
    tags = []
    n = str(name)
    for cls, kws in ASSET_CLASSES.items():
        if any(k in n for k in kws):
            tags.append(cls)
    for kws, target in CLASS_OVERRIDES:
        if not any(k in n for k in kws):
            continue
        # 只应用第一条命中的规则（规则表按具体→宽泛排序）
        if "gold" in tags:
            tags.remove("gold")
        if "commodity" in tags and target != "commodity":
            tags.remove("commodity")
        if target and target not in tags:
            tags.append(target)
        break
    return tags


def build_universe() -> Dict[str, Any]:
    """从 akshare fund_etf_spot_ths() 建 ETF 清单 + 资产类别标注。"""
    import akshare as ak

    df = ak.fund_etf_spot_ths()
    df = df.copy()
    df["基金代码"] = df["基金代码"].astype(str).str.zfill(6)
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        code = str(row["基金代码"]).zfill(6)
        name = str(row.get("基金名称", ""))
        tags = tag_asset_class(name)
        out[code] = {"name": name, "tags": tags}
    payload = {
        "source": "akshare.fund_etf_spot_ths",
        "total": len(out),
        "asset_classes": ASSET_CLASSES,
        "etfs": out,
    }
    os.makedirs(ETF_DIR, exist_ok=True)
    with open(UNIVERSE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return payload


def select_pool(universe: Dict[str, Any], classes: List[str]) -> List[str]:
    """按资产类别挑 ETF 池（每类取成交额靠前的若干只由调用方决定）。"""
    etfs = universe["etfs"]
    picked: List[str] = []
    for code, meta in etfs.items():
        if any(c in meta["tags"] for c in classes):
            picked.append(code)
    return sorted(picked)


def retag_universe() -> Dict[str, Any]:
    """**离线**重打资产类别标签：读现有 etf_universe.json，用当前的
    `tag_asset_class` 重算 tags 并写回。

    为什么需要：`data/` 在 .gitignore 里，universe json 不进版本库。
    改了分类逻辑（`CLASS_OVERRIDES`）之后必须有一条**可复现**的离线命令
    把已有清单的标签刷新，否则新克隆的仓库仍带着旧标签（且旧标签里
    gold 类混着黄金股 ETF 和有色金属期货 ETF）。

    不需要网络：ETF 清单和名称都不变，只重算标签。
    """
    if not os.path.exists(UNIVERSE_PATH):
        raise SystemExit(f"❌ 找不到 {UNIVERSE_PATH}，先跑一次 build_universe")
    with open(UNIVERSE_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    changed = []
    for code, meta in payload.get("etfs", {}).items():
        old = list(meta.get("tags", []))
        new = tag_asset_class(meta.get("name", ""))
        if sorted(old) != sorted(new):
            changed.append((code, meta.get("name", ""), old, new))
        meta["tags"] = new
    payload["asset_classes"] = ASSET_CLASSES
    payload["class_overrides"] = [
        {"keywords": list(kws), "target": target} for kws, target in CLASS_OVERRIDES
    ]
    payload.setdefault("notes", {})["retag"] = (
        "离线重打标签：黄金股/黄金产业ETF→gold_equity；有色金属期货ETF→commodity；"
        "有色金属主题/矿业股票ETF→a_share"
    )
    with open(UNIVERSE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[retag] {len(changed)} 个 ETF 类别发生变化")
    for code, name, old, new in changed:
        print(f"  {code} {name:36s} {old} → {new}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="", help="逗号分隔的 6 位 ETF 代码")
    ap.add_argument("--pool", action="store_true", help="按资产类别池抓取")
    ap.add_argument("--classes", default="gold,bond,cross_border,dividend,a_share")
    ap.add_argument("--incremental", action="store_true", help="增量：只补最近缺口")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--limit-per-class", type=int, default=0, help="每类最多抓几只（0=不限）")
    ap.add_argument("--retag", action="store_true",
                    help="离线重打资产类别标签（不联网、不抓数据）")
    args = ap.parse_args()

    if args.retag:
        retag_universe()
        return

    os.makedirs(PARQUET_DIR, exist_ok=True)
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # 清单
    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, encoding="utf-8") as fh:
            universe = json.load(fh)
    else:
        print("[etf] 拉取 ETF 清单 (akshare fund_etf_spot_ths) ...", flush=True)
        universe = build_universe()
    print(f"[etf] ETF 清单 {universe['total']} 只", flush=True)

    # 目标代码
    codes: List[str] = []
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.pool:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        codes = select_pool(universe, classes)
        if args.limit_per_class > 0:
            # 按类别限额：每类先到先得
            seen: Dict[str, int] = {}
            limited: List[str] = []
            for code in codes:
                tags = universe["etfs"].get(code, {}).get("tags", [])
                primary = next((t for t in tags if t in classes), None)
                if primary is None:
                    continue
                if seen.get(primary, 0) >= args.limit_per_class:
                    continue
                seen[primary] = seen.get(primary, 0) + 1
                limited.append(code)
            codes = limited
        print(f"[etf] 目标池 {len(codes)} 只（类别 {classes}）", flush=True)
    else:
        ap.error("需要 --codes 或 --pool")

    ok, fail = 0, 0
    t0 = time.perf_counter()
    for i, code in enumerate(codes, 1):
        existing = load_existing(code) if args.incremental else []
        if args.incremental and existing:
            last = max(r["date"] for r in existing)
            # 从 last 前 10 天重拉（防边界漏行）
            s = max(start_date, datetime.strptime(last, "%Y-%m-%d").date() - timedelta(days=10))
        else:
            s = start_date
        recs = fetch_etf(code, s, end_date)
        if not recs:
            print(f"  [{i}/{len(codes)}] {code} ❌ 无数据", flush=True)
            fail += 1
            continue
        if existing:
            merged = {r["date"]: r for r in existing}
            merged.update({r["date"]: r for r in recs})
            recs = [merged[k] for k in sorted(merged)]
        path = save_parquet(code, recs)
        if path:
            ok += 1
            print(
                f"  [{i}/{len(codes)}] {code} ✅ {len(recs)} 行 "
                f"{recs[0]['date']}~{recs[-1]['date']}",
                flush=True,
            )
        else:
            fail += 1

    print(
        f"\n[etf] 完成：成功 {ok} / 失败 {fail}，耗时 {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    print(f"[etf] 数据目录 {PARQUET_DIR}")
    print(f"[etf] 清单 {UNIVERSE_PATH}")


if __name__ == "__main__":
    main()
