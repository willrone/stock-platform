"""最新交易日行情 + 可成交性判定（自用「可下单清单」，2026-11-05）

建设项 C（docs/PERSONAL_USE_PIVOT_2026-11-05.md）：下单清单必须按**最新收盘**
重新给出参考金额，并把买不进/卖不出的标的**显式列出**，不能静默剔除。

口径（两条硬规定）：
1. 行情层 = `data/parquet/baostock_daily/<CODE_EXCHANGE>.parquet`
   未复权 OHLCV + 交易所口径涨跌幅 + is_st / trade_status，
   由 `scripts/sync_pct_chg_from_tdx.py` 维护（该层掉队会让整页价格过期）。
2. `pct_chg` 在这一层是**百分数**（写入处 `pct * 100.0`，注释「与 baostock 一致用百分数」），
   而 `execution_rules.check_tradability` 收的是**小数**（0.095 = +9.5%）→ 本模块负责换算。
   单位换算错的两种后果都很坏：+9.9% 传成 9.9 会被判成「涨跌幅异常、本日回避」，
   传成 0.099 则永远拦不住涨停。改这里前先确认口径。

与 `execution_plan.py` 的分工：`execution_plan.py` 是「重算撮合」（自己排单、算现金
约束），它没有任何调用方也没有测试；本模块只做「同一批数量、换最新价、判可成交」，
数量仍来自 `paper_portfolio.py` 的撮合结果 —— 否则下单清单会和纸面账本对不上。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from app.services.trading.execution_rules import check_tradability

# backend/app/services/trading/latest_quote.py → 上溯 4 层到项目根
# （trading → services → app → backend → <项目根>）。少一层会去找 backend/data/...
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DAILY_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
# ETF 腿（159934 / 511260 / 518880 …）不在 A 股日频层里，在 ETF 管道自己的层。
# 不接这一层，袖套的 9 笔订单会全部显示「无行情数据」。
ETF_DIR = os.path.join(ROOT, "data", "etf", "parquet")
BOARD_LOT = 100

_BAR_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "is_st",
    "trade_status",
)
_ETF_BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume")

# (path, mtime) → bar。行情文件每个交易日被覆写，用 mtime 失效。
_BAR_CACHE: Dict[str, Any] = {}
_CACHE_MAX = 800

# A 股 15:00 收盘。收盘前出现的「当日 bar」是盘中未完成 bar：
# 拿它判涨跌停会在盘中把一个暂时 +10% 的标的误判成涨停，
# 拿它当收盘价会让参考金额随盘中波动漂。→ 盘中 bar 不判可成交性，只给参考价。
_MARKET_CLOSE = (15, 10)


def _bar_is_intraday(bar_date: str) -> bool:
    try:
        d = datetime.strptime(str(bar_date)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    if d != datetime.now().date():
        return False
    now = datetime.now()
    return (now.hour, now.minute) < _MARKET_CLOSE


def _read_last_bar(path: str, columns: tuple, source: str) -> Optional[Dict[str, Any]]:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = f"{path}|{mtime}"
    if key in _BAR_CACHE:
        return _BAR_CACHE[key]
    try:
        have = set(pd.read_parquet(path).columns)
        cols = [c for c in columns if c in have]
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.sort_values("date")
    row = df.iloc[-1]
    bar: Dict[str, Any] = {
        "bar_date": str(row["date"])[:10],
        # 来源用显式标签，不要从路径里嗅「etf」字样 —— 测试临时目录名里带 etf
        # 就会把 A 股 bar 误标成 ETF（实测踩到过）
        "bar_source": source,
        "n_rows": int(len(df)),
    }
    for col in _BAR_COLUMNS:
        if col == "date":
            continue
        # 缺列显式置 None：ETF 层没有 amount/pct_chg，
        # 缺列若是「键不存在」会让上层 .get 与「值为 0」两种语义混淆
        bar[col] = _num(row[col]) if col in cols else None
    if len(_BAR_CACHE) > _CACHE_MAX:
        _BAR_CACHE.clear()
    _BAR_CACHE[key] = bar
    return bar


def bar_file(code: str, daily_dir: str = DAILY_DIR) -> str:
    """股票代码 → 行情层文件名（`000030.SZ` → `000030_SZ.parquet`）。"""
    return os.path.join(daily_dir, f"{code.replace('.', '_')}.parquet")


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_latest_bar(
    code: str,
    daily_dir: str = DAILY_DIR,
    *,
    etf_dir: str = ETF_DIR,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """读某标的在行情层里的**最后一个交易日** bar。缺文件/空文件 → None（如实缺，不猜价）。

    路由：带交易所后缀的（`600000.SH`）走 A 股层；裸 6 位代码（ETF 腿）走 ETF 层。
    任一层查不到时再回落到另一层，避免因代码写法不同而误报「无行情」。
    """
    code = str(code).strip()
    bare = code.split(".")[0]
    is_etf_shape = (
        "." not in code and len(bare) == 6 and bare[:2] in ("15", "51", "56", "58")
    )

    bars: List[Optional[Dict[str, Any]]] = []
    if is_etf_shape:
        bars.append(
            _read_last_bar(
                os.path.join(etf_dir, f"{bare}.parquet"), _ETF_BAR_COLUMNS, "etf"
            )
        )
        bars.append(
            _read_last_bar(bar_file(code, daily_dir), _BAR_COLUMNS, "baostock_daily")
        )
    else:
        bars.append(
            _read_last_bar(bar_file(code, daily_dir), _BAR_COLUMNS, "baostock_daily")
        )
        bars.append(
            _read_last_bar(
                os.path.join(etf_dir, f"{bare}.parquet"), _ETF_BAR_COLUMNS, "etf"
            )
        )

    bar = next((b for b in bars if b), None)
    if bar is None:
        return None
    bar = dict(bar)
    bar["code"] = code
    bar["intraday"] = _bar_is_intraday(bar.get("bar_date", ""))
    if use_cache:
        return bar
    return bar


def tradability_for_order(
    code: str,
    side: str,
    bar: Optional[Dict[str, Any]],
    *,
    name: str = "",
) -> Dict[str, Any]:
    """该笔订单在**最后一个有行情的交易日**能不能成交。

    Returns:
        {"tradable": bool | None, "tradable_reason": str, "price_as_of": str | None,
         "latest_price": float | None, "latest_pct_chg_pct": float | None}
        `tradable=None` 表示没有行情，**不是**「可以买」。
    """
    out: Dict[str, Any] = {
        "tradable": None,
        "tradable_reason": "",
        "price_as_of": None,
        "latest_price": None,
        "latest_pct_chg_pct": None,
    }
    if bar is None:
        out["tradable_reason"] = "无行情数据（行情层里没有该标的）"
        return out

    pct_pct = bar.get("pct_chg")
    out["price_as_of"] = bar.get("bar_date")
    out["latest_price"] = bar.get("close")
    out["latest_pct_chg_pct"] = pct_pct
    out["price_basis"] = "盘中未收盘" if bar.get("intraday") else "收盘价"
    out["quote_source"] = bar.get("bar_source")

    if bar.get("intraday"):
        # 盘中 bar：涨跌停判定不成立（暂时 +10% ≠ 封板），参考价也不是收盘价
        out["tradable"] = None
        out["tradable_reason"] = "当日行情未收盘（盘中 bar），收盘后才能判可成交性"
        return out

    t = check_tradability(
        code,
        name=name or "",
        close=bar.get("close"),
        open_price=bar.get("open"),
        high=bar.get("high"),
        low=bar.get("low"),
        # 百分数 → 小数（口径换算的唯一入口，见模块头第 2 条）
        pct_chg=(pct_pct / 100.0) if pct_pct is not None else None,
        volume=bar.get("volume"),
        amount=bar.get("amount"),
        trade_status=bar.get("trade_status"),
        is_st=bar.get("is_st"),
        pct_reliable=True,
    )
    if side == "sell":
        ok = bool(t.can_sell)
    else:
        ok = bool(t.can_buy)
    out["tradable"] = ok
    out["tradable_reason"] = "" if ok else (t.blocked_reason or t.reason or "不可成交")
    # ETF 层只有 OHLCV，没有交易所口径涨跌幅 → 涨跌停判定实际没跑。
    # 不写这一句，UI 上的「可成交」会被读成「已经判过涨跌停」。
    out["limit_check"] = pct_pct is not None
    if ok and pct_pct is None:
        out["tradable_reason"] = "有行情且当日有成交；该行情层无涨跌幅，涨跌停未判定"
    return out


def enrich_order_row(
    row: Dict[str, Any],
    side: str,
    *,
    portfolio_value: Optional[float] = None,
    daily_dir: str = DAILY_DIR,
    etf_dir: str = ETF_DIR,
    lot: int = BOARD_LOT,
) -> Dict[str, Any]:
    """给一笔买/卖订单补上：最新价、最新涨跌幅、可成交性、整手数量、参考金额、目标权重。

    **只增不改**：`price` / `shares` / `amount` 这些撮合口径字段原样保留
    （它们来自 paper_portfolio 的成交结果，改了就和纸面账本对不上）。
    """
    code = str(row.get("code", ""))
    name = str(row.get("name", "") or "")
    bar = load_latest_bar(code, daily_dir, etf_dir=etf_dir)
    shares = int(row.get("shares") or row.get("quantity") or 0)

    out: Dict[str, Any] = dict(row)
    out.update(tradability_for_order(code, side, bar, name=name))

    # 整手：纸面撮合本来就按整手下单，这里只做**校验**，不重新取整
    out["lot"] = lot
    out["lots"] = shares // lot if shares else 0
    out["lot_ok"] = bool(shares) and shares % lot == 0

    signal_amount = _num(row.get("amount"))
    out["signal_amount"] = signal_amount
    latest_price = out.get("latest_price")
    out["ref_amount_latest"] = (
        round(shares * latest_price, 2) if (latest_price and shares) else None
    )
    # 目标权重：用**信号日**成交额 / 信号日组合净值（不是最新市值口径，
    # 后者会把「这笔单占组合多少」变成「这只票今天值多少」，两件事）
    if portfolio_value and signal_amount:
        out["target_weight_pct"] = round(
            signal_amount / float(portfolio_value) * 100.0, 2
        )
    else:
        out["target_weight_pct"] = None
    return out


def enrich_orders(
    buys: Iterable[Dict[str, Any]],
    sells: Iterable[Dict[str, Any]],
    *,
    portfolio_value: Optional[float] = None,
    daily_dir: str = DAILY_DIR,
    etf_dir: str = ETF_DIR,
) -> Dict[str, Any]:
    """批量补字段 + 汇总：价格 as-of、不可成交笔数、缺行情笔数。"""
    eb = [
        enrich_order_row(
            r,
            "buy",
            portfolio_value=portfolio_value,
            daily_dir=daily_dir,
            etf_dir=etf_dir,
        )
        for r in buys
    ]
    es = [
        enrich_order_row(
            r,
            "sell",
            portfolio_value=portfolio_value,
            daily_dir=daily_dir,
            etf_dir=etf_dir,
        )
        for r in sells
    ]
    dates = [str(r["price_as_of"]) for r in (eb + es) if r.get("price_as_of")]
    all_rows: List[Dict[str, Any]] = eb + es
    # `price_as_of` 取**最旧**的一根 bar：整批清单能不能照着下单，取决于最陈旧的那一行。
    # 报最新值会把 9 行 11-04 的行情说成 11-05，那是假的新鲜。
    return {
        "buys": eb,
        "sells": es,
        "price_as_of": min(dates) if dates else None,
        "price_as_of_newest": max(dates) if dates else None,
        "n_price_dates": len(set(dates)),
        "n_not_tradable": sum(1 for r in all_rows if r.get("tradable") is False),
        "n_no_quote": sum(1 for r in all_rows if r.get("tradable") is None),
        "n_orders": len(all_rows),
    }
