"""A 股前复权 — 通达信标准加法（线性）模型。

数据源: mootdx `xdxr` 接口（通达信分红送配事件表）
- 0.02s/只，免费无 token
- **注意**: mootdx 的 K 线命令 2026-09 起已失效（返回空，#52），
  但 `xdxr` 是不同模块，实测正常

为什么用加法模型而不是乘法因子:
A 股主流软件（通达信/同花顺/大智慧）与腾讯 qfq 都用**加法模型**。
实测 000001.SZ（平安银行，纯现金分红）:
- 腾讯隐含 adj 不是常数（0.9752~0.9786），乘法模型无法拟合
- 但 raw - qfq 恒等于 0.249，正是 2026-09-24 除权的每股红利
- 用加法模型推算 06-05: 10.98 - 0.36(06-12) - 0.249(09-24) = 10.371，
  腾讯实测 10.371，**完全一致**

乘法因子（新浪 hfq.js）在纯现金分红股上会引入 0.2~0.3% 误差。

公式（与 Shiniese/ashare、通达信一致，已对真实数据验证）:
    qfq(d) = raw(d) · M(d) − N(d)

    M(d) = Π (1 + 送转_e + 配股_e)        对所有 e > d
    N(d) = Σ B_e · Π (1 + 送转 + 配股)    对所有 e > d
    B_e  = 红利_e − 配股价_e · 配股_e

xdxr 字段（category=1 除权除息行）:
    fenhong      每 10 股现金分红（元）
    peigujia     配股价（元）
    songzhuangu  每 10 股送转股数
    peigu        每 10 股配股数

用法:
    from back_test_data_service.tdx_pipeline.adjust import AdjustmentEvent, fetch_events, to_qfq

    events = fetch_events("000001.SZ")
    qfq_price = to_qfq(raw_price=10.98, events=events, date="2026-06-05")
"""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass
class AdjustmentEvent:
    """一次除权除息事件。"""
    date: str        # 除权日 YYYY-MM-DD
    cash_per_share: float   # 每股现金红利（fenhong/10）
    rights_per_share: float # 每股送转股数（(songzhuangu+peigu)/10）
    peigu_price: float      # 配股价（peigujia，无配股时 0）
    peigu_per_share: float  # 每股配股数（peigu/10）

    @property
    def has_rights(self) -> bool:
        """是否有送转/配股（影响 M 乘数）。"""
        return self.rights_per_share != 0.0


def _fetch_xdxr(code6: str, timeout: int = 15) -> List[dict]:
    """调 mootdx xdxr 拿原始事件行。

    Args:
        code6: 6 位纯数字代码

    Returns:
        原始 dict 列表，每行含 year/month/day/category/fenhong/peigujia/songzhuangu/peigu
    """
    from mootdx.quotes import Quotes
    client = Quotes.factory(market="std", timeout=timeout)
    df = client.xdxr(symbol=code6)
    if df is None or len(df) == 0:
        return []
    # 只要 category==1（除权除息），其余（股本变化等）不参与复权
    sub = df[df["category"] == 1] if "category" in df.columns else df
    return sub.to_dict("records")


def fetch_events(ts_code: str, timeout: int = 15,
                 as_of: Optional[str] = None) -> List[AdjustmentEvent]:
    """拉取单只股票的除权除息事件（升序）。

    Args:
        ts_code: Tushare 风格，如 000001.SZ
        timeout: 超时秒数
        as_of: 截止日期 YYYY-MM-DD，只保留 ≤ 该日的事件。
               缺省 = 今天。**必须过滤未来除权** — 实测 000630 的 xdxr
               含 2026-10-26 的未来除权（0.06/股），若不排除会让所有
               历史日期的前复权价被多减 0.06（0.93% 偏差）。

    Returns:
        AdjustmentEvent 列表，按日期升序

    Raises:
        ValueError: 代码格式错误
    """
    if "." not in ts_code:
        raise ValueError(f"需要 Tushare 风格代码（如 000001.SZ）: {ts_code}")
    code6 = ts_code.split(".")[0]
    if as_of is None:
        as_of = date.today().isoformat()

    try:
        rows = _fetch_xdxr(code6, timeout=timeout)
    except Exception as e:
        raise ConnectionError(f"拉取 xdxr 失败 {ts_code}: {e}") from e

    events: List[AdjustmentEvent] = []
    for r in rows:
        try:
            y, m, d = int(r["year"]), int(r["month"]), int(r["day"])
            fenhong = float(r.get("fenhong") or 0.0)
            peigujia = float(r.get("peigujia") or 0.0)
            songzhuangu = float(r.get("songzhuangu") or 0.0)
            peigu = float(r.get("peigu") or 0.0)
        except (TypeError, ValueError):
            continue  # 跳过脏行

        ev_date = f"{y:04d}-{m:02d}-{d:02d}"
        if ev_date > as_of:
            continue  # 未来除权未发生，不参与复权

        events.append(AdjustmentEvent(
            date=ev_date,
            cash_per_share=fenhong / 10.0,
            rights_per_share=(songzhuangu + peigu) / 10.0,
            peigu_price=peigujia,
            peigu_per_share=peigu / 10.0,
        ))

    events.sort(key=lambda e: e.date)
    return events


def compute_adjustment(events: List[AdjustmentEvent], date: str) -> tuple:
    """计算某日期的复权系数 (M, N)。

    Args:
        events: 升序事件表
        date: 目标日期 YYYY-MM-DD

    Returns:
        (M, N) 使得 qfq(d) = raw(d) * M - N

    实现细节:
        只考虑 e > d 的事件（除权日当天及之后用不复权价）。
        从最近往回累乘，保证数值稳定。
    """
    m = 1.0
    n = 0.0
    # 从最近的事件往回处理（保证 N 的连乘正确）
    for e in reversed(events):
        if e.date <= date:
            break
        # B_e = 红利_e − 配股价_e · 配股_e
        b_e = e.cash_per_share - e.peigu_price * e.peigu_per_share
        # 先把已有 N 乘上本次的 (1+rights)
        n = n * (1.0 + e.rights_per_share) + b_e
        m *= (1.0 + e.rights_per_share)
    return m, n


def to_qfq(raw_price: float, events: List[AdjustmentEvent], date: str) -> float:
    """不复权价 → 前复权价。

    公式: qfq(d) = raw(d) · M(d) − N(d)
    """
    m, n = compute_adjustment(events, date)
    return raw_price * m - n


def to_hfq(raw_price: float, events: List[AdjustmentEvent], date: str) -> float:
    """不复权价 → 后复权价。

    后复权以最早交易日为基准。用前复权的逆变换 + 最早日的 (M,N) 归一。
    """
    if not events:
        return raw_price
    # 后复权: hfq(d) = (qfq(d) + N_ref) / M_ref，ref = 第一个除权日之前
    first = events[0].date
    m_ref, n_ref = compute_adjustment(events, first)
    qfq_d = to_qfq(raw_price, events, date)
    if m_ref == 0:
        return qfq_d
    return (qfq_d + n_ref) / m_ref


def factor_series(ts_code: str) -> List[AdjustmentEvent]:
    """拉取事件序列（别名，语义更清晰）。"""
    return fetch_events(ts_code)
