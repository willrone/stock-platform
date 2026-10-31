"""多资产组合信号线（multi_asset_final）—— ETF 袖套腿的核心逻辑。

定稿口径（`docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md` §十 + P18 结论）：

    - A 股腿 50%：42 因子等权复合（`honest_composite` 同源），TopK 60 / n_drop 5，周频
    - ETF 袖套 50%：gold / bond / commodity 三类各 1/3，类内等权，
      固定持仓集，**每月（21 交易日）再平衡回类间等权**
    - 组合 = 两个子账户**静态 50/50 划分，腿间不再平衡**

为什么「静态划分」就是研究口径：
研究侧的组合是 `multi_asset_leg_experiment.merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])`
—— 两腿净值各自归一化到 1.0 后**线性相加**。这在账户层面等价于：
期初把资金一次性分成 50/50 两个子账户，之后各子账户自己涨，比例自然漂移、
不再互调。**不是**日频再平衡（那会让两条腿的相对权重每天都回到 50/50）。
所以研究口径本身就是实盘可实现的，不需要另立口径。

袖套内部则是**真再平衡**：每月把三类拉回各 1/3、类内 3 只等权。
P17 的结论支持这个动作（类间定期再平衡在两个窗口都同向为正）。

本模块只负责**袖套腿**。A 股腿复用 `paper_portfolio.run_line` 的既有实现
（同一套撮合逻辑只能有一份实现 —— 见 skill「撮合器与基线门」）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# backend/app/services/trading/multi_asset_service.py → 上溯 4 层到项目根
# （trading → services → app → backend → <项目根>）
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
#: **实时** ETF 行情目录（实盘/产品用；实盘周任务会全量重拉更新）
ETF_PARQUET = os.path.join(ROOT, "data", "etf", "parquet")
#: **冻结快照**（研究侧用）—— 前复权序列在分红时会整体重锚，
#: 实盘重拉会让历史价格偏移 → 研究数字无法复现，所以研究读这一份。
ETF_PARQUET_FROZEN = os.path.join(ROOT, "data", "etf", "parquet_frozen")

# ── 资产类别与固定持仓集 ────────────────────────────────────────────────
# 固定持仓集（P3 定稿）：选入依据是**资产性质**，不是当前流动性排名。
#   gold      黄金 ETF —— 与 A 股低相关、对冲货币/风险偏好
#   bond      长久期利率债 —— 负股票 beta（剔除了可转债 511380，它是「带债底的股票」）
#   commodity 商品 ETF —— 但它是「商品 beta」不是独立资产（P2 已核实 ρ）
SLEEVE_CLASSES: Tuple[str, ...] = ("gold", "bond", "commodity")

SLEEVE_HELD: Dict[str, List[str]] = {
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
    "commodity": ["159980", "159985", "159981"],
}

# 每月再平衡（21 个交易日），与研究侧 P3.build_targets(reb_every=21) 同口径
SLEEVE_REB_EVERY = 21

# 成本模型：ETF **免印花税**（只有 A 股卖出才收万 5）
# 佣金/滑点沿用研究侧 L3 的口径（0.0003 / 0.0001），
# 而 A 股腿的纸面盘用滑点 0.001（更保守）—— 两条腿成本口径不同是**故意的**：
# ETF 流动性远好于个股小盘股，用万 1 滑点是对研究数字的忠实还原。
ETF_COMMISSION = 0.0003
ETF_SLIPPAGE = 0.0001
ETF_BOARD_LOT = 100

#: 成交时允许的行情滞后上限（自然日）。
#: 估值可以无限前向填充（否则缺口日净值暴跌）；
#: 成交**不行** —— 拿 6 天前的价格下单是假的，必须挡住。
MAX_STALE_TRADE_DAYS = 3

# 袖套内部单类上限（类内等权后每只 = 1/3 类权重 / 3 只 = 1/9）
SLEEVE_MAX_WEIGHT = 1.0 / len(SLEEVE_CLASSES) / 3.0


class SleeveError(RuntimeError):
    """袖套腿的口径异常（数据缺失、持仓集退化等）。"""


# ── 数据 ────────────────────────────────────────────────────────────────
class EtfBars:
    """ETF 日线的快速索引（numpy + searchsorted），支持**前向填充**。

    为什么必须前向填充：ETF 行情文件里存在**成段的缺口**
    （实测 2023-11-28 ~ 2023-12-04 共 6 个交易日，9 只 ETF 同时缺失，
    是管道那一批抓漏了）。研究侧 `L3._px` 用 bisect 取「不晚于该日的最后一条」
    → 缺口日沿用前收盘价，净值不跳变。

    如果按精确匹配取价，缺口日会把持仓估值成 0 → **净值当天暴跌 ~98%**，
    后面所有夏普/回撤指标全废（实测踩到过）。
    → 估值一律前向填充；**成交**则不允许用过期价（见 `max_stale_days`）。
    """

    __slots__ = ("_idx", "frames")

    def __init__(self, frames: Dict[str, pd.DataFrame]):
        self.frames = frames
        self._idx: Dict[str, Dict[str, np.ndarray]] = {}
        for code, df in frames.items():
            self._idx[code] = {
                "dates": df["date"].to_numpy(dtype="datetime64[ns]"),
                "open": df["open"].to_numpy(dtype=float),
                "high": df["high"].to_numpy(dtype=float),
                "low": df["low"].to_numpy(dtype=float),
                "close": df["close"].to_numpy(dtype=float),
                "volume": df["volume"].to_numpy(dtype=float),
            }

    @property
    def codes(self) -> List[str]:
        return list(self._idx.keys())

    def get(
        self,
        code: str,
        date: pd.Timestamp,
        price_field: str = "close",
        exact: bool = False,
        max_stale_days: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """取某只 ETF 在某日的行情。

        - `exact=True`：必须当天有数据，否则 None
        - 否则前向填充（取不晚于 date 的最后一条）
        - `max_stale_days`：填充出来的价格距 date 超过这个自然日数就返回 None
          （**成交**用这个：不允许拿 6 天前的价格去下单）
        """
        e = self._idx.get(code)
        if e is None:
            return None
        target = np.datetime64(pd.Timestamp(date))
        i = int(np.searchsorted(e["dates"], target, side="right")) - 1
        if i < 0:
            return None
        # 2026-11-01 修正（潜伏 bug）：原来写成
        #     if exact or e["dates"][i] != target:
        #         if exact: return None
        #   → `exact=True` 时**无论当天有没有数据都返回 None**，
        #   「必须当天有数据」这个语义从来没生效过（这条分支此前无人调用）。
        # 现在按语义拆开：exact 只看当天；否则按 max_stale_days 卡陈旧度。
        if exact:
            if e["dates"][i] != target:
                return None
        elif max_stale_days is not None:
            stale_days = int((target - e["dates"][i]) / np.timedelta64(1, "D"))
            if stale_days > max_stale_days:
                return None
        px = float(e[price_field][i]) if price_field in e else float(e["close"][i])
        vol = float(e["volume"][i])
        return {
            "open": float(e["open"][i]),
            "high": float(e["high"][i]),
            "low": float(e["low"][i]),
            "close": float(e["close"][i]),
            "volume": vol if np.isfinite(vol) else 0.0,
            "price": px,
            "last_date": str(pd.Timestamp(e["dates"][i]).date()),
            "stale": bool(e["dates"][i] != target),
        }


def load_etf_bars(codes: List[str], base_dir: Optional[str] = None) -> EtfBars:
    """读 ETF parquet → `EtfBars`（找不到的代码直接跳过）。

    `base_dir` 默认实时目录（实盘）；验证/复现研究数字时传 `ETF_PARQUET_FROZEN`。

    调用方必须自己校验「实际拿到的持仓集 == 定稿持仓集」，
    否则会出现**持仓集退化**（少一只）而静默把权重摊到别的票上。
    """
    base_dir = base_dir or ETF_PARQUET
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        path = os.path.join(base_dir, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        frames[code] = df
    return EtfBars(frames)


def missing_codes(codes: List[str], base_dir: Optional[str] = None) -> List[str]:
    """返回没有本地数据的代码（调用方必须硬失败，不要静默降级）。"""
    base_dir = base_dir or ETF_PARQUET
    return [
        c for c in codes
        if not os.path.exists(os.path.join(base_dir, f"{c}.parquet"))
    ]


# ── 目标权重 ────────────────────────────────────────────────────────────
def sleeve_target_weights(held: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
    """袖套目标权重：三类各 1/3，类内等权。

    `held` 默认用定稿持仓集；传入自定义持仓集时（研究实验用）按同样规则展开。
    """
    held = held or SLEEVE_HELD
    w: Dict[str, float] = {}
    cls_w = 1.0 / len(SLEEVE_CLASSES)
    for cls in SLEEVE_CLASSES:
        codes = held.get(cls) or []
        if not codes:
            continue
        for c in codes:
            w[c] = cls_w / len(codes)
    return w


def sleeve_class_weights(held: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
    """三类各自的目标权重（用于展示「类间等权」是否达成）。"""
    held = held or SLEEVE_HELD
    return {
        cls: 1.0 / len(SLEEVE_CLASSES)
        for cls in SLEEVE_CLASSES
        if held.get(cls)
    }


def is_rebalance_day(index: int, anchor: int = 0) -> bool:
    """第 index 个交易日（0 基）是否是袖套调仓日。

    与研究侧 `P3.build_targets` 同口径：每 `reb_every` 个交易日调一次
    （21 个交易日 ≈ 一个月）。研究侧的网格是绝对序号 {1, 22, 43, ...}；
    实盘改成**相对锚点**（锚点 = 袖套首次建仓那一日的序号），
    这样实盘不依赖研究日历的起始日 —— 否则研究日历与实盘日历差一天，
    整个调仓网格就永久错位。

    注意：调仓日是**交易日序号**而不是「每月 1 号」（按自然月的掩码会因为
    1 号常是节假日而退化成 17 个执行日，研究侧已确认这个坑）。
    """
    return index >= anchor and (index - anchor) % SLEEVE_REB_EVERY == 0


# ── 调仓 ────────────────────────────────────────────────────────────────

def _record_buy(
    positions: Dict[str, Dict[str, Any]],
    buys: List[Dict[str, Any]],
    code: str,
    p: float,
    buy_px: float,
    shares: float,
    amount: float,
    cost: float,
    date: Any,
) -> None:
    """记一笔买入：更新持仓（加权平均成本）+ 追加成交明细。"""
    prev = positions.get(code) or {}
    prev_shares = float(prev.get("shares", 0))
    prev_cost = float(prev.get("cost", 0) or 0)
    new_shares = prev_shares + shares
    new_cost = (
        (prev_cost * prev_shares + amount) / new_shares
        if new_shares > 0 else p
    )
    positions[code] = {
        "shares": new_shares,
        "cost": round(new_cost, 6),
        "buy_date": str(date.date()),
        "last_buy_date": str(date.date()),
        "is_etf": True,
    }
    buys.append({
        "code": code,
        "price": round(p, 4),
        "shares": int(shares // ETF_BOARD_LOT * ETF_BOARD_LOT) or int(shares),
        "amount": round(amount, 2),
        "commission": round(cost, 2),
        "stamp_tax": 0.0,
        "slippage": round(amount * ETF_SLIPPAGE / (1 + ETF_SLIPPAGE), 2),
        "net": round(amount + cost, 2),
        "reason": "袖套调仓：补足目标权重",
    })


def _plan_sleeve_lots(
    target: Dict[str, float],
    value: float,
    px: Dict[str, float],
    wants: Dict[str, float],
    held_value: Dict[str, float],
    cash: float,
    board_lot: int,
) -> Dict[str, int]:
    """算出每只该买多少**手**（整手约束下最贴近「类间等权」的方案）。

    为什么必须优化，而不是逐只向下取整：
      袖套每只目标 = 净值/9（50 万 → 55,556 元），但 ETF 的**一手金额**差 66 倍：
      159981 商品 ETF 一手 ≈ 181 元，511010 国债 ETF 一手 ≈ 14,100 元
      （一手就等于单只目标的 25%）。所以"每只 1/9"在债券类根本凑不出来 ——
      逐只向下取整会让债券类系统性欠配（实测类权重偏到 32.1%）。

    做法（局部搜索）：
      1) 起点 = 逐只向下取整；
      2) 反复尝试「某只 ±1 手」的移动，取**类权重最大偏离最小**、
         且总成本 ≤ 现金的那一个；没有改进就停。

    目标函数用「三类权重相对 1/3 的最大偏离」，而不是「总额误差」——
    因为策略的承诺是**类间等权**（gold/bond/commodity 各 1/3），
    总额差 1% 无所谓，类权重偏 1 个点才是违约。

    返回 {code: 手数}（只包含需要买的标的）。
    """
    if board_lot <= 0:
        return {}

    codes = [c for c in wants if c in px]
    if not codes:
        return {}

    def _evaluate(lots_map: Dict[str, int]) -> Any:
        """→ (类权重最大偏离, 总成本, {类: 金额})"""
        class_amt: Dict[str, float] = {c: 0.0 for c in SLEEVE_CLASSES}
        cost = 0.0
        for c, l in lots_map.items():
            amt = l * board_lot * px[c] * (1 + ETF_SLIPPAGE)
            cls = next((k for k, lst in SLEEVE_HELD.items() if c in lst), None)
            if cls is not None:
                class_amt[cls] += amt
            cost += amt * (1 + ETF_COMMISSION)
        # 已持有（本次不买）的市值也计入类权重
        for cls in SLEEVE_CLASSES:
            for code in SLEEVE_HELD[cls]:
                if code not in lots_map:
                    class_amt[cls] += held_value.get(code, 0.0)
        total = sum(class_amt.values())
        if total <= 0:
            return 9.99, cost, class_amt
        dev = max(abs(class_amt[c] / total - 1 / 3) for c in SLEEVE_CLASSES)
        return dev, cost, class_amt

    # ── 起点：逐只向下取整 ────────────────────────────────────────────
    lots_map: Dict[str, int] = {}
    for c in codes:
        want = wants[c]
        buy_px = px[c] * (1 + ETF_SLIPPAGE)
        lots_map[c] = max(0, int((want / buy_px) // board_lot))

    best_dev, best_cost, _ = _evaluate(lots_map)
    # ── 局部搜索：每次取一个改进最大的 ±1 手移动 ──────────────────────
    for _ in range(200):
        move = None
        move_key = (best_dev, best_cost)
        for c in codes:
            for delta in (1, -1):
                nl = lots_map[c] + delta
                if nl < 0:
                    continue
                trial = dict(lots_map)
                trial[c] = nl
                dev, cost, _amt = _evaluate(trial)
                if cost > cash + 1e-6:
                    continue
                # 优先最小化类权重偏离；并列时取成本更低（闲置更少）
                key = (round(dev, 12), cost)
                if key < move_key:
                    move_key = key
                    move = (trial, dev, cost)
        if move is None:
            break
        lots_map, best_dev, best_cost = move[0], move[1], move[2]

    return {c: l for c, l in lots_map.items() if l > 0}


def _iter_product(per_code: List[List[Any]]) -> Any:
    """所有候选项的笛卡尔积（生成器）。"""
    if not per_code:
        yield []
        return
    head, rest = per_code[0], per_code[1:]
    for item in head:
        for tail in _iter_product(rest):
            yield [item] + tail



def rebalance_sleeve(
    positions: Dict[str, Dict[str, Any]],
    cash: float,
    bars: EtfBars,
    date: pd.Timestamp,
    target: Dict[str, float],
    price_field: str = "close",
    board_lot: int = ETF_BOARD_LOT,
    charge_sell_commission: bool = True,
    require_exact_date: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把袖套调仓到目标权重。

    撮合语义与研究侧 `L3.simulate` **逐步对齐**（同一套逻辑只能有一份实现）：
      1. 估值：现金 + Σ(股数 × 当日收盘价)
      2. 先卖：不在目标里的持仓全部卖出
      3. 买：只买**不足**的部分；现金约束下**按比例缩放全部买入额**
         （不能"先到先得" —— 那会让遍历顺序决定谁被削减，
          P6 实测同一目标权重表仅顺序不同就差 0.0055 夏普）
      4. 再平衡：卖出**超配**的部分

    成交价 = 当日收盘价 ± 滑点（与纸面盘 A 股腿同口径：成交日收盘价成交）。

    `charge_sell_commission`：卖出是否扣佣金。
      - True（**产品口径**）：买卖都扣，与 A 股腿一致，也是正确的做法。
      - False：只用于与研究侧 `L3.simulate` 做**逐点等价性验证** ——
        研究侧那段代码 `cash += proceeds`（卖出**不扣**佣金，佣金只累加进
        `total_cost`），属于研究侧的一处口径不一致，会让 3 年累计净值虚高
        约 2.8bp（卖方佣金 ≈ 0.925 倍换手 × 0.03%）。量级可忽略，
        但**验证时必须对齐**，否则会被误判成"实现不等价"。

    `require_exact_date`：**产品（纸面盘）侧必须置 True**。
      - True：成交价必须是**成交当日**的收盘价，缺当日 bar 直接抛错。
      - False（研究侧默认）：允许前向填充，落后 ≤ `MAX_STALE_TRADE_DAYS` 个自然日。
      为什么产品侧必须严格：`MAX_STALE_TRADE_DAYS = 3`（自然日）会**放过**
      「周五 → 周一」这正好 3 天的缺口 —— ETF 数据晚一天到货时，
      会拿**上周五的收盘价**当周一的成交价成交：账面上「成交了」，
      实际成交价比回测口径晚 1~3 天，净值曲线直接失真。
      研究侧不受影响：实测研究日历 35 个调仓日 × 9 只 = 315 个组合
      **全部是当日价**，所以「严格」对研究侧结果零影响。
    """
    positions = {k: dict(v) for k, v in (positions or {}).items()}
    cash = float(cash)
    buys: List[Dict[str, Any]] = []
    sells: List[Dict[str, Any]] = []

    px: Dict[str, float] = {}
    for code in set(positions) | set(target):
        bar = bars.get(code, date, price_field, max_stale_days=MAX_STALE_TRADE_DAYS)
        if bar is None:
            continue
        px[code] = float(bar["price"])

    # 目标里必须有价格，否则无法成交 → 显式失败，不静默跳过
    missing = [c for c in target if c not in px]
    if missing:
        raise SleeveError(
            f"袖套调仓失败：目标标的缺少 {date.date()} 的可用行情 {missing}"
            f"（允许最多落后 {MAX_STALE_TRADE_DAYS} 个自然日）"
        )

    # ── 产品侧严格口径：成交价必须是**成交当日**的收盘价 ────────────────
    # 见函数 docstring 的 `require_exact_date` 说明：容忍 3 个自然日会放过
    # 「周五 → 周一」的缺口 → ETF 数据晚到时拿上周五的收盘价成交（假成交）。
    if require_exact_date:
        not_exact = [c for c in target if bars.get(c, date, price_field, exact=True) is None]
        if not_exact:
            raise SleeveError(
                f"袖套调仓失败：目标标的 {not_exact} 在 {date.date()} **没有当天的行情**"
                f"（产品侧不允许用更早的收盘价成交）—— ETF 数据管道可能还没更新到成交日，"
                f"等数据到货后重跑。"
            )

    value = cash + sum(
        float(positions[c].get("shares", 0)) * px[c]
        for c in positions
        if c in px
    )

    # ── 1) 卖出不在目标里的 ────────────────────────────────────────────
    for code in list(positions.keys()):
        if code in target:
            continue
        if code not in px:
            continue
        shares = float(positions[code].get("shares", 0))
        if shares <= 0:
            del positions[code]
            continue
        p = px[code]
        sell_px = p * (1 - ETF_SLIPPAGE)
        proceeds = shares * sell_px
        commission = proceeds * ETF_COMMISSION
        cost_amt = float(positions[code].get("cost", 0) or 0) * shares
        cash += proceeds - (commission if charge_sell_commission else 0.0)
        realized = proceeds - commission - cost_amt
        sells.append({
            "code": code,
            "price": round(p, 4),
            "shares": int(shares),
            "amount": round(proceeds, 2),
            "cost_amount": round(cost_amt, 2),
            "realized_pnl": round(realized, 2),
            "commission": round(commission, 2),
            "stamp_tax": 0.0,
            "slippage": round(proceeds * ETF_SLIPPAGE / (1 - ETF_SLIPPAGE), 2),
            "net": round(proceeds - commission, 2),
            "reason": "袖套调仓：目标持仓集变更",
        })
        del positions[code]

    # ── 2) 买（只买不足的部分，pro-rata 缩放）────────────────────────
    bought_codes: set = set()  # 本次买进来的标的（第 3 步不削它们）
    wants: Dict[str, float] = {}
    for code, w in target.items():
        want = value * w
        cur = float(positions.get(code, {}).get("shares", 0)) * px[code]
        if cur < want - 1e-6:
            wants[code] = want - cur

    if wants:
        need = sum(wants.values())
        scale = 1.0
        if need > 0 and cash < need * (1 + ETF_COMMISSION):
            scale = cash / (need * (1 + ETF_COMMISSION))

        if not (board_lot and board_lot > 0):
            # ── board_lot=0：允许小数股 ─────────────────────────────────
            #    **仅用于与研究侧撮合器做逐点等价性验证**（实盘绝不可用：
            #    ETF 最小交易单位 100 股）。
            #    这条路径必须与研究侧 L3.simulate **完全一致**，包括
            #    "最后一笔按剩余现金缩放"这个细节，否则会被误判成实现不等价。
            for code, amount in wants.items():
                if amount <= 0:
                    continue
                amount *= scale
                cost = amount * ETF_COMMISSION
                if amount + cost > cash + 1e-6:
                    # ⚠️ 不能整笔跳过：pro-rata 缩放后各笔金额的浮点累加会**略微**
                    # 超出可用现金，最后一笔必然放不下 → 整笔跳过 = 1/9 的资金
                    # 永久闲置（实测首笔建仓时 11.1% ≈ 11.1 万躺在现金里）。
                    # 正确做法：按剩余现金缩放这一笔。
                    amount = max(0.0, cash / (1.0 + ETF_COMMISSION))
                    cost = amount * ETF_COMMISSION
                    if amount <= 0:
                        continue
                p = px[code]
                buy_px = p * (1 + ETF_SLIPPAGE)
                shares = amount / buy_px
                cash -= amount + cost
                bought_codes.add(code)
                _record_buy(positions, buys, code, p, buy_px, shares, amount, cost, date)
        else:
            # ── 第一遍：按整手向下取整，记下每只的"零头" ────────────────
            #   整手约束：ETF 最小交易单位 100 股（货币 ETF 例外，本袖套不含）。
            #   不取整会留下零股 → 实盘根本下不了单。
            #
            #   ⚠️ 单纯向下取整会把**高价标的系统性欠配**：
            #   511020 单价 ~103 元 → 一手 10,300 元，而单只目标权重只有
            #   55,555（1/9）的 18.5%。实测三类权重因此偏成
            #   gold 34.8% / bond 30.3% / commodity 34.9%（偏离 4.7%），
            #   "类间等权"名存实亡。
            held_value = {
                c: float(positions.get(c, {}).get("shares", 0)) * px[c]
                for c in positions if c in px
            }
            scaled_wants = {c: a * scale for c, a in wants.items() if a > 0}
            lots_plan = _plan_sleeve_lots(
                target, value, px, scaled_wants, held_value, cash, board_lot,
            )
            plan: List[Dict[str, Any]] = []
            if lots_plan:
                # 按类分组求出的最优整手方案（类间权重偏离最小）
                for code, lots in lots_plan.items():
                    p = px[code]
                    plan.append({
                        "code": code, "px": p,
                        "buy_px": p * (1 + ETF_SLIPPAGE),
                        "lots": lots, "remainder": 0.0, "remainder_value": 0.0,
                    })
            else:
                # 兜底：逐只向下取整（现金不够买任何候选组合时）
                for code, amount in scaled_wants.items():
                    p = px[code]
                    buy_px = p * (1 + ETF_SLIPPAGE)
                    raw_shares = amount / buy_px
                    lots = int(raw_shares // board_lot)
                    remainder = raw_shares - lots * board_lot
                    plan.append({
                        "code": code, "px": p, "buy_px": buy_px,
                        "lots": lots, "remainder": remainder,
                        "remainder_value": remainder * buy_px,
                    })

            # ── 第二遍：兜底的零头分配（只在类分组方案不可用时）────────
            #   正常路径下 `_plan_sleeve_lots` 已经给出按类分组的最优整手方案，
            #   这里只在它返回空（现金连最小候选都买不起）时兜底：
            #   把零头按**误差下降量**（2×缺口 − 一手金额）最大优先补成整手，
            #   只补能真正降低总误差的（gain > 0），每只最多补一手。
            #
            #   ⚠️ 这里**不能直接扣 cash** —— 第三遍执行时还会把整笔成交额
            #      从 cash 里扣一遍，双扣会让现金变成负数（实测 −4.3 万）。
            #      正确做法：这里只算「可分配的闲钱」free，执行时统一扣。
            if not lots_plan:
                def _plan_cost() -> float:
                    return sum(
                        it["lots"] * board_lot * it["buy_px"] * (1 + ETF_COMMISSION)
                        for it in plan
                    )

                free = cash - _plan_cost()
                while True:
                    best: Optional[Dict[str, Any]] = None
                    for item in plan:
                        if item["remainder"] < 0:
                            continue  # 已经补过一手
                        need_cash = board_lot * item["buy_px"] * (1 + ETF_COMMISSION)
                        if need_cash > free + 1e-6:
                            continue
                        # 补一手带来的误差下降：缺口 s → 一手 − s，下降 2s − 一手
                        gain = 2 * item["remainder_value"] - board_lot * item["buy_px"]
                        if gain <= 0:
                            continue
                        if best is None or gain > best["gain"]:
                            best = dict(item, gain=gain)
                    if best is None:
                        break
                    free -= board_lot * best["buy_px"] * (1 + ETF_COMMISSION)
                    for item in plan:
                        if item["code"] == best["code"]:
                            item["lots"] += 1
                            item["remainder"] = -1.0
                            break

            # ── 第三遍：执行 ────────────────────────────────────────────
            for item in plan:
                code = item["code"]
                p = item["px"]
                buy_px = item["buy_px"]
                shares = float(item["lots"] * board_lot)
                if shares <= 0:
                    continue
                # 按整手后的真实成交额回填现金（取整后金额会小于预算，差额留在现金里）
                amount = shares * buy_px
                cost = amount * ETF_COMMISSION
                cash -= amount + cost
                bought_codes.add(code)
                _record_buy(positions, buys, code, p, buy_px, shares, amount, cost, date)

    # ── 3) 卖出超配 ────────────────────────────────────────────────────
    #   ⚠️ 只卖**买之前就超配**的标的（研究侧 L3.simulate 第 4 步的语义：
    #      买的时候 `cur >= want` 就跳过不买，事后把超配部分削掉）。
    #      为什么不能一刀切地卖所有超配：
    #        ① 刚买进来的标的如果因为整手取整/补手而略超目标，卖掉它
    #           就把整手误差又还回去了 → 类间等权重新偏掉；
    #        ② 卖零股在 A 股/ETF **根本下不了单**（只能整手卖，
    #           零股必须一次性全卖）→ 持仓变成不可交易的状态。
    #      所以：只削买前就超配的，且**按整手**卖（卖不完的零头留着，
    #      误差 < 1 手）。
    for code in list(positions.keys()):
        if code not in target:
            continue
        if code in bought_codes:
            continue  # 本次买进来的 → 不削（保留整手误差）
        p = px[code]
        cur = float(positions[code].get("shares", 0)) * p
        want = value * target[code]
        if cur <= want + 1e-6:
            continue
        sell_px = p * (1 - ETF_SLIPPAGE)
        excess_shares = (cur - want) / sell_px
        # 整手卖出：零股不可交易
        sell_shares = (float(int(excess_shares // board_lot) * board_lot)
                       if (board_lot and board_lot > 0) else excess_shares)
        if sell_shares <= 0:
            continue
        proceeds = sell_shares * sell_px
        if proceeds <= 0:
            continue
        commission = proceeds * ETF_COMMISSION
        cash += proceeds - (commission if charge_sell_commission else 0.0)
        shares = float(positions[code].get("shares", 0))
        cost_amt = float(positions[code].get("cost", 0) or 0) * sell_shares
        realized = proceeds - commission - cost_amt
        positions[code]["shares"] = shares - sell_shares
        sells.append({
            "code": code,
            "price": round(p, 4),
            "shares": int(sell_shares),
            "amount": round(proceeds, 2),
            "cost_amount": round(cost_amt, 2),
            "realized_pnl": round(realized, 2),
            "commission": round(commission, 2),
            "stamp_tax": 0.0,
            "slippage": round(proceeds * ETF_SLIPPAGE / (1 - ETF_SLIPPAGE), 2),
            "net": round(proceeds - commission, 2),
            "reason": "袖套调仓：削减超配",
        })

    # 清掉被卖光的
    for code in list(positions.keys()):
        if float(positions[code].get("shares", 0)) <= 1e-9:
            del positions[code]

    return positions, cash, buys, sells


def sleeve_positions_value(
    positions: Dict[str, Dict[str, Any]],
    bars: EtfBars,
    date: pd.Timestamp,
    price_field: str = "close",
) -> float:
    """袖套持仓市值（价格口径与研究侧一致：前复权价，不做二次复权）。

    注意 ETF 行情是**前复权**序列（前值随分红重锚）：
    所以持仓市值必须用「股数 × 当前价」，**不能**用「股数 × 买入价 × 比值」
    ——后者在锚点重锚后会系统性错算。
    """
    total = 0.0
    for code, pos in (positions or {}).items():
        bar = bars.get(code, date, price_field)
        if bar is None:
            continue
        total += float(pos.get("shares", 0)) * float(bar["price"])
    return total


def sleeve_class_weights_now(
    positions: Dict[str, Dict[str, Any]],
    bars: EtfBars,
    date: pd.Timestamp,
) -> Dict[str, float]:
    """当前三类各占袖套的比例（用于校验「类间等权」是否被漂移破坏）。"""
    vals: Dict[str, float] = {}
    for code, pos in (positions or {}).items():
        cls = next((c for c, lst in SLEEVE_HELD.items() if code in lst), None)
        if cls is None:
            continue
        bar = bars.get(code, date)
        if bar is None:
            continue
        vals[cls] = vals.get(cls, 0.0) + float(pos.get("shares", 0)) * float(bar["price"])
    total = sum(vals.values())
    if total <= 0:
        return {c: 0.0 for c in SLEEVE_CLASSES}
    return {c: vals.get(c, 0.0) / total for c in SLEEVE_CLASSES}
