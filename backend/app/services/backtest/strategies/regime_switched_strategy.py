"""风格暴露择时开关策略（研究原型，不进生产策略表）。

在 honest_composite（基线选股线）之上加一层「二元开关」：
  ON  ：完全等于基线 —— 25 因子 honest_composite 的 TopK 目标权重；
  OFF ：清掉股票持仓，买 defense_code（默认 511010 国债ETF）权重 defense_weight
        （默认 0.95），其余留现金。

判据（AND 才防御，与 regime.py 单测契约一致）：
  进防御：J1（size spread z < z_threshold 且样本充足）AND J2（影子净值回撤超
          dd_limit 或影子 60 日滚动夏普低于 sharpe_limit），连续 confirm_off 个
          再平衡执行日才翻 OFF；
  出防御：J1 与 J2 都转好（z 样本不足视为「好」→ 默认不防御），连续 confirm_on
          个执行日才翻回 ON；
  达到 max_switches 后冻结在当前状态（在 meta 记录）。

无前视：判据在执行日 D 只用 D-1 收盘及以前的数据（影子净值同样只更新到 D-1），
与 lag1 口径（周五收盘出信号、周一收盘成交）严格对齐；asof 之后的价格不参与任何
决策（regime.py 有逐位相等的单测保证）。

引擎接线说明（这个引擎的 sizing 只有 BacktestConfig.max_position_size 一个旋钮）：
  - OFF 要「一只 ETF 占 95%」而 ON 要「20 只票各占 4.75%」，同一个 BacktestConfig
    装不下两个值。策略在 validate_signal（引擎每笔执行前都会调、且信号带 timestamp）
    里按当日角色把调用方传入的 BacktestConfig.max_position_size 改成对应值。
    只改调用方自己创建的 config 实例，不改任何引擎/生产代码。
  - 模式参数（topk/n_drop）通过 get_trade_mode_config() 逐日动态给出：
    正常 ON 日 (topk, n_drop)；OFF 日 (1, topk) 一天清完股票并买入 ETF；
    离开防御的第一天 (1, 1) 先卖掉 ETF 并买入分数最高的 1 只，第二天起正常补仓
    到 topk（引擎 initial-build 分支自动填满）。

决策日志通过 params["switch_log"]（调用方创建的 list）与 params["meta"]（dict）
传出 —— 没有全局变量。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..models import TradingSignal
from ..regime import (
    ShadowNav,
    random_switch_schedule,
    run_switch_machine,
)
from .honest_composite_strategy import _SCORES_ATTR

OFF_STOCK_SCORE = -1000.0  # OFF 日给所有股票的哨兵分（让持仓成为卖出候选）
OFF_ETF_SCORE = 1000.0  # OFF 日给防御 ETF 的最高分（买它、不买股票）
EXIT_ETF_SCORE = -1e9  # 离开防御第一天给 ETF 的最低分（先卖掉它）


DEFAULT_PARAMS: Dict[str, Any] = {
    "z_threshold": -1.0,
    "confirm_off": 5,
    "confirm_on": 10,
    "dd_limit": 0.15,
    "sharpe_limit": 0.0,
    "defense_code": "511010.SH",
    "defense_weight": 0.95,
    "max_switches": 12,
    "shadow_window": 60,
    "shadow_annualize": 244,
}


def make_regime_switched_strategy(
    cfg: Dict[str, Any],
    base_factory: Callable[[Dict[str, Any]], BaseStrategy],
    params: Dict[str, Any],
) -> "RegimeSwitchedStrategy":
    """构造开关策略。

    Args:
        cfg: 引擎 strategy_config（不含 topk/n_drop 键，避免被引擎的
             trade_mode_config 合并逻辑覆盖逐日动态参数；用 topk_base/n_drop_base）。
        base_factory: 基线策略工厂（如 lambda c: HonestCompositeStrategy(c)）。
        params: 判据参数 + 出参：
             params["switch_log"]: list  —— 逐执行日决策日志（runner 写进 JSON）
             params["meta"]: dict        —— 判据元信息（样本不足天数、冻结标记等）
             params["z_by_date"]: {iso: float|None} —— 预计算的 J1 z（runner 从
                  分层指数算好；regime 模式必需）
             params["calendar"]: [iso...] —— 交易日历（含窗口前 warmup，供 T-1 查找）
             params["engine_backtest_config"]: 调用方的 BacktestConfig 实例（用于
                  OFF/ON 单仓上限切换；None 则 OFF 口径退化为基线 4.75%）
             params["random_off_dates"]: [iso...] —— 随机开关对照的 OFF 日集合
                  （给了它就走 random 模式，不再跑判据状态机）
    """
    return RegimeSwitchedStrategy(cfg, base_factory, params)


class RegimeSwitchedStrategy(BaseStrategy):
    def __init__(
        self,
        cfg: Dict[str, Any],
        base_factory: Callable[[Dict[str, Any]], BaseStrategy],
        params: Dict[str, Any],
    ):
        super().__init__("RegimeSwitched", cfg)
        self.topk_base = int(cfg.get("topk_base", 20))
        self.n_drop_base = int(cfg.get("n_drop_base", 3))
        cash_reserve = float(cfg.get("cash_reserve_ratio", 0.05))
        self.on_max_size = (1.0 - cash_reserve) / self.topk_base

        self.params = dict(params)
        p = self.params
        self.p = {k: p.get(k, v) for k, v in DEFAULT_PARAMS.items()}
        self.defense_code = str(self.p["defense_code"])

        self._base = base_factory(cfg.get("base_config") or {})
        self._off_dates: set = set()  # OFF 执行日（datetime）
        self._exit_etf_days: set = set()  # 离开防御第一天（卖 ETF）
        self._day_role: Dict[datetime, str] = {}  # exec 日 → "off"/"exit_on"/"on"
        self._seen_dt: Optional[datetime] = None  # generate_signals 观察到的最近日期
        self._switch_log: List[Dict[str, Any]] = p.get("switch_log")
        if self._switch_log is None:
            self._switch_log = []
            p["switch_log"] = self._switch_log
        self._meta: Dict[str, Any] = p.get("meta")
        if self._meta is None:
            self._meta = {}
            p["meta"] = self._meta
        self._bt_config = p.get("engine_backtest_config")
        self._zmap: Dict[str, Optional[float]] = {
            str(k)[:10]: (None if v is None else float(v))
            for k, v in (p.get("z_by_date") or {}).items()
        }
        self._cal: List[pd.Timestamp] = [
            pd.Timestamp(x) for x in (p.get("calendar") or [])
        ]
        self._random_off: set = {str(x)[:10] for x in (p.get("random_off_dates") or [])}

    # ── 预热：基线打分 + 判据状态机 + 把 OFF/ON 日程写进 attrs ────────────
    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        codes = [c for c in stock_data.keys() if c != self.defense_code]
        base_frames = {c: stock_data[c] for c in codes}
        await self._base.prepare_backtest_data(base_frames, start_date, end_date)

        scores: Dict[str, Dict[datetime, float]] = {
            c: base_frames[c].attrs.get(_SCORES_ATTR) or {} for c in codes
        }
        exec_dates = sorted({d for m in scores.values() for d in m})
        if not exec_dates:
            self._meta["note"] = "基线没有任何执行日（数据不足），开关不生效"
            return

        closes = pd.DataFrame({c: base_frames[c]["close"] for c in codes}).sort_index()
        closes.index = pd.DatetimeIndex(closes.index)
        if self.defense_code not in stock_data:
            raise ValueError(
                f"防御代码 {self.defense_code} 不在引擎加载的数据里 —— "
                "OFF 状态将无法建仓（检查 data/qlib_data/features/day 下的 ETF 文件）"
            )
        etf_close = pd.to_numeric(
            stock_data[self.defense_code]["close"], errors="coerce"
        )
        closes[self.defense_code] = np.nan
        for ts, v in zip(etf_close.index, etf_close.to_numpy(dtype="float64")):
            closes.loc[ts, self.defense_code] = v
        closes = closes.sort_index()

        cal = list(self._cal) or sorted(closes.index)

        def prev_cal(d: pd.Timestamp) -> Optional[pd.Timestamp]:
            i = int(np.searchsorted(cal, d, side="left")) - 1
            return cal[i] if i >= 0 else None

        # 单遍顺序推进：影子净值依赖状态（J2），状态又依赖影子净值 → 必须内联
        # 状态机。每步只用 ≤D-1 收盘的数据：决策发生在「影子净值更新到 D-1」之后、
        # 「把 D 当日收益计入影子」之前。
        window_days = [d for d in cal if closes.index[0] <= d <= closes.index[-1]]
        if not window_days:
            window_days = list(closes.index)
        shadow = ShadowNav(
            window=int(self.p["shadow_window"]),
            annualize=int(self.p["shadow_annualize"]),
        )
        shadow.seed(window_days[0], 1.0)
        basket: Optional[Dict[str, float]] = None
        exec_set = {pd.Timestamp(d) for d in exec_dates}

        state = "ON"
        enter_streak = exit_streak = switch_count = 0
        frozen = False
        z_threshold = float(self.p["z_threshold"])
        dd_limit = float(self.p["dd_limit"])
        sharpe_limit = float(self.p["sharpe_limit"])
        max_switches = int(self.p["max_switches"])
        random_mode = bool(self._random_off)

        steps: List[Dict[str, Any]] = []
        insufficient_z_days = 0
        random_off_exec = {pd.Timestamp(x) for x in self._random_off}

        for d in window_days:
            x = prev_cal(d)
            if d in exec_set:
                dt = d.to_pydatetime()
                z = self._zmap.get(x.date().isoformat()) if x is not None else None
                if z is None:
                    insufficient_z_days += 1
                dd = shadow.drawdown()  # 只含 ≤D-1 的净值
                sh = shadow.rolling_sharpe()  # 只含 ≤D-1 的净值
                j1_bad = z is not None and z < z_threshold
                j1_good = not j1_bad  # z 样本不足 → 默认不防御 → 视作好
                j2_bad = (dd < -dd_limit) or (sh is not None and sh < sharpe_limit)
                j2_good = (dd >= -dd_limit) and (sh is None or sh >= sharpe_limit)
                before = state
                switched = False
                frozen_attempt = False
                if random_mode:
                    state = "OFF" if d in random_off_exec else "ON"
                    switched = state != before
                    if switched:
                        switch_count += 1
                    enter_flag = exit_flag = None
                else:
                    enter_flag = bool(j1_bad and j2_bad)  # AND 才防御
                    exit_flag = bool(j1_good and j2_good)  # 两判据都好才退出
                    enter_streak = enter_streak + 1 if enter_flag else 0
                    exit_streak = exit_streak + 1 if exit_flag else 0
                    if not frozen:
                        if state == "ON" and enter_streak >= int(self.p["confirm_off"]):
                            if switch_count >= max_switches:
                                frozen = True
                                frozen_attempt = True
                            else:
                                state = "OFF"
                                switched = True
                                switch_count += 1
                                enter_streak = exit_streak = 0
                        elif state == "OFF" and exit_streak >= int(
                            self.p["confirm_on"]
                        ):
                            if switch_count >= max_switches:
                                frozen = True
                                frozen_attempt = True
                            else:
                                state = "ON"
                                switched = True
                                switch_count += 1
                                enter_streak = exit_streak = 0
                steps.append(
                    {
                        "date": d.date().isoformat(),
                        "state_before": before,
                        "state": state,
                        "enter_flag": enter_flag,
                        "exit_flag": exit_flag,
                        "enter_streak": enter_streak,
                        "exit_streak": exit_streak,
                        "switched": switched,
                        "switch_count": switch_count,
                        "frozen": frozen,
                        "frozen_attempt": frozen_attempt,
                        "z": z,
                        "shadow_dd": dd,
                        "shadow_sharpe": sh,
                    }
                )
                if state == "OFF":
                    self._off_dates.add(dt)
                if state == "ON" and before == "OFF":
                    self._exit_etf_days.add(dt)

            # 影子净值：D-1→D 收益归属「上一个目标篮子」（引擎在 D 收盘才换仓）
            if x is not None and x in closes.index and d in closes.index:
                shadow.update(basket, x, d, closes)
            # 篮子按 D 收盘的新状态切换：OFF→ETF，ON→基线 top 分数 topk 等权
            if d in exec_set:
                basket = self._basket_after(d, state, scores)

        machine_meta = {
            "mode": "random_control" if random_mode else "regime",
            "switch_count": switch_count,
            "frozen_at_max_switches": frozen,
            "max_switches": max_switches,
            "confirm_off": self.p["confirm_off"],
            "confirm_on": self.p["confirm_on"],
            "z_threshold": self.p["z_threshold"],
            "dd_limit": self.p["dd_limit"],
            "sharpe_limit": self.p["sharpe_limit"],
            "defense_code": self.defense_code,
            "defense_weight": self.p["defense_weight"],
            "off_days": int(sum(1 for s in steps if s["state"] == "OFF")),
            "n_decision_days": len(steps),
        }

        # 事后一致性校验：把记录的 flag 喂给纯函数 run_switch_machine，
        # 状态序列必须与内联逐日一致（不一致 = 接线有漂移，直接报错而不是出数）
        if not random_mode:
            check = run_switch_machine(
                [pd.Timestamp(s["date"]) for s in steps],
                [bool(s["enter_flag"]) for s in steps],
                [bool(s["exit_flag"]) for s in steps],
                confirm_off=int(self.p["confirm_off"]),
                confirm_on=int(self.p["confirm_on"]),
                max_switches=max_switches,
            )
            mismatch = [
                (a["date"], a["state"], b["state"])
                for a, b in zip(steps, check["steps"])
                if a["state"] != b["state"]
            ]
            if mismatch:
                raise RuntimeError(
                    f"内联状态机与 run_switch_machine 不一致: {mismatch[:5]}"
                )
            machine_meta["pure_machine_check"] = "identical"

        prev_state = "ON"
        for s in steps:
            d = pd.Timestamp(s["date"]).to_pydatetime()
            if s["state"] == "OFF":
                self._day_role[d] = "off"
            else:
                self._day_role[d] = "exit_on" if prev_state == "OFF" else "on"
            prev_state = s["state"]
            self._switch_log.append(s)

        # 把开关日程写进打分 attrs：OFF 日股票全部 -1000、ETF +1000；
        # 离开防御第一天 ETF -1e9。ON 日股票分 = 基线分逐位不动。
        for c in codes:
            if _SCORES_ATTR not in base_frames[c].attrs:
                continue
            m = scores[c]
            for d in self._off_dates:
                m[d] = OFF_STOCK_SCORE
        if self.defense_code in stock_data:
            etf_scores: Dict[datetime, float] = {}
            for d in self._off_dates:
                etf_scores[d] = OFF_ETF_SCORE
            for d in self._exit_etf_days:
                etf_scores[d] = EXIT_ETF_SCORE
            stock_data[self.defense_code].attrs[_SCORES_ATTR] = etf_scores

        self._meta.update(machine_meta)
        self._meta["insufficient_z_days"] = insufficient_z_days
        self._meta["n_exec_days"] = len(exec_dates)
        self._meta["shadow_nav_series"] = shadow.series()
        self._meta["shadow_missing_price_events"] = shadow.missing_price_events
        logger.info(
            f"regime_switched: 执行日={len(exec_dates)} 切换={machine_meta['switch_count']} "
            f"OFF日={len(self._off_dates)} z样本不足日={insufficient_z_days} "
            f"冻结={machine_meta.get('frozen_at_max_switches')}"
        )

    def _basket_after(
        self, d: pd.Timestamp, state: str, scores: Dict[str, Dict[datetime, float]]
    ) -> Optional[Dict[str, float]]:
        """D 收盘决策后的影子篮子：ON=基线 top 分数的 topk 等权；OFF=ETF×defense_weight。"""
        if state == "OFF":
            return {self.defense_code: float(self.p["defense_weight"])}
        dt = d.to_pydatetime()
        # ON：此处的 map 尚未被 OFF 覆盖（覆盖发生在准备末尾）→ 原始基线分
        pairs = []
        for c, m in scores.items():
            v = m.get(dt)
            if v is not None and np.isfinite(v):
                pairs.append((v, c))
        if not pairs:
            return None
        pairs.sort(key=lambda t: (t[0], t[1]), reverse=True)
        top = [c for _, c in pairs[: self.topk_base]]
        w = 1.0 / len(top)
        return {c: w for c in top}

    # 引擎接线
    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        if self._seen_dt is None or current_date > self._seen_dt:
            self._seen_dt = current_date
        return self._base.generate_signals(data, current_date)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        return self._base.calculate_indicators(data)

    def precompute_all_signals(self, data: pd.DataFrame):
        return None  # 与基线一致：走逐日 generate_signals 路径

    def get_trade_mode(self) -> Optional[str]:
        return "topk_dropout"

    def get_trade_mode_config(self) -> Dict[str, object]:
        d = self._seen_dt
        role = self._day_role.get(d) if d is not None else None
        if role == "off":
            # 一天清完股票（n_drop=topk_base）并只买 1 个仓位（topk=1）的 ETF
            return {
                "topk": 1,
                "n_drop": self.topk_base,
                "hold_thresh": 0,
                "deal_price": "close",
            }
        if role == "exit_on":
            # 离开防御第一天：卖掉 ETF 并买入最高分 1 只
            return {"topk": 1, "n_drop": 1, "hold_thresh": 0, "deal_price": "close"}
        return {
            "topk": self.topk_base,
            "n_drop": self.n_drop_base,
            "hold_thresh": 0,
            "deal_price": "close",
        }

    def get_position_size_hint(self) -> Optional[float]:
        return self.on_max_size

    def validate_signal(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_positions: Dict[str, Any],
    ):
        # 引擎每笔执行前都会调这里，且信号带 timestamp → 按当日角色切单仓上限。
        # 只动调用方自己创建的 BacktestConfig 实例，不动引擎代码。
        if self._bt_config is not None:
            role = self._day_role.get(signal.timestamp)
            if role == "off":
                self._bt_config.max_position_size = float(self.p["defense_weight"])
            else:
                self._bt_config.max_position_size = self.on_max_size
        return self._base.validate_signal(signal, portfolio_value, current_positions)


class FixedDefenseStrategy(BaseStrategy):
    """纯防御臂：整个窗口只持 defense_code（引擎口径下单持 ETF 的真实参照）。"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__("FixedDefense", cfg)
        self.code = str(cfg.get("defense_code", "511010.SH"))
        self.weight = float(cfg.get("defense_weight", 0.95))
        self._bought = False

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        code = data.attrs.get("stock_code", "")
        if code != self.code or self._bought:
            return []
        idx = self._get_current_idx(data, current_date)
        if idx < 1:
            return []
        price = float(data["close"].iloc[idx])
        if not np.isfinite(price) or price <= 0:
            return []
        self._bought = True
        from ..models import SignalType

        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=code,
                signal_type=SignalType.BUY,
                strength=1.0,
                price=price,
                reason="纯防御臂：整窗持有国债ETF",
            )
        ]

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        return {"price": data["close"]}

    def precompute_all_signals(self, data: pd.DataFrame):
        return None

    def get_position_size_hint(self) -> Optional[float]:
        return self.weight


def build_random_schedule_from_log(
    decision_dates: List[datetime], switch_log: List[Dict[str, Any]], seed: int
) -> Dict[str, Any]:
    """随机开关对照日程：切换次数取开关版实际切换次数，OFF 区间由 seed 决定。"""
    n = sum(1 for s in switch_log if s.get("switched"))
    return random_switch_schedule(decision_dates, target_switches=n, seed=seed)
