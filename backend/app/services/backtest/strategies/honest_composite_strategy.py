"""诚实版等权复合因子 → TopK-Dropout 组合管理（正式策略 honest_composite）。

诊断8 发现: v2 复合的头条 Sharpe 1.75 大部分是"当日涨停锁死"伪迹——
  内盘因子 (HIGH0/KLEN/KUP/KLOW/LOW0/RSQR5) 的 top50 全是当日涨停票，
  排除当日涨停后这些因子几乎失效。引擎未建模锁死 → 伪收益。

本版三条纪律（与 scripts/ic_composite_10y_honest.py 完全对齐）：
  1. 剔除全部内盘 OHLC 因子 (KLEN/KUP/KLOW/LOW0/HIGH0/RSQR5)
  2. 涨停掩码: 当日涨幅 >= 9.5% 的票分数置 NaN (当日不可在收盘买入)
  3. 保留 close/volume 族 (MIN/STD/CORR/CORD/MAX), MAX 亦被掩码约束

因子池 = 25 个诚实因子 (23 个唯一公式) + 17 个 Alpha158 面板因子 = 42：
  基础 23 公式: MIN/MAX/STD/CORR/CORD，n ∈ {5,10,20,30,60}
    MIN:  +1  (低于窗口最低价越深越好 → 正 IC)
    MAX/STD/CORR/CORD: -1
  Alpha158 面板因子 (读自 panel_dir/，MultiIndex 转 date)：
    4 族 × 3 窗 (5,10,20): VSUMP (-1) / VSUMN (+1) / VSUMD (-1) / QTLD (+1)
    5 族 × 1 窗 (5): IMAX5 (-1) / IMIN5 (-1) / WVMA5 (-1) / VMA5 (+1) / VSTD5 (-1)
    符号取自 5y IC study fwd1 IC 符号
  面板数据从 2021-05-11 起，此前 17 个因子为 NaN，复合自动退化为 23 公式版。
  窗口选择依据 (ic_composite_10y_expand2 五格对比): 42f 全面优于 43f (9族×2窗) 和
  47f (4族×3窗+5族×2窗) — 其余族的 10 日窗是负贡献，不应纳入。

复合 = 等权平均( sign(z(因子)) )，nan-aware 截面 Z-Score（有效票数 ≥ min_cross）。
涨停掩码: 当日涨幅 ≥ limit_up 的票 composite = NaN。
lag: 0 = 用当日分数交易当日（默认）；1 = 用前一日分数。

TopK-Dropout 执行端由 ``TopkDropoutTradeModeExecutor`` 完成，
config 提供 topk / ndrop / hold_thresh / max_position_size。
max_position_size 默认 = (1-cash_reserve)/topk 等权，避免现金耗尽冻结。
config 可选 panel_dir 指定 Alpha158 面板因子路径（默认 ../data/qlib_data/features/day）。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.services.trading.execution_rules import (
    industry_ids,
    limit_up_mask_matrix_at,
    load_industry_map,
    load_st_flags,
    zscore_with_industry,
)

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal
from ._fundamental_panel import _load_fundamental_panel

_SCORES_ATTR = "_honest_composite_scores"

# 23 个唯一公式 (5 族 × 5 窗口)，MIN 为正号，其余为负号
_HONEST_FORMULAS: Dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    _HONEST_FORMULAS[f"MIN{_n}"] = 1.0
    _HONEST_FORMULAS[f"MAX{_n}"] = -1.0
    _HONEST_FORMULAS[f"STD{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORR{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORD{_n}"] = -1.0

# 17 个 Alpha158 面板因子（来自 qlib_data/features/day/，MultiIndex 已转 date 列）
# 4 族×3 窗 (5,10,20) + 5 族×1 窗 (5)。符号取自 5y IC study fwd1 IC 符号
# 42f 五格全面优于 43f/47f (2026-10-15 ic_composite_10y_expand2 对比):
#   IMAX/IMIN/WVMA/VMA/VSTD 的 10 日窗是负贡献，VSUMP/VSUMN/VSUMD/QTLD 的 20 日窗是正贡献
_ALPHAPANEL_FACTORS: Dict[str, float] = {
    "VSUMP5": -1.0,
    "VSUMP10": -1.0,
    "VSUMP20": -1.0,
    "VSUMN5": 1.0,
    "VSUMN10": 1.0,
    "VSUMN20": 1.0,
    "VSUMD5": -1.0,
    "VSUMD10": -1.0,
    "VSUMD20": -1.0,
    "QTLD5": 1.0,
    "QTLD10": 1.0,
    "QTLD20": 1.0,
    "IMAX5": -1.0,
    "IMIN5": -1.0,
    "WVMA5": -1.0,
    "VMA5": 1.0,
    "VSTD5": -1.0,
}


#: Alpha158 面板因子的默认目录（**绝对路径**）
#: 2026-10-26 修正：原来写的是相对 CWD 的 "../data/qlib_data/features/day"。
#: 从仓库根跑脚本时它解析到 ~/Projects/data/... → 不存在 → 17 个面板因子
#: 全 NaN → 42 因子策略**静默退化成 25 因子版**（实测 0.8796 vs
#: 真正的 42 因子 0.7992）。所有验证脚本都从仓库根跑 → 一直在跑错的东西。
#: 现在按 __file__ 上溯到项目根，任何 CWD 下都指向同一个目录。
# 路径：backend/app/services/backtest/strategies/xxx.py
#   dirname 1 = strategies, 2 = backtest, 3 = services, 4 = app,
#   5 = backend, 6 = 项目根
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)
DEFAULT_PANEL_DIR = os.path.join(_PROJECT_ROOT, "data", "qlib_data", "features", "day")


def _load_factor_weights(path: str, power: float, cap: float) -> Dict[str, float]:
    """从 ICIR 表 JSON 读因子权重。

    文件格式（factor_ic_decompose.py 产出）：
        {"factors": [{"factor": "CORR10", "icir": -0.691, ...}, ...]}
    权重 = min(|ICIR| ** power, cap) / mean(|ICIR| ** power)
    → 归一化到平均权重 1，等权时等价于不加权。
    """
    import json as _json

    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            d = _json.load(fh)
    except Exception:
        return {}
    rows = d.get("factors") or []
    raw: Dict[str, float] = {}
    for r in rows:
        name = r.get("factor")
        icir = r.get("icir")
        if not name or icir is None:
            continue
        raw[str(name)] = min(abs(float(icir)) ** power, cap)
    if not raw:
        return {}
    mean_w = sum(raw.values()) / len(raw)
    if mean_w <= 0:
        return {}
    return {k: v / mean_w for k, v in raw.items()}


class HonestCompositeStrategy(BaseStrategy):
    """25 因子等权复合横截面排名策略（TopK/Dropout 执行模式）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("HonestComposite", config)
        self.topk = int(config.get("topk", 60))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 5)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.lag = int(config.get("lag", 0))
        # 调仓日掩码：rebalance_days="weekly" → 只在周五出分（周频），
        # "daily" → 每日出分。引擎不消费 BacktestConfig.rebalance_frequency，
        # 周频必须在这里把非调仓日的分数置 NaN，执行器自然不换仓。
        # 周频能把换手降 4~5 倍（实盘成本 0.03%+0.05%印花+0.1%滑点/单边），
        # 是 A 股日频 TopK 策略在真实成本下能否落地的决定性变量。
        self.rebalance_days = str(config.get("rebalance_days", "daily")).lower()
        self.rebalance_weekday = int(config.get("rebalance_weekday", 4))  # 4=周五
        self.limit_up = float(config.get("limit_up", 0.095))
        # 2026-10-25 新增：行业中性化
        # 42 因子复合存在系统性行业暴露（高分票集中在少数行业）。
        # 开启后：因子 z-score 先按申万一级行业去均值，再在行业内重新标准化，
        # 变成「行业内选相对最强的」。逐日横截面，无未来函数。
        # 实测（3 年全市场周频）：夏普 0.965 → 1.441，回撤基本不变。
        self.industry_neutral = bool(config.get("industry_neutral", False))
        # 2026-10-26 新增：回撤刹车（组合层择时）
        # 动机：10 年回测里所有策略 2018 年都亏 24%~42% —— 没有一条线能躲开
        # 系统性熊市。这层在**选股之上**决定「满仓还是减仓」，不预测涨跌。
        #   regime_filter: ""(关闭) | "breadth"(市场宽度) | "ma"(等权指数均线)
        #   regime_threshold: 触发减仓的阈值
        #   regime_keep_ratio: 坏时保留的持仓比例（0.5 = 只留一半）
        self.regime_filter = str(config.get("regime_filter", "") or "").lower()
        self.regime_threshold = float(config.get("regime_threshold", 0.35))
        self.regime_keep_ratio = float(config.get("regime_keep_ratio", 0.5))
        self.regime_ma_window = int(config.get("regime_ma_window", 20))
        # 2026-10-27 新增：**趋势加分**（攻「逐笔胜率 67% → 75%」）
        # 思路：把「close / MA_n - 1」作为一个**加分项**加进复合分。
        # 为什么是加分而不是过滤？
        #   - 过滤（把不合格的分数置 NaN）会让持仓一旦跌破均线就被
        #     topk_dropout 引擎踢出 → 变成止损/追跌杀跌，之前实测有害
        #     （5% 止损：夏普 0.956 → 0.769）
        #   - 加分只是让趋势差的票**自然排在后面**、不进新建仓，
        #     已有持仓的排名不受影响 → 不会引入额外换手
        # trend_ma_window: 用多少日均线（0 = 关闭）
        # trend_weight: 加分权重（相对其它因子的 z-score 单位）
        self.trend_ma_window = int(config.get("trend_ma_window", 0))
        self.trend_weight = float(config.get("trend_weight", 0.0))
        # 2026-10-27 新增：**ICIR 加权**（攻夏普）
        # 动机：因子级 RankIC 拆解显示 22 个公式因子的 IC 全部显著，
        #   其中 CORR 族（价量相关性）ICIR 高达 −0.69，而**等权平均把
        #   强因子的信号稀释了** —— 组合 ICIR 只有 0.43 < 单因子 CORR10 的 0.69。
        #   42 因子版更糟：多出的 17 个面板因子把 CORR 族权重从 3/25 压到 3/42。
        # factor_icir_file: ICIR 表 JSON（factor_ic_decompose.py 产出）
        # factor_weight_power: 权重 = |ICIR|^power（0 = 等权，1 = 线性）
        # factor_weight_cap: 单因子权重上限（避免过度集中）
        self.factor_icir_file = config.get("factor_icir_file", "")
        self.factor_weight_power = float(config.get("factor_weight_power", 0.0))
        self.factor_weight_cap = float(config.get("factor_weight_cap", 3.0))
        # 2026-10-27：**因子白名单**（因子减法）
        # 动机：IC 拆解显示 22 个公式因子 IC 全部显著，但它们的 |ICIR| 差 4.5 倍
        #   （CORR10 0.69 vs STD60 0.15），等权平均把强弱拉平。
        #   而 ICIR 加权已被证伪（因子高度相关时加权反而更差）→ 正确做法是
        #   **把弱因子直接去掉**，让强因子在等权下占更大比重。
        # factor_allowlist: 只保留这些因子（None = 全部）
        # 2026-10-27：**use_panel 开关**
        # 实测：42 因子版（面板因子真生效）夏普 0.799 / 回撤 −16.46%，
        #       25 公式因子版 夏普 0.880 / 回撤 −13.37%。
        # → 17 个 Alpha158 面板因子是净负面，主线应只用 25 个公式因子。
        # 这里做成显式开关（而不是靠「panel_dir 路径失效」的副产品）。
        # 默认 False：25 公式因子版（实测夏普 0.880 / 回撤 −13.37%）优于
        # 42 因子版（0.799 / −16.46%）。要用面板因子要显式 use_panel=True。
        self.use_panel = bool(config.get("use_panel", False))
        # 2026-10-27：**暴露中性化**（Barra APT 风格逐日截面 OLS 残差化）
        # 暴露：size = 20 日 log 成交额均值（平台无市值数据，是流动性/规模混合代理）
        #       illiq = Amihud(2002) ILLIQ 20 日均值取 log1p
        # 中性化：逐日截面 OLS（y=复合分, X=[size, illiq, 1]）取残差
        #
        # ⚠️⚠️ 2026-10-31 更正：**默认关闭，而且在生产口径下是负面的 —— 不要开启。**
        #   原注释写的是"本轮唯一跨段稳定有效，段1/2/3 Δ夏普 +0.222/+0.241/+0.307，
        #   每段回撤改善 2~3pp"。**那组数字来自研究实现**
        #   （methods_research_v2.neutralize_score，**42f 口径**），不是本策略的生产口径。
        #   真实引擎口径（25f，data/backtest/neutseg_seg*_*.json）是：
        #       seg1 0.7121→0.5716 (Δ −0.140)
        #       seg2 1.0419→1.2422 (Δ +0.200)
        #       seg3 1.3374→1.0685 (Δ −0.269)   ← 三段里两段负面
        #   3 年整体（real_lag1_neutral.json，旧引擎）：0.880→0.757，
        #   回撤 −13.37%→−20.47%（回撤反而恶化 7.1pp）
        #   → 只有 42f 口径支持它。生产口径不支持。
        #   另：5 年窗口上失效（0.635→0.548），regime 依赖也成立。
        #   裁决：docs/backtest/P8_EXPOSURE_NEUTRAL_2026-10-31.md
        #   ⚠️ 教训：**引用别人的"改进有效"结论前，必须确认它是同一口径**
        #      （同因子集 + 同引擎版本 + 同窗口切分），否则会把 42f 的乐观数字
        #      当成 25f 的结论 —— 本仓库已经踩过一次。
        self.exposure_neutral = bool(config.get("exposure_neutral", False))
        self.neutral_exposures = list(
            config.get("neutral_exposures", ("size", "illiq"))
        )
        self.neutral_window = int(config.get("neutral_window", 20))
        # 2026-10-27：**财务因子**（唯一还没试过的「新信息」路径）
        # IC 验证（5513 只，3 年，PIT 按报告期+120 天）：
        #   gross_margin −0.0103(t−4.45) / roe_yoy −0.0097(t−3.90) /
        #   asset_yoy −0.0079(t−3.40) / cfps +0.0070(t+3.19) /
        #   inventory_turnover +0.0050(t+2.63) / asset_turnover +0.0048(t+2.58)
        # ⚠️ 经典质量因子（roe / net_margin / debt_ratio / eps_adj）**全部无效**
        #    （|t| < 0.5）→ A 股 fwd5 上基本面不是 alpha 因子
        # ⚠️ 显著的 3 个负 IC 因子本质是「低估值/低关注度」反向代理，不是质量
        # 面板：npz（date × stock 的 float32 矩阵），由 build_fundamental_panel.py 生成
        self.fundamental_panel = config.get("fundamental_panel") or None
        # fundamental_factors: {因子名: 符号}，符号 +1 = 值越高越好，−1 = 值越低越好
        # （与 zscore_with_industry 的符号语义一致：z = (x-mu)/sd * sign）
        _ff = config.get("fundamental_factors") or {}
        if isinstance(_ff, dict):
            self.fundamental_factors = {str(k): float(v) for k, v in _ff.items()}
        else:
            self.fundamental_factors = {str(t[0]): float(t[1]) for t in _ff}
        self.fundamental_weight = float(config.get("fundamental_weight", 1.0))
        self.factor_allowlist = config.get("factor_allowlist", None)
        if self.factor_allowlist is not None:
            self.factor_allowlist = set(self.factor_allowlist)
        self._factor_weights: Dict[str, float] = {}
        if self.factor_icir_file and self.factor_weight_power > 0:
            self._factor_weights = _load_factor_weights(
                self.factor_icir_file,
                self.factor_weight_power,
                self.factor_weight_cap,
            )
        self.score_scale = float(config.get("score_scale", 10.0))
        self.panel_dir = config.get("panel_dir", DEFAULT_PANEL_DIR)

        # max_position_size: 默认等权公式 (1-cash_reserve)/topk
        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )

        if self.topk <= 0:
            raise ValueError("honest_composite 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("honest_composite 策略要求 n_drop > 0")
        logger.info(
            f"honest_composite topk={self.topk} n_drop={self.n_drop} "
            f"lag={self.lag} limit_up={self.limit_up} "
            f"mps={self.max_position_size:.6f}"
        )

    # ── TopK/Dropout 执行契约 ──────────────────────────────────────────
    def get_trade_mode(self) -> str:
        return "topk_dropout"

    def get_trade_mode_config(self) -> Dict[str, object]:
        return {
            "topk": self.topk,
            "n_drop": self.n_drop,
            "hold_thresh": self.hold_thresh,
            "deal_price": "close",
        }

    def get_position_size_hint(self) -> Optional[float]:
        """等权上限 = (1-现金预留)/topk，与 TopK 持仓数匹配。

        不设这个上限时引擎默认 0.2，topk=60 会把现金一次性耗尽、组合冻结
        （实测只买到 9 只）。
        """
        return self.max_position_size

    # ── 全市场横截面打分（回测准备阶段一次性完成） ─────────────────────
    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        codes = [
            code
            for code, df in stock_data.items()
            if df is not None
            and not df.empty
            and "close" in df.columns
            and "high" in df.columns
            and "low" in df.columns
            and "volume" in df.columns
        ]
        if not codes:
            logger.warning("honest_composite: 无可用股票数据，跳过打分")
            return

        indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
        master = indexes[0].append(indexes[1:]) if len(indexes) > 1 else indexes[0]
        master = master.unique().sort_values()
        n_dates, n_stocks = len(master), len(codes)

        def to_matrix(column: str) -> np.ndarray:
            out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
            for j, code in enumerate(codes):
                df = stock_data[code]
                positions = master.get_indexer(pd.DatetimeIndex(df.index))
                ok = positions >= 0
                values = pd.to_numeric(df[column], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                out[positions[ok], j] = values[ok]
            return out

        frame_opts = {"index": master, "dtype": np.float64}
        close = pd.DataFrame(to_matrix("close"), **frame_opts)
        vol = pd.DataFrame(to_matrix("volume"), **frame_opts)

        # 计算 23 个诚实因子公式
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = close.pct_change(fill_method=None)
            vchg = vol.pct_change(fill_method=None)
            cands: Dict[str, pd.DataFrame] = {}
            for n in (5, 10, 20, 30, 60):
                cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
                cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
                cands[f"STD{n}"] = close.rolling(n).std() / close
                cands[f"CORR{n}"] = close.rolling(n).corr(vol)
                cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

        # 涨停掩码: 当日涨幅 >= limit_up 的票不可在收盘买入
        day_ret = close.pct_change(fill_method=None).to_numpy()
        # 2026-10-25 修正：原来用统一的 self.limit_up（默认 0.095）全板通用。
        # 实测问题：182 只 20% 板（创业板/科创板）股票有 5380 个交易日涨幅落在
        # 9.5%~20%（并未涨停）却被误剔；ST 股触及 5% 涨停则完全没拦。
        # 改为逐票按板块判定（主板 10% / 创业板科创板 20% / 北交所 30%）。
        # 2026-10-26 再补：ST / 退市整理期按 5% 判定
        # （数据源 data/stocks/st_names.json，375 只 ST + 94 只退市整理）。
        # 2026-10-31 改为**按交易日 PIT** 的涨跌停幅度：
        # 创业板 2020-08-24 注册制之前是 10%，现行 `limit_pct` 用 20%
        # → 长历史窗口（2020-08 之前）会把"真的涨停买不到"的票放过去，
        #   成交价不可实现 → 长历史回测被系统性污染。
        # 2020-08-24 之后的窗口与原实现**逐位相同**（回归门见
        # `backend/scripts/p10_limit_pit_regression.py`）。
        limit_up_mask = limit_up_mask_matrix_at(
            day_ret, close.columns, master, is_st=load_st_flags(list(close.columns))
        )

        # nan-aware 等权复合 — 23 个诚实公式 + 18 个 Alpha158 面板因子
        min_cross = max(2, min(30, n_stocks // 10))
        acc = np.zeros((n_dates, n_stocks), dtype=np.float64)
        wsum = np.zeros((n_dates, n_stocks), dtype=np.float64)

        # 行业中性：把代码映射到申万一级行业 id
        _ind_ids = None
        _n_ind = 0
        if self.industry_neutral:
            _imap = load_industry_map()
            _ind_ids = industry_ids(codes, _imap)
            _n_ind = int(_ind_ids.max()) + 1 if len(_ind_ids) else 0
            logger.info(
                f"honest_composite 行业中性: {int((_ind_ids >= 0).sum())}/{len(codes)} "
                f"只有行业分类, {_n_ind} 个行业"
            )

        def zscore_add(
            mat: np.ndarray, sign: float, min_cross: int, weight: float = 1.0
        ) -> None:
            z = zscore_with_industry(
                mat, sign, min_cross, _ind_ids, _n_ind, self.industry_neutral
            )
            good = np.isfinite(z)
            acc[good] += np.nan_to_num(z)[good] * weight
            wsum[good] += weight

        # 23 个诚实公式（ICIR 加权时按因子名取权重）
        _n_used = 0
        for formula, sign in _HONEST_FORMULAS.items():
            if (
                self.factor_allowlist is not None
                and formula not in self.factor_allowlist
            ):
                continue
            zscore_add(
                cands[formula].to_numpy(dtype=np.float64),
                sign,
                min_cross,
                self._factor_weights.get(formula, 1.0),
            )
            _n_used += 1

        # 18 个 Alpha158 面板因子 — 每个因子构建一次全截面矩阵
        for factor_name, sign in (
            _ALPHAPANEL_FACTORS.items() if self.use_panel else ()
        ):
            if (
                self.factor_allowlist is not None
                and factor_name not in self.factor_allowlist
            ):
                continue
            _pw = self._factor_weights.get(factor_name, 1.0)
            mat = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
            for j, code in enumerate(codes):
                try:
                    pf = pd.read_parquet(
                        os.path.join(
                            self.panel_dir, code.replace(".", "_") + ".parquet"
                        )
                    ).reset_index()
                    pf = pf.set_index("date").sort_index()
                    pf = pf.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
                    pf = pf[~pf.index.duplicated(keep="first")]
                    pos = master.get_indexer(pd.DatetimeIndex(pf.index))
                    ok = pos >= 0
                    mat[pos[ok], j] = pd.to_numeric(
                        pf[factor_name], errors="coerce"
                    ).to_numpy(dtype=np.float64, na_value=np.nan)[ok]
                except Exception:
                    continue
            zscore_add(mat, sign, min_cross, _pw)

        composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)

        # ── 财务因子并入（2026-10-27）────────────────────────────
        # 方式：与价量复合分**等权平均**（两路各自 z-score 后各占 50% 权重）。
        # ⚠️ 不做 ICIR 加权/减法 —— 那些操作在本项目已全部证伪
        #    （提高集中度 = 降夏普 + 升回撤）。
        if self.fundamental_panel and self.fundamental_factors:
            _facc = np.zeros((n_dates, n_stocks), dtype=np.float64)
            _fwsum = np.zeros((n_dates, n_stocks), dtype=np.float64)
            _fz = _load_fundamental_panel(
                self.fundamental_panel, master, codes, self.fundamental_factors
            )
            for _fname, _sign in self.fundamental_factors.items():
                _m = _fz.get(_fname)
                if _m is None:
                    continue
                _z = zscore_with_industry(
                    _m, _sign, min_cross, _ind_ids, _n_ind, self.industry_neutral
                )
                _good = np.isfinite(_z)
                _facc[_good] += _z[_good]
                _fwsum[_good] += 1.0
            _fcomp = np.where(
                _fwsum > 0, _facc / np.where(_fwsum > 0, _fwsum, 1.0), np.nan
            )
            # 与价量分等权合成（两路都有效时）
            _both = np.isfinite(composite) & np.isfinite(_fcomp)
            _merged = np.where(
                _both,
                0.5 * composite + 0.5 * _fcomp,
                np.where(np.isfinite(composite), composite, _fcomp),
            )
            _n_fund = int(np.isfinite(_fcomp).sum())
            logger.info(
                f"honest_composite 财务因子并入: "
                f"{len(self.fundamental_factors)} 个因子，有效格 {_n_fund}"
            )
            # 静默失效防护：面板加载失败会表现为「变体与基线逐位相同」，
            # 极易被误读成「财务因子没用」。这里直接抛错。
            if _n_fund < n_dates * n_stocks * 0.10:
                raise ValueError(
                    f"财务因子面板有效格只有 {_n_fund} / {n_dates * n_stocks}"
                    f"（<10%）→ 面板未正确加载，拒绝静默退回基线"
                )
            composite = _merged

        composite[limit_up_mask] = np.nan  # 涨停掩码: 当日不可买

        # ── 暴露中性化（2026-10-27）──────────────────────────────
        # 逐日截面 OLS：y = 复合分, X = [暴露..., 1] → 取残差。
        # 暴露全部只用「当日及之前」的数据 → 无未来函数。
        if self.exposure_neutral and self.neutral_exposures:
            _exps = {}
            # ⚠️ 单位：qlib 的 `$volume` 单位是**手**（1 手 = 100 股），
            # 实测 qlib/baostock 的 volume 比恒为 0.0100。
            # 成交额 = close × volume × 100，否则 illiq 会差 100 倍
            # （实测：漏乘 100 时原生引擎夏普 0.855 vs 实验室 1.012）
            _amount = close * vol * 100.0
            if "size" in self.neutral_exposures:
                with np.errstate(divide="ignore", invalid="ignore"):
                    _log_amt = np.log(
                        np.where(_amount.to_numpy() > 0, _amount.to_numpy(), np.nan)
                    )
                _exps["size"] = (
                    pd.DataFrame(_log_amt)
                    .rolling(self.neutral_window, min_periods=5)
                    .mean()
                    .to_numpy()
                )
            if "illiq" in self.neutral_exposures:
                with np.errstate(divide="ignore", invalid="ignore"):
                    _r = close / close.shift(1) - 1.0
                    _daily = (_r.abs() / _amount.where(_amount > 0)).to_numpy()
                _ill = (
                    pd.DataFrame(_daily)
                    .rolling(
                        self.neutral_window,
                        min_periods=max(5, self.neutral_window // 2),
                    )
                    .mean()
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    _exps["illiq"] = np.log1p(_ill.to_numpy())
            if _exps:
                _names = list(_exps)
                _resid = np.full_like(composite, np.nan)
                for _t in range(n_dates):
                    _y = composite[_t]
                    _X = np.column_stack([_exps[k][_t] for k in _names])
                    _m = np.isfinite(_y) & np.isfinite(_X).all(axis=1)
                    if int(_m.sum()) < 50:
                        continue
                    _A = np.column_stack([_X[_m], np.ones(int(_m.sum()))])
                    _beta, *_ = np.linalg.lstsq(_A, _y[_m], rcond=None)
                    _resid[_t, _m] = _y[_m] - _A @ _beta
                composite = _resid
                logger.info(
                    f"honest_composite 暴露中性化: {','.join(_names)} "
                    f"（窗口 {self.neutral_window}）"
                )

        # ── 趋势加分（2026-10-27）────────────────────────────────
        # 逐日横截面 z-score(close / MA_n - 1)，乘权重加到复合分上。
        # 全部是「当日及之前」的数据 → 无未来函数。
        if self.trend_ma_window > 0 and self.trend_weight != 0:
            ma = close.rolling(self.trend_ma_window, min_periods=1).mean()
            trend_raw = (close / ma.where(ma > 0) - 1.0).to_numpy(dtype=np.float64)
            # 逐日横截面 z-score（nan-aware），复用同一套 min_cross 门槛
            _ok = np.isfinite(trend_raw)
            _cnt = _ok.sum(axis=1, keepdims=True)
            _mu = np.where(
                _cnt > 0,
                np.where(_ok, trend_raw, 0.0).sum(axis=1, keepdims=True)
                / np.maximum(_cnt, 1),
                0.0,
            )
            _dv = trend_raw - _mu
            _sd = np.sqrt(
                np.where(
                    _cnt > 1,
                    (np.where(_ok, _dv, 0.0) ** 2).sum(axis=1, keepdims=True)
                    / np.maximum(_cnt - 1, 1),
                    0.0,
                )
            )
            _tz = np.where(_sd > 1e-12, _dv / np.where(_sd > 1e-12, _sd, 1.0), np.nan)
            _tz = np.where(_cnt >= min_cross, _tz, np.nan)
            composite = composite + self.trend_weight * _tz

        # 调仓日掩码: 非调仓日分数全部置 NaN → 执行器无分可依, 不换仓。
        if self.rebalance_days == "weekly":
            weekday = master.dayofweek  # 0=周一 … 4=周五
            composite[weekday != self.rebalance_weekday] = np.nan
        elif self.rebalance_days == "monthly":
            composite[master.day != 1] = np.nan

        # ── 回撤刹车（组合层择时）────────────────────────────────────
        # 10 年回测显示所有策略 2018 年都亏 24%~42%，没有一条能躲开系统性熊市。
        # 这层不预测涨跌，只在「市场状态差」时把持仓砍掉一部分（其余资金留现金）。
        #
        # 两种状态判定（都用当日及之前的数据，无未来函数）：
        #   breadth  市场宽度 = 当日上涨股票占比 < 阈值 → 减仓
        #   ma       等权指数收盘 < 其 N 日均线 → 减仓
        #
        # 减仓方式：把超出保留比例的分数置 NaN。目标权重仍是 (1-0.05)/topk，
        # 所以保留一半持仓 ≈ 总仓位降到一半（其余是现金）。
        if self.regime_filter:
            if self.regime_filter == "breadth":
                with np.errstate(invalid="ignore"):
                    breadth = np.nanmean((day_ret > 0).astype(float), axis=1)
                bad = breadth < self.regime_threshold
            elif self.regime_filter == "ma":
                mkt = np.nanmean(close.to_numpy(dtype=np.float64), axis=1)
                ma = (
                    pd.Series(mkt).rolling(self.regime_ma_window, min_periods=20).mean()
                )
                bad = mkt < ma.to_numpy()
            else:
                bad = np.zeros(n_dates, dtype=bool)

            n_keep = max(1, int(round(self.topk * self.regime_keep_ratio)))
            n_bad = 0
            for k in np.where(bad)[0]:
                row = composite[k]
                valid = np.where(np.isfinite(row))[0]
                if len(valid) <= n_keep:
                    continue
                order = valid[np.argsort(row[valid])[::-1]]
                drop = order[n_keep:]
                composite[k, drop] = np.nan
                n_bad += 1
            if n_bad:
                logger.info(
                    f"honest_composite 回撤刹车({self.regime_filter}"
                    f" thr={self.regime_threshold} keep={self.regime_keep_ratio}): "
                    f"{n_bad}/{n_dates} 天减仓至 {n_keep} 只"
                )

        # lag: 0 = 当日分数，1 = 前一日分数
        if self.lag > 0:
            lagged = np.full_like(composite, np.nan)
            if self.lag < n_dates:
                lagged[self.lag :] = composite[: n_dates - self.lag]
            composite = lagged

        # 存储每股分数
        stored = 0
        for j, code in enumerate(codes):
            column = composite[:, j]
            score_map: Dict[datetime, float] = {}
            for k in range(n_dates):
                value = column[k]
                if np.isfinite(value):
                    score_map[master[k].to_pydatetime()] = float(value)
            stock_data[code].attrs[_SCORES_ATTR] = score_map
            stored += len(score_map)
        if self.factor_allowlist is not None:
            logger.info(
                f"honest_composite 因子白名单: 用 {_n_used} 个公式因子"
                f"（全量 {len(_HONEST_FORMULAS)}）"
            )
        logger.info(
            f"honest_composite 横截面打分完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, 最小截面要求={min_cross}, "
            f"涨停掩码={self.limit_up}, lag={self.lag}"
        )

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        return None

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        scores = data.attrs.get(_SCORES_ATTR) or {}
        index = pd.DatetimeIndex(data.index)
        score_series = pd.Series(
            [scores.get(ts.to_pydatetime()) for ts in index],
            index=index,
            dtype="float64",
        )
        return {"price": data["close"], "score": score_series}

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        scores = data.attrs.get(_SCORES_ATTR)
        if not scores:
            return []
        score = scores.get(current_date)
        if score is None or not np.isfinite(score):
            return []

        current_idx = self._get_current_idx(data, current_date)
        if current_idx < 0:
            return []
        price = float(data["close"].iloc[current_idx])
        if not np.isfinite(price) or price <= 0:
            return []
        stock_code = data.attrs.get("stock_code", "UNKNOWN")

        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=stock_code,
                signal_type=SignalType.BUY,
                strength=min(1.0, abs(score) / self.score_scale),
                price=price,
                reason=(
                    f"诚实复合因子排名分 {score:.4f} "
                    f"(TopK={self.topk}, n_drop={self.n_drop}, lag={self.lag})"
                ),
                metadata={
                    "ranking_score": score,
                    "signal_role": "ranking_score",
                    "trade_mode": self.get_trade_mode(),
                    "topk": self.topk,
                    "n_drop": self.n_drop,
                    "hold_thresh": self.hold_thresh,
                    "deal_price": "close",
                },
            )
        ]
