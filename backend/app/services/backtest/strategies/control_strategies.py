"""对照实验专用策略（只在 ablation 脚本里注册，不进生产策略表）。

背景（2026-11-06）：纸面盘 honest_42f 的 60 只持仓，在全池的 60 日已实现
波动分位中位 0.958 —— 即全市场最安静的 4% 那一批，60 只里 60 只都比池中位
数更不波动。因此「跑赢全池等权」可能是低波动 beta，不是选股能力。

要判定这件事，需要与 honest_42f **同 topk、同 n_drop、同调仓频率、同成本、
同可成交性** 的对照组：
  - 对照 C：纯低波动 TopK（factor_ranking weights={"low_vol":1.0}，无需新代码）
  - 对照 D：随机 TopK（本文件）—— 用来回答「任何选股是不是都强于随机」

随机对照的分数只由 (固定种子, 日期) 决定，与任何行情字段无关，因此它就是
噪声本身：任何真实策略若打不过它，说明它没有信息量，只有换手。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from .factor_ranking_strategy import FactorRankingStrategy, _SCORES_ATTR


class RandomTopKControlStrategy(FactorRankingStrategy):
    """固定种子的随机横截面分数（TopK/Dropout 执行，与真实策略同契约）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.seed = int(config.get("seed", 20261106))

    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        codes = [
            code
            for code, df in stock_data.items()
            if df is not None and not df.empty and "close" in df.columns
        ]
        if not codes:
            logger.warning("random_topk_control: 无可用股票数据，跳过打分")
            return

        indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
        master = indexes[0].append(indexes[1:]) if len(indexes) > 1 else indexes[0]
        master = master.unique().sort_values()
        n_dates, n_stocks = len(master), len(codes)

        composite = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        rows = np.arange(n_dates)
        if self.rebalance_days == "weekly":
            mask = master.dayofweek == self.rebalance_weekday
        elif self.rebalance_days == "monthly":
            mask = master.day == 1
        else:
            mask = np.ones(n_dates, dtype=bool)
        # 每个调仓日一份独立随机截面：分数与行情无关 → 纯噪声。
        for k in rows[mask]:
            rng = np.random.default_rng(self.seed + int(master[k].toordinal()))
            composite[k, :] = rng.random(n_stocks)

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
        logger.info(
            f"random_topk_control 打分完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, seed={self.seed}, topk={self.topk}, n_drop={self.n_drop}"
        )
