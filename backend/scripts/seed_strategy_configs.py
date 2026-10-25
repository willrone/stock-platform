#!/usr/bin/env python3
"""Seed default reusable strategy configurations.

The presets are intentionally conservative assets for smoke / benchmark / research
workflows. They make strategy parameters visible in the platform instead of only
living in code or one-off request payloads.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import Base, SessionLocal, sync_engine
from app.models.strategy_config_models import StrategyConfig

DEFAULT_PRESET_USER_ID = "default-presets"

DEFAULT_STRATEGY_CONFIG_PRESETS: list[dict[str, Any]] = [
    {
        "config_name": "benchmark/moving_average_5_20_threshold_005",
        "strategy_name": "moving_average",
        "parameters": {
            "preset_layer": "benchmark",
            "short_window": 5,
            "long_window": 20,
            "signal_threshold": 0.005,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "旧版 MA 基准（5/20 + 0.5% 阈值）。全市场 3 年 -43.1%/夏普 -0.498 —— 保留仅作历史对比，当前不建议使用。",
    },
    {
        "config_name": "benchmark/moving_average_20_60",
        "strategy_name": "moving_average",
        "parameters": {
            "preset_layer": "benchmark",
            "short_window": 20,
            "long_window": 60,
            "signal_threshold": 0.0,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "MA 修复版（20/60 + 无阈值），全市场 3 年 -1.5%/夏普 0.110/回撤 -36.8%。已从旧版 -43.1% 大幅改善，但仍然不赚钱 —— 趋势跟随在 A 股不 work，不进纸面盘。",
    },
    {
        "config_name": "benchmark/rsi_optimized_default",
        "strategy_name": "rsi",
        "parameters": {
            "preset_layer": "benchmark",
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "trend_ma_period": 50,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "RSI 优化版默认 benchmark 配置，保留趋势过滤口径。",
    },
    {
        "config_name": "benchmark/macd_default",
        "strategy_name": "macd",
        "parameters": {
            "preset_layer": "benchmark",
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "经典 MACD 12/26/9 benchmark 配置。",
    },
    {
        "config_name": "benchmark/bollinger_20_2",
        "strategy_name": "bollinger",
        "parameters": {
            "preset_layer": "benchmark",
            "period": 20,
            "std_dev": 2,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "布林带 20 日、2 倍标准差 benchmark 配置。",
    },
    {
        "config_name": "research/portfolio_technical_vote_v1",
        "strategy_name": "portfolio",
        "parameters": {
            "preset_layer": "research",
            "integration_method": "weighted_voting",
            "strategies": [
                {
                    "name": "moving_average",
                    "weight": 0.34,
                    "config": {
                        "short_window": 5,
                        "long_window": 20,
                        "signal_threshold": 0.005,
                    },
                },
                {
                    "name": "rsi",
                    "weight": 0.33,
                    "config": {
                        "rsi_period": 14,
                        "oversold_threshold": 30,
                        "overbought_threshold": 70,
                    },
                },
                {
                    "name": "macd",
                    "weight": 0.33,
                    "config": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                },
            ],
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "技术指标加权投票 research 配置，用于后续优化和 walk-forward 对照。",
    },
    {
        "config_name": "benchmark/honest_composite_topk60",
        "strategy_name": "honest_composite",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 60,
            "n_drop": 6,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "max_position_size": 0.015833,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "25 诚实因子等权复合 (MIN/MAX/STD/CORR/CORD) + TopK-Dropout top60/drop6。剔除内盘 OHLC 因子 + 涨停掩码，max_position_size 等权=(1-0.05)/60，防现金耗尽冻结。",
    },
    {
        "config_name": "benchmark/rsi_71421_topk30",
        "strategy_name": "rsi_71421",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 30,
            "n_drop": 3,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "max_position_size": 0.031667,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "RSI 7/14/21 均值回归 + 低波动 + 量价等权复合 + TopK-Dropout top30/drop3。1年回测夏普0.537(原版22f的0.111的5倍), 收益+7.03%, 回撤-15.7%。治本版主力策略。",
    },
    {
        "config_name": "benchmark/gbm_voltgt_topk60",
        "strategy_name": "gbm_voltgt",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 60,
            "n_drop": 5,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "max_position_size": 0.015833,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
            "top_n_factors": 0,
            "horizon": 1,
        },
        "description": "gbm_voltgt 默认 42 因子 + 1天周期 + 波动率目标仓位。1年回测夏普1.155。",
    },
    {
        "config_name": "benchmark/gbm_voltgt_top100",
        "strategy_name": "gbm_voltgt",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 60,
            "n_drop": 5,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "max_position_size": 0.015833,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
            "top_n_factors": 100,
            "horizon": 1,
        },
        "description": "gbm_voltgt IC选top100因子 + 1天周期 + 波动率目标仓位。3年回测夏普1.334(比默认42因子的1.173高14%)。因子筛选管线自动选IC最高的100个因子。",
    },
    {
        "config_name": "benchmark/honest_composite_topk60_weekly",
        "strategy_name": "honest_composite",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 60,
            "n_drop": 6,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "rebalance_days": "weekly",
            "rebalance_weekday": 4,
            "max_position_size": 0.015833,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "42f 等权复合 + 周频调仓(周五)。3年全市场: 周频夏普0.965 vs 日频0.718, 交易6660→1390笔(4.8x), 回撤-20.3%→-17.6%。周频是实盘成本下的更优口径。",
    },
    {
        "config_name": "benchmark/rsi_71421_topk30_weekly",
        "strategy_name": "rsi_71421",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 30,
            "n_drop": 3,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "rebalance_days": "weekly",
            "rebalance_weekday": 4,
            "max_position_size": 0.031667,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "RSI 7/14/21 均值回归 + 周频调仓(周五)。3年全市场: 周频夏普0.803 vs 日频0.328, 收益+13.2%→+37.8%, 回撤-26.7%→-16.3%, 交易4092→828笔。",
    },
    {
        "config_name": "benchmark/factor_ranking_topk50_weekly",
        "strategy_name": "factor_ranking",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 50,
            "n_drop": 5,
            "hold_thresh": 0,
            "limit_up": 0.095,
            "rebalance_days": "weekly",
            "rebalance_weekday": 4,
            "max_position_size": 0.019,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "动量/低波动/低振幅横截面复合 + 周频调仓(周五)。3年全市场: 周频夏普0.768 vs 日频0.720, 回撤-15.2%→-14.2%, 交易5054→1160笔。",
    },
    {
        "config_name": "benchmark/gbm_voltgt_topk60_weekly",
        "strategy_name": "gbm_voltgt",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 60,
            "n_drop": 5,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "rebalance_days": "weekly",
            "rebalance_weekday": 4,
            "max_position_size": 0.015833,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
            "top_n_factors": 0,
            "horizon": 5,
        },
        "description": "GBM 元模型 + vol-target + 周频调仓。3年全市场: 周频夏普0.835 vs 日频0.780, 回撤-33.7%→-29.8%, 交易2930→620笔。",
    },
    {
        "config_name": "benchmark/low_volatility_topk30_weekly",
        "strategy_name": "low_volatility",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 30,
            "n_drop": 3,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "rebalance_days": "weekly",
            "rebalance_weekday": 4,
            "max_position_size": 0.031667,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "低波动横截面排序 + TopK/Dropout + 周频。2026-10-25 修复: 旧版只买不卖、3年只成交8笔(7只票躺满3年), 现为横截面选股。",
    },
    {
        "config_name": "benchmark/multi_factor_topk30_weekly",
        "strategy_name": "multi_factor",
        "parameters": {
            "preset_layer": "benchmark",
            "topk": 30,
            "n_drop": 3,
            "hold_thresh": 0,
            "lag": 0,
            "limit_up": 0.095,
            "rebalance_days": "weekly",
            "rebalance_weekday": 4,
            "max_position_size": 0.031667,
            "cash_reserve_ratio": 0.05,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "value/momentum/low_vol 三因子横截面等权 + TopK/Dropout + 周频。2026-10-25 修复: 旧版用时序标准化导致只买不卖、3年只成交117笔。",
    },
    {
        "config_name": "model/topk_dropout_k10_drop2",
        "strategy_name": "model_topk_dropout",
        "parameters": {
            "preset_layer": "model",
            "topk": 10,
            "n_drop": 2,
            "trade_mode": "topk_dropout",
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
        },
        "description": "模型排序类 TopK-Dropout 默认配置：持仓 top10，每期替换 2 只。",
    },
]


def build_config_id(config_name: str) -> str:
    """Build a stable deterministic config id from the preset name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", config_name.strip()).strip("-").lower()
    return f"preset-{slug}"


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False
    )


def seed_strategy_configs(session: Session) -> dict[str, int]:
    """Insert or update default strategy presets in a sync SQLAlchemy session."""
    created = 0
    updated = 0
    unchanged = 0

    for preset in DEFAULT_STRATEGY_CONFIG_PRESETS:
        config_id = build_config_id(preset["config_name"])
        existing = session.execute(
            select(StrategyConfig).where(StrategyConfig.config_id == config_id)
        ).scalar_one_or_none()

        if existing is None:
            session.add(
                StrategyConfig(
                    config_id=config_id,
                    config_name=preset["config_name"],
                    strategy_name=preset["strategy_name"],
                    parameters=preset["parameters"],
                    description=preset["description"],
                    user_id=DEFAULT_PRESET_USER_ID,
                )
            )
            created += 1
            continue

        needs_update = (
            existing.config_name != preset["config_name"]
            or existing.strategy_name != preset["strategy_name"]
            or not _json_equal(existing.parameters, preset["parameters"])
            or existing.description != preset["description"]
            or existing.user_id != DEFAULT_PRESET_USER_ID
        )
        if needs_update:
            existing.config_name = preset["config_name"]
            existing.strategy_name = preset["strategy_name"]
            existing.parameters = preset["parameters"]
            existing.description = preset["description"]
            existing.user_id = DEFAULT_PRESET_USER_ID
            updated += 1
        else:
            unchanged += 1

    session.commit()
    return {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "total": len(DEFAULT_STRATEGY_CONFIG_PRESETS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed default strategy configuration presets")
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create metadata tables before seeding; useful for local SQLite bootstrap.",
    )
    args = parser.parse_args()

    if args.create_tables:
        Base.metadata.create_all(bind=sync_engine)

    with SessionLocal() as session:
        result = seed_strategy_configs(session)

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
