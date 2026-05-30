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
        "description": "固定 5/20 均线、0.5% 阈值的 MA 基准配置，用于标准横向比较。",
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
