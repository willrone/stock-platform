"""Default strategy config seed script tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.strategy_config_models import StrategyConfig

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "seed_strategy_configs.py"


def load_seed_module() -> Any:
    spec = importlib.util.spec_from_file_location("seed_strategy_configs", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_strategy_config_presets_cover_required_layers() -> None:
    module = load_seed_module()

    names = {preset["config_name"] for preset in module.DEFAULT_STRATEGY_CONFIG_PRESETS}

    assert "benchmark/moving_average_5_20_threshold_005" in names
    assert "benchmark/rsi_optimized_default" in names
    assert "benchmark/macd_default" in names
    assert "benchmark/bollinger_20_2" in names
    assert "research/portfolio_technical_vote_v1" in names
    assert "model/topk_dropout_k10_drop2" in names
    assert all("preset_layer" in preset["parameters"] for preset in module.DEFAULT_STRATEGY_CONFIG_PRESETS)


def test_seed_strategy_configs_is_idempotent(tmp_path: Path) -> None:
    module = load_seed_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'strategy-configs.db'}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        first = module.seed_strategy_configs(session)
        second = module.seed_strategy_configs(session)
        configs = session.execute(select(StrategyConfig)).scalars().all()

    # 2026-11-02：原来硬编码 6（脚本的 presets 从 6 涨到 17 后这条断言就过期了），
    # 而且**这个目录一直被 pyproject 的 noreversedirs=["scripts"] 排除在采集之外**
    # → 没人跑过它 → 过期没人发现。断言改成按 presets 实际条数（不写死魔数），
    #   真正的不变量是"幂等"：第一次全建，第二次全不改。
    n = len(module.DEFAULT_STRATEGY_CONFIG_PRESETS)
    assert first == {"created": n, "updated": 0, "unchanged": 0, "total": n}
    assert second == {"created": 0, "updated": 0, "unchanged": n, "total": n}
    assert len(configs) == n
    assert {config.config_id for config in configs} == {
        module.build_config_id(preset["config_name"])
        for preset in module.DEFAULT_STRATEGY_CONFIG_PRESETS
    }


def test_seed_strategy_configs_updates_existing_preset(tmp_path: Path) -> None:
    module = load_seed_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'strategy-configs.db'}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    config_id = module.build_config_id("benchmark/moving_average_5_20_threshold_005")
    with Session() as session:
        session.add(
            StrategyConfig(
                config_id=config_id,
                config_name="benchmark/moving_average_5_20_threshold_005",
                strategy_name="moving_average",
                parameters={"short_window": 1},
                description="stale",
                user_id="default-presets",
            )
        )
        session.commit()

        result = module.seed_strategy_configs(session)
        refreshed = session.get(StrategyConfig, config_id)

    assert result["updated"] == 1
    assert refreshed is not None
    assert refreshed.parameters["short_window"] == 5
    assert refreshed.parameters["signal_threshold"] == 0.005
    assert refreshed.description != "stale"
