from app.services.qlib.official_workflow import (
    DEFAULT_OFFICIAL_SEGMENTS,
    OfficialDataset,
    OfficialMarket,
    build_official_lightgbm_workflow_config,
)


def test_default_official_segments_match_qlib_lightgbm_benchmark() -> None:
    assert DEFAULT_OFFICIAL_SEGMENTS.train == ("2008-01-01", "2014-12-31")
    assert DEFAULT_OFFICIAL_SEGMENTS.valid == ("2015-01-01", "2016-12-31")
    assert DEFAULT_OFFICIAL_SEGMENTS.test == ("2017-01-01", "2020-08-01")


def test_alpha158_official_workflow_config_matches_reference_defaults() -> None:
    config = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA158,
        market=OfficialMarket.CSI300,
    )

    assert config.market == "csi300"
    assert config.benchmark == "SH000300"
    assert config.topk == 50
    assert config.n_drop == 5
    assert config.account == 100000000
    assert config.open_cost == 0.0005
    assert config.close_cost == 0.0015
    assert config.min_cost == 5.0
    assert config.handler_class == "Alpha158"
    assert config.feature_count == 158
    assert config.label_expression == "Ref($close, -2) / Ref($close, -1) - 1"
    assert config.learn_processors == [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]


def test_alpha360_official_workflow_config_uses_csranknorm() -> None:
    config = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA360,
        market=OfficialMarket.CSI300,
    )

    assert config.handler_class == "Alpha360"
    assert config.feature_count == 360
    assert config.learn_processors == [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]


def test_csi500_official_workflow_switches_benchmark() -> None:
    config = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA158,
        market=OfficialMarket.CSI500,
    )

    assert config.market == "csi500"
    assert config.benchmark == "SH000905"
