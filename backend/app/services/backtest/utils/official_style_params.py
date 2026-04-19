from __future__ import annotations

from math import ceil
from typing import Any

_RANKING_STRATEGIES = {
    "topk_dropout",
    "model_topk_dropout",
    "official_topk_dropout",
    "model_ranking",
    "ranking",
}


def derive_official_style_topk_dropout_params(pool_size: int) -> dict[str, int]:
    if pool_size < 2:
        raise ValueError("official-style TopkDropout 至少需要 2 只股票")

    target_hold_ratio = 50 / 300
    target_drop_ratio = 5 / 50

    topk = min(pool_size - 1, max(2, ceil(pool_size * target_hold_ratio)))
    n_drop = max(1, ceil(topk * target_drop_ratio))
    hold_thresh = 0 if pool_size <= 3 else max(1, min(2, ceil((pool_size - topk) * 0.15)))

    return {
        "topk": int(topk),
        "n_drop": int(n_drop),
        "hold_thresh": int(hold_thresh),
    }


def apply_official_style_topk_dropout_params(
    *,
    strategy_name: str,
    stock_codes: list[str] | None,
    strategy_config: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_config = dict(strategy_config or {})
    normalized_name = str(strategy_name or "").lower()
    official_style_enabled = bool(normalized_config.get("official_style"))

    if not official_style_enabled or normalized_name not in _RANKING_STRATEGIES:
        return normalized_config

    pool_size = len(stock_codes or [])
    if pool_size < 2:
        return normalized_config

    derived = derive_official_style_topk_dropout_params(pool_size)
    for key, value in derived.items():
        normalized_config.setdefault(key, value)

    return normalized_config
