#!/usr/bin/env python

"""CLI: 组合策略（portfolio）小宇宙参数/结构优化

用法示例：
  cd backend
  source venv/bin/activate
  PYTHONPATH=$PWD python scripts/optimize_portfolio.py --universe-size 200 --seed 42 --trials 30 --lambda-dd 1.0

默认：随机抽样股票池（可复现），目标函数为软惩罚：
  score = annualized_return - lambda_dd * abs(max_drawdown)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from app.core.config import settings
from app.services.data.stock_data_loader import StockDataLoader
from app.services.backtest.optimization.portfolio_hyperparameter_optimizer import (
    PortfolioHyperparameterOptimizer,
    PortfolioOptConfig,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--lambda-dd", type=float, default=1.0)
    p.add_argument("--max-strategies", type=int, default=3)
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2021-12-31")
    p.add_argument("--out", type=str, default="data/optimization/portfolio_last.json")

    args = p.parse_args()

    start_date = datetime.fromisoformat(args.start)
    end_date = datetime.fromisoformat(args.end)

    loader = StockDataLoader(data_root=str(settings.DATA_ROOT_PATH))

    # 获取可用股票列表：优先从 qlib 预计算特征目录取（更稳定，避免抽到原始行情缺失的股票）
    import os

    all_codes = []
    qlib_day_dir = os.path.join(str(settings.DATA_ROOT_PATH), "qlib_data", "features", "day")
    if os.path.isdir(qlib_day_dir):
        for name in os.listdir(qlib_day_dir):
            if not name.endswith(".parquet"):
                continue
            # 文件名形如 000001_SZ.parquet -> 000001.SZ
            code = name.replace(".parquet", "").replace("_", ".")
            all_codes.append(code)
    else:
        # 兜底：扫原始 parquet 行情目录
        stock_parquet_dir = os.path.join(str(settings.DATA_ROOT_PATH), "parquet", "stock_data")
        if os.path.isdir(stock_parquet_dir):
            for name in os.listdir(stock_parquet_dir):
                if not name.endswith(".parquet"):
                    continue
                code = name.replace(".parquet", "").replace("_", ".")
                all_codes.append(code)

    all_codes = [c for c in all_codes if isinstance(c, str) and c]
    all_codes = sorted(set(all_codes))

    if len(all_codes) < 10:
        raise SystemExit(f"可用股票数量过少: {len(all_codes)}，请先确认数据目录 settings.DATA_ROOT_PATH={settings.DATA_ROOT_PATH}")

    opt = PortfolioHyperparameterOptimizer()

    cfg = PortfolioOptConfig(
        lambda_drawdown=args.lambda_dd,
        max_strategies=args.max_strategies,
        allow_strategy_selection=True,
    )

    # progress: print every 10 trials (Top3 snapshot)
    progress_state = {"best": []}  # list of dicts sorted by score desc

    def on_progress(info):
        # info: {trial, score, annualized_return, max_drawdown, chosen}
        try:
            rec = {
                "trial": int(info.get("trial")),
                "score": float(info.get("score")),
                "annualized_return": float(info.get("annualized_return")),
                "max_drawdown": float(info.get("max_drawdown")),
                "chosen": info.get("chosen"),
            }
        except Exception:
            return

        best = progress_state["best"]
        best.append(rec)
        best.sort(key=lambda x: x["score"], reverse=True)
        del best[10:]

        if (rec["trial"] + 1) % 10 == 0:
            print(f"\n[progress] trials={rec['trial']+1}/{args.trials}")
            for i, b in enumerate(best[:3], 1):
                print(
                    f"  Top{i}: score={b['score']:.4f} ann={b['annualized_return']:.4f} "
                    f"mdd={b['max_drawdown']:.4f} chosen={b['chosen']}"
                )

    result = __import__("asyncio").run(
        opt.optimize(
            all_stock_codes=all_codes,
            start_date=start_date,
            end_date=end_date,
            universe_size=args.universe_size,
            universe_seed=args.seed,
            n_trials=args.trials,
            cfg=cfg,
            progress_callback=on_progress,
        )
    )

    # 写出结果
    import os

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== BEST ===")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2)[:2000])
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
