# 默认策略配置预设

这些配置由 `backend/scripts/seed_strategy_configs.py` 写入 `/api/v1/strategy-configs` 使用的 `strategy_configs` 表，目标是把“可复用策略参数”资产化，而不是只散落在代码或临时请求里。

## 使用方式

```bash
cd backend
PYTHONPATH=. .venv-py313/bin/python scripts/seed_strategy_configs.py --create-tables
```

脚本是幂等的：

- 首次运行会创建 6 条默认配置。
- 再次运行不会重复插入。
- 如果默认配置内容升级，会按稳定 `config_id` 更新已有 preset。

运行后可检查：

```bash
curl -s http://127.0.0.1:18082/api/v1/strategy-configs | python -m json.tool
```

## 分层口径

| 层级 | 用途 | 说明 |
|---|---|---|
| `smoke/*` | 链路检查 | 跑得快，只验证 API / 数据 / 回测链路，不评价收益。当前 smoke 由 `make smoke-backtest` 直接发起短样本请求。 |
| `benchmark/*` | 固定基准 | 参数固定，用于跨策略、跨窗口横向比较。 |
| `research/*` | 研究候选 | 可进入参数扰动、优化、WFO、样本外验证。 |
| `model/*` | 模型排序 | 统一承载模型驱动策略参数。 |

## 当前预置配置

| config_name | strategy_name | 关键参数 | 用途 |
|---|---|---|---|
| `benchmark/moving_average_5_20_threshold_005` | `moving_average` | `short_window=5`, `long_window=20`, `signal_threshold=0.005` | MA 标准 benchmark；默认 0.5% 阈值，避免历史 2% 阈值过滤过多金叉/死叉。 |
| `benchmark/rsi_optimized_default` | `rsi` | `rsi_period=14`, `oversold_threshold=30`, `overbought_threshold=70`, `trend_ma_period=50` | RSI 优化版默认 benchmark。 |
| `benchmark/macd_default` | `macd` | `fast_period=12`, `slow_period=26`, `signal_period=9` | 经典 MACD benchmark。 |
| `benchmark/bollinger_20_2` | `bollinger` | `period=20`, `std_dev=2` | 经典布林带 benchmark。 |
| `research/portfolio_technical_vote_v1` | `portfolio` | MA / RSI / MACD 加权投票 | 技术指标组合 research 候选，用于后续优化和 walk-forward 对照。 |
| `model/topk_dropout_k10_drop2` | `model_topk_dropout` | `topk=10`, `n_drop=2`, `trade_mode=topk_dropout` | 模型排序类默认配置。 |

## 注意

- `benchmark/*` 只代表固定比较口径，不代表“能赚钱”。
- `smoke` 短样本只验证链路；如果 0 交易，`make smoke-backtest` 会输出：`短样本仅验证链路，不评价策略收益`。
- 后续进入可信策略评估时，这些配置应配合数据质量、交易仿真、样本外、稳健性和报告解释层一起使用。
