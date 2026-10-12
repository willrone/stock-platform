# 全策略统一基线（Full-Market 3-Year Baseline）

> 2026-10-12 立项并跑完。回答「每个策略有没有回测基准数据」：**有**——
> 16/20 策略在同一宇宙、同一窗口、同一费率下跑完，全部对齐同一条等权基准；
> 4 个 ML 策略因模型不可用登记原因（不跑）。

## 1. 统一口径（所有策略完全相同）

| 项 | 值 |
|---|---|
| 窗口 | **2023-05-12 → 2026-05-08**（723 交易日，本地 qlib 数据止于 05-08） |
| 宇宙 | `data/qlib_data/features/day/*.parquet` 全量 **5513 只**（沪深北） |
| 统计套利宇宙 | 500 只（O(N²) 配对扫描约束；固定清单 `baseline_universe_500.txt`） |
| 初始资金 | 1,000,000 |
| 费率 | commission 0.0003 / slippage 0.0001 |
| 仓位 | max_position 0.2 / 现金预留 5% / 整数倍 100 |
| 止盈止损 | 5% / 15%（factor_ranking 例外，见 §4） |
| 调仓 | 日频 |
| 策略参数 | 全部代码默认值 |
| **基准** | **全市场 5513 只等权买入持有**：+56.0% / 夏普 0.74 / 回撤 -29.9%（只算一次，全部策略共用同一序列） |

## 2. 排行榜（16 策略 + 基准，统一窗口）

| 策略 | 宇宙 | 总收益 | 夏普 | 最大回撤 | 交易笔数 | vs 等权超额(年化) | IR |
|---|---|---|---|---|---|---|---|
| 策略 | 宇宙 | 总收益 | 夏普 | 最大回撤 | 交易笔数 | vs等权超额(年化) | IR |
|---|---|---|---|---|---|---|---|
| stochastic | 5513 | 112.4% | 0.898757 | -26.7% | 1929 | +13.3% | 0.467495 |
| bollinger | 5513 | 80.2% | 0.611192 | -27.3% | 2490 | +6.0% | 0.268783 |
| obv | 5513 | 77.2% | 0.549064 | -46.7% | 28351 | +5.3% | 0.291814 |
| value_factor | 5513 | 65.8% | 0.09324 | -75.1% | 1854 | +2.5% | 0.671761 |
| pairs_trading | 500 | 65.1% | 0.101295 | -58.6% | 838 | +2.3% | 0.872477 |
| multi_factor | 5513 | 57.9% | 0.21153 | -40.3% | 116 | +0.5% | 0.342051 |
| macd | 5513 | 49.9% | 0.348912 | -48.2% | 13448 | -1.6% | 0.10733 |
| kdj | 5513 | 47.7% | 0.356834 | -35.9% | 1079 | -2.2% | 0.071556 |
| momentum_factor | 5513 | 44.5% | 0.105465 | -57.1% | 2374 | -3.1% | 0.575492 |
| low_volatility | 5513 | 40.1% | 0.245693 | -47.0% | 8 | -4.3% | 0.100187 |
| factor_ranking | 5513 | 38.5% | 0.924055 | -10.6% | 3326 | -4.7% | -0.327642 |
| cci | 5513 | 37.0% | 0.442244 | -39.8% | 5654 | -5.2% | -0.249691 |
| mean_reversion | 500 | 21.6% | 0.039104 | -60.5% | 483 | -9.7% | 0.733204 |
| cointegration | 500 | 21.6% | 0.199757 | -44.0% | 176 | -9.7% | -0.288019 |
| rsi | 5513 | 19.6% | 0.216664 | -33.4% | 2486 | -10.3% | -0.422968 |
| moving_average | 5513 | -38.1% | -0.444713 | -54.7% | 2533 | -32.2% | -1.457109 |
| **等权基准** | **5513** | **56.0%** | **0.736993** | **-29.9%** | — | — | — |

## 3. 数据层根因修复（本次关键发现）

**症状**：bollinger / cointegration 全市场回测 **0 信号、0 交易**，看似"策略无效"。

**根因**（数据文件级 bug，非策略逻辑）：
`data/qlib_data/features/day/*.parquet` 每个文件里存了**两个 stock_code 条目**
（点格式 `000001.SZ` + 下划线 `000001_sz`），两者列覆盖**互补**：
- 点条目：close/high/low/BOLL*/KDJ*/RSI*/MACD 有效，但 **MA5~MA60/STD5~STD30 全 NaN**（STD60 有效）
- 下划线条目：MA*/STD* 有效，但 close 全 NaN

`QlibFormatConverter._filter_by_stock_code` 只取点条目 → 188 列模式下
bollinger/cointegration 映射到的 `MA{period}`/`STD{period}` 列**整列 NaN** →
`_extract_indicators_from_precomputed` 只检查"列是否存在"不检查全 NaN →
策略不触发 rolling 回退 → percent_b/zscore 全 NaN → 静默 0 信号。

**修复**：`core/base_strategy.py::_extract_indicators_from_precomputed` 增加回归防护——
映射列整列为 NaN 时返回 None 让策略 fallback 到现场计算（rolling），并打 warning。
验证（150 只探针，188 列路径）：bollinger 0→5838 信号、cointegration 0→3644 信号，
其余策略（rsi/macd/kdj/obv 等）数值不变，无回归。

**遗留（数据层，未动）**：双条目本身是 qlib 特征构建时点的历史产物
（`000001_sz` 条目 1304 行 vs 点条目 1210 行，下划线条目 close 全空）。
彻底修需重跑特征构建（`qlib_bin_converter` 系列）去重；本次只加了消费端防护，
不动数据文件以免影响其它已验证结果。

## 4. 策略级口径例外

| 策略 | 例外项 | 原因 |
|---|---|---|
| factor_ranking | max_position 0.2→**0.019**、止盈止损关闭 | TopK(50) 等权组合：100 股/手约束下 0.2 单仓上限导致整组合建仓被拒（8 笔假象）；换仓由排名驱动不需逐股止盈止损。与 `FACTOR_RANKING_STRATEGY.md` §3 推荐口径一致 |

## 5. 未跑策略（ML 4 个）

| 策略 | 原因 |
|---|---|
| ml_ensemble_lgb_xgb_riskctl | official-baseline pkl `joblib.load` 段错误 rc=139；模型 csi300/50 只域 + 训练窗口覆盖回测期（前视） |
| lightgbm | 同上 |
| model_signal | 依赖 official-baseline 预测引擎，pkl 加载段错误 + 域不匹配 + 前视 |
| model_topk_dropout | 同上 |

要用需先 walk-forward 重训（另立项）。

## 6. 产物与复现

| 产物 | 路径 |
|---|---|
| 基线 runner | `backend/scripts/run_strategy_baseline.py`（断点续跑：已完成策略默认跳过，`--rerun` 强制） |
| 基线数据 | `data/strategy_baseline/baseline_full_market_3y.json`（含每策略 metrics + vs 等权基准对齐 + perf） |
| 等权基准序列 | 同上 `benchmark.daily_value`（723 点） |
| 统计套利宇宙清单 | `data/strategy_baseline/baseline_universe_500.txt` |
| DB 注册 | `backend/scripts/register_strategy_baseline.py` → backtest_results/backtest_benchmarks/tasks 三表 |

```bash
cd backend
# 全量（约 35min, 断点续跑）
env -u PYTHONPATH .venv-py313/bin/python3 scripts/run_strategy_baseline.py --batch all
# 只补单策略
env -u PYTHONPATH .venv-py313/bin/python3 scripts/run_strategy_baseline.py --strategies rsi
# 注册进 DB
env -u PYTHONPATH .venv-py313/bin/python3 scripts/register_strategy_baseline.py
```

**已知口径差异**：等权基准 5513 只 = +56.0%（本文档口径）vs 10-08 文档 5033 只 = +56.7%
——差异来自宇宙大小（5513 含更多北交所低波动票），两者不可直接混引；本文档一律 5513 口径。
