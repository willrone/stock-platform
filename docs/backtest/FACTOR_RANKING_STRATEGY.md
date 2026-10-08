# FACTOR_RANKING 横截面因子排名策略（TopK/Dropout）

> 2026-10-08 新增。全市场选股的标准范式：横截面打分 → 每日排名 → 持有前 K 名 → 每日换出最差 n_drop 只。
> 解决阈值信号类因子策略（multi_factor 等）在全市场场景下"碰信号就买、无组合纪律"的结构性缺陷。

## 1. 背景与动机

同窗口（2023-05-12 → 2026-05-08）全市场回测暴露的问题：

| 策略 | 总收益 | 夏普 | 最大回撤 | 备注 |
|---|---|---|---|---|
| 全市场等权基准 | +56.7% | 0.63 | -31.2% | 无脑等权买入持有 |
| multi_factor（修复前口径） | +64.3% | 0.56 | -22.6% | 5033 只仅成交 323 只，期末 140 个碎仓 |
| multi_factor（修复后口径） | +60.8% | 0.24 | -39.3% | 仓位上限修正后暴露真实集中度：26 仓、波动 73% |
| momentum_factor | +20.4% | 0.10 | -50.5% | |
| portfolio 组合（加权投票） | +17.8% | 0.12 | -50.5% | 投票集成被差成分拖垮 |

阈值信号框架的天花板（夏普 0.56）低于无脑等权（0.63）。根因：逐股独立出信号、
无横截面比较、无组合构建纪律。行业标准做法（qlib TopkDropout、券商金工复合打分研究）
是横截面排名 + TopK 等权 + Dropout 换仓。

## 2. 架构

```
prepare_backtest_data (回测准备阶段, 一次性)
  ├─ 全市场矩阵化 close/high/low（master 日期对齐）
  ├─ 三因子: momentum(126日, 跳过20日) / low_vol(-std60) / low_amp(-振幅10)
  ├─ 按日期横截面 Z-Score → 复合分 = 加权平均(截面有效因子)
  └─ 每股存 score_map {datetime: score} 到 data.attrs

generate_signals (主循环逐股查表, O(1))
  └─ 输出 ranking_score 信号（metadata.ranking_score）

TopkDropoutTradeModeExecutor (现成执行端, trade_modes.py)
  ├─ 每日按 score 全市场排名
  ├─ 持仓 < topk → 补齐前 topk 名
  ├─ 排名跌出前 topk 的持仓 → 最差 n_drop 只卖出
  └─ 未持有的前 n_drop 名 → 买入
```

注册名：`factor_ranking`（`StrategyFactory`）。

## 3. 参数（runner: `scripts/run_bt_factor_ranking.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| topk | 50 | 持股数量 |
| n_drop | 5 | 每日换出数量（换手率 ≈ 2×5/50 = 20%/日上限） |
| hold_thresh | 0 | 卖出前最少持有天数 |
| momentum_window / skip | 126 / 20 | 动量回看与跳过期 |
| vol_window / amp_window | 60 / 10 | 低波/低振幅窗口 |
| weights | {momentum, low_vol, low_amp: 1.0} | 因子权重 |

建议 BacktestConfig：`initial_cash=1,000,000`（100 万让单仓 ~2 万，规避 100 股
手数买不起高价股）、`max_position_size=0.019`（50×1.9%≈95%）、
`stop_loss/take_profit` 关闭（换仓由排名驱动，不需逐股止盈止损）。

## 4. 全市场回测结果（5033 只生效宇宙, 723 交易日, wall 199s）

| 指标 | factor_ranking | multi_factor（修复后） | 等权基准 |
|---|---|---|---|
| 总收益 | +38.5% | +60.8% | +56.7% |
| 年化 | +11.5% | +17.2% | +16.3% |
| 年化波动 | **12.5%** | 73.1% | 25.9% |
| 夏普 | **0.92** | 0.24 | 0.63 |
| 最大回撤 | **-10.6%** | -39.3% | -31.2% |
| 期末持仓 | 恒定 50 | 26（现金 199 元） | — |
| 交易笔数 / PF | 3326 / 1.50 | 96 / 4.71 | — |

注：multi_factor 为单仓上限修复后的同口径重跑；修复前口径为 +64.3%/0.56/-22.6%
（微仓分散掩盖了真实集中度），两口径不可与修复后混比。

收益略低于阈值基线，但风险指标全面占优：夏普 0.92 vs 0.56/0.63，回撤不到基线一半。

## 5. 配套修复：单仓上限计算 bug（portfolio_manager）

`_execute_buy` 原实现 `get_portfolio_value({stock_code: price})` 只传单只要买的
股票 → 其它持仓按 0 计价 → 单仓上限 = `max_position_size × 现金` → 每笔买入后
现金几何衰减，资金无法充分部署（TopK 策略实测只部署 63%，现金三年单调上涨）。
修复：`execute_signal` 将完整 `current_prices` 传入 `_execute_buy`。
`portfolio_manager.py`（旧管理器）同口径修复。

修复效果（300 只冒烟对照）：期末现金 59.7万 → 15.8万（部署 40% → 84%），
总收益 +14.7% → +30.8%，夏普 0.543 → 0.587。全市场 multi_factor 同口径重跑：
+64.3%→+60.8%、夏普 0.56→0.24、回撤 -22.6%→-39.3%（修复暴露了真实的持仓集中度，
修复前的微仓分散是仓位上限低估造成的假象）。

> 影响面：所有使用 max_position_size 的回测。现金/手数为紧约束的策略
> （如 10 万本金的 multi_factor）受影响微小；受限仓位策略（TopK）影响显著。
> 2026-10-08 之前的历史回测数字与此口径不完全可比。

## 6. 验证

- 单元测试：`tests/unit/backtest/` 35 个文件全绿（逐文件独立子进程）
- lint：black / isort / flake8 全过
- 冒烟：300 只 6.2s；全市场 5513 只 199s

```bash
cd backend
env -u PYTHONPATH .venv/bin/python3 scripts/run_bt_factor_ranking.py      # 全市场
env -u PYTHONPATH .venv/bin/python3 scripts/run_bt_factor_ranking.py 300  # 冒烟
```
