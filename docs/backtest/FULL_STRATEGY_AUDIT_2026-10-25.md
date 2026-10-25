# 全策略代码审计 + 统一口径复验报告

**日期**: 2026-10-25
**范围**: `backend/app/services/backtest/strategies/` 全部 22 个实现类（30 个注册名，含别名）
**口径**: 3 年全市场 5513 只（2023-05-12 → 2026-05-08）/ 100 万 / 佣金 0.0003 + 滑点 0.0001 / 现金预留 5% / 100 股整手 / 日频调仓

---

## 一、结论摘要

1. **所有策略的注册代码都能跑，但 9 处真实缺陷**让其中 5 条的��测数字"能看不能用"。
2. **最致命的是单仓上限从未传到引擎**：`max_position_size` 默认 0.2，TopK 等权组合
   （topk=30/50/60）建仓到现金耗尽即冻结，topk=60 实测只买到 **9 只**。
   修复后全部 TopK 策略的持仓数与交易数回到设计值。
3. **qlib 的 `MA*`/`STD*` 不是价格均线**，是归一化因子值（实测相关系数 0.087），
   Bollinger / Cointegration 拿它算指标 = 用垃圾指标。修复后 Bollinger 夏普
   1.335 → **0.924**，回撤 -41.1% → **-27.0%**。
4. **ML 类 4 条策略 `joblib.load` 段错误 rc=139**（numpy 2.x `multiarray.scalar`
   签名变更 + LightGBM Booster 原生句柄）。修复后 5 个 official-baseline 模型
   全部可加载，3 条策略从"完全不可用"变成能跑。
5. **周频全面优于日频**（4/4 条 TopK 策略），这是实盘成本约束下的决定性变量。
6. **记忆库中"MAX 符号修正"的结论被 10 年 PIT 复验证伪**：只在 22 因子核心版
   成立，叠加 Alpha158 面板因子后是负贡献。

---

## 二、代码审计：9 个真实缺陷

### 致命级

#### D1. 策略自算的单仓上限从未传给引擎

- **位置**: `backend/app/services/backtest/core/base_strategy.py`（新增钩子）、
  `backend/app/services/backtest/execution/backtest_executor.py:277-294`、
  `strategies/{honest_composite,rsi_71421,gbm_voltgt,factor_ranking}_strategy.py`
- **现象**: 策略内部算 `max_position_size=(1-cash_reserve)/topk`，但引擎的
  `PortfolioManagerArray._execute_buy` 只读 `config.max_position_size`
  （默认 0.2）。策略的 `self.max_position_size` 只出现在 logger 里。
- **实证**（500 只 / 2024H1，honest_composite topk60）：
  | | 交易数 | 持仓 | 收益 | 夏普 |
  |---|---|---|---|---|
  | 不传 mps（引擎 0.2） | 9 | 9 | -23.2% | -1.123 |
  | 显式传等权 mps | 559 | 59 | -12.4% | -0.947 |
- **影响**: 任何走 API/脚本/直接调 executor 且没手动传 `max_position_size` 的
  TopK 策略，都会静默退化成重仓押注少数几只票。
- **修复**: `BaseStrategy.get_position_size_hint()` 新钩子 + executor 在
  `"max_position_size" not in strategy_config` 时采用策略建议值（显式传值以调用方为准）。
- **回归**: `backend/tests/unit/backtest/test_strategy_position_size_hint.py`（8 例）

#### D2. qlib 的 `MA*` / `STD*` 是归一化因子值，不是价格均线

- **位置**: `backend/app/services/backtest/strategies/strategies.py`
  Bollinger `calculate_indicators`（原 46-58 行）、Cointegration（1115-1127 行）
- **实证**（`000001_SZ.parquet`）:
  | | qlib 列 | 真实值 | 相关系数 |
  |---|---|---|---|
  | MA20 | 1.073 | 45.49 | **0.087** |
  | STD20 | 0.047 | 1.99 | — |
  | RSI14 | 60.15 | 60.1533 | **1.000000**（最大差 0.0003）|
- **补充**: 5463 个 parquet 中仅 272 个（5.0%）的 `MA20`/`STD20` 是全量填充，
  95% 是部分填充 → 同一策略内有的票走预计算、有的走 fallback，行为不一致。
- **修复**: 两个策略统一改用真实 `close` 计算 rolling。
- **效果**: Bollinger 3 年全市场 夏普 1.335 → **0.924**，回撤 -41.1% → **-27.0%**。
- **注意**: `RSI14` 是正确的，`rsi_optimized.py` 用它**无需改**。

#### D3. ML 类策略 `joblib.load` 段错误 rc=139

- **位置**: `prediction_engine.py::_load_model_file`、
  `qlib/unified_qlib_training_engine.py::load_qlib_model`、
  `strategies/lightgbm_strategy.py::_load_model`
- **根因**（二分定位，非猜测）:
  1. pickle 用 `numpy._core.multiarray.scalar` 反序列化，numpy 2.4.4 签名已改成
     `scalar(dtype, obj)`，老 pickle 是 `scalar(obj, dtype)` → 参数错位；
  2. 修好 1 之后仍崩在 LightGBM `Booster.__setstate__` 的原生句柄恢复。
- **段错误无法 try/except**，必须在 unpickle 阶段规避。
- **修复**: 新增 `backend/app/core/model_loader.py`
  （`safe_pickle_load` / `safe_joblib_load` + 兼容 unpickler），
  3 处调用点统一改用。
- **效果**: 5 个 official-baseline 模型全部加载成功（rc=0，booster 有树、
  158 特征、predict 可用）；3 条 ML 策略从"完全跑不起来"→ 能跑。
- **注**: `model_loader` 必须放 `app/core/` 而非 `app/services/backtest/` ——
  放 backtest 会形成 prediction → backtest → strategies → prediction 循环导入。

### 严重级

#### D4. `low_volatility` 只生成 BUY、从不生成 SELL

- **位置**: `strategies.py::LowVolatilityStrategy.generate_signals`（原 1627-1666 行）
- **现象**: 触发条件是 `current_rar > 0`（风险调整收益为正，**不看波动率高低**），
  只有 BUY 分支。3 年全市场 **8 笔成交，全是 BUY**，7 只票各持有 660 天。
- **本质**: 不是策略表现，是"7 只股票满仓躺 3 年"。夏普 0.579 无意义。
- **修复**: 重写为全市场横截面 z-score（低波动 + 低 RAR 波动，均 sign -1）
  → TopK 等权持仓 → 掉出 TopK 按 n_drop 卖出，与 `honest_composite` 同语义。

#### D5. `multi_factor` 用时序标准化 → 只买不卖

- **位置**: `strategies.py::MultiFactorStrategy`
- **现象**: 分数是**个股自身** 60/252 日滚动分位（时序），不是横截面。
  绝大多数股票长期为正 → 持续 BUY；SELL 只在"综合分转负"时触发，极少。
  3 年 117 笔成交（BUY 78 / SELL 39），持仓中位数 15。
- **修复**: 三因子（value / momentum / low_volatility）改**横截面** z-score
  等权合成 → TopK/Dropout。

#### D6. `StockDataLoader` 返回 58% NaN 价格

- **位置**: `backend/app/services/data/stock_data_loader.py::load_stock_data`
- **成因**: qlib 预计算 parquet 是两段拼接 —— block1 = 交易日 OHLCV + 传统指标，
  block2 = Alpha158（含节假日行，无价格）。实测 `600000_SH`：2514 行里
  1304 行（51.9%）OHLCV 全 NaN；`600000.SH` 2024Q1 = 37/64 行无价格。
- **影响**: 下游特征提取器直接抛"列 open 缺失值过多: 136/252"；
  任何依赖 `StockDataLoader` 的路径都拿到 58% NaN 的价格。
- **修复**: 数据层剔除无价格行（`dropna(subset=["open","close","volume"])`）；
  `qlib/enhanced_qlib_provider.py` 合并 Alpha158 后同样剔除。

#### D7. `lightgbm` 策略 0 交易但"看起来跑通了"

- **位置**: `strategies/lightgbm_strategy.py::_get_prediction_series`
- **现象**: 只认本地 `lgb_model.txt`（本机不存在）→ `FileNotFoundError` 被上层
  吞掉 → 整条策略 0 信号 0 交易。回测"成功"返回但什么也没做。
- **修复**: 本地 booster 不可用时回落到基类 `PredictionEngine` 路径（用 `model_id`）。

#### D8. `lightgbm` 策略 `str / str` TypeError

- **位置**: `strategies/lightgbm_strategy.py:38`
- **成因**: `settings.MODEL_STORAGE_PATH` 是 `str`，代码用 `settings.MODEL_STORAGE_PATH / "lgb_model.txt"`。
- **修复**: 先 `Path(...)` 再拼接。

### 中等

#### D9. 引擎零消费 `stop_loss_pct` / `take_profit_pct` / `rebalance_frequency`

- **证据**: `grep -rn "stop_loss_pct|take_profit_pct" backend/app/services/backtest/`
  在 `core/` + `execution/` 下**零命中**；`rebalance_frequency` 只在
  `strategies.py:1311` 被一个因子策略读去做日志。
- **影响**: 基线配置里的"止亏 5% / 止盈 15%"是摆设，任何"加止损能改善"的
  结论都不成立（历史上止损 A/B 已被证伪，但结论本身是错的）。
- **处理**: 周频改由**策略侧分数掩码**实现（`rebalance_days="weekly"` →
  非周五分数置 NaN，执行器自然不换仓）；止盈止损在实验 runner 里统一传 0。

---

## 三、统一口径全策略复验（3 年全市场，修复 mps 后）

| 策略 | 收益 | 夏普 | 回撤 | 交易数 | 持仓 | 状态 |
|---|---|---|---|---|---|---|
| bollinger | +150.6% | 1.335 | -41.1% | 2354 | 159 | ⚠️ 修复 MA/STD 后 → **0.924 / -27.0%** |
| stochastic | +102.4% | 1.086 | -26.1% | 1865 | 201 | |
| macd | +125.0% | 1.051 | -46.7% | 13921 | 147 | |
| gbm_voltgt | +59.6% | 0.780 | -33.7% | 2930 | 60 | |
| cci | +53.6% | 0.746 | -40.1% | 6174 | 242 | |
| factor_ranking | +30.5% | 0.720 | -15.2% | 5054 | 50 | |
| honest_composite | +33.1% | 0.718 | -20.3% | 6660 | 60 | |
| momentum_factor | +81.2% | 0.697 | -40.1% | 2238 | 119 | |
| multi_factor | +61.0% | 0.659 | -25.3% | 117 | 36 | ⚠️ 修复后重跑中 |
| kdj | +47.7% | 0.647 | -35.9% | 1074 | 158 | |
| pairs_trading | +61.6% | 0.606 | -44.8% | 850 | 19 | 500 只宇宙 |
| low_volatility | +40.1% | 0.579 | -33.4% | **8** | 7 | ⚠️ 修复后重跑中 |
| obv | +46.2% | 0.573 | -41.1% | 28540 | 260 | |
| value_factor | +52.5% | 0.549 | -29.4% | 1624 | 151 | |
| ml_ensemble_lgb_xgb_riskctl | +28.3% | 0.460 | -45.1% | 31930 | 204 | 跑 100+ min |
| mean_reversion | +28.0% | 0.448 | -47.9% | 483 | 15 | 500 只宇宙 |
| cointegration | +21.7% | 0.370 | -44.0% | 174 | 15 | 500 只宇宙 |
| rsi_71421 | +13.2% | 0.328 | -26.7% | 4092 | 30 | |
| rsi | +15.5% | 0.326 | -33.4% | 2408 | 162 | |
| moving_average | **-43.1%** | **-0.498** | -54.3% | 2579 | 151 | 唯一负收益 |
| lightgbm / model_signal / model_topk_dropout | — | — | — | — | — | 段错误已修，3 年全市场复跑未完成 |

**基准**（全市场等权买入持有）: +56.0% / 夏普 0.737 / 回撤 -29.9%

**读数注意**:
- `moving_average` 是唯一负收益策略（-43.1%），策略方向本身可能有问题，需单独诊断。
- 技术类策略（macd/stochastic/cci/obv）持仓 147~260 只，`max_position_size=0.2`
  下每只仅约 0.5% 仓位、近半资金闲置 —— 这是下一处可查的优化点。
- `ml_ensemble_lgb_xgb_riskctl` 交易 31930 笔（3 年 723 天 = 44 笔/天），换手不可信。

---

## 四、优化一：周频 vs 日频（唯一变量是调仓频率）

引擎不消费 `rebalance_frequency`，周频由策略侧分数掩码实现。

| 策略 | 日频夏普 | 周频夏普 | 日频交易 | 周频交易 | 日频回撤 | 周频回撤 |
|---|---|---|---|---|---|---|
| honest_composite | 0.718 | **0.965** | 6660 | 1390 | -20.3% | **-17.6%** |
| rsi_71421 | 0.328 | **0.803** | 4092 | 828 | -26.7% | **-16.3%** |
| gbm_voltgt | 0.780 | **0.835** | 2930 | 620 | -33.7% | **-29.8%** |
| factor_ranking | 0.720 | **0.768** | 5054 | 1160 | -15.2% | **-14.2%** |

**4/4 全面更优。** 换手砍 4~5 倍（佣金 0.03% + 印花税 0.05% + 滑点 0.1%/单边），
省下的成本远超放弃的择时频率。rsi_71421 收益 +13.2% → **+37.8%**。

落地：4 条 TopK 策略均新增 `rebalance_days` 参数
（`daily` 默认 / `weekly`=周五 / `monthly`=月初），周频掩码在 `prepare_backtest_data`
里把非调仓日分数置 NaN，执行器无分可依即不换仓。

---

## 五、优化二：因子符号修正（证伪，未落地）

**设计**: 10 年 PIT 宇宙（4329 只，2016-05-12 → 2026-05-08），
只改因子符号表，其余（窗口/宇宙/费率/执行）全同。
基线格 `base_42f_topk60` = **0.8201**，与记忆库 0.8193 吻合 → 基线复现门通过。

| 变体 | 因子数 | topk60 夏普 | topk30 夏普 | OOS(16-21) | in-sample(21-26) |
|---|---|---|---|---|---|
| base_42f（现状） | 42 | **0.8201** | **0.8865** | **0.4963** | 1.0831 |
| max3060_plus（MAX30/60→+1） | 42 | 0.6907 | 0.7223 | 0.3338 | 0.9842 |
| signfix_22f_core（剔 MAX5/10/20 + MAX30/60 +1） | 22+17 | 0.3893 | 0.2853 | 0.1324 | 0.6022 |
| signfix_22f_core_noPanel | 22 | 0.3219 | 0.1904 | 0.0956 | 0.5150 |

**结论：记忆库中"MAX 符号修正 = edge"的结论只在 22 因子核心版成立；
叠加 17 个 Alpha158 面板因子后，MAX 的 -1 符号反而更优。**
即：42f 版本不需要符号修正，**不做这个改动**。

档案：`data/strategy_baseline/ic_composite_10y_signfix.json`
脚本：`backend/scripts/ic_composite_10y_signfix.py`

---

## 六、最终纸面盘策略决定

**4 条独立线，各 100 万，各持仓、各账本，同起始日、同宇宙、同成本模型。**

| 线 | 因子源 | 周频夏普 | 回撤 | 持仓 | 角色 |
|---|---|---|---|---|---|
| `honest_42f` | 42 因子等权（23 基础 + 17 Alpha158） | **0.965** | -17.6% | 60 | 主力 |
| `gbm_voltgt` | LightGBM 元模型 + vol-target | 0.835 | -29.8% | 60 | 进攻 |
| `rsi_71421` | RSI 7/14/21 均值回归 + 低波动 + 量价 | 0.803 | -16.3% | 30 | 稳健 |
| `factor_ranking` | 动量/低波动/低振幅 | 0.768 | -14.2% | 50 | 低回撤 |

**入选理由**: 因子源互不重叠（等权复合 / 元模型 / RSI 超跌 / 量价动量），
失败时不会一起死；全是周频低换手；回撤可控。

**未入选**:
- `22f` —— 周频夏普 0.685，10 年 OOS 仅 0.041，作为 4 条线之一价值不足
- 所有日频线 —— 实盘成本下不可做
- `lightgbm` / `model_signal` / `model_topk_dropout` —— 模型是 csi300/50 只域
  且训练窗覆盖回测期（前视），不进纸面盘
- `moving_average` —— 唯一负收益策略

**落地**:
- 脚本 `backend/scripts/paper_portfolio.py`（4 条线独立持仓 + 独立账本 +
  ST 过滤 + 真实成本 + 整手 + 涨停/跌停不可成交检查）
- 周五 cron 已从单一 `weekly_signal.py` 切到 `paper_portfolio.py`
  （`scripts/daily_tdx_update.sh`）
- 输出 `data/paper/{strategy}_holdings.json` / `{strategy}_ledger.csv` /
  `portfolio.json` + `frontend/public/paper-portfolio.json`

---

## 七、ML 类策略修复验证

`app/core/model_loader.py` 落地后,3 条 ML 策略从"段错误 rc=139 完全不可用"
变成能跑出真实数字：

| 策略 | 修复前 | 修复后（3 年全市场） |
|---|---|---|
| lightgbm | 段错误 rc=139 | **+27.1% / 夏普 0.430 / 回撤 -33.4% / 692 笔** |
| model_signal | 段错误 rc=139 | 复跑中（qlib 路径逐票预测，耗时 > 100 min） |
| model_topk_dropout | 段错误 rc=139 | 待复跑 |

**但这 3 条仍不进纸面盘**，原因与修复无关：
- 模型域是 csi300 / 50 只，本平台是全市场 5513 只
- 训练窗口覆盖回测期（前视）
- 逐票 PredictionEngine 极慢（lightgbm 单条 3 年全市场 ~40 min CPU）

它们的定位是「证明模型加载链路已修好」，不是候选策略。

---

## 八、moving_average 负收益根因诊断（全市场 5513 只 / 3 年）

`moving_average` 是唯一负收益策略（-43.1%），做了根因拆解。

**第一步：500 只子集上的分解**（4 格，每格只改一个变量）

| 变体 | 收益 | 夏普 | 回撤 | 交易数 |
|---|---|---|---|---|
| base 5/20 阈值0.5% | -5.6% | 0.187 | -66.2% | 690 |
| 成本归零 | -5.2% | 0.189 | -66.1% | 725 |
| 阈值=0 | **+22.1%** | 0.390 | -65.5% | 1974 |
| 长窗口 20/60 | **+61.3%** | 0.622 | -46.6% | 79 |
| 反向（死叉买） | +36.2% | 0.470 | -51.2% | 8 |

结论：负收益**不是交易成本造成的**（成本归零几乎不变），是**信号质量**问题。

**第二步：全市场复验 → 500 只的乐观结论被推翻**

| 变体 | 500 只 | 全市场 5513 |
|---|---|---|
| base 5/20 阈值0.5% | -5.6% | -43.1% |
| 阈值=0（5/20） | +22.1% | **-43.3%**（交易 10355 笔）|
| 20/60 阈值0.5% | **+61.3%** | **-23.1%**（交易 61 笔）|
| 20/60 阈值0 | — | **-1.5%**（夏普 +0.110，回撤 -36.8%）|
| 10/60 阈值0 | — | -18.4% |

**500 只清单是排序尾部选取、不是随机样本** → 500 只上 +61.3% 的"最优格"
在全市场只有 -23.1%。这是本项目第四次踩到同一条规则。

**最终处置**：
- 默认参数改为全市场最优格（20/60 + 阈值 0）：-43.1% → **-1.5%**，夏普 -0.498 → +0.110
- 但**仍然不赚钱**，趋势跟随在 A 股不 work（与"市场 MA200 全进全出"证伪、
  "residual_mom 最差"同源）→ **不列入纸面盘**，preset 描述里明确标注不建议使用
- 保留旧 preset（5/20 + 0.5%）作历史对比

runner：`backend/scripts/moving_average_fix.py`
档案：`data/strategy_baseline/moving_average_fix_3y.json`

---

## 九、本次未完成 / 下一步

1. `model_signal` / `model_topk_dropout` 的 3 年全市场复跑（耗时很长）
2. 技术类策略持仓 147~260 只、`max_position_size=0.2` 下每只仅约 0.5% 仓位、
   近半资金闲置 —— 下一处可查的优化点
3. `multi_factor` 修复后夏普偏低（周频 0.411）的原因：旧版"7 只躺 3 年"的
   假夏普被剔除后真实表现更差，需要判断该策略是否还有保留价值
4. `ml_ensemble_lgb_xgb_riskctl` 单条 3 年全市场跑 100+ min 且交易 31930 笔
   （44 笔/天），换手不可信，未做周频对照
