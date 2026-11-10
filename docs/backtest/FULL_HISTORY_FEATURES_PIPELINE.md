# qlib 官方全历史 → 平台 42 因子特征管线

**产出**：`data/qlib_data/features_full/{code}_{MKT}.parquet`
**脚本**：`backend/scripts/build_full_history_features.py`
**对账报告**：`data/backtest/full_history_reconcile.json`
**日期**：2026-10-26

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 复权方案 | **不做复权**。qlib 官方数据没有 `$factor` 文件，存的就是未复权真实价；features/day 的 `$close` 同样未复权 → 两者基准天然一致 |
| 口径对账 | **通过**。可比口径（单段参考文件）：17 个面板因子 + 25 个基础公式 + 5 个价格列，相关系数 **1.0**，maxdiff **0.0**（逐位相等） |
| 覆盖 | **5513 / 5513** 只（qlib 官方 instruments 全量），从上市首日到退市日，构建失败 0 |
| 起始日期 | median **2016-07-12**、min **1990-12-19**；**78.5%** 的股票早于 2021-05-11 起始（features/day 只有 4.9%） |
| 耗时 | **23.2 分钟**（8 进程，4.0 只/s），20 GB / 16,539,903 行 |
| PIT | 滚动窗口只用历史（截断不变性实测 maxdiff = 0.0）；唯二的例外是 `LABEL0`（标签，非因子）与首行 warmup 填 0 |

---

## 1. 复权方案：不做复权（已核实，非假设）

### 1.1 任务前提需要修正

任务书假设「features/day 里的 `$close` 是复权过的，需要先查清」。**实测结论相反**：

- qlib 官方目录 `data/qlib_official_data/features/{code}_{mkt}/` 只有
  `close/high/low/open/volume` 五个 `.day.bin`，**没有 `$factor`**。
- qlib 官方 `.bin` 的 close 与 `data/parquet/stock_data` 的 close **逐日相等**：
  - 抽样 200 只，max abs diff = **2.93e-05**（float32 存储精度）
  - 5513 只的 `.bin` 稠密值长度 == `[上市日, 退市日]` 区间内 stock_data 行数，
    抽样 200/200 全部命中
- features/day 的 `$close` 与 stock_data 的 close 的 ratio **恒为 1.0**
  → features/day 存的也是**未复权价**

### 1.2 结论

**新管线不做任何复权**，与 features/day 完全同基准。这不是妥协，而是唯一正确的选择：
若对新管线做复权而 features/day 没做，两者因子口径会立刻分叉，反而无法对比。

⚠️ 附带风险提示（不影响本任务，但影响回测解读）：**未复权价在除权日有跳空**，
所有价格类滚动因子（MA/STD/CORR/QTLD/IMAX…）在除权日附近会有伪信号。
这是平台**既有**性质（features/day 也是未复权），本管线忠实沿用，
以保证与现有回测结果可比。若将来要修，必须同时重建 features/day 与 features_full。

---

## 2. qlib 官方 .bin 的日期轴不可用（重要陷阱）

`data/qlib_official_data` 的 `.bin` 文件头里存的是 `start_index`（float32）。
本仓库这批文件**全部是 0**（5513/5513），但文件内容是**按该股票自身的交易日
序列稠密排列**的（值与 stock_data 逐位相等，maxdiff ≤ 2.9e-5）。

后果：**qlib 自己的 `D.features()` 会给出错误日期**。实测：

| instrument | qlib `D.features()` 读出的区间 | 真实区间 |
|---|---|---|
| `600000_sh` | 1990-12-19 ~ 2016-11-10 | 1999-11-10 ~ 2026-05-08 |
| `300750_sz` | 1998-05-26 ~ 2006-05-11 | 2018-06-11 ~ 2026-05-08 |
| `000002_sz` | 8637 行（对齐官方日历） | 8345 行 |

所以**不能**用 `D.features()` 取官方数据的日期。

### 本管线的处理

- **日期轴**：`data/parquet/stock_data` 的日期（平台权威，逐日与 .bin 稠密值
  对齐校验，maxdiff ≤ 1.2e-4）
- **区间**：qlib 官方 `instruments/all.txt` 的上市日 / 退市日做 PIT 截断
- **交叉校验**：每只股票都拿官方 `.bin` 的稠密 close 逐日比对
  stock_data 的 close；超出 float32 容差就**跳过该股并记录**，绝不静默出数
- qlib 官方日历（1990-12-19 ~ 2026-05-08，8637 天）仅作参考，不作日期源

---

## 3. 因子口径

复用平台自己的 `OfflineFactorPrecomputeService`，与既有重建脚本
`back_test_data_service/tdx_pipeline/rebuild_qlib_features.py` **完全同一套代码**：

1. `convert_parquet_to_qlib` + `_compute_qlib_base_indicators` → 9 列（$open..$volume, RET1/5/10/20）
2. Alpha158 **表达式引擎**（`force_expression_engine=True`）→ 158 列
3. `_compute_technical_indicators` → 16 列
4. `_compute_fundamental_features` → 9 列
5. `LABEL0 = Ref($close,-2)/Ref($close,-1) - 1` → 1 列

合计 **188 列**，与 features/day 的列集合**完全一致**（逐列差集为空）。
输出 MultiIndex `(stock_code, date)`，level-0 = `000002_SZ` 形式，
文件名 `{code}_{MKT}.parquet` → `DataLoader._load_from_precomputed` 可直接读。

### 25 个基础公式的口径

`honest_composite_strategy.prepare_backtest_data` 里的 25 个公式（MIN/MAX/STD/CORR/CORD ×
5/10/20/30/60，共 5 族 × 5 窗 = 25；策略源码里的注释写「23 个」是**陈旧注释**，
实际 `_HONEST_FORMULAS` 字典有 25 项）是由**策略在运行时从 OHLCV 现算的**，不落盘。
加上 17 个 Alpha158 面板因子 = **42 因子**（策略类名里写的「25 因子」也是陈旧命名）：

```
MIN{n}  = close.rolling(n).min() / close - 1.0
MAX{n}  = close.rolling(n).max() / close - 1.0
STD{n}  = close.rolling(n).std() / close
CORR{n} = close.rolling(n).corr(volume)
CORD{n} = ret.rolling(n).corr(vchg)      # ret/vchg = pct_change(fill_method=None)
```

注意这 25 个公式是在**横截面矩阵**（master 日期索引 × 股票）上算的，
即 `close.rolling(n)` 是按列滚动、跨越该股缺失的日期。引擎喂进来的 OHLCV
来自 parquet，所以只要 parquet 的 OHLCV 与日期轴一致，这 25 个公式就一致。

新管线输出的 `$open/$high/$low/$close/$volume` 与 features/day 逐位相等
（maxdiff = 0.0），因此这 25 个公式在两条管线上算出来的值也逐位相等
（对账中 base_formula_corr_min = 0.9999999999999998，float32 舍入级）。

---

## 4. 口径对账（`data/backtest/full_history_reconcile.json`）

### 4.1 参考文件的两种结构 —— 必须分开对账

`features/day` 的 5513 个文件里：

| 结构 | 数量 | block1 | block2 |
|---|---|---|---|
| **单段** | 272 | 交易日格，价格 + 面板因子同格，全历史 | — |
| **两段拼接** | 5241 | 交易日格：有 `$close`，**无**面板因子 | 业务日格（含节假日行）：有面板因子，**无价格** |

两段拼接的成因见 `docs/backtest/FULL_STRATEGY_AUDIT_2026-10-25.md` 的 D6。
策略的去重逻辑（`pf[~pf.index.duplicated(keep="first")]`）会让同一天只剩其中一段，
**结果「同一天既有价格又有面板因子」的日期数 = 0**（实测 002864_SZ：$close 非空 593 天、
VSUMP5 非空 690 天、两者同时非空 **0** 天）。

→ **两段文件的参考面板因子与「交易日格」口径在数值上本就不可比**。

### 4.2 两段文件的 block2 是怎么算出来的（已复现）

用 qlib **原生** `D.features()` 在 `data/qlib_data` provider 上算，得到的
Alpha158 表达式结果与 block2 **逐位相等**：

```
300014_SZ, block2 vs qlib native D.features()（业务日格）
  VSUMP5   n=1278 maxdiff=0.000e+00 corr=1.00000000
  VSUMD10  n=1303 maxdiff=0.000e+00 corr=1.00000000
  QTLD5    n=1210 maxdiff=0.000e+00 corr=1.00000000
  IMAX5    n=1289 maxdiff=0.000e+00 corr=1.00000000
  VMA5     n=1210 maxdiff=0.000e+00 corr=1.00000000
  VSTD5    n=1199 maxdiff=0.000e+00 corr=1.00000000
  WVMA5    n=1247 maxdiff=0.000e+00 corr=1.00000000
  VSUMN5   n=1278 maxdiff=0.000e+00 corr=1.00000000
```

即 block2 的滚动窗口**跨过了节假日**（业务日格 vs 交易日格），
所以 VSUMP/QTLD/IMAX 这类窗口因子与交易日格口径天然不同。
这是**参考数据的既有缺陷**，不是新管线的口径错误。

### 4.3 对账结果

对账脚本 `backend/scripts/build_full_history_features.py --reconcile-only --reconcile-n 20`
按参考结构**分层抽样**（单段 10 只 + 两段 10 只），结果在
`data/backtest/full_history_reconcile.json`。

**可比口径（单段参考文件）—— 逐位相等**：

| 指标 | 数值 |
|---|---|
| 17 个面板因子相关系数 min / mean | **0.9999999999999998 / 1.0** |
| 17 个面板因子 mean abs diff max | **0.0** |
| 面板因子 std 比值 mean | **1.0** |
| 25 个基础公式相关系数 min | **0.9999999999999998** |
| 价格列 ($open/$high/$low/$close/$volume) max abs diff | **0.0** |
| 158 个 Alpha158 列相关系数 min | **0.9999999999999998** |
| NaN 位置模式是否完全一致 | **是** |

抽样 4 只单段参考股的全历史逐列比对（`000157_SZ` / `000543_SZ` / `000004_SZ` / `000007_SZ`）：

```
000543_SZ: MA5/VSUMP5/QTLD5/VSTD5/$close/RET1/ATR14 → maxdiff=0.0, NaN 模式一致
000004_SZ: 同上 → maxdiff=0.0, NaN 模式一致
```

**即新管线在可比口径上与 features/day 逐位相等，不是「高度相关」而是「完全相同」。**

**不可比口径（两段参考文件）**：面板因子相关系数在 -0.94 ~ +0.98 之间
（`VSUMN20` -0.002、`VSUMP20` -0.030 最低；`VMA5` 0.981、`VSTD5` 0.958 最高），
差异来源已在 §4.2 证明是参考数据的业务日格 + qlib 原生算子所致。
价格列与 25 个基础公式仍然完全一致（maxdiff = 0.0 / corr = 1.0）。

---

## 5. 覆盖与起始日期

见 `data/backtest/full_history_features_progress.json`（每只股票的 rows/start/end）。

- 覆盖股票数：**5513**（qlib 官方 `instruments/all.txt` 全量）
- 起始日期 = 该股上市首日，结束日期 = 退市日（未退市的到官方数据末日 2026-05-08）
- 与 features/day 的对比：features/day 里 5241 只从 2021-05-11 才开始，
  新管线把这部分补齐到各自的上市日

（具体统计见 §7 「产出统计」。）

---

## 6. PIT 正确性

### 6.1 静态检查

- 17 个面板因子的表达式全部是**后向**滚动窗口
  （`Sum/Mean/Std/Quantile/IdxMax/IdxMin` 都是 rolling window，
  `Ref(x, n)` 全部 n > 0）
- 25 个基础公式是 `rolling(n)` + `pct_change`，只用历史
- 唯一的未来引用是 `LABEL0`（`shift(-2)/shift(-1)`）—— 它是**标签**不是因子，
  与 features/day 行为完全一致（末两行必为 NaN）
- 每只股票的样本区间 = `[上市日, 退市日]`，退市后不再有数据（PIT 宇宙）
- 序列开头窗口不足 → 引擎填 0（与 features/day 完全一致，对账已验证
  NaN 位置模式逐位相同）

### 6.2 经验性验证（截断不变性）

把输入序列在 T 时刻截断，重算因子，**T 之前的每个因子值必须逐位不变**
（若某个因子用到了 T 之后的数据，截断后它必然改变）。

对 `000002_SZ`（8345 行，1991-01-29 起）在三个截断点实测：

```
cut at 2008-07-10 (4171 行) → 之前 4170 行，17 个面板因子 maxdiff = 0.000e+00
cut at 2017-09-25 (6256 行) → 之前 6255 行，17 个面板因子 maxdiff = 0.000e+00
cut at 2022-11-22 (7507 行) → 之前 7506 行，17 个面板因子 maxdiff = 0.000e+00
```

→ **无未来函数**。

---

## 7. 产出统计

**构建结果**：`ok=5507 fail=0`，耗时 **23.2 分钟**（8 进程，4.0 只/s）。
加上前面 6 只冒烟样本，共 **5513 / 5513** 只 qlib 官方 instruments 全部成功。

| 指标 | features_full（新） | features/day（现有） |
|---|---|---|
| 股票数 | 5513 | 5513 |
| 起始日期 min | **1990-12-19** | 1991-01-14 |
| 起始日期 median | **2016-07-12** | 2021-05-11 |
| 起始日期 mean | **2013-06-17** | — |
| 早于 2021-05-11 起始 | **4330 只 (78.5%)** | 272 只 (4.9%) |
| 早于 2000-01-01 起始 | **756 只** | — |
| 早于 2010-01-01 起始 | **1529 只** | — |
| 结束日期 median | 2026-05-08 | 2026-05-08 / 2026-10-16 |
| 每只行数 min / median / mean / max | 2 / 2286 / 3000 / 8504 | — |
| 总行数 | **16,539,903** | — |
| 磁盘 | 20 GB | — |

**起始日期确实远早于 2021-05-11**（median 提前近 5 年，78.5% 的股票有 2021 年前的数据）。

边界情况已核对（都是正确的 PIT 行为）：

- `rows=2`：`920200_BJ`，2026-05-07 上市（qlib 官方 all.txt 上市日 = 2026-05-07，
  .bin 里正好 2 个有效值）→ 新股
- `end=2026-04-13`：`000638_SZ`，2026-04-13 退市 → 退市股在退市日之后不再有数据
- `end=2026-04-27`：`000004_SZ` 退市；`600193_SH` 退市

**全量一致性验证**（5513/5513 全过）：

| 检查 | 不合格数 |
|---|---|
| qlib 官方 .bin 有效值个数 == 我们的行数 | **0** |
| 结束日期 <= qlib 官方 all.txt 退市日 | **0** |
| 起始日期 >= qlib 官方 all.txt 上市日 | **0** |

→ 新管线的样本区间与 qlib 官方数据**逐只精确对齐**。

另抽样 40 个文件做结构审计：188 列、17 个面板因子齐全、5 个 OHLCV 齐全、
无 `$close` NaN、MultiIndex 层级正确 → **0 问题**。

---

## 8. 运行方式

```bash
cd /Users/rone/Projects/stock-platform

# 冒烟：先跑少量验证
env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/build_full_history_features.py --limit 5

# 全量（断点续跑，8 进程）
env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/build_full_history_features.py --resume --workers 8

# 只做口径对账
env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/build_full_history_features.py --reconcile-only \
    --reconcile-n 12
```

参数：`--limit N` / `--resume` / `--workers N` / `--stocks CODE...` /
`--out-dir DIR` / `--reconcile-only` / `--reconcile-n N` /
`--reconcile-start` / `--reconcile-end` / `--out JSON`

### 实现要点

- **多进程用 `fork`**，进度通过**分片 JSON 文件**回传（不用 multiprocessing.Queue，
  避免大批量结果把父子进程卡死）
- **启动期固化 `sys.argv[0]` 为绝对路径**：脚本会把 cwd 切到 `backend/`，
  而 `app/services/qlib/native_env.py` 在 macOS 上用
  `os.execvpe(sys.executable, [sys.executable, *sys.argv])` **重新执行本脚本**
  （为让 dyld 启动时就带上 Homebrew libomp）。若 `argv[0]` 是相对路径，
  换 cwd 后子进程会静默死掉（`can't open file .../backend/backend/scripts/...`），
  父进程却收到 0 结果。脚本在 import app.* 之前先做 libomp re-exec，
  之后 fork 出的子进程继承 `DYLD_LIBRARY_PATH` 会直接 early-return，不会递归 fork。

---

## 9. 未做的事（受约束）

- 未修改 `honest_composite_strategy.py`、回测引擎、纸面盘脚本
- 未改动 `data/qlib_data/features/day/`（增量管道产物）
- 未做 git commit
- 未把 `features_full` 接入回测（接线方式需另行确认，见下）

## 10. 接入回测的下一步（建议，未执行）

`honest_composite_strategy` 的 `panel_dir` 默认是 `../data/qlib_data/features/day`，
回测引擎的 `DataLoader.qlib_data_path` 也是 `QLIB_DATA_PATH/features/day`。
要跑 10 年 PIT 回测，需要把这两个入口指向 `features_full`（例如加一个
`QLIB_FULL_HISTORY` 配置项或做目录软链），并注意：

1. `features_full` 的价格未复权（与 features/day 一致）→ 现有回测可比
2. `features_full` 的面板因子在**交易日格**上算，覆盖率 100%；
   features/day 的两段文件覆盖率只有 ~53% 且与价格日期错位 →
   **换过去会改变 42f composite 的实际取值**，这不是 bug 修复而是口径变更，
   必须在回测报告里显式声明
3. 旧的 2021+ 回测结果（`honest_composite` 三年 +33.1% / 夏普 0.718）
   是基于 features/day 算的，**不能**与 features_full 的结果直接对比
