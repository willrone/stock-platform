# 方案 B：纸面盘日频净值曲线（待批准，尚未实现）

> 状态：**待你拍板**。本文只写方案与风险，不含任何代码改动。
> 依据：2026-11-03 只读审计（`backend/scripts/paper_portfolio.py` 字段语义清单 + 并发写审计）。

## 一、要解决的问题

现在纸面盘只有**周频**净值（`nav_history`，每周一个点）和一份**日频** `*_nav_daily.json`
（由 `backend/scripts/paper_daily_nav.py` 产出，但多资产线被显式排除）。

缺的是：**任意历史日**的持仓快照 + 连续日频净值曲线。缺了它就没法回答：
"如果那天收盘我就调仓会怎样""这条线和买入持有差多少""回撤曲线长什么样"。

## 二、动手前必须先解决的三个坑（审计已证实）

### 坑 1：ledger 表头与数据行列数不一致（结构性 bug）

`append_ledger` 用**首次写入那批 rows 的首行键序**决定表头，且**只在文件不存在时写表头**。
- 首行是 BUY → 13 列
- 首行是 SELL → 15 列（多 `cost_amount` / `realized_pnl`）

实测 8 个 ledger 里 6 个表头是 13 列，但 2026-11-02 的行是 15 列。
`csv.DictReader` 按位置错位读 → `cash_after` 读到的是 `cost_amount`，
`portfolio_value` 读到的是 `realized_pnl`。

**已被污染的既有代码**：`paper_portfolio.py:1291-1310` 的 `_ledger_cash_timeline`
实测对 22f 返回 `2026-11-02: 31360.0`，真值 `78942.93`。

→ **重建必须按行长度分派解析（13 列 / 15 列），禁用 DictReader**；
顺带修掉 `_ledger_cash_timeline` 这个既有 bug。

### 坑 2：ledger 的 `portfolio_value` 不能当历史净值点的权威值

2026-11-02 那批行的 `portfolio_value` 用的是 `adj_ratio = 1.0`（未乘后复权比值）：

| 线 | ledger.pv | holdings.last_value | 差额 |
|---|---|---|---|
| 22f | 1,001,156.93 | 978,982.93 | **+22,174.00** |
| rsi_71421 | 1,013,439.91 | 984,151.90 | +29,288.01 |
| honest_42f_neu | 998,438.49 | 936,671.49 | +61,767.00 |

→ 净值点一律用 `total_return_portfolio_value(positions, cash, adj_close, as_of)` 重算，
**不读 ledger 的 portfolio_value**。

### 坑 3：多资产线合并文件里的 leg=a 段是过期快照

`multi_asset_final_holdings.json` 的 A 腿是从 `multi_asset_final_a_holdings.json`
**拷贝**的副本；A 腿后来被 `fix_adj_cost_backfill.py` 修好了，合并文件没跟着修。
实测 **60/60 只**的 `adj_cost` 是 `NaN`、`buy_date` 是信号日（2026-10-30）而非成交日。

→ 重建多资产线**必须从 `multi_asset_final_a_holdings.json` 取 A 腿**，不信合并文件。

## 三、实现分四步

### 第 1 步：解析层（新建 `backend/scripts/ledger_replay.py`）

- `read_ledger(path)`：按行长度分派，返回统一的 dict 列表
- `replay_positions(strategy, as_of)`：逐笔累加重建任意日持仓
- 顺带修 `_ledger_cash_timeline`

**语义选择（必须二选一并写进台账）**：
- `run_line` 的实现是**全清仓 + 从 positions 删除**（`paper_portfolio.py:996-1021`）
- `paper_daily_nav.py` 用的是**移动平均成本**（部分卖时每股 cost 不变）

现有数据全是全清仓，两种实现在现有数据上**结果完全相同**（实测 7 条线逐位 0 差异），
所以这个选择目前**无法被数据证伪**。→ 选 `run_line` 语义（与生产代码一致）。

### 第 2 步：估值层

- 单线：`total_return_portfolio_value`（`paper_portfolio.py:423-497`）
- 袖套：`sleeve_positions_value`（`multi_asset_service.py:684-702`）—— **不能套 A 腿逻辑**

### 第 3 步：产出层

扩展 `paper_daily_nav.py`：
- 修 DictReader → 按位置解析
- 加 `--as-of` / `--range` 输出任意区间
- 输出 `data/paper/<strategy>_nav_daily.json`（已有格式，不改 schema）

### 第 4 步：多资产线（口径拼接，最难）

A 腿用**后复权比值**，袖套用**前复权价直接乘股数**（前复权序列会随分红重锚，做比值会系统性错算）。
两套口径**拼不到一条有经济意义的曲线上**。

可选：
- **B1**：多资产线**不做**日频（保持现状排除），只做 6 条单线
- **B2**：把袖套也改成后复权比值口径 → 需要袖套历史后复权数据（当前只有前复权 bars）
- **B3**：袖套按现金/份额记账（用 `shares × 前复权价`，并接受口径差异，**在图上分段标注**）

**我的推荐：B1**。理由：B2 需要引入袖套后复权数据源（工作量大、且 §12.22 C1 记着袖套历史价有数据源差异）；
B3 产出的曲线在袖套调仓区间内不可解释，容易被误读成策略表现。

## 四、验证锚点（不达标就不算做完）

| 项 | 阈值 | 当前实测参考 |
|---|---|---|
| 重放 positions（shares/cost/buy_date）vs holdings 文件 | **逐位 0 差异** | 7 条线实测 0 差异 ✓ |
| 重放 cash vs ledger `cash_after` | 残差 **≤ 0.03 元** | 6 条线实测 ≤0.03 ✓ |
| 末点净值 vs `nav_history` 末点 | 差 **≤ 0.05 元** | 实测 0.00~0.02 ✓ |
| 日频曲线的行情覆盖率 | 无空洞（缺数日必须显式标注，不能静默跳过） | 待测 |
| 已有门禁（`test_paper_daily_nav.py` 等） | 全绿 | — |

## 五、风险与已知限制

1. **停牌/新股**：某日无行情 → 必须显式标注该日缺数，不能静默用前值填充（否则曲线会骗人）
2. **分红除权**：后复权序列是递推出来的，历史值会随数据更新漂移 → 净值曲线**不是可复现的历史事实**，是"按当前后复权序列回算的结果"。这一点必须写在输出文件的元数据里。
3. **袖套 21 个交易日调一次仓**：调仓间隔内持仓不变、只变价格 —— 日频盯市逻辑要区分这两种变化
4. **多资产线 A 腿 50 万 + 袖套 50 万**：合并账本的 `cash_after`/`portfolio_value` 是两腿之和、全行同值，不能当逐笔语义用

## 六、工作量估计

| 步骤 | 内容 | 估计 |
|---|---|---|
| 1 | 解析层 + 修 `_ledger_cash_timeline` | 小 |
| 2 | 估值层接线 | 小 |
| 3 | 日频曲线 + 覆盖标注 | 中 |
| 4 | 多资产线 | **不做（B1）** |

## 七、要你拍板的

1. **B1 / B2 / B3** 选哪个（我推荐 **B1**）
2. **"全清仓"vs"移动平均"** 语义选哪个（我推荐跟 `run_line` 的**全清仓**，与生产代码一致）
3. 日频曲线要不要**进纸面盘 API**（`/api/v1/trading/paper/*`），还是先只落文件

**拍板后我再动手。** 在此之前不改任何行为。
