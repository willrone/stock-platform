# 回测引擎可成交性下沉（P0-1 / P0-2 / P0-3）—— 2026-10-26

## 背景
子代理对照 qlib / RQAlpha / LEAN / nautilus / zipline / backtrader 后的结论：

> **G1 是当前最大的回测-实盘口径裂缝**。纸面盘（`paper_portfolio.py`）有 5 项
> 可成交性检查（ST/退市、停牌、涨停不可买、跌停不可卖、印花税），
> **回测引擎一项都没有**。回测能买涨停、卖跌停 → 回测夏普虚高，
> 纸面盘跑出来的收益永远达不到回测预期。

子代理给的 A/B 预期：同一 42f 策略，回测加/不加可成交性检查，
10 年夏普差值应落在 **0.05~0.15**（下降的部分就是以前虚高的部分）。

## 做了什么

### P0-1 可成交性检查下沉到回测引擎
- 新增 `backend/app/services/backtest/execution/tradability.py`
  - `build_tradability(stock_data, trading_dates, codes)` → 可成交性矩阵
  - 矩阵维度 (n_dates × n_stocks)：`can_buy` / `can_sell` / `max_buy_amount` / 原因
- `PortfolioManagerArray` 增加 `tradability` 属性，`execute_signal` 入口查表拦截
- `BacktestExecutor` 在交易日历算出来后构建矩阵并注入
- `BacktestConfig.enable_tradability`（默认 True）可关闭回到旧口径
- **口径与纸面盘/对外信号共用** `app/services/trading/execution_rules.py`

### P0-2 涨跌停阈值按板块动态 + ST 5%
- 新增 `backend/scripts/fetch_st_names.py`：用 baostock `query_stock_basic`
  抓全市场股票名称 → `data/stocks/st_names.json`
  （实测 9022 只，其中 **375 只 ST**、**94 只退市整理**）
- `execution_rules.load_st_flags()` 提供 ST 标记数组
- 回测策略层（`honest_composite_strategy`）与纸面盘（`paper_portfolio`）的
  涨停掩码都改成 `limit_up_mask_matrix(day_ret, codes, is_st=...)`
- 原来纸面盘硬编码 `LIMIT_UP = 0.095` 全板通用：
  - 20% 板涨 9.5%~20%（未涨停）被误剔 → 砍 alpha
  - ST 股 5% 涨停拦不住 → 买入即亏损
  - 纸面盘里那份重复的 `limit_pct()` 是死代码，已删除

### P0-3 跌停无法出场 → 挂起而不是丢弃
- `PortfolioManagerArray._pending_sells`：跌停/停牌时卖单**挂着等解封**
- `flush_pending_sells(current_date, current_prices)`：每个交易日先尝试执行挂起卖单
- `BacktestConfig.enable_pending_sells`（默认 True）
- 理由：实盘里连续一字跌停的票，策略想跑却跑不掉。
  回测若丢弃 → 低估「跑不掉」的真实风险；若直接成交 → 完全虚假。

### 流动性约束（子代理 P1-4）
- 买入金额 ≤ 当日成交额 × `participation_rate`（默认 0.5%）
- 超出部分**按上限截断数量**，不是记成买满额
- 当日成交额 < `min_turnover`（默认 500 万）→ 不进信号

## 踩到的坑（都已修）

### 坑 1：qlib 的 `$volume` 单位是「手」，不是「股」
`qlib_volume / baostock_volume = 0.0100`（恒定）。
不乘 100 会让成交额少算 100 倍 → **77% 的股票被误判「流动性不足」**：
冒烟测试从 240 笔掉到 28 笔。修正后 234 笔。

### 坑 2：`_tradability` 变量残留
矩阵构建从 `PortfolioManagerArray` 创建处移到交易日历之后（那里才有
`trading_dates`），但创建处的引用没删干净 → `NameError` → 4 组 A/B 全废。

### 坑 3：execute_code 的 `str.replace` 静默不匹配
两次「参数加好」的提示是假的 —— 目标串没匹配上，代码没写进去，
但打印了成功。**纪律：核心改动后必须跑一次冒烟验证再上完整 A/B。**
为此新增 `backend/scripts/smoke_tradability.py`：
短窗口 + 开关开/关，断言「笔数必须不同」，否则判定检查没生效。

## 冒烟测试结果（800 只，2025-01-01 → 2025-06-30）

| | 夏普 | 收益 | 回撤 | 笔数 |
|---|---|---|---|---|
| 关可成交性 | 1.693 | +10.31% | −6.83% | 240 |
| 开可成交性 | 1.560 | +9.41% | −6.83% | 234 |

夏普下降 0.13 —— 落在子代理预期的 0.05~0.15 区间内。
