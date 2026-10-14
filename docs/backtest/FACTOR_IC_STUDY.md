# 因子 zoo 横截面 IC 普查与复合策略 (信号侧研究弧)

日期: 2026-10-13/14 · 分支 feat/vwap-below-optin · 脚本 `backend/scripts/factor_ic_study_5y.py`,
`ic_composite_panel5y{,_cells}.py`, `ic_composite_10y_v2.py`, `ic_composite_10y_honest{,_cells}.py`
· 结果 `data/strategy_baseline/{factor_ic_study_5y,ic_composite_panel5y,ic_composite_panel5y_cells,
ic_composite_10y_v2,ic_composite_10y_cells,ic_composite_10y_honest,ic_composite_10y_honest_cells}.json`

## 背景与决策

组合层 4 类杠杆 (择时/包装/条件配权/扩腿) 10 年 DSR 全灭后 (见 EXPANDED_LEGS_10Y.md),
转向从未系统勘探的**信号层**: 对引擎内 qlib Alpha158 面板 (188 列) 做全市场因子普查,
再把幸存者复合成 TopK/Dropout 策略。

## 数据事实 (2026-10-13 实测)

- 面板每股 parquet 纵向拼两块: block1 = $OHLCV+传统指标 (仅交易日, 1210 天);
  block2 = Alpha158 特征 (含节假日行)。窗口 2021-05-11 → 2026-05-08。
- LABEL0 与 close[-2]/close[-1]-1 相关 0.9995 → qlib 前向标签确认。
- 前向收益必须在 block1 交易日历上 shift, 否则跨节假日错位。

## 门 1: IC 普查 (factor_ic_study_5y.py)

~156 特征 × 5513 只 × 1210 天, fwd1/5/20 RankIC。
三道门: G1 fwd1 BH-FDR q<0.05 · G2 两子期同号 (split 2023-05-09) · G3 三 horizon 同号。

**结果: 49/156 三门全过**, 整族幸存 (非散点噪声):
STD×5(全负) · CORD×5 · CORR×5 · MAX×5 · MIN×5 · VSUMP/N/D 各×3 · QTLD×3 · KLEN/KUP/KLOW/LOW0/HIGH0/IMAX/IMIN/WVMA/VMA/VSTD/RSQR 单个。

## 门 2: 复合 → 回测 (三轮迭代, 两次证伪)

### v1 (废弃): OHLCV 重算因子定义错位
MIN20 面板定义 = min/close-1 (rc -0.94 vs 常见 close/min-1), CORR20 = corr(close, volume) 而非
corr(ret, mkt) → 方向错位, 10y Sharpe -0.53 是**伪证**。教训: 复合前必须公式匹配验证
(rank-corr ≥ 0.85 且方向保持, 31/49 幸存者可从 OHLCV 忠实重算)。

### v2 (头条证伪): 涨停锁死伪迹
公式匹配后 10y 复合 Sharpe 1.75 (OOS 2016-21 也有 1.36) — 远超历史所有腿, 触发"异常好=先怀疑"。
逐因子分解: HIGH0 top50 年化 +161.7%, KLEN +118.2%, KUP +67.2%, RSQR5 +47.3%。
**排除当日涨停 (≥9.5%) 后**: HIGH0 **-16.1%**, KLEN +2.5%, LOW0 -28.9%, MAX 族 -30~-59%。
内盘 OHLC 因子的 top50 全是当日涨停锁死票 — 收盘价根本买不进 (无卖盘), 引擎未建模锁死 → 伪收益。
lag1 执行也救不回 (信号仍由当日 high/low 构造)。

### 诚实版 (当前结论): ic_composite_10y_honest.py
三条纪律: ①剔除全部内盘因子 (KLEN/KUP/KLOW/LOW0/HIGH0/RSQR5) ②涨停掩码 (当日涨幅≥9.5% 分数置 NaN)
③保留 close/volume 族 23 公式, 研究 IC 符号, 截面 zscore 等权, TopK50/Drop5, 真实成本。

| 变体 | Sharpe | 总收益 | mdd | OOS 16-21 | 窗内 21-26 |
|---|---|---|---|---|---|
| lag0 | **0.629** | +105% | -38.6% | 0.575 | 0.679 |
| lag1 | 0.620 | +105% | -41.9% | 0.545 | 0.687 |
| topk40 | 0.566 | +88% | -38.5% | 0.622 | 0.525 |
| topk60 | **0.680** ✓ | +117% | -37.8% | 0.607 | 0.747 |
| ndrop3 | 0.663 | +110% | -37.0% | 0.570 | 0.746 |
| ndrop7 | 0.607 | +101% | -39.1% | 0.523 | 0.682 |

注: topk40/topk60 已改用等权 `max_position_size=(1-cash_reserve)/topk`, 与 topk50 同口径 95% 可投现金 (详见下节).

年度 (lag0): 2016 +18.1, 2017 -3.4, 2018 -30.9, 2019 +31.4, 2020 +20.4, 2021 +18.3,
2022 -1.8, 2023 +1.0, 2024 +18.5, 2025 +12.0, 2026 +2.2。
面板原生路径 (ic_composite_panel5y, 49 特征直接取面板值) 5y 窗 Sharpe 0.632, 与诚实版窗内 0.679
交叉验证一致。

**判定: 过三门 (topk60 邻域门已解).** topk60 邻域原先崩溃 (0.146) 是 **配置×topk 交互 bug, 非信号衰减** — 详见下节。修复后五格同向 (topk40 0.566 / topk50 0.629 / topk60 0.680 / ndrop3 0.663 / ndrop7 0.607, 全 ≥0.5), 邻域门过。诚实结论不变:
信号层存在真实但温和的横截面信息 (OOS 0.575 一致), 幅度约为等权基线 (0.435) 的 1.4×, 远小于头条 1.75。
0.63 量级在本弧 ~40 次尝试的多重检验下 DSR 大概率仍不过线, 但邻域一致性已坐实信号真实性。

## 引擎级发现: topk60 邻域崩塌 — 配置×topk 现金耗尽（已解 2026-10-14）

**根因**: TopK/Dropout 执行器单仓上限 = 组合市值 × `max_position_size` (按组合计价, 非现金),
买单上限再取 `min(单仓上限, 可投现金=cash×0.95)`. `max_position_size` 写死 0.019 → 单仓 1.9% × topk:
- topk50: 50×1.9% = **95%** = 可投现金, 恰好饱和, 买满 50 只 ✓
- topk60: 60×1.9% = **114%** > 95% 可投现金 → 买满 ~50 只后现金不够买第 51 只 → 整手取整后数量=0 → **买单失败**
- 后果: topk60 组合**永久冻结在 ~55 只** (前 5 只建仓后只剩 442 元现金), 之后每天 n_drop 换出 5 只但买单 99.4% 失败 (9619/9675), 持仓长期 <60, 换仓空转, 收益只剩 8% (Sharpe 0.146).

**逐日证据 (10y 2404 天)**:

| cell | 有买单成功的天数 | 买单成功 | 买单失败 | 失败率 | 最后持仓日 | 持仓数 | Sharpe |
|---|---|---|---|---|---|---|---|
| topk50 mps0.019 | 2316 | 10785 | 258 | 2.3% | 2026-05-08 | 50 | **0.629** |
| topk60 mps0.019 | **2** | 56 | **9619** | **99.4%** | **2016-06-16** | 56 | **0.146** |
| topk60 mps0.0158 | 2299 | 10610 | 302 | 2.8% | 2026-05-08 | 60 | **0.680** |

topk60 冻结: 10 年里只有 **2 天**有成功买单 (建仓日 2016-06-13 买 55 只, 2016-06-16 又买 1 只), 之后每天买单全失败。

**修复**: 不改引擎, 改 cell 配置 — `max_position_size = (1 - cash_reserve) / topk` 等权, 保证 topk×单仓 = 可投现金:
topk40 → 0.95/40=0.02375, topk50 → 0.019 (不变), topk60 → 0.95/60=0.015833. 重跑 4 cell:

| cell | Sharpe | 总收益 | mdd | OOS 16-21 | 窗内 21-26 | n_trades |
|---|---|---|---|---|---|---|
| topk40 | 0.566 | +88% | -38.5% | 0.622 | 0.525 | 21720 |
| topk50 (基准) | 0.629 | +105% | -38.6% | 0.575 | 0.679 | 21520 |
| **topk60 (修复)** | **0.680** | **+117%** | -37.8% | **0.607** | **0.747** | 21160 |
| ndrop3 | 0.663 | +110% | -37.0% | 0.570 | 0.746 | 13400 |
| ndrop7 | 0.607 | +101% | -39.1% | 0.523 | 0.682 | 29024 |

**topk60 修复后 Sharpe 0.680 ≥ 0.5 且与邻域 (topk40 0.566 / ndrop3 0.663 / ndrop7 0.607) 同向** — 邻域门过.
topk 有效域: `max_position_size × topk ≤ (1 - cash_reserve_ratio)`, 即单仓上限须 ≤ 可投现金/topk.
topk50 恰在边界 (95%=95%), topk>50 必须降 max_position_size 否则现金耗尽冻结.

**新工程陷阱**: 单仓上限按组合市值 × max_position_size 算 (非现金), 且 `topk × max_position_size` 必须 ≤ 可投现金比例, 否则 TopK 组合在 topk>~50 时现金耗尽永久冻结 (买单 99%+ 失败, 持仓永久 <topk). 换 topk 邻居格必须同步调 max_position_size = (1-cash_reserve)/topk.

## 引擎级发现: factor_ranking 旧三因子 zscore NaN 传播缺陷（已修复 2026-10-14）

`factor_ranking_strategy.py` 非 vwap 分支曾用 `np.where(valid,matrix,np.nan).mean()/std()`
(ndarray 语义, **NaN 传播**), vwap_below 已在 PR#16 修为 nan-aware。
后果: 全宇宙 (4329 只) 任一票任一天停牌 → 该日整个截面分数全 NaN → 全市场停打分。
停牌密度 2021-2023 高 → 官方三因子对照组在这三年**零交易零收益** (实测 valid-score 仅 2024 起),
对照组 "5y Sharpe 0.518" 实为 ~2.3 年结果。

**修复（用户批准方案 A）**: 全因子统一 nanmean/nanstd, 与 vwap_below 对齐。
回归测试 `tests/unit/backtest/test_factor_ranking_nan_scoring.py` (3 例: 停牌票不杀截面 /
分数=手算 nan-aware zscore / 全 NaN 行仍无分数)。
验证: golden 7/7 PASS + guard PASS (golden 无 factor_ranking 用例, 预期不变);
backtest 单测 37/38 (唯一红为已知 hypothesis flaky)。

**修复后重跑的官方数字**:
- 3 年全市场 (2023-05→2026-05, 5513): +30.5% / 夏普 **0.72** / 回撤 -15.2% / 5054 笔
  (旧 +38.5% / 0.98 / -10.6% / 3326 笔作废; 隐性挑日交易消失 → 笔数 +52%, 夏普 -0.26)
- 5 年对照 (2021-05→2026-05, point-in-time 4329): 夏普 **-0.238** / -17.1% / 7800 笔
  (旧 "0.518" 作废; 2021-2023 不再是零交易而是亏损: 2021 -4.3 / 2022 -23.0 / 2023 -13.7)
- 修复后分数覆盖 2021-2026 全年份 (对照组逐年 200/200 有效票)

**榜内影响**: factor_ranking 3 年风险调整排名 第 1 → 第 8; STRATEGY_BASELINE.md 已同步。
10 年 DSR 终审结论不受影响 (0.72 < 0.98, 只会更低)。IC 复合对比基线相应更新:
诚实复合 lag0 0.629 (10y) vs 官方三因子 -0.238 (5y 同窗同宇宙), 优势从 0.11 扩大到 ~0.87。

## 其他工程教训

- `pd.concat([df1,df2],axis=1).max(axis=1)` 在同名列上触发列 join 笛卡尔展开 (80 列→2504 列),
  逐元素运算必须用 `np.maximum/np.minimum`。
- 假设"异常好=先怀疑"再次救命: 1.75 头条经两层分解 (逐因子 top50 + 涨停排除) 五小时内定位为伪迹。
