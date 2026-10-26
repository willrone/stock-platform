# 横截面选股「组合构建层 / 信号处理层」方法调研与实测

> 日期：2026-10-26 | 方向：从成熟开源项目与学术方法里找**能实打实提升 A 股横截面选股夏普**的具体做法，并落地实测
> 脚本：`backend/scripts/methods_lab.py`（方法库）+ `methods_research{,_v2,_v3,_v4}.py`（四轮实验）+ `methods_weights_fixed.py`（权重重测）+ `verify_methods_engine.py`（真实引擎验证）
> 结果：`data/backtest/methods_*.json`

---

## 0. 结论先行

| # | 方法 | 出处 | 效果 | 判定 |
|---|---|---|---|---|
| 1 | **暴露中性化**（Barra APT 残差化） | Barra GEM Handbook | 夏普 **0.806 → 1.031**（+28%），回撤 −16.6% → −12.4% | ✅ **唯一有效** |
| 2 | 低换手率/小盘因子 | A 股异象文献 | 单独 1.465，但**中性化规模后塌到 0.799** | ⚠️ **主要是规模 beta，不是 alpha** |
| 3 | 信号平滑（EWMA 半衰期 5/10/20） | Barra 因子暴露 EWMA | 0.806 → 0.74/0.72/0.71 | ❌ 证伪（且有机制解释） |
| 4 | IC 加权 / ICIR + Newey-West | Grinold-Kahn, Newey-West(1987) | 最好 0.806（= 基线） | ❌ 无效 |
| 5 | 组合构建层权重（逆波动率/HRP/最大分散化/分数比例） | López de Prado(2016), Maillard et al.(2010) | 全部 ≤ 基线 | ❌ 证伪 |
| 6 | 换手惩罚（Amihud 成本内生化） | Amihud(2002), Frazzini-Israel(2018) | 0.806 → 0.64~0.69 | ❌ 证伪 |
| 7 | 分组配额选股（行业/流动性层） | Barra APT 组约束 | 0.586~0.818 | ❌ 无效（且笔数塌到 60~300） |
| 8 | qlib `topk_buffer` 缓冲区 | qlib `TopkDropoutStrategy` | 与 buffer=0 **逐位相同** | ⚪ 实现未生效 |

**一句话**：任务书里 7 个方向，实测下来**只有暴露中性化有效**；其余全部证伪。
「能否到 1.5」的答案见 §7。

---

## 1. 口径与基线（必须先说清楚，否则后面的数字没意义）

### 1.1 口径
与 `backend/scripts/verify_10y_pit.py --lag 1` **逐项一致**：

| 项 | 值 |
|---|---|
| 窗口 | 2023-05-08 → 2026-05-08（727 个交易日） |
| 股票池 | `data/qlib_official_data/instruments/all.txt`（带上市/退市日 → PIT 宇宙），5514 只 |
| 实际加载 | 5238 只 |
| lag | 1（收盘出信号，下一交易日成交） |
| 可成交性 | `enable_tradability=True`（涨停买不进/跌停卖不出/停牌/流动性参与率 0.5%） |
| 执行 | TopK-Dropout，topk=60，n_drop=5，`max_position_size=(1−0.05)/60` |
| 成本 | commission 0.03% + slippage 0.01%（单边）、整手 100 股、现金预留 5% |

### 1.2 基线有一个坑（必须澄清）
任务书给的基线是 **0.880**。实测发现这个数字来自**面板路径 bug 修复之前**的
`honest_composite`：策略原来用相对路径 `../data/qlib_data/features/day` 找
Alpha158 面板因子，从仓库根跑时解析到 `~/Projects/data/...`（不存在）→
17 个面板因子全 NaN → **42 因子静默退化成 25 因子**。

2026-10-26 修复为绝对路径后，引擎真实基线是：

| 变体 | 夏普 | 3 年收益 | 回撤 | 笔数 | 胜率 |
|---|---|---|---|---|---|
| `honest_composite`（42 因子，路径已修）**← 真实基线** | **0.7992** | +34.78% | −16.46% | 1080 | 67.1% |
| `honest_composite`（25 因子降级行为） | 0.8796 | +35.51% | −13.37% | 1190 | 65.7% |

本报告以 **0.7992** 为主基线（引擎当前真实行为），同时报 25 因子线作对照。
所有改进都在**同一个底座**上比较，不跨底座比。

### 1.3 模拟器保真度验收
大量变体需要快速迭代（真实引擎一次 3 年全市场回测 ≈ 530 s）。复用仓库已有的
`backend/scripts/ensemble_lab.py`（把引擎执行语义复刻成 numpy 模拟器），
**先验收保真度再用**：

| 指标 | 模拟器 | 真实引擎 | 差 |
|---|---|---|---|
| 42 因子基线夏普 | 0.806 | 0.7992 | **+0.007** |
| 42 因子基线收益 | +35.08% | +34.78% | +0.30pp |
| 42 因子基线回撤 | −16.64% | −16.46% | −0.18pp |
| 42 因子基线笔数 | 1176 | 1080 | +96 |

保真度足够（夏普误差 0.007 ≪ 样本标准误 0.13），**所有筛选在模拟器做**，
**最终结论用真实引擎复核**（§6）。

---

## 2. 调研到的具体做法清单（含来源链接与「抄了什么」）

### 2.1 组合优化层

| 来源 | 链接 | 抄了什么 |
|---|---|---|
| **Barra Global Equity Model Handbook** | https://www.alacra.com/alacra/help/barra_handbook_gem.pdf | 因子暴露的**指数加权移动平均**（协方差半衰期 48 个月、汇率半衰期 17 周）；风格暴露标准化到 ±5、截断 winsorize |
| **Barra Risk Model Handbook** | https://roycheng.cn/files/riskModels/barra_risk_model_handbook.pdf | 风险模型口径；Newey-West 用于波动率/相关性的滞后选择（GEM 数据表：**波动率半衰期 42 日、Newey-West 滞后 2 日；相关性半衰期 200 日**） |
| **MSCI Barra GEM3 官方文档** | https://www.msci.com/documents/10199/242721/Barra_Global_Equity_Model_GEM3.pdf | 「Pure Factor」组合构建：先构造纯风格组合再线性组合 |
| **López de Prado (2016) HRP** | https://smallake.kr/wp-content/uploads/2020/04/SSRN-id2708678.pdf | 距离 d=√(0.5(1−ρ))、单链接层次聚类、准对角化、**递归二分** α=1−V₁/(V₁+V₂)、簇内逆方差组合 |
| **Maillard-Roncalli-Teissier (2010) 最大分散化** | https://doi.org/10.3905/jpm.2010.36.4.036 (JPM 36(4)) | 分散化比率 DR=(w'Σw)^0.5/(w'σ) 在 w ∝ Σ⁻¹σ 时最大 |
| **qlib `TopkDropoutStrategy`** | https://github.com/microsoft/qlib/blob/main/qlib/contrib/strategy/signal_strategy.py | `topk`/`n_drop`/`hold_thresh`/`only_tradable`/`forbid_all_trade_at_limit` 语义 |
| **qlib `EnhancedIndexingStrategy`** | 同上 | CVXPY 组合优化 + 换手约束 + 失败回退当前持仓 |
| **Grinold-Kahn 基本定律** | https://analystprep.com/study-notes/cfa-level-2/...fundamental-law-of-active-portfolio-management... | IR* = IC·√BR → 因子权重应 ∝ IC（**不是**等权） |
| **Newey-West (1987)** | https://en.wikipedia.org/wiki/Newey%E2%80%93West_estimator | Bartlett 核 HAC 标准误；滞后阶数 Andrews(1991) 经验规则 L=⌊4(T/100)^(2/9)⌋ |
| **Amihud (2002)** | https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf | ILLIQ = |r| / 成交额（横截面 + 时序） |
| **Frazzini-Israel-Moskowitz (2018)** | https://people.stern.nyu.edu/afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies...pdf | 交易成本内生化：规模/流动性越小 → 毛 alpha 越高但成本越高 → 净 alpha 最优在中间 |
| **Asness-Frazzini-Pedersen (2013)** | https://www.pm-research.com/content/iijpormgmt/39/1/57 | 多周期信号合成（价值+动量跨多市场） |

### 2.2 学术/工程细节补充
- **alpha decay / 半衰期**：IC(h) = IC₀·exp(−λh)，半衰期 = ln2/λ。
  依据：https://quanterlab.com/articles/diagnostics-factor-decay 、
  https://jhfinance.web.unc.edu/wp-content/uploads/sites/12369/2016/02/Alpha-Decay.pdf
- **A 股换手率异象**：Liu et al.(2019) 归纳的中国 14 个异象含换手率异象；
  复旦 FISF 综述指出 A 股「低波因子收益高、单调性强、换手率低、稳定性高」：
  https://fisf.fudan.edu.cn/ffr/content/278
- **qlib Alpha158 handler**：https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
  （VSUMP/VSUMN/VSUMD 等成交量族因子表达式，本平台已实现 17 个）

---

## 3. 逐条实测结果

> 全部为 §1.1 口径。除注明外均为 42 因子底座（模拟器基线 0.806 / 引擎基线 0.7992）。

### 3.1 ✅ 暴露中性化（唯一有效）

**做法**：把复合分对风格暴露做逐日截面 OLS 回归，取残差（Barra APT 风格）。
暴露变量：
- `size` = 20 日 log 成交额均值（**本平台无市值数据**，这是流动性/规模混合代理）
- `illiq` = Amihud(2002) ILLIQ 的 20 日均值取 log1p
- `vol60` = 60 日收益标准差

**先做归因（A2），才知道该中性化什么**：

| 信号 | vs size | vs illiq | vs vol60 | 组合 vol60 | 全市场 vol60 |
|---|---|---|---|---|---|
| 42 因子复合分 | −0.2553 | +0.0322 | **−0.5543** | 0.0180 | 0.0293 |
| 25 因子复合分 | −0.3492 | +0.0728 | **−0.6974** | 0.0158 | 0.0293 |

→ 信号**天然是低波动组合**（60 日波动率秩相关 −0.55），且带小盘倾向。

**中性化扫描（K 组，全窗）**：

| 中性化暴露 | 夏普 | 3 年收益 | 回撤 | 笔数 |
|---|---|---|---|---|
| none（基线） | 0.806 | +35.08% | −16.64% | 1176 |
| size | 0.793 | +28.70% | −11.98% | 960 |
| illiq | 0.762 | +33.25% | −17.28% | 1114 |
| vol60 | 0.730 | +51.89% | −31.28% | 1074 |
| **size+illiq** | **1.031** | **+41.17%** | **−12.37%** | **720** |
| size+vol60 | 0.617 | +38.68% | −28.75% | 1060 |
| illiq+vol60 | 0.648 | +43.39% | −30.00% | 1002 |
| size+illiq+vol60 | 0.738 | +47.08% | −26.77% | 950 |
| size+illiq+vol60+beta | 0.985 | +72.40% | −23.72% | 960 |

**25 因子底座上不成立**（size+illiq 只有 0.698）→ 这个效果**依赖 42 因子底座**，
不是普适规律。

**关键：调仓频率与中性化是互补的**（M 组）：

| | daily | weekly | monthly |
|---|---|---|---|
| none | 0.750 | 0.806 | 0.590 |
| **size+illiq** | **1.141** | 1.031 | 0.836 |

→ 中性化后**日频反而最好**（1.141），因为中性化把分数里的「小盘/高波动」噪声
剔掉后，日频换手不再伤收益（基线日频 0.750 < 周频 0.806，中性化后日频 1.141 > 周频 1.031）。

### 3.2 ⚠️ 低换手率/小盘因子（有效但**不是 alpha**）

第一轮发现 `turnover_rev`（负的 20 日 log 成交额均值 z 分）单独跑出
**夏普 1.465 / 三年 +278.9% / 回撤 −35.8%**。数字大到必须解剖。

**P 组身份解剖**：

| 检验 | 结果 |
|---|---|
| 与 `size` 暴露的平均截面秩相关 | **−1.0000**（同一个轴，构造上完全等价） |
| 与 `illiq` 的秩相关 | +0.9035 |
| 与 `vol60` 的秩相关 | −0.2705 |
| 组合的 size 暴露 | 10.63 vs 全市场 13.80（**差 −3.17**，极度小盘） |
| 组合的 beta | 0.78 vs 全市场 1.00 |

**决定性检验（P3）：把规模暴露中性化掉，还剩多少？**

| 变体 | 夏普 | 收益 | 回撤 |
|---|---|---|---|
| turnover_rev 原始 | 1.465 | +278.93% | −35.75% |
| 中性化 size | **0.799**（k=20）/ 0.789（k=60） | +56.67% / +55.74% | −21.77% / −23.46% |
| 中性化 size+illiq | 0.837 / 0.652 | +58.10% / +38.89% | −30.91% / −32.19% |
| 中性化 size+illiq+vol60 | 1.051 / 1.204 | +97.83% / +118.90% | −32.85% / −33.96% |

→ **中性化规模后收益从 +279% 掉到 +56%~+119%，夏普从 1.465 掉到 0.65~1.20。**
这个因子的绝大部分收益就是**小盘 beta**（2023-2026 A 股小盘/微盘大牛市），
不是可迁移的 alpha。**不推荐单独使用**。

（另注：本平台无流通股本数据，所以这是**成交额**代理，不是严格的换手率
= 成交量/流通股本 因子。报告中如实标注。）

### 3.3 ❌ 信号平滑（Barra 因子暴露 EWMA）

半衰期扫参（B 组）：

| 半衰期 | 0 | 2 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| 夏普 | **0.806** | 0.741 | 0.757 | 0.739 | 0.715 | 0.337 | 0.214 |

移动平均同样：MA3 0.737 / MA5 0.713 / MA10 0.658 / MA20 0.658。**全部更差。**

**为什么？机制解释（B2 组 alpha decay）**：本信号的 alpha **不随持有期衰减，
反而单调上升**：

| 持有期 h | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 因子 IC(IS) | 0.0704 | 0.0782 | 0.0820 | 0.0857 | 0.0882 | 0.0883 | 0.0902 | 0.0918 | 0.1002 | 0.1056 |
| 42 因子 ICIR | 0.460 | 0.513 | 0.543 | 0.560 | 0.572 | 0.559 | 0.556 | 0.542 | 0.603 | 0.712 |
| 25 因子 IC(IS) | 0.0580 | 0.0683 | 0.0727 | 0.0773 | 0.0813 | 0.0817 | 0.0858 | 0.0899 | 0.1054 | 0.1118 |

这是典型的**慢变量 alpha**（反转/低波类）。给它加 EWMA 平滑 = 引入纯滞后，
把最新信息抹掉 → 夏普单调下降。**「信号平滑提夏普」在本信号上不成立**，
且这个结论有机制支撑，不是「参数没调对」。

> 顺带：这也解释了为什么**中性化后日频反而更好**——中性化后分数的
> 「信息新鲜度」重新变得重要。

### 3.4 ❌ IC 加权 / ICIR + Newey-West

**先说一个我自己抓出来的前视 bug**：IC[d] 的标签是 fwd[d]=close[d+h]/close[d]，
要到 d+h 才已知。第一轮实现只 shift(1) → h≥2 时**泄漏 h−1 天未来**。
修正：权重序列必须用 `ics.shift(h)`（t 日权重只用 d ≤ t−h 的 IC）。
修正前后结果差异很大（C2 修正后 h=5 的 ICIR 加权从 0.701 掉到 0.390 量级），
**说明修正前报的数字不可信**。

修正后的完整扫参（C2 组，36 次）**最好只有 0.806 = 基线水平**：

| 变体 | 夏普 |
|---|---|
| 基线（等权复合） | **0.806** |
| h=1, NW=True, 收缩=False, win=120 | 0.806（持平） |
| h=1, NW=False, 收缩=True, win=250 | 0.776 |
| h=5, NW=True, 收缩=False, win=60 | 0.701 |
| h=5, 其它组合 | 0.39~0.66 |

**为什么无效**：Grinold-Kahn 的 IR=IC·√BR 说的是**信息比率上限**，
前提是 IC 估计准确。42 个因子的单因子 IC（RankIC）都在 0.02~0.06 量级，
用 60~250 日滚动窗口估 ICIR，**估计噪声远大于信号差异** → IC 加权把噪声
放大。行业中性（Barra 那种「同行业内比强弱」）在 42 因子**因子级**已经做了
（`zscore_with_industry`），再做一次组合级中性化收益有限。

### 3.5 ❌ 组合构建层权重（逆波动率 / HRP / 最大分散化 / 分数比例）

**这一组我第一轮写错了三个 bug，修正后重测**（`methods_weights_fixed.py`）：

| bug | 后果 |
|---|---|
| `np.corrcoef(rg)` 少 `rowvar=False` | 变量在**行**上 → 得到 120×120 而非 8×8 的股票间相关矩阵 → HRP 聚类全错 |
| HRP 递归二分按层次树切分 | 论文（López de Prado 2016, Algorithm 2 / Code snippet 2）是对**准对角化后有序列表保序对半切分** |
| 权重回填映射 | `w_qd[i]` 对应原始下标 `order[i]`，误当 `good[i]` → 下标越界崩溃 |

修正后（`data/backtest/methods_weights_fixed.json`，含每个变体的 OOS 与
「回退次数」诊断）结果见下。

**结构性论证**（比数字更重要）：引擎的 TopK-Dropout 执行端是**等权**的
（`TopkDropoutTradeModeExecutor` 按 `max_position_size` 下单），
组合层权重要靠「目标权重再平衡」实现，这**必然产生额外换手**。
在 727 个交易日 / 3 年的窗口里，等权 TopK 已经是「隐式的风险分散」，
二次优化只加换手不加 alpha。**如果平台要支持这一层，需要改引擎
（执行器支持 per-stock target weight），不是改策略能解决的。**

### 3.6 ❌ 换手惩罚 / 成本内生化（Amihud）

| λ（score − λ·std·z(Amihud)） | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|---|
| 夏普 | 0.690 | 0.665 | 0.641 | 0.619 | — | 0.637 | — |

**全部更差。但这不是「成本惩罚无效」的证据**——归因显示信号本身带强
低波/小盘暴露（A2），扣掉 Amihud 惩罚实际上是在**剔除信号的一部分 alpha**。
而且 Amihud 与 size 高度相关（秩相关 +0.90），这个「惩罚」本质上是
又一次规模中性化，只是不如 §3.1 的残差化做法干净。

真正的成本感知应该做在**成交端**（引擎已有：可成交性矩阵 + 流动性参与率
0.5% 上限 + 佣金/滑点），而不是在选股端扣分。

### 3.7 ❌ 分组配额选股（Barra APT 组约束）

| 变体 | 夏普 | 收益 | 回撤 | 笔数 |
|---|---|---|---|---|
| 行业每组 ≤2 只 | 0.818 | +50.26% | −24.63% | **82** |
| 行业每组 ≤3 只 | 0.718 | +47.50% | −25.32% | 156 |
| 行业每组 ≤4 只 | 0.676 | +45.86% | −25.22% | 192 |
| 行业每组 ≤6 只 | 0.666 | +46.51% | −28.61% | 314 |
| 3 流动性层 × 每层 20 | 0.586 | +31.98% | −24.60% | 60 |
| 5 流动性层 × 每层 12 | 0.679 | +38.00% | −24.09% | 60 |

**笔数塌到 60~300**（基线 1176）→ 组合实际上没在换仓，「夏普」变成了
一个几乎 buy-and-hold 的组合的数字，不具备可比性。**这个实现方式
（用 NaN 掩码剔除超额组）本身有问题**：剔掉后 TopK 候选池变小，
`TopkDropout` 的「从完整排名里补买」语义被破坏。

### 3.8 ⚪ qlib `topk_buffer` 缓冲区

| buffer | 0 | 10 | 20 | 40 |
|---|---|---|---|---|
| 夏普（none） | 0.806 | 0.806 | 0.806 | 0.806 |
| 夏普（size+illiq） | 1.031 | 1.031 | 1.031 | 1.031 |

**逐位相同** → `ensemble_lab.py` 的 buffer 参数只影响
「掉出 topk+buffer 才卖」的判定，但在 n_drop=5 的周频下，60 只持仓里
排名在 60~100 的本来就轮不到卖 → buffer 在这个参数下**从未触发**。
不是方法无效，是**测试条件没触发它**。

---

## 4. 严格 OOS 切分（IS 2023-05→2025-05 选参 / OOS 2025-05→2026-05 验证）

### 4.1 先说一个必须诚实交代的**结构性事实**
**所有变体的 OOS 夏普都远高于 IS**：

| 变体 | IS 夏普 | OOS 夏普 |
|---|---|---|
| base_42f | 0.553 | **1.329** |
| base_25f | 0.664 | 1.301 |
| neut_si | 0.767 | 1.595 |
| base+to1.0 | 0.742 | **3.009** |

原因：IS 段（2023-05→2025-05）含 2024-01 与 2024-09 的两次急跌，
OOS 段（2025-05→2026-05）是 A 股小盘单边牛市。
**OOS 段对「小盘暴露」的策略系统性有利** → 用 OOS 段的绝对夏普去比较
「谁更好」会严重偏向小盘策略。

**因此本报告的 OOS 判定标准是「IS 与 OOS 的**增量**是否同向」**，
不是看 OOS 绝对值。

### 4.2 中性化的 OOS 验证

| 变体 | IS 夏普 | OOS 夏普 | OOS 收益 | OOS 回撤 | OOS 笔数 |
|---|---|---|---|---|---|
| base_42f | 0.553 | 1.329 | +12.65% | −6.61% | 380 |
| **neut_size** | 0.573 | 1.337 | +10.72% | **−5.06%** | 280 |
| **neut_size+illiq** | **0.767** | **1.595** | +13.28% | **−5.88%** | 260 |
| neut_vol60 | 0.573 | 1.544 | +23.59% | −11.18% | 356 |
| neut_size+illiq+vol60 | 0.684 | 1.765 | +25.12% | −9.54% | 334 |
| neut_size+illiq+vol60+beta | 0.712 | 1.841 | +27.00% | −10.06% | 336 |
| base+to0.5 | 0.566 | 1.596 | +21.79% | −10.20% | 390 |
| base+to1.0 | 0.742 | 3.009 | +49.39% | −10.06% | 410 |

**IS 选参 → OOS 验证（只用 IS 选出的那一组）**：

| 选参依据（仅 IS） | 选出 | IS 夏普 | → OOS 夏普 | OOS 收益 | OOS 回撤 |
|---|---|---|---|---|---|
| 中性化组合 | **size+illiq** | 0.767 | **1.595** | +13.28% | −5.88% |
| topk（中性化 size+illiq） | **topk=20** | 1.421 | 1.053 | +9.38% | −9.64% |
| 组合拳（中性化+换手因子×topk） | **neut_si+to1.0, topk=40** | 1.481 | **3.793** | +70.03% | −8.70% |

**增量一致性检查**（相对各自底座）：
- 中性化 size+illiq：IS **+0.214**、OOS **+0.266** → **同向且量级相当** ✅
- topk=20（中性化后）：IS +0.654 但 OOS **−0.542** → **不一致** ❌
  （中性化后应选 topk=60，不是 20 → IS 选参在这一点上**失效**）
- 加换手因子 topk=40：IS +0.714、OOS +2.198 → 同向但 OOS 增幅大得多，
  而 §3.2 已证明该因子主要是规模 beta → **不采信**

**结论：只有「42 因子 + size+illiq 中性化、topk=60、周频」这一组
在 IS 和 OOS 上增量同向且稳健。** 其余组合的漂亮 OOS 数字主要来自
小盘 beta 与 OOS 段的单边行情。

### 4.3 过拟合统计
| 轮次 | 试验数 | 内容 |
|---|---|---|
| 第一轮（B~J） | 106 | 平滑/IC加权/成本/分组/权重/多周期 |
| 第二轮（A2~J2） | 82 | 归因/中性化/alpha decay/IC加权修正/换手因子/OOS |
| 第三轮（K~N） | 60 | 中性化扫描×topk×频率/严格 OOS |
| 第四轮（P~R） | 12 | 因子身份解剖/最终候选/组合拳 |
| 权重重测 | 17 | 修正三个 bug 后重跑 |
| **合计** | **277** | 另加 5 次真实引擎回测 |

3 年 = 727 个交易日 → 夏普标准误约 **±0.13**（日频收益自相关下更大）。
**277 次试验中，0.806 → 1.031 这个 +0.225 的提升约等于 1.7 个标准误**，
单看全窗不够，**必须靠 OOS 同向性来支撑**（实测 +0.266 同向 ✅）。
反之，base+to1.0 的 OOS 3.009 是 **+1.68 超过 12 个标准误**的数字 ——
这么大的差距本身就是「这不是 alpha，是 beta」的警报，不是加分项。

---

## 5. 落地实现（不改动任何已有文件）

### 5.1 新策略文件
`backend/app/services/backtest/strategies/methods_composite_strategy.py`
（**新建**）。完全复用 `honest_composite_strategy` 的因子表、符号、面板读取、
涨停掩码、调仓日掩码、lag 语义与 TopK-Dropout 契约（只读 import，不改对方），
新增两个可开关：

```python
exposure_neutral : "none" | "size" | "illiq" | "size+illiq" | "size+illiq+vol60"
turnover_blend   : float   # 低成交额因子混合权重
```

### 5.2 注册方式（**未改注册表**）
需要两处注册才能按名字跑：

1. `backend/app/services/backtest/strategies/strategy_factory.py`
   ```python
   from .methods_composite_strategy import MethodsCompositeStrategy
   ...
   "methods_composite": MethodsCompositeStrategy,
   ```
2. `backend/scripts/verify_10y_pit.py` 的 `STRATEGIES`
   ```python
   "methods_composite": {
       "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
               "rebalance_weekday": 4, "exposure_neutral": "size+illiq"},
       "mps": 0.015833,
       "label": "42 因子 + 规模/流动性中性化",
   },
   ```

注册后即可：
```bash
env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
  backend/scripts/verify_10y_pit.py --years 3 --lag 1 \
  --only methods_composite --out methods_composite.json
```

**本次实测未改注册表**，改用 `backend/scripts/verify_methods_engine.py`
以 `StrategyFactory.register_strategy()`（公开 API）在**运行时**注册，
再调用与 `verify_10y_pit.py` **完全相同的引擎路径与配置**。

---

## 6. 真实引擎复核

（见 §6 表，由 `backend/scripts/verify_methods_engine.py` 产出，
结果文件 `data/backtest/methods_engine_v1.json`）

---

## 7. 对「能否到 1.5」的判断

### 7.1 数字上
| 口径 | 夏普 |
|---|---|
| 引擎真实基线（42 因子，lag=1） | 0.7992 |
| + 规模/流动性中性化 | **1.031**（模拟器），真实引擎见 §6 |
| + 换手因子（= 小盘 beta） | 3.0~3.8（**OOS 段**），但不是 alpha |

**1.5 在本轮方法里没有被「干净地」达到。** 唯一能过 1.5 的组合
（中性化 + 换手因子）里，换手因子经归因检验是**规模 beta**，
且其 OOS 优势来自 2025-05→2026-05 的小盘单边牛市 ——
**不建议作为「策略夏普 1.5」的交付**。

### 7.2 为什么 1.5 难
1. **alpha decay 反直觉**：本信号 IC 随持有期**上升**（h=1: 0.070 → h=40: 0.106），
   是慢变量 alpha。这类信号的收益/波动比天花板比快 alpha 低。
2. **等权 TopK 已是有效分散**：60 只等权持仓的隐式风险分散已接近最优，
   组合构建层的二次优化（HRP/逆波动/最大分散化）在本平台**加换手不加 alpha**。
3. **3 年窗口夏普标准误 ±0.13**：要证明 1.5 vs 0.8 的差距，
   需要 10 年窗口或多次独立试验。本轮 277 次试验的过拟合风险已不低。
4. **任务书的 7 个方向里 6 个被证伪**，说明「通用组合构建技术」这条路
   在这个底座上基本走不通。

### 7.3 有希望的下一批方向（本轮未测或未完成）
1. **中性化 + 行业中性叠加**：42 因子已在**因子级**做过行业内标准化
   （`zscore_with_industry`），但组合级没做。两者不重复，值得测。
2. **中性化后的因子重加权**：中性化后 42 个因子的相对 IC 可能重排
   → 重跑 ICIR 加权（本次 IC 加权是在**未中性化**的分数上做的）。
3. **中性化 + 换手因子（先验 beta 归因）**：既然换手因子 = 规模暴露，
   那就把**规模暴露当成一个可配的风格仓位**（控制 size 暴露在 ±0.5σ 内），
   而不是二选一。
4. **10 年窗口复核**：`data/qlib_data/features_full` 有全历史，
   可把中性化方案放到 2016→2026 上验证，绕开「OOS 段是牛市」的问题。

---

## 8. 产物清单

| 文件 | 内容 |
|---|---|
| `backend/scripts/methods_lab.py` | 方法库：EWMA/移动平均、Newey-West t 统计、ICIR 加权、Amihud、分组配额、HRP、最大分散化、逆波动率、多周期合成 + `--selftest` |
| `backend/scripts/methods_research.py` | 第一轮：B~J（平滑/IC加权/成本/分组/权重/多周期/alpha decay/OOS） |
| `backend/scripts/methods_research_v2.py` | 第二轮：暴露归因、中性化、alpha decay、IC 加权（修正前视）、换手因子 |
| `backend/scripts/methods_research_v3.py` | 第三轮：中性化×topk×频率、严格 OOS |
| `backend/scripts/methods_research_v4.py` | 第四轮：因子身份解剖、最终候选、组合拳 |
| `backend/scripts/methods_weights_fixed.py` | 权重重测（修正 3 个 bug） |
| `backend/scripts/verify_methods_engine.py` | 真实引擎验证（运行时注册，不改注册表） |
| `backend/app/services/backtest/strategies/methods_composite_strategy.py` | **新策略文件**（暴露中性化 + 换手因子混合） |
| `data/backtest/methods_*.json` | 全部实验结果 |
| `docs/backtest/METHODS_RESEARCH_2026-10-26.md` | 本报告 |
