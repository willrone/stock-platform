# 回测平台流程图 (Mermaid格式)

本文档包含所有回测流程的Mermaid流程图，可以在支持Mermaid的工具中渲染（如GitHub、VS Code、Typora等）。

---

## 1. 回测流程整体图

```mermaid
flowchart TD
    Start([用户提交回测任务]) --> CreateTask[创建任务记录<br/>状态: CREATED<br/>进度: 0%]
    
    CreateTask --> QueueTask[加入后台执行队列]
    
    QueueTask --> Stage1[阶段1: 初始化<br/>0-10%<br/>生成回测ID<br/>启动进度监控]
    
    Stage1 --> Stage2[阶段2: 数据加载<br/>10-25%<br/>加载股票历史数据<br/>验证数据完整性<br/>支持并行加载]
    
    Stage2 --> Stage3[阶段3: 策略设置<br/>25-30%<br/>创建交易策略实例<br/>初始化组合管理器<br/>配置初始资金/手续费/滑点]
    
    Stage3 --> Stage4[阶段4: 回测执行<br/>30-90%<br/>获取交易日历<br/>逐日回测循环]
    
    Stage4 --> Stage5[阶段5: 结果计算<br/>90-95%<br/>计算收益率指标<br/>计算风险指标<br/>生成交易报告]
    
    Stage5 --> Stage6[阶段6: 完成<br/>95-100%<br/>保存回测结果<br/>更新任务状态为COMPLETED<br/>推送完成通知]
    
    Stage6 --> End([回测完成])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style Stage1 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Stage2 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Stage3 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Stage4 fill:#ed8936,stroke:#333,stroke-width:2px,color:#fff
    style Stage5 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Stage6 fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
```

---

## 2. 回测执行详细流程图

```mermaid
flowchart TD
    Start([开始回测执行]) --> GetCalendar[获取交易日历<br/>合并所有股票交易日期<br/>排序并过滤日期范围]
    
    GetCalendar --> ValidateDays{交易日数 >= 20?}
    
    ValidateDays -->|否| Error1[抛出错误:<br/>数据不足]
    ValidateDays -->|是| LoopStart[主循环开始<br/>for current_date in trading_dates]
    
    LoopStart --> GetPrice[获取当前价格<br/>从所有股票数据中<br/>提取收盘价]
    
    GetPrice --> GenSignals[生成交易信号<br/>对每只股票调用<br/>strategy.generate_signals]
    
    GenSignals --> ValidateSignals[验证信号有效性<br/>检查信号强度<br/>检查持仓限制]
    
    ValidateSignals --> ExecuteSignals[执行信号<br/>买入/卖出操作<br/>更新持仓和资金]
    
    ExecuteSignals --> UpdatePortfolio[更新持仓和资金<br/>计算新的持仓价值<br/>计算可用资金]
    
    UpdatePortfolio --> RecordTrades[记录交易历史<br/>保存交易记录到列表<br/>包括时间/价格/数量]
    
    RecordTrades --> UpdateProgress[更新进度<br/>计算进度百分比<br/>通过WebSocket推送]
    
    UpdateProgress --> CheckEnd{是否还有<br/>交易日?}
    
    CheckEnd -->|是| LoopStart
    CheckEnd -->|否| CalcResults[计算回测结果<br/>收益率/风险指标<br/>交易统计]
    
    CalcResults --> End([回测执行完成])
    
    Error1 --> End
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style LoopStart fill:#ed8936,stroke:#333,stroke-width:2px,color:#fff
    style GetPrice fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style GenSignals fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style ExecuteSignals fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style Error1 fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
```

---

## 3. 策略分类架构图

```mermaid
graph TD
    Root[量化交易策略] --> Tech[技术分析策略]
    Root --> Stat[统计套利策略]
    Root --> Factor[因子投资策略]
    
    Tech --> MA[移动平均策略<br/>MovingAverage]
    Tech --> RSI[RSI策略<br/>RSI]
    Tech --> MACD[MACD策略<br/>MACD]
    Tech --> BB[布林带策略<br/>BollingerBands]
    Tech --> STOCH[随机指标策略<br/>Stochastic]
    Tech --> CCI[CCI策略<br/>CCI]
    
    Stat --> Pairs[配对交易策略<br/>PairsTrading]
    Stat --> MeanRev[均值回归策略<br/>MeanReversion]
    Stat --> Coint[协整策略<br/>Cointegration]
    
    Factor --> Value[价值因子策略<br/>ValueFactor]
    Factor --> Momentum[动量因子策略<br/>MomentumFactor]
    Factor --> LowVol[低波动因子策略<br/>LowVolatility]
    Factor --> Multi[多因子组合策略<br/>MultiFactor]
    
    style Root fill:#667eea,stroke:#333,stroke-width:3px,color:#fff
    style Tech fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Stat fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style Factor fill:#ed8936,stroke:#333,stroke-width:2px,color:#fff
    style MA fill:#e2e8f0,stroke:#333,stroke-width:1px
    style RSI fill:#e2e8f0,stroke:#333,stroke-width:1px
    style MACD fill:#e2e8f0,stroke:#333,stroke-width:1px
    style BB fill:#e2e8f0,stroke:#333,stroke-width:1px
    style STOCH fill:#e2e8f0,stroke:#333,stroke-width:1px
    style CCI fill:#e2e8f0,stroke:#333,stroke-width:1px
    style Pairs fill:#c6f6d5,stroke:#333,stroke-width:1px
    style MeanRev fill:#c6f6d5,stroke:#333,stroke-width:1px
    style Coint fill:#c6f6d5,stroke:#333,stroke-width:1px
    style Value fill:#fed7aa,stroke:#333,stroke-width:1px
    style Momentum fill:#fed7aa,stroke:#333,stroke-width:1px
    style LowVol fill:#fed7aa,stroke:#333,stroke-width:1px
    style Multi fill:#fed7aa,stroke:#333,stroke-width:1px
```

---

## 4. 移动平均策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcSMA_Short[计算短期移动平均线<br/>SMA_short = 5日均线]
    
    CalcSMA_Short --> CalcSMA_Long[计算长期移动平均线<br/>SMA_long = 20日均线]
    
    CalcSMA_Long --> CalcDiff[计算移动平均差值<br/>ma_diff = SMA_short - SMA_long / SMA_long]
    
    CalcDiff --> CheckPrev{前一日<br/>ma_diff <= 0?}
    
    CheckPrev -->|是| CheckCurr1{当前<br/>ma_diff > 0<br/>且 abs > 阈值?}
    CheckPrev -->|否| CheckPrev2{前一日<br/>ma_diff >= 0?}
    
    CheckCurr1 -->|是| BuySignal[生成买入信号<br/>金叉: 短期上穿长期<br/>信号强度 = abs ma_diff * 10]
    CheckCurr1 -->|否| NoSignal1[无信号]
    
    CheckPrev2 -->|是| CheckCurr2{当前<br/>ma_diff < 0<br/>且 abs > 阈值?}
    CheckPrev2 -->|否| NoSignal2[无信号]
    
    CheckCurr2 -->|是| SellSignal[生成卖出信号<br/>死叉: 短期下穿长期<br/>信号强度 = abs ma_diff * 10]
    CheckCurr2 -->|否| NoSignal3[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    NoSignal3 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckCurr1 fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckCurr2 fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
```

---

## 5. RSI策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcRSI[计算RSI指标<br/>周期: 14日<br/>方法: Wilder平滑]
    
    CalcRSI --> CalcTrendMA[计算趋势均线<br/>50日均线<br/>用于判断趋势]
    
    CalcTrendMA --> DetectTrend[检测趋势方向<br/>上升趋势/下降趋势/横盘]
    
    DetectTrend --> DetectDivergence[检测背离<br/>看涨背离: 价格创新低<br/>但RSI未创新低<br/>看跌背离: 价格创新高<br/>但RSI未创新高]
    
    DetectDivergence --> CheckRSI{检查RSI穿越}
    
    CheckRSI --> CheckBuy{RSI从超卖区域<br/>< 30 向上穿越?}
    CheckRSI --> CheckSell{RSI从超买区域<br/>> 70 向下穿越?}
    
    CheckBuy -->|是| CheckTrendBuy{趋势对齐?}
    CheckSell -->|是| CheckTrendSell{趋势对齐?}
    
    CheckTrendBuy -->|上升趋势| CheckRSILevel1{RSI在<br/>40-55之间?}
    CheckTrendBuy -->|非下降趋势| CheckDivergence1{看涨背离?}
    
    CheckRSILevel1 -->|是| BuySignal1[生成买入信号<br/>上升趋势回调买入<br/>强度增强]
    CheckDivergence1 -->|是| BuySignal2[生成买入信号<br/>超卖反弹 + 背离<br/>强度增强1.3倍]
    CheckDivergence1 -->|否| BuySignal3[生成买入信号<br/>超卖反弹]
    
    CheckTrendSell -->|下降趋势| CheckRSILevel2{RSI在<br/>45-60之间?}
    CheckTrendSell -->|非上升趋势| CheckDivergence2{看跌背离?}
    
    CheckRSILevel2 -->|是| SellSignal1[生成卖出信号<br/>下降趋势反弹卖出<br/>强度增强]
    CheckDivergence2 -->|是| SellSignal2[生成卖出信号<br/>超买回调 + 背离<br/>强度增强1.3倍]
    CheckDivergence2 -->|否| SellSignal3[生成卖出信号<br/>超买回调]
    
    BuySignal1 --> CheckVolume
    BuySignal2 --> CheckVolume
    BuySignal3 --> CheckVolume
    SellSignal1 --> CheckVolume
    SellSignal2 --> CheckVolume
    SellSignal3 --> CheckVolume
    
    CheckVolume{检查成交量确认<br/>可选}
    
    CheckVolume --> Output[输出交易信号]
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal1 fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal2 fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal3 fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal1 fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal2 fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal3 fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style DetectTrend fill:#9f7aea,stroke:#333,stroke-width:2px,color:#fff
    style DetectDivergence fill:#9f7aea,stroke:#333,stroke-width:2px,color:#fff
```

---

## 6. 组合策略信号整合流程图

```mermaid
flowchart TD
    Start([组合策略开始]) --> Strategy1[策略1: RSI<br/>权重: 0.4<br/>生成信号1]
    Start --> Strategy2[策略2: MACD<br/>权重: 0.3<br/>生成信号2]
    Start --> Strategy3[策略3: 布林带<br/>权重: 0.3<br/>生成信号3]
    
    Strategy1 --> Signal1[信号1:<br/>类型: 买入/卖出/无<br/>强度: 0.0-1.0]
    Strategy2 --> Signal2[信号2:<br/>类型: 买入/卖出/无<br/>强度: 0.0-1.0]
    Strategy3 --> Signal3[信号3:<br/>类型: 买入/卖出/无<br/>强度: 0.0-1.0]
    
    Signal1 --> Integrator[信号整合器]
    Signal2 --> Integrator
    Signal3 --> Integrator
    
    Integrator --> Method{选择整合方法}
    
    Method -->|加权投票| WeightedVoting[加权投票<br/>final_signal =<br/>sum signal * weight]
    Method -->|多数投票| MajorityVoting[多数投票<br/>final_signal =<br/>majority signals]
    Method -->|加权平均| WeightedAvg[加权平均<br/>final_signal =<br/>weighted_avg signals]
    
    WeightedVoting --> FinalSignal[最终交易信号<br/>类型: 买入/卖出/无<br/>综合强度: 0.0-1.0<br/>各策略贡献详情]
    MajorityVoting --> FinalSignal
    WeightedAvg --> FinalSignal
    
    FinalSignal --> End([输出最终信号])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style Strategy1 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Strategy2 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Strategy3 fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Integrator fill:#ed8936,stroke:#333,stroke-width:2px,color:#fff
    style FinalSignal fill:#48bb78,stroke:#333,stroke-width:3px,color:#fff
```

---

## 7. 数据加载流程图

```mermaid
flowchart TD
    Start([开始数据加载]) --> CheckStocks{股票数量 > 3<br/>且启用并行?}
    
    CheckStocks -->|是| ParallelLoad[并行加载数据<br/>ThreadPoolExecutor<br/>max_workers=CPU核心数]
    CheckStocks -->|否| SequentialLoad[顺序加载数据]
    
    ParallelLoad --> LoadStock1[加载股票1数据<br/>StockDataLoader<br/>从CSV文件读取]
    ParallelLoad --> LoadStock2[加载股票2数据]
    ParallelLoad --> LoadStockN[加载股票N数据]
    
    SequentialLoad --> LoadStock1
    
    LoadStock1 --> Validate1[验证数据完整性<br/>检查必需列:<br/>open, high, low, close, volume]
    LoadStock2 --> Validate2[验证数据完整性]
    LoadStockN --> ValidateN[验证数据完整性]
    
    Validate1 --> MergeData[合并所有股票数据<br/>stock_data: Dict<br/>key: stock_code<br/>value: DataFrame]
    Validate2 --> MergeData
    ValidateN --> MergeData
    
    MergeData --> CheckData{数据量 >= 20<br/>个交易日?}
    
    CheckData -->|否| Error[抛出错误:<br/>数据不足]
    CheckData -->|是| End([数据加载完成])
    
    Error --> End
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style ParallelLoad fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style Error fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
```

---

## 8. 任务创建流程图

```mermaid
flowchart TD
    Start([用户访问任务创建页面]) --> FillForm[填写任务表单<br/>任务名称<br/>任务类型: backtest]
    
    FillForm --> SelectStocks[选择股票代码列表<br/>支持多选<br/>建议1-10只股票]
    
    SelectStocks --> ConfigParams[配置回测参数<br/>开始日期/结束日期<br/>初始资金<br/>手续费率/滑点率]
    
    ConfigParams --> SelectStrategy{选择策略类型}
    
    SelectStrategy -->|单策略| SingleStrategy[选择单策略<br/>moving_average<br/>rsi, macd等]
    SelectStrategy -->|组合策略| PortfolioStrategy[配置组合策略<br/>选择多个策略<br/>设置权重<br/>选择整合方法]
    
    SingleStrategy --> ConfigStrategy[配置策略参数<br/>根据策略类型<br/>设置相应参数]
    PortfolioStrategy --> ConfigStrategy
    
    ConfigStrategy --> ValidateForm{验证表单<br/>数据有效性}
    
    ValidateForm -->|无效| ShowError[显示错误信息<br/>提示用户修正]
    ValidateForm -->|有效| SubmitTask[提交任务<br/>POST /api/v1/tasks]
    
    ShowError --> ConfigParams
    
    SubmitTask --> CreateTask[后端创建任务记录<br/>status: CREATED<br/>progress: 0%<br/>保存到数据库]
    
    CreateTask --> QueueTask[加入后台执行队列<br/>background_tasks.add_task]
    
    QueueTask --> Redirect[跳转到任务详情页<br/>路径: /tasks/task_id]
    
    Redirect --> Monitor[监控回测进度<br/>WebSocket实时推送]
    
    Monitor --> End([任务创建完成])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SubmitTask fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style CreateTask fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style ShowError fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
```

---

## 9. MACD策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcEMA_Fast[计算快速EMA<br/>周期: 12日<br/>EMA_fast = 12日指数移动平均]
    
    CalcEMA_Fast --> CalcEMA_Slow[计算慢速EMA<br/>周期: 26日<br/>EMA_slow = 26日指数移动平均]
    
    CalcEMA_Slow --> CalcMACD[计算MACD线<br/>MACD = EMA_fast - EMA_slow]
    
    CalcMACD --> CalcSignal[计算信号线<br/>周期: 9日<br/>MACD_signal = MACD的9日EMA]
    
    CalcSignal --> CalcHist[计算MACD柱状图<br/>MACD_hist = MACD - MACD_signal]
    
    CalcHist --> CheckPrev{前一日<br/>MACD_hist <= 0?}
    
    CheckPrev -->|是| CheckCurr1{当前<br/>MACD_hist > 0?}
    CheckPrev -->|否| CheckPrev2{前一日<br/>MACD_hist >= 0?}
    
    CheckCurr1 -->|是| BuySignal[生成买入信号<br/>MACD金叉: MACD上穿信号线<br/>信号强度 = abs MACD_hist * 100]
    CheckCurr1 -->|否| NoSignal1[无信号]
    
    CheckPrev2 -->|是| CheckCurr2{当前<br/>MACD_hist < 0?}
    CheckPrev2 -->|否| NoSignal2[无信号]
    
    CheckCurr2 -->|是| SellSignal[生成卖出信号<br/>MACD死叉: MACD下穿信号线<br/>信号强度 = abs MACD_hist * 100]
    CheckCurr2 -->|否| NoSignal3[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    NoSignal3 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckCurr1 fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckCurr2 fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
```

---

## 10. 布林带策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcSMA[计算移动平均线<br/>周期: 20日<br/>SMA = 20日简单移动平均]
    
    CalcSMA --> CalcStd[计算标准差<br/>周期: 20日<br/>Std = 20日价格标准差]
    
    CalcStd --> CalcUpper[计算上轨<br/>Upper = SMA + Std * 2]
    
    CalcUpper --> CalcLower[计算下轨<br/>Lower = SMA - Std * 2]
    
    CalcLower --> CalcPercentB[计算PercentB<br/>PercentB = 价格 - 下轨 / 上轨 - 下轨]
    
    CalcPercentB --> CheckPrev{前一日<br/>PercentB <= 0?}
    
    CheckPrev -->|是| CheckCurr1{当前<br/>PercentB > 0?}
    CheckPrev -->|否| CheckPrev2{前一日<br/>PercentB >= 1?}
    
    CheckCurr1 -->|是| BuySignal[生成买入信号<br/>价格突破下轨反弹<br/>信号强度 = PercentB]
    CheckCurr1 -->|否| NoSignal1[无信号]
    
    CheckPrev2 -->|是| CheckCurr2{当前<br/>PercentB < 1?}
    CheckPrev2 -->|否| NoSignal2[无信号]
    
    CheckCurr2 -->|是| SellSignal[生成卖出信号<br/>价格突破上轨回调<br/>信号强度 = 1 - PercentB]
    CheckCurr2 -->|否| NoSignal3[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    NoSignal3 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckCurr1 fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckCurr2 fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
```

---

## 11. 随机指标策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcLowest[计算最低价<br/>周期: 14日<br/>Lowest = 14日最低价]
    
    CalcLowest --> CalcHighest[计算最高价<br/>周期: 14日<br/>Highest = 14日最高价]
    
    CalcHighest --> CalcK[计算K值<br/>K = 100 * 收盘价 - 最低价 / 最高价 - 最低价]
    
    CalcK --> CalcD[计算D值<br/>周期: 3日<br/>D = K的3日移动平均]
    
    CalcD --> CheckK{当前K值<br/>和前一日的K值}
    
    CheckK --> CheckBuy{K < 20<br/>且前一日K < 20<br/>且K上穿D?}
    CheckK --> CheckSell{K > 80<br/>且前一日K > 80<br/>且K下穿D?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>随机指标超卖金叉<br/>信号强度 = 20 - K / 20]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>随机指标超买死叉<br/>信号强度 = K - 80 / 20]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckK fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckBuy fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
    style CheckSell fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
```

---

## 12. CCI策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcTypicalPrice[计算典型价格<br/>Typical = 最高价 + 最低价 + 收盘价 / 3]
    
    CalcTypicalPrice --> CalcSMA[计算移动平均<br/>周期: 20日<br/>SMA = 20日典型价格移动平均]
    
    CalcSMA --> CalcMeanDev[计算平均偏差<br/>周期: 20日<br/>MeanDev = 20日平均绝对偏差]
    
    CalcMeanDev --> CalcCCI[计算CCI指标<br/>CCI = 典型价格 - SMA / 0.015 * MeanDev]
    
    CalcCCI --> CheckPrev{前一日CCI}
    
    CheckPrev --> CheckBuy{当前CCI < -100<br/>且前一日CCI >= -100?}
    CheckPrev --> CheckSell{当前CCI > 100<br/>且前一日CCI <= 100?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>CCI超卖反弹<br/>信号强度 = abs CCI / 100]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>CCI超买回调<br/>信号强度 = CCI / 100]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 13. 配对交易策略流程图

```mermaid
flowchart TD
    Start([输入股票对数据]) --> FindPairs[寻找配对股票<br/>计算相关性<br/>阈值: 0.8]
    
    FindPairs --> CalcSpread[计算价差<br/>Spread = 股票1价格 - 股票2价格]
    
    CalcSpread --> CalcZScore[计算Z-score<br/>周期: 20日<br/>Z-score = Spread - Mean / Std]
    
    CalcZScore --> CalcMomentum[计算相对强度<br/>Momentum_5d = 5日动量<br/>Momentum_20d = 20日动量<br/>Relative = Momentum_20d - Momentum_5d]
    
    CalcMomentum --> CheckPrev{前一日Z-score}
    
    CheckPrev --> CheckBuy{前一日Z-score <= -2<br/>且当前Z-score > -2<br/>且相对强度 < -0.02?}
    CheckPrev --> CheckSell{前一日Z-score >= 2<br/>且当前Z-score < 2<br/>且相对强度 > 0.02?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>价差回归，买入弱势股<br/>信号强度 = abs Z-score / 2]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>价差偏离，卖出强势股<br/>信号强度 = abs Z-score / 2]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 14. 均值回归策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcSMA[计算移动平均<br/>周期: 20日<br/>SMA = 20日简单移动平均]
    
    CalcSMA --> CalcStd[计算标准差<br/>周期: 20日<br/>Std = 20日价格标准差]
    
    CalcStd --> CalcZScore[计算Z-score<br/>Z-score = 价格 - SMA / Std]
    
    CalcZScore --> CheckPrev{前一日Z-score}
    
    CheckPrev --> CheckBuy{前一日Z-score <= -2<br/>且当前Z-score > -2?}
    CheckPrev --> CheckSell{前一日Z-score >= 2<br/>且当前Z-score < 2?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>价格回归均值<br/>信号强度 = abs Z-score / 2]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>价格偏离均值<br/>信号强度 = abs Z-score / 2]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 15. 协整策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcReturns[计算收益率<br/>Returns = 价格变化率]
    
    CalcReturns --> EstimateHalfLife[估计半衰期<br/>使用OLS回归<br/>HalfLife = -ln2 / beta]
    
    EstimateHalfLife --> CalcSMA[计算移动平均<br/>周期: 60日<br/>SMA = 60日价格移动平均]
    
    CalcSMA --> CalcStd[计算标准差<br/>周期: 60日<br/>Std = 60日价格标准差]
    
    CalcStd --> CalcZScore[计算Z-score<br/>Z-score = 价格 - SMA / Std]
    
    CalcZScore --> CalcMeanRev[计算均值回归强度<br/>MeanRev = -ln2 / HalfLife]
    
    CalcMeanRev --> CheckPrev{前一日Z-score}
    
    CheckPrev --> CheckBuy{前一日Z-score <= -2<br/>且当前Z-score > -2<br/>且均值回归强度 < 0?}
    CheckPrev --> CheckSell{前一日Z-score >= 2<br/>且当前Z-score < 2<br/>且均值回归强度 < 0?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>协整信号：价格回归均衡<br/>信号强度 = abs Z-score / 2]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>协整信号：价格偏离均衡<br/>信号强度 = abs Z-score / 2]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 16. 价值因子策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcReturns[计算收益率<br/>Returns = 价格变化率]
    
    CalcReturns --> CalcVolatility[计算波动率<br/>周期: 21日<br/>Volatility = 21日收益率标准差]
    
    CalcVolatility --> EstimatePE[估计PE比率<br/>PE = 1 / 年化收益率 + 0.001]
    
    EstimatePE --> EstimatePB[估计PB比率<br/>PB = 1 / 波动率 * 0.5 + 1]
    
    EstimatePB --> EstimatePS[估计PS比率<br/>PS = 1 / 波动率 * 0.3 + 1.5]
    
    EstimatePS --> EstimateEV[估计EV/EBITDA<br/>EV = 1 / 波动率 * 0.4 + 5]
    
    EstimateEV --> NormalizeFactors[标准化因子<br/>对PE/PB/PS/EV进行<br/>252日滚动标准化]
    
    NormalizeFactors --> CalcValueScore[计算价值评分<br/>ValueScore = -PE*0.25 - PB*0.25<br/>- PS*0.25 - EV*0.25]
    
    CalcValueScore --> CheckPrev{前一日价值评分}
    
    CheckPrev --> CheckBuy{前一日ValueScore <= 0<br/>且当前ValueScore > 0?}
    CheckPrev --> CheckSell{前一日ValueScore >= 0<br/>且当前ValueScore < 0?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>价值因子评分转正<br/>信号强度 = ValueScore]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>价值因子评分转负<br/>信号强度 = abs ValueScore]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 17. 动量因子策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcMomentum1M[计算1月动量<br/>周期: 21日<br/>Momentum_1M = 价格 / 21日前价格 - 1]
    
    CalcMomentum1M --> CalcMomentum3M[计算3月动量<br/>周期: 63日<br/>Momentum_3M = 价格 / 63日前价格 - 1]
    
    CalcMomentum3M --> CalcMomentum6M[计算6月动量<br/>周期: 126日<br/>Momentum_6M = 价格 / 126日前价格 - 1]
    
    CalcMomentum6M --> NormalizeMomentum[标准化动量<br/>对每个动量进行<br/>滚动标准化]
    
    NormalizeMomentum --> CalcCombined[计算综合动量<br/>Momentum = Momentum_1M*0.5<br/>+ Momentum_3M*0.3<br/>+ Momentum_6M*0.2]
    
    CalcCombined --> CheckPrev{前一日动量}
    
    CheckPrev --> CheckBuy{前一日Momentum <= 0<br/>且当前Momentum > 0?}
    CheckPrev --> CheckSell{前一日Momentum >= 0<br/>且当前Momentum < 0?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>动量转正<br/>信号强度 = abs Momentum]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>动量转负<br/>信号强度 = abs Momentum]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 18. 低波动因子策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcReturns[计算日收益率<br/>Returns = 价格变化率]
    
    CalcReturns --> CalcVolatility[计算历史波动率<br/>周期: 63日<br/>Volatility = 63日收益率标准差 * sqrt252]
    
    CalcVolatility --> CalcRiskAdjReturn[计算风险调整收益<br/>周期: 21日<br/>RAR = 21日平均收益 / 21日标准差]
    
    CalcRiskAdjReturn --> CheckRAR{风险调整收益<br/>RAR > 0?}
    
    CheckRAR -->|是| BuySignal[生成买入信号<br/>低波动高收益<br/>信号强度 = RAR / 5]
    CheckRAR -->|否| NoSignal[无信号]
    
    BuySignal --> Output[输出交易信号]
    NoSignal --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style CheckRAR fill:#ea580c,stroke:#333,stroke-width:2px,color:#fff
```

---

## 19. 多因子组合策略流程图

```mermaid
flowchart TD
    Start([输入历史价格数据]) --> CalcValue[计算价值因子<br/>ValueScore = -价格相对均线偏离度]
    
    CalcValue --> CalcMomentum[计算动量因子<br/>MomentumScore = 多周期动量标准化]
    
    CalcMomentum --> CalcLowVol[计算低波动因子<br/>LowVolScore = -波动率标准化]
    
    CalcLowVol --> NormalizeFactors[标准化各因子<br/>对每个因子进行滚动标准化]
    
    NormalizeFactors --> WeightFactors[加权组合因子<br/>根据配置的权重<br/>CombinedScore = Value*w1<br/>+ Momentum*w2<br/>+ LowVol*w3]
    
    WeightFactors --> SmoothScore[平滑综合评分<br/>周期: 5日<br/>CombinedScore = 5日移动平均]
    
    SmoothScore --> CheckPrev{前一日综合评分}
    
    CheckPrev --> CheckBuy{前一日CombinedScore <= 0<br/>且当前CombinedScore > 0?}
    CheckPrev --> CheckSell{前一日CombinedScore >= 0<br/>且当前CombinedScore < 0?}
    
    CheckBuy -->|是| BuySignal[生成买入信号<br/>多因子综合评分转正<br/>信号强度 = CombinedScore]
    CheckBuy -->|否| NoSignal1[无信号]
    
    CheckSell -->|是| SellSignal[生成卖出信号<br/>多因子综合评分转负<br/>信号强度 = abs CombinedScore]
    CheckSell -->|否| NoSignal2[无信号]
    
    BuySignal --> Output[输出交易信号]
    SellSignal --> Output
    NoSignal1 --> Output
    NoSignal2 --> Output
    
    Output --> End([结束])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style BuySignal fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SellSignal fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
    style CheckPrev fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckBuy fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
    style CheckSell fill:#ea580c,color:#fff,stroke:#333,stroke-width:2px
```

---

## 使用说明

### 如何查看这些流程图

1. **GitHub**: 直接在GitHub上查看，自动渲染Mermaid图表
2. **VS Code**: 安装 "Markdown Preview Mermaid Support" 插件
3. **Typora**: 原生支持Mermaid，直接打开即可查看
4. **在线工具**: 
   - https://mermaid.live/ - 在线Mermaid编辑器
   - https://mermaid-js.github.io/mermaid-live-editor/ - 官方在线编辑器

### 如何导出为图片

1. 使用在线工具 (mermaid.live):
   - 复制Mermaid代码
   - 粘贴到在线编辑器
   - 点击"Actions" → "Download PNG/SVG"

2. 使用VS Code:
   - 安装 "Markdown Preview Mermaid Support" 插件
   - 右键点击图表 → "导出为图片"

3. 使用命令行工具:
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   mmdc -i flowchart.mmd -o flowchart.png
   ```

---

## 文件说明

- **回测平台PPT_Marp.md**: Marp格式的完整PPT，可直接生成演示文稿
- **回测平台PPT_简洁版.md**: 简洁版PPT，22页幻灯片
- **回测平台流程图_Mermaid.md**: 所有流程图的Mermaid源码
- **回测平台PPT讲解.md**: 详细的讲解文档（原始版本）
