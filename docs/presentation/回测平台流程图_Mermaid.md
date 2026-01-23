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
    style CheckPrev fill:#fbbf24,stroke:#333,stroke-width:2px
    style CheckCurr1 fill:#fbbf24,stroke:#333,stroke-width:2px
    style CheckCurr2 fill:#fbbf24,stroke:#333,stroke-width:2px
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
    
    QueueTask --> Redirect[跳转到任务详情页<br/>/tasks/[id]]
    
    Redirect --> Monitor[监控回测进度<br/>WebSocket实时推送]
    
    Monitor --> End([任务创建完成])
    
    style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
    style End fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style SubmitTask fill:#4299e1,stroke:#333,stroke-width:2px,color:#fff
    style CreateTask fill:#48bb78,stroke:#333,stroke-width:2px,color:#fff
    style ShowError fill:#f56565,stroke:#333,stroke-width:2px,color:#fff
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
