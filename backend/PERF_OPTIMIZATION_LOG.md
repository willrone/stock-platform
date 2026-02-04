# 回测性能优化日志 (Phase 0/Phase 1)

## 目标
- Phase 0: 500只×3年回测 ≤ 5分钟
- Phase 1: 500只×3年回测 ≤ 3分钟

## 当前进展

### 2024-02-04 - 初始问题修复

#### 发现的问题
1. **接口不一致问题（严重）**
   - 位置：`backend/app/services/backtest/execution/backtest_executor.py` 第 997-1013 行
   - 问题：`get_precomputed_signal_fast` 函数在创建 `TradingSignal` 对象时缺少必需的 `price` 参数
   - 症状：大量 `TradingSignal.__init__() missing 1 required positional argument: 'price'` 警告
   - 影响：预计算信号无法正常使用，导致回测失败

#### 已完成的修复
1. **修复 TradingSignal 创建问题**
   - 文件：`backend/app/services/backtest/execution/backtest_executor.py`
   - 修改：在 `get_precomputed_signal_fast` 函数中添加价格获取逻辑
   - 代码变更：
     ```python
     # 获取当前价格
     try:
         data = stock_data.get(stock_code)
         if data is not None and date in data.index:
             current_price = float(data.loc[date, "close"])
         else:
             current_price = 0.0
     except Exception:
         current_price = 0.0
     
     return [TradingSignal(
         signal_type=signal,
         stock_code=stock_code,
         timestamp=date,
         price=current_price,  # 添加此参数
         strength=1.0,
         reason=f"Precomputed signal"
     )]
     ```

#### 正在进行
- 运行基准测试：`bench_backtest_500_3y.py --sizes 10,50,100`
- 等待获取基线性能数据

## 下一步计划

### Phase 0 优化策略
1. **主循环优化（优先级：高）**
   - 信号数组化：将 `signal[stock_i, t]` 改为 numpy 数组
   - 数据对齐：close/open/valid_mask 使用 ndarray
   - 减少 DataFrame 操作

2. **持仓管理优化（优先级：高）**
   - positions 从 dict 改为数组
   - buy/sell 使用 numpy.where
   - trade records 使用 list append（延迟转换）

3. **Portfolio History 优化（优先级：中）**
   - 默认降采样或关闭详细历史
   - 仅保留 equity 曲线

4. **分段计时（已完成）**
   - 已在 backtest_executor 中添加 perf_counter 分段计时
   - 输出：load / precompute / main loop / metrics

### 性能基线（已测量 - 2024-02-04）

#### 修复后的基线性能
| 股票数 | 总耗时(s) | 数据加载(s) | 预计算(s) | 主循环(s) | 指标计算(s) |
|--------|-----------|-------------|-----------|-----------|-------------|
| 10     | 24.957    | 0.157       | 0.408     | 24.381    | 0.001       |
| 50     | 39.997    | 0.223       | 0.620     | 39.141    | 0.001       |
| 100    | 41.141    | 0.406       | 0.594     | 40.126    | 0.002       |

#### 性能分析
1. **主循环占比极高**：主循环耗时占总耗时的 97%+
2. **预计算有效**：信号预计算耗时很低（0.4-0.6秒）
3. **扩展性问题**：从50到100股票，耗时几乎不变，说明存在瓶颈
4. **推算500股票耗时**：按当前趋势，500股票预计需要 200-250秒（3-4分钟）

#### 瓶颈识别
- **主循环**是最大瓶颈（占97%+耗时）
- 需要优化的重点：
  1. 每日信号生成（即使有预计算，fallback路径仍在消耗时间）
  2. 交易执行逻辑
  3. 持仓管理（dict操作）
  4. Portfolio快照记录

## 风险与注意事项
1. **接口兼容性**：修改信号处理逻辑时需确保与现有策略兼容
2. **数据完整性**：数组化后需确保边界检查和 NaN 处理
3. **测试覆盖**：每次优化后需运行完整的回测验证

## 验收标准
1. 基准脚本 `backend/scripts/bench_backtest_500_3y.py` 可一键运行
2. 输出包含 wall time + 分段耗时（load / precompute / main loop / metrics）
3. 500只×3年回测时间 ≤ 5分钟（Phase 0）
4. 500只×3年回测时间 ≤ 3分钟（Phase 1）
