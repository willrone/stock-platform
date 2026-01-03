# 交互式图表组件

本目录包含了回测结果可视化的交互式图表组件，实现了任务5的所有要求。

## 组件列表

### 1. EquityCurveChart - 收益曲线图表
- **功能**: 显示组合价值和收益率随时间的变化
- **特性**:
  - 支持缩放和平移操作
  - 时间范围选择（1个月、3个月、6个月、1年、全部）
  - 权益曲线和收益率曲线切换
  - 基准对比功能
  - 交互式工具栏（放大、缩小、重置）

### 2. DrawdownChart - 回撤曲线图表
- **功能**: 显示回撤曲线并标注最大回撤期间
- **特性**:
  - 回撤曲线可视化
  - 最大回撤期间高亮标注
  - 最大回撤点标记
  - 回撤统计信息展示
  - 支持缩放和交互操作

### 3. MonthlyHeatmapChart - 月度收益热力图
- **功能**: 显示每月收益率的热力图分布
- **特性**:
  - 月度收益热力图展示
  - 年份筛选功能
  - 季节性模式识别
  - 月度统计详情表格
  - 图表导出功能

### 4. InteractiveChartsContainer - 交互式图表容器
- **功能**: 整合所有图表组件的容器
- **特性**:
  - 标签页式布局
  - 数据缓存机制
  - 错误处理和重试
  - 加载状态管理
  - API数据获取

## 技术实现

### 图表库
- **ECharts**: 主要图表渲染引擎
- **React**: 组件框架
- **TypeScript**: 类型安全

### 数据流
1. 从后端API获取图表数据
2. 数据缓存和错误处理
3. 数据转换和格式化
4. 图表渲染和交互

### API集成
- 使用 `BacktestService` 获取图表数据
- 支持数据缓存和强制刷新
- 错误处理和重试机制

## 使用方式

```tsx
import { InteractiveChartsContainer } from '../components/charts';

<InteractiveChartsContainer
  taskId="your-task-id"
  backtestData={backtestData}
/>
```

## 数据格式

### 收益曲线数据
```typescript
{
  dates: string[];
  portfolioValues: number[];
  returns: number[];
  dailyReturns: number[];
}
```

### 回撤数据
```typescript
{
  dates: string[];
  drawdowns: number[];
  maxDrawdown: number;
  maxDrawdownDate: string;
  maxDrawdownDuration: number;
}
```

### 月度热力图数据
```typescript
{
  monthlyReturns: Array<{
    year: number;
    month: number;
    return: number;
    date: string;
  }>;
  years: number[];
  months: number[];
}
```

## 特性说明

### 交互功能
- **缩放**: 支持鼠标滚轮和工具栏缩放
- **平移**: 支持鼠标拖拽平移
- **时间范围**: 快速选择不同时间段
- **数据悬停**: 显示详细数据点信息
- **图表导出**: 支持PNG格式导出

### 响应式设计
- 自适应不同屏幕尺寸
- 移动端友好的交互体验
- 动态调整图表尺寸

### 性能优化
- 数据缓存机制
- 懒加载和虚拟化
- 防抖和节流处理

## 集成说明

这些组件已经集成到任务详情页面（`/tasks/[id]`）的"交互式图表"标签页中，替换了原有的基础图表展示，提供了更丰富的交互体验和数据分析功能。

## 需求对应

- ✅ **需求 2.1**: 收益曲线图表组件，支持缩放和时间范围选择
- ✅ **需求 2.2**: 回撤曲线图表，标注最大回撤期间  
- ✅ **需求 6.1**: 月度收益热力图组件
- ✅ 支持基准对比和交互式操作
- ✅ 响应式设计和性能优化