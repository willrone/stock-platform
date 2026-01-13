/**
 * 交互式图表容器组件
 * 整合收益曲线、回撤曲线和月度热力图
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Tabs,
  Tab,
  Button,
  Spinner,
  Alert,
  Select,
  SelectItem,
} from '@heroui/react';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Calendar,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';

import EquityCurveChart from './EquityCurveChart';
import DrawdownChart from './DrawdownChart';
import MonthlyHeatmapChart from './MonthlyHeatmapChart';
import { BacktestService, TradeRecord } from '../../services/backtestService';
import TradingViewChart from './TradingViewChart';

interface InteractiveChartsContainerProps {
  taskId: string;
  backtestData?: any;
  stockCode?: string;
  stockCodes?: string[];
}

interface ChartData {
  equityCurve?: {
    dates: string[];
    portfolioValues: number[];
    returns: number[];
    dailyReturns: number[];
  };
  drawdownCurve?: {
    dates: string[];
    drawdowns: number[];
    maxDrawdown: number;
    maxDrawdownDate: string;
    maxDrawdownDuration: number;
  };
  monthlyHeatmap?: {
    monthlyReturns: Array<{
      year: number;
      month: number;
      return: number;
      date: string;
    }>;
    years: number[];
    months: number[];
  };
  benchmarkData?: {
    dates: string[];
    values: number[];
    returns: number[];
  };
}

export default function InteractiveChartsContainer({
  taskId,
  backtestData,
  stockCode,
  stockCodes,
}: InteractiveChartsContainerProps) {
  const [chartData, setChartData] = useState<ChartData>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('equity');
  const [tradeRecords, setTradeRecords] = useState<TradeRecord[]>([]);
  const [tradeLoading, setTradeLoading] = useState(false);
  const [selectedStock, setSelectedStock] = useState<string>('');

  // 加载图表数据
  const loadChartData = async (forceRefresh = false) => {
    console.log(`[InteractiveChartsContainer] 开始加载图表数据: taskId=${taskId}, forceRefresh=${forceRefresh}`);
    
    try {
      setLoading(true);
      setError(null);

      console.log(`[InteractiveChartsContainer] 并行加载三种图表数据...`);
      
      // 并行加载所有图表数据，允许部分失败
      const [equityResult, drawdownResult, heatmapResult] = await Promise.allSettled([
        BacktestService.getChartData(taskId, 'equity_curve', forceRefresh),
        BacktestService.getChartData(taskId, 'drawdown_curve', forceRefresh),
        BacktestService.getChartData(taskId, 'monthly_heatmap', forceRefresh),
      ]);

      // 处理每个结果
      const equityData = equityResult.status === 'fulfilled' ? equityResult.value : null;
      const drawdownData = drawdownResult.status === 'fulfilled' ? drawdownResult.value : null;
      const heatmapData = heatmapResult.status === 'fulfilled' ? heatmapResult.value : null;

      if (equityResult.status === 'rejected') {
        console.warn(`[InteractiveChartsContainer] 权益曲线数据加载失败:`, equityResult.reason);
      }
      if (drawdownResult.status === 'rejected') {
        console.warn(`[InteractiveChartsContainer] 回撤曲线数据加载失败:`, drawdownResult.reason);
      }
      if (heatmapResult.status === 'rejected') {
        console.warn(`[InteractiveChartsContainer] 月度热力图数据加载失败:`, heatmapResult.reason);
      }

      // 如果所有数据都加载失败，尝试从现有回测数据生成
      if (!equityData && !drawdownData && !heatmapData) {
        console.log(`[InteractiveChartsContainer] 所有API数据加载失败，尝试从现有回测数据生成...`);
        generateChartDataFromBacktest();
        return;
      }

      console.log(`[InteractiveChartsContainer] 基础图表数据加载结果:`, {
        equityData: !!equityData,
        drawdownData: !!drawdownData,
        heatmapData: !!heatmapData
      });

      // 尝试加载基准数据（可选）
      let benchmarkData;
      try {
        console.log(`[InteractiveChartsContainer] 尝试加载基准数据...`);
        const benchmark = await BacktestService.getBenchmarkData(taskId);
        if (benchmark && benchmark.benchmark_returns) {
          benchmarkData = {
            dates: benchmark.benchmark_returns.map((r: any) => r.date),
            values: benchmark.benchmark_returns.map((r: any) => r.cumulative_return * 100000), // 假设初始值
            returns: benchmark.benchmark_returns.map((r: any) => r.return),
          };
          console.log(`[InteractiveChartsContainer] 基准数据加载成功`);
        } else {
          console.log(`[InteractiveChartsContainer] 基准数据为空`);
        }
      } catch (benchmarkError) {
        console.warn('[InteractiveChartsContainer] 无法加载基准数据:', benchmarkError);
      }

      const finalChartData = {
        equityCurve: equityData as ChartData['equityCurve'],
        drawdownCurve: drawdownData as ChartData['drawdownCurve'],
        monthlyHeatmap: heatmapData as ChartData['monthlyHeatmap'],
        benchmarkData,
      };
      
      console.log(`[InteractiveChartsContainer] 所有图表数据设置完成:`, finalChartData);
      setChartData(finalChartData);

    } catch (err: any) {
      console.error('[InteractiveChartsContainer] 加载图表数据失败:', err);
      // 如果API加载失败，尝试从现有回测数据生成
      console.log(`[InteractiveChartsContainer] 尝试从现有回测数据生成图表...`);
      generateChartDataFromBacktest();
      if (!chartData || Object.keys(chartData).length === 0) {
        setError(err.message || '加载图表数据失败');
      }
    } finally {
      setLoading(false);
    }
  };

  // 从现有回测数据生成图表数据（备用方案）
  const generateChartDataFromBacktest = () => {
    if (!backtestData) return;

    try {
      // 生成权益曲线数据
      const portfolioHistory = backtestData.portfolio_history || [];
      if (portfolioHistory.length > 0) {
        // 按日期排序，确保数据顺序正确
        const sortedHistory = [...portfolioHistory].sort((a: any, b: any) => {
          const dateA = new Date(a.date || a.snapshot_date).getTime();
          const dateB = new Date(b.date || b.snapshot_date).getTime();
          return dateA - dateB;
        });
        
        const equityCurveData = {
          dates: sortedHistory.map((h: any) => h.date || h.snapshot_date),
          portfolioValues: sortedHistory.map((h: any) => h.portfolio_value),
          returns: sortedHistory.map((h: any) => h.total_return || 0),
          dailyReturns: sortedHistory.map((h: any) => h.daily_return || 0),
        };
        
        console.log(`[InteractiveChartsContainer] 从回测数据生成权益曲线: 数据量=${sortedHistory.length}, 日期范围=${equityCurveData.dates[0]} 至 ${equityCurveData.dates[equityCurveData.dates.length - 1]}`);

        // 生成回撤数据
        const values = equityCurveData.portfolioValues;
        const drawdowns: number[] = [];
        let peak = values[0];
        let maxDrawdown = 0;
        let maxDrawdownDate = equityCurveData.dates[0];

        values.forEach((value: number, index: number) => {
          if (value > peak) {
            peak = value;
          }
          const drawdown = (peak - value) / peak * 100;
          drawdowns.push(-drawdown); // 负值表示回撤
          
          if (drawdown > Math.abs(maxDrawdown)) {
            maxDrawdown = -drawdown;
            maxDrawdownDate = equityCurveData.dates[index];
          }
        });

        const drawdownCurveData = {
          dates: equityCurveData.dates,
          drawdowns,
          maxDrawdown,
          maxDrawdownDate,
          maxDrawdownDuration: 0, // 简化计算
        };

        // 生成月度热力图数据（简化版）
        const monthlyHeatmapData = {
          monthlyReturns: [],
          years: [],
          months: Array.from({ length: 12 }, (_, i) => i + 1),
        };

        setChartData({
          equityCurve: equityCurveData,
          drawdownCurve: drawdownCurveData,
          monthlyHeatmap: monthlyHeatmapData,
        });
      }
    } catch (err) {
      console.error('从回测数据生成图表数据失败:', err);
    }
  };

  useEffect(() => {
    if (taskId) {
      // 首次加载时强制刷新，确保获取完整数据（包括所有年份的数据）
      loadChartData(true).catch(() => {
        // 如果API加载失败，尝试从现有回测数据生成
        generateChartDataFromBacktest();
        setLoading(false);
      });
    }
  }, [taskId]);

  useEffect(() => {
    console.log(`[InteractiveChartsContainer] 更新selectedStock:`, { stockCode, stockCodes, currentSelected: selectedStock });
    
    if (stockCode) {
      console.log(`[InteractiveChartsContainer] 使用stockCode: ${stockCode}`);
      setSelectedStock(stockCode);
    } else if (stockCodes && stockCodes.length > 0) {
      const defaultStock = stockCodes[0];
      console.log(`[InteractiveChartsContainer] 使用stockCodes[0]: ${defaultStock}`);
      setSelectedStock((prev) => {
        // 如果已经有选中的股票且该股票仍在列表中，保持选中
        if (prev && stockCodes.includes(prev)) {
          return prev;
        }
        return defaultStock;
      });
    } else {
      console.warn(`[InteractiveChartsContainer] 没有可用的股票代码`);
    }
  }, [stockCode, stockCodes]);

  useEffect(() => {
    if (!taskId || !selectedStock) {
      setTradeRecords([]);
      return;
    }

    const fetchTrades = async () => {
      try {
        setTradeLoading(true);
        const tradesResponse = await BacktestService.getTradeRecords(taskId, {
          stockCode: selectedStock,
          limit: 1000,
          orderBy: 'timestamp',
          orderDesc: false,
        });
        setTradeRecords(tradesResponse.trades);
      } catch (tradeError) {
        console.warn('[InteractiveChartsContainer] 无法加载交易记录:', tradeError);
        setTradeRecords([]);
      } finally {
        setTradeLoading(false);
      }
    };

    fetchTrades();
  }, [taskId, selectedStock]);

  // 刷新数据
  const handleRefresh = () => {
    loadChartData(true);
  };

  if (loading) {
    return (
      <Card>
        <CardBody>
          <div className="flex flex-col items-center justify-center py-12">
            <Spinner size="lg" />
            <p className="mt-4 text-default-500">正在加载图表数据...</p>
          </div>
        </CardBody>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardBody>
          <Alert
            color="danger"
            variant="flat"
            startContent={<AlertCircle className="w-5 h-5" />}
            endContent={
              <Button
                size="sm"
                variant="flat"
                color="danger"
                startContent={<RefreshCw className="w-4 h-4" />}
                onPress={handleRefresh}
              >
                重试
              </Button>
            }
          >
            <div>
              <p className="font-medium">图表数据加载失败</p>
              <p className="text-sm">{error}</p>
            </div>
          </Alert>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 图表标签页 */}
      <Card>
        <CardHeader className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-semibold">交互式图表分析</h2>
          </div>

          <div className="flex items-center gap-2">
            {stockCodes && stockCodes.length > 0 && (
              <Select
                selectedKeys={selectedStock ? [selectedStock] : []}
                onSelectionChange={(keys) => {
                  const selected = Array.from(keys)[0] as string;
                  setSelectedStock(selected);
                }}
                aria-label="选择股票"
                size="sm"
                className="min-w-[160px]"
              >
                {stockCodes.map((code) => (
                  <SelectItem key={code}>{code}</SelectItem>
                ))}
              </Select>
            )}
          
            <Button
              size="sm"
              variant="flat"
              startContent={<RefreshCw className="w-4 h-4" />}
              onPress={handleRefresh}
              isLoading={loading}
            >
              刷新数据
            </Button>
          </div>
        </CardHeader>

        <CardBody>
          <Tabs
            selectedKey={activeTab}
            onSelectionChange={(key) => setActiveTab(key as string)}
            variant="underlined"
            classNames={{
              tabList: "gap-6 w-full relative rounded-none p-0 border-b border-divider",
              cursor: "w-full bg-primary",
              tab: "max-w-fit px-0 h-12",
              tabContent: "group-data-[selected=true]:text-primary"
            }}
          >
            <Tab
              key="price"
              title={
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4" />
                  <span>价格走势</span>
                </div>
              }
            >
              {selectedStock ? (
                <TradingViewChart
                  stockCode={selectedStock}
                  startDate={(() => {
                    // 尝试从多个可能的位置获取开始日期
                    const startDate = backtestData?.start_date || 
                                     backtestData?.period?.start_date ||
                                     backtestData?.backtest_config?.start_date;
                    console.log(`[InteractiveChartsContainer] TradingViewChart startDate:`, startDate, 'from backtestData:', backtestData);
                    return startDate;
                  })()}
                  endDate={(() => {
                    // 尝试从多个可能的位置获取结束日期
                    const endDate = backtestData?.end_date || 
                                   backtestData?.period?.end_date ||
                                   backtestData?.backtest_config?.end_date;
                    console.log(`[InteractiveChartsContainer] TradingViewChart endDate:`, endDate, 'from backtestData:', backtestData);
                    return endDate;
                  })()}
                  trades={tradeRecords}
                  height={420}
                />
              ) : (
                <div className="flex items-center justify-center py-12 text-default-500">
                  <AlertCircle className="w-6 h-6 mr-2" />
                  <span>暂无股票代码</span>
                </div>
              )}
              {tradeLoading && (
                <div className="mt-3 text-sm text-default-500">
                  交易记录加载中...
                </div>
              )}
            </Tab>

            <Tab
              key="equity"
              title={
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4" />
                  <span>收益曲线</span>
                </div>
              }
            >
              {chartData.equityCurve ? (
                <EquityCurveChart
                  taskId={taskId}
                  data={chartData.equityCurve}
                  benchmarkData={chartData.benchmarkData}
                  loading={false}
                />
              ) : (
                <div className="flex items-center justify-center py-12 text-default-500">
                  <AlertCircle className="w-6 h-6 mr-2" />
                  <span>暂无收益曲线数据</span>
                </div>
              )}
            </Tab>

            <Tab
              key="drawdown"
              title={
                <div className="flex items-center space-x-2">
                  <TrendingDown className="w-4 h-4" />
                  <span>回撤分析</span>
                </div>
              }
            >
              {chartData.drawdownCurve ? (
                <DrawdownChart
                  taskId={taskId}
                  data={chartData.drawdownCurve}
                  loading={false}
                />
              ) : (
                <div className="flex items-center justify-center py-12 text-default-500">
                  <AlertCircle className="w-6 h-6 mr-2" />
                  <span>暂无回撤分析数据</span>
                </div>
              )}
            </Tab>

            <Tab
              key="monthly"
              title={
                <div className="flex items-center space-x-2">
                  <Calendar className="w-4 h-4" />
                  <span>月度热力图</span>
                </div>
              }
            >
              {chartData.monthlyHeatmap && chartData.monthlyHeatmap.monthlyReturns.length > 0 ? (
                <MonthlyHeatmapChart
                  taskId={taskId}
                  data={chartData.monthlyHeatmap}
                  loading={false}
                />
              ) : (
                <div className="flex items-center justify-center py-12 text-default-500">
                  <AlertCircle className="w-6 h-6 mr-2" />
                  <span>暂无月度收益数据</span>
                </div>
              )}
            </Tab>
          </Tabs>
        </CardBody>
      </Card>
    </div>
  );
}
