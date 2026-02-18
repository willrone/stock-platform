/**
 * 交互式图表容器组件
 * 整合收益曲线、回撤曲线和月度热力图
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Tabs,
  Tab,
  Button,
  CircularProgress,
  Alert,
  Select,
  MenuItem,
  Box,
  Typography,
  FormControl,
  InputLabel,
} from '@mui/material';
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
import { BacktestService, TradeRecord, SignalRecord } from '../../services/backtestService';
import TradingViewChart from './TradingViewChart';

interface InteractiveChartsContainerProps {
  taskId: string;
  backtestData?: Record<string, unknown>;
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
  const [signalRecords, setSignalRecords] = useState<SignalRecord[]>([]);
  const [tradeLoading, setTradeLoading] = useState(false);
  const [signalLoading, setSignalLoading] = useState(false);
  const [selectedStock, setSelectedStock] = useState<string>('');

  // 加载图表数据
  const loadChartData = async (forceRefresh = false) => {
    console.log(
      `[InteractiveChartsContainer] 开始加载图表数据: taskId=${taskId}, forceRefresh=${forceRefresh}`
    );

    try {
      setLoading(true);
      setError(null);

      console.log('[InteractiveChartsContainer] 并行加载三种图表数据...');

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
        console.warn('[InteractiveChartsContainer] 权益曲线数据加载失败:', equityResult.reason);
      }
      if (drawdownResult.status === 'rejected') {
        console.warn('[InteractiveChartsContainer] 回撤曲线数据加载失败:', drawdownResult.reason);
      }
      if (heatmapResult.status === 'rejected') {
        console.warn('[InteractiveChartsContainer] 月度热力图数据加载失败:', heatmapResult.reason);
      }

      // 如果所有数据都加载失败，尝试从现有回测数据生成
      if (!equityData && !drawdownData && !heatmapData) {
        console.log('[InteractiveChartsContainer] 所有API数据加载失败，尝试从现有回测数据生成...');
        generateChartDataFromBacktest();
        return;
      }

      console.log('[InteractiveChartsContainer] 基础图表数据加载结果:', {
        equityData: !!equityData,
        drawdownData: !!drawdownData,
        heatmapData: !!heatmapData,
      });

      // 尝试加载基准数据（可选）
      let benchmarkData;
      try {
        console.log('[InteractiveChartsContainer] 尝试加载基准数据...');
        const benchmark = await BacktestService.getBenchmarkData(taskId);
        if (benchmark && benchmark.benchmark_returns) {
          benchmarkData = {
            dates: benchmark.benchmark_returns.map((r: { date: string; cumulative_return: number; return: number }) => r.date),
            values: benchmark.benchmark_returns.map((r: { date: string; cumulative_return: number; return: number }) => r.cumulative_return * 100000), // 假设初始值
            returns: benchmark.benchmark_returns.map((r: { date: string; cumulative_return: number; return: number }) => r.return),
          };
          console.log('[InteractiveChartsContainer] 基准数据加载成功');
        } else {
          console.log('[InteractiveChartsContainer] 基准数据为空');
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

      console.log('[InteractiveChartsContainer] 所有图表数据设置完成:', finalChartData);
      setChartData(finalChartData);
    } catch (err: unknown) {
      console.error('[InteractiveChartsContainer] 加载图表数据失败:', err);
      // 如果API加载失败，尝试从现有回测数据生成
      console.log('[InteractiveChartsContainer] 尝试从现有回测数据生成图表...');
      generateChartDataFromBacktest();
      if (!chartData || Object.keys(chartData).length === 0) {
        setError(err instanceof Error ? err.message : '加载图表数据失败');
      }
    } finally {
      setLoading(false);
    }
  };

  // 从现有回测数据生成图表数据（备用方案）
  const generateChartDataFromBacktest = () => {
    if (!backtestData) {
      return;
    }

    try {
      // 生成权益曲线数据
      const portfolioHistory = backtestData.portfolio_history || [];
      if (portfolioHistory.length > 0) {
        // 按日期排序，确保数据顺序正确
        const sortedHistory = [...portfolioHistory].sort((a: Record<string, unknown>, b: Record<string, unknown>) => {
          const dateA = new Date((a.date || a.snapshot_date) as string).getTime();
          const dateB = new Date((b.date || b.snapshot_date) as string).getTime();
          return dateA - dateB;
        });

        const equityCurveData = {
          dates: sortedHistory.map((h: Record<string, unknown>) => (h.date || h.snapshot_date) as string),
          portfolioValues: sortedHistory.map((h: Record<string, unknown>) => h.portfolio_value as number),
          returns: sortedHistory.map((h: Record<string, unknown>) => (h.total_return || 0) as number),
          dailyReturns: sortedHistory.map((h: Record<string, unknown>) => (h.daily_return || 0) as number),
        };

        console.log(
          `[InteractiveChartsContainer] 从回测数据生成权益曲线: 数据量=${
            sortedHistory.length
          }, 日期范围=${equityCurveData.dates[0]} 至 ${
            equityCurveData.dates[equityCurveData.dates.length - 1]
          }`
        );

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
          const drawdown = ((peak - value) / peak) * 100;
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
    console.log('[InteractiveChartsContainer] 更新selectedStock:', {
      stockCode,
      stockCodes,
      currentSelected: selectedStock,
    });

    if (stockCode) {
      console.log(`[InteractiveChartsContainer] 使用stockCode: ${stockCode}`);
      setSelectedStock(stockCode);
    } else if (stockCodes && stockCodes.length > 0) {
      const defaultStock = stockCodes[0];
      console.log(`[InteractiveChartsContainer] 使用stockCodes[0]: ${defaultStock}`);
      setSelectedStock(prev => {
        // 如果已经有选中的股票且该股票仍在列表中，保持选中
        if (prev && stockCodes.includes(prev)) {
          return prev;
        }
        return defaultStock;
      });
    } else {
      console.warn('[InteractiveChartsContainer] 没有可用的股票代码');
    }
  }, [stockCode, stockCodes]);

  useEffect(() => {
    if (!taskId || !selectedStock) {
      setTradeRecords([]);
      setSignalRecords([]);
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

    const fetchSignals = async () => {
      try {
        setSignalLoading(true);
        const signalsResponse = await BacktestService.getSignalRecords(taskId, {
          stockCode: selectedStock,
          limit: 1000,
          orderBy: 'timestamp',
          orderDesc: false,
        });
        setSignalRecords(signalsResponse.signals);
      } catch (signalError) {
        console.warn('[InteractiveChartsContainer] 无法加载信号记录:', signalError);
        setSignalRecords([]);
      } finally {
        setSignalLoading(false);
      }
    };

    fetchTrades();
    fetchSignals();
  }, [taskId, selectedStock]);

  // 刷新数据
  const handleRefresh = () => {
    loadChartData(true);
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 6,
            }}
          >
            <CircularProgress size={48} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              正在加载图表数据...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert
            severity="error"
            action={
              <Button
                size="small"
                color="error"
                startIcon={<RefreshCw size={16} />}
                onClick={handleRefresh}
              >
                重试
              </Button>
            }
            icon={<AlertCircle size={20} />}
          >
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              图表数据加载失败
            </Typography>
            <Typography variant="caption">{error}</Typography>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 图表标签页 */}
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <BarChart3 size={20} color="#1976d2" />
              <Typography
                variant="h5"
                component="h2"
                sx={{ fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem', md: '1.5rem' } }}
              >
                交互式图表分析
              </Typography>
            </Box>
          }
          action={
            <Box
              sx={{
                display: 'flex',
                flexDirection: { xs: 'column', sm: 'row' },
                alignItems: { xs: 'flex-end', sm: 'center' },
                gap: 1,
              }}
            >
              {stockCodes && stockCodes.length > 0 && (
                <FormControl size="small" sx={{ minWidth: 160 }}>
                  <InputLabel>选择股票</InputLabel>
                  <Select
                    value={selectedStock || ''}
                    label="选择股票"
                    onChange={e => setSelectedStock(e.target.value)}
                  >
                    {stockCodes.map(code => (
                      <MenuItem key={code} value={code}>
                        {code}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              <Button
                size="small"
                variant="outlined"
                startIcon={<RefreshCw size={16} />}
                onClick={handleRefresh}
                disabled={loading}
              >
                刷新数据
              </Button>
            </Box>
          }
        />

        <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
          <Tabs
            value={activeTab}
            onChange={(e, newValue) => setActiveTab(newValue)}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
          >
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TrendingUp size={16} />
                  <span>价格走势</span>
                </Box>
              }
              value="price"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TrendingUp size={16} />
                  <span>收益曲线</span>
                </Box>
              }
              value="equity"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TrendingDown size={16} />
                  <span>回撤分析</span>
                </Box>
              }
              value="drawdown"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Calendar size={16} />
                  <span>月度热力图</span>
                </Box>
              }
              value="monthly"
            />
          </Tabs>

          <Box sx={{ mt: 2 }}>
            {activeTab === 'price' && (
              <Box>
                {selectedStock ? (
                  <TradingViewChart
                    stockCode={selectedStock}
                    taskId={taskId}
                    startDate={(() => {
                      const startDate =
                        backtestData?.start_date ||
                        backtestData?.period?.start_date ||
                        backtestData?.backtest_config?.start_date;
                      console.log(
                        '[InteractiveChartsContainer] TradingViewChart startDate:',
                        startDate,
                        'from backtestData:',
                        backtestData
                      );
                      return startDate;
                    })()}
                    endDate={(() => {
                      const endDate =
                        backtestData?.end_date ||
                        backtestData?.period?.end_date ||
                        backtestData?.backtest_config?.end_date;
                      console.log(
                        '[InteractiveChartsContainer] TradingViewChart endDate:',
                        endDate,
                        'from backtestData:',
                        backtestData
                      );
                      return endDate;
                    })()}
                    trades={tradeRecords}
                    signals={signalRecords}
                    height={420}
                  />
                ) : (
                  <Box
                    sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 6 }}
                  >
                    <AlertCircle size={24} color="#666" style={{ marginRight: 8 }} />
                    <Typography variant="body2" color="text.secondary">
                      暂无股票代码
                    </Typography>
                  </Box>
                )}
                {tradeLoading && (
                  <Box sx={{ mt: 1.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      交易记录加载中...
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {activeTab === 'equity' && (
              <Box>
                {chartData.equityCurve ? (
                  <EquityCurveChart
                    taskId={taskId}
                    data={chartData.equityCurve}
                    benchmarkData={chartData.benchmarkData}
                    loading={false}
                  />
                ) : (
                  <Box
                    sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 6 }}
                  >
                    <AlertCircle size={24} color="#666" style={{ marginRight: 8 }} />
                    <Typography variant="body2" color="text.secondary">
                      暂无收益曲线数据
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {activeTab === 'drawdown' && (
              <Box>
                {chartData.drawdownCurve ? (
                  <DrawdownChart taskId={taskId} data={chartData.drawdownCurve} loading={false} />
                ) : (
                  <Box
                    sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 6 }}
                  >
                    <AlertCircle size={24} color="#666" style={{ marginRight: 8 }} />
                    <Typography variant="body2" color="text.secondary">
                      暂无回撤分析数据
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {activeTab === 'monthly' && (
              <Box>
                {chartData.monthlyHeatmap && chartData.monthlyHeatmap.monthlyReturns.length > 0 ? (
                  <MonthlyHeatmapChart
                    taskId={taskId}
                    data={chartData.monthlyHeatmap}
                    loading={false}
                  />
                ) : (
                  <Box
                    sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 6 }}
                  >
                    <AlertCircle size={24} color="#666" style={{ marginRight: 8 }} />
                    <Typography variant="body2" color="text.secondary">
                      暂无月度收益数据
                    </Typography>
                  </Box>
                )}
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
