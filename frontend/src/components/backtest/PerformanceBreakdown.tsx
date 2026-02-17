/**
 * 绩效分解组件
 * 实现月度和年度绩效分解、季节性分析
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Select,
  MenuItem,
  Button,
  Tooltip,
  Box,
  Typography,
  FormControl,
  InputLabel,
  TableContainer,
  Paper,
} from '@mui/material';
import * as echarts from 'echarts';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Target,
  Award,
  Clock,
  Calendar,
} from 'lucide-react';

interface MonthlyPerformance {
  year: number;
  month: number;
  return_rate: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  trading_days: number;
}

interface YearlyPerformance {
  year: number;
  annual_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  sortino_ratio: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
}

interface SeasonalAnalysis {
  monthly_avg_returns: number[]; // 12个月的平均收益率
  monthly_win_rates: number[]; // 12个月的胜率
  quarterly_performance: {
    q1: number;
    q2: number;
    q3: number;
    q4: number;
  };
  best_month: {
    month: number;
    avg_return: number;
  };
  worst_month: {
    month: number;
    avg_return: number;
  };
}

interface BenchmarkComparison {
  dates: string[];
  strategy_returns: number[];
  benchmark_returns: number[];
  excess_returns: number[];
  tracking_error: number;
  information_ratio: number;
  beta: number;
  alpha: number;
  correlation: number;
}

interface PerformanceBreakdownProps {
  taskId: string;
  monthlyPerformance: MonthlyPerformance[];
  yearlyPerformance: YearlyPerformance[];
  seasonalAnalysis: SeasonalAnalysis;
  benchmarkComparison: BenchmarkComparison;
}

export function PerformanceBreakdown({
  taskId,
  monthlyPerformance,
  yearlyPerformance,
  seasonalAnalysis,
  benchmarkComparison,
}: PerformanceBreakdownProps) {
  const [selectedYear, setSelectedYear] = useState<string>('all');
  const [selectedMetric, setSelectedMetric] = useState<'return' | 'volatility' | 'sharpe'>(
    'return'
  );
  const [activeTab, setActiveTab] = useState<string>('heatmap');

  // 图表引用
  const heatmapChartRef = useRef<HTMLDivElement>(null);
  const seasonalChartRef = useRef<HTMLDivElement>(null);
  const benchmarkChartRef = useRef<HTMLDivElement>(null);
  const heatmapChartInstance = useRef<echarts.ECharts | null>(null);
  const seasonalChartInstance = useRef<echarts.ECharts | null>(null);
  const benchmarkChartInstance = useRef<echarts.ECharts | null>(null);

  // 月份名称
  const monthNames = [
    '1月',
    '2月',
    '3月',
    '4月',
    '5月',
    '6月',
    '7月',
    '8月',
    '9月',
    '10月',
    '11月',
    '12月',
  ];

  // 季度名称（保留以备将来使用）
  // const quarterNames = ['Q1', 'Q2', 'Q3', 'Q4'];

  // 可用年份
  const availableYears = useMemo(() => {
    const yearSet = new Set(monthlyPerformance.map(item => item.year));
    const years = Array.from(yearSet).sort();
    return years;
  }, [monthlyPerformance]);

  // 热力图数据
  const heatmapData = useMemo(() => {
    if (!monthlyPerformance.length) {
      return [];
    }

    const filteredData =
      selectedYear === 'all'
        ? monthlyPerformance
        : monthlyPerformance.filter(item => item.year.toString() === selectedYear);

    return filteredData.map(item => {
      let value;
      switch (selectedMetric) {
        case 'return':
          value = item.return_rate;
          break;
        case 'volatility':
          value = item.volatility;
          break;
        case 'sharpe':
          value = item.sharpe_ratio;
          break;
        default:
          value = item.return_rate;
      }

      return [item.month - 1, item.year, value];
    });
  }, [monthlyPerformance, selectedYear, selectedMetric]);

  // 年度统计
  const yearlyStats = useMemo(() => {
    if (!yearlyPerformance.length) {
      return null;
    }

    const totalReturn = yearlyPerformance.reduce((sum, year) => sum + year.annual_return, 0);
    const avgReturn = totalReturn / yearlyPerformance.length;
    const avgVolatility =
      yearlyPerformance.reduce((sum, year) => sum + year.volatility, 0) / yearlyPerformance.length;
    const avgSharpe =
      yearlyPerformance.reduce((sum, year) => sum + year.sharpe_ratio, 0) /
      yearlyPerformance.length;

    const bestYear = yearlyPerformance.reduce((best, year) =>
      year.annual_return > best.annual_return ? year : best
    );
    const worstYear = yearlyPerformance.reduce((worst, year) =>
      year.annual_return < worst.annual_return ? year : worst
    );

    return {
      avgReturn,
      avgVolatility,
      avgSharpe,
      bestYear,
      worstYear,
      totalYears: yearlyPerformance.length,
    };
  }, [yearlyPerformance]);

  // 初始化热力图
  useEffect(() => {
    if (!heatmapChartRef.current || !heatmapData.length) {
      return;
    }

    if (heatmapChartInstance.current) {
      heatmapChartInstance.current.dispose();
    }

    heatmapChartInstance.current = echarts.init(heatmapChartRef.current);

    const yearSet = new Set(heatmapData.map(item => item[1]));
    const years = Array.from(yearSet).sort();

    const option = {
      title: {
        text: '月度表现热力图',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        position: 'top',
        formatter: function (params: { data: [number, number, number] }) {
          const year = params.data[1];
          const value = params.data[2];

          let unit = '';
          let label = '';
          switch (selectedMetric) {
            case 'return':
              unit = '%';
              label = '收益率';
              break;
            case 'volatility':
              unit = '%';
              label = '波动率';
              break;
            case 'sharpe':
              unit = '';
              label = '夏普比率';
              break;
          }

          return `${year}年${month}<br/>${label}: ${(value * 100).toFixed(2)}${unit}`;
        },
      },
      grid: {
        height: '50%',
        top: '10%',
      },
      xAxis: {
        type: 'category',
        data: monthNames,
        splitArea: {
          show: true,
        },
      },
      yAxis: {
        type: 'category',
        data: years,
        splitArea: {
          show: true,
        },
      },
      visualMap: {
        min: Math.min(...heatmapData.map(item => item[2])),
        max: Math.max(...heatmapData.map(item => item[2])),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
        inRange: {
          color: [
            '#313695',
            '#4575b4',
            '#74add1',
            '#abd9e9',
            '#e0f3f8',
            '#ffffbf',
            '#fee090',
            '#fdae61',
            '#f46d43',
            '#d73027',
            '#a50026',
          ],
        },
      },
      series: [
        {
          name: selectedMetric,
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: function (params: { data: [number, number, number] }) {
            },
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
    };

    heatmapChartInstance.current.setOption(option);

    const handleResize = () => {
      if (heatmapChartInstance.current) {
        heatmapChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [heatmapData, selectedMetric, monthNames]);

  // 初始化季节性分析图表
  useEffect(() => {
    if (!seasonalChartRef.current || !seasonalAnalysis) {
      return;
    }

    if (seasonalChartInstance.current) {
      seasonalChartInstance.current.dispose();
    }

    seasonalChartInstance.current = echarts.init(seasonalChartRef.current);

    const option = {
      title: {
        text: '季节性表现分析',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
      },
      legend: {
        data: ['平均收益率', '胜率'],
        top: '10%',
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '20%',
        containLabel: true,
      },
      xAxis: [
        {
          type: 'category',
          data: monthNames,
          axisPointer: {
            type: 'shadow',
          },
        },
      ],
      yAxis: [
        {
          type: 'value',
          name: '收益率 (%)',
          position: 'left',
          axisLabel: {
            formatter: '{value}%',
          },
        },
        {
          type: 'value',
          name: '胜率 (%)',
          position: 'right',
          axisLabel: {
            formatter: '{value}%',
          },
        },
      ],
      series: [
        {
          name: '平均收益率',
          type: 'bar',
          yAxisIndex: 0,
          data: seasonalAnalysis.monthly_avg_returns.map(val => (val * 100).toFixed(2)),
          itemStyle: {
            color: '#3b82f6',
          },
        },
        {
          name: '胜率',
          type: 'line',
          yAxisIndex: 1,
          data: seasonalAnalysis.monthly_win_rates.map(val => (val * 100).toFixed(2)),
          itemStyle: {
            color: '#10b981',
          },
          lineStyle: {
            width: 3,
          },
        },
      ],
    };

    seasonalChartInstance.current.setOption(option);

    const handleResize = () => {
      if (seasonalChartInstance.current) {
        seasonalChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [seasonalAnalysis, monthNames]);

  // 初始化基准对比图表
  useEffect(() => {
    if (!benchmarkChartRef.current || !benchmarkComparison) {
      return;
    }

    if (benchmarkChartInstance.current) {
      benchmarkChartInstance.current.dispose();
    }

    benchmarkChartInstance.current = echarts.init(benchmarkChartRef.current);

    const option = {
      title: {
        text: '策略 vs 基准表现对比',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
      },
      legend: {
        data: ['策略收益', '基准收益', '超额收益'],
        top: '10%',
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '20%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: benchmarkComparison.dates,
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%',
        },
      },
      series: [
        {
          name: '策略收益',
          type: 'line',
          data: benchmarkComparison.strategy_returns.map(val => (val * 100).toFixed(2)),
          itemStyle: {
            color: '#3b82f6',
          },
          smooth: true,
        },
        {
          name: '基准收益',
          type: 'line',
          data: benchmarkComparison.benchmark_returns.map(val => (val * 100).toFixed(2)),
          itemStyle: {
            color: '#6b7280',
          },
          smooth: true,
        },
        {
          name: '超额收益',
          type: 'line',
          data: benchmarkComparison.excess_returns.map(val => (val * 100).toFixed(2)),
          itemStyle: {
            color: '#10b981',
          },
          smooth: true,
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(16, 185, 129, 0.3)' },
              { offset: 1, color: 'rgba(16, 185, 129, 0.1)' },
            ]),
          },
        },
      ],
    };

    benchmarkChartInstance.current.setOption(option);

    const handleResize = () => {
      if (benchmarkChartInstance.current) {
        benchmarkChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [benchmarkComparison]);

  // 格式化函数
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatRatio = (value: number) => value.toFixed(3);

  if (!monthlyPerformance.length || !yearlyPerformance.length) {
    return (
      <Card>
        <CardContent
          sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <Calendar size={48} color="#999" style={{ margin: '0 auto 8px' }} />
            <Typography variant="body2" color="text.secondary">
              暂无绩效分解数据
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 年度统计概览 */}
      {yearlyStats && (
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
            gap: 2,
          }}
        >
          <Card>
            <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp size={20} color="#1976d2" />
                <Box sx={{ overflow: 'hidden' }}>
                  <Typography variant="caption" color="text.secondary">
                    平均年化收益
                  </Typography>
                  <Typography
                    sx={{
                      fontWeight: 600,
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                      overflow: 'hidden',
                      wordBreak: 'break-word',
                    }}
                  >
                    {formatPercent(yearlyStats.avgReturn)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Activity size={20} color="#9c27b0" />
                <Box sx={{ overflow: 'hidden' }}>
                  <Typography variant="caption" color="text.secondary">
                    平均波动率
                  </Typography>
                  <Typography
                    sx={{
                      fontWeight: 600,
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                      overflow: 'hidden',
                      wordBreak: 'break-word',
                    }}
                  >
                    {formatPercent(yearlyStats.avgVolatility)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Target size={20} color="#2e7d32" />
                <Box sx={{ overflow: 'hidden' }}>
                  <Typography variant="caption" color="text.secondary">
                    平均夏普比率
                  </Typography>
                  <Typography
                    sx={{
                      fontWeight: 600,
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                      overflow: 'hidden',
                      wordBreak: 'break-word',
                    }}
                  >
                    {formatRatio(yearlyStats.avgSharpe)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Clock size={20} color="#ed6c02" />
                <Box sx={{ overflow: 'hidden' }}>
                  <Typography variant="caption" color="text.secondary">
                    回测年数
                  </Typography>
                  <Typography
                    sx={{
                      fontWeight: 600,
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                      overflow: 'hidden',
                      wordBreak: 'break-word',
                    }}
                  >
                    {yearlyStats.totalYears} 年
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* 最佳和最差年份 */}
      {yearlyStats && (
        <Box
          sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}
        >
          <Card>
            <CardHeader
              sx={{ px: { xs: 1.5, md: 2 }, pt: { xs: 1.5, md: 2 }, pb: 0 }}
              title={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Award size={20} color="#2e7d32" />
                  <Typography
                    sx={{
                      fontWeight: 600,
                      color: 'success.main',
                      fontSize: { xs: '1rem', md: '1.25rem' },
                    }}
                  >
                    最佳年份
                  </Typography>
                </Box>
              }
            />
            <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ overflow: 'hidden' }}>
                  <Typography sx={{ fontWeight: 600, fontSize: { xs: '1.5rem', md: '2rem' } }}>
                    {yearlyStats.bestYear.year}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    年化收益率
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'right', overflow: 'hidden' }}>
                  <Typography
                    sx={{
                      fontWeight: 600,
                      color: 'success.main',
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                      overflow: 'hidden',
                      wordBreak: 'break-word',
                    }}
                  >
                    {formatPercent(yearlyStats.bestYear.annual_return)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    夏普: {formatRatio(yearlyStats.bestYear.sharpe_ratio)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardHeader
              sx={{ px: { xs: 1.5, md: 2 }, pt: { xs: 1.5, md: 2 }, pb: 0 }}
              title={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TrendingDown size={20} color="#d32f2f" />
                  <Typography
                    sx={{
                      fontWeight: 600,
                      color: 'error.main',
                      fontSize: { xs: '1rem', md: '1.25rem' },
                    }}
                  >
                    最差年份
                  </Typography>
                </Box>
              }
            />
            <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ overflow: 'hidden' }}>
                  <Typography sx={{ fontWeight: 600, fontSize: { xs: '1.5rem', md: '2rem' } }}>
                    {yearlyStats.worstYear.year}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    年化收益率
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'right', overflow: 'hidden' }}>
                  <Typography
                    sx={{
                      fontWeight: 600,
                      color: 'error.main',
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                      overflow: 'hidden',
                      wordBreak: 'break-word',
                    }}
                  >
                    {formatPercent(yearlyStats.worstYear.annual_return)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    夏普: {formatRatio(yearlyStats.worstYear.sharpe_ratio)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}

      <Box>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Calendar size={16} />
                <span>月度热力图</span>
              </Box>
            }
            value="heatmap"
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <BarChart3 size={16} />
                <span>季节性分析</span>
              </Box>
            }
            value="seasonal"
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Target size={16} />
                <span>年度表现</span>
              </Box>
            }
            value="yearly"
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Activity size={16} />
                <span>基准对比</span>
              </Box>
            }
            value="benchmark"
          />
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {activeTab === 'heatmap' && (
            <Card>
              <CardHeader
                title={
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      width: '100%',
                    }}
                  >
                    <Typography
                      variant="h6"
                      component="h3"
                      sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
                    >
                      月度表现热力图
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>年份</InputLabel>
                        <Select
                          value={selectedYear}
                          label="年份"
                          onChange={e => setSelectedYear(e.target.value)}
                        >
                          <MenuItem value="all">全部</MenuItem>
                          {availableYears.map(year => (
                            <MenuItem key={year} value={year.toString()}>
                              {year}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>

                      <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>指标</InputLabel>
                        <Select
                          value={selectedMetric}
                          label="指标"
                          onChange={e =>
                            setSelectedMetric(e.target.value as 'return' | 'volatility' | 'sharpe')
                          }
                        >
                          <MenuItem value="return">收益率</MenuItem>
                          <MenuItem value="volatility">波动率</MenuItem>
                          <MenuItem value="sharpe">夏普比率</MenuItem>
                        </Select>
                      </FormControl>
                    </Box>
                  </Box>
                }
              />
              <CardContent>
                <Box ref={heatmapChartRef} sx={{ height: 400, width: '100%' }} />
              </CardContent>
            </Card>
          )}

          {activeTab === 'seasonal' && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* 季节性图表 */}
              <Card>
                <CardHeader title="月度季节性表现" />
                <CardContent>
                  <Box ref={seasonalChartRef} sx={{ height: 400, width: '100%' }} />
                </CardContent>
              </Card>

              {/* 季度统计 */}
              {seasonalAnalysis && (
                <Card>
                  <CardHeader title="季度表现统计" />
                  <CardContent>
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                        gap: 2,
                        mb: 3,
                      }}
                    >
                      {Object.entries(seasonalAnalysis.quarterly_performance).map(
                        ([quarter, performance]) => (
                          <Box
                            key={quarter}
                            sx={{
                              textAlign: 'center',
                              p: { xs: 1.5, md: 2 },
                              border: 1,
                              borderColor: 'divider',
                              borderRadius: 1,
                            }}
                          >
                            <Typography variant="caption" color="text.secondary">
                              {quarter.toUpperCase()}
                            </Typography>
                            <Typography
                              sx={{
                                fontWeight: 600,
                                fontSize: { xs: '1.1rem', md: '1.5rem' },
                                overflow: 'hidden',
                                wordBreak: 'break-word',
                                color: performance >= 0 ? 'success.main' : 'error.main',
                              }}
                            >
                              {formatPercent(performance)}
                            </Typography>
                          </Box>
                        )
                      )}
                    </Box>

                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
                        gap: 2,
                      }}
                    >
                      <Box sx={{ p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: 500, color: 'success.dark', mb: 1 }}
                        >
                          最佳月份
                        </Typography>
                        <Box
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                          }}
                        >
                          <Typography variant="body2" sx={{ color: 'success.dark' }}>
                            {monthNames[seasonalAnalysis.best_month.month - 1]}
                          </Typography>
                          <Typography
                            sx={{
                              fontWeight: 600,
                              color: 'success.dark',
                              fontSize: { xs: '1rem', md: '1.25rem' },
                            }}
                          >
                            {formatPercent(seasonalAnalysis.best_month.avg_return)}
                          </Typography>
                        </Box>
                      </Box>

                      <Box sx={{ p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: 500, color: 'error.dark', mb: 1 }}
                        >
                          最差月份
                        </Typography>
                        <Box
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                          }}
                        >
                          <Typography variant="body2" sx={{ color: 'error.dark' }}>
                            {monthNames[seasonalAnalysis.worst_month.month - 1]}
                          </Typography>
                          <Typography
                            sx={{
                              fontWeight: 600,
                              color: 'error.dark',
                              fontSize: { xs: '1rem', md: '1.25rem' },
                            }}
                          >
                            {formatPercent(seasonalAnalysis.worst_month.avg_return)}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Box>
          )}

          {activeTab === 'yearly' && (
            <Card>
              <CardHeader title="年度绩效统计" />
              <CardContent sx={{ p: { xs: 0, md: 0 } }}>
                <TableContainer component={Paper} variant="outlined" sx={{ overflowX: 'auto' }}>
                  <Table
                    sx={{
                      '& .MuiTableCell-root': {
                        px: { xs: 1, md: 2 },
                        py: { xs: 0.75, md: 1 },
                        fontSize: { xs: '0.75rem', md: '0.875rem' },
                      },
                    }}
                  >
                    <TableHead>
                      <TableRow>
                        <TableCell>年份</TableCell>
                        <TableCell align="right">年化收益</TableCell>
                        <TableCell align="right">波动率</TableCell>
                        <TableCell align="right">夏普比率</TableCell>
                        <TableCell align="right">最大回撤</TableCell>
                        <TableCell align="right">卡玛比率</TableCell>
                        <TableCell align="right">胜率</TableCell>
                        <TableCell align="right">交易次数</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {yearlyPerformance.map(year => (
                        <TableRow key={year.year} hover>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {year.year}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              sx={{
                                fontFamily: 'monospace',
                                color: year.annual_return >= 0 ? 'success.main' : 'error.main',
                              }}
                            >
                              {formatPercent(year.annual_return)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {formatPercent(year.volatility)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {formatRatio(year.sharpe_ratio)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              sx={{ fontFamily: 'monospace', color: 'error.main' }}
                            >
                              {formatPercent(Math.abs(year.max_drawdown))}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {formatRatio(year.calmar_ratio)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {formatPercent(year.win_rate)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {year.total_trades}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {activeTab === 'benchmark' && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* 基准对比图表 */}
              <Card>
                <CardHeader title="策略 vs 基准表现" />
                <CardContent>
                  <Box ref={benchmarkChartRef} sx={{ height: 400, width: '100%' }} />
                </CardContent>
              </Card>

              {/* 基准对比统计 */}
              {benchmarkComparison && (
                <Card>
                  <CardHeader title="相对基准统计" />
                  <CardContent>
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                        gap: 2,
                        mb: 2,
                      }}
                    >
                      <Box
                        sx={{
                          textAlign: 'center',
                          p: { xs: 1.5, md: 2 },
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          跟踪误差
                        </Typography>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            fontSize: { xs: '1.1rem', md: '1.5rem' },
                            overflow: 'hidden',
                            wordBreak: 'break-word',
                          }}
                        >
                          {formatPercent(benchmarkComparison.tracking_error)}
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          textAlign: 'center',
                          p: { xs: 1.5, md: 2 },
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          信息比率
                        </Typography>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            fontSize: { xs: '1.1rem', md: '1.5rem' },
                            overflow: 'hidden',
                            wordBreak: 'break-word',
                          }}
                        >
                          {formatRatio(benchmarkComparison.information_ratio)}
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          textAlign: 'center',
                          p: { xs: 1.5, md: 2 },
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          Beta系数
                        </Typography>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            fontSize: { xs: '1.1rem', md: '1.5rem' },
                            overflow: 'hidden',
                            wordBreak: 'break-word',
                          }}
                        >
                          {formatRatio(benchmarkComparison.beta)}
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          textAlign: 'center',
                          p: { xs: 1.5, md: 2 },
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          Alpha系数
                        </Typography>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            fontSize: { xs: '1.1rem', md: '1.5rem' },
                            overflow: 'hidden',
                            wordBreak: 'break-word',
                            color: benchmarkComparison.alpha >= 0 ? 'success.main' : 'error.main',
                          }}
                        >
                          {formatPercent(benchmarkComparison.alpha)}
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ mt: 2, p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
                      <Box
                        sx={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <Typography variant="body2" sx={{ color: 'primary.dark' }}>
                          相关系数
                        </Typography>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            color: 'primary.dark',
                            fontSize: { xs: '1rem', md: '1.25rem' },
                          }}
                        >
                          {formatRatio(benchmarkComparison.correlation)}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
}
