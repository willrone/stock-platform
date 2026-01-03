/**
 * 绩效分解组件
 * 实现月度和年度绩效分解、季节性分析
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Card,
  CardBody,
  CardHeader,
  Tabs,
  Tab,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Select,
  SelectItem,
  Button,
  Tooltip,
} from '@heroui/react';
import * as echarts from 'echarts';
import { 
  Calendar, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Activity,
  Target,
  Award,
  Clock
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
  benchmarkComparison
}: PerformanceBreakdownProps) {
  const [selectedYear, setSelectedYear] = useState<string>('all');
  const [selectedMetric, setSelectedMetric] = useState<'return' | 'volatility' | 'sharpe'>('return');
  
  // 图表引用
  const heatmapChartRef = useRef<HTMLDivElement>(null);
  const seasonalChartRef = useRef<HTMLDivElement>(null);
  const benchmarkChartRef = useRef<HTMLDivElement>(null);
  const heatmapChartInstance = useRef<echarts.ECharts | null>(null);
  const seasonalChartInstance = useRef<echarts.ECharts | null>(null);
  const benchmarkChartInstance = useRef<echarts.ECharts | null>(null);

  // 月份名称
  const monthNames = [
    '1月', '2月', '3月', '4月', '5月', '6月',
    '7月', '8月', '9月', '10月', '11月', '12月'
  ];

  // 季度名称
  const quarterNames = ['Q1', 'Q2', 'Q3', 'Q4'];

  // 可用年份
  const availableYears = useMemo(() => {
    const yearSet = new Set(monthlyPerformance.map(item => item.year));
    const years = Array.from(yearSet).sort();
    return years;
  }, [monthlyPerformance]);

  // 热力图数据
  const heatmapData = useMemo(() => {
    if (!monthlyPerformance.length) return [];
    
    const filteredData = selectedYear === 'all' 
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
    if (!yearlyPerformance.length) return null;
    
    const totalReturn = yearlyPerformance.reduce((sum, year) => sum + year.annual_return, 0);
    const avgReturn = totalReturn / yearlyPerformance.length;
    const avgVolatility = yearlyPerformance.reduce((sum, year) => sum + year.volatility, 0) / yearlyPerformance.length;
    const avgSharpe = yearlyPerformance.reduce((sum, year) => sum + year.sharpe_ratio, 0) / yearlyPerformance.length;
    
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
      totalYears: yearlyPerformance.length
    };
  }, [yearlyPerformance]);

  // 初始化热力图
  useEffect(() => {
    if (!heatmapChartRef.current || !heatmapData.length) return;

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
        formatter: function (params: any) {
          const month = monthNames[params.data[0]];
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
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        },
      },
      series: [
        {
          name: selectedMetric,
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: function (params: any) {
              return (params.data[2] * 100).toFixed(1);
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
    if (!seasonalChartRef.current || !seasonalAnalysis) return;

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
    if (!benchmarkChartRef.current || !benchmarkComparison) return;

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
        <CardBody className="flex items-center justify-center h-64">
          <div className="text-center text-gray-500">
            <Calendar className="w-12 h-12 mx-auto mb-2" />
            <p>暂无绩效分解数据</p>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 年度统计概览 */}
      {yearlyStats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-500" />
                <div>
                  <p className="text-sm text-gray-500">平均年化收益</p>
                  <p className="text-xl font-bold">{formatPercent(yearlyStats.avgReturn)}</p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-500" />
                <div>
                  <p className="text-sm text-gray-500">平均波动率</p>
                  <p className="text-xl font-bold">{formatPercent(yearlyStats.avgVolatility)}</p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-green-500" />
                <div>
                  <p className="text-sm text-gray-500">平均夏普比率</p>
                  <p className="text-xl font-bold">{formatRatio(yearlyStats.avgSharpe)}</p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-orange-500" />
                <div>
                  <p className="text-sm text-gray-500">回测年数</p>
                  <p className="text-xl font-bold">{yearlyStats.totalYears} 年</p>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      )}

      {/* 最佳和最差年份 */}
      {yearlyStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <h4 className="text-lg font-semibold text-green-600 flex items-center gap-2">
                <Award className="w-5 h-5" />
                最佳年份
              </h4>
            </CardHeader>
            <CardBody className="pt-0">
              <div className="flex justify-between items-center">
                <div>
                  <p className="text-2xl font-bold">{yearlyStats.bestYear.year}</p>
                  <p className="text-sm text-gray-500">年化收益率</p>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-green-600">
                    {formatPercent(yearlyStats.bestYear.annual_return)}
                  </p>
                  <p className="text-sm text-gray-500">
                    夏普: {formatRatio(yearlyStats.bestYear.sharpe_ratio)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <h4 className="text-lg font-semibold text-red-600 flex items-center gap-2">
                <TrendingDown className="w-5 h-5" />
                最差年份
              </h4>
            </CardHeader>
            <CardBody className="pt-0">
              <div className="flex justify-between items-center">
                <div>
                  <p className="text-2xl font-bold">{yearlyStats.worstYear.year}</p>
                  <p className="text-sm text-gray-500">年化收益率</p>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-red-600">
                    {formatPercent(yearlyStats.worstYear.annual_return)}
                  </p>
                  <p className="text-sm text-gray-500">
                    夏普: {formatRatio(yearlyStats.worstYear.sharpe_ratio)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      )}

      <Tabs defaultSelectedKey="heatmap" className="w-full">
        {/* 月度热力图 */}
        <Tab key="heatmap" title={
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            月度热力图
          </div>
        }>
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">月度表现热力图</h3>
                <div className="flex gap-2">
                  <select
                    value={selectedYear}
                    onChange={(e) => setSelectedYear(e.target.value)}
                    className="px-3 py-1 border rounded text-sm"
                  >
                    <option value="all">全部</option>
                    {availableYears.map(year => (
                      <option key={year} value={year.toString()}>
                        {year}
                      </option>
                    ))}
                  </select>
                  
                  <select
                    value={selectedMetric}
                    onChange={(e) => setSelectedMetric(e.target.value as 'return' | 'volatility' | 'sharpe')}
                    className="px-3 py-1 border rounded text-sm"
                  >
                    <option value="return">收益率</option>
                    <option value="volatility">波动率</option>
                    <option value="sharpe">夏普比率</option>
                  </select>
                </div>
              </div>
            </CardHeader>
            <CardBody>
              <div ref={heatmapChartRef} style={{ height: '400px', width: '100%' }} />
            </CardBody>
          </Card>
        </Tab>

        {/* 季节性分析 */}
        <Tab key="seasonal" title={
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            季节性分析
          </div>
        }>
          <div className="space-y-6">
            {/* 季节性图表 */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">月度季节性表现</h3>
              </CardHeader>
              <CardBody>
                <div ref={seasonalChartRef} style={{ height: '400px', width: '100%' }} />
              </CardBody>
            </Card>

            {/* 季度统计 */}
            {seasonalAnalysis && (
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">季度表现统计</h3>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(seasonalAnalysis.quarterly_performance).map(([quarter, performance]) => (
                      <div key={quarter} className="text-center p-4 border rounded-lg">
                        <p className="text-sm text-gray-500">{quarter.toUpperCase()}</p>
                        <p className={`text-xl font-bold ${performance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatPercent(performance)}
                        </p>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-green-50 rounded-lg">
                      <h4 className="font-medium text-green-800 mb-2">最佳月份</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-green-700">
                          {monthNames[seasonalAnalysis.best_month.month - 1]}
                        </span>
                        <span className="font-bold text-green-800">
                          {formatPercent(seasonalAnalysis.best_month.avg_return)}
                        </span>
                      </div>
                    </div>
                    
                    <div className="p-4 bg-red-50 rounded-lg">
                      <h4 className="font-medium text-red-800 mb-2">最差月份</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-red-700">
                          {monthNames[seasonalAnalysis.worst_month.month - 1]}
                        </span>
                        <span className="font-bold text-red-800">
                          {formatPercent(seasonalAnalysis.worst_month.avg_return)}
                        </span>
                      </div>
                    </div>
                  </div>
                </CardBody>
              </Card>
            )}
          </div>
        </Tab>

        {/* 年度表现表格 */}
        <Tab key="yearly" title={
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4" />
            年度表现
          </div>
        }>
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">年度绩效统计</h3>
            </CardHeader>
            <CardBody className="p-0">
              <Table aria-label="年度绩效表格">
                <TableHeader>
                  <TableColumn>年份</TableColumn>
                  <TableColumn>年化收益</TableColumn>
                  <TableColumn>波动率</TableColumn>
                  <TableColumn>夏普比率</TableColumn>
                  <TableColumn>最大回撤</TableColumn>
                  <TableColumn>卡玛比率</TableColumn>
                  <TableColumn>胜率</TableColumn>
                  <TableColumn>交易次数</TableColumn>
                </TableHeader>
                <TableBody>
                  {yearlyPerformance.map((year) => (
                    <TableRow key={year.year}>
                      <TableCell>
                        <span className="font-bold">{year.year}</span>
                      </TableCell>
                      <TableCell>
                        <span className={`font-mono ${year.annual_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatPercent(year.annual_return)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{formatPercent(year.volatility)}</span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{formatRatio(year.sharpe_ratio)}</span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono text-red-600">
                          {formatPercent(Math.abs(year.max_drawdown))}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{formatRatio(year.calmar_ratio)}</span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{formatPercent(year.win_rate)}</span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{year.total_trades}</span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardBody>
          </Card>
        </Tab>

        {/* 基准对比 */}
        <Tab key="benchmark" title={
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            基准对比
          </div>
        }>
          <div className="space-y-6">
            {/* 基准对比图表 */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">策略 vs 基准表现</h3>
              </CardHeader>
              <CardBody>
                <div ref={benchmarkChartRef} style={{ height: '400px', width: '100%' }} />
              </CardBody>
            </Card>

            {/* 基准对比统计 */}
            {benchmarkComparison && (
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">相对基准统计</h3>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-4 border rounded-lg">
                      <p className="text-sm text-gray-500">跟踪误差</p>
                      <p className="text-xl font-bold">{formatPercent(benchmarkComparison.tracking_error)}</p>
                    </div>
                    
                    <div className="text-center p-4 border rounded-lg">
                      <p className="text-sm text-gray-500">信息比率</p>
                      <p className="text-xl font-bold">{formatRatio(benchmarkComparison.information_ratio)}</p>
                    </div>
                    
                    <div className="text-center p-4 border rounded-lg">
                      <p className="text-sm text-gray-500">Beta系数</p>
                      <p className="text-xl font-bold">{formatRatio(benchmarkComparison.beta)}</p>
                    </div>
                    
                    <div className="text-center p-4 border rounded-lg">
                      <p className="text-sm text-gray-500">Alpha系数</p>
                      <p className={`text-xl font-bold ${benchmarkComparison.alpha >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatPercent(benchmarkComparison.alpha)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <div className="flex justify-between items-center">
                      <span className="text-blue-700">相关系数</span>
                      <span className="font-bold text-blue-800">
                        {formatRatio(benchmarkComparison.correlation)}
                      </span>
                    </div>
                  </div>
                </CardBody>
              </Card>
            )}
          </div>
        </Tab>
      </Tabs>
    </div>
  );
}