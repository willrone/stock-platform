/**
 * 收益曲线图表组件
 * 支持缩放、时间范围选择和基准对比
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  ButtonGroup,
  Switch,
  Tooltip,
  Select,
  MenuItem,
  Box,
  Typography,
  FormControl,
  InputLabel,
  CircularProgress,
  IconButton,
} from '@mui/material';
import { ZoomIn, ZoomOut, RotateCcw, TrendingUp, Info, Calendar } from 'lucide-react';

interface EquityCurveChartProps {
  taskId: string;
  data: {
    dates: string[];
    portfolioValues: number[];
    returns: number[];
    dailyReturns: number[];
  };
  benchmarkData?: {
    dates: string[];
    values: number[];
    returns: number[];
  };
  loading?: boolean;
  height?: number;
}

interface TimeRange {
  label: string;
  value: string;
  days: number;
}

const TIME_RANGES: TimeRange[] = [
  { label: '1个月', value: '1M', days: 30 },
  { label: '3个月', value: '3M', days: 90 },
  { label: '6个月', value: '6M', days: 180 },
  { label: '1年', value: '1Y', days: 365 },
  { label: '全部', value: 'ALL', days: -1 },
];

export default function EquityCurveChart({
  taskId,
  data,
  benchmarkData,
  loading = false,
  height = 400,
}: EquityCurveChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState('ALL');
  const [chartType, setChartType] = useState<'value' | 'return'>('value');

  // Defensive: normalize possibly-empty data to avoid runtime errors in tests / partial API.
  const safeData = {
    dates: (data as Partial<EquityCurveChartProps['data']>)?.dates || [],
    portfolioValues: (data as Partial<EquityCurveChartProps['data']>)?.portfolioValues || [],
    returns: (data as Partial<EquityCurveChartProps['data']>)?.returns || [],
    dailyReturns: (data as Partial<EquityCurveChartProps['data']>)?.dailyReturns || [],
  };

  // 处理时间范围筛选
  const getFilteredData = () => {
    if (!safeData.dates.length || selectedTimeRange === 'ALL') {
      return safeData;
    }

    const timeRange = TIME_RANGES.find(r => r.value === selectedTimeRange);
    if (!timeRange || timeRange.days === -1) {
      return data;
    }

    const endIndex = safeData.dates.length - 1;
    const startIndex = Math.max(0, endIndex - timeRange.days);

    return {
      dates: safeData.dates.slice(startIndex),
      portfolioValues: safeData.portfolioValues.slice(startIndex),
      returns: safeData.returns.slice(startIndex),
      dailyReturns: safeData.dailyReturns.slice(startIndex),
    };
  };

  // 处理基准数据筛选
  const getFilteredBenchmarkData = () => {
    const safeBenchmark = benchmarkData
      ? {
          dates: (benchmarkData as Partial<NonNullable<EquityCurveChartProps['benchmarkData']>>)?.dates || [],
          values: (benchmarkData as Partial<NonNullable<EquityCurveChartProps['benchmarkData']>>)?.values || [],
          returns: (benchmarkData as Partial<NonNullable<EquityCurveChartProps['benchmarkData']>>)?.returns || [],
        }
      : undefined;

    if (!safeBenchmark || !safeBenchmark.dates.length || selectedTimeRange === 'ALL') {
      return safeBenchmark;
    }

    const timeRange = TIME_RANGES.find(r => r.value === selectedTimeRange);
    if (!timeRange || timeRange.days === -1) {
      return benchmarkData;
    }

    const endIndex = safeBenchmark.dates.length - 1;
    const startIndex = Math.max(0, endIndex - timeRange.days);

    return {
      dates: safeBenchmark.dates.slice(startIndex),
      values: safeBenchmark.values.slice(startIndex),
      returns: safeBenchmark.returns.slice(startIndex),
    };
  };

  // 初始化和更新图表
  useEffect(() => {
    if (!chartRef.current || loading) {
      return;
    }

    // 销毁现有图表实例
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    // 创建新的图表实例
    chartInstance.current = echarts.init(chartRef.current);

    const filteredData = getFilteredData();
    const filteredBenchmarkData = getFilteredBenchmarkData();

    // 准备图表数据
    const series: Record<string, unknown>[] = [];

    if (chartType === 'value') {
      // 权益曲线
      series.push({
        name: '组合价值',
        type: 'line',
        data: filteredData.portfolioValues,
        lineStyle: {
          color: '#10b981',
          width: 2,
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(16, 185, 129, 0.3)' },
            { offset: 1, color: 'rgba(16, 185, 129, 0.1)' },
          ]),
        },
        symbol: 'none',
        smooth: true,
      });

      // 基准曲线
      if (showBenchmark && filteredBenchmarkData) {
        series.push({
          name: '基准指数',
          type: 'line',
          data: filteredBenchmarkData.values,
          lineStyle: {
            color: '#f59e0b',
            width: 2,
            type: 'dashed',
          },
          symbol: 'none',
          smooth: true,
        });
      }
    } else {
      // 收益率曲线
      series.push({
        name: '组合收益率',
        type: 'line',
        data: filteredData.returns.map(r => r * 100), // 转换为百分比
        lineStyle: {
          color: '#10b981',
          width: 2,
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(16, 185, 129, 0.3)' },
            { offset: 1, color: 'rgba(16, 185, 129, 0.1)' },
          ]),
        },
        symbol: 'none',
        smooth: true,
      });

      // 基准收益率
      if (showBenchmark && filteredBenchmarkData) {
        series.push({
          name: '基准收益率',
          type: 'line',
          data: filteredBenchmarkData.returns.map(r => r * 100),
          lineStyle: {
            color: '#f59e0b',
            width: 2,
            type: 'dashed',
          },
          symbol: 'none',
          smooth: true,
        });
      }
    }

    const option = {
      title: {
        text: chartType === 'value' ? '权益曲线' : '收益率曲线',
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
          label: {
            backgroundColor: '#6a7985',
          },
        },
        formatter: function (params: { axisValue: string; value: number; color: string; seriesName: string }[]) {
          const date = params[0].axisValue;
          let content = `<div style="margin-bottom: 4px;">${date}</div>`;

          params.forEach((param: { axisValue: string; value: number; color: string; seriesName: string }) => {
            const value = param.value;
            const color = param.color;

            if (chartType === 'value') {
              content += `<div style="color: ${color};">
                ${param.seriesName}: ¥${value.toLocaleString()}
              </div>`;
            } else {
              content += `<div style="color: ${color};">
                ${param.seriesName}: ${value.toFixed(2)}%
              </div>`;
            }
          });

          return content;
        },
      },
      legend: {
        data: series.map(s => s.name),
        top: 30,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '15%',
        containLabel: true,
      },
      toolbox: {
        feature: {
          dataZoom: {
            yAxisIndex: [0],
            xAxisIndex: [0],
          },
          restore: {},
          saveAsImage: {},
        },
        right: 20,
        top: 10,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: filteredData.dates,
        axisLabel: {
          formatter: function (value: string) {
            const date = new Date(value);
            return date.toLocaleDateString('zh-CN', {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
            });
          },
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        scale: false,
        min: (() => {
          if (chartType === 'value') {
            const allValues = [...filteredData.portfolioValues];
            if (showBenchmark && filteredBenchmarkData) {
              allValues.push(...filteredBenchmarkData.values);
            }
            if (allValues.length > 0) {
              const min = Math.min(...allValues);
              const max = Math.max(...allValues);
              const range = max - min;
              return min - range * 0.1;
            }
          } else {
            const allReturns = filteredData.returns.map(r => r * 100);
            if (showBenchmark && filteredBenchmarkData) {
              allReturns.push(...filteredBenchmarkData.returns.map(r => r * 100));
            }
            if (allReturns.length > 0) {
              const min = Math.min(...allReturns);
              const max = Math.max(...allReturns);
              const range = max - min;
              return min - Math.max(range * 0.1, 0.5);
            }
          }
          return undefined;
        })(),
        max: (() => {
          if (chartType === 'value') {
            const allValues = [...filteredData.portfolioValues];
            if (showBenchmark && filteredBenchmarkData) {
              allValues.push(...filteredBenchmarkData.values);
            }
            if (allValues.length > 0) {
              const min = Math.min(...allValues);
              const max = Math.max(...allValues);
              const range = max - min;
              return max + range * 0.1;
            }
          } else {
            const allReturns = filteredData.returns.map(r => r * 100);
            if (showBenchmark && filteredBenchmarkData) {
              allReturns.push(...filteredBenchmarkData.returns.map(r => r * 100));
            }
            if (allReturns.length > 0) {
              const min = Math.min(...allReturns);
              const max = Math.max(...allReturns);
              const range = max - min;
              return max + Math.max(range * 0.1, 0.5);
            }
          }
          return undefined;
        })(),
        axisLabel: {
          formatter: function (value: number) {
            if (chartType === 'value') {
              if (value >= 10000) {
                return `¥${(value / 10000).toFixed(1)}万`;
              } else {
                return `¥${(value / 1000).toFixed(0)}K`;
              }
            } else {
              return `${value.toFixed(2)}%`;
            }
          },
        },
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
          xAxisIndex: [0],
          yAxisIndex: [0],
        },
        {
          type: 'slider',
          start: 0,
          end: 100,
          height: 30,
          bottom: 20,
          xAxisIndex: [0],
        },
      ],
      series: series,
    };

    chartInstance.current.setOption(option);

    // 监听dataZoom事件，动态更新Y轴范围
    const handleDataZoom = () => {
      if (!chartInstance.current) {
        return;
      }

      if (!(chartInstance.current as echarts.ECharts & { getOption?: () => Record<string, unknown> })?.getOption) {
        return;
      }

      const option = (chartInstance.current as echarts.ECharts & { getOption: () => Record<string, unknown> }).getOption();
      const dataZoom = option.dataZoom as { xAxisIndex?: number | number[]; start?: number; end?: number }[];

      if (!dataZoom || dataZoom.length === 0) {
        return;
      }

      const xDataZoom = dataZoom.find((dz: { xAxisIndex?: number | number[]; start?: number; end?: number }) => {
        if (dz.xAxisIndex !== undefined) {
          return Array.isArray(dz.xAxisIndex) ? dz.xAxisIndex.includes(0) : dz.xAxisIndex === 0;
        }
        return false;
      });

      if (xDataZoom) {
        const startPercent = xDataZoom.start || 0;
        const endPercent = xDataZoom.end || 100;

        const dataLength = filteredData.dates.length;
        const startIndex = Math.max(0, Math.floor((startPercent / 100) * dataLength));
        const endIndex = Math.min(dataLength, Math.ceil((endPercent / 100) * dataLength));

        let visibleValues: number[] = [];

        if (chartType === 'value') {
          visibleValues = filteredData.portfolioValues.slice(startIndex, endIndex);
          if (showBenchmark && filteredBenchmarkData) {
            const visibleBenchmarkValues = filteredBenchmarkData.values.slice(startIndex, endIndex);
            visibleValues.push(...visibleBenchmarkValues);
          }
        } else {
          const visibleReturns = filteredData.returns
            .slice(startIndex, endIndex)
            .map((r: number) => r * 100);
          visibleValues = visibleReturns;
          if (showBenchmark && filteredBenchmarkData) {
            const visibleBenchmarkReturns = filteredBenchmarkData.returns
              .slice(startIndex, endIndex)
              .map((r: number) => r * 100);
            visibleValues.push(...visibleBenchmarkReturns);
          }
        }

        if (visibleValues.length > 0) {
          const minValue = Math.min(...visibleValues);
          const maxValue = Math.max(...visibleValues);
          const range = maxValue - minValue;

          let padding = range * 0.1;
          if (range < 0.01) {
            padding = Math.max(0.5, Math.abs(minValue) * 0.1);
          }

          chartInstance.current.setOption(
            {
              yAxis: {
                min: minValue - padding,
                max: maxValue + padding,
              },
            },
            false
          );
        }
      }
    };

    // ECharts instance in tests may be a partial mock.
    if ((chartInstance.current as echarts.ECharts & { on?: unknown })?.on) {
      (chartInstance.current as echarts.ECharts & { on: (event: string, handler: () => void) => void }).on('dataZoom', handleDataZoom);
    }

    setTimeout(() => {
      handleDataZoom();
    }, 100);

    const handleResize = () => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      if ((chartInstance.current as echarts.ECharts & { off?: unknown })?.off) {
        (chartInstance.current as echarts.ECharts & { off: (event: string, handler: () => void) => void }).off('dataZoom', handleDataZoom);
      }
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.dispose();
      }
    };
  }, [data, benchmarkData, showBenchmark, selectedTimeRange, chartType, loading]);

  // 图表操作函数
  const handleZoomIn = () => {
    if (chartInstance.current) {
      chartInstance.current.dispatchAction({
        type: 'dataZoom',
        xAxisIndex: [0],
        start: 25,
        end: 75,
      });
      chartInstance.current.dispatchAction({
        type: 'dataZoom',
        yAxisIndex: [0],
        start: 25,
        end: 75,
      });
    }
  };

  const handleZoomOut = () => {
    if (chartInstance.current) {
      chartInstance.current.dispatchAction({
        type: 'dataZoom',
        xAxisIndex: [0],
        start: 0,
        end: 100,
      });
      chartInstance.current.dispatchAction({
        type: 'dataZoom',
        yAxisIndex: [0],
        start: 0,
        end: 100,
      });
    }
  };

  const handleReset = () => {
    if (chartInstance.current) {
      chartInstance.current.dispatchAction({
        type: 'restore',
      });
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height }}>
            <CircularProgress size={32} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box
              sx={{
                display: 'flex',
                flexDirection: { xs: 'column', sm: 'row' },
                alignItems: { xs: 'flex-start', sm: 'center' },
                justifyContent: 'space-between',
                width: '100%',
                gap: 1,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp size={20} color="#1976d2" />
                <Typography
                  variant="h6"
                  component="h3"
                  sx={{ fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' } }}
                >
                  收益曲线分析
                </Typography>
                <Tooltip title="显示组合价值或收益率随时间的变化">
                  <IconButton size="small">
                    <Info size={16} />
                  </IconButton>
                </Tooltip>
              </Box>

              {/* 图表控制按钮 */}
              <ButtonGroup size="small" variant="outlined">
                <Button onClick={handleZoomIn} startIcon={<ZoomIn size={16} />}>
                  放大
                </Button>
                <Button onClick={handleZoomOut} startIcon={<ZoomOut size={16} />}>
                  缩小
                </Button>
                <Button onClick={handleReset} startIcon={<RotateCcw size={16} />}>
                  重置
                </Button>
              </ButtonGroup>
            </Box>

            {/* 图表选项 */}
            <Box
              sx={{
                display: 'flex',
                flexDirection: { xs: 'column', sm: 'row' },
                alignItems: { xs: 'flex-start', sm: 'center' },
                justifyContent: 'space-between',
                width: '100%',
                gap: 1,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {/* 时间范围选择 */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Calendar size={16} color="#666" />
                  <FormControl size="small" sx={{ minWidth: 128 }}>
                    <InputLabel>选择时间范围</InputLabel>
                    <Select
                      value={selectedTimeRange}
                      label="选择时间范围"
                      onChange={e => setSelectedTimeRange(e.target.value)}
                    >
                      {TIME_RANGES.map(range => (
                        <MenuItem key={range.value} value={range.value}>
                          {range.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                {/* 图表类型切换 */}
                <ButtonGroup size="small" variant="outlined">
                  <Button
                    variant={chartType === 'value' ? 'contained' : 'outlined'}
                    onClick={() => setChartType('value')}
                  >
                    权益曲线
                  </Button>
                  <Button
                    variant={chartType === 'return' ? 'contained' : 'outlined'}
                    onClick={() => setChartType('return')}
                  >
                    收益率
                  </Button>
                </ButtonGroup>
              </Box>

              {/* 基准对比开关 */}
              {benchmarkData && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    基准对比
                  </Typography>
                  <Switch
                    size="small"
                    checked={showBenchmark}
                    onChange={e => setShowBenchmark(e.target.checked)}
                  />
                </Box>
              )}
            </Box>
          </Box>
        }
      />

      <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
        <Box sx={{ overflowX: 'auto' }}>
          <Box
            ref={chartRef}
            sx={{ height, width: '100%', minHeight: { xs: 300, sm: 400 }, minWidth: 400 }}
          />
        </Box>
      </CardContent>
    </Card>
  );
}
