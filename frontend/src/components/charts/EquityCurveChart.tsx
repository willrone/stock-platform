/**
 * 收益曲线图表组件
 * 支持缩放、时间范围选择和基准对比
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  ButtonGroup,
  Switch,
  Tooltip,
  Select,
  SelectItem,
} from '@heroui/react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  TrendingUp,
  Info,
  Calendar,
} from 'lucide-react';

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

  // 处理时间范围筛选
  const getFilteredData = () => {
    if (!data.dates.length || selectedTimeRange === 'ALL') {
      return data;
    }

    const timeRange = TIME_RANGES.find(r => r.value === selectedTimeRange);
    if (!timeRange || timeRange.days === -1) {
      return data;
    }

    const endIndex = data.dates.length - 1;
    const startIndex = Math.max(0, endIndex - timeRange.days);

    return {
      dates: data.dates.slice(startIndex),
      portfolioValues: data.portfolioValues.slice(startIndex),
      returns: data.returns.slice(startIndex),
      dailyReturns: data.dailyReturns.slice(startIndex),
    };
  };

  // 处理基准数据筛选
  const getFilteredBenchmarkData = () => {
    if (!benchmarkData || !benchmarkData.dates.length || selectedTimeRange === 'ALL') {
      return benchmarkData;
    }

    const timeRange = TIME_RANGES.find(r => r.value === selectedTimeRange);
    if (!timeRange || timeRange.days === -1) {
      return benchmarkData;
    }

    const endIndex = benchmarkData.dates.length - 1;
    const startIndex = Math.max(0, endIndex - timeRange.days);

    return {
      dates: benchmarkData.dates.slice(startIndex),
      values: benchmarkData.values.slice(startIndex),
      returns: benchmarkData.returns.slice(startIndex),
    };
  };

  // 初始化和更新图表
  useEffect(() => {
    if (!chartRef.current || loading) return;

    // 销毁现有图表实例
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    // 创建新的图表实例
    chartInstance.current = echarts.init(chartRef.current);

    const filteredData = getFilteredData();
    const filteredBenchmarkData = getFilteredBenchmarkData();

    // 准备图表数据
    const series: any[] = [];

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
        formatter: function (params: any) {
          const date = params[0].axisValue;
          let content = `<div style="margin-bottom: 4px;">${date}</div>`;
          
          params.forEach((param: any) => {
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
        bottom: '3%',
        top: '15%',
        containLabel: true,
      },
      toolbox: {
        feature: {
          dataZoom: {
            yAxisIndex: 'none',
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
            return new Date(value).toLocaleDateString('zh-CN', {
              month: 'short',
              day: 'numeric',
            });
          },
        },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          formatter: function (value: number) {
            if (chartType === 'value') {
              return `¥${(value / 1000).toFixed(0)}K`;
            } else {
              return `${value.toFixed(1)}%`;
            }
          },
        },
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          start: 0,
          end: 100,
          height: 30,
          bottom: 20,
        },
      ],
      series: series,
    };

    chartInstance.current.setOption(option);

    // 响应式调整
    const handleResize = () => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
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
        start: 25,
        end: 75,
      });
    }
  };

  const handleZoomOut = () => {
    if (chartInstance.current) {
      chartInstance.current.dispatchAction({
        type: 'dataZoom',
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
        <CardBody>
          <div className="flex items-center justify-center" style={{ height }}>
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-col space-y-4">
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-semibold">收益曲线分析</h3>
            <Tooltip content="显示组合价值或收益率随时间的变化">
              <Info className="w-4 h-4 text-default-400 cursor-help" />
            </Tooltip>
          </div>

          {/* 图表控制按钮 */}
          <div className="flex items-center space-x-2">
            <ButtonGroup size="sm" variant="flat">
              <Button
                onPress={handleZoomIn}
                startContent={<ZoomIn className="w-4 h-4" />}
              >
                放大
              </Button>
              <Button
                onPress={handleZoomOut}
                startContent={<ZoomOut className="w-4 h-4" />}
              >
                缩小
              </Button>
              <Button
                onPress={handleReset}
                startContent={<RotateCcw className="w-4 h-4" />}
              >
                重置
              </Button>
            </ButtonGroup>
          </div>
        </div>

        {/* 图表选项 */}
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center space-x-4">
            {/* 时间范围选择 */}
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-default-500" />
              <Select
                size="sm"
                placeholder="选择时间范围"
                selectedKeys={[selectedTimeRange]}
                onSelectionChange={(keys) => {
                  const selected = Array.from(keys)[0] as string;
                  setSelectedTimeRange(selected);
                }}
                className="w-32"
                items={TIME_RANGES}
              >
                {(range) => (
                  <SelectItem key={range.value}>
                    {range.label}
                  </SelectItem>
                )}
              </Select>
            </div>

            {/* 图表类型切换 */}
            <ButtonGroup size="sm" variant="flat">
              <Button
                color={chartType === 'value' ? 'primary' : 'default'}
                onPress={() => setChartType('value')}
              >
                权益曲线
              </Button>
              <Button
                color={chartType === 'return' ? 'primary' : 'default'}
                onPress={() => setChartType('return')}
              >
                收益率
              </Button>
            </ButtonGroup>
          </div>

          {/* 基准对比开关 */}
          {benchmarkData && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-default-500">基准对比</span>
              <Switch
                size="sm"
                isSelected={showBenchmark}
                onValueChange={setShowBenchmark}
              />
            </div>
          )}
        </div>
      </CardHeader>

      <CardBody>
        <div
          ref={chartRef}
          style={{ height, width: '100%' }}
          className="min-h-[400px]"
        />
      </CardBody>
    </Card>
  );
}