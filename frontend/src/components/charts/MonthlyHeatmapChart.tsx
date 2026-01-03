/**
 * 月度收益热力图组件
 * 显示每月收益率的热力图，便于识别季节性模式
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import {
  Card,
  CardHeader,
  CardBody,
  Select,
  SelectItem,
  Chip,
  Tooltip,
  Button,
} from '@heroui/react';
import {
  Calendar,
  Info,
  TrendingUp,
  TrendingDown,
  Download,
} from 'lucide-react';

interface MonthlyHeatmapChartProps {
  taskId: string;
  data: {
    monthlyReturns: Array<{
      year: number;
      month: number;
      return: number;
      date: string;
    }>;
    years: number[];
    months: number[];
  };
  loading?: boolean;
  height?: number;
}

const MONTH_NAMES = [
  '1月', '2月', '3月', '4月', '5月', '6月',
  '7月', '8月', '9月', '10月', '11月', '12月'
];

export default function MonthlyHeatmapChart({
  taskId,
  data,
  loading = false,
  height = 400,
}: MonthlyHeatmapChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [selectedYear, setSelectedYear] = useState<string>('all');
  const [showStatistics, setShowStatistics] = useState(true);

  // 处理年份筛选
  const getFilteredData = () => {
    if (selectedYear === 'all') {
      return data.monthlyReturns;
    }
    return data.monthlyReturns.filter(item => item.year === parseInt(selectedYear));
  };

  // 计算月度统计
  const getMonthlyStatistics = () => {
    const filteredData = getFilteredData();
    const monthlyStats = Array.from({ length: 12 }, (_, index) => {
      const month = index + 1;
      const monthData = filteredData.filter(item => item.month === month);
      
      if (monthData.length === 0) {
        return {
          month,
          avgReturn: 0,
          count: 0,
          positiveCount: 0,
          negativeCount: 0,
          maxReturn: 0,
          minReturn: 0,
        };
      }

      const returns = monthData.map(item => item.return);
      const positiveReturns = returns.filter(r => r > 0);
      const negativeReturns = returns.filter(r => r < 0);

      return {
        month,
        avgReturn: returns.reduce((sum, r) => sum + r, 0) / returns.length,
        count: returns.length,
        positiveCount: positiveReturns.length,
        negativeCount: negativeReturns.length,
        maxReturn: Math.max(...returns),
        minReturn: Math.min(...returns),
      };
    });

    return monthlyStats;
  };

  // 初始化和更新图表
  useEffect(() => {
    if (!chartRef.current || loading || !data.monthlyReturns.length) return;

    // 销毁现有图表实例
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    // 创建新的图表实例
    chartInstance.current = echarts.init(chartRef.current);

    const filteredData = getFilteredData();

    // 准备热力图数据
    const heatmapData: any[] = [];
    const years = selectedYear === 'all' ? data.years : [parseInt(selectedYear)];
    
    years.forEach((year, yearIndex) => {
      for (let month = 1; month <= 12; month++) {
        const monthData = filteredData.find(item => 
          item.year === year && item.month === month
        );
        
        if (monthData) {
          heatmapData.push([
            month - 1, // x轴：月份（0-11）
            yearIndex,  // y轴：年份索引
            monthData.return, // 值：收益率
            monthData.date, // 额外信息：日期
          ]);
        }
      }
    });

    // 计算颜色范围
    const returns = heatmapData.map(item => item[2]);
    const maxReturn = Math.max(...returns);
    const minReturn = Math.min(...returns);
    const maxAbs = Math.max(Math.abs(maxReturn), Math.abs(minReturn));

    const option = {
      title: {
        text: selectedYear === 'all' ? '月度收益热力图' : `${selectedYear}年月度收益热力图`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        position: 'top',
        formatter: function (params: any) {
          const [monthIndex, yearIndex, returnValue, date] = params.data;
          const month = MONTH_NAMES[monthIndex];
          const year = years[yearIndex];
          
          return `
            <div style="margin-bottom: 4px;">${year}年${month}</div>
            <div style="color: ${returnValue >= 0 ? '#10b981' : '#ef4444'};">
              收益率: ${returnValue.toFixed(2)}%
            </div>
          `;
        },
      },
      grid: {
        height: '60%',
        top: '15%',
        left: '10%',
        right: '10%',
      },
      xAxis: {
        type: 'category',
        data: MONTH_NAMES,
        splitArea: {
          show: true,
        },
      },
      yAxis: {
        type: 'category',
        data: years.map(year => `${year}年`),
        splitArea: {
          show: true,
        },
      },
      visualMap: {
        min: -maxAbs,
        max: maxAbs,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: ['#ef4444', '#ffffff', '#10b981'], // 红-白-绿
        },
        text: ['高收益', '低收益'],
        textStyle: {
          color: '#333',
        },
      },
      series: [
        {
          name: '月度收益率',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: function (params: any) {
              return `${params.data[2].toFixed(1)}%`;
            },
            fontSize: 10,
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
  }, [data, selectedYear, loading]);

  // 导出图表
  const handleExportChart = () => {
    if (chartInstance.current) {
      const url = chartInstance.current.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#fff',
      });
      
      const link = document.createElement('a');
      link.download = `monthly-heatmap-${taskId}-${selectedYear}.png`;
      link.href = url;
      link.click();
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

  const monthlyStats = getMonthlyStatistics();
  const bestMonth = monthlyStats.reduce((best, current) => 
    current.avgReturn > best.avgReturn ? current : best
  );
  const worstMonth = monthlyStats.reduce((worst, current) => 
    current.avgReturn < worst.avgReturn ? current : worst
  );

  return (
    <Card>
      <CardHeader className="flex flex-col space-y-4">
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center space-x-2">
            <Calendar className="w-5 h-5 text-primary" />
            <h3 className="text-lg font-semibold">月度收益热力图</h3>
            <Tooltip content="显示每月收益率的分布，帮助识别季节性模式">
              <Info className="w-4 h-4 text-default-400 cursor-help" />
            </Tooltip>
          </div>

          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              variant="flat"
              startContent={<Download className="w-4 h-4" />}
              onPress={handleExportChart}
            >
              导出图表
            </Button>
          </div>
        </div>

        {/* 年份选择和统计开关 */}
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center space-x-4">
            <Select
              size="sm"
              placeholder="选择年份"
              selectedKeys={[selectedYear]}
              onSelectionChange={(keys) => {
                const selected = Array.from(keys)[0] as string;
                setSelectedYear(selected);
              }}
              className="w-32"
              items={[{ key: 'all', label: '全部年份' }, ...data.years.map(year => ({ key: year.toString(), label: `${year}年` }))]}
            >
              {(item) => (
                <SelectItem key={item.key}>
                  {item.label}
                </SelectItem>
              )}
            </Select>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="showStatistics"
              checked={showStatistics}
              onChange={(e) => setShowStatistics(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="showStatistics" className="text-sm text-default-600">
              显示月度统计
            </label>
          </div>
        </div>

        {/* 月度统计信息 */}
        {showStatistics && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between p-3 bg-success-50 rounded-lg">
              <div>
                <p className="text-sm text-default-500">表现最佳月份</p>
                <div className="flex items-center space-x-2">
                  <p className="text-lg font-bold text-success">
                    {MONTH_NAMES[bestMonth.month - 1]}
                  </p>
                  <Chip color="success" variant="flat" size="sm">
                    +{bestMonth.avgReturn.toFixed(2)}%
                  </Chip>
                </div>
              </div>
              <TrendingUp className="w-6 h-6 text-success" />
            </div>

            <div className="flex items-center justify-between p-3 bg-danger-50 rounded-lg">
              <div>
                <p className="text-sm text-default-500">表现最差月份</p>
                <div className="flex items-center space-x-2">
                  <p className="text-lg font-bold text-danger">
                    {MONTH_NAMES[worstMonth.month - 1]}
                  </p>
                  <Chip color="danger" variant="flat" size="sm">
                    {worstMonth.avgReturn.toFixed(2)}%
                  </Chip>
                </div>
              </div>
              <TrendingDown className="w-6 h-6 text-danger" />
            </div>
          </div>
        )}
      </CardHeader>

      <CardBody>
        <div
          ref={chartRef}
          style={{ height, width: '100%' }}
          className="min-h-[400px]"
        />

        {/* 月度详细统计表格 */}
        {showStatistics && (
          <div className="mt-6">
            <h4 className="text-md font-semibold mb-3">月度统计详情</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">月份</th>
                    <th className="text-right p-2">平均收益率</th>
                    <th className="text-right p-2">数据点数</th>
                    <th className="text-right p-2">正收益次数</th>
                    <th className="text-right p-2">负收益次数</th>
                    <th className="text-right p-2">最高收益率</th>
                    <th className="text-right p-2">最低收益率</th>
                  </tr>
                </thead>
                <tbody>
                  {monthlyStats.map((stat) => (
                    <tr key={stat.month} className="border-b hover:bg-default-50">
                      <td className="p-2 font-medium">
                        {MONTH_NAMES[stat.month - 1]}
                      </td>
                      <td className={`p-2 text-right font-medium ${
                        stat.avgReturn >= 0 ? 'text-success' : 'text-danger'
                      }`}>
                        {stat.avgReturn.toFixed(2)}%
                      </td>
                      <td className="p-2 text-right">{stat.count}</td>
                      <td className="p-2 text-right text-success">{stat.positiveCount}</td>
                      <td className="p-2 text-right text-danger">{stat.negativeCount}</td>
                      <td className="p-2 text-right text-success">
                        {stat.maxReturn.toFixed(2)}%
                      </td>
                      <td className="p-2 text-right text-danger">
                        {stat.minReturn.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  );
}