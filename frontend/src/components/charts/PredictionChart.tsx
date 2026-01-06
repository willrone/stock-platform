/**
 * 预测分析图表组件
 * 使用ECharts显示预测结果、置信区间和技术指标
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { Card, CardBody, Chip, Spinner } from '@heroui/react';
import { TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';
import { PredictionResult, TaskService } from '../../services/taskService';

interface PredictionChartProps {
  taskId: string;
  stockCode: string;
  prediction?: PredictionResult;
}

export default function PredictionChart({ taskId, stockCode, prediction }: PredictionChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [loading, setLoading] = useState(true);
  const [seriesData, setSeriesData] = useState<Array<{date: string; actual: number; predicted: number}>>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  // 获取预测序列数据
  useEffect(() => {
    const fetchPredictionSeries = async () => {
      if (!stockCode || !taskId) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setLoadError(null);
        const response = await TaskService.getPredictionSeries(taskId, stockCode);
        setSeriesData(response.series || []);
      } catch (error: any) {
        console.error('获取预测序列失败:', error);
        setSeriesData([]);
        setLoadError(error.message || '获取预测序列失败');
      } finally {
        setLoading(false);
      }
    };

    fetchPredictionSeries();
  }, [stockCode, taskId]);

  useEffect(() => {
    if (!chartRef.current || loading) return;

    // 初始化图表
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }
    
    chartInstance.current = echarts.init(chartRef.current);
    const chartDates = seriesData.map(item => item.date);
    const actualSeries = seriesData.map(item => item.actual);
    const predictedSeries = seriesData.map(item => item.predicted);

    // 图表配置
    const option = {
      title: {
        text: `${stockCode} 预测分析`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        },
        formatter: function(params: any) {
          let result = `${params[0].axisValue}<br/>`;
          params.forEach((param: any) => {
            const color = param.color;
            const seriesName = param.seriesName;
            const value = param.value;
            result += `<span style="color:${color}">●</span> ${seriesName}: ¥${value}<br/>`;
          });
          return result;
        }
      },
      legend: {
        data: ['实际价格', '预测价格'],
        top: 30
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: chartDates,
        axisLine: {
          lineStyle: {
            color: '#ccc'
          }
        },
        axisLabel: {
          formatter: function(value: string) {
            return new Date(value).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
          }
        }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: {
          lineStyle: {
            color: '#ccc'
          }
        },
        splitLine: {
          lineStyle: {
            color: '#f0f0f0'
          }
        },
        axisLabel: {
          formatter: '¥{value}'
        }
      },
      series: [
        {
          name: '实际价格',
          type: 'line',
          data: actualSeries,
          lineStyle: {
            color: '#3b82f6',
            width: 2
          },
          itemStyle: {
            color: '#3b82f6'
          },
          symbol: 'circle',
          symbolSize: 4
        },
        {
          name: '预测价格',
          type: 'line',
          data: predictedSeries,
          lineStyle: {
            color: '#10b981',
            width: 3,
            type: 'dashed'
          },
          itemStyle: {
            color: '#10b981'
          },
          symbol: 'diamond',
          symbolSize: 6
        }
      ]
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
  }, [stockCode, seriesData, loading]);

  if (!prediction) {
    return (
      <Card>
        <CardBody>
          <div className="flex items-center justify-center h-64 text-default-500">
            <AlertCircle className="w-8 h-8 mr-2" />
            <span>暂无预测数据</span>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* 预测摘要 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-2">
              {prediction.predicted_direction > 0 ? (
                <TrendingUp className="w-6 h-6 text-success" />
              ) : prediction.predicted_direction < 0 ? (
                <TrendingDown className="w-6 h-6 text-danger" />
              ) : (
                <Target className="w-6 h-6 text-warning" />
              )}
            </div>
            <p className="text-sm text-default-500">预测方向</p>
            <p className="font-semibold">
              {prediction.predicted_direction > 0 ? '上涨' : 
               prediction.predicted_direction < 0 ? '下跌' : '持平'}
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="text-center">
            <p className="text-2xl font-bold text-primary">
              {(prediction.predicted_return * 100).toFixed(2)}%
            </p>
            <p className="text-sm text-default-500">预测收益率</p>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="text-center">
            <p className="text-2xl font-bold text-secondary">
              {(prediction.confidence_score * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-default-500">置信度</p>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="text-center">
            <p className="text-2xl font-bold text-danger">
              {prediction.risk_assessment?.value_at_risk 
                ? (prediction.risk_assessment.value_at_risk * 100).toFixed(2) 
                : '--'}%
            </p>
            <p className="text-sm text-default-500">风险价值(VaR)</p>
          </CardBody>
        </Card>
      </div>

      {/* 预测图表 */}
      <Card>
        <CardBody>
          {loading ? (
            <div className="flex items-center justify-center h-96">
              <Spinner size="lg" />
            </div>
          ) : loadError ? (
            <div className="flex items-center justify-center h-96 text-default-500">
              <AlertCircle className="w-8 h-8 mr-2" />
              <span>{loadError}</span>
            </div>
          ) : (
            <div
              ref={chartRef}
              style={{ height: '400px', width: '100%' }}
            />
          )}
        </CardBody>
      </Card>

      {/* 技术指标摘要 */}
      <Card>
        <CardBody>
          <h4 className="text-lg font-semibold mb-4">技术指标分析</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {prediction.technical_indicators && Object.entries(prediction.technical_indicators).map(([key, value]) => (
              <div key={key} className="text-center">
                <p className="text-sm text-default-500 capitalize">{key.replace('_', ' ')}</p>
                <Chip variant="flat" size="sm">
                  {typeof value === 'number' ? value.toFixed(2) : String(value)}
                </Chip>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>

      {/* 风险评估详情 */}
      <Card>
        <CardBody>
          <h4 className="text-lg font-semibold mb-4">风险评估</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-default-600">置信区间</span>
              <span className="font-medium">
                [{(prediction.confidence_interval.lower * 100).toFixed(2)}%, {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-default-600">风险价值 (VaR)</span>
              <span className="font-medium text-danger">
                {prediction.risk_assessment?.value_at_risk 
                  ? (prediction.risk_assessment.value_at_risk * 100).toFixed(2) 
                  : '--'}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-default-600">最大回撤</span>
              <span className="font-medium text-warning">
                {prediction.risk_assessment?.max_drawdown 
                  ? (prediction.risk_assessment.max_drawdown * 100).toFixed(2) 
                  : '--'}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-default-600">夏普比率</span>
              <span className="font-medium text-success">
                {prediction.risk_assessment?.sharpe_ratio 
                  ? prediction.risk_assessment.sharpe_ratio.toFixed(3) 
                  : '--'}
              </span>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
