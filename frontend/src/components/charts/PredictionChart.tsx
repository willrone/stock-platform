/**
 * 预测分析图表组件
 * 使用ECharts显示预测结果、置信区间和技术指标
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { Card, CardBody, Chip, Spinner } from '@heroui/react';
import { TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';
import { PredictionResult } from '../../services/taskService';
import { DataService } from '../../services/dataService';

interface PredictionChartProps {
  stockCode: string;
  prediction?: PredictionResult;
}

export default function PredictionChart({ stockCode, prediction }: PredictionChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [loading, setLoading] = useState(true);
  const [historicalData, setHistoricalData] = useState<Array<{date: string; close: number}>>([]);

  // 获取历史价格数据
  useEffect(() => {
    const fetchHistoricalData = async () => {
      if (!stockCode) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        // 获取最近60天的历史数据用于显示
        const endDate = new Date();
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - 60);
        
        const response = await DataService.getStockData(
          stockCode,
          startDate.toISOString().split('T')[0],
          endDate.toISOString().split('T')[0]
        );
        
        // 解析数据
        const apiData: any = response.data;
        const dataArray = (apiData?.data && Array.isArray(apiData.data)) 
          ? apiData.data 
          : (Array.isArray(apiData) ? apiData : []);
        
        if (dataArray.length > 0) {
          const formatted = dataArray
            .map((item: any) => ({
              date: item.date ? item.date.split('T')[0] : item.date,
              close: Number(item.close) || 0,
            }))
            .filter((item: any) => item.date && item.close > 0)
            .sort((a: any, b: any) => a.date.localeCompare(b.date));
          
          setHistoricalData(formatted);
        }
      } catch (error) {
        console.error('获取历史价格数据失败:', error);
        setHistoricalData([]);
      } finally {
        setLoading(false);
      }
    };

    if (stockCode) {
      fetchHistoricalData();
    }
  }, [stockCode]);

  useEffect(() => {
    if (!chartRef.current || !prediction || loading) return;

    // 初始化图表
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }
    
    chartInstance.current = echarts.init(chartRef.current);

    // 生成图表数据（使用真实历史数据和预测数据）
    const generateChartData = () => {
      const predictionDays = 5; // 预测未来5天
      
      // 使用真实历史数据
      const historicalPrices = historicalData.map(item => item.close);
      const historicalDates = historicalData.map(item => item.date);
      
      // 如果没有历史数据，使用最近30天的模拟数据作为后备
      let dates = [...historicalDates];
      let historical = [...historicalPrices];
      
      if (historical.length === 0) {
        const basePrice = 100;
        for (let i = 30; i >= 0; i--) {
          const date = new Date();
          date.setDate(date.getDate() - i);
          dates.push(date.toISOString().split('T')[0]);
          historical.push(basePrice);
        }
      }
      
      // 获取当前价格（历史数据的最后一个）
      const currentPrice = historical.length > 0 ? historical[historical.length - 1] : 100;
      const splitIndex = historical.length - 1;
      
      // 预测数据
      const predictionData = [];
      const upperBound = [];
      const lowerBound = [];
      
      for (let i = 1; i <= predictionDays; i++) {
        const date = new Date();
        date.setDate(date.getDate() + i);
        dates.push(date.toISOString().split('T')[0]);
        
        // 基于预测收益率计算预测价格
        const predictedPrice = currentPrice * (1 + prediction.predicted_return * (i / predictionDays));
        predictionData.push(Number(predictedPrice.toFixed(2)));
        
        // 置信区间
        const confidence = prediction.confidence_interval;
        upperBound.push(Number((predictedPrice * (1 + confidence.upper)).toFixed(2)));
        lowerBound.push(Number((predictedPrice * (1 + confidence.lower)).toFixed(2)));
      }
      
      return {
        dates,
        historical,
        prediction: predictionData,
        upperBound,
        lowerBound,
        splitIndex
      };
    };

    const chartData = generateChartData();

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
        data: ['历史价格', '预测价格', '置信区间上限', '置信区间下限'],
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
        data: chartData.dates,
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
          name: '历史价格',
          type: 'line',
          data: [...chartData.historical, ...new Array(chartData.prediction.length).fill(null)],
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
          data: [...new Array(chartData.splitIndex + 1).fill(null), ...chartData.prediction],
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
        },
        {
          name: '置信区间上限',
          type: 'line',
          data: [...new Array(chartData.splitIndex + 1).fill(null), ...chartData.upperBound],
          lineStyle: {
            color: '#f59e0b',
            width: 1,
            type: 'dotted'
          },
          itemStyle: {
            color: '#f59e0b'
          },
          symbol: 'none'
        },
        {
          name: '置信区间下限',
          type: 'line',
          data: [...new Array(chartData.splitIndex + 1).fill(null), ...chartData.lowerBound],
          lineStyle: {
            color: '#f59e0b',
            width: 1,
            type: 'dotted'
          },
          itemStyle: {
            color: '#f59e0b'
          },
          symbol: 'none',
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {
                offset: 0,
                color: 'rgba(245, 158, 11, 0.1)'
              },
              {
                offset: 1,
                color: 'rgba(245, 158, 11, 0.05)'
              }
            ])
          },
          stack: 'confidence'
        }
      ],
      markLine: {
        data: [
          {
            xAxis: chartData.splitIndex,
            lineStyle: {
              color: '#ef4444',
              type: 'solid',
              width: 2
            },
            label: {
              formatter: '预测起点',
              position: 'insideEndTop'
            }
          }
        ]
      }
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
  }, [stockCode, prediction, historicalData, loading]);

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