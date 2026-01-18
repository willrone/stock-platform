/**
 * 预测分析图表组件
 * 使用ECharts显示预测结果、置信区间和技术指标
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { Card, CardContent, Chip, CircularProgress, Box, Typography } from '@mui/material';
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
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}>
            <AlertCircle size={32} style={{ marginRight: 8 }} />
            <Typography variant="body2" color="text.secondary">暂无预测数据</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 预测摘要 */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2 }}>
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
              {prediction.predicted_direction > 0 ? (
                <TrendingUp size={24} color="#2e7d32" />
              ) : prediction.predicted_direction < 0 ? (
                <TrendingDown size={24} color="#d32f2f" />
              ) : (
                <Target size={24} color="#ed6c02" />
              )}
            </Box>
            <Typography variant="caption" color="text.secondary">预测方向</Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {prediction.predicted_direction > 0 ? '上涨' : 
               prediction.predicted_direction < 0 ? '下跌' : '持平'}
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
              {(prediction.predicted_return * 100).toFixed(2)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">预测收益率</Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ fontWeight: 600, color: 'secondary.main' }}>
              {(prediction.confidence_score * 100).toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">置信度</Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
              {prediction.risk_assessment?.value_at_risk 
                ? (prediction.risk_assessment.value_at_risk * 100).toFixed(2) 
                : '--'}%
            </Typography>
            <Typography variant="caption" color="text.secondary">风险价值(VaR)</Typography>
          </CardContent>
        </Card>
      </Box>

      {/* 预测图表 */}
      <Card>
        <CardContent>
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 384 }}>
              <CircularProgress size={48} />
            </Box>
          ) : loadError ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 384 }}>
              <AlertCircle size={32} style={{ marginRight: 8 }} />
              <Typography variant="body2" color="text.secondary">{loadError}</Typography>
            </Box>
          ) : (
            <Box
              ref={chartRef}
              sx={{ height: 400, width: '100%' }}
            />
          )}
        </CardContent>
      </Card>

      {/* 技术指标摘要 */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2 }}>
            技术指标分析
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
            {prediction.technical_indicators && Object.entries(prediction.technical_indicators).map(([key, value]) => (
              <Box key={key} sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                  {key.replace('_', ' ')}
                </Typography>
                <Chip label={typeof value === 'number' ? value.toFixed(2) : String(value)} size="small" sx={{ mt: 0.5 }} />
              </Box>
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* 风险评估详情 */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2 }}>
            风险评估
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">置信区间</Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                [{(prediction.confidence_interval.lower * 100).toFixed(2)}%, {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">风险价值 (VaR)</Typography>
              <Typography variant="body2" sx={{ fontWeight: 500, color: 'error.main' }}>
                {prediction.risk_assessment?.value_at_risk 
                  ? (prediction.risk_assessment.value_at_risk * 100).toFixed(2) 
                  : '--'}%
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">最大回撤</Typography>
              <Typography variant="body2" sx={{ fontWeight: 500, color: 'warning.main' }}>
                {prediction.risk_assessment?.max_drawdown 
                  ? (prediction.risk_assessment.max_drawdown * 100).toFixed(2) 
                  : '--'}%
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">夏普比率</Typography>
              <Typography variant="body2" sx={{ fontWeight: 500, color: 'success.main' }}>
                {prediction.risk_assessment?.sharpe_ratio 
                  ? prediction.risk_assessment.sharpe_ratio.toFixed(3) 
                  : '--'}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
