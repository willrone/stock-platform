/**
 * 回撤曲线图表组件
 * 显示回撤曲线并标注最大回撤期间
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import {
  Card,
  CardContent,
  CardHeader,
  Chip,
  Tooltip,
  Button,
  ButtonGroup,
  Box,
  Typography,
  Checkbox,
  FormControlLabel,
  CircularProgress,
  IconButton,
} from '@mui/material';
import {
  TrendingDown,
  Info,
  AlertTriangle,
  Calendar,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from 'lucide-react';

interface DrawdownChartProps {
  taskId: string;
  data: {
    dates: string[];
    drawdowns: number[];
    maxDrawdown: number;
    maxDrawdownDate: string;
    maxDrawdownDuration: number;
  };
  loading?: boolean;
  height?: number;
}

export default function DrawdownChart({
  taskId,
  data,
  loading = false,
  height = 400,
}: DrawdownChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [showMaxDrawdownPeriod, setShowMaxDrawdownPeriod] = useState(true);

  // 找到最大回撤期间的索引
  const getMaxDrawdownPeriod = () => {
    if (!data.dates.length || !data.maxDrawdownDate) {
      return { startIndex: -1, endIndex: -1 };
    }

    const maxDrawdownIndex = data.dates.findIndex(date => date === data.maxDrawdownDate);
    if (maxDrawdownIndex === -1) {
      return { startIndex: -1, endIndex: -1 };
    }

    // 向前找到回撤开始点（回撤为0的点）
    let startIndex = maxDrawdownIndex;
    for (let i = maxDrawdownIndex - 1; i >= 0; i--) {
      if (Math.abs(data.drawdowns[i]) < 0.001) { // 接近0
        startIndex = i;
        break;
      }
    }

    // 向后找到回撤结束点（回撤恢复到接近0的点）
    let endIndex = maxDrawdownIndex;
    for (let i = maxDrawdownIndex + 1; i < data.drawdowns.length; i++) {
      if (Math.abs(data.drawdowns[i]) < 0.001) { // 接近0
        endIndex = i;
        break;
      }
    }

    return { startIndex, endIndex };
  };

  // 初始化和更新图表
  useEffect(() => {
    if (!chartRef.current || loading || !data.dates.length) return;

    // 销毁现有图表实例
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    // 创建新的图表实例
    chartInstance.current = echarts.init(chartRef.current);

    const { startIndex, endIndex } = getMaxDrawdownPeriod();

    // 准备图表数据
    const series: any[] = [
      {
        name: '回撤',
        type: 'line',
        data: data.drawdowns,
        lineStyle: {
          color: '#ef4444',
          width: 2,
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(239, 68, 68, 0.3)' },
            { offset: 1, color: 'rgba(239, 68, 68, 0.1)' },
          ]),
        },
        symbol: 'none',
        smooth: true,
      },
    ];

    // 添加最大回撤期间的标注
    const markAreas: any[] = [];
    const markLines: any[] = [];

    if (showMaxDrawdownPeriod && startIndex !== -1 && endIndex !== -1) {
      // 标注最大回撤期间
      markAreas.push({
        name: '最大回撤期间',
        itemStyle: {
          color: 'rgba(239, 68, 68, 0.2)',
        },
        data: [
          [
            { xAxis: data.dates[startIndex] },
            { xAxis: data.dates[endIndex] },
          ],
        ],
      });

      // 标注最大回撤点
      markLines.push({
        name: '最大回撤点',
        data: [
          {
            xAxis: data.maxDrawdownDate,
            yAxis: data.maxDrawdown,
            label: {
              formatter: `最大回撤: ${Math.abs(data.maxDrawdown).toFixed(2)}%`,
              position: 'insideEndTop',
            },
            lineStyle: {
              color: '#dc2626',
              type: 'dashed',
              width: 2,
            },
          },
        ],
      });
    }

    series[0].markArea = { data: markAreas };
    series[0].markLine = { data: markLines };

    const option = {
      title: {
        text: '回撤曲线',
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
          const drawdown = params[0].value;
          
          return `
            <div style="margin-bottom: 4px;">${date}</div>
            <div style="color: #ef4444;">
              回撤: ${Math.abs(drawdown).toFixed(2)}%
            </div>
          `;
        },
      },
      legend: {
        data: ['回撤'],
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
        data: data.dates,
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
        max: 0,
        min: function (value: any) {
          return Math.floor(value.min * 1.1);
        },
        axisLabel: {
          formatter: function (value: number) {
            return `${Math.abs(value).toFixed(1)}%`;
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
  }, [data, showMaxDrawdownPeriod, loading]);

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

  // 获取回撤评级
  const getDrawdownRating = (maxDrawdown: number) => {
    const absDrawdown = Math.abs(maxDrawdown);
    if (absDrawdown <= 5) return { text: '优秀', color: 'success' as const };
    if (absDrawdown <= 10) return { text: '良好', color: 'primary' as const };
    if (absDrawdown <= 20) return { text: '一般', color: 'warning' as const };
    return { text: '较差', color: 'error' as const };
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

  const drawdownRating = getDrawdownRating(data.maxDrawdown);

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingDown size={20} color="#d32f2f" />
                <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                  回撤分析
                </Typography>
                <Tooltip title="显示组合价值从峰值下跌的幅度">
                  <IconButton size="small">
                    <Info size={16} />
                  </IconButton>
                </Tooltip>
              </Box>

              {/* 图表控制按钮 */}
              <ButtonGroup size="small" variant="outlined">
                <Button
                  onClick={handleZoomIn}
                  startIcon={<ZoomIn size={16} />}
                >
                  放大
                </Button>
                <Button
                  onClick={handleZoomOut}
                  startIcon={<ZoomOut size={16} />}
                >
                  缩小
                </Button>
                <Button
                  onClick={handleReset}
                  startIcon={<RotateCcw size={16} />}
                >
                  重置
                </Button>
              </ButtonGroup>
            </Box>

            {/* 回撤统计信息 */}
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1.5, bgcolor: 'error.light', borderRadius: 1 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    最大回撤
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                      {Math.abs(data.maxDrawdown).toFixed(2)}%
                    </Typography>
                    <Chip label={drawdownRating.text} color={drawdownRating.color} size="small" />
                  </Box>
                </Box>
                <AlertTriangle size={24} color="#d32f2f" />
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1.5, bgcolor: 'warning.light', borderRadius: 1 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    最大回撤日期
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    {data.maxDrawdownDate ? 
                      new Date(data.maxDrawdownDate).toLocaleDateString('zh-CN') : 
                      '未知'
                    }
                  </Typography>
                </Box>
                <Calendar size={24} color="#ed6c02" />
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1.5, bgcolor: 'secondary.light', borderRadius: 1 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    回撤持续天数
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                    {data.maxDrawdownDuration || 0} 天
                  </Typography>
                </Box>
                <TrendingDown size={24} color="#9c27b0" />
              </Box>
            </Box>

            {/* 显示选项 */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showMaxDrawdownPeriod}
                    onChange={(e) => setShowMaxDrawdownPeriod(e.target.checked)}
                    size="small"
                  />
                }
                label="标注最大回撤期间"
              />
            </Box>
          </Box>
        }
      />

      <CardContent>
        <Box
          ref={chartRef}
          sx={{ height, width: '100%', minHeight: 400 }}
        />
      </CardContent>
    </Card>
  );
}
