/**
 * 优化结果可视化组件
 * 包含参数重要性、优化历史曲线、Pareto front等
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Card, CardHeader, CardContent, Tabs, Tab, Box, Typography } from '@mui/material';
import { OptimizationResult } from '../../services/optimizationService';
import * as echarts from 'echarts';

interface OptimizationVisualizationProps {
  result: OptimizationResult;
}

export default function OptimizationVisualization({ result }: OptimizationVisualizationProps) {
  const historyChartRef = useRef<HTMLDivElement>(null);
  const importanceChartRef = useRef<HTMLDivElement>(null);
  const paretoChartRef = useRef<HTMLDivElement>(null);
  const historyChartInstance = useRef<echarts.ECharts | null>(null);
  const importanceChartInstance = useRef<echarts.ECharts | null>(null);
  const paretoChartInstance = useRef<echarts.ECharts | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('history');

  useEffect(() => {
    console.log('[OptimizationVisualization] optimization_history:', result.optimization_history);
    console.log(
      '[OptimizationVisualization] optimization_history length:',
      result.optimization_history?.length
    );
    console.log('[OptimizationVisualization] selectedTab:', selectedTab);

    // 如果不在历史Tab，不初始化
    if (selectedTab !== 'history') {
      return;
    }

    if (
      !historyChartRef.current ||
      !result.optimization_history ||
      result.optimization_history.length === 0
    ) {
      console.log('[OptimizationVisualization] 跳过初始化：容器或数据不存在');
      return;
    }

    const initChart = () => {
      if (!historyChartRef.current) {
        console.log('[OptimizationVisualization] historyChartRef.current 为空');
        return;
      }

      // 检查容器是否有尺寸
      const rect = historyChartRef.current.getBoundingClientRect();
      console.log('[OptimizationVisualization] 容器尺寸:', rect.width, rect.height);

      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        console.log('[OptimizationVisualization] 容器尺寸为0，延迟重试');
        setTimeout(initChart, 100);
        return;
      }

      // 如果已有图表实例，先销毁
      if (historyChartInstance.current) {
        historyChartInstance.current.dispose();
        historyChartInstance.current = null;
      }

      console.log('[OptimizationVisualization] 开始初始化历史图表');
      const chart = echarts.init(historyChartRef.current);
      historyChartInstance.current = chart;

      const completedTrials = result.optimization_history.filter(
        t => (t.state === 'complete' || t.state === 'finished') && t.score !== undefined
      );

      console.log('[OptimizationVisualization] completedTrials:', completedTrials.length);

      if (completedTrials.length === 0) {
        console.warn('[OptimizationVisualization] 没有完成的试验数据');
        return;
      }

      const data = completedTrials.map(t => ({
        value: [t.trial_number, t.score!],
        name: `Trial ${t.trial_number}`,
      }));

      // 计算最佳得分曲线
      let bestScore = -Infinity;
      const bestScores = completedTrials.map(t => {
        if (t.score! > bestScore) {
          bestScore = t.score!;
        }
        return [t.trial_number, bestScore];
      });

      console.log(
        '[OptimizationVisualization] 数据点数量:',
        data.length,
        '最佳得分曲线点数:',
        bestScores.length
      );

      const option = {
        title: {
          text: '优化历史曲线',
          left: 'center',
        },
        tooltip: {
          trigger: 'axis',
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: '15%',
          containLabel: true,
        },
        xAxis: {
          type: 'value',
          name: '试验编号',
        },
        yAxis: {
          type: 'value',
          name: '得分',
        },
        series: [
          {
            name: '试验得分',
            type: 'scatter',
            data: data.map(d => d.value),
            symbolSize: 6,
          },
          {
            name: '最佳得分',
            type: 'line',
            data: bestScores,
            lineStyle: {
              color: '#18c964',
              width: 2,
            },
          },
        ],
      };

      chart.setOption(option);
      console.log('[OptimizationVisualization] 历史图表设置完成');

      // 响应式调整
      const handleResize = () => {
        if (historyChartInstance.current) {
          historyChartInstance.current.resize();
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
      };
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 200);

    return () => {
      clearTimeout(timer);
      if (historyChartInstance.current) {
        historyChartInstance.current.dispose();
        historyChartInstance.current = null;
      }
    };
  }, [result.optimization_history, selectedTab]);

  useEffect(() => {
    console.log('[OptimizationVisualization] param_importance:', result.param_importance);
    console.log(
      '[OptimizationVisualization] param_importance keys:',
      result.param_importance ? Object.keys(result.param_importance) : []
    );
    console.log('[OptimizationVisualization] selectedTab:', selectedTab);

    // 如果不在重要性Tab，不初始化
    if (selectedTab !== 'importance') {
      return;
    }

    if (
      !importanceChartRef.current ||
      !result.param_importance ||
      Object.keys(result.param_importance).length === 0
    ) {
      console.log('[OptimizationVisualization] 跳过初始化：容器或数据不存在');
      return;
    }

    const initChart = () => {
      if (!importanceChartRef.current) {
        console.log('[OptimizationVisualization] importanceChartRef.current 为空');
        return;
      }

      // 检查容器是否有尺寸
      const rect = importanceChartRef.current.getBoundingClientRect();
      console.log('[OptimizationVisualization] 重要性容器尺寸:', rect.width, rect.height);

      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        console.log('[OptimizationVisualization] 重要性容器尺寸为0，延迟重试');
        setTimeout(initChart, 100);
        return;
      }

      // 如果已有图表实例，先销毁
      if (importanceChartInstance.current) {
        importanceChartInstance.current.dispose();
        importanceChartInstance.current = null;
      }

      console.log('[OptimizationVisualization] 开始初始化参数重要性图表');
      const chart = echarts.init(importanceChartRef.current);
      importanceChartInstance.current = chart;

      const entries = Object.entries(result.param_importance || {}).sort((a, b) => b[1] - a[1]);

      console.log('[OptimizationVisualization] param_importance entries:', entries.length);

      const option = {
        title: {
          text: '参数重要性',
          left: 'center',
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow',
          },
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: '15%',
          containLabel: true,
        },
        xAxis: {
          type: 'value',
          name: '重要性',
        },
        yAxis: {
          type: 'category',
          data: entries.map(e => e[0]),
          inverse: true,
        },
        series: [
          {
            name: '重要性',
            type: 'bar',
            data: entries.map(e => e[1]),
            itemStyle: {
              color: '#0070f3',
            },
          },
        ],
      };

      chart.setOption(option);
      console.log('[OptimizationVisualization] 参数重要性图表设置完成');

      // 响应式调整
      const handleResize = () => {
        if (importanceChartInstance.current) {
          importanceChartInstance.current.resize();
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
      };
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 200);

    return () => {
      clearTimeout(timer);
      if (importanceChartInstance.current) {
        importanceChartInstance.current.dispose();
        importanceChartInstance.current = null;
      }
    };
  }, [result.param_importance, selectedTab]);

  useEffect(() => {
    if (paretoChartRef.current && result.pareto_front && result.pareto_front.length > 0) {
      const chart = echarts.init(paretoChartRef.current);

      const paretoData = result.pareto_front.map(p => p.objectives);
      const otherData = result.optimization_history
        .filter(t => t.objectives && (t.state === 'complete' || t.state === 'finished'))
        .map(t => t.objectives!)
        .filter(
          obj =>
            result.pareto_front &&
            result.pareto_front.some(
              p => p.objectives[0] === obj[0] && p.objectives[1] === obj[1]
            ) === false
        );

      const option = {
        title: {
          text: '帕累托前沿',
          left: 'center',
        },
        tooltip: {
          trigger: 'item',
        },
        xAxis: {
          type: 'value',
          name: result.objective_metric[0] || '目标1',
        },
        yAxis: {
          type: 'value',
          name: result.objective_metric[1] || '目标2',
        },
        series: [
          {
            name: '帕累托前沿',
            type: 'scatter',
            data: paretoData,
            symbolSize: 10,
            itemStyle: {
              color: '#18c964',
            },
          },
          {
            name: '其他解',
            type: 'scatter',
            data: otherData,
            symbolSize: 6,
            itemStyle: {
              color: '#888',
            },
          },
        ],
      };

      chart.setOption(option);

      return () => {
        chart.dispose();
      };
    }
  }, [result.pareto_front, result.objective_metric]);

  return (
    <Box>
      <Tabs
        value={selectedTab}
        onChange={(e, newValue) => {
          setSelectedTab(newValue);
          console.log('[OptimizationVisualization] Tab切换:', newValue);
        }}
        aria-label="可视化标签页"
      >
        <Tab label="优化历史" value="history" />
        <Tab label="参数重要性" value="importance" />
        {result.pareto_front && result.pareto_front.length > 0 && (
          <Tab label="帕累托前沿" value="pareto" />
        )}
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {selectedTab === 'history' && (
          <Card>
            <CardContent>
              {result.optimization_history && result.optimization_history.length > 0 ? (
                <Box ref={historyChartRef} sx={{ width: '100%', height: 400 }} />
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    暂无优化历史数据
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ mt: 1, display: 'block' }}
                  >
                    数据: {JSON.stringify(result.optimization_history)}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        )}

        {selectedTab === 'importance' && (
          <Card>
            <CardContent>
              {result.param_importance && Object.keys(result.param_importance).length > 0 ? (
                <Box ref={importanceChartRef} sx={{ width: '100%', height: 400 }} />
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    暂无参数重要性数据（仅单目标优化任务提供）
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ mt: 1, display: 'block' }}
                  >
                    数据: {JSON.stringify(result.param_importance)}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        )}

        {selectedTab === 'pareto' && result.pareto_front && result.pareto_front.length > 0 && (
          <Card>
            <CardContent>
              <Box ref={paretoChartRef} sx={{ width: '100%', height: 400 }} />
            </CardContent>
          </Card>
        )}
      </Box>
    </Box>
  );
}
