/**
 * 优化结果可视化组件
 * 包含参数重要性、优化历史曲线、Pareto front等
 */

'use client';

import React, { useEffect, useRef } from 'react';
import { Card, CardHeader, CardBody, Tabs, Tab } from '@heroui/react';
import { OptimizationResult } from '../../services/optimizationService';
import * as echarts from 'echarts';

interface OptimizationVisualizationProps {
  result: OptimizationResult;
}

export default function OptimizationVisualization({
  result,
}: OptimizationVisualizationProps) {
  const historyChartRef = useRef<HTMLDivElement>(null);
  const importanceChartRef = useRef<HTMLDivElement>(null);
  const paretoChartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (historyChartRef.current && result.optimization_history.length > 0) {
      const chart = echarts.init(historyChartRef.current);
      
      const completedTrials = result.optimization_history.filter(
        t => t.state === 'finished' && t.score !== undefined
      );
      
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

      const option = {
        title: {
          text: '优化历史曲线',
          left: 'center',
        },
        tooltip: {
          trigger: 'axis',
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

      return () => {
        chart.dispose();
      };
    }
  }, [result.optimization_history]);

  useEffect(() => {
    if (importanceChartRef.current && result.param_importance) {
      const chart = echarts.init(importanceChartRef.current);
      
      const entries = Object.entries(result.param_importance).sort(
        (a, b) => b[1] - a[1]
      );

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

      return () => {
        chart.dispose();
      };
    }
  }, [result.param_importance]);

  useEffect(() => {
    if (paretoChartRef.current && result.pareto_front && result.pareto_front.length > 0) {
      const chart = echarts.init(paretoChartRef.current);
      
      const paretoData = result.pareto_front.map(p => p.objectives);
      const otherData = result.optimization_history
        .filter(t => t.objectives && t.state === 'finished')
        .map(t => t.objectives!)
        .filter(obj => !result.pareto_front.some(p => 
          p.objectives[0] === obj[0] && p.objectives[1] === obj[1]
        ));

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
    <Tabs aria-label="可视化标签页">
      <Tab key="history" title="优化历史">
        <Card className="mt-4">
          <CardBody>
            <div
              ref={historyChartRef}
              style={{ width: '100%', height: '400px' }}
            />
          </CardBody>
        </Card>
      </Tab>

      {result.param_importance && (
        <Tab key="importance" title="参数重要性">
          <Card className="mt-4">
            <CardBody>
              <div
                ref={importanceChartRef}
                style={{ width: '100%', height: '400px' }}
              />
            </CardBody>
          </Card>
        </Tab>
      )}

      {result.pareto_front && result.pareto_front.length > 0 && (
        <Tab key="pareto" title="帕累托前沿">
          <Card className="mt-4">
            <CardBody>
              <div
                ref={paretoChartRef}
                style={{ width: '100%', height: '400px' }}
              />
            </CardBody>
          </Card>
        </Tab>
      )}
    </Tabs>
  );
}

