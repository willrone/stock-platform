/**
 * ECharts 初始化和管理 Hook
 */

import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

export const useECharts = (
  data: any,
  options: echarts.EChartsOption | ((data: any) => echarts.EChartsOption),
  dependencies: any[] = [],
  isActive: boolean = true
) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current || !data || data.length === 0 || !isActive) {
      return;
    }

    const initChart = () => {
      if (!chartRef.current || !data || data.length === 0) {
        return;
      }

      // 检查容器是否有尺寸
      const rect = chartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (chartInstance.current) {
        chartInstance.current.dispose();
      }

      chartInstance.current = echarts.init(chartRef.current);

      const chartOptions = typeof options === 'function' ? options(data) : options;
      chartInstance.current.setOption(chartOptions);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, [data, isActive, ...dependencies]);

  return chartRef;
};
