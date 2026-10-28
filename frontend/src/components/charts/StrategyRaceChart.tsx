'use client';

/**
 * 策略赛马对比图（多策略净值曲线叠一张图）
 *
 * 【P0-04】原「策略赛马」页只有 6 张卡片（市值/收益率/持仓数），没有净值曲线，
 * 用户无法直观看出「谁在什么时候领先、谁后来翻车」。
 *
 * 口径说明（重要）：
 * - 曲线是 **月频复利还原**（36 点），不是日频精确净值 → 阶梯感是预期的，
 *   不能当成日频曲线解读。组件标题下方强制展示该口径。
 * - 净值起点 = 1.0，即「期初 100 万变成多少倍」。
 * - 主线策略（is_mainline）加粗显示，其余策略用较细的线。
 */

import React, { useEffect, useMemo, useRef } from 'react';
// ⚠️ 只做**类型**导入（编译期被擦除，不进 server bundle）。
// 运行时的 echarts 必须在 useEffect 里动态 import ——
// zrender 在模块加载期就会做 UA 探测（`'ontouchstart' in window`），
// 而 Node ≥21 定义了全局 `navigator` 却没有 `window`，
// 于是 zrender 走进 `detect()` 分支并抛 `ReferenceError: window is not defined`。
// 顶层 `import * as echarts from 'echarts'` 会让任何静态预渲染该组件的页面构建失败。
import type * as EChartsNS from 'echarts';
import { Box, Typography, useTheme } from '@mui/material';

export interface RaceNavPoint {
  date: string;
  value: number;
}

export interface RaceSeries {
  key: string;
  name: string;
  isMainline?: boolean;
  nav: RaceNavPoint[];
}

interface StrategyRaceChartProps {
  series: RaceSeries[];
  height?: number;
  /** 无数据时的提示文案 */
  emptyMessage?: string;
}

/** 策略配色：主线最醒目，其余用可区分的中性色 */
const SERIES_COLORS = [
  '#10b981', // 主线 —— 绿
  '#3b82f6', // 蓝
  '#a855f7', // 紫
  '#f59e0b', // 琥珀
  '#ef4444', // 红（对照策略）
  '#06b6d4', // 青
];

export default function StrategyRaceChart({
  series,
  height = 340,
  emptyMessage = '暂无净值数据',
}: StrategyRaceChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<EChartsNS.ECharts | null>(null);
  const theme = useTheme();

  // 全部策略共享同一条日期轴（取所有日期的并集，按月排序）
  const axis = useMemo(() => {
    const set = new Set<string>();
    series.forEach((s) => s.nav.forEach((p) => set.add(p.date)));
    return Array.from(set).sort();
  }, [series]);

  const indexOf = useMemo(() => {
    const m = new Map<string, number>();
    axis.forEach((d, i) => m.set(d, i));
    return m;
  }, [axis]);

  useEffect(() => {
    if (!chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
      chartInstance.current = null;
    }

    const usable = series.filter((s) => s.nav && s.nav.length > 0);
    if (usable.length === 0 || axis.length === 0) return;

    const isDark = theme.palette.mode === 'dark';
    const textColor = isDark ? 'rgba(255,255,255,0.72)' : 'rgba(0,0,0,0.65)';
    const axisColor = isDark ? 'rgba(255,255,255,0.16)' : 'rgba(0,0,0,0.12)';

    // 动态 import echarts：避免 zrender 在 SSR 期探测 window（见文件头注释）
    let disposed = false;
    let chart: EChartsNS.ECharts | null = null;
    let onResize: (() => void) | null = null;

    void import('echarts').then((echarts) => {
      // 组件在 import 完成前已卸载 / ref 已失效 → 直接放弃渲染
      if (disposed || !chartRef.current) return;

      chart = echarts.init(chartRef.current);
      chartInstance.current = chart;

      const echartsSeries: EChartsNS.SeriesOption[] = usable.map((s, i) => {
        const color = SERIES_COLORS[i % SERIES_COLORS.length];
        // 对齐到统一日期轴：缺失的月份用 null（不连线断点）
        const byDate = new Map(s.nav.map((p) => [p.date, p.value]));
        const data: (number | null)[] = axis.map((d) => {
          const v = byDate.get(d);
          return v === undefined ? null : v;
        });
        return {
          name: s.name,
          type: 'line',
          data,
          connectNulls: true,
          symbol: 'none',
          smooth: false,
          lineStyle: {
            color,
            width: s.isMainline ? 3 : 1.8,
            type: s.isMainline ? 'solid' : 'solid',
            opacity: s.isMainline ? 1 : 0.85,
          },
          itemStyle: { color },
          emphasis: { focus: 'series' },
        } as EChartsNS.SeriesOption;
      });

      // 1.0 基准线（期初本金）
      echartsSeries.push({
        name: '期初本金',
        type: 'line',
        data: axis.map(() => 1.0),
        symbol: 'none',
        lineStyle: { color: axisColor, width: 1, type: 'dashed' },
        tooltip: { show: false },
      } as EChartsNS.SeriesOption);

      chart.setOption({
        animation: false,
        grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
        legend: {
          type: 'scroll',
          top: 0,
          textStyle: { color: textColor, fontSize: 12 },
          inactiveColor: isDark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.25)',
        },
        tooltip: {
          trigger: 'axis',
          valueFormatter: (v: unknown) =>
            typeof v === 'number' ? `${v.toFixed(3)} 倍` : String(v ?? '—'),
        },
        xAxis: {
          type: 'category',
          data: axis,
          boundaryGap: false,
          axisLine: { lineStyle: { color: axisColor } },
          axisLabel: {
            color: textColor,
            fontSize: 11,
            formatter: (v: string) => (v.length >= 7 ? v.slice(2) : v),
          },
          axisTick: { show: false },
        },
        yAxis: {
          type: 'value',
          scale: true,
          name: '净值（倍）',
          nameTextStyle: { color: textColor, fontSize: 11 },
          axisLabel: {
            color: textColor,
            fontSize: 11,
            formatter: (v: number) => `${v.toFixed(2)}×`,
          },
          splitLine: { lineStyle: { color: axisColor, type: 'dashed' } },
        },
        series: echartsSeries,
      });

      onResize = () => chart?.resize();
      window.addEventListener('resize', onResize);
    });

    return () => {
      disposed = true;
      if (onResize) {
        window.removeEventListener('resize', onResize);
        onResize = null;
      }
      chart?.dispose();
      chart = null;
      chartInstance.current = null;
    };
  }, [series, axis, theme.palette.mode]);

  const hasData = series.some((s) => s.nav && s.nav.length > 0);

  return (
    <Box>
      <Box
        ref={chartRef}
        role="img"
        aria-label="多策略净值曲线对比图"
        sx={{ width: '100%', height }}
      />
      {!hasData && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {emptyMessage}
        </Typography>
      )}
    </Box>
  );
}
