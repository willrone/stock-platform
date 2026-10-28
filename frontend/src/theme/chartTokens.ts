/**
 * 图表色板唯一真相源
 * ---------------------------------------------------------------------------
 * 图表组件（ECharts / lightweight-charts / Recharts）不再各自维护色板，
 * 全部从这里取色，保证同一语义在全站图表里颜色一致。
 *
 * 【涨跌语义】A 股：**红涨绿跌**
 *  - up   = 上涨 / 正收益（红）
 *  - down = 下跌 / 负收益（绿）
 *  - flat = 平盘
 *
 * 与 MUI 主题的关系：MUI 侧的涨跌色在 `src/theme/tokens.ts` 的
 * `semanticColors`，本文件是它的「图表专用」投影（额外提供 alpha 渐变、
 * 分类色序列、网格线、坐标轴色等图表专用值）。
 */

import { semanticColors } from './tokens';

/* ------------------------------------------------------------------ *
 * 工具
 * ------------------------------------------------------------------ */

/** 给 hex 颜色加 alpha，返回 rgba(...) */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const full =
    h.length === 3
      ? h
          .split('')
          .map(c => c + c)
          .join('')
      : h;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ------------------------------------------------------------------ *
 * 涨跌色（图表投影）
 * ------------------------------------------------------------------ */

export const chartSemantic = {
  /** 上涨 / 正收益 —— 红 */
  up: semanticColors.up,
  upLight: semanticColors.upLight,
  upDark: semanticColors.upDark,
  upFill: withAlpha(semanticColors.up, 0.30),
  upFillFaint: withAlpha(semanticColors.up, 0.10),
  /** 下跌 / 负收益 —— 绿 */
  down: semanticColors.down,
  downLight: semanticColors.downLight,
  downDark: semanticColors.downDark,
  downFill: withAlpha(semanticColors.down, 0.30),
  downFillFaint: withAlpha(semanticColors.down, 0.10),
  /** 平盘 */
  flat: semanticColors.flat,
} as const;

/* ------------------------------------------------------------------ *
 * 分类色序列（多系列 / 多策略时按序取用）
 * ------------------------------------------------------------------ */

export const chartSeries = [
  '#006FEE', // 蓝（主色）
  '#F59E0B', // 琥珀
  '#7C3AED', // 紫
  '#0EA5E9', // 天蓝
  '#16A34A', // 绿
  '#DB2777', // 品红
  '#EF4444', // 红
  '#0891B2', // 青
  '#CA8A04', // 金
  '#64748B', // 石板灰
] as const;

/** 固定策略色（TradingView 信号标记，保证同一策略永远是同一颜色） */
export const chartStrategyColors: Record<string, string> = {
  // 基础技术策略
  moving_average: '#006FEE', // 蓝
  rsi: '#F59E0B', // 琥珀
  macd: '#16A34A', // 绿
  // 高级技术分析策略
  bollinger: '#7C3AED', // 紫
  stochastic: '#0891B2', // 青
  cci: '#EF4444', // 红
  // 统计套利策略
  pairs_trading: '#92400E', // 棕
  mean_reversion: '#64748B', // 蓝灰
  cointegration: '#DB2777', // 品红
  // 因子投资策略
  value_factor: '#4F46E5', // 靛蓝
  momentum_factor: '#CA8A04', // 金
  low_volatility: '#0EA5E9', // 天蓝
  multi_factor: '#DC2626', // 深红
};

/** 扩展调色板（未映射策略的稳定哈希分配） */
export const chartExtendedPalette: readonly string[] = [
  '#006FEE',
  '#F59E0B',
  '#16A34A',
  '#DC2626',
  '#7C3AED',
  '#92400E',
  '#DB2777',
  '#64748B',
  '#CA8A04',
  '#0891B2',
  '#4F46E5',
  '#65A30D',
  '#B45309',
  '#0D9488',
  '#BE185D',
  '#CBD5E1',
  '#A3E635',
  '#34D399',
  '#F472B6',
  '#818CF8',
  '#E11D48',
  '#2DD4BF',
  '#FACC15',
  '#38BDF8',
  '#A78BFA',
];

/* ------------------------------------------------------------------ *
 * 坐标轴 / 网格 / 提示框
 * ------------------------------------------------------------------ */

export const chartAxis = {
  /** 网格线（分割线） */
  splitLine: '#E2E8F0',
  /** 轴线 */
  axisLine: '#CBD5E1',
  /** 刻度文字 */
  label: '#64748B',
  /** 强调文字（tooltip 标题等） */
  labelStrong: '#334155',
} as const;

export const chartGrid = {
  /** ECharts splitLine */
  splitLine: '#EDF1F6',
  /** lightweight-charts 网格 */
  light: '#F0F3F8',
} as const;

export const chartTooltip = {
  background: '#FFFFFF',
  border: '#E2E8F0',
  text: '#0F172A',
  /** 覆盖层遮罩（loading / 空态） */
  overlay: 'rgba(255, 255, 255, 0.8)',
  shadow: '0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 10px 20px -6px rgba(15, 23, 42, 0.12)',
} as const;

/** 通用中性色（图标、提示文字） */
export const chartNeutral = {
  icon: '#64748B',
  iconMuted: '#94A3B8',
  text: '#475569',
  surface: '#F8FAFC',
} as const;

/* ------------------------------------------------------------------ *
 * 语义辅助函数
 * ------------------------------------------------------------------ */

/** 数值 → 图表涨跌色 */
export function chartColorOf(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return chartSemantic.flat;
  if (value > 0) return chartSemantic.up;
  if (value < 0) return chartSemantic.down;
  return chartSemantic.flat;
}

/** 数值 → 图表涨跌填充色（渐变用） */
export function chartFillOf(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return withAlpha(chartSemantic.flat, 0.1);
  }
  if (value > 0) return chartSemantic.upFill;
  if (value < 0) return chartSemantic.downFill;
  return withAlpha(chartSemantic.flat, 0.1);
}

/** 数值 → 图表涨跌淡填充色 */
export function chartFaintFillOf(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return withAlpha(chartSemantic.flat, 0.1);
  }
  if (value > 0) return chartSemantic.upFillFaint;
  if (value < 0) return chartSemantic.downFillFaint;
  return withAlpha(chartSemantic.flat, 0.1);
}

/** 涨跌色 → MUI sx 可用的 palette 路径（与 theme.tokens.semanticColorOf 保持一致） */
export function chartSemanticTokenOf(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'semantics.flat';
  if (value > 0) return 'semantics.up';
  if (value < 0) return 'semantics.down';
  return 'semantics.flat';
}
