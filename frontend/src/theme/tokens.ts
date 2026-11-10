/**
 * 设计 Token 唯一真相源 (Single Source of Truth)
 * ---------------------------------------------------------------------------
 * 全站所有颜色 / 间距 / 圆角 / 阴影 / 字号都从这里取，禁止在业务代码里硬编码。
 *
 * 消费方：
 *  - src/theme/muiTheme.tsx        MUI 主题
 *  - src/theme/mobileOverrides.ts  移动端主题覆盖
 *  - src/theme/chartTokens.ts      图表色板
 *  - src/app/globals.css           CSS 变量（与本文件的 HSL 值一一对应）
 *
 * 【已移除的 UI 库】
 *  - HeroUI：全站 0 引用，只注入 128KB 死 CSS
 *  - antd：4 个页面已迁移到 MUI，ConfigProvider 主题（antdTheme.ts）一并删除
 *
 * 【关键语义约定】
 *  - A 股涨跌：**红涨绿跌**。涨 = semantics.up（红），跌 = semantics.down（绿）。
 *  - success / error 仅用于「操作成功 / 操作失败」，**不得**再当涨跌色使用。
 *  - 只做浅色主题，暂无暗色模式。
 */

/* ------------------------------------------------------------------ *
 * 颜色
 * ------------------------------------------------------------------ */

/** 品牌主色（与原 antd / HeroUI 主色对齐） */
export const primaryColors = {
  50: '#E6F1FE',
  100: '#CCE3FD',
  200: '#99C7FB',
  300: '#66AAF9',
  400: '#338EF7',
  500: '#006FEE',
  600: '#005BC4',
  700: '#004493',
  800: '#002E62',
  900: '#001731',
  DEFAULT: '#006FEE',
  light: '#338EF7',
  dark: '#005BC4',
  contrastText: '#FFFFFF',
  /** 浅色底（选中态 / 徽标底） */
  subtle: '#E6F1FE',
} as const;

/** 辅助色（紫） */
export const secondaryColors = {
  main: '#7C3AED',
  light: '#A78BFA',
  dark: '#5B21B6',
  contrastText: '#FFFFFF',
  subtle: '#F3EEFF',
} as const;

/** 中性色阶 */
export const neutralColors = {
  0: '#FFFFFF',
  50: '#F8FAFC',
  100: '#F1F5F9',
  200: '#E2E8F0',
  300: '#CBD5E1',
  400: '#94A3B8',
  500: '#64748B',
  600: '#475569',
  700: '#334155',
  800: '#1E293B',
  900: '#0F172A',
} as const;

/** 文字色 */
export const textColors = {
  primary: '#0F172A',
  secondary: '#475569',
  tertiary: '#64748B',
  disabled: '#94A3B8',
  inverse: '#FFFFFF',
} as const;

/** 背景色 */
export const backgroundColors = {
  default: '#F8FAFC',
  paper: '#FFFFFF',
  subtle: '#F1F5F9',
  overlay: 'rgba(15, 23, 42, 0.45)',
} as const;

/** 边框色 */
export const borderColors = {
  default: '#E2E8F0',
  strong: '#CBD5E1',
  subtle: '#F1F5F9',
} as const;

/**
 * 状态色 —— 仅用于「操作成功 / 失败 / 警告 / 提示」
 * ⚠️ 不要用它们表达涨跌，涨跌请用 semantics。
 */
export const statusColors = {
  success: { main: '#16A34A', light: '#22C55E', dark: '#15803D', subtle: '#EAF7EE' },
  error: { main: '#DC2626', light: '#EF4444', dark: '#B91C1C', subtle: '#FDECEC' },
  warning: { main: '#ED6C02', light: '#F59E0B', dark: '#C2410C', subtle: '#FFF4E6' },
  info: { main: '#0288D1', light: '#38BDF8', dark: '#0369A1', subtle: '#E6F4FC' },
} as const;

/**
 * 涨跌语义色 —— A 股习惯：**红涨绿跌**
 *
 * 用法（sx）：`sx={{ color: 'semantics.up' }}` / `sx={{ color: 'semantics.down' }}`
 * ⚠️ 不要用 `color="semantics.up"`（Typography 的 color prop 只能取 palette 的一级键）
 */
export const semanticColors = {
  /** 上涨 / 正收益 / 买入 */
  up: '#D32F2F',
  upLight: '#EF5350',
  upDark: '#C62828',
  /** 上涨的极浅底色（表格斑马 / 徽标底） */
  upSoft: 'rgba(211, 47, 47, 0.10)',
  /** 下跌 / 负收益 / 卖出 */
  down: '#2E7D32',
  downLight: '#4CAF50',
  downDark: '#1B5E20',
  downSoft: 'rgba(46, 125, 50, 0.10)',
  /** 平盘 / 不涨不跌 */
  flat: '#64748B',
  flatSoft: 'rgba(100, 116, 139, 0.10)',
} as const;

/** 供 MUI 主题直接消费的 palette 语义片段 */
export const paletteSemantics = {
  up: semanticColors.up,
  upLight: semanticColors.upLight,
  upDark: semanticColors.upDark,
  upSoft: semanticColors.upSoft,
  down: semanticColors.down,
  downLight: semanticColors.downLight,
  downDark: semanticColors.downDark,
  downSoft: semanticColors.downSoft,
  flat: semanticColors.flat,
  flatSoft: semanticColors.flatSoft,
} as const;

/* ------------------------------------------------------------------ *
 * 间距
 * ------------------------------------------------------------------ */

/** 间距 scale（单位 px，4 的倍数） */
export const spacing = {
  none: 0,
  xxs: 2,
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
  huge: 40,
  giant: 48,
} as const;

/* ------------------------------------------------------------------ *
 * 圆角（收敛到 3 档）
 * ------------------------------------------------------------------ */

export const radii = {
  sm: 4,
  md: 8,
  lg: 12,
} as const;

export type RadiusToken = keyof typeof radii;

/* ------------------------------------------------------------------ *
 * 阴影（收敛到 2 档）
 * ------------------------------------------------------------------ */

export const shadows = {
  /** 卡片 / 浮层：极轻 */
  sm: '0 1px 2px 0 rgba(15, 23, 42, 0.06), 0 1px 3px 0 rgba(15, 23, 42, 0.08)',
  /** 弹层 / 下拉 / 悬浮卡片 */
  lg: '0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 10px 20px -6px rgba(15, 23, 42, 0.12)',
} as const;

export type ShadowToken = keyof typeof shadows;

/* ------------------------------------------------------------------ *
 * 字体与字号
 * ------------------------------------------------------------------ */

export const fontFamily = [
  '-apple-system',
  'BlinkMacSystemFont',
  '"Segoe UI"',
  'Roboto',
  '"PingFang SC"',
  '"Hiragino Sans GB"',
  '"Microsoft YaHei"',
  '"Helvetica Neue"',
  'Arial',
  'sans-serif',
].join(',');

export const fontFamilyMono = [
  '"SF Mono"',
  'Menlo',
  'Consolas',
  '"Roboto Mono"',
  '"Liberation Mono"',
  'monospace',
].join(',');

/**
 * 字号 scale
 *
 * 中文行高建议：正文 1.6 ~ 1.75，标题 1.3 ~ 1.5（中文方块字需要比英文更松）。
 */
export const typography = {
  /** 页面主标题 */
  h1: { fontSize: '2rem', lineHeight: 1.35, fontWeight: 600, letterSpacing: '-0.01em' },
  /** 区块标题 */
  h2: { fontSize: '1.5rem', lineHeight: 1.4, fontWeight: 600, letterSpacing: '-0.01em' },
  /** 卡片标题 */
  h3: { fontSize: '1.25rem', lineHeight: 1.45, fontWeight: 600 },
  /** 小节标题 */
  h4: { fontSize: '1.125rem', lineHeight: 1.5, fontWeight: 600 },
  h5: { fontSize: '1rem', lineHeight: 1.6, fontWeight: 600 },
  h6: { fontSize: '0.875rem', lineHeight: 1.6, fontWeight: 600 },
  /** 次级标题 */
  subtitle1: { fontSize: '1rem', lineHeight: 1.6, fontWeight: 500 },
  subtitle2: { fontSize: '0.875rem', lineHeight: 1.6, fontWeight: 500 },
  /** 正文 */
  body1: { fontSize: '0.9375rem', lineHeight: 1.7, fontWeight: 400 },
  body2: { fontSize: '0.8125rem', lineHeight: 1.65, fontWeight: 400 },
  /** 辅助文字 / 表格备注 */
  caption: { fontSize: '0.75rem', lineHeight: 1.6, fontWeight: 400 },
  /** 按钮文字 */
  button: { fontSize: '0.875rem', lineHeight: 1.5, fontWeight: 500 },
  /** 极小标签 */
  overline: { fontSize: '0.6875rem', lineHeight: 1.5, fontWeight: 600, letterSpacing: '0.08em' },
} as const;

export type TypographyVariant = keyof typeof typography;

/**
 * 移动端字号覆盖（sm 断点以下）
 * 移动端不缩小正文的可读性下限：body1 保持 0.9375rem。
 */
export const mobileTypography = {
  h4: { fontSize: '1.0625rem' },
  h5: { fontSize: '1rem' },
  h6: { fontSize: '0.875rem' },
  body1: { fontSize: '0.9375rem' },
  body2: { fontSize: '0.8125rem' },
  caption: { fontSize: '0.75rem' },
  overline: { fontSize: '0.6875rem' },
} as const;

/* ------------------------------------------------------------------ *
 * 控件尺寸
 * ------------------------------------------------------------------ */

export const controlSizes = {
  /** 触摸目标最小高度（44px，Apple HIG / Material 建议） */
  touchTarget: 44,
  touchTargetSmall: 40,
  touchTargetLarge: 48,
  inputHeight: 44,
} as const;

/* ------------------------------------------------------------------ *
 * 工具函数
 * ------------------------------------------------------------------ */

/**
 * 根据数值返回涨跌语义 token 名（用于 sx 的 color）。
 *
 * ```ts
 * sx={{ color: semanticColorOf(v) }}   // → 'semantics.up' / 'semantics.down'
 * ```
 */
export function semanticColorOf(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'semantics.flat';
  }
  if (value > 0) {
    return 'semantics.up';
  }
  if (value < 0) {
    return 'semantics.down';
  }
  return 'semantics.flat';
}

/** 根据数值返回涨跌语义色值（用于图表 / 非 sx 场景） */
export function semanticColorValue(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return semanticColors.flat;
  }
  if (value > 0) {
    return semanticColors.up;
  }
  if (value < 0) {
    return semanticColors.down;
  }
  return semanticColors.flat;
}
