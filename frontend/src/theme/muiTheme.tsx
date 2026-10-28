'use client';

/**
 * MUI 主题配置
 * ---------------------------------------------------------------------------
 * 本文件**只做装配**，所有具体数值都来自 `src/theme/tokens.ts`（唯一真相源）。
 * 新增/修改颜色、间距、圆角、阴影、字号，请改 tokens.ts，不要在这里写魔法值。
 *
 * 【涨跌语义】A 股习惯：**红涨绿跌**
 *  - theme.palette.semantics.up   → 红，正收益 / 上涨 / 买入
 *  - theme.palette.semantics.down → 绿，负收益 / 下跌 / 卖出
 *  - theme.palette.success / error 只用于「操作成功 / 失败」
 */

import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ReactNode } from 'react';
import { getMobileOverrides } from './mobileOverrides';
import {
  backgroundColors,
  borderColors,
  fontFamily,
  primaryColors,
  radii,
  secondaryColors,
  semanticColors,
  shadows,
  statusColors,
  textColors,
  typography,
} from './tokens';

/**
 * MUI palette 的语义扩展。
 * `semantics.up` / `semantics.down` 是纯字符串色值，因此可直接用于
 * `sx={{ color: 'semantics.up' }}`（MUI 的 sx 会去 theme.palette 里查路径）。
 */
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

// 创建基础主题
const baseTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: primaryColors.DEFAULT,
      light: primaryColors.light,
      dark: primaryColors.dark,
      contrastText: primaryColors.contrastText,
    },
    secondary: {
      main: secondaryColors.main,
      light: secondaryColors.light,
      dark: secondaryColors.dark,
      contrastText: secondaryColors.contrastText,
    },
    // success / error 只表示「操作成功 / 操作失败」，不要当涨跌色用
    success: {
      main: statusColors.success.main,
      light: statusColors.success.light,
      dark: statusColors.success.dark,
    },
    error: {
      main: statusColors.error.main,
      light: statusColors.error.light,
      dark: statusColors.error.dark,
    },
    warning: {
      main: statusColors.warning.main,
      light: statusColors.warning.light,
      dark: statusColors.warning.dark,
    },
    info: {
      main: statusColors.info.main,
      light: statusColors.info.light,
      dark: statusColors.info.dark,
    },
    background: {
      default: backgroundColors.default,
      paper: backgroundColors.paper,
    },
    text: {
      primary: textColors.primary,
      secondary: textColors.secondary,
      disabled: textColors.disabled,
    },
    divider: borderColors.default,
    // 涨跌语义色（红涨绿跌）
    semantics: paletteSemantics,
  },
  typography: {
    fontFamily,
    h1: { ...typography.h1 },
    h2: { ...typography.h2 },
    h3: { ...typography.h3 },
    h4: { ...typography.h4 },
    h5: { ...typography.h5 },
    h6: { ...typography.h6 },
    subtitle1: { ...typography.subtitle1 },
    subtitle2: { ...typography.subtitle2 },
    body1: { ...typography.body1 },
    body2: { ...typography.body2 },
    caption: { ...typography.caption },
    button: { ...typography.button },
    overline: { ...typography.overline },
  },
  shape: {
    borderRadius: radii.md,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: shadows.sm,
          borderRadius: radii.md,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

// 合并移动端优化配置
const mobileOverrides = getMobileOverrides(baseTheme);
const theme = createTheme(baseTheme, mobileOverrides);

interface MUIThemeProviderProps {
  children: ReactNode;
}

export function MUIThemeProvider({ children }: MUIThemeProviderProps) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}

export default theme;
