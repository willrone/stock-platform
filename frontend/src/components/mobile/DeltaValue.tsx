'use client';

/**
 * 涨跌数值（颜色 + 图标双冗余）
 *
 * 解决 P0-9：涨跌只靠红绿传达信息 —— 色觉障碍用户（红绿色盲约占男性 8%）
 * 无法区分。WCAG 1.4.1 要求「不能仅用颜色传达信息」，所以这里同时给出
 * ↑ / ↓ / — 图标与正负号。
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';

interface DeltaValueProps {
  /** 数值；null/undefined → 显示「暂无」 */
  value?: number | null;
  /** 后缀，默认 '%' */
  suffix?: string;
  /** 小数位，默认 2 */
  digits?: number;
  /** 颜色语义：正=success / 负=error */
  invert?: boolean;
  /** 字号 */
  variant?: 'body2' | 'h6' | 'caption' | 'subtitle1';
  /** 无数据时的占位文案 */
  placeholder?: string;
}

export const DeltaValue: React.FC<DeltaValueProps> = ({
  value,
  suffix = '%',
  digits = 2,
  invert = false,
  variant = 'body2',
  placeholder = '暂无',
}) => {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return (
      <Typography variant={variant} color="text.secondary">
        {placeholder}
      </Typography>
    );
  }

  const positive = value > 0;
  const negative = value < 0;
  const isGood = invert ? negative : positive;
  const isBad = invert ? positive : negative;
  const color = isGood ? 'success.main' : isBad ? 'error.main' : 'text.secondary';

  const Icon = positive ? ArrowUp : negative ? ArrowDown : Minus;

  return (
    <Box
      component="span"
      sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, color }}
      role="img"
      aria-label={`${positive ? '上涨' : negative ? '下跌' : '持平'} ${Math.abs(value).toFixed(
        digits
      )}${suffix}`}
    >
      <Icon size={variant === 'h6' ? 16 : 13} aria-hidden="true" />
      <Typography variant={variant} sx={{ color, fontWeight: 600 }}>
        {positive ? '+' : ''}
        {value.toFixed(digits)}
        {suffix}
      </Typography>
    </Box>
  );
};

export default DeltaValue;
