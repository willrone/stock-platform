/**
 * 加载动画组件
 *
 * 提供统一的加载状态显示，包括：
 * - 旋转动画
 * - 自定义文本
 * - 不同尺寸
 * - 居中布局
 */

'use client';

import React from 'react';
import { CircularProgress, Box, Typography, SxProps, Theme } from '@mui/material';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  fullScreen?: boolean;
}

const sizeMap = {
  sm: 24,
  md: 40,
  lg: 56,
};

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  text = '加载中...',
  size = 'lg',
  className = '',
  fullScreen = false,
}) => {
  const containerSx: SxProps<Theme> = fullScreen
    ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.paper',
        zIndex: 9999,
      }
    : {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
      };

  return (
    <Box sx={containerSx} className={className}>
      <CircularProgress size={sizeMap[size]} />
      {text && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          {text}
        </Typography>
      )}
    </Box>
  );
};

// 页面级加载组件
export const PageLoading: React.FC<{ text?: string }> = ({ text }) => (
  <LoadingSpinner text={text} fullScreen />
);

// 内联加载组件
export const InlineLoading: React.FC<{ text?: string; size?: 'sm' | 'md' | 'lg' }> = ({
  text,
  size = 'md',
}) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
    <CircularProgress size={sizeMap[size]} />
    {text && (
      <Typography variant="body2" color="text.secondary">
        {text}
      </Typography>
    )}
  </Box>
);

// 按钮加载状态
export const ButtonLoading: React.FC = () => <Loader2 size={16} className="animate-spin" />;
