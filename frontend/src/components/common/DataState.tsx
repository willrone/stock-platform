'use client';

/**
 * DataState —— 统一的「加载 / 空数据 / 失败」三态组件（P1-13）
 *
 * 各页面不再自己写加载圈、空态图、错误提示，统一用这个：
 *
 * ```tsx
 * import { DataState } from '@/components/common/DataState';
 *
 * <DataState
 *   loading={loading}
 *   error={error}          // 错误信息字符串；为 null/undefined 时不展示
 *   empty={items.length === 0}
 *   onRetry={refetch}      // 失败重试 / 空态跳转按钮
 *   loadingText="加载策略信号中…"
 *   emptyTitle="暂无信号"
 *   emptyText="今天还没有产生信号，明天再来看看"
 *   errorText="加载失败，请检查网络后重试"
 *   retryText="重试"
 * />
 * // 三态互斥：loading > error > empty；都不满足时渲染 null（不占位）
 * ```
 *
 * 说明：
 * - 组件自身可点击元素都带 focus-visible 焦点指示（P1-14，
 *   全站级 focusVisible outline 需主题层在 muiTheme.tsx / globals.css 补齐）
 * - 空态的 action 按钮（actionLabel/onAction）用于「空数据时给个入口」，
 *   如「暂无任务 → 创建任务」
 */

import React from 'react';
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import { Inbox } from 'lucide-react';

export interface DataStateProps {
  /** 加载中（优先级最高） */
  loading: boolean;
  /** 错误信息；非空时展示错误态 */
  error?: string | null;
  /** 无数据（!loading 且 !error 时生效） */
  empty?: boolean;
  /** 失败重试 / 空态动作 */
  onRetry?: () => void;
  loadingText?: string;
  emptyTitle?: string;
  emptyText?: string;
  errorText?: string;
  retryText?: string;
  actionLabel?: string;
  onAction?: () => void;
}

/** 键盘焦点指示：主题层 focusVisible 补齐前的组件级兜底 */
const focusOutline = {
  '&:focus-visible': {
    outline: '2px solid primary.main',
    outlineOffset: 2,
    borderRadius: 1,
  },
};

export const DataState: React.FC<DataStateProps> = ({
  loading,
  error,
  empty = false,
  onRetry,
  loadingText = '加载中…',
  emptyTitle = '暂无数据',
  emptyText,
  errorText = '加载失败，请重试',
  retryText = '重试',
  actionLabel,
  onAction,
}) => {
  if (loading) {
    return (
      <Box
        role="status"
        aria-live="polite"
        sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 8 }}
      >
        <CircularProgress aria-label={loadingText} />
        <Typography variant="body2" color="text.secondary">
          {loadingText}
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ py: 4 }} role="alert">
        <Alert severity="error" variant="filled">
          {errorText || error}
        </Alert>
        {onRetry && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Button variant="contained" onClick={onRetry} sx={focusOutline}>
              {retryText}
            </Button>
          </Box>
        )}
      </Box>
    );
  }

  if (empty) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1,
          py: 8,
          textAlign: 'center',
        }}
      >
        <Inbox size={48} color="#9e9e9e" aria-hidden />
        <Typography variant="h6" color="text.secondary">
          {emptyTitle}
        </Typography>
        {emptyText && (
          <Typography variant="body2" color="text.disabled">
            {emptyText}
          </Typography>
        )}
        {actionLabel && onAction && (
          <Button variant="contained" onClick={onAction} sx={{ ...focusOutline, mt: 1 }}>
            {actionLabel}
          </Button>
        )}
      </Box>
    );
  }

  return null;
};
