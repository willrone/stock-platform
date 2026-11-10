'use client';

/**
 * 错误态组件（带重试入口）
 *
 * 解决 P0-8：之前 today-signal:252 / signals:657 的失败提示只有一句文案，
 * 用户没有任何自救入口，只能刷新整个页面。
 *
 * 同时解决「后端 detail 丢失」：后端 HTTPException 会返回 detail 字段
 * （见 backend/app/main.py 的 http_exception_handler），这里原样透传。
 */

import React from 'react';
import { Alert, AlertTitle, Box, Button, Collapse, Stack, Typography } from '@mui/material';
import { AlertTriangle, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import { MIN_TOUCH_TARGET } from './safeArea';

interface ErrorStateProps {
  /** 简短标题，例如「加载信号失败」 */
  title?: string;
  /** 用户可读的错误信息（后端 detail 优先） */
  message?: string | null;
  /** 原始错误对象 / detail 结构，用于展开查看 */
  detail?: unknown;
  /** 重试回调；不传则不显示重试按钮 */
  onRetry?: () => void;
  /** 重试按钮文案 */
  retryLabel?: string;
  /** 重试中 */
  retrying?: boolean;
  severity?: 'error' | 'warning';
}

const stringifyDetail = (detail: unknown): string => {
  if (detail == null) {
    return '';
  }
  if (typeof detail === 'string') {
    return detail;
  }
  try {
    return JSON.stringify(detail, null, 2);
  } catch {
    return String(detail);
  }
};

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = '加载失败',
  message,
  detail,
  onRetry,
  retryLabel = '重试',
  retrying = false,
  severity = 'error',
}) => {
  const [open, setOpen] = React.useState(false);
  const detailText = stringifyDetail(detail);
  // 后端 detail 已经通过 message 展示时，展开区只补充状态码等信息
  const showDetail = detailText && detailText !== message;

  return (
    <Alert
      severity={severity}
      icon={<AlertTriangle size={20} />}
      sx={{ mb: 2 }}
      action={
        onRetry ? (
          <Button
            onClick={onRetry}
            disabled={retrying}
            startIcon={<RefreshCw size={16} />}
            sx={{
              minHeight: MIN_TOUCH_TARGET,
              minWidth: MIN_TOUCH_TARGET,
              whiteSpace: 'nowrap',
            }}
          >
            {retrying ? '重试中…' : retryLabel}
          </Button>
        ) : undefined
      }
    >
      <AlertTitle sx={{ fontWeight: 700 }}>{title}</AlertTitle>
      <Typography variant="body2" component="div">
        {message || '请求失败，请稍后重试。'}
      </Typography>

      {showDetail && (
        <>
          <Box
            component="button"
            type="button"
            onClick={() => setOpen(v => !v)}
            sx={{
              mt: 1,
              p: 0,
              border: 0,
              background: 'transparent',
              color: 'inherit',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 0.5,
              minHeight: MIN_TOUCH_TARGET,
              textDecoration: 'underline',
              font: 'inherit',
            }}
            aria-expanded={open}
          >
            {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {open ? '收起技术细节' : '查看技术细节'}
          </Box>
          <Collapse in={open} unmountOnExit>
            <Box
              component="pre"
              sx={{
                mt: 1,
                mb: 0,
                p: 1,
                borderRadius: 1,
                bgcolor: 'action.hover',
                fontSize: 12,
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
                maxHeight: 220,
                overflow: 'auto',
              }}
            >
              {detailText}
            </Box>
          </Collapse>
        </>
      )}
    </Alert>
  );
};

interface EmptyStateProps {
  title?: string;
  message?: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title = '暂无数据', message, action }) => (
  <Stack
    alignItems="center"
    justifyContent="center"
    spacing={1.5}
    sx={{ py: 6, px: 2, textAlign: 'center' }}
  >
    <Typography variant="h6" color="text.secondary">
      {title}
    </Typography>
    {message && (
      <Typography variant="body2" color="text.secondary">
        {message}
      </Typography>
    )}
    {action}
  </Stack>
);

export default ErrorState;
