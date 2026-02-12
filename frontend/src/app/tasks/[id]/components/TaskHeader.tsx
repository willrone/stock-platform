/**
 * 页面标题和操作按钮组件
 */

import React from 'react';
import { Box, Typography, Button, IconButton, Chip } from '@mui/material';
import { ArrowLeft, RefreshCw, Copy, Play, Download, Trash2 } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';

interface TaskHeaderProps {
  task: Task;
  refreshing: boolean;
  onBack: () => void;
  onRefresh: () => void;
  onRebuild: () => void;
  onRetry: () => void;
  onExport: () => void;
  onDelete: () => void;
}

export function TaskHeader({
  task,
  refreshing,
  onBack,
  onRefresh,
  onRebuild,
  onRetry,
  onExport,
  onDelete,
}: TaskHeaderProps) {
  const getStatusChip = (status: Task['status']) => {
    const statusConfig = {
      created: { color: 'default' as const, text: '已创建' },
      running: { color: 'primary' as const, text: '运行中' },
      completed: { color: 'success' as const, text: '已完成' },
      failed: { color: 'error' as const, text: '失败' },
    };

    const config = statusConfig[status] || statusConfig.created;
    return <Chip label={config.text} color={config.color} size="small" />;
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: { xs: 1.5, md: 2 } }}>
      {/* 标题行 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
        <IconButton onClick={onBack} size="small" sx={{ flexShrink: 0, minWidth: 44, minHeight: 44 }}>
          <ArrowLeft size={20} />
        </IconButton>
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="h4"
              component="h1"
              sx={{
                fontWeight: 600,
                fontSize: { xs: '1rem', sm: '1.5rem', md: '2.125rem' },
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                minWidth: 0,
              }}
            >
              {task.task_name}
            </Typography>
            {getStatusChip(task.status)}
          </Box>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{
              fontSize: { xs: '0.625rem', sm: '0.75rem' },
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              display: 'block',
            }}
          >
            任务ID: {task.task_id}
          </Typography>
        </Box>
      </Box>

      {/* 操作按钮 - 移动端横向滚动 */}
      <Box
        sx={{
          display: 'flex',
          gap: { xs: 0.5, sm: 1 },
          overflowX: { xs: 'auto', md: 'visible' },
          pb: { xs: 0.5, md: 0 },
          ml: { xs: 0, md: 5 },
          WebkitOverflowScrolling: 'touch',
          '&::-webkit-scrollbar': { height: 0 },
        }}
      >
        <Button
          variant="outlined"
          startIcon={<RefreshCw size={16} />}
          onClick={onRefresh}
          disabled={refreshing}
          size="small"
          sx={{ flexShrink: 0, minHeight: 44, px: { xs: 1.5, sm: 2 }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
        >
          刷新
        </Button>

        <Button
          variant="outlined"
          color="secondary"
          startIcon={<Copy size={16} />}
          onClick={onRebuild}
          size="small"
          sx={{ flexShrink: 0, minHeight: 44, px: { xs: 1.5, sm: 2 }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
        >
          重建
        </Button>

        {task.status === 'failed' && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<Play size={16} />}
            onClick={onRetry}
            size="small"
            sx={{ flexShrink: 0, minHeight: 44, px: { xs: 1.5, sm: 2 }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
          >
            重试
          </Button>
        )}

        {task.status === 'completed' && (
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<Download size={16} />}
            onClick={onExport}
            size="small"
            sx={{ flexShrink: 0, minHeight: 44, px: { xs: 1.5, sm: 2 }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
          >
            导出
          </Button>
        )}

        <Button
          variant="outlined"
          color="error"
          startIcon={<Trash2 size={16} />}
          onClick={onDelete}
          size="small"
          sx={{ flexShrink: 0, minHeight: 44, px: { xs: 1.5, sm: 2 }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
        >
          删除
        </Button>
      </Box>
    </Box>
  );
}
