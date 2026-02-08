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
    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <IconButton onClick={onBack} size="small">
          <ArrowLeft size={20} />
        </IconButton>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
            {task.task_name}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            任务ID: {task.task_id}
          </Typography>
        </Box>
        {getStatusChip(task.status)}
      </Box>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button
          variant="outlined"
          startIcon={<RefreshCw size={16} />}
          onClick={onRefresh}
          disabled={refreshing}
        >
          刷新
        </Button>

        <Button
          variant="outlined"
          color="secondary"
          startIcon={<Copy size={16} />}
          onClick={onRebuild}
        >
          重建任务
        </Button>

        {task.status === 'failed' && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<Play size={16} />}
            onClick={onRetry}
          >
            重新运行
          </Button>
        )}

        {task.status === 'completed' && (
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<Download size={16} />}
            onClick={onExport}
          >
            导出结果
          </Button>
        )}

        <Button
          variant="outlined"
          color="error"
          startIcon={<Trash2 size={16} />}
          onClick={onDelete}
        >
          删除
        </Button>
      </Box>
    </Box>
  );
}
