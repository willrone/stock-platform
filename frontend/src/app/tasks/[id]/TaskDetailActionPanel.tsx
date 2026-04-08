import React from 'react';
import { Button } from '@mui/material';
import { Copy, Download, Play, RefreshCw, Trash2 } from 'lucide-react';

import type { Task } from '../../../types/task';

interface TaskDetailActionPanelProps {
  task: Task;
  refreshing: boolean;
  onRefresh: () => void;
  onRetry: () => void;
  onExport: () => void;
  onRebuild: () => void;
  onDelete: () => void;
}

export function TaskDetailActionPanel({
  task,
  refreshing,
  onRefresh,
  onRetry,
  onExport,
  onRebuild,
  onDelete,
}: TaskDetailActionPanelProps): React.ReactNode {
  return (
    <>
      <Button
        variant="outlined"
        startIcon={<RefreshCw size={16} />}
        onClick={onRefresh}
        disabled={refreshing}
      >
        刷新
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

      <Button variant="outlined" startIcon={<Copy size={16} />} onClick={onRebuild}>
        重建任务
      </Button>
      <Button variant="outlined" color="error" startIcon={<Trash2 size={16} />} onClick={onDelete}>
        删除
      </Button>
    </>
  );
}
