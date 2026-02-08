/**
 * 任务侧边栏组件
 */

import React from 'react';
import { Card, CardHeader, CardContent, Box, Typography, Button } from '@mui/material';
import { RefreshCw, Play, Download, Trash2 } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';
import BacktestTaskStatus from '@/components/backtest/BacktestTaskStatus';

interface TaskSidebarProps {
  task: Task;
  refreshing: boolean;
  onRefresh: () => void;
  onRetry: () => void;
  onExport: () => void;
  onDelete: () => void;
  onStop?: () => void;
}

export function TaskSidebar({
  task,
  refreshing,
  onRefresh,
  onRetry,
  onExport,
  onDelete,
  onStop,
}: TaskSidebarProps) {
  // 回测任务使用专用状态组件
  if (task.task_type === 'backtest') {
    return (
      <BacktestTaskStatus
        task={task}
        onRetry={onRetry}
        onStop={onStop || (() => {})}
        loading={refreshing}
      />
    );
  }

  // 预测任务��边栏
  return (
    <>
      {/* 统计信息 */}
      {task.results && (
        <Card>
          <CardHeader title="统计信息" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                  {task.results.total_stocks}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总股票数
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                  {task.results.successful_predictions}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  成功预测
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                  {((task.results.average_confidence || 0) * 100).toFixed(1)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  平均置信度
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* 快速操作 */}
      <Card>
        <CardHeader title="快速操作" />
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Button
              variant="outlined"
              startIcon={<RefreshCw size={16} />}
              onClick={onRefresh}
              disabled={refreshing}
              fullWidth
            >
              刷新状态
            </Button>

            {task.status === 'failed' && (
              <Button
                variant="contained"
                color="primary"
                startIcon={<Play size={16} />}
                onClick={onRetry}
                fullWidth
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
                fullWidth
              >
                导出结果
              </Button>
            )}

            <Button
              variant="outlined"
              color="error"
              startIcon={<Trash2 size={16} />}
              onClick={onDelete}
              fullWidth
            >
              删除任务
            </Button>
          </Box>
        </CardContent>
      </Card>
    </>
  );
}
