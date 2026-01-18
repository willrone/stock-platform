/**
 * 优化任务列表组件
 */

'use client';

import React from 'react';
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  LinearProgress,
  Button,
  Tooltip,
  Box,
  Typography,
  IconButton,
} from '@mui/material';
import { Eye, RefreshCw } from 'lucide-react';
import { OptimizationTask } from '../../services/optimizationService';

interface OptimizationTaskListProps {
  tasks: OptimizationTask[];
  onTaskSelect: (taskId: string) => void;
  onRefresh: () => void;
}

export default function OptimizationTaskList({
  tasks,
  onTaskSelect,
  onRefresh,
}: OptimizationTaskListProps) {
  const getStatusColor = (status: string): "success" | "primary" | "error" | "default" => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'failed':
        return 'error';
      case 'created':
      case 'queued':
        return 'default';
      default:
        return 'default';
    }
  };

  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      created: '已创建',
      queued: '排队中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      cancelled: '已取消',
    };
    return statusMap[status] || status;
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          优化任务列表
        </Typography>
        <Button
          size="small"
          variant="outlined"
          onClick={onRefresh}
          startIcon={<RefreshCw size={16} />}
        >
          刷新
        </Button>
      </Box>

      {tasks.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body2" color="text.secondary">
            暂无优化任务，请创建新任务
          </Typography>
        </Box>
      ) : (
        <Box sx={{ overflowX: 'auto' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>任务名称</TableCell>
                <TableCell>策略</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>进度</TableCell>
                <TableCell>试验数</TableCell>
                <TableCell>最佳得分</TableCell>
                <TableCell>创建时间</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tasks.map((task) => (
                <TableRow key={task.task_id} hover>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {task.task_name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip label={task.strategy_name} size="small" />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={getStatusText(task.status)}
                      color={getStatusColor(task.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ width: 96 }}>
                      <LinearProgress
                        variant="determinate"
                        value={task.progress}
                        color={task.status === 'failed' ? 'error' : 'primary'}
                        sx={{ height: 6, borderRadius: 3 }}
                      />
                    </Box>
                  </TableCell>
                  <TableCell>
                    {task.status === 'completed' ? (
                      <Typography variant="body2">
                        {task.n_trials} / {task.n_trials}
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        - / {task.n_trials}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {task.best_score !== undefined ? (
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'success.main' }}>
                        {task.best_score.toFixed(4)}
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {task.created_at
                        ? new Date(task.created_at).toLocaleString('zh-CN')
                        : '-'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Tooltip title="查看详情">
                      <IconButton
                        size="small"
                        onClick={() => onTaskSelect(task.task_id)}
                      >
                        <Eye size={16} />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}
    </Box>
  );
}
