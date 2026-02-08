/**
 * 任务进度显示组件
 */

import React from 'react';
import { Card, CardHeader, CardContent, Box, LinearProgress, Typography } from '@mui/material';
import { AlertTriangle } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';

interface TaskProgressProps {
  task: Task;
}

export function TaskProgress({ task }: TaskProgressProps) {
  return (
    <Card>
      <CardHeader title="任务进度" />
      <CardContent>
        <Box sx={{ mb: 2 }}>
          <LinearProgress
            variant="determinate"
            value={task.progress}
            color={task.status === 'failed' ? 'error' : 'primary'}
            sx={{ height: 10, borderRadius: 5 }}
          />
        </Box>
        {task.task_type === 'hyperparameter_optimization' && task.optimization_info && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              已完成轮次: {task.optimization_info.completed_trials} /{' '}
              {task.optimization_info.n_trials}
            </Typography>
          </Box>
        )}
        {task.status === 'running' && (
          <Typography variant="caption" color="text.secondary">
            任务正在执行中，请耐心等待...
          </Typography>
        )}
        {task.status === 'failed' && task.error_message && (
          <Box
            sx={{
              bgcolor: 'error.light',
              border: 1,
              borderColor: 'error.main',
              borderRadius: 1,
              p: 2,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <AlertTriangle size={20} color="#d32f2f" style={{ marginTop: 2 }} />
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, color: 'error.dark' }}>
                  任务执行失败
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: 'error.dark', mt: 0.5, display: 'block' }}
                >
                  {task.error_message}
                </Typography>
              </Box>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
