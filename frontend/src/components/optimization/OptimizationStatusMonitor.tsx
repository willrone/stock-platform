/**
 * 优化任务状态监控组件
 */

'use client';

import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  LinearProgress,
  Chip,
  Box,
  Typography,
} from '@mui/material';
import { OptimizationStatus } from '../../services/optimizationService';

interface OptimizationStatusMonitorProps {
  status: OptimizationStatus;
  task: any;
}

export default function OptimizationStatusMonitor({
  status,
  task,
}: OptimizationStatusMonitorProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
          gap: 2,
        }}
      >
        <Card>
          <CardContent>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                {status.completed_trials || 0} / {status.n_trials || 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                已完成试验
              </Typography>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                {status.running_trials || 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                运行中
              </Typography>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                {status.pruned_trials || 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                已剪枝
              </Typography>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                {status.failed_trials || 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                失败
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>

      <Card>
        <CardHeader title="优化进度" />
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 1,
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  {`${(status.progress || 0).toFixed(1)}%`}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={status.progress || 0}
                color={getStatusColor(status.status) as any}
                sx={{ height: 10, borderRadius: 5 }}
              />
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" color="text.secondary">
                状态:
              </Typography>
              <Chip
                label={status.status}
                color={getStatusColor(status.status) as any}
                size="small"
              />
            </Box>
          </Box>
        </CardContent>
      </Card>

      {status.best_score !== undefined && status.best_score !== null && (
        <Card>
          <CardHeader title="最佳结果" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  最佳得分:
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                  {(status.best_score || 0).toFixed(4)}
                </Typography>
              </Box>
              {status.best_trial_number !== undefined && (
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    最佳试验编号:
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    #{status.best_trial_number}
                  </Typography>
                </Box>
              )}
              {status.best_params && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                    最佳参数:
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 1 }}>
                    {Object.entries(status.best_params).map(([key, value]) => (
                      <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="caption" color="text.secondary">
                          {key}:
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 500 }}>
                          {String(value)}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
