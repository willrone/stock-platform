'use client';

import React from 'react';
import { Card, CardContent, Box, Typography, Chip, LinearProgress, Button } from '@mui/material';
import { Sparkles, TrendingUp, Calendar } from 'lucide-react';

interface OptimizationTask {
  task_id: string;
  task_name: string;
  status: string;
  progress?: number;
  model_id?: string;
  best_score?: number;
  created_at: string;
  search_space?: any;
}

interface MobileOptimizationCardProps {
  task: OptimizationTask;
  onViewDetails: (id: string) => void;
}

export const MobileOptimizationCard: React.FC<MobileOptimizationCardProps> = ({
  task,
  onViewDetails,
}) => {
  const getStatusColor = (
    status: string
  ): 'primary' | 'success' | 'error' | 'warning' | 'default' => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running':
        return '运行中';
      case 'completed':
        return '已完成';
      case 'failed':
        return '失败';
      case 'pending':
        return '待运行';
      default:
        return status;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Card
      sx={{
        mb: 2,
        borderRadius: 3,
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        transition: 'all 0.2s',
      }}
    >
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        {/* 标题行 */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            mb: 1.5,
          }}
        >
          <Box sx={{ flex: 1, pr: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
              <Sparkles size={16} color="#9c27b0" />
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  fontSize: '1.05rem',
                  lineHeight: 1.3,
                }}
              >
                {task.task_name}
              </Typography>
            </Box>
            {task.model_id && (
              <Typography variant="caption" color="text.secondary">
                模型: {task.model_id}
              </Typography>
            )}
          </Box>
          <Chip
            label={getStatusText(task.status)}
            color={getStatusColor(task.status)}
            size="small"
            sx={{
              fontWeight: 600,
              fontSize: '0.75rem',
            }}
          />
        </Box>

        {/* 最佳得分 */}
        {task.best_score != null && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
            <TrendingUp size={16} color="#4caf50" />
            <Typography variant="body2" color="text.secondary">
              最佳得分:
            </Typography>
            <Typography variant="body2" fontWeight={600} color="success.main">
              {task.best_score.toFixed(4)}
            </Typography>
          </Box>
        )}

        {/* 时间 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1.5 }}>
          <Calendar size={14} color="#666" />
          <Typography variant="body2" color="text.secondary">
            {formatDate(task.created_at)}
          </Typography>
        </Box>

        {/* 进度条 */}
        {task.status === 'running' && task.progress != null && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                优化进度
              </Typography>
              <Typography variant="caption" fontWeight={600} color="primary">
                {task.progress}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={task.progress}
              sx={{
                height: 6,
                borderRadius: 3,
                bgcolor: 'action.hover',
              }}
            />
          </Box>
        )}

        {/* 查看详情按钮 */}
        <Button
          variant="outlined"
          size="medium"
          fullWidth
          onClick={() => onViewDetails(task.task_id)}
          sx={{
            borderRadius: 2,
            minHeight: 44,
            textTransform: 'none',
            fontWeight: 500,
          }}
        >
          查看详情
        </Button>
      </CardContent>
    </Card>
  );
};
