'use client';

import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Chip,
  LinearProgress,
  IconButton,
  Button,
} from '@mui/material';
import { Eye, Trash2, TrendingUp, Calendar } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface BacktestTask {
  task_id: string;
  task_name: string;
  status: string;
  progress?: number;
  stock_codes?: string[];
  config?: any;
  created_at: string;
}

interface MobileBacktestCardProps {
  task: BacktestTask;
  onDelete: (id: string) => void;
}

export const MobileBacktestCard: React.FC<MobileBacktestCardProps> = ({ task, onDelete }) => {
  const router = useRouter();

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
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) {
      return `${diffMins}分钟前`;
    } else if (diffHours < 24) {
      return `${diffHours}小时前`;
    } else if (diffDays < 7) {
      return `${diffDays}天前`;
    } else {
      return date.toLocaleDateString('zh-CN', {
        month: 'short',
        day: 'numeric',
      });
    }
  };

  const getStrategyName = () => {
    return (
      task.config?.strategy_name ||
      task.config?.strategy_type ||
      task.config?.backtest_config?.strategy_name ||
      '未知策略'
    );
  };

  const getBacktestPeriod = () => {
    const config = task.config?.backtest_config || task.config;
    if (config?.start_date && config?.end_date) {
      const start = config.start_date.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3');
      const end = config.end_date.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3');
      return `${start} ~ ${end}`;
    }
    return '-';
  };

  return (
    <Card
      sx={{
        mb: 2,
        borderRadius: 3,
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        transition: 'all 0.2s',
        '&:active': {
          transform: 'scale(0.98)',
        },
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
          <Typography
            variant="h6"
            sx={{
              fontWeight: 600,
              fontSize: '1.1rem',
              flex: 1,
              pr: 1,
              lineHeight: 1.3,
            }}
          >
            {task.task_name}
          </Typography>
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

        {/* 策略信息 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
          <TrendingUp size={16} color="#666" />
          <Typography variant="body2" color="text.secondary">
            策略:
          </Typography>
          <Typography variant="body2" fontWeight={500}>
            {getStrategyName()}
          </Typography>
        </Box>

        {/* 回测期间 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
          <Calendar size={16} color="#666" />
          <Typography variant="body2" color="text.secondary">
            {getBacktestPeriod()}
          </Typography>
        </Box>

        {/* 股票数量和时间 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
          <Typography variant="body2" color="text.secondary">
            {task.stock_codes?.length || 0} 只股票
          </Typography>
          <Typography variant="body2" color="text.secondary">
            •
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {formatDate(task.created_at)}
          </Typography>
        </Box>

        {/* 进度条 */}
        {task.status === 'running' && task.progress != null && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                进度
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

        {/* 操作按钮 */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="medium"
            startIcon={<Eye size={18} />}
            onClick={() => router.push(`/backtest/${task.task_id}`)}
            sx={{
              flex: 1,
              borderRadius: 2,
              minHeight: 44,
              textTransform: 'none',
              fontWeight: 500,
            }}
          >
            查看详情
          </Button>

          <IconButton
            size="medium"
            onClick={e => {
              e.stopPropagation();
              onDelete(task.task_id);
            }}
            sx={{
              border: 1,
              borderColor: 'error.main',
              color: 'error.main',
              borderRadius: 2,
              minHeight: 44,
              minWidth: 44,
              '&:hover': {
                bgcolor: 'error.light',
                borderColor: 'error.dark',
              },
            }}
          >
            <Trash2 size={20} />
          </IconButton>
        </Box>
      </CardContent>
    </Card>
  );
};
