'use client';

import React, { useState, useEffect } from 'react';
import { LinearProgress, Chip, Tooltip, Box, Typography, CircularProgress } from '@mui/material';
import {
  PlayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  StopIcon,
} from '@heroicons/react/24/outline';

import {
  BacktestProgressData,
  getBacktestProgressWebSocketManager,
} from '../../services/BacktestProgressWebSocket';

interface BacktestProgressIndicatorProps {
  taskId: string;
  taskStatus: string;
  className?: string;
  showDetails?: boolean;
}

const stageDisplayNames: Record<string, string> = {
  initialization: '初始化',
  data_loading: '数据加载',
  strategy_setup: '策略设置',
  backtest_execution: '回测执行',
  metrics_calculation: '指标计算',
  report_generation: '报告生成',
  data_storage: '数据存储',
};

export default function BacktestProgressIndicator({
  taskId,
  taskStatus,
  className = '',
  showDetails = false,
}: BacktestProgressIndicatorProps) {
  const [progressData, setProgressData] = useState<BacktestProgressData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 只在任务运行中时连接WebSocket
  const shouldConnect = taskStatus === 'running' || taskStatus === 'pending';

  useEffect(() => {
    if (!shouldConnect) {
      return;
    }

    const manager = getBacktestProgressWebSocketManager();

    const initConnection = async () => {
      try {
        await manager.connect(taskId, {
          onProgress: (data: BacktestProgressData) => {
            setProgressData(data);
            setError(null);
          },
          onError: errorData => {
            setError(errorData.error_message);
          },
          onCompletion: () => {
            // 任务完成，断开连接
            manager.disconnect(taskId);
          },
          onCancellation: () => {
            // 任务取消，断开连接
            manager.disconnect(taskId);
          },
          onConnection: (connected: boolean) => {
            setIsConnected(connected);
          },
        });
      } catch (error) {
        console.error('连接回测进度WebSocket失败:', error);
        setError('连接失败');
      }
    };

    initConnection();

    return () => {
      manager.disconnect(taskId);
    };
  }, [taskId, shouldConnect]);

  // 根据任务状态显示不同的内容
  if (taskStatus === 'completed') {
    return (
      <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CheckCircleIcon className="w-4 h-4" style={{ color: '#2e7d32' }} />
        <Chip label="已完成" size="small" color="success" />
      </Box>
    );
  }

  if (taskStatus === 'failed') {
    return (
      <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ExclamationTriangleIcon className="w-4 h-4" style={{ color: '#d32f2f' }} />
        <Chip label="失败" size="small" color="error" />
      </Box>
    );
  }

  if (taskStatus === 'cancelled') {
    return (
      <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <StopIcon className="w-4 h-4" style={{ color: '#ed6c02' }} />
        <Chip label="已取消" size="small" color="warning" />
      </Box>
    );
  }

  if (!shouldConnect) {
    return (
      <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Chip label={taskStatus} size="small" />
      </Box>
    );
  }

  // 显示错误状态
  if (error) {
    return (
      <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ExclamationTriangleIcon className="w-4 h-4" style={{ color: '#d32f2f' }} />
        <Tooltip title={error}>
          <Chip label="连接错误" size="small" color="error" />
        </Tooltip>
      </Box>
    );
  }

  // 显示进度信息
  if (progressData) {
    const currentStageDisplay =
      stageDisplayNames[progressData.current_stage] || progressData.current_stage;

    if (showDetails) {
      return (
        <Box className={className} sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PlayIcon className="w-4 h-4" style={{ color: '#1976d2' }} />
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                运行中
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              {progressData.overall_progress.toFixed(1)}%
            </Typography>
          </Box>

          <LinearProgress
            variant="determinate"
            value={progressData.overall_progress}
            color="primary"
            sx={{ height: 6, borderRadius: 3 }}
          />

          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" color="text.secondary">
              {currentStageDisplay}
            </Typography>
            {progressData.processed_days > 0 && progressData.total_days > 0 && (
              <Typography variant="caption" color="text.secondary">
                {progressData.processed_days}/{progressData.total_days} 天
              </Typography>
            )}
          </Box>

          {progressData.current_date && (
            <Typography variant="caption" color="text.secondary">
              当前: {progressData.current_date}
            </Typography>
          )}
        </Box>
      );
    } else {
      // 简化显示
      return (
        <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PlayIcon className="w-4 h-4" style={{ color: '#1976d2' }} />
          <Tooltip
            title={
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                <Typography variant="caption">
                  进度: {progressData.overall_progress.toFixed(1)}%
                </Typography>
                <Typography variant="caption">阶段: {currentStageDisplay}</Typography>
                {progressData.processed_days > 0 && progressData.total_days > 0 && (
                  <Typography variant="caption">
                    处理: {progressData.processed_days}/{progressData.total_days} 天
                  </Typography>
                )}
                {progressData.current_date && (
                  <Typography variant="caption">当前日期: {progressData.current_date}</Typography>
                )}
              </Box>
            }
          >
            <Chip
              label={`运行中 ${progressData.overall_progress.toFixed(0)}%`}
              size="small"
              color="primary"
            />
          </Tooltip>
        </Box>
      );
    }
  }

  // 连接中或等待状态
  return (
    <Box className={className} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <CircularProgress size={16} />
      <Chip label={isConnected ? '准备中' : '连接中'} size="small" color="primary" />
    </Box>
  );
}
