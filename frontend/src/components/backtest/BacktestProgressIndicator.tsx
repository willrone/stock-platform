'use client';

import React, { useState, useEffect } from 'react';
import { Progress } from '@heroui/progress';
import { Chip } from '@heroui/chip';
import { Tooltip } from '@heroui/tooltip';
import { 
  PlayIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  StopIcon
} from '@heroicons/react/24/outline';

import {
  BacktestProgressData,
  getBacktestProgressWebSocketManager
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
  data_storage: '数据存储'
};

export default function BacktestProgressIndicator({
  taskId,
  taskStatus,
  className = '',
  showDetails = false
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
          onError: (errorData) => {
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
          }
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
      <div className={`flex items-center gap-2 ${className}`}>
        <CheckCircleIcon className="w-4 h-4 text-success-500" />
        <Chip size="sm" color="success" variant="flat">
          已完成
        </Chip>
      </div>
    );
  }

  if (taskStatus === 'failed') {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <ExclamationTriangleIcon className="w-4 h-4 text-danger-500" />
        <Chip size="sm" color="danger" variant="flat">
          失败
        </Chip>
      </div>
    );
  }

  if (taskStatus === 'cancelled') {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <StopIcon className="w-4 h-4 text-warning-500" />
        <Chip size="sm" color="warning" variant="flat">
          已取消
        </Chip>
      </div>
    );
  }

  if (!shouldConnect) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <Chip size="sm" color="default" variant="flat">
          {taskStatus}
        </Chip>
      </div>
    );
  }

  // 显示错误状态
  if (error) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <ExclamationTriangleIcon className="w-4 h-4 text-danger-500" />
        <Tooltip content={error}>
          <Chip size="sm" color="danger" variant="flat">
            连接错误
          </Chip>
        </Tooltip>
      </div>
    );
  }

  // 显示进度信息
  if (progressData) {
    const currentStageDisplay = stageDisplayNames[progressData.current_stage] || progressData.current_stage;
    
    if (showDetails) {
      return (
        <div className={`space-y-2 ${className}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <PlayIcon className="w-4 h-4 text-primary-500 animate-pulse" />
              <span className="text-sm font-medium">运行中</span>
            </div>
            <span className="text-xs text-default-500">
              {progressData.overall_progress.toFixed(1)}%
            </span>
          </div>
          
          <Progress 
            value={progressData.overall_progress} 
            size="sm"
            color="primary"
          />
          
          <div className="flex justify-between text-xs text-default-500">
            <span>{currentStageDisplay}</span>
            {progressData.processed_days > 0 && progressData.total_days > 0 && (
              <span>{progressData.processed_days}/{progressData.total_days} 天</span>
            )}
          </div>
          
          {progressData.current_date && (
            <div className="text-xs text-default-400">
              当前: {progressData.current_date}
            </div>
          )}
        </div>
      );
    } else {
      // 简化显示
      return (
        <div className={`flex items-center gap-2 ${className}`}>
          <PlayIcon className="w-4 h-4 text-primary-500 animate-pulse" />
          <Tooltip 
            content={
              <div className="space-y-1">
                <div>进度: {progressData.overall_progress.toFixed(1)}%</div>
                <div>阶段: {currentStageDisplay}</div>
                {progressData.processed_days > 0 && progressData.total_days > 0 && (
                  <div>处理: {progressData.processed_days}/{progressData.total_days} 天</div>
                )}
                {progressData.current_date && (
                  <div>当前日期: {progressData.current_date}</div>
                )}
              </div>
            }
          >
            <Chip size="sm" color="primary" variant="flat">
              运行中 {progressData.overall_progress.toFixed(0)}%
            </Chip>
          </Tooltip>
        </div>
      );
    }
  }

  // 连接中或等待状态
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="w-4 h-4 border-2 border-primary-200 border-t-primary-500 rounded-full animate-spin"></div>
      <Chip size="sm" color="primary" variant="flat">
        {isConnected ? '准备中' : '连接中'}
      </Chip>
    </div>
  );
}