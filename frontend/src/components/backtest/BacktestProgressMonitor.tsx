'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardBody, CardHeader } from '@heroui/card';
import { Progress } from '@heroui/progress';
import { Button } from '@heroui/button';
import { Chip } from '@heroui/chip';
import { Divider } from '@heroui/divider';
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from '@heroui/modal';
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ChartBarIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';

import {
  BacktestProgressWebSocket,
  BacktestProgressData,
  BacktestProgressStage,
  BacktestErrorData,
  BacktestCompletionData,
  BacktestCancellationData,
  getBacktestProgressWebSocketManager
} from '../../services/BacktestProgressWebSocket';

interface BacktestProgressMonitorProps {
  taskId: string;
  onComplete?: (results: any) => void;
  onError?: (error: string) => void;
  onCancel?: () => void;
  className?: string;
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

const stageIcons: Record<string, React.ReactNode> = {
  initialization: <PlayIcon className="w-4 h-4" />,
  data_loading: <ChartBarIcon className="w-4 h-4" />,
  strategy_setup: <PlayIcon className="w-4 h-4" />,
  backtest_execution: <PlayIcon className="w-4 h-4" />,
  metrics_calculation: <ChartBarIcon className="w-4 h-4" />,
  report_generation: <ChartBarIcon className="w-4 h-4" />,
  data_storage: <CheckCircleIcon className="w-4 h-4" />
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'success';
    case 'running': return 'primary';
    case 'failed': return 'danger';
    case 'cancelled': return 'warning';
    default: return 'default';
  }
};

const getStatusText = (status: string) => {
  switch (status) {
    case 'pending': return '等待中';
    case 'running': return '运行中';
    case 'completed': return '已完成';
    case 'failed': return '失败';
    case 'cancelled': return '已取消';
    default: return '未知';
  }
};

export default function BacktestProgressMonitor({
  taskId,
  onComplete,
  onError,
  onCancel,
  className = ''
}: BacktestProgressMonitorProps) {
  const [progressData, setProgressData] = useState<BacktestProgressData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [connection, setConnection] = useState<BacktestProgressWebSocket | null>(null);
  
  const { isOpen: isCancelModalOpen, onOpen: onCancelModalOpen, onClose: onCancelModalClose } = useDisclosure();

  // 格式化时间
  const formatDuration = useCallback((durationStr: string | undefined) => {
    if (!durationStr) return '--';
    
    try {
      // 解析类似 "0:05:23.456789" 的格式
      const parts = durationStr.split(':');
      if (parts.length >= 3) {
        const hours = parseInt(parts[0]);
        const minutes = parseInt(parts[1]);
        const seconds = Math.floor(parseFloat(parts[2]));
        
        if (hours > 0) {
          return `${hours}小时${minutes}分${seconds}秒`;
        } else if (minutes > 0) {
          return `${minutes}分${seconds}秒`;
        } else {
          return `${seconds}秒`;
        }
      }
      return durationStr;
    } catch {
      return durationStr;
    }
  }, []);

  // 格式化日期时间
  const formatDateTime = useCallback((dateTimeStr: string | undefined) => {
    if (!dateTimeStr) return '--';
    
    try {
      const date = new Date(dateTimeStr);
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return dateTimeStr;
    }
  }, []);

  // 格式化数值
  const formatNumber = useCallback((value: number, decimals: number = 2) => {
    if (typeof value !== 'number' || isNaN(value)) return '--';
    return value.toLocaleString('zh-CN', { 
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals 
    });
  }, []);

  // 初始化WebSocket连接
  useEffect(() => {
    const manager = getBacktestProgressWebSocketManager();
    
    const initConnection = async () => {
      try {
        const conn = await manager.connect(taskId, {
          onProgress: (data: BacktestProgressData) => {
            setProgressData(data);
            setError(null);
          },
          onError: (errorData: BacktestErrorData) => {
            setError(errorData.error_message);
            onError?.(errorData.error_message);
          },
          onCompletion: (completionData: BacktestCompletionData) => {
            onComplete?.(completionData.results);
          },
          onCancellation: (cancellationData: BacktestCancellationData) => {
            onCancel?.();
          },
          onConnection: (connected: boolean) => {
            setIsConnected(connected);
          }
        });
        
        setConnection(conn);
      } catch (error) {
        console.error('连接回测进度WebSocket失败:', error);
        setError('连接失败，请刷新页面重试');
      }
    };

    initConnection();

    return () => {
      manager.disconnect(taskId);
    };
  }, [taskId, onComplete, onError, onCancel]);

  // 取消回测
  const handleCancelBacktest = useCallback(() => {
    if (connection) {
      connection.cancelBacktest('用户手动取消');
      onCancelModalClose();
    }
  }, [connection, onCancelModalClose]);

  // 渲染阶段进度
  const renderStageProgress = useCallback((stage: BacktestProgressStage) => {
    return (
      <div key={stage.name} className="flex items-center gap-3 p-3 rounded-lg bg-content2">
        <div className="flex-shrink-0">
          {stageIcons[stage.name] || <ClockIcon className="w-4 h-4" />}
        </div>
        
        <div className="flex-grow min-w-0">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium truncate">
              {stageDisplayNames[stage.name] || stage.description}
            </span>
            <Chip 
              size="sm" 
              color={getStatusColor(stage.status)}
              variant="flat"
            >
              {getStatusText(stage.status)}
            </Chip>
          </div>
          
          {stage.status === 'running' && (
            <Progress 
              value={stage.progress} 
              size="sm"
              color="primary"
              className="mb-1"
            />
          )}
          
          {stage.details && Object.keys(stage.details).length > 0 && (
            <div className="text-xs text-default-500 space-y-1">
              {stage.details.processed_days && stage.details.total_days && (
                <div>进度: {stage.details.processed_days}/{stage.details.total_days} 天</div>
              )}
              {stage.details.current_date && (
                <div>当前日期: {stage.details.current_date}</div>
              )}
              {stage.details.signals_generated !== undefined && (
                <div>信号数: {stage.details.signals_generated}</div>
              )}
              {stage.details.trades_executed !== undefined && (
                <div>交易数: {stage.details.trades_executed}</div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }, []);

  if (!progressData && !error) {
    return (
      <Card className={className}>
        <CardBody className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-default-500">连接回测进度监控中...</p>
        </CardBody>
      </Card>
    );
  }

  return (
    <>
      <Card className={className}>
        <CardHeader className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-semibold">回测进度监控</h3>
            <p className="text-sm text-default-500">
              连接状态: {isConnected ? '已连接' : '未连接'}
            </p>
          </div>
          
          {progressData && progressData.overall_progress < 100 && !error && (
            <Button
              color="danger"
              variant="light"
              size="sm"
              startContent={<StopIcon className="w-4 h-4" />}
              onPress={onCancelModalOpen}
            >
              取消回测
            </Button>
          )}
        </CardHeader>

        <CardBody className="space-y-6">
          {error && (
            <div className="flex items-center gap-2 p-3 bg-danger-50 border border-danger-200 rounded-lg">
              <ExclamationTriangleIcon className="w-5 h-5 text-danger-500 flex-shrink-0" />
              <span className="text-danger-700 text-sm">{error}</span>
            </div>
          )}

          {progressData && (
            <>
              {/* 总体进度 */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="font-medium">总体进度</span>
                  <span className="text-sm text-default-500">
                    {formatNumber(progressData.overall_progress, 1)}%
                  </span>
                </div>
                <Progress 
                  value={progressData.overall_progress} 
                  color="primary"
                  size="lg"
                  className="mb-2"
                />
                <div className="text-sm text-default-500">
                  当前阶段: {stageDisplayNames[progressData.current_stage] || progressData.current_stage}
                </div>
              </div>

              <Divider />

              {/* 执行统计 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary">
                    {progressData.processed_days}
                  </div>
                  <div className="text-xs text-default-500">已处理天数</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-success">
                    {progressData.trades_executed}
                  </div>
                  <div className="text-xs text-default-500">执行交易</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-warning">
                    {progressData.signals_generated}
                  </div>
                  <div className="text-xs text-default-500">生成信号</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-secondary">
                    {formatNumber(progressData.portfolio_value, 0)}
                  </div>
                  <div className="text-xs text-default-500">组合价值</div>
                </div>
              </div>

              {/* 时间信息 */}
              {(progressData.elapsed_time || progressData.estimated_completion || progressData.processing_speed > 0) && (
                <>
                  <Divider />
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    {progressData.elapsed_time && (
                      <div>
                        <span className="text-default-500">已用时间: </span>
                        <span className="font-medium">{formatDuration(progressData.elapsed_time)}</span>
                      </div>
                    )}
                    
                    {progressData.estimated_completion && (
                      <div>
                        <span className="text-default-500">预计完成: </span>
                        <span className="font-medium">{formatDateTime(progressData.estimated_completion)}</span>
                      </div>
                    )}
                    
                    {progressData.processing_speed > 0 && (
                      <div>
                        <span className="text-default-500">处理速度: </span>
                        <span className="font-medium">{formatNumber(progressData.processing_speed, 2)} 天/秒</span>
                      </div>
                    )}
                  </div>
                </>
              )}

              {/* 当前处理信息 */}
              {progressData.current_date && (
                <>
                  <Divider />
                  <div className="text-sm">
                    <span className="text-default-500">当前处理日期: </span>
                    <span className="font-medium">{progressData.current_date}</span>
                  </div>
                </>
              )}

              {/* 警告信息 */}
              {progressData.warnings_count > 0 && (
                <>
                  <Divider />
                  <div className="flex items-center gap-2">
                    <ExclamationTriangleIcon className="w-4 h-4 text-warning-500" />
                    <span className="text-sm text-warning-600">
                      发现 {progressData.warnings_count} 个警告
                    </span>
                  </div>
                </>
              )}

              {/* 阶段详情 */}
              <Divider />
              <div className="space-y-3">
                <h4 className="font-medium">执行阶段</h4>
                <div className="space-y-2">
                  {progressData.stages.map(renderStageProgress)}
                </div>
              </div>
            </>
          )}
        </CardBody>
      </Card>

      {/* 取消确认模态框 */}
      <Modal isOpen={isCancelModalOpen} onClose={onCancelModalClose}>
        <ModalContent>
          <ModalHeader>确认取消回测</ModalHeader>
          <ModalBody>
            <p>确定要取消当前的回测任务吗？</p>
            <p className="text-sm text-default-500">
              取消后将无法恢复当前的计算进度，需要重新开始回测。
            </p>
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={onCancelModalClose}>
              继续回测
            </Button>
            <Button color="danger" onPress={handleCancelBacktest}>
              确认取消
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}