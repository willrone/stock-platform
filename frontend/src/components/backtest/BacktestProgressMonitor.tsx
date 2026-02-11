'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  LinearProgress,
  Button,
  Chip,
  Divider,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Box,
  Typography,
  CircularProgress,
} from '@mui/material';
import {
  PlayIcon,
  StopIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';

import {
  BacktestProgressWebSocket,
  BacktestProgressData,
  BacktestProgressStage,
  BacktestErrorData,
  BacktestCompletionData,
  BacktestCancellationData,
  getBacktestProgressWebSocketManager,
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
  data_storage: '数据存储',
};

const stageIcons: Record<string, React.ReactNode> = {
  initialization: <PlayIcon className="w-4 h-4" />,
  data_loading: <ChartBarIcon className="w-4 h-4" />,
  strategy_setup: <PlayIcon className="w-4 h-4" />,
  backtest_execution: <PlayIcon className="w-4 h-4" />,
  metrics_calculation: <ChartBarIcon className="w-4 h-4" />,
  report_generation: <ChartBarIcon className="w-4 h-4" />,
  data_storage: <CheckCircleIcon className="w-4 h-4" />,
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'success';
    case 'running':
      return 'primary';
    case 'failed':
      return 'error';
    case 'cancelled':
      return 'warning';
    default:
      return 'default';
  }
};

const getStatusText = (status: string) => {
  switch (status) {
    case 'pending':
      return '等待中';
    case 'running':
      return '运行中';
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
    case 'cancelled':
      return '已取消';
    default:
      return '未知';
  }
};

export default function BacktestProgressMonitor({
  taskId,
  onComplete,
  onError,
  onCancel,
  className = '',
}: BacktestProgressMonitorProps) {
  const [progressData, setProgressData] = useState<BacktestProgressData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [, setWarnings] = useState<string[]>([]);
  const [connection, setConnection] = useState<BacktestProgressWebSocket | null>(null);

  const [isCancelModalOpen, setIsCancelModalOpen] = useState(false);
  const onCancelModalOpen = () => setIsCancelModalOpen(true);
  const onCancelModalClose = () => setIsCancelModalOpen(false);

  // 格式化时间
  const formatDuration = useCallback((durationStr: string | undefined) => {
    if (!durationStr) {
      return '--';
    }

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
    if (!dateTimeStr) {
      return '--';
    }

    try {
      const date = new Date(dateTimeStr);
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return dateTimeStr;
    }
  }, []);

  // 格式化数值
  const formatNumber = useCallback((value: number, decimals: number = 2) => {
    if (typeof value !== 'number' || isNaN(value)) {
      return '--';
    }
    return value.toLocaleString('zh-CN', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
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
          onCancellation: (_cancellationData: BacktestCancellationData) => {
            onCancel?.();
          },
          onConnection: (connected: boolean) => {
            setIsConnected(connected);
          },
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
      <Box
        key={stage.name}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          p: 1.5,
          borderRadius: 1,
          bgcolor: 'grey.50',
        }}
      >
        <Box sx={{ flexShrink: 0 }}>
          {stageIcons[stage.name] || <ClockIcon className="w-4 h-4" />}
        </Box>

        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Box
            sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}
          >
            <Typography
              variant="body2"
              sx={{
                fontWeight: 500,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {stageDisplayNames[stage.name] || stage.description}
            </Typography>
            <Chip
              label={getStatusText(stage.status)}
              size="small"
              color={getStatusColor(stage.status) as any}
            />
          </Box>

          {stage.status === 'running' && (
            <Box sx={{ mb: 0.5 }}>
              <LinearProgress
                variant="determinate"
                value={stage.progress}
                color="primary"
                sx={{ height: 6, borderRadius: 3 }}
              />
            </Box>
          )}

          {stage.details && Object.keys(stage.details).length > 0 && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {stage.details.processed_days && stage.details.total_days && (
                <Typography variant="caption" color="text.secondary">
                  进度: {stage.details.processed_days}/{stage.details.total_days} 天
                </Typography>
              )}
              {stage.details.current_date && (
                <Typography variant="caption" color="text.secondary">
                  当前日期: {stage.details.current_date}
                </Typography>
              )}
              {stage.details.signals_generated !== undefined && (
                <Typography variant="caption" color="text.secondary">
                  信号数: {stage.details.signals_generated}
                </Typography>
              )}
              {stage.details.trades_executed !== undefined && (
                <Typography variant="caption" color="text.secondary">
                  交易数: {stage.details.trades_executed}
                </Typography>
              )}
            </Box>
          )}
        </Box>
      </Box>
    );
  }, []);

  if (!progressData && !error) {
    return (
      <Card className={className}>
        <CardContent sx={{ textAlign: 'center', py: 4 }}>
          <CircularProgress size={32} sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            连接回测进度监控中...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card className={className}>
        <CardHeader
          title={
            <Box>
              <Typography variant="h6" component="h3" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                回测进度监控
              </Typography>
              <Typography variant="caption" color="text.secondary">
                连接状态: {isConnected ? '已连接' : '未连接'}
              </Typography>
            </Box>
          }
          action={
            progressData &&
            progressData.overall_progress < 100 &&
            !error && (
              <Button
                variant="outlined"
                color="error"
                size="small"
                startIcon={<StopIcon className="w-4 h-4" />}
                onClick={onCancelModalOpen}
              >
                取消回测
              </Button>
            )
          }
        />

        <CardContent sx={{ p: { xs: 1.5, sm: 2, md: 3 } }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {error && (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  p: 1.5,
                  bgcolor: 'error.light',
                  border: 1,
                  borderColor: 'error.main',
                  borderRadius: 1,
                }}
              >
                <ExclamationTriangleIcon
                  className="w-5 h-5"
                  style={{ color: '#d32f2f', flexShrink: 0 }}
                />
                <Typography variant="body2" sx={{ color: 'error.dark' }}>
                  {error}
                </Typography>
              </Box>
            )}

            {progressData && (
              <>
                {/* 总体进度 */}
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                  <Box
                    sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      总体进度
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {formatNumber(progressData.overall_progress, 1)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={progressData.overall_progress}
                    color="primary"
                    sx={{ height: 10, borderRadius: 5, mb: 1 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    当前阶段:{' '}
                    {stageDisplayNames[progressData.current_stage] || progressData.current_stage}
                  </Typography>
                </Box>

                <Divider />

                {/* 执行统计 */}
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                    gap: 2,
                  }}
                >
                  <Box sx={{ textAlign: 'center', overflow: 'hidden', wordBreak: 'break-word' }}>
                    <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main', fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' } }}>
                      {progressData.processed_days}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      已处理天数
                    </Typography>
                  </Box>

                  <Box sx={{ textAlign: 'center', overflow: 'hidden', wordBreak: 'break-word' }}>
                    <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' } }}>
                      {progressData.trades_executed}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      执行交易
                    </Typography>
                  </Box>

                  <Box sx={{ textAlign: 'center', overflow: 'hidden', wordBreak: 'break-word' }}>
                    <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main', fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' } }}>
                      {progressData.signals_generated}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      生成信号
                    </Typography>
                  </Box>

                  <Box sx={{ textAlign: 'center', overflow: 'hidden', wordBreak: 'break-word' }}>
                    <Typography variant="h4" sx={{ fontWeight: 600, color: 'secondary.main', fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' } }}>
                      {formatNumber(progressData.portfolio_value, 0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      组合价值
                    </Typography>
                  </Box>
                </Box>

                {/* 时间信息 */}
                {(progressData.elapsed_time ||
                  progressData.estimated_completion ||
                  progressData.processing_speed > 0) && (
                  <>
                    <Divider />
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
                        gap: 2,
                      }}
                    >
                      {progressData.elapsed_time && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            已用时间:{' '}
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {formatDuration(progressData.elapsed_time)}
                          </Typography>
                        </Box>
                      )}

                      {progressData.estimated_completion && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            预计完成:{' '}
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {formatDateTime(progressData.estimated_completion)}
                          </Typography>
                        </Box>
                      )}

                      {progressData.processing_speed > 0 && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            处理速度:{' '}
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {formatNumber(progressData.processing_speed, 2)} 天/秒
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </>
                )}

                {/* 当前处理信息 */}
                {progressData.current_date && (
                  <>
                    <Divider />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        当前处理日期:{' '}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {progressData.current_date}
                      </Typography>
                    </Box>
                  </>
                )}

                {/* 警告信息 */}
                {progressData.warnings_count > 0 && (
                  <>
                    <Divider />
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <ExclamationTriangleIcon className="w-4 h-4" style={{ color: '#ed6c02' }} />
                      <Typography variant="body2" sx={{ color: 'warning.dark' }}>
                        发现 {progressData.warnings_count} 个警告
                      </Typography>
                    </Box>
                  </>
                )}

                {/* 阶段详情 */}
                <Divider />
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                    执行阶段
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {progressData.stages.map(renderStageProgress)}
                  </Box>
                </Box>
              </>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* 取消确认模态框 */}
      <Dialog open={isCancelModalOpen} onClose={onCancelModalClose}>
        <DialogTitle>确认取消回测</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 1 }}>
            确定要取消当前的回测任务吗？
          </Typography>
          <Typography variant="caption" color="text.secondary">
            取消后将无法恢复当前的计算进度，需要重新开始回测。
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button variant="outlined" onClick={onCancelModalClose}>
            继续回测
          </Button>
          <Button variant="contained" color="error" onClick={handleCancelBacktest}>
            确认取消
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
