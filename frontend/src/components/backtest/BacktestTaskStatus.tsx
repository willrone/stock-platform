/**
 * 回测任务状态组件
 * 显示回测任务的状态、进度和基础信息
 */

'use client';

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Chip,
  LinearProgress,
  Divider,
  Button,
  Box,
  Typography,
  Alert,
  IconButton,
} from '@mui/material';
import {
  Clock,
  Calendar,
  Settings,
  AlertTriangle,
  CheckCircle,
  Play,
  Pause,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { Task } from '../../stores/useTaskStore';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface BacktestTaskStatusProps {
  task: Task;
  onRetry?: () => void;
  onStop?: () => void;
  loading?: boolean;
}

export default function BacktestTaskStatus({
  task,
  onRetry,
  onStop,
  loading = false,
}: BacktestTaskStatusProps) {
  const [selectedStocksPage, setSelectedStocksPage] = useState(1); // 已选股票分页
  // 获取状态配置
  const getStatusConfig = (status: Task['status']) => {
    const configs = {
      created: {
        color: 'default' as const,
        text: '已创建',
        icon: <Clock size={16} />,
        description: '任务已创建，等待执行',
      },
      running: {
        color: 'primary' as const,
        text: '运行中',
        icon: <Play size={16} />,
        description: '回测正在执行中，请耐心等待',
      },
      completed: {
        color: 'success' as const,
        text: '已完成',
        icon: <CheckCircle size={16} />,
        description: '回测执行完成，可以查看结果',
      },
      failed: {
        color: 'error' as const,
        text: '执行失败',
        icon: <AlertTriangle size={16} />,
        description: '回测执行失败，请检查配置或重新运行',
      },
    };

    return configs[status] || configs.created;
  };

  // 格式化时间
  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // 计算执行时长
  const getExecutionDuration = () => {
    if (!task.created_at) {
      return null;
    }

    const startTime = new Date(task.created_at);
    const endTime = task.completed_at ? new Date(task.completed_at) : new Date();
    const duration = endTime.getTime() - startTime.getTime();

    const hours = Math.floor(duration / (1000 * 60 * 60));
    const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((duration % (1000 * 60)) / 1000);

    if (hours > 0) {
      return `${hours}小时${minutes}分钟${seconds}秒`;
    } else if (minutes > 0) {
      return `${minutes}分钟${seconds}秒`;
    } else {
      return `${seconds}秒`;
    }
  };

  // 策略名称中英文映射
  const getStrategyDisplayName = (strategyName: string): string => {
    const strategyNameMap: Record<string, string> = {
      // 基础技术分析策略
      moving_average: '移动平均策略',
      rsi: 'RSI策略',
      macd: 'MACD策略',
      // 新增技术分析策略
      bollinger: '布林带策略',
      stochastic: '随机指标策略',
      cci: 'CCI策略',
      // 统计套利策略
      pairs_trading: '配对交易策略',
      mean_reversion: '均值回归策略',
      cointegration: '协整策略',
      // 因子投资策略
      value_factor: '价值因子策略',
      momentum_factor: '动量因子策略',
      low_volatility: '低波动因子策略',
      multi_factor: '多因子组合策略',
    };

    return (
      strategyNameMap[strategyName] ||
      strategyName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    );
  };

  // 获取回测配置信息
  const getBacktestConfig = () => {
    const result = task.result;
    const config = result?.backtest_config || task.config?.backtest_config || {};

    // 获取原始策略名称并转换为中文
    const rawStrategyName =
      config.strategy_name || result?.strategy_name || task.model_id || '默认策略';

    return {
      startDate: config.start_date || result?.start_date || result?.startDate || '未设置',
      endDate: config.end_date || result?.end_date || result?.endDate || '未设置',
      initialCash: config.initial_cash || result?.initial_cash || result?.initialCash || 100000,
      commissionRate: (config.commission_rate || result?.commission_rate || 0.001) * 100,
      slippageRate: (config.slippage_rate || result?.slippage_rate || 0.001) * 100,
      strategyName: getStrategyDisplayName(rawStrategyName),
    };
  };

  const statusConfig = getStatusConfig(task.status);
  const duration = getExecutionDuration();
  const backtestConfig = getBacktestConfig();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 任务状态卡片 */}
      <Card>
        <CardHeader
          title="任务状态"
          titleTypographyProps={{ sx: { fontSize: { xs: '1rem', sm: '1.25rem' } } }}
          action={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {statusConfig.icon}
              <Chip label={statusConfig.text} color={statusConfig.color} size="small" />
            </Box>
          }
        />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* 进度条 */}
          <Box>
            <Box
              sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}
            >
              <Typography variant="body2" color="text.secondary">
                执行进度
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {task.progress}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={task.progress}
              color={task.status === 'failed' ? 'error' : 'primary'}
              sx={{ height: 8, borderRadius: 4, mb: 1 }}
            />
            <Typography variant="body2" color="text.secondary">
              {statusConfig.description}
            </Typography>
          </Box>

          {/* 错误信息 */}
          {task.status === 'failed' && task.error_message && (
            <Alert severity="error" icon={<AlertTriangle size={20} />}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                执行失败
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                {task.error_message}
              </Typography>
            </Alert>
          )}

          {/* 操作按钮 */}
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {task.status === 'failed' && onRetry && (
              <Button
                variant="outlined"
                color="primary"
                startIcon={<RotateCcw size={16} />}
                onClick={onRetry}
                disabled={loading}
                size="small"
                sx={{ minHeight: 44 }}
              >
                重新运行
              </Button>
            )}

            {task.status === 'running' && onStop && (
              <Button
                variant="outlined"
                color="warning"
                startIcon={<Pause size={16} />}
                onClick={onStop}
                disabled={loading}
                size="small"
                sx={{ minHeight: 44 }}
              >
                停止任务
              </Button>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* 任务基础信息 */}
      <Card>
        <CardHeader title="基础信息" titleTypographyProps={{ sx: { fontSize: { xs: '1rem', sm: '1.25rem' } } }} />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(2, 1fr)' },
              gap: 1.5,
            }}
          >
            <Box sx={{ gridColumn: { xs: '1 / -1', md: 'auto' } }}>
              <Typography variant="caption" color="text.secondary">
                任务名称
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 500,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {task.task_name}
              </Typography>
            </Box>

            <Box sx={{ gridColumn: { xs: '1 / -1', md: 'auto' } }}>
              <Typography variant="caption" color="text.secondary">
                任务ID
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  fontFamily: 'monospace',
                  wordBreak: 'break-all',
                  fontSize: { xs: '0.65rem', sm: '0.875rem' },
                  lineHeight: 1.4,
                }}
              >
                {task.task_id}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                策略模型
              </Typography>
              <Chip
                label={backtestConfig.strategyName}
                color="secondary"
                size="small"
                sx={{ mt: 0.5, maxWidth: '100%' }}
              />
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                股票数量
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {task.stock_codes.length} 只
              </Typography>
            </Box>
          </Box>

          <Divider />

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(2, 1fr)' },
              gap: 1.5,
            }}
          >
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <Calendar size={14} />
                <Typography variant="caption" color="text.secondary">
                  创建时间
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ fontWeight: 500, fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
                {formatDateTime(task.created_at)}
              </Typography>
            </Box>

            {task.completed_at && (
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                  <CheckCircle size={14} />
                  <Typography variant="caption" color="text.secondary">
                    完成时间
                  </Typography>
                </Box>
                <Typography variant="body2" sx={{ fontWeight: 500, fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
                  {formatDateTime(task.completed_at)}
                </Typography>
              </Box>
            )}

            {duration && (
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                  <Clock size={14} />
                  <Typography variant="caption" color="text.secondary">
                    执行时长
                  </Typography>
                </Box>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {duration}
                </Typography>
              </Box>
            )}
          </Box>

          <Divider />

          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              选择的股票
            </Typography>
            <Box
              sx={{
                height: { xs: 160, sm: 200 },
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
                p: { xs: 1, sm: 1.5 },
              }}
            >
              {task.stock_codes && task.stock_codes.length > 0 ? (
                <>
                  <Box
                    sx={{
                      flex: 1,
                      overflowY: 'auto',
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: 1,
                      alignContent: 'flex-start',
                      pb: 1,
                    }}
                  >
                    {(() => {
                      const STOCKS_PER_PAGE = 12;
                      const totalPages = Math.ceil(task.stock_codes.length / STOCKS_PER_PAGE);
                      const startIndex = (selectedStocksPage - 1) * STOCKS_PER_PAGE;
                      const endIndex = startIndex + STOCKS_PER_PAGE;
                      const currentStocks = task.stock_codes.slice(startIndex, endIndex);

                      return currentStocks.map(code => (
                        <Chip key={code} label={code} size="small" />
                      ));
                    })()}
                  </Box>

                  {(() => {
                    const STOCKS_PER_PAGE = 12;
                    const totalPages = Math.ceil(task.stock_codes.length / STOCKS_PER_PAGE);

                    if (totalPages > 1) {
                      return (
                        <Box
                          sx={{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            gap: 1,
                            pt: 1,
                            borderTop: '1px solid',
                            borderColor: 'divider',
                          }}
                        >
                          <IconButton
                            size="small"
                            disabled={selectedStocksPage === 1}
                            onClick={() => setSelectedStocksPage(prev => Math.max(1, prev - 1))}
                          >
                            <ChevronLeft size={16} />
                          </IconButton>

                          <Typography variant="caption" color="text.secondary">
                            第 {selectedStocksPage} / {totalPages} 页
                          </Typography>

                          <IconButton
                            size="small"
                            disabled={selectedStocksPage >= totalPages}
                            onClick={() =>
                              setSelectedStocksPage(prev => Math.min(totalPages, prev + 1))
                            }
                          >
                            <ChevronRight size={16} />
                          </IconButton>
                        </Box>
                      );
                    }
                    return null;
                  })()}

                  <Box
                    sx={{
                      pt: 1,
                      mt: 1,
                      borderTop: '1px solid',
                      borderColor: 'divider',
                      display: 'flex',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      已选择 <strong>{task.stock_codes.length}</strong> 只股票
                    </Typography>
                  </Box>
                </>
              ) : (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    暂无选择的股票
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* 回测配置信息 */}
      <Card>
        <CardHeader avatar={<Settings size={24} />} title="回测配置" titleTypographyProps={{ sx: { fontSize: { xs: '1rem', sm: '1.25rem' } } }} />
        <CardContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(2, 1fr)' },
              gap: 1.5,
            }}
          >
            <Box>
              <Typography variant="caption" color="text.secondary">
                回测开始日期
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {backtestConfig.startDate}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                回测结束日期
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {backtestConfig.endDate}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                初始资金
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                ¥{backtestConfig.initialCash.toLocaleString()}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                手续费率
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {backtestConfig.commissionRate.toFixed(3)}%
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                滑点率
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {backtestConfig.slippageRate.toFixed(3)}%
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
