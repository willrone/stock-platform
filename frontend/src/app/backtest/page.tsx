'use client';

/**
 * 策略回测页面
 *
 * 显示回测任务列表和管理界面
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
  Box,
  Typography,
  IconButton,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  LinearProgress,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  Plus,
  RefreshCw,
  Trash2,
  Eye,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { Task } from '../../stores/useTaskStore';
import { TaskService } from '../../services/taskService';
import { wsService } from '../../services/websocket';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { MobileBacktestCard } from '../../components/mobile/MobileBacktestCard';
import { useSnackbarStore } from '../../stores/useSnackbarStore';

export default function BacktestPage() {
  const router = useRouter();
  const showSnackbar = useSnackbarStore();
  /**
   * 【state P0-8】服务端回测任务列表由本页本地持有，不放全局 store，
   * 避免与 /tasks 页互相污染。
   */
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  /** 局部更新单条任务（WebSocket 实时进度用） */
  const updateTask = useCallback((taskId: string, patch: Partial<Task>) => {
    setTasks(prev => prev.map(t => (t.task_id === taskId ? { ...t, ...patch } : t)));
  }, []);

  // 过滤出回测任务
  const backtestTasks = tasks.filter(task => {
    // 优先使用 task_type 字段
    if (task.task_type === 'backtest') {
      return true;
    }
    // 如果没有 task_type，通过 config 判断（兼容旧数据）
    if (task.config?.backtest_config || task.config?.strategy_name) {
      return true;
    }
    // 如果 result 中包含回测相关字段，也认为是回测任务
    if (task.result || task.backtest_results || task.results?.backtest_results) {
      const result = task.result || task.backtest_results || task.results?.backtest_results;
      if (result && typeof result === 'object') {
        const backtestKeys = [
          'equity_curve',
          'drawdown_curve',
          'portfolio',
          'risk_metrics',
          'trade_history',
          'dates',
        ];
        return backtestKeys.some(key => key in result);
      }
    }
    return false;
  });

  // 加载任务列表
  const loadTasks = async () => {
    try {
      setRefreshing(true);
      // 获取所有任务（不限制数量，确保获取所有回测任务）
      const result = await TaskService.getTasks(undefined, 1000, 0);
      setTasks(result.tasks ?? []);
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : '加载回测任务失败';
      console.error('加载回测任务失败:', error);
      setLoadError(errMsg);
      showSnackbar.showError(errMsg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // 初始化加载
  useEffect(() => {
    loadTasks();
  }, []);

  // WebSocket实时更新
  useEffect(() => {
    const handleTaskProgress = (data: { task_id: string; progress: number; status: string }) => {
      updateTask(data.task_id, {
        progress: data.progress,
        status: data.status as Task['status'],
      });
    };

    const handleTaskCompleted = (data: { task_id: string; results: any }) => {
      updateTask(data.task_id, {
        status: 'completed',
        progress: 100,
        results: data.results,
        completed_at: new Date().toISOString(),
      });
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      updateTask(data.task_id, {
        status: 'failed',
        error_message: data.error,
      });
    };

    wsService.on('task:progress', handleTaskProgress);
    wsService.on('task:completed', handleTaskCompleted);
    wsService.on('task:failed', handleTaskFailed);

    return () => {
      wsService.off('task:progress', handleTaskProgress);
      wsService.off('task:completed', handleTaskCompleted);
      wsService.off('task:failed', handleTaskFailed);
    };
  }, [updateTask]);

  // 获取任务状态标签
  const getStatusChip = (status: Task['status']) => {
    const statusConfig = {
      created: { label: '已创建', color: 'default' as const, icon: Clock },
      queued: { label: '排队中', color: 'default' as const, icon: Clock },
      running: { label: '运行中', color: 'primary' as const, icon: Activity },
      completed: { label: '已完成', color: 'success' as const, icon: CheckCircle },
      failed: { label: '失败', color: 'error' as const, icon: XCircle },
      cancelled: { label: '已取消', color: 'warning' as const, icon: AlertTriangle },
      paused: { label: '已暂停', color: 'warning' as const, icon: AlertTriangle },
    };

    const config = statusConfig[status] || statusConfig.created;
    const IconComponent = config.icon;

    return (
      <Chip
        icon={<IconComponent size={16} />}
        label={config.label}
        color={config.color}
        size="small"
      />
    );
  };

  // 处理查看任务详情
  const handleViewTask = (taskId: string) => {
    router.push(`/tasks/${taskId}`);
  };

  // 处理创建新任务
  const handleCreateTask = () => {
    router.push('/tasks/create?type=backtest');
  };

  // 处理删除任务
  const handleDeleteTask = async () => {
    if (!selectedTask) {
      return;
    }

    try {
      await TaskService.deleteTask(selectedTask.task_id);
      setIsDeleteOpen(false);
      setSelectedTask(null);
      loadTasks();
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : '删除任务失败';
      console.error('删除任务失败:', error);
      showSnackbar.showError(errMsg);
    }
  };

  // 格式化日期
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN');
  };

  /**
   * 获取回测结果摘要
   *
   * 【P0-03】列表接口的指标来源是 `task.result_summary`（后端 TaskSummaryDTO 提供），
   * 旧的实现读 `task.result` —— 列表接口不返回该字段，导致已完成回测的
   * 收益/夏普列永远显示 "-"（看起来像没跑）。
   *
   * 缺失指标一律返回 null，UI 显示 "—"，**绝不用 0 冒充**（0 会被读成"收益为零"）。
   */
  const getBacktestSummary = (task: Task) => {
    const summary = task.result_summary;
    const detail =
      (task.results?.backtest_results as Record<string, unknown> | undefined) ??
      (task.backtest_results as Record<string, unknown> | undefined) ??
      (task.result as Record<string, unknown> | undefined);

    const pick = (key: string): number | null => {
      const raw = summary?.[key] ?? detail?.[key];
      return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
    };

    const totalReturn = pick('total_return');
    const sharpeRatio = pick('sharpe_ratio');
    const maxDrawdown = pick('max_drawdown');
    const winRate = pick('win_rate');

    // 一个指标都拿不到 → 视为无摘要
    if (totalReturn === null && sharpeRatio === null && maxDrawdown === null && winRate === null) {
      return null;
    }

    return {
      totalReturn,
      sharpeRatio,
      maxDrawdown,
      winRate,
      annualizedReturn: pick('annualized_return'),
      totalTrades: pick('total_trades'),
      turnover: pick('turnover'),
    };
  };

  /** 数值 → 展示文本；null 显示破折号（不是 0，也不是 undefined） */
  const fmtMetric = (value: number | null | undefined, digits = 2): string =>
    value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits);

  if (loading) {
    return <LoadingSpinner text="加载回测任务..." />;
  }

  if (loadError && backtestTasks.length === 0) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* 页面标题和操作 */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Activity size={32} />
            <Box>
              <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
                策略回测
              </Typography>
              <Typography variant="caption" color="text.secondary">
                管理和查看回测任务
              </Typography>
            </Box>
          </Box>
        </Box>

        <Alert
          severity="error"
          variant="filled"
          sx={{ borderRadius: 2 }}
          action={
            <Button color="inherit" size="small" onClick={loadTasks}>
              重试
            </Button>
          }
        >
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            加载失败
          </Typography>
          <Typography variant="caption">{loadError}</Typography>
        </Alert>

        {/* 骨架屏 */}
        <Card>
          <CardHeader title="回测任务" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[1, 2, 3].map(i => (
                <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2 }}>
                  <Skeleton variant="circular" width={32} height={32} />
                  <Box sx={{ flex: 1 }}>
                    <Skeleton variant="text" width="40%" />
                    <Skeleton variant="text" width="25%" />
                  </Box>
                  <Skeleton variant="rounded" width={80} height={24} />
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题和操作 */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Activity size={32} />
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
              策略回测
            </Typography>
            <Typography variant="caption" color="text.secondary">
              管理和查看回测任务
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshCw size={16} />}
            onClick={loadTasks}
            disabled={refreshing}
          >
            刷新
          </Button>
          <Button
            variant="contained"
            color="primary"
            startIcon={<Plus size={16} />}
            onClick={handleCreateTask}
          >
            创建回测任务
          </Button>
        </Box>
      </Box>

      {/* 回测任务列表 */}
      <Card>
        <CardHeader title={`回测任务 (${backtestTasks.length})`} />
        <CardContent>
          {backtestTasks.length === 0 ? (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                py: 8,
                gap: 2,
              }}
            >
              <Activity size={64} color="#999" />
              <Typography variant="body1" color="text.secondary">
                暂无回测任务
              </Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<Plus size={16} />}
                onClick={handleCreateTask}
              >
                创建第一个回测任务
              </Button>
            </Box>
          ) : (
            <Box>
              {/* 移动端：卡片列表 */}
              <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                {backtestTasks.map(task => (
                  <MobileBacktestCard
                    key={task.task_id}
                    task={task}
                    onDelete={id => {
                      setSelectedTask(backtestTasks.find(t => t.task_id === id) || null);
                      setIsDeleteOpen(true);
                    }}
                  />
                ))}
              </Box>

              {/* 桌面端：表格 */}
              <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>任务名称</TableCell>
                        <TableCell>策略</TableCell>
                        <TableCell>股票数量</TableCell>
                        <TableCell>回测期间</TableCell>
                        <TableCell>状态</TableCell>
                        <TableCell>进度</TableCell>
                        <TableCell>创建时间</TableCell>
                        <TableCell align="right">操作</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {backtestTasks.map(task => {
                        const summary = getBacktestSummary(task);
                        const strategyName =
                          task.config?.strategy_name ||
                          task.config?.strategy_config?.strategy_name ||
                          '未知策略';
                        const startDate = task.config?.start_date || '';
                        const endDate = task.config?.end_date || '';

                        return (
                          <TableRow key={task.task_id} hover>
                            <TableCell>
                              <Box>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {task.task_name}
                                </Typography>
                                {task.description && (
                                  <Typography variant="caption" color="text.secondary">
                                    {task.description}
                                  </Typography>
                                )}
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip label={strategyName} size="small" variant="outlined" />
                            </TableCell>
                            <TableCell>{task.stock_codes?.length || 0}</TableCell>
                            <TableCell>
                              {startDate && endDate ? (
                                <Typography variant="caption">
                                  {startDate.split('T')[0]} ~ {endDate.split('T')[0]}
                                </Typography>
                              ) : (
                                <Typography variant="caption" color="text.secondary">
                                  -
                                </Typography>
                              )}
                            </TableCell>
                            <TableCell>{getStatusChip(task.status)}</TableCell>
                            <TableCell>
                              {task.status === 'running' ? (
                                <Box sx={{ width: 100 }}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={task.progress || 0}
                                    sx={{ height: 8, borderRadius: 4 }}
                                  />
                                  <Typography variant="caption" color="text.secondary">
                                    {task.progress || 0}%
                                  </Typography>
                                </Box>
                              ) : task.status === 'completed' && summary ? (
                                <Box>
                                  {/* 【P0-03 + 语义色】收益列改读 result_summary；
                                      涨跌色走 palette.semantics（红涨绿跌），
                                      不用 success.main（那是"操作成功"色）。 */}
                                  <Typography
                                    variant="caption"
                                    sx={{
                                      color:
                                        summary.totalReturn === null
                                          ? 'text.secondary'
                                          : summary.totalReturn >= 0
                                            ? 'semantics.up'
                                            : 'semantics.down',
                                    }}
                                  >
                                    总收益:{' '}
                                    {summary.totalReturn === null
                                      ? '—'
                                      : `${(summary.totalReturn * 100).toFixed(2)}%`}
                                  </Typography>
                                  <br />
                                  <Typography variant="caption" color="text.secondary">
                                    夏普: {fmtMetric(summary.sharpeRatio)}
                                  </Typography>
                                </Box>
                              ) : (
                                <Typography variant="caption" color="text.secondary">
                                  —
                                </Typography>
                              )}
                            </TableCell>
                            <TableCell>
                              <Typography variant="caption">
                                {formatDate(task.created_at)}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                                <IconButton
                                  size="small"
                                  color="primary"
                                  onClick={() => handleViewTask(task.task_id)}
                                  title="查看详情"
                                >
                                  <Eye size={16} />
                                </IconButton>
                                {task.status !== 'running' && (
                                  <IconButton
                                    size="small"
                                    color="error"
                                    onClick={() => {
                                      setSelectedTask(task);
                                      setIsDeleteOpen(true);
                                    }}
                                    title="删除任务"
                                  >
                                    <Trash2 size={16} />
                                  </IconButton>
                                )}
                              </Box>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* 删除确认对话框 */}
      <Dialog open={isDeleteOpen} onClose={() => setIsDeleteOpen(false)}>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AlertTriangle size={20} color="#d32f2f" />
            <Typography variant="h6" component="span">
              确认删除
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            确定要删除回测任务 “{selectedTask?.task_name}” 吗？此操作不可撤销。
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button variant="outlined" onClick={() => setIsDeleteOpen(false)}>
            取消
          </Button>
          <Button variant="contained" color="error" onClick={handleDeleteTask}>
            删除
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
