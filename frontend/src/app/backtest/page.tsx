/**
 * 策略回测页面
 *
 * 显示回测任务列表和管理界面
 */

'use client';

import React, { useEffect, useState } from 'react';
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
} from '@mui/material';
import {
  Plus,
  RefreshCw,
  Play,
  Trash2,
  Eye,
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useTaskStore, Task } from '../../stores/useTaskStore';
import { TaskService } from '../../services/taskService';
import { wsService } from '../../services/websocket';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { MobileBacktestCard } from '../../components/mobile/MobileBacktestCard';

export default function BacktestPage() {
  const router = useRouter();
  const { tasks, setTasks, updateTask } = useTaskStore();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

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
        const backtestKeys = ['equity_curve', 'drawdown_curve', 'portfolio', 'risk_metrics', 'trade_history', 'dates'];
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
      // 使用setTasks更新整个任务列表
      setTasks(result.tasks, result.total);
    } catch (error) {
      console.error('加载回测任务失败:', error);
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
      running: { label: '运行中', color: 'primary' as const, icon: Activity },
      completed: { label: '已完成', color: 'success' as const, icon: CheckCircle },
      failed: { label: '失败', color: 'error' as const, icon: XCircle },
      cancelled: { label: '已取消', color: 'warning' as const, icon: AlertTriangle },
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
    if (!selectedTask) return;

    try {
      await TaskService.deleteTask(selectedTask.task_id);
      setIsDeleteOpen(false);
      setSelectedTask(null);
      loadTasks();
    } catch (error) {
      console.error('删除任务失败:', error);
      alert('删除任务失败: ' + (error instanceof Error ? error.message : String(error)));
    }
  };

  // 格式化日期
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN');
  };

  // 获取回测结果摘要
  const getBacktestSummary = (task: Task) => {
    const backtestData =
      task.results?.backtest_results || task.backtest_results || task.result;
    if (!backtestData) {
      return null;
    }

    return {
      totalReturn: backtestData.total_return || 0,
      sharpeRatio: backtestData.sharpe_ratio || 0,
      maxDrawdown: backtestData.max_drawdown || 0,
      winRate: backtestData.win_rate || 0,
    };
  };

  if (loading) {
    return <LoadingSpinner text="加载回测任务..." />;
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
                    onDelete={(id) => {
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
                      task.config?.strategy_name || task.config?.strategy_config?.strategy_name || '未知策略';
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
                              <Typography variant="caption" color="success.main">
                                总收益: {(summary.totalReturn * 100).toFixed(2)}%
                              </Typography>
                              <br />
                              <Typography variant="caption" color="text.secondary">
                                夏普: {summary.sharpeRatio.toFixed(2)}
                              </Typography>
                            </Box>
                          ) : (
                            <Typography variant="caption" color="text.secondary">
                              -
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
            确定要删除回测任务 "{selectedTask?.task_name}" 吗？此操作不可撤销。
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
