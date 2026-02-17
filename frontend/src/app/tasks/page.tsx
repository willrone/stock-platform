/**
 * 任务管理页面
 *
 * 显示任务列表，支持：
 * - 任务创建
 * - 任务状态筛选
 * - 任务进度监控
 * - 实时状态更新
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  LinearProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Pagination,
  Tooltip,
  Box,
  Typography,
  IconButton,
  InputAdornment,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { Plus, RefreshCw, Search, Eye, Play, Pause, Trash2, Filter, BarChart3 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useTaskStore, Task } from '../../stores/useTaskStore';
import { TaskService } from '../../services/taskService';
import { wsService } from '../../services/websocket';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { MobileTaskCard } from '../../components/mobile/MobileTaskCard';

export default function TasksPage() {
  const router = useRouter();
  const {
    tasks,
    total,
    currentPage,
    pageSize,
    statusFilter,
    loading,
    setTasks,
    setLoading,
    setPagination,
    setStatusFilter,
    updateTask,
  } = useTaskStore();

  const [searchText, setSearchText] = useState('');
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());
  const [taskToDelete, setTaskToDelete] = useState<string | null>(null);
  const [stats, setStats] = useState({
    total: 0,
    running: 0,
    completed: 0,
    failed: 0,
  });
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isBatchDeleteOpen, setIsBatchDeleteOpen] = useState(false);
  const [deleteForce, setDeleteForce] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // 加载任务列表
  const loadTasks = async (page = currentPage, size = pageSize, status = statusFilter) => {
    setLoading(true);
    try {
      const offset = (page - 1) * size;
      const result = await TaskService.getTasks(status || undefined, size, offset);
      setTasks(result.tasks, result.total);
    } catch (error) {
      console.error('加载任务失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 加载任务统计信息
  const loadStats = async () => {
    try {
      const statsData = await TaskService.getTaskStats();
      setStats({
        total: statsData.total,
        running: statsData.running,
        completed: statsData.completed,
        failed: statsData.failed,
      });
    } catch (error) {
      console.error('加载任务统计失败:', error);
      // 如果API失败，使用本地计算作为后备
      setStats({
        total: tasks.length,
        running: tasks.filter(t => t.status === 'running').length,
        completed: tasks.filter(t => t.status === 'completed').length,
        failed: tasks.filter(t => t.status === 'failed').length,
      });
    }
  };

  // 初始化加载
  useEffect(() => {
    loadTasks();
    loadStats();
  }, []);

  // 当任务列表更新时，也更新统计（作为后备）
  useEffect(() => {
    if (tasks.length > 0 && stats.total === 0) {
      setStats({
        total: tasks.length,
        running: tasks.filter(t => t.status === 'running').length,
        completed: tasks.filter(t => t.status === 'completed').length,
        failed: tasks.filter(t => t.status === 'failed').length,
      });
    }
  }, [tasks]);

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
      console.log(`任务 ${data.task_id} 已完成`);
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      updateTask(data.task_id, {
        status: 'failed',
        error_message: data.error,
      });
      console.error(`任务 ${data.task_id} 执行失败`);
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
      created: { color: 'default' as const, text: '已创建' },
      running: { color: 'primary' as const, text: '运行中' },
      completed: { color: 'success' as const, text: '已完成' },
      failed: { color: 'error' as const, text: '失败' },
    };

    const config = statusConfig[status] || statusConfig.created;
    return <Chip label={config.text} color={config.color} size="small" />;
  };

  // 处理分页变化
  const handlePageChange = (event: React.ChangeEvent<unknown>, page: number) => {
    setPagination(page, pageSize);
    loadTasks(page, pageSize, statusFilter);
  };

  // 处理状态筛选
  const handleStatusFilter = (status: string | null) => {
    setStatusFilter(status);
    setPagination(1, pageSize);
    loadTasks(1, pageSize, status);
  };

  // 刷新任务列表
  const handleRefresh = () => {
    loadTasks();
  };

  // 创建新任务
  const handleCreateTask = () => {
    router.push('/tasks/create');
  };

  // 查看任务详情
  const handleViewTask = (taskId: string) => {
    router.push(`/tasks/${taskId}`);
  };

  // 删除任务
  const handleDeleteTask = async () => {
    if (!taskToDelete) {
      return;
    }

    setDeleteError(null);
    try {
      await TaskService.deleteTask(taskToDelete, deleteForce);
      console.log('任务删除成功');
      loadTasks();
      setTaskToDelete(null);
      setIsDeleteOpen(false);
      setDeleteForce(false);
    } catch (error: any) {
      console.error('删除任务失败:', error);
      const errorMessage = error?.response?.data?.detail || error?.message || '删除任务失败';
      setDeleteError(errorMessage);

      // 如果错误提示需要使用强制删除，自动勾选强制删除选项
      if (errorMessage.includes('强制删除') || errorMessage.includes('正在运行中')) {
        setDeleteForce(true);
      }
    }
  };

  // 批量删除任务
  const handleBatchDelete = async () => {
    const taskIds = Array.from(selectedKeys);
    if (taskIds.length === 0) {
      return;
    }

    try {
      await TaskService.batchDeleteTasks(taskIds);
      console.log(`成功删除 ${taskIds.length} 个任务`);
      setSelectedKeys(new Set());
      loadTasks();
    } catch (error) {
      console.error('批量删除失败');
    } finally {
      setIsBatchDeleteOpen(false);
    }
  };

  // 重新运行任务
  const handleRetryTask = async (taskId: string) => {
    try {
      await TaskService.retryTask(taskId);
      console.log('任务已重新启动');
      loadTasks();
    } catch (error) {
      console.error('重新运行失败');
    }
  };

  // 过滤任务
  const filteredTasks = tasks.filter(task =>
    task.task_name.toLowerCase().includes(searchText.toLowerCase())
  );

  if (loading && tasks.length === 0) {
    return <LoadingSpinner text="加载任务列表..." />;
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题和统计 */}
      <Box>
        <Typography
          variant="h4"
          component="h1"
          sx={{ fontWeight: 600, mb: 2, fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' } }}
        >
          任务管理
        </Typography>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
            gap: 2,
          }}
        >
          <Card>
            <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 } }}>
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                <BarChart3 size={24} color="#1976d2" />
              </Box>
              <Typography
                variant="h4"
                sx={{ fontWeight: 600, fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' } }}
              >
                {stats.total}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
              >
                总任务数
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 } }}>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 600,
                  color: 'primary.main',
                  fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' },
                }}
              >
                {stats.running}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
              >
                运行中
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 } }}>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 600,
                  color: 'success.main',
                  fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' },
                }}
              >
                {stats.completed}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
              >
                已完成
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 } }}>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 600,
                  color: 'error.main',
                  fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' },
                }}
              >
                {stats.failed}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
              >
                失败
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* 操作栏 */}
      <Card>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              justifyContent: 'space-between',
              alignItems: { xs: 'stretch', md: 'center' },
              gap: 2,
            }}
          >
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<Plus size={16} />}
                onClick={handleCreateTask}
              >
                创建任务
              </Button>
              <Button
                variant="outlined"
                startIcon={<RefreshCw size={16} />}
                onClick={handleRefresh}
                disabled={loading}
              >
                刷新
              </Button>
              {selectedKeys.size > 0 && (
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Trash2 size={16} />}
                  onClick={() => setIsBatchDeleteOpen(true)}
                >
                  批量删除 ({selectedKeys.size})
                </Button>
              )}
            </Box>

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <TextField
                placeholder="搜索任务名称"
                size="small"
                value={searchText}
                onChange={e => setSearchText(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search size={16} />
                    </InputAdornment>
                  ),
                }}
                sx={{ width: 200 }}
              />
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>筛选状态</InputLabel>
                <Select
                  value={statusFilter || ''}
                  label="筛选状态"
                  onChange={e => handleStatusFilter(e.target.value || null)}
                  startAdornment={<Filter size={16} />}
                >
                  <MenuItem value="">全部</MenuItem>
                  <MenuItem value="created">已创建</MenuItem>
                  <MenuItem value="running">运行中</MenuItem>
                  <MenuItem value="completed">已完成</MenuItem>
                  <MenuItem value="failed">失败</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* 任务列表 */}
      <Card>
        <CardContent>
          {/* 移动端：卡片列表 */}
          <Box sx={{ display: { xs: 'block', md: 'none' } }}>
            {filteredTasks.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  {statusFilter ? '该状态下暂无任务' : '暂无任务'}
                </Typography>
              </Box>
            ) : (
              filteredTasks.map(task => (
                <MobileTaskCard
                  key={task.task_id}
                  task={task}
                  onDelete={id => {
                    setTaskToDelete(id);
                    setIsDeleteOpen(true);
                  }}
                  onToggle={id => {
                    console.log('暂停功能开发中', id);
                  }}
                />
              ))
            )}
          </Box>

          {/* 桌面端：表格 */}
          <Box sx={{ display: { xs: 'none', md: 'block' }, overflowX: 'auto' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>任务名称</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>进度</TableCell>
                  <TableCell>股票数量</TableCell>
                  <TableCell>模型</TableCell>
                  <TableCell>创建时间</TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredTasks.map(task => (
                  <TableRow key={task.task_id} hover>
                    <TableCell>
                      <Button
                        variant="text"
                        onClick={() => handleViewTask(task.task_id)}
                        sx={{ textTransform: 'none', p: 0, minWidth: 0 }}
                      >
                        {task.task_name}
                      </Button>
                    </TableCell>
                    <TableCell>{getStatusChip(task.status)}</TableCell>
                    <TableCell>
                      <Box sx={{ width: 80 }}>
                        <LinearProgress
                          variant="determinate"
                          value={task.progress}
                          color={task.status === 'failed' ? 'error' : 'primary'}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {task.stock_codes?.length || 0}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {task.model_id ? (
                        <Chip label={task.model_id} size="small" />
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          {task.task_type === 'backtest'
                            ? '回测任务'
                            : task.task_type === 'hyperparameter_optimization'
                              ? '超参优化'
                              : '-'}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(task.created_at).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="查看详情">
                          <IconButton size="small" onClick={() => handleViewTask(task.task_id)}>
                            <Eye size={16} />
                          </IconButton>
                        </Tooltip>

                        {task.status === 'running' && (
                          <Tooltip title="暂停任务">
                            <IconButton
                              size="small"
                              onClick={() => {
                                console.log('暂停功能开发中');
                              }}
                            >
                              <Pause size={16} />
                            </IconButton>
                          </Tooltip>
                        )}

                        {task.status === 'failed' && (
                          <Tooltip title="重新运行">
                            <IconButton size="small" onClick={() => handleRetryTask(task.task_id)}>
                              <Play size={16} />
                            </IconButton>
                          </Tooltip>
                        )}

                        <Tooltip title="删除任务">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => {
                              setTaskToDelete(task.task_id);
                              setIsDeleteOpen(true);
                            }}
                          >
                            <Trash2 size={16} />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>

          {/* 分页 */}
          {total > pageSize && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Pagination
                count={Math.ceil(total / pageSize)}
                page={currentPage}
                onChange={handlePageChange}
                color="primary"
              />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* 删除确认对话框 */}
      <Dialog
        open={isDeleteOpen}
        onClose={() => {
          setIsDeleteOpen(false);
          setDeleteForce(false);
          setDeleteError(null);
        }}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>确认删除</DialogTitle>
        <DialogContent>
          <Typography sx={{ mb: 2 }}>确定要删除这个任务吗？此操作不可撤销。</Typography>

          {deleteError && (
            <Box
              sx={{
                p: 2,
                mb: 2,
                bgcolor: 'error.light',
                color: 'error.contrastText',
                borderRadius: 1,
              }}
            >
              <Typography variant="body2">{deleteError}</Typography>
            </Box>
          )}

          <FormControlLabel
            control={
              <Switch
                checked={deleteForce}
                onChange={e => setDeleteForce(e.target.checked)}
                color="error"
              />
            }
            label={
              <Typography variant="body2">
                强制删除（用于删除运行中的任务或存在关联数据的任务）
              </Typography>
            }
          />
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setIsDeleteOpen(false);
              setDeleteForce(false);
              setDeleteError(null);
            }}
          >
            取消
          </Button>
          <Button onClick={handleDeleteTask} color="error" variant="contained">
            {deleteForce ? '强制删除' : '删除'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 批量删除确认对话框 */}
      <Dialog open={isBatchDeleteOpen} onClose={() => setIsBatchDeleteOpen(false)}>
        <DialogTitle>批量删除确认</DialogTitle>
        <DialogContent>
          <Typography>确定要删除选中的 {selectedKeys.size} 个任务吗？此操作不可撤销。</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsBatchDeleteOpen(false)}>取消</Button>
          <Button onClick={handleBatchDelete} color="error" variant="contained">
            删除
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
