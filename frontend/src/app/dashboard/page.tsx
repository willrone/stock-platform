/**
 * 仪表板页面
 *
 * 显示系统概览信息，包括：
 * - 任务统计
 * - 系统状态
 * - 最近活动
 * - 快速操作
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Avatar,
  Box,
  Typography,
  LinearProgress,
} from '@mui/material';
import {
  Bot,
  CheckCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  Database,
  Wifi,
  Plus,
  Eye,
  Activity,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useTaskStore } from '../../stores/useTaskStore';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

export default function DashboardPage() {
  const router = useRouter();
  const { tasks } = useTaskStore();

  const [loading, setLoading] = useState(true);
  const [recentTasks, setRecentTasks] = useState<any[]>([]);
  const [systemStats, setSystemStats] = useState({
    totalTasks: 0,
    runningTasks: 0,
    completedTasks: 0,
    failedTasks: 0,
    dataFiles: 0,
    systemHealth: 'good' as 'good' | 'warning' | 'error',
  });

  // 加载仪表板数据
  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true);

        // 加载最近任务
        const tasksResult = await TaskService.getTasks(undefined, 5, 0);
        setRecentTasks(tasksResult.tasks);

        // 计算统计数据
        const stats = {
          totalTasks: tasks.length,
          runningTasks: tasks.filter(t => t.status === 'running').length,
          completedTasks: tasks.filter(t => t.status === 'completed').length,
          failedTasks: tasks.filter(t => t.status === 'failed').length,
          dataFiles: 156, // 模拟数据
          systemHealth: 'good' as const,
        };
        setSystemStats(stats);
      } catch (error) {
        console.error('加载仪表板数据失败:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, [tasks]);

  // 获取任务状态颜色
  const getTaskStatusColor = (status: string): 'primary' | 'success' | 'error' | 'default' => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  // 获取任务状态文本
  const getTaskStatusText = (status: string) => {
    switch (status) {
      case 'running':
        return '运行中';
      case 'completed':
        return '已完成';
      case 'failed':
        return '失败';
      default:
        return '已创建';
    }
  };

  // 获取任务状态图标
  const getTaskStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Clock size={16} />;
      case 'completed':
        return <CheckCircle size={16} />;
      case 'failed':
        return <AlertTriangle size={16} />;
      default:
        return <Bot size={16} />;
    }
  };

  if (loading) {
    return <LoadingSpinner text="加载仪表板数据..." />;
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题 */}
      <Box>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 1 }}>
          仪表板
        </Typography>
        <Typography variant="body2" color="text.secondary">
          系统概览和快速操作
        </Typography>
      </Box>

      {/* 统计卡片 */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' },
          gap: 2,
        }}
      >
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'primary.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <Bot size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {systemStats.totalTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  总任务数
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'success.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <CheckCircle size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                  {systemStats.completedTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  已完成
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'warning.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <Clock size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                  {systemStats.runningTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  运行中
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'error.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <AlertTriangle size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                  {systemStats.failedTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  失败
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      <Box
        sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 3, mt: 3 }}
      >
        {/* 最近任务 */}
        <Box>
          <Card>
            <CardHeader
              title="最近任务"
              action={
                <Button size="small" onClick={() => router.push('/tasks')}>
                  查看全部
                </Button>
              }
            />
            <CardContent>
              {recentTasks.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Bot size={48} color="#ccc" style={{ margin: '0 auto 16px' }} />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    暂无任务
                  </Typography>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<Plus size={16} />}
                    onClick={() => router.push('/tasks/create')}
                  >
                    创建任务
                  </Button>
                </Box>
              ) : (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {recentTasks.map(task => (
                    <Box
                      key={task.task_id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        p: 2,
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        cursor: 'pointer',
                        '&:hover': { bgcolor: 'action.hover' },
                      }}
                      onClick={() => router.push(`/tasks/${task.task_id}`)}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Avatar sx={{ bgcolor: 'grey.200', width: 32, height: 32 }}>
                          {getTaskStatusIcon(task.status)}
                        </Avatar>
                        <Box>
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            {task.task_name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {task.stock_codes?.length || 0} 只股票
                          </Typography>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Chip
                          label={getTaskStatusText(task.status)}
                          color={getTaskStatusColor(task.status)}
                          size="small"
                        />
                        {task.status === 'running' && (
                          <Box sx={{ width: 64 }}>
                            <LinearProgress variant="determinate" value={task.progress} />
                          </Box>
                        )}
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>

        {/* 系统状态和快速操作 */}
        <Box>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 系统状态 */}
            <Card>
              <CardHeader title="系统状态" />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box
                  sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Wifi size={16} color="#2e7d32" />
                    <Typography variant="body2">API服务</Typography>
                  </Box>
                  <Chip label="正常" color="success" size="small" />
                </Box>

                <Box
                  sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Database size={16} color="#2e7d32" />
                    <Typography variant="body2">数据服务</Typography>
                  </Box>
                  <Chip label="正常" color="success" size="small" />
                </Box>

                <Box
                  sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Activity size={16} color="#ed6c02" />
                    <Typography variant="body2">模型服务</Typography>
                  </Box>
                  <Chip label="负载高" color="warning" size="small" />
                </Box>

                <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      系统负载
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      65%
                    </Typography>
                  </Box>
                  <LinearProgress variant="determinate" value={65} />
                </Box>
              </CardContent>
            </Card>

            {/* 快速操作 */}
            <Card>
              <CardHeader title="快速操作" />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<Plus size={16} />}
                  onClick={() => router.push('/tasks/create')}
                  fullWidth
                >
                  创建预测任务
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<Eye size={16} />}
                  onClick={() => router.push('/tasks')}
                  fullWidth
                >
                  查看所有任务
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<Database size={16} />}
                  onClick={() => router.push('/data')}
                  fullWidth
                >
                  数据管理
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<TrendingUp size={16} />}
                  onClick={() => router.push('/predictions')}
                  fullWidth
                >
                  预测分析
                </Button>
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
