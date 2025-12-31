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
  CardHeader,
  CardBody,
  Button,
  Progress,
  Chip,
  Avatar,
} from '@heroui/react';
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
import { useDataStore } from '../../stores/useDataStore';
import { useAppStore } from '../../stores/useAppStore';
import { TaskService } from '../../services/taskService';
import { DataService } from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

export default function DashboardPage() {
  const router = useRouter();
  const { tasks } = useTaskStore();
  const { systemStatus } = useDataStore();
  
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
  const getTaskStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'failed': return 'danger';
      default: return 'default';
    }
  };

  // 获取任务状态文本
  const getTaskStatusText = (status: string) => {
    switch (status) {
      case 'running': return '运行中';
      case 'completed': return '已完成';
      case 'failed': return '失败';
      default: return '已创建';
    }
  };

  // 获取任务状态图标
  const getTaskStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Clock className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <AlertTriangle className="w-4 h-4" />;
      default: return <Bot className="w-4 h-4" />;
    }
  };

  if (loading) {
    return <LoadingSpinner text="加载仪表板数据..." />;
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-2xl font-bold mb-2">仪表板</h1>
        <p className="text-default-500">系统概览和快速操作</p>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardBody className="flex items-center space-x-4">
            <div className="p-3 bg-primary-100 rounded-lg">
              <Bot className="w-6 h-6 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold">{systemStats.totalTasks}</p>
              <p className="text-sm text-default-500">总任务数</p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="flex items-center space-x-4">
            <div className="p-3 bg-success-100 rounded-lg">
              <CheckCircle className="w-6 h-6 text-success" />
            </div>
            <div>
              <p className="text-2xl font-bold text-success">{systemStats.completedTasks}</p>
              <p className="text-sm text-default-500">已完成</p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="flex items-center space-x-4">
            <div className="p-3 bg-warning-100 rounded-lg">
              <Clock className="w-6 h-6 text-warning" />
            </div>
            <div>
              <p className="text-2xl font-bold text-warning">{systemStats.runningTasks}</p>
              <p className="text-sm text-default-500">运行中</p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="flex items-center space-x-4">
            <div className="p-3 bg-danger-100 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-danger" />
            </div>
            <div>
              <p className="text-2xl font-bold text-danger">{systemStats.failedTasks}</p>
              <p className="text-sm text-default-500">失败</p>
            </div>
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 最近任务 */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="flex justify-between items-center">
              <h3 className="text-lg font-semibold">最近任务</h3>
              <Button
                size="sm"
                variant="light"
                onPress={() => router.push('/tasks')}
              >
                查看全部
              </Button>
            </CardHeader>
            <CardBody>
              {recentTasks.length === 0 ? (
                <div className="text-center py-8">
                  <Bot className="w-12 h-12 text-default-300 mx-auto mb-4" />
                  <p className="text-default-500">暂无任务</p>
                  <Button
                    color="primary"
                    className="mt-4"
                    startContent={<Plus className="w-4 h-4" />}
                    onPress={() => router.push('/tasks/create')}
                  >
                    创建任务
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {recentTasks.map((task) => (
                    <div
                      key={task.task_id}
                      className="flex items-center justify-between p-3 border border-divider rounded-lg hover:bg-default-50 cursor-pointer"
                      onClick={() => router.push(`/tasks/${task.task_id}`)}
                    >
                      <div className="flex items-center space-x-3">
                        <Avatar
                          size="sm"
                          fallback={getTaskStatusIcon(task.status)}
                          className="bg-default-100"
                        />
                        <div>
                          <p className="font-medium">{task.task_name}</p>
                          <p className="text-sm text-default-500">
                            {task.stock_codes?.length || 0} 只股票
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Chip
                          color={getTaskStatusColor(task.status)}
                          variant="flat"
                          size="sm"
                        >
                          {getTaskStatusText(task.status)}
                        </Chip>
                        {task.status === 'running' && (
                          <Progress
                            value={task.progress}
                            size="sm"
                            className="w-16"
                          />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardBody>
          </Card>
        </div>

        {/* 系统状态和快速操作 */}
        <div className="space-y-6">
          {/* 系统状态 */}
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">系统状态</h3>
            </CardHeader>
            <CardBody className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Wifi className="w-4 h-4 text-success" />
                  <span className="text-sm">API服务</span>
                </div>
                <Chip color="success" variant="flat" size="sm">正常</Chip>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Database className="w-4 h-4 text-success" />
                  <span className="text-sm">数据服务</span>
                </div>
                <Chip color="success" variant="flat" size="sm">正常</Chip>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-warning" />
                  <span className="text-sm">模型服务</span>
                </div>
                <Chip color="warning" variant="flat" size="sm">负载高</Chip>
              </div>
              
              <div className="pt-2 border-t border-divider">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-default-600">系统负载</span>
                  <span className="text-sm font-medium">65%</span>
                </div>
                <Progress value={65} size="sm" />
              </div>
            </CardBody>
          </Card>

          {/* 快速操作 */}
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">快速操作</h3>
            </CardHeader>
            <CardBody className="space-y-3">
              <Button
                color="primary"
                startContent={<Plus className="w-4 h-4" />}
                onPress={() => router.push('/tasks/create')}
                fullWidth
              >
                创建预测任务
              </Button>
              
              <Button
                variant="light"
                startContent={<Eye className="w-4 h-4" />}
                onPress={() => router.push('/tasks')}
                fullWidth
              >
                查看所有任务
              </Button>
              
              <Button
                variant="light"
                startContent={<Database className="w-4 h-4" />}
                onPress={() => router.push('/data')}
                fullWidth
              >
                数据管理
              </Button>
              
              <Button
                variant="light"
                startContent={<TrendingUp className="w-4 h-4" />}
                onPress={() => router.push('/predictions')}
                fullWidth
              >
                预测分析
              </Button>
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );}