/**
 * 超参优化页面
 * 
 * 提供完整的超参优化功能：
 * - 创建优化任务
 * - 查看任务列表
 * - 监控优化状态
 * - 查看优化结果和可视化
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Tabs,
  Tab,
  Button,
} from '@heroui/react';
import { useRouter } from 'next/navigation';
import { OptimizationService, OptimizationTask } from '../../services/optimizationService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import CreateOptimizationTaskForm from '../../components/optimization/CreateOptimizationTaskForm';
import OptimizationTaskList from '../../components/optimization/OptimizationTaskList';
import OptimizationTaskDetail from '../../components/optimization/OptimizationTaskDetail';

export default function OptimizationPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('list');
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<OptimizationTask[]>([]);
  const [loading, setLoading] = useState(false);

  // 加载任务列表
  const loadTasks = async () => {
    setLoading(true);
    try {
      const result = await OptimizationService.getTasks();
      setTasks(result.tasks);
    } catch (error) {
      console.error('加载优化任务列表失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTasks();
  }, []);

  const handleTaskCreated = () => {
    loadTasks();
    setActiveTab('list');
  };

  const handleTaskSelected = (taskId: string) => {
    setSelectedTaskId(taskId);
    setActiveTab('detail');
  };

  return (
    <div className="container mx-auto px-4 py-6 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">超参优化</h1>
        <p className="text-default-500 mt-2">
          使用 Optuna 对策略参数进行优化，寻找最佳参数配置
        </p>
      </div>

      <Tabs
        selectedKey={activeTab}
        onSelectionChange={(key) => setActiveTab(key as string)}
        aria-label="优化任务标签页"
      >
        <Tab key="list" title="任务列表">
          <Card className="mt-4">
            <CardBody>
              {loading ? (
                <LoadingSpinner text="加载任务列表..." />
              ) : (
                <OptimizationTaskList
                  tasks={tasks}
                  onTaskSelect={handleTaskSelected}
                  onRefresh={loadTasks}
                />
              )}
            </CardBody>
          </Card>
        </Tab>

        <Tab key="create" title="创建任务">
          <Card className="mt-4">
            <CardBody>
              <CreateOptimizationTaskForm onTaskCreated={handleTaskCreated} />
            </CardBody>
          </Card>
        </Tab>

        <Tab key="detail" title="任务详情" isDisabled={!selectedTaskId}>
          {selectedTaskId && (
            <Card className="mt-4">
              <CardBody>
                <OptimizationTaskDetail
                  taskId={selectedTaskId}
                  onBack={() => setActiveTab('list')}
                />
              </CardBody>
            </Card>
          )}
        </Tab>
      </Tabs>
    </div>
  );
}

