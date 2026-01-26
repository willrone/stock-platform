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
import { Card, CardContent, Tabs, Tab, Box, Typography } from '@mui/material';
import { OptimizationService, OptimizationTask } from '../../services/optimizationService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import CreateOptimizationTaskForm from '../../components/optimization/CreateOptimizationTaskForm';
import OptimizationTaskList from '../../components/optimization/OptimizationTaskList';
import OptimizationTaskDetail from '../../components/optimization/OptimizationTaskDetail';

export default function OptimizationPage() {
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
    <Box sx={{ maxWidth: 1400, mx: 'auto', px: 3, py: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          超参优化
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          使用 Optuna 对策略参数进行优化，寻找最佳参数配置
        </Typography>
      </Box>

      <Tabs
        value={activeTab}
        onChange={(e, newValue) => setActiveTab(newValue)}
        aria-label="优化任务标签页"
      >
        <Tab label="任务列表" value="list" />
        <Tab label="创建任务" value="create" />
        {selectedTaskId && <Tab label="任务详情" value="detail" />}
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {activeTab === 'list' && (
          <Card>
            <CardContent>
              {loading ? (
                <LoadingSpinner text="加载任务列表..." />
              ) : (
                <OptimizationTaskList
                  tasks={tasks}
                  onTaskSelect={handleTaskSelected}
                  onRefresh={loadTasks}
                />
              )}
            </CardContent>
          </Card>
        )}

        {activeTab === 'create' && (
          <Card>
            <CardContent>
              <CreateOptimizationTaskForm onTaskCreated={handleTaskCreated} />
            </CardContent>
          </Card>
        )}

        {activeTab === 'detail' && selectedTaskId && (
          <Card>
            <CardContent>
              <OptimizationTaskDetail taskId={selectedTaskId} onBack={() => setActiveTab('list')} />
            </CardContent>
          </Card>
        )}
      </Box>
    </Box>
  );
}
