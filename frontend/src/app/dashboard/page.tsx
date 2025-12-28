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
  Row, 
  Col, 
  Card, 
  Statistic, 
  Progress, 
  List, 
  Avatar, 
  Button, 
  Space,
  Typography,
  Tag,
  Alert,
} from 'antd';
import {
  RobotOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  RiseOutlined,
  DatabaseOutlined,
  ApiOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useTaskStore } from '../../stores/useTaskStore';
import { useDataStore } from '../../stores/useDataStore';
import { useAppStore } from '../../stores/useAppStore';
import { TaskService } from '../../services/taskService';
import { DataService } from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

const { Title, Text } = Typography;

export default function DashboardPage() {
  const router = useRouter();
  const { tasks } = useTaskStore();
  const { systemStatus } = useDataStore();
  const { loading, setLoading } = useAppStore();
  
  const [taskStats, setTaskStats] = useState({
    total: 0,
    completed: 0,
    running: 0,
    failed: 0,
    success_rate: 0,
  });
  
  const [recentTasks, setRecentTasks] = useState<any[]>([]);

  // 加载仪表板数据
  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // 并行加载数据
      const [statsResult, tasksResult, statusResult] = await Promise.all([
        TaskService.getTaskStats(),
        TaskService.getTasks(undefined, 5, 0), // 获取最近5个任务
        DataService.getSystemStatus(),
      ]);

      setTaskStats(statsResult);
      setRecentTasks(tasksResult.tasks);
      
    } catch (error) {
      console.error('加载仪表板数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 获取任务状态颜色
  const getTaskStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'processing';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  // 获取任务状态文本
  const getTaskStatusText = (status: string) => {
    switch (status) {
      case 'completed':
        return '已完成';
      case 'running':
        return '运行中';
      case 'failed':
        return '失败';
      case 'created':
        return '已创建';
      default:
        return status;
    }
  };

  if (loading) {
    return <LoadingSpinner text="加载仪表板数据..." />;
  }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>仪表板</Title>
        <Text type="secondary">系统概览和快速操作</Text>
      </div>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={taskStats.total}
              prefix={<RobotOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="已完成"
              value={taskStats.completed}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="运行中"
              value={taskStats.running}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="成功率"
              value={taskStats.success_rate}
              suffix="%"
              prefix={<RiseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 系统状态 */}
        <Col xs={24} lg={12}>
          <Card title="系统状态" extra={<Button size="small">刷新</Button>}>
            {systemStatus ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <ApiOutlined />
                    <Text>API服务</Text>
                  </Space>
                  <Tag color={systemStatus.api_server.status === 'healthy' ? 'green' : 'red'}>
                    {systemStatus.api_server.status === 'healthy' ? '正常' : '异常'}
                  </Tag>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <DatabaseOutlined />
                    <Text>数据服务</Text>
                  </Space>
                  <Tag color={systemStatus.data_service.status === 'healthy' ? 'green' : 'red'}>
                    {systemStatus.data_service.status === 'healthy' ? '正常' : '异常'}
                  </Tag>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <RobotOutlined />
                    <Text>预测引擎</Text>
                  </Space>
                  <Tag color={systemStatus.prediction_engine.status === 'healthy' ? 'green' : 'red'}>
                    {systemStatus.prediction_engine.status === 'healthy' ? '正常' : '异常'}
                  </Tag>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <Text>活跃模型</Text>
                  </Space>
                  <Text strong>{systemStatus.prediction_engine.active_models}</Text>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <Text>运行任务</Text>
                  </Space>
                  <Text strong>{systemStatus.task_manager.running_tasks}</Text>
                </div>
              </Space>
            ) : (
              <Alert message="系统状态加载中..." type="info" />
            )}
          </Card>
        </Col>

        {/* 最近任务 */}
        <Col xs={24} lg={12}>
          <Card 
            title="最近任务" 
            extra={
              <Button 
                type="link" 
                size="small"
                onClick={() => router.push('/tasks')}
              >
                查看全部
              </Button>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={recentTasks}
              renderItem={(task) => (
                <List.Item
                  actions={[
                    <Tag key="status" color={getTaskStatusColor(task.status)}>
                      {getTaskStatusText(task.status)}
                    </Tag>
                  ]}
                >
                  <List.Item.Meta
                    avatar={<Avatar icon={<RobotOutlined />} />}
                    title={
                      <Button 
                        type="link" 
                        size="small"
                        onClick={() => router.push(`/tasks/${task.task_id}`)}
                        style={{ padding: 0, height: 'auto' }}
                      >
                        {task.task_name}
                      </Button>
                    }
                    description={
                      <Space>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {new Date(task.created_at).toLocaleString()}
                        </Text>
                        {task.status === 'running' && (
                          <Progress 
                            percent={task.progress} 
                            size="small" 
                            style={{ width: 100 }}
                          />
                        )}
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* 快速操作 */}
      <Card title="快速操作" style={{ marginTop: 16 }}>
        <Space wrap>
          <Button 
            type="primary" 
            icon={<RobotOutlined />}
            onClick={() => router.push('/tasks/create')}
          >
            创建预测任务
          </Button>
          <Button 
            icon={<DatabaseOutlined />}
            onClick={() => router.push('/data-management')}
          >
            数据管理
          </Button>
          <Button 
            icon={<RiseOutlined />}
            onClick={() => router.push('/predictions')}
          >
            预测分析
          </Button>
          <Button 
            icon={<ApiOutlined />}
            onClick={() => router.push('/settings')}
          >
            系统设置
          </Button>
        </Space>
      </Card>
    </div>
  );
}