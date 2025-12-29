/**
 * 任务详情页面
 * 
 * 显示任务的详细信息，包括：
 * - 任务基本信息
 * - 实时进度更新
 * - 预测结果展示
 * - 操作控制
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Tag,
  Progress,
  Button,
  Space,
  Descriptions,
  Table,
  Alert,
  Statistic,
  message,
  Modal,
} from 'antd';
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  DownloadOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useRouter, useParams } from 'next/navigation';
import { useTaskStore, Task } from '../../../stores/useTaskStore';
import { TaskService, PredictionResult } from '../../../services/taskService';
import { wsService } from '../../../services/websocket';
import { LoadingSpinner } from '../../../components/common/LoadingSpinner';

const { Title, Text } = Typography;

export default function TaskDetailPage() {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;

  const { currentTask, setCurrentTask, updateTask } = useTaskStore();
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  // 加载任务详情
  const loadTaskDetail = async () => {
    try {
      const task = await TaskService.getTaskDetail(taskId);
      setCurrentTask(task);
      
      // 如果任务已完成，加载预测结果
      if (task.status === 'completed' && task.results) {
        const results = await TaskService.getTaskResults(taskId);
        setPredictions(results);
      }
    } catch (error) {
      message.error('加载任务详情失败');
      console.error('加载任务详情失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 初始化加载
  useEffect(() => {
    if (taskId) {
      loadTaskDetail();
      // 订阅任务更新
      wsService.subscribeToTask(taskId);
    }

    return () => {
      if (taskId) {
        wsService.unsubscribeFromTask(taskId);
      }
    };
  }, [taskId]);

  // WebSocket实时更新
  useEffect(() => {
    const handleTaskProgress = (data: { task_id: string; progress: number; status: string }) => {
      if (data.task_id === taskId) {
        updateTask(data.task_id, {
          progress: data.progress,
          status: data.status as Task['status'],
        });
        
        if (currentTask) {
          setCurrentTask({
            ...currentTask,
            progress: data.progress,
            status: data.status as Task['status'],
          });
        }
      }
    };

    const handleTaskCompleted = async (data: { task_id: string; results: any }) => {
      if (data.task_id === taskId) {
        const updatedTask = {
          ...currentTask!,
          status: 'completed' as const,
          progress: 100,
          results: data.results,
          completed_at: new Date().toISOString(),
        };
        
        setCurrentTask(updatedTask);
        updateTask(data.task_id, updatedTask);
        
        // 加载预测结果
        try {
          const results = await TaskService.getTaskResults(taskId);
          setPredictions(results);
        } catch (error) {
          console.error('加载预测结果失败:', error);
        }
        
        message.success('任务执行完成');
      }
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      if (data.task_id === taskId) {
        const updatedTask = {
          ...currentTask!,
          status: 'failed' as const,
          error_message: data.error,
        };
        
        setCurrentTask(updatedTask);
        updateTask(data.task_id, updatedTask);
        message.error('任务执行失败');
      }
    };

    wsService.on('task:progress', handleTaskProgress);
    wsService.on('task:completed', handleTaskCompleted);
    wsService.on('task:failed', handleTaskFailed);

    return () => {
      wsService.off('task:progress', handleTaskProgress);
      wsService.off('task:completed', handleTaskCompleted);
      wsService.off('task:failed', handleTaskFailed);
    };
  }, [taskId, currentTask, updateTask, setCurrentTask]);

  // 刷新任务
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadTaskDetail();
    setRefreshing(false);
  };

  // 重新运行任务
  const handleRetry = async () => {
    try {
      await TaskService.retryTask(taskId);
      message.success('任务已重新启动');
      await loadTaskDetail();
    } catch (error) {
      message.error('重新运行失败');
    }
  };

  // 删除任务
  const handleDelete = () => {
    Modal.confirm({
      title: '确认删除',
      icon: <ExclamationCircleOutlined />,
      content: '确定要删除这个任务吗？此操作不可撤销。',
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await TaskService.deleteTask(taskId);
          message.success('任务删除成功');
          router.push('/tasks');
        } catch (error) {
          message.error('删除任务失败');
        }
      },
    });
  };

  // 导出结果
  const handleExport = async () => {
    try {
      const blob = await TaskService.exportTaskResults(taskId, 'csv');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `task_${taskId}_results.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      message.success('结果导出成功');
    } catch (error) {
      message.error('导出失败');
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  // 获取状态标签
  const getStatusTag = (status: Task['status']) => {
    const statusConfig = {
      created: { color: 'default', text: '已创建' },
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
    };
    
    const config = statusConfig[status] || statusConfig.created;
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 预测结果表格列
  const predictionColumns = [
    {
      title: '股票代码',
      dataIndex: 'stock_code',
      key: 'stock_code',
    },
    {
      title: '预测方向',
      dataIndex: 'predicted_direction',
      key: 'predicted_direction',
      render: (direction: number) => (
        <Tag color={direction > 0 ? 'green' : direction < 0 ? 'red' : 'default'}>
          {direction > 0 ? '上涨' : direction < 0 ? '下跌' : '持平'}
        </Tag>
      ),
    },
    {
      title: '预测收益率',
      dataIndex: 'predicted_return',
      key: 'predicted_return',
      render: (value: number) => (
        <Text style={{ color: value > 0 ? '#52c41a' : value < 0 ? '#ff4d4f' : undefined }}>
          {(value * 100).toFixed(2)}%
        </Text>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          format={(percent) => `${percent?.toFixed(1)}%`}
        />
      ),
    },
    {
      title: '置信区间',
      key: 'confidence_interval',
      render: (_: any, record: PredictionResult) => (
        <Text type="secondary">
          [{(record.confidence_interval.lower * 100).toFixed(2)}%, {(record.confidence_interval.upper * 100).toFixed(2)}%]
        </Text>
      ),
    },
    {
      title: 'VaR',
      dataIndex: ['risk_assessment', 'value_at_risk'],
      key: 'var',
      render: (value: number) => (
        <Text style={{ color: '#ff4d4f' }}>
          {(value * 100).toFixed(2)}%
        </Text>
      ),
    },
  ];

  if (loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!currentTask) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Text>任务不存在或已被删除</Text>
        <br />
        <Button type="primary" onClick={handleBack} style={{ marginTop: 16 }}>
          返回任务列表
        </Button>
      </div>
    );
  }

  return (
    <div>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={handleBack}>
            返回
          </Button>
          <Title level={2} style={{ margin: 0 }}>
            {currentTask.task_name}
          </Title>
          {getStatusTag(currentTask.status)}
        </Space>
      </div>

      <Row gutter={24}>
        <Col span={16}>
          {/* 任务进度 */}
          <Card title="任务进度" style={{ marginBottom: 16 }}>
            <Progress
              percent={currentTask.progress}
              status={currentTask.status === 'failed' ? 'exception' : undefined}
              strokeWidth={8}
            />
            {currentTask.status === 'running' && (
              <Text type="secondary" style={{ marginTop: 8, display: 'block' }}>
                任务正在执行中，请耐心等待...
              </Text>
            )}
            {currentTask.status === 'failed' && currentTask.error_message && (
              <Alert
                message="任务执行失败"
                description={currentTask.error_message}
                type="error"
                style={{ marginTop: 16 }}
              />
            )}
          </Card>

          {/* 任务信息 */}
          <Card title="任务信息" style={{ marginBottom: 16 }}>
            <Descriptions column={2}>
              <Descriptions.Item label="任务ID">
                {currentTask.task_id}
              </Descriptions.Item>
              <Descriptions.Item label="模型">
                <Tag>{currentTask.model_id}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="股票数量">
                {currentTask.stock_codes.length}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {new Date(currentTask.created_at).toLocaleString()}
              </Descriptions.Item>
              {currentTask.completed_at && (
                <Descriptions.Item label="完成时间">
                  {new Date(currentTask.completed_at).toLocaleString()}
                </Descriptions.Item>
              )}
            </Descriptions>

            <div style={{ marginTop: 16 }}>
              <Text strong>选择的股票:</Text>
              <div style={{ marginTop: 8 }}>
                <Space wrap>
                  {currentTask.stock_codes.map(code => (
                    <Tag key={code}>{code}</Tag>
                  ))}
                </Space>
              </div>
            </div>
          </Card>

          {/* 预测结果 */}
          {currentTask.status === 'completed' && predictions.length > 0 && (
            <Card
              title="预测结果"
              extra={
                <Button
                  icon={<DownloadOutlined />}
                  onClick={handleExport}
                >
                  导出结果
                </Button>
              }
            >
              <Table
                columns={predictionColumns}
                dataSource={predictions}
                rowKey="stock_code"
                pagination={false}
                size="small"
              />
            </Card>
          )}
        </Col>

        <Col span={8}>
          {/* 操作面板 */}
          <Card title="操作" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                icon={<ReloadOutlined />}
                onClick={handleRefresh}
                loading={refreshing}
                block
              >
                刷新状态
              </Button>
              
              {currentTask.status === 'failed' && (
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleRetry}
                  block
                >
                  重新运行
                </Button>
              )}
              
              {currentTask.status === 'completed' && (
                <Button
                  icon={<DownloadOutlined />}
                  onClick={handleExport}
                  block
                >
                  导出结果
                </Button>
              )}
              
              <Button
                danger
                icon={<DeleteOutlined />}
                onClick={handleDelete}
                block
              >
                删除任务
              </Button>
            </Space>
          </Card>

          {/* 统计信息 */}
          {currentTask.results && (
            <Card title="统计信息">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="总股票数"
                    value={currentTask.results.total_stocks}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="成功预测"
                    value={currentTask.results.successful_predictions}
                  />
                </Col>
                <Col span={24} style={{ marginTop: 16 }}>
                  <Statistic
                    title="平均置信度"
                    value={currentTask.results.average_confidence}
                    precision={1}
                    suffix="%"
                    formatter={(value) => `${((value as number) * 100).toFixed(1)}%`}
                  />
                </Col>
              </Row>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
}