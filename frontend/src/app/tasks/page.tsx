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
  Table,
  Button,
  Space,
  Tag,
  Progress,
  Select,
  Input,
  Modal,
  message,
  Tooltip,
  Typography,
  Row,
  Col,
  Statistic,
} from 'antd';
import {
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useTaskStore, Task } from '../../stores/useTaskStore';
import { TaskService } from '../../services/taskService';
import { wsService } from '../../services/websocket';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;

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
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);

  // 加载任务列表
  const loadTasks = async (page = currentPage, size = pageSize, status = statusFilter) => {
    setLoading(true);
    try {
      const offset = (page - 1) * size;
      const result = await TaskService.getTasks(status || undefined, size, offset);
      setTasks(result.tasks, result.total);
    } catch (error) {
      message.error('加载任务列表失败');
      console.error('加载任务失败:', error);
    } finally {
      setLoading(false);
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
      message.success(`任务 ${data.task_id} 已完成`);
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      updateTask(data.task_id, {
        status: 'failed',
        error_message: data.error,
      });
      message.error(`任务 ${data.task_id} 执行失败`);
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

  // 处理分页变化
  const handleTableChange = (pagination: any) => {
    const { current, pageSize: size } = pagination;
    setPagination(current, size);
    loadTasks(current, size, statusFilter);
  };

  // 处理状态筛选
  const handleStatusFilter = (value: string) => {
    setStatusFilter(value || null);
    setPagination(1, pageSize);
    loadTasks(1, pageSize, value || null);
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
  const handleDeleteTask = async (taskId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个任务吗？此操作不可撤销。',
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await TaskService.deleteTask(taskId);
          message.success('任务删除成功');
          loadTasks();
        } catch (error) {
          message.error('删除任务失败');
        }
      },
    });
  };

  // 批量删除任务
  const handleBatchDelete = () => {
    if (selectedRowKeys.length === 0) {
      message.warning('请选择要删除的任务');
      return;
    }

    Modal.confirm({
      title: '批量删除确认',
      content: `确定要删除选中的 ${selectedRowKeys.length} 个任务吗？此操作不可撤销。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await TaskService.batchDeleteTasks(selectedRowKeys);
          message.success(`成功删除 ${selectedRowKeys.length} 个任务`);
          setSelectedRowKeys([]);
          loadTasks();
        } catch (error) {
          message.error('批量删除失败');
        }
      },
    });
  };

  // 表格列定义
  const columns = [
    {
      title: '任务名称',
      dataIndex: 'task_name',
      key: 'task_name',
      render: (text: string, record: Task) => (
        <Button
          type="link"
          onClick={() => handleViewTask(record.task_id)}
          style={{ padding: 0, height: 'auto' }}
        >
          {text}
        </Button>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: Task['status']) => getStatusTag(status),
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 120,
      render: (progress: number, record: Task) => (
        <Progress
          percent={progress}
          size="small"
          status={record.status === 'failed' ? 'exception' : undefined}
        />
      ),
    },
    {
      title: '股票数量',
      dataIndex: 'stock_codes',
      key: 'stock_count',
      width: 100,
      render: (stockCodes: string[]) => (
        <Text>{stockCodes?.length || 0}</Text>
      ),
    },
    {
      title: '模型',
      dataIndex: 'model_id',
      key: 'model_id',
      width: 120,
      render: (modelId: string) => (
        <Tag>{modelId}</Tag>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (time: string) => (
        <Text type="secondary">
          {new Date(time).toLocaleString()}
        </Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_: any, record: Task) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handleViewTask(record.task_id)}
            />
          </Tooltip>
          {record.status === 'running' && (
            <Tooltip title="暂停任务">
              <Button
                type="text"
                icon={<PauseCircleOutlined />}
                onClick={() => {
                  // TODO: 实现暂停功能
                  message.info('暂停功能开发中');
                }}
              />
            </Tooltip>
          )}
          {record.status === 'failed' && (
            <Tooltip title="重新运行">
              <Button
                type="text"
                icon={<PlayCircleOutlined />}
                onClick={async () => {
                  try {
                    await TaskService.retryTask(record.task_id);
                    message.success('任务已重新启动');
                    loadTasks();
                  } catch (error) {
                    message.error('重新运行失败');
                  }
                }}
              />
            </Tooltip>
          )}
          <Tooltip title="删除任务">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteTask(record.task_id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 计算统计数据
  const stats = {
    total: tasks.length,
    running: tasks.filter(t => t.status === 'running').length,
    completed: tasks.filter(t => t.status === 'completed').length,
    failed: tasks.filter(t => t.status === 'failed').length,
  };

  if (loading && tasks.length === 0) {
    return <LoadingSpinner text="加载任务列表..." />;
  }

  return (
    <div>
      {/* 页面标题和统计 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>任务管理</Title>
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col span={6}>
            <Card>
              <Statistic title="总任务数" value={stats.total} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中"
                value={stats.running}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成"
                value={stats.completed}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="失败"
                value={stats.failed}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>
      </div>

      {/* 操作栏 */}
      <Card style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleCreateTask}
              >
                创建任务
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={handleRefresh}
                loading={loading}
              >
                刷新
              </Button>
              {selectedRowKeys.length > 0 && (
                <Button
                  danger
                  icon={<DeleteOutlined />}
                  onClick={handleBatchDelete}
                >
                  批量删除 ({selectedRowKeys.length})
                </Button>
              )}
            </Space>
          </Col>
          <Col>
            <Space>
              <Search
                placeholder="搜索任务名称"
                allowClear
                style={{ width: 200 }}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                onSearch={(value) => {
                  // TODO: 实现搜索功能
                  message.info('搜索功能开发中');
                }}
              />
              <Select
                placeholder="筛选状态"
                allowClear
                style={{ width: 120 }}
                value={statusFilter}
                onChange={handleStatusFilter}
              >
                <Option value="created">已创建</Option>
                <Option value="running">运行中</Option>
                <Option value="completed">已完成</Option>
                <Option value="failed">失败</Option>
              </Select>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 任务列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={tasks}
          rowKey="task_id"
          loading={loading}
          rowSelection={{
            selectedRowKeys,
            onChange: (keys: React.Key[]) => setSelectedRowKeys(keys as string[]),
          }}
          pagination={{
            current: currentPage,
            pageSize,
            total,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          onChange={handleTableChange}
        />
      </Card>
    </div>
  );
}