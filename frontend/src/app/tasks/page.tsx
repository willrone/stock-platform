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
  CardHeader,
  CardBody,
  Button,
  Input,
  Select,
  SelectItem,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Progress,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Pagination,
  Tooltip,
} from '@heroui/react';
import {
  Plus,
  RefreshCw,
  Search,
  Eye,
  Play,
  Pause,
  Trash2,
  Filter,
  BarChart3,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useTaskStore, Task } from '../../stores/useTaskStore';
import { TaskService } from '../../services/taskService';
import { wsService } from '../../services/websocket';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

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
  const { isOpen: isDeleteOpen, onOpen: onDeleteOpen, onClose: onDeleteClose } = useDisclosure();
  const { isOpen: isBatchDeleteOpen, onOpen: onBatchDeleteOpen, onClose: onBatchDeleteClose } = useDisclosure();

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
      failed: { color: 'danger' as const, text: '失败' },
    };
    
    const config = statusConfig[status] || statusConfig.created;
    return <Chip color={config.color} variant="flat" size="sm">{config.text}</Chip>;
  };

  // 处理分页变化
  const handlePageChange = (page: number) => {
    setPagination(page, pageSize);
    loadTasks(page, pageSize, statusFilter);
  };

  // 处理状态筛选
  const handleStatusFilter = (keys: Set<string>) => {
    const status = Array.from(keys)[0] || null;
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
    if (!taskToDelete) return;
    
    try {
      await TaskService.deleteTask(taskToDelete);
      console.log('任务删除成功');
      loadTasks();
    } catch (error) {
      console.error('删除任务失败');
    } finally {
      setTaskToDelete(null);
      onDeleteClose();
    }
  };

  // 批量删除任务
  const handleBatchDelete = async () => {
    const taskIds = Array.from(selectedKeys);
    if (taskIds.length === 0) return;

    try {
      await TaskService.batchDeleteTasks(taskIds);
      console.log(`成功删除 ${taskIds.length} 个任务`);
      setSelectedKeys(new Set());
      loadTasks();
    } catch (error) {
      console.error('批量删除失败');
    } finally {
      onBatchDeleteClose();
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


  if (loading && tasks.length === 0) {
    return <LoadingSpinner text="加载任务列表..." />;
  }

  return (
    <div className="space-y-6">
      {/* 页面标题和统计 */}
      <div>
        <h1 className="text-2xl font-bold mb-4">任务管理</h1>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardBody className="text-center">
              <div className="flex items-center justify-center mb-2">
                <BarChart3 className="w-6 h-6 text-primary" />
              </div>
              <p className="text-2xl font-bold">{stats.total}</p>
              <p className="text-sm text-default-500">总任务数</p>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="text-center">
              <p className="text-2xl font-bold text-primary">{stats.running}</p>
              <p className="text-sm text-default-500">运行中</p>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="text-center">
              <p className="text-2xl font-bold text-success">{stats.completed}</p>
              <p className="text-sm text-default-500">已完成</p>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="text-center">
              <p className="text-2xl font-bold text-danger">{stats.failed}</p>
              <p className="text-sm text-default-500">失败</p>
            </CardBody>
          </Card>
        </div>
      </div>

      {/* 操作栏 */}
      <Card>
        <CardBody>
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-4 md:space-y-0">
            <div className="flex flex-wrap gap-2">
              <Button
                color="primary"
                startContent={<Plus className="w-4 h-4" />}
                onPress={handleCreateTask}
              >
                创建任务
              </Button>
              <Button
                variant="light"
                startContent={<RefreshCw className="w-4 h-4" />}
                onPress={handleRefresh}
                isLoading={loading}
              >
                刷新
              </Button>
              {selectedKeys.size > 0 && (
                <Button
                  color="danger"
                  variant="light"
                  startContent={<Trash2 className="w-4 h-4" />}
                  onPress={onBatchDeleteOpen}
                >
                  批量删除 ({selectedKeys.size})
                </Button>
              )}
            </div>
            
            <div className="flex flex-wrap gap-2">
              <Input
                placeholder="搜索任务名称"
                startContent={<Search className="w-4 h-4" />}
                value={searchText}
                onValueChange={setSearchText}
                className="w-48"
              />
              <Select
                placeholder="筛选状态"
                startContent={<Filter className="w-4 h-4" />}
                selectedKeys={statusFilter ? [statusFilter] : []}
                onSelectionChange={(keys) => {
                  const selectedKeys = Array.from(keys);
                  handleStatusFilter(new Set(selectedKeys.map(String)));
                }}
                className="w-32"
              >
                <SelectItem key="created">已创建</SelectItem>
                <SelectItem key="running">运行中</SelectItem>
                <SelectItem key="completed">已完成</SelectItem>
                <SelectItem key="failed">失败</SelectItem>
              </Select>
            </div>
          </div>
        </CardBody>
      </Card>

      {/* 任务列表 */}
      <Card>
        <CardBody>
          <Table
            aria-label="任务列表"
            selectionMode="multiple"
            selectedKeys={selectedKeys}
            onSelectionChange={(keys) => setSelectedKeys(new Set(Array.from(keys).map(String)))}
          >
            <TableHeader>
              <TableColumn>任务名称</TableColumn>
              <TableColumn>状态</TableColumn>
              <TableColumn>进度</TableColumn>
              <TableColumn>股票数量</TableColumn>
              <TableColumn>模型</TableColumn>
              <TableColumn>创建时间</TableColumn>
              <TableColumn>操作</TableColumn>
            </TableHeader>
            <TableBody>
              {tasks.map((task) => (
                <TableRow key={task.task_id}>
                  <TableCell>
                    <Button
                      variant="light"
                      onPress={() => handleViewTask(task.task_id)}
                      className="p-0 h-auto min-w-0 justify-start"
                    >
                      {task.task_name}
                    </Button>
                  </TableCell>
                  <TableCell>
                    {getStatusChip(task.status)}
                  </TableCell>
                  <TableCell>
                    <Progress
                      value={task.progress}
                      color={task.status === 'failed' ? 'danger' : 'primary'}
                      size="sm"
                      className="w-20"
                    />
                  </TableCell>
                  <TableCell>
                    <span className="text-default-600">{task.stock_codes?.length || 0}</span>
                  </TableCell>
                  <TableCell>
                    <Chip variant="flat" size="sm">{task.model_id}</Chip>
                  </TableCell>
                  <TableCell>
                    <span className="text-default-500 text-sm">
                      {new Date(task.created_at).toLocaleString()}
                    </span>
                  </TableCell>
                  <TableCell>
                    <div className="flex space-x-1">
                      <Tooltip content="查看详情">
                        <Button
                          isIconOnly
                          variant="light"
                          size="sm"
                          onPress={() => handleViewTask(task.task_id)}
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                      </Tooltip>
                      
                      {task.status === 'running' && (
                        <Tooltip content="暂停任务">
                          <Button
                            isIconOnly
                            variant="light"
                            size="sm"
                            onPress={() => {
                              console.log('暂停功能开发中');
                            }}
                          >
                            <Pause className="w-4 h-4" />
                          </Button>
                        </Tooltip>
                      )}
                      
                      {task.status === 'failed' && (
                        <Tooltip content="重新运行">
                          <Button
                            isIconOnly
                            variant="light"
                            size="sm"
                            onPress={() => handleRetryTask(task.task_id)}
                          >
                            <Play className="w-4 h-4" />
                          </Button>
                        </Tooltip>
                      )}
                      
                      <Tooltip content="删除任务">
                        <Button
                          isIconOnly
                          variant="light"
                          size="sm"
                          color="danger"
                          onPress={() => {
                            setTaskToDelete(task.task_id);
                            onDeleteOpen();
                          }}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </Tooltip>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          
          {/* 分页 */}
          {total > pageSize && (
            <div className="flex justify-center mt-4">
              <Pagination
                total={Math.ceil(total / pageSize)}
                page={currentPage}
                onChange={handlePageChange}
                showControls
              />
            </div>
          )}
        </CardBody>
      </Card>

      {/* 删除确认对话框 */}
      <Modal isOpen={isDeleteOpen} onClose={onDeleteClose}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>确认删除</ModalHeader>
              <ModalBody>
                <p>确定要删除这个任务吗？此操作不可撤销。</p>
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onPress={onClose}>
                  取消
                </Button>
                <Button color="danger" onPress={handleDeleteTask}>
                  删除
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>

      {/* 批量删除确认对话框 */}
      <Modal isOpen={isBatchDeleteOpen} onClose={onBatchDeleteClose}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>批量删除确认</ModalHeader>
              <ModalBody>
                <p>确定要删除选中的 {selectedKeys.size} 个任务吗？此操作不可撤销。</p>
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onPress={onClose}>
                  取消
                </Button>
                <Button color="danger" onPress={handleBatchDelete}>
                  删除
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  );}