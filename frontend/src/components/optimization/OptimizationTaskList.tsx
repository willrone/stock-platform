/**
 * 优化任务列表组件
 */

'use client';

import React from 'react';
import {
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Progress,
  Button,
  Tooltip,
} from '@heroui/react';
import { Eye, RefreshCw } from 'lucide-react';
import { OptimizationTask } from '../../services/optimizationService';

interface OptimizationTaskListProps {
  tasks: OptimizationTask[];
  onTaskSelect: (taskId: string) => void;
  onRefresh: () => void;
}

export default function OptimizationTaskList({
  tasks,
  onTaskSelect,
  onRefresh,
}: OptimizationTaskListProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'failed':
        return 'danger';
      case 'created':
      case 'queued':
        return 'default';
      default:
        return 'default';
    }
  };

  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      created: '已创建',
      queued: '排队中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      cancelled: '已取消',
    };
    return statusMap[status] || status;
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">优化任务列表</h3>
        <Button
          size="sm"
          variant="light"
          onPress={onRefresh}
          startContent={<RefreshCw size={16} />}
        >
          刷新
        </Button>
      </div>

      {tasks.length === 0 ? (
        <div className="text-center py-8 text-default-500">
          暂无优化任务，请创建新任务
        </div>
      ) : (
        <Table aria-label="优化任务列表">
          <TableHeader>
            <TableColumn>任务名称</TableColumn>
            <TableColumn>策略</TableColumn>
            <TableColumn>状态</TableColumn>
            <TableColumn>进度</TableColumn>
            <TableColumn>试验数</TableColumn>
            <TableColumn>最佳得分</TableColumn>
            <TableColumn>创建时间</TableColumn>
            <TableColumn>操作</TableColumn>
          </TableHeader>
          <TableBody>
            {tasks.map((task) => (
              <TableRow key={task.task_id}>
                <TableCell>
                  <div className="font-medium">{task.task_name}</div>
                </TableCell>
                <TableCell>
                  <Chip size="sm" variant="flat">
                    {task.strategy_name}
                  </Chip>
                </TableCell>
                <TableCell>
                  <Chip color={getStatusColor(task.status)} size="sm" variant="flat">
                    {getStatusText(task.status)}
                  </Chip>
                </TableCell>
                <TableCell>
                  <Progress
                    value={task.progress}
                    color={task.status === 'failed' ? 'danger' : 'primary'}
                    size="sm"
                    className="w-24"
                  />
                </TableCell>
                <TableCell>
                  {task.status === 'completed' ? (
                    <span>{task.n_trials} / {task.n_trials}</span>
                  ) : (
                    <span>- / {task.n_trials}</span>
                  )}
                </TableCell>
                <TableCell>
                  {task.best_score !== undefined ? (
                    <span className="font-medium text-success">
                      {task.best_score.toFixed(4)}
                    </span>
                  ) : (
                    <span className="text-default-400">-</span>
                  )}
                </TableCell>
                <TableCell>
                  <span className="text-sm text-default-500">
                    {task.created_at
                      ? new Date(task.created_at).toLocaleString('zh-CN')
                      : '-'}
                  </span>
                </TableCell>
                <TableCell>
                  <Tooltip content="查看详情">
                    <Button
                      size="sm"
                      variant="light"
                      isIconOnly
                      onPress={() => onTaskSelect(task.task_id)}
                    >
                      <Eye size={16} />
                    </Button>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}

