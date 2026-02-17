/**
 * 任务 WebSocket 实时更新 Hook
 */

import { useEffect } from 'react';
import { useTaskStore, Task } from '@/stores/useTaskStore';
import { TaskService } from '@/services/taskService';
import { wsService } from '@/services/websocket';

interface UseTaskWebSocketProps {
  taskId: string;
  currentTask: Task | null;
  onTaskCompleted?: () => void;
}

export function useTaskWebSocket({ taskId, currentTask, onTaskCompleted }: UseTaskWebSocketProps) {
  const { updateTask, setCurrentTask } = useTaskStore();

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

    const handleTaskCompleted = async (data: { task_id: string; results: Record<string, unknown> }) => {
      if (data.task_id === taskId) {
        // 重新加载任务详情以获取完整数据
        try {
          const task = await TaskService.getTaskDetail(taskId);
          const updatedTask = {
            ...task,
            status: 'completed' as const,
            progress: 100,
            completed_at: new Date().toISOString(),
          };

          setCurrentTask(updatedTask);
          updateTask(data.task_id, updatedTask);

          // 触发完成回调
          if (onTaskCompleted) {
            onTaskCompleted();
          }

          console.log('任务执行完成');
        } catch (error) {
          console.error('加载任务详情失败:', error);
        }
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
        console.error('任务执行失败');
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
  }, [taskId, currentTask, updateTask, setCurrentTask, onTaskCompleted]);
}
