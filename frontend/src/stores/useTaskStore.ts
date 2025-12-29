/**
 * 任务管理状态
 * 
 * 管理预测任务的状态，包括：
 * - 任务列表
 * - 任务详情
 * - 任务创建和管理
 * - 实时状态更新
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export interface Task {
  task_id: string;
  task_name: string;
  status: 'created' | 'running' | 'completed' | 'failed';
  progress: number;
  stock_codes: string[];
  model_id: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  results?: {
    total_stocks: number;
    successful_predictions: number;
    average_confidence: number;
    predictions: Array<{
      stock_code: string;
      predicted_direction: number;
      predicted_return?: number;
      confidence_score: number;
      confidence_interval?: {
        lower: number;
        upper: number;
      };
      risk_assessment?: {
        value_at_risk: number;
        volatility: number;
      };
    }>;
  };
}

interface TaskState {
  // 任务数据
  tasks: Task[];
  currentTask: Task | null;
  
  // 分页和过滤
  total: number;
  currentPage: number;
  pageSize: number;
  statusFilter: string | null;
  
  // 加载状态
  loading: boolean;
  creating: boolean;
  
  // Actions
  setTasks: (tasks: Task[], total: number) => void;
  addTask: (task: Task) => void;
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  setCurrentTask: (task: Task | null) => void;
  setLoading: (loading: boolean) => void;
  setCreating: (creating: boolean) => void;
  setPagination: (page: number, pageSize: number) => void;
  setStatusFilter: (status: string | null) => void;
  clearTasks: () => void;
}

export const useTaskStore = create<TaskState>()(
  devtools(
    (set, get) => ({
      // 初始状态
      tasks: [],
      currentTask: null,
      total: 0,
      currentPage: 1,
      pageSize: 10,
      statusFilter: null,
      loading: false,
      creating: false,

      // Actions
      setTasks: (tasks, total) => set({
        tasks,
        total,
        loading: false,
      }, false, 'setTasks'),

      addTask: (task) => set((state) => ({
        tasks: [task, ...state.tasks],
        total: state.total + 1,
      }), false, 'addTask'),

      updateTask: (taskId, updates) => set((state) => ({
        tasks: state.tasks.map(task =>
          task.task_id === taskId ? { ...task, ...updates } : task
        ),
        currentTask: state.currentTask?.task_id === taskId
          ? { ...state.currentTask, ...updates }
          : state.currentTask,
      }), false, 'updateTask'),

      setCurrentTask: (task) => set({ currentTask: task }, false, 'setCurrentTask'),

      setLoading: (loading) => set({ loading }, false, 'setLoading'),

      setCreating: (creating) => set({ creating }, false, 'setCreating'),

      setPagination: (currentPage, pageSize) => set({
        currentPage,
        pageSize,
      }, false, 'setPagination'),

      setStatusFilter: (statusFilter) => set({
        statusFilter,
        currentPage: 1, // 重置到第一页
      }, false, 'setStatusFilter'),

      clearTasks: () => set({
        tasks: [],
        currentTask: null,
        total: 0,
        currentPage: 1,
      }, false, 'clearTasks'),
    }),
    {
      name: 'task-store',
    }
  )
);