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
  task_type?: string; // 添加任务类型字段
  status: 'created' | 'running' | 'completed' | 'failed';
  progress: number;
  stock_codes: string[];
  model_id: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  result?: Record<string, unknown>; // 添加原始结果字段
  backtest_results?: Record<string, unknown>; // 添加顶层回测结果字段
  config?: {
    backtest_config?: {
      strategy_name?: string;
      strategy_config?: Record<string, unknown>;
      start_date?: string;
      end_date?: string;
      initial_cash?: number;
      commission_rate?: number;
      slippage_rate?: number;
    };
    prediction_config?: Record<string, unknown>;
    optimization_config?: Record<string, unknown>;
    stock_codes?: string[];
    model_id?: string;
    [key: string]: unknown;
  };
  results?: {
    total_stocks: number;
    successful_predictions: number;
    average_confidence: number;
    backtest_results?: Record<string, unknown>;
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
  optimization_info?: {
    n_trials: number;
    completed_trials: number;
    running_trials?: number;
    pruned_trials?: number;
    failed_trials?: number;
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
    (set, _get) => ({
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
      setTasks: (tasks, total) =>
        set(
          {
            tasks,
            total,
            loading: false,
          },
          false,
          'setTasks'
        ),

      addTask: task =>
        set(
          state => ({
            tasks: [task, ...state.tasks],
            total: state.total + 1,
          }),
          false,
          'addTask'
        ),

      updateTask: (taskId, updates) =>
        set(
          state => ({
            tasks: state.tasks.map(task =>
              task.task_id === taskId ? { ...task, ...updates } : task
            ),
            currentTask:
              state.currentTask?.task_id === taskId
                ? { ...state.currentTask, ...updates }
                : state.currentTask,
          }),
          false,
          'updateTask'
        ),

      setCurrentTask: task => set({ currentTask: task }, false, 'setCurrentTask'),

      setLoading: loading => set({ loading }, false, 'setLoading'),

      setCreating: creating => set({ creating }, false, 'setCreating'),

      setPagination: (currentPage, pageSize) =>
        set(
          {
            currentPage,
            pageSize,
          },
          false,
          'setPagination'
        ),

      setStatusFilter: statusFilter =>
        set(
          {
            statusFilter,
            currentPage: 1, // 重置到第一页
          },
          false,
          'setStatusFilter'
        ),

      clearTasks: () =>
        set(
          {
            tasks: [],
            currentTask: null,
            total: 0,
            currentPage: 1,
          },
          false,
          'clearTasks'
        ),
    }),
    {
      name: 'task-store',
    }
  )
);
