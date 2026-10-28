/**
 * 任务管理 UI 状态
 *
 * 【重要 · state P0-8】本 store **只保存跨页面的纯 UI 态**（分页 / 筛选 / 排序 /
 * 加载标志）。服务端数据（任务列表、total、当前任务详情）**一律由页面本地持有**。
 *
 * 原因：旧的实现把服务端任务列表放进模块级单例，导致
 *   A. `/tasks` 写入当前页 10 条 → `/backtest` 写入全量 1000 条 → 两页互相污染；
 *   B. `total` 与实际渲染条数来自不同请求，"共 N 条"对不上；
 *   C. `/tasks` 设置的 `statusFilter` / `currentPage` 会带进 `/backtest`；
 *   D. `updateTask` 用 `task.task_id === taskId` 匹配，task_id 缺失时整批更新不到。
 *
 * 保留 `Task` 类型的再导出，供 `components/backtest/BacktestTaskStatus.tsx` 等
 * 仍从本模块引入类型的文件使用。
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export type { Task } from '../types/task';

/** 任务列表排序字段 */
export type TaskSortField = 'created_at' | 'updated_at' | 'name' | 'status';
export type TaskSortOrder = 'asc' | 'desc';

interface TaskState {
  // ── 纯 UI 态（可跨页保留）──────────────────────────────
  /** 当前页码（1 起） */
  currentPage: number;
  /** 每页条数 */
  pageSize: number;
  /** 状态筛选（null = 全部） */
  statusFilter: string | null;
  /** 排序字段 */
  sortBy: TaskSortField;
  /** 排序方向 */
  sortOrder: TaskSortOrder;
  /** 是否有任务正在创建（跨页展示"创建中"） */
  creating: boolean;

  // ── Actions ────────────────────────────────────────────
  setPagination: (page: number, pageSize: number) => void;
  setStatusFilter: (status: string | null) => void;
  setSort: (sortBy: TaskSortField, sortOrder: TaskSortOrder) => void;
  setCreating: (creating: boolean) => void;
  /** 重置筛选 / 分页（离开列表页时调用，避免状态带入下一页） */
  resetUiState: () => void;
}

export const useTaskStore = create<TaskState>()(
  devtools(
    set => ({
      currentPage: 1,
      pageSize: 10,
      statusFilter: null,
      sortBy: 'created_at',
      sortOrder: 'desc',
      creating: false,

      setPagination: (currentPage, pageSize) =>
        set({ currentPage, pageSize }, false, 'setPagination'),

      setStatusFilter: statusFilter =>
        set(
          {
            statusFilter,
            currentPage: 1, // 换筛选条件回到第一页
          },
          false,
          'setStatusFilter'
        ),

      setSort: (sortBy, sortOrder) => set({ sortBy, sortOrder }, false, 'setSort'),

      setCreating: creating => set({ creating }, false, 'setCreating'),

      resetUiState: () =>
        set(
          {
            currentPage: 1,
            statusFilter: null,
            sortBy: 'created_at',
            sortOrder: 'desc',
          },
          false,
          'resetUiState'
        ),
    }),
    {
      name: 'task-store',
    }
  )
);
