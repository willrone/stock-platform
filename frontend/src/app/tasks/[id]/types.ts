import type React from 'react';

import type {
  BacktestDetailedResult,
  BacktestOverviewData,
  BacktestSummaryData,
} from '../../../types/backtest';
import type {
  BacktestDetailedPerformanceViewModel,
  BacktestDetailedRiskViewModel,
} from '../../../services/backtestDataAdapter';
import type { PredictionResult, Task } from '../../../types/task';
import type { StrategyConfigInfo } from './taskDetailUtils';

/**
 * 任务详情加载失败的分类
 *
 * - `not_found`   404 —— 任务确实不存在
 * - `unauthorized` 401 —— 会话过期/无权限，需要重新登录
 * - `rate_limited` 429 —— 被限流，可退避后重试
 * - `network`     其它错误（网络/服务异常），可重试
 */
export type TaskDetailErrorKind = 'not_found' | 'unauthorized' | 'rate_limited' | 'network';

export interface TaskDetailLoadError {
  kind: TaskDetailErrorKind;
  message: string;
  /** HTTP 状态码（若有） */
  status?: number;
}

export interface TaskDetailPageModel {
  taskId: string;
  currentTask: Task | null;
  loading: boolean;
  /**
   * 【P1-04】加载失败的分类信息。
   * 旧的实现把任何失败都显示成"任务不存在或已被删除"——
   * 网络抖动、401 会话过期、429 限流都会被误报成"任务没了"。
   * kind 决定 UI 文案与是否可重试。
   */
  loadError: TaskDetailLoadError | null;
  predictions: PredictionResult[];
  refreshing: boolean;
  selectedStock: string;
  setSelectedStock: React.Dispatch<React.SetStateAction<string>>;
  backtestDetailedData: BacktestDetailedResult | null;
  backtestSummaryData: BacktestSummaryData | null;
  backtestOverviewData: BacktestOverviewData | null;
  adaptedRiskData: BacktestDetailedRiskViewModel | null;
  adaptedPerformanceData: BacktestDetailedPerformanceViewModel | null;
  loadingBacktestData: boolean;
  selectedBacktestTab: string;
  setSelectedBacktestTab: React.Dispatch<React.SetStateAction<string>>;
  selectedPredictionTab: string;
  setSelectedPredictionTab: React.Dispatch<React.SetStateAction<string>>;
  isDeleteOpen: boolean;
  isSaveConfigOpen: boolean;
  deleteForce: boolean;
  setDeleteForce: React.Dispatch<React.SetStateAction<boolean>>;
  savingConfig: boolean;
  /** 【P3-03】导出进行中 */
  exporting: boolean;
  /** 【P3-03】导出失败原因（非 null → UI 展示错误） */
  exportError: string | null;
  selectedStocksPage: number;
  setSelectedStocksPage: React.Dispatch<React.SetStateAction<number>>;
  strategyConfigInfo: StrategyConfigInfo | null;
  configSnapshot: Record<string, any>;
  configSnapshotLoading: boolean;
  loadBacktestDetailedData: (force?: boolean) => Promise<void>;
  loadTaskDetail: () => Promise<void>;
  handleRefresh: () => Promise<void>;
  handleRetry: () => Promise<void>;
  handleDelete: () => Promise<void>;
  handleExport: () => Promise<void>;
  handleSaveConfig: (configName: string, description: string) => Promise<void>;
  handleBack: () => void;
  handleRebuild: () => void;
  openDeleteDialog: () => void;
  closeDeleteDialog: () => void;
  openSaveConfigDialog: () => void;
  closeSaveConfigDialog: () => void;
}
