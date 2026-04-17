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

export interface TaskDetailPageModel {
  taskId: string;
  currentTask: Task | null;
  loading: boolean;
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
  selectedStocksPage: number;
  setSelectedStocksPage: React.Dispatch<React.SetStateAction<number>>;
  strategyConfigInfo: StrategyConfigInfo | null;
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
