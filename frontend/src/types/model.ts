/**
 * 模型相关 API 合同类型。
 *
 * Backend source:
 * - backend/app/api/v1/model_dto.py
 * - backend/app/api/v1/models.py
 * - backend/app/api/v1/training_progress.py
 */

export type ModelStatus =
  | 'active'
  | 'inactive'
  | 'training'
  | 'ready'
  | 'failed'
  | 'cancelled'
  | 'deployed';

export interface ModelPerformanceMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  [key: string]: number | string | boolean | null | undefined | Record<string, unknown> | unknown[];
}

export interface ModelTrainingInfo {
  training_data_period?: Record<string, string>;
  hyperparameters?: Record<string, unknown>;
  stock_codes?: string[];
}

export interface Model {
  model_id: string;
  model_name: string;
  model_type: string;
  version: string;
  accuracy: number;
  created_at: string;
  status: ModelStatus;
  description?: string;
  training_progress?: number;
  training_stage?: string;
  performance_metrics?: ModelPerformanceMetrics;
  training_info?: ModelTrainingInfo;
}

export interface TrainingProgressSnapshot {
  task_id: string;
  status: string;
  progress_percentage: number;
  created_at: string;
  updated_at: string;
  elapsed_time: number;
  current_epoch: number;
  total_epochs: number;
  current_batch: number;
  total_batches: number;
  current_loss: number | null;
  best_loss: number | null;
  current_accuracy: number | null;
  best_accuracy: number | null;
  learning_rate: number | null;
  estimated_remaining: number | null;
  stage: string;
}
