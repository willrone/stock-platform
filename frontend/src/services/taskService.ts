/**
 * 任务管理服务
 *
 * 处理与任务相关的API调用，包括：
 * - 任务创建
 * - 任务查询
 * - 任务状态更新
 * - 任务结果获取
 */

import { apiRequest } from './api';
import type { CreateTaskRequest, PredictionResult, Task, TaskListResponse } from '../types/task';

export type { CreateTaskRequest, PredictionResult, Task, TaskListResponse } from '../types/task';

/** 导出格式。`csv` 为前端期望值，当前后端只保证 `excel` / `pdf` 可用。 */
export type TaskExportFormat = 'excel' | 'csv' | 'pdf';

/** 后端导出接口返回的数据体（标准信封的 data 部分） */
export interface TaskExportPayload {
  download_url?: string;
  file_name?: string;
  file_size?: number;
}

/** 导出结果：要么是文件流（Blob），要么是下载路径 */
export interface TaskExportResult {
  blob: Blob | null;
  fileName: string;
  format: TaskExportFormat;
  downloadUrl: string | null;
  /** 是否因为后端不支持所选格式而降级 */
  fallback: boolean;
}

export interface BatchDeleteFailure {
  taskId: string;
  message: string;
}

export interface BatchDeleteResult {
  succeeded: string[];
  failed: BatchDeleteFailure[];
}

/** 从任意错误对象里取 HTTP 状态码 */
const extractStatus = (error: unknown): number | undefined => {
  if (error && typeof error === 'object') {
    const candidate = error as {
      status?: unknown;
      response?: { status?: unknown };
    };
    if (typeof candidate.status === 'number') {
      return candidate.status;
    }
    if (typeof candidate.response?.status === 'number') {
      return candidate.response.status;
    }
  }
  return undefined;
};

/** 后端返回「不支持的导出格式」时，可降级重试 */
const isUnsupportedFormatError = (error: unknown): boolean => {
  const status = extractStatus(error);
  if (status !== 400 && status !== 422) {
    return false;
  }
  const message = error instanceof Error ? error.message : '';
  return message.includes('导出格式') || message.includes('format');
};

/** 端点不存在（404 / 405）时，可降级为其它实现 */
const isEndpointMissingError = (error: unknown): boolean => {
  const status = extractStatus(error);
  return status === 404 || status === 405 || status === 501;
};

/** 构造带 token 的请求头（用于直接 fetch 下载文件，绕过 axios 的响应归一化） */
const buildAuthHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {};
  if (typeof window !== 'undefined') {
    const token = window.localStorage.getItem('access_token');
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }
  return headers;
};

/** 触发浏览器下载 */
const triggerBrowserDownload = (url: string, fileName: string): void => {
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.rel = 'noopener';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// 任务服务类
export class TaskService {
  /**
   * 创建新的预测任务
   */
  static async createTask(request: CreateTaskRequest): Promise<Task> {
    return apiRequest.post<Task>('/tasks', request);
  }

  /**
   * 获取任务列表
   *
   * @param status 状态筛选
   * @param limit 分页大小
   * @param offset 分页偏移
   * @param taskType 任务类型筛选（如 `backtest`）。后端支持时生效；
   *   不支持时后端会忽略该参数，前端仍需自行按 task_type 过滤。
   */
  static async getTasks(
    status?: string,
    limit: number = 20,
    offset: number = 0,
    taskType?: string
  ): Promise<TaskListResponse> {
    const params: Record<string, string | number> = { limit, offset };
    if (status) {
      params.status = status;
    }
    if (taskType) {
      params.task_type = taskType;
    }

    return apiRequest.get<TaskListResponse>('/tasks', params);
  }

  /**
   * 获取任务详情
   */
  static async getTaskDetail(taskId: string): Promise<Task> {
    return apiRequest.get<Task>(`/tasks/${taskId}`);
  }

  /**
   * 删除任务
   * @param taskId 任务ID
   * @param force 是否强制删除运行中的任务
   */
  static async deleteTask(taskId: string, force: boolean = false): Promise<void> {
    const url = force ? `/tasks/${taskId}?force=true` : `/tasks/${taskId}`;
    return apiRequest.delete(url);
  }

  /**
   * 重新运行任务
   */
  static async retryTask(taskId: string): Promise<Task> {
    return apiRequest.post<Task>(`/tasks/${taskId}/retry`);
  }

  /**
   * 停止运行中的任务
   */
  static async stopTask(taskId: string): Promise<Task> {
    return apiRequest.post<Task>(`/tasks/${taskId}/stop`);
  }

  /**
   * 获取任务的预测结果
   */
  static async getTaskResults(taskId: string): Promise<PredictionResult[]> {
    const task = await this.getTaskDetail(taskId);
    // 转换数据格式以匹配PredictionResult接口
    return (
      task.results?.predictions?.map(pred => ({
        stock_code: pred.stock_code,
        predicted_direction: pred.predicted_direction,
        predicted_return: pred.predicted_return || 0,
        confidence_score: pred.confidence_score,
        confidence_interval: pred.confidence_interval || { lower: 0, upper: 0 },
        risk_assessment: {
          value_at_risk: pred.risk_assessment?.value_at_risk || 0,
          volatility: pred.risk_assessment?.volatility || 0,
          max_drawdown: (pred.risk_assessment as any)?.max_drawdown || 0,
          sharpe_ratio: (pred.risk_assessment as any)?.sharpe_ratio || 0,
        },
      })) || []
    );
  }

  /**
   * 获取预测任务的历史预测序列
   */
  static async getPredictionSeries(
    taskId: string,
    stockCode: string,
    lookbackDays?: number
  ): Promise<{
    stock_code: string;
    series: Array<{ date: string; actual: number; predicted: number }>;
  }> {
    const params: Record<string, string | number> = { stock_code: stockCode };
    if (lookbackDays !== undefined && lookbackDays > 0) {
      params.lookback_days = lookbackDays;
    }
    return apiRequest.get(`/tasks/${taskId}/prediction-series`, params);
  }

  /**
   * 导出任务结果。
   *
   * 后端契约：`POST /tasks/{task_id}/export`，body `{ format }`，
   * 响应 `StandardResponse`，`data` 为 `{ download_url, file_name, file_size }`。
   *
   * 前端做了两层兼容：
   * 1. 后端若返回文件路径（`download_url`）→ 走浏览器下载；
   * 2. 后端若直接返回文件流（Blob）→ 直接存盘。
   *
   * 格式降级：当前后端只接受 `pdf` / `excel`（`csv` 会 400「不支持的导出格式」），
   * 因此当请求的格式被拒时，自动回退到 `excel` 并在返回值里如实告知实际格式。
   */
  static async exportTaskResults(
    taskId: string,
    format: TaskExportFormat = 'excel'
  ): Promise<TaskExportResult> {
    const requestedFormat = format;

    let payload: TaskExportPayload | Blob | null = null;
    let actualFormat = requestedFormat;

    try {
      payload = await apiRequest.post<TaskExportPayload | Blob>(
        `/tasks/${taskId}/export`,
        { format: requestedFormat }
      );
    } catch (error) {
      // 后端不支持该格式时降级到 excel（后端已确认支持的最小集合）
      if (requestedFormat !== 'excel' && isUnsupportedFormatError(error)) {
        payload = await apiRequest.post<TaskExportPayload | Blob>(`/tasks/${taskId}/export`, {
          format: 'excel',
        });
        actualFormat = 'excel';
      } else {
        throw error;
      }
    }

    if (payload instanceof Blob) {
      return {
        blob: payload,
        fileName: `task_${taskId}_results.${actualFormat}`,
        format: actualFormat,
        downloadUrl: null,
        fallback: actualFormat !== requestedFormat,
      };
    }

    const downloadUrl = payload?.download_url ?? null;
    const fileName = payload?.file_name ?? `task_${taskId}_results.${actualFormat}`;

    if (!downloadUrl) {
      throw new Error('导出接口未返回下载地址');
    }

    return {
      blob: null,
      fileName,
      format: actualFormat,
      downloadUrl,
      fallback: actualFormat !== requestedFormat,
    };
  }

  /**
   * 触发浏览器下载导出的结果文件。
   *
   * 优先用 `download_url`（后端返回的文件路径），必要时用文件流。
   * 走 `<a download>` 而不是 `window.open`，避免被浏览器拦截新窗口。
   */
  static async downloadExportResult(result: TaskExportResult): Promise<void> {
    if (result.blob) {
      const url = window.URL.createObjectURL(result.blob);
      try {
        triggerBrowserDownload(url, result.fileName);
      } finally {
        window.URL.revokeObjectURL(url);
      }
      return;
    }

    if (!result.downloadUrl) {
      throw new Error('没有可下载的导出文件');
    }

    // 后端返回的 download_url 是后端相对路径（如 /api/v1/files/download/x.xlsx），
    // 需要带 Authorization 头拉取后再存盘，否则拿到的是登录页 HTML。
    const response = await fetch(result.downloadUrl, {
      headers: buildAuthHeaders(),
    });
    if (!response.ok) {
      throw new Error(`下载导出文件失败（HTTP ${response.status}）`);
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    try {
      triggerBrowserDownload(url, result.fileName);
    } finally {
      window.URL.revokeObjectURL(url);
    }
  }

  /**
   * 批量删除任务。
   *
   * 后端契约（期望）：`POST /tasks/batch-delete`，body `{ task_ids: string[] }`。
   * 若后端尚未提供该端点（404/405），自动降级为循环调用已存在的
   * `DELETE /tasks/{task_id}`，并如实返回成功/失败明细。
   */
  static async batchDeleteTasks(
    taskIds: string[],
    force: boolean = false
  ): Promise<BatchDeleteResult> {
    if (taskIds.length === 0) {
      return { succeeded: [], failed: [] };
    }

    try {
      await apiRequest.post('/tasks/batch-delete', { task_ids: taskIds, force });
      return { succeeded: [...taskIds], failed: [] };
    } catch (error) {
      // 端点不存在（404/405）→ 降级为逐条删除
      if (!isEndpointMissingError(error)) {
        throw error;
      }
    }

    const succeeded: string[] = [];
    const failed: BatchDeleteFailure[] = [];

    for (const taskId of taskIds) {
      try {
        await TaskService.deleteTask(taskId, force);
        succeeded.push(taskId);
      } catch (error) {
        failed.push({
          taskId,
          message: error instanceof Error ? error.message : '删除失败',
        });
      }
    }

    return { succeeded, failed };
  }

  /**
   * 获取任务统计信息
   */
  static async getTaskStats(): Promise<{
    total: number;
    completed: number;
    running: number;
    failed: number;
    success_rate: number;
  }> {
    return apiRequest.get('/tasks/stats');
  }

  /**
   * 重建任务（基于已有任务的配置创建新任务）
   */
  static async rebuildTask(
    taskId: string,
    params: { task_name?: string; config_override?: Record<string, any> }
  ): Promise<Task> {
    return apiRequest.post<Task>(`/tasks/${taskId}/rebuild`, params);
  }

  /**
   * 获取任务创建时的完整配置快照
   * 从 GET /api/v1/tasks/{taskId}/config 获取
   * 用于在任务详情页展示参数快照面板
   */
  static async getTaskConfig(taskId: string): Promise<Record<string, any>> {
    try {
      return await apiRequest.get<Record<string, any>>(`/tasks/${taskId}/config`);
    } catch (error: any) {
      // 旧任务可能没有保存快照，返回 404 时静默处理
      if (error?.status === 404) {
        return {};
      }
      throw error;
    }
  }
}
