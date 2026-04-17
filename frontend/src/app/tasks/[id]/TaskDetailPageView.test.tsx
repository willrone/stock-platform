import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

import { TaskDetailPageView } from './TaskDetailPageView';
import type { TaskDetailPageModel } from './types';

jest.mock('../../../components/common/LoadingSpinner', () => ({
  LoadingSpinner: ({ text }: { text: string }) => <div>{text}</div>,
}));

jest.mock('../../../components/backtest/SaveStrategyConfigDialog', () => ({
  SaveStrategyConfigDialog: ({ isOpen }: { isOpen: boolean }) =>
    isOpen ? <div>保存策略配置对话框</div> : null,
}));

jest.mock('./TaskDetailContent', () => ({
  TaskDetailContent: () => <div>任务详情内容区</div>,
}));

function createModel(overrides: Partial<TaskDetailPageModel> = {}): TaskDetailPageModel {
  return {
    taskId: 'task-123',
    currentTask: {
      task_id: 'task-123',
      task_name: '回测任务A',
      task_type: 'backtest',
      status: 'completed',
      progress: 100,
      stock_codes: ['000001.SZ'],
      model_id: 'model-a',
      created_at: '2026-04-08T00:00:00.000Z',
    },
    loading: false,
    predictions: [],
    refreshing: false,
    selectedStock: '',
    setSelectedStock: jest.fn(),
    backtestDetailedData: null,
    backtestSummaryData: null,
    backtestOverviewData: null,
    adaptedRiskData: null,
    adaptedPerformanceData: null,
    loadingBacktestData: false,
    selectedBacktestTab: 'overview',
    setSelectedBacktestTab: jest.fn(),
    selectedPredictionTab: 'chart',
    setSelectedPredictionTab: jest.fn(),
    isDeleteOpen: false,
    isSaveConfigOpen: false,
    deleteForce: false,
    setDeleteForce: jest.fn(),
    savingConfig: false,
    selectedStocksPage: 1,
    setSelectedStocksPage: jest.fn(),
    strategyConfigInfo: null,
    loadBacktestDetailedData: jest.fn(async () => undefined),
    loadTaskDetail: jest.fn(async () => undefined),
    handleRefresh: jest.fn(async () => undefined),
    handleRetry: jest.fn(async () => undefined),
    handleDelete: jest.fn(async () => undefined),
    handleExport: jest.fn(async () => undefined),
    handleSaveConfig: jest.fn(async () => undefined),
    handleBack: jest.fn(),
    handleRebuild: jest.fn(),
    openDeleteDialog: jest.fn(),
    closeDeleteDialog: jest.fn(),
    openSaveConfigDialog: jest.fn(),
    closeSaveConfigDialog: jest.fn(),
    ...overrides,
  };
}

describe('TaskDetailPageView', () => {
  it('应该渲染标题、动作按钮和内容区域', () => {
    const model = createModel();

    render(<TaskDetailPageView model={model} />);

    expect(screen.getByText('回测任务A')).toBeInTheDocument();
    expect(screen.getByText('任务ID: task-123')).toBeInTheDocument();
    expect(screen.getByText('导出结果')).toBeInTheDocument();
    expect(screen.getByText('重建任务')).toBeInTheDocument();
    expect(screen.getByText('任务详情内容区')).toBeInTheDocument();
  });

  it('应该在空任务时渲染兜底态并支持返回', () => {
    const handleBack = jest.fn();
    const model = createModel({ currentTask: null, handleBack });

    render(<TaskDetailPageView model={model} />);

    expect(screen.getByText('任务不存在或已被删除')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '返回任务列表' }));
    expect(handleBack).toHaveBeenCalledTimes(1);
  });

  it('应该打开删除弹窗并执行删除', () => {
    const handleDelete = jest.fn(async () => undefined);
    const closeDeleteDialog = jest.fn();
    const model = createModel({
      isDeleteOpen: true,
      handleDelete,
      closeDeleteDialog,
    });

    render(<TaskDetailPageView model={model} />);

    fireEvent.click(screen.getByRole('button', { name: '删除' }));
    expect(handleDelete).toHaveBeenCalledTimes(1);
    expect(closeDeleteDialog).toHaveBeenCalledTimes(1);
  });
});
