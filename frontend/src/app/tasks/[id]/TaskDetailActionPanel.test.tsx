import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

import { TaskDetailActionPanel } from './TaskDetailActionPanel';

describe('TaskDetailActionPanel', () => {
  it('completed 任务应展示导出和重建动作', () => {
    const onRefresh = jest.fn();
    const onExport = jest.fn();
    const onRebuild = jest.fn();

    render(
      <TaskDetailActionPanel
        task={{
          task_id: 'task-1',
          task_name: '任务1',
          status: 'completed',
          progress: 100,
          stock_codes: [],
          model_id: 'model-1',
          created_at: '2026-04-08T00:00:00.000Z',
        }}
        refreshing={false}
        onRefresh={onRefresh}
        onRetry={jest.fn()}
        onExport={onExport}
        onRebuild={onRebuild}
        onDelete={jest.fn()}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: '刷新' }));
    fireEvent.click(screen.getByRole('button', { name: '导出结果' }));
    fireEvent.click(screen.getByRole('button', { name: '重建任务' }));

    expect(onRefresh).toHaveBeenCalledTimes(1);
    expect(onExport).toHaveBeenCalledTimes(1);
    expect(onRebuild).toHaveBeenCalledTimes(1);
  });

  it('failed 任务应展示重新运行动作', () => {
    const onRetry = jest.fn();

    render(
      <TaskDetailActionPanel
        task={{
          task_id: 'task-2',
          task_name: '任务2',
          status: 'failed',
          progress: 30,
          stock_codes: [],
          model_id: 'model-2',
          created_at: '2026-04-08T00:00:00.000Z',
        }}
        refreshing={false}
        onRefresh={jest.fn()}
        onRetry={onRetry}
        onExport={jest.fn()}
        onRebuild={jest.fn()}
        onDelete={jest.fn()}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: '重新运行' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
