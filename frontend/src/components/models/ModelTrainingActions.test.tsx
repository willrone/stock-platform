import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { ModelListTable } from './ModelListTable';
import { LiveTrainingModal } from './LiveTrainingModal';
import type { Model } from '../../types/model';

const trainingModel: Model = {
  model_id: 'model-training-1',
  model_name: '训练中的模型',
  model_type: 'lightgbm',
  version: '1.0.0',
  accuracy: 0.82,
  created_at: '2026-04-11T12:00:00Z',
  status: 'training',
  training_progress: 48,
  training_stage: 'training',
};

const readyModel: Model = {
  ...trainingModel,
  model_id: 'model-ready-1',
  model_name: '已完成模型',
  status: 'ready',
};

describe('model training action copy', () => {
  it('训练中模型的删除提示不应误导为取消训练', () => {
    render(
      <ModelListTable
        models={[trainingModel, readyModel]}
        trainingProgress={{}}
        getStatusColor={() => 'warning'}
        getStatusText={status => status}
        getStageText={stage => stage}
        onShowTrainingReport={jest.fn()}
        onShowLiveTraining={jest.fn()}
        onCreateBacktest={jest.fn()}
        onDeleteModel={jest.fn()}
        deleting={false}
      />
    );

    expect(
      screen.getAllByRole('button', { name: '删除模型记录（不会停止后台训练）' }).length
    ).toBeGreaterThan(0);
    expect(screen.queryByRole('button', { name: '取消训练并删除该模型' })).not.toBeInTheDocument();
  });

  it('实时训练弹窗应提供可点击的停止训练动作', () => {
    const onStopTraining = jest.fn();

    render(
      <LiveTrainingModal
        isOpen
        onClose={jest.fn()}
        onStopTraining={onStopTraining}
        stopping={false}
        modelId={trainingModel.model_id}
        models={[trainingModel]}
        trainingProgress={{
          [trainingModel.model_id]: {
            progress: 48,
            stage: 'training',
            message: '正在训练',
            metrics: {},
          },
        }}
        getStageText={stage => stage}
      />
    );

    const stopButton = screen.getByRole('button', { name: '停止训练' });
    expect(stopButton).toBeEnabled();

    fireEvent.click(stopButton);
    expect(onStopTraining).toHaveBeenCalledWith(trainingModel.model_id);
  });
});
