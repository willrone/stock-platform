import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { TrainingReportModal } from './TrainingReportModal';
import { DataService } from '../../services/dataService';

jest.mock('echarts-for-react', () => () => <div data-testid="echarts" />);

jest.mock('../../services/dataService', () => ({
  DataService: {
    getTrainingReport: jest.fn(),
  },
}));

const mockedGetTrainingReport = DataService.getTrainingReport as jest.MockedFunction<
  typeof DataService.getTrainingReport
>;

describe('TrainingReportModal bridge summaries', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders cost-vs-gross and per-stock bridge insights from evaluation report', async () => {
    mockedGetTrainingReport.mockResolvedValue({
      model_id: 'model-bridge-1',
      model_name: '桥接模型',
      model_type: 'lightgbm',
      training_summary: {
        training_duration: 12,
        total_samples: 1234,
        train_samples: 800,
        validation_samples: 200,
        test_samples: 234,
      },
      performance_metrics: {
        accuracy: 0.61,
      },
      hyperparameters: {},
      training_data_info: {
        stock_codes: ['600519.SH'],
        start_date: '2020-01-01',
        end_date: '2020-08-01',
      },
      cost_vs_gross_gap_summary: {
        task_count: 2,
        largest_cost_gap: {
          task_name: 'alpha360-2020-short-window',
          window_label: '2020-01-01→2020-08-01',
          gross_minus_net_value_gap: 56077.57,
        },
        best_gross_return: {
          task_name: 'alpha360-testfull',
          total_return_without_cost: 0.2056,
        },
        best_net_return: {
          task_name: 'alpha360-smoke',
          total_return: -0.0263,
        },
      },
      per_stock_ranking_preference: {
        best_overall: {
          stock_code: '600519.SH',
          total_pnl: 90630.16,
          signal_count: 331,
        },
        worst_overall: {
          stock_code: '000001.SZ',
          total_pnl: -109778.5,
          signal_count: 404,
        },
        stocks: [
          {
            stock_code: '600519.SH',
            task_mentions: 2,
            positive_task_count: 2,
            negative_task_count: 0,
            total_pnl: 90630.16,
            signal_count: 331,
          },
        ],
      },
      ranking_overlap_summary: {
        available: false,
        windows: [],
      },
      event_replay_summary: {
        available: false,
        events: [],
      },
    });

    render(<TrainingReportModal isOpen onClose={jest.fn()} modelId="model-bridge-1" />);

    expect(await screen.findByText('正式任务桥接洞察')).toBeInTheDocument();
    expect(screen.getByText('成本 vs gross gap')).toBeInTheDocument();
    expect(screen.getByText('最大成本吞噬窗口')).toBeInTheDocument();
    expect(screen.getByText('alpha360-2020-short-window')).toBeInTheDocument();
    expect(screen.getByText('个股偏好 / 个股贡献')).toBeInTheDocument();
    expect(screen.getAllByText('600519.SH').length).toBeGreaterThan(0);
    expect(screen.getByText('000001.SZ')).toBeInTheDocument();

    await waitFor(() => {
      expect(mockedGetTrainingReport).toHaveBeenCalledWith('model-bridge-1');
    });
  });
});
