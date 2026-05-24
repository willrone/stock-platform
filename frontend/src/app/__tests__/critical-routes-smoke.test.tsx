import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';

import DataManagementPage from '../data/page';
import MonitoringPage from '../monitoring/page';
import { DataService } from '../../services/dataService';

jest.mock('../../services/dataService', () => ({
  DataService: {
    getLocalStockList: jest.fn(),
    getDataServiceStatus: jest.fn(),
    getRemoteStockList: jest.fn(),
    getRemoteDataSummary: jest.fn(),
    getRemoteServiceLogs: jest.fn(),
    getErrorStatistics: jest.fn(),
    getAnomalies: jest.fn(),
    getDataQuality: jest.fn(),
  },
}));

jest.mock('../../services/websocket', () => ({
  wsService: {
    on: jest.fn(),
    off: jest.fn(),
  },
}));

jest.mock('../../components/monitoring/SystemHealthCard', () => ({
  SystemHealthCard: () => <div data-testid="system-health-card">系统健康卡片</div>,
}));

jest.mock('../../components/monitoring/PerformanceMetricsCard', () => ({
  PerformanceMetricsCard: () => <div data-testid="performance-metrics-card">性能指标卡片</div>,
}));

const mockedDataService = DataService as jest.Mocked<typeof DataService>;

describe('关键路由 smoke 测试', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockedDataService.getLocalStockList.mockResolvedValue({
      stocks: [],
      stock_codes: [],
      total_stocks: 0,
    });
    mockedDataService.getDataServiceStatus.mockResolvedValue({
      service_url: 'http://127.0.0.1:5002',
      is_connected: true,
      last_check: '2026-05-24T00:00:00Z',
      response_time: 25,
    });
    mockedDataService.getRemoteStockList.mockResolvedValue({
      stocks: [],
      stock_codes: [],
      total_stocks: 0,
    });
    mockedDataService.getRemoteDataSummary.mockResolvedValue({
      total_stocks: 0,
      total_records: 0,
      complete_stocks: 0,
      incomplete_stocks: 0,
      missing_stocks: 0,
      last_update: '2026-05-24T00:00:00Z',
    });
    mockedDataService.getRemoteServiceLogs.mockResolvedValue({
      content: '',
      lines: 0,
    });

    mockedDataService.getErrorStatistics.mockResolvedValue({
      time_range_hours: 24,
      total_error_types: 0,
      total_errors: 0,
      error_statistics: [],
    });
    mockedDataService.getAnomalies.mockResolvedValue({
      total_anomalies: 0,
      by_severity: { high: 0, medium: 0, low: 0 },
      anomalies: [],
      detection_time: '2026-05-24T00:00:00Z',
    });
    mockedDataService.getDataQuality.mockResolvedValue({
      overall_score: 100,
      checks: [],
    });
  });

  it('渲染 /data 页面并触发关键数据接口', async () => {
    render(<DataManagementPage />);

    expect(await screen.findByTestId('data-route-smoke')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '数据管理' })).toBeInTheDocument();

    await waitFor(() => {
      expect(mockedDataService.getLocalStockList).toHaveBeenCalledTimes(1);
      expect(mockedDataService.getDataServiceStatus).toHaveBeenCalledTimes(1);
      expect(mockedDataService.getRemoteStockList).toHaveBeenCalledTimes(1);
    });
  });

  it('渲染 /monitoring 页面并触发关键监控接口', async () => {
    render(<MonitoringPage />);

    expect(await screen.findByTestId('monitoring-route-smoke')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '系统监控' })).toBeInTheDocument();
    expect(screen.getByTestId('system-health-card')).toBeInTheDocument();
    expect(screen.getByTestId('performance-metrics-card')).toBeInTheDocument();

    await waitFor(() => {
      expect(mockedDataService.getErrorStatistics).toHaveBeenCalledWith(24);
      expect(mockedDataService.getAnomalies).toHaveBeenCalledTimes(1);
      expect(mockedDataService.getDataQuality).toHaveBeenCalledTimes(1);
    });
  });
});
