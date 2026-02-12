/**
 * 回测结果标签页组件
 */

import React, { useState } from 'react';
import { Card, CardContent, Box, Tabs, Tab, Typography } from '@mui/material';
import {
  BarChart3,
  LineChart,
  FileText,
  AlertTriangle,
  PieChart,
  Calendar,
  Activity,
  TrendingUp,
} from 'lucide-react';
import { Task } from '@/stores/useTaskStore';
import BacktestOverview from '@/components/backtest/BacktestOverview';
import { CostAnalysis } from '@/components/backtest/CostAnalysis';
import { TradeHistoryTable } from '@/components/backtest/TradeHistoryTable';
import { SignalHistoryTable } from '@/components/backtest/SignalHistoryTable';
import dynamic from 'next/dynamic';

// 动态导入图表组件
const InteractiveChartsContainer = dynamic(
  () => import('@/components/charts/InteractiveChartsContainer'),
  {
    ssr: false,
    loading: () => <div className="h-96 flex items-center justify-center">加载交互式图表中...</div>,
  }
);

const PositionAnalysis = dynamic(
  () => import('@/components/backtest/PositionAnalysis').then(mod => ({ default: mod.PositionAnalysis })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载持仓分析中...</div> }
);

const RiskAnalysis = dynamic(
  () => import('@/components/backtest/RiskAnalysis').then(mod => ({ default: mod.RiskAnalysis })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载风险分析中...</div> }
);

const PerformanceBreakdown = dynamic(
  () => import('@/components/backtest/PerformanceBreakdown').then(mod => ({ default: mod.PerformanceBreakdown })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载绩效分析中...</div> }
);

interface BacktestTabsProps {
  task: Task;
  taskId: string;
  selectedStock: string;
  backtestDetailedData: any;
  adaptedRiskData: any;
  adaptedPerformanceData: any;
  loadingBacktestData: boolean;
  onTabChange?: (tab: string) => void;
  renderStrategyConfig: () => React.ReactNode;
  renderPerformanceMonitor: () => React.ReactNode;
}

export function BacktestTabs({
  task,
  taskId,
  selectedStock,
  backtestDetailedData,
  adaptedRiskData,
  adaptedPerformanceData,
  loadingBacktestData,
  onTabChange,
  renderStrategyConfig,
  renderPerformanceMonitor,
}: BacktestTabsProps) {
  const [selectedTab, setSelectedTab] = useState<string>('overview');

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
    setSelectedTab(newValue);
    if (onTabChange) {
      onTabChange(newValue);
    }
  };

  const backtestData = task.result || task.results?.backtest_results || task.backtest_results;

  return (
    <Card>
      <CardContent sx={{ px: { xs: 1, sm: 2, md: 3 }, py: { xs: 1.5, sm: 2 } }}>
        <Box>
          <Tabs
            value={selectedTab}
            onChange={handleTabChange}
            aria-label="回测结果展示"
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{
              minHeight: { xs: 40, sm: 48 },
              '& .MuiTab-root': {
                minHeight: { xs: 40, sm: 48 },
                minWidth: { xs: 'auto', sm: 90 },
                px: { xs: 1, sm: 2 },
                fontSize: { xs: '0.75rem', sm: '0.875rem' },
              },
            }}
          >
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <BarChart3 size={14} />
                  <span>概览</span>
                </Box>
              }
              value="overview"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <LineChart size={14} />
                  <span>图表</span>
                </Box>
              }
              value="charts"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <FileText size={14} />
                  <span>交易</span>
                </Box>
              }
              value="trades"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <AlertTriangle size={14} />
                  <span>信号</span>
                </Box>
              }
              value="signals"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <PieChart size={14} />
                  <span>持仓</span>
                </Box>
              }
              value="positions"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Calendar size={14} />
                  <span>月度</span>
                </Box>
              }
              value="monthly"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Activity size={14} />
                  <span>风险</span>
                </Box>
              }
              value="risk"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <TrendingUp size={14} />
                  <span>绩效</span>
                </Box>
              }
              value="performance"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Activity size={14} />
                  <span>性能</span>
                </Box>
              }
              value="perf_monitor"
            />
          </Tabs>

          <Box sx={{ mt: 2 }}>
            {selectedTab === 'overview' && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {renderStrategyConfig()}
                <BacktestOverview backtestData={backtestData} loading={loadingBacktestData} />
                <CostAnalysis backtestData={backtestData} loading={loadingBacktestData} />
              </Box>
            )}

            {selectedTab === 'charts' && (
              <Box sx={{ mt: 2 }}>
                <InteractiveChartsContainer
                  taskId={taskId}
                  stockCode={selectedStock || task?.stock_codes?.[0]}
                  stockCodes={task?.stock_codes || []}
                  backtestData={backtestData}
                />
              </Box>
            )}

            {selectedTab === 'trades' && (
              <Box sx={{ mt: 2 }}>
                <TradeHistoryTable
                  taskId={taskId}
                  onTradeClick={trade => {
                    console.log('查看交易详情:', trade);
                  }}
                />
              </Box>
            )}

            {selectedTab === 'signals' && (
              <Box sx={{ mt: 2 }}>
                <SignalHistoryTable
                  taskId={taskId}
                  onSignalClick={signal => {
                    console.log('查看信号详情:', signal);
                  }}
                />
              </Box>
            )}

            {selectedTab === 'positions' && (
              <Box sx={{ mt: 2 }}>
                {loadingBacktestData ? (
                  <div className="text-center text-default-500 py-8">
                    <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>持仓分析数据加载中...</p>
                  </div>
                ) : backtestDetailedData?.position_analysis ? (
                  <PositionAnalysis
                    positionAnalysis={backtestDetailedData.position_analysis}
                    stockCodes={task.stock_codes || []}
                    taskId={taskId}
                  />
                ) : (
                  <div className="text-center text-default-500 py-8">
                    <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>暂无持仓分析数据</p>
                  </div>
                )}
              </Box>
            )}

            {selectedTab === 'monthly' && (
              <Box sx={{ mt: 2 }}>
                {backtestDetailedData?.monthly_returns ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography variant="h6" component="h4" sx={{ fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' } }}>
                      月度收益热力图
                    </Typography>
                    <Box sx={{ overflowX: 'auto' }}>
                      <Box
                        sx={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(12, 1fr)',
                          gap: 0.5,
                          minWidth: 500,
                        }}
                      >
                      {backtestDetailedData.monthly_returns.map((monthData: any) => (
                        <Box
                          key={`${monthData.year}-${monthData.month}`}
                          sx={{
                            p: 1,
                            textAlign: 'center',
                            fontSize: '0.75rem',
                            borderRadius: 1,
                            bgcolor:
                              monthData.monthly_return >= 0 ? 'success.light' : 'error.light',
                            color:
                              monthData.monthly_return >= 0 ? 'success.dark' : 'error.dark',
                          }}
                          title={`${monthData.year}年${monthData.month}月: ${(
                            monthData.monthly_return * 100
                          ).toFixed(2)}%`}
                        >
                          {monthData.month}月
                          <br />
                          {(monthData.monthly_return * 100).toFixed(1)}%
                        </Box>
                      ))}
                      </Box>
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Calendar size={48} color="#999" style={{ margin: '0 auto 16px' }} />
                    <Typography variant="body2" color="text.secondary">
                      月度分析数据加载中...
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {selectedTab === 'risk' && (
              <Box sx={{ mt: 2 }}>
                {adaptedRiskData ? (
                  <RiskAnalysis
                    taskId={taskId}
                    riskMetrics={adaptedRiskData.riskMetrics}
                    returnDistribution={adaptedRiskData.returnDistribution}
                    rollingMetrics={adaptedRiskData.rollingMetrics}
                  />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />
                    <Typography variant="body2" color="text.secondary">
                      风险分析数据加载中...
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {selectedTab === 'performance' && (
              <Box sx={{ mt: 2 }}>
                {adaptedPerformanceData ? (
                  <PerformanceBreakdown
                    taskId={taskId}
                    monthlyPerformance={adaptedPerformanceData.monthlyPerformance}
                    yearlyPerformance={adaptedPerformanceData.yearlyPerformance}
                    seasonalAnalysis={adaptedPerformanceData.seasonalAnalysis}
                    benchmarkComparison={adaptedPerformanceData.benchmarkComparison}
                  />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <TrendingUp size={48} color="#999" style={{ margin: '0 auto 16px' }} />
                    <Typography variant="body2" color="text.secondary">
                      绩效分解数据加载中...
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {selectedTab === 'perf_monitor' && renderPerformanceMonitor()}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
