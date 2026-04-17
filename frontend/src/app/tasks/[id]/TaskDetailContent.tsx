'use client';

import React from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Typography,
} from '@mui/material';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Calendar,
  ChevronLeft,
  ChevronRight,
  Download,
  FileText,
  LineChart,
  PieChart,
  Play,
  RefreshCw,
  Save,
  Trash2,
  TrendingUp,
} from 'lucide-react';
import dynamic from 'next/dynamic';

import BacktestOverview from '../../../components/backtest/BacktestOverview';
import BacktestProgressMonitor from '../../../components/backtest/BacktestProgressMonitor';
import BacktestTaskStatus from '../../../components/backtest/BacktestTaskStatus';
import { CostAnalysis } from '../../../components/backtest/CostAnalysis';
import { SignalHistoryTable } from '../../../components/backtest/SignalHistoryTable';
import { TradeHistoryTable } from '../../../components/backtest/TradeHistoryTable';
import {
  getPredictionIcon,
  getPredictionText,
  getStrategyDisplayName,
  renderStrategyParameters,
} from './taskDetailUtils';
import type { TaskDetailPageModel } from './types';

const TradingViewChart = dynamic(() => import('../../../components/charts/TradingViewChart'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载图表中...</div>,
});

const PredictionChart = dynamic(() => import('../../../components/charts/PredictionChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载预测图表中...</div>,
});

const BacktestChart = dynamic(() => import('../../../components/charts/BacktestChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载回测图表中...</div>,
});

const InteractiveChartsContainer = dynamic(
  () => import('../../../components/charts/InteractiveChartsContainer'),
  {
    ssr: false,
    loading: () => <div className="h-96 flex items-center justify-center">加载交互式图表中...</div>,
  }
);

const PositionAnalysis = dynamic(
  () =>
    import('../../../components/backtest/PositionAnalysis').then(mod => ({
      default: mod.PositionAnalysis,
    })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载持仓分析中...</div> }
);

const RiskAnalysis = dynamic(
  () =>
    import('../../../components/backtest/RiskAnalysis').then(mod => ({
      default: mod.RiskAnalysis,
    })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载风险分析中...</div> }
);

const PerformanceBreakdown = dynamic(
  () =>
    import('../../../components/backtest/PerformanceBreakdown').then(mod => ({
      default: mod.PerformanceBreakdown,
    })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载绩效分析中...</div> }
);

interface TaskDetailContentProps {
  model: TaskDetailPageModel;
}

export function TaskDetailContent({ model }: TaskDetailContentProps): React.ReactNode {
  const { currentTask } = model;

  if (!currentTask) {
    return null;
  }

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 3 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {currentTask.task_type === 'backtest' &&
        (currentTask.status === 'running' || currentTask.status === 'created') ? (
          <BacktestProgressMonitor
            taskId={model.taskId}
            onComplete={() => {
              void model.loadTaskDetail();
            }}
            onError={() => {
              void model.loadTaskDetail();
            }}
            onCancel={() => {
              void model.loadTaskDetail();
            }}
          />
        ) : (
          <Card>
            <CardHeader title="任务进度" />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={currentTask.progress}
                  color={currentTask.status === 'failed' ? 'error' : 'primary'}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>
              {currentTask.task_type === 'hyperparameter_optimization' &&
                currentTask.optimization_info && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      已完成轮次: {currentTask.optimization_info.completed_trials} /{' '}
                      {currentTask.optimization_info.n_trials}
                    </Typography>
                  </Box>
                )}
              {currentTask.status === 'running' && (
                <Typography variant="caption" color="text.secondary">
                  任务正在执行中，请耐心等待...
                </Typography>
              )}
              {currentTask.status === 'failed' && currentTask.error_message && (
                <Box
                  sx={{
                    bgcolor: 'error.light',
                    border: 1,
                    borderColor: 'error.main',
                    borderRadius: 1,
                    p: 2,
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                    <AlertTriangle size={20} color="#d32f2f" style={{ marginTop: 2 }} />
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'error.dark' }}>
                        任务执行失败
                      </Typography>
                      <Typography
                        variant="caption"
                        sx={{ color: 'error.dark', mt: 0.5, display: 'block' }}
                      >
                        {currentTask.error_message}
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        )}

        {currentTask.task_type === 'backtest' ? (
          <Card>
            <CardContent>
              <Box>
                <Tabs
                  value={model.selectedBacktestTab}
                  onChange={(_event, newValue) => {
                    const tabKey = newValue as string;
                    model.setSelectedBacktestTab(tabKey);

                    if (
                      tabKey === 'positions' &&
                      currentTask.task_type === 'backtest' &&
                      currentTask.status === 'completed' &&
                      !model.backtestDetailedData &&
                      !model.loadingBacktestData
                    ) {
                      void model.loadBacktestDetailedData();
                    }
                  }}
                  aria-label="回测结果展示"
                  variant="scrollable"
                  scrollButtons="auto"
                >
                  <Tab label={<TabLabel icon={<BarChart3 size={16} />} text="概览" />} value="overview" />
                  <Tab label={<TabLabel icon={<LineChart size={16} />} text="交互式图表" />} value="charts" />
                  <Tab label={<TabLabel icon={<FileText size={16} />} text="交易记录" />} value="trades" />
                  <Tab label={<TabLabel icon={<AlertTriangle size={16} />} text="信号记录" />} value="signals" />
                  <Tab label={<TabLabel icon={<PieChart size={16} />} text="持仓分析" />} value="positions" />
                  <Tab label={<TabLabel icon={<Calendar size={16} />} text="月度分析" />} value="monthly" />
                  <Tab label={<TabLabel icon={<Activity size={16} />} text="风险分析" />} value="risk" />
                  <Tab label={<TabLabel icon={<TrendingUp size={16} />} text="绩效分解" />} value="performance" />
                  <Tab label={<TabLabel icon={<Activity size={16} />} text="性能分析" />} value="perf_monitor" />
                </Tabs>

                <Box sx={{ mt: 2 }}>
                  {model.selectedBacktestTab === 'overview' && (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {model.strategyConfigInfo && (
                        <Card>
                          <CardHeader
                            title={
                              <Box
                                sx={{
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  width: '100%',
                                }}
                              >
                                <Box>
                                  <Typography variant="h6" component="h4" sx={{ fontWeight: 600 }}>
                                    策略配置
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    策略: {getStrategyDisplayName(model.strategyConfigInfo.strategyName)}
                                  </Typography>
                                </Box>
                                <Button
                                  variant="outlined"
                                  color="primary"
                                  size="small"
                                  startIcon={<Save size={16} />}
                                  onClick={model.openSaveConfigDialog}
                                  disabled={
                                    !model.strategyConfigInfo.strategyName ||
                                    model.strategyConfigInfo.strategyName === '未知策略' ||
                                    Object.keys(model.strategyConfigInfo.parameters).length === 0
                                  }
                                >
                                  保存配置
                                </Button>
                              </Box>
                            }
                          />
                          <CardContent>
                            {Object.keys(model.strategyConfigInfo.parameters).length > 0 ? (
                              renderStrategyParameters(model.strategyConfigInfo.parameters)
                            ) : (
                              <Typography variant="caption" color="text.secondary">
                                暂无策略参数配置
                              </Typography>
                            )}
                          </CardContent>
                        </Card>
                      )}
                      <BacktestOverview
                        backtestData={model.backtestOverviewData}
                        loading={model.loadingBacktestData}
                      />
                      <CostAnalysis
                        backtestData={model.backtestSummaryData}
                        loading={model.loadingBacktestData}
                      />
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'charts' && (
                    <Box sx={{ mt: 2 }}>
                      <InteractiveChartsContainer
                        taskId={model.taskId}
                        stockCode={model.selectedStock || currentTask.stock_codes?.[0]}
                        stockCodes={currentTask.stock_codes || []}
                        backtestData={model.backtestSummaryData}
                      />
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'trades' && (
                    <Box sx={{ mt: 2 }}>
                      <TradeHistoryTable taskId={model.taskId} onTradeClick={() => undefined} />
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'signals' && (
                    <Box sx={{ mt: 2 }}>
                      <SignalHistoryTable taskId={model.taskId} onSignalClick={() => undefined} />
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'positions' && (
                    <Box sx={{ mt: 2 }}>
                      {renderPositionAnalysis(model, currentTask)}
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'monthly' && (
                    <Box sx={{ mt: 2 }}>
                      {model.backtestDetailedData?.monthly_returns ? (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <Typography variant="h6" component="h4" sx={{ fontWeight: 600 }}>
                            月度收益热力图
                          </Typography>
                          <Box
                            sx={{
                              display: 'grid',
                              gridTemplateColumns: 'repeat(12, 1fr)',
                              gap: 0.5,
                            }}
                          >
                            {model.backtestDetailedData.monthly_returns.map(monthData => (
                              <Box
                                key={`${monthData.year}-${monthData.month}`}
                                sx={{
                                  p: 1,
                                  textAlign: 'center',
                                  fontSize: '0.75rem',
                                  borderRadius: 1,
                                  bgcolor:
                                    monthData.monthly_return >= 0
                                      ? 'success.light'
                                      : 'error.light',
                                  color:
                                    monthData.monthly_return >= 0
                                      ? 'success.dark'
                                      : 'error.dark',
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
                      ) : (
                        <EmptyState icon={<Calendar size={48} color="#999" style={{ margin: '0 auto 16px' }} />} message="月度分析数据加载中..." />
                      )}
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'risk' && (
                    <Box sx={{ mt: 2 }}>
                      {model.adaptedRiskData ? (
                        <RiskAnalysis
                          taskId={model.taskId}
                          riskMetrics={model.adaptedRiskData.riskMetrics}
                          returnDistribution={model.adaptedRiskData.returnDistribution}
                          rollingMetrics={model.adaptedRiskData.rollingMetrics}
                        />
                      ) : (
                        <EmptyState icon={<Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />} message="风险分析数据加载中..." />
                      )}
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'performance' && (
                    <Box sx={{ mt: 2 }}>
                      {model.adaptedPerformanceData ? (
                        <PerformanceBreakdown
                          taskId={model.taskId}
                          monthlyPerformance={model.adaptedPerformanceData.monthlyPerformance}
                          yearlyPerformance={model.adaptedPerformanceData.yearlyPerformance}
                          seasonalAnalysis={model.adaptedPerformanceData.seasonalAnalysis}
                          benchmarkComparison={model.adaptedPerformanceData.benchmarkComparison}
                        />
                      ) : (
                        <EmptyState icon={<TrendingUp size={48} color="#999" style={{ margin: '0 auto 16px' }} />} message="绩效分解数据加载中..." />
                      )}
                    </Box>
                  )}

                  {model.selectedBacktestTab === 'perf_monitor' && (
                    <Box sx={{ mt: 2 }}>
                      {renderPerformanceMonitor(currentTask)}
                    </Box>
                  )}
                </Box>
              </Box>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card>
              <CardHeader title="任务信息" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                    {currentTask.task_type === 'hyperparameter_optimization' ? (
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          已完成轮次
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                          {currentTask.optimization_info?.completed_trials ?? 0} /{' '}
                          {currentTask.optimization_info?.n_trials ?? 0}
                        </Typography>
                      </Box>
                    ) : (
                      <>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            模型
                          </Typography>
                          <Chip label={currentTask.model_id} color="secondary" size="small" sx={{ mt: 0.5 }} />
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            股票数量
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                            {currentTask.stock_codes.length}
                          </Typography>
                        </Box>
                      </>
                    )}
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        创建时间
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                        {new Date(currentTask.created_at).toLocaleString()}
                      </Typography>
                    </Box>
                    {currentTask.completed_at && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          完成时间
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                          {new Date(currentTask.completed_at).toLocaleString()}
                        </Typography>
                      </Box>
                    )}
                  </Box>

                  <Divider />

                  <Box>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                      选择的股票
                    </Typography>
                    <Box
                      sx={{
                        height: 200,
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        p: 1.5,
                      }}
                    >
                      {currentTask.stock_codes && currentTask.stock_codes.length > 0 ? (
                        <>
                          <Box
                            sx={{
                              flex: 1,
                              overflowY: 'auto',
                              display: 'flex',
                              flexWrap: 'wrap',
                              gap: 1,
                              alignContent: 'flex-start',
                              pb: 1,
                            }}
                          >
                            {paginateStocks(currentTask.stock_codes, model.selectedStocksPage).map(code => (
                              <Chip key={code} label={code} size="small" />
                            ))}
                          </Box>

                          {renderStockPagination(currentTask.stock_codes, model)}

                          <Box
                            sx={{
                              pt: 1,
                              mt: 1,
                              borderTop: '1px solid',
                              borderColor: 'divider',
                              display: 'flex',
                              justifyContent: 'center',
                            }}
                          >
                            <Typography variant="body2" color="text.secondary">
                              已选择 <strong>{currentTask.stock_codes.length}</strong> 只股票
                            </Typography>
                          </Box>
                        </>
                      ) : (
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                          }}
                        >
                          <Typography variant="body2" color="text.secondary">
                            暂无选择的股票
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Box>
                </Box>
              </CardContent>
            </Card>

            {currentTask.status === 'completed' && model.predictions.length > 0 && (
              <Card>
                <CardHeader
                  title={
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        width: '100%',
                      }}
                    >
                      <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                        预测结果
                      </Typography>
                      <FormControl size="small" sx={{ minWidth: 192 }}>
                        <InputLabel>选择股票</InputLabel>
                        <Select
                          value={model.selectedStock || ''}
                          label="选择股票"
                          onChange={event => model.setSelectedStock(event.target.value)}
                        >
                          {model.predictions.map(prediction => (
                            <MenuItem key={prediction.stock_code} value={prediction.stock_code}>
                              {prediction.stock_code}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Box>
                  }
                />
                <CardContent>
                  <Box>
                    <Tabs
                      value={model.selectedPredictionTab}
                      onChange={(_event, newValue) => model.setSelectedPredictionTab(newValue)}
                      aria-label="预测结果展示"
                    >
                      <Tab label={<TabLabel icon={<LineChart size={16} />} text="价格走势" />} value="chart" />
                      <Tab label={<TabLabel icon={<BarChart3 size={16} />} text="预测分析" />} value="prediction" />
                      {currentTask.task_type === 'backtest' && (
                        <Tab label={<TabLabel icon={<Activity size={16} />} text="回测结果" />} value="backtest" />
                      )}
                      <Tab label="数据表格" value="table" />
                    </Tabs>

                    <Box sx={{ mt: 2 }}>
                      {model.selectedPredictionTab === 'chart' && model.selectedStock && (
                        <TradingViewChart
                          stockCode={model.selectedStock}
                          prediction={model.predictions.find(
                            prediction => prediction.stock_code === model.selectedStock
                          )}
                        />
                      )}

                      {model.selectedPredictionTab === 'prediction' && model.selectedStock && (
                        <PredictionChart
                          taskId={model.taskId}
                          stockCode={model.selectedStock}
                          prediction={model.predictions.find(
                            prediction => prediction.stock_code === model.selectedStock
                          )}
                        />
                      )}

                      {model.selectedPredictionTab === 'backtest' && currentTask.task_type === 'backtest' && (
                        <BacktestChart
                          stockCode={model.selectedStock || currentTask.stock_codes?.[0] || ''}
                          backtestData={model.backtestSummaryData}
                        />
                      )}

                      {model.selectedPredictionTab === 'table' && (
                        <TableContainer component={Paper}>
                          <Table aria-label="预测结果表格">
                            <TableHead>
                              <TableRow>
                                <TableCell>股票代码</TableCell>
                                <TableCell>预测方向</TableCell>
                                <TableCell>预测收益率</TableCell>
                                <TableCell>置信度</TableCell>
                                <TableCell>置信区间</TableCell>
                                <TableCell>VaR</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {model.predictions.map(prediction => (
                                <TableRow key={prediction.stock_code}>
                                  <TableCell>
                                    <Chip label={prediction.stock_code} size="small" />
                                  </TableCell>
                                  <TableCell>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      {getPredictionIcon(prediction.predicted_direction)}
                                      <Typography variant="body2">
                                        {getPredictionText(prediction.predicted_direction)}
                                      </Typography>
                                    </Box>
                                  </TableCell>
                                  <TableCell>
                                    <Typography
                                      variant="body2"
                                      sx={{
                                        color:
                                          prediction.predicted_return > 0
                                            ? 'success.main'
                                            : prediction.predicted_return < 0
                                              ? 'error.main'
                                              : 'text.secondary',
                                      }}
                                    >
                                      {(prediction.predicted_return * 100).toFixed(2)}%
                                    </Typography>
                                  </TableCell>
                                  <TableCell>
                                    <Box sx={{ width: 80 }}>
                                      <LinearProgress
                                        variant="determinate"
                                        value={prediction.confidence_score * 100}
                                        sx={{ height: 8, borderRadius: 4 }}
                                      />
                                    </Box>
                                  </TableCell>
                                  <TableCell>
                                    <Typography variant="caption" color="text.secondary">
                                      [
                                      {(prediction.confidence_interval.lower * 100).toFixed(2)}%,{' '}
                                      {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
                                    </Typography>
                                  </TableCell>
                                  <TableCell>
                                    <Typography variant="body2" color="error.main">
                                      {(prediction.risk_assessment.value_at_risk * 100).toFixed(2)}%
                                    </Typography>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      )}
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {currentTask.task_type === 'backtest' ? (
          <BacktestTaskStatus
            task={currentTask}
            onRetry={() => void model.handleRetry()}
            onStop={() => undefined}
            loading={model.refreshing}
          />
        ) : (
          <>
            {currentTask.results && (
              <Card>
                <CardHeader title="统计信息" />
                <CardContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <StatItem value={currentTask.results.total_stocks} label="总股票数" color="primary.main" />
                    <StatItem value={currentTask.results.successful_predictions} label="成功预测" color="success.main" />
                    <StatItem
                      value={`${((currentTask.results.average_confidence || 0) * 100).toFixed(1)}%`}
                      label="平均置信度"
                      color="secondary.main"
                    />
                  </Box>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader title="快速操作" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshCw size={16} />}
                    onClick={() => void model.handleRefresh()}
                    disabled={model.refreshing}
                    fullWidth
                  >
                    刷新状态
                  </Button>

                  {currentTask.status === 'failed' && (
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<Play size={16} />}
                      onClick={() => void model.handleRetry()}
                      fullWidth
                    >
                      重新运行
                    </Button>
                  )}

                  {currentTask.status === 'completed' && (
                    <Button
                      variant="outlined"
                      color="secondary"
                      startIcon={<Download size={16} />}
                      onClick={() => void model.handleExport()}
                      fullWidth
                    >
                      导出结果
                    </Button>
                  )}

                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<Trash2 size={16} />}
                    onClick={model.openDeleteDialog}
                    fullWidth
                  >
                    删除任务
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </>
        )}
      </Box>
    </Box>
  );
}

function TabLabel({ icon, text }: { icon: React.ReactNode; text: string }): React.ReactNode {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
      {icon}
      <span>{text}</span>
    </Box>
  );
}

function EmptyState({ icon, message }: { icon: React.ReactNode; message: string }): React.ReactNode {
  return (
    <Box sx={{ textAlign: 'center', py: 4 }}>
      {icon}
      <Typography variant="body2" color="text.secondary">
        {message}
      </Typography>
    </Box>
  );
}

function StatItem({ value, label, color }: { value: React.ReactNode; label: string; color: string }): React.ReactNode {
  return (
    <Box sx={{ textAlign: 'center' }}>
      <Typography variant="h4" sx={{ fontWeight: 600, color }}>
        {value}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}

function paginateStocks(stockCodes: string[], page: number): string[] {
  const stocksPerPage = 12;
  const startIndex = (page - 1) * stocksPerPage;
  return stockCodes.slice(startIndex, startIndex + stocksPerPage);
}

function renderStockPagination(
  stockCodes: string[],
  model: TaskDetailPageModel
): React.ReactNode {
  const stocksPerPage = 12;
  const totalPages = Math.ceil(stockCodes.length / stocksPerPage);

  if (totalPages <= 1) {
    return null;
  }

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 1,
        pt: 1,
        borderTop: '1px solid',
        borderColor: 'divider',
      }}
    >
      <IconButton
        size="small"
        disabled={model.selectedStocksPage === 1}
        onClick={() => model.setSelectedStocksPage(previous => Math.max(1, previous - 1))}
      >
        <ChevronLeft size={16} />
      </IconButton>

      <Typography variant="caption" color="text.secondary">
        第 {model.selectedStocksPage} / {totalPages} 页
      </Typography>

      <IconButton
        size="small"
        disabled={model.selectedStocksPage >= totalPages}
        onClick={() => model.setSelectedStocksPage(previous => Math.min(totalPages, previous + 1))}
      >
        <ChevronRight size={16} />
      </IconButton>
    </Box>
  );
}

function renderPositionAnalysis(
  model: TaskDetailPageModel,
  currentTask: NonNullable<TaskDetailPageModel['currentTask']>
): React.ReactNode {
  if (model.loadingBacktestData) {
    return renderNoDataState('持仓分析数据加载中...');
  }

  if (model.backtestDetailedData === null) {
    return renderNoDataState('暂无持仓分析数据');
  }

  const positionAnalysis = model.backtestDetailedData?.position_analysis;
  if (positionAnalysis === null || positionAnalysis === undefined) {
    return renderNoDataState('暂无持仓分析数据');
  }

  if (typeof positionAnalysis === 'object' && !Array.isArray(positionAnalysis)) {
    if (positionAnalysis.stock_performance !== undefined) {
      const stockPerformance = positionAnalysis.stock_performance;
      if (Array.isArray(stockPerformance) && stockPerformance.length > 0) {
        return (
          <PositionAnalysis
            positionAnalysis={positionAnalysis}
            stockCodes={currentTask.stock_codes || []}
            taskId={model.taskId}
          />
        );
      }
      return renderNoDataState('暂无持仓分析数据');
    }

    if (Object.keys(positionAnalysis).length === 0) {
      return renderNoDataState('暂无持仓分析数据');
    }

    return (
      <PositionAnalysis
        positionAnalysis={positionAnalysis}
        stockCodes={currentTask.stock_codes || []}
        taskId={model.taskId}
      />
    );
  }

  if (Array.isArray(positionAnalysis)) {
    if (positionAnalysis.length > 0) {
      return <PositionAnalysis positionAnalysis={positionAnalysis} stockCodes={currentTask.stock_codes || []} />;
    }
    return renderNoDataState('暂无持仓分析数据');
  }

  return renderNoDataState('暂无持仓分析数据');
}

function renderNoDataState(message: string): React.ReactNode {
  return (
    <div className="text-center text-default-500 py-8">
      <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
      <p>{message}</p>
    </div>
  );
}

function renderPerformanceMonitor(
  currentTask: NonNullable<TaskDetailPageModel['currentTask']>
): React.ReactNode {
  const backtestData =
    currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results;
  const performanceAnalysis = backtestData?.performance_analysis as any;

  if (!backtestData) {
    return (
      <EmptyState
        icon={<Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />}
        message="暂无回测结果数据，无法展示性能分析。"
      />
    );
  }

  if (!performanceAnalysis) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />
        <Typography variant="body2" color="text.secondary">
          当前回测未启用性能监控，或后端尚未写入性能报告。
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          请确认后端创建回测执行器时已开启 enable_performance_profiling，并返回
          performance_analysis 字段。
        </Typography>
      </Box>
    );
  }

  const summary = performanceAnalysis.summary || {};
  const stages = performanceAnalysis.stages || {};
  const functionCalls = performanceAnalysis.function_calls || {};
  const parallel = performanceAnalysis.parallel_efficiency || {};

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Card>
        <CardHeader title="整体性能概要" />
        <CardContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
              gap: 2,
            }}
          >
            <Box>
              <Typography variant="caption" color="text.secondary">
                总执行时间
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {(summary.total_time || 0).toFixed(2)} 秒
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                总信号数 / 交易数
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {(summary.total_signals || 0).toLocaleString()} /{' '}
                {(summary.total_trades || 0).toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                处理速度
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {(summary.days_per_second || 0).toFixed(2)} 天/秒
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="阶段耗时与资源占用" />
        <CardContent>
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>阶段</TableCell>
                  <TableCell align="right">耗时 (秒)</TableCell>
                  <TableCell align="right">占比</TableCell>
                  <TableCell align="right">峰值内存 (MB)</TableCell>
                  <TableCell align="right">平均 CPU (%)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(stages).map(([name, data]: [string, any]) => (
                  <TableRow key={name}>
                    <TableCell>
                      <Typography variant="body2">
                        {name === 'total_backtest' ? '整体回测' : name}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">{(data.duration || 0).toFixed(2)}</TableCell>
                    <TableCell align="right">{(data.percentage || 0).toFixed(1)}%</TableCell>
                    <TableCell align="right">
                      {(data.memory_peak_mb ?? data.memory_after_mb ?? 0).toFixed(2)}
                    </TableCell>
                    <TableCell align="right">{(data.cpu_avg_percent || 0).toFixed(1)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {Object.keys(functionCalls).length > 0 && (
        <Card>
          <CardHeader title="最耗时的函数 (Top 10)" />
          <CardContent>
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>函数名</TableCell>
                    <TableCell align="right">调用次数</TableCell>
                    <TableCell align="right">总耗时 (秒)</TableCell>
                    <TableCell align="right">平均耗时 (毫秒)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(functionCalls)
                    .slice(0, 10)
                    .map(([name, data]: [string, any]) => (
                      <TableRow key={name}>
                        <TableCell>{name}</TableCell>
                        <TableCell align="right">{data.call_count || 0}</TableCell>
                        <TableCell align="right">{(data.total_time || 0).toFixed(4)}</TableCell>
                        <TableCell align="right">{((data.avg_time || 0) * 1000).toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {Object.keys(parallel).length > 0 && (
        <Card>
          <CardHeader title="并行化效率" />
          <CardContent>
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>操作</TableCell>
                    <TableCell align="right">顺序时间 (秒)</TableCell>
                    <TableCell align="right">并行时间 (秒)</TableCell>
                    <TableCell align="right">加速比</TableCell>
                    <TableCell align="right">效率 (%)</TableCell>
                    <TableCell align="right">Worker 数</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(parallel).map(([name, data]: [string, any]) => (
                    <TableRow key={name}>
                      <TableCell>{name}</TableCell>
                      <TableCell align="right">{(data.sequential_time || 0).toFixed(4)}</TableCell>
                      <TableCell align="right">{(data.parallel_time || 0).toFixed(4)}</TableCell>
                      <TableCell align="right">{(data.speedup || 0).toFixed(2)}x</TableCell>
                      <TableCell align="right">{(data.efficiency_percent || 0).toFixed(1)}</TableCell>
                      <TableCell align="right">{data.worker_count || 0}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
