/**
 * 任务详情页面
 * 
 * 显示任务的详细信息，包括：
 * - 任务基本信息
 * - 实时进度更新
 * - 预测结果展示（TradingView图表 + ECharts）
 * - 操作控制
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Button,
  Progress,
  Chip,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Divider,
  Spacer,
  Select,
  SelectItem,
  Tabs,
  Tab,
} from '@heroui/react';
import {
  ArrowLeft,
  RefreshCw,
  Play,
  Download,
  Trash2,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  BarChart3,
  LineChart,
  Activity,
  PieChart,
  Calendar,
  FileText,
} from 'lucide-react';
import { useRouter, useParams } from 'next/navigation';
import { useTaskStore, Task } from '../../../stores/useTaskStore';
import { TaskService, PredictionResult } from '../../../services/taskService';
import { BacktestService } from '../../../services/backtestService';
import { BacktestDataAdapter } from '../../../services/backtestDataAdapter';
import { wsService } from '../../../services/websocket';
import { LoadingSpinner } from '../../../components/common/LoadingSpinner';
import BacktestOverview from '../../../components/backtest/BacktestOverview';
import BacktestTaskStatus from '../../../components/backtest/BacktestTaskStatus';
import BacktestProgressMonitor from '../../../components/backtest/BacktestProgressMonitor';
import { TradeHistoryTable } from '../../../components/backtest/TradeHistoryTable';
import { PositionAnalysis } from '../../../components/backtest/PositionAnalysis';
import { RiskAnalysis } from '../../../components/backtest/RiskAnalysis';
import { PerformanceBreakdown } from '../../../components/backtest/PerformanceBreakdown';
import dynamic from 'next/dynamic';

// 动态导入图表组件
const TradingViewChart = dynamic(() => import('../../../components/charts/TradingViewChart'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载图表中...</div>
});

const PredictionChart = dynamic(() => import('../../../components/charts/PredictionChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载预测图表中...</div>
});

const BacktestChart = dynamic(() => import('../../../components/charts/BacktestChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载回测图表中...</div>
});

const InteractiveChartsContainer = dynamic(() => import('../../../components/charts/InteractiveChartsContainer'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载交互式图表中...</div>
});

export default function TaskDetailPage() {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;

  const { currentTask, setCurrentTask, updateTask } = useTaskStore();
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [backtestDetailedData, setBacktestDetailedData] = useState<any>(null);
  const [adaptedRiskData, setAdaptedRiskData] = useState<any>(null);
  const [adaptedPerformanceData, setAdaptedPerformanceData] = useState<any>(null);
  const [loadingBacktestData, setLoadingBacktestData] = useState(false);
  const { isOpen: isDeleteOpen, onOpen: onDeleteOpen, onClose: onDeleteClose } = useDisclosure();
  const [deleteForce, setDeleteForce] = useState(false);

  // 加载回测详细数据
  const loadBacktestDetailedData = async () => {
    if (!currentTask || currentTask.task_type !== 'backtest' || currentTask.status !== 'completed') {
      return;
    }

    setLoadingBacktestData(true);
    try {
      console.log('[TaskDetail] 开始加载回测详细数据...');
      const detailedResult = await BacktestService.getDetailedResult(taskId);
      console.log('[TaskDetail] 后端返回的详细数据:', detailedResult);
      console.log('[TaskDetail] position_analysis 数据:', detailedResult?.position_analysis);
      console.log('[TaskDetail] position_analysis 类型:', typeof detailedResult?.position_analysis);
      console.log('[TaskDetail] position_analysis 是否为数组:', Array.isArray(detailedResult?.position_analysis));
      if (detailedResult?.position_analysis && typeof detailedResult.position_analysis === 'object' && !Array.isArray(detailedResult.position_analysis)) {
        console.log('[TaskDetail] position_analysis.stock_performance:', detailedResult.position_analysis.stock_performance);
        console.log('[TaskDetail] stock_performance 长度:', detailedResult.position_analysis.stock_performance?.length);
      }
      
      setBacktestDetailedData(detailedResult);
      
      // 使用数据适配器转换数据格式
      console.log('[TaskDetail] 开始适配风险分析数据...');
      const riskMetrics = BacktestDataAdapter.adaptRiskMetrics(detailedResult);
      const returnDistribution = BacktestDataAdapter.generateReturnDistribution(detailedResult);
      const rollingMetrics = BacktestDataAdapter.generateRollingMetrics(detailedResult);
      
      setAdaptedRiskData({
        riskMetrics,
        returnDistribution,
        rollingMetrics
      });
      console.log('[TaskDetail] 风险分析数据适配完成:', { riskMetrics, returnDistribution, rollingMetrics });
      
      console.log('[TaskDetail] 开始适配绩效分解数据...');
      const monthlyPerformance = BacktestDataAdapter.adaptMonthlyPerformance(detailedResult);
      const yearlyPerformance = BacktestDataAdapter.generateYearlyPerformance(detailedResult);
      const seasonalAnalysis = BacktestDataAdapter.generateSeasonalAnalysis(detailedResult);
      const benchmarkComparison = BacktestDataAdapter.generateBenchmarkComparison(detailedResult);
      
      setAdaptedPerformanceData({
        monthlyPerformance,
        yearlyPerformance,
        seasonalAnalysis,
        benchmarkComparison
      });
      console.log('[TaskDetail] 绩效分解数据适配完成:', { monthlyPerformance, yearlyPerformance, seasonalAnalysis, benchmarkComparison });
      
    } catch (error) {
      console.error('[TaskDetail] 加载回测详细数据失败:', error);
      // 如果详细数据不存在，使用基础回测数据
      setBacktestDetailedData(null);
      setAdaptedRiskData(null);
      setAdaptedPerformanceData(null);
    } finally {
      setLoadingBacktestData(false);
    }
  };

  // 加载任务详情
  const loadTaskDetail = async () => {
    try {
      const task = await TaskService.getTaskDetail(taskId);
      console.log('加载的任务详情:', {
        task_id: task.task_id,
        task_type: task.task_type,
        status: task.status,
        results: task.results,
        backtest_results: task.backtest_results,
        result: task.result
      });
      setCurrentTask(task);
      
      // 如果任务已完成，加载预测结果
      if (task.status === 'completed' && task.results) {
        // 如果是预测任务，加载预测结果
        if (task.task_type === 'prediction') {
          const results = await TaskService.getTaskResults(taskId);
          setPredictions(results);
          // 默认选择第一个股票
          if (results.length > 0) {
            setSelectedStock(results[0].stock_code);
          }
        } else if (task.task_type === 'backtest') {
          // 如果是回测任务，确保回测结果已加载
          console.log('回测任务详情:', {
            'results.backtest_results': task.results?.backtest_results,
            'backtest_results': task.backtest_results,
            'result': task.result
          });
          // 如果没有股票代码，使用任务配置中的股票代码
          if (!selectedStock && task.stock_codes && task.stock_codes.length > 0) {
            setSelectedStock(task.stock_codes[0]);
          }
          // 加载回测详细数据
          await loadBacktestDetailedData();
        }
      }
    } catch (error) {
      console.error('加载任务详情失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 初始化加载
  useEffect(() => {
    if (taskId) {
      loadTaskDetail();
      // 订阅任务更新
      wsService.subscribeToTask(taskId);
    }

    return () => {
      if (taskId) {
        wsService.unsubscribeFromTask(taskId);
      }
    };
  }, [taskId]);

  // WebSocket实时更新
  useEffect(() => {
    const handleTaskProgress = (data: { task_id: string; progress: number; status: string }) => {
      if (data.task_id === taskId) {
        updateTask(data.task_id, {
          progress: data.progress,
          status: data.status as Task['status'],
        });
        
        if (currentTask) {
          setCurrentTask({
            ...currentTask,
            progress: data.progress,
            status: data.status as Task['status'],
          });
        }
      }
    };

    const handleTaskCompleted = async (data: { task_id: string; results: any }) => {
      if (data.task_id === taskId) {
        // 重新加载任务详情以获取完整数据
        try {
          const task = await TaskService.getTaskDetail(taskId);
          const updatedTask = {
            ...task,
            status: 'completed' as const,
            progress: 100,
            completed_at: new Date().toISOString(),
          };
          
          setCurrentTask(updatedTask);
          updateTask(data.task_id, updatedTask);
          
          // 如果是预测任务，加载预测结果
          if (task.task_type === 'prediction') {
            const results = await TaskService.getTaskResults(taskId);
            setPredictions(results);
            if (results.length > 0) {
              setSelectedStock(results[0].stock_code);
            }
          } else if (task.task_type === 'backtest') {
            // 回测任务，确保回测结果已加载
            console.log('回测任务完成，回测结果:', task.results?.backtest_results);
            // 重新加载回测详细数据
            await loadBacktestDetailedData();
          }
        } catch (error) {
          console.error('加载任务详情失败:', error);
        }
        
        console.log('任务执行完成');
      }
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      if (data.task_id === taskId) {
        const updatedTask = {
          ...currentTask!,
          status: 'failed' as const,
          error_message: data.error,
        };
        
        setCurrentTask(updatedTask);
        updateTask(data.task_id, updatedTask);
        console.error('任务执行失败');
      }
    };

    wsService.on('task:progress', handleTaskProgress);
    wsService.on('task:completed', handleTaskCompleted);
    wsService.on('task:failed', handleTaskFailed);

    return () => {
      wsService.off('task:progress', handleTaskProgress);
      wsService.off('task:completed', handleTaskCompleted);
      wsService.off('task:failed', handleTaskFailed);
    };
  }, [taskId, currentTask, updateTask, setCurrentTask]);

  // 刷新任务
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadTaskDetail();
    setRefreshing(false);
  };

  // 重新运行任务
  const handleRetry = async () => {
    try {
      await TaskService.retryTask(taskId);
      console.log('任务已重新启动');
      await loadTaskDetail();
    } catch (error) {
      console.error('重新运行失败');
    }
  };

  // 删除任务
  const handleDelete = async () => {
    try {
      await TaskService.deleteTask(taskId, deleteForce);
      console.log(`任务删除成功${deleteForce ? '（强制删除）' : ''}`);
      router.push('/tasks');
    } catch (error: any) {
      console.error('删除任务失败:', error);
      // 如果任务正在运行或可能是僵尸任务，显示强制删除选项
      if (error.message?.includes('正在运行中') || 
          error.message?.includes('运行中') ||
          currentTask?.status === 'running') {
        setDeleteForce(true);
        onDeleteOpen();
      }
    }
  };

  // 导出结果
  const handleExport = async () => {
    try {
      const blob = await TaskService.exportTaskResults(taskId, 'csv');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `task_${taskId}_results.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      console.log('结果导出成功');
    } catch (error) {
      console.error('导出失败');
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  // 获取状态标签
  const getStatusChip = (status: Task['status']) => {
    const statusConfig = {
      created: { color: 'default' as const, text: '已创建' },
      running: { color: 'primary' as const, text: '运行中' },
      completed: { color: 'success' as const, text: '已完成' },
      failed: { color: 'danger' as const, text: '失败' },
    };
    
    const config = statusConfig[status] || statusConfig.created;
    return <Chip color={config.color} variant="flat">{config.text}</Chip>;
  };

  // 获取预测方向图标
  const getPredictionIcon = (direction: number) => {
    if (direction > 0) return <TrendingUp className="w-4 h-4 text-success" />;
    if (direction < 0) return <TrendingDown className="w-4 h-4 text-danger" />;
    return <Minus className="w-4 h-4 text-default-500" />;
  };

  // 获取预测方向文本
  const getPredictionText = (direction: number) => {
    if (direction > 0) return '上涨';
    if (direction < 0) return '下跌';
    return '持平';
  };

  if (loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!currentTask) {
    return (
      <div className="flex flex-col items-center justify-center min-h-96 space-y-4">
        <p className="text-default-500">任务不存在或已被删除</p>
        <Button color="primary" onPress={handleBack}>
          返回任务列表
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button
            isIconOnly
            variant="light"
            onPress={handleBack}
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold">{currentTask.task_name}</h1>
            <p className="text-default-500 text-sm">任务ID: {currentTask.task_id}</p>
          </div>
          {getStatusChip(currentTask.status)}
        </div>
        
        <div className="flex space-x-2">
          <Button
            variant="light"
            startContent={<RefreshCw className="w-4 h-4" />}
            onPress={handleRefresh}
            isLoading={refreshing}
          >
            刷新
          </Button>
          
          {currentTask.status === 'failed' && (
            <Button
              color="primary"
              startContent={<Play className="w-4 h-4" />}
              onPress={handleRetry}
            >
              重新运行
            </Button>
          )}
          
          {currentTask.status === 'completed' && (
            <Button
              color="secondary"
              startContent={<Download className="w-4 h-4" />}
              onPress={handleExport}
            >
              导出结果
            </Button>
          )}
          
          <Button
            color="danger"
            variant="light"
            startContent={<Trash2 className="w-4 h-4" />}
            onPress={onDeleteOpen}
          >
            删除
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 主要内容区域 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 任务进度 */}
          {currentTask.task_type === 'backtest' && (currentTask.status === 'running' || currentTask.status === 'created') ? (
            <BacktestProgressMonitor
              taskId={taskId}
              onComplete={(results) => {
                console.log('回测完成:', results);
                // 刷新任务数据
                loadTaskDetail();
              }}
              onError={(error) => {
                console.error('回测错误:', error);
                // 刷新任务数据以获取最新状态
                loadTaskDetail();
              }}
              onCancel={() => {
                console.log('回测已取消');
                // 刷新任务数据
                loadTaskDetail();
              }}
            />
          ) : (
            /* 通用任务进度显示 */
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">任务进度</h3>
              </CardHeader>
              <CardBody>
                <Progress
                  value={currentTask.progress}
                  color={currentTask.status === 'failed' ? 'danger' : 'primary'}
                  className="mb-4"
                />
                {currentTask.status === 'running' && (
                  <p className="text-default-500 text-sm">
                    任务正在执行中，请耐心等待...
                  </p>
                )}
                {currentTask.status === 'failed' && currentTask.error_message && (
                  <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
                    <div className="flex items-start space-x-2">
                      <AlertTriangle className="w-5 h-5 text-danger mt-0.5" />
                      <div>
                        <p className="font-medium text-danger">任务执行失败</p>
                        <p className="text-sm text-danger-600 mt-1">{currentTask.error_message}</p>
                      </div>
                    </div>
                  </div>
                )}
              </CardBody>
            </Card>
          )}

          {/* 根据任务类型显示不同内容 */}
          {currentTask.task_type === 'backtest' ? (
            /* 回测任务专用标签页 */
            <Card>
              <CardBody>
                <Tabs aria-label="回测结果展示" className="w-full">
                  <Tab key="overview" title={
                    <div className="flex items-center space-x-2">
                      <BarChart3 className="w-4 h-4" />
                      <span>概览</span>
                    </div>
                  }>
                    <div className="mt-4">
                      <BacktestOverview 
                        backtestData={currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results}
                        loading={loadingBacktestData}
                      />
                    </div>
                  </Tab>
                  
                  <Tab key="charts" title={
                    <div className="flex items-center space-x-2">
                      <LineChart className="w-4 h-4" />
                      <span>交互式图表</span>
                    </div>
                  }>
                    <div className="mt-4">
                      <InteractiveChartsContainer 
                        taskId={taskId}
                        stockCode={selectedStock || currentTask?.stock_codes?.[0]}
                        stockCodes={currentTask?.stock_codes || []}
                        backtestData={(() => {
                          // 尝试从多个位置获取回测数据
                          const data = currentTask?.results?.backtest_results || 
                                     currentTask?.backtest_results ||
                                     (currentTask?.task_type === 'backtest' ? currentTask?.result : null);
                          return data;
                        })()}
                      />
                    </div>
                  </Tab>
                  
                  <Tab key="trades" title={
                    <div className="flex items-center space-x-2">
                      <FileText className="w-4 h-4" />
                      <span>交易记录</span>
                    </div>
                  }>
                    <div className="mt-4">
                      <TradeHistoryTable 
                        taskId={taskId}
                        onTradeClick={(trade) => {
                          console.log('查看交易详情:', trade);
                        }}
                      />
                    </div>
                  </Tab>
                  
                  <Tab key="positions" title={
                    <div className="flex items-center space-x-2">
                      <PieChart className="w-4 h-4" />
                      <span>持仓分析</span>
                    </div>
                  }>
                    <div className="mt-4">
                      {(() => {
                        // 如果数据正在加载，显示加载中
                        if (loadingBacktestData || backtestDetailedData === null) {
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>持仓分析数据加载中...</p>
                            </div>
                          );
                        }
                        
                        const posAnalysis = backtestDetailedData?.position_analysis;
                        
                        // 如果position_analysis为null或undefined，显示无数据
                        if (posAnalysis === null || posAnalysis === undefined) {
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>暂无持仓分析数据</p>
                            </div>
                          );
                        }
                        
                        // 检查新格式（对象，包含stock_performance）
                        if (typeof posAnalysis === 'object' && !Array.isArray(posAnalysis)) {
                          // 检查是否有stock_performance字段
                          if (posAnalysis.stock_performance !== undefined) {
                            const stockPerf = posAnalysis.stock_performance;
                            if (Array.isArray(stockPerf) && stockPerf.length > 0) {
                              return (
                                <PositionAnalysis 
                                  positionAnalysis={posAnalysis}
                                  stockCodes={currentTask.stock_codes || []}
                                />
                              );
                            }
                            // stock_performance存在但为空数组
                            return (
                              <div className="text-center text-default-500 py-8">
                                <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                <p>暂无持仓分析数据</p>
                              </div>
                            );
                          }
                          // 对象格式但没有stock_performance字段，可能是空对象或其他格式
                          // 检查是否是完全空对象
                          if (Object.keys(posAnalysis).length === 0) {
                            return (
                              <div className="text-center text-default-500 py-8">
                                <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                <p>暂无持仓分析数据</p>
                              </div>
                            );
                          }
                          // 有其他字段但没有stock_performance，尝试直接使用（兼容其他可能的格式）
                          return (
                            <PositionAnalysis 
                              positionAnalysis={posAnalysis}
                              stockCodes={currentTask.stock_codes || []}
                            />
                          );
                        }
                        
                        // 检查旧格式（数组）
                        if (Array.isArray(posAnalysis)) {
                          if (posAnalysis.length > 0) {
                            return (
                              <PositionAnalysis 
                                positionAnalysis={posAnalysis}
                                stockCodes={currentTask.stock_codes || []}
                              />
                            );
                          }
                          // 数组为空
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>暂无持仓分析数据</p>
                            </div>
                          );
                        }
                        
                        // 其他情况，显示无数据
                        return (
                          <div className="text-center text-default-500 py-8">
                            <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                            <p>暂无持仓分析数据</p>
                          </div>
                        );
                      })()}
                    </div>
                  </Tab>
                  
                  <Tab key="monthly" title={
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4" />
                      <span>月度分析</span>
                    </div>
                  }>
                    <div className="mt-4">
                      {backtestDetailedData?.monthly_returns ? (
                        <div className="space-y-4">
                          <h4 className="text-lg font-semibold">月度收益热力图</h4>
                          <div className="grid grid-cols-12 gap-1">
                            {backtestDetailedData.monthly_returns.map((monthData: any) => (
                              <div
                                key={`${monthData.year}-${monthData.month}`}
                                className={`
                                  p-2 text-center text-xs rounded
                                  ${monthData.monthly_return >= 0 
                                    ? 'bg-success-100 text-success-800' 
                                    : 'bg-danger-100 text-danger-800'
                                  }
                                `}
                                title={`${monthData.year}年${monthData.month}月: ${(monthData.monthly_return * 100).toFixed(2)}%`}
                              >
                                {monthData.month}月
                                <br />
                                {(monthData.monthly_return * 100).toFixed(1)}%
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="text-center text-default-500 py-8">
                          <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
                          <p>月度分析数据加载中...</p>
                        </div>
                      )}
                    </div>
                  </Tab>
                  
                  <Tab key="risk" title={
                    <div className="flex items-center space-x-2">
                      <Activity className="w-4 h-4" />
                      <span>风险分析</span>
                    </div>
                  }>
                    <div className="mt-4">
                      {adaptedRiskData ? (
                        <RiskAnalysis 
                          taskId={taskId}
                          riskMetrics={adaptedRiskData.riskMetrics}
                          returnDistribution={adaptedRiskData.returnDistribution}
                          rollingMetrics={adaptedRiskData.rollingMetrics}
                        />
                      ) : (
                        <div className="text-center text-default-500 py-8">
                          <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                          <p>风险分析数据加载中...</p>
                        </div>
                      )}
                    </div>
                  </Tab>
                  
                  <Tab key="performance" title={
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="w-4 h-4" />
                      <span>绩效分解</span>
                    </div>
                  }>
                    <div className="mt-4">
                      {adaptedPerformanceData ? (
                        <PerformanceBreakdown 
                          taskId={taskId}
                          monthlyPerformance={adaptedPerformanceData.monthlyPerformance}
                          yearlyPerformance={adaptedPerformanceData.yearlyPerformance}
                          seasonalAnalysis={adaptedPerformanceData.seasonalAnalysis}
                          benchmarkComparison={adaptedPerformanceData.benchmarkComparison}
                        />
                      ) : (
                        <div className="text-center text-default-500 py-8">
                          <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                          <p>绩效分解数据加载中...</p>
                        </div>
                      )}
                    </div>
                  </Tab>
                </Tabs>
              </CardBody>
            </Card>
          ) : (
            /* 预测任务的原有内容 */
            <>
              {/* 任务信息 */}
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">任务信息</h3>
                </CardHeader>
                <CardBody className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-default-500">模型</p>
                      <Chip variant="flat" color="secondary">{currentTask.model_id}</Chip>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">股票数量</p>
                      <p className="font-medium">{currentTask.stock_codes.length}</p>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">创建时间</p>
                      <p className="font-medium">{new Date(currentTask.created_at).toLocaleString()}</p>
                    </div>
                    {currentTask.completed_at && (
                      <div>
                        <p className="text-sm text-default-500">完成时间</p>
                        <p className="font-medium">{new Date(currentTask.completed_at).toLocaleString()}</p>
                      </div>
                    )}
                  </div>

                  <Divider />

                  <div>
                    <p className="text-sm text-default-500 mb-2">选择的股票</p>
                    <div className="flex flex-wrap gap-2">
                      {currentTask.stock_codes.map(code => (
                        <Chip key={code} variant="flat" size="sm">{code}</Chip>
                      ))}
                    </div>
                  </div>
                </CardBody>
              </Card>

              {/* 预测结果和图表 */}
              {currentTask.status === 'completed' && predictions.length > 0 && (
                <Card>
                  <CardHeader className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold">预测结果</h3>
                    <Select
                      placeholder="选择股票"
                      selectedKeys={selectedStock ? [selectedStock] : []}
                      onSelectionChange={(keys) => {
                        const selected = Array.from(keys)[0] as string;
                        setSelectedStock(selected);
                      }}
                      className="w-48"
                    >
                      {predictions.map((prediction) => (
                        <SelectItem key={prediction.stock_code}>
                          {prediction.stock_code}
                        </SelectItem>
                      ))}
                    </Select>
                  </CardHeader>
                  <CardBody>
                    <Tabs aria-label="预测结果展示">
                      <Tab key="chart" title={
                        <div className="flex items-center space-x-2">
                          <LineChart className="w-4 h-4" />
                          <span>价格走势</span>
                        </div>
                      }>
                        {selectedStock && (
                          <TradingViewChart 
                            stockCode={selectedStock}
                            prediction={predictions.find(p => p.stock_code === selectedStock)}
                          />
                        )}
                      </Tab>
                      
                      <Tab key="prediction" title={
                        <div className="flex items-center space-x-2">
                          <BarChart3 className="w-4 h-4" />
                          <span>预测分析</span>
                        </div>
                      }>
                        {selectedStock && (
                          <PredictionChart 
                            taskId={taskId}
                            stockCode={selectedStock}
                            prediction={predictions.find(p => p.stock_code === selectedStock)}
                          />
                        )}
                      </Tab>
                      
                      {currentTask.task_type === 'backtest' && (
                        <Tab key="backtest" title={
                          <div className="flex items-center space-x-2">
                            <Activity className="w-4 h-4" />
                            <span>回测结果</span>
                          </div>
                        }>
                          <BacktestChart 
                            stockCode={selectedStock || (currentTask?.stock_codes?.[0] || '')}
                            backtestData={(() => {
                              // 尝试从多个位置获取回测数据
                              const data = currentTask?.results?.backtest_results || 
                                         currentTask?.backtest_results ||
                                         (currentTask?.task_type === 'backtest' ? currentTask?.result : null);
                              console.log('回测数据来源检查:', {
                                'results.backtest_results': currentTask?.results?.backtest_results,
                                'backtest_results': currentTask?.backtest_results,
                                'result': currentTask?.result,
                                '最终数据': data
                              });
                              return data;
                            })()}
                          />
                        </Tab>
                      )}
                      
                      <Tab key="table" title="数据表格">
                        <Table aria-label="预测结果表格">
                          <TableHeader>
                            <TableColumn>股票代码</TableColumn>
                            <TableColumn>预测方向</TableColumn>
                            <TableColumn>预测收益率</TableColumn>
                            <TableColumn>置信度</TableColumn>
                            <TableColumn>置信区间</TableColumn>
                            <TableColumn>VaR</TableColumn>
                          </TableHeader>
                          <TableBody>
                            {predictions.map((prediction) => (
                              <TableRow key={prediction.stock_code}>
                                <TableCell>
                                  <Chip variant="flat" size="sm">{prediction.stock_code}</Chip>
                                </TableCell>
                                <TableCell>
                                  <div className="flex items-center space-x-2">
                                    {getPredictionIcon(prediction.predicted_direction)}
                                    <span>{getPredictionText(prediction.predicted_direction)}</span>
                                  </div>
                                </TableCell>
                                <TableCell>
                                  <span className={
                                    prediction.predicted_return > 0 
                                      ? 'text-success' 
                                      : prediction.predicted_return < 0 
                                        ? 'text-danger' 
                                        : 'text-default-500'
                                  }>
                                    {(prediction.predicted_return * 100).toFixed(2)}%
                                  </span>
                                </TableCell>
                                <TableCell>
                                  <Progress
                                    value={prediction.confidence_score * 100}
                                    size="sm"
                                    className="w-20"
                                  />
                                </TableCell>
                                <TableCell>
                                  <span className="text-sm text-default-500">
                                    [{(prediction.confidence_interval.lower * 100).toFixed(2)}%, {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
                                  </span>
                                </TableCell>
                                <TableCell>
                                  <span className="text-danger">
                                    {(prediction.risk_assessment.value_at_risk * 100).toFixed(2)}%
                                  </span>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Tab>
                    </Tabs>
                  </CardBody>
                </Card>
              )}
            </>
          )}
        </div>

        {/* 侧边栏 */}
        <div className="space-y-6">
          {/* 根据任务类型显示不同的侧边栏内容 */}
          {currentTask.task_type === 'backtest' ? (
            /* 回测任务侧边栏 */
            <BacktestTaskStatus 
              task={currentTask}
              onRetry={handleRetry}
              onStop={() => {/* TODO: 实现停止功能 */}}
              loading={refreshing}
            />
          ) : (
            /* 预测任务侧边栏 */
            <>
              {/* 统计信息 */}
              {currentTask.results && (
                <Card>
                  <CardHeader>
                    <h3 className="text-lg font-semibold">统计信息</h3>
                  </CardHeader>
                  <CardBody className="space-y-4">
                    <div className="grid grid-cols-1 gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-primary">{currentTask.results.total_stocks}</p>
                        <p className="text-sm text-default-500">总股票数</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-success">{currentTask.results.successful_predictions}</p>
                        <p className="text-sm text-default-500">成功预测</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-secondary">
                          {((currentTask.results.average_confidence || 0) * 100).toFixed(1)}%
                        </p>
                        <p className="text-sm text-default-500">平均置信度</p>
                      </div>
                    </div>
                  </CardBody>
                </Card>
              )}

              {/* 快速操作 */}
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">快速操作</h3>
                </CardHeader>
                <CardBody className="space-y-3">
                  <Button
                    variant="light"
                    startContent={<RefreshCw className="w-4 h-4" />}
                    onPress={handleRefresh}
                    isLoading={refreshing}
                    fullWidth
                  >
                    刷新状态
                  </Button>
                  
                  {currentTask.status === 'failed' && (
                    <Button
                      color="primary"
                      startContent={<Play className="w-4 h-4" />}
                      onPress={handleRetry}
                      fullWidth
                    >
                      重新运行
                    </Button>
                  )}
                  
                  {currentTask.status === 'completed' && (
                    <Button
                      color="secondary"
                      startContent={<Download className="w-4 h-4" />}
                      onPress={handleExport}
                      fullWidth
                    >
                      导出结果
                    </Button>
                  )}
                  
                  <Button
                    color="danger"
                    variant="light"
                    startContent={<Trash2 className="w-4 h-4" />}
                    onPress={onDeleteOpen}
                    fullWidth
                  >
                    删除任务
                  </Button>
                </CardBody>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* 删除确认对话框 */}
      <Modal isOpen={isDeleteOpen} onClose={onDeleteClose}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader className="flex flex-col gap-1">
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="w-5 h-5 text-danger" />
                  <span>确认删除</span>
                </div>
              </ModalHeader>
              <ModalBody>
                <p>确定要删除这个任务吗？此操作不可撤销。</p>
                {currentTask?.status === 'running' && (
                  <div className="mt-4 p-3 bg-warning-50 border border-warning-200 rounded-lg">
                    <p className="text-sm text-warning-700 mb-2">
                      ⚠️ 该任务当前正在运行中
                    </p>
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={deleteForce}
                        onChange={(e) => setDeleteForce(e.target.checked)}
                        className="w-4 h-4 text-danger rounded"
                      />
                      <span className="text-sm font-medium">强制删除（将中断正在运行的任务）</span>
                    </label>
                  </div>
                )}
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onPress={() => {
                  setDeleteForce(false);
                  onClose();
                }}>
                  取消
                </Button>
                <Button color="danger" onPress={() => {
                  handleDelete();
                  onClose();
                }}>
                  {deleteForce ? '强制删除' : '删除'}
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  );}
