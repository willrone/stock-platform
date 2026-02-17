/**
 * 任务详情页面 - 重构版本
 *
 * 显示任务的详细信息，包括：
 * - 任务基本信息
 * - 实时进度更新
 * - 预测结果展示（TradingView图表 + ECharts）
 * - 回测结果展示（多维度分析）
 * - 操作控制
 */

'use client';

import React, { useState } from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useRouter, useParams } from 'next/navigation';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import BacktestProgressMonitor from '@/components/backtest/BacktestProgressMonitor';
import { SaveStrategyConfigDialog } from '@/components/backtest/SaveStrategyConfigDialog';
import { StrategyConfigService } from '@/services/strategyConfigService';

// Hooks
import { useTaskDetail } from './hooks/useTaskDetail';
import { useTaskWebSocket } from './hooks/useTaskWebSocket';
import { useBacktestData } from './hooks/useBacktestData';
import { useTaskActions } from './hooks/useTaskActions';

// Components
import { TaskHeader } from './components/TaskHeader';
import { TaskProgress } from './components/TaskProgress';
import { TaskInfo } from './components/TaskInfo';
import { StrategyConfig } from './components/StrategyConfig';
import { BacktestTabs } from './components/BacktestTabs';
import { PredictionTabs } from './components/PredictionTabs';
import { TaskSidebar } from './components/TaskSidebar';
import { DeleteTaskDialog } from './components/DeleteTaskDialog';
import { PerformanceMonitor } from './components/PerformanceMonitor';

export default function TaskDetailPage() {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;

  // 自定义 Hooks
  const { currentTask, loading, predictions, selectedStock, setSelectedStock, loadTaskDetail } =
    useTaskDetail(taskId);

  const {
    backtestDetailedData,
    adaptedRiskData,
    adaptedPerformanceData,
    loadingBacktestData,
    loadBacktestDetailedData,
  } = useBacktestData(taskId, currentTask);

  const {
    refreshing,
    isDeleteOpen,
    deleteForce,
    setIsDeleteOpen,
    setDeleteForce,
    handleRefresh,
    handleRetry,
    handleDelete,
    handleExport,
    handleRebuild,
  } = useTaskActions(taskId, currentTask, loadTaskDetail);

  // WebSocket 实时更新
  useTaskWebSocket({
    taskId,
    currentTask,
    onTaskCompleted: async () => {
      await loadTaskDetail();
      if (currentTask?.task_type === 'backtest') {
        await loadBacktestDetailedData(true);
      }
    },
  });

  // 保存策略���置
  const [isSaveConfigOpen, setIsSaveConfigOpen] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);

  const handleSaveConfig = async (configName: string, description: string) => {
    if (!currentTask || currentTask.task_type !== 'backtest') {
      throw new Error('无法获取策略配置信息');
    }

    const cfg = currentTask.config;
    const bc = cfg?.backtest_config || cfg;
    const backtestData =
      currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results;
    const resultBc = backtestData?.backtest_config;

    let strategyName =
      bc?.strategy_name ??
      cfg?.strategy_name ??
      resultBc?.strategy_name ??
      (backtestData as any)?.strategy_name ??
      '未知策略';

    const parameters: Record<string, any> =
      bc?.strategy_config != null
        ? bc.strategy_config
        : cfg?.strategy_config != null
          ? cfg.strategy_config
          : resultBc?.strategy_config != null
            ? resultBc.strategy_config
            : {};

    if (strategyName === '未知策略' && Array.isArray((parameters as any)?.strategies)) {
      strategyName = 'portfolio';
    }

    setSavingConfig(true);
    try {
      await StrategyConfigService.createConfig({
        config_name: configName,
        strategy_name: strategyName,
        parameters: parameters,
        description: description,
      });
      console.log('策略配置保存成功');
    } catch (error: any) {
      console.error('保存策略配置失败:', error);
      throw error;
    } finally {
      setSavingConfig(false);
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  if (loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!currentTask) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 384,
          gap: 2,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          任务不存在或已被删除
        </Typography>
        <Button variant="contained" color="primary" onClick={handleBack}>
          返回任务列表
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: { xs: 2, sm: 3 } }}>
      {/* 页面标题 */}
      <TaskHeader
        task={currentTask}
        refreshing={refreshing}
        onBack={handleBack}
        onRefresh={handleRefresh}
        onRebuild={handleRebuild}
        onRetry={handleRetry}
        onExport={handleExport}
        onDelete={() => setIsDeleteOpen(true)}
      />

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' },
          gap: { xs: 2, lg: 3 },
        }}
      >
        {/* 主要内容区域 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: { xs: 2, lg: 3 }, minWidth: 0 }}>
          {/* 任务进度 */}
          {currentTask.task_type === 'backtest' &&
          (currentTask.status === 'running' || currentTask.status === 'created') ? (
            <BacktestProgressMonitor
              taskId={taskId}
              onComplete={results => {
                console.log('回测完成:', results);
                loadTaskDetail();
              }}
              onError={error => {
                console.error('回测错误:', error);
                loadTaskDetail();
              }}
              onCancel={() => {
                console.log('回测已取消');
                loadTaskDetail();
              }}
            />
          ) : (
            <TaskProgress task={currentTask} />
          )}

          {/* 根据任务类型显示不同内容 */}
          {currentTask.task_type === 'backtest' ? (
            /* 回测任务 */
            <BacktestTabs
              task={currentTask}
              taskId={taskId}
              selectedStock={selectedStock}
              backtestDetailedData={backtestDetailedData}
              adaptedRiskData={adaptedRiskData}
              adaptedPerformanceData={adaptedPerformanceData}
              loadingBacktestData={loadingBacktestData}
              onTabChange={tab => {
                console.log('[TaskDetail] 切换到页签:', tab);
                if (
                  tab === 'positions' &&
                  currentTask.status === 'completed' &&
                  !backtestDetailedData &&
                  !loadingBacktestData
                ) {
                  console.log('[TaskDetail] 切换到持仓分析页签，触发数据加载');
                  loadBacktestDetailedData();
                }
              }}
              renderStrategyConfig={() => (
                <StrategyConfig task={currentTask} onSaveConfig={() => setIsSaveConfigOpen(true)} />
              )}
              renderPerformanceMonitor={() => <PerformanceMonitor task={currentTask} />}
            />
          ) : (
            /* 预测任务 */
            <>
              <TaskInfo task={currentTask} />
              {currentTask.status === 'completed' && predictions.length > 0 && (
                <PredictionTabs
                  taskId={taskId}
                  predictions={predictions}
                  selectedStock={selectedStock}
                  onStockChange={setSelectedStock}
                  backtestData={
                    currentTask.results?.backtest_results ||
                    currentTask.backtest_results ||
                    (currentTask.task_type === 'backtest' ? currentTask.result : null)
                  }
                  showBacktestTab={currentTask.task_type === 'backtest'}
                />
              )}
            </>
          )}
        </Box>

        {/* 侧边栏 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: { xs: 2, lg: 3 }, minWidth: 0 }}>
          <TaskSidebar
            task={currentTask}
            refreshing={refreshing}
            onRefresh={handleRefresh}
            onRetry={handleRetry}
            onExport={handleExport}
            onDelete={() => setIsDeleteOpen(true)}
          />
        </Box>
      </Box>

      {/* 保存策略配置对话框 */}
      {currentTask.task_type === 'backtest' && (
        <SaveStrategyConfigDialog
          isOpen={isSaveConfigOpen}
          onClose={() => setIsSaveConfigOpen(false)}
          strategyName={
            currentTask.config?.backtest_config?.strategy_name ||
            currentTask.config?.strategy_name ||
            'portfolio'
          }
          parameters={
            currentTask.config?.backtest_config?.strategy_config ||
            currentTask.config?.strategy_config ||
            {}
          }
          onSave={handleSaveConfig}
          loading={savingConfig}
        />
      )}

      {/* 删除确认对话框 */}
      <DeleteTaskDialog
        isOpen={isDeleteOpen}
        task={currentTask}
        deleteForce={deleteForce}
        onClose={() => {
          setDeleteForce(false);
          setIsDeleteOpen(false);
        }}
        onConfirm={() => {
          handleDelete();
          setIsDeleteOpen(false);
        }}
        onForceChange={setDeleteForce}
      />
    </Box>
  );
}
