/**
 * 优化任务详情组件
 * 包含状态监控和结果可视化
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Chip,
  Progress,
  Tabs,
  Tab,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  useDisclosure,
} from '@heroui/react';
import { ArrowLeft, RefreshCw, Save } from 'lucide-react';
import { OptimizationService, OptimizationStatus, OptimizationResult } from '../../services/optimizationService';
import { StrategyConfigService } from '../../services/strategyConfigService';
import { LoadingSpinner } from '../common/LoadingSpinner';
import OptimizationStatusMonitor from './OptimizationStatusMonitor';
import OptimizationVisualization from './OptimizationVisualization';
import { SaveStrategyConfigDialog } from '../backtest/SaveStrategyConfigDialog';

interface OptimizationTaskDetailProps {
  taskId: string;
  onBack: () => void;
}

export default function OptimizationTaskDetail({
  taskId,
  onBack,
}: OptimizationTaskDetailProps) {
  const [loading, setLoading] = useState(true);
  const [task, setTask] = useState<any>(null);
  const [status, setStatus] = useState<OptimizationStatus | null>(null);
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const { isOpen: isSaveConfigOpen, onOpen: onSaveConfigOpen, onClose: onSaveConfigClose } = useDisclosure();
  const [savingConfig, setSavingConfig] = useState(false);

  const loadTask = async () => {
    try {
      const taskData = await OptimizationService.getTask(taskId);
      setTask(taskData);
      if (taskData.result) {
        console.log('[OptimizationTaskDetail] task result:', taskData.result);
        console.log('[OptimizationTaskDetail] optimization_history:', taskData.result.optimization_history);
        console.log('[OptimizationTaskDetail] param_importance:', taskData.result.param_importance);
        setResult(taskData.result);
      } else {
        console.warn('[OptimizationTaskDetail] task result is empty:', taskData);
      }
    } catch (error) {
      console.error('加载任务详情失败:', error);
    }
  };

  const loadStatus = async () => {
    try {
      const statusData = await OptimizationService.getStatus(taskId);
      setStatus(statusData);
    } catch (error) {
      console.error('加载任务状态失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([loadTask(), loadStatus()]);
      setLoading(false);
    };
    loadData();

    // 如果任务正在运行，定期刷新状态
    const interval = setInterval(() => {
      if (task?.status === 'running') {
        loadStatus();
        loadTask();
      }
    }, 5000); // 每5秒刷新一次

    return () => clearInterval(interval);
  }, [taskId, task?.status]);

  // 保存策略配置
  const handleSaveConfig = async (configName: string, description: string) => {
    if (!result?.best_params || !task?.strategy_name) {
      throw new Error('无法获取最佳参数或策略名称');
    }

    setSavingConfig(true);
    try {
      await StrategyConfigService.createConfig({
        config_name: configName,
        strategy_name: task.strategy_name,
        parameters: result.best_params,
        description: description || `来自超参优化任务: ${task.task_name}`,
      });
      console.log('策略配置保存成功');
    } catch (error: any) {
      console.error('保存策略配置失败:', error);
      throw error;
    } finally {
      setSavingConfig(false);
    }
  };

  if (loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!task) {
    return (
      <div className="text-center py-8">
        <p className="text-default-500">任务不存在</p>
        <Button onPress={onBack} className="mt-4">
          返回列表
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            variant="light"
            isIconOnly
            onPress={onBack}
          >
            <ArrowLeft />
          </Button>
          <div>
            <h2 className="text-2xl font-bold">{task.task_name}</h2>
            <p className="text-default-500 text-sm">
              策略: {task.strategy_name} | 创建时间: {task.created_at ? new Date(task.created_at).toLocaleString('zh-CN') : '-'}
            </p>
          </div>
        </div>
        <Button
          variant="light"
          startContent={<RefreshCw size={16} />}
          onPress={() => {
            loadTask();
            loadStatus();
          }}
        >
          刷新
        </Button>
      </div>

      <Tabs aria-label="任务详情标签页">
        <Tab key="status" title="运行状态">
          <Card className="mt-4">
            <CardBody>
              {status && (
                <OptimizationStatusMonitor
                  status={status}
                  task={task}
                />
              )}
            </CardBody>
          </Card>
        </Tab>

        <Tab key="results" title="优化结果" isDisabled={task.status !== 'completed'}>
          <Card className="mt-4">
            <CardBody>
              {result ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <Card>
                      <CardBody>
                        <div className="text-center">
                          <p className="text-2xl font-bold text-success">
                            {result.best_score?.toFixed(4) || '-'}
                          </p>
                          <p className="text-sm text-default-500">最佳得分</p>
                        </div>
                      </CardBody>
                    </Card>
                    <Card>
                      <CardBody>
                        <div className="text-center">
                          <p className="text-2xl font-bold">
                            {result.completed_trials} / {result.n_trials}
                          </p>
                          <p className="text-sm text-default-500">完成试验数</p>
                        </div>
                      </CardBody>
                    </Card>
                    <Card>
                      <CardBody>
                        <div className="text-center">
                          <p className="text-2xl font-bold">
                            {result.optimization_metadata?.duration_seconds
                              ? `${Math.round(result.optimization_metadata.duration_seconds / 60)} 分钟`
                              : '-'}
                          </p>
                          <p className="text-sm text-default-500">优化耗时</p>
                        </div>
                      </CardBody>
                    </Card>
                  </div>

                  {result.best_params && (
                    <Card>
                      <CardHeader className="flex justify-between items-center">
                        <h3 className="text-lg font-semibold">最佳参数</h3>
                        <Button
                          color="primary"
                          variant="flat"
                          size="sm"
                          startContent={<Save className="w-4 h-4" />}
                          onPress={onSaveConfigOpen}
                        >
                          保存为配置
                        </Button>
                      </CardHeader>
                      <CardBody>
                        <div className="grid grid-cols-2 gap-4">
                          {Object.entries(result.best_params).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-default-600">{key}:</span>
                              <span className="font-medium">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </CardBody>
                    </Card>
                  )}

                  <OptimizationVisualization result={result} />
                </div>
              ) : (
                <div className="text-center py-8 text-default-500">
                  任务尚未完成，暂无结果
                </div>
              )}
            </CardBody>
          </Card>
        </Tab>
      </Tabs>

      {/* 保存配置对话框 */}
      {result?.best_params && task?.strategy_name && (
        <SaveStrategyConfigDialog
          isOpen={isSaveConfigOpen}
          onClose={onSaveConfigClose}
          strategyName={task.strategy_name}
          parameters={result.best_params}
          onSave={handleSaveConfig}
          loading={savingConfig}
        />
      )}
    </div>
  );
}

