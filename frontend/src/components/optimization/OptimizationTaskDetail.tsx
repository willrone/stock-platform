/**
 * 优化任务详情组件
 * 包含状态监控和结果可视化
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Button,
  Chip,
  LinearProgress,
  Tabs,
  Tab,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Box,
  Typography,
  TableContainer,
  Paper,
} from '@mui/material';
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
  const [isSaveConfigOpen, setIsSaveConfigOpen] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>('status');
  const onSaveConfigOpen = () => setIsSaveConfigOpen(true);
  const onSaveConfigClose = () => setIsSaveConfigOpen(false);

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
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body2" color="text.secondary">任务不存在</Typography>
        <Button onClick={onBack} variant="outlined" sx={{ mt: 2 }}>
          返回列表
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Button
            variant="outlined"
            onClick={onBack}
            sx={{ minWidth: 40, px: 1 }}
          >
            <ArrowLeft size={20} />
          </Button>
          <Box>
            <Typography variant="h4" component="h2" sx={{ fontWeight: 600 }}>
              {task.task_name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              策略: {task.strategy_name} | 创建时间: {task.created_at ? new Date(task.created_at).toLocaleString('zh-CN') : '-'}
            </Typography>
          </Box>
        </Box>
        <Button
          variant="outlined"
          startIcon={<RefreshCw size={16} />}
          onClick={() => {
            loadTask();
            loadStatus();
          }}
        >
          刷新
        </Button>
      </Box>

      <Box>
        <Tabs value={selectedTab} onChange={(e, newValue) => setSelectedTab(newValue)} aria-label="任务详情标签页">
          <Tab label="运行状态" value="status" />
          <Tab label="优化结果" value="results" disabled={task.status !== 'completed'} />
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {selectedTab === 'status' && (
            <Card>
              <CardContent>
                {status && (
                  <OptimizationStatusMonitor
                    status={status}
                    task={task}
                  />
                )}
              </CardContent>
            </Card>
          )}

          {selectedTab === 'results' && (
            <Card>
              <CardContent>
                {result ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
                      <Card>
                        <CardContent>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                              {result.best_score?.toFixed(4) || '-'}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">最佳得分</Typography>
                          </Box>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h4" sx={{ fontWeight: 600 }}>
                              {result.completed_trials} / {result.n_trials}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">完成试验数</Typography>
                          </Box>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h4" sx={{ fontWeight: 600 }}>
                              {result.optimization_metadata?.duration_seconds
                                ? `${Math.round(result.optimization_metadata.duration_seconds / 60)} 分钟`
                                : '-'}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">优化耗时</Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Box>

                    {result.best_params && (
                      <Card>
                        <CardHeader
                          title={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                              <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                                最佳参数
                              </Typography>
                              <Button
                                variant="outlined"
                                size="small"
                                startIcon={<Save size={16} />}
                                onClick={onSaveConfigOpen}
                              >
                                保存为配置
                              </Button>
                            </Box>
                          }
                        />
                        <CardContent>
                          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                            {Object.entries(result.best_params).map(([key, value]) => (
                              <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="body2" color="text.secondary">{key}:</Typography>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>{String(value)}</Typography>
                              </Box>
                            ))}
                          </Box>
                        </CardContent>
                      </Card>
                    )}

                    <OptimizationVisualization result={result} />
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      任务尚未完成，暂无结果
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>

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
    </Box>
  );
}

