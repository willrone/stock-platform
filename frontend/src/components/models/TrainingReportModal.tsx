/**
 * 训练报告详情弹窗组件
 * 
 * 显示模型训练完成后的详细报告，包括训练概览、曲线图、特征重要性等
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  Card,
  CardContent,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Typography,
} from '@mui/material';
import { ExpandMore } from '@mui/icons-material';
import {
  Brain,
  TrendingUp,
  BarChart3,
  Download,
  Clock,
  Target,
  Zap,
  Activity,
} from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { DataService } from '../../services/dataService';

interface TrainingReport {
  model_id: string;
  model_name: string;
  model_type: string;
  version?: string;
  created_at?: string;
  training_summary?: {
    training_duration: number;
    total_samples?: number;
    train_samples?: number;
    validation_samples?: number;
    test_samples?: number;
    epochs?: number;
    batch_size?: number;
    learning_rate?: number;
  };
  performance_metrics?: {
    accuracy: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    rmse?: number;
    mae?: number;
    r2?: number;
    mse?: number;
    sharpe_ratio?: number;
    total_return?: number;
    max_drawdown?: number;
    win_rate?: number;
  };
  training_history?: Array<{
    epoch?: number;
    train_loss?: number;
    val_loss?: number;
    train_accuracy?: number;
    val_accuracy?: number;
    timestamp?: string;
  }>;
  feature_importance?: Array<{
    feature_name: string;
    importance: number;
    rank?: number;
  }> | Record<string, number>;
  feature_correlation?: {
    target_correlations?: Record<string, number>;
    high_correlation_pairs?: Array<{
      feature1: string;
      feature2: string;
      correlation: number;
    }>;
    avg_target_correlation?: number;
    max_target_correlation?: number;
    error?: string;
  };
  hyperparameter_tuning?: {
    strategy?: string;
    trials?: number;
    best_score?: number;
    best_hyperparameters?: Record<string, any> | null;
  };
  hyperparameters: Record<string, any>;
  training_data_info: {
    stock_codes: string[];
    start_date: string;
    end_date: string;
  };
  recommendations?: string[];
  // 兼容旧格式
  training_duration?: number;
  total_epochs?: number;
  best_epoch?: number;
  final_metrics?: Record<string, number>;
}

interface TrainingReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelId: string | null;
}

export const TrainingReportModal: React.FC<TrainingReportModalProps> = ({
  isOpen,
  onClose,
  modelId
}) => {
  const [report, setReport] = useState<TrainingReport | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen && modelId) {
      fetchTrainingReport(modelId);
    }
  }, [isOpen, modelId]);

  const fetchTrainingReport = async (id: string) => {
    setLoading(true);
    try {
      const response = await DataService.getTrainingReport(id);
      // 转换后端数据格式为前端格式
      const report: TrainingReport = {
        model_id: response.model_id || id,
        model_name: response.model_name || '未知模型',
        model_type: response.model_type || 'unknown',
        version: response.version,
        created_at: response.created_at,
        training_summary: response.training_summary,
        performance_metrics: response.performance_metrics,
        training_history: response.training_history,
        feature_importance: response.feature_importance,
        feature_correlation: response.feature_correlation,
        hyperparameter_tuning: response.hyperparameter_tuning,
        hyperparameters: response.hyperparameters || {},
        training_data_info: response.training_data_info || {
          stock_codes: [],
          start_date: '',
          end_date: ''
        },
        recommendations: response.recommendations,
        // 兼容字段
        training_duration: response.training_summary?.training_duration || response.training_duration,
        total_epochs: response.training_summary?.epochs || response.total_epochs,
        final_metrics: response.performance_metrics || response.final_metrics,
      };
      
      setReport(report);
    } catch (error: any) {
      console.error('获取训练报告失败:', error);
      // 如果API失败，显示错误信息
      if (error?.response?.status === 404) {
        alert('该模型尚未生成评估报告，请等待训练完成');
      } else {
        alert('获取训练报告失败: ' + (error?.message || '未知错误'));
      }
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}小时${minutes}分钟${secs}秒`;
    } else if (minutes > 0) {
      return `${minutes}分钟${secs}秒`;
    } else {
      return `${secs}秒`;
    }
  };

  const downloadReport = () => {
    if (!report) return;
    
    try {
      // 准备下载的数据
      const reportData = {
        model_id: report.model_id,
        model_name: report.model_name,
        model_type: report.model_type,
        version: report.version,
        created_at: report.created_at || new Date().toISOString(),
        training_summary: report.training_summary,
        performance_metrics: report.performance_metrics,
        training_history: report.training_history,
        feature_importance: report.feature_importance,
        feature_correlation: report.feature_correlation,
        hyperparameter_tuning: report.hyperparameter_tuning,
        hyperparameters: report.hyperparameters,
        training_data_info: report.training_data_info,
        recommendations: report.recommendations,
      };
      
      // 转换为JSON字符串
      const jsonString = JSON.stringify(reportData, null, 2);
      
      // 创建Blob并下载
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `training_report_${report.model_id}_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('下载报告失败:', error);
      alert('下载报告失败，请稍后重试');
    }
  };

  if (loading) {
    return (
      <Dialog open={isOpen} onClose={onClose} maxWidth="xl" fullWidth>
        <DialogContent sx={{ py: 4 }}>
          <LoadingSpinner />
        </DialogContent>
      </Dialog>
    );
  }

  const correlationEntries = report?.feature_correlation?.target_correlations
    ? Object.entries(report.feature_correlation.target_correlations)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 15)
    : [];
  const highCorrelationPairs = report?.feature_correlation?.high_correlation_pairs || [];

  return (
    <Dialog 
      open={isOpen} 
      onClose={onClose} 
      maxWidth="xl" 
      fullWidth
      scroll="paper"
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Brain size={20} />
          <span>训练报告 - {report?.model_name}</span>
        </Box>
      </DialogTitle>
      <DialogContent>
          {report ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* 训练概览 */}
              <Box>
                <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                  训练概览
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                  <Card>
                    <CardContent sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Clock size={16} color="#1976d2" />
                        <Typography variant="caption" color="text.secondary">训练时长</Typography>
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {formatDuration(report.training_summary?.training_duration || report.training_duration || 0)}
                      </Typography>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Target size={16} color="#2e7d32" />
                        <Typography variant="caption" color="text.secondary">最终准确率</Typography>
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {((report.performance_metrics?.accuracy || report.final_metrics?.accuracy || 0) * 100).toFixed(2)}%
                      </Typography>
                    </CardContent>
                  </Card>
                  
                  {(report.training_summary?.epochs || report.total_epochs) && (
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Activity size={16} color="#9c27b0" />
                          <Typography variant="caption" color="text.secondary">总轮次</Typography>
                        </Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {report.training_summary?.epochs || report.total_epochs}
                        </Typography>
                      </CardContent>
                    </Card>
                  )}
                  
                  {report.training_summary?.total_samples && (
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Zap size={16} color="#ed6c02" />
                          <Typography variant="caption" color="text.secondary">训练样本</Typography>
                        </Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {report.training_summary.total_samples.toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              </Box>
              
              {/* 性能指标 */}
              <Box>
                <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                  性能指标
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                  {Object.entries(report.performance_metrics || report.final_metrics || {}).map(([key, value]) => {
                    // 格式化显示值
                    let displayValue: string;
                    if (value === null || value === undefined) {
                      displayValue = 'N/A';
                    } else if (typeof value === 'number') {
                      if (key.includes('ratio') || key.includes('rate') || key.includes('accuracy') || key.includes('precision') || key.includes('recall') || key.includes('f1') || key.includes('win_rate')) {
                        displayValue = (value * 100).toFixed(2) + '%';
                      } else if (key.includes('drawdown') || key.includes('return')) {
                        displayValue = (value * 100).toFixed(2) + '%';
                      } else {
                        displayValue = value.toFixed(4);
                      }
                    } else {
                      displayValue = String(value);
                    }
                    
                    return (
                      <Box key={key} sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          {key.toUpperCase().replace(/_/g, ' ')}
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 600, color: value === null || value === undefined ? 'text.disabled' : 'text.primary' }}>
                          {displayValue}
                        </Typography>
                      </Box>
                    );
                  })}
                </Box>
              </Box>

              {/* 训练曲线 */}
              {report.training_history && report.training_history.length > 0 && (
                <Box>
                  <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                    训练曲线
                  </Typography>
                  <Box sx={{ height: 320, bgcolor: 'grey.50', borderRadius: 1, p: 2 }}>
                    <ReactECharts
                      option={{
                        title: {
                          text: '训练过程指标变化',
                          left: 'center',
                          textStyle: { fontSize: 14 }
                        },
                        tooltip: {
                          trigger: 'axis',
                          axisPointer: { type: 'cross' }
                        },
                        legend: {
                          data: ['训练损失', '验证损失', '训练准确率', '验证准确率'].filter((_, idx) => {
                            const hasLoss = report.training_history?.some(h => h.train_loss !== undefined || h.val_loss !== undefined);
                            const hasAccuracy = report.training_history?.some(h => h.train_accuracy !== undefined || h.val_accuracy !== undefined);
                            if (idx < 2) return hasLoss;
                            return hasAccuracy;
                          }),
                          bottom: 0
                        },
                        grid: {
                          left: '3%',
                          right: '4%',
                          bottom: '15%',
                          containLabel: true
                        },
                        xAxis: {
                          type: 'category',
                          boundaryGap: false,
                          data: report.training_history.map((h, idx) => h.epoch !== undefined ? `Epoch ${h.epoch}` : idx + 1)
                        },
                        yAxis: [
                          {
                            type: 'value',
                            name: '损失',
                            position: 'left',
                            show: report.training_history.some(h => h.train_loss !== undefined || h.val_loss !== undefined)
                          },
                          {
                            type: 'value',
                            name: '准确率',
                            position: 'right',
                            min: 0,
                            max: 1,
                            show: report.training_history.some(h => h.train_accuracy !== undefined || h.val_accuracy !== undefined)
                          }
                        ],
                        series: [
                          ...(report.training_history.some(h => h.train_loss !== undefined) ? [{
                            name: '训练损失',
                            type: 'line',
                            yAxisIndex: 0,
                            data: report.training_history.map(h => h.train_loss).filter(v => v !== undefined),
                            smooth: true,
                            itemStyle: { color: '#5470c6' }
                          }] : []),
                          ...(report.training_history.some(h => h.val_loss !== undefined) ? [{
                            name: '验证损失',
                            type: 'line',
                            yAxisIndex: 0,
                            data: report.training_history.map(h => h.val_loss).filter(v => v !== undefined),
                            smooth: true,
                            itemStyle: { color: '#91cc75' }
                          }] : []),
                          ...(report.training_history.some(h => h.train_accuracy !== undefined) ? [{
                            name: '训练准确率',
                            type: 'line',
                            yAxisIndex: 1,
                            data: report.training_history.map(h => h.train_accuracy).filter(v => v !== undefined),
                            smooth: true,
                            itemStyle: { color: '#fac858' }
                          }] : []),
                          ...(report.training_history.some(h => h.val_accuracy !== undefined) ? [{
                            name: '验证准确率',
                            type: 'line',
                            yAxisIndex: 1,
                            data: report.training_history.map(h => h.val_accuracy).filter(v => v !== undefined),
                            smooth: true,
                            itemStyle: { color: '#ee6666' }
                          }] : [])
                        ]
                      }}
                      style={{ height: '100%', width: '100%' }}
                    />
                  </Box>
                </Box>
              )}

              {/* 特征重要性 */}
              {report.feature_importance && (
                (Array.isArray(report.feature_importance) ? report.feature_importance.length > 0 : Object.keys(report.feature_importance).length > 0) && (
                <Box>
                  <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                    特征重要性
                  </Typography>
                  
                  {/* 特征重要性图表 */}
                  <Box sx={{ height: 256, bgcolor: 'grey.50', borderRadius: 1, p: 2, mb: 2 }}>
                    <ReactECharts
                      option={{
                        title: {
                          text: 'Top 10 特征重要性',
                          left: 'center',
                          textStyle: { fontSize: 14 }
                        },
                        tooltip: {
                          trigger: 'axis',
                          axisPointer: { type: 'shadow' }
                        },
                        grid: {
                          left: '15%',
                          right: '4%',
                          bottom: '10%',
                          containLabel: true
                        },
                        xAxis: {
                          type: 'value',
                          name: '重要性'
                        },
                        yAxis: {
                          type: 'category',
                          data: (Array.isArray(report.feature_importance) 
                            ? report.feature_importance
                                .sort((a: any, b: any) => (b.importance || b) - (a.importance || a))
                                .slice(0, 10)
                                .map((f: any) => f.feature_name || f)
                            : Object.entries(report.feature_importance)
                                .sort(([,a], [,b]) => (b as number) - (a as number))
                                .slice(0, 10)
                                .map(([name]) => name)
                          ).reverse()
                        },
                        series: [{
                          name: '重要性',
                          type: 'bar',
                          data: (Array.isArray(report.feature_importance)
                            ? report.feature_importance
                                .sort((a: any, b: any) => (b.importance || b) - (a.importance || a))
                                .slice(0, 10)
                                .map((f: any) => f.importance || f)
                            : Object.entries(report.feature_importance)
                                .sort(([,a], [,b]) => (b as number) - (a as number))
                                .slice(0, 10)
                                .map(([, importance]) => importance as number)
                          ).reverse(),
                          itemStyle: {
                            color: '#5470c6'
                          }
                        }]
                      }}
                      style={{ height: '100%', width: '100%' }}
                    />
                  </Box>
                  
                  {/* 特征重要性列表 */}
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                      特征重要性排序
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {(Array.isArray(report.feature_importance)
                        ? report.feature_importance
                            .sort((a: any, b: any) => (b.importance || b) - (a.importance || a))
                            .slice(0, 10)
                        : Object.entries(report.feature_importance)
                            .sort(([,a], [,b]) => (b as number) - (a as number))
                            .slice(0, 10)
                            .map(([feature_name, importance]) => ({ feature_name, importance: importance as number }))
                      ).map((item: any) => {
                        const feature = item.feature_name || item[0] || '';
                        const importance = item.importance !== undefined ? item.importance : (item[1] || 0);
                        const maxImportance = Array.isArray(report.feature_importance)
                          ? Math.max(...report.feature_importance.map((f: any) => f.importance || f))
                          : Math.max(...Object.values(report.feature_importance as Record<string, number>));
                        
                        return (
                          <Box key={feature} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Typography variant="body2">{feature}</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Box sx={{ width: 80, bgcolor: 'grey.200', borderRadius: '999px', height: 8 }}>
                                <Box 
                                  sx={{ 
                                    bgcolor: 'primary.main', 
                                    height: 8, 
                                    borderRadius: '999px',
                                    width: `${(importance / maxImportance) * 100}%` 
                                  }}
                                />
                              </Box>
                              <Typography variant="caption" color="text.secondary" sx={{ width: 48 }}>
                                {(importance * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </Box>
                        );
                      })}
                    </Box>
                  </Box>
                </Box>
              ))}

              {/* 特征相关性 */}
              {report.feature_correlation && !report.feature_correlation.error && correlationEntries.length > 0 && (
                <Box>
                  <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                    特征相关性
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2, mb: 2 }}>
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          平均相关性
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {(report.feature_correlation.avg_target_correlation || 0).toFixed(4)}
                        </Typography>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          最大相关性
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {(report.feature_correlation.max_target_correlation || 0).toFixed(4)}
                        </Typography>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          高相关特征对
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {highCorrelationPairs.length}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Box>

                  <Box sx={{ height: 256, bgcolor: 'grey.50', borderRadius: 1, p: 2, mb: 2 }}>
                    <ReactECharts
                      option={{
                        title: {
                          text: 'Top 15 特征-目标相关性',
                          left: 'center',
                          textStyle: { fontSize: 14 }
                        },
                        tooltip: {
                          trigger: 'axis',
                          axisPointer: { type: 'shadow' },
                          formatter: (params: any) => {
                            const item = Array.isArray(params) ? params[0] : params;
                            return `${item.name}: ${item.value.toFixed(4)}`;
                          }
                        },
                        grid: {
                          left: '20%',
                          right: '4%',
                          bottom: '10%',
                          containLabel: true
                        },
                        xAxis: {
                          type: 'value',
                          name: '|corr|'
                        },
                        yAxis: {
                          type: 'category',
                          data: correlationEntries.map(([name]) => name).reverse()
                        },
                        series: [{
                          name: '相关性',
                          type: 'bar',
                          data: correlationEntries.map(([, value]) => Number(value)).reverse(),
                          itemStyle: {
                            color: '#73c0de'
                          }
                        }]
                      }}
                      style={{ height: '100%', width: '100%' }}
                    />
                  </Box>

                  {highCorrelationPairs.length > 0 && (
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                        高相关特征对（|corr| &gt; 0.8）
                      </Typography>
                      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 1 }}>
                        {highCorrelationPairs.slice(0, 6).map((pair, idx) => (
                          <Typography key={`${pair.feature1}-${pair.feature2}-${idx}`} variant="body2" color="text.secondary">
                            {pair.feature1} × {pair.feature2}: {pair.correlation.toFixed(3)}
                          </Typography>
                        ))}
                      </Box>
                    </Box>
                  )}
                </Box>
              )}

              {/* 超参数调优 */}
              {report.hyperparameter_tuning && (
                <Box>
                  <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                    超参数调优
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2, mb: 2 }}>
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          搜索策略
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {report.hyperparameter_tuning.strategy || 'unknown'}
                        </Typography>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          试验次数
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {report.hyperparameter_tuning.trials ?? 0}
                        </Typography>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                          最佳得分
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {report.hyperparameter_tuning.best_score !== undefined && report.hyperparameter_tuning.best_score !== null
                            ? report.hyperparameter_tuning.best_score.toFixed(4)
                            : 'N/A'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Box>
                  {report.hyperparameter_tuning.best_hyperparameters && (
                    <Card>
                      <CardContent>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                          最佳超参数
                        </Typography>
                        <Box component="pre" sx={{ fontSize: '0.75rem', whiteSpace: 'pre-wrap', m: 0 }}>
                          {JSON.stringify(report.hyperparameter_tuning.best_hyperparameters, null, 2)}
                        </Box>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              )}

              {/* 建议和改进 */}
              {report.recommendations && report.recommendations.length > 0 && (
                <Box>
                  <Typography variant="h6" component="h3" sx={{ fontWeight: 600, mb: 2 }}>
                    建议和改进
                  </Typography>
                  <Card>
                    <CardContent>
                      <Box component="ul" sx={{ m: 0, pl: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
                        {report.recommendations.map((rec, idx) => (
                          <Typography key={idx} component="li" variant="body2" sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                            <Box component="span" sx={{ color: 'primary.main', mt: 0.5 }}>•</Box>
                            <Box component="span">{rec}</Box>
                          </Typography>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              )}

              {/* 详细配置 */}
              <Box>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        超参数配置
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        查看训练时使用的超参数
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                      <Box component="pre" sx={{ fontSize: '0.875rem', overflowX: 'auto', m: 0 }}>
                        {JSON.stringify(report.hyperparameters, null, 2)}
                      </Box>
                    </Box>
                  </AccordionDetails>
                </Accordion>
                
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        训练数据信息
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        查看训练数据的详细信息
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box>
                        <Typography variant="caption" color="text.secondary" component="span">
                          股票代码:{' '}
                        </Typography>
                        <Typography variant="body2" component="span">
                          {report.training_data_info.stock_codes?.join(', ') || '无'}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary" component="span">
                          数据范围:{' '}
                        </Typography>
                        <Typography variant="body2" component="span">
                          {report.training_data_info.start_date || '未知'} 至 {report.training_data_info.end_date || '未知'}
                        </Typography>
                      </Box>
                      {report.training_summary && (
                        <>
                          <Box>
                            <Typography variant="caption" color="text.secondary" component="span">
                              训练样本数:{' '}
                            </Typography>
                            <Typography variant="body2" component="span">
                              {report.training_summary.train_samples?.toLocaleString() || '未知'}
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="text.secondary" component="span">
                              验证样本数:{' '}
                            </Typography>
                            <Typography variant="body2" component="span">
                              {report.training_summary.validation_samples?.toLocaleString() || '未知'}
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="text.secondary" component="span">
                              测试样本数:{' '}
                            </Typography>
                            <Typography variant="body2" component="span">
                              {report.training_summary.test_samples?.toLocaleString() || '未知'}
                            </Typography>
                          </Box>
                          {report.training_summary.batch_size && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" component="span">
                                批次大小:{' '}
                              </Typography>
                              <Typography variant="body2" component="span">
                                {report.training_summary.batch_size}
                              </Typography>
                            </Box>
                          )}
                          {report.training_summary.learning_rate && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" component="span">
                                学习率:{' '}
                              </Typography>
                              <Typography variant="body2" component="span">
                                {report.training_summary.learning_rate}
                              </Typography>
                            </Box>
                          )}
                        </>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Box>
            </Box>
          ) : (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body2" color="text.secondary">
                无法加载训练报告
              </Typography>
            </Box>
          )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>关闭</Button>
        {report && (
          <Button 
            variant="contained"
            color="primary"
            startIcon={<Download size={16} />}
            onClick={downloadReport}
          >
            下载报告
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};
