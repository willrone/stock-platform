/**
 * 训练报告详情弹窗组件
 * 
 * 显示模型训练完成后的详细报告，包括训练概览、曲线图、特征重要性等
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Card,
  CardBody,
  Chip,
  Accordion,
  AccordionItem,
} from '@heroui/react';
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
      <Modal isOpen={isOpen} onClose={onClose} size="5xl">
        <ModalContent>
          <ModalBody className="py-8">
            <LoadingSpinner />
          </ModalBody>
        </ModalContent>
      </Modal>
    );
  }

  return (
    <Modal 
      isOpen={isOpen} 
      onClose={onClose} 
      size="5xl" 
      scrollBehavior="inside"
    >
      <ModalContent>
        <ModalHeader>
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            <span>训练报告 - {report?.model_name}</span>
          </div>
        </ModalHeader>
        <ModalBody>
          {report ? (
            <div className="space-y-6">
              {/* 训练概览 */}
              <div>
                <h3 className="text-lg font-semibold mb-3">训练概览</h3>
                <div className="grid grid-cols-4 gap-4">
                  <Card>
                    <CardBody className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Clock className="w-4 h-4 text-blue-500" />
                        <span className="text-sm text-default-500">训练时长</span>
                      </div>
                      <div className="text-lg font-semibold">
                        {formatDuration(report.training_summary?.training_duration || report.training_duration || 0)}
                      </div>
                    </CardBody>
                  </Card>
                  
                  <Card>
                    <CardBody className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Target className="w-4 h-4 text-green-500" />
                        <span className="text-sm text-default-500">最终准确率</span>
                      </div>
                      <div className="text-lg font-semibold">
                        {((report.performance_metrics?.accuracy || report.final_metrics?.accuracy || 0) * 100).toFixed(2)}%
                      </div>
                    </CardBody>
                  </Card>
                  
                  {(report.training_summary?.epochs || report.total_epochs) && (
                    <Card>
                      <CardBody className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Activity className="w-4 h-4 text-purple-500" />
                          <span className="text-sm text-default-500">总轮次</span>
                        </div>
                        <div className="text-lg font-semibold">
                          {report.training_summary?.epochs || report.total_epochs}
                        </div>
                      </CardBody>
                    </Card>
                  )}
                  
                  {report.training_summary?.total_samples && (
                    <Card>
                      <CardBody className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4 text-orange-500" />
                          <span className="text-sm text-default-500">训练样本</span>
                        </div>
                        <div className="text-lg font-semibold">
                          {report.training_summary.total_samples.toLocaleString()}
                        </div>
                      </CardBody>
                    </Card>
                  )}
                </div>
              </div>
              
              {/* 性能指标 */}
              <div>
                <h3 className="text-lg font-semibold mb-3">性能指标</h3>
                <div className="grid grid-cols-2 gap-4">
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
                      <div key={key} className="bg-default-50 p-4 rounded-lg">
                        <div className="text-sm text-default-500 mb-1">
                          {key.toUpperCase().replace(/_/g, ' ')}
                        </div>
                        <div className={`text-xl font-semibold ${value === null || value === undefined ? 'text-default-400' : ''}`}>
                          {displayValue}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 训练曲线 */}
              {report.training_history && report.training_history.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">训练曲线</h3>
                  <div className="h-80 bg-default-50 rounded-lg p-4">
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
                  </div>
                </div>
              )}

              {/* 特征重要性 */}
              {report.feature_importance && (
                (Array.isArray(report.feature_importance) ? report.feature_importance.length > 0 : Object.keys(report.feature_importance).length > 0) && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">特征重要性</h3>
                  
                  {/* 特征重要性图表 */}
                  <div className="h-64 bg-default-50 rounded-lg p-4 mb-4">
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
                  </div>
                  
                  {/* 特征重要性列表 */}
                  <div className="mt-4">
                    <h4 className="font-medium mb-2">特征重要性排序</h4>
                    <div className="space-y-2">
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
                          <div key={feature} className="flex items-center justify-between">
                            <span className="text-sm">{feature}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-default-200 rounded-full h-2">
                                <div 
                                  className="bg-primary h-2 rounded-full"
                                  style={{ width: `${(importance / maxImportance) * 100}%` }}
                                />
                              </div>
                              <span className="text-xs text-default-500 w-12">
                                {(importance * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ))}

              {/* 建议和改进 */}
              {report.recommendations && report.recommendations.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">建议和改进</h3>
                  <Card>
                    <CardBody>
                      <ul className="space-y-2">
                        {report.recommendations.map((rec, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-primary mt-1">•</span>
                            <span className="text-sm">{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </CardBody>
                  </Card>
                </div>
              )}

              {/* 详细配置 */}
              <Accordion>
                <AccordionItem title="超参数配置" subtitle="查看训练时使用的超参数">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <pre className="text-sm overflow-x-auto">
                      {JSON.stringify(report.hyperparameters, null, 2)}
                    </pre>
                  </div>
                </AccordionItem>
                
                <AccordionItem title="训练数据信息" subtitle="查看训练数据的详细信息">
                  <div className="space-y-2">
                    <div>
                      <span className="text-sm text-default-500">股票代码: </span>
                      <span>{report.training_data_info.stock_codes?.join(', ') || '无'}</span>
                    </div>
                    <div>
                      <span className="text-sm text-default-500">数据范围: </span>
                      <span>{report.training_data_info.start_date || '未知'} 至 {report.training_data_info.end_date || '未知'}</span>
                    </div>
                    {report.training_summary && (
                      <>
                        <div>
                          <span className="text-sm text-default-500">训练样本数: </span>
                          <span>{report.training_summary.train_samples?.toLocaleString() || '未知'}</span>
                        </div>
                        <div>
                          <span className="text-sm text-default-500">验证样本数: </span>
                          <span>{report.training_summary.validation_samples?.toLocaleString() || '未知'}</span>
                        </div>
                        <div>
                          <span className="text-sm text-default-500">测试样本数: </span>
                          <span>{report.training_summary.test_samples?.toLocaleString() || '未知'}</span>
                        </div>
                        {report.training_summary.batch_size && (
                          <div>
                            <span className="text-sm text-default-500">批次大小: </span>
                            <span>{report.training_summary.batch_size}</span>
                          </div>
                        )}
                        {report.training_summary.learning_rate && (
                          <div>
                            <span className="text-sm text-default-500">学习率: </span>
                            <span>{report.training_summary.learning_rate}</span>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </AccordionItem>
              </Accordion>
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="text-default-500">无法加载训练报告</div>
            </div>
          )}
        </ModalBody>
        <ModalFooter>
          <Button variant="light" onPress={onClose}>
            关闭
          </Button>
          {report && (
            <Button 
              color="primary"
              startContent={<Download className="w-4 h-4" />}
              onPress={downloadReport}
            >
              下载报告
            </Button>
          )}
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};