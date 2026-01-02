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
import { LoadingSpinner } from '../common/LoadingSpinner';

interface TrainingReport {
  model_id: string;
  model_name: string;
  model_type: string;
  training_duration: number;
  total_epochs?: number;
  best_epoch?: number;
  final_metrics: Record<string, number>;
  training_history?: Array<{
    epoch: number;
    train_loss?: number;
    val_loss?: number;
    train_accuracy?: number;
    val_accuracy?: number;
  }>;
  feature_importance?: Record<string, number>;
  hyperparameters: Record<string, any>;
  training_data_info: {
    stock_codes: string[];
    start_date: string;
    end_date: string;
  };
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
      // TODO: 实现API调用获取训练报告
      // const response = await DataService.getTrainingReport(id);
      // setReport(response);
      
      // 模拟数据
      const mockReport: TrainingReport = {
        model_id: id,
        model_name: '测试模型',
        model_type: 'lightgbm',
        training_duration: 180.5,
        total_epochs: 100,
        best_epoch: 85,
        final_metrics: {
          accuracy: 0.8234,
          mse: 0.0156,
          mae: 0.0892,
          r2: 0.7845
        },
        training_history: [
          { epoch: 1, train_loss: 0.15, val_loss: 0.18, train_accuracy: 0.65, val_accuracy: 0.62 },
          { epoch: 2, train_loss: 0.12, val_loss: 0.16, train_accuracy: 0.72, val_accuracy: 0.68 },
          { epoch: 3, train_loss: 0.10, val_loss: 0.14, train_accuracy: 0.78, val_accuracy: 0.74 },
          // ... 更多历史数据
        ],
        feature_importance: {
          'RSI14': 0.15,
          'MACD': 0.12,
          'MA20': 0.10,
          'BOLL_UPPER': 0.08,
          'ATR14': 0.07,
          'VOLUME_MA_RATIO': 0.06,
        },
        hyperparameters: {
          learning_rate: 0.05,
          num_leaves: 210,
          max_depth: 8,
          feature_fraction: 0.85
        },
        training_data_info: {
          stock_codes: ['000001.SZ', '000002.SZ'],
          start_date: '2023-01-01',
          end_date: '2023-12-31'
        }
      };
      
      setReport(mockReport);
    } catch (error) {
      console.error('获取训练报告失败:', error);
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
    
    // TODO: 实现报告下载功能
    alert('报告下载功能开发中...');
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
                        {formatDuration(report.training_duration)}
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
                        {(report.final_metrics.accuracy * 100).toFixed(2)}%
                      </div>
                    </CardBody>
                  </Card>
                  
                  {report.total_epochs && (
                    <Card>
                      <CardBody className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Activity className="w-4 h-4 text-purple-500" />
                          <span className="text-sm text-default-500">总轮次</span>
                        </div>
                        <div className="text-lg font-semibold">
                          {report.total_epochs}
                        </div>
                      </CardBody>
                    </Card>
                  )}
                  
                  {report.best_epoch && (
                    <Card>
                      <CardBody className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4 text-orange-500" />
                          <span className="text-sm text-default-500">最佳轮次</span>
                        </div>
                        <div className="text-lg font-semibold">
                          {report.best_epoch}
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
                  {Object.entries(report.final_metrics).map(([key, value]) => (
                    <div key={key} className="bg-default-50 p-4 rounded-lg">
                      <div className="text-sm text-default-500 mb-1">
                        {key.toUpperCase()}
                      </div>
                      <div className="text-xl font-semibold">
                        {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 训练曲线占位符 */}
              {report.training_history && report.training_history.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">训练曲线</h3>
                  <div className="h-80 bg-default-50 rounded-lg flex items-center justify-center">
                    <div className="text-center text-default-500">
                      <TrendingUp className="w-8 h-8 mx-auto mb-2" />
                      <p>训练曲线图</p>
                      <p className="text-sm">(图表组件开发中)</p>
                    </div>
                  </div>
                </div>
              )}

              {/* 特征重要性 */}
              {report.feature_importance && Object.keys(report.feature_importance).length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">特征重要性</h3>
                  <div className="h-64 bg-default-50 rounded-lg flex items-center justify-center">
                    <div className="text-center text-default-500">
                      <BarChart3 className="w-8 h-8 mx-auto mb-2" />
                      <p>特征重要性图表</p>
                      <p className="text-sm">(图表组件开发中)</p>
                    </div>
                  </div>
                  
                  {/* 特征重要性列表 */}
                  <div className="mt-4">
                    <h4 className="font-medium mb-2">特征重要性排序</h4>
                    <div className="space-y-2">
                      {Object.entries(report.feature_importance)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 10)
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center justify-between">
                            <span className="text-sm">{feature}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-default-200 rounded-full h-2">
                                <div 
                                  className="bg-primary h-2 rounded-full"
                                  style={{ width: `${(importance * 100)}%` }}
                                />
                              </div>
                              <span className="text-xs text-default-500 w-12">
                                {(importance * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
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
                      <span>{report.training_data_info.stock_codes.join(', ')}</span>
                    </div>
                    <div>
                      <span className="text-sm text-default-500">数据范围: </span>
                      <span>{report.training_data_info.start_date} 至 {report.training_data_info.end_date}</span>
                    </div>
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