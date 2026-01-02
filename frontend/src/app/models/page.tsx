/**
 * 模型管理页面
 * 
 * 提供模型创建、查看和管理功能
 */

'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Input,
  Select,
  SelectItem,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Progress,
  Tooltip,
} from '@heroui/react';
import {
  Plus,
  Brain,
  TrendingUp,
  Calendar,
  Settings,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import {
  Accordion,
  AccordionItem,
  Checkbox,
} from '@heroui/react';
import { DataService } from '../../services/dataService';
import { useDataStore, Model } from '../../stores/useDataStore';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { StockSelector } from '../../components/tasks/StockSelector';
import { TrainingReportModal } from '../../components/models/TrainingReportModal';
import { getTrainingProgressWebSocket, cleanupTrainingProgressWebSocket } from '../../services/TrainingProgressWebSocket';

export default function ModelsPage() {
  const router = useRouter();
  const { models, setModels } = useDataStore();
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { isOpen: isTrainingReportOpen, onOpen: onTrainingReportOpen, onClose: onTrainingReportClose } = useDisclosure();
  const { isOpen: isLiveTrainingOpen, onOpen: onLiveTrainingOpen, onClose: onLiveTrainingClose } = useDisclosure();
  const [trainingReportModelId, setTrainingReportModelId] = useState<string | null>(null);
  const [liveTrainingModelId, setLiveTrainingModelId] = useState<string | null>(null);
  
  // WebSocket连接状态
  const [wsConnected, setWsConnected] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<Record<string, any>>({});
  
  const [formData, setFormData] = useState({
    model_name: '',
    model_type: 'lightgbm',
    stock_codes: [] as string[],
    start_date: '',
    end_date: '',
    description: '',
    hyperparameters: {} as Record<string, any>,
    enable_hyperparameter_tuning: false,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // 加载模型列表
  const loadModels = async () => {
    try {
      setLoading(true);
      const data = await DataService.getModels();
      setModels(data.models || data);
    } catch (error) {
      console.error('加载模型列表失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 设置WebSocket连接
  const setupWebSocketConnection = async () => {
    try {
      const wsClient = getTrainingProgressWebSocket();
      
      // 连接WebSocket
      await wsClient.connect();
      setWsConnected(true);
      
      // 订阅所有训练进度更新
      const unsubscribe = wsClient.subscribeToAll((data) => {
        handleWebSocketMessage(data);
      });
      
      // 保存取消订阅函数以便清理
      (window as any).unsubscribeTrainingProgress = unsubscribe;
      
    } catch (error) {
      console.error('设置WebSocket连接失败:', error);
      setWsConnected(false);
    }
  };

  // 处理WebSocket消息
  const handleWebSocketMessage = (data: any) => {
    if (data.type === 'model:training:progress') {
      setTrainingProgress(prev => ({
        ...prev,
        [data.model_id]: {
          progress: data.progress,
          stage: data.stage,
          message: data.message,
          metrics: data.metrics || {},
          timestamp: data.timestamp
        }
      }));
      
      // 更新模型列表中的进度
      const updatedModels = models.map((model: Model): Model => 
        model.model_id === data.model_id 
          ? { 
              ...model, 
              training_progress: data.progress,
              training_stage: data.stage,
              status: (data.progress >= 100 ? 'ready' : 'training') as Model['status']
            }
          : model
      );
      setModels(updatedModels);
    } else if (data.type === 'model:training:completed') {
      // 训练完成，刷新模型列表
      loadModels();
      setTrainingProgress(prev => {
        const updated = { ...prev };
        delete updated[data.model_id];
        return updated;
      });
    } else if (data.type === 'model:training:failed') {
      // 训练失败，更新状态
      const updatedModels = models.map((model: Model): Model => 
        model.model_id === data.model_id 
          ? { ...model, status: 'failed', training_stage: 'failed' }
          : model
      );
      setModels(updatedModels);
      setTrainingProgress(prev => {
        const updated = { ...prev };
        delete updated[data.model_id];
        return updated;
      });
    }
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready':
        return 'success';
      case 'active':
        return 'primary';
      case 'deployed':
        return 'secondary';
      case 'training':
        return 'warning';
      case 'failed':
        return 'danger';
      default:
        return 'default';
    }
  };

  // 获取状态文本
  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      'ready': '就绪',
      'active': '活跃',
      'deployed': '已部署',
      'training': '训练中',
      'failed': '失败',
    };
    return statusMap[status] || status;
  };

  // 获取训练阶段文本
  const getStageText = (stage: string) => {
    const stageMap: Record<string, string> = {
      'initializing': '初始化中',
      'preparing': '准备数据',
      'configuring': '配置模型',
      'preprocessing': '数据预处理',
      'training': '模型训练',
      'evaluating': '模型评估',
      'saving': '保存模型',
      'completed': '训练完成',
      'failed': '训练失败',
      'hyperparameter_tuning': '超参数调优'
    };
    return stageMap[stage] || stage;
  };

  // 显示实时训练详情
  const showLiveTrainingDetails = (modelId: string) => {
    setLiveTrainingModelId(modelId);
    onLiveTrainingOpen();
  };

  // 显示训练报告
  const showTrainingReport = (modelId: string) => {
    setTrainingReportModelId(modelId);
    onTrainingReportOpen();
  };

  // 验证表单
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.model_name.trim()) {
      newErrors.model_name = '请输入模型名称';
    }
    
    if (formData.stock_codes.length === 0) {
      newErrors.stock_codes = '请至少选择一只股票用于训练';
    }
    
    if (!formData.start_date) {
      newErrors.start_date = '请选择训练数据开始日期';
    }
    
    if (!formData.end_date) {
      newErrors.end_date = '请选择训练数据结束日期';
    }
    
    if (formData.start_date && formData.end_date && formData.start_date >= formData.end_date) {
      newErrors.end_date = '结束日期必须晚于开始日期';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // 提交创建模型表单
  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    setCreating(true);
    try {
      const result = await DataService.createModel(formData);
      console.log('模型创建成功:', result);
      
      // 重置表单
      setFormData({
        model_name: '',
        model_type: 'lightgbm',
        stock_codes: [],
        start_date: '',
        end_date: '',
        description: '',
        hyperparameters: {},
        enable_hyperparameter_tuning: false,
      });
      setErrors({});
      
      // 关闭对话框并刷新列表
      onClose();
      await loadModels();
      
      alert('模型创建成功！训练任务已开始，您可以在模型列表中查看进度。');
    } catch (error: any) {
      console.error('创建模型失败:', error);
      alert(error?.message || '创建模型失败，请稍后重试');
    } finally {
      setCreating(false);
    }
  };

  useEffect(() => {
    loadModels();
    setupWebSocketConnection();
    
    return () => {
      // 清理WebSocket连接
      cleanupTrainingProgressWebSocket();
    };
  }, []);

  if (loading && models.length === 0) {
    return <LoadingSpinner text="加载模型列表..." />;
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8" />
            模型管理
          </h1>
          <p className="text-default-500 mt-2">创建和管理预测模型</p>
        </div>
        <Button
          color="primary"
          startContent={<Plus className="w-4 h-4" />}
          onPress={onOpen}
        >
          创建模型
        </Button>
      </div>

      {/* 模型列表 */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <h2 className="text-lg font-semibold">模型列表</h2>
            <span className="text-default-500">({models.length} 个)</span>
          </div>
        </CardHeader>
        <CardBody>
          {models.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="w-16 h-16 mx-auto text-default-300 mb-4" />
              <p className="text-default-500 text-lg mb-2">暂无模型</p>
              <p className="text-default-400 text-sm mb-4">
                创建您的第一个预测模型开始使用
              </p>
              <Button
                color="primary"
                startContent={<Plus className="w-4 h-4" />}
                onPress={onOpen}
              >
                创建模型
              </Button>
            </div>
          ) : (
            <Table aria-label="模型列表">
              <TableHeader>
                <TableColumn>模型名称</TableColumn>
                <TableColumn>类型</TableColumn>
                <TableColumn>准确率</TableColumn>
                <TableColumn>状态</TableColumn>
                <TableColumn>创建时间</TableColumn>
                <TableColumn>操作</TableColumn>
              </TableHeader>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.model_id}>
                    <TableCell>
                      <div className="flex flex-col">
                        <span className="font-medium">{model.model_name}</span>
                        {model.description && (
                          <span className="text-xs text-default-500">
                            {model.description}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>{model.model_type}</TableCell>
                    <TableCell>
                      <span className="font-medium">
                        {(model.accuracy * 100).toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1">
                        <Chip
                          size="sm"
                          color={getStatusColor(model.status)}
                          variant="flat"
                        >
                          {getStatusText(model.status)}
                        </Chip>
                        
                        {/* 训练进度显示 */}
                        {model.status === 'training' && (
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <Progress 
                                value={trainingProgress[model.model_id]?.progress || model.training_progress || 0}
                                size="sm"
                                className="flex-1"
                                color="primary"
                              />
                              <span className="text-xs text-default-500 min-w-[40px]">
                                {(trainingProgress[model.model_id]?.progress || model.training_progress || 0).toFixed(0)}%
                              </span>
                            </div>
                            
                            {/* 训练阶段和消息 */}
                            <div className="text-xs text-default-400">
                              {trainingProgress[model.model_id]?.message || 
                               getStageText(trainingProgress[model.model_id]?.stage || model.training_stage)}
                            </div>
                            
                            {/* 实时指标 */}
                            {trainingProgress[model.model_id]?.metrics && Object.keys(trainingProgress[model.model_id].metrics).length > 0 && (
                              <div className="flex gap-2 text-xs">
                                {Object.entries(trainingProgress[model.model_id].metrics).slice(0, 2).map(([key, value]) => (
                                  <Tooltip key={key} content={key}>
                                    <span className="text-default-500">
                                      {key}: {typeof value === 'number' ? value.toFixed(3) : String(value)}
                                    </span>
                                  </Tooltip>
                                ))}
                              </div>
                            )}
                            
                            {/* 查看实时详情按钮 */}
                            <Button
                              size="sm"
                              variant="light"
                              color="primary"
                              className="text-xs h-6"
                              onPress={() => showLiveTrainingDetails(model.model_id)}
                            >
                              查看实时详情
                            </Button>
                          </div>
                        )}
                        
                        {/* 非训练状态的简单显示 */}
                        {model.status !== 'training' && model.training_stage && (
                          <span className="text-xs text-default-400">
                            {getStageText(model.training_stage)}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {new Date(model.created_at).toLocaleDateString('zh-CN')}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        {model.status === 'ready' && (
                          <Button
                            size="sm"
                            variant="light"
                            color="primary"
                            onPress={() => showTrainingReport(model.model_id)}
                          >
                            查看报告
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardBody>
      </Card>

      {/* 创建模型对话框 */}
      <Modal isOpen={isOpen} onClose={onClose} size="4xl" scrollBehavior="inside">
        <ModalContent>
          <ModalHeader>
            <div className="flex items-center space-x-2">
              <Plus className="w-5 h-5" />
              <span>创建新模型</span>
            </div>
          </ModalHeader>
          <ModalBody>
            <div className="space-y-4">
              <Input
                label="模型名称"
                placeholder="请输入模型名称"
                value={formData.model_name}
                onValueChange={(value) =>
                  setFormData((prev) => ({ ...prev, model_name: value }))
                }
                isInvalid={!!errors.model_name}
                errorMessage={errors.model_name}
                isRequired
              />

              <Select
                label="模型类型"
                selectedKeys={[formData.model_type]}
                onSelectionChange={(keys) => {
                  const type = Array.from(keys)[0] as string;
                  setFormData((prev) => ({ ...prev, model_type: type }));
                }}
                description="选择要训练的模型类型（基于Qlib框架统一训练）"
              >
                <SelectItem key="lightgbm" description="推荐：高效的梯度提升模型，适合表格数据">
                  LightGBM (推荐)
                </SelectItem>
                <SelectItem key="xgboost" description="经典的梯度提升模型，性能稳定">
                  XGBoost
                </SelectItem>
                <SelectItem key="linear_regression" description="简单的线性回归模型，训练快速">
                  线性回归
                </SelectItem>
                <SelectItem key="transformer" description="Transformer模型，适合复杂时序模式">
                  Transformer
                </SelectItem>
              </Select>

              <StockSelector
                value={formData.stock_codes}
                onChange={(codes) =>
                  setFormData((prev) => ({ ...prev, stock_codes: codes }))
                }
              />
              {errors.stock_codes && (
                <p className="text-danger text-sm mt-1">{errors.stock_codes}</p>
              )}

              <div className="grid grid-cols-2 gap-4">
                <Input
                  type="date"
                  label="训练数据开始日期"
                  value={formData.start_date}
                  onValueChange={(value) =>
                    setFormData((prev) => ({ ...prev, start_date: value }))
                  }
                  isInvalid={!!errors.start_date}
                  errorMessage={errors.start_date}
                  isRequired
                />
                <Input
                  type="date"
                  label="训练数据结束日期"
                  value={formData.end_date}
                  onValueChange={(value) =>
                    setFormData((prev) => ({ ...prev, end_date: value }))
                  }
                  isInvalid={!!errors.end_date}
                  errorMessage={errors.end_date}
                  isRequired
                />
              </div>

              <Input
                label="模型描述（可选）"
                placeholder="请输入模型描述"
                value={formData.description}
                onValueChange={(value) =>
                  setFormData((prev) => ({ ...prev, description: value }))
                }
              />

              <Checkbox
                isSelected={formData.enable_hyperparameter_tuning}
                onValueChange={(checked) =>
                  setFormData((prev) => ({ ...prev, enable_hyperparameter_tuning: checked }))
                }
              >
                启用自动超参数调优
              </Checkbox>
            </div>
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={onClose}>
              取消
            </Button>
            <Button
              color="primary"
              onPress={handleSubmit}
              isLoading={creating}
              startContent={!creating ? <Plus className="w-4 h-4" /> : undefined}
            >
              {creating ? '创建中...' : '创建模型'}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* 训练报告详情弹窗 */}
      <TrainingReportModal
        isOpen={isTrainingReportOpen}
        onClose={onTrainingReportClose}
        modelId={trainingReportModelId}
      />

      {/* 实时训练监控弹窗 */}
      <Modal isOpen={isLiveTrainingOpen} onClose={onLiveTrainingClose} size="5xl" scrollBehavior="inside">
        <ModalContent>
          <ModalHeader>
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5" />
              <span>实时训练监控</span>
              {liveTrainingModelId && (
                <Chip size="sm" color="primary" variant="flat">
                  {models.find(m => m.model_id === liveTrainingModelId)?.model_name || '未知模型'}
                </Chip>
              )}
            </div>
          </ModalHeader>
          <ModalBody>
            {liveTrainingModelId && trainingProgress[liveTrainingModelId] ? (
              <div className="space-y-6">
                {/* 顶部：实时指标概览 */}
                <div className="grid grid-cols-2 gap-6">
                  {/* 左侧：实时指标 */}
                  <div className="space-y-4">
                    <h4 className="font-semibold">实时指标</h4>
                    <div className="grid grid-cols-2 gap-4">
                      {trainingProgress[liveTrainingModelId].metrics && Object.entries(trainingProgress[liveTrainingModelId].metrics).map(([key, value]) => (
                        <Card key={key} className="p-3">
                          <div className="text-sm text-default-500">{key}</div>
                          <div className="text-lg font-semibold">
                            {typeof value === 'number' ? value.toFixed(4) : String(value)}
                          </div>
                        </Card>
                      ))}
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>总体进度</span>
                        <span>{trainingProgress[liveTrainingModelId].progress?.toFixed(1)}%</span>
                      </div>
                      <Progress value={trainingProgress[liveTrainingModelId].progress || 0} />
                      <div className="text-xs text-default-500">
                        当前阶段: {getStageText(trainingProgress[liveTrainingModelId].stage)}
                      </div>
                      {trainingProgress[liveTrainingModelId].message && (
                        <div className="text-xs text-default-400">
                          {trainingProgress[liveTrainingModelId].message}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* 右侧：训练曲线占位符 */}
                  <div className="space-y-4">
                    <h4 className="font-semibold">训练曲线</h4>
                    <div className="h-64 bg-default-50 rounded-lg flex items-center justify-center">
                      <div className="text-center text-default-500">
                        <TrendingUp className="w-8 h-8 mx-auto mb-2" />
                        <p>训练曲线图</p>
                        <p className="text-sm">(开发中)</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* 训练日志占位符 */}
                <div>
                  <h4 className="font-semibold mb-2">训练日志</h4>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg h-32 overflow-y-auto text-sm font-mono">
                    <div>[{new Date().toLocaleTimeString()}] 训练进度: {trainingProgress[liveTrainingModelId].progress?.toFixed(1)}%</div>
                    <div>[{new Date().toLocaleTimeString()}] 当前阶段: {getStageText(trainingProgress[liveTrainingModelId].stage)}</div>
                    {trainingProgress[liveTrainingModelId].message && (
                      <div>[{new Date().toLocaleTimeString()}] {trainingProgress[liveTrainingModelId].message}</div>
                    )}
                    <div className="text-gray-500">更多日志功能开发中...</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="text-default-500">
                  {liveTrainingModelId ? '暂无训练数据' : '请选择一个训练中的模型'}
                </div>
              </div>
            )}
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={onLiveTrainingClose}>
              关闭
            </Button>
            {liveTrainingModelId && trainingProgress[liveTrainingModelId] && (
              <Button 
                color="danger" 
                variant="light"
                onPress={() => {
                  // TODO: 实现停止训练功能
                  alert('停止训练功能开发中...');
                }}
              >
                停止训练
              </Button>
            )}
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}