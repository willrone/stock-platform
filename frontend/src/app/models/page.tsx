/**
 * 模型管理页面
 * 
 * 提供模型创建、查看和管理功能
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
} from '@heroui/react';
import {
  Plus,
  Brain,
  TrendingUp,
  Trash2,
} from 'lucide-react';
import { DataService } from '../../services/dataService';
import { useDataStore, Model } from '../../stores/useDataStore';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { TrainingReportModal } from '../../components/models/TrainingReportModal';
import { FeatureSelector } from '../../components/models/FeatureSelector';
import { ModelListTable } from '../../components/models/ModelListTable';
import { LiveTrainingModal } from '../../components/models/LiveTrainingModal';
import { CreateModelForm } from '../../components/models/CreateModelForm';
import { getTrainingProgressWebSocket, cleanupTrainingProgressWebSocket } from '../../services/TrainingProgressWebSocket';

export default function ModelsPage() {
  const { models, setModels } = useDataStore();
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { isOpen: isTrainingReportOpen, onOpen: onTrainingReportOpen, onClose: onTrainingReportClose } = useDisclosure();
  const { isOpen: isLiveTrainingOpen, onOpen: onLiveTrainingOpen, onClose: onLiveTrainingClose } = useDisclosure();
  const { isOpen: isDeleteModalOpen, onOpen: onDeleteModalOpen, onClose: onDeleteModalClose } = useDisclosure();
  const [trainingReportModelId, setTrainingReportModelId] = useState<string | null>(null);
  const [liveTrainingModelId, setLiveTrainingModelId] = useState<string | null>(null);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  
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
    num_iterations: 100, // 训练迭代次数（epochs）
    selected_features: [] as string[], // 选择的特征列表
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [useAllFeatures, setUseAllFeatures] = useState(true); // 是否使用所有特征

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

  // 显示删除确认对话框
  const showDeleteConfirm = (modelId: string) => {
    setDeletingModelId(modelId);
    onDeleteModalOpen();
  };

  // 删除模型
  const handleDeleteModel = async () => {
    if (!deletingModelId) return;

    setDeleting(true);
    try {
      await DataService.deleteModel(deletingModelId);
      // 刷新模型列表
      await loadModels();
      // 关闭对话框
      onDeleteModalClose();
      setDeletingModelId(null);
      alert('模型删除成功');
    } catch (error: any) {
      console.error('删除模型失败:', error);
      alert(error?.message || '删除模型失败，请稍后重试');
    } finally {
      setDeleting(false);
    }
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
      // 确保num_iterations被包含在hyperparameters中
      const submitData = {
        ...formData,
        hyperparameters: {
          ...formData.hyperparameters,
          num_iterations: formData.num_iterations || 100
        },
        // 如果使用所有特征，不传递selected_features（或传递null）
        selected_features: useAllFeatures ? undefined : (formData.selected_features.length > 0 ? formData.selected_features : undefined)
      };
      const result = await DataService.createModel(submitData);
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
        num_iterations: 100,
        selected_features: [],
      });
      setErrors({});
      setUseAllFeatures(true);
      
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

  // 处理表单数据变化
  const handleFormDataChange = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
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
            <ModelListTable
              models={models}
              trainingProgress={trainingProgress}
              getStatusColor={getStatusColor}
              getStatusText={getStatusText}
              getStageText={getStageText}
              onShowTrainingReport={showTrainingReport}
              onShowLiveTraining={showLiveTrainingDetails}
              onDeleteModel={showDeleteConfirm}
              deleting={deleting}
            />
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
            <CreateModelForm
              formData={formData}
              errors={errors}
              useAllFeatures={useAllFeatures}
              onFormDataChange={handleFormDataChange}
              onUseAllFeaturesChange={setUseAllFeatures}
              onSelectedFeaturesChange={(features) => handleFormDataChange('selected_features', features)}
            />
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

      {/* 删除确认对话框 */}
      <Modal isOpen={isDeleteModalOpen} onClose={onDeleteModalClose}>
        <ModalContent>
          <ModalHeader>
            <div className="flex items-center space-x-2">
              <Trash2 className="w-5 h-5 text-danger" />
              <span>确认删除模型</span>
            </div>
          </ModalHeader>
          <ModalBody>
            <p>您确定要删除此模型吗？此操作不可撤销。</p>
            {deletingModelId && (
              <p className="text-sm text-default-500 mt-2">
                模型ID: {deletingModelId}
              </p>
            )}
          </ModalBody>
          <ModalFooter>
            <Button
              variant="light"
              onPress={onDeleteModalClose}
              isDisabled={deleting}
            >
              取消
            </Button>
            <Button
              color="danger"
              onPress={handleDeleteModel}
              isLoading={deleting}
              startContent={!deleting && <Trash2 className="w-4 h-4" />}
            >
              {deleting ? '删除中...' : '确认删除'}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* 实时训练监控弹窗 */}
      <LiveTrainingModal
        isOpen={isLiveTrainingOpen}
        onClose={onLiveTrainingClose}
        modelId={liveTrainingModelId}
        models={models}
        trainingProgress={trainingProgress}
        getStageText={getStageText}
      />
    </div>
  );
}