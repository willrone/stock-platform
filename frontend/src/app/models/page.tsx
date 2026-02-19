/**
 * 模型管理页面
 *
 * 提供模型创建、查看和管理功能
 */

'use client';

import React, { useCallback, useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Box,
  Typography,
} from '@mui/material';
import { Plus, Brain, TrendingUp, Trash2 } from 'lucide-react';
import { DataService } from '../../services/dataService';
import { useDataStore, Model } from '../../stores/useDataStore';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { ModelListTable } from '../../components/models/ModelListTable';

const TrainingReportModal = dynamic(
  () => import('../../components/models/TrainingReportModal').then(mod => ({ default: mod.TrainingReportModal })),
  { ssr: false }
);
import { LiveTrainingModal } from '../../components/models/LiveTrainingModal';
import { CreateModelForm } from '../../components/models/CreateModelForm';
import {
  getTrainingProgressWebSocket,
  cleanupTrainingProgressWebSocket,
} from '../../services/TrainingProgressWebSocket';

export default function ModelsPage() {
  const { models, setModels } = useDataStore();
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [isTrainingReportOpen, setIsTrainingReportOpen] = useState(false);
  const [isLiveTrainingOpen, setIsLiveTrainingOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [trainingReportModelId, setTrainingReportModelId] = useState<string | null>(null);
  const [liveTrainingModelId, setLiveTrainingModelId] = useState<string | null>(null);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  // WebSocket连接状态
  const [, setWsConnected] = useState(false);
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
    num_iterations: 1000,
    selected_features: [] as string[],
    feature_set: 'alpha158',
    label_type: 'regression',
    binary_threshold: 0.0,
    split_method: 'purged_cv',
    train_end_date: '',
    val_end_date: '',
    // 滚动训练（P2）
    enable_rolling: false,
    rolling_window_type: 'sliding',
    rolling_step: 60,
    rolling_train_window: 480,
    rolling_valid_window: 60,
    enable_sample_decay: true,
    sample_decay_rate: 0.999,
    // CSRankNorm 标签变换
    enable_cs_rank_norm: false,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [useAllFeatures, setUseAllFeatures] = useState(true); // 是否使用所有特征

  // 加载模型列表
  const loadModels = useCallback(async () => {
    try {
      setLoading(true);
      const data = await DataService.getModels();
      setModels(data.models || data);
    } catch (error) {
      console.error('加载模型列表失败:', error);
    } finally {
      setLoading(false);
    }
  }, [setModels]);

  // 处理WebSocket消息
  const handleWebSocketMessage = useCallback((data: any) => {
    if (data.type === 'model:training:progress') {
      setTrainingProgress(prev => ({
        ...prev,
        [data.model_id]: {
          progress: data.progress,
          stage: data.stage,
          message: data.message,
          metrics: data.metrics || {},
          timestamp: data.timestamp,
        },
      }));

      // 更新模型列表中的进度
      const updatedModels = models.map(
        (model: Model): Model =>
          model.model_id === data.model_id
            ? {
                ...model,
                training_progress: data.progress,
                training_stage: data.stage,
                status: (data.progress >= 100 ? 'ready' : 'training') as Model['status'],
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
      const updatedModels = models.map(
        (model: Model): Model =>
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
  }, [models, setModels, loadModels]);

  // 获取状态颜色
  const getStatusColor = (
    status: string
  ): 'success' | 'primary' | 'secondary' | 'warning' | 'error' | 'default' => {
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
        return 'error';
      default:
        return 'default';
    }
  };

  // 获取状态文本
  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      ready: '就绪',
      active: '活跃',
      deployed: '已部署',
      training: '训练中',
      failed: '失败',
    };
    return statusMap[status] || status;
  };

  // 获取训练阶段文本
  const getStageText = (stage: string) => {
    const stageMap: Record<string, string> = {
      initializing: '初始化中',
      preparing: '准备数据',
      configuring: '配置模型',
      preprocessing: '数据预处理',
      training: '模型训练',
      evaluating: '模型评估',
      saving: '保存模型',
      completed: '训练完成',
      failed: '训练失败',
      hyperparameter_tuning: '超参数调优',
    };
    return stageMap[stage] || stage;
  };

  // 显示实时训练详情
  const showLiveTrainingDetails = (modelId: string) => {
    setLiveTrainingModelId(modelId);
    setIsLiveTrainingOpen(true);
  };

  // 显示训练报告
  const showTrainingReport = (modelId: string) => {
    setTrainingReportModelId(modelId);
    setIsTrainingReportOpen(true);
  };

  // 显示删除确认对话框
  const showDeleteConfirm = (modelId: string) => {
    setDeletingModelId(modelId);
    setIsDeleteModalOpen(true);
  };

  // 删除模型
  const handleDeleteModel = async () => {
    if (!deletingModelId) {
      return;
    }

    setDeleting(true);
    try {
      await DataService.deleteModel(deletingModelId);
      // 刷新模型列表
      await loadModels();
      // 关闭对话框
      setIsDeleteModalOpen(false);
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
          num_iterations: formData.num_iterations || 1000,
        },
        // 如果使用所有特征，不传递selected_features（或传递null）
        selected_features: useAllFeatures
          ? undefined
          : formData.selected_features.length > 0
            ? formData.selected_features
            : undefined,
        // 新增训练选项
        feature_set: formData.feature_set,
        label_type: formData.label_type,
        binary_threshold: formData.label_type === 'binary' ? formData.binary_threshold : undefined,
        split_method: formData.split_method,
        train_end_date:
          formData.split_method === 'hardcut' && formData.train_end_date
            ? formData.train_end_date
            : undefined,
        val_end_date:
          formData.split_method === 'hardcut' && formData.val_end_date
            ? formData.val_end_date
            : undefined,
        // 滚动训练（P2）
        enable_rolling: formData.enable_rolling,
        rolling_window_type: formData.enable_rolling ? formData.rolling_window_type : undefined,
        rolling_step: formData.enable_rolling ? formData.rolling_step : undefined,
        rolling_train_window: formData.enable_rolling ? formData.rolling_train_window : undefined,
        rolling_valid_window: formData.enable_rolling ? formData.rolling_valid_window : undefined,
        enable_sample_decay: formData.enable_rolling ? formData.enable_sample_decay : undefined,
        sample_decay_rate: formData.enable_rolling ? formData.sample_decay_rate : undefined,
        // CSRankNorm 标签变换
        enable_cs_rank_norm: formData.enable_cs_rank_norm,
      };
      await DataService.createModel(submitData);

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
        num_iterations: 1000,
        selected_features: [],
        feature_set: 'alpha158',
        label_type: 'regression',
        binary_threshold: 0.0,
        split_method: 'purged_cv',
        train_end_date: '',
        val_end_date: '',
        // 滚动训练（P2）
        enable_rolling: false,
        rolling_window_type: 'sliding',
        rolling_step: 60,
        rolling_train_window: 480,
        rolling_valid_window: 60,
        enable_sample_decay: true,
        sample_decay_rate: 0.999,
        // CSRankNorm 标签变换
        enable_cs_rank_norm: false,
      });
      setErrors({});
      setUseAllFeatures(true);

      // 关闭对话框并刷新列表
      setIsOpen(false);
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
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  useEffect(() => {
    loadModels();

    // 设置WebSocket连接
    const setupWebSocketConnection = async () => {
      try {
        const wsClient = getTrainingProgressWebSocket();

        // 连接WebSocket
        await wsClient.connect();
        setWsConnected(true);

        // 订阅所有训练进度更新
        const unsubscribe = wsClient.subscribeToAll(data => {
          handleWebSocketMessage(data);
        });

        // 保存取消订阅函数以便清理
        if (typeof window !== 'undefined') {
          (window as any).unsubscribeTrainingProgress = unsubscribe;
        }
      } catch (error) {
        console.error('设置WebSocket连接失败:', error);
        setWsConnected(false);
      }
    };
    setupWebSocketConnection();

    return () => {
      // 清理WebSocket连接
      cleanupTrainingProgressWebSocket();
    };
  }, [loadModels, handleWebSocketMessage]);

  if (loading && models.length === 0) {
    return <LoadingSpinner text="加载模型列表..." />;
  }

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', px: { xs: 1.5, sm: 2, md: 3 }, py: { xs: 2, md: 3 } }}>
      {/* 页面标题 */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', sm: 'row' },
          justifyContent: 'space-between',
          alignItems: { xs: 'flex-start', sm: 'center' },
          gap: 2,
          mb: 3,
        }}
      >
        <Box>
          <Typography
            variant="h4"
            component="h1"
            sx={{
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' },
            }}
          >
            <Brain size={32} />
            模型管理
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            创建和管理预测模型
          </Typography>
        </Box>
        <Button
          variant="contained"
          color="primary"
          startIcon={<Plus size={16} />}
          onClick={() => setIsOpen(true)}
        >
          创建模型
        </Button>
      </Box>

      {/* 模型列表 */}
      <Card>
        <CardHeader
          avatar={<TrendingUp size={24} />}
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>模型列表</span>
              <Typography variant="body2" color="text.secondary">
                ({models.length} 个)
              </Typography>
            </Box>
          }
        />
        <CardContent>
          {models.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 6 }}>
              <Brain size={64} color="#ccc" style={{ margin: '0 auto 16px' }} />
              <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                暂无模型
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                创建您的第一个预测模型开始使用
              </Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<Plus size={16} />}
                onClick={() => setIsOpen(true)}
              >
                创建模型
              </Button>
            </Box>
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
        </CardContent>
      </Card>

      {/* 创建模型对话框 */}
      <Dialog
        open={isOpen}
        onClose={() => setIsOpen(false)}
        maxWidth="lg"
        fullWidth
        sx={{ '& .MuiDialog-paper': { m: { xs: 1, sm: 2 } } }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Plus size={20} />
            <span>创建新模型</span>
          </Box>
        </DialogTitle>
        <DialogContent>
          <CreateModelForm
            formData={formData}
            errors={errors}
            useAllFeatures={useAllFeatures}
            onFormDataChange={handleFormDataChange}
            onUseAllFeaturesChange={setUseAllFeatures}
            onSelectedFeaturesChange={features =>
              handleFormDataChange('selected_features', features)
            }
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsOpen(false)}>取消</Button>
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={creating}
            startIcon={!creating ? <Plus size={16} /> : undefined}
          >
            {creating ? '创建中...' : '创建模型'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 训练报告详情弹窗 */}
      <TrainingReportModal
        isOpen={isTrainingReportOpen}
        onClose={() => setIsTrainingReportOpen(false)}
        modelId={trainingReportModelId}
      />

      {/* 删除确认对话框 */}
      <Dialog open={isDeleteModalOpen} onClose={() => setIsDeleteModalOpen(false)}>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Trash2 size={20} color="#d32f2f" />
            <span>确认删除模型</span>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography>您确定要删除此模型吗？此操作不可撤销。</Typography>
          {deletingModelId && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              模型ID: {deletingModelId}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsDeleteModalOpen(false)} disabled={deleting}>
            取消
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleDeleteModel}
            disabled={deleting}
            startIcon={!deleting ? <Trash2 size={16} /> : undefined}
          >
            {deleting ? '删除中...' : '确认删除'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 实时训练监控弹窗 */}
      <LiveTrainingModal
        isOpen={isLiveTrainingOpen}
        onClose={() => setIsLiveTrainingOpen(false)}
        modelId={liveTrainingModelId}
        models={models}
        trainingProgress={trainingProgress}
        getStageText={getStageText}
      />
    </Box>
  );
}
