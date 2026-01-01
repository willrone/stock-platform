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
import { DataService } from '../../services/dataService';
import { useDataStore } from '../../stores/useDataStore';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { StockSelector } from '../../components/tasks/StockSelector';

export default function ModelsPage() {
  const router = useRouter();
  const { models, setModels } = useDataStore();
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { isOpen: isDetailOpen, onOpen: onDetailOpen, onClose: onDetailClose } = useDisclosure();
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null);
  const [retrainingModelId, setRetrainingModelId] = useState<string | null>(null);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);
  
  const [formData, setFormData] = useState({
    model_name: '',
    model_type: 'random_forest',
    stock_codes: [] as string[],
    start_date: '',
    end_date: '',
    description: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // 加载模型列表
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const result = await DataService.getModels();
      setModels(result.models);
    } catch (error) {
      console.error('加载模型列表失败:', error);
    } finally {
      setLoading(false);
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

  // 创建模型
  const handleCreateModel = async () => {
    if (!validateForm()) {
      return;
    }

    setCreating(true);
    try {
      const result = await DataService.createModel({
        model_name: formData.model_name,
        model_type: formData.model_type,
        stock_codes: formData.stock_codes,
        start_date: formData.start_date,
        end_date: formData.end_date,
        description: formData.description,
      });
      
      console.log('模型创建成功:', result);
      onClose();
      resetForm();
      await loadModels();
    } catch (error) {
      console.error('创建模型失败:', error);
      setErrors({ submit: '创建模型失败，请稍后重试' });
    } finally {
      setCreating(false);
    }
  };

  // 重置表单
  const resetForm = () => {
    setFormData({
      model_name: '',
      model_type: 'random_forest',
      stock_codes: [],
      start_date: '',
      end_date: '',
      description: '',
    });
    setErrors({});
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready':
      case 'active':
      case 'deployed':
        return 'success';
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

  // 重建模型任务
  const handleRetrainModel = async (model: Model) => {
    if (!confirm(`确定要重新训练模型 "${model.model_name}" 吗？这将创建一个新的训练任务。`)) {
      return;
    }

    setRetrainingModelId(model.model_id);
    try {
      // 获取模型详情以获取训练参数
      const detail = await DataService.getModelDetail(model.model_id);
      
      // 从详情中提取训练参数
      const trainingInfo = (detail as any).training_info || {};
      const trainingDataPeriod = trainingInfo.training_data_period || {};
      const stockCodes = trainingInfo.stock_codes || [];
      
      if (!stockCodes || stockCodes.length === 0) {
        alert('无法获取模型的训练股票代码，无法重建任务');
        return;
      }

      if (!trainingDataPeriod.start || !trainingDataPeriod.end) {
        alert('无法获取模型的训练日期范围，无法重建任务');
        return;
      }

      // 创建新的训练任务，使用相同的参数
      // 清理模型名称，移除不允许的字符
      const timestamp = new Date().toLocaleString('zh-CN', { 
        month: '2-digit', 
        day: '2-digit', 
        hour: '2-digit', 
        minute: '2-digit' 
      }).replace(/\//g, '-').replace(/:/g, '-'); // 替换斜杠和冒号为横线
      
      const result = await DataService.createModel({
        model_name: `${model.model_name}_重建_${timestamp}`,
        model_type: model.model_type,
        stock_codes: stockCodes,
        start_date: trainingDataPeriod.start.split('T')[0], // 只取日期部分
        end_date: trainingDataPeriod.end.split('T')[0],
        description: `重建自模型: ${model.model_name}`,
        hyperparameters: trainingInfo.hyperparameters || {},
        parent_model_id: model.model_id, // 标记为原模型的子版本
      });
      
      console.log('模型重建任务创建成功:', result);
      alert('模型重建任务已创建，正在后台训练中');
      await loadModels();
    } catch (error) {
      console.error('重建模型失败:', error);
      alert('重建模型失败，请稍后重试');
    } finally {
      setRetrainingModelId(null);
    }
  };

  // 删除模型
  const handleDeleteModel = async (model: Model) => {
    if (!confirm(`确定要删除模型 "${model.model_name}" 吗？此操作不可恢复。`)) {
      return;
    }

    // 不能删除正在训练中的模型
    if (model.status === 'training') {
      alert('无法删除正在训练中的模型，请等待训练完成或取消训练');
      return;
    }

    setDeletingModelId(model.model_id);
    try {
      await DataService.deleteModel(model.model_id);
      alert('模型删除成功');
      await loadModels();
    } catch (error: any) {
      console.error('删除模型失败:', error);
      alert(error?.message || '删除模型失败，请稍后重试');
    } finally {
      setDeletingModelId(null);
    }
  };

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
                <TableColumn>版本</TableColumn>
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
                    <TableCell>{model.version}</TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1">
                        <Chip
                          size="sm"
                          color={getStatusColor(model.status)}
                          variant="flat"
                        >
                          {getStatusText(model.status)}
                        </Chip>
                        {model.status === 'training' && model.training_progress !== undefined && (
                          <span className="text-xs text-default-500">
                            进度: {model.training_progress.toFixed(0)}%
                          </span>
                        )}
                        {model.training_stage && (
                          <span className="text-xs text-default-400">
                            {model.training_stage}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {new Date(model.created_at).toLocaleDateString('zh-CN')}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="light"
                          onPress={async () => {
                            setLoadingModelId(model.model_id);
                            setDetailLoading(true);
                            try {
                              const detail = await DataService.getModelDetail(model.model_id);
                              setSelectedModel(detail);
                              onDetailOpen();
                            } catch (error) {
                              console.error('加载模型详情失败:', error);
                              alert('加载模型详情失败，请稍后重试');
                            } finally {
                              setDetailLoading(false);
                              setLoadingModelId(null);
                            }
                          }}
                          isLoading={detailLoading && loadingModelId === model.model_id}
                        >
                          查看
                        </Button>
                        <Button
                          size="sm"
                          variant="light"
                          color="primary"
                          startContent={<RefreshCw className="w-4 h-4" />}
                          onPress={() => handleRetrainModel(model)}
                          isLoading={retrainingModelId === model.model_id}
                          isDisabled={model.status === 'training'}
                        >
                          重建
                        </Button>
                        <Button
                          size="sm"
                          variant="light"
                          color="danger"
                          startContent={<Trash2 className="w-4 h-4" />}
                          onPress={() => handleDeleteModel(model)}
                          isLoading={deletingModelId === model.model_id}
                          isDisabled={model.status === 'training'}
                        >
                          删除
                        </Button>
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
                description="选择要训练的模型类型"
              >
                <SelectItem key="random_forest" description="传统机器学习模型">随机森林</SelectItem>
                <SelectItem key="linear_regression" description="传统机器学习模型">线性回归</SelectItem>
                <SelectItem key="xgboost" description="传统机器学习模型">XGBoost</SelectItem>
                <SelectItem key="lightgbm" description="传统机器学习模型">LightGBM</SelectItem>
                <SelectItem key="lstm" description="深度学习模型">LSTM</SelectItem>
                <SelectItem key="transformer" description="深度学习模型">Transformer</SelectItem>
                <SelectItem key="timesnet" description="深度学习模型">TimesNet</SelectItem>
                <SelectItem key="patchtst" description="深度学习模型">PatchTST</SelectItem>
                <SelectItem key="informer" description="深度学习模型">Informer</SelectItem>
              </Select>

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

              <div>
                <label className="text-sm font-medium text-foreground mb-2 block">
                  训练股票选择 <span className="text-danger">*</span>
                </label>
                <StockSelector
                  value={formData.stock_codes}
                  onChange={(stocks) => {
                    setFormData((prev) => ({ ...prev, stock_codes: stocks }));
                    if (errors.stock_codes) {
                      setErrors((prev) => ({ ...prev, stock_codes: '' }));
                    }
                  }}
                  placeholder="搜索股票代码或名称"
                />
                {errors.stock_codes && (
                  <p className="text-danger text-sm mt-1">{errors.stock_codes}</p>
                )}
              </div>

              <Input
                label="模型描述（可选）"
                placeholder="请输入模型描述"
                value={formData.description}
                onValueChange={(value) =>
                  setFormData((prev) => ({ ...prev, description: value }))
                }
              />

              {errors.submit && (
                <div className="text-danger text-sm">{errors.submit}</div>
              )}
            </div>
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={onClose}>
              取消
            </Button>
            <Button
              color="primary"
              onPress={handleCreateModel}
              isLoading={creating}
            >
              创建模型
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* 模型详情对话框 */}
      <Modal isOpen={isDetailOpen} onClose={onDetailClose} size="3xl" scrollBehavior="inside">
        <ModalContent>
          <ModalHeader>
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5" />
              <span>模型详情</span>
            </div>
          </ModalHeader>
          <ModalBody>
            {selectedModel ? (
              <div className="space-y-6">
                {/* 基本信息 */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">基本信息</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-default-500">模型名称</p>
                      <p className="text-base font-medium">{selectedModel.model_name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">模型类型</p>
                      <p className="text-base font-medium">{selectedModel.model_type}</p>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">版本</p>
                      <p className="text-base font-medium">{selectedModel.version}</p>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">状态</p>
                      <Chip
                        size="sm"
                        color={
                          selectedModel.status === 'ready' || selectedModel.status === 'active' || selectedModel.status === 'deployed'
                            ? 'success'
                            : selectedModel.status === 'training'
                            ? 'warning'
                            : selectedModel.status === 'failed'
                            ? 'danger'
                            : 'default'
                        }
                        variant="flat"
                      >
                        {selectedModel.status === 'ready' ? '就绪' :
                         selectedModel.status === 'active' ? '活跃' :
                         selectedModel.status === 'deployed' ? '已部署' :
                         selectedModel.status === 'training' ? '训练中' :
                         selectedModel.status === 'failed' ? '失败' :
                         selectedModel.status}
                      </Chip>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">准确率</p>
                      <p className="text-base font-medium">{(selectedModel.accuracy * 100).toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-default-500">创建时间</p>
                      <p className="text-base font-medium">
                        {new Date(selectedModel.created_at).toLocaleString('zh-CN')}
                      </p>
                    </div>
                  </div>
                </div>

                {/* 性能指标 */}
                {selectedModel.performance_metrics && (
                  <div>
                    <h3 className="text-lg font-semibold mb-3">性能指标</h3>
                    <div className="grid grid-cols-3 gap-4">
                      {selectedModel.performance_metrics.accuracy !== undefined && (
                        <div>
                          <p className="text-sm text-default-500">准确率</p>
                          <p className="text-base font-medium">
                            {typeof selectedModel.performance_metrics.accuracy === 'number' 
                              ? (selectedModel.performance_metrics.accuracy * 100).toFixed(2) + '%'
                              : String(selectedModel.performance_metrics.accuracy)}
                          </p>
                        </div>
                      )}
                      {(selectedModel.performance_metrics as any).rmse !== undefined && (
                        <div>
                          <p className="text-sm text-default-500">RMSE</p>
                          <p className="text-base font-medium">
                            {typeof (selectedModel.performance_metrics as any).rmse === 'number'
                              ? (selectedModel.performance_metrics as any).rmse.toFixed(4)
                              : String((selectedModel.performance_metrics as any).rmse)}
                          </p>
                        </div>
                      )}
                      {(selectedModel.performance_metrics as any).mae !== undefined && (
                        <div>
                          <p className="text-sm text-default-500">MAE</p>
                          <p className="text-base font-medium">
                            {typeof (selectedModel.performance_metrics as any).mae === 'number'
                              ? (selectedModel.performance_metrics as any).mae.toFixed(4)
                              : String((selectedModel.performance_metrics as any).mae)}
                          </p>
                        </div>
                      )}
                      {selectedModel.performance_metrics.sharpe_ratio !== undefined && (
                        <div>
                          <p className="text-sm text-default-500">夏普比率</p>
                          <p className="text-base font-medium">
                            {typeof selectedModel.performance_metrics.sharpe_ratio === 'number'
                              ? selectedModel.performance_metrics.sharpe_ratio.toFixed(4)
                              : String(selectedModel.performance_metrics.sharpe_ratio)}
                          </p>
                        </div>
                      )}
                      {selectedModel.performance_metrics.max_drawdown !== undefined && (
                        <div>
                          <p className="text-sm text-default-500">最大回撤</p>
                          <p className="text-base font-medium">
                            {typeof selectedModel.performance_metrics.max_drawdown === 'number'
                              ? (selectedModel.performance_metrics.max_drawdown * 100).toFixed(2) + '%'
                              : String(selectedModel.performance_metrics.max_drawdown)}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 描述 */}
                {selectedModel.description && (
                  <div>
                    <h3 className="text-lg font-semibold mb-3">描述</h3>
                    <p className="text-base text-default-600">{selectedModel.description}</p>
                  </div>
                )}
              </div>
            ) : (
              <LoadingSpinner />
            )}
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={onDetailClose}>
              关闭
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}

