/**
 * 任务创建页面
 * 
 * 提供任务创建表单，包括：
 * - 股票选择器
 * - 模型选择
 * - 参数配置
 * - 预测设置
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  TextField,
  Select,
  MenuItem,
  Switch,
  Slider,
  Chip,
  Divider,
  Box,
  Typography,
  FormControl,
  InputLabel,
  FormHelperText,
  IconButton,
} from '@mui/material';
import {
  ArrowLeft,
  Settings,
  Target,
  TrendingUp,
  Shield,
  Info,
  Activity,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useDataStore } from '../../../stores/useDataStore';
import { useTaskStore } from '../../../stores/useTaskStore';
import { TaskService, CreateTaskRequest } from '../../../services/taskService';
import { DataService } from '../../../services/dataService';
import { StockSelector } from '../../../components/tasks/StockSelector';
import { LoadingSpinner } from '../../../components/common/LoadingSpinner';
import { StrategyConfigForm, StrategyParameter } from '../../../components/backtest/StrategyConfigForm';
import { StrategyConfigService, StrategyConfig } from '../../../services/strategyConfigService';

export default function CreateTaskPage() {
  const router = useRouter();
  const { models, selectedModel, setModels, setSelectedModel } = useDataStore();
  const { setCreating } = useTaskStore();

  const [loading, setLoading] = useState(false);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [taskType, setTaskType] = useState<'prediction' | 'backtest'>('prediction');
  const [availableStrategies, setAvailableStrategies] = useState<Array<{key: string; name: string; description: string; parameters?: Record<string, StrategyParameter>}>>([]);
  const [strategyConfig, setStrategyConfig] = useState<Record<string, any>>({});
  const [savedConfigs, setSavedConfigs] = useState<StrategyConfig[]>([]);
  const [loadingConfigs, setLoadingConfigs] = useState(false);
  const [configFormKey, setConfigFormKey] = useState(0);
  const [formData, setFormData] = useState({
    task_name: '',
    description: '',
    model_id: '',
    horizon: 'short_term' as 'intraday' | 'short_term' | 'medium_term',
    confidence_level: 95,
    risk_assessment: true,
    // 回测配置
    strategy_name: 'moving_average',
    start_date: '',
    end_date: '',
    initial_cash: 100000,
    commission_rate: 0.0003,
    slippage_rate: 0.0001,
    enable_performance_profiling: false,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // 加载模型列表
  useEffect(() => {
    const loadModels = async () => {
      try {
        const result = await DataService.getModels();
        setModels(result.models);
        if (result.models.length > 0 && !selectedModel) {
          setSelectedModel(result.models[0]);
          setFormData(prev => ({ ...prev, model_id: result.models[0].model_id }));
        }
      } catch (error) {
        console.error('加载模型列表失败');
      }
    };

    loadModels();
  }, [setModels, setSelectedModel, selectedModel]);

  // 加载可用策略列表
  useEffect(() => {
    const loadStrategies = async () => {
      if (taskType === 'backtest') {
        try {
          const response = await fetch('/api/v1/backtest/strategies');
          const data = await response.json();
          if (data.success && data.data) {
            setAvailableStrategies(data.data);
            // 如果当前策略不在列表中，设置为第一个策略
            if (data.data.length > 0 && !data.data.find((s: any) => s.key === formData.strategy_name)) {
              updateFormData('strategy_name', data.data[0].key);
            }
          }
        } catch (error) {
          console.error('加载策略列表失败:', error);
        }
      }
    };

    loadStrategies();
  }, [taskType]);

  // 加载已保存的配置列表
  useEffect(() => {
    const loadSavedConfigs = async () => {
      if (taskType === 'backtest' && formData.strategy_name) {
        setLoadingConfigs(true);
        try {
          const response = await StrategyConfigService.getConfigs(formData.strategy_name);
          setSavedConfigs(response.configs);
        } catch (error) {
          console.error('加载已保存配置失败:', error);
        } finally {
          setLoadingConfigs(false);
        }
      } else {
        setSavedConfigs([]);
      }
    };

    loadSavedConfigs();
  }, [taskType, formData.strategy_name]);

  // 加载配置详情
  const handleLoadConfig = async (configId: string) => {
    try {
      const config = await StrategyConfigService.getConfig(configId);
      setStrategyConfig(config.parameters);
      // 通过更新key强制重新渲染组件，传入新的values
      setConfigFormKey(prev => prev + 1);
    } catch (error) {
      console.error('加载配置失败:', error);
    }
  };

  // 表单验证
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.task_name.trim()) {
      newErrors.task_name = '请输入任务名称';
    }
    
    if (selectedStocks.length === 0) {
      newErrors.stock_codes = '请至少选择一只股票';
    }
    
    if (taskType === 'prediction') {
      if (!formData.model_id) {
        newErrors.model_id = '请选择预测模型';
      }
    } else {  // backtest
      if (!formData.start_date) {
        newErrors.start_date = '请选择回测开始日期';
      }
      if (!formData.end_date) {
        newErrors.end_date = '请选择回测结束日期';
      }
      if (formData.start_date && formData.end_date && formData.start_date >= formData.end_date) {
        newErrors.end_date = '结束日期必须晚于开始日期';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // 提交表单
  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setCreating(true);

    try {
      const request: CreateTaskRequest = {
        task_name: formData.task_name,
        task_type: taskType,
        stock_codes: selectedStocks,
        ...(taskType === 'prediction' ? {
          model_id: formData.model_id,
          prediction_config: {
            horizon: formData.horizon,
            confidence_level: formData.confidence_level / 100,
            risk_assessment: formData.risk_assessment,
          },
        } : {
          backtest_config: {
            strategy_name: formData.strategy_name,
            start_date: formData.start_date,
            end_date: formData.end_date,
            initial_cash: formData.initial_cash,
            commission_rate: formData.commission_rate,
            slippage_rate: formData.slippage_rate,
            strategy_config: strategyConfig,
            enable_performance_profiling: formData.enable_performance_profiling,
          },
        }),
      };

      const task = await TaskService.createTask(request);
      console.log('任务创建成功');
      router.push(`/tasks/${task.task_id}`);
    } catch (error) {
      console.error('创建任务失败:', error);
    } finally {
      setLoading(false);
      setCreating(false);
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  // 更新表单数据
  const updateFormData = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // 清除对应字段的错误
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <IconButton onClick={handleBack}>
          <ArrowLeft size={20} />
        </IconButton>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
            创建任务
          </Typography>
          <Typography variant="body2" color="text.secondary">
            配置股票预测或回测任务参数
          </Typography>
        </Box>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 3 }}>
        {/* 主要配置区域 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* 基本信息 */}
          <Card>
            <CardHeader
              avatar={<Settings size={20} />}
              title="基本信息"
            />
            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth required>
                <InputLabel>任务类型</InputLabel>
                <Select
                  value={taskType}
                  label="任务类型"
                  onChange={(e) => setTaskType(e.target.value as 'prediction' | 'backtest')}
                >
                  <MenuItem value="prediction">预测任务</MenuItem>
                  <MenuItem value="backtest">回测任务</MenuItem>
                </Select>
              </FormControl>
              
              <TextField
                label="任务名称"
                placeholder="请输入任务名称"
                value={formData.task_name}
                onChange={(e) => updateFormData('task_name', e.target.value)}
                error={!!errors.task_name}
                helperText={errors.task_name}
                required
                fullWidth
              />
              
              <TextField
                label="任务描述"
                placeholder="请输入任务描述（可选）"
                value={formData.description}
                onChange={(e) => updateFormData('description', e.target.value)}
                multiline
                rows={3}
                fullWidth
              />
            </CardContent>
          </Card>

          {/* 股票选择 */}
          <Card>
            <CardHeader
              avatar={<Target size={20} />}
              title="股票选择"
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <StockSelector
                  value={selectedStocks}
                  onChange={(stocks) => {
                    setSelectedStocks(stocks);
                    if (errors.stock_codes) {
                      setErrors(prev => ({ ...prev, stock_codes: '' }));
                    }
                  }}
                  maxCount={20}
                  placeholder="搜索股票代码或名称"
                />
                {errors.stock_codes && (
                  <FormHelperText error>{errors.stock_codes}</FormHelperText>
                )}
              </Box>
            </CardContent>
          </Card>

          {/* 模型和参数 / 回测配置 */}
          {taskType === 'prediction' ? (
            <Card>
              <CardHeader
                avatar={<TrendingUp size={20} />}
                title="模型和参数"
              />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <FormControl fullWidth required error={!!errors.model_id}>
                  <InputLabel>预测模型</InputLabel>
                  <Select
                    value={formData.model_id}
                    label="预测模型"
                    onChange={(e) => {
                      const modelId = e.target.value;
                      updateFormData('model_id', modelId);
                      const model = models.find(m => m.model_id === modelId);
                      setSelectedModel(model || null);
                    }}
                  >
                    {models
                      .filter(model => model.status === 'ready' || model.status === 'active')
                      .map(model => (
                        <MenuItem key={model.model_id} value={model.model_id}>
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {model.model_name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              准确率: {(model.accuracy * 100).toFixed(1)}% | 类型: {model.model_type}
                            </Typography>
                          </Box>
                        </MenuItem>
                      ))}
                  </Select>
                  {errors.model_id && <FormHelperText>{errors.model_id}</FormHelperText>}
                </FormControl>

                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                  <FormControl fullWidth>
                    <InputLabel>预测时间维度</InputLabel>
                    <Select
                      value={formData.horizon}
                      label="预测时间维度"
                      onChange={(e) => updateFormData('horizon', e.target.value)}
                    >
                      <MenuItem value="intraday">日内 (1-4小时)</MenuItem>
                      <MenuItem value="short_term">短期 (1-5天)</MenuItem>
                      <MenuItem value="medium_term">中期 (1-4周)</MenuItem>
                    </Select>
                  </FormControl>

                  <Box>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      置信水平: {formData.confidence_level}%
                    </Typography>
                    <Slider
                      value={formData.confidence_level}
                      onChange={(e, value) => updateFormData('confidence_level', value)}
                      min={80}
                      max={99}
                      step={1}
                      marks
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Shield size={20} color="#1976d2" />
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        风险评估
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        启用风险评估和VaR计算
                      </Typography>
                    </Box>
                  </Box>
                  <Switch
                    checked={formData.risk_assessment}
                    onChange={(e) => updateFormData('risk_assessment', e.target.checked)}
                  />
                </Box>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader
                avatar={<Activity size={20} />}
                title="回测配置"
              />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <FormControl fullWidth required disabled={availableStrategies.length === 0}>
                  <InputLabel>交易策略</InputLabel>
                  <Select
                    value={formData.strategy_name}
                    label="交易策略"
                    onChange={(e) => updateFormData('strategy_name', e.target.value)}
                  >
                    {availableStrategies.map(strategy => (
                      <MenuItem key={strategy.key} value={strategy.key}>
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {strategy.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {strategy.description}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {/* 策略配置表单 */}
                {formData.strategy_name && (() => {
                  const selectedStrategy = availableStrategies.find(s => s.key === formData.strategy_name);
                  if (selectedStrategy && selectedStrategy.parameters) {
                    return (
                      <StrategyConfigForm
                        key={`${formData.strategy_name}-${configFormKey}`}
                        strategyName={formData.strategy_name}
                        parameters={selectedStrategy.parameters}
                        values={configFormKey > 0 && Object.keys(strategyConfig).length > 0 ? strategyConfig : undefined}
                        onChange={(newConfig) => {
                          setStrategyConfig(newConfig);
                        }}
                        onLoadConfig={handleLoadConfig}
                        savedConfigs={savedConfigs.map(c => ({
                          config_id: c.config_id,
                          config_name: c.config_name,
                          created_at: c.created_at,
                        }))}
                        loading={loadingConfigs}
                      />
                    );
                  }
                  return null;
                })()}
                
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                  <TextField
                    type="date"
                    label="回测开始日期"
                    value={formData.start_date}
                    onChange={(e) => updateFormData('start_date', e.target.value)}
                    error={!!errors.start_date}
                    helperText={errors.start_date}
                    required
                    fullWidth
                    InputLabelProps={{ shrink: true }}
                  />
                  
                  <TextField
                    type="date"
                    label="回测结束日期"
                    value={formData.end_date}
                    onChange={(e) => updateFormData('end_date', e.target.value)}
                    error={!!errors.end_date}
                    helperText={errors.end_date}
                    required
                    fullWidth
                    InputLabelProps={{ shrink: true }}
                  />
                </Box>
                
                <TextField
                  type="number"
                  label="初始资金"
                  value={formData.initial_cash}
                  onChange={(e) => updateFormData('initial_cash', parseFloat(e.target.value) || 100000)}
                  inputProps={{ min: 1000, step: 1000 }}
                  fullWidth
                />
                
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                  <TextField
                    type="number"
                    label="手续费率"
                    value={formData.commission_rate}
                    onChange={(e) => updateFormData('commission_rate', parseFloat(e.target.value) || 0.0003)}
                    inputProps={{ min: 0, max: 0.01, step: 0.0001 }}
                    fullWidth
                  />
                  
                  <TextField
                    type="number"
                    label="滑点率"
                    value={formData.slippage_rate}
                    onChange={(e) => updateFormData('slippage_rate', parseFloat(e.target.value) || 0.0001)}
                    inputProps={{ min: 0, max: 0.01, step: 0.0001 }}
                    fullWidth
                  />
                </Box>

                <Divider />

                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Activity size={20} color="#1976d2" />
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        性能监控
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        启用后将收集回测各阶段耗时/CPU/内存，并在任务详情页「性能分析」Tab 展示
                      </Typography>
                    </Box>
                  </Box>
                  <Switch
                    checked={formData.enable_performance_profiling}
                    onChange={(e) => updateFormData('enable_performance_profiling', e.target.checked)}
                  />
                </Box>
              </CardContent>
            </Card>
          )}

          {/* 提交按钮 */}
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              color="primary"
              size="large"
              onClick={handleSubmit}
              disabled={loading}
              fullWidth
            >
              {loading ? '创建中...' : '创建任务'}
            </Button>
            <Button
              variant="outlined"
              size="large"
              onClick={handleBack}
            >
              取消
            </Button>
          </Box>
        </Box>

        {/* 侧边栏信息 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* 模型信息 */}
          {selectedModel && (
            <Card>
              <CardHeader title="模型信息" />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    模型名称
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {selectedModel.model_name}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    模型类型
                  </Typography>
                  <Chip label={selectedModel.model_type} size="small" color="secondary" sx={{ mt: 0.5 }} />
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    准确率
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {(selectedModel.accuracy * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    版本
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {selectedModel.version}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    创建时间
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {new Date(selectedModel.created_at).toLocaleDateString()}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* 预测说明 */}
          <Card>
            <CardHeader
              avatar={<Info size={20} />}
              title="预测说明"
            />
            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  时间维度说明:
                </Typography>
                <Box component="ul" sx={{ m: 0, pl: 2, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <Typography component="li" variant="body2" color="text.secondary">
                    日内: 适合短线交易
                  </Typography>
                  <Typography component="li" variant="body2" color="text.secondary">
                    短期: 适合波段操作
                  </Typography>
                  <Typography component="li" variant="body2" color="text.secondary">
                    中期: 适合趋势投资
                  </Typography>
                </Box>
              </Box>
              <Divider />
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  置信水平:
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  置信水平越高，预测区间越宽，但可靠性越高
                </Typography>
              </Box>
              <Divider />
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  风险评估:
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  包含VaR计算和波动率分析，帮助评估投资风险
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
}
