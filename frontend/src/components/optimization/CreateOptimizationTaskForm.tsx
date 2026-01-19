/**
 * 创建超参优化任务表单
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Button,
  TextField,
  Select,
  MenuItem,
  Switch,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Chip,
  Box,
  Typography,
  FormControl,
  InputLabel,
  FormHelperText,
} from '@mui/material';
import { Save, Loader2 } from 'lucide-react';
import { OptimizationService, CreateOptimizationTaskRequest, ParamSpaceConfig } from '../../services/optimizationService';
import { StockSelector } from '../tasks/StockSelector';
import { apiRequest } from '../../services/api';

interface CreateOptimizationTaskFormProps {
  onTaskCreated: () => void;
}

interface Strategy {
  key: string;
  name: string;
  description: string;
  parameters?: Record<string, any>;
}

export default function CreateOptimizationTaskForm({ onTaskCreated }: CreateOptimizationTaskFormProps) {
  const [loading, setLoading] = useState(false);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [formData, setFormData] = useState({
    task_name: '',
    strategy_name: '',
    start_date: '',
    end_date: '',
    objective_metric: 'sharpe' as string | string[],
    direction: 'maximize' as 'maximize' | 'minimize',
    n_trials: 50,
    optimization_method: 'tpe',
    timeout: undefined as number | undefined,
  });
  const [paramSpace, setParamSpace] = useState<Record<string, ParamSpaceConfig>>({});
  const [objectiveWeights, setObjectiveWeights] = useState<Record<string, number>>({
    sharpe_ratio: 0.6,
    total_return: 0.4,
  });

  // 加载策略列表
  useEffect(() => {
    const loadStrategies = async () => {
      try {
        const strategiesList = await apiRequest.get<Strategy[]>('/backtest/strategies');
        console.log('加载到策略数量:', strategiesList?.length || 0);
        if (strategiesList && Array.isArray(strategiesList)) {
          setStrategies(strategiesList);
        } else {
          console.warn('策略列表数据格式不正确:', strategiesList);
          setStrategies([]);
        }
      } catch (error) {
        console.error('加载策略列表失败:', error);
        setStrategies([]);
      }
    };
    loadStrategies();
  }, []);

  // 当选择策略时，加载默认参数空间
  useEffect(() => {
    if (formData.strategy_name) {
      const strategy = strategies.find(s => s.key === formData.strategy_name);
      if (strategy?.parameters) {
        const defaultSpace: Record<string, ParamSpaceConfig> = {};
        Object.entries(strategy.parameters).forEach(([key, param]: [string, any]) => {
          // 只处理可以优化的参数类型：int 和 float
          if (param.type === 'int' && param.min !== undefined && param.max !== undefined) {
            defaultSpace[key] = {
              type: 'int',
              low: param.min,
              high: param.max,
              default: param.default,
              enabled: true,
            };
          } else if (param.type === 'float' && param.min !== undefined && param.max !== undefined) {
            defaultSpace[key] = {
              type: 'float',
              low: param.min,
              high: param.max,
              default: param.default,
              enabled: true,
            };
          } else if (param.type === 'categorical' && param.options && Array.isArray(param.options)) {
            // 处理分类参数
            defaultSpace[key] = {
              type: 'categorical',
              choices: param.options,
              default: param.default,
              enabled: true,
            };
          }
        });
        console.log('加载的参数空间:', defaultSpace);
        setParamSpace(defaultSpace);
      } else {
        // 如果策略没有参数定义，清空参数空间
        setParamSpace({});
      }
    } else {
      // 未选择策略时，清空参数空间
      setParamSpace({});
    }
  }, [formData.strategy_name, strategies]);

  const handleSubmit = async () => {
    if (!formData.task_name || !formData.strategy_name || selectedStocks.length === 0 || !formData.start_date || !formData.end_date) {
      alert('请填写所有必填字段');
      return;
    }

    setLoading(true);
    try {
      // 转换日期格式：从 YYYY-MM-DD 转换为 ISO 格式（带时区）
      // 注意：Date 构造函数会将本地时间转换为 UTC，需要手动处理
      const startDate = formData.start_date 
        ? `${formData.start_date}T00:00:00` 
        : '';
      const endDate = formData.end_date 
        ? `${formData.end_date}T23:59:59` 
        : '';
      
      const request: CreateOptimizationTaskRequest = {
        task_name: formData.task_name,
        strategy_name: formData.strategy_name,
        stock_codes: selectedStocks,
        start_date: startDate,
        end_date: endDate,
        param_space: paramSpace,
        objective_config: {
          objective_metric: formData.objective_metric,
          direction: formData.direction,
          objective_weights: formData.objective_metric === 'custom' ? objectiveWeights : undefined,
        },
        n_trials: formData.n_trials,
        optimization_method: formData.optimization_method,
        timeout: formData.timeout,
      };
      
      console.log('创建优化任务请求:', request);

      await OptimizationService.createTask(request);
      onTaskCreated();
      
      // 重置表单
      setFormData({
        task_name: '',
        strategy_name: '',
        start_date: '',
        end_date: '',
        objective_metric: 'sharpe',
        direction: 'maximize',
        n_trials: 50,
        optimization_method: 'tpe',
        timeout: undefined,
      });
      setSelectedStocks([]);
      setParamSpace({});
    } catch (error) {
      console.error('创建优化任务失败:', error);
      alert('创建优化任务失败: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setLoading(false);
    }
  };

  const updateParamSpace = (paramName: string, field: keyof ParamSpaceConfig, value: any) => {
    setParamSpace(prev => ({
      ...prev,
      [paramName]: {
        ...prev[paramName],
        [field]: value,
      },
    }));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box>
        <Typography variant="h5" component="h2" sx={{ fontWeight: 600, mb: 1 }}>
          创建超参优化任务
        </Typography>
        <Typography variant="body2" color="text.secondary">
          配置优化参数，寻找策略的最佳参数组合
        </Typography>
      </Box>

      <Card>
        <CardHeader title="基本信息" />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <TextField
            label="任务名称"
            placeholder="输入任务名称"
            value={formData.task_name}
            onChange={(e) => setFormData(prev => ({ ...prev, task_name: e.target.value }))}
            required
            fullWidth
          />

          <FormControl fullWidth required disabled={strategies.length === 0}>
            <InputLabel>选择策略</InputLabel>
            <Select
              value={formData.strategy_name}
              label="选择策略"
              onChange={(e) => setFormData(prev => ({ ...prev, strategy_name: e.target.value }))}
            >
              {strategies.map((strategy) => (
                <MenuItem key={strategy.key} value={strategy.key}>
                  {strategy.name || strategy.key}
                </MenuItem>
              ))}
            </Select>
            {strategies.length === 0 && (
              <FormHelperText>正在加载策略列表...</FormHelperText>
            )}
          </FormControl>

          <Box>
            <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
              选择股票
            </Typography>
            <StockSelector
              value={selectedStocks}
              onChange={setSelectedStocks}
            />
          </Box>

          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
            <TextField
              type="date"
              label="开始日期"
              value={formData.start_date}
              onChange={(e) => setFormData(prev => ({ ...prev, start_date: e.target.value }))}
              required
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
            <TextField
              type="date"
              label="结束日期"
              value={formData.end_date}
              onChange={(e) => setFormData(prev => ({ ...prev, end_date: e.target.value }))}
              required
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardHeader 
          title="参数空间配置" 
          subheader="配置需要优化的策略参数范围。只有数值型参数（整数、浮点数）和分类参数可以进行优化。"
        />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {!formData.strategy_name ? (
            <Typography variant="body2" color="text.secondary">
              请先选择策略，系统将自动加载该策略的可优化参数
            </Typography>
          ) : Object.entries(paramSpace).length === 0 ? (
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                当前策略没有可优化的参数（需要数值型参数：整数或浮点数）
              </Typography>
              <Typography variant="caption" color="text.secondary">
                提示：某些策略的参数可能是 JSON 对象、字符串或布尔值，这些参数类型不支持自动优化。
                您可以在创建回测任务时手动配置这些参数。
              </Typography>
            </Box>
          ) : (
            Object.entries(paramSpace).map(([paramName, config]) => {
              const strategy = strategies.find(s => s.key === formData.strategy_name);
              const paramInfo = strategy?.parameters?.[paramName];
              const paramDescription = paramInfo?.description || '';
              
              return (
                <Box key={paramName} sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {paramName}
                      </Typography>
                      {paramDescription && (
                        <Typography variant="caption" color="text.secondary">
                          {paramDescription}
                        </Typography>
                      )}
                      {config.default !== undefined && (
                        <Chip 
                          label={`默认值: ${config.default}`} 
                          size="small" 
                          sx={{ mt: 0.5, mr: 1 }}
                        />
                      )}
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2">启用优化</Typography>
                      <Switch
                        checked={config.enabled}
                        onChange={(e) => updateParamSpace(paramName, 'enabled', e.target.checked)}
                      />
                    </Box>
                  </Box>
                  {config.enabled && (
                    <>
                      {config.type === 'categorical' && config.choices ? (
                        <FormControl fullWidth>
                          <InputLabel>可选值</InputLabel>
                          <Select
                            multiple
                            value={config.choices || []}
                            label="可选值"
                            renderValue={(selected) => (
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                {selected.map((value: any) => (
                                  <Chip key={value} label={value} size="small" />
                                ))}
                              </Box>
                            )}
                            disabled
                          >
                            {config.choices.map((choice: any) => (
                              <MenuItem key={choice} value={choice}>
                                {choice}
                              </MenuItem>
                            ))}
                          </Select>
                          <FormHelperText>
                            分类参数：将从以上值中选择
                          </FormHelperText>
                        </FormControl>
                      ) : (
                        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                          <TextField
                            type="number"
                            label="最小值"
                            value={config.low?.toString() || ''}
                            onChange={(e) => updateParamSpace(paramName, 'low', parseFloat(e.target.value))}
                            fullWidth
                            helperText={config.type === 'int' ? '整数' : '浮点数'}
                          />
                          <TextField
                            type="number"
                            label="最大值"
                            value={config.high?.toString() || ''}
                            onChange={(e) => updateParamSpace(paramName, 'high', parseFloat(e.target.value))}
                            fullWidth
                            helperText={config.type === 'int' ? '整数' : '浮点数'}
                          />
                        </Box>
                      )}
                    </>
                  )}
                </Box>
              );
            })
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="优化目标" />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <FormControl fullWidth>
            <InputLabel>目标指标</InputLabel>
            <Select
              value={Array.isArray(formData.objective_metric) ? formData.objective_metric[0] : formData.objective_metric}
              label="目标指标"
              onChange={(e) => {
                setFormData(prev => ({ 
                  ...prev, 
                  objective_metric: e.target.value
                }));
              }}
            >
              <MenuItem value="sharpe">夏普比率 (Sharpe Ratio)</MenuItem>
              <MenuItem value="calmar">卡玛比率 (Calmar Ratio)</MenuItem>
              <MenuItem value="ic">信息系数 (IC)</MenuItem>
              <MenuItem value="custom">自定义组合</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel>优化方向</InputLabel>
            <Select
              value={formData.direction}
              label="优化方向"
              onChange={(e) => {
                const selected = e.target.value as 'maximize' | 'minimize';
                setFormData(prev => ({ ...prev, direction: selected }));
              }}
            >
              <MenuItem value="maximize">最大化</MenuItem>
              <MenuItem value="minimize">最小化</MenuItem>
            </Select>
          </FormControl>

          {formData.objective_metric === 'custom' && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                自定义权重
              </Typography>
              <TextField
                type="number"
                label="夏普比率权重"
                value={objectiveWeights.sharpe_ratio.toString()}
                onChange={(e) => setObjectiveWeights(prev => ({ ...prev, sharpe_ratio: parseFloat(e.target.value) }))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="总收益率权重"
                value={objectiveWeights.total_return.toString()}
                onChange={(e) => setObjectiveWeights(prev => ({ ...prev, total_return: parseFloat(e.target.value) }))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
            </Box>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="优化配置" />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <TextField
            type="number"
            label="试验次数"
            value={formData.n_trials.toString()}
            onChange={(e) => setFormData(prev => ({ ...prev, n_trials: parseInt(e.target.value) || 50 }))}
            inputProps={{ min: 10, max: 200 }}
            fullWidth
          />

          <FormControl fullWidth>
            <InputLabel>优化方法</InputLabel>
            <Select
              value={formData.optimization_method}
              label="优化方法"
              onChange={(e) => setFormData(prev => ({ ...prev, optimization_method: e.target.value }))}
            >
              <MenuItem value="tpe">TPE (Tree-structured Parzen Estimator)</MenuItem>
              <MenuItem value="random">随机搜索</MenuItem>
              <MenuItem value="nsga2">NSGA-II (多目标)</MenuItem>
              <MenuItem value="motpe">MOTPE (多目标)</MenuItem>
            </Select>
          </FormControl>

          <TextField
            type="number"
            label="超时时间（秒，可选）"
            value={formData.timeout?.toString() || ''}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              timeout: e.target.value ? parseInt(e.target.value) : undefined 
            }))}
            placeholder="不设置则无超时限制"
            fullWidth
          />
        </CardContent>
      </Card>

      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={loading}
          startIcon={!loading && <Save size={16} />}
        >
          {loading ? '创建中...' : '创建优化任务'}
        </Button>
      </Box>
    </Box>
  );
}
