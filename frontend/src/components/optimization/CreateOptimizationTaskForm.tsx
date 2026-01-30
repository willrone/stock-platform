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
import {
  OptimizationService,
  CreateOptimizationTaskRequest,
  ParamSpaceConfig,
} from '../../services/optimizationService';
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

export default function CreateOptimizationTaskForm({
  onTaskCreated,
}: CreateOptimizationTaskFormProps) {
  const [loading, setLoading] = useState(false);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [formData, setFormData] = useState({
    task_name: '',
    // optimization_mode: single strategy vs portfolio (ensemble)
    optimization_mode: 'single' as 'single' | 'portfolio',
    // for single mode
    strategy_name: '',
    // for portfolio mode
    portfolio_strategies: [] as string[],
    start_date: '',
    end_date: '',
    objective_metric: 'stability' as string | string[],
    direction: 'maximize' as 'maximize' | 'minimize',
    n_trials: 50,
    optimization_method: 'tpe',
    timeout: undefined as number | undefined,
  });
  const [paramSpace, setParamSpace] = useState<Record<string, ParamSpaceConfig>>({});
  const [objectiveWeights, setObjectiveWeights] = useState<Record<string, number>>({
    sharpe_ratio: 0.4,
    total_return: 0.2,
    win_rate: 0.1,
    profit_factor: 0.1,
    information_ratio: 0.1,
    cost_ratio: 0.1,
  });

  const isMultiObjectiveMethod =
    formData.optimization_method === 'nsga2' || formData.optimization_method === 'motpe';

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

  // 当选择策略/模式时，加载默认参数空间
  useEffect(() => {
    // Portfolio mode: build a flattened param space based on selected sub-strategies.
    if (formData.optimization_mode === 'portfolio') {
      const defaultSpace: Record<string, ParamSpaceConfig> = {
        // integration method for portfolio
        integration_method: {
          type: 'categorical',
          choices: ['weighted_voting'],
          default: 'weighted_voting',
          enabled: true,
        },
      };

      // For each chosen sub-strategy: enable switch + weight + its numeric/categorical params.
      for (const key of formData.portfolio_strategies || []) {
        defaultSpace[`use__${key}`] = {
          type: 'categorical',
          choices: [0, 1],
          default: 1,
          enabled: true,
        };
        defaultSpace[`weight__${key}`] = {
          type: 'float',
          low: 0.0,
          high: 1.0,
          default: 0.5,
          enabled: true,
        };

        const st = strategies.find(s => s.key === key);
        const params = st?.parameters || {};
        Object.entries(params).forEach(([pname, p]: [string, any]) => {
          const full = `${key}__${pname}`;
          if (p?.type === 'int' && p.min !== undefined && p.max !== undefined) {
            defaultSpace[full] = {
              type: 'int',
              low: p.min,
              high: p.max,
              default: p.default,
              enabled: true,
            };
          } else if (p?.type === 'float' && p.min !== undefined && p.max !== undefined) {
            defaultSpace[full] = {
              type: 'float',
              low: p.min,
              high: p.max,
              default: p.default,
              enabled: true,
            };
          } else if (p?.type === 'categorical' && Array.isArray(p.options)) {
            defaultSpace[full] = {
              type: 'categorical',
              choices: p.options,
              default: p.default,
              enabled: true,
            };
          }
        });
      }

      setParamSpace(defaultSpace);
      return;
    }

    // Single-strategy mode: same as before
    if (formData.strategy_name) {
      const strategy = strategies.find(s => s.key === formData.strategy_name);
      if (strategy?.parameters) {
        const defaultSpace: Record<string, ParamSpaceConfig> = {};
        Object.entries(strategy.parameters).forEach(([key, param]: [string, any]) => {
          // 只处理可以优化的参数类型：int/float/categorical
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
            defaultSpace[key] = {
              type: 'categorical',
              choices: param.options,
              default: param.default,
              enabled: true,
            };
          }
        });
        setParamSpace(defaultSpace);
      } else {
        setParamSpace({});
      }
      return;
    }

    // 未选择策略时，清空参数空间
    setParamSpace({});
  }, [formData.optimization_mode, formData.strategy_name, formData.portfolio_strategies, strategies]);

  const handleSubmit = async () => {
    const strategyValid =
      formData.optimization_mode === 'single'
        ? Boolean(formData.strategy_name)
        : (formData.portfolio_strategies?.length || 0) > 0;

    if (!formData.task_name || !strategyValid || selectedStocks.length === 0 || !formData.start_date || !formData.end_date) {
      alert('请填写所有必填字段');
      return;
    }

    setLoading(true);
    try {
      // 转换日期格式：从 YYYY-MM-DD 转换为 ISO 格式（带时区）
      // 注意：Date 构造函数会将本地时间转换为 UTC，需要手动处理
      const startDate = formData.start_date ? `${formData.start_date}T00:00:00` : '';
      const endDate = formData.end_date ? `${formData.end_date}T23:59:59` : '';

      // Plan A: 固定 topk_buffer，不把 topk/buffer/max_changes_per_day 作为待优化参数。
      const fixedTradeConfig = {
        trade_mode: 'topk_buffer',
        topk: 10,
        buffer: 20,
        max_changes_per_day: 2,
      };

      const filteredParamSpace: Record<string, ParamSpaceConfig> = { ...paramSpace };
      delete filteredParamSpace.topk;
      delete filteredParamSpace.buffer;
      delete filteredParamSpace.max_changes_per_day;

      const request: CreateOptimizationTaskRequest = {
        task_name: formData.task_name,
        // backend will route portfolio optimization by strategy_name="portfolio"
        strategy_name: formData.optimization_mode === 'portfolio' ? 'portfolio' : formData.strategy_name,
        stock_codes: selectedStocks,
        start_date: startDate,
        end_date: endDate,
        param_space: filteredParamSpace,
        objective_config: {
          objective_metric: formData.objective_metric,
          direction: formData.direction,
          objective_weights: formData.objective_metric === 'custom' ? objectiveWeights : undefined,
        },
        n_trials: formData.n_trials,
        optimization_method: formData.optimization_method,
        timeout: formData.timeout,
        backtest_config: formData.optimization_mode === 'portfolio' ? fixedTradeConfig : undefined,
      };

      console.log('创建优化任务请求:', request);

      await OptimizationService.createTask(request);
      onTaskCreated();

      // 重置表单
      setFormData({
        task_name: '',
        optimization_mode: 'single',
        strategy_name: '',
        portfolio_strategies: [],
        start_date: '',
        end_date: '',
        objective_metric: 'stability',
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
            onChange={e => setFormData(prev => ({ ...prev, task_name: e.target.value }))}
            required
            fullWidth
          />

          <FormControl fullWidth required>
            <InputLabel>优化类型</InputLabel>
            <Select
              value={formData.optimization_mode}
              label="优化类型"
              onChange={e =>
                setFormData(prev => ({
                  ...prev,
                  optimization_mode: e.target.value as 'single' | 'portfolio',
                  // reset selections when switching
                  strategy_name: '',
                  portfolio_strategies: [],
                }))
              }
            >
              <MenuItem value="single">单策略优化</MenuItem>
              <MenuItem value="portfolio">组合策略优化（自由搭配）</MenuItem>
            </Select>
            <FormHelperText>
              单策略：像以前一样选一个策略优化；组合策略：选择多个子策略并一起优化权重与子策略参数（交易执行固定 topk_buffer）。
            </FormHelperText>
          </FormControl>

          {formData.optimization_mode === 'single' ? (
            <FormControl fullWidth required disabled={strategies.length === 0}>
              <InputLabel>选择策略</InputLabel>
              <Select
                value={formData.strategy_name}
                label="选择策略"
                onChange={e => setFormData(prev => ({ ...prev, strategy_name: e.target.value }))}
              >
                {strategies.map(strategy => (
                  <MenuItem key={strategy.key} value={strategy.key}>
                    {strategy.name || strategy.key}
                  </MenuItem>
                ))}
              </Select>
              {strategies.length === 0 && <FormHelperText>正在加载策略列表...</FormHelperText>}
            </FormControl>
          ) : (
            <FormControl fullWidth required disabled={strategies.length === 0}>
              <InputLabel>选择子策略（可多选）</InputLabel>
              <Select
                multiple
                value={formData.portfolio_strategies}
                label="选择子策略（可多选）"
                onChange={e =>
                  setFormData(prev => ({
                    ...prev,
                    portfolio_strategies: (e.target.value as string[]) || [],
                  }))
                }
                renderValue={selected => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map(v => {
                      const st = strategies.find(s => s.key === v);
                      return <Chip key={v} label={st?.name || v} size="small" />;
                    })}
                  </Box>
                )}
              >
                {strategies.map(strategy => (
                  <MenuItem key={strategy.key} value={strategy.key}>
                    {strategy.name || strategy.key}
                  </MenuItem>
                ))}
              </Select>
              {strategies.length === 0 && <FormHelperText>正在加载策略列表...</FormHelperText>}
              {formData.portfolio_strategies.length === 0 && (
                <FormHelperText>请至少选择 1 个子策略</FormHelperText>
              )}
            </FormControl>
          )}

          <Box>
            <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
              选择股票
            </Typography>
            <StockSelector value={selectedStocks} onChange={setSelectedStocks} />
          </Box>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
              gap: 2,
            }}
          >
            <TextField
              type="date"
              label="开始日期"
              value={formData.start_date}
              onChange={e => setFormData(prev => ({ ...prev, start_date: e.target.value }))}
              required
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
            <TextField
              type="date"
              label="结束日期"
              value={formData.end_date}
              onChange={e => setFormData(prev => ({ ...prev, end_date: e.target.value }))}
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
          {formData.optimization_mode === 'single' && !formData.strategy_name ? (
            <Typography variant="body2" color="text.secondary">
              请先选择策略，系统将自动加载该策略的可优化参数
            </Typography>
          ) : formData.optimization_mode === 'portfolio' && (formData.portfolio_strategies?.length || 0) === 0 ? (
            <Typography variant="body2" color="text.secondary">
              请先选择至少 1 个子策略，系统将自动展开组合策略的参数空间
            </Typography>
          ) : Object.entries(paramSpace).length === 0 ? (
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                当前配置没有可优化的参数（需要数值型参数：整数/浮点数或分类参数）
              </Typography>
              <Typography variant="caption" color="text.secondary">
                提示：某些策略的参数可能是 JSON 对象、字符串或布尔值，这些参数类型不支持自动优化。
                组合策略模式目前只会自动展开 int/float/categorical。
              </Typography>
            </Box>
          ) : (
            Object.entries(paramSpace).map(([paramName, config]) => {
              // single mode: show per-param description from selected strategy
              const baseStrategy = strategies.find(s => s.key === formData.strategy_name);
              const paramInfo = baseStrategy?.parameters?.[paramName];
              const paramDescription = paramInfo?.description || '';

              return (
                <Box
                  key={paramName}
                  sx={{
                    p: 2,
                    border: 1,
                    borderColor: 'divider',
                    borderRadius: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                  }}
                >
                  <Box
                    sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                  >
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
                        onChange={e => updateParamSpace(paramName, 'enabled', e.target.checked)}
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
                            renderValue={selected => (
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
                          <FormHelperText>分类参数：将从以上值中选择</FormHelperText>
                        </FormControl>
                      ) : (
                        <Box
                          sx={{
                            display: 'grid',
                            gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
                            gap: 2,
                          }}
                        >
                          <TextField
                            type="number"
                            label="最小值"
                            value={config.low?.toString() || ''}
                            onChange={e =>
                              updateParamSpace(paramName, 'low', parseFloat(e.target.value))
                            }
                            fullWidth
                            helperText={config.type === 'int' ? '整数' : '浮点数'}
                          />
                          <TextField
                            type="number"
                            label="最大值"
                            value={config.high?.toString() || ''}
                            onChange={e =>
                              updateParamSpace(paramName, 'high', parseFloat(e.target.value))
                            }
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
              multiple={isMultiObjectiveMethod}
              value={
                isMultiObjectiveMethod
                  ? Array.isArray(formData.objective_metric)
                    ? formData.objective_metric
                    : [formData.objective_metric]
                  : Array.isArray(formData.objective_metric)
                    ? formData.objective_metric[0]
                    : formData.objective_metric
              }
              label="目标指标"
              onChange={e => {
                const value = e.target.value;
                setFormData(prev => ({
                  ...prev,
                  objective_metric: isMultiObjectiveMethod
                    ? Array.isArray(value)
                      ? value
                      : [value]
                    : Array.isArray(value)
                      ? value[0]
                      : value,
                }));
              }}
              renderValue={selected =>
                isMultiObjectiveMethod && Array.isArray(selected) ? (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value: any) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                ) : (
                  (selected as any)
                )
              }
            >
              <MenuItem value="sharpe">夏普比率 (Sharpe Ratio)</MenuItem>
              <MenuItem value="calmar">卡玛比率 (Calmar Ratio)</MenuItem>
              <MenuItem value="stability">稳定赚钱 (Stability)</MenuItem>
              <MenuItem value="ic">信息系数 (IC)</MenuItem>
              <MenuItem value="ic_ir">信息比率 (IC_IR)</MenuItem>
              <MenuItem value="total_return">总收益率 (Total Return)</MenuItem>
              <MenuItem value="annualized_return">年化收益率 (Annualized Return)</MenuItem>
              <MenuItem value="win_rate">胜率 (Win Rate)</MenuItem>
              <MenuItem value="profit_factor">盈亏比 (Profit Factor)</MenuItem>
              <MenuItem value="max_drawdown">最大回撤 (Max Drawdown)</MenuItem>
              <MenuItem value="cost">交易成本占比 (Cost Ratio)</MenuItem>
              <MenuItem value="custom">自定义组合</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel>优化方向</InputLabel>
            <Select
              value={formData.direction}
              label="优化方向"
              onChange={e => {
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
                value={objectiveWeights.sharpe_ratio?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    sharpe_ratio: parseFloat(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="总收益率权重"
                value={objectiveWeights.total_return?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    total_return: parseFloat(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="胜率权重"
                value={objectiveWeights.win_rate?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    win_rate: parseFloat(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="盈亏比权重 (Profit Factor)"
                value={objectiveWeights.profit_factor?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    profit_factor: parseFloat(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="信息比率权重 (IC_IR)"
                value={objectiveWeights.information_ratio?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    information_ratio: parseFloat(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="最大回撤权重"
                value={objectiveWeights.max_drawdown?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    max_drawdown: parseFloat(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
              <TextField
                type="number"
                label="交易成本权重 (Cost Ratio)"
                value={objectiveWeights.cost_ratio?.toString() ?? '0'}
                onChange={e =>
                  setObjectiveWeights(prev => ({
                    ...prev,
                    cost_ratio: parseFloat(e.target.value) || 0,
                  }))
                }
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
            onChange={e =>
              setFormData(prev => ({ ...prev, n_trials: parseInt(e.target.value) || 50 }))
            }
            inputProps={{ min: 10, max: 200 }}
            fullWidth
          />

          <FormControl fullWidth>
            <InputLabel>优化方法</InputLabel>
            <Select
              value={formData.optimization_method}
              label="优化方法"
              onChange={e => {
                const method = e.target.value as string;
                setFormData(prev => {
                  const nextIsMulti = method === 'nsga2' || method === 'motpe';
                  let objective_metric = prev.objective_metric;
                  if (nextIsMulti && !Array.isArray(objective_metric)) {
                    objective_metric = [objective_metric];
                  } else if (!nextIsMulti && Array.isArray(objective_metric)) {
                    objective_metric = objective_metric[0] || 'sharpe';
                  }
                  return {
                    ...prev,
                    optimization_method: method,
                    objective_metric,
                  };
                });
              }}
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
            onChange={e =>
              setFormData(prev => ({
                ...prev,
                timeout: e.target.value ? parseInt(e.target.value) : undefined,
              }))
            }
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
