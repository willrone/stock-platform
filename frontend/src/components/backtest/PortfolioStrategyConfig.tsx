/**
 * 组合策略配置组件
 *
 * 提供组合策略的配置界面，包括：
 * - 添加/删除策略
 * - 配置每个策略的权重
 * - 配置每个策略的参数
 * - 权重归一化显示
 * - 权重约束验证
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Button,
  Select,
  MenuItem,
  Box,
  Typography,
  FormControl,
  InputLabel,
  FormHelperText,
  IconButton,
  Chip,
  Slider,
  Alert,
  Paper,
} from '@mui/material';
import { Plus, Trash2, Settings, TrendingUp, AlertCircle } from 'lucide-react';
import { StrategyConfigForm, StrategyParameter } from './StrategyConfigForm';

export interface PortfolioStrategyItem {
  name: string;
  weight: number;
  config: Record<string, any>;
}

export interface PortfolioStrategyConfigProps {
  availableStrategies: Array<{
    key: string;
    name: string;
    description: string;
    parameters?: Record<string, StrategyParameter>;
  }>;
  portfolioConfig?: {
    strategies: PortfolioStrategyItem[];
    integration_method?: string;
  };
  onChange: (config: { strategies: PortfolioStrategyItem[]; integration_method: string }) => void;
  constraints?: {
    maxWeight?: number;
    grossLeverage?: number;
    minStrategies?: number;
    maxStrategies?: number;
  };
}

export function PortfolioStrategyConfig({
  availableStrategies,
  portfolioConfig,
  onChange,
  constraints = {
    maxWeight: 0.5,
    grossLeverage: 1.0,
    minStrategies: 1,
    maxStrategies: 10,
  },
}: PortfolioStrategyConfigProps) {
  const [strategies, setStrategies] = useState<PortfolioStrategyItem[]>(
    portfolioConfig?.strategies || []
  );
  const [integrationMethod, setIntegrationMethod] = useState<string>(
    portfolioConfig?.integration_method || 'weighted_voting'
  );
  const [expandedStrategy, setExpandedStrategy] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // 计算归一化后的权重
  const calculateNormalizedWeights = (items: PortfolioStrategyItem[]): Record<string, number> => {
    const totalWeight = items.reduce((sum, item) => sum + item.weight, 0);
    if (totalWeight === 0) {
      return {};
    }
    const normalized: Record<string, number> = {};
    items.forEach((item, index) => {
      normalized[`${item.name}_${index}`] = item.weight / totalWeight;
    });
    return normalized;
  };

  const normalizedWeights = calculateNormalizedWeights(strategies);

  const getDefaultConfigForStrategy = (strategyName: string) => {
    const strategyInfo = availableStrategies.find(s => s.key === strategyName);
    if (!strategyInfo?.parameters) {
      return {};
    }
    const defaults: Record<string, any> = {};
    Object.entries(strategyInfo.parameters).forEach(([key, param]) => {
      defaults[key] = param.default;
    });
    return defaults;
  };

  // 验证权重约束
  const validateWeights = (items: PortfolioStrategyItem[]): Record<string, string> => {
    const newErrors: Record<string, string> = {};
    // const totalWeight = items.reduce((sum, item) => sum + item.weight, 0); // 暂时未使用
    const normalized = calculateNormalizedWeights(items);

    items.forEach((item, index) => {
      const key = `${item.name}_${index}`;
      const normWeight = normalized[key] || 0;

      // 检查最大权重限制
      if (constraints.maxWeight && normWeight > constraints.maxWeight) {
        newErrors[key] = `权重不能超过 ${(constraints.maxWeight * 100).toFixed(0)}%`;
      }

      // 检查权重非负
      if (item.weight < 0) {
        newErrors[key] = '权重不能为负';
      }
    });

    // 检查总杠杆
    if (constraints.grossLeverage) {
      const grossLeverage = items.reduce((sum, item) => sum + Math.abs(item.weight), 0);
      if (grossLeverage > constraints.grossLeverage) {
        newErrors['gross_leverage'] = `总杠杆不能超过 ${constraints.grossLeverage}`;
      }
    }

    return newErrors;
  };

  // 添加策略
  const handleAddStrategy = () => {
    if (strategies.length >= (constraints.maxStrategies || 10)) {
      return;
    }

    const firstAvailableStrategy = availableStrategies.find(
      s => !strategies.some(item => item.name === s.key)
    );

    if (!firstAvailableStrategy) {
      return;
    }

    const newStrategy: PortfolioStrategyItem = {
      name: firstAvailableStrategy.key,
      weight: 1.0,
      config: getDefaultConfigForStrategy(firstAvailableStrategy.key),
    };

    const newStrategies = [...strategies, newStrategy];
    setStrategies(newStrategies);
    updateConfig(newStrategies, integrationMethod);
  };

  // 删除策略
  const handleRemoveStrategy = (index: number) => {
    if (strategies.length <= (constraints.minStrategies || 1)) {
      return;
    }

    const newStrategies = strategies.filter((_, i) => i !== index);
    setStrategies(newStrategies);
    updateConfig(newStrategies, integrationMethod);
  };

  // 更新策略权重
  const handleWeightChange = (index: number, weight: number) => {
    const newStrategies = [...strategies];
    newStrategies[index].weight = Math.max(0, weight);
    setStrategies(newStrategies);
    updateConfig(newStrategies, integrationMethod);
  };

  // 更新策略配置
  const handleStrategyConfigChange = (index: number, config: Record<string, any>) => {
    const newStrategies = [...strategies];
    newStrategies[index].config = config;
    setStrategies(newStrategies);
    updateConfig(newStrategies, integrationMethod);
  };

  // 更新策略名称
  const handleStrategyNameChange = (index: number, name: string) => {
    const newStrategies = [...strategies];
    newStrategies[index].name = name;
    newStrategies[index].config = getDefaultConfigForStrategy(name);
    setStrategies(newStrategies);
    updateConfig(newStrategies, integrationMethod);
  };

  // 更新配置并通知父组件
  const updateConfig = (newStrategies: PortfolioStrategyItem[], method: string) => {
    const newErrors = validateWeights(newStrategies);
    setErrors(newErrors);

    onChange({
      strategies: newStrategies,
      integration_method: method,
    });
  };

  // 缺失配置时回填默认参数
  useEffect(() => {
    if (availableStrategies.length === 0 || strategies.length === 0) {
      return;
    }
    let updated = false;
    const nextStrategies = strategies.map(item => {
      if (item.config && Object.keys(item.config).length > 0) {
        return item;
      }
      const defaults = getDefaultConfigForStrategy(item.name);
      if (Object.keys(defaults).length === 0) {
        return item;
      }
      updated = true;
      return { ...item, config: defaults };
    });
    if (updated) {
      setStrategies(nextStrategies);
      updateConfig(nextStrategies, integrationMethod);
    }
  }, [availableStrategies, strategies, integrationMethod]);

  // 当integrationMethod变化时更新
  useEffect(() => {
    updateConfig(strategies, integrationMethod);
  }, [integrationMethod]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 整合方法选择 */}
      <FormControl fullWidth>
        <InputLabel>信号整合方法</InputLabel>
        <Select
          value={integrationMethod}
          label="信号整合方法"
          onChange={e => setIntegrationMethod(e.target.value)}
        >
          <MenuItem value="weighted_voting">加权投票</MenuItem>
        </Select>
        <FormHelperText>选择如何整合多个策略的信号</FormHelperText>
      </FormControl>

      {/* 权重约束提示 */}
      {constraints.maxWeight && (
        <Alert severity="info" icon={<AlertCircle size={16} />}>
          单个策略权重限制: {(constraints.maxWeight * 100).toFixed(0)}%， 总杠杆限制:{' '}
          {constraints.grossLeverage || 1.0}
        </Alert>
      )}

      {/* 策略列表 */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {strategies.map((strategy, index) => {
          const strategyInfo = availableStrategies.find(s => s.key === strategy.name);
          const normalizedWeight = normalizedWeights[`${strategy.name}_${index}`] || 0;
          const isExpanded = expandedStrategy === `${strategy.name}_${index}`;
          const strategyErrors = errors[`${strategy.name}_${index}`] || errors['gross_leverage'];

          return (
            <Card key={`${strategy.name}_${index}`} variant="outlined">
              <CardHeader
                title={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TrendingUp size={18} />
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {strategyInfo?.name || strategy.name}
                    </Typography>
                    <Chip
                      label={`${(normalizedWeight * 100).toFixed(1)}%`}
                      size="small"
                      color={
                        normalizedWeight > (constraints.maxWeight || 0.5) ? 'error' : 'primary'
                      }
                    />
                  </Box>
                }
                action={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <IconButton
                      size="small"
                      onClick={() =>
                        setExpandedStrategy(isExpanded ? null : `${strategy.name}_${index}`)
                      }
                    >
                      <Settings size={18} />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleRemoveStrategy(index)}
                      disabled={strategies.length <= (constraints.minStrategies || 1)}
                    >
                      <Trash2 size={18} />
                    </IconButton>
                  </Box>
                }
              />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {/* 策略选择 */}
                  <FormControl fullWidth>
                    <InputLabel>策略</InputLabel>
                    <Select
                      value={strategy.name}
                      label="策略"
                      onChange={e => handleStrategyNameChange(index, e.target.value)}
                    >
                      {availableStrategies.map(s => (
                        <MenuItem key={s.key} value={s.key}>
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {s.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {s.description}
                            </Typography>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  {/* 权重配置 */}
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        权重
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        原始: {strategy.weight.toFixed(2)} | 归一化:{' '}
                        {(normalizedWeight * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Slider
                      value={strategy.weight}
                      onChange={(_, value) => handleWeightChange(index, value as number)}
                      min={0}
                      max={10}
                      step={0.1}
                      marks={[
                        { value: 0, label: '0' },
                        { value: 5, label: '5' },
                        { value: 10, label: '10' },
                      ]}
                    />
                    {strategyErrors && <FormHelperText error>{strategyErrors}</FormHelperText>}
                  </Box>

                  {/* 策略参数配置（展开时显示） */}
                  {isExpanded && strategyInfo?.parameters && (
                    <>
                      <Divider />
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        策略参数配置
                      </Typography>
                      <StrategyConfigForm
                        strategyName={strategy.name}
                        parameters={strategyInfo.parameters}
                        values={strategy.config}
                        onChange={config => handleStrategyConfigChange(index, config)}
                      />
                    </>
                  )}
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {/* 添加策略按钮 */}
      <Button
        variant="outlined"
        startIcon={<Plus size={18} />}
        onClick={handleAddStrategy}
        disabled={strategies.length >= (constraints.maxStrategies || 10)}
        fullWidth
      >
        添加策略
      </Button>

      {/* 权重汇总 */}
      {strategies.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            权重汇总
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            {strategies.map((strategy, index) => {
              const normalizedWeight = normalizedWeights[`${strategy.name}_${index}`] || 0;
              return (
                <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">
                    {availableStrategies.find(s => s.key === strategy.name)?.name || strategy.name}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {(normalizedWeight * 100).toFixed(1)}%
                  </Typography>
                </Box>
              );
            })}
            <Divider sx={{ my: 0.5 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                总计
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {Object.values(normalizedWeights).reduce((sum, w) => sum + w, 0) * 100}%
              </Typography>
            </Box>
          </Box>
        </Paper>
      )}
    </Box>
  );
}
