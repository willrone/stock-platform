/**
 * 策略配置表单组件
 *
 * 根据策略的参数定义动态渲染表单字段
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  TextField,
  Select,
  MenuItem,
  Switch,
  Slider,
  Button,
  Tooltip,
  Chip,
  Box,
  Typography,
  FormControl,
  InputLabel,
  FormHelperText,
  IconButton,
} from '@mui/material';
import { Info, RotateCcw } from 'lucide-react';
import { DataService } from '../../services/dataService';
import { Model } from '../../stores/useDataStore';

export interface StrategyParameter {
  type: 'int' | 'float' | 'boolean' | 'string' | 'json';
  default: unknown;
  description?: string;
  min?: number;
  max?: number;
  options?: string[];
}

export interface StrategyConfigFormProps {
  strategyName: string;
  parameters: Record<string, StrategyParameter>;
  values?: Record<string, unknown>;
  onChange?: (values: Record<string, unknown>) => void;
  onLoadConfig?: (configId: string) => void;
  savedConfigs?: Array<{ config_id: string; config_name: string; created_at: string }>;
  loading?: boolean;
}

export function StrategyConfigForm({
  strategyName,
  parameters,
  values: externalValues,
  onChange,
  onLoadConfig,
  savedConfigs = [],
  loading = false,
}: StrategyConfigFormProps) {
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const prevStrategyRef = React.useRef(strategyName);
  const onChangeRef = React.useRef(onChange);
  const isMountedRef = React.useRef(false);

  // 检测参数名是否为模型ID字段
  const isModelIdField = (key: string) => key.toLowerCase().endsWith('model_id');

  // 根据字段名推断模型类型过滤条件
  const getModelTypeFilter = (key: string): string | null => {
    const k = key.toLowerCase();
    if (k.includes('lgb') || k.includes('lightgbm')) return 'lightgbm';
    if (k.includes('xgb') || k.includes('xgboost')) return 'xgboost';
    return null;
  };

  // 加载可用模型列表（仅当策略参数中包含 model_id 字段时）
  useEffect(() => {
    const hasModelIdParam = Object.keys(parameters).some(isModelIdField);
    if (!hasModelIdParam) return;

    let cancelled = false;
    const loadModels = async () => {
      try {
        const result = await DataService.getModels();
        if (!cancelled) {
          setAvailableModels(result.models.filter(m => m.status === 'ready'));
        }
      } catch (e) {
        console.error('加载模型列表失败:', e);
      }
    };
    loadModels();
    return () => { cancelled = true; };
  }, [parameters]);

  // 更新onChange ref
  React.useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  // 初始化默认值（只在策略变化或首次挂载时）
  useEffect(() => {
    const strategyChanged = prevStrategyRef.current !== strategyName;

    if (strategyChanged || !isMountedRef.current) {
      prevStrategyRef.current = strategyName;
      isMountedRef.current = true;

      // 策略变化或首次挂载时，使用externalValues（如果提供）或默认值
      let initialValues: Record<string, unknown> = {};
      if (externalValues && Object.keys(externalValues).length > 0) {
        initialValues = externalValues;
        setValues(initialValues);
      } else {
        // 生成默认值
        Object.entries(parameters).forEach(([key, param]) => {
          initialValues[key] = param.default;
        });
        setValues(initialValues);
      }

      // 通知父组件初始值（使用微任务避免在渲染期间调用）
      if (onChangeRef.current && Object.keys(initialValues).length > 0) {
        Promise.resolve().then(() => {
          onChangeRef.current?.(initialValues);
        });
      }
    }
    // 不监听externalValues，避免循环更新
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyName]);

  // 处理参数值变化
  const handleValueChange = React.useCallback(
    (key: string, value: unknown) => {
      const param = parameters[key];
      if (!param) {
        return;
      }

      // 验证值
      let validatedValue = value;
      if (param.type === 'int') {
        validatedValue =
          typeof value === 'number' ? Math.round(value) : parseInt(String(value), 10);
        if (isNaN(validatedValue)) {
          setErrors(prev => ({ ...prev, [key]: '请输入有效的整数' }));
          return;
        }
        if (param.min !== undefined && validatedValue < param.min) {
          validatedValue = param.min;
        }
        if (param.max !== undefined && validatedValue > param.max) {
          validatedValue = param.max;
        }
      } else if (param.type === 'float') {
        validatedValue = typeof value === 'number' ? value : parseFloat(String(value));
        if (isNaN(validatedValue)) {
          setErrors(prev => ({ ...prev, [key]: '请输入有效的数字' }));
          return;
        }
        if (param.min !== undefined && validatedValue < param.min) {
          validatedValue = param.min;
        }
        if (param.max !== undefined && validatedValue > param.max) {
          validatedValue = param.max;
        }
      } else if (param.type === 'json') {
        try {
          validatedValue = typeof value === 'string' ? JSON.parse(value) : value;
        } catch (e) {
          setErrors(prev => ({ ...prev, [key]: '请输入有效的JSON格式' }));
          return;
        }
      }

      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[key];
        return newErrors;
      });

      setValues(prev => {
        // 避免不必要的更新
        if (prev[key] === validatedValue) {
          return prev;
        }
        const newValues = { ...prev, [key]: validatedValue };
        // 调用onChange，使用ref避免闭包问题
        if (onChangeRef.current) {
          // 使用微任务，确保在状态更新后调用
          Promise.resolve().then(() => {
            onChangeRef.current?.(newValues);
          });
        }
        return newValues;
      });
    },
    [parameters]
  );

  // 重置为默认值
  const handleReset = () => {
    const defaults: Record<string, unknown> = {};
    Object.entries(parameters).forEach(([key, param]) => {
      defaults[key] = param.default;
    });
    setValues(defaults);
    setErrors({});
    // 通知父组件
    if (onChangeRef.current) {
      Promise.resolve().then(() => {
        onChangeRef.current?.(defaults);
      });
    }
  };

  // 渲染参数输入
  const renderParameterInput = (key: string, param: StrategyParameter) => {
    const value = values[key] ?? param.default;
    const error = errors[key];

    switch (param.type) {
      case 'int':
      case 'float':
        const numValue =
          typeof value === 'number'
            ? value
            : param.type === 'int'
              ? parseInt(String(value || param.default), 10)
              : parseFloat(String(value || param.default));
        const safeValue = isNaN(numValue) ? param.default : numValue;

        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <TextField
                type="number"
                value={safeValue?.toString() || ''}
                onChange={e => {
                  const numVal =
                    param.type === 'int'
                      ? parseInt(e.target.value, 10)
                      : parseFloat(e.target.value);
                  if (!isNaN(numVal)) {
                    handleValueChange(key, numVal);
                  }
                }}
                error={!!error}
                helperText={error}
                inputProps={{
                  step: param.type === 'float' ? 0.001 : 1,
                  min: param.min,
                  max: param.max,
                }}
                sx={{ flex: 1 }}
              />
              {param.min !== undefined && param.max !== undefined && (
                <Box sx={{ flex: 1 }}>
                  <Slider
                    value={safeValue}
                    onChange={(e, val) => {
                      const numVal = Number(val);
                      if (!isNaN(numVal)) {
                        const finalValue = param.type === 'int' ? Math.round(numVal) : numVal;
                        setValues(prev => {
                          if (prev[key] === finalValue) {
                            return prev;
                          }
                          const newValues = { ...prev, [key]: finalValue };
                          if (onChangeRef.current) {
                            Promise.resolve().then(() => {
                              onChangeRef.current?.(newValues);
                            });
                          }
                          return newValues;
                        });
                        setErrors(prev => {
                          const newErrors = { ...prev };
                          delete newErrors[key];
                          return newErrors;
                        });
                      }
                    }}
                    min={param.min}
                    max={param.max}
                    step={param.type === 'float' ? 0.001 : 1}
                    valueLabelDisplay="auto"
                  />
                </Box>
              )}
            </Box>
            {(param.min !== undefined || param.max !== undefined) && (
              <Typography variant="caption" color="text.secondary">
                范围: {param.min ?? '无限制'} - {param.max ?? '无限制'}
              </Typography>
            )}
          </Box>
        );

      case 'boolean':
        return <Switch checked={value} onChange={e => handleValueChange(key, e.target.checked)} />;

      case 'string':
        // 模型ID字段：渲染为模型选择下拉框
        if (isModelIdField(key)) {
          const typeFilter = getModelTypeFilter(key);
          const filteredModels = typeFilter
            ? availableModels.filter(m => m.model_type === typeFilter)
            : availableModels;

          return (
            <FormControl fullWidth error={!!error}>
              <InputLabel>{param.description || key}</InputLabel>
              <Select
                value={value || ''}
                label={param.description || key}
                onChange={e => handleValueChange(key, e.target.value)}
                displayEmpty
              >
                <MenuItem value="">
                  <Typography variant="body2" color="text.secondary">
                    留空使用默认模型
                  </Typography>
                </MenuItem>
                {filteredModels.length > 0 ? (
                  filteredModels.map(model => (
                    <MenuItem key={model.model_id} value={model.model_id}>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {model.model_name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          准确率: {(model.accuracy * 100).toFixed(1)}% | {model.model_type} | {new Date(model.created_at).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))
                ) : (
                  <MenuItem disabled>
                    <Typography variant="body2" color="text.secondary">
                      暂无{typeFilter ? ` ${typeFilter} 类型的` : ''}可用模型
                    </Typography>
                  </MenuItem>
                )}
              </Select>
              {error && <FormHelperText>{error}</FormHelperText>}
            </FormControl>
          );
        }

        if (param.options && param.options.length > 0) {
          return (
            <FormControl fullWidth error={!!error}>
              <InputLabel>{key}</InputLabel>
              <Select
                value={value || ''}
                label={key}
                onChange={e => handleValueChange(key, e.target.value)}
              >
                {param.options.map(option => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
              {error && <FormHelperText>{error}</FormHelperText>}
            </FormControl>
          );
        }
        return (
          <TextField
            value={value?.toString() || ''}
            onChange={e => handleValueChange(key, e.target.value)}
            error={!!error}
            helperText={error}
            fullWidth
          />
        );

      case 'json':
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <TextField
              multiline
              rows={3}
              value={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
              onChange={e => handleValueChange(key, e.target.value)}
              error={!!error}
              helperText={error || '请输入有效的JSON格式'}
              placeholder='例如: [1, 2, 3] 或 {"key": "value"}'
              fullWidth
            />
          </Box>
        );

      default:
        return null;
    }
  };

  if (Object.keys(parameters).length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="h6"
              component="span"
              sx={{ fontSize: { xs: '0.95rem', md: '1.25rem' } }}
            >
              策略配置参数
            </Typography>
            <Chip label={strategyName} size="small" color="secondary" />
          </Box>
        }
        action={
          <Button
            size="small"
            variant="outlined"
            startIcon={<RotateCcw size={16} />}
            onClick={handleReset}
          >
            重置为默认值
          </Button>
        }
      />
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* 加载已保存配置 */}
        {onLoadConfig && (
          <Box
            sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1, border: 1, borderColor: 'divider' }}
          >
            <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
              加载已保存配置
            </Typography>
            {loading ? (
              <Typography variant="body2" color="text.secondary">
                加载中...
              </Typography>
            ) : savedConfigs.length > 0 ? (
              <FormControl fullWidth size="small">
                <InputLabel>选择已保存的配置</InputLabel>
                <Select
                  value=""
                  label="选择已保存的配置"
                  onChange={e => {
                    const configId = e.target.value;
                    if (configId && onLoadConfig) {
                      onLoadConfig(configId);
                    }
                  }}
                  disabled={loading}
                >
                  {savedConfigs.map(config => (
                    <MenuItem key={config.config_id} value={config.config_id}>
                      {config.config_name} ({new Date(config.created_at).toLocaleDateString()})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ) : (
              <Typography variant="body2" color="text.secondary">
                暂无已保存的配置。完成回测或超参优化后可以保存配置。
              </Typography>
            )}
          </Box>
        )}

        {/* 参数表单 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {Object.entries(parameters).map(([key, param]) => (
            <Box key={key} sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {key}
                </Typography>
                {param.description && (
                  <Tooltip title={param.description}>
                    <IconButton size="small">
                      <Info size={16} />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
              {renderParameterInput(key, param)}
              {param.description && (
                <Typography variant="caption" color="text.secondary">
                  {param.description}
                </Typography>
              )}
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
}
