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

export interface StrategyParameter {
  type: 'int' | 'float' | 'boolean' | 'string' | 'json';
  default: any;
  description?: string;
  min?: number;
  max?: number;
  options?: string[];
}

export interface StrategyConfigFormProps {
  strategyName: string;
  parameters: Record<string, StrategyParameter>;
  values?: Record<string, any>;
  onChange?: (values: Record<string, any>) => void;
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
  const [values, setValues] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const prevStrategyRef = React.useRef(strategyName);
  const onChangeRef = React.useRef(onChange);
  const isMountedRef = React.useRef(false);

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
      let initialValues: Record<string, any> = {};
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
    (key: string, value: any) => {
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
    const defaults: Record<string, any> = {};
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
            <Typography variant="h6" component="span">
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
