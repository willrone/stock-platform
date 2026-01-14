/**
 * 策略配置表单组件
 * 
 * 根据策略的参数定义动态渲染表单字段
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Input,
  Select,
  SelectItem,
  Switch,
  Slider,
  Button,
  Tooltip,
  Chip,
} from '@heroui/react';
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
      if (externalValues && Object.keys(externalValues).length > 0) {
        setValues(externalValues);
      } else {
        const defaults: Record<string, any> = {};
        Object.entries(parameters).forEach(([key, param]) => {
          defaults[key] = param.default;
        });
        setValues(defaults);
      }
    }
    // 不监听externalValues，避免循环更新
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyName]);

  // 处理参数值变化
  const handleValueChange = React.useCallback((key: string, value: any) => {
    const param = parameters[key];
    if (!param) return;

    // 验证值
    let validatedValue = value;
    if (param.type === 'int') {
      validatedValue = typeof value === 'number' ? Math.round(value) : parseInt(String(value), 10);
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
  }, [parameters]);

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
        const numValue = typeof value === 'number' ? value : (param.type === 'int' ? parseInt(String(value || param.default), 10) : parseFloat(String(value || param.default)));
        const safeValue = isNaN(numValue) ? param.default : numValue;
        
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <Input
                type="number"
                value={safeValue?.toString() || ''}
                onValueChange={(val) => {
                  const numVal = param.type === 'int' ? parseInt(val, 10) : parseFloat(val);
                  if (!isNaN(numVal)) {
                    handleValueChange(key, numVal);
                  }
                }}
                isInvalid={!!error}
                errorMessage={error}
                step={param.type === 'float' ? 0.001 : 1}
                min={param.min}
                max={param.max}
                className="flex-1"
              />
              {param.min !== undefined && param.max !== undefined && (
                <Slider
                  value={safeValue}
                  onChange={(val) => {
                    // 滑块直接更新，避免通过handleValueChange的延迟
                    const numVal = Number(val);
                    if (!isNaN(numVal)) {
                      const finalValue = param.type === 'int' ? Math.round(numVal) : numVal;
                      // 直接更新state
                      setValues(prev => {
                        if (prev[key] === finalValue) return prev;
                        const newValues = { ...prev, [key]: finalValue };
                        // 调用onChange
                        if (onChangeRef.current) {
                          Promise.resolve().then(() => {
                            onChangeRef.current?.(newValues);
                          });
                        }
                        return newValues;
                      });
                      // 清除错误
                      setErrors(prev => {
                        const newErrors = { ...prev };
                        delete newErrors[key];
                        return newErrors;
                      });
                    }
                  }}
                  minValue={param.min}
                  maxValue={param.max}
                  step={param.type === 'float' ? 0.001 : 1}
                  className="flex-1"
                  aria-label={`${key} 滑块`}
                />
              )}
            </div>
            {(param.min !== undefined || param.max !== undefined) && (
              <p className="text-xs text-default-500">
                范围: {param.min ?? '无限制'} - {param.max ?? '无限制'}
              </p>
            )}
          </div>
        );

      case 'boolean':
        return (
          <Switch
            isSelected={value}
            onValueChange={(val) => handleValueChange(key, val)}
          />
        );

      case 'string':
        if (param.options && param.options.length > 0) {
          return (
            <Select
              selectedKeys={value ? [value] : []}
              onSelectionChange={(keys) => {
                const selected = Array.from(keys)[0] as string;
                handleValueChange(key, selected);
              }}
              isInvalid={!!error}
              errorMessage={error}
            >
              {param.options.map(option => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </Select>
          );
        }
        return (
          <Input
            value={value?.toString() || ''}
            onValueChange={(val) => handleValueChange(key, val)}
            isInvalid={!!error}
            errorMessage={error}
          />
        );

      case 'json':
        return (
          <div className="space-y-2">
            <Input
              type="textarea"
              value={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
              onValueChange={(val) => handleValueChange(key, val)}
              isInvalid={!!error}
              errorMessage={error}
              placeholder='例如: [1, 2, 3] 或 {"key": "value"}'
              minRows={3}
            />
            <p className="text-xs text-default-500">请输入有效的JSON格式</p>
          </div>
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
      <CardHeader className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <h3 className="text-lg font-semibold">策略配置参数</h3>
          <Chip variant="flat" size="sm" color="secondary">
            {strategyName}
          </Chip>
        </div>
        <Button
          size="sm"
          variant="light"
          startContent={<RotateCcw className="w-4 h-4" />}
          onPress={handleReset}
        >
          重置为默认值
        </Button>
      </CardHeader>
      <CardBody className="space-y-4">
        {/* 加载已保存配置 */}
        {savedConfigs.length > 0 && onLoadConfig && (
          <div className="mb-4">
            <Select
              label="加载已保存配置"
              placeholder="选择已保存的配置"
              onSelectionChange={(keys) => {
                const configId = Array.from(keys)[0] as string;
                if (configId && onLoadConfig) {
                  onLoadConfig(configId);
                }
              }}
              isDisabled={loading}
            >
              {savedConfigs.map(config => (
                <SelectItem key={config.config_id} value={config.config_id}>
                  {config.config_name} ({new Date(config.created_at).toLocaleDateString()})
                </SelectItem>
              ))}
            </Select>
          </div>
        )}

        {/* 参数表单 */}
        <div className="space-y-4">
          {Object.entries(parameters).map(([key, param]) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium">{key}</label>
                {param.description && (
                  <Tooltip content={param.description}>
                    <Info className="w-4 h-4 text-default-400" />
                  </Tooltip>
                )}
              </div>
              {renderParameterInput(key, param)}
              {param.description && (
                <p className="text-xs text-default-500">{param.description}</p>
              )}
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}
