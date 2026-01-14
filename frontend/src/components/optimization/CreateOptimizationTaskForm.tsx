/**
 * 创建超参优化任务表单
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Button,
  Input,
  Select,
  SelectItem,
  DatePicker,
  Switch,
  Card,
  CardHeader,
  CardBody,
  Divider,
  Chip,
} from '@heroui/react';
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
          if (param.type === 'int') {
            defaultSpace[key] = {
              type: 'int',
              low: param.min || 1,
              high: param.max || 100,
              default: param.default,
              enabled: true,
            };
          } else if (param.type === 'float') {
            defaultSpace[key] = {
              type: 'float',
              low: param.min || 0.0,
              high: param.max || 1.0,
              default: param.default,
              enabled: true,
            };
          }
        });
        setParamSpace(defaultSpace);
      }
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
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">创建超参优化任务</h2>
        <p className="text-default-500">配置优化参数，寻找策略的最佳参数组合</p>
      </div>

      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">基本信息</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          <Input
            label="任务名称"
            placeholder="输入任务名称"
            value={formData.task_name}
            onChange={(e) => setFormData(prev => ({ ...prev, task_name: e.target.value }))}
            isRequired
          />

          <Select
            label="选择策略"
            placeholder={strategies.length === 0 ? "加载中..." : "选择要优化的策略"}
            selectedKeys={formData.strategy_name ? [formData.strategy_name] : []}
            onSelectionChange={(keys) => {
              const selected = Array.from(keys)[0] as string;
              setFormData(prev => ({ ...prev, strategy_name: selected }));
            }}
            isRequired
            isDisabled={strategies.length === 0}
          >
            {strategies.map((strategy) => (
              <SelectItem key={strategy.key} value={strategy.key}>
                {strategy.name || strategy.key}
              </SelectItem>
            ))}
          </Select>
          {strategies.length === 0 && (
            <p className="text-sm text-default-500">正在加载策略列表...</p>
          )}

          <div>
            <label className="text-sm font-medium mb-2 block">选择股票</label>
            <StockSelector
              value={selectedStocks}
              onChange={setSelectedStocks}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Input
              type="date"
              label="开始日期"
              value={formData.start_date}
              onChange={(e) => setFormData(prev => ({ ...prev, start_date: e.target.value }))}
              isRequired
            />
            <Input
              type="date"
              label="结束日期"
              value={formData.end_date}
              onChange={(e) => setFormData(prev => ({ ...prev, end_date: e.target.value }))}
              isRequired
            />
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">参数空间配置</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          {Object.entries(paramSpace).length === 0 ? (
            <p className="text-default-500">请先选择策略</p>
          ) : (
            Object.entries(paramSpace).map(([paramName, config]) => (
              <div key={paramName} className="p-4 border border-divider rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">{paramName}</h4>
                  <Switch
                    isSelected={config.enabled}
                    onValueChange={(checked) => updateParamSpace(paramName, 'enabled', checked)}
                  >
                    启用优化
                  </Switch>
                </div>
                {config.enabled && (
                  <div className="grid grid-cols-2 gap-4">
                    <Input
                      type="number"
                      label="最小值"
                      value={config.low?.toString() || ''}
                      onChange={(e) => updateParamSpace(paramName, 'low', parseFloat(e.target.value))}
                    />
                    <Input
                      type="number"
                      label="最大值"
                      value={config.high?.toString() || ''}
                      onChange={(e) => updateParamSpace(paramName, 'high', parseFloat(e.target.value))}
                    />
                  </div>
                )}
              </div>
            ))
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">优化目标</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          <Select
            label="目标指标"
            selectedKeys={Array.isArray(formData.objective_metric) ? formData.objective_metric : [formData.objective_metric]}
            onSelectionChange={(keys) => {
              const selected = Array.from(keys);
              setFormData(prev => ({ 
                ...prev, 
                objective_metric: selected.length === 1 ? selected[0] as string : selected as string[]
              }));
            }}
          >
            <SelectItem key="sharpe" value="sharpe">夏普比率 (Sharpe Ratio)</SelectItem>
            <SelectItem key="calmar" value="calmar">卡玛比率 (Calmar Ratio)</SelectItem>
            <SelectItem key="ic" value="ic">信息系数 (IC)</SelectItem>
            <SelectItem key="custom" value="custom">自定义组合</SelectItem>
          </Select>

          <Select
            label="优化方向"
            selectedKeys={[formData.direction]}
            onSelectionChange={(keys) => {
              const selected = Array.from(keys)[0] as 'maximize' | 'minimize';
              setFormData(prev => ({ ...prev, direction: selected }));
            }}
          >
            <SelectItem key="maximize" value="maximize">最大化</SelectItem>
            <SelectItem key="minimize" value="minimize">最小化</SelectItem>
          </Select>

          {formData.objective_metric === 'custom' && (
            <div className="space-y-2">
              <label className="text-sm font-medium">自定义权重</label>
              <Input
                type="number"
                label="夏普比率权重"
                value={objectiveWeights.sharpe_ratio.toString()}
                onChange={(e) => setObjectiveWeights(prev => ({ ...prev, sharpe_ratio: parseFloat(e.target.value) }))}
                min={0}
                max={1}
                step={0.1}
              />
              <Input
                type="number"
                label="总收益率权重"
                value={objectiveWeights.total_return.toString()}
                onChange={(e) => setObjectiveWeights(prev => ({ ...prev, total_return: parseFloat(e.target.value) }))}
                min={0}
                max={1}
                step={0.1}
              />
            </div>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">优化配置</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          <Input
            type="number"
            label="试验次数"
            value={formData.n_trials.toString()}
            onChange={(e) => setFormData(prev => ({ ...prev, n_trials: parseInt(e.target.value) || 50 }))}
            min={10}
            max={200}
          />

          <Select
            label="优化方法"
            selectedKeys={[formData.optimization_method]}
            onSelectionChange={(keys) => {
              const selected = Array.from(keys)[0] as string;
              setFormData(prev => ({ ...prev, optimization_method: selected }));
            }}
          >
            <SelectItem key="tpe" value="tpe">TPE (Tree-structured Parzen Estimator)</SelectItem>
            <SelectItem key="random" value="random">随机搜索</SelectItem>
            <SelectItem key="nsga2" value="nsga2">NSGA-II (多目标)</SelectItem>
            <SelectItem key="motpe" value="motpe">MOTPE (多目标)</SelectItem>
          </Select>

          <Input
            type="number"
            label="超时时间（秒，可选）"
            value={formData.timeout?.toString() || ''}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              timeout: e.target.value ? parseInt(e.target.value) : undefined 
            }))}
            placeholder="不设置则无超时限制"
          />
        </CardBody>
      </Card>

      <div className="flex justify-end gap-4">
        <Button
          color="primary"
          onPress={handleSubmit}
          isLoading={loading}
          startContent={!loading && <Save />}
        >
          {loading ? '创建中...' : '创建优化任务'}
        </Button>
      </div>
    </div>
  );
}

