/**
 * 回测任务状态组件
 * 显示回测任务的状态、进度和基础信息
 */

'use client';

import React from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Chip,
  Progress,
  Divider,
  Button,
} from '@heroui/react';
import {
  Clock,
  Calendar,
  Settings,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Play,
  Pause,
  RotateCcw,
} from 'lucide-react';
import { Task } from '../../stores/useTaskStore';

interface BacktestTaskStatusProps {
  task: Task;
  onRetry?: () => void;
  onStop?: () => void;
  loading?: boolean;
}

export default function BacktestTaskStatus({ 
  task, 
  onRetry, 
  onStop, 
  loading = false 
}: BacktestTaskStatusProps) {
  // 获取状态配置
  const getStatusConfig = (status: Task['status']) => {
    const configs = {
      created: {
        color: 'default' as const,
        text: '已创建',
        icon: <Clock className="w-4 h-4" />,
        description: '任务已创建，等待执行'
      },
      running: {
        color: 'primary' as const,
        text: '运行中',
        icon: <Play className="w-4 h-4" />,
        description: '回测正在执行中，请耐心等待'
      },
      completed: {
        color: 'success' as const,
        text: '已完成',
        icon: <CheckCircle className="w-4 h-4" />,
        description: '回测执行完成，可以查看结果'
      },
      failed: {
        color: 'danger' as const,
        text: '执行失败',
        icon: <AlertTriangle className="w-4 h-4" />,
        description: '回测执行失败，请检查配置或重新运行'
      },
    };
    
    return configs[status] || configs.created;
  };

  // 格式化时间
  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // 计算执行时长
  const getExecutionDuration = () => {
    if (!task.created_at) return null;
    
    const startTime = new Date(task.created_at);
    const endTime = task.completed_at ? new Date(task.completed_at) : new Date();
    const duration = endTime.getTime() - startTime.getTime();
    
    const hours = Math.floor(duration / (1000 * 60 * 60));
    const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((duration % (1000 * 60)) / 1000);
    
    if (hours > 0) {
      return `${hours}小时${minutes}分钟${seconds}秒`;
    } else if (minutes > 0) {
      return `${minutes}分钟${seconds}秒`;
    } else {
      return `${seconds}秒`;
    }
  };

  // 策略名称中英文映射
  const getStrategyDisplayName = (strategyName: string): string => {
    const strategyNameMap: Record<string, string> = {
      // 基础技术分析策略
      'moving_average': '移动平均策略',
      'rsi': 'RSI策略',
      'macd': 'MACD策略',
      // 新增技术分析策略
      'bollinger': '布林带策略',
      'stochastic': '随机指标策略',
      'cci': 'CCI策略',
      // 统计套利策略
      'pairs_trading': '配对交易策略',
      'mean_reversion': '均值回归策略',
      'cointegration': '协整策略',
      // 因子投资策略
      'value_factor': '价值因子策略',
      'momentum_factor': '动量因子策略',
      'low_volatility': '低波动因子策略',
      'multi_factor': '多因子组合策略',
    };
    
    return strategyNameMap[strategyName] || strategyName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // 获取回测配置信息
  const getBacktestConfig = () => {
    const result = task.result;
    const config = result?.backtest_config || {};
    
    // 获取原始策略名称并转换为中文
    const rawStrategyName = config.strategy_name || result?.strategy_name || task.model_id || '默认策略';
    
    return {
      startDate: config.start_date || result?.start_date || result?.startDate || '未设置',
      endDate: config.end_date || result?.end_date || result?.endDate || '未设置',
      initialCash: config.initial_cash || result?.initial_cash || result?.initialCash || 100000,
      commissionRate: (config.commission_rate || result?.commission_rate || 0.001) * 100,
      slippageRate: (config.slippage_rate || result?.slippage_rate || 0.001) * 100,
      strategyName: getStrategyDisplayName(rawStrategyName),
    };
  };

  const statusConfig = getStatusConfig(task.status);
  const duration = getExecutionDuration();
  const backtestConfig = getBacktestConfig();

  return (
    <div className="space-y-6">
      {/* 任务状态卡片 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <h3 className="text-lg font-semibold">任务状态</h3>
            <div className="flex items-center space-x-2">
              {statusConfig.icon}
              <Chip color={statusConfig.color} variant="flat">
                {statusConfig.text}
              </Chip>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          {/* 进度条 */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-default-500">执行进度</span>
              <span className="text-sm font-medium">{task.progress}%</span>
            </div>
            <Progress
              value={task.progress}
              color={task.status === 'failed' ? 'danger' : 'primary'}
              className="mb-2"
            />
            <p className="text-sm text-default-500">{statusConfig.description}</p>
          </div>

          {/* 错误信息 */}
          {task.status === 'failed' && task.error_message && (
            <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
              <div className="flex items-start space-x-2">
                <AlertTriangle className="w-5 h-5 text-danger mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-danger">执行失败</p>
                  <p className="text-sm text-danger-600 mt-1">{task.error_message}</p>
                </div>
              </div>
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex space-x-2">
            {task.status === 'failed' && onRetry && (
              <Button
                color="primary"
                variant="flat"
                startContent={<RotateCcw className="w-4 h-4" />}
                onPress={onRetry}
                isLoading={loading}
                size="sm"
              >
                重新运行
              </Button>
            )}
            
            {task.status === 'running' && onStop && (
              <Button
                color="warning"
                variant="flat"
                startContent={<Pause className="w-4 h-4" />}
                onPress={onStop}
                isLoading={loading}
                size="sm"
              >
                停止任务
              </Button>
            )}
          </div>
        </CardBody>
      </Card>

      {/* 任务基础信息 */}
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">基础信息</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-default-500 mb-1">任务名称</p>
              <p className="font-medium">{task.task_name}</p>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">任务ID</p>
              <p className="font-mono text-sm">{task.task_id}</p>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">策略模型</p>
              <Chip variant="flat" color="secondary" size="sm">
                {backtestConfig.strategyName}
              </Chip>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">股票数量</p>
              <p className="font-medium">{task.stock_codes.length} 只</p>
            </div>
          </div>

          <Divider />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-default-500 mb-1 flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                创建时间
              </p>
              <p className="font-medium">{formatDateTime(task.created_at)}</p>
            </div>
            
            {task.completed_at && (
              <div>
                <p className="text-sm text-default-500 mb-1 flex items-center">
                  <CheckCircle className="w-4 h-4 mr-1" />
                  完成时间
                </p>
                <p className="font-medium">{formatDateTime(task.completed_at)}</p>
              </div>
            )}
            
            {duration && (
              <div>
                <p className="text-sm text-default-500 mb-1 flex items-center">
                  <Clock className="w-4 h-4 mr-1" />
                  执行时长
                </p>
                <p className="font-medium">{duration}</p>
              </div>
            )}
          </div>

          <Divider />

          <div>
            <p className="text-sm text-default-500 mb-2">选择的股票</p>
            <div className="flex flex-wrap gap-2">
              {task.stock_codes.map(code => (
                <Chip key={code} variant="flat" size="sm">
                  {code}
                </Chip>
              ))}
            </div>
          </div>
        </CardBody>
      </Card>

      {/* 回测配置信息 */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Settings className="w-5 h-5" />
            <h3 className="text-lg font-semibold">回测配置</h3>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-default-500 mb-1">回测开始日期</p>
              <p className="font-medium">{backtestConfig.startDate}</p>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">回测结束日期</p>
              <p className="font-medium">{backtestConfig.endDate}</p>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">初始资金</p>
              <p className="font-medium">¥{backtestConfig.initialCash.toLocaleString()}</p>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">手续费率</p>
              <p className="font-medium">{backtestConfig.commissionRate.toFixed(3)}%</p>
            </div>
            
            <div>
              <p className="text-sm text-default-500 mb-1">滑点率</p>
              <p className="font-medium">{backtestConfig.slippageRate.toFixed(3)}%</p>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}