import React from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { Minus, TrendingDown, TrendingUp } from 'lucide-react';

import type { Task } from '../../../types/task';

export interface StrategyConfigInfo {
  strategyName: string;
  parameters: Record<string, any>;
}

const DEFAULT_PORTFOLIO_STRATEGIES = [
  { name: 'bollinger', weight: 1, config: { period: 20, std_dev: 2, entry_threshold: 0.02 } },
  { name: 'cci', weight: 1, config: { period: 20, oversold: -100, overbought: 100 } },
  { name: 'macd', weight: 1, config: { fast_period: 12, slow_period: 26, signal_period: 9 } },
];

export function getStrategyConfig(currentTask: Task | null): StrategyConfigInfo | null {
  if (!currentTask || currentTask.task_type !== 'backtest') {
    return null;
  }

  const cfg = currentTask.config;
  const backtestConfig = cfg?.backtest_config;
  const backtestData =
    currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results;
  const resultBacktestConfig = backtestData?.backtest_config;

  let strategyName =
    backtestConfig?.strategy_name ??
    cfg?.strategy_name ??
    resultBacktestConfig?.strategy_name ??
    backtestData?.strategy_name ??
    '未知策略';

  const parameters: Record<string, any> =
    backtestConfig?.strategy_config != null
      ? backtestConfig.strategy_config
      : cfg?.strategy_config != null
        ? cfg.strategy_config
        : resultBacktestConfig?.strategy_config != null
          ? resultBacktestConfig.strategy_config
          : {};

  if (strategyName === '未知策略' && Array.isArray(parameters.strategies)) {
    strategyName = 'portfolio';
  }

  return { strategyName, parameters };
}

export function getStrategyDisplayName(strategyName: string): string {
  return strategyName === 'portfolio' ? '组合策略' : strategyName;
}

export function renderStrategyParameters(parameters: Record<string, any>): React.ReactNode {
  const rawStrategies = Array.isArray(parameters.strategies) ? parameters.strategies : null;
  const strategies =
    rawStrategies === null
      ? null
      : rawStrategies.length > 0
        ? rawStrategies
        : DEFAULT_PORTFOLIO_STRATEGIES;

  if (strategies && strategies.length > 0) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
          <Chip
            size="small"
            color="secondary"
            label={`组合策略 · ${strategies.length} 个${
              rawStrategies?.length === 0 ? '（默认）' : ''
            }`}
          />
          <Chip
            size="small"
            variant="outlined"
            label={`信号整合: ${parameters.integration_method || 'weighted_voting'}`}
          />
        </Box>
        <Box
          sx={{
            display: 'grid',
            gap: 2,
            gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' },
          }}
        >
          {strategies.map((strategy: any, index: number) => (
            <Box
              key={`${strategy?.name || 'strategy'}-${index}`}
              sx={{
                border: 1,
                borderColor: 'divider',
                borderRadius: 2,
                p: 2,
                bgcolor: 'background.paper',
                boxShadow: '0 4px 14px rgba(15, 23, 42, 0.06)',
              }}
            >
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  mb: 1,
                }}
              >
                <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                  {strategy?.name || `策略${index + 1}`}
                </Typography>
                <Chip
                  size="small"
                  color="primary"
                  label={`权重 ${
                    typeof strategy?.weight === 'number'
                      ? strategy.weight.toFixed(2)
                      : strategy?.weight ?? '-'
                  }`}
                />
              </Box>
              {strategy?.config && Object.keys(strategy.config).length > 0 ? (
                <Box
                  component="pre"
                  sx={{
                    fontSize: '0.75rem',
                    color: 'text.secondary',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'monospace',
                    m: 0,
                    p: 1.5,
                    borderRadius: 1,
                    bgcolor: 'grey.50',
                    border: 1,
                    borderColor: 'divider',
                    maxHeight: 200,
                    overflow: 'auto',
                  }}
                >
                  {JSON.stringify(strategy.config, null, 2)}
                </Box>
              ) : (
                <Box
                  sx={{
                    borderRadius: 1,
                    bgcolor: 'grey.50',
                    border: 1,
                    borderColor: 'divider',
                    p: 1.5,
                  }}
                >
                  <Typography variant="caption" color="text.secondary">
                    暂无参数
                  </Typography>
                </Box>
              )}
            </Box>
          ))}
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ bgcolor: 'grey.100', borderRadius: 1, p: 1.5 }}>
      <Box
        component="pre"
        sx={{
          fontSize: '0.75rem',
          color: 'text.secondary',
          whiteSpace: 'pre-wrap',
          fontFamily: 'monospace',
          m: 0,
        }}
      >
        {Object.entries(parameters)
          .map(([key, value]) => {
            if (typeof value === 'object' && value !== null) {
              return `${key}: ${JSON.stringify(value, null, 2)}`;
            }
            return `${key}: ${value}`;
          })
          .join('\n')}
      </Box>
    </Box>
  );
}

export function getStatusChip(status: Task['status']): React.ReactNode {
  const statusConfig = {
    created: { color: 'default' as const, text: '已创建' },
    queued: { color: 'default' as const, text: '排队中' },
    running: { color: 'primary' as const, text: '运行中' },
    completed: { color: 'success' as const, text: '已完成' },
    failed: { color: 'error' as const, text: '失败' },
    cancelled: { color: 'warning' as const, text: '已取消' },
    paused: { color: 'warning' as const, text: '已暂停' },
  };

  const config = statusConfig[status] || statusConfig.created;
  return <Chip label={config.text} color={config.color} size="small" />;
}

export function getPredictionIcon(direction: number): React.ReactNode {
  if (direction > 0) {
    return <TrendingUp className="w-4 h-4 text-success" />;
  }
  if (direction < 0) {
    return <TrendingDown className="w-4 h-4 text-danger" />;
  }
  return <Minus className="w-4 h-4 text-default-500" />;
}

export function getPredictionText(direction: number): string {
  if (direction > 0) {
    return '上涨';
  }
  if (direction < 0) {
    return '下跌';
  }
  return '持平';
}
