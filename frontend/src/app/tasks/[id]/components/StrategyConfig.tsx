/**
 * 策略配置展示组件
 */

import React from 'react';
import { Card, CardHeader, CardContent, Box, Typography, Chip, Button } from '@mui/material';
import { Save } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';

interface StrategyConfigProps {
  task: Task;
  onSaveConfig: () => void;
}

// 组合策略默认配置（与后端一致）
const DEFAULT_PORTFOLIO_STRATEGIES = [
  { name: 'bollinger', weight: 1, config: { period: 20, std_dev: 2, entry_threshold: 0.02 } },
  { name: 'cci', weight: 1, config: { period: 20, oversold: -100, overbought: 100 } },
  { name: 'macd', weight: 1, config: { fast_period: 12, slow_period: 26, signal_period: 9 } },
];

export function StrategyConfig({ task, onSaveConfig }: StrategyConfigProps) {
  // 获取��略配置信息
  const getStrategyConfig = () => {
    if (!task || task.task_type !== 'backtest') {
      return null;
    }

    const cfg = task.config;
    const bc = cfg?.backtest_config;
    const backtestData = task.result || task.results?.backtest_results || task.backtest_results;
    const resultBc = backtestData?.backtest_config;

    let strategyName =
      bc?.strategy_name ??
      cfg?.strategy_name ??
      resultBc?.strategy_name ??
      (backtestData as any)?.strategy_name ??
      '未知策略';

    const parameters: Record<string, any> =
      bc?.strategy_config != null
        ? bc.strategy_config
        : cfg?.strategy_config != null
          ? cfg.strategy_config
          : resultBc?.strategy_config != null
            ? resultBc.strategy_config
            : {};

    if (strategyName === '未知策略' && Array.isArray((parameters as any)?.strategies)) {
      strategyName = 'portfolio';
    }

    return { strategyName, parameters };
  };

  const getStrategyDisplayName = (strategyName: string) => {
    return strategyName === 'portfolio' ? '组合策略' : strategyName;
  };

  const renderStrategyParameters = (parameters: Record<string, any>) => {
    const raw = Array.isArray(parameters.strategies) ? parameters.strategies : null;
    const strategies = raw === null ? null : raw.length > 0 ? raw : DEFAULT_PORTFOLIO_STRATEGIES;

    if (strategies && strategies.length > 0) {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
            <Chip
              size="small"
              color="secondary"
              label={`组合策略 · ${strategies.length} 个${raw?.length === 0 ? '（默认）' : ''}`}
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
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' },
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
  };

  const configInfo = getStrategyConfig();
  if (!configInfo) {
    return null;
  }

  return (
    <Card>
      <CardHeader
        title={
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              justifyContent: 'space-between',
              alignItems: { xs: 'flex-start', sm: 'center' },
              gap: 1,
              width: '100%',
            }}
          >
            <Box>
              <Typography
                variant="h6"
                component="h4"
                sx={{ fontWeight: 600, fontSize: { xs: '0.9rem', sm: '1.25rem' } }}
              >
                策略配置
              </Typography>
              <Typography variant="caption" color="text.secondary">
                策略: {getStrategyDisplayName(configInfo.strategyName)}
              </Typography>
            </Box>
            <Button
              variant="outlined"
              color="primary"
              size="small"
              startIcon={<Save size={14} />}
              onClick={onSaveConfig}
              disabled={
                !configInfo.strategyName ||
                configInfo.strategyName === '未知策略' ||
                Object.keys(configInfo.parameters).length === 0
              }
              sx={{ flexShrink: 0, minHeight: 36 }}
            >
              保存配置
            </Button>
          </Box>
        }
      />
      <CardContent>
        {Object.keys(configInfo.parameters).length > 0 ? (
          renderStrategyParameters(configInfo.parameters)
        ) : (
          <Typography variant="caption" color="text.secondary">
            暂无策略参数配置
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}
