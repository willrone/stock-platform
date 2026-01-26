/**
 * 组合策略结果展示组件
 *
 * 展示组合策略回测结果，包括：
 * - 策略贡献度可视化
 * - 策略权重分布图表
 * - 信号来源标识
 */

'use client';

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Box,
  Typography,
  Chip,
  Paper,
  Divider,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, PieChart as PieChartIcon, BarChart3 } from 'lucide-react';

export interface PortfolioStrategyInfo {
  name: string;
  weight: number;
  contribution?: {
    return_contribution?: number;
    sharpe_contribution?: number;
    trade_count?: number;
  };
}

export interface PortfolioBacktestResult {
  is_portfolio?: boolean;
  portfolio_info?: {
    strategies: PortfolioStrategyInfo[];
    correlation_matrix?: number[][];
  };
  // 标准回测结果字段
  strategy_name?: string;
  portfolio?: {
    total_return: number;
    annualized_return: number;
  };
  risk_metrics?: {
    sharpe_ratio: number;
    max_drawdown: number;
  };
  trading_stats?: {
    total_trades: number;
    win_rate: number;
  };
}

export interface PortfolioStrategyResultsProps {
  backtestData: PortfolioBacktestResult | null;
  loading?: boolean;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658'];

export function PortfolioStrategyResults({
  backtestData,
  loading = false,
}: PortfolioStrategyResultsProps) {
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography>加载中...</Typography>
        </CardContent>
      </Card>
    );
  }

  if (!backtestData || !backtestData.is_portfolio || !backtestData.portfolio_info) {
    return null; // 不是组合策略，不显示此组件
  }

  const portfolioInfo = backtestData.portfolio_info;
  const strategies = portfolioInfo.strategies || [];

  if (strategies.length === 0) {
    return null;
  }

  // 准备权重分布数据
  const weightData = strategies.map((strategy, index) => ({
    name: strategy.name,
    weight: strategy.weight * 100,
    color: COLORS[index % COLORS.length],
  }));

  // 准备贡献度数据（如果有）
  const contributionData = strategies
    .filter(s => s.contribution)
    .map((strategy, index) => ({
      name: strategy.name,
      return_contribution: (strategy.contribution?.return_contribution || 0) * 100,
      sharpe_contribution: strategy.contribution?.sharpe_contribution || 0,
      trade_count: strategy.contribution?.trade_count || 0,
      color: COLORS[index % COLORS.length],
    }));

  // 计算总权重
  const totalWeight = strategies.reduce((sum, s) => sum + s.weight, 0);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 组合策略信息卡片 */}
      <Card>
        <CardHeader
          avatar={<PieChartIcon size={24} />}
          title="组合策略信息"
          subheader={`包含 ${strategies.length} 个策略`}
        />
        <CardContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
              gap: 2,
            }}
          >
            {strategies.map((strategy, index) => (
              <Box key={strategy.name}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        bgcolor: COLORS[index % COLORS.length],
                      }}
                    />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {strategy.name}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    权重: {(strategy.weight * 100).toFixed(1)}%
                  </Typography>
                  {strategy.contribution && (
                    <>
                      {strategy.contribution.return_contribution !== undefined && (
                        <Typography variant="body2" color="text.secondary">
                          收益贡献: {(strategy.contribution.return_contribution * 100).toFixed(2)}%
                        </Typography>
                      )}
                      {strategy.contribution.trade_count !== undefined && (
                        <Typography variant="body2" color="text.secondary">
                          交易次数: {strategy.contribution.trade_count}
                        </Typography>
                      )}
                    </>
                  )}
                </Paper>
              </Box>
            ))}
          </Box>
          <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
            <Typography variant="body2" color="text.secondary">
              总权重: {(totalWeight * 100).toFixed(1)}%
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* 权重分布图表 */}
      <Card>
        <CardHeader avatar={<PieChartIcon size={24} />} title="策略权重分布" />
        <CardContent>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={weightData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(props: any) => {
                    const { name, percent } = props;
                    return `${name}: ${((percent || 0) * 100).toFixed(1)}%`;
                  }}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="weight"
                >
                  {weightData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* 策略贡献度对比（如果有数据） */}
      {contributionData.length > 0 && (
        <>
          <Card>
            <CardHeader avatar={<BarChart3 size={24} />} title="策略收益贡献对比" />
            <CardContent>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={contributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: '收益贡献 (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="return_contribution" fill="#8884d8" name="收益贡献 (%)" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardHeader avatar={<TrendingUp size={24} />} title="策略交易次数对比" />
            <CardContent>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={contributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: '交易次数', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="trade_count" fill="#82ca9d" name="交易次数" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </>
      )}

      {/* 组合策略总体表现 */}
      {backtestData.portfolio && (
        <Card>
          <CardHeader avatar={<TrendingUp size={24} />} title="组合策略总体表现" />
          <CardContent>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                gap: 2,
              }}
            >
              <Box>
                <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    总收益率
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {(backtestData.portfolio.total_return * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Box>
              <Box>
                <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    年化收益率
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {(backtestData.portfolio.annualized_return * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Box>
              {backtestData.risk_metrics && (
                <>
                  <Box>
                    <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        夏普比率
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {backtestData.risk_metrics.sharpe_ratio.toFixed(2)}
                      </Typography>
                    </Paper>
                  </Box>
                  <Box>
                    <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        最大回撤
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                        {(backtestData.risk_metrics.max_drawdown * 100).toFixed(2)}%
                      </Typography>
                    </Paper>
                  </Box>
                </>
              )}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
