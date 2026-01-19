/**
 * 成本分析组件
 * 展示有成本/无成本收益对比和交易成本明细
 */

'use client';

import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab,
  Chip,
  Tooltip,
  Box,
  Typography,
  LinearProgress,
  Skeleton,
  IconButton,
} from '@mui/material';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Info,
  BarChart3,
  PieChart,
} from 'lucide-react';
import * as echarts from 'echarts';

interface CostAnalysisProps {
  backtestData: any;
  loading?: boolean;
}

export function CostAnalysis({ backtestData, loading = false }: CostAnalysisProps) {
  // 提取有成本/无成本收益数据
  const costComparison = useMemo(() => {
    if (!backtestData) return null;

    const excessReturnWithCost = backtestData.excess_return_with_cost;
    const excessReturnWithoutCost = backtestData.excess_return_without_cost;
    const costStats = backtestData.cost_statistics;
    const portfolioHistory = backtestData.portfolio_history || [];

    return {
      withCost: excessReturnWithCost || {},
      withoutCost: excessReturnWithoutCost || {},
      costStats: costStats || {},
      portfolioHistory: portfolioHistory,
    };
  }, [backtestData]);

  // 计算成本影响
  const costImpact = useMemo(() => {
    if (!costComparison) return null;

    const withCost = costComparison.withCost.annualized_return || 0;
    const withoutCost = costComparison.withoutCost.annualized_return || 0;
    const impact = withoutCost - withCost;

    return {
      impact: impact,
      impactPercent: (impact / Math.abs(withoutCost)) * 100,
      costRatio: costComparison.costStats.cost_ratio || 0,
    };
  }, [costComparison]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Skeleton variant="text" width="25%" />
              <Skeleton variant="rectangular" height={80} />
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  if (!costComparison || !backtestData) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              暂无成本分析数据
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 成本对比概览 */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2 }}>
        {/* 含成本年化收益 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <TrendingDown size={20} color="#d32f2f" />
            </Box>
            <Tooltip title="考虑交易成本后的年化超额收益率">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  含成本年化收益
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 600,
                color: (costComparison.withCost.annualized_return || 0) >= 0 ? 'success.main' : 'error.main',
              }}
            >
              {((costComparison.withCost.annualized_return || 0) * 100).toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        {/* 无成本年化收益 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <TrendingUp size={20} color="#2e7d32" />
            </Box>
            <Tooltip title="不考虑交易成本的年化超额收益率">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  无成本年化收益
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 600,
                color: (costComparison.withoutCost.annualized_return || 0) >= 0 ? 'success.main' : 'error.main',
              }}
            >
              {((costComparison.withoutCost.annualized_return || 0) * 100).toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        {/* 成本影响 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <DollarSign size={20} color="#ed6c02" />
            </Box>
            <Tooltip title="交易成本对年化收益的影响">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  成本影响
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
              {(costImpact?.impact || 0) * 100 >= 0 ? '-' : '+'}
              {Math.abs((costImpact?.impact || 0) * 100).toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        {/* 成本占比 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <PieChart size={20} color="#9c27b0" />
            </Box>
            <Tooltip title="总交易成本占初始资金的比例">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  成本占比
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h4" sx={{ fontWeight: 600 }}>
              {((costImpact?.costRatio || 0) * 100).toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* 详细对比 */}
      <Card>
        <CardHeader title="有成本/无成本收益对比" />
        <CardContent>
          <Tabs value="metrics">
            <Tab label="指标对比" value="metrics" />
            <Tab label="交易成本明细" value="costs" />
          </Tabs>

          <Box sx={{ mt: 2 }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: 'error.main' }}>
                  含成本指标
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      平均收益:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {((costComparison.withCost.mean || 0) * 100).toFixed(4)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      标准差:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {((costComparison.withCost.std || 0) * 100).toFixed(4)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      年化收益:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {((costComparison.withCost.annualized_return || 0) * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      信息比率:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {(costComparison.withCost.information_ratio || 0).toFixed(3)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      最大回撤:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'error.main' }}>
                      {((costComparison.withCost.max_drawdown || 0) * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>
                  无成本指标
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      平均收益:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {((costComparison.withoutCost.mean || 0) * 100).toFixed(4)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      标准差:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {((costComparison.withoutCost.std || 0) * 100).toFixed(4)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      年化收益:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {((costComparison.withoutCost.annualized_return || 0) * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      信息比率:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {(costComparison.withoutCost.information_ratio || 0).toFixed(3)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      最大回撤:
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'error.main' }}>
                      {((costComparison.withoutCost.max_drawdown || 0) * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Box>

            {/* 交易成本明细 */}
            <Box sx={{ mt: 3, display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    总手续费
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    ¥{(costComparison.costStats.total_commission || 0).toLocaleString('zh-CN', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </Typography>
                </CardContent>
              </Card>

              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    总滑点成本
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    ¥{(costComparison.costStats.total_slippage || 0).toLocaleString('zh-CN', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </Typography>
                </CardContent>
              </Card>

              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    总交易成本
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    ¥{(costComparison.costStats.total_cost || 0).toLocaleString('zh-CN', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </Typography>
                </CardContent>
              </Card>
            </Box>

            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  成本占比: {((costComparison.costStats.cost_ratio || 0) * 100).toFixed(2)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min((costComparison.costStats.cost_ratio || 0) * 100, 100)}
                color="warning"
                sx={{ height: 10, borderRadius: 5 }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
