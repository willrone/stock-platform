/**
 * 回测概览组件
 * 展示回测任务的关键指标和概览信息
 */

'use client';

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Chip,
  LinearProgress,
  Tooltip,
  Box,
  Typography,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  DollarSign,
  Activity,
  AlertTriangle,
  Info,
} from 'lucide-react';

interface BacktestOverviewProps {
  backtestData: any;
  loading?: boolean;
}

interface BacktestMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
  // 新增：交易分布与时间分段的概要统计（如果后端提供则展示）
  tradePnlMean: number;
  tradePnlMedian: number;
  tradePnlStd: number;
  monthlyReturnMean: number;
  monthlyReturnStd: number;
  positiveMonths: number;
  negativeMonths: number;
  stocksTraded: number;
}

export default function BacktestOverview({ backtestData, loading = false }: BacktestOverviewProps) {
  // 处理回测数据，提取关键指标
  const processMetrics = (): BacktestMetrics => {
    if (!backtestData) {
      return {
        totalReturn: 0,
        annualizedReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        volatility: 0,
        winRate: 0,
        totalTrades: 0,
        profitFactor: 0,
        tradePnlMean: 0,
        tradePnlMedian: 0,
        tradePnlStd: 0,
        monthlyReturnMean: 0,
        monthlyReturnStd: 0,
        positiveMonths: 0,
        negativeMonths: 0,
        stocksTraded: 0,
      };
    }

    const additional = backtestData || {};

    return {
      totalReturn: (backtestData.total_return || 0) * 100,
      annualizedReturn: (backtestData.annualized_return || 0) * 100,
      sharpeRatio: backtestData.sharpe_ratio || 0,
      maxDrawdown: (backtestData.max_drawdown || 0) * 100,
      volatility: (backtestData.volatility || 0) * 100,
      winRate: (backtestData.win_rate || 0) * 100,
      totalTrades: backtestData.total_trades || 0,
      profitFactor: backtestData.profit_factor || 0,
      tradePnlMean: (additional.trade_pnl_mean || 0) * 100,
      tradePnlMedian: (additional.trade_pnl_median || 0) * 100,
      tradePnlStd: (additional.trade_pnl_std || 0) * 100,
      monthlyReturnMean: (additional.monthly_return_mean || 0) * 100,
      monthlyReturnStd: (additional.monthly_return_std || 0) * 100,
      positiveMonths: additional.positive_months || 0,
      negativeMonths: additional.negative_months || 0,
      stocksTraded: additional.stocks_traded || 0,
    };
  };

  const metrics = processMetrics();

  // 获取收益率颜色
  const getReturnColor = (value: number): string => {
    if (value > 0) return 'success.main';
    if (value < 0) return 'error.main';
    return 'text.secondary';
  };

  // 获取收益率图标
  const getReturnIcon = (value: number) => {
    if (value > 0) return <TrendingUp size={20} color="#2e7d32" />;
    if (value < 0) return <TrendingDown size={20} color="#d32f2f" />;
    return <Activity size={20} color="#666" />;
  };

  // 获取夏普比率评级
  const getSharpeRating = (sharpe: number) => {
    if (sharpe >= 2) return { text: '优秀', color: 'success' as const };
    if (sharpe >= 1) return { text: '良好', color: 'primary' as const };
    if (sharpe >= 0.5) return { text: '一般', color: 'warning' as const };
    return { text: '较差', color: 'error' as const };
  };

  if (loading) {
    return (
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2 }}>
        {Array.from({ length: 8 }).map((_, index) => (
          <Card key={index}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Skeleton variant="circular" width={32} height={32} sx={{ mx: 'auto', mb: 2 }} />
              <Skeleton variant="text" width="60%" sx={{ mx: 'auto', mb: 1 }} />
              <Skeleton variant="text" width="40%" sx={{ mx: 'auto' }} />
            </CardContent>
          </Card>
        ))}
      </Box>
    );
  }

  if (!backtestData) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 128, color: 'text.secondary' }}>
            <AlertTriangle size={32} style={{ marginRight: 8 }} />
            <Typography>暂无回测数据</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  const sharpeRating = getSharpeRating(metrics.sharpeRatio);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 核心指标卡片 */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2 }}>
        {/* 总收益率 */}
        <Card sx={{ '&:hover': { boxShadow: 4 }, transition: 'box-shadow 0.3s' }}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              {getReturnIcon(metrics.totalReturn)}
            </Box>
            <Tooltip title="策略在整个回测期间的总收益率">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  总收益率
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h4" sx={{ fontWeight: 600, color: getReturnColor(metrics.totalReturn) }}>
              {metrics.totalReturn >= 0 ? '+' : ''}{metrics.totalReturn.toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        {/* 年化收益率 */}
        <Card sx={{ '&:hover': { boxShadow: 4 }, transition: 'box-shadow 0.3s' }}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <BarChart3 size={20} color="#1976d2" />
            </Box>
            <Tooltip title="将总收益率按年化计算的收益率">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  年化收益率
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h4" sx={{ fontWeight: 600, color: getReturnColor(metrics.annualizedReturn) }}>
              {metrics.annualizedReturn >= 0 ? '+' : ''}{metrics.annualizedReturn.toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        {/* 夏普比率 */}
        <Card sx={{ '&:hover': { boxShadow: 4 }, transition: 'box-shadow 0.3s' }}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <Target size={20} color="#9c27b0" />
            </Box>
            <Tooltip title="衡量风险调整后收益的指标，数值越高越好">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  夏普比率
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                {metrics.sharpeRatio.toFixed(3)}
              </Typography>
              <Chip label={sharpeRating.text} color={sharpeRating.color} size="small" />
            </Box>
          </CardContent>
        </Card>

        {/* 最大回撤 */}
        <Card sx={{ '&:hover': { boxShadow: 4 }, transition: 'box-shadow 0.3s' }}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <TrendingDown size={20} color="#d32f2f" />
            </Box>
            <Tooltip title="策略在回测期间的最大亏损幅度">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  最大回撤
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
              -{Math.abs(metrics.maxDrawdown).toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* 次要指标卡片 */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2 }}>
        {/* 波动率 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <Activity size={20} color="#ed6c02" />
            </Box>
            <Tooltip title="策略收益的波动程度，数值越低越稳定">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  波动率
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {metrics.volatility.toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        {/* 胜率 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <Target size={20} color="#2e7d32" />
            </Box>
            <Tooltip title="盈利交易占总交易次数的比例">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  胜率
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {metrics.winRate.toFixed(1)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.winRate}
                color={metrics.winRate >= 50 ? 'success' : 'warning'}
                sx={{ height: 6, borderRadius: 3 }}
              />
            </Box>
          </CardContent>
        </Card>

        {/* 交易次数 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <BarChart3 size={20} color="#1976d2" />
            </Box>
            <Tooltip title="回测期间的总交易次数">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  交易次数
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {metrics.totalTrades}
            </Typography>
          </CardContent>
        </Card>

        {/* 盈亏比 */}
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              <DollarSign size={20} color="#9c27b0" />
            </Box>
            <Tooltip title="平均盈利交易与平均亏损交易的比值">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1, cursor: 'help' }}>
                <Typography variant="body2" color="text.secondary">
                  盈亏比
                </Typography>
                <Info size={12} style={{ marginLeft: 4 }} />
              </Box>
            </Tooltip>
            <Typography
              variant="h5"
              sx={{ fontWeight: 600, color: metrics.profitFactor >= 1 ? 'success.main' : 'error.main' }}
            >
              {metrics.profitFactor.toFixed(2)}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* 扩展统计：交易分布与时间分段摘要（如果有数据才显示） */}
      {(metrics.tradePnlMean !== 0 ||
        metrics.tradePnlMedian !== 0 ||
        metrics.monthlyReturnMean !== 0 ||
        metrics.stocksTraded !== 0) && (
        <Card>
          <CardHeader title="交易与时间分段摘要" />
          <CardContent>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' },
                gap: 2,
              }}
            >
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  平均单笔盈亏
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 600,
                    color: metrics.tradePnlMean >= 0 ? 'success.main' : 'error.main',
                  }}
                >
                  {metrics.tradePnlMean >= 0 ? '+' : ''}
                  {metrics.tradePnlMean.toFixed(2)}%
                </Typography>
              </Box>

              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  单笔盈亏中位数
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 600,
                    color: metrics.tradePnlMedian >= 0 ? 'success.main' : 'error.main',
                  }}
                >
                  {metrics.tradePnlMedian >= 0 ? '+' : ''}
                  {metrics.tradePnlMedian.toFixed(2)}%
                </Typography>
              </Box>

              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  月度平均收益 / 波动
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {metrics.monthlyReturnMean.toFixed(2)}% ± {metrics.monthlyReturnStd.toFixed(2)}%
                </Typography>
              </Box>

              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  交易股票数 / 正负月份
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {metrics.stocksTraded}
                  <Typography
                    component="span"
                    variant="body2"
                    color="text.secondary"
                    sx={{ ml: 0.5 }}
                  >
                    （{metrics.positiveMonths} 正 / {metrics.negativeMonths} 负）
                  </Typography>
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* 风险评估总结 */}
      <Card>
        <CardHeader title="风险评估总结" />
        <CardContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
            <Box sx={{ textAlign: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    mr: 1,
                    bgcolor:
                      metrics.sharpeRatio >= 1
                        ? 'success.main'
                        : metrics.sharpeRatio >= 0.5
                        ? 'warning.main'
                        : 'error.main',
                  }}
                />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  风险调整收益
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {metrics.sharpeRatio >= 1
                  ? '良好'
                  : metrics.sharpeRatio >= 0.5
                  ? '一般'
                  : '需要改进'}
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    mr: 1,
                    bgcolor:
                      Math.abs(metrics.maxDrawdown) <= 10
                        ? 'success.main'
                        : Math.abs(metrics.maxDrawdown) <= 20
                        ? 'warning.main'
                        : 'error.main',
                  }}
                />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  回撤控制
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {Math.abs(metrics.maxDrawdown) <= 10
                  ? '优秀'
                  : Math.abs(metrics.maxDrawdown) <= 20
                  ? '良好'
                  : '需要改进'}
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    mr: 1,
                    bgcolor:
                      metrics.winRate >= 50
                        ? 'success.main'
                        : metrics.winRate >= 40
                        ? 'warning.main'
                        : 'error.main',
                  }}
                />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  交易胜率
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {metrics.winRate >= 50
                  ? '良好'
                  : metrics.winRate >= 40
                  ? '一般'
                  : '需要改进'}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
