/**
 * 统计卡片组件
 */

import React from 'react';
import { Box, Card, CardContent, Typography } from '@mui/material';
import {
  Target,
  TrendingUp,
  Award,
  TrendingDown,
} from 'lucide-react';
import { formatCurrency, formatPercent } from '@/utils/backtest/formatters';

interface StatisticsCardsProps {
  statistics: {
    totalStocks: number;
    profitableStocks: number;
    totalReturn: number;
    avgWinRate: number;
    avgHoldingPeriod: number;
    bestPerformer: any;
    worstPerformer: any;
  };
}

export const StatisticsCards: React.FC<StatisticsCardsProps> = ({ statistics }) => {
  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
        gap: 2,
      }}
    >
      <Card>
        <CardContent sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Target size={20} color="#1976d2" />
            <Box>
              <Typography variant="caption" color="text.secondary">
                持仓股票
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {statistics.totalStocks}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUp size={20} color="#2e7d32" />
            <Box>
              <Typography variant="caption" color="text.secondary">
                盈利股票
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                {statistics.profitableStocks}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                ({((statistics.profitableStocks / statistics.totalStocks) * 100).toFixed(1)}%)
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Award size={20} color="#9c27b0" />
            <Box>
              <Typography variant="caption" color="text.secondary">
                平均胜率
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {formatPercent(statistics.avgWinRate)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUp size={20} color="#ed6c02" />
            <Box>
              <Typography variant="caption" color="text.secondary">
                总收益
              </Typography>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 600,
                  color: statistics.totalReturn >= 0 ? 'success.main' : 'error.main',
                }}
              >
                {formatCurrency(statistics.totalReturn)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};
