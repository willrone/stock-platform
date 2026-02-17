/**
 * 统计卡片组件
 */

import React from 'react';
import { Box, Card, CardContent, Typography } from '@mui/material';
import { Target, TrendingUp, Award, TrendingDown } from 'lucide-react';
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
        <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Target size={20} color="#1976d2" />
            <Box sx={{ overflow: 'hidden' }}>
              <Typography variant="caption" color="text.secondary">
                持仓股票
              </Typography>
              <Typography
                sx={{
                  fontWeight: 600,
                  fontSize: { xs: '1.25rem', md: '1.5rem' },
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
                {statistics.totalStocks}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUp size={20} color="#2e7d32" />
            <Box sx={{ overflow: 'hidden' }}>
              <Typography variant="caption" color="text.secondary">
                盈利股票
              </Typography>
              <Typography
                sx={{
                  fontWeight: 600,
                  color: 'success.main',
                  fontSize: { xs: '1.25rem', md: '1.5rem' },
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
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
        <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Award size={20} color="#9c27b0" />
            <Box sx={{ overflow: 'hidden' }}>
              <Typography variant="caption" color="text.secondary">
                平均胜率
              </Typography>
              <Typography
                sx={{
                  fontWeight: 600,
                  fontSize: { xs: '1.25rem', md: '1.5rem' },
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
                {formatPercent(statistics.avgWinRate)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUp size={20} color="#ed6c02" />
            <Box sx={{ overflow: 'hidden' }}>
              <Typography variant="caption" color="text.secondary">
                总收益
              </Typography>
              <Typography
                sx={{
                  fontWeight: 600,
                  color: statistics.totalReturn >= 0 ? 'success.main' : 'error.main',
                  fontSize: { xs: '1.25rem', md: '1.5rem' },
                  overflow: 'hidden',
                  wordBreak: 'break-word',
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
