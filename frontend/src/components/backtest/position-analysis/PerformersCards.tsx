/**
 * 最佳和最差表现者卡片组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { Award, TrendingDown } from 'lucide-react';
import { formatCurrency, formatPercent } from '@/utils/backtest/formatters';
import { PositionData } from '@/utils/backtest/positionDataUtils';

interface PerformersCardsProps {
  bestPerformer: PositionData | null;
  worstPerformer: PositionData | null;
}

export const PerformersCards: React.FC<PerformersCardsProps> = ({
  bestPerformer,
  worstPerformer,
}) => {
  if (!bestPerformer || !worstPerformer) {
    return null;
  }

  return (
    <Box
      sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}
    >
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Award size={20} color="#2e7d32" />
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                最佳表现
              </Typography>
            </Box>
          }
        />
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="h6" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                {bestPerformer.stock_code}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {bestPerformer.stock_name}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'right' }}>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                {formatCurrency(bestPerformer.total_return)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                胜率: {formatPercent(bestPerformer.win_rate)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TrendingDown size={20} color="#d32f2f" />
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                最差表现
              </Typography>
            </Box>
          }
        />
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="h6" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                {worstPerformer.stock_code}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {worstPerformer.stock_name}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'right' }}>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main' }}>
                {formatCurrency(worstPerformer.total_return)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                胜率: {formatPercent(worstPerformer.win_rate)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};
