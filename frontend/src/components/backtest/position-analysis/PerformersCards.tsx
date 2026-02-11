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
          sx={{ pb: 0, px: { xs: 1.5, md: 2 }, pt: { xs: 1.5, md: 2 } }}
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Award size={20} color="#2e7d32" />
              <Typography sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '1rem', md: '1.25rem' } }}>
                最佳表现
              </Typography>
            </Box>
          }
        />
        <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
            <Box sx={{ overflow: 'hidden', minWidth: 0 }}>
              <Typography sx={{ fontFamily: 'monospace', fontWeight: 600, fontSize: { xs: '1rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {bestPerformer.stock_code}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {bestPerformer.stock_name}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'right', overflow: 'hidden', flexShrink: 0 }}>
              <Typography sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '1.25rem', md: '1.5rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
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
          sx={{ pb: 0, px: { xs: 1.5, md: 2 }, pt: { xs: 1.5, md: 2 } }}
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TrendingDown size={20} color="#d32f2f" />
              <Typography sx={{ fontWeight: 600, color: 'error.main', fontSize: { xs: '1rem', md: '1.25rem' } }}>
                最差表现
              </Typography>
            </Box>
          }
        />
        <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
            <Box sx={{ overflow: 'hidden', minWidth: 0 }}>
              <Typography sx={{ fontFamily: 'monospace', fontWeight: 600, fontSize: { xs: '1rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {worstPerformer.stock_code}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {worstPerformer.stock_name}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'right', overflow: 'hidden', flexShrink: 0 }}>
              <Typography sx={{ fontWeight: 600, color: 'error.main', fontSize: { xs: '1.25rem', md: '1.5rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
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
