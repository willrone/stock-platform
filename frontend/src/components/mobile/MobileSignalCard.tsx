'use client';

import React from 'react';
import { 
  Card, 
  CardContent, 
  Box, 
  Typography, 
  Chip,
} from '@mui/material';
import { TrendingUp, TrendingDown, Calendar, DollarSign } from 'lucide-react';

interface SignalRow {
  stock_code: string;
  stock_name?: string;
  signal: string;
  price: number;
  change_percent?: number;
  signal_time: string;
  strategy?: string;
}

interface MobileSignalCardProps {
  signal: SignalRow;
}

export const MobileSignalCard: React.FC<MobileSignalCardProps> = ({ signal }) => {
  const getSignalColor = (sig: string): 'success' | 'error' | 'warning' | 'default' => {
    const s = sig.toLowerCase();
    if (s.includes('buy') || s.includes('买入') || s === '1') return 'success';
    if (s.includes('sell') || s.includes('卖出') || s === '-1') return 'error';
    if (s.includes('hold') || s.includes('持有') || s === '0') return 'warning';
    return 'default';
  };

  const getSignalText = (sig: string) => {
    const s = sig.toLowerCase();
    if (s.includes('buy') || s === '1') return '买入';
    if (s.includes('sell') || s === '-1') return '卖出';
    if (s.includes('hold') || s === '0') return '持有';
    return sig;
  };

  const getSignalIcon = (sig: string) => {
    const s = sig.toLowerCase();
    if (s.includes('buy') || s === '1') return <TrendingUp size={18} />;
    if (s.includes('sell') || s === '-1') return <TrendingDown size={18} />;
    return null;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const signalColor = getSignalColor(signal.signal);
  const isPositive = signal.change_percent && signal.change_percent > 0;

  return (
    <Card 
      sx={{ 
        mb: 1.5, 
        borderRadius: 2,
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
        borderLeft: 4,
        borderLeftColor: signalColor + '.main',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* 标题行 */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Box>
            <Typography variant="body1" fontWeight={600} sx={{ fontSize: '0.95rem' }}>
              {signal.stock_code}
            </Typography>
            {signal.stock_name && (
              <Typography variant="caption" color="text.secondary">
                {signal.stock_name}
              </Typography>
            )}
          </Box>
          <Chip 
            label={getSignalText(signal.signal)}
            color={signalColor}
            size="small"
            icon={getSignalIcon(signal.signal) as any}
            sx={{ 
              fontWeight: 600,
              fontSize: '0.75rem',
            }}
          />
        </Box>

        {/* 价格和涨跌 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <DollarSign size={14} color="#666" />
            <Typography variant="body2" fontWeight={600}>
              ¥{signal.price.toFixed(2)}
            </Typography>
          </Box>
          
          {signal.change_percent != null && (
            <Typography 
              variant="body2" 
              fontWeight={600}
              color={isPositive ? 'success.main' : 'error.main'}
            >
              {isPositive ? '+' : ''}{signal.change_percent.toFixed(2)}%
            </Typography>
          )}
        </Box>

        {/* 时间和策略 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Calendar size={14} color="#666" />
            <Typography variant="caption" color="text.secondary">
              {formatDate(signal.signal_time)}
            </Typography>
          </Box>
          
          {signal.strategy && (
            <>
              <Typography variant="caption" color="text.secondary">•</Typography>
              <Typography variant="caption" color="text.secondary">
                {signal.strategy}
              </Typography>
            </>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};
