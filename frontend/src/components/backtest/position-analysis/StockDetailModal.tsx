/**
 * 股票详情模态框组件
 */

import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  Box,
  Typography,
  LinearProgress,
} from '@mui/material';
import { formatCurrency, formatPercent } from '@/utils/backtest/formatters';
import { PositionData } from '@/utils/backtest/positionDataUtils';

interface StockDetailModalProps {
  open: boolean;
  onClose: () => void;
  stock: PositionData | null;
}

export const StockDetailModal: React.FC<StockDetailModalProps> = ({ open, onClose, stock }) => {
  if (!stock) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>股票详细分析</DialogTitle>
      <DialogContent sx={{ p: { xs: 2, md: 3 } }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ textAlign: 'center', borderBottom: 1, borderColor: 'divider', pb: 2 }}>
            <Typography variant="h4" sx={{ fontFamily: 'monospace', fontWeight: 600, fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
              {stock.stock_code}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {stock.stock_name}
            </Typography>
          </Box>

          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                总收益
              </Typography>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 600,
                  fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' },
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                  color:
                    stock.total_return > 0
                      ? 'success.main'
                      : stock.total_return < 0
                        ? 'error.main'
                        : 'text.secondary',
                }}
              >
                {formatCurrency(stock.total_return)}
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                胜率
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {formatPercent(stock.win_rate)}
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                交易次数
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {stock.trade_count}
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                平均持仓期
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2.125rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {stock.avg_holding_period} 天
              </Typography>
            </Box>
          </Box>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 2,
              pt: 2,
              borderTop: 1,
              borderColor: 'divider',
            }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                盈利交易
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '1.1rem', md: '1.5rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {stock.winning_trades}
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                亏损交易
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main', fontSize: { xs: '1.1rem', md: '1.5rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                {stock.losing_trades}
              </Typography>
            </Box>
          </Box>

          {/* 扩展信息 */}
          {(stock.avg_win !== undefined ||
            stock.avg_loss !== undefined ||
            stock.profit_factor !== undefined) && (
            <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                盈亏分析
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                {stock.avg_win !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      平均盈利
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      {formatCurrency(stock.avg_win)}
                    </Typography>
                  </Box>
                )}
                {stock.avg_loss !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      平均亏损
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main', fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      {formatCurrency(stock.avg_loss)}
                    </Typography>
                  </Box>
                )}
                {stock.largest_win !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      最大盈利
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      {formatCurrency(stock.largest_win)}
                    </Typography>
                  </Box>
                )}
                {stock.largest_loss !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      最大亏损
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main', fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      {formatCurrency(stock.largest_loss)}
                    </Typography>
                  </Box>
                )}
                {stock.profit_factor !== undefined && (
                  <Box sx={{ gridColumn: 'span 2' }}>
                    <Typography variant="caption" color="text.secondary">
                      盈亏比
                    </Typography>
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 600,
                        fontSize: { xs: '0.95rem', md: '1.25rem' },
                        overflow: 'hidden',
                        wordBreak: 'break-word',
                        color: stock.profit_factor >= 1 ? 'success.main' : 'error.main',
                      }}
                    >
                      {stock.profit_factor.toFixed(2)}
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          )}

          {/* 价格分析 */}
          {(stock.avg_buy_price !== undefined ||
            stock.avg_sell_price !== undefined ||
            stock.price_improvement !== undefined) && (
            <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                价格分析
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                {stock.avg_buy_price !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      平均买入价
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      ¥{stock.avg_buy_price.toFixed(2)}
                    </Typography>
                  </Box>
                )}
                {stock.avg_sell_price !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      平均卖出价
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      ¥{stock.avg_sell_price.toFixed(2)}
                    </Typography>
                  </Box>
                )}
                {stock.price_improvement !== undefined && (
                  <Box sx={{ gridColumn: 'span 2' }}>
                    <Typography variant="caption" color="text.secondary">
                      价格改善率
                    </Typography>
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 600,
                        fontSize: { xs: '0.95rem', md: '1.25rem' },
                        overflow: 'hidden',
                        wordBreak: 'break-word',
                        color: stock.price_improvement >= 0 ? 'success.main' : 'error.main',
                      }}
                    >
                      {(stock.price_improvement * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          )}

          {/* 持仓期详情 */}
          {(stock.max_holding_period !== undefined ||
            stock.min_holding_period !== undefined) && (
            <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                持仓期详情
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    平均持仓期
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                    {stock.avg_holding_period} 天
                  </Typography>
                </Box>
                {stock.max_holding_period !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      最长持仓期
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      {stock.max_holding_period} 天
                    </Typography>
                  </Box>
                )}
                {stock.min_holding_period !== undefined && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      最短持仓期
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' }, overflow: 'hidden', wordBreak: 'break-word' }}>
                      {stock.min_holding_period} 天
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          )}

          <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 1,
              }}
            >
              <Typography variant="caption" color="text.secondary">
                胜率进度
              </Typography>
              <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                {formatPercent(stock.win_rate)}
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={stock.win_rate * 100}
              color={stock.win_rate >= 0.5 ? 'success' : 'error'}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button variant="contained" color="primary" onClick={onClose}>
          关闭
        </Button>
      </DialogActions>
    </Dialog>
  );
};
