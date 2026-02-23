/**
 * 回测结果图表组件
 * 显示策略回测的收益曲线、交易记录和性能指标
 */

'use client';

import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import {
  Card,
  CardContent,
  CardHeader,
  Chip,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Box,
  Typography,
  TableContainer,
  Paper,
} from '@mui/material';
import { TrendingUp, TrendingDown, DollarSign, BarChart3, AlertCircle } from 'lucide-react';

interface BacktestChartProps {
  stockCode: string;
  backtestData?: any;
}

interface TradeRecord {
  date: string;
  action: 'buy' | 'sell';
  price: number;
  quantity: number;
  pnl: number;
}

interface BacktestMetrics {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
}

export default function BacktestChart({ stockCode, backtestData }: BacktestChartProps) {
  const equityChartRef = useRef<HTMLDivElement>(null);
  const drawdownChartRef = useRef<HTMLDivElement>(null);
  const equityChartInstance = useRef<echarts.ECharts | null>(null);
  const drawdownChartInstance = useRef<echarts.ECharts | null>(null);

  // 处理回测数据（优先使用真实数据，否则生成模拟数据）
  const processBacktestData = () => {
    // 调试日志
    console.log('BacktestChart - backtestData:', backtestData);

    // 如果提供了真实回测数据，使用真实数据
    // 检查多种可能的数据格式
    const hasRealData =
      backtestData &&
      ((backtestData.portfolio && backtestData.risk_metrics) ||
        (backtestData.equity_curve && backtestData.equity_curve.length > 0) ||
        backtestData.total_return !== undefined ||
        backtestData.sharpe_ratio !== undefined);

    if (hasRealData) {
      console.log('BacktestChart - 使用真实回测数据');
      // 兼容多种数据格式
      const portfolio = backtestData.portfolio || {
        initial_cash: backtestData.initial_cash || 100000,
        final_value: backtestData.final_value || backtestData.initial_cash || 100000,
        total_return: backtestData.total_return || 0,
        annualized_return: backtestData.annualized_return || 0,
      };

      const riskMetrics = backtestData.risk_metrics || {
        volatility: backtestData.volatility || 0,
        sharpe_ratio: backtestData.sharpe_ratio || 0,
        max_drawdown: backtestData.max_drawdown || 0,
      };

      const tradingStats = backtestData.trading_stats || {
        total_trades: backtestData.total_trades || 0,
        win_rate: backtestData.win_rate || 0,
        profit_factor: backtestData.profit_factor || 0,
      };

      const tradeHistory = backtestData.trade_history || [];

      // 从真实数据构建图表数据
      const equityCurve = backtestData.equity_curve || [];
      const drawdownCurve = backtestData.drawdown_curve || [];
      const dates = backtestData.dates || [];

      // 转换交易记录格式
      const trades: TradeRecord[] = tradeHistory.map((trade: any) => ({
        date: trade.date || trade.trade_date,
        action: trade.action || (trade.side === 'buy' ? 'buy' : 'sell'),
        price: trade.price || trade.execution_price || 0,
        quantity: trade.quantity || trade.shares || 0,
        pnl: trade.pnl || trade.profit_loss || 0,
      }));

      const metrics: BacktestMetrics = {
        total_return: (portfolio.total_return || 0) * 100,
        sharpe_ratio: riskMetrics.sharpe_ratio || 0,
        max_drawdown: (riskMetrics.max_drawdown || 0) * 100,
        win_rate: (tradingStats.win_rate || 0) * 100,
        total_trades: tradingStats.total_trades || 0,
        profit_factor: tradingStats.profit_factor || 0,
      };

      return {
        dates:
          dates.length > 0
            ? dates
            : equityCurve.map((_: any, i: number) => {
                const date = new Date();
                date.setDate(date.getDate() - (equityCurve.length - i));
                return date.toISOString().split('T')[0];
              }),
        equityCurve: equityCurve.length > 0 ? equityCurve : [portfolio.initial_cash || 100000],
        drawdownCurve: drawdownCurve.length > 0 ? drawdownCurve : [0],
        trades: trades.slice(-10), // 最近10笔交易
        metrics,
      };
    }

    // 否则生成模拟回测数据
    console.log('BacktestChart - 未找到真实数据，生成模拟数据');
    return generateMockBacktestData();
  };

  // 生成模拟回测数据
  const generateMockBacktestData = () => {
    const days = 252; // 一年交易日
    const initialCapital = 100000;
    const equityCurve = [initialCapital];
    const drawdownCurve = [0];
    const dates = [];
    const trades: TradeRecord[] = [];

    let currentEquity = initialCapital;
    let peak = initialCapital;
    let position = 0;
    let lastPrice = 100;

    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      dates.push(date.toISOString().split('T')[0]);

      // 模拟价格变动
      const priceChange = (Math.random() - 0.5) * 0.04;
      const currentPrice = lastPrice * (1 + priceChange);

      // 模拟交易信号（简单策略）
      if (position === 0 && Math.random() < 0.05) {
        // 买入信号
        const quantity = Math.floor((currentEquity * 0.1) / currentPrice);
        if (quantity > 0) {
          position = quantity;
          trades.push({
            date: dates[i],
            action: 'buy',
            price: currentPrice,
            quantity,
            pnl: 0,
          });
        }
      } else if (position > 0 && Math.random() < 0.03) {
        // 卖出信号
        const sellValue = position * currentPrice;
        const buyValue = position * trades[trades.length - 1].price;
        const pnl = sellValue - buyValue;

        trades.push({
          date: dates[i],
          action: 'sell',
          price: currentPrice,
          quantity: position,
          pnl,
        });

        currentEquity += pnl;
        position = 0;
      }

      // 计算当前权益
      if (position > 0) {
        const positionValue = position * currentPrice;
        const cash = currentEquity - position * trades[trades.length - 1].price;
        currentEquity = cash + positionValue;
      }

      equityCurve.push(currentEquity);

      // 计算回撤
      if (currentEquity > peak) {
        peak = currentEquity;
      }
      const drawdown = ((peak - currentEquity) / peak) * 100;
      drawdownCurve.push(drawdown);

      lastPrice = currentPrice;
    }

    // 计算性能指标
    const totalReturn = ((currentEquity - initialCapital) / initialCapital) * 100;
    const returns = equityCurve
      .slice(1)
      .map((equity, i) => (equity - equityCurve[i]) / equityCurve[i]);
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const returnStd = Math.sqrt(
      returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length
    );
    const sharpeRatio = (avgReturn / returnStd) * Math.sqrt(252);
    const maxDrawdown = Math.max(...drawdownCurve);

    const winningTrades = trades.filter(t => t.action === 'sell' && t.pnl > 0).length;
    const totalSellTrades = trades.filter(t => t.action === 'sell').length;
    const winRate = totalSellTrades > 0 ? (winningTrades / totalSellTrades) * 100 : 0;

    const grossProfit = trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;

    const metrics: BacktestMetrics = {
      total_return: totalReturn,
      sharpe_ratio: sharpeRatio,
      max_drawdown: maxDrawdown,
      win_rate: winRate,
      total_trades: totalSellTrades,
      profit_factor: profitFactor,
    };

    return {
      dates,
      equityCurve: equityCurve.slice(1),
      drawdownCurve: drawdownCurve.slice(1),
      trades: trades.slice(-10), // 最近10笔交易
      metrics,
    };
  };

  useEffect(() => {
    const data = processBacktestData();

    // 权益曲线图表
    if (equityChartRef.current) {
      if (equityChartInstance.current) {
        equityChartInstance.current.dispose();
      }

      equityChartInstance.current = echarts.init(equityChartRef.current);

      const equityOption = {
        title: {
          text: '权益曲线',
          left: 'center',
          textStyle: {
            fontSize: 14,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'axis',
          formatter: function (params: any) {
            const value = params[0].value;
            const date = params[0].axisValue;
            return `${date}<br/>权益: ¥${value.toLocaleString()}`;
          },
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: '15%',
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: data.dates,
          axisLabel: {
            formatter: function (value: string) {
              return new Date(value).toLocaleDateString('zh-CN', {
                month: 'short',
                day: 'numeric',
              });
            },
          },
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: function (value: number) {
              return `¥${(value / 1000).toFixed(0)}K`;
            },
          },
        },
        series: [
          {
            name: '权益',
            type: 'line',
            data: data.equityCurve,
            lineStyle: {
              color: '#10b981',
              width: 2,
            },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                {
                  offset: 0,
                  color: 'rgba(16, 185, 129, 0.3)',
                },
                {
                  offset: 1,
                  color: 'rgba(16, 185, 129, 0.1)',
                },
              ]),
            },
            symbol: 'none',
          },
        ],
      };

      equityChartInstance.current.setOption(equityOption);
    }

    // 回撤图表
    if (drawdownChartRef.current) {
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.dispose();
      }

      drawdownChartInstance.current = echarts.init(drawdownChartRef.current);

      const drawdownOption = {
        title: {
          text: '回撤曲线',
          left: 'center',
          textStyle: {
            fontSize: 14,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'axis',
          formatter: function (params: any) {
            const value = params[0].value;
            const date = params[0].axisValue;
            return `${date}<br/>回撤: ${value.toFixed(2)}%`;
          },
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: '15%',
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: data.dates,
          axisLabel: {
            formatter: function (value: string) {
              return new Date(value).toLocaleDateString('zh-CN', {
                month: 'short',
                day: 'numeric',
              });
            },
          },
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: '{value}%',
          },
          max: 0,
          min: function (value: any) {
            return Math.floor(value.min * 1.1);
          },
        },
        series: [
          {
            name: '回撤',
            type: 'line',
            data: data.drawdownCurve.map((d: number) => -d),
            lineStyle: {
              color: '#ef4444',
              width: 2,
            },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                {
                  offset: 0,
                  color: 'rgba(239, 68, 68, 0.3)',
                },
                {
                  offset: 1,
                  color: 'rgba(239, 68, 68, 0.1)',
                },
              ]),
            },
            symbol: 'none',
          },
        ],
      };

      drawdownChartInstance.current.setOption(drawdownOption);
    }

    // 响应式调整
    const handleResize = () => {
      if (equityChartInstance.current) {
        equityChartInstance.current.resize();
      }
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (equityChartInstance.current) {
        equityChartInstance.current.dispose();
      }
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.dispose();
      }
    };
  }, [stockCode, backtestData]);

  const data = processBacktestData();

  if (!backtestData && !data) {
    return (
      <Card>
        <CardContent>
          <Box
            sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}
          >
            <AlertCircle size={32} style={{ marginRight: 8 }} />
            <Typography variant="body2" color="text.secondary">
              暂无回测数据
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 性能指标概览 */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2, 1fr)', sm: 'repeat(3, 1fr)', lg: 'repeat(6, 1fr)' },
          gap: { xs: 1, sm: 2 },
        }}
      >
        <Card>
          <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
            <DollarSign size={24} color="#1976d2" style={{ margin: '0 auto 8px' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
              总收益率
            </Typography>
            <Typography
              sx={{
                fontWeight: 600,
                color: data.metrics.total_return >= 0 ? 'success.main' : 'error.main',
                fontSize: { xs: '1rem', sm: '1.25rem' },
                wordBreak: 'break-word',
              }}
            >
              {data.metrics.total_return.toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
            <TrendingUp size={24} color="#9c27b0" style={{ margin: '0 auto 8px' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
              夏���比率
            </Typography>
            <Typography sx={{ fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' }, wordBreak: 'break-word' }}>
              {data.metrics.sharpe_ratio.toFixed(3)}
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
            <TrendingDown size={24} color="#d32f2f" style={{ margin: '0 auto 8px' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
              最大回撤
            </Typography>
            <Typography sx={{ fontWeight: 600, color: 'error.main', fontSize: { xs: '1rem', sm: '1.25rem' }, wordBreak: 'break-word' }}>
              {data.metrics.max_drawdown.toFixed(2)}%
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
            <BarChart3 size={24} color="#ed6c02" style={{ margin: '0 auto 8px' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
              胜率
            </Typography>
            <Typography sx={{ fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' }, wordBreak: 'break-word' }}>
              {data.metrics.win_rate.toFixed(1)}%
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
            <Typography sx={{ fontWeight: 600, color: 'primary.main', fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2rem' }, mb: 0.5, wordBreak: 'break-word' }}>
              {data.metrics.total_trades}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
              总交易次数
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ textAlign: 'center', p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
            <Typography sx={{ fontWeight: 600, color: 'secondary.main', fontSize: { xs: '1rem', sm: '1.25rem' }, mb: 0.5, wordBreak: 'break-word' }}>
              {data.metrics.profit_factor.toFixed(2)}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
              盈亏比
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* 图表区域 */}
      <Box
        sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: { xs: 2, sm: 3 } }}
      >
        <Card>
          <CardContent sx={{ p: { xs: 1, sm: 2 }, overflow: 'hidden' }}>
            <Box sx={{ overflowX: 'auto' }}>
              <Box ref={equityChartRef} sx={{ height: { xs: 250, sm: 300 }, width: '100%', minWidth: 350 }} />
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent sx={{ p: { xs: 1, sm: 2 }, overflow: 'hidden' }}>
            <Box sx={{ overflowX: 'auto' }}>
              <Box ref={drawdownChartRef} sx={{ height: { xs: 250, sm: 300 }, width: '100%', minWidth: 350 }} />
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* 交易记录 */}
      <Card>
        <CardHeader title="最近交易记录" titleTypographyProps={{ sx: { fontSize: { xs: '1rem', sm: '1.25rem' } } }} />
        <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
          <TableContainer component={Paper} variant="outlined" sx={{ overflowX: 'auto' }}>
            <Table size="small" sx={{ minWidth: 450 }}>
              <TableHead>
                <TableRow>
                  <TableCell>日期</TableCell>
                  <TableCell>操作</TableCell>
                  <TableCell>价格</TableCell>
                  <TableCell>数量</TableCell>
                  <TableCell>盈亏</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.trades.map((trade, index) => (
                  <TableRow key={index} hover>
                    <TableCell>{new Date(trade.date).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <Chip
                        label={trade.action === 'buy' ? '买入' : '卖出'}
                        color={trade.action === 'buy' ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>¥{trade.price.toFixed(2)}</TableCell>
                    <TableCell>{trade.quantity}</TableCell>
                    <TableCell>
                      {trade.action === 'sell' && (
                        <Typography
                          variant="body2"
                          sx={{ color: trade.pnl >= 0 ? 'success.main' : 'error.main' }}
                        >
                          ¥{trade.pnl.toFixed(2)}
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}
