/**
 * TradingView图表组件
 * 使用lightweight-charts库显示股票价格走势
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  SeriesMarkerPosition,
  SeriesMarkerShape,
} from 'lightweight-charts';
import {
  Card,
  CardContent,
  Button,
  ButtonGroup,
  CircularProgress,
  Box,
  Typography,
} from '@mui/material';
import { TrendingUp, BarChart3, AlertCircle } from 'lucide-react';
import { DataService } from '@/services/dataService';

import { PredictionResult } from '../../services/taskService';

interface TradingViewChartProps {
  stockCode: string;
  height?: number;
  prediction?: PredictionResult;
  startDate?: string;
  endDate?: string;
  trades?: Array<{
    trade_id?: string;
    stock_code?: string;
    action: 'BUY' | 'SELL';
    price?: number;
    timestamp: string;
  }>;
  signals?: Array<{
    signal_id?: string;
    stock_code?: string;
    signal_type: 'BUY' | 'SELL';
    price: number;
    timestamp: string;
    executed?: boolean;
    strategy_name?: string;
    strategy_id?: string;
  }>;
  showSignals?: boolean;
  showTrades?: boolean;
}

interface PriceData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function TradingViewChart({
  stockCode,
  height = 400,
  prediction,
  startDate,
  endDate,
  trades,
  signals,
  showSignals = true,
  showTrades = true,
}: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M'>('1D');
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('candlestick');
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [internalShowSignals, setInternalShowSignals] = useState(showSignals);
  const [internalShowTrades, setInternalShowTrades] = useState(showTrades);

  // 获取股票数据
  const fetchStockData = async () => {
    if (!stockCode) {
      console.warn('[TradingViewChart] 股票代码为空，无法加载数据');
      setPriceData([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      // 处理日期格式
      const fallbackEnd = new Date();
      const fallbackStart = new Date('2020-01-01');

      let resolvedStart: Date;
      let resolvedEnd: Date;

      // 处理startDate：可能是ISO字符串或undefined
      if (startDate) {
        resolvedStart = new Date(startDate);
        if (isNaN(resolvedStart.getTime())) {
          console.warn(`[TradingViewChart] 无效的startDate格式: ${startDate}，使用默认值`);
          resolvedStart = fallbackStart;
        }
      } else {
        resolvedStart = fallbackStart;
      }

      // 处理endDate：可能是ISO字符串或undefined
      if (endDate) {
        resolvedEnd = new Date(endDate);
        if (isNaN(resolvedEnd.getTime())) {
          console.warn(`[TradingViewChart] 无效的endDate格式: ${endDate}，使用默认值`);
          resolvedEnd = fallbackEnd;
        }
      } else {
        resolvedEnd = fallbackEnd;
      }

      console.log(
        `[TradingViewChart] 开始获取股票数据: ${stockCode}, 时间范围: ${
          resolvedStart.toISOString().split('T')[0]
        } 至 ${resolvedEnd.toISOString().split('T')[0]}`
      );

      // 调用真实API获取数据
      const response = await DataService.getStockData(
        stockCode,
        resolvedStart.toISOString().split('T')[0],
        resolvedEnd.toISOString().split('T')[0]
      );

      console.log('[TradingViewChart] API响应:', response);

      // 转换数据格式
      // DataService.getStockData 返回格式: { stock_code, data: [...], last_updated }
      // 其中 data 字段已经是后端返回的数据数组
      const dataArray: any[] = Array.isArray(response?.data) ? response.data : [];

      console.log(`[TradingViewChart] 解析后的数据数组长度: ${dataArray.length}`, {
        responseType: typeof response,
        responseKeys: response ? Object.keys(response) : [],
        hasDataArray: Array.isArray(dataArray),
        dataArrayLength: dataArray.length,
        firstItem: dataArray[0],
      });

      if (dataArray.length > 0) {
        let formattedData: PriceData[] = dataArray
          .map((item: any) => ({
            time: item.date ? item.date.split('T')[0] : item.date, // 只取日期部分
            open: Number(item.open) || 0,
            high: Number(item.high) || 0,
            low: Number(item.low) || 0,
            close: Number(item.close) || 0,
            volume: Number(item.volume) || 0,
          }))
          .filter((item: PriceData) => item.time); // 过滤掉无效数据

        // 根据timeframe进行数据采样
        if (timeframe === '1W' && formattedData.length > 0) {
          // 周线：每周取最后一个交易日的数据
          const weeklyData: PriceData[] = [];
          let currentWeek = '';
          let lastItem: PriceData | null = null;

          for (const item of formattedData) {
            const date = new Date(item.time);
            const weekKey = `${date.getFullYear()}-W${getWeekNumber(date)}`;

            if (weekKey !== currentWeek) {
              if (currentWeek && lastItem) {
                // 保存上一周的最后一条数据
                weeklyData.push(lastItem);
              }
              currentWeek = weekKey;
            }
            lastItem = item;
          }
          // 添加最后一周的数据
          if (lastItem) {
            weeklyData.push(lastItem);
          }
          formattedData = weeklyData;
        } else if (timeframe === '1M' && formattedData.length > 0) {
          // 月线：每月取最后一个交易日的数据
          const monthlyData: PriceData[] = [];
          let currentMonth = '';
          let lastItem: PriceData | null = null;

          for (const item of formattedData) {
            const date = new Date(item.time);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(
              2,
              '0'
            )}`;

            if (monthKey !== currentMonth) {
              if (currentMonth && lastItem) {
                // 保存上一月的最后一条数据
                monthlyData.push(lastItem);
              }
              currentMonth = monthKey;
            }
            lastItem = item;
          }
          // 添加最后一月的数据
          if (lastItem) {
            monthlyData.push(lastItem);
          }
          formattedData = monthlyData;
        }

        // 按时间排序（确保数据按时间顺序）
        formattedData.sort((a, b) => a.time.localeCompare(b.time));

        console.log(
          `[TradingViewChart] 成功加载 ${formattedData.length} 条${
            timeframe === '1D' ? '日' : timeframe === '1W' ? '周' : '月'
          }线数据，时间范围: ${formattedData[0]?.time} 至 ${formattedData[formattedData.length - 1]
            ?.time}`
        );
        setPriceData(formattedData);
      } else {
        console.warn('[TradingViewChart] 未获取到股票数据，返回空数据。', {
          stockCode,
          startDate: resolvedStart.toISOString().split('T')[0],
          endDate: resolvedEnd.toISOString().split('T')[0],
          response,
          dataArrayLength: dataArray.length,
        });
        setPriceData([]);
      }
    } catch (error: any) {
      console.error('[TradingViewChart] 获取股票数据失败:', error);
      console.error('[TradingViewChart] 错误详情:', {
        message: error?.message,
        stack: error?.stack,
        response: error?.response,
        status: error?.status,
      });
      setPriceData([]);
      // 可以在这里添加用户友好的错误提示
    } finally {
      setLoading(false);
    }
  };

  // 获取ISO周数
  const getWeekNumber = (date: Date): number => {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  };

  // 添加买卖标记的辅助函数
  const addTradingMarkers = () => {
    if (!prediction || !candlestickSeriesRef.current || priceData.length === 0) {
      return;
    }

    const buyThreshold = 0.02; // 2%收益率阈值
    const confidenceThreshold = 0.6; // 60%置信度阈值

    // 买入信号
    if (
      prediction.predicted_direction > 0 &&
      prediction.predicted_return > buyThreshold &&
      prediction.confidence_score > confidenceThreshold
    ) {
      const lastData = priceData[priceData.length - 1];
      candlestickSeriesRef.current.createPriceLine({
        price: lastData.close,
        color: '#10b981',
        lineWidth: 2,
        lineStyle: 0, // 实线
        axisLabelVisible: true,
        title: `买入 - 预测上涨${(prediction.predicted_return * 100).toFixed(2)}%`,
      });
    }

    // 卖出信号
    if (
      prediction.predicted_direction < 0 &&
      prediction.predicted_return < -buyThreshold &&
      prediction.confidence_score > confidenceThreshold
    ) {
      const lastData = priceData[priceData.length - 1];
      candlestickSeriesRef.current.createPriceLine({
        price: lastData.close,
        color: '#ef4444',
        lineWidth: 2,
        lineStyle: 0, // 实线
        axisLabelVisible: true,
        title: `卖出 - 预测下跌${(Math.abs(prediction.predicted_return) * 100).toFixed(2)}%`,
      });
    }
  };

  const addTradeMarkers = () => {
    // 这个函数现在被addSignalMarkers取代，因为它会合并交易和信号标记
    // 保留函数签名以防其他地方调用，但实际逻辑在addSignalMarkers中
    if (candlestickSeriesRef.current) {
      addSignalMarkers();
    }
  };

  const addSignalMarkers = () => {
    if (!candlestickSeriesRef.current) {
      return;
    }

    // 合并交易标记和信号标记
    const allMarkers: any[] = [];

    // 为多策略信号准备颜色映射（按 strategy_name / strategy_id）
    // 常用策略固定颜色映射（保证视觉一致性和易于识别）
    const fixedColorMap: Record<string, string> = {
      // 基础技术策略
      moving_average: '#2196F3', // 蓝色
      rsi: '#FF9800', // 橙色
      macd: '#4CAF50', // 绿色

      // 高级技术分析策略
      bollinger: '#9C27B0', // 紫色
      stochastic: '#00BCD4', // 青色
      cci: '#FF5722', // 深橙红

      // 统计套利策略
      pairs_trading: '#795548', // 棕色
      mean_reversion: '#607D8B', // 蓝灰
      cointegration: '#E91E63', // 粉红

      // 因子投资策略
      value_factor: '#3F51B5', // 靛蓝
      momentum_factor: '#FFC107', // 琥珀
      low_volatility: '#009688', // 青绿
      multi_factor: '#F44336', // 红色
    };

    // 扩展调色板（用于未映射的策略，使用稳定哈希分配）
    const extendedPalette = [
      '#1f77b4',
      '#ff7f0e',
      '#2ca02c',
      '#d62728',
      '#9467bd',
      '#8c564b',
      '#e377c2',
      '#7f7f7f',
      '#bcbd22',
      '#17becf',
      '#aec7e8',
      '#ffbb78',
      '#98df8a',
      '#ff9896',
      '#c5b0d5',
      '#c49c94',
      '#f7b6d3',
      '#c7c7c7',
      '#dbdb8d',
      '#9edae5',
      '#6b6ecf',
      '#b5cf6b',
      '#bd9e39',
      '#e7ba52',
      '#637939',
    ];

    // 基于字符串生成稳定哈希值（用于未映射的策略）
    const stringHash = (str: string): number => {
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash = hash & hash; // Convert to 32bit integer
      }
      return Math.abs(hash);
    };

    const strategyColorMap: Record<string, string> = {};
    const strategyKeys: string[] = [];
    if (internalShowSignals && signals && signals.length > 0) {
      for (const s of signals) {
        const key = s.strategy_id || s.strategy_name;
        if (key && !strategyKeys.includes(key)) {
          strategyKeys.push(key);
        }
      }
    }

    // 为每个策略分配颜色：优先使用固定映射，否则使用稳定哈希
    strategyKeys.forEach(key => {
      const normalizedKey = key.toLowerCase().trim();
      if (fixedColorMap[normalizedKey]) {
        strategyColorMap[key] = fixedColorMap[normalizedKey];
      } else {
        // 使用哈希值从扩展调色板中选择颜色，确保同一策略总是得到相同颜色
        const hash = stringHash(normalizedKey);
        strategyColorMap[key] = extendedPalette[hash % extendedPalette.length];
      }
    });

    // 先添加交易标记（如果显示）
    if (internalShowTrades && trades && trades.length > 0) {
      const tradeMarkers = trades
        .filter(trade => trade.timestamp)
        .map(trade => ({
          time: trade.timestamp.split('T')[0],
          position: (trade.action === 'BUY' ? 'belowBar' : 'aboveBar') as SeriesMarkerPosition,
          color: trade.action === 'BUY' ? '#10b981' : '#ef4444',
          shape: (trade.action === 'BUY' ? 'arrowUp' : 'arrowDown') as SeriesMarkerShape,
          text: trade.action === 'BUY' ? '买入' : '卖出',
        }));
      allMarkers.push(...tradeMarkers);
    }

    // 添加信号标记（如果显示）
    if (internalShowSignals && signals && signals.length > 0) {
      const signalMarkers = signals
        .filter(signal => signal.timestamp)
        .map(signal => ({
          time: signal.timestamp.split('T')[0],
          position: (signal.signal_type === 'BUY'
            ? 'belowBar'
            : 'aboveBar') as SeriesMarkerPosition,
          shape: (signal.signal_type === 'BUY' ? 'circle' : 'circle') as SeriesMarkerShape, // 使用圆形区分信号
          color: (() => {
            const key = signal.strategy_id || signal.strategy_name;
            if (key && strategyColorMap[key]) {
              // 根据执行与否调整透明度
              const base = strategyColorMap[key];
              if (signal.executed === false) {
                // 未执行加一点透明度
                return base + '80';
              }
              return base;
            }
            // 回退到旧的按信号方向配色
            return signal.signal_type === 'BUY'
              ? signal.executed
                ? '#4caf50'
                : '#8bc34a'
              : signal.executed
                ? '#f44336'
                : '#ff9800';
          })(),
          text: `${signal.strategy_name ? signal.strategy_name + '·' : ''}${
            signal.signal_type === 'BUY' ? '买' : '卖'
          }信号${signal.executed ? '' : '(未执行)'}`,
          size: signal.executed ? 0.8 : 0.6, // 已执行信号更大
        }));
      allMarkers.push(...signalMarkers);
    }

    // 按时间排序（必须升序）
    allMarkers.sort((a, b) => {
      const timeA = new Date(a.time).getTime();
      const timeB = new Date(b.time).getTime();
      return timeA - timeB;
    });

    candlestickSeriesRef.current.setMarkers(allMarkers);
  };

  // 初始化图表
  const initChart = () => {
    if (!chartContainerRef.current) {
      return;
    }

    // 清理现有图表
    if (chartRef.current) {
      chartRef.current.remove();
    }

    // 创建新图表
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#333',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#cccccc',
        scaleMargins: {
          top: 0.1,
          bottom: 0.3,
        },
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // 添加价格系列
    if (chartType === 'candlestick') {
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#ef5350', // 红色，上涨
        downColor: '#26a69a', // 绿色，下跌
        borderVisible: false,
        wickUpColor: '#ef5350',
        wickDownColor: '#26a69a',
      });
      candlestickSeriesRef.current = candlestickSeries;
    } else {
      const lineSeries = chart.addLineSeries({
        color: '#2962FF',
        lineWidth: 2,
      });
      candlestickSeriesRef.current = lineSeries as any;
    }

    // 添加成交量系列
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });
    volumeSeriesRef.current = volumeSeries;
    chart.priceScale('').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // 设置数据
    updateChartData();

    // 添加买卖标记
    addTradingMarkers();
    addTradeMarkers();
    addSignalMarkers();

    // 响应式调整
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  };

  // 更新图表数据
  const updateChartData = () => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current) {
      return;
    }

    if (chartType === 'candlestick') {
      const candlestickData = priceData.map(item => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));
      candlestickSeriesRef.current.setData(candlestickData);
    } else {
      const lineData = priceData.map(item => ({
        time: item.time,
        value: item.close,
      }));
      candlestickSeriesRef.current.setData(lineData);
    }

    const volumeData = priceData.map(item => ({
      time: item.time,
      value: item.volume,
      color: item.close >= item.open ? '#ef535080' : '#26a69a80', // 上涨时红色，下跌时绿色
    }));
    volumeSeriesRef.current.setData(volumeData);
  };

  // 初始化
  useEffect(() => {
    fetchStockData();
  }, [stockCode, timeframe, startDate, endDate]);

  useEffect(() => {
    // 即使数据为空也初始化图表，以便显示空状态
    if (chartContainerRef.current) {
      initChart();
    }
  }, [priceData.length > 0, chartType, height]); // 当有数据时重新初始化

  useEffect(() => {
    updateChartData();
    // 当数据或预测结果更新时，重新添加买卖标记
    if (priceData.length > 0 && candlestickSeriesRef.current) {
      addTradingMarkers();
      addTradeMarkers();
      addSignalMarkers();
    }
  }, [priceData, chartType, prediction, trades, signals, internalShowSignals, internalShowTrades]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
              {stockCode} 价格走势
            </Typography>
            <Typography variant="body2" color="text.secondary">
              当前价格: ¥
              {priceData.length > 0 ? priceData[priceData.length - 1]?.close.toFixed(2) : '--'}
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <ButtonGroup size="small" variant="outlined">
              <Button
                variant={timeframe === '1D' ? 'contained' : 'outlined'}
                onClick={() => setTimeframe('1D')}
              >
                日线
              </Button>
              <Button
                variant={timeframe === '1W' ? 'contained' : 'outlined'}
                onClick={() => setTimeframe('1W')}
              >
                周线
              </Button>
              <Button
                variant={timeframe === '1M' ? 'contained' : 'outlined'}
                onClick={() => setTimeframe('1M')}
              >
                月线
              </Button>
            </ButtonGroup>

            <ButtonGroup size="small" variant="outlined">
              <Button
                variant={chartType === 'candlestick' ? 'contained' : 'outlined'}
                onClick={() => setChartType('candlestick')}
                startIcon={<BarChart3 size={16} />}
              >
                K线
              </Button>
              <Button
                variant={chartType === 'line' ? 'contained' : 'outlined'}
                onClick={() => setChartType('line')}
                startIcon={<TrendingUp size={16} />}
              >
                线图
              </Button>
            </ButtonGroup>

            {(signals && signals.length > 0) || (trades && trades.length > 0) ? (
              <ButtonGroup size="small" variant="outlined">
                <Button
                  variant={internalShowSignals ? 'contained' : 'outlined'}
                  onClick={() => {
                    setInternalShowSignals(!internalShowSignals);
                  }}
                  startIcon={<AlertCircle size={16} />}
                >
                  信号
                </Button>
                <Button
                  variant={internalShowTrades ? 'contained' : 'outlined'}
                  onClick={() => {
                    setInternalShowTrades(!internalShowTrades);
                  }}
                  startIcon={<TrendingUp size={16} />}
                >
                  交易
                </Button>
              </ButtonGroup>
            ) : null}
          </Box>
        </Box>

        <Box sx={{ position: 'relative' }}>
          {loading && (
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'rgba(255,255,255,0.8)',
                zIndex: 10,
              }}
            >
              <CircularProgress size={48} />
            </Box>
          )}
          {!loading && priceData.length === 0 && (
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'rgba(255,255,255,0.8)',
                zIndex: 10,
              }}
            >
              <AlertCircle size={48} color="#999" style={{ marginBottom: 8 }} />
              <Typography variant="body2" color="text.secondary">
                暂无数据
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                股票代码: {stockCode}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                时间范围: {startDate || '默认'} 至 {endDate || '现在'}
              </Typography>
            </Box>
          )}
          <Box ref={chartContainerRef} sx={{ height: `${height}px`, width: '100%' }} />
        </Box>
      </CardContent>
    </Card>
  );
}
