/**
 * TradingView图表组件
 * 使用lightweight-charts库显示股票价格走势
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { Card, CardBody, Button, ButtonGroup, Spinner } from '@heroui/react';
import { Calendar, TrendingUp, BarChart3 } from 'lucide-react';

interface TradingViewChartProps {
  stockCode: string;
  height?: number;
}

interface PriceData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function TradingViewChart({ stockCode, height = 400 }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M'>('1D');
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('candlestick');
  const [priceData, setPriceData] = useState<PriceData[]>([]);

  // 获取股票数据
  const fetchStockData = async () => {
    setLoading(true);
    try {
      // 这里应该调用实际的API获取股票数据
      // 暂时使用模拟数据
      const mockData: PriceData[] = generateMockData(stockCode, timeframe);
      setPriceData(mockData);
    } catch (error) {
      console.error('获取股票数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 生成模拟数据
  const generateMockData = (code: string, tf: string): PriceData[] => {
    const data: PriceData[] = [];
    const basePrice = 100 + Math.random() * 50;
    const days = tf === '1D' ? 30 : tf === '1W' ? 52 : 12;
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      if (tf === '1D') {
        date.setDate(date.getDate() - (days - i));
      } else if (tf === '1W') {
        date.setDate(date.getDate() - (days - i) * 7);
      } else {
        date.setMonth(date.getMonth() - (days - i));
      }
      
      const prevClose = i === 0 ? basePrice : data[i - 1].close;
      const change = (Math.random() - 0.5) * 0.1;
      const open = prevClose * (1 + change * 0.5);
      const close = open * (1 + change);
      const high = Math.max(open, close) * (1 + Math.random() * 0.05);
      const low = Math.min(open, close) * (1 - Math.random() * 0.05);
      const volume = Math.floor(Math.random() * 1000000) + 100000;
      
      data.push({
        time: date.toISOString().split('T')[0],
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume,
      });
    }
    
    return data;
  };

  // 初始化图表
  const initChart = () => {
    if (!chartContainerRef.current) return;

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
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
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

    // 设置数据
    updateChartData();

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
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;

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
      color: item.close >= item.open ? '#26a69a80' : '#ef535080',
    }));
    volumeSeriesRef.current.setData(volumeData);
  };

  // 初始化
  useEffect(() => {
    fetchStockData();
  }, [stockCode, timeframe]);

  useEffect(() => {
    if (priceData.length > 0) {
      initChart();
    }
  }, [priceData, chartType, height]);

  useEffect(() => {
    updateChartData();
  }, [priceData, chartType]);

  return (
    <Card>
      <CardBody>
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-lg font-semibold">{stockCode} 价格走势</h3>
            <p className="text-sm text-default-500">
              当前价格: ¥{priceData.length > 0 ? priceData[priceData.length - 1]?.close.toFixed(2) : '--'}
            </p>
          </div>
          
          <div className="flex space-x-2">
            <ButtonGroup size="sm">
              <Button
                variant={timeframe === '1D' ? 'solid' : 'light'}
                onPress={() => setTimeframe('1D')}
              >
                日线
              </Button>
              <Button
                variant={timeframe === '1W' ? 'solid' : 'light'}
                onPress={() => setTimeframe('1W')}
              >
                周线
              </Button>
              <Button
                variant={timeframe === '1M' ? 'solid' : 'light'}
                onPress={() => setTimeframe('1M')}
              >
                月线
              </Button>
            </ButtonGroup>
            
            <ButtonGroup size="sm">
              <Button
                variant={chartType === 'candlestick' ? 'solid' : 'light'}
                onPress={() => setChartType('candlestick')}
                startContent={<BarChart3 className="w-4 h-4" />}
              >
                K线
              </Button>
              <Button
                variant={chartType === 'line' ? 'solid' : 'light'}
                onPress={() => setChartType('line')}
                startContent={<TrendingUp className="w-4 h-4" />}
              >
                线图
              </Button>
            </ButtonGroup>
          </div>
        </div>
        
        <div className="relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
              <Spinner size="lg" />
            </div>
          )}
          <div
            ref={chartContainerRef}
            style={{ height: `${height}px` }}
            className="w-full"
          />
        </div>
      </CardBody>
    </Card>
  );
}