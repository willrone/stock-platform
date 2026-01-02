/**
 * TradingView图表组件
 * 使用lightweight-charts库显示股票价格走势
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { Card, CardBody, Button, ButtonGroup, Spinner } from '@heroui/react';
import { Calendar, TrendingUp, BarChart3 } from 'lucide-react';
import { DataService } from '@/services/dataService';

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
    if (!stockCode) {
      setPriceData([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      // 获取完整的数据范围：从2020年1月1日到现在
      const endDate = new Date();
      const startDate = new Date('2020-01-01');
      
      // 调用真实API获取数据
      const response = await DataService.getStockData(
        stockCode,
        startDate.toISOString().split('T')[0],
        endDate.toISOString().split('T')[0]
      );
      
      // 转换数据格式
      // DataService.getStockData返回的格式: { stock_code, data: { stock_code, start_date, end_date, data_points, data: [...] }, last_updated }
      // 所以实际数据在 response.data.data 中
      const apiData: any = response.data;
      const dataArray = (apiData?.data && Array.isArray(apiData.data)) 
        ? apiData.data 
        : (Array.isArray(apiData) ? apiData : []);
      
      if (dataArray.length > 0) {
        let formattedData: PriceData[] = dataArray.map((item: any) => ({
          time: item.date ? item.date.split('T')[0] : item.date, // 只取日期部分
          open: Number(item.open) || 0,
          high: Number(item.high) || 0,
          low: Number(item.low) || 0,
          close: Number(item.close) || 0,
          volume: Number(item.volume) || 0,
        })).filter((item: PriceData) => item.time); // 过滤掉无效数据
        
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
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            
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
        
        console.log(`成功加载 ${formattedData.length} 条${timeframe === '1D' ? '日' : timeframe === '1W' ? '周' : '月'}线数据，时间范围: ${formattedData[0]?.time} 至 ${formattedData[formattedData.length - 1]?.time}`);
        setPriceData(formattedData);
      } else {
        console.warn('未获取到股票数据，返回空数据');
        setPriceData([]);
      }
    } catch (error) {
      console.error('获取股票数据失败:', error);
      setPriceData([]);
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
    return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
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