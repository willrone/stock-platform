/**
 * 持仓分析组件
 * 展示各股票表现排行、持仓权重饼图和柱状图
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  Select,
  MenuItem,
  Tabs,
  Tab,
  LinearProgress,
  Tooltip,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Box,
  Typography,
  FormControl,
  InputLabel,
  TableContainer,
  Paper,
  TableSortLabel,
} from '@mui/material';
import * as echarts from 'echarts';
import { TrendingUp, TrendingDown, PieChart as PieChartIcon, BarChart3, Target, Award, DollarSign } from 'lucide-react';
import { BacktestService, PortfolioSnapshot } from '@/services/backtestService';

interface PositionData {
  stock_code: string;
  stock_name: string;
  total_return: number;
  trade_count: number;
  win_rate: number;
  avg_holding_period: number;
  winning_trades: number;
  losing_trades: number;
  // 扩展字段
  avg_return_per_trade?: number;
  return_ratio?: number;
  trade_frequency?: number;
  avg_win?: number;
  avg_loss?: number;
  largest_win?: number;
  largest_loss?: number;
  profit_factor?: number;
  max_holding_period?: number;
  min_holding_period?: number;
  avg_buy_price?: number;
  avg_sell_price?: number;
  price_improvement?: number;
  total_volume?: number;
  total_commission?: number;
  commission_ratio?: number;
}

interface EnhancedPositionAnalysis {
  stock_performance: PositionData[];
  position_weights?: any;
  trading_patterns?: any;
  holding_periods?: any;
  concentration_risk?: any;
}

interface PositionAnalysisProps {
  positionAnalysis: PositionData[] | EnhancedPositionAnalysis;
  stockCodes: string[];
  taskId?: string; // 任务ID，用于获取组合快照数据
}

interface SortConfig {
  key: keyof PositionData;
  direction: 'asc' | 'desc';
}

export function PositionAnalysis({ positionAnalysis, stockCodes, taskId }: PositionAnalysisProps) {
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'total_return',
    direction: 'desc'
  });
  const [selectedMetric, setSelectedMetric] = useState<keyof PositionData>('total_return');
  const [selectedStock, setSelectedStock] = useState<PositionData | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('table');
  
  // 组合快照数据状态
  const [portfolioSnapshots, setPortfolioSnapshots] = useState<PortfolioSnapshot[]>([]);
  const [loadingSnapshots, setLoadingSnapshots] = useState(false);
  
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const onDetailOpen = () => setIsDetailOpen(true);
  const onDetailClose = () => setIsDetailOpen(false);
  
  // 图表引用
  const pieChartRef = useRef<HTMLDivElement>(null);
  const barChartRef = useRef<HTMLDivElement>(null);
  const treemapChartRef = useRef<HTMLDivElement>(null);
  const weightChartRef = useRef<HTMLDivElement>(null);
  const tradingPatternChartRef = useRef<HTMLDivElement>(null);
  const holdingPeriodChartRef = useRef<HTMLDivElement>(null);
  const capitalChartRef = useRef<HTMLDivElement>(null);
  const pieChartInstance = useRef<echarts.ECharts | null>(null);
  const barChartInstance = useRef<echarts.ECharts | null>(null);
  const treemapChartInstance = useRef<echarts.ECharts | null>(null);
  const weightChartInstance = useRef<echarts.ECharts | null>(null);
  const tradingPatternChartInstance = useRef<echarts.ECharts | null>(null);
  const holdingPeriodChartInstance = useRef<echarts.ECharts | null>(null);
  const capitalChartInstance = useRef<echarts.ECharts | null>(null);

  // 数据格式转换：兼容新旧两种格式
  const normalizedData = useMemo(() => {
    console.log('[PositionAnalysis] 接收到的 positionAnalysis:', positionAnalysis);
    console.log('[PositionAnalysis] positionAnalysis 类型:', typeof positionAnalysis);
    console.log('[PositionAnalysis] 是否为数组:', Array.isArray(positionAnalysis));
    
    if (!positionAnalysis) {
      console.log('[PositionAnalysis] positionAnalysis 为空');
      return null;
    }
    
    // 如果是数组格式（旧格式），直接使用
    if (Array.isArray(positionAnalysis)) {
      console.log('[PositionAnalysis] 使用数组格式，长度:', positionAnalysis.length);
      return {
        stock_performance: positionAnalysis,
        position_weights: undefined,
        trading_patterns: undefined,
        holding_periods: undefined,
        concentration_risk: undefined
      };
    }
    
    // 如果是对象格式（新格式），检查是否有 stock_performance
    if (typeof positionAnalysis === 'object' && positionAnalysis !== null) {
      console.log('[PositionAnalysis] 使用对象格式');
      console.log('[PositionAnalysis] 对象键:', Object.keys(positionAnalysis));
      console.log('[PositionAnalysis] stock_performance:', positionAnalysis.stock_performance);
      console.log('[PositionAnalysis] stock_performance 类型:', typeof positionAnalysis.stock_performance);
      console.log('[PositionAnalysis] stock_performance 长度:', Array.isArray(positionAnalysis.stock_performance) ? positionAnalysis.stock_performance.length : 'N/A');
      
      // 确保 stock_performance 存在且是数组
      if (positionAnalysis.stock_performance && Array.isArray(positionAnalysis.stock_performance)) {
        return positionAnalysis as EnhancedPositionAnalysis;
      } else {
        console.warn('[PositionAnalysis] stock_performance 不存在或不是数组');
        return null;
      }
    }
    
    console.warn('[PositionAnalysis] 未知的数据格式');
    return null;
  }, [positionAnalysis]);

  // 获取股票表现数据
  const stockPerformance = useMemo(() => {
    return normalizedData?.stock_performance || [];
  }, [normalizedData]);

  // 排序后的持仓数据
  const sortedPositions = useMemo(() => {
    if (!stockPerformance || stockPerformance.length === 0) return [];
    
    return [...stockPerformance].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortConfig.direction === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      return 0;
    });
  }, [stockPerformance, sortConfig]);

  // 饼图数据（基于总收益）
  const pieChartData = useMemo(() => {
    if (!stockPerformance || stockPerformance.length === 0) return [];
    
    return stockPerformance
      .filter(pos => Math.abs(pos.total_return) > 0)
      .map(pos => ({
        name: pos.stock_code,
        value: Math.abs(pos.total_return),
        originalValue: pos.total_return,
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10); // 只显示前10个
  }, [stockPerformance]);

  // 柱状图数据
  const barChartData = useMemo(() => {
    if (!stockPerformance || stockPerformance.length === 0) return [];
    
    return stockPerformance
      .map(pos => ({
        stock_code: pos.stock_code,
        total_return: pos.total_return,
        win_rate: pos.win_rate * 100,
        trade_count: pos.trade_count,
        avg_holding_period: pos.avg_holding_period
      }))
      .sort((a, b) => b.total_return - a.total_return)
      .slice(0, 15); // 显示前15个
  }, [stockPerformance]);

  // 树状图数据（用于权重可视化）
  const treemapData = useMemo(() => {
    if (!stockPerformance || stockPerformance.length === 0) return [];
    
    return stockPerformance
      .filter(pos => Math.abs(pos.total_return) > 0)
      .map(pos => ({
        name: pos.stock_code,
        value: Math.abs(pos.total_return),
        originalValue: pos.total_return,
        itemStyle: {
          color: pos.total_return >= 0 ? '#10b981' : '#ef4444'
        }
      }))
      .sort((a, b) => b.value - a.value);
  }, [stockPerformance]);

  // 持仓权重数据（基于真实权重）
  const weightChartData = useMemo(() => {
    const weights = normalizedData?.position_weights?.current_weights;
    if (!weights || Object.keys(weights).length === 0) return null;
    
    return Object.entries(weights)
      .map(([stock_code, weight]) => ({
        name: stock_code,
        value: (weight as number) * 100, // 转换为百分比
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 15);
  }, [normalizedData]);

  // 统计信息
  const statistics = useMemo(() => {
    if (!stockPerformance || stockPerformance.length === 0) {
      return {
        totalStocks: 0,
        profitableStocks: 0,
        totalReturn: 0,
        avgWinRate: 0,
        avgHoldingPeriod: 0,
        bestPerformer: null,
        worstPerformer: null
      };
    }
    
    const totalStocks = stockPerformance.length;
    const profitableStocks = stockPerformance.filter(pos => pos.total_return > 0).length;
    const totalReturn = stockPerformance.reduce((sum, pos) => sum + pos.total_return, 0);
    const avgWinRate = stockPerformance.reduce((sum, pos) => sum + pos.win_rate, 0) / totalStocks;
    const avgHoldingPeriod = stockPerformance.reduce((sum, pos) => sum + pos.avg_holding_period, 0) / totalStocks;
    
    const bestPerformer = stockPerformance.reduce((best, pos) => 
      pos.total_return > best.total_return ? pos : best
    );
    const worstPerformer = stockPerformance.reduce((worst, pos) => 
      pos.total_return < worst.total_return ? pos : worst
    );
    
    return {
      totalStocks,
      profitableStocks,
      totalReturn,
      avgWinRate,
      avgHoldingPeriod,
      bestPerformer,
      worstPerformer
    };
  }, [stockPerformance]);

  // 初始化饼图
  useEffect(() => {
    if (!pieChartRef.current || pieChartData.length === 0) return;
    
    // 如果不在饼图Tab，不初始化
    if (selectedTab !== 'pie') return;

    const initChart = () => {
      if (!pieChartRef.current || pieChartData.length === 0) return;
      
      // 检查容器是否有尺寸
      const rect = pieChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (pieChartInstance.current) {
        pieChartInstance.current.dispose();
      }

      pieChartInstance.current = echarts.init(pieChartRef.current);

      const option = {
        title: {
          text: '持仓权重分布',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'item',
          formatter: function (params: any) {
            const percentage = ((params.value / pieChartData.reduce((sum, item) => sum + item.value, 0)) * 100).toFixed(1);
            return `${params.name}<br/>收益: ¥${params.data.originalValue.toFixed(2)}<br/>占比: ${percentage}%`;
          },
        },
        legend: {
          orient: 'vertical',
          left: 'left',
          data: pieChartData.map(item => item.name),
        },
        series: [
          {
            name: '持仓权重',
            type: 'pie',
            radius: '50%',
            data: pieChartData,
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)',
              },
            },
          },
        ],
      };

      pieChartInstance.current.setOption(option);

      const handleResize = () => {
        if (pieChartInstance.current) {
          pieChartInstance.current.resize();
        }
      };

      window.addEventListener('resize', handleResize);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (pieChartInstance.current) {
        pieChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [pieChartData, selectedTab]);

  // 初始化柱状图
  useEffect(() => {
    if (!barChartRef.current || barChartData.length === 0) return;
    
    // 如果不在柱状图Tab，不初始化
    if (selectedTab !== 'bar') return;

    const initChart = () => {
      if (!barChartRef.current || barChartData.length === 0) return;
      
      // 检查容器是否有尺寸
      const rect = barChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (barChartInstance.current) {
        barChartInstance.current.dispose();
      }

      barChartInstance.current = echarts.init(barChartRef.current);

    const getDataByMetric = () => {
      switch (selectedMetric) {
        case 'total_return':
          return barChartData.map(item => item.total_return);
        case 'win_rate':
          return barChartData.map(item => item.win_rate);
        case 'trade_count':
          return barChartData.map(item => item.trade_count);
        case 'avg_holding_period':
          return barChartData.map(item => item.avg_holding_period);
        default:
          return barChartData.map(item => item.total_return);
      }
    };

    const getMetricName = () => {
      switch (selectedMetric) {
        case 'total_return':
          return '总收益';
        case 'win_rate':
          return '胜率';
        case 'trade_count':
          return '交易次数';
        case 'avg_holding_period':
          return '平均持仓期';
        default:
          return '总收益';
      }
    };

    const option = {
      title: {
        text: '股票表现对比',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: function (params: any) {
          const param = params[0];
          const value = param.value;
          
          if (selectedMetric === 'total_return') {
            return `${param.name}<br/>${getMetricName()}: ¥${value.toFixed(2)}`;
          } else if (selectedMetric === 'win_rate') {
            return `${param.name}<br/>${getMetricName()}: ${value.toFixed(2)}%`;
          } else if (selectedMetric === 'avg_holding_period') {
            return `${param.name}<br/>${getMetricName()}: ${value} 天`;
          }
          return `${param.name}<br/>${getMetricName()}: ${value}`;
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: barChartData.map(item => item.stock_code),
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: function (value: number) {
            if (selectedMetric === 'total_return') {
              return `¥${(value / 1000).toFixed(0)}K`;
            } else if (selectedMetric === 'win_rate') {
              return `${value.toFixed(0)}%`;
            } else if (selectedMetric === 'avg_holding_period') {
              return `${value}天`;
            }
            return value.toString();
          },
        },
      },
      series: [
        {
          name: getMetricName(),
          type: 'bar',
          data: getDataByMetric(),
          itemStyle: {
            color: '#3b82f6',
          },
        },
      ],
    };

      barChartInstance.current.setOption(option);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (barChartInstance.current) {
        barChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [barChartData, selectedMetric, selectedTab]);

  // 初始化树状图
  useEffect(() => {
    if (!treemapChartRef.current || treemapData.length === 0) return;
    
    // 如果不在树状图Tab，不初始化
    if (selectedTab !== 'treemap') return;

    const initChart = () => {
      if (!treemapChartRef.current || treemapData.length === 0) return;
      
      // 检查容器是否有尺寸
      const rect = treemapChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (treemapChartInstance.current) {
        treemapChartInstance.current.dispose();
      }

      treemapChartInstance.current = echarts.init(treemapChartRef.current);

      const option = {
        title: {
          text: '持仓权重树状图',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'item',
          formatter: function (params: any) {
            const percentage = ((params.value / treemapData.reduce((sum, item) => sum + item.value, 0)) * 100).toFixed(1);
            return `${params.name}<br/>收益: ¥${params.data.originalValue.toFixed(2)}<br/>占比: ${percentage}%`;
          },
        },
        series: [
          {
            name: '持仓权重',
            type: 'treemap',
            data: treemapData,
            roam: false,
            nodeClick: false,
            breadcrumb: {
              show: false,
            },
            label: {
              show: true,
              formatter: function (params: any) {
                const percentage = ((params.value / treemapData.reduce((sum, item) => sum + item.value, 0)) * 100).toFixed(1);
                return `${params.name}\n${percentage}%`;
              },
              color: '#fff',
              fontWeight: 'bold',
            },
            upperLabel: {
              show: false,
            },
          },
        ],
      };

      treemapChartInstance.current.setOption(option);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (treemapChartInstance.current) {
        treemapChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [treemapData, selectedTab]);

  // 初始化持仓权重图表
  useEffect(() => {
    if (!weightChartRef.current || !weightChartData || weightChartData.length === 0) return;
    
    // 如果不在持仓权重Tab，不初始化
    if (selectedTab !== 'weight') return;

    const initChart = () => {
      if (!weightChartRef.current || !weightChartData || weightChartData.length === 0) return;
      
      // 检查容器是否有尺寸
      const rect = weightChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (weightChartInstance.current) {
        weightChartInstance.current.dispose();
      }

      weightChartInstance.current = echarts.init(weightChartRef.current);

    const option = {
      title: {
        text: '持仓权重分布（真实权重）',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params: any) {
          return `${params.name}<br/>权重: ${params.value.toFixed(2)}%`;
        },
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        data: weightChartData.map(item => item.name),
      },
      series: [
        {
          name: '持仓权重',
          type: 'pie',
          radius: '50%',
          data: weightChartData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
    };

      weightChartInstance.current.setOption(option);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (weightChartInstance.current) {
        weightChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [weightChartData, selectedTab]);

  // 初始化交易模式图表
  useEffect(() => {
    if (!tradingPatternChartRef.current || !normalizedData?.trading_patterns) return;
    
    // 如果不在交易模式Tab，不初始化
    if (selectedTab !== 'trading') return;

    const initChart = () => {
      if (!tradingPatternChartRef.current || !normalizedData?.trading_patterns) return;
      
      // 检查容器是否有尺寸
      const rect = tradingPatternChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (tradingPatternChartInstance.current) {
        tradingPatternChartInstance.current.dispose();
      }

      tradingPatternChartInstance.current = echarts.init(tradingPatternChartRef.current);

      const patterns = normalizedData.trading_patterns;
      const option: any = {
        title: {
          text: '交易模式分析',
          left: 'center',
        },
        tooltip: {
          trigger: 'axis',
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '15%',
          top: '15%',
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: patterns.time_patterns?.monthly_distribution?.map((m: any) => `${m.month}月`) || [],
        },
        yAxis: {
          type: 'value',
        },
        series: [
          {
            name: '交易次数',
            type: 'bar',
            data: patterns.time_patterns?.monthly_distribution?.map((m: any) => m.count) || [],
            itemStyle: {
              color: '#3b82f6',
            },
          },
        ],
      };

      tradingPatternChartInstance.current.setOption(option);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (tradingPatternChartInstance.current) {
        tradingPatternChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [normalizedData?.trading_patterns, selectedTab]);

  // 初始化持仓时间图表
  useEffect(() => {
    if (!holdingPeriodChartRef.current || !normalizedData?.holding_periods) return;
    
    // 如果不在持仓时间Tab，不初始化
    if (selectedTab !== 'holding') return;

    const initChart = () => {
      if (!holdingPeriodChartRef.current || !normalizedData?.holding_periods) return;
      
      // 检查容器是否有尺寸
      const rect = holdingPeriodChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (holdingPeriodChartInstance.current) {
        holdingPeriodChartInstance.current.dispose();
      }

      holdingPeriodChartInstance.current = echarts.init(holdingPeriodChartRef.current);

      const periods = normalizedData.holding_periods;
      const option = {
        title: {
          text: '持仓时间分布',
          left: 'center',
        },
        tooltip: {
          trigger: 'item',
        },
        series: [
          {
            name: '持仓时间',
            type: 'pie',
            radius: '50%',
            data: [
              {
                value: periods.short_term_positions,
                name: '短期（≤7天）',
                itemStyle: { color: '#3b82f6' },
              },
              {
                value: periods.medium_term_positions,
                name: '中期（7-30天）',
                itemStyle: { color: '#10b981' },
              },
              {
                value: periods.long_term_positions,
                name: '长期（>30天）',
                itemStyle: { color: '#f59e0b' },
              },
            ],
          },
        ],
      };

      holdingPeriodChartInstance.current.setOption(option);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (holdingPeriodChartInstance.current) {
        holdingPeriodChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [normalizedData?.holding_periods, selectedTab]);

  // 获取组合快照数据
  useEffect(() => {
    if (!taskId) return;

    const loadSnapshots = async () => {
      setLoadingSnapshots(true);
      try {
        const result = await BacktestService.getPortfolioSnapshots(taskId, undefined, undefined, 10000);
        if (result && result.snapshots) {
          // 按日期排序
          const sorted = [...result.snapshots].sort((a, b) => 
            new Date(a.snapshot_date).getTime() - new Date(b.snapshot_date).getTime()
          );
          setPortfolioSnapshots(sorted);
        }
      } catch (error) {
        console.error('获取组合快照数据失败:', error);
      } finally {
        setLoadingSnapshots(false);
      }
    };

    loadSnapshots();
  }, [taskId]);

  // 资金分配图表数据
  const capitalChartData = useMemo(() => {
    if (!portfolioSnapshots || portfolioSnapshots.length === 0) return null;

    const dates: string[] = [];
    const totalCapital: number[] = [];
    const positionCapital: number[] = [];
    const freeCapital: number[] = [];

    portfolioSnapshots.forEach(snapshot => {
      dates.push(snapshot.snapshot_date);
      totalCapital.push(snapshot.portfolio_value);
      freeCapital.push(snapshot.cash);
      positionCapital.push(snapshot.portfolio_value - snapshot.cash);
    });

    return {
      dates,
      totalCapital,
      positionCapital,
      freeCapital
    };
  }, [portfolioSnapshots]);

  // 初始化资金分配折线图
  useEffect(() => {
    if (!capitalChartRef.current || !capitalChartData || capitalChartData.dates.length === 0) return;
    
    // 如果不在资金分配Tab，不初始化
    if (selectedTab !== 'capital') return;

    const initChart = () => {
      if (!capitalChartRef.current || !capitalChartData || capitalChartData.dates.length === 0) return;
      
      // 检查容器是否有尺寸
      const rect = capitalChartRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        // 如果容器还没有尺寸，延迟重试
        setTimeout(initChart, 100);
        return;
      }

      if (capitalChartInstance.current) {
        capitalChartInstance.current.dispose();
      }

      capitalChartInstance.current = echarts.init(capitalChartRef.current);

      const option = {
        title: {
          text: '资金分配趋势',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'axis',
          formatter: function (params: any) {
            let result = `${params[0].axisValue}<br/>`;
            params.forEach((param: any) => {
              result += `${param.marker}${param.seriesName}: ¥${param.value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}<br/>`;
            });
            return result;
          },
        },
        legend: {
          data: ['总资金', '持仓资金', '空闲资金'],
          top: 30,
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
          boundaryGap: false,
          data: capitalChartData.dates,
          axisLabel: {
            rotate: 45,
            formatter: function (value: string) {
              return value.split('T')[0]; // 只显示日期部分
            },
          },
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: function (value: number) {
              if (value >= 10000) {
                return `¥${(value / 10000).toFixed(1)}万`;
              }
              return `¥${value.toFixed(0)}`;
            },
          },
        },
        series: [
          {
            name: '总资金',
            type: 'line',
            data: capitalChartData.totalCapital,
            smooth: true,
            itemStyle: {
              color: '#3b82f6', // 蓝色
            },
            areaStyle: {
              opacity: 0.1,
            },
          },
          {
            name: '持仓资金',
            type: 'line',
            data: capitalChartData.positionCapital,
            smooth: true,
            itemStyle: {
              color: '#10b981', // 绿色
            },
            areaStyle: {
              opacity: 0.1,
            },
          },
          {
            name: '空闲资金',
            type: 'line',
            data: capitalChartData.freeCapital,
            smooth: true,
            itemStyle: {
              color: '#f59e0b', // 橙色
            },
            areaStyle: {
              opacity: 0.1,
            },
          },
        ],
      };

      capitalChartInstance.current.setOption(option);
    };

    // 延迟初始化，确保容器已渲染
    const timer = setTimeout(initChart, 100);

    const handleResize = () => {
      if (capitalChartInstance.current) {
        capitalChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [capitalChartData, selectedTab]);

  // 处理排序
  const handleSort = (key: keyof PositionData) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  // 处理股票详情点击
  const handleStockClick = (stock: PositionData) => {
    setSelectedStock(stock);
    onDetailOpen();
  };

  // 格式化函数
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (!stockPerformance || stockPerformance.length === 0) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Target size={48} color="#999" style={{ margin: '0 auto 8px' }} />
            <Typography variant="body2" color="text.secondary">暂无持仓分析数据</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 统计概览 */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
        <Card>
          <CardContent sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Target size={20} color="#1976d2" />
              <Box>
                <Typography variant="caption" color="text.secondary">持仓股票</Typography>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>{statistics.totalStocks}</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TrendingUp size={20} color="#2e7d32" />
              <Box>
                <Typography variant="caption" color="text.secondary">盈利股票</Typography>
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
                <Typography variant="caption" color="text.secondary">平均胜率</Typography>
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
                <Typography variant="caption" color="text.secondary">总收益</Typography>
                <Typography variant="h5" sx={{ fontWeight: 600, color: statistics.totalReturn >= 0 ? 'success.main' : 'error.main' }}>
                  {formatCurrency(statistics.totalReturn)}
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* 最佳和最差表现者 */}
      {statistics.bestPerformer && statistics.worstPerformer && (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
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
                    {statistics.bestPerformer.stock_code}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {statistics.bestPerformer.stock_name}
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatCurrency(statistics.bestPerformer.total_return)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    胜率: {formatPercent(statistics.bestPerformer.win_rate)}
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
                    {statistics.worstPerformer.stock_code}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {statistics.worstPerformer.stock_name}
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {formatCurrency(statistics.worstPerformer.total_return)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    胜率: {formatPercent(statistics.worstPerformer.win_rate)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* 图表展示 */}
      <Box>
        <Tabs 
          value={selectedTab}
          onChange={(e, newValue) => setSelectedTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Target size={16} />
              <span>表格视图</span>
            </Box>
          } value="table" />
          <Tab label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <PieChartIcon size={16} />
              <span>饼图</span>
            </Box>
          } value="pie" />
          <Tab label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <BarChart3 size={16} />
              <span>柱状图</span>
            </Box>
          } value="bar" />
          <Tab label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <BarChart3 size={16} />
              <span>树状图</span>
            </Box>
          } value="treemap" />
          {normalizedData?.position_weights && (
            <Tab label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <BarChart3 size={16} />
                <span>权重分析</span>
              </Box>
            } value="weight" />
          )}
          {normalizedData?.trading_patterns && (
            <Tab label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <BarChart3 size={16} />
                <span>交易模式</span>
              </Box>
            } value="trading" />
          )}
          {normalizedData?.holding_periods && (
            <Tab label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <BarChart3 size={16} />
                <span>持仓期分析</span>
              </Box>
            } value="holding" />
          )}
          {taskId && (
            <Tab label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <DollarSign size={16} />
                <span>资金分析</span>
              </Box>
            } value="capital" />
          )}
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {selectedTab === 'table' && (
            <Card>
              <CardContent sx={{ p: 0 }}>
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
                  <Table stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>
                          <TableSortLabel
                            active={sortConfig.key === 'stock_code'}
                            direction={sortConfig.key === 'stock_code' ? sortConfig.direction : 'asc'}
                            onClick={() => handleSort('stock_code')}
                          >
                            股票代码
                          </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                          <TableSortLabel
                            active={sortConfig.key === 'total_return'}
                            direction={sortConfig.key === 'total_return' ? sortConfig.direction : 'asc'}
                            onClick={() => handleSort('total_return')}
                          >
                            总收益
                          </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                          <TableSortLabel
                            active={sortConfig.key === 'trade_count'}
                            direction={sortConfig.key === 'trade_count' ? sortConfig.direction : 'asc'}
                            onClick={() => handleSort('trade_count')}
                          >
                            交易次数
                          </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                          <TableSortLabel
                            active={sortConfig.key === 'win_rate'}
                            direction={sortConfig.key === 'win_rate' ? sortConfig.direction : 'asc'}
                            onClick={() => handleSort('win_rate')}
                          >
                            胜率
                          </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                          <TableSortLabel
                            active={sortConfig.key === 'avg_holding_period'}
                            direction={sortConfig.key === 'avg_holding_period' ? sortConfig.direction : 'asc'}
                            onClick={() => handleSort('avg_holding_period')}
                          >
                            平均持仓期
                          </TableSortLabel>
                        </TableCell>
                        <TableCell>表现</TableCell>
                        <TableCell>操作</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sortedPositions.map((position) => (
                        <TableRow key={position.stock_code} hover>
                          <TableCell>
                            <Box>
                              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                {position.stock_code}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {position.stock_name}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontFamily: 'monospace', 
                                fontWeight: 500,
                                color: position.total_return > 0 ? 'success.main' : 
                                       position.total_return < 0 ? 'error.main' : 'text.secondary'
                              }}
                            >
                              {formatCurrency(position.total_return)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {position.trade_count}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={position.win_rate * 100}
                                sx={{ width: 60, height: 8, borderRadius: 4 }}
                                color={position.win_rate >= 0.5 ? 'success' : 'error'}
                              />
                              <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                                {formatPercent(position.win_rate)}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {position.avg_holding_period} 天
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={`${position.winning_trades}胜${position.losing_trades}负`}
                              size="small"
                              color={position.winning_trades > position.losing_trades ? 'success' : 'error'}
                            />
                          </TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => handleStockClick(position)}
                            >
                              详情
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {selectedTab === 'pie' && (
            <Card>
              <CardHeader title="持仓权重分布（按收益绝对值）" />
              <CardContent>
                <Box ref={pieChartRef} sx={{ height: 400, width: '100%' }} />
              </CardContent>
            </Card>
          )}

          {selectedTab === 'bar' && (
            <Card>
              <CardHeader
                title={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                      股票表现对比
                    </Typography>
                    <FormControl size="small" sx={{ minWidth: 128 }}>
                      <InputLabel>指标</InputLabel>
                      <Select
                        value={selectedMetric}
                        label="指标"
                        onChange={(e) => setSelectedMetric(e.target.value as keyof PositionData)}
                      >
                        <MenuItem value="total_return">总收益</MenuItem>
                        <MenuItem value="win_rate">胜率</MenuItem>
                        <MenuItem value="trade_count">交易次数</MenuItem>
                        <MenuItem value="avg_holding_period">持仓期</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                }
              />
              <CardContent>
                <Box ref={barChartRef} sx={{ height: 400, width: '100%' }} />
              </CardContent>
            </Card>
          )}

          {selectedTab === 'treemap' && (
            <Card>
              <CardHeader
                title={
                  <Box>
                    <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                      持仓权重树状图
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      绿色表示盈利，红色表示亏损，面积大小表示收益绝对值
                    </Typography>
                  </Box>
                }
              />
              <CardContent>
                <Box ref={treemapChartRef} sx={{ height: 400, width: '100%' }} />
              </CardContent>
            </Card>
          )}

          {selectedTab === 'weight' && normalizedData?.position_weights && (
            <Card>
              <CardHeader
                title={
                  <Box>
                    <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                      持仓权重分析
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      基于真实持仓权重的分布
                    </Typography>
                  </Box>
                }
              />
              <CardContent>
                {weightChartData && weightChartData.length > 0 ? (
                  <Box ref={weightChartRef} sx={{ height: 400, width: '100%' }} />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">暂无持仓权重数据</Typography>
                  </Box>
                )}
                {/* 集中度指标 */}
                {normalizedData.position_weights.concentration_metrics?.averages && (
                  <Box sx={{ mt: 3, display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">HHI指数</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {normalizedData.position_weights.concentration_metrics.averages.avg_hhi.toFixed(3)}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">有效股票数</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {normalizedData.position_weights.concentration_metrics.averages.avg_effective_stocks.toFixed(1)}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">前3大集中度</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {(normalizedData.position_weights.concentration_metrics.averages.avg_top_3_concentration * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">前5大集中度</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {(normalizedData.position_weights.concentration_metrics.averages.avg_top_5_concentration * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {selectedTab === 'trading' && normalizedData?.trading_patterns && (
            <Card>
              <CardHeader title="交易模式分析" />
              <CardContent>
                <Box ref={tradingPatternChartRef} sx={{ height: 400, width: '100%' }} />
                {/* 交易模式统计 */}
                <Box sx={{ mt: 3, display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                  {normalizedData.trading_patterns.size_patterns && (
                    <>
                      <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary">平均交易规模</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          ¥{(normalizedData.trading_patterns.size_patterns.avg_trade_size / 10000).toFixed(2)}万
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary">总交易量</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          ¥{(normalizedData.trading_patterns.size_patterns.total_volume / 10000).toFixed(2)}万
                        </Typography>
                      </Box>
                    </>
                  )}
                  {normalizedData.trading_patterns.frequency_patterns && (
                    <>
                      <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary">平均间隔</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {normalizedData.trading_patterns.frequency_patterns.avg_interval_days.toFixed(1)}天
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary">月度交易次数</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {normalizedData.trading_patterns.frequency_patterns.avg_monthly_trades.toFixed(1)}
                        </Typography>
                      </Box>
                    </>
                  )}
                </Box>
              </CardContent>
            </Card>
          )}

          {selectedTab === 'holding' && normalizedData?.holding_periods && (
            <Card>
              <CardHeader title="持仓时间分析" />
              <CardContent>
                <Box ref={holdingPeriodChartRef} sx={{ height: 400, width: '100%' }} />
                {/* 持仓时间统计 */}
                <Box sx={{ mt: 3, display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                  <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary">平均持仓期</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {normalizedData.holding_periods.avg_holding_period.toFixed(1)}天
                    </Typography>
                  </Box>
                  <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary">中位数持仓期</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {normalizedData.holding_periods.median_holding_period.toFixed(1)}天
                    </Typography>
                  </Box>
                  <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary">短期持仓</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main' }}>
                      {normalizedData.holding_periods.short_term_positions}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">≤7天</Typography>
                  </Box>
                  <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary">长期持仓</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                      {normalizedData.holding_periods.long_term_positions}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">&gt;30天</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {selectedTab === 'capital' && taskId && (
            <Card>
              <CardHeader
                title={
                  <Box>
                    <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                      资金分配趋势
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      展示每天的持仓资金、空闲资金和总资金变化
                    </Typography>
                  </Box>
                }
              />
              <CardContent>
                {loadingSnapshots ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">加载资金分配数据中...</Typography>
                  </Box>
                ) : capitalChartData && capitalChartData.dates.length > 0 ? (
                  <Box ref={capitalChartRef} sx={{ height: 400, width: '100%' }} />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">暂无资金分配数据</Typography>
                  </Box>
                )}
                {/* 资金统计信息 */}
                {capitalChartData && capitalChartData.dates.length > 0 && (
                  <Box sx={{ mt: 3, display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'primary.light', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">平均总资金</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.dark' }}>
                        ¥{(capitalChartData.totalCapital.reduce((a, b) => a + b, 0) / capitalChartData.totalCapital.length / 10000).toFixed(2)}万
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'success.light', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">平均持仓资金</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.dark' }}>
                        ¥{(capitalChartData.positionCapital.reduce((a, b) => a + b, 0) / capitalChartData.positionCapital.length / 10000).toFixed(2)}万
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'warning.light', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">平均空闲资金</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'warning.dark' }}>
                        ¥{(capitalChartData.freeCapital.reduce((a, b) => a + b, 0) / capitalChartData.freeCapital.length / 10000).toFixed(2)}万
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">平均持仓比例</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {((capitalChartData.positionCapital.reduce((a, b) => a + b, 0) / capitalChartData.totalCapital.reduce((a, b) => a + b, 0)) * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>

      {/* 股票详情模态框 */}
      <Dialog open={isDetailOpen} onClose={onDetailClose} maxWidth="md" fullWidth>
        <DialogTitle>股票详细分析</DialogTitle>
        <DialogContent>
          {selectedStock && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ textAlign: 'center', borderBottom: 1, borderColor: 'divider', pb: 2 }}>
                <Typography variant="h4" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                  {selectedStock.stock_code}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedStock.stock_name}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">总收益</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: selectedStock.total_return > 0 ? 'success.main' : selectedStock.total_return < 0 ? 'error.main' : 'text.secondary' }}>
                    {formatCurrency(selectedStock.total_return)}
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">胜率</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>
                    {formatPercent(selectedStock.win_rate)}
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">交易次数</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>{selectedStock.trade_count}</Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">平均持仓期</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>{selectedStock.avg_holding_period} 天</Typography>
                </Box>
              </Box>
              
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">盈利交易</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {selectedStock.winning_trades}
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">亏损交易</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {selectedStock.losing_trades}
                  </Typography>
                </Box>
              </Box>

              {/* 扩展信息 */}
              {(selectedStock.avg_win !== undefined || selectedStock.avg_loss !== undefined || selectedStock.profit_factor !== undefined) && (
                <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2 }}>
                    盈亏分析
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                    {selectedStock.avg_win !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">平均盈利</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                          {formatCurrency(selectedStock.avg_win)}
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.avg_loss !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">平均亏损</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                          {formatCurrency(selectedStock.avg_loss)}
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.largest_win !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">最大盈利</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                          {formatCurrency(selectedStock.largest_win)}
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.largest_loss !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">最大亏损</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                          {formatCurrency(selectedStock.largest_loss)}
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.profit_factor !== undefined && (
                      <Box sx={{ gridColumn: 'span 2' }}>
                        <Typography variant="caption" color="text.secondary">盈亏比</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600, color: selectedStock.profit_factor >= 1 ? 'success.main' : 'error.main' }}>
                          {selectedStock.profit_factor.toFixed(2)}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Box>
              )}

              {/* 价格分析 */}
              {(selectedStock.avg_buy_price !== undefined || selectedStock.avg_sell_price !== undefined || selectedStock.price_improvement !== undefined) && (
                <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2 }}>
                    价格分析
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                    {selectedStock.avg_buy_price !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">平均买入价</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          ¥{selectedStock.avg_buy_price.toFixed(2)}
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.avg_sell_price !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">平均卖出价</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          ¥{selectedStock.avg_sell_price.toFixed(2)}
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.price_improvement !== undefined && (
                      <Box sx={{ gridColumn: 'span 2' }}>
                        <Typography variant="caption" color="text.secondary">价格改善率</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600, color: selectedStock.price_improvement >= 0 ? 'success.main' : 'error.main' }}>
                          {(selectedStock.price_improvement * 100).toFixed(2)}%
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Box>
              )}

              {/* 持仓期详情 */}
              {(selectedStock.max_holding_period !== undefined || selectedStock.min_holding_period !== undefined) && (
                <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2 }}>
                    持仓期详情
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
                    <Box>
                      <Typography variant="caption" color="text.secondary">平均持仓期</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {selectedStock.avg_holding_period} 天
                      </Typography>
                    </Box>
                    {selectedStock.max_holding_period !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">最长持仓期</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {selectedStock.max_holding_period} 天
                        </Typography>
                      </Box>
                    )}
                    {selectedStock.min_holding_period !== undefined && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">最短持仓期</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {selectedStock.min_holding_period} 天
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Box>
              )}
              
              <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">胜率进度</Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                    {formatPercent(selectedStock.win_rate)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={selectedStock.win_rate * 100}
                  color={selectedStock.win_rate >= 0.5 ? 'success' : 'error'}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button variant="contained" color="primary" onClick={onDetailClose}>
            关闭
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}