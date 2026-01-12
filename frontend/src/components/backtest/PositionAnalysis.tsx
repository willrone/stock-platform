/**
 * 持仓分析组件
 * 展示各股票表现排行、持仓权重饼图和柱状图
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Card,
  CardBody,
  CardHeader,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  Select,
  SelectItem,
  Tabs,
  Tab,
  Progress,
  Tooltip,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
} from '@heroui/react';
import * as echarts from 'echarts';
import { TrendingUp, TrendingDown, PieChart as PieChartIcon, BarChart3, Target, Award } from 'lucide-react';

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
}

interface SortConfig {
  key: keyof PositionData;
  direction: 'asc' | 'desc';
}

export function PositionAnalysis({ positionAnalysis, stockCodes }: PositionAnalysisProps) {
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'total_return',
    direction: 'desc'
  });
  const [selectedMetric, setSelectedMetric] = useState<keyof PositionData>('total_return');
  const [selectedStock, setSelectedStock] = useState<PositionData | null>(null);
  
  const { isOpen: isDetailOpen, onOpen: onDetailOpen, onClose: onDetailClose } = useDisclosure();
  
  // 图表引用
  const pieChartRef = useRef<HTMLDivElement>(null);
  const barChartRef = useRef<HTMLDivElement>(null);
  const treemapChartRef = useRef<HTMLDivElement>(null);
  const weightChartRef = useRef<HTMLDivElement>(null);
  const tradingPatternChartRef = useRef<HTMLDivElement>(null);
  const holdingPeriodChartRef = useRef<HTMLDivElement>(null);
  const pieChartInstance = useRef<echarts.ECharts | null>(null);
  const barChartInstance = useRef<echarts.ECharts | null>(null);
  const treemapChartInstance = useRef<echarts.ECharts | null>(null);
  const weightChartInstance = useRef<echarts.ECharts | null>(null);
  const tradingPatternChartInstance = useRef<echarts.ECharts | null>(null);
  const holdingPeriodChartInstance = useRef<echarts.ECharts | null>(null);

  // 数据格式转换：兼容新旧两种格式
  const normalizedData = useMemo(() => {
    if (!positionAnalysis) return null;
    
    // 如果是数组格式（旧格式），直接使用
    if (Array.isArray(positionAnalysis)) {
      return {
        stock_performance: positionAnalysis,
        position_weights: undefined,
        trading_patterns: undefined,
        holding_periods: undefined,
        concentration_risk: undefined
      };
    }
    
    // 如果是对象格式（新格式），直接使用
    return positionAnalysis as EnhancedPositionAnalysis;
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

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [pieChartData]);

  // 初始化柱状图
  useEffect(() => {
    if (!barChartRef.current || barChartData.length === 0) return;

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

    const handleResize = () => {
      if (barChartInstance.current) {
        barChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [barChartData, selectedMetric]);

  // 初始化树状图
  useEffect(() => {
    if (!treemapChartRef.current || treemapData.length === 0) return;

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

    const handleResize = () => {
      if (treemapChartInstance.current) {
        treemapChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [treemapData]);

  // 初始化持仓权重图表
  useEffect(() => {
    if (!weightChartRef.current || !weightChartData || weightChartData.length === 0) return;

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

    const handleResize = () => {
      if (weightChartInstance.current) {
        weightChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [weightChartData]);

  // 初始化交易模式图表
  useEffect(() => {
    if (!tradingPatternChartRef.current || !normalizedData?.trading_patterns) return;

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

    const handleResize = () => {
      if (tradingPatternChartInstance.current) {
        tradingPatternChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [normalizedData?.trading_patterns]);

  // 初始化持仓时间图表
  useEffect(() => {
    if (!holdingPeriodChartRef.current || !normalizedData?.holding_periods) return;

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

    const handleResize = () => {
      if (holdingPeriodChartInstance.current) {
        holdingPeriodChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [normalizedData?.holding_periods]);

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
        <CardBody className="flex items-center justify-center h-64">
          <div className="text-center text-gray-500">
            <Target className="w-12 h-12 mx-auto mb-2" />
            <p>暂无持仓分析数据</p>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 统计概览 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardBody className="p-4">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-500" />
              <div>
                <p className="text-sm text-gray-500">持仓股票</p>
                <p className="text-xl font-bold">{statistics.totalStocks}</p>
              </div>
            </div>
          </CardBody>
        </Card>
        
        <Card>
          <CardBody className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-500" />
              <div>
                <p className="text-sm text-gray-500">盈利股票</p>
                <p className="text-xl font-bold text-green-600">
                  {statistics.profitableStocks}
                </p>
                <p className="text-xs text-gray-400">
                  ({((statistics.profitableStocks / statistics.totalStocks) * 100).toFixed(1)}%)
                </p>
              </div>
            </div>
          </CardBody>
        </Card>
        
        <Card>
          <CardBody className="p-4">
            <div className="flex items-center gap-2">
              <Award className="w-5 h-5 text-purple-500" />
              <div>
                <p className="text-sm text-gray-500">平均胜率</p>
                <p className="text-xl font-bold">
                  {formatPercent(statistics.avgWinRate)}
                </p>
              </div>
            </div>
          </CardBody>
        </Card>
        
        <Card>
          <CardBody className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-orange-500" />
              <div>
                <p className="text-sm text-gray-500">总收益</p>
                <p className={`text-xl font-bold ${statistics.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatCurrency(statistics.totalReturn)}
                </p>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>

      {/* 最佳和最差表现者 */}
      {statistics.bestPerformer && statistics.worstPerformer && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <h4 className="text-lg font-semibold text-green-600 flex items-center gap-2">
                <Award className="w-5 h-5" />
                最佳表现
              </h4>
            </CardHeader>
            <CardBody className="pt-0">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-mono font-bold text-lg">{statistics.bestPerformer.stock_code}</p>
                  <p className="text-sm text-gray-500">{statistics.bestPerformer.stock_name}</p>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-green-600">
                    {formatCurrency(statistics.bestPerformer.total_return)}
                  </p>
                  <p className="text-sm text-gray-500">
                    胜率: {formatPercent(statistics.bestPerformer.win_rate)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <h4 className="text-lg font-semibold text-red-600 flex items-center gap-2">
                <TrendingDown className="w-5 h-5" />
                最差表现
              </h4>
            </CardHeader>
            <CardBody className="pt-0">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-mono font-bold text-lg">{statistics.worstPerformer.stock_code}</p>
                  <p className="text-sm text-gray-500">{statistics.worstPerformer.stock_name}</p>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-red-600">
                    {formatCurrency(statistics.worstPerformer.total_return)}
                  </p>
                  <p className="text-sm text-gray-500">
                    胜率: {formatPercent(statistics.worstPerformer.win_rate)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      )}

      {/* 图表展示 */}
      <Tabs defaultSelectedKey="table" className="w-full">
        <Tab key="table" title={
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4" />
            表格视图
          </div>
        }>
          <Card>
            <CardBody className="p-0">
              <Table
                aria-label="持仓分析表格"
                isHeaderSticky
                classNames={{
                  wrapper: "max-h-[600px]",
                }}
              >
                <TableHeader>
                  <TableColumn 
                    key="stock_code"
                    allowsSorting
                    className="cursor-pointer"
                    onClick={() => handleSort('stock_code')}
                  >
                    <div className="flex items-center gap-1">
                      股票代码
                      {sortConfig.key === 'stock_code' && (
                        <span className="text-xs">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </TableColumn>
                  <TableColumn 
                    key="total_return"
                    allowsSorting
                    className="cursor-pointer"
                    onClick={() => handleSort('total_return')}
                  >
                    <div className="flex items-center gap-1">
                      总收益
                      {sortConfig.key === 'total_return' && (
                        <span className="text-xs">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </TableColumn>
                  <TableColumn 
                    key="trade_count"
                    allowsSorting
                    className="cursor-pointer"
                    onClick={() => handleSort('trade_count')}
                  >
                    <div className="flex items-center gap-1">
                      交易次数
                      {sortConfig.key === 'trade_count' && (
                        <span className="text-xs">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </TableColumn>
                  <TableColumn 
                    key="win_rate"
                    allowsSorting
                    className="cursor-pointer"
                    onClick={() => handleSort('win_rate')}
                  >
                    <div className="flex items-center gap-1">
                      胜率
                      {sortConfig.key === 'win_rate' && (
                        <span className="text-xs">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </TableColumn>
                  <TableColumn 
                    key="avg_holding_period"
                    allowsSorting
                    className="cursor-pointer"
                    onClick={() => handleSort('avg_holding_period')}
                  >
                    <div className="flex items-center gap-1">
                      平均持仓期
                      {sortConfig.key === 'avg_holding_period' && (
                        <span className="text-xs">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </TableColumn>
                  <TableColumn key="performance">表现</TableColumn>
                  <TableColumn key="actions">操作</TableColumn>
                </TableHeader>
                <TableBody items={sortedPositions}>
                  {(position) => (
                    <TableRow key={position.stock_code}>
                      <TableCell>
                        <div>
                          <p className="font-mono font-medium">{position.stock_code}</p>
                          <p className="text-sm text-gray-500">{position.stock_name}</p>
                        </div>
                      </TableCell>
                      <TableCell>
                        <span 
                          className={`font-mono font-medium ${
                            position.total_return > 0 ? 'text-green-600' : 
                            position.total_return < 0 ? 'text-red-600' : 'text-gray-600'
                          }`}
                        >
                          {formatCurrency(position.total_return)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{position.trade_count}</span>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Progress
                            value={position.win_rate * 100}
                            className="max-w-[60px]"
                            color={position.win_rate >= 0.5 ? 'success' : 'danger'}
                            size="sm"
                          />
                          <span className="text-sm font-mono">
                            {formatPercent(position.win_rate)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className="font-mono">{position.avg_holding_period} 天</span>
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1">
                          <Chip
                            size="sm"
                            variant="flat"
                            color={position.winning_trades > position.losing_trades ? 'success' : 'danger'}
                          >
                            {position.winning_trades}胜{position.losing_trades}负
                          </Chip>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Button
                          size="sm"
                          variant="light"
                          onPress={() => handleStockClick(position)}
                        >
                          详情
                        </Button>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardBody>
          </Card>
        </Tab>

        <Tab key="pie" title={
          <div className="flex items-center gap-2">
            <PieChartIcon className="w-4 h-4" />
            饼图
          </div>
        }>
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">持仓权重分布（按收益绝对值）</h3>
            </CardHeader>
            <CardBody>
              <div ref={pieChartRef} style={{ height: '400px', width: '100%' }} />
            </CardBody>
          </Card>
        </Tab>

        <Tab key="bar" title={
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            柱状图
          </div>
        }>
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">股票表现对比</h3>
                <Select
                  size="sm"
                  selectedKeys={[selectedMetric]}
                  onSelectionChange={(keys) => {
                    const selected = Array.from(keys)[0] as keyof PositionData;
                    setSelectedMetric(selected);
                  }}
                  className="w-32"
                >
                  <SelectItem key="total_return">总收益</SelectItem>
                  <SelectItem key="win_rate">胜率</SelectItem>
                  <SelectItem key="trade_count">交易次数</SelectItem>
                  <SelectItem key="avg_holding_period">持仓期</SelectItem>
                </Select>
              </div>
            </CardHeader>
            <CardBody>
              <div ref={barChartRef} style={{ height: '400px', width: '100%' }} />
            </CardBody>
          </Card>
        </Tab>

        <Tab key="treemap" title={
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4" />
            树状图
          </div>
        }>
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">持仓权重树状图</h3>
              <p className="text-sm text-gray-500">绿色表示盈利，红色表示亏损，面积大小表示收益绝对值</p>
            </CardHeader>
            <CardBody>
              <div ref={treemapChartRef} style={{ height: '400px', width: '100%' }} />
            </CardBody>
          </Card>
        </Tab>

        {/* 持仓权重分析 */}
        {normalizedData?.position_weights && (
          <Tab key="weights" title={
            <div className="flex items-center gap-2">
              <PieChartIcon className="w-4 h-4" />
              持仓权重
            </div>
          }>
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">持仓权重分析</h3>
                <p className="text-sm text-gray-500">基于真实持仓权重的分布</p>
              </CardHeader>
              <CardBody>
                {weightChartData && weightChartData.length > 0 ? (
                  <div ref={weightChartRef} style={{ height: '400px', width: '100%' }} />
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <p>暂无持仓权重数据</p>
                  </div>
                )}
                {/* 集中度指标 */}
                {normalizedData.position_weights.concentration_metrics?.averages && (
                  <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-500">HHI指数</p>
                      <p className="text-lg font-bold">
                        {normalizedData.position_weights.concentration_metrics.averages.avg_hhi.toFixed(3)}
                      </p>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-500">有效股票数</p>
                      <p className="text-lg font-bold">
                        {normalizedData.position_weights.concentration_metrics.averages.avg_effective_stocks.toFixed(1)}
                      </p>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-500">前3大集中度</p>
                      <p className="text-lg font-bold">
                        {(normalizedData.position_weights.concentration_metrics.averages.avg_top_3_concentration * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-500">前5大集中度</p>
                      <p className="text-lg font-bold">
                        {(normalizedData.position_weights.concentration_metrics.averages.avg_top_5_concentration * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                )}
              </CardBody>
            </Card>
          </Tab>
        )}

        {/* 交易模式分析 */}
        {normalizedData?.trading_patterns && (
          <Tab key="trading-patterns" title={
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              交易模式
            </div>
          }>
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">交易模式分析</h3>
              </CardHeader>
              <CardBody>
                <div ref={tradingPatternChartRef} style={{ height: '400px', width: '100%' }} />
                {/* 交易模式统计 */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  {normalizedData.trading_patterns.size_patterns && (
                    <>
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <p className="text-sm text-gray-500">平均交易规模</p>
                        <p className="text-lg font-bold">
                          ¥{(normalizedData.trading_patterns.size_patterns.avg_trade_size / 10000).toFixed(2)}万
                        </p>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <p className="text-sm text-gray-500">总交易量</p>
                        <p className="text-lg font-bold">
                          ¥{(normalizedData.trading_patterns.size_patterns.total_volume / 10000).toFixed(2)}万
                        </p>
                      </div>
                    </>
                  )}
                  {normalizedData.trading_patterns.frequency_patterns && (
                    <>
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <p className="text-sm text-gray-500">平均间隔</p>
                        <p className="text-lg font-bold">
                          {normalizedData.trading_patterns.frequency_patterns.avg_interval_days.toFixed(1)}天
                        </p>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <p className="text-sm text-gray-500">月度交易次数</p>
                        <p className="text-lg font-bold">
                          {normalizedData.trading_patterns.frequency_patterns.avg_monthly_trades.toFixed(1)}
                        </p>
                      </div>
                    </>
                  )}
                </div>
              </CardBody>
            </Card>
          </Tab>
        )}

        {/* 持仓时间分析 */}
        {normalizedData?.holding_periods && (
          <Tab key="holding-periods" title={
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              持仓时间
            </div>
          }>
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">持仓时间分析</h3>
              </CardHeader>
              <CardBody>
                <div ref={holdingPeriodChartRef} style={{ height: '400px', width: '100%' }} />
                {/* 持仓时间统计 */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <p className="text-sm text-gray-500">平均持仓期</p>
                    <p className="text-lg font-bold">
                      {normalizedData.holding_periods.avg_holding_period.toFixed(1)}天
                    </p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <p className="text-sm text-gray-500">中位数持仓期</p>
                    <p className="text-lg font-bold">
                      {normalizedData.holding_periods.median_holding_period.toFixed(1)}天
                    </p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <p className="text-sm text-gray-500">短期持仓</p>
                    <p className="text-lg font-bold text-blue-600">
                      {normalizedData.holding_periods.short_term_positions}
                    </p>
                    <p className="text-xs text-gray-500">≤7天</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <p className="text-sm text-gray-500">长期持仓</p>
                    <p className="text-lg font-bold text-green-600">
                      {normalizedData.holding_periods.long_term_positions}
                    </p>
                    <p className="text-xs text-gray-500">&gt;30天</p>
                  </div>
                </div>
              </CardBody>
            </Card>
          </Tab>
        )}

        {/* 风险集中度分析 */}
        {normalizedData?.concentration_risk && (
          <Tab key="concentration" title={
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4" />
              风险集中度
            </div>
          }>
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">风险集中度分析</h3>
              </CardHeader>
              <CardBody>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* 交易集中度 */}
                  {normalizedData.concentration_risk.trade_concentration && (
                    <div className="space-y-4">
                      <h4 className="font-semibold text-lg">交易集中度</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-blue-50 rounded">
                          <p className="text-sm text-gray-600">HHI指数</p>
                          <p className="text-2xl font-bold">
                            {normalizedData.concentration_risk.trade_concentration.hhi.toFixed(3)}
                          </p>
                        </div>
                        <div className="p-4 bg-green-50 rounded">
                          <p className="text-sm text-gray-600">有效股票数</p>
                          <p className="text-2xl font-bold">
                            {normalizedData.concentration_risk.trade_concentration.effective_stocks.toFixed(1)}
                          </p>
                        </div>
                        <div className="p-4 bg-yellow-50 rounded">
                          <p className="text-sm text-gray-600">前3大权重</p>
                          <p className="text-2xl font-bold">
                            {(normalizedData.concentration_risk.trade_concentration.top_3_weight * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="p-4 bg-purple-50 rounded">
                          <p className="text-sm text-gray-600">前5大权重</p>
                          <p className="text-2xl font-bold">
                            {(normalizedData.concentration_risk.trade_concentration.top_5_weight * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  {/* 持仓集中度 */}
                  {normalizedData.concentration_risk.position_concentration && (
                    <div className="space-y-4">
                      <h4 className="font-semibold text-lg">持仓集中度</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-blue-50 rounded">
                          <p className="text-sm text-gray-600">HHI指数</p>
                          <p className="text-2xl font-bold">
                            {normalizedData.concentration_risk.position_concentration.hhi.toFixed(3)}
                          </p>
                        </div>
                        <div className="p-4 bg-green-50 rounded">
                          <p className="text-sm text-gray-600">有效持仓数</p>
                          <p className="text-2xl font-bold">
                            {normalizedData.concentration_risk.position_concentration.effective_positions.toFixed(1)}
                          </p>
                        </div>
                        <div className="p-4 bg-yellow-50 rounded">
                          <p className="text-sm text-gray-600">前3大权重</p>
                          <p className="text-2xl font-bold">
                            {(normalizedData.concentration_risk.position_concentration.top_3_weight * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="p-4 bg-purple-50 rounded">
                          <p className="text-sm text-gray-600">前5大权重</p>
                          <p className="text-2xl font-bold">
                            {(normalizedData.concentration_risk.position_concentration.top_5_weight * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardBody>
            </Card>
          </Tab>
        )}
      </Tabs>

      {/* 股票详情模态框 */}
      <Modal isOpen={isDetailOpen} onClose={onDetailClose} size="lg">
        <ModalContent>
          <ModalHeader>股票详细分析</ModalHeader>
          <ModalBody>
            {selectedStock && (
              <div className="space-y-4">
                <div className="text-center border-b pb-4">
                  <h3 className="text-2xl font-bold font-mono">{selectedStock.stock_code}</h3>
                  <p className="text-gray-600">{selectedStock.stock_name}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-500">总收益</p>
                    <p className={`text-2xl font-bold ${
                      selectedStock.total_return > 0 ? 'text-green-600' : 
                      selectedStock.total_return < 0 ? 'text-red-600' : 'text-gray-600'
                    }`}>
                      {formatCurrency(selectedStock.total_return)}
                    </p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-500">胜率</p>
                    <p className="text-2xl font-bold">
                      {formatPercent(selectedStock.win_rate)}
                    </p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-500">交易次数</p>
                    <p className="text-2xl font-bold">{selectedStock.trade_count}</p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-500">平均持仓期</p>
                    <p className="text-2xl font-bold">{selectedStock.avg_holding_period} 天</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                  <div className="text-center">
                    <p className="text-sm text-gray-500">盈利交易</p>
                    <p className="text-xl font-bold text-green-600">{selectedStock.winning_trades}</p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-500">亏损交易</p>
                    <p className="text-xl font-bold text-red-600">{selectedStock.losing_trades}</p>
                  </div>
                </div>

                {/* 扩展信息 */}
                {(selectedStock.avg_win !== undefined || selectedStock.avg_loss !== undefined || selectedStock.profit_factor !== undefined) && (
                  <div className="pt-4 border-t">
                    <h4 className="text-lg font-semibold mb-3">盈亏分析</h4>
                    <div className="grid grid-cols-2 gap-4">
                      {selectedStock.avg_win !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">平均盈利</p>
                          <p className="text-lg font-bold text-green-600">{formatCurrency(selectedStock.avg_win)}</p>
                        </div>
                      )}
                      {selectedStock.avg_loss !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">平均亏损</p>
                          <p className="text-lg font-bold text-red-600">{formatCurrency(selectedStock.avg_loss)}</p>
                        </div>
                      )}
                      {selectedStock.largest_win !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">最大盈利</p>
                          <p className="text-lg font-bold text-green-600">{formatCurrency(selectedStock.largest_win)}</p>
                        </div>
                      )}
                      {selectedStock.largest_loss !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">最大亏损</p>
                          <p className="text-lg font-bold text-red-600">{formatCurrency(selectedStock.largest_loss)}</p>
                        </div>
                      )}
                      {selectedStock.profit_factor !== undefined && (
                        <div className="col-span-2">
                          <p className="text-sm text-gray-500">盈亏比</p>
                          <p className={`text-lg font-bold ${selectedStock.profit_factor >= 1 ? 'text-green-600' : 'text-red-600'}`}>
                            {selectedStock.profit_factor.toFixed(2)}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 价格分析 */}
                {(selectedStock.avg_buy_price !== undefined || selectedStock.avg_sell_price !== undefined || selectedStock.price_improvement !== undefined) && (
                  <div className="pt-4 border-t">
                    <h4 className="text-lg font-semibold mb-3">价格分析</h4>
                    <div className="grid grid-cols-2 gap-4">
                      {selectedStock.avg_buy_price !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">平均买入价</p>
                          <p className="text-lg font-bold">¥{selectedStock.avg_buy_price.toFixed(2)}</p>
                        </div>
                      )}
                      {selectedStock.avg_sell_price !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">平均卖出价</p>
                          <p className="text-lg font-bold">¥{selectedStock.avg_sell_price.toFixed(2)}</p>
                        </div>
                      )}
                      {selectedStock.price_improvement !== undefined && (
                        <div className="col-span-2">
                          <p className="text-sm text-gray-500">价格改善率</p>
                          <p className={`text-lg font-bold ${selectedStock.price_improvement >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {(selectedStock.price_improvement * 100).toFixed(2)}%
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 持仓期详情 */}
                {(selectedStock.max_holding_period !== undefined || selectedStock.min_holding_period !== undefined) && (
                  <div className="pt-4 border-t">
                    <h4 className="text-lg font-semibold mb-3">持仓期详情</h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-sm text-gray-500">平均持仓期</p>
                        <p className="text-lg font-bold">{selectedStock.avg_holding_period} 天</p>
                      </div>
                      {selectedStock.max_holding_period !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">最长持仓期</p>
                          <p className="text-lg font-bold">{selectedStock.max_holding_period} 天</p>
                        </div>
                      )}
                      {selectedStock.min_holding_period !== undefined && (
                        <div>
                          <p className="text-sm text-gray-500">最短持仓期</p>
                          <p className="text-lg font-bold">{selectedStock.min_holding_period} 天</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                <div className="pt-4 border-t">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-500">胜率进度</span>
                    <span className="text-sm font-mono">{formatPercent(selectedStock.win_rate)}</span>
                  </div>
                  <Progress
                    value={selectedStock.win_rate * 100}
                    color={selectedStock.win_rate >= 0.5 ? 'success' : 'danger'}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </ModalBody>
          <ModalFooter>
            <Button color="primary" onPress={onDetailClose}>
              关闭
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}