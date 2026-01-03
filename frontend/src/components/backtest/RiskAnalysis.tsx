/**
 * 风险分析组件
 * 实现扩展风险指标展示、收益分布直方图和正态性检验
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Card,
  CardBody,
  CardHeader,
  Tabs,
  Tab,
  Progress,
  Chip,
  Tooltip,
  Select,
  SelectItem,
  Button,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
} from '@heroui/react';
import * as echarts from 'echarts';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  PieChart as PieChartIcon, 
  AlertTriangle, 
  Target,
  Calculator,
  Activity,
  Info
} from 'lucide-react';

interface RiskMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  information_ratio: number;
  max_drawdown: number;
  avg_drawdown: number;
  drawdown_recovery_time: number;
  volatility_daily: number;
  volatility_monthly: number;
  volatility_annual: number;
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  beta: number;
  alpha: number;
  tracking_error: number;
  upside_capture: number;
  downside_capture: number;
}

interface ReturnDistribution {
  daily_returns: number[];
  monthly_returns: number[];
  return_bins: number[];
  return_frequencies: number[];
  normality_test: {
    statistic: number;
    p_value: number;
    is_normal: boolean;
  };
  skewness: number;
  kurtosis: number;
  percentiles: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  };
}

interface RollingMetrics {
  dates: string[];
  rolling_sharpe: number[];
  rolling_volatility: number[];
  rolling_drawdown: number[];
  rolling_beta: number[];
  window_size: number;
}

interface RiskAnalysisProps {
  taskId: string;
  riskMetrics: RiskMetrics;
  returnDistribution: ReturnDistribution;
  rollingMetrics: RollingMetrics;
}

export function RiskAnalysis({ 
  taskId, 
  riskMetrics, 
  returnDistribution, 
  rollingMetrics 
}: RiskAnalysisProps) {
  const [selectedMetric, setSelectedMetric] = useState<keyof RollingMetrics>('rolling_sharpe');
  const [selectedDistribution, setSelectedDistribution] = useState<'daily' | 'monthly'>('daily');
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<'95' | '99'>('95');
  
  const { isOpen: isDetailOpen, onOpen: onDetailOpen, onClose: onDetailClose } = useDisclosure();
  const [selectedRiskMetric, setSelectedRiskMetric] = useState<string | null>(null);
  
  // 图表引用
  const distributionChartRef = useRef<HTMLDivElement>(null);
  const rollingChartRef = useRef<HTMLDivElement>(null);
  const distributionChartInstance = useRef<echarts.ECharts | null>(null);
  const rollingChartInstance = useRef<echarts.ECharts | null>(null);

  // 风险指标分类
  const riskCategories = useMemo(() => ({
    performance: {
      title: '绩效指标',
      icon: TrendingUp,
      color: 'text-blue-500',
      metrics: [
        { key: 'sharpe_ratio', name: '夏普比率', description: '风险调整后收益指标', format: 'ratio' },
        { key: 'sortino_ratio', name: '索提诺比率', description: '下行风险调整收益', format: 'ratio' },
        { key: 'calmar_ratio', name: '卡玛比率', description: '年化收益/最大回撤', format: 'ratio' },
        { key: 'information_ratio', name: '信息比率', description: '超额收益/跟踪误差', format: 'ratio' },
      ]
    },
    risk: {
      title: '风险指标',
      icon: AlertTriangle,
      color: 'text-red-500',
      metrics: [
        { key: 'max_drawdown', name: '最大回撤', description: '历史最大亏损幅度', format: 'percent' },
        { key: 'avg_drawdown', name: '平均回撤', description: '平均回撤幅度', format: 'percent' },
        { key: 'volatility_annual', name: '年化波动率', description: '收益率标准差', format: 'percent' },
        { key: 'var_95', name: 'VaR(95%)', description: '95%置信度风险价值', format: 'percent' },
      ]
    },
    market: {
      title: '市场相关性',
      icon: Target,
      color: 'text-purple-500',
      metrics: [
        { key: 'beta', name: 'Beta系数', description: '相对市场的敏感度', format: 'ratio' },
        { key: 'alpha', name: 'Alpha系数', description: '超额收益能力', format: 'percent' },
        { key: 'tracking_error', name: '跟踪误差', description: '相对基准的波动', format: 'percent' },
        { key: 'upside_capture', name: '上涨捕获率', description: '牛市中的表现', format: 'percent' },
      ]
    }
  }), []);

  // 收益分布数据
  const distributionData = useMemo(() => {
    if (!returnDistribution) return { returns: [], bins: [], frequencies: [] };
    
    const returns = selectedDistribution === 'daily' 
      ? returnDistribution.daily_returns 
      : returnDistribution.monthly_returns;
    
    return {
      returns,
      bins: returnDistribution.return_bins,
      frequencies: returnDistribution.return_frequencies
    };
  }, [returnDistribution, selectedDistribution]);

  // 滚动指标数据
  const rollingData = useMemo(() => {
    if (!rollingMetrics) return { dates: [], values: [] };
    
    return {
      dates: rollingMetrics.dates,
      values: rollingMetrics[selectedMetric] || []
    };
  }, [rollingMetrics, selectedMetric]);

  // 初始化收益分布图表
  useEffect(() => {
    if (!distributionChartRef.current || !distributionData.bins.length) return;

    if (distributionChartInstance.current) {
      distributionChartInstance.current.dispose();
    }

    distributionChartInstance.current = echarts.init(distributionChartRef.current);

    const option = {
      title: {
        text: `${selectedDistribution === 'daily' ? '日' : '月'}收益率分布`,
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
          return `收益率区间: ${(param.name * 100).toFixed(2)}%<br/>频次: ${param.value}`;
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
        data: distributionData.bins.map(bin => (bin * 100).toFixed(1)),
        name: '收益率 (%)',
        nameLocation: 'middle',
        nameGap: 30,
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        name: '频次',
        nameLocation: 'middle',
        nameGap: 40,
      },
      series: [
        {
          name: '频次',
          type: 'bar',
          data: distributionData.frequencies,
          itemStyle: {
            color: '#3b82f6',
          },
          markLine: {
            data: [
              {
                name: '均值',
                xAxis: distributionData.returns.reduce((a, b) => a + b, 0) / distributionData.returns.length * 100,
                lineStyle: {
                  color: '#ef4444',
                  type: 'dashed',
                },
                label: {
                  formatter: '均值',
                },
              },
            ],
          },
        },
      ],
    };

    distributionChartInstance.current.setOption(option);

    const handleResize = () => {
      if (distributionChartInstance.current) {
        distributionChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [distributionData, selectedDistribution]);

  // 初始化滚动指标图表
  useEffect(() => {
    if (!rollingChartRef.current || !rollingData.dates.length) return;

    if (rollingChartInstance.current) {
      rollingChartInstance.current.dispose();
    }

    rollingChartInstance.current = echarts.init(rollingChartRef.current);

    const getMetricName = () => {
      switch (selectedMetric) {
        case 'rolling_sharpe':
          return '滚动夏普比率';
        case 'rolling_volatility':
          return '滚动波动率';
        case 'rolling_drawdown':
          return '滚动回撤';
        case 'rolling_beta':
          return '滚动Beta';
        default:
          return '滚动指标';
      }
    };

    const option = {
      title: {
        text: `${getMetricName()} (${rollingMetrics?.window_size || 60}日窗口)`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        formatter: function (params: any) {
          const param = params[0];
          const value = param.value;
          
          if (selectedMetric === 'rolling_volatility' || selectedMetric === 'rolling_drawdown') {
            return `${param.axisValue}<br/>${getMetricName()}: ${(value * 100).toFixed(2)}%`;
          }
          return `${param.axisValue}<br/>${getMetricName()}: ${value.toFixed(3)}`;
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
        data: rollingData.dates,
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: function (value: number) {
            if (selectedMetric === 'rolling_volatility' || selectedMetric === 'rolling_drawdown') {
              return `${(value * 100).toFixed(1)}%`;
            }
            return value.toFixed(2);
          },
        },
      },
      series: [
        {
          name: getMetricName(),
          type: 'line',
          data: rollingData.values,
          smooth: true,
          itemStyle: {
            color: '#10b981',
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(16, 185, 129, 0.3)' },
              { offset: 1, color: 'rgba(16, 185, 129, 0.1)' },
            ]),
          },
        },
      ],
    };

    rollingChartInstance.current.setOption(option);

    const handleResize = () => {
      if (rollingChartInstance.current) {
        rollingChartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [rollingData, selectedMetric, rollingMetrics?.window_size]);

  // 格式化函数
  const formatValue = (value: number, format: string) => {
    switch (format) {
      case 'percent':
        return `${(value * 100).toFixed(2)}%`;
      case 'ratio':
        return value.toFixed(3);
      case 'days':
        return `${value.toFixed(0)} 天`;
      default:
        return value.toFixed(2);
    }
  };

  const getRiskLevel = (key: string, value: number) => {
    // 根据不同指标判断风险等级
    switch (key) {
      case 'sharpe_ratio':
        if (value > 1.5) return { level: 'excellent', color: 'success' };
        if (value > 1.0) return { level: 'good', color: 'primary' };
        if (value > 0.5) return { level: 'fair', color: 'warning' };
        return { level: 'poor', color: 'danger' };
      
      case 'max_drawdown':
        if (Math.abs(value) < 0.05) return { level: 'excellent', color: 'success' };
        if (Math.abs(value) < 0.10) return { level: 'good', color: 'primary' };
        if (Math.abs(value) < 0.20) return { level: 'fair', color: 'warning' };
        return { level: 'poor', color: 'danger' };
      
      case 'volatility_annual':
        if (value < 0.10) return { level: 'excellent', color: 'success' };
        if (value < 0.15) return { level: 'good', color: 'primary' };
        if (value < 0.25) return { level: 'fair', color: 'warning' };
        return { level: 'poor', color: 'danger' };
      
      default:
        return { level: 'neutral', color: 'default' };
    }
  };

  const handleRiskMetricClick = (key: string, name: string, description: string) => {
    setSelectedRiskMetric(`${name}: ${description}`);
    onDetailOpen();
  };

  if (!riskMetrics || !returnDistribution || !rollingMetrics) {
    return (
      <Card>
        <CardBody className="flex items-center justify-center h-64">
          <div className="text-center text-gray-500">
            <Calculator className="w-12 h-12 mx-auto mb-2" />
            <p>暂无风险分析数据</p>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 风险指标概览 */}
      <div className="space-y-4">
        {Object.entries(riskCategories).map(([categoryKey, category]) => {
          const IconComponent = category.icon;
          
          return (
            <Card key={categoryKey}>
              <CardHeader className="pb-2">
                <h3 className={`text-lg font-semibold flex items-center gap-2 ${category.color}`}>
                  <IconComponent className="w-5 h-5" />
                  {category.title}
                </h3>
              </CardHeader>
              <CardBody className="pt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {category.metrics.map((metric) => {
                    const value = riskMetrics[metric.key as keyof RiskMetrics];
                    const riskLevel = getRiskLevel(metric.key, value);
                    
                    return (
                      <div
                        key={metric.key}
                        className="p-4 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors"
                        onClick={() => handleRiskMetricClick(metric.key, metric.name, metric.description)}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <p className="text-sm text-gray-500">{metric.name}</p>
                            <p className="text-xl font-bold">
                              {formatValue(value, metric.format)}
                            </p>
                          </div>
                          <Chip
                            size="sm"
                            color={riskLevel.color as any}
                            variant="flat"
                          >
                            {riskLevel.level}
                          </Chip>
                        </div>
                        <Tooltip content={metric.description}>
                          <p className="text-xs text-gray-400 truncate">
                            {metric.description}
                          </p>
                        </Tooltip>
                      </div>
                    );
                  })}
                </div>
              </CardBody>
            </Card>
          );
        })}
      </div>

      {/* VaR 和 CVaR 特殊展示 */}
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold flex items-center gap-2 text-orange-500">
            <AlertTriangle className="w-5 h-5" />
            风险价值 (VaR) 分析
          </h3>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium">风险价值 (VaR)</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">95% 置信度</span>
                  <span className="font-mono font-medium text-red-600">
                    {formatValue(riskMetrics.var_95, 'percent')}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">99% 置信度</span>
                  <span className="font-mono font-medium text-red-600">
                    {formatValue(riskMetrics.var_99, 'percent')}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-medium">条件风险价值 (CVaR)</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">95% 置信度</span>
                  <span className="font-mono font-medium text-red-600">
                    {formatValue(riskMetrics.cvar_95, 'percent')}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">99% 置信度</span>
                  <span className="font-mono font-medium text-red-600">
                    {formatValue(riskMetrics.cvar_99, 'percent')}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardBody>
      </Card>

      {/* 收益分布和正态性检验 */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              收益分布分析
            </h3>
            <select
              value={selectedDistribution}
              onChange={(e) => setSelectedDistribution(e.target.value as 'daily' | 'monthly')}
              className="px-3 py-1 border rounded text-sm"
            >
              <option value="daily">日收益</option>
              <option value="monthly">月收益</option>
            </select>
          </div>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 分布图表 */}
            <div className="lg:col-span-2">
              <div ref={distributionChartRef} style={{ height: '300px', width: '100%' }} />
            </div>
            
            {/* 统计信息 */}
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">分布特征</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">偏度</span>
                    <span className="font-mono">{returnDistribution.skewness.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">峰度</span>
                    <span className="font-mono">{returnDistribution.kurtosis.toFixed(3)}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">分位数</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">5%</span>
                    <span className="font-mono">{formatValue(returnDistribution.percentiles.p5, 'percent')}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">25%</span>
                    <span className="font-mono">{formatValue(returnDistribution.percentiles.p25, 'percent')}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">50%</span>
                    <span className="font-mono">{formatValue(returnDistribution.percentiles.p50, 'percent')}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">75%</span>
                    <span className="font-mono">{formatValue(returnDistribution.percentiles.p75, 'percent')}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">95%</span>
                    <span className="font-mono">{formatValue(returnDistribution.percentiles.p95, 'percent')}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">正态性检验</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">检验统计量</span>
                    <span className="font-mono text-sm">{returnDistribution.normality_test.statistic.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">P值</span>
                    <span className="font-mono text-sm">{returnDistribution.normality_test.p_value.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">正态分布</span>
                    <Chip
                      size="sm"
                      color={returnDistribution.normality_test.is_normal ? 'success' : 'danger'}
                      variant="flat"
                    >
                      {returnDistribution.normality_test.is_normal ? '是' : '否'}
                    </Chip>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardBody>
      </Card>

      {/* 滚动指标图表 */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Activity className="w-5 h-5" />
              滚动风险指标
            </h3>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as keyof RollingMetrics)}
              className="px-3 py-1 border rounded text-sm"
            >
              <option value="rolling_sharpe">滚动夏普比率</option>
              <option value="rolling_volatility">滚动波动率</option>
              <option value="rolling_drawdown">滚动回撤</option>
              <option value="rolling_beta">滚动Beta</option>
            </select>
          </div>
        </CardHeader>
        <CardBody>
          <div ref={rollingChartRef} style={{ height: '400px', width: '100%' }} />
        </CardBody>
      </Card>

      {/* 风险指标详情模态框 */}
      <Modal isOpen={isDetailOpen} onClose={onDetailClose} size="lg">
        <ModalContent>
          <ModalHeader className="flex items-center gap-2">
            <Info className="w-5 h-5" />
            风险指标详情
          </ModalHeader>
          <ModalBody>
            {selectedRiskMetric && (
              <div className="space-y-4">
                <p className="text-gray-700">{selectedRiskMetric}</p>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-2">计算说明</h4>
                  <p className="text-sm text-gray-600">
                    该指标基于历史回测数据计算得出，用于评估策略的风险收益特征。
                    请结合其他指标综合分析，避免单一指标判断。
                  </p>
                </div>
                
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-2 text-blue-800">使用建议</h4>
                  <p className="text-sm text-blue-700">
                    建议将该指标与同类策略或基准进行对比，
                    并考虑市场环境变化对指标的影响。
                  </p>
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