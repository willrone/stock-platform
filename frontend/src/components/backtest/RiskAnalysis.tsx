/**
 * 风险分析组件
 * 实现扩展风险指标展示、收益分布直方图和正态性检验
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab,
  LinearProgress,
  Chip,
  Tooltip,
  Select,
  MenuItem,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Box,
  Typography,
  FormControl,
  InputLabel,
} from '@mui/material';
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
  
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const onDetailOpen = () => setIsDetailOpen(true);
  const onDetailClose = () => setIsDetailOpen(false);
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
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Calculator size={48} color="#999" style={{ margin: '0 auto 8px' }} />
            <Typography variant="body2" color="text.secondary">暂无风险分析数据</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 风险指标概览 */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {Object.entries(riskCategories).map(([categoryKey, category]) => {
          const IconComponent = category.icon;
          
          return (
            <Card key={categoryKey}>
              <CardHeader
                title={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <IconComponent size={20} />
                    <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                      {category.title}
                    </Typography>
                  </Box>
                }
              />
              <CardContent>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2 }}>
                  {category.metrics.map((metric) => {
                    const value = riskMetrics[metric.key as keyof RiskMetrics];
                    const riskLevel = getRiskLevel(metric.key, value);
                    
                    return (
                      <Box
                        key={metric.key}
                        sx={{ 
                          p: 2, 
                          border: 1, 
                          borderColor: 'divider', 
                          borderRadius: 1, 
                          cursor: 'pointer',
                          '&:hover': { bgcolor: 'grey.50' },
                          transition: 'background-color 0.2s'
                        }}
                        onClick={() => handleRiskMetricClick(metric.key, metric.name, metric.description)}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                          <Box>
                            <Typography variant="caption" color="text.secondary">{metric.name}</Typography>
                            <Typography variant="h5" sx={{ fontWeight: 600 }}>
                              {formatValue(value, metric.format)}
                            </Typography>
                          </Box>
                          <Chip
                            label={riskLevel.level}
                            size="small"
                            color={riskLevel.color as any}
                          />
                        </Box>
                        <Tooltip title={metric.description}>
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {metric.description}
                          </Typography>
                        </Tooltip>
                      </Box>
                    );
                  })}
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {/* VaR 和 CVaR 特殊展示 */}
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AlertTriangle size={20} color="#ed6c02" />
              <Typography variant="h6" component="h3" sx={{ fontWeight: 600, color: 'warning.main' }}>
                风险价值 (VaR) 分析
              </Typography>
            </Box>
          }
        />
        <CardContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>风险价值 (VaR)</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary">95% 置信度</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500, color: 'error.main' }}>
                    {formatValue(riskMetrics.var_95, 'percent')}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary">99% 置信度</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500, color: 'error.main' }}>
                    {formatValue(riskMetrics.var_99, 'percent')}
                  </Typography>
                </Box>
              </Box>
            </Box>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>条件风险价值 (CVaR)</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary">95% 置信度</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500, color: 'error.main' }}>
                    {formatValue(riskMetrics.cvar_95, 'percent')}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary">99% 置信度</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500, color: 'error.main' }}>
                    {formatValue(riskMetrics.cvar_99, 'percent')}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* 收益分布和正态性检验 */}
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <BarChart3 size={20} />
                <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                  收益分布分析
                </Typography>
              </Box>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>分布类型</InputLabel>
                <Select
                  value={selectedDistribution}
                  label="分布类型"
                  onChange={(e) => setSelectedDistribution(e.target.value as 'daily' | 'monthly')}
                >
                  <MenuItem value="daily">日收益</MenuItem>
                  <MenuItem value="monthly">月收益</MenuItem>
                </Select>
              </FormControl>
            </Box>
          }
        />
        <CardContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 3 }}>
            {/* 分布图表 */}
            <Box>
              <Box ref={distributionChartRef} sx={{ height: 300, width: '100%' }} />
            </Box>
            
            {/* 统计信息 */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  分布特征
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">偏度</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {returnDistribution.skewness.toFixed(3)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">峰度</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {returnDistribution.kurtosis.toFixed(3)}
                    </Typography>
                  </Box>
                </Box>
              </Box>
              
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  分位数
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {[5, 25, 50, 75, 95].map(percentile => (
                    <Box key={percentile} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary">{percentile}%</Typography>
                      <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                        {formatValue(returnDistribution.percentiles[`p${percentile}` as keyof typeof returnDistribution.percentiles], 'percent')}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
              
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  正态性检验
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">检验统计量</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {returnDistribution.normality_test.statistic.toFixed(3)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">P值</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {returnDistribution.normality_test.p_value.toFixed(4)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">正态分布</Typography>
                    <Chip
                      label={returnDistribution.normality_test.is_normal ? '是' : '否'}
                      size="small"
                      color={returnDistribution.normality_test.is_normal ? 'success' : 'error'}
                    />
                  </Box>
                </Box>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* 滚动指标图表 */}
      <Card>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Activity size={20} />
                <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                  滚动风险指标
                </Typography>
              </Box>
              <FormControl size="small" sx={{ minWidth: 160 }}>
                <InputLabel>指标类型</InputLabel>
                <Select
                  value={selectedMetric}
                  label="指标类型"
                  onChange={(e) => setSelectedMetric(e.target.value as keyof RollingMetrics)}
                >
                  <MenuItem value="rolling_sharpe">滚动夏普比率</MenuItem>
                  <MenuItem value="rolling_volatility">滚动波动率</MenuItem>
                  <MenuItem value="rolling_drawdown">滚动回撤</MenuItem>
                  <MenuItem value="rolling_beta">滚动Beta</MenuItem>
                </Select>
              </FormControl>
            </Box>
          }
        />
        <CardContent>
          <Box ref={rollingChartRef} sx={{ height: 400, width: '100%' }} />
        </CardContent>
      </Card>

      {/* 风险指标详情模态框 */}
      <Dialog open={isDetailOpen} onClose={onDetailClose} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Info size={20} />
            风险指标详情
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedRiskMetric && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="body2">{selectedRiskMetric}</Typography>
              
              <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                  计算说明
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  该指标基于历史回测数据计算得出，用于评估策略的风险收益特征。
                  请结合其他指标综合分析，避免单一指标判断。
                </Typography>
              </Box>
              
              <Box sx={{ bgcolor: 'primary.light', p: 2, borderRadius: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1, color: 'primary.dark' }}>
                  使用建议
                </Typography>
                <Typography variant="body2" sx={{ color: 'primary.dark' }}>
                    建议将该指标与同类策略或基准进行对比，
                    并考虑市场环境变化对指标的影响。
                </Typography>
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