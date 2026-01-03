/**
 * 回测概览组件
 * 展示回测任务的关键指标和概览信息
 */

'use client';

import React from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Chip,
  Progress,
  Tooltip,
} from '@heroui/react';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  DollarSign,
  Activity,
  AlertTriangle,
  Info,
} from 'lucide-react';

interface BacktestOverviewProps {
  backtestData: any;
  loading?: boolean;
}

interface BacktestMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
}

export default function BacktestOverview({ backtestData, loading = false }: BacktestOverviewProps) {
  // 处理回测数据，提取关键指标
  const processMetrics = (): BacktestMetrics => {
    if (!backtestData) {
      return {
        totalReturn: 0,
        annualizedReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        volatility: 0,
        winRate: 0,
        totalTrades: 0,
        profitFactor: 0,
      };
    }

    return {
      totalReturn: (backtestData.total_return || 0) * 100,
      annualizedReturn: (backtestData.annualized_return || 0) * 100,
      sharpeRatio: backtestData.sharpe_ratio || 0,
      maxDrawdown: (backtestData.max_drawdown || 0) * 100,
      volatility: (backtestData.volatility || 0) * 100,
      winRate: (backtestData.win_rate || 0) * 100,
      totalTrades: backtestData.total_trades || 0,
      profitFactor: backtestData.profit_factor || 0,
    };
  };

  const metrics = processMetrics();

  // 获取收益率颜色
  const getReturnColor = (value: number) => {
    if (value > 0) return 'text-success';
    if (value < 0) return 'text-danger';
    return 'text-default-500';
  };

  // 获取收益率图标
  const getReturnIcon = (value: number) => {
    if (value > 0) return <TrendingUp className="w-5 h-5 text-success" />;
    if (value < 0) return <TrendingDown className="w-5 h-5 text-danger" />;
    return <Activity className="w-5 h-5 text-default-500" />;
  };

  // 获取夏普比率评级
  const getSharpeRating = (sharpe: number) => {
    if (sharpe >= 2) return { text: '优秀', color: 'success' as const };
    if (sharpe >= 1) return { text: '良好', color: 'primary' as const };
    if (sharpe >= 0.5) return { text: '一般', color: 'warning' as const };
    return { text: '较差', color: 'danger' as const };
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, index) => (
          <Card key={index}>
            <CardBody className="text-center">
              <div className="animate-pulse">
                <div className="w-8 h-8 bg-default-200 rounded-full mx-auto mb-3"></div>
                <div className="h-4 bg-default-200 rounded mb-2"></div>
                <div className="h-6 bg-default-200 rounded"></div>
              </div>
            </CardBody>
          </Card>
        ))}
      </div>
    );
  }

  if (!backtestData) {
    return (
      <Card>
        <CardBody>
          <div className="flex items-center justify-center h-32 text-default-500">
            <AlertTriangle className="w-8 h-8 mr-2" />
            <span>暂无回测数据</span>
          </div>
        </CardBody>
      </Card>
    );
  }

  const sharpeRating = getSharpeRating(metrics.sharpeRatio);

  return (
    <div className="space-y-6">
      {/* 核心指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 总收益率 */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              {getReturnIcon(metrics.totalReturn)}
            </div>
            <Tooltip content="策略在整个回测期间的总收益率">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                总收益率
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className={`text-2xl font-bold ${getReturnColor(metrics.totalReturn)}`}>
              {metrics.totalReturn >= 0 ? '+' : ''}{metrics.totalReturn.toFixed(2)}%
            </p>
          </CardBody>
        </Card>

        {/* 年化收益率 */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <BarChart3 className="w-5 h-5 text-primary" />
            </div>
            <Tooltip content="将总收益率按年化计算的收益率">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                年化收益率
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className={`text-2xl font-bold ${getReturnColor(metrics.annualizedReturn)}`}>
              {metrics.annualizedReturn >= 0 ? '+' : ''}{metrics.annualizedReturn.toFixed(2)}%
            </p>
          </CardBody>
        </Card>

        {/* 夏普比率 */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <Target className="w-5 h-5 text-secondary" />
            </div>
            <Tooltip content="衡量风险调整后收益的指标，数值越高越好">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                夏普比率
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <div className="flex items-center justify-center space-x-2">
              <p className="text-2xl font-bold">{metrics.sharpeRatio.toFixed(3)}</p>
              <Chip color={sharpeRating.color} variant="flat" size="sm">
                {sharpeRating.text}
              </Chip>
            </div>
          </CardBody>
        </Card>

        {/* 最大回撤 */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <TrendingDown className="w-5 h-5 text-danger" />
            </div>
            <Tooltip content="策略在回测期间的最大亏损幅度">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                最大回撤
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className="text-2xl font-bold text-danger">
              -{Math.abs(metrics.maxDrawdown).toFixed(2)}%
            </p>
          </CardBody>
        </Card>
      </div>

      {/* 次要指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 波动率 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <Activity className="w-5 h-5 text-warning" />
            </div>
            <Tooltip content="策略收益的波动程度，数值越低越稳定">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                波动率
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className="text-xl font-bold">{metrics.volatility.toFixed(2)}%</p>
          </CardBody>
        </Card>

        {/* 胜率 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <Target className="w-5 h-5 text-success" />
            </div>
            <Tooltip content="盈利交易占总交易次数的比例">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                胜率
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <div className="space-y-2">
              <p className="text-xl font-bold">{metrics.winRate.toFixed(1)}%</p>
              <Progress 
                value={metrics.winRate} 
                color={metrics.winRate >= 50 ? 'success' : 'warning'}
                size="sm"
              />
            </div>
          </CardBody>
        </Card>

        {/* 交易次数 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <BarChart3 className="w-5 h-5 text-primary" />
            </div>
            <Tooltip content="回测期间的总交易次数">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                交易次数
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className="text-xl font-bold">{metrics.totalTrades}</p>
          </CardBody>
        </Card>

        {/* 盈亏比 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <DollarSign className="w-5 h-5 text-secondary" />
            </div>
            <Tooltip content="平均盈利交易与平均亏损交易的比值">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                盈亏比
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className={`text-xl font-bold ${metrics.profitFactor >= 1 ? 'text-success' : 'text-danger'}`}>
              {metrics.profitFactor.toFixed(2)}
            </p>
          </CardBody>
        </Card>
      </div>

      {/* 风险评估总结 */}
      <Card>
        <CardHeader>
          <h4 className="text-lg font-semibold">风险评估总结</h4>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  metrics.sharpeRatio >= 1 ? 'bg-success' : 
                  metrics.sharpeRatio >= 0.5 ? 'bg-warning' : 'bg-danger'
                }`}></div>
                <span className="text-sm font-medium">风险调整收益</span>
              </div>
              <p className="text-xs text-default-500">
                {metrics.sharpeRatio >= 1 ? '良好' : 
                 metrics.sharpeRatio >= 0.5 ? '一般' : '需要改进'}
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  Math.abs(metrics.maxDrawdown) <= 10 ? 'bg-success' : 
                  Math.abs(metrics.maxDrawdown) <= 20 ? 'bg-warning' : 'bg-danger'
                }`}></div>
                <span className="text-sm font-medium">回撤控制</span>
              </div>
              <p className="text-xs text-default-500">
                {Math.abs(metrics.maxDrawdown) <= 10 ? '优秀' : 
                 Math.abs(metrics.maxDrawdown) <= 20 ? '良好' : '需要改进'}
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  metrics.winRate >= 50 ? 'bg-success' : 
                  metrics.winRate >= 40 ? 'bg-warning' : 'bg-danger'
                }`}></div>
                <span className="text-sm font-medium">交易胜率</span>
              </div>
              <p className="text-xs text-default-500">
                {metrics.winRate >= 50 ? '良好' : 
                 metrics.winRate >= 40 ? '一般' : '需要改进'}
              </p>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}