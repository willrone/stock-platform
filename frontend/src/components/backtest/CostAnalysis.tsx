/**
 * 成本分析组件
 * 展示有成本/无成本收益对比和交易成本明细
 */

'use client';

import React, { useMemo } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Tabs,
  Tab,
  Chip,
  Tooltip,
} from '@heroui/react';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Info,
  BarChart3,
  PieChart,
} from 'lucide-react';
import * as echarts from 'echarts';

interface CostAnalysisProps {
  backtestData: any;
  loading?: boolean;
}

export function CostAnalysis({ backtestData, loading = false }: CostAnalysisProps) {
  // 提取有成本/无成本收益数据
  const costComparison = useMemo(() => {
    if (!backtestData) return null;

    const excessReturnWithCost = backtestData.excess_return_with_cost;
    const excessReturnWithoutCost = backtestData.excess_return_without_cost;
    const costStats = backtestData.cost_statistics;
    const portfolioHistory = backtestData.portfolio_history || [];

    return {
      withCost: excessReturnWithCost || {},
      withoutCost: excessReturnWithoutCost || {},
      costStats: costStats || {},
      portfolioHistory: portfolioHistory,
    };
  }, [backtestData]);

  // 计算成本影响
  const costImpact = useMemo(() => {
    if (!costComparison) return null;

    const withCost = costComparison.withCost.annualized_return || 0;
    const withoutCost = costComparison.withoutCost.annualized_return || 0;
    const impact = withoutCost - withCost;

    return {
      impact: impact,
      impactPercent: (impact / Math.abs(withoutCost)) * 100,
      costRatio: costComparison.costStats.cost_ratio || 0,
    };
  }, [costComparison]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Card>
          <CardBody>
            <div className="animate-pulse space-y-4">
              <div className="h-4 bg-default-200 rounded w-1/4"></div>
              <div className="h-20 bg-default-200 rounded"></div>
            </div>
          </CardBody>
        </Card>
      </div>
    );
  }

  if (!costComparison || !backtestData) {
    return (
      <Card>
        <CardBody>
          <div className="text-center text-default-500 py-8">
            暂无成本分析数据
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 成本对比概览 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 含成本年化收益 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <TrendingDown className="w-5 h-5 text-danger" />
            </div>
            <Tooltip content="考虑交易成本后的年化超额收益率">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                含成本年化收益
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className={`text-2xl font-bold ${
              (costComparison.withCost.annualized_return || 0) >= 0 ? 'text-success' : 'text-danger'
            }`}>
              {((costComparison.withCost.annualized_return || 0) * 100).toFixed(2)}%
            </p>
          </CardBody>
        </Card>

        {/* 无成本年化收益 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <TrendingUp className="w-5 h-5 text-success" />
            </div>
            <Tooltip content="不考虑交易成本的年化超额收益率">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                无成本年化收益
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className={`text-2xl font-bold ${
              (costComparison.withoutCost.annualized_return || 0) >= 0 ? 'text-success' : 'text-danger'
            }`}>
              {((costComparison.withoutCost.annualized_return || 0) * 100).toFixed(2)}%
            </p>
          </CardBody>
        </Card>

        {/* 成本影响 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <DollarSign className="w-5 h-5 text-warning" />
            </div>
            <Tooltip content="交易成本对年化收益的影响">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                成本影响
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className="text-2xl font-bold text-warning">
              {(costImpact?.impact || 0) * 100 >= 0 ? '-' : '+'}
              {Math.abs((costImpact?.impact || 0) * 100).toFixed(2)}%
            </p>
          </CardBody>
        </Card>

        {/* 成本占比 */}
        <Card>
          <CardBody className="text-center">
            <div className="flex items-center justify-center mb-3">
              <PieChart className="w-5 h-5 text-secondary" />
            </div>
            <Tooltip content="总交易成本占初始资金的比例">
              <p className="text-sm text-default-500 mb-2 cursor-help flex items-center justify-center">
                成本占比
                <Info className="w-3 h-3 ml-1" />
              </p>
            </Tooltip>
            <p className="text-2xl font-bold">
              {((costImpact?.costRatio || 0) * 100).toFixed(2)}%
            </p>
          </CardBody>
        </Card>
      </div>

      {/* 详细对比 */}
      <Card>
        <CardHeader>
          <h4 className="text-lg font-semibold">有成本/无成本收益对比</h4>
        </CardHeader>
        <CardBody>
          <Tabs>
            <Tab key="metrics" title="指标对比">
              <div className="mt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <h5 className="font-semibold text-danger">含成本指标</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-default-500">平均收益:</span>
                        <span className="font-mono">
                          {((costComparison.withCost.mean || 0) * 100).toFixed(4)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">标准差:</span>
                        <span className="font-mono">
                          {((costComparison.withCost.std || 0) * 100).toFixed(4)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">年化收益:</span>
                        <span className="font-mono">
                          {((costComparison.withCost.annualized_return || 0) * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">信息比率:</span>
                        <span className="font-mono">
                          {(costComparison.withCost.information_ratio || 0).toFixed(3)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">最大回撤:</span>
                        <span className="font-mono text-danger">
                          {((costComparison.withCost.max_drawdown || 0) * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h5 className="font-semibold text-success">无成本指标</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-default-500">平均收益:</span>
                        <span className="font-mono">
                          {((costComparison.withoutCost.mean || 0) * 100).toFixed(4)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">标准差:</span>
                        <span className="font-mono">
                          {((costComparison.withoutCost.std || 0) * 100).toFixed(4)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">年化收益:</span>
                        <span className="font-mono">
                          {((costComparison.withoutCost.annualized_return || 0) * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">信息比率:</span>
                        <span className="font-mono">
                          {(costComparison.withoutCost.information_ratio || 0).toFixed(3)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-500">最大回撤:</span>
                        <span className="font-mono text-danger">
                          {((costComparison.withoutCost.max_drawdown || 0) * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Tab>

            <Tab key="costs" title="交易成本明细">
              <div className="mt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardBody className="text-center">
                      <p className="text-sm text-default-500 mb-2">总手续费</p>
                      <p className="text-2xl font-bold">
                        ¥{(costComparison.costStats.total_commission || 0).toLocaleString('zh-CN', {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2
                        })}
                      </p>
                    </CardBody>
                  </Card>

                  <Card>
                    <CardBody className="text-center">
                      <p className="text-sm text-default-500 mb-2">总滑点成本</p>
                      <p className="text-2xl font-bold">
                        ¥{(costComparison.costStats.total_slippage || 0).toLocaleString('zh-CN', {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2
                        })}
                      </p>
                    </CardBody>
                  </Card>

                  <Card>
                    <CardBody className="text-center">
                      <p className="text-sm text-default-500 mb-2">总交易成本</p>
                      <p className="text-2xl font-bold text-warning">
                        ¥{(costComparison.costStats.total_cost || 0).toLocaleString('zh-CN', {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2
                        })}
                      </p>
                    </CardBody>
                  </Card>
                </div>

                <div className="mt-4">
                  <p className="text-sm text-default-500 mb-2">
                    成本占比: {((costComparison.costStats.cost_ratio || 0) * 100).toFixed(2)}%
                  </p>
                  <div className="w-full bg-default-100 rounded-full h-2.5">
                    <div
                      className="bg-warning h-2.5 rounded-full"
                      style={{
                        width: `${Math.min((costComparison.costStats.cost_ratio || 0) * 100, 100)}%`
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </Tab>
          </Tabs>
        </CardBody>
      </Card>
    </div>
  );
}
