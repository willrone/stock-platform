/**
 * 持仓分析组件（重构版）
 * 展示各股票表现排行、持仓权重饼图和柱状图
 */

import React, { useState } from 'react';
import { Box, Card, CardContent, Tabs, Tab, Typography } from '@mui/material';
import { Target, PieChart as PieChartIcon, BarChart3, DollarSign } from 'lucide-react';
import {
  PositionData,
  EnhancedPositionAnalysis,
  SortConfig,
} from '@/utils/backtest/positionDataUtils';
import { usePositionAnalysisData } from '@/hooks/backtest/usePositionAnalysisData';
import { usePortfolioSnapshots } from '@/hooks/backtest/usePortfolioSnapshots';
import { StatisticsCards } from './position-analysis/StatisticsCards';
import { PerformersCards } from './position-analysis/PerformersCards';
import { PositionTable } from './position-analysis/PositionTable';
import { PieChart } from './position-analysis/PieChart';
import { BarChart } from './position-analysis/BarChart';
import { TreemapChart } from './position-analysis/TreemapChart';
import { WeightChart } from './position-analysis/WeightChart';
import { TradingPatternChart } from './position-analysis/TradingPatternChart';
import { HoldingPeriodChart } from './position-analysis/HoldingPeriodChart';
import { CapitalChart } from './position-analysis/CapitalChart';
import { StockDetailModal } from './position-analysis/StockDetailModal';

interface PositionAnalysisProps {
  positionAnalysis: PositionData[] | EnhancedPositionAnalysis;
  stockCodes: string[];
  taskId?: string;
}

export function PositionAnalysis({ positionAnalysis, stockCodes, taskId }: PositionAnalysisProps) {
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'total_return',
    direction: 'desc',
  });
  const [selectedMetric, setSelectedMetric] = useState<keyof PositionData>('total_return');
  const [selectedStock, setSelectedStock] = useState<PositionData | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('table');
  const [isDetailOpen, setIsDetailOpen] = useState(false);

  // 获取组合快照数据
  const { portfolioSnapshots, loadingSnapshots } = usePortfolioSnapshots(taskId);

  // 处理和计算所有数据
  const {
    normalizedData,
    stockPerformance,
    sortedPositions,
    pieChartData,
    barChartData,
    treemapData,
    weightChartData,
    capitalChartData,
    statistics,
  } = usePositionAnalysisData(positionAnalysis, sortConfig, portfolioSnapshots);

  // 处理排序
  const handleSort = (key: keyof PositionData) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  // 处理股票详情点击
  const handleStockClick = (stock: PositionData) => {
    setSelectedStock(stock);
    setIsDetailOpen(true);
  };

  // 空数据处理
  if (!stockPerformance || stockPerformance.length === 0) {
    return (
      <Card>
        <CardContent
          sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 256 }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <Target size={48} color="#999" style={{ margin: '0 auto 8px' }} />
            <Typography variant="body2" color="text.secondary">
              暂无持仓分析数据
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 统计概览 */}
      <StatisticsCards statistics={statistics} />

      {/* 最佳和最差表现者 */}
      <PerformersCards
        bestPerformer={statistics.bestPerformer}
        worstPerformer={statistics.worstPerformer}
      />

      {/* 图表展示 */}
      <Box>
        <Tabs
          value={selectedTab}
          onChange={(e, newValue) => setSelectedTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          allowScrollButtonsMobile
        >
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Target size={16} />
                <span>表格视图</span>
              </Box>
            }
            value="table"
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <PieChartIcon size={16} />
                <span>饼图</span>
              </Box>
            }
            value="pie"
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <BarChart3 size={16} />
                <span>柱状图</span>
              </Box>
            }
            value="bar"
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <BarChart3 size={16} />
                <span>树状图</span>
              </Box>
            }
            value="treemap"
          />
          {normalizedData?.position_weights && (
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <BarChart3 size={16} />
                  <span>权重分析</span>
                </Box>
              }
              value="weight"
            />
          )}
          {normalizedData?.trading_patterns && (
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <BarChart3 size={16} />
                  <span>交易模式</span>
                </Box>
              }
              value="trading"
            />
          )}
          {normalizedData?.holding_periods && (
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <BarChart3 size={16} />
                  <span>持仓期分析</span>
                </Box>
              }
              value="holding"
            />
          )}
          {taskId && (
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <DollarSign size={16} />
                  <span>资金分析</span>
                </Box>
              }
              value="capital"
            />
          )}
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {selectedTab === 'table' && (
            <PositionTable
              sortedPositions={sortedPositions}
              sortConfig={sortConfig}
              onSort={handleSort}
              onStockClick={handleStockClick}
            />
          )}

          {selectedTab === 'pie' && (
            <PieChart data={pieChartData} isActive={selectedTab === 'pie'} />
          )}

          {selectedTab === 'bar' && (
            <BarChart
              data={barChartData}
              selectedMetric={selectedMetric}
              onMetricChange={setSelectedMetric}
              isActive={selectedTab === 'bar'}
            />
          )}

          {selectedTab === 'treemap' && (
            <TreemapChart data={treemapData} isActive={selectedTab === 'treemap'} />
          )}

          {selectedTab === 'weight' && normalizedData?.position_weights && (
            <WeightChart
              data={weightChartData}
              concentrationMetrics={normalizedData.position_weights.concentration_metrics}
              isActive={selectedTab === 'weight'}
            />
          )}

          {selectedTab === 'trading' && normalizedData?.trading_patterns && (
            <TradingPatternChart
              tradingPatterns={normalizedData.trading_patterns}
              isActive={selectedTab === 'trading'}
            />
          )}

          {selectedTab === 'holding' && normalizedData?.holding_periods && (
            <HoldingPeriodChart
              holdingPeriods={normalizedData.holding_periods}
              isActive={selectedTab === 'holding'}
            />
          )}

          {selectedTab === 'capital' && taskId && (
            <CapitalChart
              data={capitalChartData}
              loading={loadingSnapshots}
              isActive={selectedTab === 'capital'}
            />
          )}
        </Box>
      </Box>

      {/* 股票详情模态框 */}
      <StockDetailModal
        open={isDetailOpen}
        onClose={() => setIsDetailOpen(false)}
        stock={selectedStock}
      />
    </Box>
  );
}
