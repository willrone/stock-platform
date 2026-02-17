/**
 * 预测结果标签页组件
 */

import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Box,
  Tabs,
  Tab,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
  Chip,
  LinearProgress,
} from '@mui/material';
import { LineChart, BarChart3, Activity, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { PredictionResult } from '@/services/taskService';
import dynamic from 'next/dynamic';

// 动态导入图表组件
const TradingViewChart = dynamic(() => import('@/components/charts/TradingViewChart'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载图表中...</div>,
});

const PredictionChart = dynamic(() => import('@/components/charts/PredictionChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载预测图表中...</div>,
});

const BacktestChart = dynamic(() => import('@/components/charts/BacktestChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载回测图表中...</div>,
});

interface PredictionTabsProps {
  taskId: string;
  predictions: PredictionResult[];
  selectedStock: string;
  onStockChange: (stock: string) => void;
  backtestData?: any;
  showBacktestTab?: boolean;
}

export function PredictionTabs({
  taskId,
  predictions,
  selectedStock,
  onStockChange,
  backtestData,
  showBacktestTab = false,
}: PredictionTabsProps) {
  const [selectedTab, setSelectedTab] = useState<string>('chart');

  // 获取预测方向图标
  const getPredictionIcon = (direction: number) => {
    if (direction > 0) {
      return <TrendingUp className="w-4 h-4 text-success" />;
    }
    if (direction < 0) {
      return <TrendingDown className="w-4 h-4 text-danger" />;
    }
    return <Minus className="w-4 h-4 text-default-500" />;
  };

  // 获取预测方向文本
  const getPredictionText = (direction: number) => {
    if (direction > 0) {
      return '上涨';
    }
    if (direction < 0) {
      return '下跌';
    }
    return '持平';
  };

  return (
    <Card>
      <CardHeader
        title={
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              justifyContent: 'space-between',
              alignItems: { xs: 'flex-start', sm: 'center' },
              gap: 1,
              width: '100%',
            }}
          >
            <Typography
              variant="h6"
              component="h3"
              sx={{ fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' } }}
            >
              预测结果
            </Typography>
            <FormControl size="small" sx={{ minWidth: { xs: '100%', sm: 192 } }}>
              <InputLabel>选择股票</InputLabel>
              <Select
                value={selectedStock || ''}
                label="选择股票"
                onChange={e => onStockChange(e.target.value)}
              >
                {predictions.map(prediction => (
                  <MenuItem key={prediction.stock_code} value={prediction.stock_code}>
                    {prediction.stock_code}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        }
      />
      <CardContent>
        <Box>
          <Tabs
            value={selectedTab}
            onChange={(e, newValue) => setSelectedTab(newValue)}
            aria-label="预测结果展示"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <LineChart size={16} />
                  <span>价格走势</span>
                </Box>
              }
              value="chart"
            />
            <Tab
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <BarChart3 size={16} />
                  <span>预测分析</span>
                </Box>
              }
              value="prediction"
            />
            {showBacktestTab && (
              <Tab
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Activity size={16} />
                    <span>回测结果</span>
                  </Box>
                }
                value="backtest"
              />
            )}
            <Tab label="数据表格" value="table" />
          </Tabs>

          <Box sx={{ mt: 2 }}>
            {selectedTab === 'chart' && selectedStock && (
              <TradingViewChart
                stockCode={selectedStock}
                prediction={predictions.find(p => p.stock_code === selectedStock)}
              />
            )}

            {selectedTab === 'prediction' && selectedStock && (
              <PredictionChart
                taskId={taskId}
                stockCode={selectedStock}
                prediction={predictions.find(p => p.stock_code === selectedStock)}
              />
            )}

            {selectedTab === 'backtest' && showBacktestTab && (
              <BacktestChart stockCode={selectedStock} backtestData={backtestData} />
            )}

            {selectedTab === 'table' && (
              <TableContainer component={Paper} sx={{ overflowX: 'auto' }}>
                <Table aria-label="预测结果表格" sx={{ minWidth: 600 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        股票代码
                      </TableCell>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        预测方向
                      </TableCell>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        预测收益率
                      </TableCell>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        置信度
                      </TableCell>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        置信区间
                      </TableCell>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        VaR
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map(prediction => (
                      <TableRow key={prediction.stock_code}>
                        <TableCell>
                          <Chip label={prediction.stock_code} size="small" />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getPredictionIcon(prediction.predicted_direction)}
                            <Typography variant="body2">
                              {getPredictionText(prediction.predicted_direction)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography
                            variant="body2"
                            sx={{
                              color:
                                prediction.predicted_return > 0
                                  ? 'success.main'
                                  : prediction.predicted_return < 0
                                    ? 'error.main'
                                    : 'text.secondary',
                            }}
                          >
                            {(prediction.predicted_return * 100).toFixed(2)}%
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ width: 80 }}>
                            <LinearProgress
                              variant="determinate"
                              value={prediction.confidence_score * 100}
                              sx={{ height: 8, borderRadius: 4 }}
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption" color="text.secondary">
                            [{(prediction.confidence_interval.lower * 100).toFixed(2)}%,{' '}
                            {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="error.main">
                            {(prediction.risk_assessment.value_at_risk * 100).toFixed(2)}%
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
