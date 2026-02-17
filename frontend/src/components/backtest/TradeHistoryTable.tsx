/**
 * 交易记录表格组件
 * 支持分页、排序、筛选功能
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Pagination,
  TextField,
  Select,
  MenuItem,
  Button,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Tooltip,
  Card,
  CardContent,
  CircularProgress,
  Box,
  Typography,
  IconButton,
  InputAdornment,
  FormControl,
  InputLabel,
  TableSortLabel,
} from '@mui/material';
import {
  Search,
  Filter,
  Download,
  TrendingUp,
  TrendingDown,
  Calendar,
  DollarSign,
} from 'lucide-react';
import { BacktestService, TradeRecord, TradeStatistics } from '../../services/backtestService';

interface TradeHistoryTableProps {
  taskId: string;
  onTradeClick?: (trade: TradeRecord) => void;
}

interface TradeFilters {
  stockCode: string;
  action: 'ALL' | 'BUY' | 'SELL';
  startDate: string;
  endDate: string;
  minPnl: string;
  maxPnl: string;
}

interface SortConfig {
  key: keyof TradeRecord;
  direction: 'asc' | 'desc';
}

export function TradeHistoryTable({ taskId, onTradeClick }: TradeHistoryTableProps) {
  const [trades, setTrades] = useState<TradeRecord[]>([]);
  const [statistics, setStatistics] = useState<TradeStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 分页状态
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [, setTotalCount] = useState(0);
  const itemsPerPage = 50;

  // 排序状态
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'timestamp',
    direction: 'desc',
  });

  // 筛选状态
  const [filters, setFilters] = useState<TradeFilters>({
    stockCode: '',
    action: 'ALL',
    startDate: '',
    endDate: '',
    minPnl: '',
    maxPnl: '',
  });

  // 搜索状态
  const [searchTerm, setSearchTerm] = useState('');

  // 模态框状态
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [selectedTrade, setSelectedTrade] = useState<TradeRecord | null>(null);

  // 获取交易数据
  const fetchTrades = async () => {
    try {
      setLoading(true);
      setError(null);

      const options = {
        offset: (currentPage - 1) * itemsPerPage,
        limit: itemsPerPage,
        orderBy: sortConfig.key,
        orderDesc: sortConfig.direction === 'desc',
        stockCode: filters.stockCode || undefined,
        action: filters.action !== 'ALL' ? filters.action : undefined,
        startDate: filters.startDate || undefined,
        endDate: filters.endDate || undefined,
      };

      const [tradesResponse, statsResponse] = await Promise.all([
        BacktestService.getTradeRecords(taskId, options),
        BacktestService.getTradeStatistics(taskId),
      ]);

      setTrades(tradesResponse.trades);
      setTotalCount(tradesResponse.pagination.count);
      setTotalPages(Math.ceil(tradesResponse.pagination.count / itemsPerPage));
      setStatistics(statsResponse);
    } catch (err: unknown) {
      console.error('获取交易记录失败:', err);
      setError(err instanceof Error ? err.message : '获取交易记录失败');
    } finally {
      setLoading(false);
    }
  };

  // 初始加载和依赖更新
  useEffect(() => {
    if (taskId) {
      fetchTrades();
    }
  }, [taskId, currentPage, sortConfig, filters]);

  // 筛选后的交易记录（用于搜索）
  const filteredTrades = useMemo(() => {
    if (!searchTerm) {
      return trades;
    }

    const term = searchTerm.toLowerCase();
    return trades.filter(
      trade =>
        trade.stock_code.toLowerCase().includes(term) ||
        trade.trade_id.toLowerCase().includes(term) ||
        trade.action.toLowerCase().includes(term)
    );
  }, [trades, searchTerm]);

  // 处理排序
  const handleSort = (key: keyof TradeRecord) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
    setCurrentPage(1); // 重置到第一页
  };

  // 处理筛选
  const handleFilterChange = (key: keyof TradeFilters, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setCurrentPage(1); // 重置到第一页
  };

  // 重置筛选
  const resetFilters = () => {
    setFilters({
      stockCode: '',
      action: 'ALL',
      startDate: '',
      endDate: '',
      minPnl: '',
      maxPnl: '',
    });
    setSearchTerm('');
    setCurrentPage(1);
  };

  // 处理交易详情点击
  const handleTradeClick = (trade: TradeRecord) => {
    setSelectedTrade(trade);
    setIsDetailOpen(true);
    onTradeClick?.(trade);
  };

  // 导出交易记录
  const handleExport = async () => {
    try {
      // 获取所有交易记录
      const allTrades = await BacktestService.getTradeRecords(taskId, {
        limit: 10000,
        stockCode: filters.stockCode || undefined,
        action: filters.action !== 'ALL' ? filters.action : undefined,
        startDate: filters.startDate || undefined,
        endDate: filters.endDate || undefined,
      });

      // 转换为CSV格式
      const csvContent = [
        ['交易ID', '股票代码', '操作', '数量', '价格', '时间', '手续费', '盈亏'].join(','),
        ...allTrades.trades.map(trade =>
          [
            trade.trade_id,
            trade.stock_code,
            trade.action,
            trade.quantity,
            trade.price,
            trade.timestamp,
            trade.commission,
            trade.pnl,
          ].join(',')
        ),
      ].join('\n');

      // 下载文件
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `交易记录_${taskId}_${new Date().toISOString().split('T')[0]}.csv`;
      link.click();
    } catch (err: unknown) {
      console.error('导出交易记录失败:', err);
    }
  };

  // 格式化数值
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN');
  };

  if (loading && trades.length === 0) {
    return (
      <Card>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 256,
            }}
          >
            <CircularProgress size={48} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              加载交易记录中...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 256,
            }}
          >
            <TrendingDown size={48} color="#d32f2f" style={{ marginBottom: 8 }} />
            <Typography variant="body2" color="error" sx={{ mb: 2 }}>
              {error}
            </Typography>
            <Button variant="outlined" color="primary" onClick={fetchTrades}>
              重试
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, p: { xs: 1, md: 0 } }}>
      {/* 统计信息卡片 */}
      {statistics && (
        <Box
          sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2 }}
        >
          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp size={20} color="#1976d2" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    总交易次数
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{ fontWeight: 600, fontSize: { xs: '1.1rem', md: '1.5rem' } }}
                  >
                    {statistics.total_trades}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp size={20} color="#2e7d32" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    胜率
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{
                      fontWeight: 600,
                      color: 'success.main',
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                    }}
                  >
                    {formatPercent(statistics.win_rate)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DollarSign size={20} color="#9c27b0" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    盈亏比
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{ fontWeight: 600, fontSize: { xs: '1.1rem', md: '1.5rem' } }}
                  >
                    {(statistics.profit_factor ?? 0).toFixed(2)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DollarSign size={20} color="#ed6c02" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    总盈亏
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{
                      fontWeight: 600,
                      color: statistics.total_pnl >= 0 ? 'success.main' : 'error.main',
                      fontSize: { xs: '1.1rem', md: '1.5rem' },
                    }}
                  >
                    {formatCurrency(statistics.total_pnl)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* 搜索和筛选工具栏 */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', sm: 'row' },
          gap: 2,
          alignItems: { xs: 'stretch', sm: 'center' },
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ display: 'flex', gap: 1, width: { xs: '100%', sm: 'auto' } }}>
          <TextField
            placeholder="搜索股票代码或交易ID..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            size="small"
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search size={16} />
                </InputAdornment>
              ),
            }}
            sx={{ width: { xs: '100%', sm: 256 } }}
          />
          <Button
            variant="outlined"
            onClick={() => setIsFilterOpen(true)}
            startIcon={<Filter size={16} />}
          >
            筛选
          </Button>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button variant="outlined" onClick={handleExport} startIcon={<Download size={16} />}>
            导出
          </Button>
          <Button variant="outlined" onClick={resetFilters}>
            重置
          </Button>
        </Box>
      </Box>

      {/* 交易记录表格 */}
      <Card>
        <CardContent sx={{ p: { xs: 1, md: 2 } }}>
          <Box sx={{ overflowX: 'auto', maxHeight: 600 }}>
            <Table
              stickyHeader
              sx={{
                '& .MuiTableCell-root': {
                  padding: { xs: '4px 8px', md: '16px' },
                  fontSize: { xs: '0.75rem', md: '0.875rem' },
                },
              }}
            >
              <TableHead>
                <TableRow>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'timestamp'}
                      direction={sortConfig.key === 'timestamp' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('timestamp')}
                    >
                      时间
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'stock_code'}
                      direction={sortConfig.key === 'stock_code' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('stock_code')}
                    >
                      股票代码
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>操作</TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'quantity'}
                      direction={sortConfig.key === 'quantity' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('quantity')}
                    >
                      数量
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'price'}
                      direction={sortConfig.key === 'price' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('price')}
                    >
                      价格
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>手续费</TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'pnl'}
                      direction={sortConfig.key === 'pnl' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('pnl')}
                    >
                      盈亏
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <CircularProgress size={24} />
                    </TableCell>
                  </TableRow>
                ) : filteredTrades.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body2" color="text.secondary">
                        暂无交易记录
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredTrades.map(trade => (
                    <TableRow key={trade.id} hover>
                      <TableCell>
                        <Tooltip title={formatDateTime(trade.timestamp)}>
                          <Typography variant="body2">
                            {new Date(trade.timestamp).toLocaleDateString()}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{ fontFamily: 'monospace', fontWeight: 500 }}
                        >
                          {trade.stock_code}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={trade.action === 'BUY' ? '买入' : '卖出'}
                          color={trade.action === 'BUY' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {trade.quantity.toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          ¥{trade.price.toFixed(2)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{ fontFamily: 'monospace', color: 'text.secondary' }}
                        >
                          ¥{trade.commission.toFixed(2)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{
                            fontFamily: 'monospace',
                            fontWeight: 500,
                            color:
                              trade.pnl > 0
                                ? 'success.main'
                                : trade.pnl < 0
                                  ? 'error.main'
                                  : 'text.secondary',
                          }}
                        >
                          {formatCurrency(trade.pnl)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleTradeClick(trade)}
                        >
                          详情
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </Box>
        </CardContent>
      </Card>

      {/* 分页 */}
      {totalPages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Pagination
            count={totalPages}
            page={currentPage}
            onChange={(e, page) => setCurrentPage(page)}
            color="primary"
          />
        </Box>
      )}

      {/* 筛选模态框 */}
      <Dialog open={isFilterOpen} onClose={() => setIsFilterOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>筛选交易记录</DialogTitle>
        <DialogContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
              gap: 2,
              mt: 1,
            }}
          >
            <TextField
              label="股票代码"
              placeholder="输入股票代码"
              value={filters.stockCode}
              onChange={e => handleFilterChange('stockCode', e.target.value)}
              fullWidth
            />

            <FormControl fullWidth>
              <InputLabel>操作类型</InputLabel>
              <Select
                value={filters.action}
                label="操作类型"
                onChange={e => handleFilterChange('action', e.target.value as string)}
              >
                <MenuItem value="ALL">全部</MenuItem>
                <MenuItem value="BUY">买入</MenuItem>
                <MenuItem value="SELL">卖出</MenuItem>
              </Select>
            </FormControl>

            <TextField
              type="date"
              label="开始日期"
              value={filters.startDate}
              onChange={e => handleFilterChange('startDate', e.target.value)}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />

            <TextField
              type="date"
              label="结束日期"
              value={filters.endDate}
              onChange={e => handleFilterChange('endDate', e.target.value)}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />

            <TextField
              type="number"
              label="最小盈亏"
              placeholder="输入最小盈亏金额"
              value={filters.minPnl}
              onChange={e => handleFilterChange('minPnl', e.target.value)}
              fullWidth
            />

            <TextField
              type="number"
              label="最大盈亏"
              placeholder="输入最大盈亏金额"
              value={filters.maxPnl}
              onChange={e => handleFilterChange('maxPnl', e.target.value)}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={resetFilters}>重置</Button>
          <Button variant="contained" color="primary" onClick={() => setIsFilterOpen(false)}>
            应用筛选
          </Button>
        </DialogActions>
      </Dialog>

      {/* 交易详情模态框 */}
      <Dialog open={isDetailOpen} onClose={() => setIsDetailOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>交易详情</DialogTitle>
        <DialogContent>
          {selectedTrade && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    交易ID
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedTrade.trade_id}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    股票代码
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {selectedTrade.stock_code}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    操作类型
                  </Typography>
                  <Chip
                    label={selectedTrade.action === 'BUY' ? '买入' : '卖出'}
                    color={selectedTrade.action === 'BUY' ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    交易时间
                  </Typography>
                  <Typography variant="body2">{formatDateTime(selectedTrade.timestamp)}</Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    数量
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedTrade.quantity.toLocaleString()}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    价格
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    ¥{selectedTrade.price.toFixed(2)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    手续费
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    ¥{selectedTrade.commission.toFixed(2)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    盈亏
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      fontFamily: 'monospace',
                      fontWeight: 500,
                      color:
                        selectedTrade.pnl > 0
                          ? 'success.main'
                          : selectedTrade.pnl < 0
                            ? 'error.main'
                            : 'text.secondary',
                    }}
                  >
                    {formatCurrency(selectedTrade.pnl)}
                  </Typography>
                </Box>
              </Box>

              {selectedTrade.holding_days && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    持仓天数
                  </Typography>
                  <Typography variant="body2">{selectedTrade.holding_days} 天</Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button variant="contained" color="primary" onClick={() => setIsDetailOpen(false)}>
            关闭
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
