/**
 * 信号记录表格组件
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
import { Search, Filter, Download, AlertCircle, CheckCircle, XCircle, Calendar, Zap } from 'lucide-react';
import { BacktestService, SignalRecord, SignalStatistics } from '../../services/backtestService';

interface SignalHistoryTableProps {
  taskId: string;
  onSignalClick?: (signal: SignalRecord) => void;
}

interface SignalFilters {
  stockCode: string;
  signalType: 'ALL' | 'BUY' | 'SELL';
  executed: 'ALL' | 'EXECUTED' | 'UNEXECUTED';
  startDate: string;
  endDate: string;
  minStrength: string;
  maxStrength: string;
}

interface SortConfig {
  key: keyof SignalRecord;
  direction: 'asc' | 'desc';
}

export function SignalHistoryTable({ taskId, onSignalClick }: SignalHistoryTableProps) {
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [statistics, setStatistics] = useState<SignalStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // 分页状态
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const itemsPerPage = 50;
  
  // 排序状态
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'timestamp',
    direction: 'desc'
  });
  
  // 筛选状态
  const [filters, setFilters] = useState<SignalFilters>({
    stockCode: '',
    signalType: 'ALL',
    executed: 'ALL',
    startDate: '',
    endDate: '',
    minStrength: '',
    maxStrength: ''
  });
  
  // 搜索状态
  const [searchTerm, setSearchTerm] = useState('');
  
  // 模态框状态
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<SignalRecord | null>(null);

  // 获取信号数据
  const fetchSignals = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const options = {
        offset: (currentPage - 1) * itemsPerPage,
        limit: itemsPerPage,
        orderBy: sortConfig.key,
        orderDesc: sortConfig.direction === 'desc',
        stockCode: filters.stockCode || undefined,
        signalType: filters.signalType !== 'ALL' ? filters.signalType : undefined,
        executed: filters.executed === 'EXECUTED' ? true : filters.executed === 'UNEXECUTED' ? false : undefined,
        startDate: filters.startDate || undefined,
        endDate: filters.endDate || undefined,
      };
      
      const [signalsResponse, statsResponse] = await Promise.all([
        BacktestService.getSignalRecords(taskId, options),
        BacktestService.getSignalStatistics(taskId)
      ]);
      
      setSignals(signalsResponse.signals);
      setTotalCount(signalsResponse.pagination.count);
      setTotalPages(Math.ceil(signalsResponse.pagination.count / itemsPerPage));
      setStatistics(statsResponse);
      
    } catch (err: any) {
      console.error('获取信号记录失败:', err);
      setError(err.message || '获取信号记录失败');
    } finally {
      setLoading(false);
    }
  };

  // 初始加载和依赖更新
  useEffect(() => {
    if (taskId) {
      fetchSignals();
    }
  }, [taskId, currentPage, sortConfig, filters]);

  // 筛选后的信号记录（用于搜索）
  const filteredSignals = useMemo(() => {
    if (!searchTerm) return signals;
    
    const term = searchTerm.toLowerCase();
    return signals.filter(signal => 
      signal.stock_code.toLowerCase().includes(term) ||
      signal.signal_id.toLowerCase().includes(term) ||
      signal.signal_type.toLowerCase().includes(term) ||
      (signal.reason && signal.reason.toLowerCase().includes(term))
    );
  }, [signals, searchTerm]);

  // 处理排序
  const handleSort = (key: keyof SignalRecord) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
    setCurrentPage(1); // 重置到第一页
  };

  // 处理筛选
  const handleFilterChange = (key: keyof SignalFilters, value: string | boolean) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setCurrentPage(1); // 重置到第一页
  };

  // 重置筛选
  const resetFilters = () => {
    setFilters({
      stockCode: '',
      signalType: 'ALL',
      executed: 'ALL',
      startDate: '',
      endDate: '',
      minStrength: '',
      maxStrength: ''
    });
    setSearchTerm('');
    setCurrentPage(1);
  };

  // 处理信号详情点击
  const handleSignalClick = (signal: SignalRecord) => {
    setSelectedSignal(signal);
    setIsDetailOpen(true);
    onSignalClick?.(signal);
  };

  // 导出信号记录
  const handleExport = async () => {
    try {
      // 获取所有信号记录
      const allSignals = await BacktestService.getSignalRecords(taskId, {
        limit: 10000,
        stockCode: filters.stockCode || undefined,
        signalType: filters.signalType !== 'ALL' ? filters.signalType : undefined,
        executed: filters.executed === 'EXECUTED' ? true : filters.executed === 'UNEXECUTED' ? false : undefined,
        startDate: filters.startDate || undefined,
        endDate: filters.endDate || undefined,
      });
      
      // 转换为CSV格式
      const csvContent = [
        ['信号ID', '股票代码', '信号类型', '价格', '强度', '时间', '是否执行', '原因'].join(','),
        ...allSignals.signals.map(signal => [
          signal.signal_id,
          signal.stock_code,
          signal.signal_type,
          signal.price,
          signal.strength,
          signal.timestamp,
          signal.executed ? '是' : '否',
          signal.reason || ''
        ].join(','))
      ].join('\n');
      
      // 下载文件
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `信号记录_${taskId}_${new Date().toISOString().split('T')[0]}.csv`;
      link.click();
      
    } catch (err: any) {
      console.error('导出信号记录失败:', err);
    }
  };

  // 格式化数值
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

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN');
  };

  if (loading && signals.length === 0) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 256 }}>
            <CircularProgress size={48} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              加载信号记录中...
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
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 256 }}>
            <XCircle size={48} color="#d32f2f" style={{ marginBottom: 8 }} />
            <Typography variant="body2" color="error" sx={{ mb: 2 }}>
              {error}
            </Typography>
            <Button variant="outlined" color="primary" onClick={fetchSignals}>
              重试
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 统计信息卡片 */}
      {statistics && (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2 }}>
          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Zap size={20} color="#1976d2" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    总信号数
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {statistics.total_signals}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircle size={20} color="#2e7d32" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    执行率
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatPercent(statistics.execution_rate)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AlertCircle size={20} color="#ed6c02" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    买入信号
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {statistics.buy_signals}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AlertCircle size={20} color="#d32f2f" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    卖出信号
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    {statistics.sell_signals}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* 搜索和筛选工具栏 */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2, alignItems: { xs: 'stretch', sm: 'center' }, justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', gap: 1, width: { xs: '100%', sm: 'auto' } }}>
          <TextField
            placeholder="搜索股票代码或信号ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
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
          <Button
            variant="outlined"
            onClick={handleExport}
            startIcon={<Download size={16} />}
          >
            导出
          </Button>
          <Button
            variant="outlined"
            onClick={resetFilters}
          >
            重置
          </Button>
        </Box>
      </Box>

      {/* 信号记录表格 */}
      <Card>
        <CardContent sx={{ p: 0 }}>
          <Box sx={{ overflowX: 'auto', maxHeight: 600 }}>
            <Table stickyHeader>
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
                  <TableCell>信号类型</TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'price'}
                      direction={sortConfig.key === 'price' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('price')}
                    >
                      价格
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.key === 'strength'}
                      direction={sortConfig.key === 'strength' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('strength')}
                    >
                      强度
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>是否执行</TableCell>
                  <TableCell>原因</TableCell>
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
                ) : filteredSignals.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body2" color="text.secondary">
                        暂无信号记录
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredSignals.map((signal) => (
                    <TableRow key={signal.id} hover>
                      <TableCell>
                        <Tooltip title={formatDateTime(signal.timestamp)}>
                          <Typography variant="body2">
                            {new Date(signal.timestamp).toLocaleDateString()}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                          {signal.stock_code}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={signal.signal_type === 'BUY' ? '买入' : '卖出'}
                          color={signal.signal_type === 'BUY' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          ¥{signal.price.toFixed(2)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {(signal.strength * 100).toFixed(1)}%
                          </Typography>
                          <Box
                            sx={{
                              width: 40,
                              height: 4,
                              bgcolor: 'grey.300',
                              borderRadius: 1,
                              overflow: 'hidden'
                            }}
                          >
                            <Box
                              sx={{
                                width: `${signal.strength * 100}%`,
                                height: '100%',
                                bgcolor: signal.signal_type === 'BUY' ? 'success.main' : 'error.main'
                              }}
                            />
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={signal.executed ? '已执行' : '未执行'}
                          color={signal.executed ? 'success' : 'default'}
                          size="small"
                          icon={signal.executed ? <CheckCircle size={16} /> : <XCircle size={16} />}
                        />
                      </TableCell>
                      <TableCell>
                        <Tooltip title={signal.reason || '无原因'}>
                          <Typography
                            variant="body2"
                            sx={{
                              maxWidth: 200,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            {signal.reason || '-'}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleSignalClick(signal)}
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
        <DialogTitle>筛选信号记录</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2, mt: 1 }}>
            <TextField
              label="股票代码"
              placeholder="输入股票代码"
              value={filters.stockCode}
              onChange={(e) => handleFilterChange('stockCode', e.target.value)}
              fullWidth
            />
            
            <FormControl fullWidth>
              <InputLabel>信号类型</InputLabel>
              <Select
                value={filters.signalType}
                label="信号类型"
                onChange={(e) => handleFilterChange('signalType', e.target.value as any)}
              >
                <MenuItem value="ALL">全部</MenuItem>
                <MenuItem value="BUY">买入</MenuItem>
                <MenuItem value="SELL">卖出</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth>
              <InputLabel>执行状态</InputLabel>
              <Select
                value={filters.executed}
                label="执行状态"
                onChange={(e) => handleFilterChange('executed', e.target.value as any)}
              >
                <MenuItem value="ALL">全部</MenuItem>
                <MenuItem value="EXECUTED">已执行</MenuItem>
                <MenuItem value="UNEXECUTED">未执行</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              type="date"
              label="开始日期"
              value={filters.startDate}
              onChange={(e) => handleFilterChange('startDate', e.target.value)}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
            
            <TextField
              type="date"
              label="结束日期"
              value={filters.endDate}
              onChange={(e) => handleFilterChange('endDate', e.target.value)}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
            
            <TextField
              type="number"
              label="最小强度"
              placeholder="0.0 - 1.0"
              value={filters.minStrength}
              onChange={(e) => handleFilterChange('minStrength', e.target.value)}
              fullWidth
              inputProps={{ min: 0, max: 1, step: 0.1 }}
            />
            
            <TextField
              type="number"
              label="最大强度"
              placeholder="0.0 - 1.0"
              value={filters.maxStrength}
              onChange={(e) => handleFilterChange('maxStrength', e.target.value)}
              fullWidth
              inputProps={{ min: 0, max: 1, step: 0.1 }}
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

      {/* 信号详情模态框 */}
      <Dialog open={isDetailOpen} onClose={() => setIsDetailOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>信号详情</DialogTitle>
        <DialogContent>
          {selectedSignal && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    信号ID
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedSignal.signal_id}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    股票代码
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {selectedSignal.stock_code}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    信号类型
                  </Typography>
                  <Chip
                    label={selectedSignal.signal_type === 'BUY' ? '买入' : '卖出'}
                    color={selectedSignal.signal_type === 'BUY' ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    信号时间
                  </Typography>
                  <Typography variant="body2">
                    {formatDateTime(selectedSignal.timestamp)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    价格
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    ¥{selectedSignal.price.toFixed(2)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    强度
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {(selectedSignal.strength * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    是否执行
                  </Typography>
                  <Chip
                    label={selectedSignal.executed ? '已执行' : '未执行'}
                    color={selectedSignal.executed ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
              </Box>
              
              {selectedSignal.reason && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    信号原因
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5 }}>
                    {selectedSignal.reason}
                  </Typography>
                </Box>
              )}
              
              {selectedSignal.metadata && Object.keys(selectedSignal.metadata).length > 0 && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    元数据
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, fontFamily: 'monospace', fontSize: '0.75rem' }}>
                    {JSON.stringify(selectedSignal.metadata, null, 2)}
                  </Typography>
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
