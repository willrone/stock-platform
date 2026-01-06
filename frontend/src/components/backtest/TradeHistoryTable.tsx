/**
 * 交易记录表格组件
 * 支持分页、排序、筛选功能
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Pagination,
  Input,
  Select,
  SelectItem,
  Button,
  Chip,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Tooltip,
  Card,
  CardBody,
  Spinner,
} from '@heroui/react';
import { Search, Filter, Download, TrendingUp, TrendingDown, Calendar, DollarSign } from 'lucide-react';
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
  const [totalCount, setTotalCount] = useState(0);
  const itemsPerPage = 50;
  
  // 排序状态
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'timestamp',
    direction: 'desc'
  });
  
  // 筛选状态
  const [filters, setFilters] = useState<TradeFilters>({
    stockCode: '',
    action: 'ALL',
    startDate: '',
    endDate: '',
    minPnl: '',
    maxPnl: ''
  });
  
  // 搜索状态
  const [searchTerm, setSearchTerm] = useState('');
  
  // 模态框状态
  const { isOpen: isFilterOpen, onOpen: onFilterOpen, onClose: onFilterClose } = useDisclosure();
  const { isOpen: isDetailOpen, onOpen: onDetailOpen, onClose: onDetailClose } = useDisclosure();
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
        BacktestService.getTradeStatistics(taskId)
      ]);
      
      setTrades(tradesResponse.trades);
      setTotalCount(tradesResponse.pagination.count);
      setTotalPages(Math.ceil(tradesResponse.pagination.count / itemsPerPage));
      setStatistics(statsResponse);
      
    } catch (err: any) {
      console.error('获取交易记录失败:', err);
      setError(err.message || '获取交易记录失败');
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
    if (!searchTerm) return trades;
    
    const term = searchTerm.toLowerCase();
    return trades.filter(trade => 
      trade.stock_code.toLowerCase().includes(term) ||
      trade.trade_id.toLowerCase().includes(term) ||
      trade.action.toLowerCase().includes(term)
    );
  }, [trades, searchTerm]);

  // 处理排序
  const handleSort = (key: keyof TradeRecord) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
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
      maxPnl: ''
    });
    setSearchTerm('');
    setCurrentPage(1);
  };

  // 处理交易详情点击
  const handleTradeClick = (trade: TradeRecord) => {
    setSelectedTrade(trade);
    onDetailOpen();
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
        ...allTrades.trades.map(trade => [
          trade.trade_id,
          trade.stock_code,
          trade.action,
          trade.quantity,
          trade.price,
          trade.timestamp,
          trade.commission,
          trade.pnl
        ].join(','))
      ].join('\n');
      
      // 下载文件
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `交易记录_${taskId}_${new Date().toISOString().split('T')[0]}.csv`;
      link.click();
      
    } catch (err: any) {
      console.error('导出交易记录失败:', err);
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

  if (loading && trades.length === 0) {
    return (
      <Card>
        <CardBody className="flex items-center justify-center h-64">
          <Spinner size="lg" />
          <p className="mt-4 text-gray-600">加载交易记录中...</p>
        </CardBody>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardBody className="flex items-center justify-center h-64">
          <div className="text-center">
            <TrendingDown className="w-12 h-12 text-red-500 mx-auto mb-2" />
            <p className="text-red-600">{error}</p>
            <Button 
              color="primary" 
              variant="light" 
              onPress={fetchTrades}
              className="mt-2"
            >
              重试
            </Button>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* 统计信息卡片 */}
      {statistics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-500" />
                <div>
                  <p className="text-sm text-gray-500">总交易次数</p>
                  <p className="text-xl font-bold">{statistics.total_trades}</p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-500" />
                <div>
                  <p className="text-sm text-gray-500">胜率</p>
                  <p className="text-xl font-bold text-green-600">
                    {formatPercent(statistics.win_rate)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-purple-500" />
                <div>
                  <p className="text-sm text-gray-500">盈亏比</p>
                  <p className="text-xl font-bold">
                    {(statistics.profit_factor ?? 0).toFixed(2)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody className="p-4">
              <div className="flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-orange-500" />
                <div>
                  <p className="text-sm text-gray-500">总盈亏</p>
                  <p className={`text-xl font-bold ${statistics.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatCurrency(statistics.total_pnl)}
                  </p>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      )}

      {/* 搜索和筛选工具栏 */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="flex gap-2 w-full sm:w-auto">
          <Input
            placeholder="搜索股票代码或交易ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            startContent={<Search className="w-4 h-4" />}
            className="w-full sm:w-64"
          />
          <Button
            variant="bordered"
            onPress={onFilterOpen}
            startContent={<Filter className="w-4 h-4" />}
          >
            筛选
          </Button>
        </div>
        
        <div className="flex gap-2">
          <Button
            variant="bordered"
            onPress={handleExport}
            startContent={<Download className="w-4 h-4" />}
          >
            导出
          </Button>
          <Button
            variant="light"
            onPress={resetFilters}
          >
            重置
          </Button>
        </div>
      </div>

      {/* 交易记录表格 */}
      <Card>
        <CardBody className="p-0">
          <Table
            aria-label="交易记录表格"
            isHeaderSticky
            classNames={{
              wrapper: "max-h-[600px]",
            }}
          >
            <TableHeader>
              <TableColumn 
                key="timestamp" 
                allowsSorting
                className="cursor-pointer"
                onClick={() => handleSort('timestamp')}
              >
                <div className="flex items-center gap-1">
                  时间
                  {sortConfig.key === 'timestamp' && (
                    <span className="text-xs">
                      {sortConfig.direction === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </div>
              </TableColumn>
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
              <TableColumn key="action">操作</TableColumn>
              <TableColumn 
                key="quantity"
                allowsSorting
                className="cursor-pointer"
                onClick={() => handleSort('quantity')}
              >
                <div className="flex items-center gap-1">
                  数量
                  {sortConfig.key === 'quantity' && (
                    <span className="text-xs">
                      {sortConfig.direction === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </div>
              </TableColumn>
              <TableColumn 
                key="price"
                allowsSorting
                className="cursor-pointer"
                onClick={() => handleSort('price')}
              >
                <div className="flex items-center gap-1">
                  价格
                  {sortConfig.key === 'price' && (
                    <span className="text-xs">
                      {sortConfig.direction === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </div>
              </TableColumn>
              <TableColumn key="commission">手续费</TableColumn>
              <TableColumn 
                key="pnl"
                allowsSorting
                className="cursor-pointer"
                onClick={() => handleSort('pnl')}
              >
                <div className="flex items-center gap-1">
                  盈亏
                  {sortConfig.key === 'pnl' && (
                    <span className="text-xs">
                      {sortConfig.direction === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </div>
              </TableColumn>
              <TableColumn key="actions">操作</TableColumn>
            </TableHeader>
            <TableBody
              items={filteredTrades}
              isLoading={loading}
              loadingContent={<Spinner />}
              emptyContent="暂无交易记录"
            >
              {(trade) => (
                <TableRow key={trade.id}>
                  <TableCell>
                    <Tooltip content={formatDateTime(trade.timestamp)}>
                      <span className="text-sm">
                        {new Date(trade.timestamp).toLocaleDateString()}
                      </span>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono font-medium">
                      {trade.stock_code}
                    </span>
                  </TableCell>
                  <TableCell>
                    <Chip
                      color={trade.action === 'BUY' ? 'success' : 'danger'}
                      variant="flat"
                      size="sm"
                    >
                      {trade.action === 'BUY' ? '买入' : '卖出'}
                    </Chip>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono">
                      {trade.quantity.toLocaleString()}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono">
                      ¥{trade.price.toFixed(2)}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono text-gray-600">
                      ¥{trade.commission.toFixed(2)}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span 
                      className={`font-mono font-medium ${
                        trade.pnl > 0 ? 'text-green-600' : 
                        trade.pnl < 0 ? 'text-red-600' : 'text-gray-600'
                      }`}
                    >
                      {formatCurrency(trade.pnl)}
                    </span>
                  </TableCell>
                  <TableCell>
                    <Button
                      size="sm"
                      variant="light"
                      onPress={() => handleTradeClick(trade)}
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

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="flex justify-center">
          <Pagination
            total={totalPages}
            page={currentPage}
            onChange={setCurrentPage}
            showControls
            showShadow
            color="primary"
          />
        </div>
      )}

      {/* 筛选模态框 */}
      <Modal isOpen={isFilterOpen} onClose={onFilterClose} size="2xl">
        <ModalContent>
          <ModalHeader>筛选交易记录</ModalHeader>
          <ModalBody>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input
                label="股票代码"
                placeholder="输入股票代码"
                value={filters.stockCode}
                onChange={(e) => handleFilterChange('stockCode', e.target.value)}
              />
              
              <Select
                label="操作类型"
                selectedKeys={[filters.action]}
                onChange={(e) => handleFilterChange('action', e.target.value as any)}
              >
                <SelectItem key="ALL">全部</SelectItem>
                <SelectItem key="BUY">买入</SelectItem>
                <SelectItem key="SELL">卖出</SelectItem>
              </Select>
              
              <Input
                type="date"
                label="开始日期"
                value={filters.startDate}
                onChange={(e) => handleFilterChange('startDate', e.target.value)}
              />
              
              <Input
                type="date"
                label="结束日期"
                value={filters.endDate}
                onChange={(e) => handleFilterChange('endDate', e.target.value)}
              />
              
              <Input
                type="number"
                label="最小盈亏"
                placeholder="输入最小盈亏金额"
                value={filters.minPnl}
                onChange={(e) => handleFilterChange('minPnl', e.target.value)}
              />
              
              <Input
                type="number"
                label="最大盈亏"
                placeholder="输入最大盈亏金额"
                value={filters.maxPnl}
                onChange={(e) => handleFilterChange('maxPnl', e.target.value)}
              />
            </div>
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={resetFilters}>
              重置
            </Button>
            <Button color="primary" onPress={onFilterClose}>
              应用筛选
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* 交易详情模态框 */}
      <Modal isOpen={isDetailOpen} onClose={onDetailClose} size="lg">
        <ModalContent>
          <ModalHeader>交易详情</ModalHeader>
          <ModalBody>
            {selectedTrade && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">交易ID</p>
                    <p className="font-mono">{selectedTrade.trade_id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">股票代码</p>
                    <p className="font-mono font-medium">{selectedTrade.stock_code}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">操作类型</p>
                    <Chip
                      color={selectedTrade.action === 'BUY' ? 'success' : 'danger'}
                      variant="flat"
                      size="sm"
                    >
                      {selectedTrade.action === 'BUY' ? '买入' : '卖出'}
                    </Chip>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">交易时间</p>
                    <p>{formatDateTime(selectedTrade.timestamp)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">数量</p>
                    <p className="font-mono">{selectedTrade.quantity.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">价格</p>
                    <p className="font-mono">¥{selectedTrade.price.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">手续费</p>
                    <p className="font-mono">¥{selectedTrade.commission.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">盈亏</p>
                    <p 
                      className={`font-mono font-medium ${
                        selectedTrade.pnl > 0 ? 'text-green-600' : 
                        selectedTrade.pnl < 0 ? 'text-red-600' : 'text-gray-600'
                      }`}
                    >
                      {formatCurrency(selectedTrade.pnl)}
                    </p>
                  </div>
                </div>
                
                {selectedTrade.holding_days && (
                  <div>
                    <p className="text-sm text-gray-500">持仓天数</p>
                    <p>{selectedTrade.holding_days} 天</p>
                  </div>
                )}
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
