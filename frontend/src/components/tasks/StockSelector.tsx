/**
 * 股票选择器组件
 *
 * 提供股票搜索和选择功能，包括：
 * - 股票搜索
 * - 全量股票列表展示
 * - 分页浏览
 * - 已选股票管理
 */

'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  TextField,
  Pagination,
  Box,
  Typography,
  IconButton,
  InputAdornment,
  CircularProgress,
} from '@mui/material';
import { X, Search, Trash2, ChevronLeft, ChevronRight, Shuffle } from 'lucide-react';
import { DataService } from '../../services/dataService';

interface StockOption {
  code: string;
  name: string;
  market: string;
}

interface StockSelectorProps {
  value?: string[];
  onChange?: (stocks: string[]) => void;
  maxCount?: number;
  placeholder?: string;
}

const ITEMS_PER_PAGE = 10;
const SELECTED_STOCKS_PER_PAGE = 12; // 已选股票每页显示数量
const SELECTED_STOCKS_CARD_HEIGHT = 200; // 已选股票卡片固定高度

export const StockSelector: React.FC<StockSelectorProps> = ({
  value = [],
  onChange,
  maxCount = 50,
  placeholder = '搜索股票代码或名称',
}) => {
  const [searchValue, setSearchValue] = useState('');
  const [allStocks, setAllStocks] = useState<StockOption[]>([]);
  const [loadingAllStocks, setLoadingAllStocks] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedStocksPage, setSelectedStocksPage] = useState(1); // 已选股票分页
  const [randomCount, setRandomCount] = useState<string>('10');

  // 加载本地股票列表（只显示已存储的股票数据）
  useEffect(() => {
    const loadAllStocks = async () => {
      setLoadingAllStocks(true);
      try {
        console.log('[StockSelector] 开始加载本地股票列表...');
        const result = await DataService.getLocalStockList();
        console.log(`[StockSelector] 获取到 ${result.total_stocks} 只本地股票`);

        if (!result.stocks || result.stocks.length === 0) {
          console.warn('[StockSelector] 本地股票列表为空');
          setAllStocks([]);
          return;
        }

        // 转换为标准格式
        const formattedStocks = result.stocks.map((stock: any) => {
          const stockCode = stock.ts_code || stock.code || '';
          const stockName = stock.name || stockCode;
          const market = stockCode.includes('.SZ')
            ? '深圳'
            : stockCode.includes('.SH')
              ? '上海'
              : '未知';

          return {
            code: stockCode,
            name: stockName,
            market: market,
          };
        });

        console.log(`[StockSelector] 格式化后 ${formattedStocks.length} 只股票`);
        setAllStocks(formattedStocks);
      } catch (error) {
        console.error('[StockSelector] 加载本地股票列表失败:', error);
        setAllStocks([]);
      } finally {
        setLoadingAllStocks(false);
      }
    };

    loadAllStocks();
  }, []);

  // 处理搜索输入
  const handleSearchChange = (value: string) => {
    setSearchValue(value);
    // 搜索时重置到第一页
    setCurrentPage(1);
  };

  // 根据搜索关键词过滤股票列表
  const filteredStocks = useMemo(() => {
    if (!searchValue || searchValue.trim() === '') {
      return allStocks;
    }

    const keyword = searchValue.toLowerCase().trim();
    return allStocks.filter(
      stock =>
        stock.code.toLowerCase().includes(keyword) || stock.name.toLowerCase().includes(keyword)
    );
  }, [allStocks, searchValue]);

  // 添加股票
  const handleAddStock = (stockCode: string) => {
    if (value.includes(stockCode)) {
      console.log('股票已存在');
      return;
    }

    const newValue = [...value, stockCode];
    onChange?.(newValue);
  };

  // 随机选择股票
  const handleRandomSelect = () => {
    const count = parseInt(randomCount, 10);

    if (isNaN(count) || count <= 0) {
      console.warn('请输入有效的数量');
      return;
    }

    // 从过滤后的股票列表中随机选择（如果没有搜索，则从全部列表选择）
    const availableStocks = filteredStocks.filter(stock => !value.includes(stock.code));

    if (availableStocks.length === 0) {
      console.warn('没有可选的股票');
      return;
    }

    // 如果请求的数量大于可用数量，则选择所有可用股票
    const selectCount = Math.min(count, availableStocks.length);

    // 随机打乱数组并选择前N个
    const shuffled = [...availableStocks].sort(() => Math.random() - 0.5);
    const selectedStocks = shuffled.slice(0, selectCount).map(stock => stock.code);

    // 添加到已选列表
    const newValue = [...value, ...selectedStocks];
    onChange?.(newValue);

    console.log(`随机选择了 ${selectedStocks.length} 只股票`);
  };

  // 移除股票
  const handleRemoveStock = (stockCode: string) => {
    const newValue = value.filter(code => code !== stockCode);
    onChange?.(newValue);

    // 如果删除后当前页没有股票了，且不是第一页，则回到上一页
    const selectedStocksTotalPages = Math.ceil(newValue.length / SELECTED_STOCKS_PER_PAGE);
    if (selectedStocksPage > selectedStocksTotalPages && selectedStocksTotalPages > 0) {
      setSelectedStocksPage(selectedStocksTotalPages);
    }
  };

  // 清空所有股票
  const handleClearAll = () => {
    onChange?.([]);
    setSelectedStocksPage(1); // 重置分页
    console.log('已清空所有股票');
  };

  // 获取股票显示名称
  const getStockDisplayName = (code: string) => {
    const stock = allStocks.find(s => s.code === code);
    return stock ? `${code} ${stock.name}` : code;
  };

  // 计算分页数据（基于过滤后的列表）
  const totalPages = Math.ceil(filteredStocks.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentStocks = filteredStocks.slice(startIndex, endIndex);

  // 当过滤后的列表变化时，如果当前页超出范围，重置到第一页
  useEffect(() => {
    if (currentPage > totalPages && totalPages > 0) {
      setCurrentPage(1);
    }
  }, [totalPages, currentPage]);

  // 处理分页变化
  const handlePageChange = (event: React.ChangeEvent<unknown>, page: number) => {
    setCurrentPage(page);
    // 滚动到顶部
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 搜索框 */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          placeholder={placeholder}
          value={searchValue}
          onChange={e => handleSearchChange(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search size={16} />
              </InputAdornment>
            ),
            endAdornment: searchValue ? (
              <InputAdornment position="end">
                <IconButton size="small" onClick={() => handleSearchChange('')}>
                  <X size={16} />
                </IconButton>
              </InputAdornment>
            ) : null,
          }}
          fullWidth
        />
      </Box>

      {/* 已选股票 */}
      <Card>
        <CardHeader
          title="已选股票"
          action={
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              {/* 随机选择 */}
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <TextField
                  type="number"
                  size="small"
                  value={randomCount}
                  onChange={e => setRandomCount(e.target.value)}
                  placeholder="数量"
                  sx={{ width: 80 }}
                  inputProps={{ min: 1 }}
                />
                <Button
                  size="small"
                  variant="outlined"
                  color="primary"
                  startIcon={<Shuffle size={16} />}
                  onClick={handleRandomSelect}
                  disabled={filteredStocks.length === 0 || loadingAllStocks}
                >
                  随机选择
                </Button>
              </Box>
              {value.length > 0 && (
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  startIcon={<Trash2 size={14} />}
                  onClick={handleClearAll}
                >
                  清空
                </Button>
              )}
            </Box>
          }
        />
        <CardContent sx={{ p: 2, pb: 1 }}>
          <Box
            sx={{
              height: SELECTED_STOCKS_CARD_HEIGHT,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {value.length === 0 ? (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                }}
              >
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                  请搜索或从下方列表选择股票，或使用随机选择功能
                </Typography>
              </Box>
            ) : (
              <>
                {/* 已选股票列表区域（固定高度，可滚动） */}
                <Box
                  sx={{
                    flex: 1,
                    overflowY: 'auto',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 1,
                    alignContent: 'flex-start',
                    pb: 1,
                  }}
                >
                  {(() => {
                    const selectedStocksTotalPages = Math.ceil(
                      value.length / SELECTED_STOCKS_PER_PAGE
                    );
                    const selectedStartIndex = (selectedStocksPage - 1) * SELECTED_STOCKS_PER_PAGE;
                    const selectedEndIndex = selectedStartIndex + SELECTED_STOCKS_PER_PAGE;
                    const currentSelectedStocks = value.slice(selectedStartIndex, selectedEndIndex);

                    return currentSelectedStocks.map(code => (
                      <Chip
                        key={code}
                        label={getStockDisplayName(code)}
                        onDelete={() => handleRemoveStock(code)}
                        size="small"
                      />
                    ));
                  })()}
                </Box>

                {/* 分页控制（如果需要） */}
                {(() => {
                  const selectedStocksTotalPages = Math.ceil(
                    value.length / SELECTED_STOCKS_PER_PAGE
                  );

                  if (selectedStocksTotalPages > 1) {
                    return (
                      <Box
                        sx={{
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center',
                          gap: 1,
                          pt: 1,
                          borderTop: '1px solid',
                          borderColor: 'divider',
                        }}
                      >
                        <IconButton
                          size="small"
                          disabled={selectedStocksPage === 1}
                          onClick={() => setSelectedStocksPage(prev => Math.max(1, prev - 1))}
                        >
                          <ChevronLeft size={16} />
                        </IconButton>

                        <Typography variant="caption" color="text.secondary">
                          第 {selectedStocksPage} / {selectedStocksTotalPages} 页
                        </Typography>

                        <IconButton
                          size="small"
                          disabled={selectedStocksPage >= selectedStocksTotalPages}
                          onClick={() =>
                            setSelectedStocksPage(prev =>
                              Math.min(selectedStocksTotalPages, prev + 1)
                            )
                          }
                        >
                          <ChevronRight size={16} />
                        </IconButton>
                      </Box>
                    );
                  }
                  return null;
                })()}
              </>
            )}
          </Box>

          {/* 底部显示已选择股票数量 */}
          {value.length > 0 && (
            <Box
              sx={{
                pt: 1.5,
                mt: 1,
                borderTop: '1px solid',
                borderColor: 'divider',
                display: 'flex',
                justifyContent: 'center',
              }}
            >
              <Typography variant="body2" color="text.secondary">
                已选择 <strong>{value.length}</strong> 只股票
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* 本地股票列表 */}
      <Card>
        <CardHeader
          title={
            <Box>
              <Typography variant="h6" component="span">
                本地股票列表
              </Typography>
              {searchValue ? (
                <Typography variant="body2" color="text.secondary" component="span" sx={{ ml: 1 }}>
                  (搜索: "{searchValue}", 找到 {filteredStocks.length} 只，共 {allStocks.length} 只)
                </Typography>
              ) : (
                <Typography variant="body2" color="text.secondary" component="span" sx={{ ml: 1 }}>
                  (共 {allStocks.length} 只本地股票)
                </Typography>
              )}
            </Box>
          }
        />
        <CardContent>
          {loadingAllStocks ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress size={32} />
            </Box>
          ) : allStocks.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body2" color="text.secondary">
                暂无本地股票数据
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                请先在数据管理页面同步股票数据
              </Typography>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* 股票列表 */}
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' },
                  gap: 1,
                }}
              >
                {currentStocks.map(stock => {
                  const isSelected = value.includes(stock.code);

                  return (
                    <Button
                      key={stock.code}
                      size="small"
                      variant={isSelected ? 'contained' : 'outlined'}
                      color={isSelected ? 'primary' : 'inherit'}
                      disabled={isSelected}
                      onClick={() => handleAddStock(stock.code)}
                      sx={{
                        justifyContent: 'flex-start',
                        height: 'auto',
                        py: 1,
                        textTransform: 'none',
                      }}
                    >
                      <Box
                        sx={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                          width: '100%',
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {stock.code}
                          </Typography>
                          <Chip label={stock.market} size="small" variant="outlined" />
                        </Box>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{
                            mt: 0.5,
                            width: '100%',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {stock.name}
                        </Typography>
                      </Box>
                    </Button>
                  );
                })}
              </Box>

              {/* 分页控制 */}
              {totalPages > 1 && (
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    gap: 1,
                    pt: 2,
                  }}
                >
                  <IconButton
                    size="small"
                    disabled={currentPage === 1}
                    onClick={() =>
                      handlePageChange({} as React.ChangeEvent<unknown>, currentPage - 1)
                    }
                  >
                    <ChevronLeft size={16} />
                  </IconButton>

                  <Pagination
                    count={totalPages}
                    page={currentPage}
                    onChange={handlePageChange}
                    size="small"
                    color="primary"
                  />

                  <IconButton
                    size="small"
                    disabled={currentPage === totalPages}
                    onClick={() =>
                      handlePageChange({} as React.ChangeEvent<unknown>, currentPage + 1)
                    }
                  >
                    <ChevronRight size={16} />
                  </IconButton>
                </Box>
              )}

              {/* 分页信息 */}
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ textAlign: 'center', pt: 1 }}
              >
                显示第 {startIndex + 1} - {Math.min(endIndex, filteredStocks.length)} 条，共{' '}
                {filteredStocks.length} 条
                {searchValue && filteredStocks.length < allStocks.length && (
                  <span style={{ marginLeft: 8 }}>(全部 {allStocks.length} 条)</span>
                )}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};
