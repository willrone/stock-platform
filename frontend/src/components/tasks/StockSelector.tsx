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
  CardHeader,
  CardBody,
  Button,
  Chip,
  Input,
  Pagination,
} from '@heroui/react';
import {
  X,
  Search,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Shuffle,
} from 'lucide-react';
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
          const market = stockCode.includes('.SZ') ? '深圳' : stockCode.includes('.SH') ? '上海' : '未知';
          
          return {
            code: stockCode,
            name: stockName,
            market: market
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
    return allStocks.filter(stock =>
      stock.code.toLowerCase().includes(keyword) ||
      stock.name.toLowerCase().includes(keyword)
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
  };

  // 清空所有股票
  const handleClearAll = () => {
    onChange?.([]);
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
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    // 滚动到顶部
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="space-y-4">
      {/* 搜索框 */}
      <div className="flex gap-2">
        <Input
          placeholder={placeholder}
          startContent={<Search className="w-4 h-4" />}
          value={searchValue}
          onValueChange={handleSearchChange}
          endContent={
            searchValue ? (
              <Button
                isIconOnly
                size="sm"
                variant="light"
                onPress={() => handleSearchChange('')}
                className="min-w-0"
              >
                <X className="w-4 h-4" />
              </Button>
            ) : null
          }
          className="flex-1"
        />
      </div>

      {/* 已选股票 */}
      <Card>
        <CardHeader className="flex justify-between items-center">
          <span>已选股票 ({value.length} 只)</span>
          <div className="flex gap-2 items-center">
            {/* 随机选择 */}
            <div className="flex gap-2 items-center">
              <Input
                type="number"
                size="sm"
                value={randomCount}
                onValueChange={setRandomCount}
                placeholder="数量"
                className="w-20"
                min={1}
              />
              <Button
                size="sm"
                variant="flat"
                color="primary"
                startContent={<Shuffle className="w-4 h-4" />}
                onPress={handleRandomSelect}
                isDisabled={filteredStocks.length === 0 || loadingAllStocks}
              >
                随机选择
              </Button>
            </div>
            {value.length > 0 && (
              <Button
                size="sm"
                variant="light"
                color="danger"
                startContent={<Trash2 className="w-3 h-3" />}
                onPress={handleClearAll}
              >
                清空
              </Button>
            )}
          </div>
        </CardHeader>
        <CardBody>
          <div className="min-h-16">
            {value.length === 0 ? (
              <p className="text-default-500 text-center py-4">请搜索或从下方列表选择股票，或使用随机选择功能</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {value.map(code => (
                  <Chip
                    key={code}
                    onClose={() => handleRemoveStock(code)}
                    variant="flat"
                  >
                    {getStockDisplayName(code)}
                  </Chip>
                ))}
              </div>
            )}
          </div>
        </CardBody>
      </Card>

      {/* 本地股票列表 */}
      <Card>
        <CardHeader className="flex justify-between items-center">
          <span>
            本地股票列表 
            {searchValue ? (
              <span className="text-default-500">
                (搜索: "{searchValue}", 找到 {filteredStocks.length} 只，共 {allStocks.length} 只)
              </span>
            ) : (
              <span className="text-default-500">(共 {allStocks.length} 只本地股票)</span>
            )}
          </span>
        </CardHeader>
        <CardBody>
          {loadingAllStocks ? (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : allStocks.length === 0 ? (
            <div className="text-center py-8 text-default-500">
              <p>暂无本地股票数据</p>
              <p className="text-sm mt-2">请先在数据管理页面同步股票数据</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* 股票列表 */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {currentStocks.map(stock => {
                  const isSelected = value.includes(stock.code);
                  
                  return (
                    <Button
                      key={stock.code}
                      size="sm"
                      variant={isSelected ? "solid" : "bordered"}
                      color={isSelected ? "primary" : "default"}
                      isDisabled={isSelected}
                      onPress={() => handleAddStock(stock.code)}
                      className="justify-start h-auto py-2"
                    >
                      <div className="flex flex-col items-start w-full">
                        <div className="flex items-center space-x-2 w-full">
                          <span className="font-medium text-sm">{stock.code}</span>
                          <Chip size="sm" variant="flat" className="text-xs">
                            {stock.market}
                          </Chip>
                        </div>
                        <span className="text-xs text-default-500 mt-1 truncate w-full">
                          {stock.name}
                        </span>
                      </div>
                    </Button>
                  );
                })}
              </div>

              {/* 分页控制 */}
              {totalPages > 1 && (
                <div className="flex justify-center items-center gap-2 pt-4">
                  <Button
                    size="sm"
                    variant="light"
                    isDisabled={currentPage === 1}
                    onPress={() => handlePageChange(currentPage - 1)}
                    startContent={<ChevronLeft className="w-4 h-4" />}
                  >
                    上一页
                  </Button>
                  
                  <Pagination
                    total={totalPages}
                    page={currentPage}
                    onChange={handlePageChange}
                    size="sm"
                    showControls
                  />
                  
                  <Button
                    size="sm"
                    variant="light"
                    isDisabled={currentPage === totalPages}
                    onPress={() => handlePageChange(currentPage + 1)}
                    endContent={<ChevronRight className="w-4 h-4" />}
                  >
                    下一页
                  </Button>
                </div>
              )}

              {/* 分页信息 */}
              <div className="text-center text-sm text-default-500 pt-2">
                显示第 {startIndex + 1} - {Math.min(endIndex, filteredStocks.length)} 条，共 {filteredStocks.length} 条
                {searchValue && filteredStocks.length < allStocks.length && (
                  <span className="ml-2">(全部 {allStocks.length} 条)</span>
                )}
              </div>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
};
