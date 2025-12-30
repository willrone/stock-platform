/**
 * 股票选择器组件
 * 
 * 提供股票搜索和选择功能，包括：
 * - 股票搜索
 * - 热门股票推荐
 * - 已选股票管理
 * - 批量操作
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Input,
  Button,
  Chip,
  Autocomplete,
  AutocompleteItem,
  Tooltip,
} from '@heroui/react';
import {
  Plus,
  X,
  Star,
  Search,
  Trash2,
} from 'lucide-react';
import { DataService } from '../../services/dataService';

interface StockOption {
  code: string;
  name: string;
  market: string;
  change_percent?: number;
  volume?: number;
}

interface StockSelectorProps {
  value?: string[];
  onChange?: (stocks: string[]) => void;
  maxCount?: number;
  placeholder?: string;
}

export const StockSelector: React.FC<StockSelectorProps> = ({
  value = [],
  onChange,
  maxCount = 50,
  placeholder = '搜索股票代码或名称',
}) => {
  const [searchValue, setSearchValue] = useState('');
  const [stockOptions, setStockOptions] = useState<StockOption[]>([]);
  const [popularStocks, setPopularStocks] = useState<StockOption[]>([]);
  const [searching, setSearching] = useState(false);
  const [loadingPopular, setLoadingPopular] = useState(false);

  // 加载热门股票
  useEffect(() => {
    const loadPopularStocks = async () => {
      setLoadingPopular(true);
      try {
        const stocks = await DataService.getPopularStocks();
        // 确保数据格式正确
        const formattedStocks = stocks.map(stock => ({
          ...stock,
          market: (stock as any).market || '未知'
        }));
        setPopularStocks(formattedStocks);
      } catch (error) {
        // 使用模拟数据
        const mockPopular: StockOption[] = [
          { code: '000001.SZ', name: '平安银行', market: '深圳', change_percent: 2.5, volume: 1000000 },
          { code: '000002.SZ', name: '万科A', market: '深圳', change_percent: -1.2, volume: 800000 },
          { code: '600000.SH', name: '浦发银行', market: '上海', change_percent: 1.8, volume: 1200000 },
          { code: '600036.SH', name: '招商银行', market: '上海', change_percent: 3.2, volume: 1500000 },
          { code: '000858.SZ', name: '五粮液', market: '深圳', change_percent: 0.8, volume: 600000 },
          { code: '600519.SH', name: '贵州茅台', market: '上海', change_percent: -0.5, volume: 400000 },
          { code: '000725.SZ', name: '京东方A', market: '深圳', change_percent: 4.1, volume: 2000000 },
          { code: '600887.SH', name: '伊利股份', market: '上海', change_percent: 1.5, volume: 700000 },
        ];
        setPopularStocks(mockPopular);
      } finally {
        setLoadingPopular(false);
      }
    };

    loadPopularStocks();
  }, []);

  // 搜索股票
  const handleSearch = async (searchText: string) => {
    setSearchValue(searchText);
    
    if (!searchText || searchText.length < 2) {
      setStockOptions([]);
      return;
    }

    setSearching(true);
    try {
      const results = await DataService.searchStocks(searchText);
      setStockOptions(results);
    } catch (error) {
      // 使用模拟搜索结果
      const mockResults = popularStocks.filter(stock =>
        stock.code.toLowerCase().includes(searchText.toLowerCase()) ||
        stock.name.includes(searchText)
      );
      setStockOptions(mockResults);
    } finally {
      setSearching(false);
    }
  };

  // 添加股票
  const handleAddStock = (stockCode: string) => {
    if (value.includes(stockCode)) {
      console.log('股票已存在');
      return;
    }

    if (value.length >= maxCount) {
      console.log(`最多只能选择 ${maxCount} 只股票`);
      return;
    }

    const newValue = [...value, stockCode];
    onChange?.(newValue);
    setSearchValue('');
    setStockOptions([]);
  };

  // 移除股票
  const handleRemoveStock = (stockCode: string) => {
    const newValue = value.filter(code => code !== stockCode);
    onChange?.(newValue);
  };

  // 添加热门股票
  const handleAddPopularStock = (stock: StockOption) => {
    handleAddStock(stock.code);
  };

  // 批量添加热门股票
  const handleAddPopularBatch = () => {
    const availableStocks = popularStocks
      .filter(stock => !value.includes(stock.code))
      .slice(0, Math.min(5, maxCount - value.length));

    if (availableStocks.length === 0) {
      console.log('没有可添加的热门股票');
      return;
    }

    const newValue = [...value, ...availableStocks.map(stock => stock.code)];
    onChange?.(newValue);
    console.log(`已添加 ${availableStocks.length} 只热门股票`);
  };

  // 清空所有股票
  const handleClearAll = () => {
    onChange?.([]);
    console.log('已清空所有股票');
  };

  // 获取股票显示名称
  const getStockDisplayName = (code: string) => {
    const stock = popularStocks.find(s => s.code === code);
    return stock ? `${code} ${stock.name}` : code;
  };

  return (
    <div className="space-y-4">
      {/* 搜索框 */}
      <div className="flex gap-2">
        <Autocomplete
          placeholder={placeholder}
          startContent={<Search className="w-4 h-4" />}
          value={searchValue}
          onInputChange={handleSearch}
          onSelectionChange={(key) => {
            if (key) {
              handleAddStock(key as string);
            }
          }}
          isLoading={searching}
          className="flex-1"
        >
          {stockOptions.map((stock) => (
            <AutocompleteItem key={stock.code}>
              <div className="flex justify-between items-center w-full">
                <div className="flex items-center space-x-2">
                  <span className="font-medium">{stock.code}</span>
                  <span className="text-default-500">{stock.name}</span>
                  <Chip size="sm" variant="flat">{stock.market}</Chip>
                </div>
                {stock.change_percent !== undefined && (
                  <span className={
                    stock.change_percent > 0 
                      ? 'text-success' 
                      : stock.change_percent < 0 
                        ? 'text-danger' 
                        : 'text-default-500'
                  }>
                    {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                  </span>
                )}
              </div>
            </AutocompleteItem>
          ))}
        </Autocomplete>
        
        <Button
          startContent={<Star className="w-4 h-4" />}
          onPress={handleAddPopularBatch}
          isLoading={loadingPopular}
          isDisabled={value.length >= maxCount}
          variant="light"
        >
          添加热门
        </Button>
      </div>

      {/* 已选股票 */}
      <Card>
        <CardHeader className="flex justify-between items-center">
          <span>已选股票 ({value.length}/{maxCount})</span>
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
        </CardHeader>
        <CardBody>
          <div className="min-h-16">
            {value.length === 0 ? (
              <p className="text-default-500 text-center py-4">请搜索并选择股票</p>
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

      {/* 热门股票推荐 */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Star className="w-4 h-4" />
            <span>热门股票</span>
          </div>
        </CardHeader>
        <CardBody>
          {loadingPopular ? (
            <div className="flex justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {popularStocks.map(stock => (
                <Tooltip
                  key={stock.code}
                  content={
                    <div className="p-2">
                      <div className="font-medium">{stock.name}</div>
                      <div className="text-sm">市场: {stock.market}</div>
                      {stock.change_percent !== undefined && (
                        <div className="text-sm">
                          涨跌幅: {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                        </div>
                      )}
                      {stock.volume && (
                        <div className="text-sm">
                          成交量: {(stock.volume / 10000).toFixed(1)}万
                        </div>
                      )}
                    </div>
                  }
                >
                  <Button
                    size="sm"
                    variant={value.includes(stock.code) ? "solid" : "light"}
                    color={value.includes(stock.code) ? "primary" : "default"}
                    isDisabled={value.includes(stock.code) || value.length >= maxCount}
                    onPress={() => handleAddPopularStock(stock)}
                    className="w-full justify-start"
                  >
                    <div className="flex flex-col items-start">
                      <span className="text-xs">{stock.code}</span>
                      {stock.change_percent !== undefined && (
                        <span className={`text-xs ${
                          stock.change_percent > 0 
                            ? 'text-success' 
                            : stock.change_percent < 0 
                              ? 'text-danger' 
                              : 'text-default-500'
                        }`}>
                          {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </Button>
                </Tooltip>
              ))}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
};