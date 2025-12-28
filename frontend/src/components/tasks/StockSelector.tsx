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
  AutoComplete,
  Button,
  Tag,
  Space,
  Typography,
  Row,
  Col,
  Input,
  Divider,
  message,
  Tooltip,
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  StarOutlined,
  SearchOutlined,
  ClearOutlined,
} from '@ant-design/icons';
import { DataService } from '../../services/dataService';

const { Text } = Typography;

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
      message.warning('股票已存在');
      return;
    }

    if (value.length >= maxCount) {
      message.warning(`最多只能选择 ${maxCount} 只股票`);
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
      message.warning('没有可添加的热门股票');
      return;
    }

    const newValue = [...value, ...availableStocks.map(stock => stock.code)];
    onChange?.(newValue);
    message.success(`已添加 ${availableStocks.length} 只热门股票`);
  };

  // 清空所有股票
  const handleClearAll = () => {
    onChange?.([]);
    message.success('已清空所有股票');
  };

  // 获取股票显示名称
  const getStockDisplayName = (code: string) => {
    const stock = popularStocks.find(s => s.code === code);
    return stock ? `${code} ${stock.name}` : code;
  };

  return (
    <div>
      {/* 搜索框 */}
      <Row gutter={8} style={{ marginBottom: 16 }}>
        <Col flex={1}>
          <AutoComplete
            value={searchValue}
            placeholder={placeholder}
            onSearch={handleSearch}
            onSelect={handleAddStock}
            options={stockOptions.map(stock => ({
              value: stock.code,
              label: (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Text strong>{stock.code}</Text>
                    <Text style={{ marginLeft: 8 }}>{stock.name}</Text>
                    <Tag style={{ marginLeft: 8 }}>
                      {stock.market}
                    </Tag>
                  </div>
                  {stock.change_percent !== undefined && (
                    <Text
                      style={{
                        color: stock.change_percent > 0 ? '#52c41a' : stock.change_percent < 0 ? '#ff4d4f' : undefined
                      }}
                    >
                      {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                    </Text>
                  )}
                </div>
              ),
            }))}
            style={{ width: '100%' }}
          />
        </Col>
        <Col>
          <Button
            icon={<StarOutlined />}
            onClick={handleAddPopularBatch}
            loading={loadingPopular}
            disabled={value.length >= maxCount}
          >
            添加热门
          </Button>
        </Col>
      </Row>

      {/* 已选股票 */}
      <Card
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>已选股票 ({value.length}/{maxCount})</span>
            {value.length > 0 && (
              <Button
                type="text"
                size="small"
                icon={<ClearOutlined />}
                onClick={handleClearAll}
                danger
              >
                清空
              </Button>
            )}
          </div>
        }
        size="small"
        style={{ marginBottom: 16 }}
      >
        <div style={{ minHeight: 60 }}>
          {value.length === 0 ? (
            <Text type="secondary">请搜索并选择股票</Text>
          ) : (
            <Space wrap>
              {value.map(code => (
                <Tag
                  key={code}
                  closable
                  onClose={() => handleRemoveStock(code)}
                  style={{ marginBottom: 4 }}
                >
                  {getStockDisplayName(code)}
                </Tag>
              ))}
            </Space>
          )}
        </div>
      </Card>

      {/* 热门股票推荐 */}
      <Card title="热门股票" size="small" loading={loadingPopular}>
        <Row gutter={[8, 8]}>
          {popularStocks.map(stock => (
            <Col key={stock.code}>
              <Tooltip
                title={
                  <div>
                    <div>{stock.name}</div>
                    <div>市场: {stock.market}</div>
                    {stock.change_percent !== undefined && (
                      <div>涨跌幅: {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%</div>
                    )}
                    {stock.volume && (
                      <div>成交量: {(stock.volume / 10000).toFixed(1)}万</div>
                    )}
                  </div>
                }
              >
                <Button
                  size="small"
                  type={value.includes(stock.code) ? 'primary' : 'default'}
                  disabled={value.includes(stock.code) || value.length >= maxCount}
                  onClick={() => handleAddPopularStock(stock)}
                  style={{
                    borderColor: stock.change_percent && stock.change_percent > 0 ? '#52c41a' : 
                                stock.change_percent && stock.change_percent < 0 ? '#ff4d4f' : undefined
                  }}
                >
                  {stock.code}
                  {stock.change_percent !== undefined && (
                    <span
                      style={{
                        marginLeft: 4,
                        color: stock.change_percent > 0 ? '#52c41a' : stock.change_percent < 0 ? '#ff4d4f' : undefined
                      }}
                    >
                      {stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(1)}%
                    </span>
                  )}
                </Button>
              </Tooltip>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );
};