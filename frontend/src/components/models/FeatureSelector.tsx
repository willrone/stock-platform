/**
 * 特征选择组件
 * 
 * 用于模型创建时的特征选择功能
 */

'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Button, Checkbox, Input, Tooltip, Accordion, AccordionItem } from '@heroui/react';
import { Search, Info } from 'lucide-react';
import { DataService } from '../../services/dataService';
import { 
  groupAlphaFactorsByCategory, 
  parseAlphaFactor, 
  getAlphaFactorDisplayName,
  ALPHA_FACTOR_CATEGORIES 
} from '../../utils/alphaFactorInfo';

interface FeatureSelectorProps {
  stockCodes: string[];
  startDate: string;
  endDate: string;
  selectedFeatures: string[];
  onFeaturesChange: (features: string[]) => void;
  useAllFeatures: boolean;
  onUseAllFeaturesChange: (useAll: boolean) => void;
}

export function FeatureSelector({
  stockCodes,
  startDate,
  endDate,
  selectedFeatures,
  onFeaturesChange,
  useAllFeatures,
  onUseAllFeaturesChange,
}: FeatureSelectorProps) {
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [featureCategories, setFeatureCategories] = useState<{
    base_features: string[];
    indicator_features: string[];
    fundamental_features: string[];
    alpha_features: string[];
  }>({
    base_features: [],
    indicator_features: [],
    fundamental_features: [],
    alpha_features: [],
  });
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  const [alphaSearchTerm, setAlphaSearchTerm] = useState('');
  const [expandedAlphaCategories, setExpandedAlphaCategories] = useState<Set<string>>(new Set());

  // 加载可用特征列表
  const loadAvailableFeatures = async () => {
    try {
      setLoadingFeatures(true);
      // 如果已选择股票和日期，使用实际数据获取特征
      if (stockCodes.length > 0 && startDate && endDate) {
        const result = await DataService.getAvailableFeatures({
          stock_code: stockCodes[0],
          start_date: startDate,
          end_date: endDate,
        });
        console.log('获取到的特征数据（实际）:', result);
        
        const features = result.features || [];
        const categories = result.feature_categories || {
          base_features: [],
          indicator_features: [],
          fundamental_features: [],
          alpha_features: [],
        };
        
        setAvailableFeatures(features);
        setFeatureCategories(categories);
      } else {
        // 否则获取理论特征列表
        const result = await DataService.getAvailableFeatures();
        console.log('获取到的特征数据（理论）:', result);
        
        const features = result.features || [];
        const categories = result.feature_categories || {
          base_features: [],
          indicator_features: [],
          fundamental_features: [],
          alpha_features: [],
        };
        
        // 如果分类为空但features有数据，尝试手动分类
        if (features.length > 0 && Object.values(categories).every(arr => arr.length === 0)) {
          const baseFeatures = ['open', 'high', 'low', 'close', 'volume'];
          const indicatorFeatures = features.filter(f => 
            f.includes('ma_') || f.includes('rsi') || f.includes('macd') || 
            f.includes('bb_') || f.includes('sma') || f.includes('ema') ||
            f.includes('stoch') || f.includes('kdj') || f.includes('atr') ||
            f.includes('williams') || f.includes('cci') || f.includes('momentum') ||
            f.includes('roc') || f.includes('sar') || f.includes('adx') ||
            f.includes('vwap') || f.includes('obv') || f.includes('volume_rsi')
          );
          const fundamentalFeatures = features.filter(f => 
            f.includes('price_change') || f.includes('volume_change') || 
            f.includes('volatility') || f.includes('price_position') ||
            f.includes('volume_ma_ratio')
          );
          const alphaFeatures = features.filter(f => f.startsWith('alpha_'));
          
          setFeatureCategories({
            base_features: features.filter(f => baseFeatures.includes(f)),
            indicator_features: indicatorFeatures,
            fundamental_features: fundamentalFeatures,
            alpha_features: alphaFeatures,
          });
        } else {
          setFeatureCategories(categories);
        }
        
        setAvailableFeatures(features);
      }
    } catch (error) {
      console.error('加载可用特征列表失败:', error);
      // 如果加载失败，使用默认特征列表
      const defaultFeatures = [
        'open', 'high', 'low', 'close', 'volume',
        'ma_5', 'ma_10', 'ma_20', 'ma_60',
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower'
      ];
      setAvailableFeatures(defaultFeatures);
      setFeatureCategories({
        base_features: ['open', 'high', 'low', 'close', 'volume'],
        indicator_features: ['ma_5', 'ma_10', 'ma_20', 'ma_60', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower'],
        fundamental_features: [],
        alpha_features: [],
      });
    } finally {
      setLoadingFeatures(false);
    }
  };

  // 当股票代码或日期变化时，重新加载特征列表
  useEffect(() => {
    loadAvailableFeatures();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stockCodes, startDate, endDate]);

  const handleFeatureToggle = (feature: string, checked: boolean) => {
    if (checked) {
      onFeaturesChange([...selectedFeatures, feature]);
    } else {
      onFeaturesChange(selectedFeatures.filter((f) => f !== feature));
    }
  };

  // Alpha因子分组和过滤
  const groupedAlphaFactors = useMemo(() => {
    if (featureCategories.alpha_features.length === 0) return {};
    
    // 过滤Alpha因子
    const filtered = alphaSearchTerm
      ? featureCategories.alpha_features.filter(f => 
          f.toLowerCase().includes(alphaSearchTerm.toLowerCase()) ||
          getAlphaFactorDisplayName(f).toLowerCase().includes(alphaSearchTerm.toLowerCase())
        )
      : featureCategories.alpha_features;
    
    return groupAlphaFactorsByCategory(filtered);
  }, [featureCategories.alpha_features, alphaSearchTerm]);

  // 批量选择/取消选择某个类别的Alpha因子
  const handleCategoryToggle = (category: string, checked: boolean) => {
    const categoryFactors = groupedAlphaFactors[category] || [];
    if (checked) {
      // 添加该类别的所有因子（排除已选择的）
      const newFeatures = categoryFactors.filter(f => !selectedFeatures.includes(f));
      onFeaturesChange([...selectedFeatures, ...newFeatures]);
    } else {
      // 移除该类别的所有因子
      onFeaturesChange(selectedFeatures.filter(f => !categoryFactors.includes(f)));
    }
  };

  // 检查某个类别的因子是否全部被选择
  const isCategoryFullySelected = (category: string): boolean => {
    const categoryFactors = groupedAlphaFactors[category] || [];
    if (categoryFactors.length === 0) return false;
    return categoryFactors.every(f => selectedFeatures.includes(f));
  };

  // 检查某个类别的因子是否部分被选择
  const isCategoryPartiallySelected = (category: string): boolean => {
    const categoryFactors = groupedAlphaFactors[category] || [];
    if (categoryFactors.length === 0) return false;
    const selectedCount = categoryFactors.filter(f => selectedFeatures.includes(f)).length;
    return selectedCount > 0 && selectedCount < categoryFactors.length;
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">特征选择</label>
        <Button
          size="sm"
          variant="light"
          onPress={loadAvailableFeatures}
          isLoading={loadingFeatures}
        >
          {loadingFeatures ? '加载中...' : '刷新特征列表'}
        </Button>
      </div>
      
      <Checkbox
        isSelected={useAllFeatures}
        onValueChange={(checked) => {
          onUseAllFeaturesChange(checked);
          if (checked) {
            onFeaturesChange([]);
          }
        }}
      >
        使用所有可用特征（推荐）
      </Checkbox>

      {!useAllFeatures && (
        <div className="space-y-3 mt-3 p-4 border border-default-200 rounded-lg max-h-96 overflow-y-auto">
          {loadingFeatures ? (
            <div className="text-center py-4 text-default-500">加载特征列表中...</div>
          ) : availableFeatures.length === 0 && Object.values(featureCategories).every(arr => arr.length === 0) ? (
            <div className="text-center py-4 text-default-500">
              <p>暂无可用特征</p>
              <p className="text-xs mt-2">请先选择股票和日期范围，或点击"刷新特征列表"按钮</p>
            </div>
          ) : (
            <>
              {/* 基础价格特征 */}
              {featureCategories.base_features.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-default-700">基础价格特征</h4>
                  <div className="flex flex-wrap gap-2">
                    {featureCategories.base_features.map((feature) => (
                      <Checkbox
                        key={feature}
                        size="sm"
                        isSelected={selectedFeatures.includes(feature)}
                        onValueChange={(checked) => handleFeatureToggle(feature, checked)}
                      >
                        {feature}
                      </Checkbox>
                    ))}
                  </div>
                </div>
              )}

              {/* 技术指标特征 */}
              {featureCategories.indicator_features.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-default-700">技术指标特征</h4>
                  <div className="flex flex-wrap gap-2">
                    {featureCategories.indicator_features.map((feature) => (
                      <Checkbox
                        key={feature}
                        size="sm"
                        isSelected={selectedFeatures.includes(feature)}
                        onValueChange={(checked) => handleFeatureToggle(feature, checked)}
                      >
                        {feature}
                      </Checkbox>
                    ))}
                  </div>
                </div>
              )}

              {/* 基本面特征 */}
              {featureCategories.fundamental_features.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-default-700">基本面特征</h4>
                  <div className="flex flex-wrap gap-2">
                    {featureCategories.fundamental_features.map((feature) => (
                      <Checkbox
                        key={feature}
                        size="sm"
                        isSelected={selectedFeatures.includes(feature)}
                        onValueChange={(checked) => handleFeatureToggle(feature, checked)}
                      >
                        {feature}
                      </Checkbox>
                    ))}
                  </div>
                </div>
              )}

              {/* 如果所有分类都为空，显示所有特征 */}
              {Object.values(featureCategories).every(arr => arr.length === 0) && availableFeatures.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-default-700">所有可用特征 ({availableFeatures.length})</h4>
                  <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto">
                    {availableFeatures.slice(0, 100).map((feature) => (
                      <Checkbox
                        key={feature}
                        size="sm"
                        isSelected={selectedFeatures.includes(feature)}
                        onValueChange={(checked) => handleFeatureToggle(feature, checked)}
                      >
                        {feature}
                      </Checkbox>
                    ))}
                    {availableFeatures.length > 100 && (
                      <div className="text-xs text-default-500 w-full">
                        还有 {availableFeatures.length - 100} 个特征...
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Alpha因子特征 - 优化显示 */}
              {featureCategories.alpha_features.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-default-700">
                        Alpha因子特征 ({featureCategories.alpha_features.length}个)
                      </h4>
                      <Tooltip content="Alpha因子是量化投资中用于预测股票未来收益率的量化特征，包括价格收益率、移动平均、波动率等多种类型">
                        <Info className="w-4 h-4 text-default-400 cursor-help" />
                      </Tooltip>
                    </div>
                    <Button
                      size="sm"
                      variant="light"
                      onPress={() => {
                        // 切换所有Alpha因子的选择状态
                        const allAlphaSelected = featureCategories.alpha_features.every(f => selectedFeatures.includes(f));
                        if (allAlphaSelected) {
                          onFeaturesChange(selectedFeatures.filter(f => !featureCategories.alpha_features.includes(f)));
                        } else {
                          const newFeatures = featureCategories.alpha_features.filter(f => !selectedFeatures.includes(f));
                          onFeaturesChange([...selectedFeatures, ...newFeatures]);
                        }
                      }}
                    >
                      {featureCategories.alpha_features.every(f => selectedFeatures.includes(f)) ? '全不选' : '全选'}
                    </Button>
                  </div>

                  {/* 搜索框 */}
                  <Input
                    size="sm"
                    placeholder="搜索Alpha因子..."
                    value={alphaSearchTerm}
                    onValueChange={setAlphaSearchTerm}
                    startContent={<Search className="w-4 h-4 text-default-400" />}
                    classNames={{
                      input: "text-sm",
                    }}
                  />

                  {/* 按类别分组的Alpha因子 */}
                  <div className="max-h-96 overflow-y-auto border border-default-200 rounded-lg p-2">
                    {Object.keys(groupedAlphaFactors).length === 0 ? (
                      <div className="text-center py-4 text-default-500 text-sm">
                        {alphaSearchTerm ? '未找到匹配的Alpha因子' : '暂无Alpha因子'}
                      </div>
                    ) : (
                      <Accordion
                        selectionMode="multiple"
                        selectedKeys={expandedAlphaCategories}
                        onSelectionChange={(keys) => {
                          if (keys === 'all') {
                            setExpandedAlphaCategories(new Set(Object.keys(groupedAlphaFactors)));
                          } else {
                            const keysArray = Array.isArray(keys) ? keys : Array.from(keys as Set<string>);
                            setExpandedAlphaCategories(new Set(keysArray));
                          }
                        }}
                        variant="bordered"
                      >
                        {Object.entries(groupedAlphaFactors)
                          .sort(([a], [b]) => {
                            // 优先显示有名称的类别
                            const order = ['MOMENTUM', 'MEAN_REVERSION', 'VOLATILITY', 'VOLUME', 'RESI', 'MA', 'STD', 'VSTD', 'CORR', 'MAX', 'MIN', 'QTLU', 'OTHER'];
                            const indexA = order.indexOf(a);
                            const indexB = order.indexOf(b);
                            if (indexA !== -1 && indexB !== -1) return indexA - indexB;
                            if (indexA !== -1) return -1;
                            if (indexB !== -1) return 1;
                            return a.localeCompare(b);
                          })
                          .map(([category, factors]) => {
                            const categoryInfo = ALPHA_FACTOR_CATEGORIES[category as keyof typeof ALPHA_FACTOR_CATEGORIES] || ALPHA_FACTOR_CATEGORIES.OTHER;
                            const isFullySelected = isCategoryFullySelected(category);
                            const isPartiallySelected = isCategoryPartiallySelected(category);
                            
                            return (
                              <AccordionItem
                                key={category}
                                title={
                                  <div className="flex items-center justify-between w-full pr-4">
                                    <div className="flex items-center gap-2">
                                      <span>{categoryInfo.icon}</span>
                                      <span className="font-medium">{categoryInfo.name}</span>
                                      <span className="text-xs text-default-500">({factors.length}个)</span>
                                    </div>
                                    <Checkbox
                                      size="sm"
                                      isSelected={isFullySelected}
                                      isIndeterminate={isPartiallySelected}
                                      onValueChange={(checked) => handleCategoryToggle(category, checked)}
                                      onClick={(e) => e.stopPropagation()}
                                    />
                                  </div>
                                }
                                subtitle={
                                  <div className="text-xs text-default-500 mt-1">
                                    {categoryInfo.description}
                                  </div>
                                }
                              >
                                <div className="space-y-2 pt-2">
                                  <div className="grid grid-cols-2 gap-2">
                                    {factors.map((feature) => {
                                      const info = parseAlphaFactor(feature);
                                      const displayName = getAlphaFactorDisplayName(feature);
                                      
                                      return (
                                        <div key={feature} className="flex items-start gap-2">
                                          <Checkbox
                                            size="sm"
                                            isSelected={selectedFeatures.includes(feature)}
                                            onValueChange={(checked) => handleFeatureToggle(feature, checked)}
                                            classNames={{
                                              label: "text-xs",
                                            }}
                                          >
                                            <div className="flex flex-col">
                                              <span className="font-medium">{displayName}</span>
                                              <span className="text-xs text-default-500">
                                                {feature}
                                              </span>
                                            </div>
                                          </Checkbox>
                                          {info.formula && (
                                            <Tooltip content={info.formula}>
                                              <Info className="w-3 h-3 text-default-400 cursor-help mt-1 flex-shrink-0" />
                                            </Tooltip>
                                          )}
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              </AccordionItem>
                            );
                          })}
                      </Accordion>
                    )}
                  </div>

                  {/* 已选择的Alpha因子统计 */}
                  {selectedFeatures.filter(f => featureCategories.alpha_features.includes(f)).length > 0 && (
                    <div className="text-xs text-default-600 pt-2 border-t border-default-200">
                      已选择 <strong>{selectedFeatures.filter(f => featureCategories.alpha_features.includes(f)).length}</strong> 个Alpha因子
                    </div>
                  )}
                </div>
              )}

              {selectedFeatures.length > 0 && (
                <div className="mt-3 pt-3 border-t border-default-200">
                  <p className="text-sm text-default-600">
                    已选择 <strong>{selectedFeatures.length}</strong> 个特征
                  </p>
                  <Button
                    size="sm"
                    variant="light"
                    className="mt-2"
                    onPress={() => onFeaturesChange([])}
                  >
                    清空选择
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

