/**
 * 特征选择组件
 *
 * 用于模型创建时的特征选择功能
 */

'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  Button,
  Checkbox,
  TextField,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Select,
  MenuItem,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Box,
  Typography,
  IconButton,
  FormControl,
  InputLabel,
  FormHelperText,
  InputAdornment,
  CircularProgress,
} from '@mui/material';
import { ExpandMore } from '@mui/icons-material';
import { Search, Info, Save, Settings, Trash2 } from 'lucide-react';
import { DataService } from '../../services/dataService';
import {
  groupAlphaFactorsByCategory,
  parseAlphaFactor,
  getAlphaFactorDisplayName,
  ALPHA_FACTOR_CATEGORIES,
} from '../../utils/alphaFactorInfo';
import {
  getAllConfigs,
  getConfigById,
  saveCustomConfig,
  deleteCustomConfig,
  type FeatureConfig,
} from '../../utils/featureConfigs';

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
  const [expandedAlphaCategories, setExpandedAlphaCategories] = useState<string[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState<string>('');
  const [configs, setConfigs] = useState<FeatureConfig[]>([]);
  const [saveConfigName, setSaveConfigName] = useState('');
  const [saveConfigDescription, setSaveConfigDescription] = useState('');
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);

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
          const indicatorFeatures = features.filter(
            f =>
              f.includes('ma_') ||
              f.includes('rsi') ||
              f.includes('macd') ||
              f.includes('bb_') ||
              f.includes('sma') ||
              f.includes('ema') ||
              f.includes('stoch') ||
              f.includes('kdj') ||
              f.includes('atr') ||
              f.includes('williams') ||
              f.includes('cci') ||
              f.includes('momentum') ||
              f.includes('roc') ||
              f.includes('sar') ||
              f.includes('adx') ||
              f.includes('vwap') ||
              f.includes('obv') ||
              f.includes('volume_rsi')
          );
          const fundamentalFeatures = features.filter(
            f =>
              f.includes('price_change') ||
              f.includes('volume_change') ||
              f.includes('volatility') ||
              f.includes('price_position') ||
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
        'open',
        'high',
        'low',
        'close',
        'volume',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_60',
        'rsi',
        'macd',
        'macd_signal',
        'bb_upper',
        'bb_lower',
      ];
      setAvailableFeatures(defaultFeatures);
      setFeatureCategories({
        base_features: ['open', 'high', 'low', 'close', 'volume'],
        indicator_features: [
          'ma_5',
          'ma_10',
          'ma_20',
          'ma_60',
          'rsi',
          'macd',
          'macd_signal',
          'bb_upper',
          'bb_lower',
        ],
        fundamental_features: [],
        alpha_features: [],
      });
    } finally {
      setLoadingFeatures(false);
    }
  };

  // 加载配置列表
  useEffect(() => {
    setConfigs(getAllConfigs());
  }, []);

  // 当股票代码或日期变化时，重新加载特征列表
  useEffect(() => {
    loadAvailableFeatures();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stockCodes, startDate, endDate]);

  const handleFeatureToggle = (feature: string, checked: boolean) => {
    if (checked) {
      onFeaturesChange([...selectedFeatures, feature]);
    } else {
      onFeaturesChange(selectedFeatures.filter(f => f !== feature));
    }
  };

  // Alpha因子分组和过滤
  const groupedAlphaFactors = useMemo(() => {
    if (featureCategories.alpha_features.length === 0) {
      return {};
    }

    // 过滤Alpha因子
    const filtered = alphaSearchTerm
      ? featureCategories.alpha_features.filter(
          f =>
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
    if (categoryFactors.length === 0) {
      return false;
    }
    return categoryFactors.every(f => selectedFeatures.includes(f));
  };

  // 检查某个类别的因子是否部分被选择
  const isCategoryPartiallySelected = (category: string): boolean => {
    const categoryFactors = groupedAlphaFactors[category] || [];
    if (categoryFactors.length === 0) {
      return false;
    }
    const selectedCount = categoryFactors.filter(f => selectedFeatures.includes(f)).length;
    return selectedCount > 0 && selectedCount < categoryFactors.length;
  };

  // 应用配置
  const handleApplyConfig = () => {
    if (!selectedConfigId) {
      return;
    }

    const config = getConfigById(selectedConfigId);
    if (!config) {
      return;
    }

    // 过滤出实际可用的特征
    const availableConfigFeatures = config.features.filter(f => availableFeatures.includes(f));

    if (availableConfigFeatures.length > 0) {
      onFeaturesChange(availableConfigFeatures);
      onUseAllFeaturesChange(false);
    } else {
      // 如果配置中的特征都不在可用列表中，直接使用配置的特征（可能还未加载）
      onFeaturesChange(config.features);
      onUseAllFeaturesChange(false);
    }
  };

  // 保存当前选择为配置
  const handleSaveConfig = () => {
    if (!saveConfigName.trim()) {
      alert('请输入配置名称');
      return;
    }

    if (selectedFeatures.length === 0) {
      alert('请先选择一些特征');
      return;
    }

    const newConfig = saveCustomConfig({
      name: saveConfigName.trim(),
      description: saveConfigDescription.trim() || '自定义特征配置',
      features: selectedFeatures,
    });

    setConfigs(getAllConfigs());
    setSelectedConfigId(newConfig.id);
    setSaveConfigName('');
    setSaveConfigDescription('');
    setIsSaveModalOpen(false);
  };

  // 删除自定义配置
  const handleDeleteConfig = (configId: string) => {
    if (!confirm('确定要删除这个配置吗？')) {
      return;
    }

    if (deleteCustomConfig(configId)) {
      setConfigs(getAllConfigs());
      if (selectedConfigId === configId) {
        setSelectedConfigId('');
      }
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 特征配置选择区域 */}
      <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1, border: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Settings size={16} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            特征配置
          </Typography>
        </Box>
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            alignItems: { xs: 'stretch', sm: 'flex-end' },
            gap: 1,
          }}
        >
          <FormControl size="small" sx={{ flex: 1 }}>
            <InputLabel>选择预设配置</InputLabel>
            <Select
              value={selectedConfigId}
              label="选择预设配置"
              onChange={e => setSelectedConfigId(e.target.value)}
            >
              {configs.map(config => (
                <MenuItem key={config.id} value={config.id}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                    }}
                  >
                    <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {config.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {config.description}
                      </Typography>
                    </Box>
                    {config.category === 'custom' && (
                      <IconButton
                        size="small"
                        color="error"
                        onClick={e => {
                          e.stopPropagation();
                          handleDeleteConfig(config.id);
                        }}
                        sx={{ ml: 1 }}
                      >
                        <Trash2 size={12} />
                      </IconButton>
                    )}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            size="small"
            variant="contained"
            color="primary"
            onClick={handleApplyConfig}
            disabled={!selectedConfigId}
          >
            应用配置
          </Button>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Save size={16} />}
            onClick={() => setIsSaveModalOpen(true)}
            disabled={selectedFeatures.length === 0}
          >
            保存为配置
          </Button>
        </Box>
      </Box>

      {/* 保存配置模态框 */}
      <Dialog
        open={isSaveModalOpen}
        onClose={() => setIsSaveModalOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>保存特征配置</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="配置名称"
              placeholder="请输入配置名称"
              value={saveConfigName}
              onChange={e => setSaveConfigName(e.target.value)}
              required
              autoFocus
              fullWidth
            />
            <TextField
              label="配置描述"
              placeholder="请输入配置描述（可选）"
              value={saveConfigDescription}
              onChange={e => setSaveConfigDescription(e.target.value)}
              fullWidth
            />
            <Typography variant="caption" color="text.secondary">
              当前已选择 {selectedFeatures.length} 个特征
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsSaveModalOpen(false)}>取消</Button>
          <Button variant="contained" color="primary" onClick={handleSaveConfig}>
            保存
          </Button>
        </DialogActions>
      </Dialog>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            特征选择
          </Typography>
          <Button
            size="small"
            variant="outlined"
            onClick={loadAvailableFeatures}
            disabled={loadingFeatures}
            startIcon={loadingFeatures ? <CircularProgress size={16} /> : undefined}
          >
            {loadingFeatures ? '加载中...' : '刷新特征列表'}
          </Button>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Checkbox
            checked={useAllFeatures}
            onChange={e => {
              onUseAllFeaturesChange(e.target.checked);
              if (e.target.checked) {
                onFeaturesChange([]);
              }
            }}
          />
          <Typography variant="body2">使用所有可用特征（推荐）</Typography>
        </Box>

        {!useAllFeatures && (
          <Box
            sx={{
              mt: 1,
              p: 2,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
              maxHeight: 384,
              overflowY: 'auto',
            }}
          >
            {loadingFeatures ? (
              <Box sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  加载特征列表中...
                </Typography>
              </Box>
            ) : availableFeatures.length === 0 &&
              Object.values(featureCategories).every(arr => arr.length === 0) ? (
              <Box sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  暂无可用特征
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mt: 1, display: 'block' }}
                >
                  请先选择股票和日期范围，或点击"刷新特征列表"按钮
                </Typography>
              </Box>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {/* 基础价格特征 */}
                {featureCategories.base_features.length > 0 && (
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                      基础价格特征
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {featureCategories.base_features.map(feature => (
                        <Box key={feature} sx={{ display: 'flex', alignItems: 'center' }}>
                          <Checkbox
                            size="small"
                            checked={selectedFeatures.includes(feature)}
                            onChange={e => handleFeatureToggle(feature, e.target.checked)}
                          />
                          <Typography variant="body2">{feature}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}

                {/* 技术指标特征 */}
                {featureCategories.indicator_features.length > 0 && (
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                      技术指标特征
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {featureCategories.indicator_features.map(feature => (
                        <Box key={feature} sx={{ display: 'flex', alignItems: 'center' }}>
                          <Checkbox
                            size="small"
                            checked={selectedFeatures.includes(feature)}
                            onChange={e => handleFeatureToggle(feature, e.target.checked)}
                          />
                          <Typography variant="body2">{feature}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}

                {/* 基本面特征 */}
                {featureCategories.fundamental_features.length > 0 && (
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                      基本面特征
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {featureCategories.fundamental_features.map(feature => (
                        <Box key={feature} sx={{ display: 'flex', alignItems: 'center' }}>
                          <Checkbox
                            size="small"
                            checked={selectedFeatures.includes(feature)}
                            onChange={e => handleFeatureToggle(feature, e.target.checked)}
                          />
                          <Typography variant="body2">{feature}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}

                {/* 如果所有分类都为空，显示所有特征 */}
                {Object.values(featureCategories).every(arr => arr.length === 0) &&
                  availableFeatures.length > 0 && (
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                        所有可用特征 ({availableFeatures.length})
                      </Typography>
                      <Box
                        sx={{
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: 1,
                          maxHeight: 256,
                          overflowY: 'auto',
                        }}
                      >
                        {availableFeatures.slice(0, 100).map(feature => (
                          <Box key={feature} sx={{ display: 'flex', alignItems: 'center' }}>
                            <Checkbox
                              size="small"
                              checked={selectedFeatures.includes(feature)}
                              onChange={e => handleFeatureToggle(feature, e.target.checked)}
                            />
                            <Typography variant="body2">{feature}</Typography>
                          </Box>
                        ))}
                        {availableFeatures.length > 100 && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ width: '100%' }}
                          >
                            还有 {availableFeatures.length - 100} 个特征...
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  )}

                {/* Alpha因子特征 - 优化显示 */}
                {featureCategories.alpha_features.length > 0 && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          Alpha因子特征 ({featureCategories.alpha_features.length}个)
                        </Typography>
                        <Tooltip title="Alpha因子是量化投资中用于预测股票未来收益率的量化特征，包括价格收益率、移动平均、波动率等多种类型">
                          <IconButton size="small">
                            <Info size={16} />
                          </IconButton>
                        </Tooltip>
                      </Box>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => {
                          // 切换所有Alpha因子的选择状态
                          const allAlphaSelected = featureCategories.alpha_features.every(f =>
                            selectedFeatures.includes(f)
                          );
                          if (allAlphaSelected) {
                            onFeaturesChange(
                              selectedFeatures.filter(
                                f => !featureCategories.alpha_features.includes(f)
                              )
                            );
                          } else {
                            const newFeatures = featureCategories.alpha_features.filter(
                              f => !selectedFeatures.includes(f)
                            );
                            onFeaturesChange([...selectedFeatures, ...newFeatures]);
                          }
                        }}
                      >
                        {featureCategories.alpha_features.every(f => selectedFeatures.includes(f))
                          ? '全不选'
                          : '全选'}
                      </Button>
                    </Box>

                    {/* 搜索框 */}
                    <TextField
                      size="small"
                      placeholder="搜索Alpha因子..."
                      value={alphaSearchTerm}
                      onChange={e => setAlphaSearchTerm(e.target.value)}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <Search size={16} />
                          </InputAdornment>
                        ),
                      }}
                      fullWidth
                    />

                    {/* 按类别分组的Alpha因子 */}
                    <Box
                      sx={{
                        maxHeight: 384,
                        overflowY: 'auto',
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        p: 1,
                      }}
                    >
                      {Object.keys(groupedAlphaFactors).length === 0 ? (
                        <Box sx={{ textAlign: 'center', py: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            {alphaSearchTerm ? '未找到匹配的Alpha因子' : '暂无Alpha因子'}
                          </Typography>
                        </Box>
                      ) : (
                        <Box>
                          {Object.entries(groupedAlphaFactors)
                            .sort(([a], [b]) => {
                              // 优先显示有名称的类别
                              const order = [
                                'MOMENTUM',
                                'MEAN_REVERSION',
                                'VOLATILITY',
                                'VOLUME',
                                'RESI',
                                'MA',
                                'STD',
                                'VSTD',
                                'CORR',
                                'MAX',
                                'MIN',
                                'QTLU',
                                'OTHER',
                              ];
                              const indexA = order.indexOf(a);
                              const indexB = order.indexOf(b);
                              if (indexA !== -1 && indexB !== -1) {
                                return indexA - indexB;
                              }
                              if (indexA !== -1) {
                                return -1;
                              }
                              if (indexB !== -1) {
                                return 1;
                              }
                              return a.localeCompare(b);
                            })
                            .map(([category, factors]) => {
                              const categoryInfo =
                                ALPHA_FACTOR_CATEGORIES[
                                  category as keyof typeof ALPHA_FACTOR_CATEGORIES
                                ] || ALPHA_FACTOR_CATEGORIES.OTHER;
                              const isFullySelected = isCategoryFullySelected(category);
                              const isPartiallySelected = isCategoryPartiallySelected(category);

                              return (
                                <Accordion
                                  key={category}
                                  expanded={expandedAlphaCategories.includes(category)}
                                  onChange={(e, expanded) => {
                                    if (expanded) {
                                      setExpandedAlphaCategories([
                                        ...expandedAlphaCategories,
                                        category,
                                      ]);
                                    } else {
                                      setExpandedAlphaCategories(
                                        expandedAlphaCategories.filter(c => c !== category)
                                      );
                                    }
                                  }}
                                >
                                  <AccordionSummary expandIcon={<ExpandMore />}>
                                    <Box
                                      sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        width: '100%',
                                        pr: 2,
                                      }}
                                    >
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <span>{categoryInfo.icon}</span>
                                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                          {categoryInfo.name}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                          ({factors.length}个)
                                        </Typography>
                                      </Box>
                                      <Checkbox
                                        size="small"
                                        checked={isFullySelected}
                                        indeterminate={isPartiallySelected}
                                        onChange={e => {
                                          e.stopPropagation();
                                          handleCategoryToggle(category, !isFullySelected);
                                        }}
                                      />
                                    </Box>
                                  </AccordionSummary>
                                  <AccordionDetails>
                                    <Box>
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{ mb: 1, display: 'block' }}
                                      >
                                        {categoryInfo.description}
                                      </Typography>
                                      <Box
                                        sx={{
                                          display: 'grid',
                                          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)' },
                                          gap: 1,
                                        }}
                                      >
                                        {factors.map(feature => {
                                          const info = parseAlphaFactor(feature);
                                          const displayName = getAlphaFactorDisplayName(feature);

                                          return (
                                            <Box
                                              key={feature}
                                              sx={{
                                                display: 'flex',
                                                alignItems: 'flex-start',
                                                gap: 0.5,
                                              }}
                                            >
                                              <Checkbox
                                                size="small"
                                                checked={selectedFeatures.includes(feature)}
                                                onChange={e =>
                                                  handleFeatureToggle(feature, e.target.checked)
                                                }
                                              />
                                              <Box
                                                sx={{
                                                  display: 'flex',
                                                  flexDirection: 'column',
                                                  flex: 1,
                                                }}
                                              >
                                                <Typography
                                                  variant="body2"
                                                  sx={{ fontWeight: 500 }}
                                                >
                                                  {displayName}
                                                </Typography>
                                                <Typography
                                                  variant="caption"
                                                  color="text.secondary"
                                                >
                                                  {feature}
                                                </Typography>
                                              </Box>
                                              {info.formula && (
                                                <Tooltip title={info.formula}>
                                                  <IconButton size="small">
                                                    <Info size={12} />
                                                  </IconButton>
                                                </Tooltip>
                                              )}
                                            </Box>
                                          );
                                        })}
                                      </Box>
                                    </Box>
                                  </AccordionDetails>
                                </Accordion>
                              );
                            })}
                        </Box>
                      )}
                    </Box>

                    {/* 已选择的Alpha因子统计 */}
                    {selectedFeatures.filter(f => featureCategories.alpha_features.includes(f))
                      .length > 0 && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ pt: 1, borderTop: 1, borderColor: 'divider', display: 'block' }}
                      >
                        已选择{' '}
                        <strong>
                          {
                            selectedFeatures.filter(f =>
                              featureCategories.alpha_features.includes(f)
                            ).length
                          }
                        </strong>{' '}
                        个Alpha因子
                      </Typography>
                    )}
                  </Box>
                )}

                {selectedFeatures.length > 0 && (
                  <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: 'divider' }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      已选择 <strong>{selectedFeatures.length}</strong> 个特征
                    </Typography>
                    <Button size="small" variant="outlined" onClick={() => onFeaturesChange([])}>
                      清空选择
                    </Button>
                  </Box>
                )}
              </Box>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
