/**
 * 特征配置管理
 * 
 * 提供预设的特征组合配置，以及自定义配置的保存和加载功能
 */

export interface FeatureConfig {
  id: string;
  name: string;
  description: string;
  features: string[];
  category: 'preset' | 'custom';
  createdAt?: string;
  updatedAt?: string;
}

// 预设特征配置
export const PRESET_FEATURE_CONFIGS: FeatureConfig[] = [
  {
    id: 'basic',
    name: '基础配置',
    description: '仅包含基础价格和成交量特征，适合简单模型',
    features: ['$open', '$high', '$low', '$close', '$volume'],
    category: 'preset',
  },
  {
    id: 'technical',
    name: '技术指标配置',
    description: '基础特征 + 常用技术指标（MA、RSI、MACD、Bollinger Bands等）',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 移动平均
      'MA5', 'MA10', 'MA20', 'MA60',
      // 动量指标
      'RSI14', 'STOCH_K', 'STOCH_D', 'WILLIAMS_R', 'CCI20', 'MOMENTUM', 'ROC',
      // 趋势指标
      'MACD', 'MACD_SIGNAL', 'MACD_HIST',
      // 布林带
      'BOLL_UPPER', 'BOLL_MIDDLE', 'BOLL_LOWER',
      // 成交量指标
      'VWAP', 'OBV', 'VOLUME_RSI',
      // 波动率
      'ATR14', 'VOLATILITY',
    ],
    category: 'preset',
  },
  {
    id: 'momentum',
    name: '动量策略配置',
    description: '专注于动量因子，适合趋势跟踪策略',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 移动平均
      'MA5', 'MA10', 'MA20',
      // 动量指标
      'RSI14', 'MOMENTUM', 'ROC',
      // MACD
      'MACD', 'MACD_SIGNAL', 'MACD_HIST',
      // 价格变化
      'RET1', 'RET5', 'RET20',
      // Alpha动量因子（前30个）
      ...Array.from({ length: 30 }, (_, i) => `alpha_${String(i + 1).padStart(3, '0')}`),
    ],
    category: 'preset',
  },
  {
    id: 'mean_reversion',
    name: '均值回归配置',
    description: '专注于均值回归因子，适合反转策略',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 移动平均
      'MA5', 'MA10', 'MA20', 'MA60',
      // 布林带
      'BOLL_UPPER', 'BOLL_MIDDLE', 'BOLL_LOWER',
      // 价格位置
      'PRICE_POSITION',
      // 波动率
      'VOLATILITY', 'VOLATILITY5', 'VOLATILITY20',
      // 价格变化
      'RET1', 'RET5', 'RET20',
      // Alpha均值回归因子（61-100）
      ...Array.from({ length: 40 }, (_, i) => `alpha_${String(i + 61).padStart(3, '0')}`),
    ],
    category: 'preset',
  },
  {
    id: 'volatility',
    name: '波动率策略配置',
    description: '专注于波动率因子，适合风险管理和波动率交易',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 波动率指标
      'ATR14', 'VOLATILITY', 'HISTORICAL_VOLATILITY', 'VOLATILITY5', 'VOLATILITY20',
      // 布林带
      'BOLL_UPPER', 'BOLL_MIDDLE', 'BOLL_LOWER',
      // 成交量波动
      'VOLUME_RET1', 'VOLUME_MA_RATIO',
      // Alpha波动率因子（101-140）
      ...Array.from({ length: 40 }, (_, i) => `alpha_${String(i + 101).padStart(3, '0')}`),
    ],
    category: 'preset',
  },
  {
    id: 'volume',
    name: '成交量策略配置',
    description: '专注于成交量因子，适合量价分析策略',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 成交量指标
      'VWAP', 'OBV', 'AD_LINE', 'VOLUME_RSI', 'VOLUME_RET1', 'VOLUME_MA_RATIO',
      // 量价相关性
      'CORR5', 'CORR10', 'CORR20', 'CORR30',
      // Alpha成交量因子（141-158）
      ...Array.from({ length: 18 }, (_, i) => `alpha_${String(i + 141).padStart(3, '0')}`),
    ],
    category: 'preset',
  },
  {
    id: 'comprehensive',
    name: '综合配置',
    description: '包含基础特征、技术指标和精选Alpha因子，平衡性能和效果',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 移动平均
      'MA5', 'MA10', 'MA20', 'MA60',
      // 动量指标
      'RSI14', 'STOCH_K', 'WILLIAMS_R', 'CCI20', 'MOMENTUM', 'ROC',
      // 趋势指标
      'MACD', 'MACD_SIGNAL', 'MACD_HIST',
      // 布林带
      'BOLL_UPPER', 'BOLL_MIDDLE', 'BOLL_LOWER',
      // 成交量指标
      'VWAP', 'OBV', 'VOLUME_RSI',
      // 波动率
      'ATR14', 'VOLATILITY', 'VOLATILITY5', 'VOLATILITY20',
      // 基本面特征
      'RET1', 'RET5', 'RET20', 'VOLUME_RET1', 'VOLUME_MA_RATIO', 'PRICE_POSITION',
      // 精选Alpha因子（每个类别选前20个）
      ...Array.from({ length: 20 }, (_, i) => `alpha_${String(i + 1).padStart(3, '0')}`), // 动量
      ...Array.from({ length: 20 }, (_, i) => `alpha_${String(i + 61).padStart(3, '0')}`), // 均值回归
      ...Array.from({ length: 20 }, (_, i) => `alpha_${String(i + 101).padStart(3, '0')}`), // 波动率
      ...Array.from({ length: 18 }, (_, i) => `alpha_${String(i + 141).padStart(3, '0')}`), // 成交量
    ],
    category: 'preset',
  },
  {
    id: 'minimal',
    name: '精简配置',
    description: '精选的高效特征组合，适合快速训练和测试',
    features: [
      // 基础特征
      '$open', '$high', '$low', '$close', '$volume',
      // 核心移动平均
      'MA5', 'MA20',
      // 核心动量指标
      'RSI14', 'MACD',
      // 核心波动率
      'VOLATILITY',
      // 价格变化
      'RET1', 'RET5',
      // 精选Alpha因子（每个类别选前5个）
      ...Array.from({ length: 5 }, (_, i) => `alpha_${String(i + 1).padStart(3, '0')}`),
      ...Array.from({ length: 5 }, (_, i) => `alpha_${String(i + 61).padStart(3, '0')}`),
      ...Array.from({ length: 5 }, (_, i) => `alpha_${String(i + 101).padStart(3, '0')}`),
      ...Array.from({ length: 3 }, (_, i) => `alpha_${String(i + 141).padStart(3, '0')}`),
    ],
    category: 'preset',
  },
];

// localStorage键名
const STORAGE_KEY = 'feature_configs_custom';

/**
 * 获取所有自定义配置
 */
export function getCustomConfigs(): FeatureConfig[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('加载自定义配置失败:', error);
    return [];
  }
}

/**
 * 保存自定义配置
 */
export function saveCustomConfig(config: Omit<FeatureConfig, 'id' | 'category' | 'createdAt' | 'updatedAt'>): FeatureConfig {
  const customConfigs = getCustomConfigs();
  const newConfig: FeatureConfig = {
    ...config,
    id: `custom_${Date.now()}`,
    category: 'custom',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  
  customConfigs.push(newConfig);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(customConfigs));
  
  return newConfig;
}

/**
 * 更新自定义配置
 */
export function updateCustomConfig(configId: string, updates: Partial<Omit<FeatureConfig, 'id' | 'category'>>): boolean {
  const customConfigs = getCustomConfigs();
  const index = customConfigs.findIndex(c => c.id === configId);
  
  if (index === -1) return false;
  
  customConfigs[index] = {
    ...customConfigs[index],
    ...updates,
    updatedAt: new Date().toISOString(),
  };
  
  localStorage.setItem(STORAGE_KEY, JSON.stringify(customConfigs));
  return true;
}

/**
 * 删除自定义配置
 */
export function deleteCustomConfig(configId: string): boolean {
  const customConfigs = getCustomConfigs();
  const filtered = customConfigs.filter(c => c.id !== configId);
  
  if (filtered.length === customConfigs.length) return false;
  
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * 根据ID获取配置
 */
export function getConfigById(configId: string): FeatureConfig | undefined {
  // 先查找预设配置
  const preset = PRESET_FEATURE_CONFIGS.find(c => c.id === configId);
  if (preset) return preset;
  
  // 再查找自定义配置
  const customConfigs = getCustomConfigs();
  return customConfigs.find(c => c.id === configId);
}

/**
 * 获取所有配置（预设 + 自定义）
 */
export function getAllConfigs(): FeatureConfig[] {
  return [...PRESET_FEATURE_CONFIGS, ...getCustomConfigs()];
}

