/**
 * 格式化工具函数
 */

export const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency',
    currency: 'CNY',
    minimumFractionDigits: 2,
  }).format(value);
};

export const formatPercent = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};

export const formatNumber = (value: number, decimals: number = 2): string => {
  return value.toFixed(decimals);
};

export const formatLargeNumber = (value: number): string => {
  if (value >= 10000) {
    return `¥${(value / 10000).toFixed(1)}万`;
  }
  return `¥${value.toFixed(0)}`;
};
