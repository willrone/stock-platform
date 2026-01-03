"""
月度收益分析模块
提供详细的月度、季度、年度绩效分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from calendar import monthrange
from loguru import logger

from app.core.error_handler import TaskError, ErrorSeverity


class MonthlyAnalyzer:
    """月度分析器"""
    
    def __init__(self):
        self.month_names = {
            1: '一月', 2: '二月', 3: '三月', 4: '四月',
            5: '五月', 6: '六月', 7: '七月', 8: '八月',
            9: '九月', 10: '十月', 11: '十一月', 12: '十二月'
        }
        
        self.quarter_names = {
            1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'
        }
    
    async def analyze_monthly_performance(
        self, 
        portfolio_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        分析月度绩效表现
        
        Args:
            portfolio_history: 组合历史数据
            
        Returns:
            月度分析结果
        """
        try:
            if not portfolio_history:
                logger.warning("组合历史数据为空，无法进行月度分析")
                return {}
            
            # 转换为DataFrame
            df = pd.DataFrame(portfolio_history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 月度收益分析
            monthly_returns = self._calculate_monthly_returns(df)
            
            # 季度收益分析
            quarterly_returns = self._calculate_quarterly_returns(df)
            
            # 年度收益分析
            yearly_returns = self._calculate_yearly_returns(df)
            
            # 月度统计分析
            monthly_stats = self._calculate_monthly_statistics(monthly_returns)
            
            # 季节性分析
            seasonal_analysis = self._analyze_seasonality(monthly_returns)
            
            # 热力图数据
            heatmap_data = self._prepare_heatmap_data(monthly_returns)
            
            result = {
                'monthly_returns': monthly_returns,
                'quarterly_returns': quarterly_returns,
                'yearly_returns': yearly_returns,
                'monthly_statistics': monthly_stats,
                'seasonal_analysis': seasonal_analysis,
                'heatmap_data': heatmap_data
            }
            
            logger.info(f"月度分析完成，覆盖 {len(monthly_returns)} 个月")
            return result
            
        except Exception as e:
            logger.error(f"月度绩效分析失败: {e}", exc_info=True)
            raise TaskError(
                message=f"月度绩效分析失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e
            )
    
    def _calculate_monthly_returns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """计算月度收益"""
        
        # 按月重采样，取每月最后一个交易日的组合价值
        monthly_values = df['portfolio_value'].resample('ME').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        result = []
        cumulative_return = 0
        
        for date, monthly_return in monthly_returns.items():
            cumulative_return = (1 + cumulative_return) * (1 + monthly_return) - 1
            
            # 计算月度波动率（如果有足够的日数据）
            month_start = date.replace(day=1)
            month_end = date
            month_data = df[month_start:month_end]['portfolio_value']
            
            if len(month_data) > 1:
                daily_returns = month_data.pct_change().dropna()
                monthly_volatility = daily_returns.std() * np.sqrt(len(daily_returns))
            else:
                monthly_volatility = 0
            
            month_info = {
                'year': date.year,
                'month': date.month,
                'month_name': self.month_names[date.month],
                'date': date.strftime('%Y-%m'),
                'monthly_return': float(monthly_return),
                'cumulative_return': float(cumulative_return),
                'monthly_volatility': float(monthly_volatility),
                'portfolio_value': float(monthly_values[date]),
                'trading_days': len(month_data)
            }
            
            result.append(month_info)
        
        return result
    
    def _calculate_quarterly_returns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """计算季度收益"""
        
        # 按季度重采样
        quarterly_values = df['portfolio_value'].resample('QE').last()
        quarterly_returns = quarterly_values.pct_change().dropna()
        
        result = []
        cumulative_return = 0
        
        for date, quarterly_return in quarterly_returns.items():
            cumulative_return = (1 + cumulative_return) * (1 + quarterly_return) - 1
            
            quarter = (date.month - 1) // 3 + 1
            
            quarter_info = {
                'year': date.year,
                'quarter': quarter,
                'quarter_name': f"{date.year}{self.quarter_names[quarter]}",
                'date': date.strftime('%Y-Q%s') % quarter,
                'quarterly_return': float(quarterly_return),
                'cumulative_return': float(cumulative_return),
                'portfolio_value': float(quarterly_values[date])
            }
            
            result.append(quarter_info)
        
        return result
    
    def _calculate_yearly_returns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """计算年度收益"""
        
        # 按年重采样
        yearly_values = df['portfolio_value'].resample('YE').last()
        yearly_returns = yearly_values.pct_change().dropna()
        
        result = []
        
        for date, yearly_return in yearly_returns.items():
            # 计算年度统计
            year_start = date.replace(month=1, day=1)
            year_end = date
            year_data = df[year_start:year_end]['portfolio_value']
            
            if len(year_data) > 1:
                daily_returns = year_data.pct_change().dropna()
                yearly_volatility = daily_returns.std() * np.sqrt(252)
                yearly_sharpe = (yearly_return - 0.03) / yearly_volatility if yearly_volatility > 0 else 0
                
                # 计算年度最大回撤
                cumulative = (1 + daily_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
            else:
                yearly_volatility = 0
                yearly_sharpe = 0
                max_drawdown = 0
            
            year_info = {
                'year': date.year,
                'yearly_return': float(yearly_return),
                'yearly_volatility': float(yearly_volatility),
                'yearly_sharpe': float(yearly_sharpe),
                'max_drawdown': float(max_drawdown),
                'portfolio_value': float(yearly_values[date]),
                'trading_days': len(year_data)
            }
            
            result.append(year_info)
        
        return result
    
    def _calculate_monthly_statistics(self, monthly_returns: List[Dict]) -> Dict[str, Any]:
        """计算月度统计指标"""
        
        if not monthly_returns:
            return {}
        
        returns = [m['monthly_return'] for m in monthly_returns]
        
        # 基础统计
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        median_return = np.median(returns)
        
        # 正负月份统计
        positive_months = [r for r in returns if r > 0]
        negative_months = [r for r in returns if r < 0]
        
        positive_count = len(positive_months)
        negative_count = len(negative_months)
        total_months = len(returns)
        
        positive_ratio = positive_count / total_months if total_months > 0 else 0
        
        # 最佳和最差月份
        best_month = max(returns) if returns else 0
        worst_month = min(returns) if returns else 0
        
        # 连续盈利/亏损月份
        max_consecutive_positive = self._calculate_max_consecutive_months(returns, lambda x: x > 0)
        max_consecutive_negative = self._calculate_max_consecutive_months(returns, lambda x: x < 0)
        
        return {
            'total_months': total_months,
            'mean_monthly_return': float(mean_return),
            'median_monthly_return': float(median_return),
            'monthly_volatility': float(std_return),
            'positive_months': positive_count,
            'negative_months': negative_count,
            'positive_ratio': float(positive_ratio),
            'best_month': float(best_month),
            'worst_month': float(worst_month),
            'max_consecutive_positive': max_consecutive_positive,
            'max_consecutive_negative': max_consecutive_negative,
            'avg_positive_return': float(np.mean(positive_months)) if positive_months else 0,
            'avg_negative_return': float(np.mean(negative_months)) if negative_months else 0
        }
    
    def _calculate_max_consecutive_months(self, returns: List[float], condition) -> int:
        """计算最大连续月份数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if condition(ret):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _analyze_seasonality(self, monthly_returns: List[Dict]) -> Dict[str, Any]:
        """分析季节性模式"""
        
        if not monthly_returns:
            return {}
        
        # 按月份分组
        month_groups = {}
        for month_data in monthly_returns:
            month = month_data['month']
            if month not in month_groups:
                month_groups[month] = []
            month_groups[month].append(month_data['monthly_return'])
        
        # 计算每个月份的平均收益
        monthly_averages = {}
        for month, returns in month_groups.items():
            monthly_averages[month] = {
                'month': month,
                'month_name': self.month_names[month],
                'avg_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)),
                'count': len(returns),
                'positive_ratio': len([r for r in returns if r > 0]) / len(returns)
            }
        
        # 按季度分组
        quarter_groups = {1: [], 2: [], 3: [], 4: []}
        for month_data in monthly_returns:
            quarter = (month_data['month'] - 1) // 3 + 1
            quarter_groups[quarter].append(month_data['monthly_return'])
        
        # 计算每个季度的平均收益
        quarterly_averages = {}
        for quarter, returns in quarter_groups.items():
            if returns:
                quarterly_averages[quarter] = {
                    'quarter': quarter,
                    'quarter_name': self.quarter_names[quarter],
                    'avg_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'count': len(returns),
                    'positive_ratio': len([r for r in returns if r > 0]) / len(returns)
                }
        
        # 找出最佳和最差的月份/季度
        best_month = max(monthly_averages.values(), key=lambda x: x['avg_return']) if monthly_averages else None
        worst_month = min(monthly_averages.values(), key=lambda x: x['avg_return']) if monthly_averages else None
        
        best_quarter = max(quarterly_averages.values(), key=lambda x: x['avg_return']) if quarterly_averages else None
        worst_quarter = min(quarterly_averages.values(), key=lambda x: x['avg_return']) if quarterly_averages else None
        
        return {
            'monthly_averages': list(monthly_averages.values()),
            'quarterly_averages': list(quarterly_averages.values()),
            'best_month': best_month,
            'worst_month': worst_month,
            'best_quarter': best_quarter,
            'worst_quarter': worst_quarter
        }
    
    def _prepare_heatmap_data(self, monthly_returns: List[Dict]) -> List[Dict[str, Any]]:
        """准备热力图数据"""
        
        if not monthly_returns:
            return []
        
        # 创建年份-月份矩阵
        heatmap_data = []
        
        for month_data in monthly_returns:
            heatmap_point = {
                'year': month_data['year'],
                'month': month_data['month'],
                'month_name': month_data['month_name'],
                'return': month_data['monthly_return'],
                'return_percent': month_data['monthly_return'] * 100,
                'date': month_data['date']
            }
            heatmap_data.append(heatmap_point)
        
        return heatmap_data
    
    async def calculate_rolling_monthly_metrics(
        self, 
        portfolio_history: List[Dict], 
        window_months: int = 12
    ) -> Dict[str, Any]:
        """
        计算滚动月度指标
        
        Args:
            portfolio_history: 组合历史数据
            window_months: 滚动窗口月数
            
        Returns:
            滚动月度指标
        """
        try:
            if not portfolio_history:
                return {}
            
            # 转换为DataFrame
            df = pd.DataFrame(portfolio_history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 计算月度收益
            monthly_values = df['portfolio_value'].resample('ME').last()
            monthly_returns = monthly_values.pct_change().dropna()
            
            if len(monthly_returns) < window_months:
                logger.warning(f"月度数据不足 {len(monthly_returns)}，无法计算 {window_months} 个月的滚动指标")
                return {}
            
            # 滚动年化收益率
            rolling_annual_return = monthly_returns.rolling(window_months).apply(
                lambda x: (1 + x).prod() ** (12 / window_months) - 1, raw=False
            )
            
            # 滚动波动率
            rolling_volatility = monthly_returns.rolling(window_months).std() * np.sqrt(12)
            
            # 滚动夏普比率
            rolling_sharpe = (rolling_annual_return - 0.03) / rolling_volatility
            
            # 滚动最大回撤
            rolling_max_dd = monthly_returns.rolling(window_months).apply(
                lambda x: self._calculate_rolling_max_drawdown(x), raw=False
            )
            
            # 组织结果
            rolling_data = []
            for date in rolling_annual_return.index:
                if pd.notna(rolling_annual_return[date]):
                    rolling_point = {
                        'date': date.strftime('%Y-%m'),
                        'annual_return': float(rolling_annual_return[date]),
                        'volatility': float(rolling_volatility[date]),
                        'sharpe_ratio': float(rolling_sharpe[date]) if pd.notna(rolling_sharpe[date]) else 0,
                        'max_drawdown': float(rolling_max_dd[date]) if pd.notna(rolling_max_dd[date]) else 0
                    }
                    rolling_data.append(rolling_point)
            
            return {
                'window_months': window_months,
                'rolling_metrics': rolling_data,
                'summary': {
                    'avg_annual_return': float(rolling_annual_return.mean()),
                    'avg_volatility': float(rolling_volatility.mean()),
                    'avg_sharpe_ratio': float(rolling_sharpe.mean()),
                    'avg_max_drawdown': float(rolling_max_dd.mean())
                }
            }
            
        except Exception as e:
            logger.error(f"计算滚动月度指标失败: {e}", exc_info=True)
            return {}
    
    def _calculate_rolling_max_drawdown(self, returns: pd.Series) -> float:
        """计算滚动最大回撤"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0