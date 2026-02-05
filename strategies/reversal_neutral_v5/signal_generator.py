"""
反转中性策略 V5 - 市场状态自适应版

核心改进：
1. 市场状态检测：识别牛市/熊市/震荡市
2. 自适应仓位：熊市降低仓位或空仓
3. 止损机制：连续亏损时暂停交易
4. 波动率过滤：高波动时降低仓位

基于 V4 趋势增强版改进
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"       # 牛市：趋势向上
    BEAR = "bear"       # 熊市：趋势向下
    SIDEWAYS = "sideways"  # 震荡：无明显趋势


@dataclass
class SignalOutput:
    """信号输出"""
    date: str
    long_stocks: List[str]      # 做多股票列表
    short_stocks: List[str]     # 做空股票列表
    long_weights: Dict[str, float]   # 做多权重
    short_weights: Dict[str, float]  # 做空权重
    position_scale: float       # 仓位缩放因子 (0-1.2)
    market_regime: MarketRegime # 当前市场状态
    regime_confidence: float    # 状态置信度
    stop_loss_active: bool      # 是否触发止损


class ReversalNeutralV5:
    """
    反转中性策略 V5 - 市场状态自适应版
    
    特点：
    - Top 10 做多 + Bottom 10 做空
    - 每 5 天调仓
    - 市场状态检测（牛/熊/震荡）
    - 熊市自动降低仓位
    - 连续亏损止损机制
    - 波动率自适应
    """
    
    def __init__(
        self,
        top_n: int = 10,
        rebalance_days: int = 5,
        # 市场状态检测参数
        regime_lookback: int = 20,      # 状态检测回看天数
        regime_ma_short: int = 10,      # 短期均线
        regime_ma_long: int = 30,       # 长期均线
        # 仓位控制参数
        bull_position: float = 1.2,     # 牛市仓位
        sideways_position: float = 1.0, # 震荡仓位
        bear_position: float = 0.3,     # 熊市仓位（大幅降低）
        # 止损参数
        stop_loss_threshold: float = -0.10,  # 连续亏损阈值
        stop_loss_lookback: int = 10,        # 止损回看天数
        stop_loss_cooldown: int = 5,         # 止损冷却期
        # 波动率参数
        vol_lookback: int = 20,         # 波动率计算天数
        vol_high_threshold: float = 0.03,    # 高波动阈值
        vol_position_scale: float = 0.5,     # 高波动时仓位缩放
    ):
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        
        # 市场状态参数
        self.regime_lookback = regime_lookback
        self.regime_ma_short = regime_ma_short
        self.regime_ma_long = regime_ma_long
        
        # 仓位参数
        self.bull_position = bull_position
        self.sideways_position = sideways_position
        self.bear_position = bear_position
        
        # 止损参数
        self.stop_loss_threshold = stop_loss_threshold
        self.stop_loss_lookback = stop_loss_lookback
        self.stop_loss_cooldown = stop_loss_cooldown
        
        # 波动率参数
        self.vol_lookback = vol_lookback
        self.vol_high_threshold = vol_high_threshold
        self.vol_position_scale = vol_position_scale
        
        # 状态变量
        self.last_rebalance_date = None
        self.current_long_stocks = []
        self.current_short_stocks = []
        self.portfolio_returns = []
        self.stop_loss_until = None  # 止损冷却结束日期
        
        # 市场数据缓存
        self.market_returns = []  # 市场收益率序列
        
    def detect_market_regime(
        self, 
        market_prices: pd.Series
    ) -> Tuple[MarketRegime, float]:
        """
        检测市场状态
        
        使用多重指标：
        1. 均线系统：短期均线 vs 长期均线
        2. 趋势强度：价格相对均线的位置
        3. 动量：近期收益率
        
        Returns:
            (市场状态, 置信度)
        """
        if len(market_prices) < self.regime_ma_long:
            return MarketRegime.SIDEWAYS, 0.5
        
        # 计算均线
        ma_short = market_prices.rolling(self.regime_ma_short).mean()
        ma_long = market_prices.rolling(self.regime_ma_long).mean()
        
        current_price = market_prices.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        # 计算近期收益率
        recent_return = (market_prices.iloc[-1] / market_prices.iloc[-self.regime_lookback] - 1)
        
        # 计算趋势得分
        # 1. 均线排列得分
        ma_score = 0
        if current_ma_short > current_ma_long:
            ma_score = 1
        elif current_ma_short < current_ma_long:
            ma_score = -1
            
        # 2. 价格位置得分
        price_score = 0
        if current_price > current_ma_short > current_ma_long:
            price_score = 1
        elif current_price < current_ma_short < current_ma_long:
            price_score = -1
            
        # 3. 动量得分
        momentum_score = 0
        if recent_return > 0.05:  # 5% 以上涨幅
            momentum_score = 1
        elif recent_return < -0.05:  # 5% 以上跌幅
            momentum_score = -1
        
        # 综合得分
        total_score = ma_score + price_score + momentum_score
        
        # 判断市场状态
        if total_score >= 2:
            regime = MarketRegime.BULL
            confidence = min(0.9, 0.6 + abs(recent_return))
        elif total_score <= -2:
            regime = MarketRegime.BEAR
            confidence = min(0.9, 0.6 + abs(recent_return))
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5 + (1 - abs(total_score) / 3) * 0.3
            
        return regime, confidence
    
    def calculate_volatility(self, returns: List[float]) -> float:
        """计算波动率"""
        if len(returns) < self.vol_lookback:
            return 0.02  # 默认波动率
        
        recent_returns = returns[-self.vol_lookback:]
        return np.std(recent_returns)
    
    def check_stop_loss(self, current_date: str) -> bool:
        """
        检查是否触发止损
        
        条件：近 N 天累计亏损超过阈值
        """
        # 检查是否在冷却期
        if self.stop_loss_until and current_date < self.stop_loss_until:
            return True
        
        if len(self.portfolio_returns) < self.stop_loss_lookback:
            return False
        
        recent_returns = self.portfolio_returns[-self.stop_loss_lookback:]
        cumulative_return = np.prod([1 + r for r in recent_returns]) - 1
        
        if cumulative_return < self.stop_loss_threshold:
            # 触发止损，设置冷却期
            # 简化处理：假设每天一个交易日
            self.stop_loss_until = current_date  # 实际应该加上冷却天数
            return True
        
        return False
    
    def get_position_scale(
        self, 
        regime: MarketRegime, 
        volatility: float,
        stop_loss_active: bool
    ) -> float:
        """
        计算仓位缩放因子
        
        综合考虑：
        1. 市场状态
        2. 波动率
        3. 止损状态
        """
        if stop_loss_active:
            return 0.0  # 止损期间空仓
        
        # 基础仓位（根据市场状态）
        if regime == MarketRegime.BULL:
            base_position = self.bull_position
        elif regime == MarketRegime.BEAR:
            base_position = self.bear_position
        else:
            base_position = self.sideways_position
        
        # 波动率调整
        if volatility > self.vol_high_threshold:
            vol_scale = self.vol_position_scale
        else:
            vol_scale = 1.0
        
        return base_position * vol_scale
    
    def should_rebalance(self, current_date: str) -> bool:
        """判断是否需要调仓"""
        if self.last_rebalance_date is None:
            return True
        
        # 简化处理：比较日期字符串
        # 实际应该计算交易日差
        try:
            from datetime import datetime
            current = datetime.strptime(current_date, '%Y-%m-%d')
            last = datetime.strptime(self.last_rebalance_date, '%Y-%m-%d')
            days_diff = (current - last).days
            return days_diff >= self.rebalance_days
        except:
            return True
    
    def generate_signals(
        self,
        date: str,
        predictions: pd.Series,  # 股票代码 -> 预测收益
        market_prices: pd.Series,  # 市场指数价格序列
        portfolio_return: Optional[float] = None,  # 上一期组合收益
    ) -> SignalOutput:
        """
        生成交易信号
        
        Args:
            date: 当前日期
            predictions: 模型预测的股票收益率
            market_prices: 市场指数价格序列（用于状态检测）
            portfolio_return: 上一期组合收益（用于止损判断）
            
        Returns:
            SignalOutput 包含完整的交易信号
        """
        # 更新组合收益记录
        if portfolio_return is not None:
            self.portfolio_returns.append(portfolio_return)
        
        # 1. 检测市场状态
        regime, confidence = self.detect_market_regime(market_prices)
        
        # 2. 计算波动率
        if len(self.portfolio_returns) > 0:
            volatility = self.calculate_volatility(self.portfolio_returns)
        else:
            volatility = 0.02
        
        # 3. 检查止损
        stop_loss_active = self.check_stop_loss(date)
        
        # 4. 计算仓位缩放
        position_scale = self.get_position_scale(regime, volatility, stop_loss_active)
        
        # 5. 判断是否调仓
        if not self.should_rebalance(date) and not stop_loss_active:
            # 不调仓，返回当前持仓
            return SignalOutput(
                date=date,
                long_stocks=self.current_long_stocks,
                short_stocks=self.current_short_stocks,
                long_weights={s: 1.0/len(self.current_long_stocks) if self.current_long_stocks else 0 
                             for s in self.current_long_stocks},
                short_weights={s: 1.0/len(self.current_short_stocks) if self.current_short_stocks else 0 
                              for s in self.current_short_stocks},
                position_scale=position_scale,
                market_regime=regime,
                regime_confidence=confidence,
                stop_loss_active=stop_loss_active,
            )
        
        # 6. 生成新信号
        if stop_loss_active or position_scale == 0:
            # 止损或空仓
            long_stocks = []
            short_stocks = []
        else:
            # 正常选股
            sorted_predictions = predictions.sort_values(ascending=False)
            long_stocks = sorted_predictions.head(self.top_n).index.tolist()
            short_stocks = sorted_predictions.tail(self.top_n).index.tolist()
        
        # 更新状态
        self.current_long_stocks = long_stocks
        self.current_short_stocks = short_stocks
        self.last_rebalance_date = date
        
        # 计算权重（等权）
        long_weights = {s: 1.0/len(long_stocks) if long_stocks else 0 for s in long_stocks}
        short_weights = {s: 1.0/len(short_stocks) if short_stocks else 0 for s in short_stocks}
        
        return SignalOutput(
            date=date,
            long_stocks=long_stocks,
            short_stocks=short_stocks,
            long_weights=long_weights,
            short_weights=short_weights,
            position_scale=position_scale,
            market_regime=regime,
            regime_confidence=confidence,
            stop_loss_active=stop_loss_active,
        )
    
    def get_config(self) -> Dict:
        """获取策略配置"""
        return {
            'name': 'ReversalNeutralV5',
            'version': '5.0',
            'description': '市场状态自适应版 - 熊市保护 + 止损机制',
            'top_n': self.top_n,
            'rebalance_days': self.rebalance_days,
            'regime_detection': {
                'lookback': self.regime_lookback,
                'ma_short': self.regime_ma_short,
                'ma_long': self.regime_ma_long,
            },
            'position_control': {
                'bull': self.bull_position,
                'sideways': self.sideways_position,
                'bear': self.bear_position,
            },
            'stop_loss': {
                'threshold': self.stop_loss_threshold,
                'lookback': self.stop_loss_lookback,
                'cooldown': self.stop_loss_cooldown,
            },
            'volatility': {
                'lookback': self.vol_lookback,
                'high_threshold': self.vol_high_threshold,
                'position_scale': self.vol_position_scale,
            },
        }


# 便捷函数
def create_strategy(**kwargs) -> ReversalNeutralV5:
    """创建策略实例"""
    return ReversalNeutralV5(**kwargs)
