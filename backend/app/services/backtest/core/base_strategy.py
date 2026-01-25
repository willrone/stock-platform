"""
策略基类

定义所有策略必须实现的接口
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from ..models import TradingSignal, Position, SignalType


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.indicators = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术指标"""
        pass
    
    def _extract_indicators_from_precomputed(
        self,
        data: pd.DataFrame,
        indicator_mapping: Dict[str, str]
    ) -> Optional[Dict[str, pd.Series]]:
        """
        从预计算数据中提取指标
        
        Args:
            data: 包含预计算指标的DataFrame
            indicator_mapping: 指标名称映射 {策略需要的指标名: 预计算数据中的列名}
                              例如: {'rsi': 'RSI14', 'ma20': 'MA20'}
        
        Returns:
            提取的指标字典，如果预计算数据不可用或缺少指标则返回None
        """
        # 检查数据是否来自预计算
        if not data.attrs.get('from_precomputed', False):
            return None
        
        # 检查所有需要的指标是否都存在
        missing_indicators = []
        for strategy_name, precomputed_name in indicator_mapping.items():
            if precomputed_name not in data.columns:
                missing_indicators.append(precomputed_name)
        
        if missing_indicators:
            # 部分指标缺失，返回None让策略fallback到现场计算
            return None
        
        # 提取指标
        indicators = {}
        for strategy_name, precomputed_name in indicator_mapping.items():
            indicators[strategy_name] = data[precomputed_name]
        
        # 确保基础数据也在indicators中
        if 'close' in data.columns:
            indicators['price'] = data['close']
        if 'volume' in data.columns:
            indicators['volume'] = data['volume']
        
        return indicators
    
    def validate_signal(self, signal: TradingSignal, portfolio_value: float, 
                       current_positions: Dict[str, Position]) -> Tuple[bool, Optional[str]]:
        """
        验证信号有效性
        
        Returns:
            tuple[bool, str | None]: (是否有效, 未执行原因)
            如果验证通过返回 (True, None)
            如果验证失败返回 (False, 失败原因)
        """
        # 基础验证
        if signal.strength < 0.1:  # 信号强度太低
            return False, f"信号强度过低: {signal.strength:.2%} < 10%"
        
        # 检查持仓限制
        if signal.signal_type == SignalType.BUY:
            current_position = current_positions.get(signal.stock_code)
            if current_position and portfolio_value > 0:
                position_ratio = current_position.market_value / portfolio_value
                if position_ratio > 0.3:
                    return False, f"单股持仓过大: {position_ratio:.2%} > 30%"
        
        return True, None
