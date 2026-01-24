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
