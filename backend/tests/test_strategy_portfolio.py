"""
策略组合类单元测试
"""

import pytest
import pandas as pd
from datetime import datetime
from app.services.backtest.core.strategy_portfolio import StrategyPortfolio
from app.services.backtest.core.base_strategy import BaseStrategy
from app.services.backtest.models import TradingSignal, SignalType
from app.services.backtest.strategies.strategy_factory import StrategyFactory


class MockStrategy(BaseStrategy):
    """模拟策略用于测试"""
    
    def __init__(self, name: str, signal_type: SignalType, strength: float):
        super().__init__(name, {})
        self.signal_type = signal_type
        self.strength = strength
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime):
        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=data.index[0] if len(data) > 0 else "000001.SZ",
                signal_type=self.signal_type,
                strength=self.strength,
                price=10.0,
                reason=f"{self.name}信号"
            )
        ]
    
    def calculate_indicators(self, data: pd.DataFrame):
        return {f"{self.name}_indicator": pd.Series([1, 2, 3])}


class TestStrategyPortfolio:
    """策略组合测试类"""
    
    def test_create_portfolio(self):
        """测试创建策略组合"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7),
            MockStrategy("macd", SignalType.BUY, 0.8)
        ]
        
        portfolio = StrategyPortfolio(
            strategies=strategies,
            weights={"rsi": 0.6, "macd": 0.4}
        )
        
        assert portfolio.name.startswith("Portfolio")
        assert len(portfolio.strategies) == 2
        assert portfolio.weights["rsi"] == 0.6
        assert portfolio.weights["macd"] == 0.4
    
    def test_default_weights(self):
        """测试默认权重（平均分配）"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7),
            MockStrategy("macd", SignalType.BUY, 0.8),
            MockStrategy("bollinger", SignalType.BUY, 0.6)
        ]
        
        portfolio = StrategyPortfolio(strategies=strategies)
        
        # 权重应该归一化且平均分配
        assert abs(sum(portfolio.weights.values()) - 1.0) < 0.001
        for weight in portfolio.weights.values():
            assert abs(weight - 1.0/3) < 0.001
    
    def test_weight_normalization(self):
        """测试权重自动归一化"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7),
            MockStrategy("macd", SignalType.BUY, 0.8)
        ]
        
        portfolio = StrategyPortfolio(
            strategies=strategies,
            weights={"rsi": 2.0, "macd": 2.0}  # 总和为4，应该归一化为0.5, 0.5
        )
        
        assert abs(sum(portfolio.weights.values()) - 1.0) < 0.001
        assert abs(portfolio.weights["rsi"] - 0.5) < 0.001
        assert abs(portfolio.weights["macd"] - 0.5) < 0.001
    
    def test_generate_signals(self):
        """测试生成组合信号"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7),
            MockStrategy("macd", SignalType.BUY, 0.8)
        ]
        
        portfolio = StrategyPortfolio(
            strategies=strategies,
            weights={"rsi": 0.5, "macd": 0.5}
        )
        
        data = pd.DataFrame({
            "close": [10.0, 11.0, 12.0],
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "volume": [1000, 1100, 1200]
        }, index=[datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)])
        
        signals = portfolio.generate_signals(data, datetime(2023, 1, 1))
        
        # 应该生成整合后的信号
        assert len(signals) > 0
        # 信号应该包含来源信息
        if signals:
            assert "source_signals" in signals[0].metadata or "integration_method" in signals[0].metadata
    
    def test_add_strategy(self):
        """测试动态添加策略"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7)
        ]
        
        portfolio = StrategyPortfolio(strategies=strategies)
        assert len(portfolio.strategies) == 1
        
        portfolio.add_strategy(MockStrategy("macd", SignalType.BUY, 0.8), weight=0.5)
        assert len(portfolio.strategies) == 2
        assert abs(sum(portfolio.weights.values()) - 1.0) < 0.001
    
    def test_remove_strategy(self):
        """测试移除策略"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7),
            MockStrategy("macd", SignalType.BUY, 0.8)
        ]
        
        portfolio = StrategyPortfolio(strategies=strategies)
        assert len(portfolio.strategies) == 2
        
        portfolio.remove_strategy("rsi")
        assert len(portfolio.strategies) == 1
        assert "macd" in portfolio.weights
        assert "rsi" not in portfolio.weights
    
    def test_empty_strategies_error(self):
        """测试空策略列表错误"""
        with pytest.raises(ValueError, match="策略列表不能为空"):
            StrategyPortfolio(strategies=[])
    
    def test_invalid_weights_error(self):
        """测试无效权重错误"""
        strategies = [MockStrategy("rsi", SignalType.BUY, 0.7)]
        
        with pytest.raises(ValueError, match="权重不能为负"):
            StrategyPortfolio(strategies=strategies, weights={"rsi": -0.5})
    
    def test_calculate_indicators(self):
        """测试计算指标"""
        strategies = [
            MockStrategy("rsi", SignalType.BUY, 0.7),
            MockStrategy("macd", SignalType.BUY, 0.8)
        ]
        
        portfolio = StrategyPortfolio(strategies=strategies)
        
        data = pd.DataFrame({
            "close": [10.0, 11.0, 12.0],
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "volume": [1000, 1100, 1200]
        })
        
        indicators = portfolio.calculate_indicators(data)
        
        # 应该包含所有子策略的指标（带前缀）
        assert "rsi_indicator" in indicators or "rsi_rsi_indicator" in indicators
        assert len(indicators) >= 2
