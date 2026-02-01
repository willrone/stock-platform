"""
信号整合器单元测试
"""

import pytest
from datetime import datetime
from app.services.backtest.models import TradingSignal, SignalType
from app.services.backtest.utils.signal_integrator import SignalIntegrator


class TestSignalIntegrator:
    """信号整合器测试类"""
    
    def test_weighted_voting_buy_signals(self):
        """测试加权投票 - 买入信号"""
        integrator = SignalIntegrator(method="weighted_voting")
        
        # 创建多个买入信号
        signals = [
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.8,
                price=10.0,
                reason="RSI超卖",
                metadata={"strategy_name": "rsi"}
            ),
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.6,
                price=10.0,
                reason="MACD金叉",
                metadata={"strategy_name": "macd"}
            )
        ]
        
        weights = {"rsi": 0.6, "macd": 0.4}
        integrated = integrator.integrate(signals, weights)
        
        assert len(integrated) == 1
        assert integrated[0].signal_type == SignalType.BUY
        assert integrated[0].stock_code == "000001.SZ"
        assert 0 <= integrated[0].strength <= 1.0
        assert "rsi" in integrated[0].metadata["source_signals"][0]["strategy"]
        assert "macd" in integrated[0].metadata["source_signals"][1]["strategy"]
    
    def test_weighted_voting_conflicting_signals(self):
        """测试加权投票 - 冲突信号（买入vs卖出）"""
        integrator = SignalIntegrator(method="weighted_voting")
        
        signals = [
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.7,
                price=10.0,
                reason="RSI超卖",
                metadata={"strategy_name": "rsi"}
            ),
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.SELL,
                strength=0.8,
                price=10.0,
                reason="MACD死叉",
                metadata={"strategy_name": "macd"}
            )
        ]
        
        weights = {"rsi": 0.5, "macd": 0.5}
        integrated = integrator.integrate(signals, weights)
        
        # 应该生成一个信号（卖出信号，因为MACD权重更高）
        assert len(integrated) == 1
        assert integrated[0].signal_type == SignalType.SELL
        # 冲突时信号强度应该降低
        assert integrated[0].strength < 0.8
    
    def test_consistency_enhancement(self):
        """测试一致性增强"""
        integrator = SignalIntegrator(method="weighted_voting")
        
        # 创建3个同向信号（高一致性）
        signals = [
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.6,
                price=10.0,
                reason=f"策略{i}",
                metadata={"strategy_name": f"strategy_{i}"}
            )
            for i in range(3)
        ]
        
        weights = {f"strategy_{i}": 1.0/3 for i in range(3)}
        integrated = integrator.integrate(signals, weights, consistency_threshold=0.6)
        
        assert len(integrated) == 1
        assert integrated[0].signal_type == SignalType.BUY
        # 一致性高时应该增强信号强度
        assert integrated[0].metadata["consistency"] >= 0.6
    
    def test_empty_signals(self):
        """测试空信号列表"""
        integrator = SignalIntegrator(method="weighted_voting")
        integrated = integrator.integrate([], {})
        assert integrated == []
    
    def test_weight_normalization(self):
        """测试权重归一化"""
        integrator = SignalIntegrator(method="weighted_voting")
        
        signals = [
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.5,
                price=10.0,
                reason="测试",
                metadata={"strategy_name": "rsi"}
            )
        ]
        
        # 权重之和不为1，应该自动归一化
        weights = {"rsi": 2.0, "macd": 1.0}  # 总和为3
        integrated = integrator.integrate(signals, weights)
        
        assert len(integrated) == 1
    
    def test_multiple_stocks(self):
        """测试多股票信号整合"""
        integrator = SignalIntegrator(method="weighted_voting")
        
        signals = [
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.7,
                price=10.0,
                reason="RSI",
                metadata={"strategy_name": "rsi"}
            ),
            TradingSignal(
                timestamp=datetime(2023, 1, 1),
                stock_code="000002.SZ",
                signal_type=SignalType.BUY,
                strength=0.8,
                price=20.0,
                reason="MACD",
                metadata={"strategy_name": "macd"}
            )
        ]
        
        weights = {"rsi": 0.5, "macd": 0.5}
        integrated = integrator.integrate(signals, weights)
        
        # 应该为每只股票生成一个信号
        assert len(integrated) == 2
        stock_codes = {s.stock_code for s in integrated}
        assert "000001.SZ" in stock_codes
        assert "000002.SZ" in stock_codes
