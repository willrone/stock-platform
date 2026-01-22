"""
策略工厂组合策略创建测试
"""

import pytest
from app.services.backtest.strategies.strategy_factory import StrategyFactory
from app.services.backtest.core.strategy_portfolio import StrategyPortfolio
from app.core.error_handler import TaskError


class TestStrategyFactoryPortfolio:
    """策略工厂组合策略测试类"""
    
    def test_create_portfolio_strategy(self):
        """测试创建组合策略"""
        config = {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 0.4,
                    "config": {
                        "rsi_period": 14,
                        "oversold_threshold": 30,
                        "overbought_threshold": 70
                    }
                },
                {
                    "name": "macd",
                    "weight": 0.3,
                    "config": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                },
                {
                    "name": "bollinger",
                    "weight": 0.3,
                    "config": {
                        "period": 20,
                        "std_dev": 2
                    }
                }
            ],
            "integration_method": "weighted_voting"
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        
        assert isinstance(strategy, StrategyPortfolio)
        assert len(strategy.strategies) == 3
        assert abs(sum(strategy.weights.values()) - 1.0) < 0.001
    
    def test_create_portfolio_by_config(self):
        """测试通过配置中的strategies字段自动识别组合策略"""
        config = {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 0.5,
                    "config": {"rsi_period": 14}
                },
                {
                    "name": "macd",
                    "weight": 0.5,
                    "config": {"fast_period": 12}
                }
            ]
        }
        
        # 即使strategy_name不是"portfolio"，只要config中有strategies，也应该创建组合策略
        strategy = StrategyFactory.create_strategy("any_name", config)
        
        assert isinstance(strategy, StrategyPortfolio)
        assert len(strategy.strategies) == 2
    
    def test_portfolio_weight_normalization(self):
        """测试组合策略权重自动归一化"""
        config = {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 2.0,  # 未归一化的权重
                    "config": {"rsi_period": 14}
                },
                {
                    "name": "macd",
                    "weight": 2.0,  # 未归一化的权重
                    "config": {"fast_period": 12}
                }
            ]
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        
        # 权重应该自动归一化
        assert abs(sum(strategy.weights.values()) - 1.0) < 0.001
        assert abs(strategy.weights["rsi"] - 0.5) < 0.001
        assert abs(strategy.weights["macd"] - 0.5) < 0.001
    
    def test_portfolio_default_weight(self):
        """测试组合策略默认权重"""
        config = {
            "strategies": [
                {
                    "name": "rsi",
                    "config": {"rsi_period": 14}  # 没有指定weight
                },
                {
                    "name": "macd",
                    "config": {"fast_period": 12}  # 没有指定weight
                }
            ]
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        
        # 应该使用默认权重（平均分配）
        assert abs(sum(strategy.weights.values()) - 1.0) < 0.001
        for weight in strategy.weights.values():
            assert abs(weight - 0.5) < 0.001
    
    def test_portfolio_missing_strategies_error(self):
        """测试缺少strategies字段的错误"""
        config = {}
        
        with pytest.raises(TaskError, match="必须包含'strategies'字段"):
            StrategyFactory.create_strategy("portfolio", config)
    
    def test_portfolio_empty_strategies_error(self):
        """测试空strategies列表的错误"""
        config = {
            "strategies": []
        }
        
        with pytest.raises(TaskError, match="必须是非空列表"):
            StrategyFactory.create_strategy("portfolio", config)
    
    def test_portfolio_invalid_strategy_config(self):
        """测试无效策略配置"""
        config = {
            "strategies": [
                {
                    # 缺少name字段
                    "weight": 0.5,
                    "config": {}
                }
            ]
        }
        
        with pytest.raises(TaskError, match="必须包含'name'字段"):
            StrategyFactory.create_strategy("portfolio", config)
    
    def test_portfolio_invalid_strategy_name(self):
        """测试无效策略名称"""
        config = {
            "strategies": [
                {
                    "name": "invalid_strategy",
                    "weight": 1.0,
                    "config": {}
                }
            ]
        }
        
        with pytest.raises(TaskError, match="创建策略.*失败"):
            StrategyFactory.create_strategy("portfolio", config)
    
    def test_single_strategy_still_works(self):
        """测试单策略创建仍然正常工作"""
        config = {
            "rsi_period": 14,
            "oversold_threshold": 30
        }
        
        strategy = StrategyFactory.create_strategy("rsi", config)
        
        # 应该创建单策略，不是组合策略
        assert not isinstance(strategy, StrategyPortfolio)
        assert strategy.name == "rsi"
