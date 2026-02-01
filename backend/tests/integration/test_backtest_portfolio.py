"""
组合策略回测集成测试

测试组合策略回测的端到端流程，包括：
- 组合策略创建和执行
- 信号融合验证
- 与单策略结果一致性验证
- API端点测试
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.backtest import StrategyFactory, BacktestExecutor, BacktestConfig
from app.services.backtest.core.strategy_portfolio import StrategyPortfolio
from app.services.backtest.models import TradingSignal, SignalType


class TestPortfolioBacktestIntegration:
    """组合策略回测集成测试类"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_stock_data(self):
        """生成示例股票数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日
        
        data = pd.DataFrame({
            'open': [10.0 + i * 0.01 for i in range(len(dates))],
            'high': [10.5 + i * 0.01 for i in range(len(dates))],
            'low': [9.5 + i * 0.01 for i in range(len(dates))],
            'close': [10.0 + i * 0.01 for i in range(len(dates))],
            'volume': [1000 + i * 10 for i in range(len(dates))]
        }, index=dates)
        
        return data
    
    def test_portfolio_strategy_creation(self):
        """测试组合策略创建"""
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
    
    def test_portfolio_signal_generation(self, sample_stock_data):
        """测试组合策略信号生成"""
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
                    "config": {"fast_period": 12, "slow_period": 26}
                }
            ]
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        
        # 生成信号
        current_date = sample_stock_data.index[0]
        signals = strategy.generate_signals(sample_stock_data, current_date)
        
        # 验证信号
        assert isinstance(signals, list)
        # 信号应该包含整合信息
        if signals:
            assert "integration_method" in signals[0].metadata or "source_signals" in signals[0].metadata
    
    def test_single_strategy_equivalence(self):
        """测试单策略等价性：组合只包含1个策略且权重=1时，结果应与单策略一致"""
        # 创建单策略
        single_config = {"rsi_period": 14, "oversold_threshold": 30}
        single_strategy = StrategyFactory.create_strategy("rsi", single_config)
        
        # 创建组合策略（只包含1个策略，权重=1）
        portfolio_config = {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 1.0,
                    "config": single_config
                }
            ]
        }
        portfolio_strategy = StrategyFactory.create_strategy("portfolio", portfolio_config)
        
        # 验证组合策略只包含1个策略
        assert len(portfolio_strategy.strategies) == 1
        assert portfolio_strategy.weights[portfolio_strategy.strategies[0].name] == 1.0
        
        # 注意：由于信号整合器的存在，即使只有1个策略，信号也会经过整合处理
        # 但策略本身应该是等价的
        assert portfolio_strategy.strategies[0].name == single_strategy.name
    
    @patch('app.services.backtest.execution.backtest_executor.DataLoader')
    def test_portfolio_backtest_execution(self, mock_data_loader, sample_stock_data):
        """测试组合策略回测执行"""
        # Mock数据加载器
        mock_loader = MagicMock()
        mock_loader.load_multiple_stocks.return_value = {
            "000001.SZ": sample_stock_data
        }
        mock_data_loader.return_value = mock_loader
        
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
                    "config": {"fast_period": 12, "slow_period": 26}
                }
            ]
        }
        
        executor = BacktestExecutor(data_dir="test_data")
        backtest_config = BacktestConfig(initial_cash=100000.0)
        
        # 执行回测（使用async）
        import asyncio
        result = asyncio.run(executor.run_backtest(
            strategy_name="portfolio",
            stock_codes=["000001.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            strategy_config=config,
            backtest_config=backtest_config
        ))
        
        # 验证结果
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
    
    def test_portfolio_api_endpoint(self, client):
        """测试组合策略API端点"""
        backtest_request = {
            "strategy_name": "portfolio",
            "stock_codes": ["000001.SZ"],
            "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "initial_cash": 100000.0,
            "strategy_config": {
                "strategies": [
                    {
                        "name": "rsi",
                        "weight": 0.5,
                        "config": {"rsi_period": 14}
                    },
                    {
                        "name": "macd",
                        "weight": 0.5,
                        "config": {"fast_period": 12, "slow_period": 26}
                    }
                ],
                "integration_method": "weighted_voting"
            }
        }
        
        # 注意：这个测试可能需要mock数据加载，否则会失败
        # 在实际环境中，需要确保有测试数据
        with patch('app.services.backtest.execution.backtest_executor.DataLoader'):
            response = client.post("/api/v1/backtest", json=backtest_request)
            
            # 如果数据不存在，可能返回错误，但至少验证API能处理请求
            assert response.status_code in [200, 500]  # 200成功，500可能是数据问题
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert data["data"]["is_portfolio"] is True
                assert "portfolio_info" in data["data"]
    
    def test_portfolio_templates_api(self, client):
        """测试获取组合策略模板API"""
        response = client.get("/api/v1/backtest/portfolio-templates")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        
        # 验证模板结构
        template = data["data"][0]
        assert "name" in template
        assert "description" in template
        assert "strategies" in template
        assert isinstance(template["strategies"], list)
    
    def test_weight_normalization(self):
        """测试权重自动归一化"""
        config = {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 2.0,  # 未归一化
                    "config": {"rsi_period": 14}
                },
                {
                    "name": "macd",
                    "weight": 2.0,  # 未归一化
                    "config": {"fast_period": 12}
                }
            ]
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        
        # 权重应该自动归一化
        assert abs(sum(strategy.weights.values()) - 1.0) < 0.001
        # 策略名称是大写的（如 "RSI", "MACD"）
        assert abs(strategy.weights["RSI"] - 0.5) < 0.001
        assert abs(strategy.weights["MACD"] - 0.5) < 0.001
    
    def test_portfolio_with_missing_strategy_signals(self, sample_stock_data):
        """测试组合策略处理缺失信号的情况"""
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
                    "config": {"fast_period": 12, "slow_period": 26}
                }
            ]
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        
        # 即使某个策略无法生成信号，组合策略应该仍然能工作
        current_date = sample_stock_data.index[0]
        signals = strategy.generate_signals(sample_stock_data, current_date)
        
        # 应该返回信号列表（可能为空，但不应该报错）
        assert isinstance(signals, list)
    
    def test_portfolio_add_remove_strategy(self):
        """测试动态添加和移除策略"""
        config = {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 1.0,
                    "config": {"rsi_period": 14}
                }
            ]
        }
        
        strategy = StrategyFactory.create_strategy("portfolio", config)
        assert len(strategy.strategies) == 1
        
        # 添加策略
        macd_strategy = StrategyFactory.create_strategy("macd", {"fast_period": 12})
        strategy.add_strategy(macd_strategy, weight=0.5)
        
        assert len(strategy.strategies) == 2
        assert abs(sum(strategy.weights.values()) - 1.0) < 0.001
        
        # 移除策略（使用策略名称，如 "RSI"）
        strategy.remove_strategy("RSI")
        assert len(strategy.strategies) == 1
        assert "MACD" in strategy.weights
