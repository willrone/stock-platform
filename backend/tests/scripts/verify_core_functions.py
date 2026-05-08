#!/usr/bin/env python3
"""
核心功能验证脚本
验证已实现的核心业务功能是否正常工作
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_feature_extraction():
    """测试特征提取功能"""
    print("🔍 测试特征提取功能...")

    try:
        from app.services.feature_extractor import FeatureExtractor

        # 创建测试数据
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, 100)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # 确保价格关系正确
        data["high"] = np.maximum(data["high"], data["close"])
        data["low"] = np.minimum(data["low"], data["close"])

        # 创建特征提取器
        extractor = FeatureExtractor(cache_enabled=False)

        # 提取特征
        features = extractor.extract_features("TEST.SZ", data)

        # 验证结果
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert len(features.columns) > 0

        print(
            f"✅ 特征提取成功: 提取了 {len(features.columns)} 个特征，{len(features)} 行数据"
        )
        return True

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return False


def test_prediction_engine():
    """测试预测引擎功能"""
    print("🔮 测试预测引擎功能...")

    try:
        from app.services.prediction_engine import PredictionConfig, PredictionEngine
        from app.services.prediction_fallback import PredictionErrorHandler

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建预测引擎
            engine = PredictionEngine(
                model_dir=os.path.join(temp_dir, "models"),
                data_dir=os.path.join(temp_dir, "data"),
            )

            # 创建测试数据目录
            data_dir = Path(temp_dir) / "data" / "daily" / "000001.SZ"
            data_dir.mkdir(parents=True, exist_ok=True)

            # 创建测试数据
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            np.random.seed(42)

            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

            test_data = pd.DataFrame(
                {
                    "open": prices * (1 + np.random.normal(0, 0.005, 100)),
                    "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                    "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, 100),
                },
                index=dates,
            )

            test_data["high"] = np.maximum(test_data["high"], test_data["close"])
            test_data["low"] = np.minimum(test_data["low"], test_data["close"])

            # 保存测试数据
            test_data.to_parquet(data_dir / "2024.parquet")

            # 创建预测配置
            config = PredictionConfig(
                model_id="test_model", horizon="short_term", confidence_level=0.95
            )

            # 验证输入参数
            is_valid = engine.validate_prediction_inputs("000001.SZ", config)
            assert is_valid is True

            # 执行预测（会使用备用模型）
            result = engine.predict_single_stock("000001.SZ", config)

            # 验证预测结果
            assert result.stock_code == "000001.SZ"
            assert result.predicted_price > 0
            assert result.predicted_direction in [-1, 0, 1]
            assert 0 <= result.confidence_score <= 1
            assert (
                result.confidence_interval[0]
                <= result.predicted_price
                <= result.confidence_interval[1]
            )

            print(
                f"✅ 预测引擎成功: 预测价格 {result.predicted_price:.2f}, 置信度 {result.confidence_score:.3f}"
            )

            # 测试错误处理
            error_handler = PredictionErrorHandler()
            test_error = Exception("测试错误")

            fallback_result = error_handler.handle_prediction_error(
                test_error, "000001.SZ", test_data
            )

            assert fallback_result["error_handled"] is True
            assert fallback_result["is_fallback"] is True

            print("✅ 错误处理和降级策略正常")
            return True

    except Exception as e:
        print(f"❌ 预测引擎测试失败: {e}")
        return False


def test_task_management():
    """测试任务管理功能"""
    print("📋 测试任务管理功能...")

    try:
        from app.models.task_models import Task, TaskStatus, TaskType
        from app.services.task_queue import TaskPriority, TaskScheduler
        from app.services.websocket_manager import WebSocketManager, WebSocketMessage

        # 测试任务模型
        task = Task(
            task_name="测试任务",
            task_type=TaskType.PREDICTION.value,
            user_id="test_user",
            config={"test": "config"},
        )

        assert task.task_id is not None
        assert task.task_name == "测试任务"
        assert task.status == TaskStatus.CREATED.value

        # 测试任务队列
        scheduler = TaskScheduler(max_executors=1)

        # 注册测试处理器
        test_results = []

        def test_handler(queued_task, context):
            result = {"task_id": queued_task.task_id, "success": True}
            test_results.append(result)
            return result

        scheduler.register_task_handler(TaskType.PREDICTION, test_handler)
        scheduler.start()

        # 入队测试任务
        success = scheduler.enqueue_task(
            task_id="test_task_001",
            task_type=TaskType.PREDICTION,
            config={"test": "config"},
            user_id="test_user",
            priority=TaskPriority.NORMAL,
        )

        assert success is True

        # 等待任务执行
        import time

        time.sleep(2)

        # 验证任务执行
        assert len(test_results) == 1
        assert test_results[0]["task_id"] == "test_task_001"

        scheduler.stop()

        # 测试WebSocket管理器
        WebSocketManager()

        # 测试消息格式
        message = WebSocketMessage(
            type="task_status", data={"task_id": "test_task", "status": "completed"}
        )

        json_message = message.to_json()
        assert isinstance(json_message, str)
        assert "task_status" in json_message

        print("✅ 任务管理功能正常")
        return True

    except Exception as e:
        print(f"❌ 任务管理测试失败: {e}")
        return False


def test_backtest_engine():
    """测试回测引擎功能"""
    print("📈 测试回测引擎功能...")

    try:
        from app.services.backtest_engine import (
            BacktestConfig,
            PortfolioManager,
            SignalType,
            StrategyFactory,
            TradingSignal,
        )

        # 创建测试数据
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

        test_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, 100)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        test_data["high"] = np.maximum(test_data["high"], test_data["close"])
        test_data["low"] = np.minimum(test_data["low"], test_data["close"])
        test_data.attrs["stock_code"] = "TEST.SZ"

        # 测试策略工厂
        available_strategies = StrategyFactory.get_available_strategies()
        assert "moving_average" in available_strategies
        assert "rsi" in available_strategies
        assert "macd" in available_strategies

        # 创建移动平均策略
        strategy = StrategyFactory.create_strategy(
            "moving_average",
            {"short_window": 5, "long_window": 20, "signal_threshold": 0.02},
        )

        assert strategy.name == "MovingAverage"

        # 计算指标
        indicators = strategy.calculate_indicators(test_data)
        assert isinstance(indicators, dict)
        assert "sma_short" in indicators
        assert "sma_long" in indicators

        # 生成信号
        current_date = test_data.index[50]
        signals = strategy.generate_signals(test_data, current_date)

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.signal_type in [
                SignalType.BUY,
                SignalType.SELL,
                SignalType.HOLD,
            ]

        # 测试组合管理器
        config = BacktestConfig(initial_cash=100000)
        portfolio_manager = PortfolioManager(config)

        assert portfolio_manager.cash == 100000
        assert len(portfolio_manager.positions) == 0

        # 测试组合价值计算
        current_prices = {"TEST.SZ": 100.0}
        portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
        assert portfolio_value == 100000  # 只有现金，没有持仓

        print("✅ 回测引擎功能正常")
        return True

    except Exception as e:
        print(f"❌ 回测引擎测试失败: {e}")
        return False


def test_risk_assessment():
    """测试风险评估功能"""
    print("⚠️ 测试风险评估功能...")

    try:
        from app.services.risk_assessment import (
            ConfidenceIntervalCalculator,
            RiskAssessmentService,
            RiskMetricsCalculator,
        )

        # 创建测试数据
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

        historical_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, 100)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # 测试置信区间计算
        calculator = ConfidenceIntervalCalculator()

        interval = calculator.parametric_interval(
            predicted_price=100.0, volatility=0.02, confidence_level=0.95
        )

        assert interval.lower_bound <= 100.0 <= interval.upper_bound
        assert interval.confidence_level == 0.95
        assert interval.method == "parametric"

        # 测试风险指标计算
        returns = historical_data["close"].pct_change().dropna()

        risk_calculator = RiskMetricsCalculator()
        var_results = risk_calculator.calculate_var(returns, [0.95])

        assert isinstance(var_results, dict)
        assert 0.95 in var_results

        # 测试风险评估服务
        risk_service = RiskAssessmentService()

        result = risk_service.assess_prediction_risk(
            stock_code="TEST.SZ",
            current_price=100.0,
            predicted_price=105.0,
            historical_data=historical_data,
        )

        assert result.stock_code == "TEST.SZ"
        assert result.current_price == 100.0
        assert result.predicted_price == 105.0
        assert result.risk_rating in ["low", "medium", "high", "extreme"]
        assert len(result.confidence_intervals) > 0

        print("✅ 风险评估功能正常")
        return True

    except Exception as e:
        print(f"❌ 风险评估测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始验证核心业务功能...")
    print("=" * 50)

    test_results = []

    # 执行各项测试
    test_results.append(("特征提取", test_feature_extraction()))
    test_results.append(("预测引擎", test_prediction_engine()))
    test_results.append(("任务管理", test_task_management()))
    test_results.append(("回测引擎", test_backtest_engine()))
    test_results.append(("风险评估", test_risk_assessment()))

    print("=" * 50)
    print("📊 测试结果汇总:")

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("=" * 50)
    print(f"🎯 总结: {passed} 个测试通过, {failed} 个测试失败")

    if failed == 0:
        print("🎉 所有核心功能验证通过！系统已准备好进入下一阶段开发。")
        return 0
    else:
        print("⚠️ 部分功能存在问题，需要修复后再继续。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
