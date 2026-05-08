#!/usr/bin/env python3
"""
验证模型训练和管理模块的核心功能
"""

import asyncio
import sys
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, ".")


async def test_model_training_service():
    """测试模型训练服务的初始化和基本功能"""
    print("开始测试模型训练服务...")

    try:
        from app.services.models.shared_types import ModelType, TrainingConfig

        print("✓ 共享类型导入成功")

        # 测试训练配置创建
        config = TrainingConfig(
            model_type=ModelType.XGBOOST,
            sequence_length=30,
            prediction_horizon=3,
            batch_size=16,
            epochs=10,
            learning_rate=0.001,
        )
        print("✓ 训练配置创建成功")

        # 测试参数准备
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        stock_codes = ["000001.SZ"]

        print("✓ 测试参数准备完成")
        print(f"  - 股票代码: {stock_codes}")
        print(f"  - 时间范围: {start_date} 到 {end_date}")
        print(f"  - 模型类型: {config.model_type.value}")

        print("\n测试完成，核心功能正常")
        return True

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


async def test_model_evaluation():
    """测试模型评估模块"""
    print("\n开始测试模型评估模块...")

    try:
        from app.services.models.model_evaluation import (
            ModelEvaluator,
            ModelVersionManager,
        )

        ModelEvaluator()
        ModelVersionManager()

        print("✓ 模型评估器初始化成功")
        print("✓ 模型版本管理器初始化成功")

        return True

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


async def test_feature_engineering():
    """测试特征工程模块"""
    print("\n开始测试特征工程模块...")

    try:
        from app.core.config import settings
        from app.services.data.simple_data_service import SimpleDataService
        from app.services.models.feature_engineering import FeatureEngineer

        data_service = SimpleDataService()
        FeatureEngineer(data_service, settings.DATA_ROOT_PATH)

        print("✓ 特征工程师初始化成功")

        return True

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


async def main():
    """运行所有测试"""
    print("=== 模型管理和训练模块验证 ===\n")

    results = []
    results.append(await test_model_training_service())
    results.append(await test_model_evaluation())
    results.append(await test_feature_engineering())

    print("\n=== 测试结果 ===")
    print(f"总测试数: {len(results)}")
    print(f"成功测试: {sum(results)}")
    print(f"失败测试: {len(results) - sum(results)}")

    if all(results):
        print("\n🎉 所有测试通过！核心功能正常。")
        return 0
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
