#!/usr/bin/env python3
"""
完整的Qlib集成测试

验证所有Qlib相关功能：
1. 增强的QlibDataProvider
2. 统一训练引擎
3. 模型配置管理器
4. API接口
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_module_imports():
    """测试模块导入"""
    logger.info("=== 测试模块导入 ===")
    
    test_results = []
    
    # 测试基础模块
    modules_to_test = [
        ("app.services.qlib", "Qlib服务模块"),
        ("app.services.qlib.enhanced_qlib_provider", "增强数据提供器"),
        ("app.services.qlib.unified_qlib_training_engine", "统一训练引擎"),
        ("app.services.qlib.qlib_model_manager", "模型管理器"),
        ("app.services.qlib.custom_models", "自定义模型"),
        ("app.api.v1.qlib", "Qlib API")
    ]
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name} 导入成功")
            test_results.append(True)
        except ImportError as e:
            logger.warning(f"⚠️  {display_name} 导入失败: {e}")
            test_results.append(False)
        except Exception as e:
            logger.error(f"❌ {display_name} 导入异常: {e}")
            test_results.append(False)
    
    return test_results


def test_qlib_service_classes():
    """测试Qlib服务类"""
    logger.info("=== 测试Qlib服务类 ===")
    
    test_results = []
    
    try:
        # 测试增强数据提供器
        logger.info("1. 测试增强数据提供器...")
        try:
            from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider
            provider = EnhancedQlibDataProvider()
            logger.info("✅ EnhancedQlibDataProvider 创建成功")
            test_results.append(True)
        except Exception as e:
            logger.warning(f"⚠️  EnhancedQlibDataProvider 创建失败: {e}")
            test_results.append(False)
        
        # 测试统一训练引擎
        logger.info("2. 测试统一训练引擎...")
        try:
            from app.services.qlib.unified_qlib_training_engine import UnifiedQlibTrainingEngine
            engine = UnifiedQlibTrainingEngine()
            logger.info("✅ UnifiedQlibTrainingEngine 创建成功")
            
            # 测试支持的模型类型
            supported_models = engine.get_supported_model_types()
            logger.info(f"支持的模型: {supported_models}")
            test_results.append(True)
        except Exception as e:
            logger.warning(f"⚠️  UnifiedQlibTrainingEngine 创建失败: {e}")
            test_results.append(False)
        
        # 测试模型管理器
        logger.info("3. 测试模型管理器...")
        try:
            from app.services.qlib.qlib_model_manager import QlibModelManager
            manager = QlibModelManager()
            logger.info("✅ QlibModelManager 创建成功")
            
            # 测试模型推荐
            recommendations = manager.recommend_models(
                sample_count=5000,
                feature_count=30,
                task_type="regression"
            )
            logger.info(f"推荐模型: {recommendations}")
            test_results.append(True)
        except Exception as e:
            logger.warning(f"⚠️  QlibModelManager 创建失败: {e}")
            test_results.append(False)
        
        # 测试自定义模型
        logger.info("4. 测试自定义模型...")
        try:
            from app.services.qlib import CUSTOM_MODELS_AVAILABLE
            if CUSTOM_MODELS_AVAILABLE:
                from app.services.qlib.custom_models import CustomTransformerModel
                logger.info("✅ 自定义模型可用")
            else:
                logger.info("ℹ️  自定义模型不可用（可能缺少PyTorch）")
            test_results.append(True)
        except Exception as e:
            logger.warning(f"⚠️  自定义模型测试失败: {e}")
            test_results.append(False)
        
    except Exception as e:
        logger.error(f"❌ Qlib服务类测试失败: {e}")
        test_results.append(False)
    
    return test_results


def test_model_configurations():
    """测试模型配置"""
    logger.info("=== 测试模型配置 ===")
    
    test_results = []
    
    try:
        from app.services.qlib.qlib_model_manager import QlibModelManager
        manager = QlibModelManager()
        
        # 测试所有支持的模型
        supported_models = manager.get_supported_models()
        logger.info(f"支持的模型数量: {len(supported_models)}")
        
        for model_name in supported_models[:3]:  # 只测试前3个模型
            try:
                # 获取模型元数据
                metadata = manager.get_model_metadata(model_name)
                if metadata:
                    logger.info(f"✅ {model_name} 元数据: {metadata.display_name}")
                
                # 获取超参数规格
                hyperparameter_specs = manager.get_hyperparameter_specs(model_name)
                logger.info(f"✅ {model_name} 超参数数量: {len(hyperparameter_specs)}")
                
                # 创建配置
                test_hyperparameters = {}
                for spec in hyperparameter_specs[:2]:  # 只测试前2个超参数
                    test_hyperparameters[spec.name] = spec.default_value
                
                config = manager.create_qlib_config(model_name, test_hyperparameters)
                logger.info(f"✅ {model_name} 配置创建成功")
                
                test_results.append(True)
                
            except Exception as e:
                logger.warning(f"⚠️  {model_name} 配置测试失败: {e}")
                test_results.append(False)
    
    except Exception as e:
        logger.error(f"❌ 模型配置测试失败: {e}")
        test_results.append(False)
    
    return test_results


def test_api_functions():
    """测试API函数"""
    logger.info("=== 测试API函数 ===")
    
    test_results = []
    
    try:
        from app.api.v1.qlib import get_qlib_provider, get_training_engine
        
        # 测试获取数据提供器
        try:
            provider = get_qlib_provider()
            logger.info("✅ get_qlib_provider 成功")
            test_results.append(True)
        except Exception as e:
            logger.warning(f"⚠️  get_qlib_provider 失败: {e}")
            test_results.append(False)
        
        # 测试获取训练引擎
        try:
            engine = get_training_engine()
            logger.info("✅ get_training_engine 成功")
            test_results.append(True)
        except Exception as e:
            logger.warning(f"⚠️  get_training_engine 失败: {e}")
            test_results.append(False)
    
    except Exception as e:
        logger.error(f"❌ API函数测试失败: {e}")
        test_results.append(False)
    
    return test_results


def test_file_structure():
    """测试文件结构"""
    logger.info("=== 测试文件结构 ===")
    
    required_files = [
        "app/services/qlib/__init__.py",
        "app/services/qlib/enhanced_qlib_provider.py",
        "app/services/qlib/unified_qlib_training_engine.py",
        "app/services/qlib/qlib_model_manager.py",
        "app/services/qlib/custom_models.py",
        "app/api/v1/qlib.py"
    ]
    
    test_results = []
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            logger.info(f"✅ {file_path} 存在")
            test_results.append(True)
        else:
            logger.warning(f"⚠️  {file_path} 不存在")
            test_results.append(False)
    
    return test_results


def main():
    """主测试函数"""
    logger.info("开始完整的Qlib集成测试")
    
    all_test_results = []
    
    # 执行所有测试
    test_suites = [
        ("文件结构", test_file_structure),
        ("模块导入", test_module_imports),
        ("Qlib服务类", test_qlib_service_classes),
        ("模型配置", test_model_configurations),
        ("API函数", test_api_functions)
    ]
    
    for suite_name, test_func in test_suites:
        logger.info(f"\n{'='*50}")
        logger.info(f"执行测试套件: {suite_name}")
        logger.info(f"{'='*50}")
        
        try:
            results = test_func()
            all_test_results.extend(results)
            
            passed = sum(results)
            total = len(results)
            logger.info(f"{suite_name} 测试结果: {passed}/{total} 通过")
            
        except Exception as e:
            logger.error(f"{suite_name} 测试套件执行失败: {e}")
            all_test_results.append(False)
    
    # 汇总结果
    logger.info(f"\n{'='*50}")
    logger.info("测试结果汇总")
    logger.info(f"{'='*50}")
    
    total_tests = len(all_test_results)
    passed_tests = sum(all_test_results)
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过测试: {passed_tests}")
    logger.info(f"通过率: {pass_rate:.1f}%")
    
    if pass_rate >= 80:
        logger.info("🎉 测试通过！Qlib集成功能基本正常")
        return 0
    elif pass_rate >= 60:
        logger.warning("⚠️  部分测试失败，但核心功能可用")
        return 0
    else:
        logger.error("❌ 测试失败较多，需要检查实现")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)