<<<<<<< HEAD
"""
业务服务层
"""

from .data_service import StockDataService
from .data_service_simple import SimpleStockDataService
from .technical_indicators import TechnicalIndicatorCalculator
from .task_manager import TaskManager
from .parquet_manager import ParquetManager
from .data_lifecycle import DataLifecycleManager
from .model_training import ModelTrainingService, TrainingConfig, ModelType, ModelMetrics
from .modern_models import TimesNet, PatchTST, Informer

__all__ = [
    'StockDataService',
    'SimpleStockDataService', 
    'TechnicalIndicatorCalculator',
    'TaskManager',
    'ParquetManager',
    'DataLifecycleManager',
    'ModelTrainingService',
    'TrainingConfig',
    'ModelType',
    'ModelMetrics',
    'TimesNet',
    'PatchTST',
    'Informer'
=======
"""
业务服务层
"""

from .data_service import StockDataService
from .data_service_simple import SimpleStockDataService
from .technical_indicators import TechnicalIndicatorCalculator
from .task_manager import TaskManager
from .parquet_manager import ParquetManager
from .data_lifecycle import DataLifecycleManager

# 延迟导入需要额外依赖的模块
def get_model_training_service():
    """延迟导入模型训练服务"""
    from .model_training import ModelTrainingService, TrainingConfig, ModelType, ModelMetrics
    return ModelTrainingService, TrainingConfig, ModelType, ModelMetrics

def get_modern_models():
    """延迟导入现代模型"""
    from .modern_models import TimesNet, PatchTST, Informer
    return TimesNet, PatchTST, Informer

__all__ = [
    'StockDataService',
    'SimpleStockDataService', 
    'TechnicalIndicatorCalculator',
    'TaskManager',
    'ParquetManager',
    'DataLifecycleManager',
    'get_model_training_service',
    'get_modern_models'
>>>>>>> a6754c6 (feat(platform): Complete stock prediction platform with deployment and monitoring)
]