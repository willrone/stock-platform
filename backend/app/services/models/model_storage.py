"""
模型存储和版本管理服务
"""

import os
import json
import shutil
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from loguru import logger

from app.core.error_handler import ModelError, ErrorSeverity, ErrorContext, handle_async_exception
from app.core.logging_config import AuditLogger

# 导入统一的错误处理机制
try:
    from app.core.error_handler import DataError, TaskError
except ImportError:
    DataError = Exception
    TaskError = Exception


# 从shared_types.py导入共享类型
try:
    from .shared_types import ModelStatus, ModelType, ModelMetadata
    SHARED_TYPES_AVAILABLE = True
except ImportError:
    SHARED_TYPES_AVAILABLE = False
    # 如果导入失败，使用本地定义作为备选
    class ModelStatus(Enum):
        """模型状态"""
        TRAINING = "training"
        TRAINED = "trained"
        VALIDATING = "validating"
        READY = "ready"
        DEPLOYED = "deployed"
        DEPRECATED = "deprecated"
        FAILED = "failed"
    
    class ModelType(Enum):
        """模型类型"""
        XGBOOST = "xgboost"
        LIGHTGBM = "lightgbm"
        RANDOM_FOREST = "random_forest"
        LINEAR_REGRESSION = "linear_regression"
        NEURAL_NETWORK = "neural_network"
        ENSEMBLE = "ensemble"
    
    @dataclass
    class ModelMetadata:
        """模型元数据"""
        model_id: str
        model_name: str
        model_type: ModelType
        version: str
        description: str
        created_by: str
        created_at: datetime
        updated_at: datetime
        status: ModelStatus
        
        # 训练信息
        training_data_info: Dict[str, Any]
        hyperparameters: Dict[str, Any]
        training_config: Dict[str, Any]
        
        # 性能指标
        performance_metrics: Dict[str, float]
        validation_metrics: Dict[str, float]
        
        # 部署信息
        deployment_info: Optional[Dict[str, Any]] = None
        
        # 文件信息
        model_file_path: Optional[str] = None
        model_file_size: Optional[int] = None
        model_file_hash: Optional[str] = None
        
        # 依赖信息
        dependencies: Optional[Dict[str, str]] = None
        feature_columns: Optional[List[str]] = None
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            data = asdict(self)
            data['model_type'] = self.model_type.value
            data['status'] = self.status.value
            data['created_at'] = self.created_at.isoformat()
            data['updated_at'] = self.updated_at.isoformat()
            return data
        
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
            """从字典创建"""
            data = data.copy()
            data['model_type'] = ModelType(data['model_type'])
            data['status'] = ModelStatus(data['status'])
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            return cls(**data)


class ModelStorage:
    """模型存储管理器"""
    
    def __init__(self, storage_root: str = None):
        # 使用配置中的路径，如果没有提供则使用默认配置
        from app.core.config import settings
        if storage_root is None:
            storage_root = settings.MODEL_STORAGE_PATH
        
        self.storage_root = Path(storage_root)
        # 解析相对路径为绝对路径
        if not self.storage_root.is_absolute():
            backend_dir = Path(__file__).parent.parent.parent
            self.storage_root = (backend_dir / self.storage_root).resolve()
        
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.models_dir = self.storage_root / "models"
        self.metadata_dir = self.storage_root / "metadata"
        self.versions_dir = self.storage_root / "versions"
        self.backups_dir = self.storage_root / "backups"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.versions_dir, self.backups_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 模型缓存
        self.model_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, ModelMetadata] = {}
        
        logger.info(f"模型存储初始化完成: {self.storage_root}")
    
    @handle_async_exception
    def save_model(self, model: Any, metadata: ModelMetadata, 
                  overwrite: bool = False) -> bool:
        """保存模型"""
        try:
            model_id = metadata.model_id
            
            # 检查模型是否已存在
            if not overwrite and self.model_exists(model_id):
                raise ModelError(
                    message=f"模型已存在: {model_id}，使用overwrite=True强制覆盖",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(model_id=model_id)
                )
            
            # 清理model_id，移除不允许的文件名字符
            import re
            safe_model_id = re.sub(r'[<>:"/\\|?*]', '_', model_id)  # 替换不允许的字符为下划线
            safe_model_id = re.sub(r'\s+', '_', safe_model_id)  # 替换空格为下划线
            
            # 生成文件路径
            model_file_path = self.models_dir / f"{safe_model_id}.joblib"
            metadata_file_path = self.metadata_dir / f"{safe_model_id}.json"
            
            # 确保目录存在
            model_file_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 备份现有模型（如果存在）
            if overwrite and model_file_path.exists():
                self._backup_model(safe_model_id)
            
            # 保存模型文件
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_file_path)
            else:
                # 使用pickle作为备选
                import pickle
                with open(model_file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # 计算文件信息
            file_size = model_file_path.stat().st_size
            file_hash = self._calculate_file_hash(model_file_path)
            
            # 更新元数据
            metadata.model_file_path = str(model_file_path)
            metadata.model_file_size = file_size
            metadata.model_file_hash = file_hash
            metadata.updated_at = datetime.utcnow()
            
            # 保存元数据
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            self.metadata_cache[model_id] = metadata
            
            # 记录审计日志
            AuditLogger.log_user_action(
                action="save_model",
                user_id=metadata.created_by,
                resource=f"model:{model_id}",
                success=True,
                details={
                    "model_name": metadata.model_name,
                    "model_type": metadata.model_type.value,
                    "version": metadata.version,
                    "file_size": file_size
                }
            )
            
            logger.info(f"模型保存成功: {model_id}, 版本: {metadata.version}, 大小: {file_size} bytes")
            return True
            
        except Exception as e:
            raise ModelError(
                message=f"保存模型失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=metadata.model_id),
                original_exception=e
            )
    
    @handle_async_exception
    def load_model(self, model_id: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """加载模型"""
        try:
            # 获取元数据
            metadata = self.get_model_metadata(model_id, version)
            if not metadata:
                raise ModelError(
                    message=f"模型不存在: {model_id}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(model_id=model_id)
                )
            
            # 检查缓存
            cache_key = f"{model_id}_{metadata.version}"
            if cache_key in self.model_cache:
                logger.debug(f"从缓存加载模型: {model_id}")
                return self.model_cache[cache_key], metadata
            
            # 从文件加载
            model_file_path = Path(metadata.model_file_path)
            if not model_file_path.exists():
                raise ModelError(
                    message=f"模型文件不存在: {model_file_path}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(model_id=model_id)
                )
            
            # 验证文件完整性
            current_hash = self._calculate_file_hash(model_file_path)
            if current_hash != metadata.model_file_hash:
                logger.warning(f"模型文件哈希不匹配: {model_id}，可能文件已损坏")
            
            # 加载模型
            if JOBLIB_AVAILABLE:
                model = joblib.load(model_file_path)
            else:
                # 使用pickle作为备选
                import pickle
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)
            
            # 缓存模型
            self.model_cache[cache_key] = model
            
            logger.info(f"模型加载成功: {model_id}, 版本: {metadata.version}")
            return model, metadata
            
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(
                message=f"加载模型失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
                original_exception=e
            )
    
    @handle_async_exception
    def get_model_metadata(self, model_id: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        try:
            # 检查缓存
            if model_id in self.metadata_cache:
                cached_metadata = self.metadata_cache[model_id]
                if version is None or cached_metadata.version == version:
                    return cached_metadata
            
            # 从文件加载
            if version:
                metadata_file = self.versions_dir / model_id / f"{version}.json"
            else:
                metadata_file = self.metadata_dir / f"{model_id}.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            metadata = ModelMetadata.from_dict(metadata_dict)
            
            # 更新缓存
            if version is None:
                self.metadata_cache[model_id] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"获取模型元数据失败: {model_id}, 错误: {e}")
            return None
    
    @handle_async_exception
    def list_models(self, model_type: Optional[ModelType] = None, 
                   status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """列出模型"""
        try:
            models = []
            
            # 遍历元数据文件
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_dict = json.load(f)
                    
                    metadata = ModelMetadata.from_dict(metadata_dict)
                    
                    # 应用过滤条件
                    if model_type and metadata.model_type != model_type:
                        continue
                    
                    if status and metadata.status != status:
                        continue
                    
                    models.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"读取模型元数据失败: {metadata_file}, 错误: {e}")
                    continue
            
            # 按创建时间排序
            models.sort(key=lambda x: x.created_at, reverse=True)
            
            return models
            
        except Exception as e:
            raise ModelError(
                message=f"列出模型失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e
            )
    
    def model_exists(self, model_id: str) -> bool:
        """检查模型是否存在"""
        metadata_file = self.metadata_dir / f"{model_id}.json"
        return metadata_file.exists()
    
    def _backup_model(self, model_id: str):
        """备份模型"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backups_dir / f"{model_id}_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # 备份模型文件
            model_file = self.models_dir / f"{model_id}.joblib"
            if model_file.exists():
                shutil.copy2(model_file, backup_dir / f"{model_id}.joblib")
            
            # 备份元数据文件
            metadata_file = self.metadata_dir / f"{model_id}.json"
            if metadata_file.exists():
                shutil.copy2(metadata_file, backup_dir / f"{model_id}.json")
            
            logger.info(f"模型备份完成: {model_id} -> {backup_dir}")
            
        except Exception as e:
            logger.warning(f"模型备份失败: {model_id}, 错误: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            stats = {
                "total_models": 0,
                "models_by_type": {},
                "models_by_status": {},
                "total_storage_size": 0,
                "cache_size": len(self.model_cache),
                "backup_count": 0
            }
            
            # 统计模型
            models = self.list_models()
            stats["total_models"] = len(models)
            
            for model in models:
                # 按类型统计
                model_type = model.model_type.value
                stats["models_by_type"][model_type] = stats["models_by_type"].get(model_type, 0) + 1
                
                # 按状态统计
                status = model.status.value
                stats["models_by_status"][status] = stats["models_by_status"].get(status, 0) + 1
                
                # 存储大小
                if model.model_file_size:
                    stats["total_storage_size"] += model.model_file_size
            
            # 统计备份
            if self.backups_dir.exists():
                stats["backup_count"] = len(list(self.backups_dir.iterdir()))
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}


# ModelVersionManager 已移至 model_evaluation.py 文件，统一实现