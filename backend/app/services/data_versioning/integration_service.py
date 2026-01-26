"""
数据版本控制集成服务
复用现有存储基础设施，添加版本控制接口
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.data.parquet_manager import ParquetManager
from app.services.data_versioning.lineage_tracker import (
    DataLineageTracker,
    NodeType,
    TransformationType,
    data_lineage_tracker,
)
from app.services.data_versioning.version_manager import (
    DataType,
    DataVersionManager,
    VersionStatus,
    data_version_manager,
)
from app.services.events.data_sync_events import (
    DataSyncEventType,
    get_data_sync_event_manager,
)


class DataVersioningIntegrationService:
    """数据版本控制集成服务"""

    def __init__(self):
        self.version_manager = data_version_manager
        self.lineage_tracker = data_lineage_tracker
        self.event_manager = get_data_sync_event_manager()

        # 初始化时注册事件监听器
        self._register_event_listeners()

        logger.info("数据版本控制集成服务初始化完成")

    def _register_event_listeners(self):
        """注册事件监听器"""
        # 监听数据同步事件，自动创建版本
        self.event_manager.register_listener(
            DataSyncEventType.SYNC_COMPLETED, self._on_data_sync_completed
        )

    def _on_data_sync_completed(self, event_data: Dict[str, Any]):
        """数据同步完成事件处理"""
        try:
            stock_code = event_data.get("stock_code")
            file_path = event_data.get("file_path")

            if stock_code and file_path:
                # 自动创建数据版本
                version_id = self.create_data_version_from_sync(
                    stock_code=stock_code, file_path=file_path, sync_metadata=event_data
                )

                logger.info(f"自动创建数据版本: {version_id} (股票: {stock_code})")

        except Exception as e:
            logger.error(f"处理数据同步完成事件失败: {e}")

    def _on_file_downloaded(self, event_data: Dict[str, Any]):
        """文件下载事件处理"""
        try:
            file_path = event_data.get("file_path")
            stock_code = event_data.get("stock_code")

            if file_path and stock_code:
                # 创建数据血缘节点
                self._create_data_source_node(
                    stock_code=stock_code, file_path=file_path, metadata=event_data
                )

        except Exception as e:
            logger.error(f"处理文件下载事件失败: {e}")

    def create_data_version_from_sync(
        self, stock_code: str, file_path: str, sync_metadata: Dict[str, Any]
    ) -> str:
        """从数据同步创建版本"""
        version_name = f"{stock_code}_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self.version_manager.create_version(
            file_path=file_path,
            version_name=version_name,
            data_type=DataType.RAW,
            created_by="data_sync_service",
            description=f"从远端同步的股票数据: {stock_code}",
            tags=["sync", "raw_data", stock_code],
            copy_file=False,  # 不复制文件，使用原始路径
        )

    def create_processed_data_version(
        self,
        source_version_id: str,
        processed_file_path: str,
        processing_config: Dict[str, Any],
        created_by: str = "system",
    ) -> str:
        """创建处理后的数据版本"""
        source_version = self.version_manager.get_version(source_version_id)
        if not source_version:
            raise ValueError(f"源版本不存在: {source_version_id}")

        # 从源版本信息中提取股票代码
        stock_code = None
        for tag in source_version.tags:
            if tag not in ["sync", "raw_data", "processed"]:
                stock_code = tag
                break

        version_name = (
            f"processed_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # 创建处理后的版本
        processed_version_id = self.version_manager.create_version(
            file_path=processed_file_path,
            version_name=version_name,
            data_type=DataType.PROCESSED,
            created_by=created_by,
            description=f"处理后的股票数据: {stock_code}",
            tags=["processed", stock_code] if stock_code else ["processed"],
            parent_version_id=source_version_id,
            copy_file=False,
        )

        # 创建数据血缘
        self.version_manager.create_lineage(
            source_version_id=source_version_id,
            target_version_id=processed_version_id,
            transformation_type="data_processing",
            transformation_config=processing_config,
            created_by=created_by,
            description="数据处理转换",
        )

        return processed_version_id

    def create_feature_version(
        self,
        source_data_ids: List[str],
        feature_file_path: str,
        feature_config: Dict[str, Any],
        created_by: str = "feature_engine",
    ) -> str:
        """创建特征版本"""
        feature_name = feature_config.get("feature_name", "unknown_feature")
        version_name = (
            f"feature_{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # 创建特征版本
        feature_version_id = self.version_manager.create_version(
            file_path=feature_file_path,
            version_name=version_name,
            data_type=DataType.FEATURE,
            created_by=created_by,
            description=f"特征数据: {feature_name}",
            tags=["feature", feature_name],
            copy_file=False,
        )

        # 为每个源数据创建血缘
        for source_data_id in source_data_ids:
            self.version_manager.create_lineage(
                source_version_id=source_data_id,
                target_version_id=feature_version_id,
                transformation_type="feature_engineering",
                transformation_config=feature_config,
                created_by=created_by,
                description=f"特征工程: {feature_name}",
            )

        # 在血缘追踪器中创建特征节点和血缘
        self.lineage_tracker.track_feature_computation(
            source_data_ids=source_data_ids,
            feature_id=feature_version_id,
            feature_name=feature_name,
            computation_config=feature_config,
            created_by=created_by,
        )

        return feature_version_id

    def _create_data_source_node(
        self, stock_code: str, file_path: str, metadata: Dict[str, Any]
    ):
        """创建数据源节点"""
        node_id = f"data_source_{stock_code}_{Path(file_path).stem}"

        self.lineage_tracker.create_node(
            node_id=node_id,
            node_type=NodeType.DATA_SOURCE,
            name=f"股票数据源: {stock_code}",
            description=f"来自远端同步的股票数据: {stock_code}",
            properties={
                "stock_code": stock_code,
                "file_path": file_path,
                "sync_metadata": metadata,
            },
            created_by="data_sync_service",
            tags=["sync", "raw_data", stock_code],
        )

    def track_model_training_with_versions(
        self,
        training_data_version_ids: List[str],
        feature_version_ids: List[str],
        model_id: str,
        model_name: str,
        training_config: Dict[str, Any],
        created_by: str = "model_trainer",
    ) -> List[str]:
        """使用版本ID追踪模型训练"""
        # 在血缘追踪器中记录模型训练
        lineage_ids = self.lineage_tracker.track_model_training(
            training_data_ids=training_data_version_ids,
            feature_ids=feature_version_ids,
            model_id=model_id,
            model_name=model_name,
            training_config=training_config,
            created_by=created_by,
        )

        # 在版本管理器中创建血缘关系
        version_lineage_ids = []

        # 训练数据到模型的血缘
        for data_version_id in training_data_version_ids:
            lineage_id = self.version_manager.create_lineage(
                source_version_id=data_version_id,
                target_version_id=model_id,
                transformation_type="model_training",
                transformation_config=training_config,
                created_by=created_by,
                description=f"模型训练: {model_name}",
            )
            version_lineage_ids.append(lineage_id)

        # 特征到模型的血缘
        for feature_version_id in feature_version_ids:
            lineage_id = self.version_manager.create_lineage(
                source_version_id=feature_version_id,
                target_version_id=model_id,
                transformation_type="model_training",
                transformation_config=training_config,
                created_by=created_by,
                description=f"使用特征训练模型: {model_name}",
            )
            version_lineage_ids.append(lineage_id)

        return version_lineage_ids

    def get_stock_data_versions(self, stock_code: str) -> List[Dict[str, Any]]:
        """获取股票的所有数据版本"""
        versions = self.version_manager.list_versions(tags=[stock_code])
        return [version.to_dict() for version in versions]

    def get_feature_lineage_for_stock(self, stock_code: str) -> Dict[str, Any]:
        """获取股票的特征血缘"""
        # 获取股票相关的所有版本
        versions = self.version_manager.list_versions(tags=[stock_code])

        # 获取特征版本
        feature_versions = [v for v in versions if v.data_type == DataType.FEATURE]

        lineage_info = {
            "stock_code": stock_code,
            "raw_data_versions": [
                v.to_dict() for v in versions if v.data_type == DataType.RAW
            ],
            "processed_versions": [
                v.to_dict() for v in versions if v.data_type == DataType.PROCESSED
            ],
            "feature_versions": [v.to_dict() for v in feature_versions],
            "feature_lineages": [],
        }

        # 获取每个特征的血缘信息
        for feature_version in feature_versions:
            feature_lineage = self.lineage_tracker.get_feature_lineage(
                feature_version.version_id
            )
            if feature_lineage:
                lineage_info["feature_lineages"].append(feature_lineage)

        return lineage_info

    def get_model_training_lineage(self, model_id: str) -> Dict[str, Any]:
        """获取模型训练血缘"""
        # 从血缘追踪器获取模型血缘
        model_lineage = self.lineage_tracker.get_model_lineage(model_id)

        # 从版本管理器获取相关版本信息
        version_lineages = self.version_manager.get_lineage_chain(model_id)

        return {
            "model_id": model_id,
            "lineage_tracker_info": model_lineage,
            "version_lineages": [lineage.to_dict() for lineage in version_lineages],
            "total_dependencies": len(version_lineages),
        }

    def create_data_snapshot_for_training(
        self,
        training_name: str,
        data_version_ids: List[str],
        feature_version_ids: List[str],
        created_by: str = "model_trainer",
    ) -> str:
        """为模型训练创建数据快照"""
        all_version_ids = data_version_ids + feature_version_ids

        snapshot_id = self.version_manager.create_snapshot(
            snapshot_name=f"training_snapshot_{training_name}",
            version_ids=all_version_ids,
            created_by=created_by,
            description=f"模型训练数据快照: {training_name}",
            tags=["training", "snapshot", training_name],
        )

        return snapshot_id

    def get_version_comparison(
        self, version_id1: str, version_id2: str
    ) -> Dict[str, Any]:
        """比较两个版本"""
        return self.version_manager.compare_versions(version_id1, version_id2)

    def get_data_versioning_stats(self) -> Dict[str, Any]:
        """获取数据版本控制统计信息"""
        version_stats = self.version_manager.get_version_stats()
        lineage_stats = self.lineage_tracker.get_lineage_summary()

        return {
            "version_stats": version_stats,
            "lineage_stats": lineage_stats,
            "integration_info": {
                "event_listeners_registered": True,
                "auto_versioning_enabled": True,
                "lineage_tracking_enabled": True,
            },
        }

    def cleanup_old_versions(
        self, days_to_keep: int = 30, keep_tagged_versions: bool = True
    ) -> Dict[str, Any]:
        """清理旧版本"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        versions_to_delete = []
        for version in self.version_manager.list_versions():
            if version.created_at < cutoff_date:
                # 如果保留标记版本，跳过有特殊标签的版本
                if keep_tagged_versions and any(
                    tag in ["important", "baseline", "production"]
                    for tag in version.tags
                ):
                    continue

                versions_to_delete.append(version.version_id)

        deleted_count = 0
        for version_id in versions_to_delete:
            if self.version_manager.delete_version(version_id, remove_file=False):
                deleted_count += 1

        return {
            "deleted_versions": deleted_count,
            "total_candidates": len(versions_to_delete),
            "cutoff_date": cutoff_date.isoformat(),
            "keep_tagged_versions": keep_tagged_versions,
        }


# 全局集成服务实例
data_versioning_integration = DataVersioningIntegrationService()
