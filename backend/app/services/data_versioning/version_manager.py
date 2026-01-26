"""
轻量级数据版本管理器
基于文件哈希的版本标识，记录训练数据版本信息
"""
import hashlib
import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DataType(Enum):
    """数据类型"""

    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    FEATURE = "feature"
    RAW = "raw"
    PROCESSED = "processed"


class VersionStatus(Enum):
    """版本状态"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


@dataclass
class DataVersion:
    """数据版本"""

    version_id: str
    version_name: str
    data_type: DataType
    file_path: str
    file_hash: str
    file_size: int
    row_count: Optional[int]
    column_count: Optional[int]
    created_at: datetime
    created_by: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    status: VersionStatus = VersionStatus.ACTIVE
    parent_version_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version_name": self.version_name,
            "data_type": self.data_type.value,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "status": self.status.value,
            "parent_version_id": self.parent_version_id,
            "metadata": self.metadata,
        }


@dataclass
class DataLineage:
    """数据血缘"""

    lineage_id: str
    source_version_id: str
    target_version_id: str
    transformation_type: str
    transformation_config: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "source_version_id": self.source_version_id,
            "target_version_id": self.target_version_id,
            "transformation_type": self.transformation_type,
            "transformation_config": self.transformation_config,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
        }


@dataclass
class DataSnapshot:
    """数据快照"""

    snapshot_id: str
    snapshot_name: str
    version_ids: List[str]
    created_at: datetime
    created_by: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_name": self.snapshot_name,
            "version_ids": self.version_ids,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
        }


class FileHashCalculator:
    """文件哈希计算器"""

    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = "md5") -> str:
        """计算文件哈希值"""
        hash_func = hashlib.new(algorithm)

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            raise

    @staticmethod
    def calculate_dataframe_hash(df: pd.DataFrame, algorithm: str = "md5") -> str:
        """计算DataFrame哈希值"""
        hash_func = hashlib.new(algorithm)

        try:
            # 将DataFrame转换为字节流
            csv_string = df.to_csv(index=False).encode("utf-8")
            hash_func.update(csv_string)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"计算DataFrame哈希失败: {e}")
            raise

    @staticmethod
    def calculate_content_hash(content: bytes, algorithm: str = "md5") -> str:
        """计算内容哈希值"""
        hash_func = hashlib.new(algorithm)
        hash_func.update(content)
        return hash_func.hexdigest()


class DataAnalyzer:
    """数据分析器"""

    @staticmethod
    def analyze_file(file_path: str) -> Dict[str, Any]:
        """分析数据文件"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        analysis = {
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
            "row_count": None,
            "column_count": None,
            "columns": [],
            "data_types": {},
            "missing_values": {},
            "summary_stats": {},
        }

        try:
            # 根据文件类型进行分析
            if file_path.suffix.lower() in [".csv", ".tsv"]:
                df = pd.read_csv(file_path)
                analysis.update(DataAnalyzer._analyze_dataframe(df))
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                analysis.update(DataAnalyzer._analyze_dataframe(df))
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                analysis.update(DataAnalyzer._analyze_json(data))
            elif file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
                analysis.update(DataAnalyzer._analyze_dataframe(df))

        except Exception as e:
            logger.warning(f"数据分析失败 {file_path}: {e}")
            analysis["analysis_error"] = str(e)

        return analysis

    @staticmethod
    def _analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """分析DataFrame"""
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": {},
        }

        # 数值列的统计信息
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis["summary_stats"] = df[numeric_columns].describe().to_dict()

        return analysis

    @staticmethod
    def _analyze_json(data: Any) -> Dict[str, Any]:
        """分析JSON数据"""
        analysis = {}

        if isinstance(data, list):
            analysis["row_count"] = len(data)
            if data and isinstance(data[0], dict):
                analysis["columns"] = list(data[0].keys())
                analysis["column_count"] = len(analysis["columns"])
        elif isinstance(data, dict):
            analysis["columns"] = list(data.keys())
            analysis["column_count"] = len(analysis["columns"])

        return analysis


class DataVersionManager:
    """数据版本管理器"""

    def __init__(self, storage_path: str = "data/versions"):
        """
        初始化数据版本管理器

        Args:
            storage_path: 版本存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 版本存储
        self.versions: Dict[str, DataVersion] = {}
        self.lineages: Dict[str, DataLineage] = {}
        self.snapshots: Dict[str, DataSnapshot] = {}

        # 索引
        self.hash_to_version: Dict[str, str] = {}  # 哈希到版本ID的映射
        self.path_to_version: Dict[str, List[str]] = {}  # 路径到版本ID列表的映射

        # 工具
        self.hash_calculator = FileHashCalculator()
        self.data_analyzer = DataAnalyzer()

        # 线程锁
        self.lock = threading.Lock()

        # 加载现有版本
        self._load_versions()

        logger.info(f"数据版本管理器初始化完成，存储路径: {self.storage_path}")

    def create_version(
        self,
        file_path: str,
        version_name: str,
        data_type: DataType,
        created_by: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        parent_version_id: Optional[str] = None,
        copy_file: bool = True,
    ) -> str:
        """
        创建数据版本

        Args:
            file_path: 数据文件路径
            version_name: 版本名称
            data_type: 数据类型
            created_by: 创建者
            description: 描述
            tags: 标签
            parent_version_id: 父版本ID
            copy_file: 是否复制文件到版本存储

        Returns:
            版本ID
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 计算文件哈希
        file_hash = self.hash_calculator.calculate_file_hash(str(file_path))

        with self.lock:
            # 检查是否已存在相同哈希的版本
            if file_hash in self.hash_to_version:
                existing_version_id = self.hash_to_version[file_hash]
                logger.info(f"文件已存在版本: {existing_version_id}")
                return existing_version_id

            # 生成版本ID
            version_id = f"v_{int(datetime.now().timestamp())}_{file_hash[:8]}"

            # 分析数据文件
            analysis = self.data_analyzer.analyze_file(str(file_path))

            # 确定存储路径
            if copy_file:
                version_file_path = (
                    self.storage_path / f"{version_id}{file_path.suffix}"
                )
                shutil.copy2(file_path, version_file_path)
                stored_path = str(version_file_path)
            else:
                stored_path = str(file_path)

            # 创建版本对象
            version = DataVersion(
                version_id=version_id,
                version_name=version_name,
                data_type=data_type,
                file_path=stored_path,
                file_hash=file_hash,
                file_size=analysis["file_size"],
                row_count=analysis.get("row_count"),
                column_count=analysis.get("column_count"),
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                tags=tags or [],
                parent_version_id=parent_version_id,
                metadata=analysis,
            )

            # 存储版本
            self.versions[version_id] = version
            self.hash_to_version[file_hash] = version_id

            # 更新路径索引
            path_key = str(file_path.resolve())
            if path_key not in self.path_to_version:
                self.path_to_version[path_key] = []
            self.path_to_version[path_key].append(version_id)

            # 保存版本信息
            self._save_version(version)

            logger.info(f"创建数据版本: {version_name} ({version_id})")
            return version_id

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """获取版本信息"""
        return self.versions.get(version_id)

    def get_version_by_hash(self, file_hash: str) -> Optional[DataVersion]:
        """根据哈希获取版本"""
        version_id = self.hash_to_version.get(file_hash)
        return self.versions.get(version_id) if version_id else None

    def list_versions(
        self,
        data_type: Optional[DataType] = None,
        status: Optional[VersionStatus] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> List[DataVersion]:
        """列出版本"""
        versions = list(self.versions.values())

        # 过滤条件
        if data_type:
            versions = [v for v in versions if v.data_type == data_type]

        if status:
            versions = [v for v in versions if v.status == status]

        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]

        if created_by:
            versions = [v for v in versions if v.created_by == created_by]

        # 按创建时间排序
        versions.sort(key=lambda x: x.created_at, reverse=True)

        return versions

    def update_version(
        self,
        version_id: str,
        version_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[VersionStatus] = None,
    ) -> bool:
        """更新版本信息"""
        with self.lock:
            if version_id not in self.versions:
                return False

            version = self.versions[version_id]

            if version_name is not None:
                version.version_name = version_name
            if description is not None:
                version.description = description
            if tags is not None:
                version.tags = tags
            if status is not None:
                version.status = status

            # 保存更新
            self._save_version(version)

            logger.info(f"更新版本信息: {version_id}")
            return True

    def delete_version(self, version_id: str, remove_file: bool = False) -> bool:
        """删除版本"""
        with self.lock:
            if version_id not in self.versions:
                return False

            version = self.versions[version_id]

            # 删除文件
            if remove_file and Path(version.file_path).exists():
                try:
                    Path(version.file_path).unlink()
                    logger.info(f"删除版本文件: {version.file_path}")
                except Exception as e:
                    logger.error(f"删除版本文件失败: {e}")

            # 更新状态为已删除
            version.status = VersionStatus.DELETED

            # 从索引中移除
            if version.file_hash in self.hash_to_version:
                del self.hash_to_version[version.file_hash]

            # 保存更新
            self._save_version(version)

            logger.info(f"删除版本: {version_id}")
            return True

    def create_lineage(
        self,
        source_version_id: str,
        target_version_id: str,
        transformation_type: str,
        transformation_config: Dict[str, Any],
        created_by: str,
        description: str = "",
    ) -> str:
        """创建数据血缘"""
        lineage_id = (
            f"lineage_{int(datetime.now().timestamp())}_{source_version_id[:8]}"
        )

        lineage = DataLineage(
            lineage_id=lineage_id,
            source_version_id=source_version_id,
            target_version_id=target_version_id,
            transformation_type=transformation_type,
            transformation_config=transformation_config,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
        )

        with self.lock:
            self.lineages[lineage_id] = lineage
            self._save_lineage(lineage)

        logger.info(f"创建数据血缘: {source_version_id} -> {target_version_id}")
        return lineage_id

    def get_lineage_chain(
        self, version_id: str, direction: str = "both"
    ) -> List[DataLineage]:
        """获取血缘链"""
        lineages = []

        for lineage in self.lineages.values():
            if (
                direction in ["upstream", "both"]
                and lineage.target_version_id == version_id
            ):
                lineages.append(lineage)
            elif (
                direction in ["downstream", "both"]
                and lineage.source_version_id == version_id
            ):
                lineages.append(lineage)

        return lineages

    def create_snapshot(
        self,
        snapshot_name: str,
        version_ids: List[str],
        created_by: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """创建数据快照"""
        snapshot_id = f"snapshot_{int(datetime.now().timestamp())}"

        # 验证版本ID
        for version_id in version_ids:
            if version_id not in self.versions:
                raise ValueError(f"版本不存在: {version_id}")

        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            snapshot_name=snapshot_name,
            version_ids=version_ids,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or [],
        )

        with self.lock:
            self.snapshots[snapshot_id] = snapshot
            self._save_snapshot(snapshot)

        logger.info(f"创建数据快照: {snapshot_name} ({snapshot_id})")
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[DataSnapshot]:
        """获取快照"""
        return self.snapshots.get(snapshot_id)

    def list_snapshots(self, created_by: Optional[str] = None) -> List[DataSnapshot]:
        """列出快照"""
        snapshots = list(self.snapshots.values())

        if created_by:
            snapshots = [s for s in snapshots if s.created_by == created_by]

        snapshots.sort(key=lambda x: x.created_at, reverse=True)
        return snapshots

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """比较两个版本"""
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)

        if not version1 or not version2:
            raise ValueError("版本不存在")

        comparison = {
            "version1": version1.to_dict(),
            "version2": version2.to_dict(),
            "differences": {},
            "summary": {},
        }

        # 基本信息比较
        if version1.file_size != version2.file_size:
            comparison["differences"]["file_size"] = {
                "version1": version1.file_size,
                "version2": version2.file_size,
                "change": version2.file_size - version1.file_size,
            }

        if (
            version1.row_count
            and version2.row_count
            and version1.row_count != version2.row_count
        ):
            comparison["differences"]["row_count"] = {
                "version1": version1.row_count,
                "version2": version2.row_count,
                "change": version2.row_count - version1.row_count,
            }

        if (
            version1.column_count
            and version2.column_count
            and version1.column_count != version2.column_count
        ):
            comparison["differences"]["column_count"] = {
                "version1": version1.column_count,
                "version2": version2.column_count,
                "change": version2.column_count - version1.column_count,
            }

        # 哈希比较
        comparison["same_content"] = version1.file_hash == version2.file_hash

        # 生成摘要
        comparison["summary"] = {
            "has_differences": len(comparison["differences"]) > 0,
            "same_content": comparison["same_content"],
            "total_differences": len(comparison["differences"]),
        }

        return comparison

    def get_version_stats(self) -> Dict[str, Any]:
        """获取版本统计信息"""
        with self.lock:
            stats = {
                "total_versions": len(self.versions),
                "total_lineages": len(self.lineages),
                "total_snapshots": len(self.snapshots),
                "by_data_type": {},
                "by_status": {},
                "by_creator": {},
                "storage_usage": 0,
            }

            for version in self.versions.values():
                # 按数据类型统计
                data_type = version.data_type.value
                stats["by_data_type"][data_type] = (
                    stats["by_data_type"].get(data_type, 0) + 1
                )

                # 按状态统计
                status = version.status.value
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

                # 按创建者统计
                creator = version.created_by
                stats["by_creator"][creator] = stats["by_creator"].get(creator, 0) + 1

                # 存储使用量
                stats["storage_usage"] += version.file_size

            return stats

    def _load_versions(self):
        """加载现有版本"""
        try:
            versions_file = self.storage_path / "versions.json"
            if versions_file.exists():
                with open(versions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 加载版本
                for version_data in data.get("versions", []):
                    version = DataVersion(
                        version_id=version_data["version_id"],
                        version_name=version_data["version_name"],
                        data_type=DataType(version_data["data_type"]),
                        file_path=version_data["file_path"],
                        file_hash=version_data["file_hash"],
                        file_size=version_data["file_size"],
                        row_count=version_data.get("row_count"),
                        column_count=version_data.get("column_count"),
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        created_by=version_data["created_by"],
                        description=version_data.get("description", ""),
                        tags=version_data.get("tags", []),
                        status=VersionStatus(version_data.get("status", "active")),
                        parent_version_id=version_data.get("parent_version_id"),
                        metadata=version_data.get("metadata", {}),
                    )

                    self.versions[version.version_id] = version
                    self.hash_to_version[version.file_hash] = version.version_id

                # 加载血缘
                for lineage_data in data.get("lineages", []):
                    lineage = DataLineage(
                        lineage_id=lineage_data["lineage_id"],
                        source_version_id=lineage_data["source_version_id"],
                        target_version_id=lineage_data["target_version_id"],
                        transformation_type=lineage_data["transformation_type"],
                        transformation_config=lineage_data["transformation_config"],
                        created_at=datetime.fromisoformat(lineage_data["created_at"]),
                        created_by=lineage_data["created_by"],
                        description=lineage_data.get("description", ""),
                    )

                    self.lineages[lineage.lineage_id] = lineage

                # 加载快照
                for snapshot_data in data.get("snapshots", []):
                    snapshot = DataSnapshot(
                        snapshot_id=snapshot_data["snapshot_id"],
                        snapshot_name=snapshot_data["snapshot_name"],
                        version_ids=snapshot_data["version_ids"],
                        created_at=datetime.fromisoformat(snapshot_data["created_at"]),
                        created_by=snapshot_data["created_by"],
                        description=snapshot_data.get("description", ""),
                        tags=snapshot_data.get("tags", []),
                    )

                    self.snapshots[snapshot.snapshot_id] = snapshot

                logger.info(
                    f"加载了 {len(self.versions)} 个版本，{len(self.lineages)} 个血缘，{len(self.snapshots)} 个快照"
                )

        except Exception as e:
            logger.error(f"加载版本数据失败: {e}")

    def _save_version(self, version: DataVersion):
        """保存版本信息"""
        self._save_all_data()

    def _save_lineage(self, lineage: DataLineage):
        """保存血缘信息"""
        self._save_all_data()

    def _save_snapshot(self, snapshot: DataSnapshot):
        """保存快照信息"""
        self._save_all_data()

    def _save_all_data(self):
        """保存所有数据"""
        try:
            data = {
                "versions": [v.to_dict() for v in self.versions.values()],
                "lineages": [l.to_dict() for l in self.lineages.values()],
                "snapshots": [s.to_dict() for s in self.snapshots.values()],
                "last_updated": datetime.now().isoformat(),
            }

            versions_file = self.storage_path / "versions.json"
            with open(versions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存版本数据失败: {e}")


# 全局数据版本管理器实例
data_version_manager = DataVersionManager()
