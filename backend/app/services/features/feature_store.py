"""
特征存储管理器

实现特征元数据管理、缓存和版本控制功能
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class FeatureType(Enum):
    """特征类型枚举"""

    TECHNICAL_INDICATOR = "technical_indicator"
    ALPHA_FACTOR = "alpha_factor"
    FUNDAMENTAL = "fundamental"
    CUSTOM = "custom"


@dataclass
class FeatureMetadata:
    """特征元数据"""

    feature_name: str
    feature_type: FeatureType
    calculation_method: str
    dependencies: List[str]
    update_frequency: str  # daily, hourly, real_time
    version: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["feature_type"] = self.feature_type.value
        result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """从字典创建"""
        data = data.copy()
        data["feature_type"] = FeatureType(data["feature_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class FeatureCacheEntry:
    """特征缓存条目"""

    cache_key: str
    stock_codes: List[str]
    date_range: Tuple[datetime, datetime]
    feature_names: List[str]
    data_hash: str
    file_path: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_valid_for(
        self, stock_codes: List[str], date_range: Tuple[datetime, datetime]
    ) -> bool:
        """检查缓存是否对指定参数有效"""
        # 检查股票代码是否匹配
        if set(stock_codes) != set(self.stock_codes):
            return False

        # 检查日期范围是否包含
        cache_start, cache_end = self.date_range
        query_start, query_end = date_range

        return cache_start <= query_start and cache_end >= query_end


class FeatureStore:
    """特征存储管理器"""

    def __init__(self, storage_path: str = "./data/features"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.metadata_dir = self.storage_path / "metadata"
        self.cache_dir = self.storage_path / "cache"
        self.versions_dir = self.storage_path / "versions"

        for dir_path in [self.metadata_dir, self.cache_dir, self.versions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._metadata_cache: Dict[str, FeatureMetadata] = {}
        self._cache_index: Dict[str, FeatureCacheEntry] = {}

        # 配置
        self.max_cache_size = 100  # 最大缓存条目数
        self.default_cache_ttl = timedelta(hours=24)  # 默认缓存过期时间

        logger.info(f"特征存储初始化完成，存储路径: {self.storage_path}")

    async def initialize(self):
        """初始化特征存储"""
        try:
            # 加载现有元数据
            await self._load_metadata()

            # 加载缓存索引
            await self._load_cache_index()

            # 清理过期缓存
            await self._cleanup_expired_cache()

            logger.info("特征存储初始化完成")
        except Exception as e:
            logger.error(f"特征存储初始化失败: {e}")
            raise

    async def register_feature(
        self,
        feature_name: str,
        feature_type: FeatureType,
        calculation_method: str,
        dependencies: List[str],
        update_frequency: str = "daily",
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> FeatureMetadata:
        """注册新特征"""
        # 检查特征是否已存在
        if feature_name in self._metadata_cache:
            existing = self._metadata_cache[feature_name]
            logger.warning(f"特征 {feature_name} 已存在，版本: {existing.version}")
            return existing

        # 创建特征元数据
        metadata = FeatureMetadata(
            feature_name=feature_name,
            feature_type=feature_type,
            calculation_method=calculation_method,
            dependencies=dependencies,
            update_frequency=update_frequency,
            version="1.0.0",
            created_at=datetime.now(),
            description=description,
            parameters=parameters or {},
        )

        # 保存元数据
        await self._save_metadata(metadata)

        # 更新内存缓存
        self._metadata_cache[feature_name] = metadata

        logger.info(f"特征注册成功: {feature_name}, 类型: {feature_type.value}")
        return metadata

    async def get_feature_metadata(
        self, feature_name: str
    ) -> Optional[FeatureMetadata]:
        """获取特征元数据"""
        if feature_name in self._metadata_cache:
            return self._metadata_cache[feature_name]

        # 尝试从磁盘加载
        metadata_file = self.metadata_dir / f"{feature_name}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metadata = FeatureMetadata.from_dict(data)
                self._metadata_cache[feature_name] = metadata
                return metadata
            except Exception as e:
                logger.error(f"加载特征元数据失败 {feature_name}: {e}")

        return None

    async def list_features(
        self,
        feature_type: Optional[FeatureType] = None,
        update_frequency: Optional[str] = None,
    ) -> List[FeatureMetadata]:
        """列出特征"""
        features = list(self._metadata_cache.values())

        # 过滤条件
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]

        if update_frequency:
            features = [f for f in features if f.update_frequency == update_frequency]

        return features

    async def cache_features(
        self,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
        features_data: pd.DataFrame,
        feature_names: List[str],
        ttl: Optional[timedelta] = None,
    ) -> str:
        """缓存特征数据"""
        # 生成缓存键
        cache_key = self._generate_cache_key(stock_codes, date_range, feature_names)

        # 计算数据哈希
        data_hash = self._calculate_data_hash(features_data)

        # 保存数据文件
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        features_data.to_parquet(cache_file)

        # 创建缓存条目
        expires_at = None
        if ttl:
            expires_at = datetime.now() + ttl
        elif self.default_cache_ttl:
            expires_at = datetime.now() + self.default_cache_ttl

        cache_entry = FeatureCacheEntry(
            cache_key=cache_key,
            stock_codes=stock_codes,
            date_range=date_range,
            feature_names=feature_names,
            data_hash=data_hash,
            file_path=str(cache_file),
            created_at=datetime.now(),
            expires_at=expires_at,
        )

        # 更新缓存索引
        self._cache_index[cache_key] = cache_entry
        await self._save_cache_index()

        # 清理旧缓存
        await self._cleanup_old_cache()

        logger.info(f"特征数据缓存成功: {cache_key}, 数据量: {len(features_data)}")
        return cache_key

    async def get_cached_features(
        self,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
        feature_names: List[str],
    ) -> Optional[pd.DataFrame]:
        """获取缓存的特征数据"""
        # 查找匹配的缓存
        for cache_key, cache_entry in self._cache_index.items():
            if cache_entry.is_expired():
                continue

            if cache_entry.is_valid_for(stock_codes, date_range):
                # 检查特征名称是否匹配
                if set(feature_names).issubset(set(cache_entry.feature_names)):
                    try:
                        # 加载数据
                        cache_file = Path(cache_entry.file_path)
                        if cache_file.exists():
                            data = pd.read_parquet(cache_file)

                            # 更新访问统计
                            cache_entry.access_count += 1
                            cache_entry.last_accessed = datetime.now()

                            # 只返回请求的特征
                            available_features = [
                                f for f in feature_names if f in data.columns
                            ]
                            if available_features:
                                logger.info(
                                    f"命中特征缓存: {cache_key}, 特征数: {len(available_features)}"
                                )
                                return data[available_features]

                    except Exception as e:
                        logger.error(f"加载缓存数据失败 {cache_key}: {e}")
                        # 移除损坏的缓存条目
                        del self._cache_index[cache_key]

        return None

    async def update_feature_version(
        self,
        feature_name: str,
        new_version: str,
        changes: Optional[Dict[str, Any]] = None,
    ) -> FeatureMetadata:
        """更新特征版本"""
        metadata = await self.get_feature_metadata(feature_name)
        if not metadata:
            raise ValueError(f"特征不存在: {feature_name}")

        # 备份旧版本
        old_version_file = self.versions_dir / f"{feature_name}_{metadata.version}.json"
        await self._save_metadata_to_file(metadata, old_version_file)

        # 更新元数据
        metadata.version = new_version
        metadata.updated_at = datetime.now()

        if changes:
            for key, value in changes.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        # 保存新版本
        await self._save_metadata(metadata)
        self._metadata_cache[feature_name] = metadata

        logger.info(f"特征版本更新: {feature_name} -> {new_version}")
        return metadata

    async def invalidate_cache(
        self,
        stock_codes: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """使缓存失效"""
        keys_to_remove = []

        for cache_key, cache_entry in self._cache_index.items():
            should_remove = False

            # 检查股票代码
            if stock_codes and any(
                code in cache_entry.stock_codes for code in stock_codes
            ):
                should_remove = True

            # 检查特征名称
            if feature_names and any(
                name in cache_entry.feature_names for name in feature_names
            ):
                should_remove = True

            # 如果没有指定条件，清空所有缓存
            if not stock_codes and not feature_names:
                should_remove = True

            if should_remove:
                keys_to_remove.append(cache_key)
                # 删除缓存文件
                try:
                    cache_file = Path(cache_entry.file_path)
                    if cache_file.exists():
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"删除缓存文件失败 {cache_entry.file_path}: {e}")

        # 从索引中移除
        for key in keys_to_remove:
            del self._cache_index[key]

        if keys_to_remove:
            await self._save_cache_index()
            logger.info(f"缓存失效完成，移除 {len(keys_to_remove)} 个缓存条目")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_entries = len(self._cache_index)
        expired_entries = sum(
            1 for entry in self._cache_index.values() if entry.is_expired()
        )
        total_size = 0

        for entry in self._cache_index.values():
            try:
                cache_file = Path(entry.file_path)
                if cache_file.exists():
                    total_size += cache_file.stat().st_size
            except Exception:
                pass

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_hit_rate": self._calculate_hit_rate(),
        }

    def _generate_cache_key(
        self,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
        feature_names: List[str],
    ) -> str:
        """生成缓存键"""
        # 创建唯一标识符
        codes_str = "_".join(sorted(stock_codes))
        start_str = date_range[0].strftime("%Y%m%d")
        end_str = date_range[1].strftime("%Y%m%d")
        features_str = "_".join(sorted(feature_names))

        # 生成哈希
        content = f"{codes_str}_{start_str}_{end_str}_{features_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据哈希"""
        # 使用数据的形状和部分内容生成哈希
        content = f"{data.shape}_{data.dtypes.to_string()}_{str(data.head().values.tobytes())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _calculate_hit_rate(self) -> float:
        """计算缓存命中率"""
        if not self._cache_index:
            return 0.0

        total_access = sum(entry.access_count for entry in self._cache_index.values())
        if total_access == 0:
            return 0.0

        # 简化的命中率计算
        return min(1.0, total_access / len(self._cache_index))

    async def _load_metadata(self):
        """加载所有特征元数据"""
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                if metadata_file.name.startswith("_"):  # 跳过系统文件
                    continue

                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    metadata = FeatureMetadata.from_dict(data)
                    self._metadata_cache[metadata.feature_name] = metadata
                except Exception as e:
                    logger.error(f"加载元数据文件失败 {metadata_file}: {e}")

            logger.info(f"加载特征元数据完成，共 {len(self._metadata_cache)} 个特征")
        except Exception as e:
            logger.error(f"加载特征元数据失败: {e}")

    async def _load_cache_index(self):
        """加载缓存索引"""
        index_file = self.cache_dir / "_cache_index.json"
        if not index_file.exists():
            return

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for cache_key, entry_data in data.items():
                # 转换日期字符串
                entry_data["created_at"] = datetime.fromisoformat(
                    entry_data["created_at"]
                )
                if entry_data.get("expires_at"):
                    entry_data["expires_at"] = datetime.fromisoformat(
                        entry_data["expires_at"]
                    )
                if entry_data.get("last_accessed"):
                    entry_data["last_accessed"] = datetime.fromisoformat(
                        entry_data["last_accessed"]
                    )

                # 转换日期范围
                date_range = entry_data["date_range"]
                entry_data["date_range"] = (
                    datetime.fromisoformat(date_range[0]),
                    datetime.fromisoformat(date_range[1]),
                )

                self._cache_index[cache_key] = FeatureCacheEntry(**entry_data)

            logger.info(f"加载缓存索引完成，共 {len(self._cache_index)} 个缓存条目")
        except Exception as e:
            logger.error(f"加载缓存索引失败: {e}")

    async def _save_metadata(self, metadata: FeatureMetadata):
        """保存特征元数据"""
        metadata_file = self.metadata_dir / f"{metadata.feature_name}.json"
        await self._save_metadata_to_file(metadata, metadata_file)

    async def _save_metadata_to_file(self, metadata: FeatureMetadata, file_path: Path):
        """保存元数据到指定文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存特征元数据失败 {file_path}: {e}")
            raise

    async def _save_cache_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "_cache_index.json"

        try:
            # 转换为可序列化的格式
            serializable_index = {}
            for cache_key, entry in self._cache_index.items():
                entry_data = {
                    "cache_key": entry.cache_key,
                    "stock_codes": entry.stock_codes,
                    "date_range": [
                        entry.date_range[0].isoformat(),
                        entry.date_range[1].isoformat(),
                    ],
                    "feature_names": entry.feature_names,
                    "data_hash": entry.data_hash,
                    "file_path": entry.file_path,
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat()
                    if entry.expires_at
                    else None,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed.isoformat()
                    if entry.last_accessed
                    else None,
                }
                serializable_index[cache_key] = entry_data

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(serializable_index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")

    async def _cleanup_expired_cache(self):
        """清理过期缓存"""
        expired_keys = []

        for cache_key, entry in self._cache_index.items():
            if entry.is_expired():
                expired_keys.append(cache_key)
                # 删除缓存文件
                try:
                    cache_file = Path(entry.file_path)
                    if cache_file.exists():
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"删除过期缓存文件失败 {entry.file_path}: {e}")

        # 从索引中移除
        for key in expired_keys:
            del self._cache_index[key]

        if expired_keys:
            await self._save_cache_index()
            logger.info(f"清理过期缓存完成，移除 {len(expired_keys)} 个条目")

    async def _cleanup_old_cache(self):
        """清理旧缓存（基于LRU策略）"""
        if len(self._cache_index) <= self.max_cache_size:
            return

        # 按最后访问时间排序
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at,
        )

        # 移除最旧的缓存
        entries_to_remove = len(self._cache_index) - self.max_cache_size
        removed_keys = []

        for i in range(entries_to_remove):
            cache_key, entry = sorted_entries[i]
            removed_keys.append(cache_key)

            # 删除缓存文件
            try:
                cache_file = Path(entry.file_path)
                if cache_file.exists():
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"删除旧缓存文件失败 {entry.file_path}: {e}")

        # 从索引中移除
        for key in removed_keys:
            del self._cache_index[key]

        if removed_keys:
            await self._save_cache_index()
            logger.info(f"清理旧缓存完成，移除 {len(removed_keys)} 个条目")


# 全局特征存储实例
feature_store = FeatureStore()
