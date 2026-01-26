"""
训练血缘追踪器

提供模型训练的数据和配置依赖追踪功能，包括：
- 数据血缘记录
- 配置依赖追踪
- 特征工程血缘
- 模型继承关系
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from loguru import logger
from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...core.config import settings
from ...core.database import get_async_session as get_db
from ...models.task_models import ModelInfo, Task


class LineageTracker:
    """训练血缘追踪器"""

    def __init__(self):
        self.lineage_cache = {}  # 血缘关系缓存

    async def record_training_lineage(
        self,
        model_id: str,
        training_config: Dict[str, Any],
        data_sources: List[Dict[str, Any]],
        feature_config: Optional[Dict[str, Any]] = None,
        parent_models: Optional[List[str]] = None,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """
        记录训练血缘信息

        Args:
            model_id: 模型ID
            training_config: 训练配置
            data_sources: 数据源信息
            feature_config: 特征工程配置
            parent_models: 父模型列表
            db: 数据库会话

        Returns:
            bool: 记录是否成功
        """
        if db is None:
            async for session in get_db():
                return await self._record_training_lineage_impl(
                    session,
                    model_id,
                    training_config,
                    data_sources,
                    feature_config,
                    parent_models,
                )
        else:
            return await self._record_training_lineage_impl(
                db,
                model_id,
                training_config,
                data_sources,
                feature_config,
                parent_models,
            )

    async def _record_training_lineage_impl(
        self,
        db: AsyncSession,
        model_id: str,
        training_config: Dict[str, Any],
        data_sources: List[Dict[str, Any]],
        feature_config: Optional[Dict[str, Any]],
        parent_models: Optional[List[str]],
    ) -> bool:
        """记录训练血缘信息实现"""
        try:
            # 构建血缘信息
            lineage_info = {
                "training_config": training_config,
                "data_sources": data_sources,
                "feature_config": feature_config,
                "parent_models": parent_models or [],
                "data_fingerprint": self._calculate_data_fingerprint(data_sources),
                "config_fingerprint": self._calculate_config_fingerprint(
                    training_config
                ),
                "recorded_at": datetime.utcnow().isoformat(),
            }

            # 更新模型信息中的血缘数据
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.model_id == model_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                logger.error(f"模型不存在: {model_id}")
                return False

            # 更新模型的超参数字段来存储血缘信息
            current_hyperparams = model.hyperparameters or {}
            current_hyperparams["lineage"] = lineage_info

            await db.execute(
                update(ModelInfo)
                .where(ModelInfo.model_id == model_id)
                .values(hyperparameters=current_hyperparams)
            )

            await db.commit()

            # 更新缓存
            self.lineage_cache[model_id] = lineage_info

            logger.info(f"训练血缘记录成功: {model_id}")
            return True

        except Exception as e:
            logger.error(f"记录训练血缘失败: {e}")
            await db.rollback()
            return False

    def _calculate_data_fingerprint(self, data_sources: List[Dict[str, Any]]) -> str:
        """计算数据指纹"""
        # 创建数据源的唯一标识
        data_info = []
        for source in data_sources:
            info = {
                "source_type": source.get("source_type", "unknown"),
                "path": source.get("path", ""),
                "date_range": source.get("date_range", {}),
                "stock_codes": sorted(source.get("stock_codes", [])),
                "version": source.get("version", ""),
            }
            data_info.append(info)

        # 排序确保一致性
        data_info.sort(key=lambda x: str(x))

        # 计算哈希
        data_str = json.dumps(data_info, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _calculate_config_fingerprint(self, config: Dict[str, Any]) -> str:
        """计算配置指纹"""
        # 提取关键配置信息
        key_config = {
            "model_type": config.get("model_type", ""),
            "hyperparameters": config.get("hyperparameters", {}),
            "feature_config": config.get("feature_config", {}),
            "preprocessing": config.get("preprocessing", {}),
        }

        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def get_model_lineage(
        self,
        model_id: str,
        include_ancestors: bool = True,
        include_descendants: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        获取模型血缘信息

        Args:
            model_id: 模型ID
            include_ancestors: 是否包含祖先模型
            include_descendants: 是否包含后代模型
            db: 数据库会话

        Returns:
            Dict: 血缘信息
        """
        if db is None:
            async for session in get_db():
                return await self._get_model_lineage_impl(
                    session, model_id, include_ancestors, include_descendants
                )
        else:
            return await self._get_model_lineage_impl(
                db, model_id, include_ancestors, include_descendants
            )

    async def _get_model_lineage_impl(
        self,
        db: AsyncSession,
        model_id: str,
        include_ancestors: bool,
        include_descendants: bool,
    ) -> Dict[str, Any]:
        """获取模型血缘信息实现"""
        try:
            # 获取当前模型信息
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.model_id == model_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                return {"error": f"模型不存在: {model_id}"}

            # 提取血缘信息
            lineage_info = {}
            if model.hyperparameters and "lineage" in model.hyperparameters:
                lineage_info = model.hyperparameters["lineage"]

            result = {
                "model_id": model_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "lineage": lineage_info,
                "ancestors": [],
                "descendants": [],
            }

            # 获取祖先模型
            if include_ancestors:
                ancestors = await self._get_ancestor_models(db, model_id)
                result["ancestors"] = ancestors

            # 获取后代模型
            if include_descendants:
                descendants = await self._get_descendant_models(db, model_id)
                result["descendants"] = descendants

            return result

        except Exception as e:
            logger.error(f"获取模型血缘失败: {e}")
            return {"error": str(e)}

    async def _get_ancestor_models(
        self, db: AsyncSession, model_id: str, visited: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """递归获取祖先模型"""
        if visited is None:
            visited = set()

        if model_id in visited:
            return []  # 避免循环引用

        visited.add(model_id)
        ancestors = []

        try:
            # 获取当前模型
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.model_id == model_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                return ancestors

            # 检查父模型ID
            if model.parent_model_id:
                parent_result = await db.execute(
                    select(ModelInfo).where(ModelInfo.model_id == model.parent_model_id)
                )
                parent_model = parent_result.scalar_one_or_none()

                if parent_model:
                    ancestor_info = {
                        "model_id": parent_model.model_id,
                        "model_name": parent_model.model_name,
                        "model_type": parent_model.model_type,
                        "relationship": "parent",
                    }
                    ancestors.append(ancestor_info)

                    # 递归获取更上层的祖先
                    upper_ancestors = await self._get_ancestor_models(
                        db, parent_model.model_id, visited
                    )
                    ancestors.extend(upper_ancestors)

            # 检查血缘信息中的父模型
            if model.hyperparameters and "lineage" in model.hyperparameters:
                lineage = model.hyperparameters["lineage"]
                parent_models = lineage.get("parent_models", [])

                for parent_id in parent_models:
                    if parent_id not in visited:
                        parent_result = await db.execute(
                            select(ModelInfo).where(ModelInfo.model_id == parent_id)
                        )
                        parent_model = parent_result.scalar_one_or_none()

                        if parent_model:
                            ancestor_info = {
                                "model_id": parent_model.model_id,
                                "model_name": parent_model.model_name,
                                "model_type": parent_model.model_type,
                                "relationship": "lineage_parent",
                            }
                            ancestors.append(ancestor_info)

            return ancestors

        except Exception as e:
            logger.error(f"获取祖先模型失败: {e}")
            return ancestors

    async def _get_descendant_models(
        self, db: AsyncSession, model_id: str
    ) -> List[Dict[str, Any]]:
        """获取后代模型"""
        descendants = []

        try:
            # 查找以当前模型为父模型的模型
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.parent_model_id == model_id)
            )
            child_models = result.scalars().all()

            for child in child_models:
                descendant_info = {
                    "model_id": child.model_id,
                    "model_name": child.model_name,
                    "model_type": child.model_type,
                    "relationship": "child",
                }
                descendants.append(descendant_info)

            # 查找血缘信息中引用当前模型的模型
            all_models_result = await db.execute(select(ModelInfo))
            all_models = all_models_result.scalars().all()

            for model in all_models:
                if model.hyperparameters and "lineage" in model.hyperparameters:
                    lineage = model.hyperparameters["lineage"]
                    parent_models = lineage.get("parent_models", [])

                    if model_id in parent_models:
                        descendant_info = {
                            "model_id": model.model_id,
                            "model_name": model.model_name,
                            "model_type": model.model_type,
                            "relationship": "lineage_child",
                        }
                        descendants.append(descendant_info)

            return descendants

        except Exception as e:
            logger.error(f"获取后代模型失败: {e}")
            return descendants

    async def find_similar_models(
        self,
        data_fingerprint: str,
        config_fingerprint: str,
        threshold: float = 0.8,
        db: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        """
        查找相似的模型

        Args:
            data_fingerprint: 数据指纹
            config_fingerprint: 配置指纹
            threshold: 相似度阈值
            db: 数据库会话

        Returns:
            List[Dict]: 相似模型列表
        """
        if db is None:
            async for session in get_db():
                return await self._find_similar_models_impl(
                    session, data_fingerprint, config_fingerprint, threshold
                )
        else:
            return await self._find_similar_models_impl(
                db, data_fingerprint, config_fingerprint, threshold
            )

    async def _find_similar_models_impl(
        self,
        db: AsyncSession,
        data_fingerprint: str,
        config_fingerprint: str,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """查找相似模型实现"""
        similar_models = []

        try:
            # 获取所有模型
            result = await db.execute(select(ModelInfo))
            models = result.scalars().all()

            for model in models:
                if not model.hyperparameters or "lineage" not in model.hyperparameters:
                    continue

                lineage = model.hyperparameters["lineage"]
                model_data_fp = lineage.get("data_fingerprint", "")
                model_config_fp = lineage.get("config_fingerprint", "")

                # 计算相似度
                data_similarity = self._calculate_fingerprint_similarity(
                    data_fingerprint, model_data_fp
                )
                config_similarity = self._calculate_fingerprint_similarity(
                    config_fingerprint, model_config_fp
                )

                # 综合相似度
                overall_similarity = (data_similarity + config_similarity) / 2

                if overall_similarity >= threshold:
                    similar_info = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_type": model.model_type,
                        "similarity": overall_similarity,
                        "data_similarity": data_similarity,
                        "config_similarity": config_similarity,
                        "performance_metrics": model.performance_metrics,
                    }
                    similar_models.append(similar_info)

            # 按相似度排序
            similar_models.sort(key=lambda x: x["similarity"], reverse=True)

            return similar_models

        except Exception as e:
            logger.error(f"查找相似模型失败: {e}")
            return similar_models

    def _calculate_fingerprint_similarity(self, fp1: str, fp2: str) -> float:
        """计算指纹相似度"""
        if not fp1 or not fp2:
            return 0.0

        if fp1 == fp2:
            return 1.0

        # 简单的字符串相似度计算
        # 可以使用更复杂的算法如编辑距离等
        common_chars = sum(1 for a, b in zip(fp1, fp2) if a == b)
        max_len = max(len(fp1), len(fp2))

        return common_chars / max_len if max_len > 0 else 0.0

    async def get_lineage_graph(
        self, model_ids: Optional[List[str]] = None, db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        获取血缘关系图

        Args:
            model_ids: 模型ID列表，如果为None则获取所有模型
            db: 数据库会话

        Returns:
            Dict: 血缘关系图数据
        """
        if db is None:
            async for session in get_db():
                return await self._get_lineage_graph_impl(session, model_ids)
        else:
            return await self._get_lineage_graph_impl(db, model_ids)

    async def _get_lineage_graph_impl(
        self, db: AsyncSession, model_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """获取血缘关系图实现"""
        try:
            # 获取模型列表
            if model_ids:
                result = await db.execute(
                    select(ModelInfo).where(ModelInfo.model_id.in_(model_ids))
                )
            else:
                result = await db.execute(select(ModelInfo))

            models = result.scalars().all()

            nodes = []
            edges = []

            for model in models:
                # 添加节点
                node = {
                    "id": model.model_id,
                    "label": model.model_name,
                    "type": model.model_type,
                    "status": model.status,
                    "created_at": model.created_at.isoformat()
                    if model.created_at
                    else None,
                }
                nodes.append(node)

                # 添加边（父子关系）
                if model.parent_model_id:
                    edge = {
                        "source": model.parent_model_id,
                        "target": model.model_id,
                        "type": "parent_child",
                        "label": "继承",
                    }
                    edges.append(edge)

                # 添加血缘关系边
                if model.hyperparameters and "lineage" in model.hyperparameters:
                    lineage = model.hyperparameters["lineage"]
                    parent_models = lineage.get("parent_models", [])

                    for parent_id in parent_models:
                        edge = {
                            "source": parent_id,
                            "target": model.model_id,
                            "type": "lineage",
                            "label": "血缘",
                        }
                        edges.append(edge)

            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "total_models": len(nodes),
                    "total_relationships": len(edges),
                    "generated_at": datetime.utcnow().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"获取血缘关系图失败: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}


# 全局实例
lineage_tracker = LineageTracker()
