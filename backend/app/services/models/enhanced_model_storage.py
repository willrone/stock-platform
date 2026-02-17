"""
增强的模型存储服务

集成生命周期管理和血缘追踪功能，提供：
- 模型搜索和标签功能
- 生命周期管理集成
- 血缘追踪集成
- 模型版本管理
- 性能对比分析
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_session as get_db
from ...models.task_models import ModelInfo
from .lineage_tracker import lineage_tracker
from .model_lifecycle_manager import ModelStatus, model_lifecycle_manager


class EnhancedModelStorage:
    """增强的模型存储服务"""

    def __init__(self):
        self.lifecycle_manager = model_lifecycle_manager
        self.lineage_tracker = lineage_tracker

    async def create_model_with_lineage(
        self,
        model_info: Dict[str, Any],
        training_config: Dict[str, Any],
        data_sources: List[Dict[str, Any]],
        feature_config: Optional[Dict[str, Any]] = None,
        parent_models: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        创建模型并记录血缘信息

        Args:
            model_info: 模型基本信息
            training_config: 训练配置
            data_sources: 数据源信息
            feature_config: 特征工程配置
            parent_models: 父模型列表
            tags: 模型标签
            db: 数据库会话

        Returns:
            Dict: 创建结果
        """
        if db is None:
            async for session in get_db():
                return await self._create_model_with_lineage_impl(
                    session,
                    model_info,
                    training_config,
                    data_sources,
                    feature_config,
                    parent_models,
                    tags,
                )
        else:
            return await self._create_model_with_lineage_impl(
                db,
                model_info,
                training_config,
                data_sources,
                feature_config,
                parent_models,
                tags,
            )

    async def _create_model_with_lineage_impl(
        self,
        db: AsyncSession,
        model_info: Dict[str, Any],
        training_config: Dict[str, Any],
        data_sources: List[Dict[str, Any]],
        feature_config: Optional[Dict[str, Any]],
        parent_models: Optional[List[str]],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        """创建模型并记录血缘信息实现"""
        try:
            # 准备模型数据
            model_data = {
                **model_info,
                "status": ModelStatus.CREATING.value,
                "created_at": datetime.utcnow(),
            }

            # 添加标签到超参数中
            if tags:
                hyperparams = model_data.get("hyperparameters", {})
                hyperparams["tags"] = tags
                model_data["hyperparameters"] = hyperparams

            # 创建模型记录
            model = ModelInfo(**model_data)
            db.add(model)
            await db.flush()  # 获取模型ID

            # 记录血缘信息
            lineage_success = await self.lineage_tracker.record_training_lineage(
                model.model_id,
                training_config,
                data_sources,
                feature_config,
                parent_models,
                db,
            )

            if not lineage_success:
                logger.warning(f"血缘信息记录失败，但模型创建继续: {model.model_id}")

            # 记录生命周期事件
            await self.lifecycle_manager.transition_status(
                model.model_id,
                ModelStatus.TRAINING.value,
                "开始训练",
                {
                    "training_config": training_config,
                    "data_sources_count": len(data_sources),
                    "has_parent_models": bool(parent_models),
                },
                db,
            )

            await db.commit()

            result = {
                "success": True,
                "model_id": model.model_id,
                "model_name": model.model_name,
                "status": model.status,
                "lineage_recorded": lineage_success,
            }

            logger.info(f"模型创建成功: {model.model_id}")
            return result

        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            await db.rollback()
            return {"success": False, "error": str(e)}

    async def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        status_list: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        performance_threshold: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        搜索模型

        Args:
            query: 搜索关键词
            tags: 标签过滤
            model_types: 模型类型过滤
            status_list: 状态过滤
            date_range: 日期范围过滤
            performance_threshold: 性能阈值过滤
            limit: 返回数量限制
            offset: 偏移量
            db: 数据库会话

        Returns:
            Dict: 搜索结果
        """
        if db is None:
            async for session in get_db():
                return await self._search_models_impl(
                    session,
                    query,
                    tags,
                    model_types,
                    status_list,
                    date_range,
                    performance_threshold,
                    limit,
                    offset,
                )
        else:
            return await self._search_models_impl(
                db,
                query,
                tags,
                model_types,
                status_list,
                date_range,
                performance_threshold,
                limit,
                offset,
            )

    async def _search_models_impl(
        self,
        db: AsyncSession,
        query: Optional[str],
        tags: Optional[List[str]],
        model_types: Optional[List[str]],
        status_list: Optional[List[str]],
        date_range: Optional[Tuple[datetime, datetime]],
        performance_threshold: Optional[float],
        limit: int,
        offset: int,
    ) -> Dict[str, Any]:
        """搜索模型实现"""
        try:
            # 构建查询条件
            conditions = []

            # 关键词搜索
            if query:
                conditions.append(
                    or_(
                        ModelInfo.model_name.ilike(f"%{query}%"),
                        ModelInfo.model_type.ilike(f"%{query}%"),
                    )
                )

            # 模型类型过滤
            if model_types:
                conditions.append(ModelInfo.model_type.in_(model_types))

            # 状态过滤
            if status_list:
                conditions.append(ModelInfo.status.in_(status_list))

            # 日期范围过滤
            if date_range:
                start_date, end_date = date_range
                conditions.append(
                    and_(
                        ModelInfo.created_at >= start_date,
                        ModelInfo.created_at <= end_date,
                    )
                )

            # 构建基础查询
            base_query = select(ModelInfo)
            if conditions:
                base_query = base_query.where(and_(*conditions))

            # 获取总数
            count_query = select(func.count(ModelInfo.model_id))
            if conditions:
                count_query = count_query.where(and_(*conditions))

            total_result = await db.execute(count_query)
            total_count = total_result.scalar()

            # 获取模型列表
            models_query = (
                base_query.order_by(desc(ModelInfo.created_at))
                .limit(limit)
                .offset(offset)
            )
            models_result = await db.execute(models_query)
            models = models_result.scalars().all()

            # 过滤结果
            filtered_models = []
            for model in models:
                # 标签过滤
                if tags:
                    model_tags = []
                    if model.hyperparameters and "tags" in model.hyperparameters:
                        model_tags = model.hyperparameters["tags"]

                    if not any(tag in model_tags for tag in tags):
                        continue

                # 性能阈值过滤
                if performance_threshold is not None:
                    if not model.performance_metrics:
                        continue

                    accuracy = model.performance_metrics.get("accuracy", 0)
                    if accuracy < performance_threshold:
                        continue

                # 构建返回数据
                model_data = model.to_dict()

                # 添加标签信息
                if model.hyperparameters and "tags" in model.hyperparameters:
                    model_data["tags"] = model.hyperparameters["tags"]
                else:
                    model_data["tags"] = []

                # 添加血缘信息摘要
                if model.hyperparameters and "lineage" in model.hyperparameters:
                    lineage = model.hyperparameters["lineage"]
                    model_data["lineage_summary"] = {
                        "has_parents": bool(lineage.get("parent_models", [])),
                        "data_sources_count": len(lineage.get("data_sources", [])),
                        "data_fingerprint": lineage.get("data_fingerprint", ""),
                        "config_fingerprint": lineage.get("config_fingerprint", ""),
                    }

                filtered_models.append(model_data)

            return {
                "models": filtered_models,
                "total_count": total_count,
                "filtered_count": len(filtered_models),
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(filtered_models) < total_count,
            }

        except Exception as e:
            logger.error(f"搜索模型失败: {e}")
            return {
                "models": [],
                "total_count": 0,
                "filtered_count": 0,
                "error": str(e),
            }

    async def get_model_with_lineage(
        self,
        model_id: str,
        include_ancestors: bool = True,
        include_descendants: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        获取模型详情及血缘信息

        Args:
            model_id: 模型ID
            include_ancestors: 是否包含祖先模型
            include_descendants: 是否包含后代模型
            db: 数据库会话

        Returns:
            Dict: 模型详情和血缘信息
        """
        if db is None:
            async for session in get_db():
                return await self._get_model_with_lineage_impl(
                    session, model_id, include_ancestors, include_descendants
                )
        else:
            return await self._get_model_with_lineage_impl(
                db, model_id, include_ancestors, include_descendants
            )

    async def _get_model_with_lineage_impl(
        self,
        db: AsyncSession,
        model_id: str,
        include_ancestors: bool,
        include_descendants: bool,
    ) -> Dict[str, Any]:
        """获取模型详情及血缘信息实现"""
        try:
            # 获取模型基本信息
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.model_id == model_id)
            )
            model = result.scalar_one_or_none()

            if not model:
                return {"error": f"模型不存在: {model_id}"}

            # 构建基本信息
            model_data = model.to_dict()

            # 添加标签信息
            if model.hyperparameters and "tags" in model.hyperparameters:
                model_data["tags"] = model.hyperparameters["tags"]
            else:
                model_data["tags"] = []

            # 获取生命周期历史
            lifecycle_history = await self.lifecycle_manager.get_lifecycle_history(
                model_id, limit=20, db=db
            )
            model_data["lifecycle_history"] = lifecycle_history

            # 获取血缘信息
            lineage_info = await self.lineage_tracker.get_model_lineage(
                model_id, include_ancestors, include_descendants, db
            )
            model_data["lineage"] = lineage_info

            return model_data

        except Exception as e:
            logger.error(f"获取模型详情失败: {e}")
            return {"error": str(e)}

    async def compare_models(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        对比多个模型

        Args:
            model_ids: 模型ID列表
            metrics: 要对比的指标列表
            db: 数据库会话

        Returns:
            Dict: 对比结果
        """
        if db is None:
            async for session in get_db():
                return await self._compare_models_impl(session, model_ids, metrics)
        else:
            return await self._compare_models_impl(db, model_ids, metrics)

    async def _compare_models_impl(
        self, db: AsyncSession, model_ids: List[str], metrics: Optional[List[str]]
    ) -> Dict[str, Any]:
        """对比多个模型实现"""
        try:
            # 获取模型信息
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.model_id.in_(model_ids))
            )
            models = result.scalars().all()

            if len(models) != len(model_ids):
                found_ids = [m.model_id for m in models]
                missing_ids = [mid for mid in model_ids if mid not in found_ids]
                return {"error": f"模型不存在: {missing_ids}"}

            # 默认对比指标
            if not metrics:
                metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]

            comparison_data = {"models": [], "metrics_comparison": {}, "summary": {}}

            # 收集模型数据
            for model in models:
                model_info = {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "status": model.status,
                    "created_at": model.created_at.isoformat()
                    if model.created_at
                    else None,
                    "performance_metrics": model.performance_metrics or {},
                }

                # 添加标签
                if model.hyperparameters and "tags" in model.hyperparameters:
                    model_info["tags"] = model.hyperparameters["tags"]

                comparison_data["models"].append(model_info)

            # 对比指标
            for metric in metrics:
                metric_values = []
                for model in models:
                    if (
                        model.performance_metrics
                        and metric in model.performance_metrics
                    ):
                        metric_values.append(
                            {
                                "model_id": model.model_id,
                                "model_name": model.model_name,
                                "value": model.performance_metrics[metric],
                            }
                        )

                if metric_values:
                    # 排序
                    metric_values.sort(key=lambda x: x["value"], reverse=True)

                    comparison_data["metrics_comparison"][metric] = {
                        "values": metric_values,
                        "best": metric_values[0] if metric_values else None,
                        "worst": metric_values[-1] if metric_values else None,
                        "average": sum(v["value"] for v in metric_values)
                        / len(metric_values)
                        if metric_values
                        else 0,
                    }

            # 生成摘要
            if comparison_data["metrics_comparison"]:
                best_overall = {}
                for metric, data in comparison_data["metrics_comparison"].items():
                    if data["best"]:
                        model_id = data["best"]["model_id"]
                        if model_id not in best_overall:
                            best_overall[model_id] = 0
                        best_overall[model_id] += 1

                if best_overall:
                    best_model_id = max(best_overall, key=best_overall.get)
                    best_model = next(m for m in models if m.model_id == best_model_id)

                    comparison_data["summary"] = {
                        "best_overall_model": {
                            "model_id": best_model.model_id,
                            "model_name": best_model.model_name,
                            "wins_count": best_overall[best_model_id],
                            "total_metrics": len(comparison_data["metrics_comparison"]),
                        },
                        "total_models": len(models),
                        "compared_metrics": len(comparison_data["metrics_comparison"]),
                    }

            return comparison_data

        except Exception as e:
            logger.error(f"模型对比失败: {e}")
            return {"error": str(e)}

    async def get_model_recommendations(
        self,
        training_config: Dict[str, Any],
        data_sources: List[Dict[str, Any]],
        limit: int = 5,
        db: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取模型推荐

        Args:
            training_config: 训练配置
            data_sources: 数据源信息
            limit: 推荐数量限制
            db: 数据库会话

        Returns:
            List[Dict]: 推荐模型列表
        """
        if db is None:
            async for session in get_db():
                return await self._get_model_recommendations_impl(
                    session, training_config, data_sources, limit
                )
        else:
            return await self._get_model_recommendations_impl(
                db, training_config, data_sources, limit
            )

    async def _get_model_recommendations_impl(
        self,
        db: AsyncSession,
        training_config: Dict[str, Any],
        data_sources: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """获取模型推荐实现"""
        try:
            # 计算当前配置的指纹
            data_fingerprint = self.lineage_tracker._calculate_data_fingerprint(
                data_sources
            )
            config_fingerprint = self.lineage_tracker._calculate_config_fingerprint(
                training_config
            )

            # 查找相似模型
            similar_models = await self.lineage_tracker.find_similar_models(
                data_fingerprint, config_fingerprint, threshold=0.6, db=db
            )

            # 过滤并排序推荐
            recommendations = []
            for model in similar_models[:limit]:
                if model.get("performance_metrics"):
                    recommendation = {
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        "model_type": model["model_type"],
                        "similarity": model["similarity"],
                        "performance_metrics": model["performance_metrics"],
                        "recommendation_reason": self._generate_recommendation_reason(
                            model
                        ),
                    }
                    recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"获取模型推荐失败: {e}")
            return []

    def _generate_recommendation_reason(self, model: Dict[str, Any]) -> str:
        """生成推荐理由"""
        reasons = []

        similarity = model.get("similarity", 0)
        if similarity > 0.9:
            reasons.append("配置高度相似")
        elif similarity > 0.8:
            reasons.append("配置较为相似")
        else:
            reasons.append("配置部分相似")

        performance = model.get("performance_metrics", {})
        accuracy = performance.get("accuracy", 0)
        if accuracy > 0.9:
            reasons.append("历史表现优秀")
        elif accuracy > 0.8:
            reasons.append("历史表现良好")

        return "，".join(reasons)


# 全局实例
enhanced_model_storage = EnhancedModelStorage()
