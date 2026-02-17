"""
SHAP解释性库集成
实现模型预测解释功能，支持全局和局部解释性分析
"""
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


class ExplainerType(Enum):
    """解释器类型"""

    TREE = "tree"  # 树模型解释器
    LINEAR = "linear"  # 线性模型解释器
    KERNEL = "kernel"  # 核解释器
    DEEP = "deep"  # 深度学习解释器
    PERMUTATION = "permutation"  # 排列解释器
    PARTITION = "partition"  # 分区解释器


class ExplanationType(Enum):
    """解释类型"""

    LOCAL = "local"  # 局部解释
    GLOBAL = "global"  # 全局解释
    COHORT = "cohort"  # 群体解释


@dataclass
class ShapValues:
    """SHAP值"""

    values: np.ndarray  # SHAP值数组
    base_value: Union[float, np.ndarray]  # 基准值
    data: Optional[np.ndarray] = None  # 输入数据
    feature_names: Optional[List[str]] = None  # 特征名称

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist()
            if isinstance(self.values, np.ndarray)
            else self.values,
            "base_value": float(self.base_value)
            if isinstance(self.base_value, (int, float, np.number))
            else self.base_value.tolist(),
            "data": self.data.tolist() if self.data is not None else None,
            "feature_names": self.feature_names,
        }


@dataclass
class ExplanationResult:
    """解释结果"""

    explanation_id: str
    model_id: str
    model_version: str
    explanation_type: ExplanationType
    explainer_type: ExplainerType
    shap_values: ShapValues
    feature_importance: Dict[str, float]
    summary_stats: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "explanation_type": self.explanation_type.value,
            "explainer_type": self.explainer_type.value,
            "shap_values": self.shap_values.to_dict(),
            "feature_importance": self.feature_importance,
            "summary_stats": self.summary_stats,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class ShapExplainerFactory:
    """SHAP解释器工厂"""

    @staticmethod
    def create_explainer(
        model: Any,
        explainer_type: ExplainerType,
        background_data: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        创建SHAP解释器

        Args:
            model: 模型对象
            explainer_type: 解释器类型
            background_data: 背景数据
            **kwargs: 其他参数

        Returns:
            SHAP解释器
        """
        try:
            import shap
        except ImportError:
            raise ImportError("需要安装SHAP库: pip install shap")

        if explainer_type == ExplainerType.TREE:
            return shap.TreeExplainer(model, **kwargs)

        elif explainer_type == ExplainerType.LINEAR:
            return shap.LinearExplainer(model, background_data, **kwargs)

        elif explainer_type == ExplainerType.KERNEL:
            if background_data is None:
                raise ValueError("核解释器需要背景数据")
            return shap.KernelExplainer(model, background_data, **kwargs)

        elif explainer_type == ExplainerType.DEEP:
            if background_data is None:
                raise ValueError("深度解释器需要背景数据")
            return shap.DeepExplainer(model, background_data, **kwargs)

        elif explainer_type == ExplainerType.PERMUTATION:
            return shap.PermutationExplainer(model.predict, background_data, **kwargs)

        elif explainer_type == ExplainerType.PARTITION:
            return shap.PartitionExplainer(model.predict, background_data, **kwargs)

        else:
            raise ValueError(f"不支持的解释器类型: {explainer_type}")


class ModelExplainer:
    """模型解释器"""

    def __init__(
        self,
        model: Any,
        explainer_type: ExplainerType,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
    ):
        """
        初始化模型解释器

        Args:
            model: 模型对象
            explainer_type: 解释器类型
            feature_names: 特征名称
            background_data: 背景数据
        """
        self.model = model
        self.explainer_type = explainer_type
        self.feature_names = feature_names
        self.background_data = background_data

        # 创建SHAP解释器
        self.explainer = ShapExplainerFactory.create_explainer(
            model, explainer_type, background_data
        )

        logger.info(f"创建模型解释器: {explainer_type.value}")

    def explain_instance(self, instance: np.ndarray) -> ShapValues:
        """
        解释单个实例

        Args:
            instance: 输入实例

        Returns:
            SHAP值
        """
        try:
            # 确保输入是二维数组
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)

            # 计算SHAP值
            shap_values = self.explainer.shap_values(instance)

            # 处理多类分类的情况
            if isinstance(shap_values, list):
                # 对于多类分类，取第一类的SHAP值
                shap_values = shap_values[0]

            # 获取基准值
            if hasattr(self.explainer, "expected_value"):
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
            else:
                base_value = 0.0

            return ShapValues(
                values=shap_values,
                base_value=base_value,
                data=instance,
                feature_names=self.feature_names,
            )

        except Exception as e:
            logger.error(f"解释实例失败: {e}")
            raise

    def explain_batch(
        self, batch_data: np.ndarray, max_samples: int = 100
    ) -> ShapValues:
        """
        批量解释

        Args:
            batch_data: 批量数据
            max_samples: 最大样本数

        Returns:
            SHAP值
        """
        try:
            # 限制样本数量
            if len(batch_data) > max_samples:
                indices = np.random.choice(len(batch_data), max_samples, replace=False)
                batch_data = batch_data[indices]

            # 计算SHAP值
            shap_values = self.explainer.shap_values(batch_data)

            # 处理多类分类的情况
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # 获取基准值
            if hasattr(self.explainer, "expected_value"):
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
            else:
                base_value = 0.0

            return ShapValues(
                values=shap_values,
                base_value=base_value,
                data=batch_data,
                feature_names=self.feature_names,
            )

        except Exception as e:
            logger.error(f"批量解释失败: {e}")
            raise

    def get_feature_importance(self, shap_values: ShapValues) -> Dict[str, float]:
        """
        获取特征重要性

        Args:
            shap_values: SHAP值

        Returns:
            特征重要性字典
        """
        # 计算平均绝对SHAP值作为特征重要性
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

        if self.feature_names:
            return dict(zip(self.feature_names, mean_abs_shap))
        else:
            return {
                f"feature_{i}": importance for i, importance in enumerate(mean_abs_shap)
            }

    def get_summary_stats(self, shap_values: ShapValues) -> Dict[str, Any]:
        """
        获取摘要统计

        Args:
            shap_values: SHAP值

        Returns:
            摘要统计
        """
        values = shap_values.values

        stats = {
            "mean_abs_shap": float(np.mean(np.abs(values))),
            "max_abs_shap": float(np.max(np.abs(values))),
            "min_abs_shap": float(np.min(np.abs(values))),
            "std_shap": float(np.std(values)),
            "total_attribution": float(np.sum(values)),
            "positive_attribution": float(np.sum(values[values > 0])),
            "negative_attribution": float(np.sum(values[values < 0])),
            "num_features": values.shape[1] if values.ndim > 1 else len(values),
            "num_samples": values.shape[0] if values.ndim > 1 else 1,
        }

        return stats


class ShapExplainerManager:
    """SHAP解释器管理器"""

    def __init__(self, storage_path: str = "data/explanations"):
        """
        初始化解释器管理器

        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 解释器缓存
        self.explainers: Dict[str, ModelExplainer] = {}

        # 解释结果存储
        self.explanations: Dict[str, ExplanationResult] = {}
        self.max_explanations = 10000

        # 线程锁
        self.lock = threading.Lock()

        logger.info(f"SHAP解释器管理器初始化完成，存储路径: {self.storage_path}")

    def register_model_explainer(
        self,
        model_id: str,
        model: Any,
        explainer_type: ExplainerType,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
    ) -> str:
        """
        注册模型解释器

        Args:
            model_id: 模型ID
            model: 模型对象
            explainer_type: 解释器类型
            feature_names: 特征名称
            background_data: 背景数据

        Returns:
            解释器ID
        """
        explainer_id = f"{model_id}_{explainer_type.value}"

        try:
            explainer = ModelExplainer(
                model=model,
                explainer_type=explainer_type,
                feature_names=feature_names,
                background_data=background_data,
            )

            with self.lock:
                self.explainers[explainer_id] = explainer

            logger.info(f"注册模型解释器: {explainer_id}")
            return explainer_id

        except Exception as e:
            logger.error(f"注册模型解释器失败: {e}")
            raise

    def explain_prediction(
        self,
        model_id: str,
        model_version: str,
        input_data: np.ndarray,
        explainer_type: ExplainerType = ExplainerType.KERNEL,
        explanation_type: ExplanationType = ExplanationType.LOCAL,
    ) -> str:
        """
        解释预测

        Args:
            model_id: 模型ID
            model_version: 模型版本
            input_data: 输入数据
            explainer_type: 解释器类型
            explanation_type: 解释类型

        Returns:
            解释ID
        """
        explainer_id = f"{model_id}_{explainer_type.value}"

        if explainer_id not in self.explainers:
            raise ValueError(f"解释器不存在: {explainer_id}")

        explainer = self.explainers[explainer_id]

        try:
            # 根据解释类型选择方法
            if explanation_type == ExplanationType.LOCAL:
                if input_data.ndim == 1:
                    shap_values = explainer.explain_instance(input_data)
                else:
                    # 对于批量数据，只解释第一个实例
                    shap_values = explainer.explain_instance(input_data[0])
            else:
                # 全局或群体解释
                shap_values = explainer.explain_batch(input_data)

            # 计算特征重要性
            feature_importance = explainer.get_feature_importance(shap_values)

            # 计算摘要统计
            summary_stats = explainer.get_summary_stats(shap_values)

            # 创建解释结果
            explanation_id = f"exp_{int(datetime.now().timestamp())}_{model_id}"

            explanation_result = ExplanationResult(
                explanation_id=explanation_id,
                model_id=model_id,
                model_version=model_version,
                explanation_type=explanation_type,
                explainer_type=explainer_type,
                shap_values=shap_values,
                feature_importance=feature_importance,
                summary_stats=summary_stats,
                created_at=datetime.now(),
                metadata={
                    "input_shape": input_data.shape,
                    "num_features": len(feature_importance),
                },
            )

            # 存储解释结果
            with self.lock:
                self.explanations[explanation_id] = explanation_result

                # 限制存储数量
                if len(self.explanations) > self.max_explanations:
                    # 删除最旧的解释
                    oldest_id = min(
                        self.explanations.keys(),
                        key=lambda x: self.explanations[x].created_at,
                    )
                    del self.explanations[oldest_id]

            # 保存解释结果
            self._save_explanation(explanation_result)

            logger.info(f"生成解释: {explanation_id}")
            return explanation_id

        except Exception as e:
            logger.error(f"解释预测失败: {e}")
            raise

    def get_explanation(self, explanation_id: str) -> Optional[ExplanationResult]:
        """获取解释结果"""
        return self.explanations.get(explanation_id)

    def get_model_explanations(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        explanation_type: Optional[ExplanationType] = None,
        limit: int = 100,
    ) -> List[ExplanationResult]:
        """
        获取模型的解释结果

        Args:
            model_id: 模型ID
            model_version: 模型版本
            explanation_type: 解释类型
            limit: 限制数量

        Returns:
            解释结果列表
        """
        explanations = []

        for explanation in self.explanations.values():
            if explanation.model_id != model_id:
                continue

            if model_version and explanation.model_version != model_version:
                continue

            if explanation_type and explanation.explanation_type != explanation_type:
                continue

            explanations.append(explanation)

        # 按创建时间排序
        explanations.sort(key=lambda x: x.created_at, reverse=True)

        return explanations[:limit]

    def get_feature_importance_ranking(
        self, model_id: str, model_version: Optional[str] = None, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        获取特征重要性排名

        Args:
            model_id: 模型ID
            model_version: 模型版本
            top_k: 返回前K个特征

        Returns:
            特征重要性排名
        """
        explanations = self.get_model_explanations(model_id, model_version)

        if not explanations:
            return []

        # 聚合特征重要性
        feature_importance_sum = {}
        feature_count = {}

        for explanation in explanations:
            for feature, importance in explanation.feature_importance.items():
                if feature not in feature_importance_sum:
                    feature_importance_sum[feature] = 0
                    feature_count[feature] = 0

                feature_importance_sum[feature] += importance
                feature_count[feature] += 1

        # 计算平均重要性
        avg_importance = {}
        for feature in feature_importance_sum:
            avg_importance[feature] = (
                feature_importance_sum[feature] / feature_count[feature]
            )

        # 排序并返回前K个
        sorted_features = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:top_k]

    def generate_explanation_report(
        self, model_id: str, model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成解释报告

        Args:
            model_id: 模型ID
            model_version: 模型版本

        Returns:
            解释报告
        """
        explanations = self.get_model_explanations(model_id, model_version)

        if not explanations:
            return {"error": "没有找到解释结果"}

        # 基本统计
        total_explanations = len(explanations)
        explanation_types = {}
        explainer_types = {}

        for explanation in explanations:
            exp_type = explanation.explanation_type.value
            explanation_types[exp_type] = explanation_types.get(exp_type, 0) + 1

            explainer_type = explanation.explainer_type.value
            explainer_types[explainer_type] = explainer_types.get(explainer_type, 0) + 1

        # 特征重要性排名
        feature_ranking = self.get_feature_importance_ranking(
            model_id, model_version, top_k=20
        )

        # 摘要统计
        summary_stats = {
            "mean_abs_shap": [],
            "total_attribution": [],
            "positive_attribution": [],
            "negative_attribution": [],
        }

        for explanation in explanations:
            stats = explanation.summary_stats
            for key in summary_stats:
                if key in stats:
                    summary_stats[key].append(stats[key])

        # 计算平均值
        avg_stats = {}
        for key, values in summary_stats.items():
            if values:
                avg_stats[f"avg_{key}"] = np.mean(values)
                avg_stats[f"std_{key}"] = np.std(values)

        return {
            "model_id": model_id,
            "model_version": model_version,
            "total_explanations": total_explanations,
            "explanation_types": explanation_types,
            "explainer_types": explainer_types,
            "top_features": feature_ranking,
            "summary_statistics": avg_stats,
            "generated_at": datetime.now().isoformat(),
        }

    def _save_explanation(self, explanation: ExplanationResult):
        """保存解释结果"""
        try:
            explanation_file = self.storage_path / f"{explanation.explanation_id}.json"

            # 由于SHAP值可能很大，只保存摘要信息
            summary_data = {
                "explanation_id": explanation.explanation_id,
                "model_id": explanation.model_id,
                "model_version": explanation.model_version,
                "explanation_type": explanation.explanation_type.value,
                "explainer_type": explanation.explainer_type.value,
                "feature_importance": explanation.feature_importance,
                "summary_stats": explanation.summary_stats,
                "created_at": explanation.created_at.isoformat(),
                "metadata": explanation.metadata,
            }

            with open(explanation_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存解释结果失败: {e}")

    def cleanup_old_explanations(self, days: int = 30):
        """清理旧的解释结果"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)

        with self.lock:
            to_remove = []
            for explanation_id, explanation in self.explanations.items():
                if explanation.created_at < cutoff_date:
                    to_remove.append(explanation_id)

            for explanation_id in to_remove:
                del self.explanations[explanation_id]

                # 删除文件
                explanation_file = self.storage_path / f"{explanation_id}.json"
                if explanation_file.exists():
                    explanation_file.unlink()

            logger.info(f"清理了 {len(to_remove)} 个旧解释结果")


# 全局SHAP解释器管理器实例
shap_explainer_manager = ShapExplainerManager()
