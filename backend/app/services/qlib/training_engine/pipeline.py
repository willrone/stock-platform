"""Unified Qlib training pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from app.services.data.official_qlib_data_builder import OfficialQlibDataBuilder
from app.services.qlib.official_workflow import (
    OfficialDataset,
    OfficialMarket,
    create_official_dataset_adapter,
    build_official_lightgbm_workflow_config,
)


ProgressCallback = Callable[
    [str, float, str, str, Optional[Dict[str, Any]]], Awaitable[None]
]


@dataclass
class TrainingRequest:
    """训练请求上下文。"""

    model_id: str
    model_name: str
    stock_codes: List[str]
    start_date: datetime
    end_date: datetime
    config: Any
    progress_callback: Optional[ProgressCallback] = None


class QlibTrainingPipeline:
    """训练流程中的可复用管线步骤。"""

    def __init__(self, engine: Any):
        self.engine = engine

    @staticmethod
    def ensure_qlib_available(is_available: bool) -> None:
        """在训练开始前校验 Qlib 可用性。"""
        if is_available:
            return

        raise RuntimeError(
            "Qlib库未安装，无法进行模型训练。\n\n"
            "请按照以下步骤安装Qlib：\n"
            "1. 激活虚拟环境：\n"
            "   cd backend\n"
            "   source venv/bin/activate\n\n"
            "2. 安装Qlib：\n"
            "   pip install git+https://github.com/microsoft/qlib.git\n\n"
            "3. 验证安装：\n"
            "   python -c \"import qlib; print('Qlib安装成功！')\"\n\n"
            "详细安装说明请查看：backend/QLIB_INSTALLATION.md"
        )

    async def initialize_engine(self) -> None:
        """初始化训练引擎。"""
        await self.engine.initialize()

    async def prepare_dataset(self, request: TrainingRequest) -> Any:
        """准备 Qlib 数据集。"""
        workflow_mode = getattr(request.config, "workflow_mode", "enhanced_local")
        if workflow_mode == "official_replication":
            workflow = build_official_lightgbm_workflow_config(
                dataset=OfficialDataset(getattr(request.config, "official_dataset", "alpha158") or "alpha158"),
                market=OfficialMarket(getattr(request.config, "official_market", "csi300") or "csi300"),
            )
            builder = OfficialQlibDataBuilder()
            builder.prepare_stocks(request.stock_codes)
            return create_official_dataset_adapter(
                workflow,
                stock_codes=request.stock_codes,
                provider_uri=builder.official_qlib_data_path,
            )

        return await self.engine.data_provider.prepare_qlib_dataset(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            include_alpha_factors=request.config.use_alpha_factors,
            use_cache=request.config.cache_features,
        )

    @staticmethod
    def log_dataset_overview(dataset: Any) -> None:
        """记录数据集维度和质量摘要。"""
        logger.info("========== 数据集维度信息 ==========")
        if hasattr(dataset, "workflow_config") and hasattr(dataset, "segment_lengths"):
            logger.info(f"数据集形状: {dataset.shape}")
            logger.info(f"样本数: {dataset.shape[0]}")
            logger.info(f"特征数: {dataset.shape[1] if len(dataset.shape) > 1 else 0}")
            logger.info(f"官方数据集: {dataset.workflow_config.dataset.value}")
            logger.info(f"官方市场: {dataset.workflow_config.market}")
            logger.info(f"显式切分: {dataset.segment_lengths}")
            logger.info("=====================================")
            return

        logger.info(f"数据集形状: {dataset.shape}")
        logger.info(f"样本数: {dataset.shape[0]}")
        logger.info(f"特征数: {dataset.shape[1] if len(dataset.shape) > 1 else 0}")
        logger.info(f"数据维度数: {dataset.ndim}")
        if len(dataset.columns) > 0:
            logger.info(f"特征列数: {len(dataset.columns)}")
            logger.info(f"前20个特征列名: {list(dataset.columns[:20])}")
            if len(dataset.columns) > 20:
                logger.info(f"... 还有 {len(dataset.columns) - 20} 个特征列")
        logger.info(f"索引类型: {type(dataset.index).__name__}")
        if isinstance(dataset.index, pd.MultiIndex):
            logger.info(f"MultiIndex级别数: {dataset.index.nlevels}")
            logger.info(f"MultiIndex级别名称: {dataset.index.names}")
        logger.info(f"缺失值总数: {dataset.isnull().sum().sum()}")
        logger.info(f"数据类型统计: {dataset.dtypes.value_counts().to_dict()}")
        logger.info("=====================================")

    async def create_model_config(self, config: Any) -> Dict[str, Any]:
        """构建模型配置。"""
        return await self.engine._create_qlib_model_config(config)

    def analyze_feature_correlations(self, dataset: Any) -> Dict[str, Any]:
        """分析特征相关性。"""
        if hasattr(dataset, "workflow_config") and hasattr(dataset, "segment_lengths"):
            return {
                "mode": "official_replication",
                "dataset": dataset.workflow_config.dataset.value,
                "market": dataset.workflow_config.market,
                "feature_count": dataset.workflow_config.feature_count,
                "segment_lengths": dataset.segment_lengths,
            }
        return self.engine._analyze_feature_correlations(dataset)

    async def prepare_training_datasets(
        self, dataset: Any, validation_split: float, config: Any
    ) -> Tuple[Any, Any]:
        """准备训练/验证数据集。"""
        if hasattr(dataset, "workflow_config") and hasattr(dataset, "for_segment"):
            return dataset.for_segment("train"), dataset.for_segment("valid")
        return await self.engine._prepare_training_datasets(
            dataset,
            validation_split,
            config,
        )

    @staticmethod
    def prepare_test_dataset(dataset: Any) -> Any | None:
        """为支持显式 test 段的数据集构建测试视图。"""
        if hasattr(dataset, "workflow_config") and hasattr(dataset, "for_segment"):
            segment_lengths = getattr(dataset, "segment_lengths", {}) or {}
            if segment_lengths.get("test", 0) > 0:
                return dataset.for_segment("test")
        return None

    async def train(
        self,
        model_config: Dict[str, Any],
        train_dataset: Any,
        val_dataset: Any,
        request: TrainingRequest,
    ) -> Any:
        """执行模型训练。"""
        return await self.engine._train_qlib_model(
            model_config,
            train_dataset,
            val_dataset,
            request.config,
            request.progress_callback,
            request.model_id,
        )

    async def evaluate(
        self,
        model: Any,
        train_dataset: Any,
        val_dataset: Any,
        model_id: str,
        test_dataset: Any | None = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        """评估训练后模型。"""
        return await self.engine._evaluate_model(
            model,
            train_dataset,
            val_dataset,
            model_id,
            test_dataset=test_dataset,
        )

    async def extract_feature_importance(self, model: Any, model_type: Any) -> Dict[str, float]:
        """提取特征重要性。"""
        return await self.engine._extract_feature_importance(model, model_type)

    async def save_model(self, model: Any, model_id: str, model_config: Dict[str, Any]) -> str:
        """落盘模型产物。"""
        return await self.engine._save_qlib_model(model, model_id, model_config)
