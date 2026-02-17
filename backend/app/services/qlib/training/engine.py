"""
统一Qlib训练引擎 - 主引擎类

这是重构后的主入口，保持向后兼容
"""

from datetime import datetime
from typing import List

import pandas as pd
from loguru import logger

from ..enhanced_qlib_provider import EnhancedQlibDataProvider
from ..performance_monitor import get_performance_monitor
from ..qlib_model_manager import QlibModelManager
from .config import QlibTrainingConfig, QlibTrainingResult
from .dataset_preparation import prepare_training_datasets
from .evaluation import _evaluate_model
from .feature_analysis import analyze_feature_correlations, extract_feature_importance
from .model_config import create_qlib_model_config
from .model_io import load_qlib_model, save_qlib_model

# prediction 功能已集成到类方法中
from .qlib_check import QLIB_AVAILABLE
from .rolling_integration import run_rolling_training
from .training import _train_qlib_model
from .utility import (
    get_model_config_template,
    get_supported_model_types,
    get_training_recommendations,
    recommend_models,
)

# LightGBM / XGBoost 二分类目标函数映射
_BINARY_OBJECTIVE_MAP = {
    "lightgbm": {"objective": "binary", "metric": "binary_logloss"},
    "xgboost": {"objective": "binary:logistic", "eval_metric": "logloss"},
}


def _apply_binary_objective(
    model_config: dict,
    config: QlibTrainingConfig,
) -> dict:
    """当 label_type=binary 时，自动将模型 objective 切换为二分类

    Args:
        model_config: Qlib 模型配置字典
        config: 训练配置

    Returns:
        修改后的模型配置
    """
    model_type = config.model_type.value
    overrides = _BINARY_OBJECTIVE_MAP.get(model_type)
    if not overrides:
        logger.info(
            f"模型 {model_type} 无需切换二分类 objective",
        )
        return model_config

    # Qlib 模型配置中 kwargs 存放超参数
    kwargs = model_config.get("kwargs", {})
    kwargs.update(overrides)
    model_config["kwargs"] = kwargs
    logger.info(
        f"二分类模式: {model_type} objective 已切换为 {overrides}",
    )
    return model_config


class UnifiedQlibTrainingEngine:
    """统一Qlib训练引擎"""

    def __init__(self, websocket_manager=None):
        self.websocket_manager = websocket_manager
        self.data_provider = EnhancedQlibDataProvider()
        self.model_manager = QlibModelManager()
        self.early_stopping_manager = None
        self.performance_monitor = get_performance_monitor()

        logger.info("统一Qlib训练引擎初始化完成")

    async def initialize(self):
        """初始化训练引擎"""
        try:
            # 初始化Qlib环境
            await self.data_provider.initialize_qlib()
            logger.info("Qlib训练引擎初始化成功")
        except Exception as e:
            logger.error(f"Qlib训练引擎初始化失败: {e}")
            raise

    async def train_model(
        self,
        model_id: str,
        model_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: QlibTrainingConfig,
        progress_callback=None,
    ) -> QlibTrainingResult:
        """
        统一的Qlib模型训练流程

        Args:
            model_id: 模型唯一标识
            model_name: 模型名称
            stock_codes: 训练用的股票代码列表
            start_date: 训练数据开始日期
            end_date: 训练数据结束日期
            config: 训练配置
            progress_callback: 进度回调函数

        Returns:
            训练结果
        """
        logger.info(f"开始Qlib统一训练流程: {model_id}, 模型类型: {config.model_type.value}")
        start_time = datetime.now()

        # 开始整体性能监控
        self.performance_monitor.start_stage("total_training")

        try:
            # 0. 检查Qlib是否可用
            if not QLIB_AVAILABLE:
                error_msg = (
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
                logger.error(error_msg)
                if progress_callback:
                    await progress_callback(model_id, 0.0, "failed", error_msg)
                raise RuntimeError(error_msg)

            # 1. 初始化Qlib环境
            if progress_callback:
                await progress_callback(model_id, 5.0, "initializing", "初始化Qlib环境")

            self.performance_monitor.start_stage("initialize_qlib")
            await self.initialize()
            self.performance_monitor.end_stage("initialize_qlib")

            # 2. 准备数据集（根据特征集选择不同路径）
            feature_set = config.feature_set
            skip_alpha = feature_set in ("technical_62", "custom")
            if progress_callback:
                await progress_callback(
                    model_id,
                    15.0,
                    "preparing",
                    f"准备数据集 (特征集: {feature_set})",
                    {
                        "stock_count": len(stock_codes),
                        "date_range": f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}",
                        "feature_set": feature_set,
                        "label_type": config.label_type,
                        "split_method": config.split_method,
                    },
                )

            self.performance_monitor.start_stage("prepare_dataset")
            dataset = await self.data_provider.prepare_qlib_dataset(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                include_alpha_factors=config.use_alpha_factors and not skip_alpha,
                use_cache=config.cache_features,
            )
            self.performance_monitor.end_stage("prepare_dataset")

            if dataset.empty:
                raise ValueError("无法获取训练数据")

            # 记录数据集维度信息
            n_samples, n_features = (
                dataset.shape[0],
                dataset.shape[1] if dataset.ndim > 1 else 0,
            )
            logger.info(
                f"数据集: {n_samples} 样本, {n_features} 特征, "
                f"索引={type(dataset.index).__name__}, "
                f"缺失值={dataset.isnull().sum().sum()}"
            )

            # === 滚动训练分支（P2） ===
            if config.enable_rolling:
                return await self._run_rolling_training(
                    dataset=dataset,
                    config=config,
                    model_id=model_id,
                    start_time=start_time,
                    progress_callback=progress_callback,
                )

            # 3. 创建Qlib模型配置
            if progress_callback:
                await progress_callback(
                    model_id,
                    25.0,
                    "configuring",
                    "配置Qlib模型",
                    {
                        "dataset_shape": list(dataset.shape),
                        "features_count": dataset.shape[1]
                        if len(dataset.shape) > 1
                        else 0,
                        "sample_count": dataset.shape[0],
                    },
                )

            self.performance_monitor.start_stage("create_model_config")
            model_config = await create_qlib_model_config(self.model_manager, config)
            self.performance_monitor.end_stage("create_model_config")

            # 二分类时自动切换 LightGBM/XGBoost 的 loss
            if config.label_type == "binary":
                model_config = _apply_binary_objective(model_config, config)

            self.performance_monitor.start_stage("analyze_features")
            feature_correlation = analyze_feature_correlations(dataset)
            self.performance_monitor.end_stage("analyze_features")

            # 4. 数据预处理和分割
            if progress_callback:
                await progress_callback(
                    model_id,
                    35.0,
                    "preprocessing",
                    "数据预处理",
                    {"validation_split": config.validation_split},
                )

            self.performance_monitor.start_stage("prepare_training_datasets")
            train_dataset, val_dataset = await prepare_training_datasets(
                dataset, config.validation_split, config
            )
            self.performance_monitor.end_stage("prepare_training_datasets")

            # 记录数据集分割信息
            logger.info(
                f"数据集分割完成: 训练集样本数={len(train_dataset)}, 验证集样本数={len(val_dataset)}"
            )
            if hasattr(train_dataset, "data") and isinstance(
                train_dataset.data, pd.DataFrame
            ):
                logger.info(
                    f"训练集数据形状: {train_dataset.data.shape}, 特征数={len(train_dataset.data.columns)}"
                )
            if hasattr(val_dataset, "data") and isinstance(
                val_dataset.data, pd.DataFrame
            ):
                logger.info(
                    f"验证集数据形状: {val_dataset.data.shape}, 特征数={len(val_dataset.data.columns)}"
                )

            # 5. 训练模型
            if progress_callback:
                await progress_callback(
                    model_id,
                    45.0,
                    "training",
                    f"开始训练 ({feature_set} + {config.label_type})",
                    {
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "model_type": config.model_type.value,
                        "feature_set": feature_set,
                        "label_type": config.label_type,
                        "early_stopping_enabled": config.enable_early_stopping,
                    },
                )

            self.performance_monitor.start_stage("train_model")
            training_result = await _train_qlib_model(
                model_config,
                train_dataset,
                val_dataset,
                config,
                progress_callback,
                model_id,
            )
            self.performance_monitor.end_stage("train_model")

            # 解包训练结果
            if len(training_result) == 3:
                model, training_history, early_stopping_info = training_result
            else:
                # 向后兼容
                model, training_history = training_result
                early_stopping_info = {
                    "early_stopped": False,
                    "stopped_epoch": 0,
                    "best_epoch": 0,
                    "early_stopping_reason": None,
                }

            # 6. 评估模型
            if progress_callback:
                await progress_callback(model_id, 85.0, "evaluating", "评估模型性能")

            self.performance_monitor.start_stage("evaluate_model")
            training_metrics, validation_metrics = await _evaluate_model(
                model, train_dataset, val_dataset, model_id
            )
            self.performance_monitor.end_stage("evaluate_model")

            # 使用评估得到的准确率更新训练历史
            train_accuracy = training_metrics.get("accuracy", 0.0)
            val_accuracy = validation_metrics.get("accuracy", 0.0)

            # 更新训练历史中的准确率（如果历史记录存在）
            if training_history:
                for hist_entry in training_history:
                    if (
                        "train_accuracy" not in hist_entry
                        or hist_entry.get("train_accuracy", 0.0) == 0.0
                    ):
                        hist_entry["train_accuracy"] = round(train_accuracy, 4)
                    if (
                        "val_accuracy" not in hist_entry
                        or hist_entry.get("val_accuracy", 0.0) == 0.0
                    ):
                        hist_entry["val_accuracy"] = round(val_accuracy, 4)

            # 发送详细的评估结果
            if progress_callback:
                await progress_callback(
                    model_id,
                    90.0,
                    "evaluating",
                    "模型评估完成",
                    {
                        "validation_metrics": validation_metrics,
                        "training_metrics": training_metrics,
                    },
                )

            # 7. 提取特征重要性
            self.performance_monitor.start_stage("extract_feature_importance")
            feature_importance = await extract_feature_importance(
                model, config.model_type
            )
            self.performance_monitor.end_stage("extract_feature_importance")

            # 8. 保存模型
            if progress_callback:
                await progress_callback(model_id, 95.0, "saving", "保存模型")

            self.performance_monitor.start_stage("save_model")
            training_meta = {
                "feature_set": config.feature_set,
                "label_type": config.label_type,
                "binary_threshold": config.binary_threshold,
                "split_method": config.split_method,
                "prediction_horizon": config.prediction_horizon,
            }
            model_path = await save_qlib_model(
                model,
                model_id,
                model_config,
                training_meta,
            )
            self.performance_monitor.end_stage("save_model")

            # 9. 完成训练
            training_duration = (datetime.now() - start_time).total_seconds()

            if progress_callback:
                await progress_callback(
                    model_id,
                    100.0,
                    "completed",
                    "训练完成",
                    {
                        "training_duration": training_duration,
                        "final_accuracy": validation_metrics.get("accuracy", 0.0),
                        "model_path": model_path,
                        "early_stopped": early_stopping_info["early_stopped"],
                        "early_stopping_reason": early_stopping_info[
                            "early_stopping_reason"
                        ],
                    },
                )

            # 结束整体性能监控并打印摘要
            self.performance_monitor.end_stage("total_training")
            self.performance_monitor.print_summary()

            result = QlibTrainingResult(
                model_path=model_path,
                model_config=model_config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                feature_importance=feature_importance,
                training_history=training_history,
                training_duration=training_duration,
                train_samples=len(train_dataset),
                validation_samples=len(val_dataset),
                test_samples=0,  # Qlib训练引擎目前不使用测试集
                early_stopped=early_stopping_info["early_stopped"],
                stopped_epoch=early_stopping_info["stopped_epoch"],
                best_epoch=early_stopping_info["best_epoch"],
                early_stopping_reason=early_stopping_info["early_stopping_reason"],
                feature_correlation=feature_correlation,
            )

            logger.info(f"Qlib模型训练完成: {model_id}, 耗时: {training_duration:.2f}秒")
            if early_stopping_info["early_stopped"]:
                logger.info(
                    f"训练提前停止: {early_stopping_info['early_stopping_reason']}, 停止轮次: {early_stopping_info['stopped_epoch']}"
                )
            logger.info(f"验证指标: {validation_metrics}")

            return result

        except Exception as e:
            import traceback

            logger.error(f"Qlib模型训练失败: {model_id}, 错误: {e}\n{traceback.format_exc()}")
            if progress_callback:
                await progress_callback(model_id, 0.0, "failed", f"训练失败: {str(e)}")
            # 结束整体性能监控并打印摘要
            self.performance_monitor.end_stage("total_training")
            self.performance_monitor.print_summary()
            raise

    async def _run_rolling_training(
        self,
        dataset: pd.DataFrame,
        config: QlibTrainingConfig,
        model_id: str,
        start_time: datetime,
        progress_callback=None,
    ) -> QlibTrainingResult:
        """执行滚动训练流程（委托给 rolling_integration 模块）"""
        return await run_rolling_training(
            dataset=dataset,
            config=config,
            model_id=model_id,
            start_time=start_time,
            model_manager=self.model_manager,
            performance_monitor=self.performance_monitor,
            progress_callback=progress_callback,
        )

    # 代理方法 - 保持向后兼容
    async def load_qlib_model(self, model_path: str):
        """加载Qlib模型"""
        return await load_qlib_model(model_path)

    async def predict_with_qlib_model(
        self, model, model_config, stock_codes, start_date, end_date
    ):
        """使用Qlib模型进行预测"""
        try:
            # 准备预测数据
            dataset = await self.data_provider.prepare_qlib_dataset(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                include_alpha_factors=True,
                use_cache=True,
            )

            if dataset.empty:
                raise ValueError("无法获取预测数据")

            # 对齐特征
            from .model_io import _align_prediction_features

            if isinstance(dataset, pd.DataFrame):
                dataset = _align_prediction_features(model, dataset)

            # 进行预测
            predictions = model.predict(dataset)
            logger.info(f"Qlib模型预测完成: {len(predictions)} 条预测结果")
            return predictions

        except Exception as e:
            logger.error(f"Qlib模型预测失败: {e}")
            raise

    def get_supported_model_types(self):
        """获取支持的模型类型"""
        return get_supported_model_types()

    def get_model_config_template(self, model_type: str):
        """获取模型配置模板"""
        return get_model_config_template(model_type)

    def recommend_models(self, data_size: int, feature_count: int):
        """推荐模型"""
        return recommend_models(data_size, feature_count)

    def get_training_recommendations(self, model_type: str):
        """获取训练建议"""
        return get_training_recommendations(model_type)
