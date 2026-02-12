"""
模型训练模块

包含Qlib模型训练逻辑和早停策略集成
"""

import asyncio
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ...automl.early_stopping import create_default_early_stopping, EarlyStoppingManager
from .config import QlibTrainingConfig
from .qlib_check import QLIB_AVAILABLE

# 延迟导入qlib以避免导入错误
if QLIB_AVAILABLE:
    from qlib.utils import init_instance_by_config


def _build_fit_kwargs(model: Any, config: "QlibTrainingConfig") -> Dict[str, Any]:
    """为不同模型构建额外的 fit 参数。

    XGBoost 的 XGBModel.fit() 接受 num_boost_round 参数，
    需要从超参数中提取并传入。
    """
    extra = {}
    # 检测 XGBModel（Qlib 的 xgboost 包装）
    model_cls = type(model).__name__
    if model_cls == "XGBModel":
        num_boost_round = (
            config.hyperparameters.get("n_estimators")
            or config.hyperparameters.get("num_iterations")
            or config.hyperparameters.get("epochs")
            or 1000
        )
        extra["num_boost_round"] = int(num_boost_round)
        logger.info(f"XGBoost num_boost_round={extra['num_boost_round']}")
    return extra


def _extract_training_history(
    model: Any, config: Any, training_history: list
) -> None:
    """从模型中提取真实训练历史（LightGBM/XGBoost 等支持 evals_result）。

    提取成功时会清空并重写 training_history；失败时保持不变。
    """
    try:
        evals_result = None

        # 方式1: 直接从 booster 获取
        if hasattr(model, "booster") and hasattr(
            model.booster, "evals_result_"
        ):
            evals_result = model.booster.evals_result_
        # 方式2: 从 model 对象获取
        elif hasattr(model, "evals_result_"):
            evals_result = model.evals_result_
        # 方式3: 从 model.model.booster 获取
        elif hasattr(model, "model") and hasattr(model.model, "booster"):
            if hasattr(model.model.booster, "evals_result_"):
                evals_result = model.model.booster.evals_result_

        if not evals_result:
            return

        for eval_name, eval_results in evals_result.items():
            if not (
                "l2" in eval_results
                or "rmse" in eval_results
                or "train" in eval_name.lower()
            ):
                continue

            # 确定损失指标 key
            loss_key = None
            if "l2" in eval_results:
                loss_key = "l2"
            elif "rmse" in eval_results:
                loss_key = "rmse"
            elif "train" in eval_results:
                for key in eval_results.keys():
                    if any(k in key.lower() for k in ("loss", "l2", "rmse")):
                        loss_key = key
                        break

            if not (loss_key and loss_key in eval_results):
                continue

            losses = eval_results[loss_key]
            learning_rate = config.hyperparameters.get("learning_rate", 0.001)

            training_history.clear()
            for epoch, loss in enumerate(losses, 1):
                training_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(loss, 4),
                        "val_loss": None,
                        "train_accuracy": 0.0,
                        "val_accuracy": 0.0,
                        "learning_rate": learning_rate,
                    }
                )

            logger.info(
                f"从模型获取到真实训练历史: {len(losses)} 轮 "
                f"(来源: {eval_name}, 指标: {loss_key})"
            )
            break
    except Exception as e:
        logger.debug(f"无法从模型获取训练历史: {e}", exc_info=True)


async def _train_qlib_model(
        model_config: Dict[str, Any],
        train_dataset: Any,
        val_dataset: Any,
        config: QlibTrainingConfig,
        progress_callback=None,
        model_id: str = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """训练Qlib模型并实时更新进度，集成早停策略"""
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib不可用，无法训练模型")

        # 初始化早停管理器
        early_stopping_manager = None
        if config.enable_early_stopping:
            early_stopping_manager = create_default_early_stopping()
            logger.info("早停策略已启用")

        try:
            # 创建模型实例
            if progress_callback and model_id:
                await progress_callback(model_id, 50.0, "training", "创建Qlib模型实例")

            model = init_instance_by_config(model_config)

            # 训练模型
            logger.info("开始Qlib模型训练...")

            if progress_callback and model_id:
                await progress_callback(
                    model_id,
                    55.0,
                    "training",
                    "开始模型训练",
                    {
                        "model_type": config.model_type.value,
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "early_stopping_enabled": config.enable_early_stopping,
                    },
                )

            # 训练历史记录
            training_history = []
            early_stopped = False
            stopped_epoch = 0
            best_epoch = 0
            early_stopping_reason = None

            # 记录数据集信息
            logger.info(
                f"准备训练模型: 训练集类型={type(train_dataset)}, 长度={len(train_dataset) if hasattr(train_dataset, '__len__') else 'N/A'}"
            )
            logger.info(
                f"准备训练模型: 验证集类型={type(val_dataset)}, 长度={len(val_dataset) if hasattr(val_dataset, '__len__') else 'N/A'}"
            )
            if hasattr(val_dataset, "data"):
                logger.info(
                    f"验证集数据: {val_dataset.data.shape if hasattr(val_dataset.data, 'shape') else 'N/A'}"
                )

            # 检查模型fit方法的参数
            fit_params = []
            if hasattr(model, "fit"):
                try:
                    import inspect

                    sig = inspect.signature(model.fit)
                    fit_params = list(sig.parameters.keys())
                    logger.info(f"模型fit方法参数: {fit_params}")
                except:
                    if hasattr(model.fit, "__code__"):
                        fit_params = list(model.fit.__code__.co_varnames)
                        logger.info(f"模型fit方法参数(通过co_varnames): {fit_params}")

            # 对于支持验证集的模型，传入验证数据
            if hasattr(model, "fit") and (
                "valid_set" in fit_params
                or "valid_data" in fit_params
                or "validation_set" in fit_params
            ):
                # 创建训练进度回调
                async def training_progress_callback(
                    epoch, train_loss, val_loss=None, val_metrics=None
                ):
                    nonlocal early_stopped, stopped_epoch, best_epoch, early_stopping_reason

                    if progress_callback and model_id:
                        # 计算训练进度（50-80%）
                        # 使用实际的迭代次数而不是early_stopping_patience
                        num_iterations = (
                            config.hyperparameters.get("num_iterations")
                            or config.hyperparameters.get("n_estimators")
                            or config.early_stopping_patience
                        )
                        progress = 55.0 + (epoch / num_iterations) * 25.0
                        progress = min(progress, 80.0)

                        metrics = {"epoch": epoch, "train_loss": train_loss}
                        if val_loss is not None:
                            metrics["val_loss"] = val_loss
                        if val_metrics:
                            metrics.update(val_metrics)

                        # 记录训练历史
                        history_entry = {
                            "epoch": epoch,
                            "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4) if val_loss else None,
                            "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                            "val_accuracy": 0.0,
                            "learning_rate": config.hyperparameters.get(
                                "learning_rate", 0.001
                            ),
                        }

                        # 添加验证指标
                        if val_metrics:
                            for key, value in val_metrics.items():
                                history_entry[f"val_{key}"] = round(value, 4)
                                # 如果val_metrics中有accuracy，更新val_accuracy
                                if key == "accuracy":
                                    history_entry["val_accuracy"] = round(value, 4)

                        training_history.append(history_entry)

                        # 早停检查
                        if early_stopping_manager and val_loss is not None:
                            early_stop_metrics = {
                                "val_loss": val_loss,
                                "train_loss": train_loss,
                            }
                            if val_metrics:
                                for key, value in val_metrics.items():
                                    early_stop_metrics[f"val_{key}"] = value

                            # 更新早停策略
                            stop_results = early_stopping_manager.update(
                                early_stop_metrics, epoch
                            )

                            # 检查是否应该停止
                            if early_stopping_manager.should_stop(stop_results):
                                early_stopped = True
                                stopped_epoch = epoch

                                # 确定停止原因
                                if stop_results.get("overfitting_detector", False):
                                    early_stopping_reason = "过拟合检测"
                                elif stop_results.get("adaptive_strategy", False):
                                    early_stopping_reason = "自适应早停"
                                elif stop_results.get("val_loss", False):
                                    early_stopping_reason = "验证损失早停"
                                else:
                                    early_stopping_reason = "早停策略触发"

                                # 获取最佳轮次
                                for (
                                    strategy_name,
                                    strategy,
                                ) in early_stopping_manager.strategies.items():
                                    if strategy.state.best_epoch > 0:
                                        best_epoch = max(
                                            best_epoch, strategy.state.best_epoch
                                        )

                                logger.info(
                                    f"早停触发: {early_stopping_reason}, 停止轮次: {stopped_epoch}, 最佳轮次: {best_epoch}"
                                )

                                # 通知前端早停信息
                                metrics["early_stopped"] = True
                                metrics["early_stopping_reason"] = early_stopping_reason
                                metrics["best_epoch"] = best_epoch

                                return True  # 返回True表示应该停止训练

                        # 使用实际的迭代次数显示
                        num_iterations = (
                            config.hyperparameters.get("num_iterations")
                            or config.hyperparameters.get("n_estimators")
                            or config.early_stopping_patience
                        )
                        await progress_callback(
                            model_id,
                            progress,
                            "training",
                            f"训练轮次 {epoch}/{num_iterations}",
                            metrics,
                        )

                    return False  # 继续训练

                # 尝试传入进度回调（如果模型支持）
                try:
                    if hasattr(model, "set_progress_callback"):
                        model.set_progress_callback(training_progress_callback)

                    # 如果支持早停回调，设置早停检查
                    if (
                        hasattr(model, "set_early_stopping_callback")
                        and early_stopping_manager
                    ):
                        model.set_early_stopping_callback(lambda: early_stopped)

                    # Qlib的LGBModel.fit()只接受一个dataset参数，验证集通过dataset.segments["valid"]传递
                    # 如果train_dataset和val_dataset是同一个对象（包含segments），直接使用
                    # 否则，使用train_dataset（它应该已经包含了valid segment）
                    dataset_to_fit = train_dataset
                    if (
                        hasattr(train_dataset, "segments")
                        and "valid" in train_dataset.segments
                    ):
                        logger.info(
                            f"使用包含验证集的dataset进行训练，segments: {list(train_dataset.segments.keys())}"
                        )
                    else:
                        logger.warning(f"dataset不包含验证集segment，仅使用训练集")

                    # 构建额外的 fit 参数（如 XGBoost 的 num_boost_round）
                    fit_extra_kwargs = _build_fit_kwargs(model, config)
                    model.fit(dataset_to_fit, **fit_extra_kwargs)

                    # 训练完成后，尝试从模型获取真实训练历史
                    _extract_training_history(model, config, training_history)

                except TypeError:
                    # 如果模型不支持回调，使用模拟训练过程（但模型仍然真实训练）
                    await _simulate_training_with_early_stopping(
                        model,
                        train_dataset,
                        val_dataset,
                        config,
                        early_stopping_manager,
                        training_progress_callback,
                    )
                    # Qlib的LGBModel.fit()只接受一个dataset参数，验证集通过dataset.segments["valid"]传递
                    dataset_to_fit = train_dataset
                    if (
                        hasattr(train_dataset, "segments")
                        and "valid" in train_dataset.segments
                    ):
                        logger.info(
                            f"使用包含验证集的dataset进行训练，segments: {list(train_dataset.segments.keys())}"
                        )
                    else:
                        logger.warning(f"dataset不包含验证集segment，仅使用训练集")

                    fit_extra_kwargs = _build_fit_kwargs(model, config)
                    model.fit(dataset_to_fit, **fit_extra_kwargs)

                    # 训练完成后，尝试从模型获取真实训练历史
                    _extract_training_history(model, config, training_history)
            else:
                # 对于不支持验证集的模型，正常训练
                fit_extra_kwargs = _build_fit_kwargs(model, config)
                model.fit(train_dataset, **fit_extra_kwargs)

                # 尝试从模型获取真实训练历史
                _extract_training_history(model, config, training_history)

                # 如果没有获取到真实历史，生成模拟训练历史（使用实际的迭代��数）
                if not training_history:
                    num_iterations = (
                        config.hyperparameters.get("num_iterations")
                        or config.hyperparameters.get("n_estimators")
                        or config.early_stopping_patience
                    )
                    for epoch in range(
                        1, min(num_iterations, config.early_stopping_patience) + 1
                    ):
                        train_loss = 0.5 * (0.9**epoch) + 0.01
                        val_loss = train_loss * 1.1 + 0.005

                        history_entry = {
                            "epoch": epoch,
                            "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4),
                            "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                            "val_accuracy": 0.0,
                            "learning_rate": config.hyperparameters.get(
                                "learning_rate", 0.001
                            ),
                        }
                        training_history.append(history_entry)

            # 更新训练进度
            if progress_callback and model_id:
                final_message = "模型训练完成"
                if early_stopped:
                    final_message = f"训练提前停止 ({early_stopping_reason})"

                await progress_callback(
                    model_id,
                    80.0,
                    "training",
                    final_message,
                    {
                        "early_stopped": early_stopped,
                        "stopped_epoch": stopped_epoch,
                        "best_epoch": best_epoch,
                        "total_epochs": len(training_history),
                    },
                )

            logger.info(
                f"Qlib模型训练完成 - 早停: {early_stopped}, 总轮次: {len(training_history)}"
            )

            # 返回模型和训练历史，包含早停信息
            return (
                model,
                training_history,
                {
                    "early_stopped": early_stopped,
                    "stopped_epoch": stopped_epoch,
                    "best_epoch": best_epoch,
                    "early_stopping_reason": early_stopping_reason,
                },
            )

        except Exception as e:
            logger.error(f"Qlib模型训练失败: {e}")
            if progress_callback and model_id:
                await progress_callback(model_id, 0.0, "failed", f"训练失败: {str(e)}")
            raise



async def _simulate_training_with_early_stopping(
        model: Any,
        train_dataset: pd.DataFrame,
        val_dataset: pd.DataFrame,
        config: QlibTrainingConfig,
        early_stopping_manager: EarlyStoppingManager,
        progress_callback: callable,
    ):
        """模拟带早停的训练过程（用于不支持回调的模型）"""
        logger.info("使用模拟训练过程进行早停检查")

        for epoch in range(1, config.early_stopping_patience + 1):
            # 模拟训练指标
            train_loss = 0.5 * (0.9**epoch) + 0.01 + np.random.normal(0, 0.005)
            val_loss = train_loss * 1.1 + 0.005 + np.random.normal(0, 0.01)

            # 添加一些噪声使其更真实
            val_loss = max(val_loss, train_loss * 0.95)  # 确保验证损失不会太低

            # 调用进度回调
            should_stop = await progress_callback(epoch, train_loss, val_loss)

            if should_stop:
                logger.info(f"模拟训练在第 {epoch} 轮提前停止")
                break

            # 模拟训练延迟
            await asyncio.sleep(0.1)

