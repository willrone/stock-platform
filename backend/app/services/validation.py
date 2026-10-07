"""
时间序列验证框架

实现金融机器学习中标准的验证方法：
1. Walk-Forward Validation — 滑窗/扩展窗口，正向滚动评估
2. Purged K-Fold + Embargo — De Prado 推荐的清洗K折+禁区

参考：
- Marcos López de Prado, "Advances in Financial Machine Learning", Ch. 7
- QuantInsti: Walk-Forward Optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """单折验证结果"""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_size: int
    test_size: int
    metrics: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metrics is None:
            object.__setattr__(self, "metrics", {})


@dataclass(frozen=True)
class ValidationReport:
    """验证报告"""

    folds: List[FoldResult]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    worst_fold: int
    best_fold: int
    total_train_samples: int
    total_test_samples: int

    def summary(self) -> str:
        lines = [
            f"Folds: {len(self.folds)}",
            f"Mean accuracy: {self.mean_metrics.get('accuracy', 0):.4f} ± {self.std_metrics.get('accuracy', 0):.4f}",
            f"Mean Sharpe: {self.mean_metrics.get('sharpe_ratio', 0):.4f} ± {self.std_metrics.get('sharpe_ratio', 0):.4f}",
            f"Best fold: {self.best_fold}, Worst fold: {self.worst_fold}",
            f"Total train: {self.total_train_samples}, Total test: {self.total_test_samples}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------


class WindowType(Enum):
    EXPANDING = "expanding"  # 训练集从起点开始，不断扩大
    ROLLING = "rolling"  # 训练集固定大小，向前滚动


class WalkForwardValidator:
    """
    Walk-Forward 滑窗验证器

    工作方式：
    1. 数据按时间顺序排列
    2. 划分为 n_folds 个测试窗口（等大小、不重叠、正向排列）
    3. 每个测试窗口前的训练窗口大小取决于 WindowType
    4. 每折之间可选 embargo gap

    示例（expanding, 5 folds）:
    |--train1--|--test1--|
    |----train2----|--test2--|
    |-------train3-------|--test3--|
    |----------train4----------|--test4--|
    |-------------train5-------------|--test5--|

    示例（rolling, 5 folds, window=200）:
    |----train1(200)----|--test1--|
    |----train2(200)----|--test2--|  (向右滚动)
    |----train3(200)----|--test3--|
    ...
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_ratio: float = 0.15,
        window_type: WindowType = WindowType.EXPANDING,
        min_train_size: int = 100,
        embargo_ratio: float = 0.0,
    ) -> None:
        """
        Args:
            n_folds: 验证折数
            test_ratio: 测试集占总数据的比例（每个 fold）
            window_type: 训练窗口类型（expanding/rolling）
            min_train_size: 最小训练集大小（如果数据不够则跳过该 fold）
            embargo_ratio: embargo 占总数据的比例（train/test 之间的隔离带）
        """
        self.n_folds = n_folds
        self.test_ratio = test_ratio
        self.window_type = window_type
        self.min_train_size = min_train_size
        self.embargo_ratio = embargo_ratio

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成 walk-forward 切分

        Args:
            X: 特征数据 [samples, ...]
            y: 标签数据（未使用，仅保持接口兼容）

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size = max(1, int(n_samples * self.test_ratio))
        embargo_size = max(0, int(n_samples * self.embargo_ratio))

        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        for fold in range(self.n_folds):
            # 测试集位置：从后往前留 test_size 的空间
            test_end = n_samples - fold * test_size
            test_start = test_end - test_size

            if test_start <= 0:
                logger.debug(f"Fold {fold}: 跳过（测试集起始位置 <= 0）")
                continue

            # 训练集位置
            if self.window_type == WindowType.EXPANDING:
                train_start = 0
            else:
                # ROLLING: 训练集固定大小 = test_size * 3 或 min_train_size
                train_window = max(test_size * 3, self.min_train_size)
                train_start = max(0, test_start - embargo_size - train_window)

            train_end = test_start - embargo_size

            if train_end - train_start < self.min_train_size:
                logger.debug(
                    f"Fold {fold}: 跳过（训练集太小: {train_end - train_start} < {self.min_train_size}）"
                )
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        # 反转顺序使 fold 0 是最早的测试窗口
        splits.reverse()

        logger.info(
            f"Walk-Forward 分割: {len(splits)} 折, "
            f"测试集大小={test_size}, embargo={embargo_size}, "
            f"窗口类型={self.window_type.value}"
        )
        return splits

    def split_dates(
        self,
        dates: np.ndarray,
        X: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, Any, Any]]:
        """
        按日期分割，返回每个 fold 的训练/测试时间范围

        Args:
            dates: 日期数组 [samples]
            X: 特征数据

        Returns:
            List of (train_idx, test_idx, train_start_date, test_end_date)
        """
        splits = self.split(X)
        result = []
        for train_idx, test_idx in splits:
            train_start = dates[train_idx[0]] if len(train_idx) > 0 else None
            test_end = dates[test_idx[-1]] if len(test_idx) > 0 else None
            result.append((train_idx, test_idx, train_start, test_end))
        return result


# ---------------------------------------------------------------------------
# Purged K-Fold + Embargo Validator
# ---------------------------------------------------------------------------


class PurgedKFoldValidator:
    """
    Purged K-Fold + Embargo 验证器

    基于 Marcos López de Prado 的方法：

    1. **Purging（清洗）**: 移除训练集中与测试集时间重叠的样本
       - 当标签的 event time（如预测窗口到期时间）跨越到测试集时间范围时，
         该训练样本必须被移除，否则造成信息泄露。

    2. **Embargo（禁区）**: 测试集之后的一段时间也从训练集中排除
       - 金融数据通常存在自相关，紧邻测试集之后的数据如果在训练集中，
         会导致模型间接看到测试集的信息。

    示例（5 折, embargo=5%）:
    |--train--|emb|test1|--emb|--train--|emb|test2|...

    注意：这种方法仍然尊重时间顺序（不是随机 k-fold）。
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.05,
        purge_window: int = 0,
        min_train_size: int = 50,
    ) -> None:
        """
        Args:
            n_splits: 折数
            embargo_pct: embargo 占总数据的百分比 (0.0-1.0)
            purge_window: purge 窗口大小（标签前瞻期的天数，0=自动用 test_size//n_splits）
            min_train_size: 最小训练集大小
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_window = purge_window
        self.min_train_size = min_train_size

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成 purged k-fold 切分

        划分策略：
        1. 将数据等分为 n_splits 个连续的测试块
        2. 对于每个测试块，训练集 = 该块之前的数据 - purge_window - embargo
        3. 保证训练集不小于 min_train_size

        Args:
            X: 特征数据 [samples, ...]
            y: 标签数据（未使用）

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        # purge_window 默认为一个测试块大小
        test_size = n_samples // self.n_splits
        purge = self.purge_window if self.purge_window > 0 else test_size // 2

        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        for fold in range(self.n_splits):
            # 测试块位置
            test_start = fold * test_size
            test_end = test_start + test_size

            if fold == self.n_splits - 1:
                # 最后一个 fold 使用剩余所有数据
                test_end = n_samples

            # 训练集：测试块之前的数据
            train_end = test_start - purge - embargo_size

            if train_end < self.min_train_size:
                logger.debug(
                    f"Fold {fold}: 跳过（训练集太小: {train_end} < {self.min_train_size}）"
                )
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        logger.info(
            f"Purged K-Fold 分割: {len(splits)} 折, "
            f"embargo={embargo_size} ({self.embargo_pct * 100:.1f}%), "
            f"purge={purge}"
        )
        return splits


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算单折的指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        returns: 预测收益率（可选）

    Returns:
        指标字典
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    avg = "weighted"  # 统一用 weighted，兼容二分类和多分类

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, average=avg)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, average=avg)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, average=avg)),
    }

    if returns is not None and len(returns) > 0:
        ret = np.array(returns, dtype=float)
        # Sharpe ratio（假设 252 交易日）
        if np.std(ret) > 0:
            metrics["sharpe_ratio"] = float(np.mean(ret) / np.std(ret) * np.sqrt(252))
        else:
            metrics["sharpe_ratio"] = 0.0
        # 最大回撤
        cum = np.cumprod(1 + ret)
        running_max = np.maximum.accumulate(cum)
        drawdown = (cum - running_max) / running_max
        metrics["max_drawdown"] = float(np.min(drawdown))
        # 总收益
        metrics["total_return"] = float(cum[-1] - 1) if len(cum) > 0 else 0.0
        # 胜率
        metrics["win_rate"] = float(np.mean(ret > 0)) if len(ret) > 0 else 0.0

    return metrics


def aggregate_validation_results(
    fold_results: List[FoldResult],
) -> ValidationReport:
    """
    汇总多折验证结果

    Args:
        fold_results: 各折结果列表

    Returns:
        验证报告
    """
    all_metric_keys = set()
    for fr in fold_results:
        if fr.metrics:
            all_metric_keys.update(fr.metrics.keys())

    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}

    for key in sorted(all_metric_keys):
        values = [
            fr.metrics[key]  # type: ignore[index]
            for fr in fold_results
            if fr.metrics and key in fr.metrics
        ]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

    # 找最佳/最差 fold（按 accuracy）
    best_fold = 0
    worst_fold = 0
    best_acc = -1.0
    worst_acc = 2.0
    for fr in fold_results:
        acc = fr.metrics.get("accuracy", 0) if fr.metrics else 0  # type: ignore[union-attr]
        if acc > best_acc:
            best_acc = acc
            best_fold = fr.fold
        if acc < worst_acc:
            worst_acc = acc
            worst_fold = fr.fold

    total_train = sum(fr.train_size for fr in fold_results)
    total_test = sum(fr.test_size for fr in fold_results)

    return ValidationReport(
        folds=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        worst_fold=worst_fold,
        best_fold=best_fold,
        total_train_samples=total_train,
        total_test_samples=total_test,
    )
