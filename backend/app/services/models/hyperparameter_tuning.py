"""
超参数调优服务

实现自动超参数搜索功能，支持：
- 网格搜索
- 随机搜索
- 贝叶斯优化
"""

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger


def _require_numeric_bound(value: Optional[float], name: str) -> float:
    if value is None:
        raise ValueError(f"超参数 {name} 缺少数值边界")
    return value


class SearchStrategy(Enum):
    """搜索策略"""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"


@dataclass
class HyperparameterSpace:
    """超参数空间定义"""

    name: str
    param_type: str  # 'int', 'float', 'choice'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None


@dataclass
class HyperparameterTrial:
    """超参数试验结果"""

    trial_id: int
    hyperparameters: Dict[str, Any]
    score: float
    metrics: Dict[str, Any]
    status: str  # 'completed', 'failed', 'running'


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(
        self, search_strategy: SearchStrategy = SearchStrategy.GRID_SEARCH
    ) -> None:
        self.search_strategy = search_strategy
        self.trials: List[HyperparameterTrial] = []
        self.best_trial: Optional[HyperparameterTrial] = None

    def grid_search(
        self,
        param_space: Dict[str, HyperparameterSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        max_trials: Optional[int] = None,
    ) -> HyperparameterTrial:
        """网格搜索"""
        logger.info("开始网格搜索超参数")

        # 生成所有参数组合
        param_combinations = self._generate_grid_combinations(param_space)

        if max_trials and len(param_combinations) > max_trials:
            # 随机采样
            indices = np.random.choice(
                len(param_combinations), max_trials, replace=False
            )
            param_combinations = [param_combinations[i] for i in indices]

        logger.info(f"将测试 {len(param_combinations)} 组超参数")

        best_score = float("-inf")
        best_trial: Optional[HyperparameterTrial] = None

        for trial_id, params in enumerate(param_combinations):
            try:
                logger.info(f"试验 {trial_id + 1}/{len(param_combinations)}: {params}")

                # 训练模型并评估
                metrics = train_fn(params)
                score = metrics.get("score", metrics.get("accuracy", 0.0))

                trial = HyperparameterTrial(
                    trial_id=trial_id,
                    hyperparameters=params,
                    score=score,
                    metrics=metrics,
                    status="completed",
                )
                self.trials.append(trial)

                if score > best_score:
                    best_score = score
                    best_trial = trial
                    logger.info(f"发现更好的超参数组合，得分: {score:.4f}")

            except Exception as e:
                logger.error(f"试验 {trial_id} 失败: {e}")
                trial = HyperparameterTrial(
                    trial_id=trial_id,
                    hyperparameters=params,
                    score=0.0,
                    metrics={},
                    status="failed",
                )
                self.trials.append(trial)

        if best_trial is None:
            raise ValueError("网格搜索未产生成功的超参数试验")

        self.best_trial = best_trial
        return best_trial

    def random_search(
        self,
        param_space: Dict[str, HyperparameterSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 20,
    ) -> HyperparameterTrial:
        """随机搜索"""
        logger.info(f"开始随机搜索超参数，试验次数: {n_trials}")

        best_score = float("-inf")
        best_trial: Optional[HyperparameterTrial] = None

        for trial_id in range(n_trials):
            # 随机采样参数
            params = self._random_sample_params(param_space)

            try:
                logger.info(f"试验 {trial_id + 1}/{n_trials}: {params}")

                # 训练模型并评估
                metrics = train_fn(params)
                score = metrics.get("score", metrics.get("accuracy", 0.0))

                trial = HyperparameterTrial(
                    trial_id=trial_id,
                    hyperparameters=params,
                    score=score,
                    metrics=metrics,
                    status="completed",
                )
                self.trials.append(trial)

                if score > best_score:
                    best_score = score
                    best_trial = trial
                    logger.info(f"发现更好的超参数组合，得分: {score:.4f}")

            except Exception as e:
                logger.error(f"试验 {trial_id} 失败: {e}")
                trial = HyperparameterTrial(
                    trial_id=trial_id,
                    hyperparameters=params,
                    score=0.0,
                    metrics={},
                    status="failed",
                )
                self.trials.append(trial)

        if best_trial is None:
            raise ValueError("随机搜索未产生成功的超参数试验")

        self.best_trial = best_trial
        return best_trial

    def _generate_grid_combinations(
        self, param_space: Dict[str, HyperparameterSpace]
    ) -> List[Dict[str, Any]]:
        """生成网格搜索的所有参数组合"""
        param_values: Dict[str, List[Any]] = {}

        for param_name, param_space_def in param_space.items():
            if param_space_def.param_type == "int":
                min_value = _require_numeric_bound(
                    param_space_def.min_value, param_name
                )
                max_value = _require_numeric_bound(
                    param_space_def.max_value, param_name
                )
                step = int(param_space_def.step or 1)
                values = list(range(int(min_value), int(max_value) + 1, step))
            elif param_space_def.param_type == "float":
                min_value = _require_numeric_bound(
                    param_space_def.min_value, param_name
                )
                max_value = _require_numeric_bound(
                    param_space_def.max_value, param_name
                )
                float_step = _require_numeric_bound(param_space_def.step, param_name)
                values = np.arange(
                    min_value, max_value + float_step, float_step
                ).tolist()
            elif param_space_def.param_type == "choice":
                values = list(param_space_def.choices or [])
            else:
                raise ValueError(f"不支持的参数类型: {param_space_def.param_type}")

            param_values[param_name] = values

        # 生成所有组合
        combinations: List[Dict[str, Any]] = []
        for combination in product(*param_values.values()):
            combinations.append(dict(zip(param_values.keys(), combination)))

        return combinations

    def _random_sample_params(
        self, param_space: Dict[str, HyperparameterSpace]
    ) -> Dict[str, Any]:
        """随机采样参数"""
        params: Dict[str, Any] = {}

        for param_name, param_space_def in param_space.items():
            value: Any
            if param_space_def.param_type == "int":
                min_value = _require_numeric_bound(
                    param_space_def.min_value, param_name
                )
                max_value = _require_numeric_bound(
                    param_space_def.max_value, param_name
                )
                sampled_value = int(
                    np.random.randint(int(min_value), int(max_value) + 1)
                )
                value = sampled_value
            elif param_space_def.param_type == "float":
                min_value = _require_numeric_bound(
                    param_space_def.min_value, param_name
                )
                max_value = _require_numeric_bound(
                    param_space_def.max_value, param_name
                )
                sampled_float = float(np.random.uniform(min_value, max_value))
                value = sampled_float
            elif param_space_def.param_type == "choice":
                choices = list(param_space_def.choices or [])
                if not choices:
                    raise ValueError(f"超参数 {param_name} 缺少可选值")
                value = np.random.choice(choices).item()
            else:
                raise ValueError(f"不支持的参数类型: {param_space_def.param_type}")

            params[param_name] = value

        return params

    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """获取最佳超参数"""
        if self.best_trial:
            return self.best_trial.hyperparameters
        return None

    def get_trial_summary(self) -> Dict[str, Any]:
        """获取试验摘要"""
        if not self.trials:
            return {}

        completed_trials = [t for t in self.trials if t.status == "completed"]

        return {
            "total_trials": len(self.trials),
            "completed_trials": len(completed_trials),
            "failed_trials": len(self.trials) - len(completed_trials),
            "best_score": self.best_trial.score if self.best_trial else None,
            "best_hyperparameters": (
                self.best_trial.hyperparameters if self.best_trial else None
            ),
            "average_score": (
                np.mean([t.score for t in completed_trials])
                if completed_trials
                else None
            ),
            "std_score": (
                np.std([t.score for t in completed_trials])
                if completed_trials
                else None
            ),
        }
