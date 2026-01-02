"""
超参数优化器

提供自动超参数优化功能，支持：
- 贝叶斯优化
- 遗传算法
- 网格搜索
- 随机搜索
- 早停策略
"""

import asyncio
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
import json
import logging
from enum import Enum
import math

from sklearn.model_selection import ParameterGrid
from scipy.stats import uniform, randint
import optuna

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法枚举"""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    TPE = "tpe"  # Tree-structured Parzen Estimator


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self):
        self.optimization_history = {}
        self.best_params_cache = {}
    
    async def optimize_hyperparameters(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        objective_function: Callable,
        method: OptimizationMethod = OptimizationMethod.BAYESIAN,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        early_stopping_rounds: Optional[int] = 10,
        direction: str = "maximize",
        **kwargs
    ) -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            model_type: 模型类型
            param_space: 参数空间定义
            objective_function: 目标函数
            method: 优化方法
            n_trials: 试验次数
            timeout: 超时时间（秒）
            early_stopping_rounds: 早停轮数
            direction: 优化方向 ("maximize" 或 "minimize")
            **kwargs: 其他参数
            
        Returns:
            Dict: 优化结果
        """
        logger.info(f"开始超参数优化: {model_type}, 方法: {method.value}")
        
        start_time = datetime.utcnow()
        
        try:
            if method == OptimizationMethod.BAYESIAN:
                result = await self._bayesian_optimization(
                    model_type, param_space, objective_function,
                    n_trials, timeout, early_stopping_rounds, direction, **kwargs
                )
            elif method == OptimizationMethod.TPE:
                result = await self._tpe_optimization(
                    model_type, param_space, objective_function,
                    n_trials, timeout, early_stopping_rounds, direction, **kwargs
                )
            elif method == OptimizationMethod.GENETIC:
                result = await self._genetic_optimization(
                    model_type, param_space, objective_function,
                    n_trials, early_stopping_rounds, direction, **kwargs
                )
            elif method == OptimizationMethod.GRID_SEARCH:
                result = await self._grid_search_optimization(
                    model_type, param_space, objective_function, **kwargs
                )
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = await self._random_search_optimization(
                    model_type, param_space, objective_function,
                    n_trials, early_stopping_rounds, direction, **kwargs
                )
            else:
                raise ValueError(f"不支持的优化方法: {method}")
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # 添加优化元信息
            result.update({
                "optimization_method": method.value,
                "optimization_duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "model_type": model_type
            })
            
            # 缓存最佳参数
            self.best_params_cache[model_type] = result["best_params"]
            
            logger.info(f"超参数优化完成: {model_type}, 最佳得分: {result['best_score']}")
            return result
            
        except Exception as e:
            logger.error(f"超参数优化失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_type": model_type,
                "optimization_method": method.value
            }
    
    async def _bayesian_optimization(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        objective_function: Callable,
        n_trials: int,
        timeout: Optional[int],
        early_stopping_rounds: Optional[int],
        direction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """贝叶斯优化实现"""
        
        # 创建Optuna study
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 定义目标函数包装器
        def optuna_objective(trial):
            # 从参数空间采样参数
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            # 调用目标函数
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                logger.warning(f"目标函数执行失败: {e}")
                return float('-inf') if direction == "maximize" else float('inf')
        
        # 执行优化
        try:
            study.optimize(
                optuna_objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[self._create_early_stopping_callback(early_stopping_rounds)] if early_stopping_rounds else None
            )
            
            return {
                "success": True,
                "best_params": study.best_params,
                "best_score": study.best_value,
                "n_trials": len(study.trials),
                "optimization_history": [
                    {
                        "trial": i,
                        "params": trial.params,
                        "score": trial.value,
                        "state": trial.state.name
                    }
                    for i, trial in enumerate(study.trials)
                ]
            }
            
        except Exception as e:
            logger.error(f"贝叶斯优化执行失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _tpe_optimization(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        objective_function: Callable,
        n_trials: int,
        timeout: Optional[int],
        early_stopping_rounds: Optional[int],
        direction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """TPE优化实现"""
        # TPE优化与贝叶斯优化类似，使用不同的采样器
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=10,
                n_ei_candidates=24
            )
        )
        
        def optuna_objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                logger.warning(f"目标函数执行失败: {e}")
                return float('-inf') if direction == "maximize" else float('inf')
        
        try:
            study.optimize(
                optuna_objective,
                n_trials=n_trials,
                timeout=timeout
            )
            
            return {
                "success": True,
                "best_params": study.best_params,
                "best_score": study.best_value,
                "n_trials": len(study.trials),
                "optimization_history": [
                    {
                        "trial": i,
                        "params": trial.params,
                        "score": trial.value
                    }
                    for i, trial in enumerate(study.trials)
                ]
            }
            
        except Exception as e:
            logger.error(f"TPE优化执行失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _genetic_optimization(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        objective_function: Callable,
        n_trials: int,
        early_stopping_rounds: Optional[int],
        direction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """遗传算法优化实现"""
        
        population_size = kwargs.get("population_size", 20)
        mutation_rate = kwargs.get("mutation_rate", 0.1)
        crossover_rate = kwargs.get("crossover_rate", 0.8)
        n_generations = n_trials // population_size
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = self._generate_random_params(param_space)
            population.append(individual)
        
        best_score = float('-inf') if direction == "maximize" else float('inf')
        best_params = None
        optimization_history = []
        no_improvement_count = 0
        
        for generation in range(n_generations):
            # 评估种群
            scores = []
            for individual in population:
                try:
                    score = objective_function(individual)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"个体评估失败: {e}")
                    scores.append(float('-inf') if direction == "maximize" else float('inf'))
            
            # 更新最佳结果
            current_best_idx = np.argmax(scores) if direction == "maximize" else np.argmin(scores)
            current_best_score = scores[current_best_idx]
            
            improved = False
            if direction == "maximize" and current_best_score > best_score:
                best_score = current_best_score
                best_params = population[current_best_idx].copy()
                improved = True
            elif direction == "minimize" and current_best_score < best_score:
                best_score = current_best_score
                best_params = population[current_best_idx].copy()
                improved = True
            
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 记录历史
            optimization_history.append({
                "generation": generation,
                "best_score": current_best_score,
                "avg_score": np.mean(scores),
                "population_diversity": self._calculate_diversity(population)
            })
            
            # 早停检查
            if early_stopping_rounds and no_improvement_count >= early_stopping_rounds:
                logger.info(f"遗传算法早停: {no_improvement_count} 代无改进")
                break
            
            # 选择、交叉、变异
            new_population = []
            
            # 精英保留
            elite_count = max(1, population_size // 10)
            elite_indices = np.argsort(scores)
            if direction == "maximize":
                elite_indices = elite_indices[::-1]
            
            for i in range(elite_count):
                new_population.append(population[elite_indices[i]].copy())
            
            # 生成新个体
            while len(new_population) < population_size:
                # 选择父母
                parent1 = self._tournament_selection(population, scores, direction)
                parent2 = self._tournament_selection(population, scores, direction)
                
                # 交叉
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, param_space)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1, param_space)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2, param_space)
                
                new_population.extend([child1, child2])
            
            # 截断到目标大小
            population = new_population[:population_size]
        
        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "n_generations": generation + 1,
            "optimization_history": optimization_history
        }
    
    async def _grid_search_optimization(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        objective_function: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """网格搜索优化实现"""
        
        # 转换参数空间为sklearn格式
        sklearn_param_space = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "categorical":
                sklearn_param_space[param_name] = param_config["choices"]
            elif param_config["type"] in ["int", "float"]:
                # 为数值参数生成网格点
                low, high = param_config["low"], param_config["high"]
                n_points = param_config.get("n_points", 5)
                
                if param_config["type"] == "int":
                    sklearn_param_space[param_name] = list(range(low, high + 1, max(1, (high - low) // n_points)))
                else:
                    sklearn_param_space[param_name] = np.linspace(low, high, n_points).tolist()
        
        # 生成参数组合
        param_grid = ParameterGrid(sklearn_param_space)
        
        best_score = float('-inf')
        best_params = None
        optimization_history = []
        
        for i, params in enumerate(param_grid):
            try:
                score = objective_function(params)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                optimization_history.append({
                    "trial": i,
                    "params": params,
                    "score": score
                })
                
            except Exception as e:
                logger.warning(f"网格搜索评估失败: {e}")
                optimization_history.append({
                    "trial": i,
                    "params": params,
                    "score": float('-inf'),
                    "error": str(e)
                })
        
        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(param_grid),
            "optimization_history": optimization_history
        }
    
    async def _random_search_optimization(
        self,
        model_type: str,
        param_space: Dict[str, Any],
        objective_function: Callable,
        n_trials: int,
        early_stopping_rounds: Optional[int],
        direction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """随机搜索优化实现"""
        
        best_score = float('-inf') if direction == "maximize" else float('inf')
        best_params = None
        optimization_history = []
        no_improvement_count = 0
        
        for trial in range(n_trials):
            # 随机采样参数
            params = self._generate_random_params(param_space)
            
            try:
                score = objective_function(params)
                
                improved = False
                if direction == "maximize" and score > best_score:
                    best_score = score
                    best_params = params.copy()
                    improved = True
                elif direction == "minimize" and score < best_score:
                    best_score = score
                    best_params = params.copy()
                    improved = True
                
                if improved:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                optimization_history.append({
                    "trial": trial,
                    "params": params,
                    "score": score
                })
                
                # 早停检查
                if early_stopping_rounds and no_improvement_count >= early_stopping_rounds:
                    logger.info(f"随机搜索早停: {no_improvement_count} 次试验无改进")
                    break
                
            except Exception as e:
                logger.warning(f"随机搜索评估失败: {e}")
                optimization_history.append({
                    "trial": trial,
                    "params": params,
                    "score": float('-inf') if direction == "maximize" else float('inf'),
                    "error": str(e)
                })
        
        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(optimization_history),
            "optimization_history": optimization_history
        }
    
    def _generate_random_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        
        for param_name, param_config in param_space.items():
            if param_config["type"] == "float":
                low, high = param_config["low"], param_config["high"]
                if param_config.get("log", False):
                    params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    params[param_name] = np.random.uniform(low, high)
            elif param_config["type"] == "int":
                low, high = param_config["low"], param_config["high"]
                params[param_name] = np.random.randint(low, high + 1)
            elif param_config["type"] == "categorical":
                params[param_name] = np.random.choice(param_config["choices"])
        
        return params
    
    def _tournament_selection(
        self, 
        population: List[Dict], 
        scores: List[float], 
        direction: str,
        tournament_size: int = 3
    ) -> Dict[str, Any]:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        
        if direction == "maximize":
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_scores)]
        
        return population[winner_idx].copy()
    
    def _crossover(
        self, 
        parent1: Dict[str, Any], 
        parent2: Dict[str, Any], 
        param_space: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """交叉操作"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param_name in param_space.keys():
            if random.random() < 0.5:  # 50%概率交换
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], param_space: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        
        for param_name, param_config in param_space.items():
            if random.random() < 0.1:  # 10%概率变异
                if param_config["type"] == "float":
                    low, high = param_config["low"], param_config["high"]
                    # 高斯变异
                    current_value = mutated[param_name]
                    std = (high - low) * 0.1
                    new_value = np.random.normal(current_value, std)
                    mutated[param_name] = np.clip(new_value, low, high)
                elif param_config["type"] == "int":
                    low, high = param_config["low"], param_config["high"]
                    mutated[param_name] = np.random.randint(low, high + 1)
                elif param_config["type"] == "categorical":
                    mutated[param_name] = np.random.choice(param_config["choices"])
        
        return mutated
    
    def _calculate_diversity(self, population: List[Dict[str, Any]]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        # 简单的多样性度量：参数值的标准差平均值
        diversities = []
        
        # 获取所有参数名
        param_names = set()
        for individual in population:
            param_names.update(individual.keys())
        
        for param_name in param_names:
            values = []
            for individual in population:
                if param_name in individual:
                    value = individual[param_name]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if len(values) > 1:
                diversities.append(np.std(values))
        
        return np.mean(diversities) if diversities else 0.0
    
    def _create_early_stopping_callback(self, patience: int):
        """创建早停回调"""
        def callback(study, trial):
            if len(study.trials) < patience:
                return
            
            # 检查最近patience次试验是否有改进
            recent_values = [t.value for t in study.trials[-patience:] if t.value is not None]
            if len(recent_values) < patience:
                return
            
            best_recent = max(recent_values) if study.direction.name == "MAXIMIZE" else min(recent_values)
            best_overall = study.best_value
            
            if study.direction.name == "MAXIMIZE":
                if best_recent <= best_overall:
                    study.stop()
            else:
                if best_recent >= best_overall:
                    study.stop()
        
        return callback
    
    def get_default_param_space(self, model_type: str) -> Dict[str, Any]:
        """获取默认参数空间"""
        
        param_spaces = {
            "lightgbm": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "num_leaves": {"type": "int", "low": 10, "high": 300},
                "min_child_samples": {"type": "int", "low": 5, "high": 100},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
            },
            "xgboost": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
                "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0}
            },
            "transformer": {
                "d_model": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "nhead": {"type": "categorical", "choices": [4, 8, 16]},
                "num_layers": {"type": "int", "low": 2, "high": 8},
                "dropout": {"type": "float", "low": 0.1, "high": 0.5},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]}
            }
        }
        
        return param_spaces.get(model_type, {})


# 全局实例
hyperparameter_optimizer = HyperparameterOptimizer()