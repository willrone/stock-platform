"""
并行超参数搜索实现

基于多进程的并行超参数搜索，突破GIL限制，显著提升搜索速度
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from loguru import logger

from app.services.models.hyperparameter_tuning import HyperparameterTrial, HyperparameterSpace


@dataclass
class ParallelSearchConfig:
    """并行搜索配置"""
    max_workers: Optional[int] = None  # 最大工作进程数，None表示使用CPU核心数
    timeout_per_trial: Optional[float] = None  # 每个试验的超时时间（秒）
    enable_early_stopping: bool = True  # 是否启用早停
    early_stopping_patience: int = 10  # 早停耐心值
    min_trials_for_parallel: int = 4  # 最少试验数才启用并行


class ParallelHyperparameterSearch:
    """并行超参数搜索器"""
    
    def __init__(self, config: Optional[ParallelSearchConfig] = None):
        """
        初始化并行超参数搜索器
        
        Args:
            config: 并行搜索配置
        """
        self.config = config or ParallelSearchConfig()
        
        if self.config.max_workers is None:
            self.config.max_workers = mp.cpu_count()
        
        logger.info(f"并行超参数搜索器初始化完成，最大工作进程数: {self.config.max_workers}")
    
    def parallel_random_search(
        self,
        param_space: Dict[str, HyperparameterSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 50,
        progress_callback: Optional[Callable] = None
    ) -> HyperparameterTrial:
        """
        并行随机搜索超参数
        
        Args:
            param_space: 超参数空间
            train_fn: 训练函数，接受超参数字典，返回评估指标字典
            n_trials: 试验次数
            progress_callback: 进度回调函数
        
        Returns:
            最优超参数试验结果
        """
        logger.info(f"开始并行随机搜索，试验次数: {n_trials}, 工作进程数: {self.config.max_workers}")
        
        # 生成超参数组合
        trials_params = self._generate_random_trials(param_space, n_trials)
        
        # 决定是否使用并行
        use_parallel = len(trials_params) >= self.config.min_trials_for_parallel
        
        if use_parallel:
            return self._execute_parallel_search(
                trials_params, train_fn, progress_callback
            )
        else:
            return self._execute_sequential_search(
                trials_params, train_fn, progress_callback
            )
    
    def parallel_grid_search(
        self,
        param_space: Dict[str, HyperparameterSpace],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        max_trials: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> HyperparameterTrial:
        """
        并行网格搜索超参数
        
        Args:
            param_space: 超参数空间
            train_fn: 训练函数
            max_trials: 最大试验数（如果网格太大，随机采样）
            progress_callback: 进度回调函数
        
        Returns:
            最优超参数试验结果
        """
        # 生成网格组合
        trials_params = self._generate_grid_combinations(param_space)
        
        # 如果组合太多，随机采样
        if max_trials and len(trials_params) > max_trials:
            import random
            trials_params = random.sample(trials_params, max_trials)
        
        logger.info(f"开始并行网格搜索，试验次数: {len(trials_params)}, 工作进程数: {self.config.max_workers}")
        
        # 决定是否使用并行
        use_parallel = len(trials_params) >= self.config.min_trials_for_parallel
        
        if use_parallel:
            return self._execute_parallel_search(
                trials_params, train_fn, progress_callback
            )
        else:
            return self._execute_sequential_search(
                trials_params, train_fn, progress_callback
            )
    
    def _execute_parallel_search(
        self,
        trials_params: List[Dict[str, Any]],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        progress_callback: Optional[Callable] = None
    ) -> HyperparameterTrial:
        """执行并行搜索"""
        start_time = datetime.now()
        best_score = float('-inf')
        best_trial: Optional[HyperparameterTrial] = None
        completed_trials = 0
        failed_trials = 0
        
        # 使用进程池并行执行
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    _train_and_evaluate_worker,
                    train_fn,
                    params,
                    trial_id
                ): trial_id
                for trial_id, params in enumerate(trials_params)
            }
            
            # 收集结果
            for future in as_completed(futures):
                trial_id = futures[future]
                try:
                    # 获取结果（带超时）
                    if self.config.timeout_per_trial:
                        result = future.result(timeout=self.config.timeout_per_trial)
                    else:
                        result = future.result()
                    
                    # 解析结果
                    if result['success']:
                        score = result['score']
                        metrics = result['metrics']
                        
                        trial = HyperparameterTrial(
                            trial_id=trial_id,
                            hyperparameters=result['params'],
                            score=score,
                            metrics=metrics,
                            status='completed'
                        )
                        
                        completed_trials += 1
                        
                        # 更新最优结果
                        if score > best_score:
                            best_score = score
                            best_trial = trial
                            logger.info(
                                f"发现更好的超参数组合 (试验 {trial_id}): "
                                f"得分={score:.6f}, 参数={result['params']}"
                            )
                        
                        # 进度回调
                        if progress_callback:
                            progress_callback(
                                trial_id=trial_id,
                                total_trials=len(trials_params),
                                completed=completed_trials,
                                failed=failed_trials,
                                best_score=best_score,
                                best_params=best_trial.hyperparameters if best_trial else None
                            )
                    else:
                        failed_trials += 1
                        logger.warning(f"试验 {trial_id} 失败: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed_trials += 1
                    logger.error(f"试验 {trial_id} 执行异常: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"并行搜索完成: 总耗时={duration:.2f}秒, "
            f"完成={completed_trials}, 失败={failed_trials}, "
            f"最优得分={best_score:.6f}"
        )
        
        if best_trial is None:
            raise RuntimeError("所有超参数试验都失败了")
        
        return best_trial
    
    def _execute_sequential_search(
        self,
        trials_params: List[Dict[str, Any]],
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        progress_callback: Optional[Callable] = None
    ) -> HyperparameterTrial:
        """执行顺序搜索（试验数较少时使用）"""
        start_time = datetime.now()
        best_score = float('-inf')
        best_trial: Optional[HyperparameterTrial] = None
        
        for trial_id, params in enumerate(trials_params):
            try:
                logger.info(f"试验 {trial_id + 1}/{len(trials_params)}: {params}")
                
                # 训练和评估
                metrics = train_fn(params)
                score = metrics.get('score', metrics.get('accuracy', 0.0))
                
                trial = HyperparameterTrial(
                    trial_id=trial_id,
                    hyperparameters=params,
                    score=score,
                    metrics=metrics,
                    status='completed'
                )
                
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    logger.info(f"发现更好的超参数组合，得分: {score:.6f}")
                
                # 进度回调
                if progress_callback:
                    progress_callback(
                        trial_id=trial_id,
                        total_trials=len(trials_params),
                        completed=trial_id + 1,
                        failed=0,
                        best_score=best_score,
                        best_params=best_trial.hyperparameters if best_trial else None
                    )
                    
            except Exception as e:
                logger.error(f"试验 {trial_id} 失败: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"顺序搜索完成: 总耗时={duration:.2f}秒, 最优得分={best_score:.6f}")
        
        if best_trial is None:
            raise RuntimeError("所有超参数试验都失败了")
        
        return best_trial
    
    def _generate_random_trials(
        self,
        param_space: Dict[str, HyperparameterSpace],
        n_trials: int
    ) -> List[Dict[str, Any]]:
        """生成随机超参数组合"""
        import random
        
        trials = []
        for _ in range(n_trials):
            params = {}
            for param_name, space in param_space.items():
                if space.param_type == 'float':
                    params[param_name] = round(
                        random.uniform(space.min_value, space.max_value),
                        4
                    )
                elif space.param_type == 'int':
                    params[param_name] = random.randint(
                        space.min_value, space.max_value
                    )
                elif space.param_type == 'categorical':
                    params[param_name] = random.choice(space.choices)
            trials.append(params)
        
        return trials
    
    def _generate_grid_combinations(
        self,
        param_space: Dict[str, HyperparameterSpace]
    ) -> List[Dict[str, Any]]:
        """生成网格搜索超参数组合"""
        from itertools import product
        
        param_names = list(param_space.keys())
        param_values_list = []
        
        for param_name in param_names:
            space = param_space[param_name]
            if space.param_type == 'float':
                # 生成浮点数范围
                step = (space.max_value - space.min_value) / (space.step or 10)
                values = [
                    round(space.min_value + i * step, 4)
                    for i in range(int((space.max_value - space.min_value) / step) + 1)
                ]
            elif space.param_type == 'int':
                # 生成整数范围
                step = space.step or 1
                values = list(range(space.min_value, space.max_value + 1, step))
            elif space.param_type == 'categorical':
                values = space.choices
            else:
                values = [space.min_value]  # 默认值
            
            param_values_list.append(values)
        
        # 生成所有组合
        combinations = []
        for combination in product(*param_values_list):
            params = dict(zip(param_names, combination))
            combinations.append(params)
        
        return combinations


def _train_and_evaluate_worker(
    train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    params: Dict[str, Any],
    trial_id: int
) -> Dict[str, Any]:
    """
    工作进程函数：训练模型并评估
    
    注意：这个函数必须是模块级函数，才能被pickle序列化
    传递给ProcessPoolExecutor的函数必须是可序列化的
    
    Args:
        train_fn: 训练函数
        params: 超参数
        trial_id: 试验ID
    
    Returns:
        结果字典，包含success, score, metrics, params, error等字段
    """
    try:
        # 训练和评估
        metrics = train_fn(params)
        
        # 提取得分
        score = metrics.get('score', metrics.get('accuracy', metrics.get('f1_score', 0.0)))
        
        return {
            'success': True,
            'score': float(score),
            'metrics': metrics,
            'params': params,
            'trial_id': trial_id
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        return {
            'success': False,
            'score': float('-inf'),
            'metrics': {},
            'params': params,
            'trial_id': trial_id,
            'error': error_msg,
            'traceback': error_traceback
        }
