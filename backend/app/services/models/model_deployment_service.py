"""
模型评估和部署服务
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import asyncio
from enum import Enum
from loguru import logger

from ..model_storage import ModelStorage, ModelMetadata, ModelStatus, ModelType
from app.core.error_handler import ModelError, ErrorSeverity, ErrorContext
from app.core.logging_config import PerformanceLogger, AuditLogger


class DeploymentStatus(Enum):
    """部署状态"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLBACK = "rollback"
    RETIRED = "retired"


class EvaluationMetric(Enum):
    """评估指标"""
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"


@dataclass
class ModelEvaluation:
    """模型评估结果"""
    model_id: str
    evaluation_id: str
    evaluation_date: datetime
    evaluator: str
    
    # 性能指标
    performance_metrics: Dict[str, float]
    
    # 稳定性测试
    stability_metrics: Dict[str, float]
    
    # 业务指标
    business_metrics: Dict[str, float]
    
    # 评估详情
    evaluation_period: Dict[str, str]  # start_date, end_date
    test_data_info: Dict[str, Any]
    
    # 评估结论
    overall_score: float
    recommendation: str  # deploy, reject, retrain
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['evaluation_date'] = self.evaluation_date.isoformat()
        return data


@dataclass
class DeploymentConfig:
    """部署配置"""
    model_id: str
    deployment_name: str
    deployment_type: str  # production, staging, canary
    
    # 流量配置
    traffic_percentage: float = 100.0
    canary_duration: Optional[int] = None  # 金丝雀部署持续时间（小时）
    
    # 性能要求
    max_latency_ms: int = 1000
    min_accuracy: float = 0.8
    
    # 监控配置
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, float] = None
    
    # 回滚配置
    auto_rollback: bool = True
    rollback_threshold: float = 0.1  # 性能下降阈值
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'error_rate': 0.05,
                'latency_p95': 2000,
                'accuracy_drop': 0.1
            }


@dataclass
class DeploymentRecord:
    """部署记录"""
    deployment_id: str
    model_id: str
    deployment_name: str
    status: DeploymentStatus
    config: DeploymentConfig
    
    deployed_by: str
    deployed_at: datetime
    updated_at: datetime
    
    # 部署信息
    deployment_info: Dict[str, Any]
    
    # 性能监控
    performance_history: List[Dict[str, Any]]
    
    # 状态历史
    status_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['deployed_at'] = self.deployed_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['config'] = asdict(self.config)
        return data


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_storage: ModelStorage, data_dir: str = "backend/data"):
        self.model_storage = model_storage
        self.data_dir = Path(data_dir)
        
        # 评估历史
        self.evaluation_history: Dict[str, List[ModelEvaluation]] = {}
    
    def evaluate_model(self, model_id: str, evaluator: str,
                      test_data: Optional[pd.DataFrame] = None,
                      evaluation_config: Optional[Dict[str, Any]] = None) -> ModelEvaluation:
        """评估模型"""
        try:
            evaluation_start = datetime.utcnow()
            evaluation_id = f"eval_{model_id}_{evaluation_start.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"开始评估模型: {model_id}")
            
            # 加载模型
            model, metadata = self.model_storage.load_model(model_id)
            
            # 准备测试数据
            if test_data is None:
                test_data = self._prepare_test_data(metadata, evaluation_config)
            
            # 性能评估
            performance_metrics = self._evaluate_performance(model, test_data, metadata)
            
            # 稳定性评估
            stability_metrics = self._evaluate_stability(model, test_data, metadata)
            
            # 业务指标评估
            business_metrics = self._evaluate_business_metrics(model, test_data, metadata)
            
            # 计算综合评分
            overall_score = self._calculate_overall_score(
                performance_metrics, stability_metrics, business_metrics
            )
            
            # 生成建议
            recommendation = self._generate_recommendation(
                overall_score, performance_metrics, stability_metrics, business_metrics
            )
            
            # 创建评估结果
            evaluation = ModelEvaluation(
                model_id=model_id,
                evaluation_id=evaluation_id,
                evaluation_date=evaluation_start,
                evaluator=evaluator,
                performance_metrics=performance_metrics,
                stability_metrics=stability_metrics,
                business_metrics=business_metrics,
                evaluation_period={
                    'start_date': test_data.index.min().isoformat(),
                    'end_date': test_data.index.max().isoformat()
                },
                test_data_info={
                    'samples': len(test_data),
                    'features': len(test_data.columns) - 1,  # 减去目标列
                    'date_range_days': (test_data.index.max() - test_data.index.min()).days
                },
                overall_score=overall_score,
                recommendation=recommendation
            )
            
            # 保存评估结果
            self._save_evaluation(evaluation)
            
            # 记录评估历史
            if model_id not in self.evaluation_history:
                self.evaluation_history[model_id] = []
            self.evaluation_history[model_id].append(evaluation)
            
            # 记录审计日志
            AuditLogger.log_user_action(
                action="evaluate_model",
                user_id=evaluator,
                resource=f"model:{model_id}",
                success=True,
                details={
                    "evaluation_id": evaluation_id,
                    "overall_score": overall_score,
                    "recommendation": recommendation
                }
            )
            
            logger.info(f"模型评估完成: {model_id}, 评分: {overall_score:.3f}, 建议: {recommendation}")
            
            return evaluation
            
        except Exception as e:
            raise ModelError(
                message=f"模型评估失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
                original_exception=e
            )
    
    def _prepare_test_data(self, metadata: ModelMetadata, 
                          evaluation_config: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """准备测试数据"""
        try:
            # 从配置获取测试参数
            config = evaluation_config or {}
            test_days = config.get('test_days', 30)
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=test_days)
            
            # 获取训练时使用的股票代码
            stock_codes = metadata.training_data_info.get('stock_codes', ['000001.SZ'])
            
            # 加载测试数据
            all_data = []
            for stock_code in stock_codes:
                stock_data = self._load_stock_data(stock_code, start_date, end_date)
                if len(stock_data) > 0:
                    all_data.append(stock_data)
            
            if not all_data:
                raise ModelError(
                    message="无法加载测试数据",
                    severity=ErrorSeverity.HIGH
                )
            
            # 合并数据
            test_data = pd.concat(all_data, axis=0)
            test_data = test_data.sort_index()
            
            # 只保留模型需要的特征列
            feature_columns = metadata.feature_columns
            if feature_columns:
                available_columns = [col for col in feature_columns if col in test_data.columns]
                test_data = test_data[available_columns + ['close']]  # 添加目标列
            
            return test_data
            
        except Exception as e:
            logger.error(f"准备测试数据失败: {e}")
            raise
    
    def _load_stock_data(self, stock_code: str, start_date, end_date) -> pd.DataFrame:
        """加载股票数据"""
        stock_data_dir = self.data_dir / "daily" / stock_code
        
        if not stock_data_dir.exists():
            return pd.DataFrame()
        
        # 加载数据文件
        data_files = list(stock_data_dir.glob("*.parquet"))
        if not data_files:
            return pd.DataFrame()
        
        # 合并数据
        all_data = []
        for file_path in data_files:
            try:
                df = pd.read_parquet(file_path)
                if 'date' in df.columns:
                    df = df.set_index('date')
                all_data.append(df)
            except Exception as e:
                logger.warning(f"加载数据文件失败: {file_path}, 错误: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        data = pd.concat(all_data, axis=0)
        data = data.sort_index()
        data = data[~data.index.duplicated(keep='first')]
        
        # 过滤日期范围
        data = data[(data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))]
        
        return data
    
    def _evaluate_performance(self, model: Any, test_data: pd.DataFrame, 
                            metadata: ModelMetadata) -> Dict[str, float]:
        """评估模型性能"""
        try:
            # 准备特征和目标
            feature_columns = metadata.feature_columns
            if not feature_columns:
                feature_columns = [col for col in test_data.columns if col != 'close']
            
            X = test_data[feature_columns]
            y = test_data['close'].pct_change().shift(-1).dropna()  # 下一日收益率
            
            # 对齐数据
            X, y = X.align(y, join='inner')
            
            if len(X) == 0:
                return {}
            
            # 预测
            y_pred = model.predict(X)
            
            # 计算指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred))
            }
            
            # 计算方向准确率
            direction_accuracy = np.mean(np.sign(y) == np.sign(y_pred))
            metrics['direction_accuracy'] = float(direction_accuracy)
            
            return metrics
            
        except Exception as e:
            logger.error(f"性能评估失败: {e}")
            return {}
    
    def _evaluate_stability(self, model: Any, test_data: pd.DataFrame,
                          metadata: ModelMetadata) -> Dict[str, float]:
        """评估模型稳定性"""
        try:
            metrics = {}
            
            # 时间窗口稳定性测试
            window_size = min(len(test_data) // 5, 100)  # 5个时间窗口
            if window_size < 10:
                return metrics
            
            feature_columns = metadata.feature_columns
            if not feature_columns:
                feature_columns = [col for col in test_data.columns if col != 'close']
            
            window_scores = []
            for i in range(0, len(test_data) - window_size, window_size):
                window_data = test_data.iloc[i:i + window_size]
                X = window_data[feature_columns]
                y = window_data['close'].pct_change().shift(-1).dropna()
                
                X, y = X.align(y, join='inner')
                if len(X) < 5:
                    continue
                
                try:
                    y_pred = model.predict(X)
                    from sklearn.metrics import r2_score
                    score = r2_score(y, y_pred)
                    window_scores.append(score)
                except:
                    continue
            
            if window_scores:
                metrics['stability_mean'] = float(np.mean(window_scores))
                metrics['stability_std'] = float(np.std(window_scores))
                metrics['stability_min'] = float(np.min(window_scores))
                metrics['stability_max'] = float(np.max(window_scores))
            
            return metrics
            
        except Exception as e:
            logger.error(f"稳定性评估失败: {e}")
            return {}
    
    def _evaluate_business_metrics(self, model: Any, test_data: pd.DataFrame,
                                 metadata: ModelMetadata) -> Dict[str, float]:
        """评估业务指标"""
        try:
            metrics = {}
            
            feature_columns = metadata.feature_columns
            if not feature_columns:
                feature_columns = [col for col in test_data.columns if col != 'close']
            
            X = test_data[feature_columns]
            y = test_data['close'].pct_change().shift(-1).dropna()
            
            X, y = X.align(y, join='inner')
            if len(X) == 0:
                return metrics
            
            # 预测
            y_pred = model.predict(X)
            
            # 模拟交易策略
            # 简单策略：预测收益率 > 0 时买入，< 0 时卖出
            positions = np.where(y_pred > 0, 1, -1)  # 1: 买入, -1: 卖出
            strategy_returns = positions * y.values
            
            # 计算业务指标
            if len(strategy_returns) > 0:
                # 累计收益
                cumulative_returns = np.cumprod(1 + strategy_returns) - 1
                metrics['total_return'] = float(cumulative_returns[-1])
                
                # 年化收益率
                days = len(strategy_returns)
                if days > 0:
                    annual_return = (1 + cumulative_returns[-1]) ** (252 / days) - 1
                    metrics['annual_return'] = float(annual_return)
                
                # 夏普比率
                if np.std(strategy_returns) > 0:
                    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                    metrics['sharpe_ratio'] = float(sharpe_ratio)
                
                # 最大回撤
                peak = np.maximum.accumulate(1 + cumulative_returns)
                drawdown = (1 + cumulative_returns) / peak - 1
                metrics['max_drawdown'] = float(np.min(drawdown))
                
                # 胜率
                win_rate = np.mean(strategy_returns > 0)
                metrics['win_rate'] = float(win_rate)
                
                # 盈亏比
                wins = strategy_returns[strategy_returns > 0]
                losses = strategy_returns[strategy_returns < 0]
                if len(wins) > 0 and len(losses) > 0:
                    profit_factor = np.mean(wins) / abs(np.mean(losses))
                    metrics['profit_factor'] = float(profit_factor)
            
            return metrics
            
        except Exception as e:
            logger.error(f"业务指标评估失败: {e}")
            return {}
    
    def _calculate_overall_score(self, performance_metrics: Dict[str, float],
                               stability_metrics: Dict[str, float],
                               business_metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # 性能指标权重 (40%)
            if 'r2' in performance_metrics:
                score += max(0, performance_metrics['r2']) * 0.2
                weight_sum += 0.2
            
            if 'direction_accuracy' in performance_metrics:
                score += performance_metrics['direction_accuracy'] * 0.2
                weight_sum += 0.2
            
            # 稳定性指标权重 (30%)
            if 'stability_mean' in stability_metrics:
                score += max(0, stability_metrics['stability_mean']) * 0.15
                weight_sum += 0.15
            
            if 'stability_std' in stability_metrics:
                # 标准差越小越好，转换为正向指标
                stability_score = max(0, 1 - stability_metrics['stability_std'])
                score += stability_score * 0.15
                weight_sum += 0.15
            
            # 业务指标权重 (30%)
            if 'sharpe_ratio' in business_metrics:
                # 夏普比率标准化到0-1范围
                sharpe_score = max(0, min(1, (business_metrics['sharpe_ratio'] + 1) / 3))
                score += sharpe_score * 0.15
                weight_sum += 0.15
            
            if 'win_rate' in business_metrics:
                score += business_metrics['win_rate'] * 0.15
                weight_sum += 0.15
            
            # 标准化评分
            if weight_sum > 0:
                score = score / weight_sum
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"计算综合评分失败: {e}")
            return 0.0
    
    def _generate_recommendation(self, overall_score: float,
                               performance_metrics: Dict[str, float],
                               stability_metrics: Dict[str, float],
                               business_metrics: Dict[str, float]) -> str:
        """生成部署建议"""
        try:
            # 基于综合评分的基本建议
            if overall_score >= 0.8:
                base_recommendation = "deploy"
            elif overall_score >= 0.6:
                base_recommendation = "deploy"  # 可以部署但需要监控
            elif overall_score >= 0.4:
                base_recommendation = "retrain"
            else:
                base_recommendation = "reject"
            
            # 检查关键指标
            critical_issues = []
            
            # 检查方向准确率
            if performance_metrics.get('direction_accuracy', 0) < 0.5:
                critical_issues.append("方向准确率过低")
            
            # 检查稳定性
            if stability_metrics.get('stability_std', 1) > 0.3:
                critical_issues.append("模型不稳定")
            
            # 检查业务指标
            if business_metrics.get('sharpe_ratio', 0) < 0:
                critical_issues.append("夏普比率为负")
            
            # 如果有关键问题，降级建议
            if critical_issues:
                if base_recommendation == "deploy":
                    base_recommendation = "retrain"
                elif base_recommendation == "retrain":
                    base_recommendation = "reject"
            
            return base_recommendation
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return "reject"
    
    def _save_evaluation(self, evaluation: ModelEvaluation):
        """保存评估结果"""
        try:
            evaluation_dir = Path("backend/models/evaluations")
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            
            evaluation_file = evaluation_dir / f"{evaluation.evaluation_id}.json"
            
            with open(evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"评估结果已保存: {evaluation_file}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
    
    def get_evaluation_history(self, model_id: str) -> List[ModelEvaluation]:
        """获取模型评估历史"""
        return self.evaluation_history.get(model_id, [])


class ModelDeploymentService:
    """模型部署服务"""
    
    def __init__(self, model_storage: ModelStorage, evaluator: ModelEvaluator):
        self.model_storage = model_storage
        self.evaluator = evaluator
        
        # 部署记录
        self.deployments: Dict[str, DeploymentRecord] = {}
        self.active_deployments: Dict[str, str] = {}  # model_id -> deployment_id
        
        # 性能监控
        self.performance_monitor = ModelPerformanceMonitor()
        
        logger.info("模型部署服务初始化完成")
    
    def deploy_model(self, model_id: str, config: DeploymentConfig,
                    deployed_by: str, force: bool = False) -> str:
        """部署模型"""
        try:
            deployment_start = datetime.utcnow()
            deployment_id = f"deploy_{model_id}_{deployment_start.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"开始部署模型: {model_id} -> {deployment_id}")
            
            # 检查模型是否存在
            metadata = self.model_storage.get_model_metadata(model_id)
            if not metadata:
                raise ModelError(
                    message=f"模型不存在: {model_id}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(model_id=model_id)
                )
            
            # 检查模型状态
            if metadata.status not in [ModelStatus.TRAINED, ModelStatus.READY] and not force:
                raise ModelError(
                    message=f"模型状态不允许部署: {metadata.status}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(model_id=model_id)
                )
            
            # 预部署评估（如果需要）
            if not force:
                evaluation = self.evaluator.evaluate_model(model_id, deployed_by)
                if evaluation.recommendation == "reject":
                    raise ModelError(
                        message=f"模型评估不通过，建议: {evaluation.recommendation}",
                        severity=ErrorSeverity.MEDIUM,
                        context=ErrorContext(model_id=model_id)
                    )
            
            # 创建部署记录
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_id=model_id,
                deployment_name=config.deployment_name,
                status=DeploymentStatus.DEPLOYING,
                config=config,
                deployed_by=deployed_by,
                deployed_at=deployment_start,
                updated_at=deployment_start,
                deployment_info={
                    "deployment_type": config.deployment_type,
                    "traffic_percentage": config.traffic_percentage
                },
                performance_history=[],
                status_history=[{
                    "status": DeploymentStatus.DEPLOYING.value,
                    "timestamp": deployment_start.isoformat(),
                    "message": "开始部署"
                }]
            )
            
            # 执行部署
            try:
                self._execute_deployment(deployment_record)
                
                # 更新状态为已部署
                deployment_record.status = DeploymentStatus.DEPLOYED
                deployment_record.updated_at = datetime.utcnow()
                deployment_record.status_history.append({
                    "status": DeploymentStatus.DEPLOYED.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "部署成功"
                })
                
                # 更新模型状态
                metadata.status = ModelStatus.DEPLOYED
                metadata.deployment_info = {
                    "deployment_id": deployment_id,
                    "deployed_at": deployment_start.isoformat(),
                    "deployment_type": config.deployment_type
                }
                
                # 保存更新的元数据
                self.model_storage.save_model(
                    *self.model_storage.load_model(model_id), 
                    overwrite=True
                )
                
                # 记录部署
                self.deployments[deployment_id] = deployment_record
                self.active_deployments[model_id] = deployment_id
                
                # 启动性能监控
                if config.monitoring_enabled:
                    self.performance_monitor.start_monitoring(deployment_id, config)
                
                # 记录审计日志
                AuditLogger.log_user_action(
                    action="deploy_model",
                    user_id=deployed_by,
                    resource=f"model:{model_id}",
                    success=True,
                    details={
                        "deployment_id": deployment_id,
                        "deployment_type": config.deployment_type,
                        "traffic_percentage": config.traffic_percentage
                    }
                )
                
                logger.info(f"模型部署成功: {model_id} -> {deployment_id}")
                
                return deployment_id
                
            except Exception as e:
                # 部署失败，更新状态
                deployment_record.status = DeploymentStatus.FAILED
                deployment_record.updated_at = datetime.utcnow()
                deployment_record.status_history.append({
                    "status": DeploymentStatus.FAILED.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"部署失败: {str(e)}"
                })
                
                self.deployments[deployment_id] = deployment_record
                
                raise ModelError(
                    message=f"模型部署失败: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(model_id=model_id),
                    original_exception=e
                )
                
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(
                message=f"部署过程异常: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
                original_exception=e
            )
    
    def _execute_deployment(self, deployment_record: DeploymentRecord):
        """执行实际部署"""
        try:
            # 加载模型
            model, metadata = self.model_storage.load_model(deployment_record.model_id)
            
            # 根据部署类型执行不同的部署策略
            if deployment_record.config.deployment_type == "production":
                self._deploy_to_production(model, metadata, deployment_record)
            elif deployment_record.config.deployment_type == "staging":
                self._deploy_to_staging(model, metadata, deployment_record)
            elif deployment_record.config.deployment_type == "canary":
                self._deploy_canary(model, metadata, deployment_record)
            else:
                raise ValueError(f"不支持的部署类型: {deployment_record.config.deployment_type}")
            
            logger.info(f"部署执行完成: {deployment_record.deployment_id}")
            
        except Exception as e:
            logger.error(f"部署执行失败: {deployment_record.deployment_id}, 错误: {e}")
            raise
    
    def _deploy_to_production(self, model: Any, metadata: ModelMetadata, 
                            deployment_record: DeploymentRecord):
        """部署到生产环境"""
        # 这里实现生产环境部署逻辑
        # 例如：更新模型服务、重启预测服务等
        logger.info(f"部署到生产环境: {deployment_record.model_id}")
        
        # 模拟部署过程
        import time
        time.sleep(1)  # 模拟部署时间
        
        deployment_record.deployment_info.update({
            "environment": "production",
            "service_endpoint": f"/api/predictions/{deployment_record.model_id}",
            "health_check_url": f"/health/{deployment_record.deployment_id}"
        })
    
    def _deploy_to_staging(self, model: Any, metadata: ModelMetadata,
                         deployment_record: DeploymentRecord):
        """部署到测试环境"""
        logger.info(f"部署到测试环境: {deployment_record.model_id}")
        
        deployment_record.deployment_info.update({
            "environment": "staging",
            "service_endpoint": f"/api/staging/predictions/{deployment_record.model_id}",
            "health_check_url": f"/health/staging/{deployment_record.deployment_id}"
        })
    
    def _deploy_canary(self, model: Any, metadata: ModelMetadata,
                      deployment_record: DeploymentRecord):
        """金丝雀部署"""
        logger.info(f"金丝雀部署: {deployment_record.model_id}")
        
        deployment_record.deployment_info.update({
            "environment": "canary",
            "traffic_percentage": deployment_record.config.traffic_percentage,
            "canary_duration": deployment_record.config.canary_duration,
            "service_endpoint": f"/api/canary/predictions/{deployment_record.model_id}",
            "health_check_url": f"/health/canary/{deployment_record.deployment_id}"
        })
    
    def rollback_deployment(self, deployment_id: str, user_id: str, 
                          reason: str = "") -> bool:
        """回滚部署"""
        try:
            deployment_record = self.deployments.get(deployment_id)
            if not deployment_record:
                raise ModelError(
                    message=f"部署记录不存在: {deployment_id}",
                    severity=ErrorSeverity.MEDIUM
                )
            
            if deployment_record.status != DeploymentStatus.DEPLOYED:
                raise ModelError(
                    message=f"部署状态不允许回滚: {deployment_record.status}",
                    severity=ErrorSeverity.MEDIUM
                )
            
            logger.info(f"开始回滚部署: {deployment_id}, 原因: {reason}")
            
            # 执行回滚
            self._execute_rollback(deployment_record)
            
            # 更新状态
            deployment_record.status = DeploymentStatus.ROLLBACK
            deployment_record.updated_at = datetime.utcnow()
            deployment_record.status_history.append({
                "status": DeploymentStatus.ROLLBACK.value,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"回滚: {reason}",
                "user": user_id
            })
            
            # 停止监控
            self.performance_monitor.stop_monitoring(deployment_id)
            
            # 清理活跃部署记录
            model_id = deployment_record.model_id
            if self.active_deployments.get(model_id) == deployment_id:
                del self.active_deployments[model_id]
            
            # 更新模型状态
            metadata = self.model_storage.get_model_metadata(model_id)
            if metadata:
                metadata.status = ModelStatus.READY
                metadata.deployment_info = None
                # 这里应该保存更新的元数据，但为了简化暂时跳过
            
            # 记录审计日志
            AuditLogger.log_user_action(
                action="rollback_deployment",
                user_id=user_id,
                resource=f"deployment:{deployment_id}",
                success=True,
                details={"reason": reason}
            )
            
            logger.info(f"部署回滚成功: {deployment_id}")
            return True
            
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(
                message=f"回滚部署失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                original_exception=e
            )
    
    def _execute_rollback(self, deployment_record: DeploymentRecord):
        """执行回滚操作"""
        logger.info(f"执行回滚: {deployment_record.deployment_id}")
        
        # 这里实现具体的回滚逻辑
        # 例如：恢复之前的模型版本、重启服务等
        import time
        time.sleep(0.5)  # 模拟回滚时间
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """获取部署状态"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self, model_id: Optional[str] = None,
                        status: Optional[DeploymentStatus] = None) -> List[DeploymentRecord]:
        """列出部署记录"""
        deployments = list(self.deployments.values())
        
        if model_id:
            deployments = [d for d in deployments if d.model_id == model_id]
        
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        # 按部署时间排序
        deployments.sort(key=lambda x: x.deployed_at, reverse=True)
        
        return deployments
    
    def get_active_deployment(self, model_id: str) -> Optional[DeploymentRecord]:
        """获取模型的活跃部署"""
        deployment_id = self.active_deployments.get(model_id)
        if deployment_id:
            return self.deployments.get(deployment_id)
        return None


class ModelPerformanceMonitor:
    """模型性能监控器"""
    
    def __init__(self):
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def start_monitoring(self, deployment_id: str, config: DeploymentConfig):
        """开始监控部署"""
        logger.info(f"开始监控部署: {deployment_id}")
        
        # 这里可以实现实际的监控逻辑
        # 例如：定期检查模型性能、收集指标等
        self.performance_data[deployment_id] = []
    
    def stop_monitoring(self, deployment_id: str):
        """停止监控部署"""
        logger.info(f"停止监控部署: {deployment_id}")
        
        if deployment_id in self.monitoring_tasks:
            task = self.monitoring_tasks[deployment_id]
            task.cancel()
            del self.monitoring_tasks[deployment_id]
    
    def get_performance_data(self, deployment_id: str) -> List[Dict[str, Any]]:
        """获取性能数据"""
        return self.performance_data.get(deployment_id, [])