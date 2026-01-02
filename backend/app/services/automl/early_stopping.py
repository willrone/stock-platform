"""
早停策略实现
防止模型过拟合，提高训练效率
"""
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EarlyStoppingMode(Enum):
    """早停模式"""
    MIN = "min"  # 指标越小越好（如损失函数）
    MAX = "max"  # 指标越大越好（如准确率）

@dataclass
class EarlyStoppingConfig:
    """早停配置"""
    monitor: str = "val_loss"  # 监控的指标名称
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN  # 监控模式
    patience: int = 10  # 容忍轮数
    min_delta: float = 0.001  # 最小改进阈值
    restore_best_weights: bool = True  # 是否恢复最佳权重
    baseline: Optional[float] = None  # 基线值
    verbose: bool = True  # 是否输出详细信息
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'monitor': self.monitor,
            'mode': self.mode.value,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'restore_best_weights': self.restore_best_weights,
            'baseline': self.baseline,
            'verbose': self.verbose
        }

@dataclass
class EarlyStoppingState:
    """早停状态"""
    best_value: Optional[float] = None
    best_epoch: int = 0
    wait: int = 0
    stopped_epoch: int = 0
    should_stop: bool = False
    best_weights: Optional[Any] = None
    history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'should_stop': self.should_stop,
            'history': self.history
        }

class EarlyStopping:
    """早停策略实现"""
    
    def __init__(self, config: EarlyStoppingConfig):
        """
        初始化早停策略
        
        Args:
            config: 早停配置
        """
        self.config = config
        self.state = EarlyStoppingState()
        
        # 根据模式设置比较函数
        if config.mode == EarlyStoppingMode.MIN:
            self.is_better = lambda current, best: current < best - config.min_delta
        else:
            self.is_better = lambda current, best: current > best + config.min_delta
    
    def __call__(self, 
                 current_value: float, 
                 epoch: int, 
                 model_weights: Optional[Any] = None) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_value: 当前监控指标值
            epoch: 当前轮数
            model_weights: 当前模型权重（可选）
            
        Returns:
            是否应该停止训练
        """
        return self.update(current_value, epoch, model_weights)
    
    def update(self, 
               current_value: float, 
               epoch: int, 
               model_weights: Optional[Any] = None) -> bool:
        """
        更新早停状态
        
        Args:
            current_value: 当前监控指标值
            epoch: 当前轮数
            model_weights: 当前模型权重（可选）
            
        Returns:
            是否应该停止训练
        """
        # 添加到历史记录
        self.state.history.append(current_value)
        
        # 检查基线
        if self.config.baseline is not None:
            if self.config.mode == EarlyStoppingMode.MIN:
                if current_value >= self.config.baseline:
                    self.state.wait += 1
                else:
                    self.state.wait = 0
            else:
                if current_value <= self.config.baseline:
                    self.state.wait += 1
                else:
                    self.state.wait = 0
        
        # 初始化最佳值
        if self.state.best_value is None:
            self.state.best_value = current_value
            self.state.best_epoch = epoch
            if model_weights is not None and self.config.restore_best_weights:
                self.state.best_weights = self._copy_weights(model_weights)
            return False
        
        # 检查是否有改进
        if self.is_better(current_value, self.state.best_value):
            self.state.best_value = current_value
            self.state.best_epoch = epoch
            self.state.wait = 0
            
            if model_weights is not None and self.config.restore_best_weights:
                self.state.best_weights = self._copy_weights(model_weights)
            
            if self.config.verbose:
                logger.info(f"早停监控: 在第 {epoch} 轮发现更好的 {self.config.monitor}: {current_value:.6f}")
        else:
            self.state.wait += 1
            
            if self.config.verbose and self.state.wait > 0:
                logger.info(f"早停监控: {self.config.monitor} 没有改进，等待 {self.state.wait}/{self.config.patience}")
        
        # 检查是否应该停止
        if self.state.wait >= self.config.patience:
            self.state.should_stop = True
            self.state.stopped_epoch = epoch
            
            if self.config.verbose:
                logger.info(f"早停触发: 在第 {epoch} 轮停止训练，最佳 {self.config.monitor}: {self.state.best_value:.6f} (第 {self.state.best_epoch} 轮)")
            
            return True
        
        return False
    
    def _copy_weights(self, weights: Any) -> Any:
        """复制模型权重"""
        try:
            # 尝试使用深拷贝
            import copy
            return copy.deepcopy(weights)
        except Exception:
            # 如果深拷贝失败，返回原始权重
            logger.warning("无法复制模型权重，将返回原始权重")
            return weights
    
    def get_best_weights(self) -> Optional[Any]:
        """获取最佳权重"""
        return self.state.best_weights
    
    def reset(self):
        """重置早停状态"""
        self.state = EarlyStoppingState()
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'config': self.config.to_dict(),
            'state': self.state.to_dict()
        }

class OverfittingDetector:
    """过拟合检测器"""
    
    def __init__(self, 
                 patience: int = 5,
                 threshold: float = 0.1,
                 min_epochs: int = 10):
        """
        初始化过拟合检测器
        
        Args:
            patience: 容忍轮数
            threshold: 训练集和验证集性能差异阈值
            min_epochs: 最小训练轮数
        """
        self.patience = patience
        self.threshold = threshold
        self.min_epochs = min_epochs
        
        self.train_losses = []
        self.val_losses = []
        self.overfitting_count = 0
    
    def update(self, train_loss: float, val_loss: float, epoch: int) -> bool:
        """
        更新损失并检测过拟合
        
        Args:
            train_loss: 训练损失
            val_loss: 验证损失
            epoch: 当前轮数
            
        Returns:
            是否检测到过拟合
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 需要足够的训练轮数才开始检测
        if epoch < self.min_epochs:
            return False
        
        # 计算最近几轮的平均损失
        recent_epochs = min(5, len(self.train_losses))
        recent_train_loss = np.mean(self.train_losses[-recent_epochs:])
        recent_val_loss = np.mean(self.val_losses[-recent_epochs:])
        
        # 检查验证损失是否明显高于训练损失
        if recent_val_loss > recent_train_loss * (1 + self.threshold):
            self.overfitting_count += 1
        else:
            self.overfitting_count = 0
        
        # 如果连续多轮检测到过拟合，返回True
        if self.overfitting_count >= self.patience:
            logger.warning(f"检测到过拟合: 训练损失 {recent_train_loss:.6f}, 验证损失 {recent_val_loss:.6f}")
            return True
        
        return False
    
    def reset(self):
        """重置检测器状态"""
        self.train_losses = []
        self.val_losses = []
        self.overfitting_count = 0

class AdaptiveEarlyStopping:
    """自适应早停策略"""
    
    def __init__(self, 
                 initial_patience: int = 10,
                 patience_factor: float = 1.5,
                 max_patience: int = 50,
                 improvement_threshold: float = 0.01):
        """
        初始化自适应早停策略
        
        Args:
            initial_patience: 初始容忍轮数
            patience_factor: 容忍轮数增长因子
            max_patience: 最大容忍轮数
            improvement_threshold: 显著改进阈值
        """
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.max_patience = max_patience
        self.improvement_threshold = improvement_threshold
        
        self.current_patience = initial_patience
        self.best_value = None
        self.wait = 0
        self.significant_improvements = 0
    
    def update(self, current_value: float, epoch: int) -> bool:
        """
        更新自适应早停状态
        
        Args:
            current_value: 当前监控指标值
            epoch: 当前轮数
            
        Returns:
            是否应该停止训练
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # 检查是否有显著改进
        improvement = abs(current_value - self.best_value) / abs(self.best_value)
        
        if current_value < self.best_value:  # 假设越小越好
            self.best_value = current_value
            self.wait = 0
            
            # 如果改进显著，增加容忍轮数
            if improvement > self.improvement_threshold:
                self.significant_improvements += 1
                new_patience = min(
                    int(self.current_patience * self.patience_factor),
                    self.max_patience
                )
                if new_patience > self.current_patience:
                    logger.info(f"检测到显著改进，增加容忍轮数: {self.current_patience} -> {new_patience}")
                    self.current_patience = new_patience
        else:
            self.wait += 1
        
        # 检查是否应该停止
        if self.wait >= self.current_patience:
            logger.info(f"自适应早停触发: 等待 {self.wait} 轮，容忍轮数 {self.current_patience}")
            return True
        
        return False
    
    def reset(self):
        """重置自适应早停状态"""
        self.current_patience = self.initial_patience
        self.best_value = None
        self.wait = 0
        self.significant_improvements = 0

class EarlyStoppingManager:
    """早停管理器，支持多种早停策略"""
    
    def __init__(self):
        """初始化早停管理器"""
        self.strategies: Dict[str, EarlyStopping] = {}
        self.overfitting_detector = None
        self.adaptive_strategy = None
    
    def add_strategy(self, name: str, config: EarlyStoppingConfig):
        """添加早停策略"""
        self.strategies[name] = EarlyStopping(config)
        logger.info(f"添加早停策略: {name}")
    
    def add_overfitting_detector(self, 
                               patience: int = 5,
                               threshold: float = 0.1,
                               min_epochs: int = 10):
        """添加过拟合检测器"""
        self.overfitting_detector = OverfittingDetector(patience, threshold, min_epochs)
        logger.info("添加过拟合检测器")
    
    def add_adaptive_strategy(self, 
                            initial_patience: int = 10,
                            patience_factor: float = 1.5,
                            max_patience: int = 50):
        """添加自适应早停策略"""
        self.adaptive_strategy = AdaptiveEarlyStopping(
            initial_patience, patience_factor, max_patience
        )
        logger.info("添加自适应早停策略")
    
    def update(self, 
               metrics: Dict[str, float], 
               epoch: int,
               model_weights: Optional[Any] = None) -> Dict[str, bool]:
        """
        更新所有早停策略
        
        Args:
            metrics: 指标字典
            epoch: 当前轮数
            model_weights: 模型权重
            
        Returns:
            各策略的停止决策
        """
        results = {}
        
        # 更新普通早停策略
        for name, strategy in self.strategies.items():
            if strategy.config.monitor in metrics:
                should_stop = strategy.update(
                    metrics[strategy.config.monitor], 
                    epoch, 
                    model_weights
                )
                results[name] = should_stop
        
        # 更新过拟合检测器
        if self.overfitting_detector and 'train_loss' in metrics and 'val_loss' in metrics:
            overfitting_detected = self.overfitting_detector.update(
                metrics['train_loss'], 
                metrics['val_loss'], 
                epoch
            )
            results['overfitting_detector'] = overfitting_detected
        
        # 更新自适应策略
        if self.adaptive_strategy and 'val_loss' in metrics:
            adaptive_stop = self.adaptive_strategy.update(metrics['val_loss'], epoch)
            results['adaptive_strategy'] = adaptive_stop
        
        return results
    
    def should_stop(self, results: Dict[str, bool]) -> bool:
        """根据所有策略的结果决定是否停止"""
        return any(results.values())
    
    def get_best_weights(self, strategy_name: str) -> Optional[Any]:
        """获取指定策略的最佳权重"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].get_best_weights()
        return None
    
    def get_all_states(self) -> Dict[str, Any]:
        """获取所有策略的状态"""
        states = {}
        
        for name, strategy in self.strategies.items():
            states[name] = strategy.get_state()
        
        if self.overfitting_detector:
            states['overfitting_detector'] = {
                'train_losses': self.overfitting_detector.train_losses,
                'val_losses': self.overfitting_detector.val_losses,
                'overfitting_count': self.overfitting_detector.overfitting_count
            }
        
        if self.adaptive_strategy:
            states['adaptive_strategy'] = {
                'current_patience': self.adaptive_strategy.current_patience,
                'best_value': self.adaptive_strategy.best_value,
                'wait': self.adaptive_strategy.wait,
                'significant_improvements': self.adaptive_strategy.significant_improvements
            }
        
        return states
    
    def reset_all(self):
        """重置所有策略"""
        for strategy in self.strategies.values():
            strategy.reset()
        
        if self.overfitting_detector:
            self.overfitting_detector.reset()
        
        if self.adaptive_strategy:
            self.adaptive_strategy.reset()
        
        logger.info("所有早停策略已重置")

def create_default_early_stopping() -> EarlyStoppingManager:
    """创建默认的早停管理器"""
    manager = EarlyStoppingManager()
    
    # 添加验证损失早停策略
    val_loss_config = EarlyStoppingConfig(
        monitor="val_loss",
        mode=EarlyStoppingMode.MIN,
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=True
    )
    manager.add_strategy("val_loss", val_loss_config)
    
    # 添加验证准确率早停策略（如果有的话）
    val_acc_config = EarlyStoppingConfig(
        monitor="val_accuracy",
        mode=EarlyStoppingMode.MAX,
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=True
    )
    manager.add_strategy("val_accuracy", val_acc_config)
    
    # 添加过拟合检测器
    manager.add_overfitting_detector(patience=5, threshold=0.1, min_epochs=10)
    
    # 添加自适应策略
    manager.add_adaptive_strategy(initial_patience=10, patience_factor=1.5, max_patience=50)
    
    return manager