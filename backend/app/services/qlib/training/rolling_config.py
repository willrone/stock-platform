"""
滚动训练配置模块

定义滚动训练的配置类和常量。
"""

from dataclasses import dataclass
from enum import Enum


class RollingWindowType(Enum):
    """滚动窗口类型"""

    SLIDING = "sliding"  # 固定窗口滑动
    EXPANDING = "expanding"  # 扩展窗口


# === 默认常量 ===
DEFAULT_ROLLING_STEP = 60  # 每 60 个交易日滚动一次
DEFAULT_TRAIN_WINDOW = 480  # 训练窗口 480 天
DEFAULT_VALID_WINDOW = 60  # 验证窗口 60 天
DEFAULT_DECAY_RATE = 0.999  # 样本时间衰减率
MIN_TRAIN_SAMPLES = 1000  # 最小训练样本数
IC_DECAY_WARN_THRESHOLD = 0.03  # IC 衰减警告阈值


@dataclass
class RollingTrainingConfig:
    """滚动训练配置"""

    enable_rolling: bool = False
    window_type: str = "sliding"
    rolling_step: int = DEFAULT_ROLLING_STEP
    train_window: int = DEFAULT_TRAIN_WINDOW
    valid_window: int = DEFAULT_VALID_WINDOW
    enable_sample_decay: bool = True
    decay_rate: float = DEFAULT_DECAY_RATE

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "enable_rolling": self.enable_rolling,
            "window_type": self.window_type,
            "rolling_step": self.rolling_step,
            "train_window": self.train_window,
            "valid_window": self.valid_window,
            "enable_sample_decay": self.enable_sample_decay,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RollingTrainingConfig":
        """从字典创建配置"""
        return cls(
            enable_rolling=data.get("enable_rolling", False),
            window_type=data.get("window_type", "sliding"),
            rolling_step=data.get("rolling_step", DEFAULT_ROLLING_STEP),
            train_window=data.get("train_window", DEFAULT_TRAIN_WINDOW),
            valid_window=data.get("valid_window", DEFAULT_VALID_WINDOW),
            enable_sample_decay=data.get("enable_sample_decay", True),
            decay_rate=data.get("decay_rate", DEFAULT_DECAY_RATE),
        )
