"""
按使用量计费 - Laya 决策模型服务

使用 Convai Innovations 的 Laya 模型辅助交易决策。
Laya 是一个非自回归 System 1 决策模型，擅长：
- 分类选择（买/卖/持）
- 置信度评分
- 布尔概率判断（是否突破/风险高）
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import asyncio
import logging

try:
    import laya
except ImportError:  # pragma: no cover - laya 未安装时路由降级为 503
    laya = None  # type: ignore[assignment]

from app.core.config import settings
from app.services.commerce.commission_service import CommerceService

logger = logging.getLogger(__name__)


@dataclass
class LayaDecision:
    """Laya 决策结果。"""
    
    action: str
    confidence: float
    action_probabilities: Dict[str, float] = field(default_factory=dict)
    additional_results: Dict[str, Any] = field(default_factory=dict)
    model_name: str = "laya"
    latency_ms: float = 0.0
    questions_count: int = 0
    event_type: str = "ai_assisted_decision"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "action_probabilities": self.action_probabilities,
            "additional_results": self.additional_results,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "questions_count": self.questions_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def __repr__(self) -> str:
        return f"LayaDecision(action={self.action}, confidence={self.confidence:.2f})"


@dataclass
class TradingState:
    """交易状态输入。"""
    
    symbol: str = ""
    current_price: float = 0.0
    previous_close: float = 0.0
    high_52week: float = 0.0
    low_52week: float = 0.0
    rsi: float = 50.0
    macd_signal: str = "neutral"
    moving_average_20: float = 0.0
    moving_average_50: float = 0.0
    moving_average_200: float = 0.0
    volatility_20d: float = 0.0
    atr_14: float = 0.0
    volume_ratio: float = 1.0
    avg_volume_20d: float = 0.0
    pe_ratio: float = 0.0
    market_cap: float = 0.0
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    position_size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    market_regime: str = "trending"
    sector_momentum: str = "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "previous_close": self.previous_close,
            "high_52week": self.high_52week,
            "low_52week": self.low_52week,
            "rsi": self.rsi,
            "macd_signal": self.macd_signal,
            "moving_average_20": self.moving_average_20,
            "moving_average_50": self.moving_average_50,
            "moving_average_200": self.moving_average_200,
            "volatility_20d": self.volatility_20d,
            "atr_14": self.atr_14,
            "volume_ratio": self.volume_ratio,
            "avg_volume_20d": self.avg_volume_20d,
            "pe_ratio": self.pe_ratio,
            "market_cap": self.market_cap,
            "news_sentiment": self.news_sentiment,
            "social_sentiment": self.social_sentiment,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "market_regime": self.market_regime,
            "sector_momentum": self.sector_momentum,
        }


class LayaService:
    """Laya 决策模型服务（单例模式）。"""
    
    _instance: Optional["LayaService"] = None
    _agent: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._agent = None
            self._model_name = "convaiinnovations/laya"
            self._commercial_service = CommerceService()
    
    async def initialize(self) -> bool:
        if laya is None:
            logger.error("laya 包未安装，Laya 决策模型不可用（pip install laya 后重启）")
            self._initialized = False
            return False
        try:
            logger.info(f"正在加载 Laya 模型: {self._model_name}")
            loop = asyncio.get_event_loop()
            self._agent = await loop.run_in_executor(None, laya.load, self._model_name)
            self._initialized = True
            logger.info("Laya 模型加载成功 ✅")
            return True
        except Exception as e:
            logger.error(f"Laya 模型加载失败: {e}")
            self._initialized = False
            return False
    
    @property
    def is_ready(self) -> bool:
        return self._initialized and self._agent is not None
    
    def _build_questions(self, question_types: List[str]) -> Dict[str, Any]:
        questions = {}
        for qtype in question_types:
            if qtype == "action":
                questions["action"] = {
                    "type": "choice",
                    "instructions": "基于当前市场状态，应该执行什么操作？",
                    "criteria": {
                        "buy": "强烈看涨且风险可控",
                        "sell": "触发止损或看跌",
                        "hold": "趋势不明确或持有健康",
                        "reduce": "风险过高",
                        "pass": "不确定性过高"
                    }
                }
            elif qtype == "confidence":
                questions["confidence"] = {
                    "type": "score",
                    "instructions": "对当前决策的置信度",
                    "criteria": [1, 2, 3, 4, 5]
                }
            elif qtype == "is_breakout":
                questions["is_breakout"] = {
                    "type": "noul",
                    "instructions": "价格是否有效突破阻力位？"
                }
            elif qtype == "risk_high":
                questions["risk_high"] = {
                    "type": "noul",
                    "instructions": "当前持仓风险是否过高？"
                }
            elif qtype == "is_overbought":
                questions["is_overbought"] = {
                    "type": "noul",
                    "instructions": "资产是否超买？"
                }
            elif qtype == "is_oversold":
                questions["is_oversold"] = {
                    "type": "noul",
                    "instructions": "资产是否超卖？"
                }
            elif qtype == "trend_continuation":
                questions["trend_continuation"] = {
                    "type": "noul",
                    "instructions": "当前趋势是否大概率延续？"
                }
        return questions
    
    async def analyze(
        self,
        state: TradingState,
        question_types: List[str],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[LayaDecision]:
        if not self.is_ready:
            logger.warning("Laya 服务未就绪，跳过分析")
            return None
        
        start_time = datetime.now()
        state_dict = state.to_dict()
        questions = self._build_questions(question_types)
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._agent.predict, state_dict, questions
            )
            
            answers = result.get("answers", {})
            
            decision = LayaDecision(
                action=answers.get("action", {}).get("choice", "pass"),
                confidence=answers.get("action", {}).get("confidence", 0.0),
                action_probabilities=answers.get("action", {}).get("probabilities", {}),
                additional_results={k: v for k, v in answers.items() if k != "action"},
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                questions_count=len(questions)
            )
            
            if user_id:
                await self._record_usage(user_id, decision, state_dict, metadata or {})
            
            logger.info(
                f"Laya 决策完成: action={decision.action}, "
                f"confidence={decision.confidence:.2f}, "
                f"latency={decision.latency_ms:.1f}ms"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Laya 分析失败: {e}")
            return None
    
    async def _record_usage(
        self,
        user_id: str,
        decision: LayaDecision,
        state: Dict[str, Any],
        extra_metadata: Dict[str, Any]
    ):
        try:
            await self._commercial_service.record_usage(
                user_id=user_id,
                event_type="ai_assisted_decision",
                quantity=1,
                metadata={
                    "model": "laya",
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "questions_count": decision.questions_count,
                    "latency_ms": decision.latency_ms,
                    "symbol": state.get("symbol", ""),
                    **extra_metadata
                }
            )
            logger.debug(f"Laya 计费用量已记录: user={user_id}")
        except Exception as e:
            logger.warning(f"Laya 计费记录失败: {e}")
    
    async def enhance_signal(
        self,
        base_signal: Dict[str, Any],
        state: TradingState,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.is_ready:
            return {
                "original_signal": base_signal,
                "laya_enhanced": None,
                "final_action": base_signal.get("action", "hold"),
                "fallback_reason": "laya_not_ready"
            }
        
        question_types = ["action", "confidence"]
        
        signal_type = base_signal.get("type", "")
        if signal_type in ["breakout", "reversal"]:
            question_types.append("is_breakout")
        if base_signal.get("has_stop_loss"):
            question_types.append("risk_high")
        
        decision = await self.analyze(
            state=state,
            question_types=question_types,
            user_id=user_id,
            metadata={"base_signal": base_signal}
        )
        
        final_action = base_signal.get("action", "hold")
        fallback_reason = None
        
        if decision:
            if decision.confidence < 0.4:
                final_action = base_signal.get("action", "hold")
                fallback_reason = "laya_confidence_low"
            else:
                final_action = decision.action
        
        return {
            "original_signal": base_signal,
            "laya_enhanced": decision.to_dict() if decision else None,
            "final_action": final_action,
            "fallback_reason": fallback_reason
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        return {
            "service": "laya",
            "is_ready": self.is_ready,
            "model_name": self._model_name,
            "initialized": self._initialized,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# 预定义问题模板
TRADING_QUESTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "action_only": {
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "应该执行什么操作？",
                "criteria": {
                    "buy": "强烈看涨",
                    "sell": "强烈看跌",
                    "hold": "中性/观望",
                    "reduce": "风险过高",
                    "pass": "不确定性过高"
                }
            }
        },
        "description": "仅获取操作建议"
    },
    "action_with_confidence": {
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "应该执行什么操作？",
                "criteria": {
                    "buy": "强烈看涨",
                    "sell": "强烈看跌",
                    "hold": "中性/观望",
                    "reduce": "风险过高",
                    "pass": "不确定性过高"
                }
            },
            "confidence": {
                "type": "score",
                "instructions": "决策置信度",
                "criteria": [1, 2, 3, 4, 5]
            }
        },
        "description": "操作建议 + 置信度"
    },
    "full_analysis": {
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "应该执行什么操作？",
                "criteria": {
                    "buy": "强烈看涨",
                    "sell": "强烈看跌",
                    "hold": "中性/观望",
                    "reduce": "风险过高",
                    "pass": "不确定性过高"
                }
            },
            "confidence": {
                "type": "score",
                "instructions": "决策置信度",
                "criteria": [1, 2, 3, 4, 5]
            },
            "is_breakout": {
                "type": "noul",
                "instructions": "价格是否有效突破阻力位？"
            },
            "risk_high": {
                "type": "noul",
                "instructions": "当前持仓风险是否过高？"
            },
            "trend_continuation": {
                "type": "noul",
                "instructions": "当前趋势是否大概率延续？"
            }
        },
        "description": "完整分析"
    },
    "risk_assessment": {
        "questions": {
            "risk_high": {
                "type": "noul",
                "instructions": "当前持仓风险是否过高？"
            },
            "is_overbought": {
                "type": "noul",
                "instructions": "是否超买？"
            },
            "is_oversold": {
                "type": "noul",
                "instructions": "是否超卖？"
            },
            "stop_loss_hit": {
                "type": "noul",
                "instructions": "是否触及止损位？"
            }
        },
        "description": "风险评估"
    }
}