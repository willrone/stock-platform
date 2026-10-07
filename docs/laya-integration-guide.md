# Laya 决策模型集成文档

## 概览

本文档介绍如何将 HuggingFace 上的 Laya 决策模型集成到股票预测平台中，作为 AI 辅助决策的核心引擎。

## 什么是 Laya

**Laya** 是 Convai Innovations 开发的**非自回归 System 1 决策模型**：

- **类型**：系统决策模型 (Non-autoregressive Decision Model)
- **底层**：ModernBERT-large (22 层, 768 维)
- **速度**：38.4ms/问题 (批量 10 问题仅 156ms)
- **准确率**：83.8% (基准测试)
- **许可**：Apache 2.0 (开源免费)

### 核心能力

| 问题类型 | 返回值 | 应用场景 |
|---------|--------|---------|
| `choice` | 分类结果 + 概率分布 | 买/卖/持 决策 |
| `score` | 序数评分 (1-5) | 置信度评估 |
| `noul` | 布尔概率 (0-1) | 突破/风险判断 |

---

## 集成概览

### 1. 后端模块结构

```
backend/
├── app/
│   ├── services/
│   │   └── laya/
│   │       ├── __init__.py          # 核心服务
│   │       └── templates.py         # 问题模板
│   └── api/v1/
│       └── laya.py                  # API 路由
├── app/core/config.py               # 配置项
└── start_layered.py                 # 启动脚本
```

### 2. 前端 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/v1/laya/decision` | POST | 交易决策 |
| `/api/v1/laya/enhance` | POST | 信号增强 |
| `/api/v1/laya/signal/verify` | POST | 信号验证 |
| `/api/v1/laya/initialize` | POST | 初始化模型 |
| `/api/v1/laya/health` | GET | 健康检查 |

---

## 配置说明

### 环境变量

```bash
# .env 文件
LAYA_ENABLED=true              # 是否启用 Laya
LAYA_MODEL_NAME=convaiinnovations/laya  # 模型名称
LAYA_AUTO_LOAD=true           # 启动时自动加载
LAYA_MAX_BATCH_SIZE=10        # 最大批量查询
```

### 配置项 (`app/core/config.py`)

```python
# Laya 决策模型配置
LAYA_ENABLED: bool = False           # 是否启用
LAYA_MODEL_NAME: str = "convaiinnovations/laya"
LAYA_AUTO_LOAD: bool = True            # 启动时加载
LAYA_MAX_BATCH_SIZE: int = 10
```

---

## 使用指南

### 1. 快速开始

```python
from app.services.laya import LayaService, TradingState

# 初始化服务（单例模式）
laya_svc = LayaService()
await laya_svc.initialize()

# 构建交易状态
state = TradingState(
    symbol="AAPL",
    current_price=175.50,
    rsi=65.0,
    position_size=0.05,
    market_regime="trending"
)

# 执行决策
decision = await laya_svc.analyze(
    state=state,
    question_types=["action", "confidence"],
    user_id="user_123"
)

print(decision.action)      # "buy" / "sell" / "hold" / "reduce" / "pass"
print(decision.confidence)  # 0.0 - 1.0
```

### 2. 集成到回测流程

回测 API 成功完成后自动调用 Laya：

```python
# app/api/v1/backtest.py
if settings.LAYA_ENABLED:
    result["laya_enhanced"] = {
        "action": decision.action,
        "confidence": decision.confidence,
        "recommendation": "buy" if decision.confidence >= 0.6 else "hold"
    }
```

### 3. API 调用示例

**POST /api/v1/laya/decision**

```bash
curl -X POST http://localhost:8000/api/v1/laya/decision \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "symbol": "AAPL",
      "current_price": 175.50,
      "rsi": 65.0,
      "position_size": 0.05
    },
    "question_types": ["action", "confidence"]
  }'
```

**响应：**

```json
{
  "success": true,
  "data": {
    "decision": {
      "action": "buy",
      "confidence": 0.85,
      "action_probabilities": {
        "buy": 0.85,
        "sell": 0.03,
        "hold": 0.08,
        "reduce": 0.02,
        "pass": 0.02
      },
      "latency_ms": 42.5
    }
  }
}
```

---

## 计费集成

### 用量事件类型

```python
# 每次调用 Laya 决策都会计费
EVENT_TYPE = "ai_assisted_decision"

# 计费规则（在 commerce_pricing_rules 表中）
{
    "event_type": "ai_assisted_decision",
    "base_price_cents": 5,  # ¥0.05/次
    "tier_discounts": {"pro": "0.6", "enterprise": "0.3"}
}
```

### 计费元数据

```python
await commerce_service.record_usage(
    user_id=user_id,
    event_type="ai_assisted_decision",
    quantity=1,
    metadata={
        "model": "laya",
        "symbol": "AAPL",
        "action": "buy",
        "confidence": 0.85,
        "questions_count": 2,
        "latency_ms": 42.5
    }
)
```

---

## 问题模板

### 1. action_only

仅获取操作建议：

```python
questions = {
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
}
```

### 2. full_analysis

完整分析（推荐）：

```python
questions = {
    "action": {...},
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
}
```

### 3. risk_assessment

风险评估工具：

```python
questions = {
    "risk_high": {...},
    "is_overbought": {...},
    "is_oversold": {...},
    "stop_loss_hit": {...}
}
```

---

## 最佳实践

### 1. 置信度阈值

```python
# 置信度阈值建议
HIGH_CONFIDENCE = 0.7   # 高置信度
MEDIUM_CONFIDENCE = 0.5 # 中等置信度
LOW_CONFIDENCE = 0.4    # 低置信度

if decision.confidence >= HIGH_CONFIDENCE:
    action = decision.action
elif decision.confidence >= MEDIUM_CONFIDENCE:
    action = "hold"  # 谨慎行动
else:
    action = "pass"  # 观望
```

### 2. 结合多个模型

```python
# 集成多模型决策
laya_decision = await laya_svc.analyze(state, ["action", "confidence"])
llm_decision = await llm_svc.analyze(state)

# 加权融合
if laya_decision.confidence > 0.7 and llm_decision.agrees:
    final_action = laya_decision.action
else:
    final_action = "hold"
```

### 3. 监控与反馈

```python
# 记录决策结果用于模型改进
await feedback_service.record(
    decision_id=decision.id,
    action_taken=actual_action,
    outcome=trading_result,
    feedback=decision.confidence - abs(trading_result)
)
```

---

## 性能指标

### 延迟

| 场景 | 延迟 |
|------|------|
| 单问题 | 38ms |
| 10问题批量 | 156ms |
| 50问题批量 | 721ms |

### 资源占用

| 维度 | 用量 |
|------|------|
| 内存 | ~4GB (GPU) / ~6GB (CPU) |
| 存储 | ~800MB 权重文件 |
| 计算 | 受限于 BERT 推理 |

---

## 故障排查

### 1. 模型加载失败

```python
# 检查网络连接
import laya
agent = laya.load("convaiinnovations/laya")  # 需要能够访问 HuggingFace
```

**解决方案：**
- 确保能够访问 huggingface.co
- 检查网络代理设置
- 使用离线模型镜像

### 2. 内存不足

**解决方案：**
```python
# 使用小模型变体
from laya import config
config.model_name = "laya-mini"  # 如有
config.dtype = "float16"  # 降低精度省内存
```

### 3. 推理慢

**解决方案：**
```python
# 批量处理
agents = []
for question_batch in batch_questions:
    agent_pool.execute(question_batch)  # 并行推理
```

---

## 扩展规划

### 1. 金融微调

计划使用金融数据微调 Laya，使其更擅长：
- 股票筛选
- 行业轮动
- 市场情绪识别
- 事件驱动决策

### 2. 多模态输入

未来支持：
- 财报文本分析
- 新闻情绪识别
- 社交媒体情绪
- 实时行情数据

### 3. A/B 测试框架

```python
# 版本管理和 A/B 测试
decision_v1 = await laya_v1.analyze(state)
decision_v2 = await laya_v2.analyze(state)

# 记录实验结果
await experiment_service.record(
    variant="v2",
    metrics={"return": 0.05, "sharpe": 1.2, "drawdown": 0.03}
)
```

---

## 联系方式

- **HuggingFace 页面**：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- **文档与源码**：[GitHub](https://github.com/ConvaiInnovations/laya)
- **商业支持**：contact@convaiinnovations.com

---

**本集成文档最后更新：2026-09-23**