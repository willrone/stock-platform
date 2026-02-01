# 回测WebSocket导入错误修复

## 问题描述

后端服务启动失败，错误信息：
```
ImportError: cannot import name 'get_db_session' from 'app.core.database'
```

## 问题原因

在 `backend/app/api/v1/backtest_websocket.py` 中错误地导入了不存在的 `get_db_session` 函数。

## 解决方案

### 1. 修改导入语句

**修改前：**
```python
from app.core.database import get_db_session
```

**修改后：**
```python
from app.core.database import SessionLocal
```

### 2. 添加数据库会话依赖函数

在 `backtest_websocket.py` 中添加：
```python
def get_db():
    """获取数据库会话依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 3. 更新所有使用点

将所有 `Depends(get_db_session)` 替换为 `Depends(get_db)`：

- `backtest_progress_websocket` 函数
- `get_backtest_progress` 函数
- `cancel_backtest` 函数

## 修改的文件

- `backend/app/api/v1/backtest_websocket.py`

## 验证

```bash
# 编译检查
python3 -m py_compile backend/app/api/v1/backtest_websocket.py
python3 -m py_compile backend/app/api/v1/api.py

# 两个命令都应该成功执行，无错误输出
```

## 技术说明

### 为什么使用同步Session而不是AsyncSession？

1. **WebSocket端点的特殊性**: FastAPI的WebSocket端点虽然是异步的，但在处理数据库查询时，我们使用同步的TaskRepository
2. **兼容性**: 现有的TaskRepository使用同步的SQLAlchemy session
3. **简单性**: 对于简单的数据库查询（如验证任务存在），同步session足够且更简单

### 数据库会话管理

```python
# 同步session（用于WebSocket和简单查询）
SessionLocal = sessionmaker(sync_engine, expire_on_commit=False)

# 异步session（用于复杂的异步操作）
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
```

## 后续建议

如果需要在WebSocket端点中执行复杂的数据库操作，考虑：

1. 将TaskRepository改造为支持异步操作
2. 使用 `get_async_session` 依赖
3. 在WebSocket处理函数中使用 `async with` 语法

示例：
```python
from app.core.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession

@router.websocket("/ws/{task_id}")
async def backtest_progress_websocket(
    websocket: WebSocket, 
    task_id: str
):
    # 手动获取异步session
    async for session in get_async_session():
        # 使用session进行异步查询
        ...
```

## 状态

✅ 已修复
✅ 语法检查通过
⏳ 等待服务启动验证