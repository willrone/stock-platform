#!/usr/bin/env python3
"""
测试回测WebSocket端点

验证WebSocket端点是否正常工作

注意：这是一个独立脚本，需要运行中的后端服务器。
不适合作为 pytest 单元测试运行（依赖 websockets/aiohttp 连接真实服务器）。
"""

import pytest

pytest.skip(
    "此文件是独立集成测试脚本，需要运行中的后端服务器，不适合 pytest 单元测试",
    allow_module_level=True,
)

import asyncio
import json
import sys
from pathlib import Path

# 添加backend目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

import websockets
from loguru import logger


async def test_websocket_connection():
    """测试WebSocket连接"""
    
    # 测试任务ID（使用一个已存在的回测任务）
    task_id = "test_task_001"
    
    # WebSocket URL
    ws_url = f"ws://localhost:8000/api/v1/backtest/ws/{task_id}"
    
    logger.info(f"连接到WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info("WebSocket连接成功")
            
            # 接收连接确认消息
            message = await websocket.recv()
            data = json.loads(message)
            logger.info(f"收到消息: {data}")
            
            # 发送ping消息
            logger.info("发送ping消息")
            await websocket.send(json.dumps({"type": "ping"}))
            
            # 接收pong响应
            message = await websocket.recv()
            data = json.loads(message)
            logger.info(f"收到pong响应: {data}")
            
            # 请求当前进度
            logger.info("请求当前进度")
            await websocket.send(json.dumps({"type": "get_current_progress"}))
            
            # 接收进度数据
            message = await websocket.recv()
            data = json.loads(message)
            logger.info(f"收到进度数据: {data.get('type')}")
            
            logger.success("WebSocket测试成功！")
            return True
            
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"WebSocket连接失败，状态码: {e.status_code}")
        if e.status_code == 4004:
            logger.warning("任务不存在，这是预期的（测试任务ID不存在）")
            return True  # 这实际上是成功的，因为端点正常工作
        return False
    except Exception as e:
        logger.error(f"WebSocket测试失败: {e}")
        return False


async def test_http_endpoints():
    """测试HTTP端点"""
    import aiohttp
    
    base_url = "http://localhost:8000/api/v1/backtest"
    
    async with aiohttp.ClientSession() as session:
        # 测试统计端点
        logger.info("测试WebSocket统计端点")
        async with session.get(f"{base_url}/ws/stats") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"统计数据: {data}")
                logger.success("统计端点测试成功")
            else:
                logger.error(f"统计端点测试失败: {response.status}")
                return False
        
        # 测试进度端点（使用不存在的任务）
        logger.info("测试进度HTTP端点")
        task_id = "test_task_001"
        async with session.get(f"{base_url}/progress/{task_id}") as response:
            if response.status in [200, 404]:
                logger.info(f"进度端点响应: {response.status}")
                logger.success("进度端点测试成功")
            else:
                logger.error(f"进度端点测试失败: {response.status}")
                return False
    
    return True


async def main():
    """主函数"""
    logger.info("开始测试回测WebSocket端点")
    
    # 测试HTTP端点
    logger.info("\n=== 测试HTTP端点 ===")
    http_success = await test_http_endpoints()
    
    # 测试WebSocket连接
    logger.info("\n=== 测试WebSocket连接 ===")
    ws_success = await test_websocket_connection()
    
    # 总结
    logger.info("\n=== 测试总结 ===")
    logger.info(f"HTTP端点测试: {'✓ 通过' if http_success else '✗ 失败'}")
    logger.info(f"WebSocket连接测试: {'✓ 通过' if ws_success else '✗ 失败'}")
    
    if http_success and ws_success:
        logger.success("\n所有测试通过！WebSocket端点工作正常。")
        return 0
    else:
        logger.error("\n部分测试失败，请检查日志。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
