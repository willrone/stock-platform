#!/usr/bin/env python3
"""
简单的回测进度监控测试

只测试核心数据结构和逻辑，不依赖外部模块
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest


@dataclass
class BacktestProgressStage:
    """回测进度阶段"""

    stage_name: str
    stage_description: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class BacktestProgressData:
    """回测进度数据"""

    task_id: str
    backtest_id: str
    overall_progress: float = 0.0
    current_stage: str = "initializing"
    stages: List[BacktestProgressStage] = None

    # 时间信息
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    elapsed_time: Optional[timedelta] = None

    # 处理统计
    total_trading_days: int = 0
    processed_trading_days: int = 0
    current_date: Optional[str] = None
    processing_speed: float = 0.0  # 天/秒

    # 交易统计
    total_signals_generated: int = 0
    total_trades_executed: int = 0
    current_portfolio_value: float = 0.0

    # 错误信息
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = []
        if self.warnings is None:
            self.warnings = []


class SimpleBacktestProgressMonitor:
    """简化的回测进度监控器（用于测试）"""

    def __init__(self):
        self.active_backtests: Dict[str, BacktestProgressData] = {}
        self.stage_definitions = self._define_stages()

    def _define_stages(self) -> List[BacktestProgressStage]:
        """定义回测阶段"""
        return [
            BacktestProgressStage(
                stage_name="initialization", stage_description="初始化回测环境"
            ),
            BacktestProgressStage(
                stage_name="data_loading", stage_description="加载股票数据"
            ),
            BacktestProgressStage(
                stage_name="strategy_setup", stage_description="设置交易策略"
            ),
            BacktestProgressStage(
                stage_name="backtest_execution", stage_description="执行回测计算"
            ),
            BacktestProgressStage(
                stage_name="metrics_calculation", stage_description="计算绩效指标"
            ),
            BacktestProgressStage(
                stage_name="report_generation", stage_description="生成回测报告"
            ),
            BacktestProgressStage(
                stage_name="data_storage", stage_description="保存结果数据"
            ),
        ]

    async def start_backtest_monitoring(
        self, task_id: str, backtest_id: str, total_trading_days: int = 0
    ) -> BacktestProgressData:
        """开始监控回测进度"""
        progress_data = BacktestProgressData(
            task_id=task_id,
            backtest_id=backtest_id,
            start_time=datetime.utcnow(),
            total_trading_days=total_trading_days,
            stages=[
                BacktestProgressStage(
                    stage_name=stage.stage_name,
                    stage_description=stage.stage_description,
                )
                for stage in self.stage_definitions
            ],  # 复制阶段定义
        )

        self.active_backtests[task_id] = progress_data
        print(f"📊 开始监控回测进度: {task_id}, 预计交易日: {total_trading_days}")
        return progress_data

    async def update_stage(
        self,
        task_id: str,
        stage_name: str,
        progress: float = None,
        status: str = None,
        details: Dict[str, Any] = None,
    ):
        """更新阶段进度"""
        if task_id not in self.active_backtests:
            print(f"⚠️ 尝试更新不存在的回测进度: {task_id}")
            return

        progress_data = self.active_backtests[task_id]

        # 查找并更新对应阶段
        for stage in progress_data.stages:
            if stage.stage_name == stage_name:
                if progress is not None:
                    stage.progress = progress
                if status is not None:
                    stage.status = status
                    if status == "running" and stage.start_time is None:
                        stage.start_time = datetime.utcnow()
                    elif status in ["completed", "failed"] and stage.end_time is None:
                        stage.end_time = datetime.utcnow()
                if details is not None:
                    stage.details.update(details)
                break

        # 更新当前阶段
        progress_data.current_stage = stage_name

        # 计算总体进度
        await self._calculate_overall_progress(task_id)

        print(f"🔄 阶段更新: {stage_name} -> {status} ({progress}%)")

    async def update_execution_progress(
        self,
        task_id: str,
        processed_days: int,
        current_date: str = None,
        signals_generated: int = 0,
        trades_executed: int = 0,
        portfolio_value: float = 0.0,
    ):
        """更新执行进度"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]
        progress_data.processed_trading_days = processed_days

        if current_date:
            progress_data.current_date = current_date

        progress_data.total_signals_generated += signals_generated
        progress_data.total_trades_executed += trades_executed
        progress_data.current_portfolio_value = portfolio_value

        # 计算处理速度
        if progress_data.start_time:
            elapsed = datetime.utcnow() - progress_data.start_time
            progress_data.elapsed_time = elapsed
            if elapsed.total_seconds() > 0:
                progress_data.processing_speed = (
                    processed_days / elapsed.total_seconds()
                )

        # 估算完成时间
        if progress_data.processing_speed > 0 and progress_data.total_trading_days > 0:
            remaining_days = progress_data.total_trading_days - processed_days
            remaining_seconds = remaining_days / progress_data.processing_speed
            progress_data.estimated_completion = datetime.utcnow() + timedelta(
                seconds=remaining_seconds
            )

        print(
            f"⚡ 执行进度: {processed_days}/{progress_data.total_trading_days} 天, 组合价值: {portfolio_value}"
        )

    async def add_warning(self, task_id: str, warning_message: str):
        """添加警告信息"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]
        progress_data.warnings.append(
            {"message": warning_message, "timestamp": datetime.utcnow().isoformat()}
        )

        print(f"⚠️ 回测警告 {task_id}: {warning_message}")

    async def set_error(self, task_id: str, error_message: str):
        """设置错误信息"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]
        progress_data.error_message = error_message

        # 将当前阶段标记为失败
        for stage in progress_data.stages:
            if stage.status == "running":
                stage.status = "failed"
                stage.end_time = datetime.utcnow()
                stage.details["error"] = error_message
                break

        print(f"❌ 回测错误 {task_id}: {error_message}")

    async def complete_backtest(
        self, task_id: str, final_results: Dict[str, Any] = None
    ):
        """完成回测监控"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]
        progress_data.overall_progress = 100.0

        # 标记所有阶段为完成
        for stage in progress_data.stages:
            if stage.status != "failed":
                stage.status = "completed"
                if stage.end_time is None:
                    stage.end_time = datetime.utcnow()

        print(f"✅ 回测监控完成: {task_id}")

    async def cancel_backtest(self, task_id: str, reason: str = "用户取消"):
        """取消回测"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]

        # 标记当前运行的阶段为取消
        for stage in progress_data.stages:
            if stage.status == "running":
                stage.status = "cancelled"
                stage.end_time = datetime.utcnow()
                stage.details["cancellation_reason"] = reason

        # 清理监控数据
        del self.active_backtests[task_id]

        print(f"🛑 回测已取消: {task_id}, 原因: {reason}")

    def get_progress_data(self, task_id: str) -> Optional[BacktestProgressData]:
        """获取进度数据"""
        return self.active_backtests.get(task_id)

    def get_all_active_backtests(self) -> Dict[str, BacktestProgressData]:
        """获取所有活跃的回测"""
        return self.active_backtests.copy()

    async def _calculate_overall_progress(self, task_id: str):
        """计算总体进度"""
        progress_data = self.active_backtests[task_id]

        # 基于阶段权重计算总体进度
        stage_weights = {
            "initialization": 5,
            "data_loading": 15,
            "strategy_setup": 5,
            "backtest_execution": 60,  # 主要时间消耗
            "metrics_calculation": 10,
            "report_generation": 3,
            "data_storage": 2,
        }

        total_weight = sum(stage_weights.values())
        weighted_progress = 0.0

        for stage in progress_data.stages:
            weight = stage_weights.get(stage.stage_name, 1)
            if stage.status == "completed":
                weighted_progress += weight
            elif stage.status == "running":
                weighted_progress += weight * (stage.progress / 100)

        progress_data.overall_progress = min(
            weighted_progress / total_weight * 100, 100
        )


@pytest.mark.asyncio
async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能...")

    monitor = SimpleBacktestProgressMonitor()
    task_id = "test_001"

    # 开始监控
    progress_data = await monitor.start_backtest_monitoring(task_id, "bt_001", 100)
    assert progress_data.task_id == task_id
    assert len(progress_data.stages) == 7

    # 更新阶段
    await monitor.update_stage(task_id, "data_loading", 100, "completed")
    progress_data = monitor.get_progress_data(task_id)
    data_stage = next(s for s in progress_data.stages if s.stage_name == "data_loading")
    assert data_stage.status == "completed"
    assert data_stage.progress == 100

    # 更新执行进度
    await monitor.update_execution_progress(task_id, 50, "2024-01-15", 10, 5, 105000)
    progress_data = monitor.get_progress_data(task_id)
    assert progress_data.processed_trading_days == 50
    assert progress_data.current_date == "2024-01-15"
    assert progress_data.current_portfolio_value == 105000

    # 添加警告
    await monitor.add_warning(task_id, "测试警告")
    progress_data = monitor.get_progress_data(task_id)
    assert len(progress_data.warnings) == 1

    # 完成回测
    await monitor.complete_backtest(task_id)
    progress_data = monitor.get_progress_data(task_id)
    assert progress_data.overall_progress == 100.0

    print("✅ 基本功能测试通过")
    return True


@pytest.mark.asyncio
async def test_error_handling():
    """测试错误处理"""
    print("🧪 测试错误处理...")

    monitor = SimpleBacktestProgressMonitor()
    task_id = "test_error"

    # 开始监控
    await monitor.start_backtest_monitoring(task_id, "bt_error")

    # 开始一个阶段
    await monitor.update_stage(task_id, "data_loading", status="running")

    # 设置错误
    await monitor.set_error(task_id, "模拟错误")

    progress_data = monitor.get_progress_data(task_id)
    assert progress_data.error_message == "模拟错误"

    # 检查阶段状态
    data_stage = next(s for s in progress_data.stages if s.stage_name == "data_loading")
    assert data_stage.status == "failed"

    print("✅ 错误处理测试通过")
    return True


@pytest.mark.asyncio
async def test_cancellation():
    """测试取消功能"""
    print("🧪 测试取消功能...")

    monitor = SimpleBacktestProgressMonitor()
    task_id = "test_cancel"

    # 开始监控
    await monitor.start_backtest_monitoring(task_id, "bt_cancel")

    # 开始一个阶段
    await monitor.update_stage(task_id, "backtest_execution", status="running")

    # 取消回测
    await monitor.cancel_backtest(task_id, "用户取消")

    # 检查是否已从活跃列表移除
    active_backtests = monitor.get_all_active_backtests()
    assert task_id not in active_backtests

    print("✅ 取消功能测试通过")
    return True


async def main():
    """主测试函数"""
    print("🚀 回测进度监控简单测试")
    print("=" * 40)

    tests = [
        test_basic_functionality,
        test_error_handling,
        test_cancellation,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 40)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("💥 部分测试失败！")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
