"""
回测进度监控器

提供详细的回测进度跟踪和WebSocket实时推送功能
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.infrastructure.websocket_manager import websocket_manager


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


class BacktestProgressMonitor:
    """回测进度监控器"""

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
            stages=[stage for stage in self.stage_definitions],  # 复制阶段定义
        )

        self.active_backtests[task_id] = progress_data

        # 发送开始监控消息
        await self._notify_progress_update(task_id, "backtest_started")

        logger.info(f"开始监控回测进度: {task_id}, 预计交易日: {total_trading_days}")
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
            logger.warning(f"尝试更新不存在的回测进度: {task_id}")
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

        # 发送进度更新通知
        await self._notify_progress_update(task_id, "stage_updated")

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

        # 更新回测执行阶段的进度
        if progress_data.total_trading_days > 0:
            execution_progress = min(
                processed_days / progress_data.total_trading_days * 100, 100
            )
            await self.update_stage(
                task_id,
                "backtest_execution",
                progress=execution_progress,
                status="running",
                details={
                    "processed_days": processed_days,
                    "total_days": progress_data.total_trading_days,
                    "current_date": current_date,
                    "signals_generated": signals_generated,
                    "trades_executed": trades_executed,
                    "portfolio_value": portfolio_value,
                },
            )

        # 每10天或每10%进度发送一次更新
        if processed_days % 10 == 0 or (
            processed_days % max(1, progress_data.total_trading_days // 10) == 0
        ):
            await self._notify_progress_update(task_id, "execution_progress")

    async def add_warning(self, task_id: str, warning_message: str):
        """添加警告信息"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]
        progress_data.warnings.append(
            {"message": warning_message, "timestamp": datetime.utcnow().isoformat()}
        )

        # 发送警告通知
        await self._notify_progress_update(
            task_id, "warning_added", {"warning": warning_message}
        )

        logger.warning(f"回测警告 {task_id}: {warning_message}")

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

        # 发送错误通知
        await self._notify_progress_update(
            task_id, "backtest_failed", {"error": error_message}
        )

        logger.error(f"回测错误 {task_id}: {error_message}")

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

        # 发送完成通知
        await self._notify_progress_update(
            task_id, "backtest_completed", {"results": final_results or {}}
        )

        logger.info(f"回测监控完成: {task_id}")

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

        # 发送取消通知
        await self._notify_progress_update(
            task_id, "backtest_cancelled", {"reason": reason}
        )

        # 清理监控数据
        del self.active_backtests[task_id]

        logger.info(f"回测已取消: {task_id}, 原因: {reason}")

    def get_progress_data(self, task_id: str) -> Optional[BacktestProgressData]:
        """获取进度数据"""
        return self.active_backtests.get(task_id)

    def get_all_active_backtests(self) -> Dict[str, BacktestProgressData]:
        """获取所有活跃的回测"""
        return self.active_backtests.copy()

    async def cleanup_completed_backtests(self, max_age_hours: int = 24):
        """清理已完成的回测监控数据"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        to_remove = []
        for task_id, progress_data in self.active_backtests.items():
            if progress_data.start_time and progress_data.start_time < cutoff_time:
                # 检查是否已完成或失败
                all_completed = all(
                    stage.status in ["completed", "failed", "cancelled"]
                    for stage in progress_data.stages
                )
                if all_completed:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self.active_backtests[task_id]
            logger.info(f"清理已完成的回测监控数据: {task_id}")

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

    async def _notify_progress_update(
        self, task_id: str, update_type: str, extra_data: Dict[str, Any] = None
    ):
        """发送进度更新通知"""
        if task_id not in self.active_backtests:
            return

        progress_data = self.active_backtests[task_id]

        # 构建通知数据
        notification_data = {
            "task_id": task_id,
            "backtest_id": progress_data.backtest_id,
            "update_type": update_type,
            "overall_progress": progress_data.overall_progress,
            "current_stage": progress_data.current_stage,
            "processed_days": progress_data.processed_trading_days,
            "total_days": progress_data.total_trading_days,
            "current_date": progress_data.current_date,
            "processing_speed": progress_data.processing_speed,
            "estimated_completion": progress_data.estimated_completion.isoformat()
            if progress_data.estimated_completion
            else None,
            "elapsed_time": str(progress_data.elapsed_time)
            if progress_data.elapsed_time
            else None,
            "portfolio_value": progress_data.current_portfolio_value,
            "signals_generated": progress_data.total_signals_generated,
            "trades_executed": progress_data.total_trades_executed,
            "warnings_count": len(progress_data.warnings),
            "error_message": progress_data.error_message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # 添加阶段详情
        notification_data["stages"] = [
            {
                "name": stage.stage_name,
                "description": stage.stage_description,
                "progress": stage.progress,
                "status": stage.status,
                "start_time": stage.start_time.isoformat()
                if stage.start_time
                else None,
                "end_time": stage.end_time.isoformat() if stage.end_time else None,
                "details": stage.details,
            }
            for stage in progress_data.stages
        ]

        # 添加额外数据
        if extra_data:
            notification_data.update(extra_data)

        # 发送WebSocket通知（子进程中跳过，避���死锁）
        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess':
            try:
                await websocket_manager.notify_task_status(
                    task_id=task_id,
                    status=update_type,
                    progress=progress_data.overall_progress,
                    result=notification_data,
                )
            except Exception as ws_err:
                logger.debug(f"WebSocket通知失败（非致命）: {ws_err}")
        else:
            logger.debug(f"子进程中跳过WebSocket通知: task={task_id}, type={update_type}")


# 全局进度监控器实例
backtest_progress_monitor = BacktestProgressMonitor()
