"""
P2 进度桥接模块

在主进程中运行一个后台线程，从 multiprocessing.Queue 读取
各 worker 的进度更新，聚合后写入数据库。

前端 WebSocket 通过读取数据库中的 progress_data 获取实时进度。
"""

import threading
import time
from multiprocessing import Queue
from typing import Any, Dict, Optional

import psycopg2
from loguru import logger


class ProgressBridge:
    """
    进度桥接器：聚合多个 worker 的进度，写入数据库

    生命周期：
    1. start() — 启动后台线程
    2. worker 通过 Queue 发送进度
    3. stop() — 停止线程，最终刷盘
    """

    # 最小 DB 写入间隔（秒）
    MIN_FLUSH_INTERVAL = 3.0

    def __init__(
        self,
        task_id: str,
        progress_queue: Queue,
        num_workers: int,
        db_url: str,
    ):
        self._task_id = str(task_id)
        self._queue = progress_queue
        self._num_workers = num_workers
        self._db_url = db_url

        # 每个 worker 的最新进度
        self._worker_progress: Dict[int, Dict[str, Any]] = {}
        self._workers_done: set = set()
        self._workers_error: Dict[int, str] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_flush_time = 0.0

    def start(self) -> None:
        """启动后台进度���集线程"""
        self._thread = threading.Thread(
            target=self._run,
            name=f"progress-bridge-{self._task_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"ProgressBridge 启动: task={self._task_id[:8]}, "
            f"workers={self._num_workers}"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """停止后台线程"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # 最终刷盘
        self._drain_queue()
        self._flush_to_db()
        logger.info(
            f"ProgressBridge 停止: task={self._task_id[:8]}"
        )

    def _run(self) -> None:
        """后台线程主循环"""
        while not self._stop_event.is_set():
            self._drain_queue()

            now = time.monotonic()
            if now - self._last_flush_time >= self.MIN_FLUSH_INTERVAL:
                self._flush_to_db()
                self._last_flush_time = now

            # 所有 worker 完成则退出
            if len(self._workers_done) >= self._num_workers:
                self._flush_to_db()
                break

            self._stop_event.wait(timeout=0.5)

    def _drain_queue(self) -> None:
        """从队列中读取所有待处理消息"""
        while True:
            try:
                msg = self._queue.get_nowait()
            except Exception:
                break

            msg_type = msg.get("type")
            wid = msg.get("worker_id", -1)

            if msg_type == "progress":
                self._worker_progress[wid] = msg
            elif msg_type == "done":
                self._workers_done.add(wid)
            elif msg_type == "error":
                self._workers_error[wid] = msg.get("error", "")
                self._workers_done.add(wid)

    def _flush_to_db(self) -> None:
        """聚合进度并写入数据库"""
        if not self._worker_progress:
            return

        # 聚合：计算所有 worker 的加权平均进度
        total_processed = 0
        total_days = 0
        total_signals = 0
        total_trades = 0
        latest_date = ""
        total_portfolio_value = 0.0

        for wp in self._worker_progress.values():
            total_processed += wp.get("processed_days", 0)
            total_days += wp.get("total_days", 0)
            total_signals += wp.get("total_signals", 0)
            total_trades += wp.get("total_trades", 0)
            total_portfolio_value += wp.get("portfolio_value", 0.0)
            d = wp.get("current_date", "")
            if d > latest_date:
                latest_date = d

        # 整体进度 = 所有 worker 已处理天数 / 所有 worker 总天数
        if total_days > 0:
            execution_pct = total_processed / total_days * 100
        else:
            execution_pct = 0.0

        # 映射到 30%-90% 区间（回测执行阶段）
        overall_progress = 30 + (execution_pct / 100) * 60

        progress_data = {
            "processed_days": total_processed,
            "total_days": total_days,
            "current_date": latest_date,
            "total_signals": total_signals,
            "total_trades": total_trades,
            "portfolio_value": total_portfolio_value,
            "workers_done": len(self._workers_done),
            "workers_total": self._num_workers,
            "multiprocess": True,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        try:
            self._write_progress_to_db(overall_progress, progress_data)
        except Exception as e:
            logger.warning(f"ProgressBridge DB 写入失败: {e}")

    def _write_progress_to_db(
        self, progress: float, progress_data: dict
    ) -> None:
        """直接用 psycopg2 写入进度（绕过 SQLAlchemy 连接池）"""
        import json

        conn = psycopg2.connect(self._db_url)
        try:
            cur = conn.cursor()

            # 先读取现有 result
            cur.execute(
                "SELECT result FROM tasks WHERE task_id = %s",
                (self._task_id,),
            )
            row = cur.fetchone()
            if row is None:
                cur.close()
                return

            result_data = row[0] if row[0] else {}
            if not isinstance(result_data, dict):
                try:
                    result_data = json.loads(result_data)
                except Exception:
                    result_data = {}

            result_data["progress_data"] = progress_data

            # 更新进度和 result
            cur.execute(
                """
                UPDATE tasks
                SET progress = %s,
                    result = %s::jsonb
                WHERE task_id = %s
                  AND status = 'running'
                """,
                (progress, json.dumps(result_data, default=str), self._task_id),
            )
            conn.commit()
            cur.close()

            logger.debug(
                f"ProgressBridge: progress={progress:.1f}%, "
                f"days={progress_data['processed_days']}/{progress_data['total_days']}, "
                f"workers_done={progress_data['workers_done']}/{progress_data['workers_total']}"
            )
        finally:
            conn.close()
