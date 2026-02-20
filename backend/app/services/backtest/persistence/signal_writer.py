"""
流式信号写入器 — StreamSignalWriter

从 backtest_loop_executor._flush_signals_to_db 提取并封装。
在回测循环中使用，支持 buffer → flush → finalize 生命周期。
使用 psycopg2 批量写入（execute_batch），与占位行同一连接方式。
"""

import json
import time
from typing import List, Optional

from loguru import logger


class StreamSignalWriter:
    """
    流式信号写入器

    特点：
    - 内存缓冲 + 批量写入（默认 3000 条刷一次）
    - 使用 psycopg2 直连（性能最优，适合子进程环境）
    - 自动管理连接生命周期
    - 支持信号执行状态标记
    """

    def __init__(
        self,
        backtest_id: str,
        db_url: str,
        batch_size: int = 3000,
    ):
        """
        初始化流式信号写入器

        Args:
            backtest_id: 回测ID（外键关联 backtest_results）
            db_url: 同步数据库连接 URL（postgresql://...）
            batch_size: 缓冲区大小，达到后自动 flush
        """
        self._backtest_id = backtest_id
        self._db_url = db_url
        self._batch_size = batch_size
        self._buffer: List[dict] = []
        self._total_flushed = 0

    def buffer(self, signal: dict) -> None:
        """缓冲一条信号，满 batch_size 自动 flush"""
        self._buffer.append(signal)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def buffer_many(self, signals: List[dict]) -> None:
        """批量缓冲信号"""
        self._buffer.extend(signals)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> int:
        """批量写入缓冲区中的信号到 signal_records 表，返回写入条数"""
        if not self._buffer:
            return 0

        count = self._flush_to_db(self._buffer)
        self._total_flushed += count
        self._buffer.clear()
        return count

    def finalize(self) -> int:
        """flush 剩余信号，返回总写入条数"""
        if self._buffer:
            self.flush()

        if self._total_flushed > 0:
            logger.info(f"✅ 信号写入完成: 共 {self._total_flushed} 条记录")

        return self._total_flushed

    @property
    def total_written(self) -> int:
        """已写入的总信号数"""
        return self._total_flushed

    @property
    def buffer_size(self) -> int:
        """当前缓冲区大小"""
        return len(self._buffer)

    # ──────────────────────────── 内部方法 ────────────────────────────

    def _flush_to_db(self, batch: List[dict]) -> int:
        """将信号数据批量写入 DB 并返回写入条数。

        从 backtest_loop_executor._flush_signals_to_db 提取的核心逻辑。
        """
        if not batch:
            return 0

        import psycopg2
        import psycopg2.extras

        count = len(batch)

        try:
            # 预处理数据行
            insert_rows = []
            for sd in batch:
                ts = sd["timestamp"]
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

                meta = sd.get("metadata")
                meta_str = None
                if meta is not None:
                    try:
                        meta_str = json.dumps(meta, ensure_ascii=False, default=str)
                    except Exception:
                        pass

                insert_rows.append((
                    self._backtest_id,
                    sd["signal_id"],
                    sd["stock_code"],
                    sd.get("stock_name"),
                    sd["signal_type"],
                    ts_str,
                    float(sd["price"]),
                    float(sd.get("strength", 0.0)),
                    sd.get("reason"),
                    meta_str,
                    True if sd.get("executed") else False,
                    sd.get("execution_reason"),
                ))

            raw_insert_sql = """
                INSERT INTO signal_records
                    (backtest_id, signal_id, stock_code, stock_name,
                     signal_type, timestamp, price, strength, reason,
                     signal_metadata, executed, execution_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            write_batch_size = 5000
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    conn = psycopg2.connect(self._db_url)
                    try:
                        cur = conn.cursor()
                        for bi in range(0, len(insert_rows), write_batch_size):
                            psycopg2.extras.execute_batch(
                                cur,
                                raw_insert_sql,
                                insert_rows[bi: bi + write_batch_size],
                            )
                        conn.commit()
                        cur.close()
                    finally:
                        conn.close()
                    logger.debug(f"信号批量写入: {count} 条")
                    return count
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("deadlock" in err_msg or "could not serialize" in err_msg) and attempt < max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                    else:
                        logger.error(f"信号写入DB失败: {e}")
                        return 0
        except Exception as e:
            logger.error(f"信号写入预处理失败: {e}")
            return 0
