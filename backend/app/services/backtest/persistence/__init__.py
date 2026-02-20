"""
回测数据持久化模块

统一管理回测数据的写入和读取，替代分散在各处的数据库操作。
"""

from .data_contracts import BacktestSummary, SignalData, SnapshotData, TradeData
from .persistence_service import BacktestPersistenceService
from .signal_writer import StreamSignalWriter

__all__ = [
    "BacktestPersistenceService",
    "StreamSignalWriter",
    "BacktestSummary",
    "SnapshotData",
    "TradeData",
    "SignalData",
]
