"""
批量信号生成器 - 向量化批量生成所有股票×所有日期的信号

性能优化核心：
- 将 500 stocks × 750 days = 375,000 次函数调用 → 1 次批量计算
- 使用 MultiIndex DataFrame 和 groupby 向量化操作
- 预先构建 (stock_code, date) → signal 的快速查询索引

预期收益：3x 加速
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal


def _multiprocess_precompute_stock_signals(task: Tuple[str, Dict[str, Any], Dict[str, Any]]) -> Tuple[bool, str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[str]]:
    """模块级 worker：为单只股票预计算信号（用于 ProcessPoolExecutor）。

    Returns:
        (ok, stock_code, (date_ns[int64], signal_value[float32], close[float32]) | None, err)
    """
    stock_code, data_pack, strategy_info = task
    try:
        # 重建 DataFrame（尽量避免 dict(list) 的巨大开销）
        values = np.asarray(data_pack["values"], dtype=np.float64)
        columns = list(data_pack["columns"])
        index_ns = np.asarray(data_pack["index_ns"], dtype=np.int64)
        idx = pd.to_datetime(index_ns)
        df = pd.DataFrame(values, columns=columns, index=idx)
        df.attrs["stock_code"] = data_pack.get("stock_code", stock_code)

        # 重建策略对象
        from ..strategies.strategy_factory import AdvancedStrategyFactory, StrategyFactory

        strategy_name = strategy_info.get("name")
        strategy_class_name = strategy_info.get("class_name")
        strategy_config = strategy_info.get("config") or {}

        strategy = None
        names_to_try = [
            strategy_name,
            (strategy_name or "").lower(),
            strategy_class_name,
            (strategy_class_name or "").replace("Strategy", ""),
            (strategy_class_name or "").replace("Strategy", "").lower(),
        ]
        for name in names_to_try:
            if not name:
                continue
            try:
                strategy = StrategyFactory.create_strategy(name, strategy_config)
                break
            except Exception:
                try:
                    strategy = AdvancedStrategyFactory.create_strategy(name, strategy_config)
                    break
                except Exception:
                    pass

        if strategy is None:
            return (False, stock_code, None, f"无法创建策略 {strategy_name} (尝试了: {names_to_try})")

        sigs = strategy.precompute_all_signals(df)
        if sigs is None or len(sigs) == 0:
            return (False, stock_code, None, "precompute_all_signals 返回空")

        # 提取非零信号（None/0 视为无信号）
        close = df.get("close")
        if close is None:
            return (False, stock_code, None, "缺少 close 列")

        # 将 SignalType/数字统一映射为 float32（BUY=1, SELL=-1）
        # 注意：Series dtype 可能是 object
        out_dates = []
        out_vals = []
        out_close = []
        for dt, v in sigs.items():
            if v is None or v == 0 or v == SignalType.HOLD:
                continue
            if isinstance(v, SignalType):
                vv = 1.0 if v == SignalType.BUY else -1.0 if v == SignalType.SELL else 0.0
            else:
                try:
                    vv = float(v)
                except Exception:
                    continue
                if vv == 0:
                    continue
            # price
            try:
                px = float(close.loc[dt])
            except Exception:
                px = 0.0
            out_dates.append(np.int64(pd.Timestamp(dt).value))
            out_vals.append(np.float32(vv))
            out_close.append(np.float32(px))

        if not out_dates:
            return (True, stock_code, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)), None)

        return (
            True,
            stock_code,
            (np.asarray(out_dates, dtype=np.int64), np.asarray(out_vals, dtype=np.float32), np.asarray(out_close, dtype=np.float32)),
            None,
        )

    except Exception as e:
        return (False, stock_code, None, str(e))


class BatchSignalGenerator:
    """批量信号生成器"""

    def __init__(self, strategy: BaseStrategy):
        """
        Args:
            strategy: 策略实例
        """
        self.strategy = strategy
        self._signal_cache: Optional[pd.DataFrame] = None
        self._signal_index: Optional[Dict[Tuple[str, datetime], int]] = None

    def precompute_all_signals(
        self, 
        all_stocks_data: Dict[str, pd.DataFrame],
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        预计算所有股票×所有日期的信号

        Args:
            all_stocks_data: {stock_code: DataFrame} 所有股票的历史数据
            progress_callback: 进度回调函数 callback(current, total, message)

        Returns:
            是否成功预计算
        """
        try:
            start_time = pd.Timestamp.now()
            total_stocks = len(all_stocks_data)
            
            logger.info(f"开始批量预计算信号: {total_stocks} 只股票")

            # 方法 1: 尝试策略自带的批量预计算
            if hasattr(self.strategy, 'precompute_all_signals_batch'):
                logger.info("使用策略自带的批量预计算方法")
                result = self._precompute_with_strategy_batch(all_stocks_data, progress_callback)
                if result:
                    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                    logger.info(f"批量预计算完成，耗时 {elapsed:.2f}s")
                    return True

            # 方法 2: 使用策略的单股票向量化方法
            logger.info("使用逐股票向量化预计算")
            result = self._precompute_per_stock(all_stocks_data, progress_callback)
            
            if result:
                elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                total_signals = len(self._signal_cache) if self._signal_cache is not None else 0
                logger.info(
                    f"批量预计算完成: {total_stocks} 只股票, "
                    f"{total_signals} 个信号, 耗时 {elapsed:.2f}s"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"批量预计算失败: {e}", exc_info=True)
            return False

    def _precompute_with_strategy_batch(
        self,
        all_stocks_data: Dict[str, pd.DataFrame],
        progress_callback: Optional[callable] = None
    ) -> bool:
        """使用策略自带的批量预计算方法"""
        try:
            # 构建 MultiIndex DataFrame
            dfs = []
            for stock_code, df in all_stocks_data.items():
                df_copy = df.copy()
                df_copy['stock_code'] = stock_code
                dfs.append(df_copy)
            
            if not dfs:
                return False

            # 合并所有数据
            combined_df = pd.concat(dfs, ignore_index=False)
            combined_df.set_index(['stock_code', combined_df.index], inplace=True)
            combined_df.index.names = ['stock_code', 'date']

            # 调用策略的批量方法
            signals_df = self.strategy.precompute_all_signals_batch(combined_df)

            if signals_df is None or signals_df.empty:
                return False

            # 构建信号缓存
            self._build_signal_cache_from_dataframe(signals_df)
            return True

        except Exception as e:
            logger.warning(f"策略批量预计算失败: {e}")
            return False

    def _precompute_per_stock(
        self,
        all_stocks_data: Dict[str, pd.DataFrame],
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """逐股票向量化预计算。

        Phase 2: 支持多进程并行（按股票分组）
        - 对于 stock_count 较大时，使用 ProcessPoolExecutor 提升 CPU 并行度
        - macOS 默认 spawn 序列化开销大，优先尝试 fork context（若可用）

        Phase 3: 数据结构优化
        - 进程间只传递 numpy values/index_ns/columns，避免 DataFrame -> dict(list) 的巨大开销
        """
        try:
            total_stocks = len(all_stocks_data)
            if total_stocks == 0:
                return False

            # --- decide parallel mode ---
            enable_mp = total_stocks >= 32  # 经验阈值：小数量用主进程更划算

            # 预收集任务包（numpy 化，减少 pickle 体积）
            tasks: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
            for stock_code, df in all_stocks_data.items():
                try:
                    # 仅传递数值列，避免 stock_code/行业等 object 列导致 numpy->float 转换失败
                    if len(df.columns) == 0:
                        continue
                    non_numeric_cols = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
                    if non_numeric_cols:
                        df_use = df.drop(columns=non_numeric_cols, errors="ignore")
                    else:
                        df_use = df

                    # 确保包含 close（策略通常依赖 close）
                    if "close" not in df_use.columns and "close" in df.columns:
                        # 尝试转为数值
                        try:
                            df_use = df_use.assign(close=pd.to_numeric(df["close"], errors="coerce"))
                        except Exception:
                            pass

                    columns = list(df_use.columns)
                    values = df_use.to_numpy(copy=False)
                    # index -> ns int64（DatetimeIndex 的 .asi8）
                    index_ns = getattr(df.index, "asi8", None)
                    if index_ns is None:
                        index_ns = pd.to_datetime(df.index).asi8

                    tasks.append(
                        (
                            stock_code,
                            {
                                "values": values,
                                "columns": columns,
                                "index_ns": np.asarray(index_ns, dtype=np.int64),
                                "stock_code": df.attrs.get("stock_code", stock_code),
                            },
                            {
                                "name": getattr(self.strategy, "name", None),
                                "class_name": self.strategy.__class__.__name__,
                                "config": getattr(self.strategy, "config", {}) or {},
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(f"准备股票 {stock_code} 数据失败: {e}")

            if not tasks:
                return False

            signal_records: List[Dict[str, Any]] = []

            if enable_mp:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                import multiprocessing as mp

                # 选择 start method（优先 fork，避免 spawn 的序列化/重导入开销）
                mp_ctx = None
                try:
                    mp_ctx = mp.get_context("fork")
                except Exception:
                    mp_ctx = None

                max_workers = min((mp.cpu_count() or 4), 8)

                logger.info(f"🚀 批量预计算启用多进程: stocks={total_stocks}, workers={max_workers}")

                with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
                    futures = {
                        ex.submit(_multiprocess_precompute_stock_signals, t): t[0]
                        for t in tasks
                    }
                    done = 0
                    for fu in as_completed(futures):
                        done += 1
                        code = futures[fu]
                        if progress_callback:
                            progress_callback(done, total_stocks, f"预计算 {code}")
                        try:
                            ok, stock_code, packed, err = fu.result(timeout=120)
                            if not ok:
                                if err:
                                    logger.warning(f"预计算失败 {stock_code}: {err}")
                                continue
                            if packed is None:
                                continue
                            date_ns, vals, closes = packed
                            # packed 可能为空（无信号）
                            for d_ns, v, px in zip(date_ns, vals, closes):
                                signal_records.append(
                                    {
                                        "stock_code": stock_code,
                                        "date": pd.to_datetime(int(d_ns)),
                                        "signal_type": SignalType.BUY if float(v) > 0 else SignalType.SELL,
                                        "strength": float(abs(v)),
                                        "price": float(px),
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"预计算失败 {code}: {e}")
            else:
                # 单进程路径
                for idx, (stock_code, df) in enumerate(all_stocks_data.items()):
                    if progress_callback:
                        progress_callback(idx + 1, total_stocks, f"预计算 {stock_code}")

                    signals_series = self.strategy.precompute_all_signals(df)
                    if signals_series is None or signals_series.empty:
                        continue
                    close = df.get("close")
                    for date, signal_value in signals_series.items():
                        if signal_value is None or signal_value == 0 or signal_value == SignalType.HOLD:
                            continue
                        signal_records.append(
                            {
                                "stock_code": stock_code,
                                "date": date,
                                "signal_type": self._convert_signal_value(signal_value),
                                "strength": abs(float(signal_value))
                                if isinstance(signal_value, (int, float))
                                else 1.0,
                                "price": float(close.loc[date]) if close is not None and date in df.index else 0.0,
                            }
                        )

            if not signal_records:
                logger.warning("未生成任何信号")
                return False

            self._signal_cache = pd.DataFrame(signal_records)
            self._signal_cache.set_index(["stock_code", "date"], inplace=True)
            self._build_signal_index()
            return True

        except Exception as e:
            logger.error(f"逐股票预计算失败: {e}", exc_info=True)
            return False

    def _build_signal_cache_from_dataframe(self, signals_df: pd.DataFrame):
        """从 DataFrame 构建信号缓存"""
        # 假设 signals_df 已经是 MultiIndex (stock_code, date)
        # 包含列: signal_type, strength, price
        self._signal_cache = signals_df
        self._build_signal_index()

    def _build_signal_index(self):
        """构建快速查询索引"""
        if self._signal_cache is None or self._signal_cache.empty:
            self._signal_index = {}
            return

        # 构建 (stock_code, date) → row_index 的映射
        self._signal_index = {
            (stock_code, date): idx
            for idx, (stock_code, date) in enumerate(self._signal_cache.index)
        }

        logger.info(f"构建信号索引: {len(self._signal_index)} 个信号")

    def _convert_signal_value(self, value) -> SignalType:
        """转换信号值为 SignalType"""
        if isinstance(value, SignalType):
            return value
        
        if isinstance(value, (int, float)):
            if value > 0:
                return SignalType.BUY
            elif value < 0:
                return SignalType.SELL
            else:
                return SignalType.HOLD
        
        if isinstance(value, str):
            value_upper = value.upper()
            if value_upper in ('BUY', 'LONG'):
                return SignalType.BUY
            elif value_upper in ('SELL', 'SHORT'):
                return SignalType.SELL
        
        return SignalType.HOLD

    def get_signals(
        self, 
        stock_code: str, 
        current_date: datetime
    ) -> List[TradingSignal]:
        """
        快速查询指定股票和日期的信号

        Args:
            stock_code: 股票代码
            current_date: 当前日期

        Returns:
            信号列表
        """
        if self._signal_cache is None or self._signal_index is None:
            return []

        # 快速查询
        key = (stock_code, current_date)
        if key not in self._signal_index:
            return []

        try:
            # 从缓存中获取信号
            signal_row = self._signal_cache.loc[key]

            # 处理多个信号的情况（如果同一天有多个信号）
            if isinstance(signal_row, pd.DataFrame):
                signals = []
                for _, row in signal_row.iterrows():
                    signals.append(self._create_trading_signal(stock_code, current_date, row))
                return signals
            else:
                # 单个信号
                return [self._create_trading_signal(stock_code, current_date, signal_row)]

        except Exception as e:
            logger.warning(f"获取信号失败 {stock_code} @ {current_date}: {e}")
            return []

    def _create_trading_signal(
        self,
        stock_code: str,
        date: datetime,
        signal_row: pd.Series
    ) -> TradingSignal:
        """从信号行创建 TradingSignal 对象"""
        return TradingSignal(
            stock_code=stock_code,
            signal_type=signal_row['signal_type'],
            strength=float(signal_row.get('strength', 1.0)),
            price=float(signal_row.get('price', 0.0)),
            timestamp=date,
            reason=f"{self.strategy.name} 批量预计算"
        )

    def has_precomputed_signals(self) -> bool:
        """是否已预计算信号"""
        return self._signal_cache is not None and not self._signal_cache.empty

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.has_precomputed_signals():
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'stocks_count': 0
            }

        stats = {
            'total_signals': len(self._signal_cache),
            'buy_signals': (self._signal_cache['signal_type'] == SignalType.BUY).sum(),
            'sell_signals': (self._signal_cache['signal_type'] == SignalType.SELL).sum(),
            'stocks_count': self._signal_cache.index.get_level_values('stock_code').nunique()
        }

        return stats

    def clear_cache(self):
        """清除缓存"""
        self._signal_cache = None
        self._signal_index = None
        logger.info("信号缓存已清除")
