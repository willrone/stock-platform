"""
数据预处理模块
负责回测数据的预处理、索引构建、信号预计算等
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from ..core.base_strategy import BaseStrategy

# 多进程预计算 worker 函数
def _multiprocess_precompute_worker(task: Tuple) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
    """
    多进程预计算 worker 函数（模块级，可被 pickle 序列化）。

    Args:
        task: (stock_code, data_dict, strategy_info) 元组

    Returns:
        (success, stock_code, signals_dict, error_message)
    """
    stock_code, data_dict, strategy_info = task

    try:
        # 重建 DataFrame
        df = pd.DataFrame(data_dict['values'], columns=data_dict['columns'])
        df.index = pd.to_datetime(data_dict['index'])
        df.attrs['stock_code'] = data_dict['stock_code']

        # 重建策略对象
        from ..strategies.strategy_factory import StrategyFactory, AdvancedStrategyFactory

        strategy_name = strategy_info['name']  # 使用策略名称（如 "MACD"）
        strategy_class_name = strategy_info['class_name']  # 类名（如 "MACDStrategy"）
        strategy_config = strategy_info['config']

        # 尝试从工厂创建策略（尝试多种名称格式）
        strategy = None
        names_to_try = [
            strategy_name,  # 原始名称
            strategy_name.lower(),  # 小写
            strategy_class_name,  # 类名
            strategy_class_name.replace('Strategy', ''),  # 去掉 Strategy 后缀
            strategy_class_name.replace('Strategy', '').lower(),  # 去掉后缀并小写
        ]

        for name in names_to_try:
            if strategy is not None:
                break
            try:
                strategy = StrategyFactory.create_strategy(name, strategy_config)
            except Exception:
                try:
                    strategy = AdvancedStrategyFactory.create_strategy(name, strategy_config)
                except Exception:
                    pass

        if strategy is None:
            return (False, stock_code, None, f"无法创建策略 {strategy_name} (尝试了: {names_to_try})")

        # 执行向量化预计算
        signals = strategy.precompute_all_signals(df)

        if signals is not None:
            # 将 Series 转换为可序列化格式
            signals_dict = {
                'values': signals.tolist(),
                'index': [str(idx) for idx in signals.index],
            }
            return (True, stock_code, signals_dict, None)
        else:
            return (False, stock_code, None, "precompute_all_signals 返回 None")

    except Exception as e:
        return (False, stock_code, None, str(e))



class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, enable_parallel: bool = True, max_workers: int = 8, use_multiprocessing: bool = True):
        """
        初始化数据预处理器

        Args:
            enable_parallel: 是否启用并行化
            max_workers: 最大工作线程/进程数
            use_multiprocessing: 是否使用多进程
        """
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing

    def get_trading_calendar(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> List[datetime]:
        """获取交易日历"""
        # 合并所有股票的交易日期
        all_dates = set()
        for data in stock_data.values():
            all_dates.update(data.index.tolist())

        # 过滤日期范围并排序
        trading_dates = np.sort(np.array([date for date in all_dates if start_date <= date <= end_date])).tolist()

        return trading_dates


    def build_date_index(self, stock_data: Dict[str, pd.DataFrame]) -> None:
        """为每只股票建立日期->整数索引，避免回测循环中重复 get_loc。"""
        for data in stock_data.values():
            try:
                if "_date_to_idx" not in data.attrs:
                    data.attrs["_date_to_idx"] = {
                        d: i for i, d in enumerate(data.index)
                    }
            except Exception:
                pass


    def warm_indicator_cache(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> None:
        """回测开始前预计算并缓存所有股票的指标，避免首日/首股现场计算。"""
        try:
            from ..core.strategy_portfolio import StrategyPortfolio

            if isinstance(strategy, StrategyPortfolio):
                for sub in strategy.strategies:
                    self._warm_indicator_cache(sub, stock_data)
                return
        except Exception:
            pass
        for data in stock_data.values():
            try:
                strategy.get_cached_indicators(data)
            except Exception:
                pass


    def precompute_strategy_signals(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> None:
        """[性能优化] 在回测循环开始前，尝试对所有股票进行向量化信号预计算。"""
        try:
            from ..core.strategy_portfolio import StrategyPortfolio

            if isinstance(strategy, StrategyPortfolio):
                logger.info(f"🚀 Portfolio策略检测到，递归预计算 {len(strategy.strategies)} 个子策略")
                for sub in strategy.strategies:
                    self._precompute_strategy_signals(sub, stock_data)
                return
        except Exception as e:
            logger.warning(f"Portfolio策略递归预计算失败: {e}")

        # 统计预计算成功的股票数
        success_count = 0
        total_stocks = len(stock_data)

        # 并行预计算（按股票维度），显著降低整体 wall-time
        # 注：使用 ProcessPoolExecutor 可突破 GIL 限制，但需要序列化数据
        # 这里使用混合策略：CPU 密集型任务用多进程，I/O 密集型用多线程
        use_multiprocessing = getattr(self, 'use_multiprocessing', False)

        def _work_one(item):
            stock_code, data = item
            try:
                all_sigs = strategy.precompute_all_signals(data)
                if all_sigs is not None:
                    cache = data.attrs.setdefault("_precomputed_signals", {})
                    # 使用 strategy.name 作为稳定的 key，避免多进程环境下 id() 变化
                    cache[strategy.name] = all_sigs
                    return True, stock_code, None
                return False, stock_code, None
            except Exception as e:
                return False, stock_code, str(e)

        if self.enable_parallel and total_stocks >= 4:
            if use_multiprocessing:
                # 多进程模式：突破 GIL 限制，适合 CPU 密集型策略计算
                # 注意：需要将数据序列化传递，开销较大但可真正并行
                try:
                    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
                    # 多进程需要使用模块级函数，这里使用包装器
                    results = self._precompute_signals_multiprocess(
                        strategy, stock_data
                    )
                    for ok, stock_code, err in results:
                        if ok:
                            success_count += 1
                        elif err:
                            logger.warning(
                                f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                            )
                except Exception as e:
                    logger.warning(f"多进程预计算失败，回退到多线程: {e}")
                    use_multiprocessing = False

            if not use_multiprocessing:
                # 多线程模式：受 GIL 限制，但序列化开销小
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = [ex.submit(_work_one, it) for it in stock_data.items()]
                    for fu in as_completed(futures):
                        ok, stock_code, err = fu.result()
                        if ok:
                            success_count += 1
                        elif err:
                            logger.warning(
                                f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                            )
        else:
            for it in stock_data.items():
                ok, stock_code, err = _work_one(it)
                if ok:
                    success_count += 1
                elif err:
                    logger.warning(
                        f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                    )

        if success_count > 0:
            logger.info(
                f"✅ 策略 {strategy.name} 向量化预计算完成: {success_count}/{total_stocks} 只股票"
            )


    def extract_precomputed_signals_to_dict(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> Dict[Tuple[str, datetime], Any]:
        """
        [性能优化] 将预计算的信号从 DataFrame.attrs 提取到扁平字典。
        
        这样在回测循环中可以直接用 (stock_code, date) 查找信号，
        避免每次都访问 attrs 字典和 id(strategy) 查找。
        
        Returns:
            Dict[(stock_code, date), signal]: 扁平的信号字典
        """
        signal_dict = {}
        
        try:
            from ..core.strategy_portfolio import StrategyPortfolio
            from ..models import TradingSignal
            
            if isinstance(strategy, StrategyPortfolio):
                logger.info(f"🔄 Portfolio策略信号整合开始: {len(strategy.strategies)} 个子策略")
                
                # 1. 递归提取所有子策略的信号
                all_sub_signals: Dict[Tuple[str, datetime], Any] = {}
                for sub in strategy.strategies:
                    sub_signals = self._extract_precomputed_signals_to_dict(sub, stock_data)
                    all_sub_signals.update(sub_signals)
                
                logger.info(f"📊 子策略信号总数: {len(all_sub_signals)}")
                
                # 2. 按日期分组子策略信号
                from collections import defaultdict
                signals_by_date: Dict[datetime, List[TradingSignal]] = defaultdict(list)
                
                for (stock_code, date), signal_type in all_sub_signals.items():
                    # 构造 TradingSignal 对象
                    from ..models import SignalType
                    if signal_type == SignalType.BUY or signal_type == SignalType.SELL:
                        # 获取价格
                        try:
                            df = stock_data.get(stock_code)
                            if df is not None and date in df.index:
                                price = float(df.loc[date, 'close'])
                                signal = TradingSignal(
                                    timestamp=date,
                                    stock_code=stock_code,
                                    signal_type=signal_type,
                                    strength=1.0,
                                    price=price,
                                    reason="precomputed",
                                    metadata={}
                                )
                                signals_by_date[date].append(signal)
                        except Exception as e:
                            logger.warning(f"构造信号失败 {stock_code} @ {date}: {e}")
                
                # 3. 对每个日期的信号进行整合
                integrated_count = 0
                for date, signals in signals_by_date.items():
                    if signals:
                        # 调用 Portfolio 的信号整合器
                        integrated = strategy.integrator.integrate(
                            signals, 
                            strategy.weights,
                            consistency_threshold=0.6
                        )
                        
                        # 将整合后的信号添加到字典
                        for sig in integrated:
                            signal_dict[(sig.stock_code, sig.timestamp)] = sig.signal_type
                            integrated_count += 1
                
                logger.info(f"✅ Portfolio策略信号整合完成: {integrated_count} 个整合信号")
                return signal_dict
                
        except Exception as e:
            logger.warning(f"Portfolio策略信号提取失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        # 提取单个策略的信号
        # 使用 strategy.name 作为稳定的 key，避免多进程环境下 id() 变化
        strategy_key = strategy.name
        extracted_count = 0
        
        for stock_code, data in stock_data.items():
            try:
                precomputed = data.attrs.get("_precomputed_signals", {})
                signals = precomputed.get(strategy_key)
                
                if signals is not None:
                    # signals 可能是 pd.Series 或 dict
                    if isinstance(signals, pd.Series):
                        for date, signal in signals.items():
                            if signal is not None:
                                signal_dict[(stock_code, date)] = signal
                                extracted_count += 1
                    elif isinstance(signals, dict):
                        for date, signal in signals.items():
                            if signal is not None:
                                signal_dict[(stock_code, date)] = signal
                                extracted_count += 1
            except Exception as e:
                logger.warning(f"提取股票 {stock_code} 的信号失败: {e}")
        
        if extracted_count > 0:
            logger.info(
                f"✅ 策略 {strategy.name} 信号提取完成: {extracted_count} 个信号"
            )
        
        return signal_dict


    def build_aligned_arrays(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        trading_dates: List[datetime],
    ) -> Dict[str, Any]:
        """[Phase3] 将数据/信号对齐到 ndarray，减少主循环 DataFrame/字典访问。
        
        优化点：
        1. 使用 numpy 的 searchsorted 加速日期查找
        2. 批量填充数组，减少循环
        3. 使用 .values 避免 pandas 开销

        Returns:
            {
              'stock_codes': [...],
              'dates': np.ndarray[datetime64],
              'close': float64[N,T] (nan=missing),
              'open':  float64[N,T] (nan=missing),
              'valid': bool[N,T],
              'signal': int8[N,T] (1=BUY, -1=SELL, 0=NONE)
            }
        """
        stock_codes = list(stock_data.keys())
        T = len(trading_dates)
        N = len(stock_codes)

        dates64 = np.array(trading_dates, dtype='datetime64[ns]')

        # 预分配数组（Phase 3 优化：使用连续内存）
        close = np.full((N, T), np.nan, dtype=np.float64, order='C')
        open_ = np.full((N, T), np.nan, dtype=np.float64, order='C')
        valid = np.zeros((N, T), dtype=bool, order='C')
        signal = np.zeros((N, T), dtype=np.int8, order='C')

        # 如果已做向量化预计算��号，尽量直接读取 per-stock Series 并对齐到 trading_dates
        strategy_key = strategy.name  # 使用 strategy.name 作为稳定的 key

        for i, code in enumerate(stock_codes):
            df = stock_data[code]

            # Phase 3 优化：使用 numpy searchsorted 替代 pandas reindex（更快）
            try:
                # 价格对齐（使用 searchsorted 进行索引���射）
                df_dates = df.index.values
                # 使用 searchsorted 找到每个 trading_date 在 df_dates 中的位置
                indices = np.searchsorted(df_dates, trading_dates)
                # 处理越界情况
                indices = np.clip(indices, 0, len(df_dates) - 1)
                # 检查是否精确匹配
                matches = df_dates[indices] == trading_dates
                
                # 填充价格数据
                close_values = df['close'].values[indices]
                close_values[~matches] = np.nan
                close[i, :] = close_values
                
                if 'open' in df.columns:
                    open_values = df['open'].values[indices]
                    open_values[~matches] = np.nan
                    open_[i, :] = open_values
                
                # 使用向量化操作判断有效性
                valid[i, :] = matches & ~np.isnan(close_values)
                
            except Exception as e:
                # fallback: per-date fill (slow path, should be rare)
                logger.warning(f"股票 {code} 数组对齐失败，使用慢速路径: {e}")
                idx_map = df.attrs.get('_date_to_idx') if hasattr(df, 'attrs') else None
                for t, d in enumerate(trading_dates):
                    try:
                        if idx_map and d in idx_map:
                            k = int(idx_map[d])
                            close[i, t] = float(df['close'].iloc[k])
                            if 'open' in df.columns:
                                open_[i, t] = float(df['open'].iloc[k])
                            valid[i, t] = True
                        elif d in df.index:
                            k = df.index.get_loc(d)
                            close[i, t] = float(df['close'].values[k])
                            if 'open' in df.columns:
                                open_[i, t] = float(df['open'].values[k])
                            valid[i, t] = True
                    except Exception:
                        pass

            # 信号对齐（Phase 3 优化：使用 numpy searchsorted 替代 pandas reindex）
            try:
                pre = df.attrs.get('_precomputed_signals', {}) if hasattr(df, 'attrs') else {}
                sig_ser = pre.get(strategy_key)
                if isinstance(sig_ser, pd.Series):
                    # 使用 searchsorted 批量对齐
                    sig_dates = sig_ser.index.values
                    sig_indices = np.searchsorted(sig_dates, trading_dates)
                    sig_indices = np.clip(sig_indices, 0, len(sig_dates) - 1)
                    sig_matches = sig_dates[sig_indices] == trading_dates
                    
                    # 获取信号值
                    vals = sig_ser.values[sig_indices]
                    vals[~sig_matches] = None  # 不匹配的设为 None
                    
                    # 向量化映射 SignalType to int8
                    buy_mask = vals == SignalType.BUY
                    sell_mask = vals == SignalType.SELL
                    signal[i, buy_mask] = 1
                    signal[i, sell_mask] = -1
                elif isinstance(sig_ser, dict):
                    # dict 路径：转换为数组后使用向量化操作
                    sig_series = pd.Series(sig_ser)
                    sig_dates = sig_series.index.values
                    sig_indices = np.searchsorted(sig_dates, trading_dates)
                    sig_indices = np.clip(sig_indices, 0, len(sig_dates) - 1)
                    sig_matches = sig_dates[sig_indices] == trading_dates
                    
                    vals = sig_series.values[sig_indices]
                    vals[~sig_matches] = None
                    
                    buy_mask = vals == SignalType.BUY
                    sell_mask = vals == SignalType.SELL
                    signal[i, buy_mask] = 1
                    signal[i, sell_mask] = -1
            except Exception as e:
                logger.warning(f"股票 {code} 信号对齐失败: {e}")

        return {
            'stock_codes': stock_codes,
            'code_to_i': {c: idx for idx, c in enumerate(stock_codes)},
            'dates': dates64,
            'date_to_i': {d: idx for idx, d in enumerate(trading_dates)},  # [P1 优化] 日期到索引的O(1)映射
            'close': close,
            'open': open_,
            'valid': valid,
            'signal': signal,
        }



    def precompute_signals_multiprocess(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> List[Tuple[bool, str, Optional[str]]]:
        """
        [性能优化] 使用多线程进行信号预计算，避免序列化开销。

        优化 #4：改用 ThreadPoolExecutor 替代 ProcessPoolExecutor
        - 避免 DataFrame 和策略对象的序列化/反序列化开销
        - 信号预计算主要是 numpy/pandas 操作，会释放 GIL
        - 预期提升 8-12 秒
        """
        from concurrent.futures import ThreadPoolExecutor
        import pickle

        results = []

        # 优化 #4：使用多线程，直接传递 DataFrame 和策略对象，避免序列化
        def compute_signals(stock_code: str, data: pd.DataFrame) -> Tuple[bool, str, Optional[str]]:
            """线程 worker 函数"""
            try:
                signals = strategy.precompute_all_signals(data)
                if signals is not None:
                    cache = data.attrs.setdefault("_precomputed_signals", {})
                    cache[strategy.name] = signals
                    return (True, stock_code, None)
                else:
                    return (False, stock_code, "precompute_all_signals 返回 None")
            except Exception as e:
                return (False, stock_code, str(e))

        # 使用线程池并行计算
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(compute_signals, stock_code, data): stock_code
                    for stock_code, data in stock_data.items()
                }

                for future in as_completed(futures):
                    stock_code = futures[future]
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as e:
                        results.append((False, stock_code, str(e)))
        except Exception as e:
            logger.error(f"多线程预计算执行失败: {e}")
            # 返回所有任务失败
            for stock_code in stock_data.keys():
                if not any(r[1] == stock_code for r in results):
                    results.append((False, stock_code, str(e)))

        return results


