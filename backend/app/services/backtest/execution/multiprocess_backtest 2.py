"""
多进程并行回测执行器 - Phase 4 优化

核心思路：
1. 将 500 只股票分配到多个进程（8核 CPU）
2. 每个进程独立执行回测（突破 GIL 限制）
3. 最后合并结果

预期加速：5-6x
"""

import multiprocessing as mp
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager_array import PortfolioManagerArray
from ..models import BacktestConfig, SignalType, TradingSignal


def _worker_backtest(
    worker_id: int,
    stock_codes: List[str],
    stock_data_serialized: Dict[str, Dict],
    trading_dates_list: List[str],
    strategy_info: Dict[str, Any],
    backtest_config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Worker 进程执行回测
    
    Args:
        worker_id: Worker ID
        stock_codes: 分配给该 worker 的股票列表
        stock_data_serialized: 序列化的股票数据
        trading_dates_list: 交易日期列表（字符串格式）
        strategy_info: 策略配置信息
        backtest_config_dict: 回测配置字典
        
    Returns:
        回测结果字典
    """
    try:
        # 1. 重建数据结构
        stock_data = {}
        for code in stock_codes:
            data_dict = stock_data_serialized[code]
            df = pd.DataFrame(data_dict['values'], columns=data_dict['columns'])
            df.index = pd.to_datetime(data_dict['index'])
            df.attrs['stock_code'] = code
            stock_data[code] = df
        
        trading_dates = [pd.to_datetime(d) for d in trading_dates_list]
        
        # 2. 重建策略
        from ..strategies.strategy_factory import (
            AdvancedStrategyFactory,
            StrategyFactory,
        )
        
        strategy_name = strategy_info['name']
        strategy_config = strategy_info['config']
        
        strategy = None
        try:
            strategy = AdvancedStrategyFactory.create_strategy(strategy_name, strategy_config)
        except Exception:
            strategy = StrategyFactory.create_strategy(strategy_name, strategy_config)
        
        # 3. 重建回测配置
        config = BacktestConfig(**backtest_config_dict)
        
        # 4. 创建组合管理器
        portfolio_manager = PortfolioManagerArray(config, stock_codes)
        
        # 5. 预计算信号（向量化）
        logger.info(f"Worker {worker_id}: 开始预计算信号，股票数: {len(stock_codes)}")
        precompute_start = time.perf_counter()
        
        for code, data in stock_data.items():
            try:
                signals = strategy.precompute_all_signals(data)
                if signals is not None:
                    cache = data.attrs.setdefault("_precomputed_signals", {})
                    cache[id(strategy)] = signals
            except Exception as e:
                logger.warning(f"Worker {worker_id}: 预计算信号失败 {code}: {e}")
        
        precompute_time = time.perf_counter() - precompute_start
        logger.info(f"Worker {worker_id}: 预计算完成，耗时 {precompute_time:.2f}秒")
        
        # 6. 执行回测主循环
        logger.info(f"Worker {worker_id}: 开始回测主循环，交易日: {len(trading_dates)}")
        loop_start = time.perf_counter()
        
        total_signals = 0
        executed_trades = 0
        
        for i, current_date in enumerate(trading_dates):
            # 获取当前价格
            current_prices = {}
            for code, data in stock_data.items():
                if current_date in data.index:
                    try:
                        idx = data.index.get_loc(current_date)
                        current_prices[code] = float(data['close'].iloc[idx])
                    except Exception:
                        pass
            
            if not current_prices:
                continue
            
            # 生成交易信号（优先使用预计算）
            all_signals = []
            strategy_id = id(strategy)
            
            for code, data in stock_data.items():
                if current_date not in data.index:
                    continue
                
                try:
                    idx = data.index.get_loc(current_date)
                    if idx < 20:  # 跳过预热期
                        continue
                    
                    # 从预计算缓存读取信号
                    precomputed = data.attrs.get("_precomputed_signals", {})
                    sig_series = precomputed.get(strategy_id)
                    
                    if sig_series is not None and current_date in sig_series.index:
                        sig_type = sig_series.loc[current_date]
                        if isinstance(sig_type, SignalType):
                            price = current_prices.get(code, 0.0)
                            signal = TradingSignal(
                                timestamp=current_date,
                                stock_code=code,
                                signal_type=sig_type,
                                strength=0.8,
                                price=price,
                                reason=f"Precomputed signal",
                                metadata={}
                            )
                            all_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Worker {worker_id}: 生成信号失败 {code}: {e}")
            
            total_signals += len(all_signals)
            
            # 执行交易
            for signal in all_signals:
                # 验证信号
                is_valid, _ = strategy.validate_signal(
                    signal,
                    portfolio_manager.get_portfolio_value(current_prices),
                    portfolio_manager.positions,
                )
                
                if is_valid:
                    trade, _ = portfolio_manager.execute_signal(signal, current_prices)
                    if trade:
                        executed_trades += 1
            
            # 记录组合快照
            portfolio_manager.record_portfolio_snapshot(current_date, current_prices)
            
            # 定期输出进度
            if i % 50 == 0 and i > 0:
                progress = (i + 1) / len(trading_dates) * 100
                logger.info(f"Worker {worker_id}: 进度 {progress:.1f}%")
        
        loop_time = time.perf_counter() - loop_start
        logger.info(f"Worker {worker_id}: 回测完成，耗时 {loop_time:.2f}秒")
        
        # 7. 计算绩效指标
        performance_metrics = portfolio_manager.get_performance_metrics()
        
        # 8. 返回结果
        result = {
            'worker_id': worker_id,
            'stock_codes': stock_codes,
            'total_signals': total_signals,
            'executed_trades': executed_trades,
            'trading_days': len(trading_dates),
            'performance_metrics': performance_metrics,
            'equity_curve': portfolio_manager.equity_curve,
            'trades': portfolio_manager.trades,
            'final_cash': portfolio_manager.cash,
            'final_positions': {
                code: {
                    'quantity': int(portfolio_manager.quantities[i]),
                    'avg_cost': float(portfolio_manager.avg_costs[i]),
                }
                for i, code in enumerate(stock_codes)
                if portfolio_manager.quantities[i] > 0
            },
            'timing': {
                'precompute_time': precompute_time,
                'loop_time': loop_time,
                'total_time': precompute_time + loop_time,
            }
        }
        
        logger.info(f"Worker {worker_id}: 返回结果，信号数: {total_signals}, 交易数: {executed_trades}")
        return result
        
    except Exception as e:
        logger.error(f"Worker {worker_id} 执行失败: {e}", exc_info=True)
        return {
            'worker_id': worker_id,
            'error': str(e),
            'stock_codes': stock_codes,
        }


def run_multiprocess_backtest(
    strategy_name: str,
    stock_codes: List[str],
    start_date: datetime,
    end_date: datetime,
    strategy_config: Dict[str, Any],
    backtest_config: Optional[BacktestConfig] = None,
    num_workers: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    多进程并行回测
    
    Args:
        strategy_name: 策略名称
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        strategy_config: 策略配置
        backtest_config: 回测配置
        num_workers: 工作进程数（默认使用 CPU 核心数）
        data_dir: 数据目录（绝对路径，可选）
        
    Returns:
        合并后的回测结果
    """
    total_start = time.perf_counter()
    
    # 1. 确定���作进程数
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 最多 8 个进程
    
    logger.info(f"🚀 启动多进程回测: {num_workers} 个进程, {len(stock_codes)} 只股票")
    
    # 2. 使用默认配置
    if backtest_config is None:
        backtest_config = BacktestConfig()
    
    # 3. 加载数据
    logger.info("📊 加载股票数据...")
    data_load_start = time.perf_counter()
    
    from .data_loader import DataLoader

    # 使用传入的数据目录或默认绝对路径
    if data_dir is None:
        data_dir = "/Users/ronghui/Documents/GitHub/willrone/data"
    
    logger.info(f"数据目录: {data_dir}")
    data_loader = DataLoader(data_dir=data_dir, max_workers=num_workers)
    stock_data = data_loader.load_multiple_stocks(stock_codes, start_date, end_date)
    
    data_load_time = time.perf_counter() - data_load_start
    logger.info(f"✅ 数据加载完成: {len(stock_data)} 只股票, 耗时 {data_load_time:.2f}秒")
    
    # 4. 获取交易日历
    all_dates = set()
    for data in stock_data.values():
        all_dates.update(data.index.tolist())
    trading_dates = sorted([date for date in all_dates if start_date <= date <= end_date])
    
    logger.info(f"📅 交易日历: {len(trading_dates)} 天")
    
    # 5. 序列化数据（准备传递给子进程）
    logger.info("🔄 序列化数据...")
    serialize_start = time.perf_counter()
    
    stock_data_serialized = {}
    for code, data in stock_data.items():
        stock_data_serialized[code] = {
            'values': data.values.tolist(),
            'columns': data.columns.tolist(),
            'index': [str(d) for d in data.index],
            'stock_code': code,
        }
    
    trading_dates_list = [str(d) for d in trading_dates]
    
    serialize_time = time.perf_counter() - serialize_start
    logger.info(f"✅ 序列化完成, 耗时 {serialize_time:.2f}秒")
    
    # 6. 分配股票到各个进程（负载均衡）
    actual_stock_codes = list(stock_data.keys())
    stocks_per_worker = len(actual_stock_codes) // num_workers
    stock_assignments = []
    
    for i in range(num_workers):
        start_idx = i * stocks_per_worker
        if i == num_workers - 1:
            # 最后一个进程处理剩余的所有股票
            end_idx = len(actual_stock_codes)
        else:
            end_idx = (i + 1) * stocks_per_worker
        
        assigned_stocks = actual_stock_codes[start_idx:end_idx]
        stock_assignments.append(assigned_stocks)
        logger.info(f"Worker {i}: {len(assigned_stocks)} 只股票")
    
    # 7. 准备策略和配置信息
    strategy_info = {
        'name': strategy_name,
        'config': strategy_config,
    }
    
    backtest_config_dict = {
        'initial_cash': backtest_config.initial_cash,
        'commission_rate': backtest_config.commission_rate,
        'slippage_rate': backtest_config.slippage_rate,
        'max_position_size': backtest_config.max_position_size,
    }
    
    # 8. 启动多进程执行
    logger.info("🚀 启动多进程回测...")
    mp_start = time.perf_counter()
    
    # 使用 spawn 方法（更安全，避免 fork 问题）
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=num_workers) as pool:
        # 准备任务参数
        tasks = [
            (
                i,
                stock_assignments[i],
                stock_data_serialized,
                trading_dates_list,
                strategy_info,
                backtest_config_dict,
            )
            for i in range(num_workers)
        ]
        
        # 并行执行
        results = pool.starmap(_worker_backtest, tasks)
    
    mp_time = time.perf_counter() - mp_start
    logger.info(f"✅ 多进程执行完成, 耗时 {mp_time:.2f}秒")
    
    # 9. 合并结果
    logger.info("🔄 合并结果...")
    merge_start = time.perf_counter()
    
    # 检查是否有错误
    errors = [r for r in results if 'error' in r]
    if errors:
        logger.error(f"❌ {len(errors)} 个进程执行失败:")
        for err in errors:
            logger.error(f"  Worker {err['worker_id']}: {err['error']}")
    
    # 合并成功的结果
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        raise RuntimeError("所有进程都执行失败")
    
    # 合并统计数据
    total_signals = sum(r['total_signals'] for r in successful_results)
    total_trades = sum(r['executed_trades'] for r in successful_results)
    
    # 合并权益曲线（平均各进程的收益率）
    all_equity_curves = [r['equity_curve'] for r in successful_results]
    merged_equity_curve = []
    
    if all_equity_curves:
        # 按日期对齐并计算平均值
        date_to_values = {}
        for curve in all_equity_curves:
            for date, value in curve:
                if date not in date_to_values:
                    date_to_values[date] = []
                date_to_values[date].append(value)
        
        # 计算每日平均价值（而不是求和）
        for date in sorted(date_to_values.keys()):
            avg_value = sum(date_to_values[date]) / len(date_to_values[date])
            merged_equity_curve.append((date, avg_value))
    
    # 合并交易记录
    all_trades = []
    for r in successful_results:
        all_trades.extend(r['trades'])
    
    # 计算合并后的绩效指标
    if merged_equity_curve:
        values = [v for _, v in merged_equity_curve]
        returns = pd.Series(values).pct_change().dropna()
        
        total_return = (values[-1] - backtest_config.initial_cash) / backtest_config.initial_cash
        
        days = (merged_equity_curve[-1][0] - merged_equity_curve[0][0]).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        merged_metrics = {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': len(all_trades),
        }
    else:
        merged_metrics = {}
    
    merge_time = time.perf_counter() - merge_start
    total_time = time.perf_counter() - total_start
    
    logger.info(f"✅ 结果合并完成, 耗时 {merge_time:.2f}秒")
    logger.info(f"🎉 多进程回测完成! 总耗时: {total_time:.2f}秒")
    logger.info(f"📊 统计: 信号数 {total_signals}, 交易数 {total_trades}")
    logger.info(f"💰 总收益率: {merged_metrics.get('total_return', 0):.2%}")
    
    # 10. 返回结果
    result = {
        'strategy_name': strategy_name,
        'stock_codes': actual_stock_codes,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_signals': total_signals,
        'executed_trades': total_trades,
        'trading_days': len(trading_dates),
        'performance_metrics': merged_metrics,
        'equity_curve': merged_equity_curve,
        'trades': all_trades,
        'worker_results': successful_results,
        'perf_breakdown': {
            'data_loading_s': data_load_time,
            'serialize_s': serialize_time,
            'multiprocess_s': mp_time,
            'merge_s': merge_time,
            'total_wall_s': total_time,
        },
        'num_workers': num_workers,
    }
    
    return result
