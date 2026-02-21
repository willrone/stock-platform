#!/usr/bin/env python3
"""Fix 3 failing backtest tests"""
import sys

# === Fix 1: test_backtest_data_adapter_properties.py ===
path1 = "tests/unit/backtest/test_backtest_data_adapter_properties.py"
content1 = open(path1).read()

old1 = """        for position_data in position_analysis:
            # 验证基本字段完整性
            assert position_data.stock_code is not None
            assert isinstance(position_data.total_return, float)
            assert isinstance(position_data.trade_count, int)
            assert isinstance(position_data.win_rate, float)
            assert isinstance(position_data.winning_trades, int)
            assert isinstance(position_data.losing_trades, int)
            
            # 验证数值的合理性
            assert not np.isnan(position_data.total_return)
            assert not np.isnan(position_data.win_rate)
            assert 0 <= position_data.win_rate <= 1, f"胜率 {position_data.win_rate} 应该在0-1之间"
            
            # 验证交易统计的一致性
            assert position_data.trade_count == trades_per_stock, f"交易次数应该是 {trades_per_stock}"
            assert position_data.winning_trades + position_data.losing_trades <= position_data.trade_count
            
            # 验证胜率计算的准确性
            if position_data.trade_count > 0:
                expected_win_rate = position_data.winning_trades / position_data.trade_count
                win_rate_error = abs(position_data.win_rate - expected_win_rate)
                assert win_rate_error < 0.001, f"胜率计算错误: 期望 {expected_win_rate}, 实际 {position_data.win_rate}"
            
            # 验证持仓期的合理性
            assert position_data.avg_holding_period >= 0, "平均持仓期应该非负"
            assert position_data.avg_holding_period < 365, "平均持仓期应该小于一年\""""

new1 = """        # EnhancedPositionAnalysis is not iterable; iterate over stock_performance (list of dicts)
        stock_perf_list = position_analysis.stock_performance if isinstance(position_analysis, EnhancedPositionAnalysis) else position_analysis
        for position_data in stock_perf_list:
            # stock_performance items are dicts after refactor
            if isinstance(position_data, dict):
                _stock_code = position_data.get("stock_code")
                _total_return = position_data.get("total_return", 0.0)
                _trade_count = position_data.get("total_trades", 0)
                _win_rate = position_data.get("win_rate", 0.0)
                _winning_trades = position_data.get("winning_trades", 0)
                _losing_trades = position_data.get("losing_trades", 0)
                _avg_holding_period = position_data.get("avg_holding_period", 0)
            else:
                _stock_code = position_data.stock_code
                _total_return = position_data.total_return
                _trade_count = position_data.trade_count
                _win_rate = position_data.win_rate
                _winning_trades = position_data.winning_trades
                _losing_trades = position_data.losing_trades
                _avg_holding_period = position_data.avg_holding_period

            # 验证基本字段完整性
            assert _stock_code is not None
            assert isinstance(_total_return, (int, float))
            assert isinstance(_trade_count, int)
            assert isinstance(_win_rate, (int, float))
            assert isinstance(_winning_trades, int)
            assert isinstance(_losing_trades, int)
            
            # 验证数值的合理性
            assert not np.isnan(_total_return)
            assert not np.isnan(_win_rate)
            assert 0 <= _win_rate <= 1, f"胜率 {_win_rate} 应该在0-1之间"
            
            # 验证交易统计的一致性
            assert _trade_count == trades_per_stock, f"交易次数应该是 {trades_per_stock}"
            assert _winning_trades + _losing_trades <= _trade_count
            
            # 验证胜率计算的准确性
            if _trade_count > 0:
                expected_win_rate = _winning_trades / _trade_count
                win_rate_error = abs(_win_rate - expected_win_rate)
                assert win_rate_error < 0.001, f"胜率计算错误: 期望 {expected_win_rate}, 实际 {_win_rate}"
            
            # 验证持仓期的合理性
            assert _avg_holding_period >= 0, "平均持仓期应该非负"
            assert _avg_holding_period < 365, "平均持仓期应该小于一年\""""

if old1 not in content1:
    print(f"ERROR: old text not found in {path1}")
    sys.exit(1)
content1 = content1.replace(old1, new1)
open(path1, "w").write(content1)
print(f"OK: {path1} patched (position_analysis iteration fix)")


# === Fix 2: test_backtest_engine_properties.py - generate_signals needs current_date ===
path2 = "tests/unit/backtest/test_backtest_engine_properties.py"
content2 = open(path2).read()

old2 = """            strategy = StrategyFactory.create_strategy(strategy_name, config)
            sample_data = self._create_sample_stock_data()
            signals = strategy.generate_signals(sample_data)"""

new2 = """            strategy = StrategyFactory.create_strategy(strategy_name, config)
            sample_data = self._create_sample_stock_data()
            # generate_signals now requires current_date after refactor
            current_date = sample_data.index[-1].to_pydatetime() if hasattr(sample_data.index[-1], 'to_pydatetime') else sample_data.index[-1]
            signals = strategy.generate_signals(sample_data, current_date)"""

if old2 not in content2:
    print(f"ERROR: old text not found in {path2} (generate_signals)")
    sys.exit(1)
content2 = content2.replace(old2, new2)

# === Fix 3: test_backtest_engine_properties.py - execute_buy/execute_sell -> execute_signal ===
old3 = """        buy_price = 100.0
        max_shares = int((initial_cash * max_position_size) / buy_price)

        if max_shares > 0:
            portfolio_manager.execute_buy('000001.SZ', buy_price, max_shares, datetime(2024, 1, 1))

            if '000001.SZ' in portfolio_manager.positions:
                position = portfolio_manager.positions['000001.SZ']
                assert position.quantity > 0
                assert portfolio_manager.cash < initial_cash

                sell_price = 110.0
                portfolio_manager.execute_sell('000001.SZ', sell_price, position.quantity, datetime(2024, 1, 2))
                assert '000001.SZ' not in portfolio_manager.positions or portfolio_manager.positions['000001.SZ'].quantity == 0"""

new3 = """        buy_price = 100.0
        max_shares = int((initial_cash * max_position_size) / buy_price)

        if max_shares > 0:
            # execute_buy/execute_sell removed; use execute_signal with TradingSignal
            buy_signal = TradingSignal(
                timestamp=datetime(2024, 1, 1),
                stock_code='000001.SZ',
                signal_type=SignalType.BUY,
                strength=1.0,
                price=buy_price,
                reason='test buy',
            )
            trade, fail_reason = portfolio_manager.execute_signal(buy_signal, {'000001.SZ': buy_price})

            if trade is not None and '000001.SZ' in portfolio_manager.positions:
                position = portfolio_manager.positions['000001.SZ']
                assert position.quantity > 0
                assert portfolio_manager.cash < initial_cash

                sell_price = 110.0
                sell_signal = TradingSignal(
                    timestamp=datetime(2024, 1, 2),
                    stock_code='000001.SZ',
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=sell_price,
                    reason='test sell',
                )
                trade2, fail_reason2 = portfolio_manager.execute_signal(sell_signal, {'000001.SZ': sell_price})
                assert '000001.SZ' not in portfolio_manager.positions or portfolio_manager.positions['000001.SZ'].quantity == 0"""

if old3 not in content2:
    print(f"ERROR: old text not found in {path2} (execute_buy/sell)")
    sys.exit(1)
content2 = content2.replace(old3, new3)

# Also need to add SignalType import if not present
if "from app.services.backtest.models.enums import SignalType" not in content2:
    # It's already imported, check
    pass

open(path2, "w").write(content2)
print(f"OK: {path2} patched (generate_signals + execute_signal fixes)")

print("\nAll patches applied successfully!")
