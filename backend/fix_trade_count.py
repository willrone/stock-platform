#!/usr/bin/env python3
"""Fix trade_count key in test_backtest_data_adapter_properties.py"""

path = "tests/unit/backtest/test_backtest_data_adapter_properties.py"
content = open(path).read()

# The dict key is 'trade_count', not 'total_trades'
old = '                _trade_count = position_data.get("total_trades", 0)'
new = '                _trade_count = position_data.get("trade_count", 0)'

if old not in content:
    print(f"ERROR: old text not found")
    import sys; sys.exit(1)

content = content.replace(old, new)
open(path, "w").write(content)
print("OK: fixed trade_count key")
