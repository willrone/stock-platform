#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.data.simple_data_service import SimpleDataService

ds = SimpleDataService()
print("INVALID.CODE:", ds._is_valid_stock_code('INVALID.CODE'))
print("000001:", ds._is_valid_stock_code('000001'))
print("TEST:", ds._is_valid_stock_code('TEST'))
print("invalid.code:", ds._is_valid_stock_code('invalid.code'))
print("None value:", ds._is_valid_stock_code(None))
print("Empty string:", ds._is_valid_stock_code(''))