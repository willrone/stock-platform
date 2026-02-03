#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datetime import datetime
from app.services.data.simple_data_service import SimpleDataService

async def test_invalid_stock():
    ds = SimpleDataService(offline_fallback=True)
    
    print("Testing INVALID.CODE...")
    print("_is_valid_stock_code('INVALID.CODE'):", ds._is_valid_stock_code('INVALID.CODE'))
    
    # Test generate_mock_data directly
    mock_data = ds.generate_mock_data('INVALID.CODE', datetime(2023, 1, 1), datetime(2023, 1, 10))
    print("generate_mock_data result:", mock_data)
    print("Length of mock data:", len(mock_data))
    
    # Test get_stock_data
    stock_data = await ds.get_stock_data('INVALID.CODE', datetime(2023, 1, 1), datetime(2023, 1, 10))
    print("get_stock_data result:", stock_data)
    print("Is stock_data None?", stock_data is None)
    print("Length of stock_data:", len(stock_data) if stock_data else 0)

if __name__ == "__main__":
    asyncio.run(test_invalid_stock())