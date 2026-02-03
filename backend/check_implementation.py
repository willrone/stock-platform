#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.data.simple_data_service import SimpleDataService

async def check_implementation():
    ds = SimpleDataService(offline_fallback=True)
    print('Method source check:')
    import inspect
    src = inspect.getsource(ds.get_stock_data)
    if '_is_valid_stock_code' in src:
        print('Found my fix in get_stock_data method')
    else:
        print('My fix NOT found in get_stock_data method')
        print('Current method contains:', [line.strip() for line in src.split('\n') if 'invalid' in line.lower()])
    
    # Test the method directly
    result = ds._is_valid_stock_code('INVALID.CODE')
    print(f'_is_valid_stock_code(INVALID.CODE): {result}')
    
    # Test generate_mock_data
    from datetime import datetime
    mock_result = ds.generate_mock_data('INVALID.CODE', datetime(2023, 1, 1), datetime(2023, 1, 10))
    print(f'generate_mock_data for INVALID.CODE returns: {len(mock_result)} items')
    
    # Test the full flow
    stock_data = await ds.get_stock_data('INVALID.CODE', datetime(2023, 1, 1), datetime(2023, 1, 10))
    print(f'get_stock_data for INVALID.CODE returns: {stock_data}')

if __name__ == "__main__":
    asyncio.run(check_implementation())