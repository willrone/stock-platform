#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datetime import datetime
from app.services.data.simple_data_service import SimpleDataService
from app.services.prediction.technical_indicators import TechnicalIndicatorCalculator

async def test_detailed_flow():
    print("=== Testing the full flow ===")
    
    # Create services like the API does
    data_service = SimpleDataService(offline_fallback=True)
    indicators_service = TechnicalIndicatorCalculator()
    
    print("Testing INVALID.CODE...")
    
    # Step 1: Get stock data
    stock_data = await data_service.get_stock_data(
        'INVALID.CODE', 
        datetime(2023, 1, 1), 
        datetime(2023, 1, 10)
    )
    
    print(f"get_stock_data result: {stock_data}")
    print(f"Is stock_data None? {stock_data is None}")
    print(f"Length of stock_data: {len(stock_data) if stock_data else 0}")
    
    if stock_data is not None and len(stock_data) > 0:
        print("First few stock data items:")
        for i, item in enumerate(stock_data[:2]):
            print(f"  {i}: {item}")
    
    # Step 2: Calculate indicators if stock data exists
    if stock_data:
        print("\nCalculating indicators...")
        indicator_list = ["MA5", "MA10", "MA20", "RSI", "MACD"]
        
        try:
            indicator_results = indicators_service.calculate_indicators(
                stock_data, indicator_list
            )
            print(f"Indicator results: {indicator_results}")
            print(f"Number of indicator results: {len(indicator_results)}")
            
            if indicator_results:
                latest_indicators = indicator_results[-1].indicators
                print(f"Latest indicators: {latest_indicators}")
        except Exception as e:
            print(f"Error calculating indicators: {e}")
    else:
        print("\nNo stock data - would skip indicator calculation")

if __name__ == "__main__":
    asyncio.run(test_detailed_flow())