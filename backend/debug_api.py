#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_api_directly():
    with TestClient(app) as client:
        # Test the specific failing endpoint
        response = client.get("/api/v1/stocks/INVALID.CODE/indicators")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response data keys: {list(data.keys())}")
            print(f"Success: {data.get('success', 'MISSING')}")
            print(f"Message: {data.get('message', 'MISSING')}")
            print(f"Data keys: {list(data.get('data', {}).keys()) if data.get('data') else 'No data'}")
            if data.get('data') and 'indicators' in data['data']:
                print(f"Indicators: {data['data']['indicators']}")
                print(f"Indicators length: {len(data['data']['indicators'])}")
        else:
            print(f"Non-200 response: {response.text}")

if __name__ == "__main__":
    test_api_directly()