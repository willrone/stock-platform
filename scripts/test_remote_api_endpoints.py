#!/usr/bin/env python3
"""
测试远程数据服务的API端点
"""

import requests
import json

def test_endpoints():
    """测试远程数据服务的各个端点"""
    base_url = "http://192.168.3.62:5002"
    
    endpoints = [
        "/api/data/health",
        "/api/data/stock_data_status",
        "/api/data/data_summary",
        "/api/data/stock/000001.SZ/daily?start_date=2024-12-01&end_date=2024-12-31",
        "/api/data/stock_data_status/000001.SZ"
    ]
    
    print("=== 测试远程数据服务API端点 ===")
    print(f"基础URL: {base_url}")
    print()
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"测试: {endpoint}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"  状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("  ✅ 端点可用")
                try:
                    data = response.json()
                    if 'data' in data and isinstance(data['data'], list):
                        print(f"  📊 返回数据条数: {len(data['data'])}")
                    elif 'total_stocks' in data:
                        print(f"  📊 股票总数: {data['total_stocks']}")
                    elif 'status' in data:
                        print(f"  📊 服务状态: {data['status']}")
                except:
                    print(f"  📄 响应长度: {len(response.text)} 字符")
            elif response.status_code == 404:
                print("  ❌ 端点不存在")
            else:
                print(f"  ⚠️  其他错误: {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            print("  ❌ 连接失败")
        except requests.exceptions.Timeout:
            print("  ❌ 请求超时")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        
        print()

if __name__ == "__main__":
    test_endpoints()