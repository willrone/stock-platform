#!/usr/bin/env python3
"""
测试数据同步API的本地降级策略
"""

import requests
import json

def test_sync_with_local_fallback():
    """测试同步API是否能够使用本地数据作为降级策略"""
    
    # 测试数据 - 使用我们知道本地有数据的股票代码
    sync_request = {
        "stock_codes": ["000001.SZ"],
        "start_date": "2024-12-01",
        "end_date": "2024-12-31",
        "force_update": False  # 不强制更新，允许使用本地数据
    }
    
    # API端点
    url = "http://127.0.0.1:8000/api/v1/data/sync"
    
    try:
        print("=== 测试数据同步API的本地降级策略 ===")
        print(f"请求数据: {json.dumps(sync_request, indent=2, ensure_ascii=False)}")
        print()
        
        response = requests.post(url, json=sync_request, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API请求成功")
            try:
                data = response.json()
                print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
                
                if data.get('success'):
                    print("🎉 同步成功！")
                    sync_data = data.get('data', {})
                    print(f"成功同步股票数: {sync_data.get('success_count', 0)}")
                    print(f"失败股票数: {sync_data.get('failure_count', 0)}")
                    print(f"总记录数: {sync_data.get('total_records', 0)}")
                    
                    if sync_data.get('successful_syncs'):
                        print("成功同步的股票:")
                        for sync_result in sync_data['successful_syncs']:
                            print(f"  - {sync_result['stock_code']}: {sync_result['records_synced']} 条记录")
                    
                    return True
                else:
                    print("⚠️  同步失败")
                    print(f"失败原因: {data.get('message', 'N/A')}")
                    return False
                    
            except Exception as e:
                print(f"解析响应数据失败: {e}")
                print(f"原始响应: {response.text}")
                return False
        else:
            print(f"❌ API请求失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务，请确保后端正在运行")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_sync_with_local_fallback()
    
    print("\n=== 测试结果 ===")
    if success:
        print("🎉 数据同步API的本地降级策略工作正常！")
        print("即使远程数据服务不可用，系统也能使用本地数据提供服务。")
    else:
        print("❌ 本地降级策略可能存在问题，需要进一步调试。")