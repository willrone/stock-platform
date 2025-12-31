#!/usr/bin/env python3
"""
测试数据同步API修复
"""

import requests
import json

def test_remote_data_service():
    """测试远程数据服务的新API端点"""
    
    # 测试远程数据服务的股票数据API
    remote_url = "http://192.168.3.62:5002/api/data/stock/000001.SZ/daily"
    params = {
        "start_date": "2024-12-01",
        "end_date": "2024-12-31"
    }
    
    try:
        print("测试远程数据服务API...")
        print(f"请求URL: {remote_url}")
        print(f"请求参数: {json.dumps(params, indent=2)}")
        
        response = requests.get(remote_url, params=params, timeout=10)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 远程数据服务API正常")
            try:
                data = response.json()
                print(f"返回数据条数: {data.get('total_records', 0)}")
            except:
                print(f"响应文本: {response.text[:200]}...")
            return True
        else:
            print(f"❌ 远程数据服务API错误: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到远程数据服务，请确保服务正在运行")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_sync_api():
    """测试同步API是否接受正确的请求体格式"""
    
    # 测试数据
    sync_request = {
        "stock_codes": ["000001.SZ"],
        "force_update": True
    }
    
    # API端点
    url = "http://127.0.0.1:8000/api/v1/data/sync"
    
    try:
        print("\n发送同步请求...")
        print(f"请求数据: {json.dumps(sync_request, indent=2)}")
        
        response = requests.post(url, json=sync_request, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 422:
            print("❌ 仍然是422错误 - 请求格式问题")
            print(f"错误详情: {response.text}")
            return False
        elif response.status_code == 200:
            print("✅ 请求成功 - 422错误已修复")
            try:
                data = response.json()
                print(f"同步结果: {data.get('message', 'N/A')}")
                if data.get('success'):
                    print("✅ 同步成功")
                else:
                    print("⚠️  同步失败，但API格式正确")
                    print(f"失败原因: {data.get('message', 'N/A')}")
            except:
                print(f"响应文本: {response.text}")
            return True
        else:
            print(f"⚠️  其他状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务，请确保后端正在运行")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 测试数据同步API修复 ===")
    
    # 首先测试远程数据服务
    remote_success = test_remote_data_service()
    
    # 然后测试同步API
    sync_success = test_sync_api()
    
    print("\n=== 测试结果 ===")
    if remote_success and sync_success:
        print("🎉 所有测试通过！数据同步功能已完全修复。")
    elif sync_success:
        print("✅ API格式修复成功，但远程数据服务可能需要重启或配置。")
    else:
        print("❌ 仍有问题需要解决。")