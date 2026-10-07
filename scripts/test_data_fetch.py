#!/usr/bin/env python3
"""
数据服务快速测试脚本
"""
import sys, os, json, requests

BASE = os.getenv("DATA_SERVICE_BASE", "http://127.0.0.1:5002")

def test():
    print("🧪 数据服务冒烟测试\n")
    r = requests.get(f"{BASE}/api/data/health", timeout=3)
    print(f"[健康检查] {r.status_code} — {r.json().get('message', 'ok')}")

    r = requests.post(f"{BASE}/api/data/manual_fetch", json={"stock_code": "000001.SZ"}, timeout=5)
    print(f"[手动抓取] {r.status_code} — {r.text[:120] if r.text else 'ok'}")

    print("\n✅ 数据服务测试通过" if r.status_code < 500 else "\n❌ 数据服务异常")
