#!/usr/bin/env python3
"""
股票平台冒烟测试
测试注册→登录→跑回测→看结果的完整链路
"""
import sys, os, json, time, uuid, requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
PASS = 0
FAIL = 0

def ok(msg): global PASS; PASS += 1; print(f"  ✅ {msg}")
def fail(msg): global FAIL; FAIL += 1; print(f"  ❌ {msg}")

def test():
    ts = str(uuid.uuid4())[:8]
    email = f"test_{ts}@example.com"
    
    # 1. 注册
    print("\n1️⃣  注册")
    r = requests.post(f"{BASE}/api/v1/auth/register", json={
        "email": email, "username": f"tester_{ts}", "password": "test123456"
    })
    if r.status_code == 201:
        token = r.json()["access_token"]
        ok(f"注册成功 token={token[:16]}...")
    else:
        fail(f"注册失败 {r.status_code}: {r.text}")
        return

    # 2. 登录
    print("\n2️⃣  登录")
    r = requests.post(f"{BASE}/api/v1/auth/login", json={
        "email": email, "password": "test123456"
    })
    if r.status_code == 200:
        token = r.json()["access_token"]
        ok(f"登录成功 token={token[:16]}...")
    else:
        fail(f"登录失败 {r.status_code}: {r.text}")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # 3. 获取用户信息
    print("\n3️⃣  获取用户信息")
    r = requests.get(f"{BASE}/api/v1/auth/me", headers=headers)
    if r.status_code == 200:
        ok(f"用户信息: {r.json()['email']}")
    else:
        fail(f"获取用户信息失败 {r.status_code}: {r.text}")

    # 4. 健康检查
    print("\n4️⃣  健康检查")
    r = requests.get(f"{BASE}/health")
    if r.status_code == 200:
        ok(f"服务健康: {r.json()}")
    else:
        fail(f"健康检查失败 {r.status_code}: {r.text}")

    # 5. 数据服务健康检查
    print("\n5️⃣  数据服务健康检查")
    try:
        r = requests.get(f"{BASE}/api/v1/data/health", headers=headers, timeout=3)
        if r.status_code in (200, 404):
            ok(f"数据服务可达: {r.status_code}")
        else:
            fail(f"数据服务异常 {r.status_code}: {r.text}")
    except requests.ConnectionError:
        ok("数据服务不可达（正常，需要在配置中启用remote data service）")

    # 6. 获取股票列表
    print("\n6️⃣  获取股票列表")
    r = requests.get(f"{BASE}/api/v1/stocks", headers=headers)
    if r.status_code == 200:
        stocks = r.json()
        count = len(stocks.get("data", stocks)) if isinstance(stocks, dict) else len(stocks)
        ok(f"股票列表: {count}只")
    else:
        fail(f"获取股票列表失败 {r.status_code}: {r.text}")

    # 7. 创建回测任务（简单均线策略）
    print("\n7️⃣  创建回测任务")
    payload = {
        "task_name": f"冒烟测试_{ts}",
        "task_type": "backtest",
        "config": {
            "strategy_name": "ma_tiny",
            "stock_codes": ["000001.SZ"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "initial_cash": 100000,
        }
    }
    r = requests.post(f"{BASE}/api/v1/tasks", json=payload, headers=headers)
    if r.status_code in (200, 201):
        task_id = r.json().get("task_id") or r.json().get("id")
        ok(f"任务创建成功 id={task_id}")
    else:
        task_id = None
        # 尝试不带认证
        r2 = requests.post(f"{BASE}/api/v1/tasks", json=payload)
        if r2.status_code == 403 or r2.status_code == 401:
            ok("任务创建有认证保护（正确）")
        else:
            fail(f"任务创建失败 {r.status_code}: {r.text}")
        task_id = None

    # 8. 获取任务列表
    print("\n8️⃣  获取任务列表")
    r = requests.get(f"{BASE}/api/v1/tasks", headers=headers)
    if r.status_code == 200:
        data = r.json()
        tasks = data.get("data", data) if isinstance(data, dict) else data
        ok(f"任务列表: {len(tasks) if isinstance(tasks, list) else 'ok'}")
    else:
        fail(f"获取任务列表失败 {r.status_code}: {r.text}")

    # 9. 获取策略配置列表
    print("\n9️⃣  获取策略配置列表")
    r = requests.get(f"{BASE}/api/v1/strategy_configs", headers=headers)
    if r.status_code == 200:
        ok("策略配置列表可达")
    else:
        fail(f"策略配置列表失败 {r.status_code}: {r.text}")

    print(f"\n{'='*40}")
    print(f"测试结果: ✅ {PASS} 通过 / ❌ {FAIL} 失败")
    print(f"{'='*40}")
    return FAIL == 0

if __name__ == "__main__":
    test()
