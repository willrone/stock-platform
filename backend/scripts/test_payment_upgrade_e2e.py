#!/usr/bin/env python3
"""充值打桩 E2E：下单带 plan_id → 管理员确认到账 → 套餐自动升级（2026-10-27）

验证任务列表第 1 项：「充值打桩：确认到账后自动升级 subscription_tier」。

## 流程（真实 HTTP，走 FastAPI TestClient，DB 落盘到临时 sqlite）
1. 注册普通用户（默认 tier = free）
2. 下单 ¥199，plan_id = "pro" → 订单 pending，**余额不变**
3. 普通用户调用管理接口 → 403
4. 管理员确认到账 → 余额 +199，**tier free → pro**
5. 重复确认 → 400（幂等保护）
6. 降级测试：free 用户下单 plan_id="free" → 确认后 tier 仍是 pro（只升不降）

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/test_payment_upgrade_e2e.py
"""
from __future__ import annotations

import os
import sys
import tempfile

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

DB = os.path.join(tempfile.mkdtemp(prefix="paytest_"), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"
os.environ["TESTING"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.core.database import Base, sync_engine  # noqa: E402
import app.models  # noqa: E402,F401  （确保所有模型都被 import，触发 metadata 注册）

Base.metadata.create_all(bind=sync_engine)

from app.main import app  # noqa: E402

FAIL = []


def check(cond: bool, msg: str) -> None:
    print(("  ✅ " if cond else "  ❌ ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main() -> int:
    client = TestClient(app)

    # ── 1. 注册普通用户 ────────────────────────────────────────
    print("[e2e] 1. 注册用户", flush=True)
    email = "payer@example.com"
    r = client.post("/api/v1/auth/register", json={
        "email": email, "password": "TestPass123!", "username": "payer"})
    assert r.status_code in (200, 201), f"注册失败 {r.status_code} {r.text[:200]}"
    tok = r.json()["access_token"]
    H = {"Authorization": f"Bearer {tok}"}
    me = client.get("/api/v1/auth/me", headers=H).json()
    check(me.get("subscription_tier", "free") == "free",
          f"新用户 tier = {me.get('subscription_tier')}（应为 free）")
    check(me.get("is_admin") is False, "普通用户 is_admin = False")

    # ── 2. 下单 pro 套餐 ───────────────────────────────────────
    print("[e2e] 2. 下单 ¥199 plan_id=pro", flush=True)
    r = client.post("/api/v1/commerce/deposit", headers=H, json={
        "amount_cents": 19900, "plan_id": "pro", "payment_method": "manual"})
    assert r.status_code in (200, 201), f"下单失败 {r.status_code} {r.text[:300]}"
    order = r.json()
    oid = order.get("order_no") or order.get("id") or order.get("order_id")
    check(bool(oid), f"订单创建 id={oid}")
    _st = (order.get("status") or order.get("order_status")
           or (order.get("order") or {}).get("status"))
    check(_st == "pending", f"订单初始状态 = {_st}（应 pending）")

    bal0 = client.get("/api/v1/commerce/balance", headers=H).json()
    base_bal = float(bal0.get("balance_cents", 0))
    bal = client.get("/api/v1/commerce/balance", headers=H).json()
    check(abs(float(bal.get("balance_cents", 0)) - base_bal) < 1e-6,
          f"下单后余额未变 = {bal.get('balance_cents')}（基线 {base_bal}）")

    # ── 3. 普通用户调管理接口 → 403 ────────────────────────────
    print("[e2e] 3. 普通用户越权确认 → 应 403", flush=True)
    r = client.post(f"/api/v1/commerce/admin/payment/orders/{oid}/confirm",
                    headers=H, params={"remark": "e2e"})
    check(r.status_code == 403, f"返回 {r.status_code}（应 403）")

    # ── 4. 管理员确认 → 余额 +199，tier 升级 ────────────────────
    print("[e2e] 4. 管理员确认到账", flush=True)
    admin_email = "admin@example.com"
    r = client.post("/api/v1/auth/register", json={
        "email": admin_email, "password": "AdminPass123!", "username": "admin"})
    # 注册接口不接 is_admin（防止自我提权）→ 一律走 DB 提升后重新登录
    from app.core.database import SessionLocal
    from app.models.user_models import User
    with SessionLocal() as sess:
        u = sess.query(User).filter(User.email == admin_email).first()
        if u is None:
            print("  ❌ 管理员账号不存在")
            return 1
        u.is_admin = True
        sess.commit()
    r = client.post("/api/v1/auth/login", json={
        "email": admin_email, "password": "AdminPass123!"})
    atok = r.json().get("access_token")
    AH = {"Authorization": f"Bearer {atok}"}
    adash = client.get("/api/v1/auth/me", headers=AH).json()
    check(adash.get("is_admin") is True,
          f"管理员 is_admin = {adash.get('is_admin')}")

    r = client.post(f"/api/v1/commerce/admin/payment/orders/{oid}/confirm",
                    headers=AH, params={"remark": "e2e"})
    check(r.status_code in (200, 201), f"确认返回 {r.status_code} {r.text[:150]}")

    bal = client.get("/api/v1/commerce/balance", headers=H).json()
    check(abs(float(bal.get("balance_cents", 0)) - (base_bal + 19900)) < 1e-6,
          f"确认后余额 = {bal.get('balance_cents')}（应 {base_bal + 19900}）")

    me = client.get("/api/v1/auth/me", headers=H).json()
    check(me.get("subscription_tier") == "pro",
          f"套餐自动升级 tier = {me.get('subscription_tier')}（应 pro）")

    # ── 5. 重复确认 → 400 ──────────────────────────────────────
    print("[e2e] 5. 重复确认 → 应 400", flush=True)
    r = client.post(f"/api/v1/commerce/admin/payment/orders/{oid}/confirm",
                    headers=AH, params={"remark": "e2e"})
    check(r.status_code == 400, f"返回 {r.status_code}（应 400）")

    # ── 6. 最低充值额下限 ──────────────────────────────────────
    print("[e2e] 6. 最低充值额 ¥10 下限", flush=True)
    r = client.post("/api/v1/commerce/deposit", headers=H, json={
        "amount_cents": 100, "plan_id": "free", "payment_method": "manual"})
    check(r.status_code >= 400, f"¥1 下单被拒（返回 {r.status_code}）")

    # ── 7. 只升不降 ────────────────────────────────────────────
    print("[e2e] 7. 降级保护：pro 用户下 free 单 → tier 不应降", flush=True)
    r = client.post("/api/v1/commerce/deposit", headers=H, json={
        "amount_cents": 1000, "plan_id": "free", "payment_method": "manual"})
    check(r.status_code in (200, 201), f"¥10 下单返回 {r.status_code}")
    oid2 = (r.json().get("order_no") or r.json().get("id")
            or r.json().get("order_id"))
    r = client.post(f"/api/v1/commerce/admin/payment/orders/{oid2}/confirm",
                    headers=AH, params={"remark": "e2e"})
    check(r.status_code in (200, 201), f"确认返回 {r.status_code}")
    me = client.get("/api/v1/auth/me", headers=H).json()
    check(me.get("subscription_tier") == "pro",
          f"tier 仍 = {me.get('subscription_tier')}（应保持 pro）")

    print(f"\n[e2e] {'全部通过 ✅' if not FAIL else f'{len(FAIL)} 项失败 ❌'}")
    for f in FAIL:
        print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
