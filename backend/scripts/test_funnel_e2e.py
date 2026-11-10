#!/usr/bin/env python3
"""漏斗埋点端到端验证（真起服务、真发请求、真读库）。

跑法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/test_funnel_e2e.py

验证链路：
    1. 起 uvicorn（独立临时库，不碰线上 app.db）
    2. 匿名 POST /funnel/track（view_pricing，anon_id=anon-e2e）
    3. POST /auth/register（带 anon_id）→ 拿 token
       —— 断言：匿名事件已被归并到新用户
    4. 带 token POST /funnel/track（first_screen / view_signal）
    5. 把该用户提为 admin → GET /funnel/report
       —— 断言：漏斗顺序转化、付费墙计数、匿名归并后的人数
    6. 跑 CLI 报告脚本，确认 CLI 与 API 口径一致
"""

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
SCRATCH = Path("/Users/rone/.hermes/cache/scratch")
DB_PATH = SCRATCH / "funnel_e2e.db"

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name} {detail}")
        failures.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(method: str, url: str, body=None, token: str | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(DB_PATH) + suffix)
        if p.exists():
            p.unlink()

    port = free_port()
    base = f"http://127.0.0.1:{port}/api/v1"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
    env["BASELINE_LOG_LEVEL"] = "ERROR"

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=str(BACKEND),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # 等服务起来
        ready = False
        for _ in range(60):
            if server.poll() is not None:
                print("服务启动失败:\n" + server.stdout.read().decode()[-3000:])
                return 1
            try:
                status, _ = request("GET", f"{base}/health")
                if status < 500:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        check("后端启动", ready)
        if not ready:
            return 1

        # 1) 匿名埋点（注册前看了价格页）
        status, body = request(
            "POST",
            f"{base}/funnel/track",
            {"event": "view_pricing", "anon_id": "anon-e2e", "session_id": "sess-e2e"},
        )
        check("匿名 track 返回 202", status == 202, f"got {status} {body}")
        check("匿名 track 返回 ok", body.get("ok") is True, str(body))

        # 2) 未知事件名要拒绝（但不能 500）
        status, body = request(
            "POST", f"{base}/funnel/track", {"event": "not_real", "anon_id": "anon-e2e"}
        )
        check("未知事件被拒", status == 202 and body.get("ok") is False, str(body))

        # 3) 注册（带 anon_id）
        email = f"funnel-e2e-{int(time.time())}@example.com"
        status, body = request(
            "POST",
            f"{base}/auth/register",
            {
                "email": email,
                "username": f"funnel_e2e_{int(time.time())}",
                "password": "e2e-pass-123",
                "anon_id": "anon-e2e",
            },
        )
        check("注册成功", status == 201, f"got {status} {body}")
        token = body.get("access_token")
        check("拿到 access_token", bool(token))
        user_id = (body.get("user") or {}).get("id")
        check("拿到 user_id", bool(user_id))

        # 4) 带 token 埋点
        for ev, props in (
            ("first_screen", {"path": "dashboard"}),
            ("view_signal", {"source": "today-signal"}),
        ):
            status, b = request(
                "POST",
                f"{base}/funnel/track",
                {"event": ev, "anon_id": "anon-e2e", "props": props},
                token=token,
            )
            check(f"带 token track {ev}", status == 202 and b.get("ok") is True, str(b))

        # 5) 提权为 admin 后读报告
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

        status, report = request("GET", f"{base}/funnel/report?days=30", token=token)
        check("report 返回 200", status == 200, f"got {status} {str(report)[:300]}")
        if status != 200:
            return 1

        funnel = {s["step"]: s for s in report["funnel"]}
        check("漏斗含注册步", "signup" in funnel)
        check(
            "注册触达 1 人",
            funnel["signup"]["users"] == 1,
            str(funnel["signup"]),
        )
        check(
            "匿名 view_pricing 已归并到该用户（付费墙触点=1 人）",
            report["paywall"]["view_pricing_users"] == 1,
            str(report["paywall"]),
        )
        check(
            "首屏触达 1 人",
            funnel["first_screen"]["users"] == 1,
            str(funnel["first_screen"]),
        )
        check(
            "首条信号触达 1 人",
            funnel["view_signal"]["users"] == 1,
            str(funnel["view_signal"]),
        )
        check(
            "顺序转化：到首条信号仍为 1",
            funnel["view_signal"]["converted"] == 1,
            str(funnel["view_signal"]),
        )
        check(
            "顺序转化：付费墙触点=1（匿名看价格页已归并到注册用户）",
            funnel["view_pricing"]["converted"] == 1,
            str(funnel["view_pricing"]),
        )
        check(
            "顺序转化：升级点击=0（用户没点过升级 CTA）",
            funnel["click_upgrade"]["converted"] == 0,
            str(funnel["click_upgrade"]),
        )
        check("已识别用户数=1", report["totals"]["identified_users"] == 1, str(report["totals"]))
        check("匿名残留=0（已全部归并）", report["totals"]["anonymous_users"] == 0, str(report["totals"]))
        # 4 条：匿名 view_pricing + 后端 signup + first_screen + view_signal
        # （未知事件名被拒，不入库）
        check("事件总数=4", report["totals"]["events"] == 4, str(report["totals"]))

        # 6) CLI 报告与 API 口径一致
        cli = subprocess.run(
            [sys.executable, "scripts/funnel_report.py", "--json"],
            cwd=str(BACKEND),
            env=env,
            capture_output=True,
            text=True,
        )
        check("CLI 报告脚本退出码 0", cli.returncode == 0, cli.stderr[-500:])
        if cli.returncode == 0:
            cli_report = json.loads(cli.stdout)
            check(
                "CLI 与 API 的事件总数一致",
                cli_report["totals"]["events"] == report["totals"]["events"],
                f"{cli_report['totals']['events']} vs {report['totals']['events']}",
            )
            check(
                "CLI 与 API 的 funnel 触达一致",
                [s["users"] for s in cli_report["funnel"]]
                == [s["users"] for s in report["funnel"]],
            )

        # 7) 管理员可读报告；**非管理员必须读不到**（经营数据不外泄）
        status2, _ = request("GET", f"{base}/funnel/report", token=token)
        check("管理员可读报告", status2 == 200, f"got {status2}")

        email2 = f"funnel-e2e-nonadmin-{int(time.time())}@example.com"
        status3, body3 = request(
            "POST",
            f"{base}/auth/register",
            {
                "email": email2,
                "username": f"funnel_nonadmin_{int(time.time())}",
                "password": "e2e-pass-123",
            },
        )
        check("非管理员账号注册成功", status3 == 201, f"got {status3} {str(body3)[:200]}")
        status4, _ = request(
            "GET", f"{base}/funnel/report", token=body3.get("access_token")
        )
        check(
            "非管理员读报告被拒（403）",
            status4 == 403,
            f"got {status4}",
        )
        status5, _ = request("GET", f"{base}/funnel/report")
        check("未登录读报告被拒（401）", status5 == 401, f"got {status5}")

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    print()
    if failures:
        print(f"❌ 失败 {len(failures)} 项: {failures}")
        return 1
    print("✅ 漏斗 E2E 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
