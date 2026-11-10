#!/usr/bin/env python3
"""闸门自身可运行性体检 —— 「门禁的门」必须是能跑的。

为什么需要：
2026-11-01 实查发现本仓库有**三个门禁从来没真正跑过**，而且都长得像在跑：
1. `scripts/pre-commit-check.sh` + `scripts/quick-check.sh` 是 **CRLF 行尾**
   → bash 解析到 `for ... do\r` 直接语法错误 → 脚本**第一行都没执行**。
   （仓库里其余 41 个 .sh 都是 LF，只有这两个是 CRLF → 典型的
   「从 Windows 编辑器/网页复制粘贴引入」，且没有任何检查发现它。）
2. `backend/scripts/check-code-quality.sh` 的 flake8 **只覆盖 `app/`**，
   而 cron/launchd 实际执行的是 `scripts/` → F821（必崩的 NameError）没人看。
3. `backend/.venv-py313/bin/pyflakes` 的 shebang 写着 `/Users/ronghui/...`
   （旧用户名）→ 直接执行报 "bad interpreter"。

**教训**：一个门禁「有文件 / 在 CI 里被引用」不等于「它在跑」。
必须有一条独立的检查去验证**门禁本身可执行**。

本脚本检查：
- 所有 `.sh` 是 LF 行尾（CRLF → bash 语法错误）
- 所有 `.sh` 通过 `bash -n` 语法检查
- 关键闸门脚本存在且可解析

用法：
    python3 scripts/check_gates_runnable.py
"""
from __future__ import annotations

import os
import subprocess
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)

# 关键闸门（相对 ROOT）：必须存在 + 可解析
KEY_GATES = (
    "scripts/pre-commit-check.sh",
    "scripts/quick-check.sh",
    "scripts/check-code-quality.sh",
    "scripts/weekly_paper_portfolio.sh",
    "backend/scripts/audit_doc_numbers.py",
    "backend/scripts/artifact_manifest.py",
    "backend/scripts/check_undefined_names.py",
    "backend/scripts/trial_budget.py",
)


def shell_files() -> list[str]:
    out = []
    for base in (os.path.join(ROOT, "scripts"), os.path.join(BACKEND, "scripts")):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith(".sh"):
                out.append(os.path.join(base, name))
    return out


def main() -> int:
    problems: list[str] = []

    files = shell_files()
    for p in files:
        rel = os.path.relpath(p, ROOT)
        raw = open(p, "rb").read()
        if b"\r\n" in raw:
            n = raw.count(b"\r\n")
            problems.append(f"{rel}: CRLF 行尾 {n} 处 → bash 解析报错，脚本无法执行")
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(f"{rel}: bash 语法错误 → {r.stderr.strip()[:160]}")

    for rel in KEY_GATES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            problems.append(f"{rel}: 关键闸门脚本**不存在**")
            continue
        if rel.endswith(".sh"):
            r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
            if r.returncode != 0:
                problems.append(f"{rel}: 关键闸门 bash 语法错误")

    print(f"检查了 {len(files)} 个 .sh + {len(KEY_GATES)} 个关键闸门")
    if not problems:
        print("✅ 闸门可运行性体检通过：全部 .sh 为 LF 且 bash -n 通过，关键闸门齐全")
        return 0
    print(f"❌ 闸门可运行性问题 {len(problems)} 条：")
    for p in problems:
        print(f"   {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
