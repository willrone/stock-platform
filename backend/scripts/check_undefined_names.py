#!/usr/bin/env python3
"""未定义名（NameError 类）静态闸门 —— 只查 F821/F811，**零基线**。

为什么单独做：
- `scripts/check-code-quality.sh` 跑 flake8 但**只覆盖 `app/`**，
  而真正在 cron / launchd 里跑的是 `scripts/` → F821（undefined name）
  这一类**必崩**的错误从来没被门禁看过。
- 2026-11-01 实测：`scripts/paper_portfolio.py:219` 引用了已删除的
  `LIMIT_UP`（commit 3d3e34e5 删常量时漏删引用）→ 纸面盘跑到 gbm_voltgt
  那条线就 NameError 崩掉（RC=1）→ 同一次运行里后面的
  `multi_asset_final` 旗舰线**根本轮不到执行** → 旗舰线永远长不出净值点。
  这个 bug 从引入到被抓到隔了整整一个月。

为什么只查 F821/F811：
- F821 undefined name = **必然** NameError（真崩）。
- F811 redefinition = 静默覆盖，真崩的常见前兆。
- 其余（未用 import / 行太长 / 格式）都是**存量噪声**：全量 300+ 条，
  基线永远不归零 → 告警疲劳 → 门禁失效（与台账 §12.17 的
  「假警报淹没真故障」是同一个病）。
  所以这里**只对会崩的类别**设闸门，其余类别由 check-code-quality.sh
  做**存量基线**管理。

零基线 = 当前 F821 数量必须是 0；任何新增都 exit 1。
（不用「基线文件」模式：F821 本来就该是 0，允许基线就等于允许再烂下去。）

用法：
    python3 scripts/check_undefined_names.py            # 闸门（CI / pre-commit）
    python3 scripts/check_undefined_names.py --list     # 只列出，不改退出码
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 检查范围：app/（服务代码）+ scripts/（运维/研究脚本，cron/launchd 实际执行的就是这里）
TARGETS = ("app", "scripts")
SELECT = "F821,F811"


def find_flake8() -> list[list[str]]:
    """找 flake8 可执行文件。

    注意：venv 里的 `bin/flake8` 脚本 shebang 可能指向**别的用户名**
    （本机实测 `.venv-py313/bin/pyflakes` 的 shebang 写着 `/Users/ronghui/...`
    → 直接执行报 "bad interpreter"）。
    → 优先用 `<venv>/bin/python -m flake8`（绕开 shebang）。
    """
    cands: list[list[str]] = []
    for venv in (".venv", ".venv-py313", "venv"):
        for py in ("python3", "python"):
            p = os.path.join(BACKEND, venv, "bin", py)
            if os.path.exists(p):
                cands.append([p, "-m", "flake8"])
    # 系统 flake8 兜底
    cands.append(["flake8"])
    return cands


def run_flake8() -> tuple[int, str]:
    last_err = ""
    for cmd in find_flake8():
        try:
            r = subprocess.run(
                cmd + ["--select=" + SELECT] + list(TARGETS),
                cwd=BACKEND, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "", "PYTHONHOME": ""},
            )
        except (FileNotFoundError, OSError) as ex:
            last_err = f"{cmd[0]}: {ex}"
            continue
        out = (r.stdout or "") + (r.stderr or "")
        if "bad interpreter" in out or "No module named flake8" in out:
            last_err = out.strip()[:300]
            continue
        return r.returncode, out
    print(f"❌ 找不到可用的 flake8（最后一次错误：{last_err}）")
    return 127, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列出问题，不改退出码")
    args = ap.parse_args()

    rc, out = run_flake8()
    if rc == 127:
        return 1

    findings = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # 只保留真正的问题行（跳过 flake8 的统计/汇总行）
        if ": F821 " in line or ": F811 " in line:
            findings.append(line.strip())

    if not findings:
        print(f"✅ 未定义名闸门通过：{SELECT} 在 {', '.join(TARGETS)} 下 **0 条**")
        return 0

    print(f"❌ 发现 {len(findings)} 条未定义名/重复定义（F821/F811）：")
    for f in findings[:60]:
        print(f"   {f}")
    if len(findings) > 60:
        print(f"   … 另有 {len(findings) - 60} 条")
    if args.list:
        return 0
    print("\n修复指引：")
    print("  F821 undefined name → 该名字被删除/拼错但引用还在。")
    print("    真崩场景：普通表达式引用 → 运行时 NameError；")
    print("              函数签名参数注解引用 → **import 时**就崩；")
    print("              （有 `from __future__ import annotations` 的模块，"
          "注解不求值 → 只是类型检查报错，不会崩）")
    print("  F811 redefinition → 同一名字被定义两次，后者静默覆盖前者。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
