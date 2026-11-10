"""风格门禁本地预跑：push 之前先自己跑 CI 的那三条，别把红灯推上去。

背景（2026-11-10）：feat 分支 27 天没跑过 CI（CI 只在 main/develop 的 push 与 PR 上触发），
攒了 89 条 flake8 + 61 个 black 未格式化文件，合进 main 当天三条门禁全红。
本测试把同一套门禁挪到本地测试里，push 前就能看见。

工具没装就跳过（CI 的测试 job 只装 requirements.txt），但本地 venv 全装了 → 本地必跑。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]
TARGETS = {"black": ["app/"], "isort": ["--check-only", "app/"], "flake8": ["app/"]}


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.parametrize("tool", ["black", "isort", "flake8"])
def test_backend_style_gate_is_clean(tool: str):
    if not _have(tool):
        pytest.skip(f"{tool} 没装（本地开发环境应装：backend/requirements-quality.txt）")
    cmd = [sys.executable, "-m", tool, *TARGETS[tool]]
    if tool == "black":
        cmd.insert(3, "--check")
    proc = subprocess.run(cmd, cwd=BACKEND, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"{tool} 门禁红：\n{proc.stdout[-3000:]}\n{proc.stderr[-1500:]}"
    )
