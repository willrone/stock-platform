"""仓库里不许再出现「写死的仓库绝对路径」（2026-11-10 CI 红灯的根因）。

写死 `/Users/rone/Projects/stock-platform/...` 的代码本地永远绿（路径恰好对），
CI 上是 `/home/runner/work/...`，import 阶段直接 FileNotFoundError —— 而且是在 collect 阶段炸，
一次带走一整条测试线。这类 bug 靠人眼 review 拦不住，所以写成测试。

规则（只抓「这个字符串被当路径用」，不抓文档里的举例）：
- 扫 backend/app、backend/scripts、backend/tests 的 .py，用 AST 取字符串字面量（注释不会被扫到）。
- 字面量**以** `/Users/`、`/home/`、`~/Projects/`、`~/projects/` 开头，并且含仓库标记
  （`stock-platform` 或 `Documents/GitHub/willrone`）→ 红。
- 放行：模块/函数/类 docstring；以及 `.hermes/cache/scratch` 这类研究脚本临时输出目录
  （不是仓库路径，换机器只是换个缓存位置）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("app", "scripts", "tests")
ABS_STARTS = ("/Users/", "/home/", "~/Projects/", "~/projects/")
REPO_MARKERS = ("stock-platform", "Documents/GitHub/willrone")
ALLOW_SUBSTR = ".hermes/cache/scratch"
SELF = Path(__file__).name


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    ids.add(id(body[0].value))
    return ids


@pytest.mark.parametrize("sub", SCAN_DIRS)
def test_no_hardcoded_repo_absolute_paths(sub: str):
    offenders: list[str] = []
    for path in sorted((BACKEND / sub).rglob("*.py")):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        if path.name == SELF:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            offenders.append(f"{path.relative_to(BACKEND)}:0 语法错误 {exc}")
            continue
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docs:
                continue
            val = node.value
            if ALLOW_SUBSTR in val:
                continue
            if not val.startswith(ABS_STARTS):
                continue
            if any(m in val for m in REPO_MARKERS):
                offenders.append(f"{path.relative_to(BACKEND)}:{node.lineno} {val[:90]}")
    assert not offenders, (
        "发现写死的仓库绝对路径（本地绿、CI 必红）。改成由 __file__ 推导：\n"
        + "\n".join(offenders[:40])
    )
