#!/usr/bin/env python3
"""Verify backend CI collect_ignore and cleanup ledger stay in sync."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CONFTEST = ROOT / "tests" / "conftest.py"
LEDGER = REPO_ROOT / "docs" / "quality" / "backend-ci-tail-cleanup.md"


def _load_collect_ignore() -> set[str]:
    tree = ast.parse(CONFTEST.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "collect_ignore":
                    value = ast.literal_eval(node.value)
                    return set(value)
    raise RuntimeError("collect_ignore not found in tests/conftest.py")


def _load_ledger_paths() -> set[str]:
    paths: set[str] = set()
    for line in LEDGER.read_text().splitlines():
        match = re.match(r"\| `([^`]+)` \|", line)
        if match:
            paths.add(match.group(1))
    return paths


def main() -> int:
    ignored = _load_collect_ignore()
    documented = _load_ledger_paths()
    missing_in_doc = sorted(ignored - documented)
    stale_in_doc = sorted(documented - ignored)
    if missing_in_doc or stale_in_doc:
        if missing_in_doc:
            print("Ignored but not documented:")
            for item in missing_in_doc:
                print(f"  - {item}")
        if stale_in_doc:
            print("Documented but not ignored:")
            for item in stale_in_doc:
                print(f"  - {item}")
        return 1
    print(f"CI tail cleanup ledger is in sync ({len(ignored)} entries).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
