#!/usr/bin/env python3
"""Validate backend split requirement strategy guardrails."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

REQUIRED_LINES = {
    ROOT
    / "requirements.txt": [
        "numpy>=1.26.0,<2.0",
        "black==23.11.0",
        "isort==5.12.0",
        "flake8==6.1.0",
        "# safety intentionally not pinned here: install it separately in CI/security jobs",
    ],
    ROOT
    / "requirements-quality.txt": [
        "numpy>=1.26.0,<2.0",
        "black==23.11.0",
        "isort==5.12.0",
        "flake8==6.1.0",
        "bandit==1.7.5",
    ],
    ROOT
    / "requirements-test.txt": [
        "numpy>=1.26.0,<2.0",
        "pytest==7.4.3",
        "pytest-asyncio==0.21.1",
    ],
    ROOT
    / "requirements-security.txt": [
        "bandit==1.7.5",
        "safety>=3.7.0",
    ],
    ROOT
    / "requirements-ml.txt": [
        "git+https://github.com/microsoft/qlib.git",
        "torch>=2.1.0",
        "xgboost==2.0.2",
    ],
    REPO_ROOT
    / "docs"
    / "quality"
    / "backend-dependency-strategy.md": [
        "black==23.11.0",
        "numpy>=1.26.0,<2.0",
        "requirements-security.txt",
    ],
}

FORBIDDEN_LINES = {
    ROOT / "requirements-quality.txt": ["safety>=", "safety=="],
    ROOT
    / "requirements-test.txt": [
        "safety>=",
        "safety==",
        "git+https://github.com/microsoft/qlib.git",
    ],
}


def main() -> int:
    errors: list[str] = []
    for path, lines in REQUIRED_LINES.items():
        text = path.read_text()
        for line in lines:
            if line not in text:
                errors.append(
                    f"{path.relative_to(REPO_ROOT)} missing required line: {line}"
                )
    for path, fragments in FORBIDDEN_LINES.items():
        text = path.read_text()
        for fragment in fragments:
            if fragment in text:
                errors.append(
                    f"{path.relative_to(REPO_ROOT)} contains forbidden fragment: {fragment}"
                )
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Backend requirement strategy guardrails are satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
