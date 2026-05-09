"""
简化的Pytest配置
"""

import os

# CI runs the full historical `tests/` tree. Keep this hook explicit so
# `scripts/check_ci_tail_cleanup_sync.py` can detect any future temporary
# isolations and require matching ledger entries.
if os.getenv("GITHUB_ACTIONS") == "true":
    collect_ignore = []


import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
