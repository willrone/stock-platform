"""
简化的Pytest配置
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)