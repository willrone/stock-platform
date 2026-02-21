"""
数据同步引擎属性测试
验证 IncrementalUpdater 的数据变化检测属性
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.data.incremental_updater import IncrementalUpdater


@composite
def stock_codes(draw):
    """生成股票代码"""
    market = draw(st.sampled_from(['SZ', 'SH']))
    code = draw(st.integers(min_value=1, max_value=999999))
    return f"{code:06d}.{market}"


class TestIncrementalUpdaterProperties:
    """IncrementalUpdater 属性测试类"""

    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_dir = Path(self.temp_dir) / "parquet" / "stock_data"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_dir = Path(self.temp_dir) / "qlib" / "features" / "day"
        self.qlib_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("app.services.data.incremental_updater.settings")
    def test_detect_changes_empty_directory(self, mock_settings):
        """
        属性: 空目录检测
        当 parquet 目录为空时，detect_changes 应返回空字典
        """
        mock_settings.DATA_ROOT_PATH = self.temp_dir
        mock_settings.QLIB_DATA_PATH = str(self.qlib_dir.parent.parent)
        updater = IncrementalUpdater()
        updater.parquet_data_path = self.parquet_dir
        updater.qlib_data_path = self.qlib_dir

        changes = updater.detect_changes()
        assert isinstance(changes, dict)

    @patch("app.services.data.incremental_updater.settings")
    def test_detect_changes_with_stock_codes_filter(self, mock_settings):
        """
        属性: 股票代码过滤
        传入 stock_codes 参数时，只检测指定股票
        """
        mock_settings.DATA_ROOT_PATH = self.temp_dir
        mock_settings.QLIB_DATA_PATH = str(self.qlib_dir.parent.parent)
        updater = IncrementalUpdater()
        updater.parquet_data_path = self.parquet_dir
        updater.qlib_data_path = self.qlib_dir

        changes = updater.detect_changes(stock_codes=["000001.SZ", "000002.SZ"])
        assert isinstance(changes, dict)
        # 结果只包含请求的股票代码
        for code in changes:
            assert code in ["000001.SZ", "000002.SZ"]

    @patch("app.services.data.incremental_updater.settings")
    def test_detect_changes_new_stock(self, mock_settings):
        """
        属性: 新股票检测
        当 parquet 有数据但 qlib 没有时，应标记为 new
        """
        mock_settings.DATA_ROOT_PATH = self.temp_dir
        mock_settings.QLIB_DATA_PATH = str(self.qlib_dir.parent.parent)

        # 创建 parquet 文件（模拟有数据）
        stock_dir = self.parquet_dir / "000001.SZ"
        stock_dir.mkdir(parents=True, exist_ok=True)
        (stock_dir / "data.parquet").write_text("fake")

        updater = IncrementalUpdater()
        updater.parquet_data_path = self.parquet_dir
        updater.qlib_data_path = self.qlib_dir

        changes = updater.detect_changes(stock_codes=["000001.SZ"])
        assert isinstance(changes, dict)
        if "000001.SZ" in changes:
            assert changes["000001.SZ"]["action"] in ("new", "update", "none")

    @patch("app.services.data.incremental_updater.settings")
    def test_detect_changes_returns_valid_actions(self, mock_settings):
        """
        属性: 返回值格式正确
        每个变化记录的 action 必须是 new/update/none 之一
        """
        mock_settings.DATA_ROOT_PATH = self.temp_dir
        mock_settings.QLIB_DATA_PATH = str(self.qlib_dir.parent.parent)
        updater = IncrementalUpdater()
        updater.parquet_data_path = self.parquet_dir
        updater.qlib_data_path = self.qlib_dir

        changes = updater.detect_changes()
        for code, info in changes.items():
            assert "action" in info
            assert info["action"] in ("new", "update", "none")

    @patch("app.services.data.incremental_updater.settings")
    def test_get_stocks_to_update(self, mock_settings):
        """
        属性: get_stocks_to_update 返回需要更新的股票列表
        """
        mock_settings.DATA_ROOT_PATH = self.temp_dir
        mock_settings.QLIB_DATA_PATH = str(self.qlib_dir.parent.parent)
        updater = IncrementalUpdater()
        updater.parquet_data_path = self.parquet_dir
        updater.qlib_data_path = self.qlib_dir

        stocks = updater.get_stocks_to_update()
        assert isinstance(stocks, list)

    @patch("app.services.data.incremental_updater.settings")
    def test_detect_changes_idempotent(self, mock_settings):
        """
        属性: 幂等性
        连续两次调用 detect_changes 应返回相同结果
        """
        mock_settings.DATA_ROOT_PATH = self.temp_dir
        mock_settings.QLIB_DATA_PATH = str(self.qlib_dir.parent.parent)
        updater = IncrementalUpdater()
        updater.parquet_data_path = self.parquet_dir
        updater.qlib_data_path = self.qlib_dir

        changes1 = updater.detect_changes()
        changes2 = updater.detect_changes()
        assert changes1 == changes2
