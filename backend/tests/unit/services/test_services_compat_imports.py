"""app.services 兼容导入契约测试。"""

from types import SimpleNamespace

import pytest

import app.services as services


def test_getattr_loads_compatible_service_with_deprecation_warning(
    monkeypatch,
) -> None:
    """兼容导出应动态加载目标模块并保留弃用告警。"""
    sentinel = object()
    imported_modules: list[tuple[str, str]] = []

    def fake_import_module(module_name: str, package: str) -> SimpleNamespace:
        imported_modules.append((module_name, package))
        return SimpleNamespace(SimpleDataService=sentinel)

    monkeypatch.setattr(services, "import_module", fake_import_module)

    with pytest.deprecated_call(match=r"app\.services\.data\.SimpleDataService"):
        exported = services.SimpleDataService

    assert exported is sentinel
    assert imported_modules == [(".data", "app.services")]


def test_getattr_raises_attribute_error_for_unknown_service() -> None:
    """未知兼容导出名称应抛出标准 AttributeError。"""
    with pytest.raises(AttributeError, match="UnknownService"):
        _ = services.UnknownService
