"""Ordinary device helper edits invalidate persisted CuTe machine code."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from helion.runtime.cute import launcher
from helion.runtime.cute import source_dependencies

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def common_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for relative in source_dependencies._COMMON_DEPENDENCIES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\ndef helper():\n    return 1\n")
    monkeypatch.setattr(source_dependencies, "_PACKAGE_ROOT", tmp_path)
    return tmp_path


def _ordinary_key() -> str | None:
    kernel = SimpleNamespace(_helion_cute_source_hash="unchanged generated import")
    return launcher._cute_disk_cache_key(kernel, (), (128, 1, 1), (), None, None)


@pytest.mark.parametrize("relative", source_dependencies._COMMON_DEPENDENCIES)
def test_helper_edit_changes_ordinary_kernel_key(
    common_sources: Path, relative: str
) -> None:
    before = _ordinary_key()
    assert before is not None
    (common_sources / relative).write_text("def helper():\n    return 2\n")
    after = _ordinary_key()
    assert after is not None and after != before


@pytest.mark.parametrize("relative", source_dependencies._COMMON_DEPENDENCIES)
def test_missing_common_helper_disables_persisted_reuse(
    common_sources: Path, relative: str
) -> None:
    assert _ordinary_key() is not None
    (common_sources / relative).unlink()
    assert _ordinary_key() is None


def test_unrelated_helper_does_not_change_ordinary_kernel_key(
    common_sources: Path,
) -> None:
    before = _ordinary_key()
    assert before is not None
    (common_sources / "unrelated.py").write_text("def helper():\n    return 7\n")
    assert _ordinary_key() == before
