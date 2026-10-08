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


def _called_helper_key(namespace: dict[str, object]) -> str | None:
    kernel = SimpleNamespace(
        _helion_cute_source_hash="unchanged generated source",
        __wrapped__=SimpleNamespace(__globals__=namespace),
    )
    return launcher._cute_disk_cache_key(kernel, (), (128, 1, 1), (), None, None)


def test_called_helper_edit_changes_kernel_key(common_sources: Path) -> None:
    # The generated text names the helper; only its module source changes.
    pytest.importorskip("cutlass")  # reduce_helpers imports cutlass
    from helion._compiler.cute import reduce_helpers

    relative = "_compiler/cute/reduce_helpers.py"
    (common_sources / relative).write_text("def helper():\n    return 1\n")
    # reduce_helpers' own helper imports are part of its closure.
    (common_sources / "_compiler/cute/cluster_helpers.py").write_text("# cluster\n")
    namespace = {"_cute_grouped_reduce_warp": reduce_helpers._cute_grouped_reduce_warp}
    before = _called_helper_key(namespace)
    assert before is not None
    assert _called_helper_key({}) != before
    (common_sources / relative).write_text("def helper():\n    return 2\n")
    after = _called_helper_key(namespace)
    assert after is not None and after != before
    (common_sources / relative).unlink()
    assert _called_helper_key(namespace) is None


def test_cluster_reductions_key_on_cluster_helpers() -> None:
    # The cluster reductions store to peer CTAs through cluster_helpers; the
    # helper closure follows module globals, so it sees that module only
    # through reduce_helpers' module-level imports.
    pytest.importorskip("cutlass")
    from helion._compiler.cute import reduce_helpers

    for helper in (
        reduce_helpers._cute_grouped_reduce_cluster,
        reduce_helpers._cute_grouped_reduce_cluster_online_pair,
    ):
        sources = source_dependencies.referenced_helper_sources({"helper": helper})
        assert sources is not None
        assert "_compiler/cute/cluster_helpers.py" in dict(sources)
