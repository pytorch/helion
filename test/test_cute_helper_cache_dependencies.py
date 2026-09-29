"""External compiled helper edits must invalidate the correct disk artifacts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from helion.runtime.cute import launcher
from helion.runtime.cute import source_dependencies

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files = set(source_dependencies._COMMON_DEPENDENCIES)
    for paths in source_dependencies._WRAPPER_DEPENDENCIES.values():
        files.update(paths)
    for relative in files:
        filename = tmp_path / relative
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(f"# {relative}\ndef device_helper():\n    return 1\n")
    monkeypatch.setattr(source_dependencies, "_PACKAGE_ROOT", tmp_path)
    return tmp_path


def _kernel(kind: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        _helion_cute_source_hash="unchanged generated wrapper and plan",
        _helion_cute_wrapper_plans=[] if kind is None else [{"kind": kind}],
    )


def _key(kernel: SimpleNamespace) -> str | None:
    plans = tuple(repr(plan) for plan in kernel._helion_cute_wrapper_plans)
    return launcher._cute_disk_cache_key(
        kernel, (), (192, 1, 1), plans, None, "--enable-tvm-ffi", 148
    )


@pytest.mark.parametrize(
    "kind,edited,unrelated",
    [
        (
            "chunk_recurrence_sm100",
            "_compiler/cute/chunk_recurrence_sm100.py",
            "chunk_recurrence_warp_dv4",
        ),
        (
            "chunk_recurrence_warp_dv4",
            "_compiler/cute/kda_device_primitives.py",
            "chunk_recurrence_sm100",
        ),
    ],
)
def test_transitive_helper_changes_only_its_family(
    source_tree: Path, kind: str, edited: str, unrelated: str
) -> None:
    affected, unaffected = _kernel(kind), _kernel(unrelated)
    before = _key(affected), _key(unaffected)
    (source_tree / edited).write_text("def helper():\n    return 9\n")
    assert _key(affected) != before[0]
    assert _key(unaffected) == before[1]


def test_wrapper_generator_edit_invalidates_all_kinds(source_tree: Path) -> None:
    kernels = [_kernel(None), *map(_kernel, source_dependencies._WRAPPER_DEPENDENCIES)]
    before = list(map(_key, kernels))
    (source_tree / "runtime/cute/launcher.py").write_text("# new wrapper generation\n")
    assert all(old != _key(kernel) for old, kernel in zip(before, kernels, strict=True))
