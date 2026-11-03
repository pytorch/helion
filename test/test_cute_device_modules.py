"""Invariants of the CuTe device-helper modules (the Python traced into kernels).

Two rules every module the generated kernels import has to keep:

1. A module defining a ``@cute.struct`` must not carry ``from __future__ import
   annotations``: the DSL resolves the struct's field annotations at class
   creation, and stringified ``MemRange`` / ``Align`` annotations fail with a
   runtime ``TypeError``.  Only real decorators count (``_mma_dag_runtime``,
   ``_pdl_spread_runtime``, ``cute_flash`` and ``device/__init__.py`` mention
   the decorator in comments or docstrings), so the check is AST-based.
2. Every device module imports on a CPU-only process with CUDA hidden, and
   the shared-storage struct factories build their structs there: generated
   kernels and the CPU render tests import them before any CUDA context
   exists.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from helion._compiler.cute import backend as cute_backend

CUTE_DIR = Path(cute_backend.__file__).resolve().parent
REPO = CUTE_DIR.parents[2]

# The device helpers the emitters import by module path (the generated import
# strings in ``CuteBackend.library_imports`` and the flash preamble are read
# below), plus the traced-helper modules named by file: every ``*_runtime.py``
# in the package and the explicitly listed helpers.  A listed file that is
# missing fails the test (a rename must update the list, not drop coverage).
DEVICE_MODULE_EXTRAS = (
    "_flash_gemm_ptx.py",
    "chunk_recurrence_sm100.py",
    "chunk_recurrence_dv4_sm100.py",
    "gdn_recurrence_sm100.py",
    "reduce_helpers.py",
    "split_k_exchange.py",
    "tensor_copy_helpers.py",
    "thread_budget.py",
    "vec_utils.py",
)


def _device_module_files() -> list[str]:
    missing = [name for name in DEVICE_MODULE_EXTRAS if not (CUTE_DIR / name).exists()]
    assert not missing, f"device modules moved or deleted: {missing}"
    runtime = sorted(path.name for path in CUTE_DIR.glob("*_runtime.py"))
    assert len(runtime) >= 15, runtime
    return [*runtime, *DEVICE_MODULE_EXTRAS]


_IMPORT_LINE = re.compile(r"^from (helion\.[\w.]+) import ", re.MULTILINE)


def _is_cute_struct(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "struct"
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "cute"
    )


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _struct_modules() -> dict[Path, tuple[list[str], bool]]:
    """Struct-defining modules -> (struct names, has ``from __future__ import annotations``)."""
    structs: dict[Path, tuple[list[str], bool]] = {}
    for path in sorted(CUTE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(_is_cute_struct(d) for d in node.decorator_list)
        ]
        if names:
            structs[path] = (names, _has_future_annotations(tree))
    return structs


def test_struct_modules_have_no_future_annotations() -> None:
    structs = _struct_modules()
    # The two known struct modules; a new one is fine as long as it follows
    # the rule (and gets the ``I002`` exemption in pyproject.toml).
    assert {path.name for path in structs} >= {
        "_flash_runtime.py",
        "_flash_alt_runtime.py",
    }
    offenders = {
        path.name: names for path, (names, future) in structs.items() if future
    }
    assert offenders == {}, (
        "modules defining @cute.struct must not stringify their annotations"
    )
    pyproject = (REPO / "pyproject.toml").read_text()
    for path in structs:
        relative = path.relative_to(REPO).as_posix()
        # The file's per-file-ignores list must carry I002, whatever else it lists.
        entry = re.search(
            rf'^"{re.escape(relative)}" = \[([^\]]*)\]', pyproject, re.MULTILINE
        )
        assert entry is not None and '"I002"' in entry.group(1), (
            f"{relative} needs the I002 (future import) exemption in pyproject.toml"
        )


def _generated_import_modules() -> set[str]:
    """Every module the emitted kernel source imports helpers from."""
    from helion._compiler.cute import cute_flash

    modules: set[str] = set()
    backend = cute_backend.CuteBackend()
    for statement in backend.library_imports.values():
        modules.update(_IMPORT_LINE.findall(statement))
    modules.update(_IMPORT_LINE.findall(cute_flash._FLASH_PREAMBLE_IMPORTS))
    return modules


def test_device_modules_import_on_cpu_with_cuda_hidden() -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    modules = sorted(
        {f"helion._compiler.cute.{Path(name).stem}" for name in _device_module_files()}
        | _generated_import_modules()
    )
    script = f"""
import importlib, sys
import torch
assert not torch.cuda.is_available(), "CUDA must be hidden"
for name in {modules!r}:
    importlib.import_module(name)
from helion._compiler.cute._flash_runtime import flash_fa4_shared_storage
from helion._compiler.cute._flash_alt_runtime import flash_fa4_alt_shared_storage
for head_dim in (64, 128):
    flash_fa4_shared_storage(head_dim, 2)
    flash_fa4_shared_storage(head_dim, 3, epi_tma=True, separate_kv=True)
    flash_fa4_alt_shared_storage(head_dim, 3, 2)
print("ok", len({modules!r}))
"""
    pythonpath = os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"ok {len(modules)}"
