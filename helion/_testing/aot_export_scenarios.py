"""Scenarios for the standalone export of AOT compile mode, shared by the
tests of its export paths.

Compile mode writes one standalone file per kernel function and adds a
variant for each call it observes. A backend may hold the configs a kernel
selects without a search to some of its ``autotune_config_overrides``
(``BackendSearchSurface.overrides_held_without_search``), and the export is
generated from such configs, so what a kernel holds is part of the key of
every entry the AOT cache keeps an export by
(``AOTAutotuneCache._maybe_run_compile``,
``AOTAutotuneCache._compile_current_static_shape``). Each scenario walks
kernels through exports and states, after every step, what each file holds:
the overrides its variants were generated under and the calls it dispatches.
``run_export_scenario`` checks that against the files on disk, so a file that
mixes the variants of two sets of held overrides, lacks a variant its kernel
exported or counts a call as exported on what other overrides generated
fails the step that produced it.

What an export under other held overrides does to a file depends on the
path. Without a structural policy a file is written from the entry of the
kernel that exports: the file then holds that entry's variants alone. A
static entry keeps its variants for its next turn; a dynamic export keeps
one table per file, so the table of an entry that returns is generated
again. Under a structural policy a file keeps the entry that wrote it first,
and an export from another entry is refused
(``AOTAutotuneCache._may_publish_structural_variant``). A test module
supplies an ``ExportDriver`` for each export path it covers: static and
dynamic exports of stub kernels in ``test_aot_autotuning.py``, the
structural-policy exports of CuTe kernels in
``test_cute_structural_aot_export.py``.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import logging
import os
import sys
from typing import TYPE_CHECKING
from typing import Protocol
from unittest.mock import patch

import pytest

from ..runtime.structural_policy import StructuralPolicyError

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from pathlib import Path
    from types import ModuleType

# The kernels of a scenario, by what sets them apart from ``KERNEL``.
KERNEL = "kernel"
# Another kernel object over the same function: it writes the same file.
TWIN = "twin"
# The same function with its tuning data in another directory. Beside a
# source file that is the same file again.
OTHER_DIR = "other_dir"
# A function of the same name in another source file: a file of its own.
OTHER_SOURCE = "other_source"
# The same function under another structural policy: a file of its own.
# Only the structural-policy paths have it.
OTHER_POLICY = "other_policy"

# What a kernel's configs are held to: nothing, or one of two sets of
# overrides.
PLAIN: str | None = None
HELD_A = "a"
HELD_B = "b"

# The calls a scenario exports, by the size of the kernel's argument. For a
# call of ``INELIGIBLE`` the backend's hook leaves the config as it is,
# whatever the kernel holds (a shape without the property the override is
# about), so a static export generates the code of ``PLAIN`` for it. The
# sizes share a remainder modulo 8: a structural dynamic export guards its
# one variant on the alignment class of the argument.
SMALL = 37
LARGE = 45
INELIGIBLE = 53
SHAPES = (SMALL, LARGE, INELIGIBLE)


@dataclasses.dataclass(frozen=True)
class Step:
    """One export, and the state of every standalone file after it."""

    kernel: str
    held: str | None
    shape: int
    # Every file on disk, named by a kernel that writes it: the overrides
    # its variants were generated under and the calls it dispatches. Where
    # ``owned`` is given, this is the state of the paths on which a file is
    # written from the entry of the kernel that exports.
    files: Mapping[str, tuple[str | None, frozenset[int]]]
    # The same on the paths on which a file keeps the entry that wrote it
    # first, where that differs.
    owned: Mapping[str, tuple[str | None, frozenset[int]]] | None = None
    # On those paths the owner of the file refuses this export.
    refused: bool = False
    # Another kernel's export of ``(kernel, held, shape)``, run while this
    # step's code is being generated.
    meanwhile: tuple[str, str | None, int] | None = None
    # Code generation fails.
    fails: bool = False


def _files(
    **content: tuple[str | None, tuple[int, ...]],
) -> dict[str, tuple[str | None, frozenset[int]]]:
    return {
        kernel: (held, frozenset(shapes)) for kernel, (held, shapes) in content.items()
    }


def _step(
    kernel: str,
    held: str | None,
    shape: int,
    /,
    *,
    owned: Mapping[str, tuple[str | None, tuple[int, ...]]] | None = None,
    refused: bool = False,
    meanwhile: tuple[str, str | None, int] | None = None,
    fails: bool = False,
    **content: tuple[str | None, tuple[int, ...]],
) -> Step:
    return Step(
        kernel,
        held,
        shape,
        _files(**content),
        owned=None if owned is None else _files(**owned),
        refused=refused,
        meanwhile=meanwhile,
        fails=fails,
    )


SCENARIOS: dict[str, tuple[Step, ...]] = {
    # One kernel: its calls accumulate, and a repeated call changes nothing.
    "accumulate": (
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(KERNEL, PLAIN, LARGE, kernel=(PLAIN, (SMALL, LARGE))),
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL, LARGE))),
    ),
    # What a kernel holds is read off its overrides alone: a shape the hook
    # leaves alone shares the entry, and the file, of the shapes it changes.
    "eligible_and_ineligible_shapes": (
        _step(KERNEL, HELD_A, SMALL, kernel=(HELD_A, (SMALL,))),
        _step(KERNEL, HELD_A, INELIGIBLE, kernel=(HELD_A, (SMALL, INELIGIBLE))),
        _step(KERNEL, HELD_A, LARGE, kernel=(HELD_A, (SMALL, LARGE, INELIGIBLE))),
        _step(KERNEL, HELD_A, INELIGIBLE, kernel=(HELD_A, (SMALL, LARGE, INELIGIBLE))),
    ),
    # ``Kernel.reset()`` and other overrides: the file is written from the
    # entry of the new overrides, without a variant of the earlier ones, and
    # none of those is taken for exported. The return to the first overrides
    # writes the file from their entry again: a static entry kept its
    # variants, and a dynamic table is generated again. Where a file keeps
    # its first entry, the other overrides are refused.
    "reset_under_other_overrides": (
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(KERNEL, PLAIN, LARGE, kernel=(PLAIN, (SMALL, LARGE))),
        _step(
            KERNEL,
            HELD_A,
            SMALL,
            kernel=(HELD_A, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
        _step(
            KERNEL,
            HELD_A,
            LARGE,
            kernel=(HELD_A, (SMALL, LARGE)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
        _step(KERNEL, PLAIN, LARGE, kernel=(PLAIN, (SMALL, LARGE))),
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL, LARGE))),
        _step(
            KERNEL,
            HELD_B,
            SMALL,
            kernel=(HELD_B, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
    ),
    # Two kernels over one function share its file. While they hold the same
    # overrides their calls accumulate in it.
    "two_kernels_same_overrides": (
        _step(KERNEL, HELD_A, SMALL, kernel=(HELD_A, (SMALL,))),
        _step(TWIN, HELD_A, LARGE, kernel=(HELD_A, (SMALL, LARGE))),
        _step(KERNEL, HELD_A, LARGE, kernel=(HELD_A, (SMALL, LARGE))),
    ),
    # While they hold different ones, each writes the file from its own
    # entry when its turn comes: the file holds the calls of the overrides
    # that exported last, every one of them. Where a file keeps its first
    # entry, the kernel that holds other overrides is refused each time.
    "two_kernels_alternate": (
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(
            TWIN,
            HELD_A,
            SMALL,
            kernel=(HELD_A, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL,))},
        ),
        _step(KERNEL, PLAIN, LARGE, kernel=(PLAIN, (SMALL, LARGE))),
        _step(
            TWIN,
            HELD_A,
            LARGE,
            kernel=(HELD_A, (SMALL, LARGE)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
        _step(
            TWIN,
            HELD_A,
            SMALL,
            kernel=(HELD_A, (SMALL, LARGE)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
        _step(
            KERNEL,
            HELD_B,
            SMALL,
            kernel=(HELD_B, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
        _step(
            TWIN,
            HELD_A,
            SMALL,
            kernel=(HELD_A, (SMALL, LARGE)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL, LARGE))},
        ),
    ),
    # A kernel under other overrides exports while this one generates its
    # code. A variant is published once its code is generated, as when the
    # two take turns: the file holds the calls of the overrides that
    # published last, and the other's are written again when it returns.
    # Where a file keeps its first entry, the later publisher is refused.
    "other_overrides_during_code_generation": (
        _step(TWIN, HELD_A, SMALL, kernel=(HELD_A, (SMALL,))),
        _step(
            KERNEL,
            PLAIN,
            LARGE,
            meanwhile=(TWIN, HELD_A, LARGE),
            kernel=(PLAIN, (LARGE,)),
            refused=True,
            owned={"kernel": (HELD_A, (SMALL, LARGE))},
        ),
        _step(TWIN, HELD_A, LARGE, kernel=(HELD_A, (SMALL, LARGE))),
        _step(
            KERNEL,
            PLAIN,
            LARGE,
            kernel=(PLAIN, (LARGE,)),
            refused=True,
            owned={"kernel": (HELD_A, (SMALL, LARGE))},
        ),
    ),
    # The same before any file exists: the other kernel writes the first
    # file, which this one then writes from its own entry, or is refused.
    "other_overrides_during_the_first_code_generation": (
        _step(
            KERNEL,
            PLAIN,
            SMALL,
            meanwhile=(TWIN, HELD_A, SMALL),
            kernel=(PLAIN, (SMALL,)),
            refused=True,
            owned={"kernel": (HELD_A, (SMALL,))},
        ),
        _step(
            KERNEL,
            PLAIN,
            SMALL,
            kernel=(PLAIN, (SMALL,)),
            refused=True,
            owned={"kernel": (HELD_A, (SMALL,))},
        ),
    ),
    # A kernel that holds the same overrides shares the entry: both calls
    # are in the file.
    "same_overrides_during_code_generation": (
        _step(
            KERNEL,
            HELD_A,
            SMALL,
            meanwhile=(TWIN, HELD_A, LARGE),
            kernel=(HELD_A, (SMALL, LARGE)),
        ),
    ),
    # The directory of the tuning data is no part of a file beside the
    # kernel's source: two kernels with the same data in two directories
    # write one file, like any two kernels over one function.
    "two_data_directories": (
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(
            OTHER_DIR,
            HELD_A,
            LARGE,
            kernel=(HELD_A, (LARGE,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL,))},
        ),
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(OTHER_DIR, PLAIN, LARGE, kernel=(PLAIN, (SMALL, LARGE))),
    ),
    # Functions of one name in two source files write a file each, and
    # neither export takes the other's calls for its own.
    "two_source_files": (
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(
            OTHER_SOURCE,
            PLAIN,
            SMALL,
            kernel=(PLAIN, (SMALL,)),
            other_source=(PLAIN, (SMALL,)),
        ),
        _step(
            OTHER_SOURCE,
            HELD_A,
            SMALL,
            kernel=(PLAIN, (SMALL,)),
            other_source=(HELD_A, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL,)), "other_source": (PLAIN, (SMALL,))},
        ),
        _step(
            KERNEL,
            HELD_A,
            LARGE,
            kernel=(HELD_A, (LARGE,)),
            other_source=(HELD_A, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL,)), "other_source": (PLAIN, (SMALL,))},
        ),
        _step(
            OTHER_SOURCE,
            HELD_A,
            SMALL,
            kernel=(HELD_A, (LARGE,)),
            other_source=(HELD_A, (SMALL,)),
            refused=True,
            owned={"kernel": (PLAIN, (SMALL,)), "other_source": (PLAIN, (SMALL,))},
        ),
    ),
    # One function under two structural policies writes a file each: what
    # the kernel of one policy holds refuses nothing in the other's file,
    # and each file refuses the overrides its own entry is not of. Only the
    # structural-policy paths have the second policy.
    "two_policies": (
        _step(KERNEL, HELD_A, SMALL, kernel=(HELD_A, (SMALL,))),
        _step(
            OTHER_POLICY,
            PLAIN,
            SMALL,
            kernel=(HELD_A, (SMALL,)),
            other_policy=(PLAIN, (SMALL,)),
        ),
        _step(
            KERNEL,
            HELD_A,
            LARGE,
            kernel=(HELD_A, (SMALL, LARGE)),
            other_policy=(PLAIN, (SMALL,)),
        ),
        _step(
            OTHER_POLICY,
            HELD_B,
            LARGE,
            refused=True,
            kernel=(HELD_A, (SMALL, LARGE)),
            other_policy=(PLAIN, (SMALL,)),
        ),
        _step(
            KERNEL,
            HELD_A,
            SMALL,
            kernel=(HELD_A, (SMALL, LARGE)),
            other_policy=(PLAIN, (SMALL,)),
        ),
    ),
    # A variant is published once its code is generated. Where that fails,
    # the export of a kernel under other overrides changed nothing: the file
    # stays the earlier kernel's, whose calls keep accumulating in it.
    "failed_code_generation": (
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL,))),
        _step(TWIN, HELD_A, LARGE, fails=True, kernel=(PLAIN, (SMALL,))),
        _step(KERNEL, PLAIN, LARGE, kernel=(PLAIN, (SMALL, LARGE))),
        _step(KERNEL, PLAIN, SMALL, kernel=(PLAIN, (SMALL, LARGE))),
    ),
}


class ExportDriver(Protocol):
    """One export path, with kernels a scenario can name."""

    # Whether the path writes a variant per call (static shapes). A dynamic
    # export is written once and serves every call.
    static: bool
    # Whether a file keeps the entry that wrote it first and refuses the
    # others (a structural policy). Otherwise a file is written from the
    # entry of the kernel that exports.
    owned: bool
    # The kernels of ``KERNEL`` ... ``OTHER_POLICY`` the path has.
    kernels: frozenset[str]
    # Whether failing code generation fails the export.
    codegen_failure_raises: bool

    def export(
        self,
        kernel: str,
        held: str | None,
        shape: int,
        *,
        during_codegen: Callable[[], None] | None = None,
        fail: bool = False,
    ) -> None:
        """Have ``kernel``, holding ``held``, export a call of ``shape``.
        ``during_codegen`` runs once the call's code is generated and before
        it is published; ``fail`` makes code generation raise."""
        ...

    def file(self, kernel: str) -> Path:
        """The standalone file ``kernel`` writes."""
        ...

    def files(self) -> set[Path]:
        """Every standalone file on disk."""
        ...

    def dispatched(self, path: Path) -> dict[int, str | None]:
        """The calls of ``SHAPES`` the file at ``path`` dispatches, each
        with the overrides its code was generated under."""
        ...


def scenario_names(driver_kernels: frozenset[str], *, failures: bool) -> list[str]:
    """The scenarios a path with these kernels can run."""
    return [
        name
        for name, steps in SCENARIOS.items()
        if all(
            kernel in driver_kernels
            for step in steps
            for kernel in (
                step.kernel,
                *step.files,
                *(step.meanwhile[:1] if step.meanwhile else ()),
            )
        )
        and (failures or not any(step.fails for step in steps))
    ]


def _expected(
    driver: ExportDriver, held: str | None, exported: frozenset[int]
) -> dict[int, str | None]:
    if not driver.static:
        # One export for every call, generated from the kernel's configs.
        return dict.fromkeys(SHAPES, held)
    return {shape: PLAIN if shape == INELIGIBLE else held for shape in exported}


def run_export_scenario(
    name: str, driver: ExportDriver, caplog: pytest.LogCaptureFixture
) -> None:
    """Walk ``SCENARIOS[name]`` through ``driver`` and check every file
    after each step, and that no export warns."""
    caplog.set_level(logging.WARNING, logger="helion.autotuner.aot_cache")
    for number, step in enumerate(SCENARIOS[name]):
        where = f"{name}, step {number}: {step}"
        during_codegen = None
        if step.meanwhile is not None:
            other = step.meanwhile

            def during_codegen(other: tuple[str, str | None, int] = other) -> None:
                driver.export(*other)

        if step.fails:
            with pytest.raises(RuntimeError, match="failed to compile"):
                driver.export(step.kernel, step.held, step.shape, fail=True)
        elif driver.owned and step.refused:
            with pytest.raises(StructuralPolicyError, match="already owns"):
                driver.export(
                    step.kernel, step.held, step.shape, during_codegen=during_codegen
                )
        else:
            driver.export(
                step.kernel, step.held, step.shape, during_codegen=during_codegen
            )

        files = step.files
        if driver.owned and step.owned is not None:
            files = step.owned
        expected = {
            driver.file(kernel): _expected(driver, held, exported)
            for kernel, (held, exported) in files.items()
        }
        written = driver.files()
        assert written == set(expected), (where, written)
        for path, variants in expected.items():
            dispatched = driver.dispatched(path)
            assert dispatched == variants, (where, path.name, dispatched)

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "helion.autotuner.aot_cache"
        ]
        # Nothing is logged: no export drops a variant on the way.
        assert not messages, (where, messages)


def import_after_rewrite(
    path: Path, rewrite: Callable[[], object]
) -> tuple[ModuleType, ModuleType]:
    """Import the standalone file at ``path`` as an application does, by
    its name from its directory, have ``rewrite`` replace the file, and
    import it again: the two modules.

    Python checks a file's bytecode against the size of the source and its
    timestamp in whole seconds. ``rewrite`` has to write a file of the same
    size, which then gets the timestamp of the first one: a rewrite within
    that second, which is imported from the bytecode of the file replaced
    unless its writer removed it. The first import writes bytecode whatever
    ``sys.dont_write_bytecode`` says; a test is skipped where it still
    leaves none.
    """

    def imported() -> ModuleType:
        sys.modules.pop(path.stem, None)
        sys.path.insert(0, str(path.parent))
        importlib.invalidate_caches()
        try:
            return importlib.import_module(path.stem)
        finally:
            sys.path.remove(str(path.parent))
            sys.modules.pop(path.stem, None)

    written = path.stat()
    with patch.object(sys, "dont_write_bytecode", False):
        before = imported()
    if not os.path.exists(importlib.util.cache_from_source(path)):
        pytest.skip(f"importing {path.name} left no bytecode")
    rewrite()
    assert path.stat().st_size == written.st_size
    os.utime(path, ns=(written.st_atime_ns, written.st_mtime_ns))
    return before, imported()
