"""Sources compiled behind CuTe wrapper calls but absent from generated code."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# Ordinary generated kernels import these helpers without a wrapper plan.
# Include their implementations and transitive cluster support in every key;
# a generated import statement alone cannot identify the compiled device code.
# The host wrapper emitters are keyed whole as well: they write the ``@cute.jit``
# wrapper text (and ``flash_flex`` the tile counts baked into it), which the
# key does not otherwise see; so are the two tables whose results the
# compiled wrapper carries without naming their modules: ``occupancy`` (the
# persistent flash grid clamp is baked into the wrapper text as a literal)
# and ``dtypes`` (the ``make_ptr`` element type of every tensor argument).
_COMMON_DEPENDENCIES = (
    "runtime/cute/launcher.py",
    "runtime/cute/host_wrappers/__init__.py",
    "runtime/cute/host_wrappers/block_scaled.py",
    "runtime/cute/host_wrappers/context.py",
    "runtime/cute/host_wrappers/flash.py",
    "runtime/cute/host_wrappers/gathered.py",
    "runtime/cute/host_wrappers/grouped_static.py",
    "runtime/cute/host_wrappers/mma_dag.py",
    "runtime/cute/host_wrappers/recurrences.py",
    "runtime/cute/host_wrappers/tcgen05.py",
    "runtime/cute/flash_flex.py",
    "runtime/cute/occupancy.py",
    "runtime/cute/plan_facts.py",
    "runtime/cute/register_tensor.py",
    "_compiler/cute/reduce_helpers.py",
    "_compiler/cute/cluster_helpers.py",
    "_compiler/cute/device/sync.py",
    "_compiler/cute/vec_utils.py",
    "_compiler/cute/l2_policy.py",
    "_compiler/cute/cutedsl_compat.py",
    "_compiler/cute/dtypes.py",
)

# Helper kinds that are not wrapper plans: the external device kernels keyed
# through ``set_helper_source_hash`` and the ordinary kernels'
# ``_helion_cute_helper_kinds``.  The wrapper-plan kinds declare their helper
# sources on their ``HostWrapperEmitter`` (``helion.runtime.cute.host_wrappers``).
_HELPER_KIND_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "split_k_cluster": ("_compiler/cute/cluster_helpers.py",),
    "resident_reduction": ("_compiler/cute/resident_reduction_runtime.py",),
    "single_host_sum": (
        "runtime/cute/single_sum.py",
        "runtime/cute/paired_sum.py",
        "_compiler/cute/single_sum_runtime.py",
        "_compiler/cute/resident_reduction_runtime.py",
    ),
    "paired_host_sum": (
        "runtime/cute/paired_sum.py",
        "_compiler/cute/paired_sum_runtime.py",
        "_compiler/cute/resident_reduction_runtime.py",
    ),
    "full_host_sum": (
        "runtime/cute/full_sum.py",
        "runtime/cute/paired_sum.py",
        "_compiler/cute/full_sum_runtime.py",
        "_compiler/cute/resident_reduction_runtime.py",
    ),
}


def wrapper_dependencies() -> dict[str, tuple[str, ...]]:
    """Package-relative helper sources per wrapper-plan or helper kind.

    The wrapper-plan kinds come from the host wrapper emitter registry; the
    registry is imported lazily because the emitters import the compiler's
    plan modules and this module is imported by device runtime modules.
    """
    from .host_wrappers import HOST_WRAPPER_EMITTERS

    table = {
        kind: emitter.source_dependencies
        for kind, emitter in HOST_WRAPPER_EMITTERS.items()
        if emitter.source_dependencies
    }
    table.update(_HELPER_KIND_DEPENDENCIES)
    return table


def wrapper_source_dependencies(
    kinds: Iterable[str],
) -> tuple[tuple[str, str], ...] | None:
    """Fingerprint sources once while constructing a compiled launcher.

    Generated source text does not contain external helper implementations, so
    source-only kernel hashes can reload obsolete machine code after a helper
    fix. Include common device helpers and the wrapper families it uses.
    Missing source disables disk reuse; compilation still proceeds normally.
    The in-memory launcher cache avoids reading files on repeated launches.
    """
    paths = set(_COMMON_DEPENDENCIES)
    table = wrapper_dependencies()
    for kind in kinds:
        paths.update(table.get(kind, ()))
    return _fingerprints(sorted(paths))


# The launcher's SASS checks (``helion.runtime.cute.sass_check``) admit a
# kernel that declares ``_helion_cute_sass_checks`` before it is persisted;
# a disk-cache hit is trusted on its writer's verification, so the key of
# such a kernel covers the checker's source: a stricter rule re-verifies
# (recompiles) the entries persisted under the old one.
_SASS_CHECK_DEPENDENCIES = ("runtime/cute/sass_check.py",)


def sass_check_source_dependencies() -> tuple[tuple[str, str], ...] | None:
    """Fingerprint the SASS checker's source for a kernel that declares
    checks; None (disabling disk reuse) when it cannot be read."""
    return _fingerprints(_SASS_CHECK_DEPENDENCIES)


def _fingerprints(
    relative_paths: Iterable[str],
) -> tuple[tuple[str, str], ...] | None:
    result = []
    for relative in relative_paths:
        try:
            source = (_PACKAGE_ROOT / relative).read_bytes()
        except OSError:
            return None
        result.append((relative, hashlib.sha256(source).hexdigest()))
    return tuple(result)


# Modules whose functions a generated kernel can call as device code.  The
# launcher is host code, keyed whole through ``_COMMON_DEPENDENCIES``; its own
# (host-side) imports are not followed.
_DEVICE_HELPER_PACKAGES = ("helion._compiler.cute.", "helion.runtime.cute.")
_HOST_LAUNCHER_MODULE = "helion.runtime.cute.launcher"


def _defining_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    module = getattr(value, "__module__", None)
    return module if isinstance(module, str) else None


def referenced_helper_sources(
    namespace: Mapping[str, object],
) -> tuple[tuple[str, str], ...] | None:
    """Fingerprint the Helion device-helper modules a generated kernel calls.

    The generated source names helpers (``_cute_grouped_reduce_warp``, ...)
    but not their implementations, so a helper fix must change the key of
    every kernel that calls it.  ``namespace`` is the kernel's globals; each
    referenced helper module's own helper imports are followed as well.
    Missing source disables disk reuse, as in ``wrapper_source_dependencies``.
    """
    pending = [_defining_module(value) for value in namespace.values()]
    seen: set[str] = set()
    result = []
    while pending:
        name = pending.pop()
        if (
            name is None
            or name in seen
            or name == _HOST_LAUNCHER_MODULE
            or not name.startswith(_DEVICE_HELPER_PACKAGES)
        ):
            continue
        seen.add(name)
        module = sys.modules.get(name)
        if module is None:
            return None
        relative = "/".join(name.split(".")[1:])
        relative += "/__init__.py" if hasattr(module, "__path__") else ".py"
        try:
            source = (_PACKAGE_ROOT / relative).read_bytes()
        except OSError:
            return None
        result.append((relative, hashlib.sha256(source).hexdigest()))
        pending.extend(_defining_module(value) for value in vars(module).values())
    return tuple(sorted(result))


def set_helper_source_hash(kernel: object, kind: str) -> None:
    """Give an external device kernel the same transitive cache proof as wrappers.

    Called once when its module is loaded, not on repeated launches. A missing
    dependency leaves the hash None and therefore disables on-disk reuse.
    """
    assert kind in _HELPER_KIND_DEPENDENCIES
    dependencies = wrapper_source_dependencies((kind,))
    digest = (
        hashlib.sha256(repr(dependencies).encode()).hexdigest()
        if dependencies is not None
        else None
    )
    kernel._helion_cute_source_hash = digest  # type: ignore[attr-defined]
