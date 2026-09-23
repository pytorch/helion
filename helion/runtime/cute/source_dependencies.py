"""Sources compiled behind CuTe wrapper calls but absent from generated code."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# Ordinary generated kernels import these helpers without a wrapper plan.
# Include their implementations and the L2 compatibility helper in every key;
# a generated import statement alone cannot identify the compiled device code.
_COMMON_DEPENDENCIES = (
    "runtime/cute/launcher.py",
    "_compiler/cute/vec_utils.py",
    "_compiler/cute/l2_policy.py",
    "_compiler/cute/cutedsl_compat.py",
)

# Keep transitive Helion device helpers with each wrapper or ordinary kernel
# that compiles them.
# CUTLASS itself is covered by the version in the launcher's cache key. These
# are source paths relative to the package, making the digest checkout-neutral.
_WRAPPER_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "split_k_cluster": ("_compiler/cute/cluster_helpers.py",),
    "tcgen05_grouped_rna": (
        "_compiler/cute/_flash_runtime.py",
        "_compiler/cute/_mlir_compat.py",
        "_compiler/cute/tcgen05_flattened_prefix.py",
        "_compiler/cute/tcgen05_grouped_prefix.py",
        "_compiler/cute/tcgen05_operand_pipeline.py",
        "_compiler/cute/tcgen05_operand_transform.py",
        "_compiler/cute/tcgen05_storage.py",
    ),
    "tcgen05_grouped_tma_rn": (
        "_compiler/cute/tcgen05_flattened_prefix.py",
        "_compiler/cute/tcgen05_grouped_prefix.py",
        "_compiler/cute/_flash_runtime.py",
        "_compiler/cute/_mlir_compat.py",
    ),
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
    "block_scaled_mma": (
        "_compiler/cute/block_scaled_config.py",
        "_compiler/cute/block_scaled_prepare.py",
        "_compiler/cute/block_scaled_runtime.py",
    ),
    "gathered_mma_tma": (
        "_compiler/cute/_mlir_compat.py",
        "_compiler/cute/gathered_mma_runtime.py",
    ),
    "chunk_recurrence_sm100": ("_compiler/cute/chunk_recurrence_sm100.py",),
    "chunk_recurrence_warp_dv4": (
        "_compiler/cute/chunk_recurrence_dv4_sm100.py",
        "_compiler/cute/kda_device_primitives.py",
    ),
}


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
    for kind in kinds:
        paths.update(_WRAPPER_DEPENDENCIES.get(kind, ()))
    result = []
    for relative in sorted(paths):
        try:
            source = (_PACKAGE_ROOT / relative).read_bytes()
        except OSError:
            return None
        result.append((relative, hashlib.sha256(source).hexdigest()))
    return tuple(result)


def set_helper_source_hash(kernel: object, kind: str) -> None:
    """Give an external device kernel the same transitive cache proof as wrappers.

    Called once when its module is loaded, not on repeated launches. A missing
    dependency leaves the hash None and therefore disables on-disk reuse.
    """
    assert kind in _WRAPPER_DEPENDENCIES
    dependencies = wrapper_source_dependencies((kind,))
    digest = (
        hashlib.sha256(repr(dependencies).encode()).hexdigest()
        if dependencies is not None
        else None
    )
    kernel._helion_cute_source_hash = digest  # type: ignore[attr-defined]
