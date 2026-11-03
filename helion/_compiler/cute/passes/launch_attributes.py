"""Module-level statements after a CuTe kernel definition, and the launch-block
projection (``CuteBackend.finalize_kernel_def``).

The statements set the attributes ``helion.runtime.cute.launcher`` reads off
the kernel function before its first launch: the device-helper kinds whose
imports are absent from the generated source, the thread-block cluster
shape, programmatic dependent launch (``cute_pdl``), the SASS checks and
ptxas's ``minnctapersm``.  ``project_launch_block`` refreshes the host launch
call's ``block=`` once a late device lowering fixed the CTA shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...ast_extension import statement_from_string
from ...source_location import SyntheticLocation
from ..thread_block_projection import update_launch_block
from .pdl import pdl_applies

if TYPE_CHECKING:
    import ast

    from ...device_function import DeviceFunction


def module_statements(fn: DeviceFunction) -> list[ast.stmt]:
    """The statements that follow ``fn``'s kernel definition in the module."""
    result: list[ast.stmt] = []
    if fn.cute_state.lanes.resident_reduction_layouts:
        # These imported device helpers are absent from generated source.
        # Persist their dependency with this kernel through source reload.
        result.append(
            statement_from_string(
                f"{fn.name}._helion_cute_helper_kinds = ('resident_reduction',)"
            )
        )
    simt_cluster_n = max(
        fn.cute_state.lanes.simt_cluster_n, fn.cute_state.lanes.grid_cluster_n
    )
    if simt_cluster_n > 1:
        # The CuTe launcher reads this attribute to launch the kernel
        # with a (1, cluster_n, 1) thread-block cluster.
        result.append(
            statement_from_string(
                f"{fn.name}._helion_cute_cluster_shape = (1, {simt_cluster_n}, 1)"
            )
        )
    if pdl_applies(fn):
        # The CuTe launcher reads this attribute to launch the kernel
        # with programmatic dependent launch (``use_pdl``).
        with SyntheticLocation():
            result.append(
                statement_from_string(f"{fn.name}._helion_cute_use_pdl = True")
            )
    sass_checks = fn.cute_state.lanes.sass_checks
    if sass_checks:
        # The CuTe launcher verifies the compiled kernel's SASS against
        # these checks before its first launch
        # (``helion.runtime.cute.sass_check``).
        with SyntheticLocation():
            result.append(
                statement_from_string(
                    f"{fn.name}._helion_cute_sass_checks = {tuple(sass_checks)!r}"
                )
            )
    min_blocks = fn.config.config.get("cute_min_blocks_per_mp", 0)
    if fn.cute_state.tcgen05.min_blocks_per_mp is not None:
        # The tcgen05 role launch's own requirement wins over the
        # tunable: the epilogue-group block is one CTA per SM by SMEM
        # anyway, and a higher count would only cap its registers.
        min_blocks = fn.cute_state.tcgen05.min_blocks_per_mp
    if isinstance(min_blocks, int) and min_blocks > 0:
        # ptxas ``minnctapersm``: caps registers so ``min_blocks`` CTAs
        # stay resident per SM (occupancy over spills).
        result.append(
            statement_from_string(
                f"{fn.name}._helion_cute_min_blocks_per_mp = {min_blocks}"
            )
        )
    return result


def project_launch_block(fn: DeviceFunction) -> None:
    """Refresh the host launch call's ``block=`` with the CTA shape a late
    device lowering fixed (the collective register chain or an owned root)."""
    block_dims = (
        fn.cute_state.lanes.collective_register_chain_block_dims
        or fn.cute_state.lanes.owned_root_block_dims
    )
    if block_dims is not None:
        assert update_launch_block(fn.codegen.host_statements, fn.name, block_dims) > 0
