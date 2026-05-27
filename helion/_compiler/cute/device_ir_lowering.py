"""CuTe-specific device IR lowering pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from ..device_ir import collect_cute_half_atomic_output_promotions
from ..device_ir import promote_cute_root_graph_host_tensors
from ..device_ir import rewrite_cute_half_atomic_output_allocations
from ..device_ir_lowering import DeviceIRLowering

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from ..host_function import HostFunction


class CuteDeviceIRLowering(DeviceIRLowering):
    """CuTe-specific overrides for the device IR lowering pipeline.

    Adds half-precision atomic promotion in ``transform()`` and
    CuTe-specific tunable registration (vector widths, indexed
    reductions) in ``register()``.
    """

    def transform(self, device_ir: DeviceIR, func: HostFunction) -> None:
        super().transform(device_ir, func)
        _promote_cute_half_atomic_outputs(device_ir)

    def register(self, device_ir: DeviceIR) -> None:
        super().register(device_ir)
        _register_cute_vector_widths(device_ir)
        _track_cute_indexed_reductions(device_ir)


def _promote_cute_half_atomic_outputs(device_ir: DeviceIR) -> None:
    """Promote float16 atomic output tensors to float32.

    CuTe's SMEM atomics don't support float16, so we promote the
    allocation to float32 and cast back on return.  Must run before
    ``prepare_graph_lowerings`` reads ``node.meta["val"].dtype``.
    """
    from ..host_function import HostFunction

    promotions = collect_cute_half_atomic_output_promotions(device_ir.graphs)
    if promotions:
        host_fn = HostFunction.current()
        rewrite_cute_half_atomic_output_allocations(host_fn, promotions)
        promote_cute_root_graph_host_tensors(device_ir.graphs, promotions)


def _register_cute_vector_widths(device_ir: DeviceIR) -> None:
    """Register CuteVectorWidthSpec entries for tile and reduction blocks.

    For CuTe kernels without reduction dims, registers vector width
    specs for tile blocks (enables vec-aware lane loops in
    ``CuteNDTileStrategy``).  When reduction dims are present,
    registers vector width specs for each rollable reduction dim.

    The reduction-dim slot must stay at index 0 of
    ``cute_vector_widths`` to match the ``CuteReductionTileHeuristic``
    seed and user-facing API.
    """
    from ...autotuner.config_spec import CuteVectorWidthSpec

    env = CompileEnvironment.current()
    rdims = [bs for bs in env.block_sizes if bs.reduction]

    if not rdims:
        # Register vector widths for non-reduction tile blocks
        already_registered = set(env.config_spec.cute_vector_widths.valid_block_ids())
        tile_blocks = [bs for bs in env.block_sizes if not bs.reduction]
        for tile_bs in tile_blocks:
            if tile_bs.block_id in already_registered:
                continue
            if not isinstance(tile_bs.size, (int, torch.SymInt)):
                continue
            try:
                size_hint_val = int(tile_bs.size_hint())
            except (TypeError, ValueError, AttributeError, AssertionError):
                continue
            env.config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(
                    block_id=tile_bs.block_id,
                    size_hint=size_hint_val,
                )
            )
    else:
        # Register vector widths for each rollable reduction dim.
        # register_rollable_reductions() already ran via super().register()
        # and populated config_spec.reduction_loops.  We create matching
        # CuteVectorWidthSpec entries.
        for spec in env.config_spec.reduction_loops:
            env.config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(
                    block_id=spec.block_id,
                    size_hint=spec.size_hint,
                )
            )


def _track_cute_indexed_reductions(device_ir: DeviceIR) -> None:
    """Track which reduction dims are used for argmin/argmax.

    CuTe can only combine these via ``cute.arch.warp_reduction``
    (32 threads max), so the autotuner must keep their persistent
    thread count and looped chunk size within a single warp.
    """
    env = CompileEnvironment.current()
    rdims = [bs for bs in env.block_sizes if bs.reduction]
    if not rdims:
        return

    num_original_graphs = len(device_ir.graphs)
    # Roller analysis may have added sub-graphs; scan only originals.
    # Count original graphs by finding the first ReductionLoopGraphInfo.
    from ..device_ir import ReductionLoopGraphInfo

    for i, g in enumerate(device_ir.graphs):
        if isinstance(g, ReductionLoopGraphInfo):
            num_original_graphs = i
            break

    indexed_blocks: set[int] = set()
    indexed_targets = {
        torch.ops.aten.argmin.default,
        torch.ops.aten.argmax.default,
    }
    for graph_info in device_ir.graphs[:num_original_graphs]:
        for node in graph_info.graph.nodes:
            if getattr(node, "target", None) not in indexed_targets:
                continue
            args = node.args or ()
            if not args:
                continue
            val = getattr(args[0], "meta", {}).get("val")
            if val is None:
                continue
            dim_arg = args[1] if len(args) >= 2 else -1
            dim_indices = (
                [int(cast("int", d)) for d in dim_arg]
                if isinstance(dim_arg, list)
                else [int(cast("int", dim_arg))]
            )
            for dim_idx in dim_indices:
                if dim_idx < 0:
                    dim_idx += val.ndim
                if 0 <= dim_idx < val.ndim:
                    reduce_dim = val.size(dim_idx)
                    block_id = env.resolve_block_id(reduce_dim)
                    if block_id is not None:
                        indexed_blocks.add(block_id)
    env.config_spec.cute_indexed_reduction_block_ids = indexed_blocks
