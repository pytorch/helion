"""Which grid candidates the Metal autotuner times: a cap on threadgroups
that do too little work.

The shared per-axis floor (``ConfigSpec.raise_grid_block_minimums``) bounds a
root grid's program count by ``num_compute_units() * 64``.  A program count
can't tell a decode GEMV's 19968 threadgroups, each running an 832-trip K
loop, from an elementwise kernel's millions of one-element threadgroups, and
on Apple GPUs the first kind wins: few-row matmuls and GEMVs run fastest at
one or two output columns per threadgroup.  So Metal leaves grid blocks
unfloored and screens candidates on their work per threadgroup instead.

A candidate is timed when each root grid either fits the occupancy budget
(``GRID_PROGRAMS_PER_COMPUTE_UNIT`` threadgroups per GPU core) or gives every
threadgroup at least :data:`MIN_WORK` units of work: the grid elements it
covers times the trips its loops and reductions run per element
(:func:`root_trips`).  Below one SIMD group's worth of work, a threadgroup
in an over-budget grid leaves lanes idle by construction, and the grid only
grows the launch.  The threshold never exceeds the default config's own work
per threadgroup, so the default is always timed.

The screen applies to autotune candidates only (``autotune_config_is_viable``);
a config the user gives runs as given.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

from ...autotuner.config_spec import GRID_PROGRAMS_PER_COMPUTE_UNIT
from ..inductor_lowering import ReductionLowering
from ._constants import SIMD_WIDTH

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir_analysis import DeviceIRAnalysis

#: The least work per threadgroup of a grid past the occupancy budget: one
#: unit per lane of a SIMD group.
MIN_WORK = SIMD_WIDTH


@functools.cache
def gpu_cores() -> int:
    """The GPU cores of this Mac (16 on an M4 Pro).

    ``num_compute_units()`` falls back to 128 off CUDA, so it doesn't size a
    Metal budget."""
    return torch.backends.mps.get_core_count()


def occupancy_budget() -> int:
    """Threadgroups a root grid may launch whatever their work."""
    return gpu_cores() * GRID_PROGRAMS_PER_COMPUTE_UNIT


@dataclasses.dataclass(frozen=True)
class RootWork:
    """One root grid's axes and the trips each of its elements runs."""

    #: (block id, extent) of each grid axis.  An axis without a block-size
    #: spec (an ``hl.grid`` axis) always runs one element per program.
    axes: tuple[tuple[int, int], ...]
    #: The most trips a grid element runs in one graph of the root: the
    #: extents of the loops around the graph times those it reduces.  A loop
    #: whose bounds depend on data or on another tile counts as 1.
    trips: int


def root_trips(
    device_ir: DeviceIR,
    analysis: DeviceIRAnalysis,
    config_spec: ConfigSpec,
    env: CompileEnvironment,
) -> tuple[RootWork, ...]:
    """Each root grid's :class:`RootWork`."""
    fact = config_spec.kernel_grid_fact
    if fact is None:
        return ()
    grid = set(config_spec.grid_block_ids)

    def extent(block_id: int) -> int:
        size = env.block_sizes[block_id].size
        if isinstance(size, int):
            return size
        if isinstance(size, torch.SymInt) and not free_unbacked_symbols(size):
            return max(1, env.size_hint(size))
        return 1  # bounds that depend on data or on another tile

    def reduced(graph_id: int) -> set[int]:
        info = device_ir.graphs[graph_id]
        return {
            lowering.block_index
            for node in info.graph.nodes
            if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
            and lowering.block_index is not None
        }

    trips: dict[int, int] = {}
    for info in device_ir.graphs:
        root = fact.root_id_for_graph(info.graph_id)
        if root is None:
            continue
        # The graph's reductions run once per trip of every loop around it.
        blocks = reduced(info.graph_id)
        graph_id: int | None = info.graph_id
        while graph_id is not None:
            blocks |= analysis.loop_block_ids.get(graph_id, frozenset())
            graph_id = analysis.parent_of.get(graph_id)
        count = math.prod(extent(b) for b in blocks - grid)
        trips[root] = max(trips.get(root, 1), count)
    return tuple(
        RootWork(
            axes=tuple((b, extent(b)) for b in root.block_ids),
            trips=trips.get(root.root_graph_id, 1),
        )
        for root in fact.roots
    )


def _programs_and_work(root: RootWork, blocks: dict[int, int]) -> tuple[int, int]:
    """The root grid's program count and each program's work; an empty axis
    launches no programs."""
    if any(size == 0 for _, size in root.axes):
        return 0, 0
    programs = elements = 1
    for block_id, size in root.axes:
        block = min(blocks.get(block_id, 1), size)
        programs *= -(-size // block)
        elements *= block
    return programs, elements * root.trips


def grid_is_viable(
    roots: tuple[RootWork, ...],
    blocks: dict[int, int],
    default_blocks: dict[int, int],
    budget: int,
) -> bool:
    """Whether every root grid of the candidate with *blocks* fits *budget*
    or gives each threadgroup the work the module docstring asks for."""
    for root in roots:
        programs, work = _programs_and_work(root, blocks)
        if programs <= budget:
            continue
        _, default_work = _programs_and_work(root, default_blocks)
        if work < min(MIN_WORK, default_work):
            return False
    return True


def config_is_viable(
    config_spec: ConfigSpec, config: Config, roots: tuple[RootWork, ...]
) -> bool:
    """:func:`grid_is_viable` for *config*'s block sizes (a config without a
    full list of positive sizes is left to the search's own checks)."""
    sizes = config.config.get("block_sizes")
    specs = list(config_spec.block_sizes)
    if not isinstance(sizes, list) or len(sizes) != len(specs):
        return True
    if not all(type(size) is int and size > 0 for size in sizes):
        return True
    blocks = {spec.block_id: size for spec, size in zip(specs, sizes, strict=True)}
    defaults = {
        spec.block_id: spec._fragment(config_spec).default_val for spec in specs
    }
    return grid_is_viable(roots, blocks, defaults, occupancy_budget())
