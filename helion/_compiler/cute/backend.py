"""``CuteBackend``: the CuTe backend's implementation of the ``Backend`` hooks.

Role: hook surface.  Core (``helion/_compiler/backend.py``) calls the
hooks; this module answers them by forwarding into the package: the host
rewrites (``customize_ast``), the device-IR passes (``rewrite_device_ir`` ->
``ir_passes``), the search-surface registration (``search_surface``), the
tile / reduction / program-id strategies (the core per-thread planner
``per_thread_planning``, ``reduction_emitter``, ``tcgen05.scheduler_pids``), the
per-thread strategies' layout and lane-nest hooks (``tile_packing``,
``lane_cluster_split``, ``lane_loops.nest_check``), the
planner cascade (``pre_codegen``), the emitter registries (``emitters``,
``kernel_generators``), the device-body passes (``passes``), the lane
pipeline (``lane_loops.root_passes``), the state factories (``state``), the
autotune surface (``autotune``) and the runtime hooks (``runtime_hooks``).
The module-level functions before the class are the MMA / attention loop
detectors the ``per_thread_*`` hooks run for the shared per-thread tile
planner (tests patch them here).  Most imports are
function-local: ``backend_registry`` loads this module while
``compile_environment`` is still initializing.  ``README.md`` in this
directory is the map.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
from itertools import starmap
import operator
import os
import re
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import cast

import sympy
import torch
from torch.utils._sympy.symbol import SymT
from torch.utils._sympy.symbol import symbol_is_type

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..backend import Backend
from ..backend import BackendState
from ..backend import DeviceIRStage
from ..backend import _largest_divisor_at_most
from ..backend import log
from .analysis.attention_match import _attention_flash_gate_enabled
from .analysis.attention_match import _attention_flash_supported
from .analysis.attention_match import _attention_pattern_fits_root
from .analysis.attention_match import _attention_pattern_geometry
from .analysis.attention_match import _attention_softmax_pattern_head_dim
from .analysis.specialized_mma import _grouped_rank3_specialized_mma_plan
from .analysis.specialized_mma import _kernel_specialized_mma_impl
from .analysis.specialized_mma import _kernel_specialized_mma_plan
from .analysis.specialized_mma import _specialized_mma_root_mn_block_ids
from .attention_plan import flash_head_dims_supported
from .cache_policy_loads import PACKET_LOAD_HELPERS
from .direct_affine_plan import DIRECT_AFFINE_ORDINARY_SCHEDULE
from .launch_shape import LaunchShapeFlags
from .launch_shape import check_launch_shape
from .launch_shape import live_grid_thread_extents
from .launch_shape import plan_owned_launch
from .math_templates import SIGMOID_TEMPLATE
from .math_templates import argreduce_candidate_expr
from .math_templates import nan_extremum_expr
from .pdl_release import ZERO_FILL_KERNEL_PREFIX
from .pdl_release import generic_release_rendered
from .pdl_release import spread_rendered
from .state import cute_codegen_state
from .state import cute_env_state
from .state import new_codegen_state
from .state import new_device_state
from .state import new_env_state
from .tcgen05.knobs import launch_epi_warp_groups_knob
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from .thread_budget import MAX_THREADS_PER_BLOCK
from .thread_budget import check_thread_block_dims
from .thread_budget import referenced_thread_axes
from .thread_budget import thread_idx_axis

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch._inductor.ops_handler import OpsHandler

    from ...autotuner.config_priors import ValuePrior
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.llm.profile import LlmPromptProfile
    from ...autotuner.search_surface import BackendSearchSurface
    from ...runtime.config import Config
    from ...runtime.cute.structural_policy import CuteStructuralPolicy
    from ...runtime.kernel import BoundKernel
    from ...runtime.settings import Settings
    from ..aten_lowering import Lowering
    from ..aten_lowering import LoweringContext
    from ..autotuner_heuristics.registry import AutotunerHeuristicType
    from ..autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ..backend import HandlerOpOverrides
    from ..compile_environment import CompileEnvironment
    from ..device_function import Argument
    from ..device_function import DeviceBodyContext
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import GraphInfo
    from ..device_ir_analysis import DeviceIRAnalysis
    from ..emitter_protocols import KernelGenerator
    from ..emitter_protocols import RetryPolicy
    from ..emitter_protocols import RootEmitter
    from ..generate_ast import GenerateAST
    from ..helper_function import CodegenInterface
    from ..host_function import HostFunction
    from ..inductor_lowering import CodegenState
    from ..program_id import ProgramIDs
    from ..reduction_strategy import ReductionEmitter
    from ..reduction_strategy import ReductionStrategy
    from ..tile_dispatch import TileStrategyDispatch
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState
    from ..tile_strategy import PerThreadNDTileStrategy
    from ..tile_strategy import TileStrategy
    from ..tile_strategy import VecLaneWrapper
    from .attention_plan import AttentionScorePlan
    from .device_state import CuteDeviceFunctionState
    from .lane_loops.statements import LanePlacement
    from .lane_loops.statements import LaneScope
    from .state import CuteCodegenState
    from .state import CuteEnvState

    InductorOpOverrides = OpsHandler[Any]

# Integer dtypes narrower than the Int32 the DSL gives a Python int.
_NARROW_INTEGER_DTYPES = (torch.int8, torch.int16, torch.uint8)


def validate_thread_axis_accesses(statements: Sequence[ast.AST]) -> None:
    """Reject live invalid CUDA coordinates after owned-root scaffolding DCE."""
    nodes = [node for statement in statements for node in ast.walk(statement)]

    def thread_tuple(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.thread_idx"
        ) or (isinstance(node, ast.Name) and node.id in aliases)

    aliases: set[str] = set()
    while True:
        previous = len(aliases)
        for node in nodes:
            if isinstance(node, ast.Assign) and thread_tuple(node.value):
                aliases.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
        if len(aliases) == previous:
            break
    for node in nodes:
        if not isinstance(node, ast.Subscript) or not thread_tuple(node.value):
            continue
        axis = node.slice
        if not (
            isinstance(axis, ast.Constant)
            and type(axis.value) is int
            and 0 <= axis.value < 3
        ):
            raise exc.BackendUnsupported("cute", f"thread axis {ast.unparse(axis)}")


# The ``cute.arch`` calls a thread makes on its own: its coordinates and the
# launch's shape, per-thread memory operations, scalar math and the
# programmatic-dependent-launch controls (atomics by their ``atomic_`` prefix).
# Every other ``cute.arch`` call reaches other threads: shuffles, votes, warp
# reductions, barriers, mbarriers, shared memory, warp and lane indices,
# clusters, asynchronous copies.
_CUTE_ARCH_PER_THREAD_CALLS = frozenset(
    {
        "thread_idx",
        "block_idx",
        "grid_dim",
        "block_dim",
        "load",
        "store",
        "prefetch",
        "fmax",
        "fmin",
        "exp2",
        "rcp_approx",
        "cvt_f32_tf32",
        "fma_packed_f32x2",
        "mul_packed_f32x2",
        "add_packed_f32x2",
        "sub_packed_f32x2",
        "calc_packed_f32x2_op",
        "griddepcontrol_wait",
        "griddepcontrol_launch_dependents",
    }
)
# Helion's kernel-side helpers a thread calls on its own; the others
# (``_cute_grouped_reduce_*``, ``_cute_resident_*``, the rank-1 and tcgen05
# paths) combine or stage values across threads.
_CUTE_PER_THREAD_HELPERS = frozenset(
    {
        "_cute_argreduce_index",
        "_cute_atomic_max_float32",
        "_cute_atomic_min_float32",
        "_cute_float4_e2m1fn_x16_dot_float32",
        "_cute_fp8e4m3fn_to_float32",
        "_cute_gelu_erf_exact_f32x2",
        "_cute_sigmoid_approx_ftz_f32",
    }
)
# The ``cute`` entry points through which a body partitions a tile across
# threads: copies, MMA and their atoms; ``cutlass.utils`` (shared-memory and
# TMEM allocators, schedulers) and ``cutlass.pipeline`` as a whole.
_CUTE_CROSS_THREAD_ENTRY_POINTS = frozenset(
    {
        "copy",
        "gemm",
        "nvgpu",
        "autovec_copy",
        "local_partition",
        "local_tile",
        "make_copy_atom",
        "make_mma_atom",
        "make_tiled_mma",
    }
)
_CUTE_CROSS_THREAD_ENTRY_PREFIXES = ("make_tiled_copy",)


def _cross_thread_constructs(statements: Sequence[ast.AST]) -> list[str]:
    """The constructs of a kernel body that reach other threads or hand a tile to a collective.

    Each was emitted for the static launch shape: the lanes a warp reduction
    spans, the stride of a shared-memory reduction's lane index, the
    partition of a tiled copy or MMA.  A launch axis sized by a kernel
    argument holds a one in that shape, so the construct is wrong once the
    axis has more than one thread (a warp reduction over 32 column threads
    beside eight row threads sums across rows).  Fail closed: every
    ``cute.arch`` call outside ``_CUTE_ARCH_PER_THREAD_CALLS`` and every
    ``_cute_`` helper outside ``_CUTE_PER_THREAD_HELPERS`` counts.
    """
    found: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name):
                if (
                    node.id.startswith("_cute_")
                    and node.id not in _CUTE_PER_THREAD_HELPERS
                ):
                    found.add(node.id)
                continue
            if not isinstance(node, ast.Attribute):
                continue
            owner = node.value
            if isinstance(owner, ast.Name):
                if owner.id == "cute" and (
                    node.attr in _CUTE_CROSS_THREAD_ENTRY_POINTS
                    or node.attr.startswith(_CUTE_CROSS_THREAD_ENTRY_PREFIXES)
                ):
                    found.add(f"cute.{node.attr}")
                elif owner.id == "cutlass" and node.attr in ("utils", "pipeline"):
                    found.add(f"cutlass.{node.attr}")
            elif (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "cute"
                and owner.attr == "arch"
                and node.attr not in _CUTE_ARCH_PER_THREAD_CALLS
                and not node.attr.startswith("atomic_")
            ):
                found.add(f"cute.arch.{node.attr}")
    return sorted(found)


def _symbolic_axes_text(axes: dict[int, str]) -> str:
    return ", ".join(f"{axis} ({expr})" for axis, expr in sorted(axes.items()))


def _thread_axes_read_outside_leader_guards(statements: Sequence[ast.AST]) -> set[int]:
    """The launch axes whose thread index the body reads, leader guards aside.

    A leader guard compares the index with zero (``cute.arch.thread_idx()[k]
    == 0``, see ``atomic_ops._cute_leader_predicate``); it is the one read of
    an axis that holds for a single thread.
    """
    guards: set[int] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 0
                and thread_idx_axis(node.left) is not None
            ):
                guards.add(id(node.left))
    axes: set[int] = set()
    for statement in statements:
        for node in ast.walk(statement):
            axis = thread_idx_axis(node)
            if axis is not None and id(node) not in guards:
                axes.add(axis)
    return axes


def launch_drops_root_threads(
    dims: Sequence[int], roots: Sequence[Mapping[int, int]]
) -> tuple[int, int] | None:
    """The first ``(axis, extent)`` of a root loop's read axes that ``dims``
    launches narrower than its extent, or ``None``.

    ``roots`` holds, per root loop, the thread extent of each launch axis
    the root's own statements read (``CuteCodegenState.root_read_axis_sizes``).
    An axis a root never reads (a tile whose index is unused) is absent and
    may launch one thread, whatever extent a sibling root records for it.
    """
    for root in roots:
        for axis, extent in sorted(root.items()):
            if axis < len(dims) and dims[axis] < extent:
                return axis, extent
    return None


def _reject_dropped_root_threads(
    dims: Sequence[int], roots: Sequence[Mapping[int, int]]
) -> None:
    """Refuse a launch that never runs some of a root tile's threads.

    A root tile's threads index its rows and columns directly on the axes its
    statements read; a launch narrower than them never runs the rows past the
    launch, leaving their outputs unwritten.
    """
    dropped = launch_drops_root_threads(dims, roots)
    if dropped is not None:
        axis, live = dropped
        raise exc.BackendUnsupported(
            "cute",
            f"launch block {tuple(dims)} drops threads of a root tile that "
            f"reads {live} threads on axis {axis}",
        )


def _reduced_block_ids(codegen: GenerateAST) -> set[int]:
    """The block ids some lowered reduction of the kernel reduces over."""
    # Local import: see ``_multi_phase_block_dims``.
    from ..inductor_lowering import ReductionLowering

    return {
        lowering.block_index
        for graph in codegen.codegen_graphs
        for node in graph.graph.nodes
        if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
    }


def _reject_multi_root_reduction_surplus(
    device_function: DeviceFunction, dims: Sequence[int]
) -> None:
    """Refuse a multi-root launch whose width on a reduction's axis differs
    from the reduction's own lane count.

    Root loops share one launch, the elementwise max of their thread shapes,
    so a root reducing over fewer lanes than another root runs on that axis
    runs surplus lanes: they index past the reduced extent, reduce among
    themselves and race the reduction's result to the same address.  As for
    ``hl.barrier()`` phases (``_multi_phase_block_dims``), such a reduction is
    rejected rather than folded with the surplus lanes.
    """
    from ..reduction_strategy import ReductionStrategy

    tile_strategy = device_function.tile_strategy
    reduced_block_ids = _reduced_block_ids(device_function.codegen)
    for strategy in tile_strategy.strategies:
        if (
            not isinstance(strategy, ReductionStrategy)
            or strategy.block_index not in reduced_block_ids
        ):
            continue
        count = strategy._reduction_thread_count()
        axis = tile_strategy.thread_axis_for_strategy(strategy)
        if count > 1 and axis is not None and axis < len(dims) and dims[axis] != count:
            raise exc.BackendUnsupported(
                "cute",
                f"reduction block {strategy.block_index} spans {count} lanes on "
                f"thread axis {axis} but the launch {tuple(dims)} shared by the "
                f"kernel's root loops runs {dims[axis]} there; the surplus lanes "
                "would corrupt the cross-lane reduction",
            )


def _multi_phase_block_dims(
    device_function: DeviceFunction,
    *,
    exact_grid_thread_dims: tuple[int, ...],
) -> tuple[int, int, int]:
    """Launch block dims for an ``hl.barrier()`` (multi-phase) kernel.

    All phases run inside one persistent CTA, so the launch is the elementwise
    max of the thread extents the phase bodies recorded.  A phase with fewer
    lanes on an axis then runs with surplus threads: tile axes mask them, but
    a cross-lane reduction over that axis would either fold the surplus lanes
    in or leave them racing on the result, so any reduced axis whose
    phase-local extent differs from the launch is rejected loudly instead of
    silently computing garbage.
    """
    # Local imports: these modules import ``compile_environment``, which
    # reaches this module back through ``backend_registry`` -> ``..backend``
    # before ``CompileEnvironment`` exists.
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    codegen = device_function.codegen
    tile_strategy = device_function.tile_strategy
    dims = (
        max(
            codegen.max_thread_block_dims[0],
            codegen.root_thread_block_dims[0],
            codegen.referenced_thread_block_dims[0],
            exact_grid_thread_dims[0],
        ),
        max(
            codegen.max_thread_block_dims[1],
            codegen.root_thread_block_dims[1],
            codegen.referenced_thread_block_dims[1],
            exact_grid_thread_dims[1],
        ),
        max(
            codegen.max_thread_block_dims[2],
            codegen.root_thread_block_dims[2],
            codegen.referenced_thread_block_dims[2],
            exact_grid_thread_dims[2],
        ),
    )
    if functools.reduce(operator.mul, dims, 1) > MAX_THREADS_PER_BLOCK:
        raise exc.BackendUnsupported(
            "cute",
            f"hl.barrier() phases need a combined launch block of {dims} "
            f"threads, exceeding {MAX_THREADS_PER_BLOCK}",
        )

    def reject(axis: int, extent: int, what: str) -> None:
        raise exc.BackendUnsupported(
            "cute",
            f"{what} spans {extent} lanes on thread axis {axis} but the "
            f"hl.barrier() launch {dims} runs {dims[axis]} there; the surplus "
            "lanes of another phase would corrupt the cross-lane reduction",
        )

    reduced_block_ids = _reduced_block_ids(codegen)
    for strategy in tile_strategy.strategies:
        base_axis = tile_strategy.thread_axis_for_strategy(strategy)
        if base_axis is None:
            continue
        if isinstance(strategy, ReductionStrategy):
            count = strategy._reduction_thread_count()
            if count > 1 and base_axis < 3 and dims[base_axis] != count:
                reject(base_axis, count, f"reduction block {strategy.block_index}")
            continue
        for (
            block_id,
            local_axis,
            size,
            _expr,
        ) in tile_strategy._iter_strategy_thread_axes(strategy):
            if block_id not in reduced_block_ids or size is None:
                continue
            if isinstance(strategy, PerThreadNDTileStrategy):
                size = strategy.thread_extent_for_masking(block_id, size)
            axis = base_axis + local_axis
            if size > 1 and axis < 3 and dims[axis] != size:
                reject(axis, size, f"reduction over tile block {block_id}")
    for (
        subject,
        layout,
    ) in device_function.cute_state.lanes.multi_phase_lane_reduce_layouts:
        for axis, size in layout.items():
            if axis < 3 and dims[axis] != size:
                reject(axis, size, subject)
    return dims


def _pointwise_grid_thread_dims(
    live_extents: dict[int, int],
    final_thread_axes: set[int],
    referenced_dims: Sequence[int],
    *,
    has_pointwise_fact: bool,
    has_nested_device_loops: bool,
    has_synthetic_free_axes: bool,
) -> tuple[int, int, int] | None:
    if (
        not has_pointwise_fact
        or not live_extents
        or not final_thread_axes.issubset(live_extents)
        or has_nested_device_loops
        or has_synthetic_free_axes
    ):
        return None
    if any(axis < 0 or axis >= 3 for axis in final_thread_axes | live_extents.keys()):
        raise exc.BackendUnsupported(
            "cute", "pointwise grid requires more than three physical thread axes"
        )
    if any(referenced_dims[axis] > live_extents[axis] for axis in final_thread_axes):
        # A larger reference is not evidence that every producer has a surplus
        # mask. Do not add duplicate/out-of-tile writers to satisfy it.
        raise exc.BackendUnsupported(
            "cute", "pointwise grid reference exceeds its live producer extent"
        )
    dims = [
        live_extents.get(axis, 1) if axis in final_thread_axes else 1
        for axis in range(3)
    ]
    # CUDA's z dimension is capped independently of the total thread count.
    # A numerically complete rectangular grid can otherwise request z=128
    # while still fitting the 1024-thread product budget checked by the caller.
    if any(size > limit for size, limit in zip(dims, (1024, 1024, 64), strict=True)):
        raise exc.BackendUnsupported(
            "cute", f"pointwise grid launch exceeds CUDA per-axis limits: {tuple(dims)}"
        )
    return dims[0], dims[1], dims[2]


def _detect_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
) -> bool:
    """Check if a device loop contains a matmul with MMA-compatible dtypes.

    Returns True only when the loop contains a compatible addmm/dot AND
    the grid has M and N block IDs, so the MMA pipeline can map them to tile
    offsets. Leading-batch matmul may add one extra root grid axis.
    Three-level loops (grid[M] + device_loop[N] + device_loop[K]) are NOT
    supported yet.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction
    from .cute_mma import analyze_cute_mma_node

    # MMA lowering currently relies on a single grid state that carries
    # both the M and N axes. Nested grid loops like grid[M] + grid[N] do
    # not satisfy that requirement because GenerateAST.current_grid_state
    # only tracks the innermost grid.
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    if len(device_ir.grid_block_ids[0]) not in (2, 3):
        return False
    root_grid_block_ids = tuple(device_ir.grid_block_ids[0])
    root_grid_ids = set(root_grid_block_ids)
    # CuTe MMA fragment partitioning is currently keyed to physical threads.
    # When an M/N tile is partially serialized into lane loops, the same
    # fragment would be reused for multiple logical lanes and produce
    # incorrect results. A pure K reduction loop is different: it does not
    # contribute MMA fragment coordinates, so we can still enable mma_mode
    # there to suppress synthetic lane loops around the K body.
    if any(
        block_id in root_grid_ids and threads > 0 and threads < block_size
        for block_id, block_size, threads in zip(
            block_ids,
            block_sizes,
            num_threads_config,
            strict=False,
        )
    ):
        return False
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        if graph_info.block_ids != block_ids:
            continue
        for node in graph_info.graph.nodes:
            candidate = analyze_cute_mma_node(node, graphs=fn.codegen.codegen_graphs)
            if (
                candidate is not None
                and not candidate.requires_scalar_fallback
                and candidate.operands.output_block_ids == root_grid_block_ids
                and candidate.operands.k_block_id in block_ids
                and (candidate.with_acc or candidate.is_dot)
            ):
                return True
    return False


def _specialized_mma_root_thread_layout(
    root_block_ids: Sequence[int], config: Config
) -> tuple[int, int, int, int, bool, bool] | None:
    """``(bm, bn, m_threads, n_threads, m_explicit, n_explicit)`` of the root.

    An automatic thread count (``num_threads`` of 0) stands in as the block
    size, trimmed to the 1024-thread CTA budget; the flags tell the MMA
    support predicate whether the M / N counts were requested by the config.
    """
    from ..compile_environment import CompileEnvironment

    if len(root_block_ids) != 2:
        return None
    env = CompileEnvironment.current()
    root_block_sizes: list[int] = []
    root_thread_counts: list[int] = []
    root_thread_auto: list[bool] = []
    for block_id in root_block_ids:
        block_size = env.block_sizes[block_id].from_config(config)
        if not isinstance(block_size, int):
            return None
        root_block_sizes.append(block_size)
        threads = env.config_spec.num_threads.config_get(
            config.num_threads, block_id, 0
        )
        root_thread_counts.append(threads if threads > 0 else block_size)
        root_thread_auto.append(threads == 0)
    if functools.reduce(operator.mul, root_thread_counts, 1) > 1024:
        for idx in (1, 0):
            if not root_thread_auto[idx]:
                continue
            other_threads = root_thread_counts[1 - idx]
            next_threads = _largest_divisor_at_most(
                root_block_sizes[idx], max(1024 // max(other_threads, 1), 1)
            )
            root_thread_counts[idx] = next_threads
            if functools.reduce(operator.mul, root_thread_counts, 1) <= 1024:
                break
    return (
        root_block_sizes[0],
        root_block_sizes[1],
        root_thread_counts[0],
        root_thread_counts[1],
        not root_thread_auto[0],
        not root_thread_auto[1],
    )


def _specialized_mma_root_threads_support_impl(
    mma_impl: str,
    *,
    bm: int,
    bn: int,
    root_m_threads: int,
    root_n_threads: int,
    root_m_threads_explicit: bool,
    root_n_threads_explicit: bool,
    epi_warp_groups: int = 1,
) -> bool:
    from .cute_mma import _mma_active_n_threads
    from .cute_mma import _tcgen05_root_m_threads
    from .cute_mma import _tcgen05_root_n_threads

    if mma_impl == "tcgen05":
        # The tcgen05 role launch is one physical warp per role row (see
        # ``_tcgen05_root_m_threads``) and its y extent is the role warp
        # count, clamped by the SIMT N axis (``_tcgen05_root_n_threads``):
        # an explicit M or N thread count has to be exactly the width the
        # launch planning resolves, so another request takes the generic
        # SIMT path instead of launching warps the role predicates and the
        # pipeline-init barrier do not count, or too few warps for the
        # roles.  Automatic counts arrive here as the tile extents and are
        # resolved by the launch planning.
        physical_m_threads = _tcgen05_root_m_threads(bm, bn)
        if root_m_threads_explicit:
            m_threads_supported = root_m_threads == physical_m_threads
        else:
            m_threads_supported = physical_m_threads <= root_m_threads <= bm
        if root_n_threads_explicit:
            n_threads_supported = root_n_threads == _tcgen05_root_n_threads(
                bn, epi_warp_groups=epi_warp_groups
            )
        else:
            n_threads_supported = (
                _mma_active_n_threads("tcgen05") <= root_n_threads <= bn
            )
        return (
            m_threads_supported
            and bm % root_m_threads == 0
            and n_threads_supported
            and bn % root_n_threads == 0
            and root_m_threads * root_n_threads <= 1024
        )
    return mma_impl in ("warp", "universal") and (
        root_m_threads == bm and root_n_threads == bn
    )


def _detect_grouped_rank3_specialized_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    config: Config,
) -> bool:
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .cute_mma import _rank3_grouped_root_axes

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1 or len(block_ids) != 1:
        return False
    root_grid_ids = device_ir.grid_block_ids[0]
    if any(block_id in root_grid_ids for block_id in block_ids):
        return False
    env = CompileEnvironment.current()
    semantic_block_ids = env.config_spec.tcgen05.matmul_block_ids
    axes = None
    if semantic_block_ids is not None:
        m_block_id, n_block_id, k_block_id = semantic_block_ids
        if env.canonical_block_id(k_block_id) == env.canonical_block_id(block_ids[0]):
            axes = _rank3_grouped_root_axes(
                env,
                device_ir,
                m_block_id=m_block_id,
                n_block_id=n_block_id,
                k_block_id=block_ids[0],
            )
    if semantic_block_ids is not None and axes is None:
        return False
    if axes is not None:
        segment_root_grid_id = axes.segment_block_id
        mn_root_grid_ids = [axes.m_block_id, axes.n_block_id]
    elif len(root_grid_ids) == 2:
        segment_root_grid_id = None
        mn_root_grid_ids = root_grid_ids
    elif len(root_grid_ids) == 3:
        segment_root_grid_id = root_grid_ids[0]
        mn_root_grid_ids = root_grid_ids[1:]
    else:
        return False
    if segment_root_grid_id is not None:
        segment_block = env.block_sizes[segment_root_grid_id].from_config(config)
        segment_threads = env.config_spec.num_threads.config_get(
            config.num_threads, segment_root_grid_id, 0
        )
        if segment_block != 1 or segment_threads not in (0, 1):
            return False
    root_layout = _specialized_mma_root_thread_layout(mn_root_grid_ids, config)
    if root_layout is None:
        return False
    (
        bm,
        bn,
        root_m_threads,
        root_n_threads,
        root_m_threads_explicit,
        root_n_threads_explicit,
    ) = root_layout
    (bk,) = block_sizes
    if not isinstance(bk, int):
        return False
    if not _specialized_mma_root_threads_support_impl(
        "tcgen05",
        bm=bm,
        bn=bn,
        root_m_threads=root_m_threads,
        root_n_threads=root_n_threads,
        root_m_threads_explicit=root_m_threads_explicit,
        root_n_threads_explicit=root_n_threads_explicit,
        epi_warp_groups=launch_epi_warp_groups_knob(config),
    ):
        return False

    for graph_info in fn.codegen.codegen_graphs:
        if getattr(graph_info, "block_ids", None) != block_ids:
            continue
        for node in graph_info.graph.nodes:
            if (
                _grouped_rank3_specialized_mma_plan(
                    node,
                    fn=fn,
                    k_block_id=block_ids[0],
                    bk=bk,
                    config=config,
                    env=env,
                )
                is not None
            ):
                return True
    return False


def _detect_specialized_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    config: Config,
) -> bool:
    from ..host_function import HostFunction
    from .cute_mma import _choose_mma_impl
    from .cute_mma import _loop_subrange_parent_block_id
    from .cute_mma import _mma_tiles_are_static_full
    from .cute_mma import _ragged_batched_tcgen05_config
    from .cute_mma import analyze_cute_mma_node
    from .cute_mma import ensure_tcgen05_fragment_epilogue_plan
    from .mma_support import cute_fp32_dot_uses_tf32
    from .simt_gemm import simt_gemm_root_layout_supported
    from .simt_gemm import simt_mma_requested

    if _detect_grouped_rank3_specialized_mma_loop(
        fn,
        block_ids,
        block_sizes=block_sizes,
        config=config,
    ):
        return True

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(block_ids) != 1 or any(block_id in root_grid_ids for block_id in block_ids):
        return False

    # The same boundary-only proof the root strategy applies
    # (``create_loop_strategy``): axes that only bound an inner loop carry no
    # threads or lanes, so the output tile is the root minus those axes.
    from ..compile_environment import CompileEnvironment
    from ..per_thread_planning import tile_loop_inactive_block_ids
    from .loop_nesting import boundary_only_grid_blocks

    env = CompileEnvironment.current()
    inactive_root_block_ids = frozenset(
        tile_loop_inactive_block_ids(
            fn,
            list(root_grid_ids),
            boundary_only_grid_blocks(env, device_ir, fn.codegen.codegen_graphs),
        )
    )
    candidates = []
    for graph_info in fn.codegen.codegen_graphs:
        if getattr(graph_info, "block_ids", None) != block_ids:
            continue
        for node in graph_info.graph.nodes:
            candidate = analyze_cute_mma_node(node, graphs=fn.codegen.codegen_graphs)
            if (
                candidate is not None
                and not candidate.requires_scalar_fallback
                and candidate.operands.k_block_id == block_ids[0]
                and _specialized_mma_root_mn_block_ids(
                    candidate,
                    config,
                    inactive_root_block_ids=inactive_root_block_ids,
                )
                is not None
            ):
                candidates.append((node, candidate))
    if not candidates:
        return False

    root_mn_block_ids = _specialized_mma_root_mn_block_ids(
        candidates[0][1], config, inactive_root_block_ids=inactive_root_block_ids
    )
    assert root_mn_block_ids is not None
    root_layout = _specialized_mma_root_thread_layout(root_mn_block_ids, config)
    if root_layout is None:
        return False
    (
        bm,
        bn,
        root_m_threads,
        root_n_threads,
        root_m_threads_explicit,
        root_n_threads_explicit,
    ) = root_layout
    (bk,) = block_sizes
    if not isinstance(bk, int):
        return False
    # Ragged batched tiles take the flat one-CTA tcgen05 pipeline (TMA
    # zero-fill, predicated stores); the launch geometry must agree with
    # ``plan_tcgen05_pipeline``'s admission or the MMA warp never exists.
    candidates = [
        (node, candidate)
        for node, candidate in candidates
        if not candidate.operands.has_leading_passthrough
        or _mma_tiles_are_static_full(candidate.operands, bm=bm, bn=bn, bk=bk)
        or _ragged_batched_tcgen05_config(config, bn)
    ]
    if not candidates:
        return False

    # A K loop that subdivides an outer tile (``hl.tile(outer.begin,
    # outer.end)``) may only become a lane-free specialized loop through the
    # exact fp32 SIMT GEMM, which masks its steps against the loop's own end;
    # every other lowering declines it in ``plan_tcgen05_pipeline``, and a
    # declined lane-free loop would leave the scalar fallback one K element
    # per step.
    k_subrange_parent = _loop_subrange_parent_block_id(env, block_ids[0])
    k_loop_is_subrange = k_subrange_parent is not None
    k_subrange_in_root = k_subrange_parent is None or k_subrange_parent in root_grid_ids
    fp32_exact = not cute_fp32_dot_uses_tf32()
    for node, candidate in candidates:
        lhs_val = candidate.operands.lhs.source_fake
        mma_impl = _choose_mma_impl(
            lhs_val.dtype,
            bm=bm,
            bn=bn,
            bk=bk,
            config=config,
            input_device=lhs_val.device,
        )
        if candidate.requires_accumulator_seed:
            if not k_loop_is_subrange and _specialized_mma_root_threads_support_impl(
                "universal",
                bm=bm,
                bn=bn,
                root_m_threads=root_m_threads,
                root_n_threads=root_n_threads,
                root_m_threads_explicit=root_m_threads_explicit,
                root_n_threads_explicit=root_n_threads_explicit,
            ):
                return True
            continue
        if (
            mma_impl == "universal"
            and lhs_val.dtype == torch.float32
            and fp32_exact
            and k_subrange_in_root
            and lhs_val.ndim == 2
            and candidate.operands.rhs.source_fake.ndim == 2
            and not candidate.operands.has_leading_passthrough
            and simt_mma_requested(config)
            and simt_gemm_root_layout_supported(
                bm=bm,
                bn=bn,
                root_m_threads=root_m_threads,
                root_n_threads=root_n_threads,
                root_m_threads_explicit=root_m_threads_explicit,
                root_n_threads_explicit=root_n_threads_explicit,
            )
        ):
            # The exact fp32 SIMT GEMM owns the K loop (no K lanes); the
            # emitter raises rather than declining, so this agreement holds.
            return True
        if k_loop_is_subrange:
            continue
        if mma_impl != "universal" and _specialized_mma_root_threads_support_impl(
            mma_impl,
            bm=bm,
            bn=bn,
            root_m_threads=root_m_threads,
            root_n_threads=root_n_threads,
            root_m_threads_explicit=root_m_threads_explicit,
            root_n_threads_explicit=root_n_threads_explicit,
            epi_warp_groups=launch_epi_warp_groups_knob(config),
        ):
            if mma_impl == "tcgen05" and not ensure_tcgen05_fragment_epilogue_plan(
                fn,
                node,
                candidate,
                bm=bm,
                bn=bn,
                bk=bk,
                config=config,
            ):
                continue
            return True
    return False


def _attention_loop_shape(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> tuple[int, int, AttentionScorePlan] | None:
    """Recognize the flash-attention dataflow in a tile_n device loop.

    Matches BY SHAPE (not exclusivity), mirroring the spike sequence: a
    ``ForLoopGraphInfo`` over the key tile whose body holds a QK matmul
    (``aten.bmm.dtype``) feeding the online-softmax chain
    (``amax -> maximum -> sub -> exp2 -> sum``) and a ``baddbmm.default``
    (P@V) whose acc is the loop-carried output.

    Returns ``(bm, bn, score_plan)`` derived from the operand shapes -- bm =
    tile_m (query rows), bn = tile_n (key tile), score_plan = tile-local
    QK-score transforms plus head_dim -- or None when the loop is not the
    attention pattern or the validated config envelope is not met.

    This is deliberately strict so single-GEMM detection
    (``_detect_specialized_mma_loop``) is 100% untouched.
    """
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    # Attention binds to a 2-axis root grid (tile_b, tile_m), or the
    # flattened-rows form's one axis over ``B*H*M`` rows, plus one inner
    # tile_n device loop.
    if len(device_ir.grid_block_ids) != 1:
        return None
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) not in (1, 2):
        return None
    if len(block_ids) != 1 or any(bid in root_grid_ids for bid in block_ids):
        return None

    graph_info = None
    for candidate in fn.codegen.codegen_graphs:
        if isinstance(candidate, ForLoopGraphInfo) and candidate.block_ids == block_ids:
            graph_info = candidate
            break
    if graph_info is None:
        return None

    pattern = _attention_softmax_pattern_head_dim(
        graph_info.graph,
        kv_block_id=block_ids[0],
    )
    if pattern is None:
        return None
    # Operand facts without a fused body (fp8 operands on an odd KV tile count
    # or with the canonical V) or a K/V batch index off the root keep the
    # generic lowering; the search surface declines them the same way.
    if not _attention_pattern_fits_root(
        pattern, root_grid_ids, kv_block_id=block_ids[0]
    ):
        return None

    env = CompileEnvironment.current()

    # The flash kernel walks full 128-row/128-col tiles, so BOTH the query
    # (tile_m) and key/value (tile_n) sequence lengths must be exact multiples
    # of 128, and it derives num_kv from the QUERY length, so it only handles
    # SQUARE self-attention (query seq == key/value seq). Cross-attention
    # (q seq != kv seq, e.g. the 1024-query/512-kv test_attention_block_pointer
    # shape) would iterate the wrong number of KV tiles -> keep the scalar path.
    # Only statically-equal concrete sizes qualify: a dynamic (SymInt) seq is
    # rejected conservatively (size hints can collide for distinct unbacked
    # symbols). This mirrors the codegen's seq%128 gate at detection time,
    # where the device function arguments are not yet populated -- so the
    # detector never sets ``mma_mode`` / forces ``block=(128,1,1)`` on a config
    # codegen would reject.
    geometry = _attention_pattern_geometry(pattern, root_grid_ids, block_ids[0])
    if (
        geometry is None
        or geometry.q_seq != geometry.kv_seq
        or geometry.q_seq % 128 != 0
    ):
        return None
    # The flash kernel emits its OWN (batch*head, query-tile) program mapping; it
    # is only valid for the default flat pid with no L2 reordering. Persistent /
    # interleaved pids and l2_grouping remap program ids, so the emitted tile
    # indexing would address the wrong (batch, head, query-tile) -> reject.
    if config.pid_type != "flat":
        return None
    if any(grouping != 1 for grouping in config.l2_groupings):
        return None
    if any(order != [*range(len(order))] for order in config.loop_orders):
        return None
    if any(thread_count != 0 for thread_count in config.num_threads):
        return None
    cute_vector_widths = config.config.get("cute_vector_widths", [])
    if isinstance(cute_vector_widths, list) and any(
        width != 1 for width in cute_vector_widths
    ):
        return None

    # bm/bn from the config block sizes for tile_m (the root grid's last
    # axis) and tile_n (the device loop block), NOT from the grid sizes.
    bm = env.block_sizes[root_grid_ids[-1]].from_config(config)
    (bn,) = (env.block_sizes[bid].from_config(config) for bid in block_ids)
    if not isinstance(bm, int) or not isinstance(bn, int):
        return None
    # tile_b must be one (batch, head) per CTA -- the flash kernel maps the
    # b*h flatten one row per program. A tile_b block > 1 would pack multiple
    # (batch, head) rows per CTA and miscompute. Enforce what the docstrings
    # claim (previously unchecked). The flattened-rows form has no batch axis.
    if len(root_grid_ids) == 2:
        tile_b = env.block_sizes[root_grid_ids[0]].from_config(config)
        if not isinstance(tile_b, int) or tile_b != 1:
            return None
    return bm, bn, pattern.score_plan


def _detect_attention_bwd_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> bool:
    """True when the inner Q loop belongs to the fused backward-attention body.

    Gated on the bwd search surface flag (set at DeviceIR time by
    ``detect_flash_bwd_search_surface``) plus the validated config envelope
    (flat pid, no reorderings, 128x128 block sizes).
    """
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .cute_flash_bwd import match_attention_bwd

    env = CompileEnvironment.current()
    if not env.config_spec.cute_flash_bwd_search_enabled:
        return False
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) == 1 and len(device_ir.grid_block_ids[0]) == 3:
        return _detect_flex_attention_bwd_loop(fn, block_ids, config=config)
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 1:
        return False
    root_block_id = device_ir.grid_block_ids[0][0]
    if len(block_ids) != 1 or block_ids[0] == root_block_id:
        return False
    if not _attention_config_envelope_ok(config):
        return False
    bm = env.block_sizes[block_ids[0]].from_config(config)
    bn = env.block_sizes[root_block_id].from_config(config)
    if bm != 128 or bn != 128:
        return False
    match = match_attention_bwd(device_ir)
    if match is None:
        return False
    fn.cute_state.flash.bwd_match = match
    return True


def _detect_flex_attention_bwd_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> bool:
    """The FlexAttention backward envelope: every device loop of the q-head
    loop and the two Q-block list walks answers ``mma_mode`` from one
    whole-graph-set match, probed once per device function."""
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .cute_flash_bwd import _flash_bwd_gate_enabled
    from .flex_bwd_match import match_flex_attention_bwd
    from .flex_match import FLEX_PLAN_ENABLED

    # ``HELION_CUTE_FLASH_BWD=0`` disables the fused backward, dense and flex.
    if not FLEX_PLAN_ENABLED or not _flash_bwd_gate_enabled():
        return False
    device_ir = HostFunction.current().device_ir
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(block_ids) != 1 or block_ids[0] in root_grid_ids:
        return False
    if not _attention_config_envelope_ok(config):
        return False
    env = CompileEnvironment.current()
    state = fn.cute_state
    if not state.flash.flex_bwd_probed:
        state.flash.flex_bwd_probed = True
        state.flash.flex_bwd_match = match_flex_attention_bwd(
            fn.codegen.codegen_graphs, root_grid_ids, env=env
        )
    match = state.flash.flex_bwd_match
    if match is None or block_ids[0] not in match.loop_block_ids:
        return False
    if (
        env.block_sizes[match.q_block_id].from_config(config) != 128
        or env.block_sizes[match.kv_block_id].from_config(config) != 128
    ):
        return False
    if any(
        env.block_sizes[block_id].from_config(config) != 1
        for block_id in (match.b_block_id, match.hk_block_id)
    ):
        return False
    state.flash.bwd_match = match
    if state.flash.bwd_block_ids is None:
        state.flash.bwd_block_ids = list(match.loop_block_ids)
    return True


def _attention_config_envelope_ok(config: Config) -> bool:
    """The fused flash bodies emit their own program mapping and own the
    whole CTA: only the flat pid with no reordering, no L2 grouping, no
    explicit thread counts, no lane vectors and no epilogue subtiling (the
    store split rewrites the graphs the matchers read) is valid."""
    if config.pid_type != "flat":
        return False
    if config.config.get("epilogue_subtile"):
        return False
    if any(grouping != 1 for grouping in config.l2_groupings):
        return False
    if any(order != [*range(len(order))] for order in config.loop_orders):
        return False
    if any(thread_count != 0 for thread_count in config.num_threads):
        return False
    cute_vector_widths = config.config.get("cute_vector_widths", [])
    return not (
        isinstance(cute_vector_widths, list)
        and any(width != 1 for width in cute_vector_widths)
    )


def _detect_flex_attention_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> bool:
    """True when ``block_ids`` is one of the device loops of a matched
    FlexAttention envelope (``flex_match.match_flex_envelope``).

    The match runs once per device function on the codegen graph copies and
    is cached on ``cute_state``; the outer list loops and the inner KV loops
    of both walks all answer True so the SIMT planner never allocates thread
    axes for them. The config envelope mirrors the collapsed detector plus
    the unit b/h tiles and the 128x128 (tile_m, tile_n) pins.
    """
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .flex_match import FLEX_PLAN_ENABLED
    from .flex_match import match_flex_envelope

    if not FLEX_PLAN_ENABLED:
        return False
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) != 3 or len(block_ids) != 1 or block_ids[0] in root_grid_ids:
        return False
    env = CompileEnvironment.current()
    state = fn.cute_state
    if not state.flash.flex_probed:
        state.flash.flex_probed = True
        state.flash.flex_match = match_flex_envelope(
            fn.codegen.codegen_graphs, root_grid_ids, env=env
        )
    match = state.flash.flex_match
    if match is None or block_ids[0] not in match.loop_block_ids:
        return False
    if not _attention_config_envelope_ok(config):
        return False
    bm = env.block_sizes[match.q_block_id].from_config(config)
    bn = env.block_sizes[match.kv_block_id].from_config(config)
    if bm != 128 or bn != 128:
        return False
    if any(
        env.block_sizes[block_id].from_config(config) != 1
        for block_id in (match.b_block_id, match.h_block_id)
    ):
        return False
    # (The matcher admits the widths above the pair body's only where the
    # single-tile body's cooperative pair hosts the walk.)
    if not flash_head_dims_supported(match.head_dim, match.head_dim_v):
        return False
    state.flash.score_plan = match.plan
    state.flash.block_ids = list(match.loop_block_ids)
    return True


def _detect_attention_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> bool:
    """True when the tile_n loop is a fused tcgen05 flash-attention body.

    Gated behind ``HELION_CUTE_FLASH`` and the validated config envelope
    (tile_b block = 1, mma_tiler 128x128, head_dim in {64, 128}, fp16/bf16) so the
    default scalar-fallback path is unchanged while the path is incomplete.
    """
    if not _attention_flash_gate_enabled() or not _attention_flash_supported():
        return False
    from ..compile_environment import CompileEnvironment

    if CompileEnvironment.current().config_spec.cute_attention_generic_fallback_enabled:
        return False
    shape = _attention_loop_shape(fn, block_ids, config=config)
    if shape is None:
        # The FlexAttention envelope (3-axis grid, BlockMask list walks).
        if _detect_flex_attention_loop(fn, block_ids, config=config):
            return True
        # Softmax-free (gated) attention over jagged rows: same tcgen05 body
        # family, selected by the same config envelope (128-row tiles).
        from .cute_flash_gated import detect_gated_attention_loop

        gated = detect_gated_attention_loop(fn, block_ids, config=config)
        if gated is None:
            return False
        fn.cute_state.flash.gated_match = gated
        return True
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    root_block_ids = device_ir.grid_block_ids[0]
    bm, bn, score_plan = shape
    from .cute_flash import flash_attention_graph_lse_plan_valid

    if not flash_attention_graph_lse_plan_valid(
        fn,
        root_block_ids=root_block_ids,
        kv_block_id=block_ids[0],
        score_plan=score_plan,
    ):
        return False
    bk = score_plan.head_dim
    # Validated envelope mirrors the spike: 128x128 mma_tiler and one of the
    # fused head-dim pairs (64/64, 128/128, 192/128).
    if bm != 128 or bn != 128:
        return False
    if not flash_head_dims_supported(bk, score_plan.value_head_dim):
        return False
    if score_plan.asymmetric_head_dims:
        # Only the FA4 topology carries the pair: two 128-row query tiles per
        # CTA, so the sequence has to be a multiple of 256.
        env = CompileEnvironment.current()
        kv_seq = env.block_sizes[block_ids[0]].size
        if not isinstance(kv_seq, int) or kv_seq % 256 != 0:
            return False
        if score_plan.requires_ws_overlap:
            return False
    fn.cute_state.flash.score_plan = score_plan
    return True


def _is_mma_candidate_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
    grid_ids: set[int],
) -> bool:
    if not any(bid not in grid_ids for bid in block_ids):
        return False
    resolved_threads: list[int] = [
        num_threads
        if num_threads > 0
        else int(block_size)
        if isinstance(block_size, int)
        else 0
        for block_size, num_threads in zip(block_sizes, num_threads_config, strict=True)
    ]
    return _detect_mma_loop(
        fn,
        block_ids,
        block_sizes=block_sizes,
        num_threads_config=resolved_threads,
    )


def _loop_may_use_mma(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    """Whether the device loop over ``block_ids`` (nested for-loops included)
    has a matmul the MMA lowering admits."""
    from ..device_ir import iter_loop_graph_nodes
    from .cute_mma import analyze_cute_mma_node

    for node in iter_loop_graph_nodes(fn, block_ids):
        candidate = analyze_cute_mma_node(node, graphs=fn.codegen.codegen_graphs)
        if (
            candidate is not None
            and not candidate.requires_scalar_fallback
            and (candidate.with_acc or candidate.is_dot)
        ):
            return True
    return False


def _kernel_has_specialized_mma_path(
    fn: DeviceFunction,
    *,
    config: Config,
) -> bool:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    grid_ids = {
        bid for ids in HostFunction.current().device_ir.grid_block_ids for bid in ids
    }
    seen_block_ids: set[tuple[int, ...]] = set()
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = tuple(graph_info.block_ids)
        if block_ids in seen_block_ids or not any(
            bid not in grid_ids for bid in block_ids
        ):
            continue
        seen_block_ids.add(block_ids)
        block_sizes: list[int | torch.SymInt] = []
        for bid in block_ids:
            size = env.block_sizes[bid].from_config(config)
            if not isinstance(size, (int, torch.SymInt)):
                break
            block_sizes.append(size)
        else:
            if _detect_specialized_mma_loop(
                fn,
                list(block_ids),
                block_sizes=block_sizes,
                config=config,
            ):
                return True
    return False


def _loop_contains_atomic(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    """Whether the device loop over ``block_ids`` (nested for-loops included)
    calls an ``hl.atomic_*`` op."""
    from ...language.atomic_ops import ATOMIC_OPS as atomic_targets
    from ..device_ir import iter_loop_graph_nodes

    return any(
        node.op == "call_function" and node.target in atomic_targets
        for node in iter_loop_graph_nodes(fn, block_ids)
    )


# Leave broad headroom below the G1 sweep's 3600s subprocess timeout: budget
# checks happen between inline CuTe compile/benchmark units, then the selected
# config still has to compile, pass correctness, and run the final benchmark.
_CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS = 600


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend.

    One method per ``Backend`` hook this backend overrides; the bodies
    forward to the modules named in the module docstring.  A decision that
    depends on this backend belongs here or behind a hook with a no-op
    default in ``Backend``, never behind a backend-name comparison in core.
    """

    @classmethod
    def finalist_preference(cls, config: Config, source: Callable[[], str]) -> int:
        """A kernel that renders the generic early release of its programmatic
        dependents (``cute_pdl`` on the generic path, released at entry or
        from the last wave) ranks ahead of one that does not, among finalists
        the autotuner's final verification cannot tell apart, and one that
        also pads its shared memory to one wave's density
        (``cute_pdl_spread``, ``spread_rendered``) ranks ahead of both: an
        early launch of it never packs its CTAs onto the SMs its primary
        left idle, while its isolated timing reads the pad as a tie.

        The knob's own cost or gain is invisible to the kernel's isolated
        timing whenever a plain launch separates its back-to-back calls (the
        zero fill of an output: the calls are then not dependents of each
        other), while the kernel's successor in the real call sequence
        launches 0.3-1 us earlier behind a releasing primary (mamba2_ssd
        backward: dqkg alone reads 3.65 us with the knob on or off; the
        sequence reads 10.3 us per call with it on against 11.2 off).

        Whether a config renders that release is read off its generated
        module (``generic_release_rendered``), not off its keys: the tcgen05,
        warp-MMA, collective and block-scaled families own their wait and
        release (``passes.pdl.pdl_applies``; the tcgen05 kernels
        take the ``entry`` placement after their own waits,
        ``passes.pdl._tcgen05_early_pdl_release``), and on them every
        config ranks alike: their back-to-back timing sees the knob, and a
        ranking by the release there would break ties between unrelated
        knobs (matmul backward s2: a clc-persistent finalist 0.5% behind in
        the autotuner's regime, 7% behind in the catalogue's).  A config
        without the knob ranks last without a render.
        """
        if config.config.get("cute_pdl") is not True:
            return 2
        rendered = source()
        if not generic_release_rendered(rendered):
            return 2
        return 0 if spread_rendered(rendered) else 1

    def build_standalone_code(
        self,
        bound: BoundKernel[Any],
        import_lines: list[str],
        body_root: ast.Module,
    ) -> ast.Module:
        from .standalone import build_standalone_code

        return build_standalone_code(bound, import_lines, body_root)

    def collective_owns_tile(self, fn: DeviceFunction, block_id: int) -> bool:
        from ..compile_environment import CompileEnvironment
        from .grouped_row_union import physical_schedule
        from .grouped_row_union import schedule_supported
        from .tcgen05_config import facts

        env = CompileEnvironment.current()
        spec = env.config_spec._cute_tcgen05_config
        dense = facts._dense_aligned_n_plan(spec, fn.config.config)
        if (
            dense is not None
            and env.canonical_block_id(block_id) in dense.block_ids[:2]
        ):
            return (
                tuple(fn.resolved_block_size(axis) for axis in dense.block_ids)
                == dense.physical_tile
            )
        profile = physical_schedule(fn.config)
        axes = spec.matmul_block_ids
        return (
            profile is not None
            and facts.row_union_profile_supported(spec, profile.name)
            and axes is not None
            and env.canonical_block_id(block_id)
            in tuple(env.canonical_block_id(axis) for axis in axes[:2])
            and schedule_supported(fn.config, profile.source_tile)
            and tuple(fn.resolved_block_size(axis) for axis in axes)
            == profile.source_tile
        )

    def codegen_config(self, config: Config) -> Config:
        from ..compile_environment import CompileEnvironment
        from .tcgen05_config import facts

        state = CompileEnvironment.current().config_spec._cute_tcgen05_config
        return facts.project_dense_aligned_n_codegen(
            state, facts.project_row_union_codegen(state, config)
        )

    # Bump when config_value_priors() changes its candidate distribution.
    config_value_priors_version = 4

    @property
    def name(self) -> str:
        return "cute"

    @property
    def state(self) -> BackendState:
        return BackendState.BETA

    def reserved_device_names(self) -> frozenset[str]:
        # CuTe treats `_` as a write-only discard binding. Python host
        # locals can still supply values through a renamed device argument.
        return frozenset({"_"})

    @property
    def lowers_integer_powers(self) -> bool:
        return True

    @property
    def requires_proved_integer_powers(self) -> bool:
        return True

    def supports_intra_loop_barriers(self, *, in_divergent_control_flow: bool) -> bool:
        # A CuTe SIMT branch condition can vary per thread, and the
        # convergent ``sync_threads`` barrier deadlocks in a divergent
        # branch.
        return not in_divergent_control_flow

    @property
    def lowers_chunk_unbind(self) -> bool:
        return True

    def barrier_allows_pid_type(self, pid_type: str) -> bool:
        # CuTe launches the phases in order with ``pid_type="flat"``;
        # DeviceIR keeps ``flat`` only once it has proved those launches
        # (``ordered_phases_supported``).
        return pid_type == "flat"

    def local_fetch_add_fake_shape(
        self, target: torch.Tensor, index: list[object]
    ) -> torch.Tensor | None:
        from ..host_function import HostFunction

        if (
            target not in HostFunction.current().tensor_to_origin
            and target.dtype == torch.int32
            and target.ndim == 1
            and len(index) == 1
            and isinstance(index[0], torch.Tensor)
        ):
            # Local fetch-add returns one old value per logical index.
            # Normalizing a concrete index extent to a padded reduction block
            # here changes the meaning of downstream views (for example, flip
            # reverses padding).
            return torch.empty_like(index[0], dtype=target.dtype)
        return None

    def transform_launch_grid(self, fn: DeviceFunction, grid_expr: ast.AST) -> ast.AST:
        if (
            fn.cute_state.tcgen05.program_id_divisor > 1
            and not fn.cute_state.tcgen05.split_k_applied
        ):
            raise exc.InvalidConfig(
                f"{TCGEN05_SPLIT_K_CONFIG_KEY} > 1 requires the flat one-CTA "
                "tcgen05 GEMM pipeline, which this kernel did not take"
            )
        grid_multiplier = fn.cute_state.lanes.launch_grid_multiplier
        if grid_multiplier != 1:
            # A fused body running one CTA per (grid index, lane).
            assert isinstance(grid_expr, ast.Tuple) and len(grid_expr.elts) == 1
            grid_expr = expr_from_string(
                f"(({{grid}}[0]) * {grid_multiplier},)", grid=grid_expr
            )
        simt_cluster_n = max(
            fn.cute_state.lanes.simt_cluster_n, fn.cute_state.lanes.grid_cluster_n
        )
        if simt_cluster_n > 1:
            # The cluster splits each row across ``cluster_n`` CTAs on a new
            # trailing grid dim.  Only a 1-D grid can host the cluster dim.
            if not (isinstance(grid_expr, ast.Tuple) and len(grid_expr.elts) == 1):
                raise self.unsupported("cute_cluster_n > 1 requires a 1-D launch grid")
            grid_expr = expr_from_string(
                f"(*{{grid}}, {simt_cluster_n})", grid=grid_expr
            )
        return grid_expr

    @property
    def supports_augmented_assign(self) -> bool:
        return False

    @property
    def folds_static_program_counts(self) -> bool:
        return True

    @property
    def reduction_loops_allow_non_power_of_two(self) -> bool:
        return True

    @property
    def reduction_loops_allow_full_extent(self) -> bool:
        return True

    @property
    def merges_reduction_dims(self) -> bool:
        return True

    @property
    def requires_merged_reduction_dims(self) -> bool:
        return True

    @property
    def warns_fast_math_unused(self) -> bool:
        return True

    @property
    def supports_autotune_multi(self) -> bool:
        return True

    @property
    def supports_eager_prepared_call(self) -> bool:
        return True

    @property
    def pads_small_matmul_dims(self) -> bool:
        # The ordinary lowerings read and write exactly the extent (a tile
        # past a small M / N is declined or mis-folded, not masked), so the
        # frontend keeps every matmul dim's search ceiling at the extent.
        # The one lowering that runs a padded tile -- the ``cute_mma_dag``
        # warp mode over a ragged root axis (a 16-row MMA atom over M < 16:
        # zero-filled operand rows, zero side loads, guarded stores) --
        # raises that block's ceiling to its atom from its own facts
        # registration (``mma_dag_ragged_ceilings``).
        return False

    @property
    def dynamo_captures_kernel_launch(self) -> bool:
        return False

    def validate_environment(self) -> None:
        from .cutedsl_compat import check_cute_backend_requirements

        check_cute_backend_requirements()

    def rng_streams(self) -> tuple[str, ...]:
        return ("auto", "word0", "philox4")

    def config_value_priors(self, config_spec: ConfigSpec) -> dict[str, ValuePrior]:
        """Bias the random half of the initial population toward the config
        family that performs well on Blackwell tcgen05 kernels.

        This encodes, as a distribution, what the backend's former hardcoded
        per-shape seed configs all converged on: a 2-CTA, TMA-fed,
        role-local-monolithic, static-persistent, tvm-ffi launch with deep AB
        staging and a 4-warp epilogue. Keys a given kernel does not expose
        (e.g. the ``tcgen05_*`` keys on a pointwise/reduction kernel) are
        ignored, and values a fragment cannot represent are dropped, so the
        priors are safe for any cute kernel -- a non-matmul kernel just picks up
        the generic biases (TMA indexing, 8 warps) on whichever keys it has.
        """
        from ...autotuner.config_priors import weighted_choice
        from .strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
        from .strategies import TCGEN05_STRATEGY_CONFIG_KEY
        from .strategies import Tcgen05PersistenceModel
        from .strategies import Tcgen05Strategy
        from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE

        async_store_policy_weights: dict[object, float] = {"default": 1.0}
        if (
            config_spec is not None
            and config_spec.cute_l2_evict_last_store_policy_supported
        ):
            async_store_policy_weights = {"l2_evict_last": 4.0, "default": 1.0}

        priors: dict[str, ValuePrior] = {
            # Generic knobs shared by every cute kernel.
            "num_warps": weighted_choice({8: 4.0, 4: 2.0, 16: 1.0}),
            "num_stages": weighted_choice({4: 3.0, 3: 2.0, 2: 1.0}),
            "indexing": weighted_choice(
                {"tensor_descriptor": 4.0, "pointer": 1.0, "block_ptr": 1.0}
            ),
            "pid_type": weighted_choice(
                {
                    TCGEN05_TWO_CTA_SEED_PID_TYPE: 3.0,
                    "flat": 1.0,
                    "persistent_blocked": 1.0,
                }
            ),
            # In-place vector state updates: the five-stage/two-row schedule
            # is the measured B256 winner. Disabled remains represented so
            # small grids can avoid the shared-memory occupancy cost.
            "cute_async_load_stages": weighted_choice({5: 4.0, 0: 2.0, 4: 1.0, 3: 1.0}),
            "cute_async_load_lookahead": weighted_choice({4: 4.0, 3: 1.0, 2: 1.0}),
            "cute_async_load_group_rows": weighted_choice({2: 4.0, 4: 1.0}),
            "cute_async_load_cache": weighted_choice({"cg": 4.0, "ca": 1.0}),
            "cute_async_store_policy": weighted_choice(async_store_policy_weights),
            "cute_store_policy": weighted_choice(async_store_policy_weights),
            # The exact division forms are bitwise identical; the redo form
            # is the measured row-kernel winner, the plain form covers
            # kernels at the register cap (see hoist_loop_invariant_recip).
            "cute_exact_div_form": weighted_choice({"redo": 3.0, "plain": 1.0}),
            "cute_bf16x2_recurrence": weighted_choice({True: 4.0, False: 1.0}),
            "cute_proven_bounds": weighted_choice({False: 4.0, True: 1.0}),
            # tcgen05 / 2-CTA matmul knobs (absent on non-matmul kernels).
            "tcgen05_cluster_m": weighted_choice({2: 3.0, 1: 1.0}),
            "tcgen05_ab_stages": weighted_choice(
                {3: 3.0, 4: 2.0, 5: 1.0, 6: 1.0, 2: 1.0}
            ),
            "tcgen05_acc_stages": weighted_choice({2: 4.0, 1: 1.0}),
            "tcgen05_c_stages": weighted_choice({2: 3.0, 4: 2.0, 1: 1.0}),
            "tcgen05_num_epi_warps": weighted_choice({4: 3.0, 2: 1.0, 8: 1.0}),
            TCGEN05_STRATEGY_CONFIG_KEY: weighted_choice(
                {
                    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value: 3.0,
                    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value: 1.0,
                }
            ),
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: weighted_choice(
                {
                    Tcgen05PersistenceModel.STATIC_PERSISTENT.value: 3.0,
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value: 1.0,
                }
            ),
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: weighted_choice({True: 3.0, False: 1.0}),
        }
        if config_spec is not None and config_spec.cute_register_subblocks is not None:
            # The register sub-block family is admitted: keep the random
            # half of the population mostly in-family.
            priors["cute_register_subblocks"] = weighted_choice({"on": 3.0, "off": 1.0})
        if config_spec is not None and config_spec.cute_mma_dag is not None:
            # The warp-MMA recurrence owner is admitted: keep the random half
            # of the population mostly in-family (tensor-memory mode first
            # where the kernel admits it).
            facts = config_spec.cute_mma_dag_facts
            if facts is not None and facts.tcgen05 is not None:
                priors["cute_mma_dag"] = weighted_choice(
                    {"tcgen05": 4.0, "warp": 2.0, "off": 1.0}
                )
            else:
                priors["cute_mma_dag"] = weighted_choice({"warp": 4.0, "off": 1.0})
        return priors

    def autotune_config_is_viable(
        self, config_spec: ConfigSpec, config: Config
    ) -> bool:
        from .cute_mma_dag import AXIS_LEAD
        from .cute_mma_dag import AXIS_LOOP
        from .cute_mma_dag import AXIS_ROOT
        from .cute_mma_dag import MMA_DAG_AUX_KEY
        from .cute_mma_dag import MMA_DAG_AUX_REGISTER
        from .cute_mma_dag import MMA_DAG_CLUSTER_KEY
        from .cute_mma_dag import MMA_DAG_DERIVED_EXCHANGE
        from .cute_mma_dag import MMA_DAG_DERIVED_KEY
        from .cute_mma_dag import MMA_DAG_EPILOGUE_KEY
        from .cute_mma_dag import MMA_DAG_EPILOGUE_ROW
        from .cute_mma_dag import MMA_DAG_KEY
        from .cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
        from .cute_mma_dag import MMA_DAG_LOOKAHEAD_OFF
        from .cute_mma_dag import MMA_DAG_SPLIT_CLUSTER
        from .cute_mma_dag import MMA_DAG_SPLIT_KEY
        from .cute_mma_dag import MMA_DAG_TCGEN05
        from .cute_mma_dag import MMA_DAG_WARP
        from .cute_mma_dag import apply_mma_dag_cluster
        from .cute_mma_dag import mma_dag_config_reason

        facts = config_spec.cute_mma_dag_facts
        mode = config.config.get(MMA_DAG_KEY)
        if facts is None or mode not in (MMA_DAG_WARP, MMA_DAG_TCGEN05):
            return True
        # Pre-screen the register / shared-memory budget of the warp-MMA
        # dataflow owner (``normalize`` repairs or refuses the same configs)
        # under the config's own routes and layouts: the split epilogue
        # admits the 128-column tensor-memory tiles the row layout's budget
        # refuses (the indexer's 128 x 128 split at (512, 4096) screened
        # under the row layout never reached a benchmark), the lookahead
        # schedule doubles the dot slot, the pass route sizes the chains'
        # tiles.
        block_sizes = config.config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return True
        # Fixed blocks (a lead tile of one row) have no config slot: the
        # facts resolve them without a lookup.
        tunable = config_spec.block_sizes.valid_block_ids()
        sizes: dict[int, object] = {}
        for axis in facts.axes:
            if axis.kind not in (AXIS_ROOT, AXIS_LOOP, AXIS_LEAD):
                continue
            if axis.block_id not in tunable:
                if axis.kind == AXIS_LEAD:
                    # A lead fixed to one row by the kernel (``block_size=1``).
                    sizes[axis.block_id] = 1
                continue
            try:
                sizes[axis.block_id] = block_sizes[
                    config_spec.block_sizes.block_id_to_index(axis.block_id)
                ]
            except (KeyError, IndexError):
                return True
        if (
            apply_mma_dag_cluster(
                facts,
                sizes,
                config.config.get(MMA_DAG_CLUSTER_KEY, 1),
                config.config.get(MMA_DAG_SPLIT_KEY, MMA_DAG_SPLIT_CLUSTER),
            )
            is not None
        ):
            return False
        return (
            mma_dag_config_reason(
                facts,
                sizes,
                config.config.get("cute_mma_dag_warps"),
                config.config.get("cute_mma_dag_stages"),
                cast("str", mode),
                cast(
                    "str",
                    config.config.get(MMA_DAG_DERIVED_KEY, MMA_DAG_DERIVED_EXCHANGE),
                ),
                epilogue=cast(
                    "str", config.config.get(MMA_DAG_EPILOGUE_KEY, MMA_DAG_EPILOGUE_ROW)
                ),
                lookahead=cast(
                    "str",
                    config.config.get(MMA_DAG_LOOKAHEAD_KEY, MMA_DAG_LOOKAHEAD_OFF),
                ),
                aux=cast(
                    "str",
                    config.config.get(MMA_DAG_AUX_KEY, MMA_DAG_AUX_REGISTER),
                ),
            )
            is None
        )

    def pre_inductor_lowering(self, node: torch.fx.Node) -> Lowering | None:
        from ..aten_lowering import sort_lowering
        from ..aten_lowering import squeeze_lowering
        from .aten_lowering import NumericExtremumLowering
        from .uniform_comparison import UniformComparisonLowering
        from .uniform_comparison import match_uniform_float_gt

        if node.target is torch.ops.aten.sort.stable:
            # CuTe's rank/fragment sorting already breaks ties by input index.
            # Keep other backends' existing Inductor stable-sort lowering.
            return sort_lowering
        if node.target in (torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dims):
            # These overloads have the same logical-view semantics as squeeze.dim.
            # Register them before Inductor turns them into a ReinterpretView;
            # the CuTe handler uses the traced output shape, not a scalar dim.
            return squeeze_lowering
        if node.target in (torch.ops.aten.fmin.default, torch.ops.aten.fmax.default):
            return NumericExtremumLowering()
        if match_uniform_float_gt(node) is not None:
            return UniformComparisonLowering()
        return None

    def argreduce_uses_aten_lowering(self, node: torch.fx.Node) -> bool:
        from .argreduce import _should_use_cute_argreduce_lowering

        return _should_use_cute_argreduce_lowering(node)

    def emit_virtual_clone(
        self, ctx: LoweringContext, node: torch.fx.Node
    ) -> object | None:
        from .cute_reshape import codegen_cute_virtual_clone

        return codegen_cute_virtual_clone(ctx, node)

    def emit_completed_matmul_sum(
        self, ctx: LoweringContext, node: torch.fx.Node
    ) -> ast.AST | None:
        from .completed_matmul_sum import completed_matmul_sum_input

        return completed_matmul_sum_input(ctx, node)

    def reduction_active_block_id(
        self, state: CodegenState, node: torch.fx.Node, size: object
    ) -> int | None:
        from .active_blocks import match_active_reduction_block_id

        return match_active_reduction_block_id(state, node, size)

    def handler_op_overrides(self) -> HandlerOpOverrides:
        from .handler_ops import CuteHandlerOpOverrides

        return CuteHandlerOpOverrides()

    def deferred_node_codegen(
        self,
        cg: CodegenInterface,
        node: torch.fx.Node,
        operand_env: Mapping[torch.fx.Node, object],
    ) -> object | None:
        from .device_state import DEFERRED_TCGEN05_FRAGMENT_EPILOGUE

        cute_state = cg.device_function.cute_state
        if cute_state.has_tcgen05_fragment_epilogue_plan:
            if cute_state.is_deferred_tcgen05_fragment_epilogue_node(node):
                return DEFERRED_TCGEN05_FRAGMENT_EPILOGUE
            if (
                any(
                    operand_env.get(input_node) is DEFERRED_TCGEN05_FRAGMENT_EPILOGUE
                    for input_node in node.all_input_nodes
                )
                and cute_state.tcgen05_fragment_epilogue_plan_for_store(node) is None
            ):
                raise exc.BackendUnsupported(
                    self.name,
                    "deferred tcgen05 fragment epilogue escaped its committed store",
                )
        return None

    def exchanged_node_operands(
        self,
        cg: CodegenInterface,
        node: torch.fx.Node,
        operand_env: Mapping[torch.fx.Node, object],
    ) -> dict[torch.fx.Node, ast.AST]:
        # Local imports: at module level, cute.repeated_block_ids ->
        # completed_matmul_sum -> device_ir -> ``from helion import Config``
        # fails while helion/__init__ is still importing.
        from ..inductor_lowering import PointwiseLowering
        from ..rebound_dims import REBOUND_CHECK_TARGETS
        from .cute_reshape import check_pointwise_rebound_block_ids
        from .repeated_block_ids import check_repeated_block_ids
        from .view_subtile import check_view_coord_consumers

        check_repeated_block_ids(cg, node)
        check_view_coord_consumers(cg, node)
        if (
            isinstance(node.meta["lowering"], PointwiseLowering)
            or node.target in REBOUND_CHECK_TARGETS
        ):
            return check_pointwise_rebound_block_ids(cg, node, operand_env=operand_env)
        return {}

    def prepare_graph_codegen(self, cg: GenerateAST, graph: torch.fx.Graph) -> None:
        from .cute_mma import prepare_cute_collective_lane_loop_suppression
        from .simt_gemm import prepare_cute_simt_gemm_loads

        prepare_cute_collective_lane_loop_suppression(cg, graph)
        prepare_cute_simt_gemm_loads(cg, graph)

    def tile_loop_paths(
        self, fn: DeviceFunction
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        from ..host_function import HostFunction
        from .loop_nesting import tile_loop_paths

        return tile_loop_paths(
            HostFunction.current().device_ir, fn.codegen.codegen_graphs
        )

    def constexpr_live_graph_ids(self, fn: DeviceFunction) -> frozenset[int] | None:
        from ..host_function import HostFunction
        from .loop_nesting import constexpr_live_graph_ids

        return constexpr_live_graph_ids(
            HostFunction.current().device_ir, fn.codegen.codegen_graphs, fn
        )

    def constexpr_live_block_ids(
        self, fn: DeviceFunction, live_graph_ids: frozenset[int]
    ) -> frozenset[int]:
        from ..active_blocks import active_block_ids
        from ..compile_environment import CompileEnvironment
        from ..host_function import HostFunction

        return active_block_ids(
            [
                graph
                for graph in fn.codegen.codegen_graphs
                if graph.graph_id in live_graph_ids
            ],
            (
                block_id
                for ids in HostFunction.current().device_ir.grid_block_ids
                for block_id in ids
            ),
            CompileEnvironment.current(),
        )

    def shares_sibling_thread_axes(self) -> bool:
        return True

    def inactive_broadcast_alias_block_ids(
        self, fn: DeviceFunction, strategies: Sequence[TileStrategy]
    ) -> set[int]:
        from .active_blocks import inactive_broadcast_alias_block_ids

        return inactive_broadcast_alias_block_ids(fn, strategies)

    def roots_may_reserve_distinct_thread_axes(self) -> bool:
        return True

    def pointwise_op_is_pure_scalar(self, node: torch.fx.Node) -> bool:
        from ...language.inline_asm_ops import inline_asm_elementwise
        from .pure_lane_packets import is_pure_scalar_assembly

        if (
            node.target is not inline_asm_elementwise
            or len(node.args) != 6
            or node.args[5] != 1
        ):
            return False
        return is_pure_scalar_assembly(node.args[0], node.args[1], node.args[4])

    def rolls_free_iota_reductions(self) -> bool:
        return True

    def boundary_only_grid_block_ids(self, fn: DeviceFunction) -> set[int]:
        from ..compile_environment import CompileEnvironment
        from ..host_function import HostFunction
        from .loop_nesting import boundary_only_grid_blocks

        return boundary_only_grid_blocks(
            CompileEnvironment.current(),
            HostFunction.current().device_ir,
            fn.codegen.codegen_graphs,
        )

    def tile_loop_thread_count(
        self, fn: DeviceFunction, config: Config, inactive_grid_block_ids: set[int]
    ) -> int:
        # Budget simultaneously active grid/device-loop axes. Sibling
        # passes reuse axes; multiplying all their tile sizes would turn
        # otherwise legal threaded reductions into serial lane loops.
        from ..compile_environment import CompileEnvironment
        from ..host_function import HostFunction
        from .thread_budget import tile_loop_thread_count

        return tile_loop_thread_count(
            CompileEnvironment.current(),
            HostFunction.current().device_ir,
            fn.codegen.codegen_graphs,
            config,
            inactive_block_ids=inactive_grid_block_ids,
        )

    def packed_grid_block(self, config: Config) -> tuple[int, int] | None:
        from ..compile_environment import CompileEnvironment
        from .tile_packing import packed_grid_block

        return packed_grid_block(CompileEnvironment.current(), config)

    # Tensor-core policy of the shared per-thread planner
    # (``per_thread_planning.plan_per_thread_loop_strategy``).  The detectors
    # are this module's globals, read at call time (tests patch them here).

    def per_thread_mma_candidate_loop(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        *,
        block_sizes: Sequence[int | torch.SymInt],
        num_threads_config: Sequence[int],
        grid_ids: set[int],
    ) -> bool:
        return _is_mma_candidate_loop(
            fn,
            block_ids,
            block_sizes=block_sizes,
            num_threads_config=num_threads_config,
            grid_ids=grid_ids,
        )

    def per_thread_matmul_loop_has_mma_path(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> bool:
        forced_mma_impl = os.environ.get("HELION_CUTE_MMA_IMPL", "auto")
        return _kernel_has_specialized_mma_path(fn, config=config) or (
            _loop_may_use_mma(fn, block_ids)
            and not _loop_contains_atomic(fn, block_ids)
            and forced_mma_impl.strip().lower() != "auto"
        )

    def per_thread_mma_mode(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        config: Config,
        *,
        block_sizes: Sequence[int | torch.SymInt],
    ) -> bool:
        # Detect MMA-compatible K-loops: device loops containing addmm/mm
        # with float16/bfloat16 operands.
        from .cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
        from .cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY
        from .cute_warp_mma_gemm import detect_warp_mma_gemm

        if config.get(WARP_MMA_FAMILY_KEY) == MATMUL_FAMILY_WARP_MMA:
            # The register-MMA GEMM family: the dedicated codegen
            # emits the whole device body (``32 * warps`` threads);
            # a kernel that is not the plain GEMM it lowers fails
            # closed with the reason instead of running tcgen05.
            return detect_warp_mma_gemm(fn, block_ids, config=config)
        if _detect_attention_mma_loop(fn, block_ids, config=config):
            # Fused tcgen05 flash-attention: suppress synthetic lane
            # loops (mma_mode) and run a single 128-thread warpgroup
            # like the spike. The dedicated flash codegen emits the
            # whole device body; the FX-graph statement walk is bypassed.
            # The flex envelope records every loop it owns up front
            # (two nests); keep that list instead of the current loop.
            if fn.cute_state.flash.block_ids is None:
                fn.cute_state.flash.block_ids = list(block_ids)
            return True
        if _detect_attention_bwd_mma_loop(fn, block_ids, config=config):
            # Fused tcgen05 attention BACKWARD: the dedicated bwd
            # codegen emits the whole device body. The flex envelope
            # records every loop it owns up front; keep that list.
            if fn.cute_state.flash.bwd_block_ids is None:
                fn.cute_state.flash.bwd_block_ids = list(block_ids)
            return True
        return _detect_specialized_mma_loop(
            fn,
            block_ids,
            block_sizes=block_sizes,
            config=config,
        )

    def per_thread_root_thread_counts(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        config: Config,
        *,
        block_sizes: Sequence[int | torch.SymInt],
        num_threads_config: Sequence[int],
        original_num_threads_config: Sequence[int],
    ) -> list[int] | None:
        specialized_mma_plan = _kernel_specialized_mma_plan(fn, config=config)
        if specialized_mma_plan is None or specialized_mma_plan.impl != "tcgen05":
            return None
        from ..compile_environment import CompileEnvironment
        from .cute_mma import _tcgen05_root_m_threads
        from .cute_mma import _tcgen05_root_n_threads

        env = CompileEnvironment.current()
        m_axis = block_ids.index(specialized_mma_plan.m_block_id)
        n_axis = block_ids.index(specialized_mma_plan.n_block_id)
        m_block_size = block_sizes[m_axis]
        n_block_size = block_sizes[n_axis]
        # The generic SIMT budget above can resolve an automatic
        # M axis before this specialized plan is known. Preserve
        # the original auto setting: that temporary width is not
        # an explicit request for multiple lanes per warp slot.
        # TCgen05's role launch multiplies this width by its warp
        # count, so retaining (for example) 64 would double it.
        requested_m_threads = env.config_spec.num_threads.config_get(
            config.num_threads, specialized_mma_plan.m_block_id, 0
        )
        root_m_threads = (
            _tcgen05_root_m_threads(int(m_block_size), int(n_block_size))
            if requested_m_threads == 0
            and isinstance(m_block_size, int)
            and isinstance(n_block_size, int)
            else num_threads_config[m_axis]
        )
        root_n_threads = (
            _tcgen05_root_n_threads(
                int(n_block_size),
                epi_warp_groups=launch_epi_warp_groups_knob(config),
            )
            if original_num_threads_config[n_axis] == 0
            and isinstance(n_block_size, int)
            else num_threads_config[n_axis]
        )
        result = list(num_threads_config)
        result[m_axis] = root_m_threads
        result[n_axis] = root_n_threads
        return result

    def free_iota_reductions(
        self, graphs: list[GraphInfo]
    ) -> Mapping[torch.fx.Node, tuple[torch.fx.Node, ...]]:
        from ..compile_environment import CompileEnvironment
        from .reduction_coordinates import free_iota_reductions

        return free_iota_reductions(CompileEnvironment.current(), graphs)

    def customize_ast(self, hf: HostFunction) -> bool:
        """CuTe-specific AST rewrites that rewrite high-level patterns into
        equivalent forms that compile to materially faster code.

        Currently:
          * ``rewrite_online_to_3pass`` rewrites the online two-pass
            softmax pattern into the 3-pass form (max-only, then
            sum-only, then consume).  The 3-pass form's two reductions
            are independent and compile to a more efficient layout on
            the CuTe backend.
          * ``stage_unaligned_matmul_operands`` pads matmul operands whose
            strides fail the TensorMap proof through a host copy so the
            tcgen05 pipeline keeps its TMA producers.
        """
        from .online_to_3pass import rewrite_online_to_3pass

        rewritten = rewrite_online_to_3pass(hf)
        from ..compile_environment import CompileEnvironment

        plan = cute_env_state(CompileEnvironment.current()).fission_plan
        if plan is not None:
            from .materialized_fission import apply_materialized_fission

            apply_materialized_fission(hf, plan)
            rewritten = True

        from .full_slice_matmul import stripmine_full_slice_matmuls

        rewritten |= stripmine_full_slice_matmuls(hf) > 0

        from .segmented_matmul import normalize_segmented_matmuls

        rewritten |= normalize_segmented_matmuls(hf)

        from .flatten_nested_reductions import flatten_nested_reductions

        rewritten |= flatten_nested_reductions(hf) > 0

        # Last: the staging statements it prepends are not part of the pure
        # host preludes the passes above analyze.
        from .stage_aligned_operands import stage_unaligned_matmul_operands

        rewritten |= stage_unaligned_matmul_operands(hf)
        return rewritten

    def decomp_table_overrides(
        self, table: dict[torch._ops.OpOverload, Callable[..., object]]
    ) -> None:
        # CuTe lowers these native numeric extrema directly. Preserve their
        # distinction from NaN-propagating minimum/maximum during tracing.
        table.pop(torch.ops.aten.fmin.default, None)
        table.pop(torch.ops.aten.fmax.default, None)

    def rewrite_device_ir(
        self,
        stage: DeviceIRStage,
        func: HostFunction | None,
        device_ir: DeviceIR,
        analysis: DeviceIRAnalysis | None = None,
    ) -> None:
        from .ir_passes import rewrite_device_ir

        rewrite_device_ir(stage, func, device_ir, analysis)

    def register_search_surface(
        self, device_ir: DeviceIR, config_spec: ConfigSpec
    ) -> None:
        from .search_surface import register_search_surface

        register_search_surface(device_ir, config_spec)

    def register_reduction_loop_config_slots(
        self,
        env: CompileEnvironment,
        block_id: int,
        size_hint: int,
    ) -> None:
        from ..device_ir import DeviceIR
        from .search_surface import admit_register_tile_reduction

        admit_register_tile_reduction(env, DeviceIR.current(), block_id)

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..device_ir import RootGraphInfo
        from .block_scaled_mma import plan_block_scaled
        from .chunk_prepare import plan_chunk_prepare
        from .chunk_recurrence import plan_chunk_recurrence
        from .cute_mma_dag import plan_mma_dag
        from .direct_affine_candidate import discover_direct_affine_candidates
        from .execution_plans import plan_execution
        from .fixed_token_rank1_recurrence import plan_fixed_token_rank1_recurrence
        from .gdn_recurrence import plan_gdn_recurrence
        from .literal_dims import annotate_literal_dims
        from .register_region import plan_register_region
        from .selection.radix_selection import plan_radix_selection_root
        from .selection.topk import plan_topk_root
        from .single_token_rank1_recurrence import plan_single_token_rank1_recurrence
        from .split_single_token_rank1_recurrence import (
            plan_split_single_token_rank1_recurrence,
        )
        from .view_subtile import annotate_view_subtiles

        device_function = DeviceFunction.current()
        device_function.cute_state.root_plans.radix_selection_plan = (
            plan_radix_selection_root(graphs, tile_strategy)
        )
        if device_function.cute_state.root_plans.radix_selection_plan is not None:
            return
        device_function.cute_state.root_plans.topk_plan = plan_topk_root(
            graphs, tile_strategy
        )
        if device_function.cute_state.root_plans.topk_plan is not None:
            return
        device_function.cute_state.root_plans.register_region_plan = (
            plan_register_region(graphs, tile_strategy)
        )
        if device_function.cute_state.root_plans.register_region_plan is not None:
            return
        direct_affine_requested = (
            config.cute_affine_scan_schedule != DIRECT_AFFINE_ORDINARY_SCHEDULE
        )
        if direct_affine_requested:
            direct_affine_candidates = discover_direct_affine_candidates(graphs)
            device_function.cute_state.root_plans.direct_affine_candidates = (
                direct_affine_candidates
            )
            if not CompileEnvironment.current().settings.fast_math:
                raise self.unsupported("direct affine scan requires fast_math=True")
            root_graphs = tuple(
                graph for graph in graphs if isinstance(graph, RootGraphInfo)
            )
            if (
                len(direct_affine_candidates) != 1
                or len(root_graphs) != 1
                or direct_affine_candidates[0].graph_id != root_graphs[0].graph_id
            ):
                raise self.unsupported(
                    "direct affine scan requires one compatible single-root region",
                )
            annotate_view_subtiles(graphs, config)
            annotate_literal_dims(graphs, config)
            plan_execution(graphs, tile_strategy)
            return

        device_function.cute_state.root_plans.direct_affine_candidates = ()
        from .cute_register_subblocks import plan_register_subblocks

        device_function.cute_state.root_plans.register_subblocks_plan = (
            plan_register_subblocks(graphs, config)
        )
        if device_function.cute_state.root_plans.register_subblocks_plan is not None:
            return
        plan_block_scaled(graphs)
        if DeviceFunction.current().cute_state.root_plans.block_scaled_plan is not None:
            return
        plan_chunk_prepare(graphs, tile_strategy)
        if (
            DeviceFunction.current().cute_state.root_plans.chunk_prepare_plan
            is not None
        ):
            return
        plan_chunk_recurrence(graphs, tile_strategy)
        if (
            DeviceFunction.current().cute_state.root_plans.chunk_recurrence_plan
            is not None
        ):
            return
        plan_gdn_recurrence(graphs, tile_strategy)
        if (
            DeviceFunction.current().cute_state.root_plans.gdn_recurrence_plan
            is not None
        ):
            return
        device_function.cute_state.root_plans.mma_dag_plan = plan_mma_dag(
            graphs, config
        )
        if device_function.cute_state.root_plans.mma_dag_plan is not None:
            return
        plan_single_token_rank1_recurrence(graphs, tile_strategy)
        if (
            DeviceFunction.current().cute_state.root_plans.single_token_rank1_plan
            is not None
        ):
            return
        split_t1_plan = plan_split_single_token_rank1_recurrence(graphs, tile_strategy)
        DeviceFunction.current().cute_state.root_plans.split_single_token_rank1_plan = (
            split_t1_plan
        )
        if split_t1_plan is not None:
            return

        plan_fixed_token_rank1_recurrence(graphs, tile_strategy)
        if device_function.cute_state.root_plans.fixed_token_rank1_plan is not None:
            return
        annotate_view_subtiles(graphs, config)
        annotate_literal_dims(graphs, config)
        plan_execution(graphs, tile_strategy)

    def finalize_device_body(
        self, body: list[ast.stmt], ctx: DeviceBodyContext
    ) -> tuple[list[ast.stmt], list[ast.arg]]:
        # The pass tables live in ``passes.pipeline``, imported at call time:
        # ``pipeline`` imports ``compile_environment`` and ``device_function``
        # at module level, and this module is loaded through
        # ``backend_registry`` (imported at the top of ``compile_environment``)
        # while ``compile_environment`` is still initializing.
        from .passes import pdl
        from .passes.pipeline import run_device_body_passes

        fn = ctx.device_function
        body, late = pdl.prepare_device_body(fn, body, ctx.body_start)
        body, extra_args = run_device_body_passes(body, ctx)
        if late is not None:
            body = pdl.place_late_wait(body, late)
        return body, extra_args

    def finalize_kernel_def(
        self,
        fn_def: ast.FunctionDef,
        prefix: list[ast.stmt],
        ctx: DeviceBodyContext,
    ) -> list[ast.stmt]:
        from .passes import launch_attributes
        from .passes import pdl
        from .passes.pipeline import run_kernel_def_passes

        run_kernel_def_passes(fn_def, prefix, ctx)
        fn = ctx.device_function
        pdl.insert_spread_pad(fn, fn_def)
        statements = launch_attributes.module_statements(fn)
        launch_attributes.project_launch_block(fn)
        return statements

    def tensors_proven_disjoint(
        self, env: CompileEnvironment, left: torch.Tensor, right: torch.Tensor
    ) -> bool:
        from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint

        return runtime_tensors_are_proven_disjoint(env, left, right)

    def new_env_state(self) -> CuteEnvState:
        return new_env_state()

    def new_codegen_state(self) -> CuteCodegenState:
        return new_codegen_state()

    def new_device_state(self, config: Config) -> CuteDeviceFunctionState:
        return new_device_state(config)

    def root_emitters(self) -> tuple[RootEmitter, ...]:
        from .emitters import ROOT_EMITTERS

        return ROOT_EMITTERS

    def kernel_generators(self) -> tuple[KernelGenerator, ...]:
        from .kernel_generators import KERNEL_GENERATORS

        return KERNEL_GENERATORS

    def retry_policy(self) -> RetryPolicy:
        from .kernel_generators import REGENERATION

        return REGENERATION

    def reference_override(
        self, function: object, args: tuple[object, ...]
    ) -> tuple[bool, object]:
        from ..compile_environment import CompileEnvironment
        from .philox_stream import reference_override

        policy = CompileEnvironment.current().settings.cute_rng_stream
        if policy in ("auto", "philox4"):
            return reference_override(function, args, strict=policy == "philox4")
        return False, None

    def validate_implicit_rng_reference(self) -> None:
        from ..compile_environment import CompileEnvironment

        if CompileEnvironment.current().settings.cute_rng_stream == "philox4":
            raise self.unsupported("philox4 supports explicit uniform hl.rand only")

    def autotuner_heuristics(self) -> tuple[AutotunerHeuristicType, ...]:
        # Imported here: the heuristics pull the config spec and most of the
        # backend, which import this module first.
        from .autotune.heuristics import CUTE_HEURISTICS

        return CUTE_HEURISTICS

    def seed_fact_heuristics(self) -> tuple[AutotunerHeuristicType, ...]:
        from .autotune.heuristics import CUTE_SEED_FACT_HEURISTICS

        return CUTE_SEED_FACT_HEURISTICS

    def register_seed_facts(
        self, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from .autotune.heuristics import register_seed_facts

        return register_seed_facts(env, device_ir)

    def postprocess_seed_configs(
        self, env: CompileEnvironment, device_ir: DeviceIR, configs: list[Config]
    ) -> list[Config]:
        from .autotune.heuristics import postprocess_seed_configs

        return postprocess_seed_configs(env, device_ir, configs)

    def register_coverage_groups(
        self, env: CompileEnvironment, device_ir: DeviceIR
    ) -> None:
        from .autotune.heuristics import register_coverage_groups

        register_coverage_groups(env, device_ir)

    # -- runtime hooks: see ``cute/runtime_hooks.py`` -------------------------

    def on_bind(
        self,
        bound: BoundKernel[Any],
        args: tuple[object, ...],
        *,
        rewrite_ast: bool,
    ) -> None:
        from . import runtime_hooks

        runtime_hooks.plan_bound_kernel(bound, args, rewrite_ast=rewrite_ast)

    def host_structural_facts(
        self, host_function: HostFunction, config_spec: ConfigSpec
    ) -> None:
        from . import runtime_hooks

        runtime_hooks.record_host_structural_facts(host_function, config_spec)

    def before_codegen(self, bound: BoundKernel[Any], config: Config) -> None:
        from . import runtime_hooks

        runtime_hooks.begin_codegen(bound)

    def after_codegen(self, bound: BoundKernel[Any], config: Config) -> None:
        from . import runtime_hooks

        runtime_hooks.finish_codegen(bound, config)

    def after_compile(self, bound: BoundKernel[Any], config: Config) -> None:
        from . import runtime_hooks

        runtime_hooks.register_compiled(bound, config)

    def decorator_settings(self, settings: Settings) -> list[str]:
        from . import runtime_hooks

        return runtime_hooks.decorator_settings(settings)

    def structural_policy_type(self) -> type[CuteStructuralPolicy]:
        from . import runtime_hooks

        return runtime_hooks.structural_policy_type()

    def llm_prompt_profile(self) -> LlmPromptProfile:
        from .autotune.llm_profile import CuteLlmPromptProfile

        return CuteLlmPromptProfile()

    def final_rebenchmark_top_k(self) -> int | None:
        # The flash-attention search surface has a wide config space where
        # verifying more finalists materially improves the final pick; the
        # other backends do not need the extra rebenchmark cost (it ~2.5x'd
        # autotune wall-time on cheap kernels).
        return 32

    def final_rebenchmark_isolated(self) -> bool:
        # The interleaved bench lets L2 cache-POLICY state leak between
        # candidates that read the same input tensors (an ``l2_last``
        # candidate pins its inputs in L2 and every rival free-rides on the
        # hits, so the config CAUSING the speedup can never out-measure the
        # others).  Isolated mode times each finalist in its own steady
        # window -- the same regime the deployment-style do_bench measures --
        # with suspicious-result confirmation guarding against thermal drift
        # between windows.
        return True

    def config_spec_extension(self) -> BackendSearchSurface:
        # Imported here: the surface pulls the config spec and most of the
        # backend, which import this module first.
        from .autotune.search_surface import CUTE_SEARCH_SURFACE

        return CUTE_SEARCH_SURFACE

    def supports_config_key(self, key: str) -> bool:
        if key == "reduction_fusion":
            # The CuTe reduction emitters fuse on their own terms.
            return False
        return super().supports_config_key(key)

    def cross_loop_pipelines(self) -> tuple[str, ...]:
        # Implicit tile dependencies lower like explicit ``hl.barrier()``
        # phases (persistent phase loops + grid barrier).  The static/dynamic
        # pipelines' readiness counters, task events and outlined roots are
        # Triton codegen.
        return ("barrier",)

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        if (
            inductor_dtype := CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        ) is not None:
            return inductor_dtype
        if dtype is torch.float4_e2m1fn_x2:
            # PyTorch's shell dtype stores two E2M1 values in one byte.  CuTe
            # does not support scalar dereference for its 4-bit type yet, so
            # SIMT scalar loads treat the tensor as raw byte storage.
            return "cutlass.Uint8"
        if dtype is torch.float8_e8m0fnu:
            return "cutlass.Float8E8M0FNU"
        if dtype is torch.uint32:
            return "cutlass.Uint32"
        if dtype is torch.uint64:
            return "cutlass.Int64"

        raise ValueError(f"Unsupported dtype for Cute backend: {dtype}")

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype in (torch.float16, torch.bfloat16):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

    def supports_precompile(self) -> bool:
        # The CuTe DSL does not expose a Triton-style precompile entry point;
        # the autotuner has to compile + benchmark each config inline.
        return False

    def probe_long_autotune_kernels(self, config_spec: ConfigSpec) -> bool:
        # A SIMT default config can run 1000x slower than the kernel's best
        # (an fp8 attention default at 1.5-11 s per launch): the timer's
        # five-launch estimate plus its warmup then exceeds the benchmark
        # timeout, the default and every seed are dropped, and the search
        # ends in NoConfigFound. A launch that already exceeds the timing
        # window is its own sample (the regular path times one repetition of
        # such a kernel too), so probing first costs nothing in precision and
        # keeps a slow seed inside the timeout.
        return True

    def supports_parallel_render(self) -> bool:
        # Code generation only reads the device facts the render pool
        # captures (the MMA-support probe's current device, device properties
        # for shared-memory and SM budgets); the DSL JIT runs at first call in
        # the parent. Forked workers render byte-identical source.
        return True

    def setup_compile_cache_dir(self, device_index: int) -> None:
        if "CUTE_DSL_CACHE_DIR" not in os.environ:
            from ...autotuner.local_cache import helion_cute_cache_dir

            cute_dir = helion_cute_cache_dir(device_index)
            os.environ["CUTE_DSL_CACHE_DIR"] = cute_dir
            log.debug("Set CUTE_DSL_CACHE_DIR=%s", cute_dir)

    def make_ephemeral_cache(
        self,
    ) -> contextlib.AbstractContextManager[None] | None:
        if self.keep_compile_cache_requested():
            return None
        return self._ephemeral_cute_cache()

    @contextlib.contextmanager
    def _ephemeral_cute_cache(self) -> Generator[None, None, None]:
        """Redirect the CuTe DSL on-disk cache to a temporary dir during
        autotuning so candidate compilations don't pollute the real cache.

        The winning config's artifact is re-persisted from memory into the
        real cache afterward (see :meth:`finalize_ephemeral_cache`).
        """
        saved = os.environ.get("CUTE_DSL_CACHE_DIR")
        with tempfile.TemporaryDirectory(prefix="helion_cute_autotune_") as ephemeral:
            os.environ["CUTE_DSL_CACHE_DIR"] = ephemeral
            log.debug("Ephemeral CuTe cache: %s", ephemeral)
            try:
                yield
            finally:
                if saved is not None:
                    os.environ["CUTE_DSL_CACHE_DIR"] = saved
                else:
                    os.environ.pop("CUTE_DSL_CACHE_DIR", None)

    def finalize_ephemeral_cache(
        self, bound_kernel: BoundKernel[Any], config: Config
    ) -> None:
        """Persist the winning config's compiled artifact into the real cache.

        Candidate artifacts died with the ephemeral dir, but the winner's
        launcher still holds the compiled module in memory and the disk-cache
        key excludes ``CUTE_DSL_CACHE_DIR``, so re-persisting from memory
        writes the exact artifact a later process will look up.  Launchers and
        compile-cache entries are kept so the winner launches without
        recompiling.
        """
        from ...runtime.config import Config

        compiled_fn = bound_kernel._compile_cache.get(config)
        if compiled_fn is None:
            # The autotuner may return a minimized config (default values
            # stripped); the compiled entry is keyed by the full config.
            default = bound_kernel.config_spec.default_config()
            # pyrefly: ignore [bad-argument-type]
            full_config = Config(**(default.config | config.config))
            compiled_fn = bound_kernel._compile_cache.get(full_config)
        if compiled_fn is None:
            return
        kernels = self._compiled_kernels(compiled_fn, bound_kernel.kernel.name)
        if not kernels:
            return
        device_index = (
            bound_kernel.env.device.index
            if bound_kernel.env.device.index is not None
            else 0
        )
        # The ephemeral context restored CUTE_DSL_CACHE_DIR on exit; this sets
        # the real per-device dir when the user did not provide one.
        self.setup_compile_cache_dir(device_index)
        for cute_kernel in kernels:
            launchers = getattr(cute_kernel, "_helion_cute_compiled_launchers", {})
            for launcher in launchers.values():
                launcher.persist_compiled()

    @staticmethod
    def _compiled_kernels(compiled_fn: object, kernel_name: str) -> tuple[object, ...]:
        """The CuTe kernel objects the generated host function launches: the
        kernel proper and the zero-fill kernels its host zero fills launch
        as under ``cute_pdl`` (``host_zero_fill_launches``, named
        ``_helion_zero_fill_<n>_helion_<kernel_name>``), or the bundle a
        multi-stage module declares (``_helion_cute_kernels``)."""
        bundled = getattr(compiled_fn, "_helion_cute_kernels", None)
        if isinstance(bundled, tuple):
            return bundled
        fn_globals = getattr(compiled_fn, "__globals__", None)
        if not isinstance(fn_globals, dict):
            return ()
        name = f"_helion_{kernel_name}"
        kernel = fn_globals.get(name)
        if kernel is None:
            return ()
        fills = sorted(
            key
            for key in fn_globals
            if key.startswith(ZERO_FILL_KERNEL_PREFIX)
            and key.endswith(f"_{name.lstrip('_')}")
        )
        return (kernel, *(fn_globals[key] for key in fills))

    def compiled_cache_key(
        self, bound_kernel: BoundKernel[Any], compiled_fn: object
    ) -> str | None:
        keys = []
        for cute_kernel in self._compiled_kernels(
            compiled_fn, bound_kernel.kernel.name
        ):
            launchers = getattr(cute_kernel, "_helion_cute_compiled_launchers", {})
            key = next(
                (
                    key
                    for launcher in launchers.values()
                    if (key := getattr(launcher, "_cache_key", None)) is not None
                ),
                None,
            )
            if key is None:
                return None
            keys.append(key)
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        return hashlib.sha256(repr(tuple(keys)).encode("utf-8")).hexdigest()

    def annotate_compiled_module(
        self, module: object, source: str, kernel_name: str
    ) -> None:
        compiled_fn = getattr(module, kernel_name, None)
        kernels = self._compiled_kernels(compiled_fn, kernel_name)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        # Shared stage modules retain their own source hash. Two bundles may
        # reuse the same stage while differing in another stage's configuration.
        targets = (
            (compiled_fn,)
            if isinstance(getattr(compiled_fn, "_helion_cute_kernels", None), tuple)
            else (compiled_fn, *kernels)
        )
        for target in targets:
            with contextlib.suppress(AttributeError, TypeError):
                target._helion_cute_source_hash = source_hash  # type: ignore[attr-defined]

    def generated_source_hash(self, compiled_fn: object) -> str | None:
        source_hash = getattr(compiled_fn, "_helion_cute_source_hash", None)
        if isinstance(source_hash, str):
            return source_hash
        fn_name = getattr(compiled_fn, "__name__", None)
        if not isinstance(fn_name, str):
            return None
        kernels = self._compiled_kernels(compiled_fn, fn_name)
        # One module, one hash: the kernel proper and its zero-fill kernels
        # carry the same one (``annotate_compiled_module``).
        hashes = {
            getattr(cute_kernel, "_helion_cute_source_hash", None)
            for cute_kernel in kernels
        }
        if len(hashes) != 1:
            return None
        (source_hash,) = hashes
        return source_hash if isinstance(source_hash, str) else None

    def should_deduplicate_generated_sources(self, config_spec: ConfigSpec) -> bool:
        return config_spec.cute_flash_search_enabled

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        # Exceptions raised from inside the cute/cutlass DSL during compile or
        # launch are expected when an invalid config is tried; treat them as
        # benign so the autotuner moves on. Anything else (AssertionError,
        # NameError/TypeError/AttributeError from generated code, ...) is
        # almost certainly a real Helion bug and is surfaced at warn level so
        # it gets noticed without terminating tuning.
        cls = type(err)
        module = getattr(cls, "__module__", "") or ""
        if module.startswith(("cutlass", "cute")):
            return "debug"
        if isinstance(err, Exception):
            return "warn"
        return None

    # Note: this backend intentionally does NOT override get_do_bench /
    # get_interleaved_bench. CUDA-event timing once mis-read CuTe launches
    # (~5ms reported for 250ms kernels) because the pre-compiled-launcher
    # per-call ``@cute.jit`` dispatch path spent ~200ms on the host per
    # launch; with ``_CompiledCuteLauncher`` kernels launch on the Torch
    # current stream and event timing matches synchronized wall-clock timing
    # to <0.1ms on B200 across flash families (including CLC+PDL and
    # multi-second launches). Event timing avoids charging per-launch host
    # overhead to the kernel, which the wall path does.

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        original_budget = bound_kernel.settings.autotune_budget_seconds
        # "full" should mean the search algorithms run to completion unless the
        # caller explicitly provides a budget. Keep the defensive default budget
        # only for cheaper efforts where bounded local iteration is expected.
        if (
            bound_kernel.settings.autotune_budget_seconds is None
            and bound_kernel.settings.autotune_effort != "full"
        ):
            bound_kernel.settings.autotune_budget_seconds = (
                _CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS
            )
        try:
            return super().autotune(bound_kernel, args, force=force, **kwargs)
        finally:
            bound_kernel.settings.autotune_budget_seconds = original_budget

    @property
    def function_decorator(self) -> str:
        return "cute.kernel"

    @property
    def constexpr_type(self) -> str:
        return "cutlass.Constexpr"

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

    @property
    def default_launcher_name(self) -> str:
        return "_default_cute_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "operator": "import operator",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "cutlass": "import cutlass",
            "cute": "import cutlass.cute as cute",
            "ir": "from cutlass._mlir import ir",
            "mlir_math": "from cutlass._mlir.dialects import math as mlir_math",
            "_default_cute_launcher": "from helion.runtime import default_cute_launcher as _default_cute_launcher",
            "_cute_checked_block_dims": "from helion._compiler.cute.thread_budget import checked_thread_block_dims as _cute_checked_block_dims",
            "_next_power_of_2": "from helion._utils import next_power_of_2 as _next_power_of_2",
            "_cute_argreduce_index": "from helion._compiler.cute.reduce_helpers import _cute_argreduce_index",
            "_cute_nan_max": "from helion._compiler.cute.reduce_helpers import _cute_nan_max",
            "_cute_nan_min": "from helion._compiler.cute.reduce_helpers import _cute_nan_min",
            "_helion_pdl_spread_keep": "from helion._compiler.cute._pdl_spread_runtime import keep_smem_live as _helion_pdl_spread_keep",
            "_cute_aux_copy_layout": "from helion._compiler.cute.aux_copy_layout import select_aux_copy_layout as _cute_aux_copy_layout",
            "_cute_gelu_erf_exact_f32x2": (
                "from helion._compiler.cute.epilogue_helpers import "
                "gelu_erf_exact_f32x2 as _cute_gelu_erf_exact_f32x2"
            ),
            "_cute_sigmoid_approx_ftz_f32": (
                "from helion._compiler.cute.epilogue_helpers import "
                "sigmoid_approx_ftz_f32 as _cute_sigmoid_approx_ftz_f32"
            ),
            "_cute_mma_f16": "from helion._compiler.cute._flash_row_mma_runtime import mma_f16 as _cute_mma_f16",
            "_cute_mma_bf16": "from helion._compiler.cute._flash_row_mma_runtime import mma_bf16 as _cute_mma_bf16",
            "_cute_cp_async_shared_global": (
                "from cutlass.experimental.primitives import "
                "cp_async_shared_global as _cute_cp_async_shared_global"
            ),
            "_cute_ring_cp_async_16": "from helion._compiler.cute._flash_row_mma_runtime import cp_async_16 as _cute_ring_cp_async_16",
            "_cute_ring_cp_async_16_zfill": "from helion._compiler.cute._mma_dag_runtime import cp_async_16_zfill as _cute_ring_cp_async_16_zfill",
            "_cute_ring_ldmatrix_x4": "from helion._compiler.cute._flash_row_mma_runtime import ldmatrix_x4 as _cute_ring_ldmatrix_x4",
            "_cute_ring_lds_b32": "from helion._compiler.cute._mma_dag_runtime import lds_b32 as _cute_ring_lds_b32",
            "_cute_ring_lds_f32": "from helion._compiler.cute._flash_row_mma_runtime import lds_f32 as _cute_ring_lds_f32",
            "_cute_ring_sts_f32": "from helion._compiler.cute._flash_row_mma_runtime import sts_f32 as _cute_ring_sts_f32",
            "_cute_lds_v4_b32": "from helion._compiler.cute._warp_mma_runtime import lds_v4_b32 as _cute_lds_v4_b32",
            "_cute_sts_v4_b32": "from helion._compiler.cute._warp_mma_runtime import sts_v4_b32 as _cute_sts_v4_b32",
            "_cute_grouped_reduce_shared_tree": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_tree",
            "_cute_grouped_reduce_shared_two_stage": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_two_stage",
            "_cute_grouped_reduce_shared_two_stage_fragment": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_two_stage_fragment",
            "_cute_grouped_reduce_shared_columns": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_columns",
            "_cute_resident_load_vector": "from helion._compiler.cute.resident_reduction_runtime import _cute_resident_load_vector",
            "_cute_resident_store_vector": "from helion._compiler.cute.resident_reduction_runtime import _cute_resident_store_vector",
            "_cute_resident_copy_async": "from helion._compiler.cute.resident_reduction_runtime import _cute_resident_copy_async",
            "_cute_resident_sums": "from helion._compiler.cute.resident_reduction_runtime import _cute_resident_sums",
            "_cute_resident_sums_disjoint": "from helion._compiler.cute.resident_reduction_runtime import _cute_resident_sums_disjoint",
            "_cute_try_paired_sum_cast": "from helion.runtime.cute.paired_sum import try_paired_sum_cast as _cute_try_paired_sum_cast",
            "_cute_try_single_sum_cast": "from helion.runtime.cute.single_sum import try_single_sum_cast as _cute_try_single_sum_cast",
            "_cute_try_full_sum": "from helion.runtime.cute.full_sum import try_full_sum as _cute_try_full_sum",
            "_cute_remainder_fill": "from helion._compiler.cute.remainder_fill_runtime import _cute_remainder_fill",
            "_cute_grouped_reduce_warp": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_warp",
            "_cute_complete_row_tensor_copy": "from helion._compiler.cute.tensor_copy_helpers import _cute_complete_row_tensor_copy",
            "_cute_grouped_reduce_cluster": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_cluster",
            "_cute_grouped_reduce_block": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_block",
            "_cute_grouped_reduce_cluster_warp_online_pair": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_cluster_warp_online_pair",
            "_cute_grouped_reduce_cluster_online_pair": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_cluster_online_pair",
            "_cute_store_shared_remote_x4": "from helion._compiler.cute.cluster_helpers import store_shared_remote_x4 as _cute_store_shared_remote_x4",
            "_cute_issue_clc_query_nomulticast": "from helion._compiler.cute.clc_helpers import issue_clc_query_nomulticast as _cute_issue_clc_query_nomulticast",
            "_cute_python_mod": "from helion._compiler.cute.integer_helpers import python_mod as _cute_python_mod",
            "_cute_flex_tanh": "from helion._compiler.cute.flex_score_mod_runtime import tanh_f32 as _cute_flex_tanh",
            "_cute_flex_load_vec": "from helion._compiler.cute.flex_score_mod_runtime import load_vec as _cute_flex_load_vec",
            "_cute_flex_load_run": "from helion._compiler.cute.flex_score_mod_runtime import load_run as _cute_flex_load_run",
            "_cute_flex_load_window": "from helion._compiler.cute.flex_score_mod_runtime import load_window as _cute_flex_load_window",
            "_cute_flex_coop_lane": "from helion._compiler.cute.flex_score_mod_runtime import coop_lane_map as _cute_flex_coop_lane",
            "_cute_flex_coop_issue": "from helion._compiler.cute.flex_score_mod_runtime import coop_issue as _cute_flex_coop_issue",
            "_cute_flex_lane_below": "from helion._compiler.cute.flex_score_mod_runtime import lane_below as _cute_flex_lane_below",
            "_cute_flex_fmod": "from helion._compiler.cute.flex_score_mod_runtime import fmod as _cute_flex_fmod",
            "_cute_flex_remainder": "from helion._compiler.cute.flex_score_mod_runtime import remainder_f32 as _cute_flex_remainder",
            "_cute_inline_asm_elementwise": "from helion._compiler.cute.inline_asm_helpers import inline_asm_elementwise as _cute_inline_asm_elementwise",
            "_cute_fp8e4m3fn_to_float32": "from helion._compiler.cute.quantized_helpers import fp8e4m3fn_to_float32 as _cute_fp8e4m3fn_to_float32",
            "_cute_fp8e4m3fn_x2_to_float32": "from helion._compiler.cute.quantized_helpers import fp8e4m3fn_x2_to_float32 as _cute_fp8e4m3fn_x2_to_float32",
            "_cute_fold_dot": "from helion._compiler.cute.fold_helpers import fold_dot as _cute_fold_dot",
            "_cute_float4_e2m1fn_x2_to_float32": "from helion._compiler.cute.quantized_helpers import float4_e2m1fn_x2_to_float32 as _cute_float4_e2m1fn_x2_to_float32",
            "_cute_float4_e2m1fn_x16_to_float16": "from helion._compiler.cute.quantized_helpers import float4_e2m1fn_x16_to_float16 as _cute_float4_e2m1fn_x16_to_float16",
            "_cute_float4_e2m1fn_x16_dot_float32": "from helion._compiler.cute.quantized_helpers import float4_e2m1fn_x16_dot_float32 as _cute_float4_e2m1fn_x16_dot_float32",
            "_cute_bfloat16_x16_to_float16": "from helion._compiler.cute.quantized_helpers import bfloat16_x16_to_float16 as _cute_bfloat16_x16_to_float16",
            "_cute_signed_bitfield_to_bf16_packed": "from helion._compiler.cute.vec_utils import signed_bitfield_to_bf16_packed as _cute_signed_bitfield_to_bf16_packed",
            "_cute_store_u16_vec": "from helion._compiler.cute.vec_utils import store_u16_vec as _cute_store_u16_vec",
            "_cute_store_fp8e4m3fn_vec": "from helion._compiler.cute.vec_utils import store_fp8e4m3fn_vec as _cute_store_fp8e4m3fn_vec",
            "_cute_store_fp8e5m2_vec": "from helion._compiler.cute.vec_utils import store_fp8e5m2_vec as _cute_store_fp8e5m2_vec",
            "_cute_store_fp8e4m3fn_vec_l2_evict_last": "from helion._compiler.cute.packet_store_helpers import store_fp8e4m3fn_vec_l2_evict_last as _cute_store_fp8e4m3fn_vec_l2_evict_last",
            "_cute_store_fp8e5m2_vec_l2_evict_last": "from helion._compiler.cute.packet_store_helpers import store_fp8e5m2_vec_l2_evict_last as _cute_store_fp8e5m2_vec_l2_evict_last",
            "_cute_store_u32_vec": "from helion._compiler.cute.vec_utils import store_u32_vec as _cute_store_u32_vec",
            "_cute_red_add_f32_vec": "from helion._compiler.cute.vec_utils import red_add_f32_vec as _cute_red_add_f32_vec",
            "_cute_red_add_f16_vec": "from helion._compiler.cute.vec_utils import red_add_f16_vec as _cute_red_add_f16_vec",
            "_cute_red_add_bf16_vec": "from helion._compiler.cute.vec_utils import red_add_bf16_vec as _cute_red_add_bf16_vec",
            "_cute_rank1_pack_bf16x2": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_pack_bf16x2 as _cute_rank1_pack_bf16x2",
            "_cute_rank1_mul_bf16x2": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_mul_bf16x2 as _cute_rank1_mul_bf16x2",
            "_cute_rank1_add_bf16x2": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_add_bf16x2 as _cute_rank1_add_bf16x2",
            "_cute_rank1_fma_bf16x2": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_fma_bf16x2 as _cute_rank1_fma_bf16x2",
            "_cute_rank1_load_u32x2_nc": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_u32x2_nc as _cute_rank1_load_u32x2_nc",
            "_cute_rank1_load_f32x4_nc": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_f32x4_nc as _cute_rank1_load_f32x4_nc",
            "_cute_rank1_load_f32_nc": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_f32_nc as _cute_rank1_load_f32_nc",
            "_cute_rank1_load_u16_nc": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_u16_nc as _cute_rank1_load_u16_nc",
            "_cute_rank1_load_i32_nc": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_i32_nc as _cute_rank1_load_i32_nc",
            "_cute_rank1_load_i64_nc": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_i64_nc as _cute_rank1_load_i64_nc",
            "_cute_rank1_load_u32x4_if_valid": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_load_u32x4_if_valid as _cute_rank1_load_u32x4_if_valid",
            "_cute_rank1_store_u32x4_if_valid": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_store_u32x4_if_valid as _cute_rank1_store_u32x4_if_valid",
            "_cute_rank1_store_u16_or_zero": "from helion._compiler.cute.single_token_rank1_recurrence import rank1_store_u16_or_zero as _cute_rank1_store_u16_or_zero",
            **{helper.name: helper.import_line for helper in PACKET_LOAD_HELPERS},
            "_cute_store_u16x8_l2_evict_last": "from helion._compiler.cute.l2_policy import store_u16x8_l2_evict_last as _cute_store_u16x8_l2_evict_last",
            "_cute_store_u32x4_l2_evict_last": "from helion._compiler.cute.l2_policy import store_u32x4_l2_evict_last as _cute_store_u32x4_l2_evict_last",
            "_cute_store_u16x8_streaming": "from helion._compiler.cute.l2_policy import store_u16x8_streaming as _cute_store_u16x8_streaming",
            "_cute_store_u32x4_streaming": "from helion._compiler.cute.l2_policy import store_u32x4_streaming as _cute_store_u32x4_streaming",
            "_cute_store_fp8e4m3fn_vec_streaming": "from helion._compiler.cute.packet_store_helpers import store_fp8e4m3fn_vec_streaming as _cute_store_fp8e4m3fn_vec_streaming",
            "_cute_store_fp8e5m2_vec_streaming": "from helion._compiler.cute.packet_store_helpers import store_fp8e5m2_vec_streaming as _cute_store_fp8e5m2_vec_streaming",
            "_cute_grid_barrier": "from helion._compiler.cute.grid_barrier import grid_barrier as _cute_grid_barrier",
            "_cute_tmem_store_atom": "from helion._compiler.cute.split_k_exchange import tmem_store_atom as _cute_tmem_store_atom",
            "_cute_join_cast": "from helion._compiler.cute.join_cast import join_cast as _cute_join_cast",
            "_cute_exact_div_bounds_f32": "from helion._compiler.cute.exact_division import exact_div_bounds_f32 as _cute_exact_div_bounds_f32",
            "_cute_exact_div_fast_f32": "from helion._compiler.cute.exact_division import exact_div_fast_f32 as _cute_exact_div_fast_f32",
            "_cute_exact_div_redo_f32": "from helion._compiler.cute.exact_division import exact_div_redo_f32 as _cute_exact_div_redo_f32",
            "_cute_copysign": "from helion._compiler.cute.copysign import copysign as _cute_copysign",
            "_cute_atomic_max_float32": "from helion._compiler.cute.atomic_helpers import atomic_max_float32 as _cute_atomic_max_float32",
            "_cute_atomic_min_float32": "from helion._compiler.cute.atomic_helpers import atomic_min_float32 as _cute_atomic_min_float32",
            "_cute_warp_fetch_add_int32": "from helion._compiler.cute.warp_atomic_runtime import warp_fetch_add_int32 as _cute_warp_fetch_add_int32",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        if dim == 0:
            from ..device_function import DeviceFunction

            divisor = DeviceFunction.current().cute_state.tcgen05.program_id_divisor
            if divisor > 1:
                # ``tcgen05_split_k``: ``divisor`` consecutive CTAs share one
                # output tile (the split index is ``block_idx % divisor``).
                return f"{index_dtype}(cute.arch.block_idx()[0] // {divisor})"
        return f"{index_dtype}(cute.arch.block_idx()[{dim}])"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLArg
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            upcast_compute_type,
        )

        class HelionCuteDSLOpOverrides(CuteDSLOpOverrides):
            @staticmethod
            def _ftz_safe_current_node() -> bool:
                from torch._inductor.virtualized import V

                from .exp2_fastmath import FTZ_SAFE_EXP_META_KEY

                node = V.current_node
                return node is not None and bool(node.meta.get(FTZ_SAFE_EXP_META_KEY))

            @staticmethod
            def exp(x: CuteDSLArg) -> CuteDSLArg:
                # Marked sites (sum(exp(x - amax(x)))) are provably
                # unaffected by flushing denormal exp outputs; skip the
                # default lowering's denormal fixup (FSETP + 2 predicated
                # FMULs per element) — see exp2_fastmath.py.
                if (
                    HelionCuteDSLOpOverrides._ftz_safe_current_node()
                    or HelionCuteDSLOpOverrides._fastmath_on()
                ):
                    if CuteDSLOpOverrides._get_cse_var(x) is None:
                        x = CuteDSLOpOverrides._cast_expr(str(x), torch.float32)
                    return CuteDSLOpOverrides._apply_unary_op(
                        x,
                        f"cute.math.exp2({{x}} * {CuteDSLOpOverrides.LOG2_E}, fastmath=True)",
                    )
                return CuteDSLOpOverrides.exp(x)

            @staticmethod
            def exp2(x: CuteDSLArg) -> CuteDSLArg:
                if (
                    HelionCuteDSLOpOverrides._ftz_safe_current_node()
                    or HelionCuteDSLOpOverrides._fastmath_on()
                ):
                    return CuteDSLOpOverrides._apply_unary_op(
                        x, "cute.math.exp2({x}, fastmath=True)"
                    )
                # Not Inductor's exp(x * ln2) decomposition: rounding x * ln2
                # and then (x * ln2) * log2(e) in exp costs 2-3 ulp.
                return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.exp2({x})")

            @staticmethod
            def _fastmath_on() -> bool:
                from ..compile_environment import CompileEnvironment

                return CompileEnvironment.current().settings.fast_math

            @staticmethod
            def constant(value: bool | float, dtype: torch.dtype) -> str:
                from ..inductor_lowering_extra import keeps_negative_nan_literal

                # The DSL types a Python int as Int32, which widens 8- and
                # 16-bit integer math so it never wraps, and takes a negative
                # one as signed next to an unsigned value.  Eager converts the
                # scalar to the tensor's dtype, wrapping it (x_uint8 % -7
                # divides by 249).
                if isinstance(value, int) and dtype in _NARROW_INTEGER_DTYPES:
                    bits = torch.iinfo(dtype).bits
                    value &= (1 << bits) - 1
                    if dtype.is_signed and value >= 1 << (bits - 1):
                        value -= 1 << bits
                    return CuteDSLOpOverrides._cast_expr(repr(value), dtype)
                if keeps_negative_nan_literal(value):
                    return "-float('nan')"
                return CuteDSLOpOverrides.constant(value, dtype)

            @staticmethod
            def mul(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                from torch._inductor.virtualized import V

                from .scalar_recipe_rounding import FP32_MULTIPLY_ROUNDING_EXPR
                from .scalar_recipe_rounding import FP32_MULTIPLY_ROUNDING_META_KEY
                from .scalar_recipe_rounding import NARROW_MULTIPLY_ROUNDING_EXPR
                from .scalar_recipe_rounding import NARROW_MULTIPLY_ROUNDING_META_KEY

                node = V.current_node
                expected = CuteDSLOpOverrides._expected_tensor_val()
                if (
                    not HelionCuteDSLOpOverrides._fastmath_on()
                    and isinstance(node, torch.fx.Node)
                    and node.meta.get(NARROW_MULTIPLY_ROUNDING_META_KEY)
                    and expected is not None
                    and expected.dtype in NARROW_MULTIPLY_ROUNDING_EXPR
                ):
                    # The 16-bit product ``narrow_exact_products`` made of an
                    # explicitly rounded FP32 product and its cast: ``mul.rn``
                    # in the dtype keeps the single rounding and, like the
                    # FP32 form, cannot be contracted into an FMA.
                    return CuteDSLOpOverrides._apply_binary_op(
                        a, b, NARROW_MULTIPLY_ROUNDING_EXPR[expected.dtype]
                    )
                if (
                    not HelionCuteDSLOpOverrides._fastmath_on()
                    and isinstance(node, torch.fx.Node)
                    and node.meta.get(FP32_MULTIPLY_ROUNDING_META_KEY)
                    and expected is not None
                    and expected.dtype == torch.float32
                ):
                    # A separately rounded multiply must not contract with a
                    # later add across its narrowing cast: a recipe moved to
                    # its own launch, or a product an explicit cast rounds
                    # before more arithmetic (``mark_narrowed_fp32_multiplies``).
                    return CuteDSLOpOverrides._apply_binary_op(
                        a, b, FP32_MULTIPLY_ROUNDING_EXPR
                    )
                return CuteDSLOpOverrides.mul(a, b)

            @staticmethod
            def _unary_math(x: CuteDSLArg, name: str) -> CuteDSLArg:
                suffix = (
                    ", fastmath=True" if HelionCuteDSLOpOverrides._fastmath_on() else ""
                )
                return CuteDSLOpOverrides._apply_unary_op(
                    x, f"cute.math.{name}({{x}}{suffix})"
                )

            @staticmethod
            def sqrt(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "sqrt")

            @staticmethod
            def log(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "log")

            @staticmethod
            def log2(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "log2")

            @staticmethod
            def log10(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "log10")

            @staticmethod
            def cos(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "cos")

            @staticmethod
            def sin(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "sin")

            @staticmethod
            def tan(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "tan")

            @staticmethod
            def acos(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "acos")

            @staticmethod
            def asin(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "asin")

            @staticmethod
            def atan(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "atan")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def sinh(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "sinh")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def cosh(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "cosh")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def asinh(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "asinh")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def acosh(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "acosh")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def atanh(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "atanh")

            @staticmethod
            def atan2(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                suffix = (
                    ", fastmath=True" if HelionCuteDSLOpOverrides._fastmath_on() else ""
                )
                return CuteDSLOpOverrides._apply_binary_op(
                    a, b, f"cute.math.atan2({{a}}, {{b}}{suffix})"
                )

            @staticmethod
            # pyrefly: ignore [bad-override]
            def copysign(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                # Both operands take the compute type: a Python number
                # operand is untyped.
                expected = CuteDSLOpOverrides._expected_tensor_val()
                result_dtype = expected.dtype if expected is not None else None
                dtype = upcast_compute_type(result_dtype or torch.float32)
                mag = CuteDSLOpOverrides._cast_expr("{a}", dtype)
                sign = CuteDSLOpOverrides._cast_expr("{b}", dtype)
                # cute.math.copysign gives a NaN magnitude the canonical
                # positive NaN, which is what eager's rounding of a 16-bit
                # result gives; at fp32/fp64 eager keeps the copied sign.
                function = (
                    "cute.math.copysign"
                    if result_dtype in (torch.float16, torch.bfloat16)
                    else "_cute_copysign"
                )
                return CuteDSLOpOverrides._apply_binary_op(
                    a, b, f"{function}({mag}, {sign})"
                )

            @staticmethod
            def erf(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "erf")

            # Inductor decomposes erfc, expm1 and log1p into 1 - erf(x),
            # exp(x) - 1 and log(1 + x), which cancel catastrophically for
            # large x (erfc) and for x near 0 (expm1, log1p).

            @staticmethod
            # pyrefly: ignore [bad-override]
            def erfc(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "erfc")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def expm1(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "expm1")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def log1p(x: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x, "log1p")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def floor(x: CuteDSLArg) -> CuteDSLArg:
                return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.floor({x})")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def ceil(x: CuteDSLArg) -> CuteDSLArg:
                return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.ceil({x})")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def trunc(x: CuteDSLArg) -> CuteDSLArg:
                return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.trunc({x})")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def round(x: CuteDSLArg) -> CuteDSLArg:
                # torch.round rounds half to even (libdevice.nearbyint in triton).
                return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.roundeven({x})")

            @staticmethod
            def tanh(x0: CuteDSLArg) -> CuteDSLArg:
                return HelionCuteDSLOpOverrides._unary_math(x0, "tanh")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def sigmoid(x: CuteDSLArg) -> CuteDSLArg:
                # RCP.APPROX + EX2.APPROX, the same sequence triton's default
                # tl.sigmoid compiles to (one MUFU.EX2 + one MUFU.RCP).  The
                # base lowering's accurate IEEE div is a ~20-instruction
                # BRANCHY SASS expansion (BSSY/BSYNC reconvergence per
                # element) that buys no accuracy here: sigmoid's error is
                # dominated by the shared x*log2e argument rounding, and both
                # forms measure max 64 ulp / mean 3.5 ulp vs an fp64
                # reference on [-87, 87] (identical distributions), with
                # specials preserved (nan->nan, +-inf->1/0, saturation->0).
                # Only denormal OUTPUTS differ: this form flushes them to
                # zero, as triton kernels do by default.  fp16 sigmoid
                # 67108864: 4943 -> 5805 GB/s (B200, cold autotune).
                x_cse = CuteDSLOpOverrides._get_cse_var(x)
                if x_cse is not None:
                    x_fp32: CuteDSLArg = CuteDSLOpOverrides.to_dtype(
                        x_cse,
                        torch.float32,
                        use_compute_types=False,
                    )
                else:
                    x_fp32 = CuteDSLOpOverrides._cast_expr(str(x), torch.float32)
                result = CuteDSLOpOverrides._apply_unary_op(
                    x_fp32,
                    SIGMOID_TEMPLATE,
                )
                expected = CuteDSLOpOverrides._expected_tensor_val()
                expected_dtype = expected.dtype if expected is not None else None
                if expected_dtype is not None and expected_dtype != torch.float32:
                    result_cse = CuteDSLOpOverrides._get_cse_var(result)
                    if result_cse is not None:
                        return CuteDSLOpOverrides.to_dtype(
                            result_cse,
                            expected_dtype,
                            use_compute_types=False,
                        )
                    return CuteDSLOpOverrides._cast_expr(str(result), expected_dtype)
                return result

            @staticmethod
            def truediv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                # cute.math.div lowers to an NVVM intrinsic that only accepts
                # fp32 scalars; 16/64-bit operands raise at DSL trace time, so
                # keep the accurate IEEE path for every other dtype.
                expected = CuteDSLOpOverrides._expected_tensor_val()
                if (
                    HelionCuteDSLOpOverrides._fastmath_on()
                    and expected is not None
                    and expected.dtype == torch.float32
                ):
                    return CuteDSLOpOverrides._apply_binary_op(
                        a, b, "cute.math.div({a}, {b}, approx=True, ftz=True)"
                    )
                return CuteDSLOpOverrides.truediv(a, b)

            @staticmethod
            # pyrefly: ignore [bad-override]
            def sign(x: CuteDSLArg) -> CuteDSLArg:
                # ``(x > 0) - (x < 0)``: NaN and -0.0 map to +0 like torch.sign.
                expected = CuteDSLOpOverrides._expected_tensor_val()
                dtype = expected.dtype if expected is not None else torch.float32
                if dtype == torch.bool:
                    return x
                positive = CuteDSLOpOverrides._cast_expr("(({x}) > 0)", dtype)
                negative = CuteDSLOpOverrides._cast_expr("(({x}) < 0)", dtype)
                return CuteDSLOpOverrides._apply_unary_op(
                    x, f"({positive} - {negative})"
                )

            @staticmethod
            def _nonzero_divisor(b: CuteDSLArg) -> CuteDSLArg:
                # Integer division by zero is undefined behavior, and a
                # masked-off lane's divisor is its load's zero fill: the
                # compiler then drops the masks on that lane's loads and
                # stores, which write out of bounds.  Divide such a lane by
                # one (eager leaves a zero integer divisor undefined on CUDA).
                from torch._inductor.virtualized import V

                from ..compile_environment import CompileEnvironment

                expected = CuteDSLOpOverrides._expected_tensor_val()
                if expected is None or expected.dtype.is_floating_point:
                    return b
                # A provably nonzero divisor (``x // 2``, ``tile.index %
                # tile.block_size``) needs no guard.
                args = V.current_node.args
                divisor = args[1] if len(args) > 1 else None
                if isinstance(divisor, torch.fx.Node):
                    divisor = divisor.meta.get("val")
                if type(divisor) is int and divisor != 0:
                    return b
                env = CompileEnvironment.current()
                if isinstance(divisor, torch.SymInt) and env.known_nonnegative(
                    sympy.Add(divisor._sympy_(), -1)
                ):
                    return b
                # The divisor's DSL type can differ from the FX dtype (index
                # math on Int64 shape arguments).
                return CuteDSLOpOverrides._apply_unary_op(
                    b, "(({x}) if ({x}) != 0 else _cute_join_cast(1, {x}))"
                )

            @staticmethod
            def floordiv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                b = HelionCuteDSLOpOverrides._nonzero_divisor(b)
                return CuteDSLOpOverrides._apply_binary_op(a, b, "(({a}) // ({b}))")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def truncdiv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                # The DSL's integer // floors; a minus its C remainder is an
                # exact multiple of b, which floors and truncates alike.
                from torch._inductor.virtualized import ops

                return ops.floordiv(ops.sub(a, ops.mod(a, b)), b)

            @staticmethod
            def mod(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                b = HelionCuteDSLOpOverrides._nonzero_divisor(b)
                return CuteDSLOpOverrides._apply_binary_op(a, b, "(({a}) % ({b}))")

            @staticmethod
            # pyrefly: ignore [bad-override]
            def fmod(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                # CuTe DSL's % truncates like C: the result takes a's sign.
                return HelionCuteDSLOpOverrides.mod(a, b)

            @staticmethod
            def remainder(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
                expected = CuteDSLOpOverrides._expected_tensor_val()
                if (
                    expected is not None
                    and not expected.dtype.is_floating_point
                    and expected.dtype != torch.bool
                ):
                    # TensorIterator first converts wrapped scalar operands to
                    # the logical promoted tensor dtype. Keep this distinct
                    # from scalar PythonMod's ordinary Python/SDK promotion.
                    b = HelionCuteDSLOpOverrides._nonzero_divisor(b)
                    left = CuteDSLOpOverrides._cast_expr("{a}", expected.dtype)
                    right = CuteDSLOpOverrides._cast_expr("{b}", expected.dtype)
                    return CuteDSLOpOverrides._apply_binary_op(
                        a, b, f"_cute_python_mod({left}, {right})"
                    )
                # A float remainder is Python-style too: a nonzero C remainder
                # whose sign differs from the divisor's moves by one divisor.
                from torch._inductor.virtualized import ops

                result = ops.mod(a, b)
                zero = ops.constant(0, torch.int32)
                wrong_sign = ops.logical_and(
                    ops.ne(result, zero),
                    ops.ne(ops.lt(result, zero), ops.lt(b, zero)),
                )
                return ops.where(wrong_sign, ops.add(result, b), result)

            @staticmethod
            def where(
                condition: CuteDSLArg,
                a: CuteDSLArg,
                b: CuteDSLArg,
            ) -> CuteDSLArg:
                tensor_arg = (
                    HelionCuteDSLOpOverrides._get_cse_var(a)
                    or HelionCuteDSLOpOverrides._get_cse_var(b)
                    or HelionCuteDSLOpOverrides._get_cse_var(condition)
                )
                if tensor_arg is not None:
                    return CuteDSLOpOverrides.where(condition, a, b)
                return f"(({a}) if ({condition}) else ({b}))"

        return HelionCuteDSLOpOverrides()

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{dtype_str}({expr_str})"

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        from ..device_function import DeviceFunction
        from ..device_function import NoCurrentFunction

        if isinstance(x, ast.Name):
            try:
                df = DeviceFunction.current()
            except NoCurrentFunction:
                pass
            else:
                if (
                    df.cute_state.get_tcgen05_store_value(df.variable_aliases(x.id))
                    is not None
                ):
                    return x
        return super().cast_ast(x, target_dtype)

    def scalar_float_operand_ast(
        self, x: ast.AST, value: torch.SymFloat, *, computes_in_float64: bool
    ) -> ast.AST:
        # The DSL types a Python float combined with an Int as Float64
        # (``n + 0.0`` for ``ToFloat(n)``), which would turn the op's float32
        # math, and any loop carry it feeds, into Float64.  Float kernel
        # arguments are Float32 (see the launcher), and so is an expression
        # of only them and constants.
        float_symbols = (SymT.FLOAT, SymT.UNBACKED_FLOAT)
        if computes_in_float64 or all(
            symbol_is_type(symbol, float_symbols)
            for symbol in value._sympy_().free_symbols
        ):
            return x
        return self.cast_ast(x, torch.float32)

    def grid_barrier_stmt(self, sem_arg: str) -> str | None:
        # ``sem_arg`` is a TensorArg that arrives as a ``cute.Tensor``; its
        # ``.iterator`` is the underlying ``cute.Pointer`` to the semaphore.
        return f"_cute_grid_barrier({sem_arg}.iterator)"

    def _check_thread_axis(self, axis: int) -> None:
        # ``cute.arch.thread_idx()`` is an (x, y, z) triple; indexing it with
        # a fourth axis fails deep inside DSL tracing with an IndexError.
        if axis >= 3:
            raise exc.BackendUnsupported(self.name, f"thread axis {axis}")

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        from ..compile_environment import CompileEnvironment

        self._check_thread_axis(axis)
        index_dtype = CompileEnvironment.current().index_type()
        return (
            f"{offset_var} + {index_dtype}(cute.arch.thread_idx()[{axis}])"
            f" * {elements_per_thread}"
        )

    def lane_offset_expr(self, lane_var: str) -> str:
        return f"cutlass.Int32({lane_var})"

    def thread_index_expr(self, *, axis: int) -> str:
        from ..compile_environment import CompileEnvironment

        self._check_thread_axis(axis)
        index_dtype = CompileEnvironment.current().index_type()
        return f"{index_dtype}(cute.arch.thread_idx()[{axis}])"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from .printer import cute_texpr

        return cute_texpr(expr)

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(f"cutlass.Int32({begin})")
        range_args.append(f"cutlass.Int32({end})")
        if step is not None and step != "1":
            range_args.append(f"cutlass.Int32({step})")
        return f"range({', '.join(range_args)})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        # Complete root owners can discard this scalar grid setup. Validate
        # surviving coordinates after ownership lowering and dead-code removal.
        return (
            f"{offsets_var} = ({lid}) * ({block_size_var})"
            f" + {dtype}(cute.arch.thread_idx()[{axis}])"
        )

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        # Root emitters may replace the entire scalar schedule, making this
        # index dead. Validate surviving coordinates after final codegen/DCE.
        return f"{offset_var} + {dtype}(cute.arch.thread_idx()[{axis}])"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return self.grid_index_expr(offset_var, block_size_var, dtype, axis=axis)

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            index_expr = "0"
        return f"({tensor_name})[{index_expr}]"

    def max_reduction_threads(self) -> int | None:
        return 1024

    def reduction_emitter(self, strategy: ReductionStrategy) -> ReductionEmitter[Any]:
        from .reduction_emitter import reduction_emitter_for

        return reduction_emitter_for(strategy)

    def create_reduction_strategy(
        self, fn: DeviceFunction, block_id: int, reduction_loop: int | None
    ) -> TileStrategy:
        if reduction_loop is None:
            from .reduction_emitter import CutePersistentReductionStrategy

            return CutePersistentReductionStrategy(fn, block_id)
        return super().create_reduction_strategy(fn, block_id, reduction_loop)

    def max_reduction_loop(self) -> int | None:
        from .reduction_emitter import cute_looped_reduction_block_size

        max_threads = self.max_reduction_threads()
        if max_threads is None:
            return None
        return cute_looped_reduction_block_size(2**31 - 1, max_threads)

    def default_reduction_loop(self, size_hint: int, max_threads: int) -> int:
        from .reduction_emitter import cute_looped_reduction_block_size

        return cute_looped_reduction_block_size(size_hint, max_threads)

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        from ..reduction_strategy import ReductionStrategy

        if requested <= 1:
            return requested
        other_threads = 1
        for strategy in existing_strategies:
            if isinstance(strategy, ReductionStrategy):
                count = strategy._reduction_thread_count()
                if count > 0:
                    other_threads *= count
            else:
                for size in strategy.thread_block_sizes():
                    if size > 1:
                        other_threads *= size
        while other_threads * requested > MAX_THREADS_PER_BLOCK and requested > 1:
            requested //= 2
        return requested

    def create_synthetic_reduction_lanes(
        self,
        thread_count: int,
        size_hint: int,
    ) -> int | None:
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        if thread_count <= 0:
            return None
        padded_size = next_power_of_2(max(1, size_hint))
        if padded_size > thread_count:
            # Whole-warp thread counts need not divide the padded extent.
            return (padded_size + thread_count - 1) // thread_count
        return None

    def reduction_axis_first(self) -> bool:
        return True

    def supports_lane_loop_reductions(self) -> bool:
        return True

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        # A complete root owner can also discard the speculative grid mask.
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        # Masks are elided per-axis when the extent is a known multiple of
        # the block size (same rule as the Triton backend) AND the launch
        # cannot run the axis wider than the tile (see
        # ``launches_surplus_tile_threads``).  Every per-element mask costs
        # a compare + select in the SIMT lane loop, which is significant
        # for memory-bound reduction kernels.
        return False

    def launches_surplus_tile_threads(self) -> bool:
        # Mutually exclusive kernel sections (e.g. persistent stages around
        # an ``hl.barrier()``) share one launch whose block dims are the
        # elementwise max across sections, so a section can run with more
        # threads on an axis than its own tile is wide.
        return True

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        # One element per thread: tile-shaped temporaries are scalars.
        return f"{self.dtype_str(dtype)}({value_expr})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return expr

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return expr

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"({true_val}) if ({mask}) else ({false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        from ..compile_environment import CompileEnvironment

        index_type = CompileEnvironment.current().index_type()
        lhs = self.cast_expr(a, index_type)
        rhs = self.cast_expr(b, index_type)
        return f"({lhs}) if ({lhs}) < ({rhs}) else ({rhs})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        self._check_thread_axis(axis)
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return "cutlass.Int32(0)"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"_next_power_of_2({expr})"

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        # Use Python ternary instead of cute.where for max/min because
        # these operate on scalar registers, not tensors.
        #
        # Cast the incoming value to the accumulator dtype first.  The
        # accumulator is promoted to the computation dtype (fp32 for
        # half-precision inputs), but the per-iteration reduction input keeps
        # the tensor's storage dtype (e.g. bf16 for a masked half load).  The
        # CUTLASS DSL strictly type-checks the two branches of a Python ternary
        # ("Then and else blocks of ifexp return different types"), so a bare
        # ``acc if acc > val else val`` with mixed fp32/bf16 operands fails to
        # compile.  The cast is a no-op when ``val`` already matches.
        val = self.cast_expr(val, self.dtype_str(dtype))
        if reduction_type == "sum":
            return f"({acc} + {val})"
        if reduction_type in ("min", "max") and dtype == torch.int32:
            # Preserve signed ordering through unsigned keys. The CUDA compiler
            # can lose a source negation when combining a signed Int32 extrema
            # chain into VIMNMX3. Flipping the sign bit is an exact order
            # isomorphism and avoids that invalid signed fusion.
            sign_bit = "cutlass.Uint32(2147483648)"
            lhs = f"(cutlass.Uint32({acc}) ^ {sign_bit})"
            rhs = f"(cutlass.Uint32({val}) ^ {sign_bit})"
            return (
                f"cutlass.Int32(cute.math.{reduction_type}({lhs}, {rhs}) ^ {sign_bit})"
            )
        if reduction_type in ("max", "min"):
            return nan_extremum_expr(
                reduction_type, acc, val, float32=dtype == torch.float32
            )
        if reduction_type == "prod":
            return f"({acc} * {val})"
        raise exc.BackendUnsupported(self.name, f"reduction combine {reduction_type!r}")

    def _threads_for_block_size_var(self, block_size_var: str | None) -> int:
        # threads_in_group must be a Python int literal for CuTe DSL.
        from ..reduction_strategy import ReductionStrategy
        from ..tile_strategy import BlockSizeTileStrategy

        threads = 32
        strategies = self._get_strategies()
        if block_size_var is not None:
            for strategy in strategies:
                if not isinstance(strategy, ReductionStrategy):
                    continue
                strategy_bs_var = strategy.block_size_var(strategy.block_index)
                if strategy_bs_var != block_size_var:
                    continue
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc

            # Block reductions are keyed by a tile block-size var rather than a
            # ReductionStrategy var. Recover the tile width from the owning strategy.
            for strategy in strategies:
                if not isinstance(strategy, BlockSizeTileStrategy):
                    continue
                for idx, block_id in enumerate(strategy.block_ids):
                    strategy_bs_var = strategy.block_size_var(block_id)
                    if strategy_bs_var != block_size_var:
                        continue
                    thread_extent = (
                        strategy.fn.tile_strategy.thread_extent_for_block_id(block_id)
                    )
                    if thread_extent is not None and thread_extent > 0:
                        return min(thread_extent, 32)
                    block_size = strategy.block_size
                    if isinstance(block_size, list) and idx < len(block_size):
                        block_size = block_size[idx]
                    if isinstance(block_size, int) and block_size > 0:
                        return min(block_size, 32)
            return threads

        for strategy in strategies:
            if isinstance(strategy, ReductionStrategy):
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc
        return threads

    def reduction_threads_hint(self, block_size_var: str | None = None) -> int | None:
        return self._threads_for_block_size_var(block_size_var)

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> str:
        threads = (
            threads_in_group
            if threads_in_group is not None
            else self._threads_for_block_size_var(block_size_var)
        )
        tg = f", threads_in_group={threads}"
        if reduction_type == "sum":
            return f"cute.arch.warp_reduction_sum({input_name}{tg})"
        if reduction_type == "max":
            # ``op`` swaps the partial's NaN-quiet fmax for the NaN-propagating
            # max torch.amax has; the call stays a ``warp_reduction_max`` for
            # the passes that recognize it.
            return f"cute.arch.warp_reduction_max({input_name}, op=_cute_nan_max{tg})"
        if reduction_type == "min":
            return f"cute.arch.warp_reduction({input_name}, _cute_nan_min{tg})"
        if reduction_type == "prod":
            return f"cute.arch.warp_reduction({input_name}, lambda a, b: (a * b){tg})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def thread_linear_index_expr(self, axis_sizes: dict[int, int]) -> str | None:
        from ..compile_environment import CompileEnvironment

        for axis in axis_sizes:
            self._check_thread_axis(axis)
        index_dtype = CompileEnvironment.current().index_dtype
        index_type = self.index_type_str(index_dtype)
        if not axis_sizes:
            return self.cast_expr("0", index_type)
        stride = 1
        terms: list[str] = []
        for axis, size in sorted(axis_sizes.items()):
            term = self.cast_expr(f"cute.arch.thread_idx()[{axis}]", index_type)
            if stride != 1:
                term = f"({term}) * {self.cast_expr(repr(stride), index_type)}"
            terms.append(term)
            stride *= size
        return " + ".join(terms)

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        return reduction_type in {"argmin", "argmax"}

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
        threads_in_group: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> str:
        if index_dtype is None:
            raise exc.BackendUnsupported(self.name, "missing index_dtype for argreduce")
        value_reduction = "min" if reduction_type == "argmin" else "max"
        reduced_value = self.reduction_expr(
            input_name,
            value_reduction,
            dim,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
        )
        index_dtype_str = self.index_type_str(index_dtype)
        max_index = self.cast_expr(repr(torch.iinfo(index_dtype).max), index_dtype_str)
        candidate_index = argreduce_candidate_expr(
            index_value, input_name, reduced_value, max_index
        )
        reduced_index = self.reduction_expr(
            candidate_index,
            "min",
            dim,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
        )
        return self.cast_expr(reduced_index, self.dtype_str(output_dtype))

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
        dtype: torch.dtype | None = None,
    ) -> list[str]:
        if dtype is not None:
            # Both arms of the select must have the accumulator's type (fp32
            # for an fp16 input).
            value = self.cast_expr(value, self.dtype_str(dtype))
        compare = "<" if reduction_type == "argmin" else ">"
        # torch returns the first NaN: a NaN beats every number, and only an
        # earlier NaN beats a NaN accumulator.
        better = (
            f"((({acc}) == ({acc})) & "
            f"((({value}) {compare} ({acc})) | (({value}) != ({value})))) | "
            f"(((({value}) == ({acc})) | "
            f"((({value}) != ({value})) & (({acc}) != ({acc})))) & "
            f"(({index}) < ({acc_index})))"
        )
        return [
            (
                f"{acc}, {acc_index} = "
                f"(({value}), ({index})) if ({better}) else (({acc}), ({acc_index}))"
            )
        ]

    def _get_strategies(self) -> list[TileStrategy]:
        """Get the current device function's strategies."""
        from ..device_function import DeviceFunction

        try:
            return DeviceFunction.current().tile_strategy.strategies
        except Exception:
            return []

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from ..compile_environment import CompileEnvironment
        from ..device_function import ConstExprArg
        from ..device_function import DeviceFunction
        from ..host_function import HostFunction
        from ..reduction_strategy import ReductionStrategy
        from .autotune.config_keys import CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY
        from .autotune.config_keys import _cute_chunk_recurrence_config_is_safe

        device_function = DeviceFunction.current()
        codegen = device_function.codegen
        tile_strategy = device_function.tile_strategy
        final_kernel_text = "\n".join(
            ast.unparse(stmt)
            for stmt in [*device_function.preamble, *device_function.body]
        )
        final_thread_axes = {
            int(axis_text)
            for axis_text in re.findall(
                r"cute\.arch\.thread_idx\(\)\[(\d+)\]",
                final_kernel_text,
            )
        }

        def launcher_args_with_compile_options(block_arg: str) -> list[str]:
            # A body whose shared cross-warp reductions were sized for a
            # launch shape (``finalize_shared_reduce_groups``) must launch
            # with exactly that shape: any other block would alias groups.
            sized_for = device_function.cute_state.lanes.shared_reduce_launch_block
            if sized_for is not None:
                literal = re.fullmatch(r"block=\((\d+), (\d+), (\d+)\)", block_arg)
                launched = (
                    tuple(int(dim) for dim in literal.groups())
                    if literal is not None
                    else None
                )
                if launched != sized_for:
                    raise self.unsupported(
                        f"shared reductions were sized for block={sized_for} "
                        f"but the kernel launches with {block_arg}",
                    )
            # Record the static launch shape for the kernel definition's
            # codegen (``_implicit_pdl_release`` sizes the resident CTAs per
            # SM from it); an argument-sized axis leaves it unknown.
            static_block = re.fullmatch(r"block=\((\d+), (\d+), (\d+)\)", block_arg)
            device_function.cute_state.lanes.launch_block = (
                (
                    int(static_block.group(1)),
                    int(static_block.group(2)),
                    int(static_block.group(3)),
                )
                if static_block is not None
                else None
            )
            launcher_args = [block_arg]
            compile_options: list[str] = []
            # One --ptxas-options flag carries them all (a second would
            # replace the first).
            ptxas_options: list[str] = []
            recurrence_register_cap = config.get(CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY)
            if recurrence_register_cap is not None:
                recurrence_plan = (
                    device_function.cute_state.root_plans.chunk_recurrence_plan
                )
                if recurrence_plan is None or type(recurrence_register_cap) is not int:
                    raise self.unsupported("invalid matched recurrence register cap")
                if not _cute_chunk_recurrence_config_is_safe(
                    recurrence_plan.dv_partitions, recurrence_register_cap
                ):
                    raise self.unsupported(
                        "register caps are unsafe for the TMEM DV2 recurrence "
                        "schedule's dynamic register allocation",
                    )
                ptxas_options.append(
                    "--override-directive-values "
                    f"--maxrregcount={recurrence_register_cap}"
                )
            prepare_plan = device_function.cute_state.root_plans.chunk_prepare_plan
            if prepare_plan is not None and prepare_plan.schedule.startswith(
                "split_alias_cpc"
            ):
                ptxas_options.append("--override-directive-values --maxrregcount=48")
            if config.advanced_controls_file:
                ptxas_options.append(
                    f"--apply-controls {config.advanced_controls_file}"
                )
            if ptxas_options:
                compile_options.append(f"--ptxas-options='{' '.join(ptxas_options)}'")
            if config.get(TCGEN05_CUBIN_LINEINFO_CONFIG_KEY) is True:
                compile_options.append("--generate-line-info")
            # ``--enable-tvm-ffi`` is emitted in codegen only when the
            # autotune flag is True so the generated code reflects which
            # configs deliberately requested FFI. The runtime
            # (``_get_compiled_cute_launcher``) unconditionally merges
            # the flag in for the generic launcher, so configs with this
            # flag False still execute with FFI enabled — that drift is
            # intentional for now and noted here for future cleanups.
            if config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True:
                if (
                    _kernel_specialized_mma_impl(
                        device_function,
                        config=device_function.config,
                    )
                    != "tcgen05"
                ):
                    raise self.unsupported(
                        f"{TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY}=True requires "
                        "tcgen05 CuTe lowering",
                    )
                compile_options.append("--enable-tvm-ffi")
            if compile_options:
                launcher_args.append(
                    f"cute_compile_options={' '.join(compile_options)!r}"
                )
            return launcher_args

        # A whole-root plan that owns the body owns its launch (the
        # structural ``LaunchShape``); the SIMT thread-axis recovery below
        # does not apply to it.
        owned_launch = plan_owned_launch(device_function.cute_state)
        if owned_launch is not None:
            x, y, z = owned_launch.dims
            return launcher_args_with_compile_options(f"block=({x}, {y}, {z})")

        block_size_values = {
            name: int(value)
            for name, value in re.findall(
                r"^(_BLOCK_SIZE_\d+) = (\d+)$",
                final_kernel_text,
                flags=re.MULTILINE,
            )
        }
        # Block IDs include untuned grid axes and are not positions in the
        # config's block_sizes list. Use the actual constexpr bindings, which
        # also preserve names allocated for flattened or renamed tile axes.
        block_size_values.update(
            {
                argument.name: int(argument.host_str())
                for argument in device_function.arguments
                if isinstance(argument, ConstExprArg) and argument.host_str().isdigit()
            }
        )
        # Accept both the historical ``offset_<n>`` prefix (non-CuTe backends)
        # and the post-rename ``tile_offset_<n>`` prefix (CuTe backend, see the
        # CuTe DSL preprocessor counter-collision note in
        # ``tile_strategy.py``). The launch-dim recovery walks the generated
        # source to pair Helion's per-axis offsets with their thread axes; if
        # the regex misses, we fall back to ``[1, 1, 1]`` and any kernel that
        # depends on the recovery launches with too-small dims.
        offset_block_sizes = dict(
            re.findall(
                r"^\s*(?:tile_)?offset_(\d+) = .* \* (_BLOCK_SIZE_\d+)$",
                final_kernel_text,
                flags=re.MULTILINE,
            )
        )
        grid_thread_extents = device_function.cute_state.lanes.grid_thread_extents
        live_extents = live_grid_thread_extents(
            [*device_function.preamble, *device_function.body], grid_thread_extents
        )
        exact_grid_thread_dims = [live_extents.get(axis, 1) for axis in range(3)]
        offset_thread_dims = list(exact_grid_thread_dims)
        # Grid producers record exact extents independently of index spelling.
        # Keep text recovery only for older/unregistered index producers.
        # When a lane loop is active for an axis the generated index expression
        # has the form
        #   ``indices_<n> = tile_offset_<n> + Int32(thread_idx()[<axis>]) * <epT> + Int32(lane_<n>)``
        # where ``<epT>`` is ``elements_per_thread`` for that axis and the
        # outer ``for lane_<n> in range(<epT>):`` covers the residual. In
        # that case the launch-time thread extent for the axis is
        # ``block_size / <epT>``, not ``block_size``. Without dividing by
        # ``<epT>`` here the launch dim ends up at ``block_size`` while the
        # generated tile arithmetic only spans ``block_size / <epT>`` threads,
        # which means ``thread_idx[axis] >= block_size / <epT>`` writes past
        # the tile and triggers ``cudaErrorIllegalAddress`` mid-search. The
        # ``cutlass.Int32(...)`` wrapper around ``thread_idx`` is closed
        # before the ``* <epT>`` multiplier, so the closing ``)`` between
        # ``[axis]`` and ``*`` is part of the line we have to skip over.
        #
        # The launch dim along ``axis`` has to serve **every** indices line
        # that uses that axis. If two lines on the same axis emit different
        # multipliers (e.g. one block_id with ``epT=1`` and another with
        # ``epT=2``), the line with the larger ``thread_idx[axis]`` range
        # is the binding one — so we compute ``block_size // epT`` *per
        # line* and take the ``max`` across lines, rather than combining
        # multipliers across lines and dividing once. ``re.findall`` over
        # the optional-multiplier alternation cannot be made to populate
        # the multiplier group reliably (the optional ``(?:...)?`` form
        # prefers the empty match), so we scan the lines once in Python
        # to keep the per-line ``(block_size, epT)`` pair intact.
        # ``indices_line_re`` anchors on the
        # ``cutlass.Int32(cute.arch.thread_idx()[<axis>])`` form the CuTe
        # backend emits via ``CuteBackend.lane_index_expr``.
        # The trailing ``\)`` after ``\]`` is the close of the
        # ``cutlass.Int32(...)`` wrapper around ``thread_idx``; if a
        # future codegen path drops the wrapper, the regex will
        # silently fail to match and the launch dim under-dimensions
        # without a signal. ``indices_line_assert_re`` below detects
        # the "wrapper-dropped" form so we can fail loudly instead.
        indices_line_re = re.compile(
            r"^\s*(?P<index>indices_\d+) = (?:tile_)?offset_(?P<offset>\d+) \+ "
            r"[^\n]*?cute\.arch\.thread_idx\(\)\[(?P<axis>\d+)\]\)"
            r"(?:\s*\*\s*(?:cutlass\.Int32\()?(?P<ept>\d+))?",
            flags=re.MULTILINE,
        )
        # Loose form: any ``indices_<n>`` line containing ``thread_idx``
        # under any wrapping. Used only for the wrapper-invariant
        # assertion below — never consulted for launch-dim values.
        indices_line_loose_re = re.compile(
            r"^\s*(?P<index>indices_\d+) = (?:tile_)?offset_\d+ \+ "
            r"[^\n]*?cute\.arch\.thread_idx\(\)",
            flags=re.MULTILINE,
        )
        matched_lines = 0
        for line_match in indices_line_re.finditer(final_kernel_text):
            if line_match.group("index") in grid_thread_extents:
                continue
            matched_lines += 1
            offset_id = line_match.group("offset")
            axis_text = line_match.group("axis")
            multiplier_text = line_match.group("ept")
            axis = int(axis_text)
            if not (0 <= axis < len(offset_thread_dims)):
                continue
            block_name = offset_block_sizes.get(offset_id)
            block_size = block_size_values.get(block_name or "")
            if block_size is None and block_name is not None:
                # The generated suffix is a logical block ID, not an index
                # into the tunable-only config list. Fixed leading axes and
                # aliased blocks can make those indices differ. Resolve via
                # BlockSizeInfo, as the grid producer does, so recovery cannot
                # inflate an unmasked root using another root's tile width.
                block_id = int(block_name.removeprefix("_BLOCK_SIZE_"))
                block_sizes = CompileEnvironment.current().block_sizes
                if 0 <= block_id < len(block_sizes):
                    config_block_size = block_sizes[block_id].from_config(config)
                    if isinstance(config_block_size, int):
                        block_size = config_block_size
            if block_size is None:
                continue
            elements_per_thread = int(multiplier_text) if multiplier_text else 1
            if elements_per_thread <= 0:
                # ``lane_index_expr`` only emits ``* <n>`` for ``n >= 1``;
                # a ``* 0`` multiplier would mean the index expression is
                # already invalid (every thread maps to offset 0), so
                # silently falling back to ``block_size`` here would
                # mask the very class of bug this recovery exists to
                # catch. Surface it loudly.
                raise AssertionError(
                    f"launch-dim recovery: non-positive "
                    f"elements_per_thread={elements_per_thread} for axis="
                    f"{axis}"
                )
            if block_size % elements_per_thread != 0:
                # The strategy invariant ``_thread_extent_for_axis``
                # rejects non-divisible ``block_size / nt`` at
                # construction, so the regex should never see a line
                # whose ``epT`` does not divide ``block_size`` evenly.
                # If a future codegen regression breaks that invariant,
                # surface it loudly here rather than silently
                # under-dimensioning the launch.
                raise AssertionError(
                    f"launch-dim recovery: block_size={block_size} not "
                    f"divisible by elements_per_thread={elements_per_thread} "
                    f"for axis={axis}"
                )
            line_extent = block_size // elements_per_thread
            offset_thread_dims[axis] = max(offset_thread_dims[axis], line_extent)
        # Wrapper-invariant assertion: when there is at least one
        # ``indices_<n> = ... thread_idx() ...`` line and we have
        # ``_BLOCK_SIZE_<n>`` constants to consume, the strict regex
        # above must have matched at least one of them. A loose match
        # without a strict match means the codegen emitted
        # ``thread_idx[axis]`` outside the ``cutlass.Int32(...)``
        # wrapper the strict regex anchors on; in that case the
        # launch dim would silently fall back to ``[1, 1, 1]`` and
        # under-dimension every kernel that uses a thread axis.
        if (
            offset_block_sizes
            and any(
                match.group("index") not in grid_thread_extents
                for match in indices_line_loose_re.finditer(final_kernel_text)
            )
            and matched_lines == 0
        ):
            raise AssertionError(
                "launch-dim recovery: indices_<n> lines reference "
                "cute.arch.thread_idx() but none matched the "
                "cutlass.Int32(...)-wrapped form. The strict regex in "
                "_launcher_block_arg is anchored on the wrapper emitted "
                "by lane_index_expr; if the codegen producer was changed "
                "to drop the wrapper, update the regex to match the new "
                "form."
            )
        dims = tuple(codegen.max_thread_block_dims)
        root_live_dims = tuple(codegen.root_thread_block_dims)
        referenced_dims = tuple(codegen.referenced_thread_block_dims)
        static_dims = tile_strategy.thread_block_dims()
        dim_exprs = tile_strategy.thread_block_dim_exprs()
        # The axes whose thread extent is a kernel argument (``hl.tile(n,
        # block_size=bsz)`` under ``static_shapes=False``): the static shapes
        # below hold a one for them, and the launch takes their extent from
        # the host constant at the end (``_symbolic_launch_block_arg``).
        symbolic_axes = tile_strategy.symbolic_thread_axes()
        if symbolic_axes and (
            dim_exprs is None
            or any(
                axis >= len(dim_exprs) or dim_exprs[axis] != expr
                for axis, expr in symbolic_axes.items()
            )
        ):
            raise exc.BackendUnsupported(
                self.name,
                f"launch axis {_symbolic_axes_text(symbolic_axes)} is sized by a "
                "kernel argument and the kernel's strategies share no single "
                "launch shape",
            )
        # The axes each root loop reads with their thread extents.  An axis the
        # final kernel no longer reads (an index dead-code elimination removed)
        # drops nothing.
        root_read_axes = [
            {axis: size for axis, size in root.items() if axis in final_thread_axes}
            for root in cute_codegen_state(codegen).root_read_axis_sizes
        ]
        static_threads = functools.reduce(operator.mul, static_dims, 1)
        dynamic_threads = functools.reduce(operator.mul, dims, 1)
        has_nested_device_loops = any(
            getattr(graph_info, "block_ids", None) is not None
            for graph_info in codegen.codegen_graphs
        )
        root_grid_dims = [1, 1, 1]
        device_ir = HostFunction.current().device_ir
        for block_ids in device_ir.grid_block_ids:
            for block_id in block_ids:
                # Strategy widths use local axes. Reduction/free axes can
                # precede the root in the actual CUDA layout, so use the same
                # physical mapping as TileStrategyDispatch.thread_block_dims.
                # Otherwise a dead root axis can inflate a different live
                # axis whose memory accesses correctly omitted surplus masks.
                axis = tile_strategy.thread_axis_for_block_id(block_id)
                size = tile_strategy.thread_extent_for_block_id(block_id)
                if axis is not None and size is not None and 0 <= axis < 3:
                    root_grid_dims[axis] = max(root_grid_dims[axis], size)
        root_static_dims = tuple(root_grid_dims)
        root_static_threads = functools.reduce(operator.mul, root_static_dims, 1)

        def _emits_cute_gemm(stmt: ast.AST) -> bool:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "gemm"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "cute"
                ):
                    return True
            return False

        kernel_has_mma = any(
            _emits_cute_gemm(stmt)
            for stmt in [*device_function.preamble, *device_function.body]
        )

        reduction_strategies = any(
            isinstance(strategy, ReductionStrategy)
            for strategy in tile_strategy.strategies
        )

        def launch_shape_flags(*, multi_phase: bool, tcgen05: bool) -> LaunchShapeFlags:
            return LaunchShapeFlags(
                multi_phase=multi_phase,
                nested_device_loops=has_nested_device_loops,
                mma=kernel_has_mma,
                tcgen05=tcgen05,
                reduction=reduction_strategies,
                synthetic_axes=bool(
                    cute_codegen_state(codegen).synthetic_arange_axis_sizes
                ),
                symbolic_axes=bool(symbolic_axes),
                pointwise_fact=bool(
                    CompileEnvironment.current().config_spec.pointwise_facts
                ),
            )

        recorded_tcgen05_block_shape = device_function.cute_state.tcgen05.block_shape
        specialized_root_tcgen05 = recorded_tcgen05_block_shape is not None or (
            _kernel_specialized_mma_impl(device_function, config=device_function.config)
            == "tcgen05"
            and root_static_dims != (1, 1, 1)
            and root_static_threads <= MAX_THREADS_PER_BLOCK
        )
        tcgen05_compact_dims = (
            recorded_tcgen05_block_shape if specialized_root_tcgen05 else None
        )
        if len(device_ir.phases) > 1 and tcgen05_compact_dims is None:
            phase_dims = _multi_phase_block_dims(
                device_function,
                exact_grid_thread_dims=tuple(exact_grid_thread_dims),
            )
            # An argument-sized axis never reaches here: the phases are
            # separate strategy branches, which share no single launch
            # shape expression, so ``symbolic_axes`` declined above.
            check_thread_block_dims(phase_dims, context=str(phase_dims))
            _reject_dropped_root_threads(phase_dims, root_read_axes)
            check_launch_shape(
                device_function,
                phase_dims,
                flags=launch_shape_flags(multi_phase=True, tcgen05=False),
            )
            return launcher_args_with_compile_options(
                f"block=({phase_dims[0]}, {phase_dims[1]}, {phase_dims[2]})"
            )
        if referenced_dims != (1, 1, 1):
            dims = referenced_dims
        elif has_nested_device_loops:
            dims = tuple(codegen.max_thread_block_dims)
        if functools.reduce(operator.mul, dims, 1) > MAX_THREADS_PER_BLOCK:
            if (
                root_static_dims != (1, 1, 1)
                and root_static_threads <= MAX_THREADS_PER_BLOCK
            ):
                dims = root_static_dims
            elif static_dims != (1, 1, 1) and static_threads <= MAX_THREADS_PER_BLOCK:
                dims = static_dims
        recorded_dims = tuple(
            max(
                codegen.max_thread_block_dims[axis],
                root_live_dims[axis],
                referenced_dims[axis],
            )
            for axis in range(3)
        )
        if tcgen05_compact_dims is not None:
            dims = tcgen05_compact_dims
        elif specialized_root_tcgen05:
            dims = root_static_dims
        elif final_thread_axes and (
            referenced_dims != (1, 1, 1) or has_nested_device_loops
        ):
            dims = tuple(
                max(size, root_static_dims[axis], recorded_dims[axis])
                if axis in final_thread_axes
                else size
                for axis, size in enumerate(dims)
            )
            dims = tuple(
                size if axis in final_thread_axes else 1
                for axis, size in enumerate(dims)
            )
        else:
            dims = tuple(
                min(size, recorded_dims[axis]) for axis, size in enumerate(dims)
            )
        current_threads = functools.reduce(operator.mul, dims, 1)
        if current_threads > MAX_THREADS_PER_BLOCK:
            if static_dims != (1, 1, 1) and static_threads <= MAX_THREADS_PER_BLOCK:
                dims = static_dims
            elif (
                root_live_dims != (1, 1, 1)
                and functools.reduce(operator.mul, root_live_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = root_live_dims
            elif (
                referenced_dims != (1, 1, 1)
                and functools.reduce(operator.mul, referenced_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = referenced_dims
        if (
            dims != (1, 1, 1)
            and static_dims != (1, 1, 1)
            and not has_nested_device_loops
            and static_threads < dynamic_threads
            # Synthetic free-``hl.arange`` thread axes are real launch lanes that
            # the strategy's ``static_dims`` does not know about, so do not fall
            # back to ``static_dims`` (which would drop them) when they are live.
            and not cute_codegen_state(codegen).synthetic_arange_axis_sizes
            # Nor when it is narrower than the root tile's live threads: the
            # final launch guard below would refuse it.
            and launch_drops_root_threads(static_dims, root_read_axes) is None
        ):
            dims = static_dims
        if (
            dim_exprs is not None
            and dim_exprs != ("1", "1", "1")
            # ``dim_exprs`` is the strategy's static per-axis launch shape; it
            # has no entry for synthetic free-``hl.arange`` axes, so adopting it
            # wholesale would shrink those axes back to 1. Skip it when they are
            # live (the synthetic extents are already folded into ``dims``).
            and not cute_codegen_state(codegen).synthetic_arange_axis_sizes
        ):
            if all(expr.isdigit() for expr in dim_exprs):
                expr_dims = tuple(int(expr) for expr in dim_exprs)
                if functools.reduce(
                    operator.mul, expr_dims, 1
                ) <= MAX_THREADS_PER_BLOCK and all(
                    expr_dim <= current_dim
                    for expr_dim, current_dim in zip(expr_dims, dims, strict=True)
                ):
                    dims = expr_dims
            # Otherwise an axis is sized by a kernel argument (``symbolic_axes``):
            # the static ``dims`` keep a one for it through the checks below,
            # and the launch substitutes the host constant at the end.
        if offset_thread_dims != [1, 1, 1]:
            candidate_dims = tuple(
                starmap(max, zip(dims, offset_thread_dims, strict=True))
            )
            if (
                functools.reduce(operator.mul, candidate_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = candidate_dims
        if dims == (1, 1, 1):
            dynamic_dims = tuple(codegen.max_thread_block_dims)
            if (
                dynamic_dims != (1, 1, 1)
                and functools.reduce(operator.mul, dynamic_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = dynamic_dims
            else:
                dims = DeviceFunction.current().tile_strategy.thread_block_dims()
        pointwise_dims = _pointwise_grid_thread_dims(
            live_extents,
            final_thread_axes,
            referenced_dims,
            has_pointwise_fact=bool(
                CompileEnvironment.current().config_spec.pointwise_facts
            ),
            has_nested_device_loops=has_nested_device_loops,
            has_synthetic_free_axes=bool(
                cute_codegen_state(codegen).synthetic_arange_axis_sizes
            ),
        )
        if pointwise_dims is not None:
            # Register broadcasts can create reduction strategies without an
            # executable reduction. Their static axis reservation can differ
            # from the root producer's physical mapping (especially when some
            # reduction thread counts are one). For a proven pointwise body
            # whose live axes all have grid producers, those emitted producers
            # own the launch, with larger statement references rejected rather
            # than inflating an unproved surplus mask. Keep reductions/MMA and independent
            # nested/free-arange axes on their existing launch paths.
            dims = pointwise_dims
        if any(
            required > actual
            for required, actual in zip(exact_grid_thread_dims, dims, strict=True)
        ):
            raise self.unsupported(
                "grid thread extents cannot fit the launch thread budget",
            )
        # Detect the silent-truncation case: codegen has already emitted
        # thread_idx[axis] references that assume a certain per-axis
        # extent (recorded in ``referenced_thread_block_dims``), but the
        # chosen launch ``dims`` for that axis is smaller. This happens
        # when the joint requested thread count exceeds 1024 and the
        # earlier fallback paths in this function dropped axes to (1, 1, 1)
        # to fit the budget. Under-dimensioning here would leave
        # cross-thread reductions (group_span, warp_reduce) operating
        # against nonexistent lanes, silently producing wrong results
        # (e.g. softmax with M*N>1024 returning max_err on the order of
        # tens). Surface this as ``BackendUnsupported`` so the autotuner
        # skips the config and falls back to a viable one.
        #
        # Skip this check when the joint referenced thread count fits
        # within ``MAX_THREADS_PER_BLOCK``: in that case any per-axis
        # mismatch comes from a strategy that intentionally launches
        # fewer threads than the codegen "references" (e.g. an outer-
        # tile axis with a single logical lane that still appears in a
        # warp-reduction call), and the CuTe runtime degenerates the
        # reduction to the live lanes without losing data.
        #
        # Skip this check for tcgen05-specialized matmul kernels: those
        # use a custom role-warp launch shape that intentionally differs
        # from the SIMT thread-axis counts (the latter being how many
        # threads the user-visible per-element loops would expect). The
        # tcgen05 code paths know which lanes are alive on their own.
        # Also skip when the kernel has any matmul / MMA call (checks for
        # cute.gemm calls): those paths cooperate within a warp through
        # CUTLASS MMA intrinsics that don't depend on the SIMT axis layout,
        # so the strategy can intentionally launch fewer threads on a
        # reduction axis (e.g. K) than the codegen "references" through the
        # strategy's per-block thread count.

        if (
            tcgen05_compact_dims is None
            and not specialized_root_tcgen05
            and not kernel_has_mma
        ):
            # Root indices can live in the grid's outer prefix, outside the
            # statement-reference recorder. Include those live axes when
            # validating an inner loop's thread requirements: checking only
            # the inner axes can hide a launch that dropped part of the tile.
            required_dims = tuple(
                max(ref_size, root_live_dims[axis])
                if axis in final_thread_axes
                else ref_size
                for axis, ref_size in enumerate(codegen.referenced_thread_block_dims)
            )
            referenced_threads = functools.reduce(operator.mul, required_dims, 1)
            if referenced_threads > MAX_THREADS_PER_BLOCK:
                for axis, ref_size in enumerate(required_dims):
                    if ref_size > 1 and axis < len(dims) and dims[axis] < ref_size:
                        raise exc.BackendUnsupported(
                            self.name,
                            (
                                f"launch dims {tuple(dims)} under-dimension"
                                f" required_thread_block_dims={required_dims}"
                                f" (axis {axis}: launched={dims[axis]} <"
                                f" referenced={ref_size}). Codegen would access"
                                f" nonexistent threads — joint requested"
                                f" thread count {referenced_threads} >"
                                f" {MAX_THREADS_PER_BLOCK}."
                            ),
                        )

        if (
            len(device_ir.grid_block_ids) > 1
            and tcgen05_compact_dims is None
            and not specialized_root_tcgen05
            and not kernel_has_mma
        ):
            _reject_multi_root_reduction_surplus(device_function, dims)
        if (
            tcgen05_compact_dims is None
            and not specialized_root_tcgen05
            and not kernel_has_mma
        ):
            _reject_dropped_root_threads(dims, root_read_axes)
        if tcgen05_compact_dims is not None and any(
            launched < planned
            for launched, planned in zip(dims, tcgen05_compact_dims, strict=True)
        ):
            # The role launch addresses every warp of the plan's block shape
            # (MMA, TMA, scheduler and C-input warps by linear warp id, the
            # pipeline-init barrier by their thread count); a launch narrower
            # than the plan would never start some of them and deadlock on
            # the first barrier.  The MMA detection declines the thread
            # layouts that get here, so this is a last line.
            raise exc.BackendUnsupported(
                self.name,
                f"tcgen05 role launch needs block={tuple(tcgen05_compact_dims)} "
                f"but the thread layout launches block={tuple(dims)}: a role "
                "warp would never start",
            )
        check_thread_block_dims(dims, context=str(tuple(dims)))
        if symbolic_axes:
            return launcher_args_with_compile_options(
                self._symbolic_launch_block_arg(
                    dims,
                    symbolic_axes,
                    [*device_function.preamble, *device_function.body],
                )
            )
        check_launch_shape(
            device_function,
            (dims[0], dims[1], dims[2]),
            flags=launch_shape_flags(
                multi_phase=False,
                tcgen05=tcgen05_compact_dims is not None or specialized_root_tcgen05,
            ),
        )
        return launcher_args_with_compile_options(
            f"block=({dims[0]}, {dims[1]}, {dims[2]})"
        )

    def _symbolic_launch_block_arg(
        self,
        dims: tuple[int, ...],
        symbolic_axes: dict[int, str],
        statements: Sequence[ast.AST],
    ) -> str:
        """The ``block=`` argument of a launch with an argument-sized axis.

        ``dims`` is the static launch shape, checked at codegen, with a one on
        every axis of ``symbolic_axes``; the axis is launched with its host
        constant instead (``_BLOCK_SIZE_n``, one thread per element) when the
        body reads its thread index, and the tuple goes through
        ``checked_thread_block_dims`` at launch, where the value is known.
        An axis whose index the body never reads keeps the one: more threads
        would only repeat the same work.  A leader guard
        (``cute.arch.thread_idx()[axis] == 0`` around a tile-uniform atomic)
        is not such a read: it holds for the one thread, and it is what keeps
        the atomic to one thread when the axis is launched wide.

        The body must not combine threads.  Every cross-thread construct was
        emitted for the static shape (a warp reduction over 32 column threads
        beside the symbolic row axis spans one warp only while the axis has
        one thread), so a body with any is declined rather than launched
        wrong; the reduction over the symbolic dim itself is declined by
        ``BlockReductionStrategy.codegen_reduction``.
        """
        read_axes = _thread_axes_read_outside_leader_guards(statements)
        live_axes = {
            axis: expr for axis, expr in symbolic_axes.items() if axis in read_axes
        }
        if not live_axes:
            return f"block=({dims[0]}, {dims[1]}, {dims[2]})"
        for axis in live_axes:
            if dims[axis] != 1:
                raise exc.BackendUnsupported(
                    self.name,
                    f"launch axis {axis} is sized by the kernel argument "
                    f"{live_axes[axis]} but static tracking claims {dims[axis]} "
                    "threads on it",
                )
        cross_thread = _cross_thread_constructs(statements)
        if cross_thread:
            raise exc.BackendUnsupported(
                self.name,
                f"launch axis {_symbolic_axes_text(live_axes)} is sized by a "
                f"kernel argument and the body combines threads "
                f"({', '.join(cross_thread)}), which was emitted for the static "
                "launch shape",
            )
        launch = tuple(live_axes.get(axis, str(size)) for axis, size in enumerate(dims))
        return (
            f"block=_cute_checked_block_dims(({launch[0]}, {launch[1]}, {launch[2]}))"
        )

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: list[Argument] | None = None,
    ) -> list[str]:
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out

    def create_program_ids(
        self, fn: DeviceFunction, pid_type: str, block_ids: list[int]
    ) -> ProgramIDs | None:
        """The tcgen05 persistent scheduler for persistent pid types on a
        tcgen05 GEMM body; ``None`` (the generic strategies) otherwise."""
        if not pid_type.startswith("persistent"):
            return None
        from .cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
        from .cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY

        # The register-MMA family owns the whole body (one CTA per tile); the
        # tcgen05 persistent scheduler would need a tcgen05 plan it never gets.
        if fn.config.get(WARP_MMA_FAMILY_KEY) == MATMUL_FAMILY_WARP_MMA:
            return None
        if _kernel_specialized_mma_impl(fn, config=fn.config) != "tcgen05":
            return None
        from .tcgen05.scheduler_pids import Tcgen05PersistentProgramIDs

        return Tcgen05PersistentProgramIDs(is_blocked=pid_type == "persistent_blocked")

    def lower_owned_lane_loop(self, codegen: GenerateAST) -> None:
        from .lane_loops.root_passes import run_root_lane_passes

        run_root_lane_passes(codegen)

    def tracks_statement_owners(self) -> bool:
        return True

    def tile_begin_expr(self, codegen: GenerateAST, block_idx: int) -> str | None:
        from .tile_ops import cute_tile_begin_expr

        return cute_tile_begin_expr(codegen, block_idx)

    def rewrite_lowered_root(
        self, codegen: GenerateAST, grid_state: DeviceGridState, body: list[ast.AST]
    ) -> bool:
        from .direct_affine_lowering import try_lower_direct_affine_root
        from .thread_budget import record_root_read_axes

        record_root_read_axes(codegen, grid_state, body)
        return try_lower_direct_affine_root(codegen, grid_state, body)

    def explain_root_failure(
        self, codegen: GenerateAST, error: exc.BackendUnsupported
    ) -> exc.BackendUnsupported | exc.InvalidConfig:
        """Report a generic-lowering refusal after a fragment capacity decline.

        The computed-fragment owner declines a root whose shared memory exceeds
        the CTA capacity so the ordinary lowering can own it; when that lowering
        refuses instead (lane restores, repeated block ids), the config, not the
        kernel, failed: the overflow is the actionable message. Callers wrap the
        statement visit and the device function finalization.

        ``RegisterTileUnsupported`` is not a refusal but the register tile's
        regeneration signal (``generate_ast`` retries with the rolled lane) and
        passes through unchanged, decline or not.
        """
        from .register_tile_admission import RegisterTileUnsupported

        if isinstance(error, RegisterTileUnsupported):
            return error
        declined = (
            codegen.device_function.cute_state.root_plans.fragment_capacity_decline
        )
        if declined is None:
            return error
        needed, capacity = declined
        return exc.InvalidConfig(
            f"computed fragments need {needed} shared bytes, exceeding {capacity}, "
            f"and no other lowering owns this root: {error}"
        )

    def validate_codegen_config(
        self, env: CompileEnvironment, device_ir: DeviceIR, config: Config
    ) -> None:
        if config.get("cute_fragment_warp_scan", False):
            from .autotune.heuristics.cute_fragment_warp_scan import (
                validate_warp_scan_request,
            )

            validate_warp_scan_request(env, device_ir, config)
        if config.get("cute_fragment_scan_exports", False):
            from .autotune.heuristics.cute_fragment_scan_exports import (
                validate_scan_exports,
            )

            validate_scan_exports(env, device_ir, config)
        if config.get("cute_fragment_warp_producer_regions", False):
            from .autotune.heuristics.cute_fragment_warp_producer_regions import (
                validate_warp_producer_regions,
            )

            validate_warp_producer_regions(env, device_ir, config)

    def after_device_codegen(
        self, codegen: GenerateAST, host: HostFunction, config: Config
    ) -> None:
        from .cute_reshape import run_deferred_rebound_checks
        from .host_remainder_fills import append_device_remainder_fills
        from .lane_loops.root_passes import reject_lane_reads_without_lane_loops

        codegen.device_function.cute_state.finalize_tcgen05_paired_fanout_stores()
        # Re-binding checks deferred from tcgen05 epilogue chains that no
        # store drained (chains ending in an atomic or a specialized
        # store path).
        run_deferred_rebound_checks(codegen)
        if codegen.suppressed_lane_names:
            reject_lane_reads_without_lane_loops(
                codegen.device_function.body,
                codegen.suppressed_lane_names,
                codegen.device_function.dce_vars,
                backend=self,
            )
        cute_codegen_state(codegen).remainder_fills = append_device_remainder_fills(
            codegen, host, config
        )

    def finalize_host_statements(
        self, codegen: GenerateAST, host: HostFunction, statements: list[ast.AST]
    ) -> tuple[list[ast.AST], list[ast.AST]]:
        from ..compile_environment import CompileEnvironment
        from .host_metadata import can_bake_tensor_shapes
        from .host_metadata import post_kernel_metadata_statements
        from .host_remainder_fills import lower_host_remainder_fills
        from .host_sums import PAIRED_SUM_KEY
        from .host_sums import lower_host_sums
        from .host_zero_fill_launches import lower_host_zero_fill_launches
        from .host_zero_fill_launches import zero_fill_launches_apply
        from .host_zero_fills import lower_host_zero_fills

        env = CompileEnvironment.current()
        fn = codegen.device_function
        # The wrapper plans index the launcher's argument order, so they are
        # resolved before the host passes below rewrite the wrapper.
        metadata = post_kernel_metadata_statements(codegen, host)
        if not can_bake_tensor_shapes(codegen):
            statements = [
                statement_from_string(
                    f"{fn.name}._helion_cute_disable_bake_tensor_shapes = True"
                ),
                *statements,
            ]
        statements = lower_host_remainder_fills(
            statements, cute_codegen_state(codegen).remainder_fills
        )
        statements = lower_host_sums(
            host, statements, cast("str", fn.config.get(PAIRED_SUM_KEY, "off"))
        )
        statements = lower_host_zero_fills(host, statements, env, fn.config)
        fill_kernels: list[ast.AST] = []
        if zero_fill_launches_apply(fn, env, fn.config):
            # The zero fills that survive the elision launch as
            # dependent fill kernels ahead of this (``cute_pdl``)
            # kernel instead of memset nodes.
            statements, fill_kernels = lower_host_zero_fill_launches(
                host, statements, env=env, kernel_name=fn.name
            )
        return statements, [*metadata, *fill_kernels]

    def canonicalizes_reduction_loop_blocks(self) -> bool:
        return True

    def prepare_device_loop(
        self,
        graph_info: ForLoopGraphInfo,
        state: CodegenState,
        device_loop: DeviceLoopState,
    ) -> None:
        from .loop_carries import prepare_device_loop

        prepare_device_loop(graph_info, state, device_loop)

    def finish_device_loop_body(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        outputs: list[object],
        initials: dict[int, str],
    ) -> None:
        from .loop_carries import finish_device_loop_body

        finish_device_loop_body(state, device_loop, outputs, initials)

    def synthetic_thread_axis_sizes(self, codegen: GenerateAST) -> Mapping[int, int]:
        return cute_codegen_state(codegen).synthetic_arange_axis_sizes

    def consume_root_lane_loop_suppression(self, fn: DeviceFunction) -> bool:
        return fn.cute_state.consume_root_lane_loop_suppression()

    def reorder_post_loop_statements(
        self, fn: DeviceFunction, body: list[ast.AST]
    ) -> list[ast.AST]:
        return fn.cute_state.move_tcgen05_post_loop_stmts_to_end(body)

    def roots_fissioned(self, env: CompileEnvironment) -> bool:
        return cute_env_state(env).fission_plan is not None

    def split_persistent_post_loop(
        self, fn: DeviceFunction, body: list[ast.AST]
    ) -> tuple[list[ast.AST], list[ast.stmt]]:
        cute_state = fn.cute_state
        if not cute_state.has_tcgen05_post_loop_marks:
            return body, []
        statements = cast("list[ast.stmt]", body)
        post_loop = [s for s in statements if cute_state.is_tcgen05_post_loop(s)]
        return [
            s for s in statements if not cute_state.is_tcgen05_post_loop(s)
        ], post_loop

    def collective_sibling_blocks(
        self, env: CompileEnvironment, config: Config
    ) -> tuple[set[int], set[tuple[int, ...]]] | None:
        from .matmul_utils import collective_sibling_blocks

        return collective_sibling_blocks(env, config)

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from ..per_thread_planning import plan_per_thread_loop_strategy

        return plan_per_thread_loop_strategy(self, fn, block_ids, config)

    # -- thread layout and lane loops of the tile strategies -------------------

    def loop_var_prefixes(self) -> tuple[str, str]:
        # The CuTe DSL preprocessor reserves ``offset_<n>`` / ``step_<n>``
        # (see ``TileStrategy.__init__``).
        return ("tile_offset", "tile_step")

    def plans_launch_thread_axes(self) -> bool:
        return True

    def single_thread_tile_claims_axis(self) -> bool:
        return False

    def flat_strided_tile_walks_elements(self) -> bool:
        return True

    def tracks_flat_grid_origins(self) -> bool:
        return True

    def reflattens_pointwise_tiles(self) -> bool:
        return True

    def tunes_lane_vector_widths(self) -> bool:
        return True

    def thread_axis_limit(self) -> int | None:
        return 3

    def record_launch_thread_extent(
        self, fn: DeviceFunction, name: str, axis: int, extent: int
    ) -> None:
        fn.cute_state.lanes.launch_thread_extents[name] = (axis, extent)

    def record_grid_thread_extent(
        self, fn: DeviceFunction, name: str, axis: int, extent: int
    ) -> None:
        fn.cute_state.lanes.grid_thread_extents[name] = (axis, extent)

    def referenced_thread_axes(self, statements: Sequence[ast.AST]) -> list[int]:
        return referenced_thread_axes(statements)

    def record_vloop_wrapper(
        self,
        fn: DeviceFunction,
        wrapper: VecLaneWrapper,
        *,
        block_id: int,
        lane_var: str,
        index_var: str,
        mask_var: str | None,
        begin: object,
        extent: object,
        step: object,
    ) -> None:
        from ..compile_environment import CompileEnvironment
        from .device_state import CuteVloopWrapperFact

        vec_width = fn.lane_contexts[block_id].vec_width
        # ``base = begin + pid * block + ... * V`` is V-aligned only from a
        # zero origin walked in unit steps, and a V-wide chunk cannot straddle
        # an extent that is a multiple of V.  The V-loop sink
        # (``cute/sink_vector_loops.py``) finds the constexpr V-loop by this
        # fact wherever it nests.
        aligned = isinstance(begin, int) and begin == 0 and step in (None, 1)
        fn.cute_state.lanes.vloop_sink_wrappers[wrapper.vec_lane_var] = (
            CuteVloopWrapperFact(
                block_id=block_id,
                vec_width=vec_width,
                lane_var=lane_var,
                vec_lane_var=wrapper.vec_lane_var,
                base_index_var=wrapper.base_index_var,
                index_var=index_var,
                mask_var=mask_var,
                uniform_vector_mask=aligned
                and CompileEnvironment.current().specialized_multiple(
                    extent, vec_width
                ),
                chunk_base_aligned=aligned,
            )
        )

    def apply_lane_cluster_split(
        self, strategy: PerThreadNDTileStrategy, state: CodegenState
    ) -> None:
        from ..compile_environment import CompileEnvironment
        from .lane_cluster_split import apply_cluster_split

        apply_cluster_split(strategy, CompileEnvironment.current(), state)

    def check_lane_loop_nest(self, state: DeviceLoopState) -> None:
        from .lane_loops.nest_check import check_lane_loop_nest

        check_lane_loop_nest(state)

    def add_lane_nest_barriers(
        self,
        state: DeviceGridState,
        placement: list[ast.AST | LanePlacement],
        scopes: Sequence[LaneScope],
        rename_groups: Mapping[str, str],
    ) -> None:
        from .lane_loops.nest_check import add_lane_nest_barriers

        add_lane_nest_barriers(state, placement, scopes, rename_groups)

    def demote_grid_tile_vec_stores(
        self,
        state: DeviceGridState,
        body: list[ast.AST],
        placement: list[ast.AST | LanePlacement] | None,
    ) -> bool:
        """Restore the collected stores a later statement of their loop observes.

        ``placement`` distributes ``body`` around the lane loops, or is None
        when the full nest is kept.  Only the statements that stay inside a
        store's lane loop run before its flush: a lane-invariant statement
        placed after the loop follows the flush and one placed before it
        precedes the store's site, both ordered by ``distribute_lane_loops``.
        True when a site was demoted; its flush left the loop and its scalar
        store joined the body, so the caller places the body again.  Only the
        per-thread lane strategies collect vector stores.
        """
        from ..tile_strategy import PerThreadFlattenedTileStrategy
        from ..tile_strategy import PerThreadNDTileStrategy
        from .lane_loops.placement import statements_inside_loop
        from .memory_ops import demote_reordered_tile_vec_stores

        strategy = state.strategy
        if not isinstance(
            strategy, (PerThreadNDTileStrategy, PerThreadFlattenedTileStrategy)
        ):
            # Only the per-thread lane strategies collect vector stores; the
            # other strategies have none.
            return False
        demoted = False
        for context in strategy.lane_contexts_with_vec_stores():
            inside = None
            if placement is not None:
                assert context.vloop is not None
                lane_var = state.vec_wrapper_lane_var(context.vloop)
                inside = statements_inside_loop(placement, lane_var)
            if demote_reordered_tile_vec_stores(
                context, strategy.fn, body, inside=inside
            ):
                demoted = True
        return demoted

    def demote_loop_tile_vec_stores(
        self, strategy: TileStrategy, block_ids: list[int]
    ) -> None:
        from .memory_ops import demote_reordered_tile_vec_stores

        for block_id in block_ids:
            context = strategy.fn.lane_contexts[block_id]
            assert context.vloop is not None
            demote_reordered_tile_vec_stores(context, strategy.fn, context.vloop.body)
