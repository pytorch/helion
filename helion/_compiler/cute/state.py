"""Typed CuTe backend state: the objects behind the core ``backend_state`` slots.

Role: state.  Core compile objects carry one opaque ``backend_state`` slot
each, filled by the ``Backend.new_*_state`` hooks; this module defines what
the CuTe backend puts there and the namespaces the per-device-function state
is grouped into:

* ``CuteEnvState``: per ``CompileEnvironment`` (facts that live across the
  compile phases: the materialized-fission plan, resolved wrapper plans, the
  proven TMA inputs, ...).
* ``CuteCodegenState``: per ``GenerateAST`` pass (wrapper plans of the pass,
  the synthetic ``hl.arange`` thread axes, the branch path).
* ``CuteTcgen05State`` / ``CuteFlashState`` / ``CuteLaneState`` /
  ``CuteRootPlans``: the namespaces of ``CuteDeviceFunctionState``
  (``device_state.py``), one per emitter family: the tcgen05 GEMM pipeline,
  the attention bodies, lane loops / thread axes / the launch shape, and the
  whole-root plan slots the ``PlannedRootEmitter`` registry reads.  The
  lane-loop contexts themselves are core state
  (``DeviceFunction.lane_contexts``, ``lane_loop_context.py``).

This module is a leaf: it imports nothing from the emitters (``cute_mma``,
``backend``, the launcher import the state) and names plan types only under
``TYPE_CHECKING``.  CuTe code reads the env state through the typed accessor
``cute_env_state``; core (and, for one release, the emitters) read every state
through the ``cute_*`` alias attributes (``attribute_alias.AttributeAlias``).
The core defaults of the ``Backend.new_*_state`` hooks return these objects
UNSEEDED for every backend (``new_unseeded_device_state``) until core's
unconditional ``cute_state`` reads sit behind hooks; only ``CuteBackend`` seeds
the device state from the config.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from .tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY

if TYPE_CHECKING:
    import ast

    import sympy
    import torch
    from torch.fx.node import Node

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import GenerateAST
    from .attention_plan import AttentionScorePlan
    from .block_scaled_mma import BlockScaledMmaPlan
    from .chunk_prepare import CuteChunkPreparePlan
    from .chunk_recurrence import CuteChunkRecurrencePlan
    from .collective_matmul import CollectiveMmaSite
    from .completed_matmul_sum import CompletedMatmulSum
    from .cute_epilogue import Tcgen05GroupedTailEpilogueMatch
    from .cute_flash_bwd import AttentionBwdMatch
    from .cute_flash_gated import GatedAttentionMatch
    from .cute_flash_gated import GatedAttentionPlan
    from .cute_mma import _Tcgen05AuxPipelinePlan
    from .cute_mma import _Tcgen05SchedPipelinePlan
    from .cute_mma_dag import MmaDagPlan
    from .cute_register_subblocks import RegisterSubblockPlan
    from .cute_warp_mma_gemm import CuteWarpMmaGemmMatch
    from .device_state import CuteDeviceFunctionState
    from .device_state import CuteTcgen05MatmulPlan
    from .device_state import CuteTcgen05StoreValue
    from .device_state import CuteVloopLoadFact
    from .device_state import CuteVloopStoreFact
    from .device_state import CuteVloopWrapperFact
    from .device_state import Tcgen05AbStartupPrefill
    from .direct_affine_candidate import DirectAffineCandidate
    from .direct_affine_plan import DirectAffinePlan
    from .epilogue_fanout import FanoutStore
    from .fixed_token_rank1_recurrence import CuteFixedTokenRank1Plan
    from .flex_bwd_match import FlexBwdMatch
    from .flex_match import FlexMatch
    from .fragment_epilogue import Tcgen05FragmentEpiloguePlan
    from .gdn_recurrence import CuteGdnRecurrencePlan
    from .host_remainder_fills import RemainderFill
    from .iota_utils import FreeArangeLanes
    from .materialized_fission import MaterializedFissionPlan
    from .register_region import RegisterRegionPlan
    from .resident_reductions import ResidentReductionLayout
    from .resident_sequence import SequenceRegion
    from .selection.radix_selection import RadixSelectionPlan
    from .selection.topk import CuteTopKPlan
    from .simt_gemm import SimtGemmSite
    from .single_token_rank1_recurrence import CuteSingleTokenRank1Plan
    from .split_single_token_rank1_recurrence import CuteSplitSingleTokenRank1Plan


# ---------------------------------------------------------------------------
# Per CompileEnvironment / per GenerateAST
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CuteEnvState:
    """``CompileEnvironment.backend_state`` of the CuTe backend."""

    resolved_wrapper_plans: list[dict[str, object]] = dataclasses.field(
        default_factory=list
    )
    # Dynamic reduction extents whose CuTe persistent thread layout was
    # sized from the size hint; the bound kernel keys their power-of-2
    # bucket once code generation has used one.
    hint_sized_reduction_extents: set[sympy.Expr] = dataclasses.field(
        default_factory=set
    )
    fission_plan: MaterializedFissionPlan | None = None
    half_atomic_output_promotions: dict[str, torch.dtype] = dataclasses.field(
        default_factory=dict
    )
    # Loops ``expose_scaled_contractions`` rewrote; read after
    # ``prepare_graph_lowerings`` (``cute/ir_passes.py``).
    scaled_contraction_count: int = 0
    # ``expose_dot_scaled_scales`` rewired a scale; with
    # ``scaled_contraction_count`` it decides whether the codegen-active block
    # set is recomputed after graph CSE (``cute/ir_passes.py``).
    dot_scaled_scales_exposed: bool = False
    # Internal stage compilers may inherit a proved TensorMap-aligned view
    # of an owning kernel input. Only the stage builder populates this set;
    # ordinary input tensors still require their runtime cache-key proof.
    proven_tma_inputs: set[torch.Tensor] = dataclasses.field(default_factory=set)
    # Set by ``generate_ast`` while it regenerates a kernel whose
    # register-tile lane nesting the split-time lowering rejected
    # (``cute/register_tile_admission.py``); the persistent reduction
    # strategy then keeps its rolled lane nesting.
    register_tile_disabled: bool = False


@dataclasses.dataclass
class CuteCodegenState:
    """``GenerateAST.backend_state`` of the CuTe backend."""

    wrapper_plans: list[dict[str, object]] = dataclasses.field(default_factory=list)
    uses_matmul: bool = False
    # Count declarations, not False-to-True transitions: an unrelated
    # matmul path must not inherit a register chain's shape-bake exemption.
    matmul_declaration_count: int = 0
    # Synthetic per-thread axes allocated for free/unbound ``hl.arange``
    # index dims that are not bound to any tile/reduction/grid axis. Maps a
    # stable per-arange key (length, start-repr, step-repr) to the launch
    # thread axis it occupies. ``synthetic_arange_axis_sizes`` records the
    # extent of each allocated synthetic axis so the launch-dim recovery can
    # grow the thread block to cover those lanes.
    synthetic_arange_axes: dict[tuple[object, ...], int] = dataclasses.field(
        default_factory=dict
    )
    synthetic_arange_axis_sizes: dict[int, int] = dataclasses.field(
        default_factory=dict
    )
    # Positional lane classes of the free ``hl.arange`` dims, the synthetic
    # axis keys when they cover the kernel (``cute/iota_utils.py``).
    free_arange_lanes: FreeArangeLanes | None = None
    # ``(load/store node, synthetic axis key) -> index position``: one
    # synthetic lane may address only one index dim of a given access.
    synthetic_arange_access_positions: dict[tuple[Node, object], int] = (
        dataclasses.field(default_factory=dict)
    )
    # Stack of ``(if_node_id, branch_side)`` entries describing the
    # mutually-exclusive control-flow branch the current codegen is inside.
    # ``branch_side`` is 0 for the ``if`` body and 1 for the ``else`` body of
    # a given dynamic ``_if``. Two free ``hl.arange`` dims that live in
    # mutually-exclusive branches (their paths diverge at a common ``_if``)
    # may reuse the same synthetic thread axis since only one branch ever
    # runs per program instance.
    branch_path: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    # The branch path captured when each synthetic arange axis was allocated,
    # so a later arange in a mutually-exclusive branch can reuse it.
    synthetic_arange_axis_branch_paths: dict[int, list[list[tuple[int, int]]]] = (
        dataclasses.field(default_factory=dict)
    )
    # The branch path under which a strategy (e.g. a reduction) claimed a
    # thread axis. A free ``hl.arange`` in a branch mutually-exclusive with
    # every recorded user of a strategy axis may reuse that axis instead of
    # claiming a fresh one (which would silently widen the launch block and
    # turn the strategy's single-axis shared-memory reduction into a
    # cross-axis race). Mirrors ``synthetic_arange_axis_branch_paths``.
    strategy_axis_branch_paths: dict[int, list[list[tuple[int, int]]]] = (
        dataclasses.field(default_factory=dict)
    )
    # Free ``hl.arange`` dims whose (joint) thread count would exceed the
    # 1024-thread budget are chunked onto a sequential lane loop instead of
    # claiming a fresh thread axis. Maps the per-arange ``key`` to the
    # resolved per-thread coordinate expression so a load and store over the
    # same arange share one lane loop.
    synthetic_arange_lane_exprs: dict[tuple[object, ...], str] = dataclasses.field(
        default_factory=dict
    )
    # Per root loop: the thread extent of each launch axis the root's own
    # statements (index setup, body, lane loops) read.  A launch narrower
    # than one of them drops that root's threads; a root's axis no
    # statement reads needs one thread (``_launcher_block_arg``).
    root_read_axis_sizes: list[dict[int, int]] = dataclasses.field(default_factory=list)
    # The host remainder fills the device body took over
    # (``append_device_remainder_fills``); their host statements are dropped
    # when the host wrapper is finalized.
    remainder_fills: list[RemainderFill] = dataclasses.field(default_factory=list)


def cute_codegen_state(codegen: GenerateAST) -> CuteCodegenState:
    """The CuTe state of ``codegen`` (``codegen.backend_state``, typed)."""
    return cast("CuteCodegenState", codegen.backend_state)


def cute_env_state(env: CompileEnvironment) -> CuteEnvState:
    """The CuTe state of ``env`` (``env.backend_state``, typed)."""
    return cast("CuteEnvState", env.backend_state)


def new_env_state() -> CuteEnvState:
    """``CuteBackend.new_env_state``."""
    return CuteEnvState()


def new_codegen_state() -> CuteCodegenState:
    """``CuteBackend.new_codegen_state``."""
    return CuteCodegenState()


def new_unseeded_device_state() -> CuteDeviceFunctionState:
    """A bare ``CuteDeviceFunctionState``: the core default of
    ``Backend.new_device_state`` for every other backend (core still reads the
    launch facts, post-loop marks and branch path off it unconditionally).
    Config keys are not read: a Triton/Pallas config carrying
    ``tcgen05_split_k`` keeps the key ignored, as the ``backend_name == "cute"``
    gate the seed used to sit behind did."""
    from .device_state import CuteDeviceFunctionState

    return CuteDeviceFunctionState()


def new_device_state(config: Config) -> CuteDeviceFunctionState:
    """``CuteBackend.new_device_state``: the per-``DeviceFunction`` state,
    seeded from ``config``.

    ``tcgen05_split_k``: the flat grid holds ``split_k`` CTAs per output tile
    (``block_idx // split_k`` is the tile's program id); the tcgen05 pipeline
    marks the config applied or the launch codegen rejects it
    (``DeviceFunction.codegen_function_call``).
    """
    state = new_unseeded_device_state()
    split_k = config.get(TCGEN05_SPLIT_K_CONFIG_KEY, 1)
    if isinstance(split_k, int) and split_k > 1:
        state.tcgen05.program_id_divisor = split_k
        state.lanes.launch_grid_multiplier = split_k
    return state


# ---------------------------------------------------------------------------
# The namespaces of CuteDeviceFunctionState
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CuteTcgen05State:
    """The tcgen05 GEMM pipeline's per-kernel state (``cute_state.tcgen05``)."""

    store_values: dict[str, CuteTcgen05StoreValue] = dataclasses.field(
        default_factory=dict
    )
    grouped_tail_proofs: dict[Node, Tcgen05GroupedTailEpilogueMatch] = (
        dataclasses.field(default_factory=dict)
    )
    # A same-graph fanout shares one accumulator pipeline transaction.
    # The first store waits; the final store releases/advances only after
    # every output has read TMEM. Each TMA store gets separate descriptors.
    emitted_store_nodes: dict[int, set[Node | None]] = dataclasses.field(
        default_factory=dict
    )
    pending_paired_stores: dict[Node, FanoutStore] = dataclasses.field(
        default_factory=dict
    )
    # FX matmul / hl.dot / addmm nodes lowered through tcgen05. The store
    # path uses this to recognize fused epilogue chains that must use the
    # tcgen05 store splice instead of falling through to SIMT store codegen.
    matmul_fx_nodes: set[Node] = dataclasses.field(default_factory=set)
    # Pointwise ops on a tcgen05 epilogue chain whose block-id re-binding
    # check waits for the store's epilogue classifier
    # (``cute_reshape.check_pointwise_rebound_block_ids``).
    deferred_rebound_pointwise_nodes: list[Node] = dataclasses.field(
        default_factory=list
    )
    # ``build_inner_outputs_index_from_graphs`` of the codegen graphs,
    # built once for those checks (the graphs are fixed per codegen).
    rebound_inner_outputs_index: dict[int, tuple[Node | None, ...]] | None = None
    # tcgen05 matmul anchor -> registered result var. Fused epilogue walks
    # from a store value back to the anchor and reuses the store value
    # registered under this result var, even when user-visible names were
    # renamed through casts or epilogue nodes.
    matmul_fx_node_result_vars: dict[Node, str] = dataclasses.field(
        default_factory=dict
    )
    fragment_epilogue_plan: Tcgen05FragmentEpiloguePlan | None = None
    # Rejected proofs are stable for this per-config codegen state. Keep
    # the tile shape in the key so callers cannot accidentally reuse a
    # verdict if this helper is ever exercised with multiple shapes.
    rejected_fragment_epilogue_plans: set[tuple[Node, int, int, int]] = (
        dataclasses.field(default_factory=set)
    )
    matmul_plan: CuteTcgen05MatmulPlan | None = None
    # Variable-name containers allocated in cute_mma and consumed by
    # program_id / memory_ops role builders.
    sched_pipeline_plan: _Tcgen05SchedPipelinePlan | None = None
    aux_pipeline_plan: _Tcgen05AuxPipelinePlan | None = None
    ab_startup_prefill: Tcgen05AbStartupPrefill | None = None
    # Name of the CLC throttle named barrier once the TMA-load role has
    # emitted its per-tile credit (``cute_mma``); the scheduler-warp body
    # (``program_id``) syncs on it before every CLC query only then, so a
    # producer path without the credit never leaves the scheduler waiting.
    clc_throttle_barrier: str | None = None
    # The materialized-operand dependency wait was emitted inside the
    # initial AB prefetch (independent operand first); the role-local
    # prelude must not wait again ahead of those loads.
    pdl_wait_in_prefetch: bool = False
    # ``cute_pdl`` on a tcgen05 kernel: the launch carries ``use_pdl`` and
    # the generic entry wait (``passes.pdl.pdl_applies``) heads
    # the body, ahead of every prologue read.
    pdl_entry_wait: bool = False
    # One-shot clustered tcgen05 kernels: the cluster ``pipeline_init_arrive``
    # statement after which ``program_id`` re-emits the TMA-load warp's
    # role block (see ``cute_mma`` ``tcgen05_hoist_tma_role``).
    tma_role_hoist_anchor: ast.stmt | None = None
    # Row-union kernels whose TMA-load role is hoisted: the cluster
    # ``pipeline_init_wait`` the role executes once in its prelude, after
    # its scheduler setup and ahead of its persistent loop.
    tma_role_prelude_wait: str | None = None
    # ``cute_pdl`` on a persistent role-local tcgen05 kernel: the TMA-load
    # role's ``griddepcontrol_wait`` is emitted per tile right ahead of the
    # first stage (``cute_mma``), after the tile's address math, so
    # ``program_id`` leaves the role's prelude without a wait.
    tma_role_wait_per_tile: bool = False
    # One-shot TMA-load role setup (the once-per-CTA TMA partitions of
    # the whole tiled operands, ``cute_mma``) that ``program_id`` places
    # in the role's prelude ahead of its cluster join.
    tma_role_prelude_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    per_tile_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    post_loop_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    tma_load_role_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    mma_exec_role_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    epi_role_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    epi_role_prelude_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    epi_role_full_tile_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    epi_role_edge_tile_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    epi_role_tile_counter_var: str | None = None
    epi_role_tile_counter_increment_per_tile: bool = True
    cluster_shape: tuple[int, int, int] | None = None
    block_shape: tuple[int, int, int] | None = None
    # ``min_blocks_per_mp`` the tcgen05 role launch requires (the
    # epilogue warp groups' 10+-warp block runs without a ``setmaxnreg``
    # split and needs ptxas to cap the per-thread allocation at
    # ``65536 / threads``); emitted as ``_helion_cute_min_blocks_per_mp``
    # by the device function.
    min_blocks_per_mp: int | None = None
    # Active block-id remapping for matmul operand re-materialization.
    # Maps a source block_id (e.g. the rhs operand's loop-invariant
    # contraction index) to the active contraction block_id so a
    # re-lowered operand load reads ``rhs[..., k, ...]`` per contraction
    # step instead of the loop-invariant ``rhs[..., m, ...]``.  Empty
    # except while re-materializing a matmul operand load.
    matmul_operand_block_remap: dict[int, int] = dataclasses.field(default_factory=dict)
    # Active block-id -> raw index-expression override for matmul operand
    # re-materialization.  Used by the static-MN-collapse baddbmm path: the
    # rhs free (N) axis shares a block_id with the lhs M (output) axis, so
    # the standard resolver would index the rhs at the M thread index
    # (computing only the diagonal).  Re-lowering the rhs with this override
    # makes its N axis read a serial N-loop variable instead, and suppresses
    # masking for that axis (the serial loop already covers exactly [0, C)).
    # Empty except while re-materializing such an operand load.
    matmul_operand_index_override: dict[int, str] = dataclasses.field(
        default_factory=dict
    )
    # ``tcgen05_split_k``: the flat grid carries ``divisor`` CTAs per
    # output tile, so the program-id decode reads ``block_idx // divisor``
    # and the launch grid is multiplied by it
    # (``CuteLaneState.launch_grid_multiplier``).  ``split_k_applied``
    # records that the tcgen05 pipeline honored the split; the launch
    # codegen rejects the config otherwise.
    program_id_divisor: int = 1
    split_k_applied: bool = False


@dataclasses.dataclass
class CuteFlashState:
    """The attention bodies' per-kernel state (``cute_state.flash``)."""

    # Set by the backend's flash-attention detector when the fused
    # tcgen05 QK->softmax->PV path is active (HELION_CUTE_FLASH). Holds the
    # tile_n device-loop block ids. The dedicated flash codegen emits the
    # whole device body and host launch.
    block_ids: list[int] | None = None
    score_plan: AttentionScorePlan | None = None
    # Set by the flex-envelope matcher (``flex_match.py``): the
    # whole-graph-set match, probed once per device function. Every
    # device loop of the two BlockMask list walks answers ``mma_mode``
    # from it, and the flash codegen reads its operands from it.
    flex_match: FlexMatch | None = None
    flex_probed: bool = False
    # Launch block thread count for the flash path: 128 (single-warpgroup
    # Stage-3) or 256 (Stage-4 warp-spec, double-buffered-S overlap).
    threads: int = 128
    # Set by the gated (softmax-free) attention detector: the matched
    # kernel facts plus the config-selected KV tile / TMA ring depth.
    gated_match: GatedAttentionPlan | None = None
    # Config-independent gated match, probed once per device function
    # (``gated_probed`` records that the probe ran).
    gated_probe: GatedAttentionMatch | None = None
    gated_probed: bool = False
    # Set by the backward-attention detector (cute_flash_bwd.py): the
    # matched kernel facts and the inner Q-loop block ids.
    bwd_match: AttentionBwdMatch | FlexBwdMatch | None = None
    bwd_block_ids: list[int] | None = None
    # The FlexAttention backward envelope (``flex_bwd_match.py``): the
    # whole-graph-set match, probed once per device function.
    flex_bwd_match: FlexBwdMatch | None = None
    flex_bwd_probed: bool = False


@dataclasses.dataclass
class CuteLaneState:
    """Lane loops, thread axes and the launch shape (``cute_state.lanes``)."""

    # Physical (axis, thread extent) for grid index/base assignments.
    # The launcher consults only names that survive final AST lowering;
    # this does not depend on blocked/strided index-expression spelling.
    grid_thread_extents: dict[str, tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    # The same ``(axis, thread extent)`` from every grid index producer,
    # registered without the search-fact gates ``grid_thread_extents``
    # carries; read only by the structural ``LaunchShape`` fact
    # (``cute/launch_shape.py``, check mode).
    launch_thread_extents: dict[str, tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    # The launch-shape check's record for this render (``check_launch_shape``).
    launch_shape_check: dict[str, object] | None = None
    # The CTA count of a launch the launcher sizes from a wrapper plan
    # instead of the program-id grid (the register-MMA flash backward
    # runs one CTA per ``kv_rows`` KV rows: ``total_tiles`` of its
    # plan); None when the program-id grid is the launch grid.  The
    # dependent-launch release rule (``_implicit_pdl_release``) counts
    # the launch's CTAs from it.
    launch_ctas: int | None = None
    # Grid V-loop wrappers (by constexpr lane variable) and sinkable scalar
    # loads (by pointer expression) for ``cute/sink_vector_loops.py``.
    vloop_sink_wrappers: dict[str, CuteVloopWrapperFact] = dataclasses.field(
        default_factory=dict
    )
    vloop_sink_loads: dict[str, CuteVloopLoadFact] = dataclasses.field(
        default_factory=dict
    )
    vloop_sink_stores: dict[str, CuteVloopStoreFact] = dataclasses.field(
        default_factory=dict
    )
    # ``cute_vloop_sink`` kept the vectorized grid axis on thread x; when
    # no V-loop is sunk after all, codegen restarts with the knob off.
    vloop_sink_layout_applied: bool = False
    # The sinking pass unrolled a lane loop itself (``cute_lane_unroll``);
    # otherwise the loads-first unroll of ``cute/unroll_lane_loads.py``
    # applies to the sunk lane loops once their packets are placed.
    vloop_sink_unrolled: bool = False
    # ``(subject, per-axis thread extents)`` a cross-lane reduce (a staged
    # matmul product sum or a lane-loop reduction marker) assumed for the
    # launch of an ``hl.barrier()`` kernel.  The launcher rejects a final
    # block shape that differs from them on any axis.
    multi_phase_lane_reduce_layouts: list[tuple[str, dict[int, int]]] = (
        dataclasses.field(default_factory=list)
    )
    # SIMT reduction-kernel thread-block cluster width (from the
    # ``cute_cluster_n`` config knob, applied by
    # ``PerThreadNDTileStrategy`` when a lane-looped axis is split
    # across cluster CTAs).  1 = no cluster.
    simt_cluster_n: int = 1
    # A trailing grid dim of CTAs forming one thread-block cluster per
    # program id (the ``cute_mma_dag`` cluster split of a full-slice
    # axis); 1 for an ordinary launch.
    grid_cluster_n: int = 1
    # Number of DSM cluster-reduce call sites emitted; > 0 makes the
    # device function emit one mbarrier fence + cluster arrive/wait
    # after the preamble (covering every site's mbarrier init).
    simt_cluster_reduce_sites: int = 0
    # Lane loops claimed by a register/shuffle ``hl.associative_scan``
    # lowering, mapped to the direction (``True`` = reverse) their lanes
    # are visited in.  A second scan over the same lane loop must agree
    # or it falls back to the serial lowering (see ``cute/scan_ops.py``).
    scan_lane_directions: dict[str, bool] = dataclasses.field(default_factory=dict)
    # Constexpr V-loop variables that enclose a reduction over ANOTHER block
    # (``s[tile] = x[tile, :].sum(-1)`` with a vector width on ``tile``):
    # each V lane is a different output element, so
    # ``cute/hoist_warp_reduce.py`` must not fold the lanes into one
    # accumulator before the cross-thread reduce.
    unfoldable_vec_lanes: set[str] = dataclasses.field(default_factory=set)
    # Matmul ``dot_acc`` running sums whose K-loop carry is only exact
    # once ``hoist_lane_invariant_chunk_recurrence`` restructures it to
    # update once per tile (``cute/matmul_fallback.py``); a nest the pass
    # does not recognize is rejected rather than summed across tiles.
    chunk_recurrence_running_sums: set[str] = dataclasses.field(default_factory=set)
    # Names of matmul-fallback running-sum accumulators emitted as
    # ``acc = acc + product`` inside a constexpr V-loop.  The
    # ``hoist_warp_reduce`` pass reads this to reduce the FINAL value once
    # (instead of building a per-lane V-fold, which would double-count the
    # already-accumulated running sum).  See ``_emit_cute_matmul``.
    matmul_running_sums: set[str] = dataclasses.field(default_factory=set)
    # Rolled reductions over a symbolic extent expose their trip count as
    # a host-computed ``cutlass.Constexpr`` kernel parameter so the
    # two-pass load fuser can size a per-thread register cache that is
    # exact for every runtime extent.  Keyed by the roll's offset variable
    # (``roffset_N``), the loop target the AST passes see: (size-hint trip
    # count used for profitability decisions, constexpr parameter name).
    dynamic_reduction_trips: dict[str, tuple[int, str]] = dataclasses.field(
        default_factory=dict
    )
    # Cross-thread reductions over a block that a device loop's VECTORIZED
    # lane loop distributes (several packets per thread per trip).  The
    # reduce is emitted once per lane and so reduces one lane's packet;
    # ``cute/fold_lane_reductions.py`` folds the per-lane partials across
    # the unrolled lanes and runs the cross-thread combine once per trip.
    # The lane variable of each such loop, mapped to the keys of the
    # reduce calls emitted inside it (``fold_site_key``: the call with its
    # input replaced).  Two loops may emit identical calls (the grouped
    # warp reduce takes a raw thread-index expression), so the lane
    # variable is the primary key.
    lane_fold_reduce_sites: dict[str, set[str]] = dataclasses.field(
        default_factory=dict
    )
    resident_reduction_layouts: dict[str, ResidentReductionLayout] = dataclasses.field(
        default_factory=dict
    )
    # A reshape can reuse source lanes and leave its synthetic loop dead.
    # Resolve this recorded alternative only after actual loop pruning.
    reshape_lane_fallbacks: dict[str, tuple[str, int, int, str]] = dataclasses.field(
        default_factory=dict
    )
    reshape_physical_fallbacks: dict[str, tuple[int, int, str]] = dataclasses.field(
        default_factory=dict
    )
    resident_sequence_regions: dict[int, SequenceRegion] = dataclasses.field(
        default_factory=dict
    )
    completed_matmul_sums: dict[Node, CompletedMatmulSum] = dataclasses.field(
        default_factory=dict
    )
    # ``hl.atomic_add`` nodes that receive the per-K-lane partial sums of a
    # scalar matmul whose K axis is split across a serial lane loop
    # (``cute/matmul_fallback.py``), mapped to the lane variables of those
    # loops.  Such an atomic varies along the loop although its index does
    # not cover the loop's block, so the atomic lowering does not record
    # it as uniform along the loop (``atomic_ops._cute_uniform_lane_vars``).
    per_lane_atomic_lane_vars: dict[Node, set[str]] = dataclasses.field(
        default_factory=dict
    )
    # Exact fp32 SIMT GEMM sites awaiting ``lower_simt_gemm``.
    simt_gemm_sites: list[SimtGemmSite] = dataclasses.field(default_factory=list)
    collective_lane_loop_suppression_vetoed: bool = False
    suppress_root_lane_loops: bool = False
    collective_mma_sites: list[CollectiveMmaSite] = dataclasses.field(
        default_factory=list
    )
    collective_mma_static_layouts: bool = False
    # Late scalar-recipe staging may compose several contractions. These
    # statements access only fresh compiler-owned shared buffers, so later
    # sites can keep proving effects against the original global accesses.
    collective_mma_emitted_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    collective_mma_shared_results: set[str] = dataclasses.field(default_factory=set)
    collective_mma_pure_stmt_ids: set[int] = dataclasses.field(default_factory=set)
    # Collective operand loads registered for later codegen decisions.  FX
    # node object identities drive the regular load suppression; this is
    # scoped to one codegen pass where the FX graph and AST lists retain the
    # same objects.  Node names are only unique within one graph: the root
    # graph's first load is ``load`` like the K loop's operand load.
    collective_handled_loads: set[Node] = dataclasses.field(default_factory=set)
    collective_register_chain_lowered: bool = False
    collective_register_chain_block_dims: tuple[int, int, int] | None = None
    # Launch shape ``finalize_shared_reduce_groups`` sized the shared
    # cross-warp reductions for (None when it rewrote nothing); the
    # launcher refuses to emit a different ``block=`` for such a body.
    shared_reduce_launch_block: tuple[int, int, int] | None = None
    # The static ``block=`` the launcher call was emitted with
    # (``CuteBackend.launcher_keyword_args``, which runs ahead of the
    # kernel definition's codegen); None before that call, and for a
    # launch with an argument-sized axis.  ``_implicit_pdl_release``
    # estimates the resident CTAs per SM from it.
    launch_block: tuple[int, int, int] | None = None
    owned_root_block_dims: tuple[int, int, int] | None = None
    # The fused body may run one CTA per (grid index, lane): the launch
    # grid is the PID strategy's grid times this factor (``tcgen05_split_k``
    # sets it to the split as well).
    launch_grid_multiplier: int = 1
    # SASS-level checks the launcher runs on the compiled kernel before
    # its first launch (``helion.runtime.cute.sass_check``), declared by
    # the emitters whose instruction forms the DSL's assembler has
    # miscompiled (the mma_dag emitter's hinted ``cp.async`` packets);
    # emitted as the kernel's ``_helion_cute_sass_checks`` attribute.
    sass_checks: tuple[str, ...] = ()


@dataclasses.dataclass
class CuteRootPlans:
    """Whole-root plan slots (``cute_state.root_plans``).

    Each slot is written by the one planner of its family
    (``CuteBackend.pre_codegen`` or a detector) and read by the family's
    ``PlannedRootEmitter`` entry; every ``PlannedRootEmitter.slot`` names a
    field of this class (checked when the registry is built).
    """

    # Grouped two-phase lowering for structurally proven fixed-token,
    # split-input BF16 rank-1 recurrences.
    fixed_token_rank1_plan: CuteFixedTokenRank1Plan | None = None
    # Names-agnostic affine regions awaiting late address and ownership
    # proofs. Discovery alone never changes the ordinary lowering.
    direct_affine_candidates: tuple[DirectAffineCandidate, ...] = ()
    # Installed only after late generated-address and effect proofs succeed.
    # Its CTA shape is authoritative because the direct lowering replaces
    # the ordinary lane topology.
    direct_affine_plan: DirectAffinePlan | None = None
    topk_plan: CuteTopKPlan | None = None
    radix_selection_plan: RadixSelectionPlan | None = None
    register_region_plan: RegisterRegionPlan | None = None
    # Packed one-warp lowering for structurally proven split-input T=1
    # BF16 rank-1 recurrences.  It precedes the grouped fixed-token path.
    split_single_token_rank1_plan: CuteSplitSingleTokenRank1Plan | None = None
    # Whole-body packed lowering for a structurally proven single-token
    # BF16 rank-1 state recurrence. The plan is absent by default and is
    # additionally gated by the user-facing fast_math setting.
    single_token_rank1_plan: CuteSingleTokenRank1Plan | None = None
    # ``(needed, capacity)`` when the computed-fragment owner declined the
    # current root because its shared memory exceeds the CTA capacity; a
    # generic-lowering refusal of that root is then a config failure.
    fragment_capacity_decline: tuple[int, int] | None = None
    # Whole-root BT16 five-factor prepare schedule.  This is installed only
    # after the complete semantic graph and packed workspace ABI match.
    chunk_prepare_plan: CuteChunkPreparePlan | None = None
    block_scaled_plan: BlockScaledMmaPlan | None = None
    # Whole-root BT16 KDA recurrence/output schedule. Like the
    # prepare plan, this exists only after the complete semantic graph and
    # packed workspace ABI have matched.
    chunk_recurrence_plan: CuteChunkRecurrencePlan | None = None
    # Whole-root gated-delta-rule (gdn_fwd_h) chunk recurrence.  Installed
    # only after the semantic two-contraction graph match and the shape
    # admission limits of the SM100 tcgen05 schedule both hold.
    gdn_recurrence_plan: CuteGdnRecurrencePlan | None = None
    # Set by the register-MMA GEMM detector (``cute_matmul_family=
    # "warp_mma"``, cute_warp_mma_gemm.py): the matched plain GEMM and
    # its tile; the dedicated codegen emits the whole device body and
    # the launch runs ``32 * warps`` threads per CTA.
    warp_mma_gemm_plan: CuteWarpMmaGemmMatch | None = None
    # Set by ``plan_register_subblocks`` (``cute_register_subblocks=
    # "on"``, cute_register_subblocks.py): the flattened static
    # sub-block program of the root and its knobs; the dedicated
    # codegen renders the whole device body as one register program
    # per warp and the launch runs ``32 * warps`` threads per CTA.
    register_subblocks_plan: RegisterSubblockPlan | None = None
    # Set by ``plan_mma_dag`` (``cute_mma_dag="warp"``, cute_mma_dag.py):
    # the matched serial MMA state recurrence and its tile / warp /
    # stage selection; the dedicated codegen owns the matched root and
    # the launch runs ``32 * warps`` threads per CTA.
    mma_dag_plan: MmaDagPlan | None = None
    # The mma_dag emitters' 16-bit-typed values whose fp32 widening is
    # already at hand (name -> the exact fp32 expression: an unpacked
    # tile element, a widening side load): the pointwise lowering's
    # ``to_dtype(x, float32)`` takes it instead of a cast round trip.
    mma_dag_exact_f32: dict[str, str] = dataclasses.field(default_factory=dict)


ROOT_PLAN_SLOTS: frozenset[str] = frozenset(
    field.name for field in dataclasses.fields(CuteRootPlans)
)
