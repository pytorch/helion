"""Reusable resident TCgen05 stages with logical-coordinate source emission.

Unlike a whole-root template, a stage owns only an MMA and its communication.
The common expression interpreter retains every source cast, mask, coefficient
and accumulator. Rectangular contractions can exchange M/N physical ownership
without changing the logical result or the reduction order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import cast

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_execution import ChainedExecution
from .chained_mma_selection import warp_mma_shape
from .chained_result_transport import M64_WIDTHS
from .chained_scratch_layout import ScratchLayouts
from .chained_tcgen05 import _layout
from .chained_tcgen05 import _load_result
from .chained_vector_stage import VectorStageOperand
from .chained_vector_stage import emit_vector_stage
from .chained_vector_stage import emit_vector_stage_group
from .chained_warp_mma import emit_warp_mma as emit_warp_mma
from .thread_budget import MAX_THREADS_PER_BLOCK
from .warp_specialized_plan import VALID_TMEM_COLUMNS
from .warp_specialized_plan import WARP_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_body_program import RootActionBody
    from .chained_body_program import RootBodyStage
    from .chained_broadcast_retention import BroadcastOptions
    from .chained_broadcast_retention import BroadcastPlacement
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_completed_store import CompletedStoreAction
    from .chained_contraction_groups import ContractionGroup
    from .chained_island_publication import IslandConsumerPublication
    from .chained_loop_tmem_carry_transport import LoopTmemCarryTransport
    from .chained_loop_tmem_transport import LoopTmemTransport
    from .chained_matmul import ChainedMatmulPlan
    from .chained_output_lease import BoundOutputLease
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_fragment import PreparationFragment
    from .chained_preparation_reads import PreparationWarpCompletion
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_root_stage import RootStageAction
    from .chained_seed_tiles import SeedTiling
    from .chained_stage_operands import DirectStageOperands
    from .chained_tmem_accumulator import TmemAccumulatorResidency
    from .chained_tmem_drains import DrainTiling
    from .chained_tmem_drains import ScalarPublication
    from .chained_tmem_transport import TmemOperandBinding
    from .chained_vector_stage import VectorStaging
    from .prepared_tcgen_binding import PreparedTmemEdge
    from .warp_specialized_plan import SharedMemoryLayoutPlan


@dataclass(frozen=True)
class StageGeometry:
    logical: tuple[int, int, int]
    transpose: bool
    native_rows: int = 128

    def __post_init__(self) -> None:
        if type(self.native_rows) is not int or self.native_rows not in (64, 128):
            raise ValueError("unsupported native contraction rows")
        if self.native_rows == 64 and (
            type(self.logical) is not tuple
            or len(self.logical) != 3
            or type(self.transpose) is not bool
            or self.transpose
            or any(type(size) is not int for size in self.logical)
            or self.logical[0] != 64
            or self.logical[1] not in M64_WIDTHS
            or self.logical[2] <= 0
            or self.logical[2] % 16
        ):
            raise ValueError("unsupported native M64 geometry")

    @property
    def physical(self) -> tuple[int, int, int]:
        m, n, k = self.logical
        width = m if self.transpose else n
        # M128 uses the complete 32-datapath copy family, with a shorter
        # repetition for N16. Padded values are never exposed to the DAG.
        return self.native_rows, max(16, (width + 15) // 16 * 16), k

    def result_coordinates(self, row: str, column: str) -> tuple[str, str]:
        return (column, row) if self.transpose else (row, column)

    def operand(self, role: str, row: str, column: str) -> tuple[int, tuple[str, str]]:
        if self.transpose:
            return (1, (column, row)) if role == "a" else (0, (row, column))
        return (0, (row, column)) if role == "a" else (1, (column, row))


def stage_geometry(shape: tuple[int, int, int]) -> StageGeometry | None:
    m, n, k = shape
    if min(shape) <= 0 or m % 16 or n % 8 or k % 16:
        return None
    transpose = m < 128 and n == 128
    rows, columns = (n, m) if transpose else (m, n)
    if rows > 128 or columns > 256:
        return None
    return StageGeometry(shape, transpose)


def shared_bytes(
    geometries: tuple[StageGeometry, ...],
    groups: tuple[ContractionGroup, ...] | None = None,
    workspace: SharedMemoryLayoutPlan | None = None,
) -> int:
    physical = tuple(
        item.physical for item in (geometries if groups is None else groups)
    )
    allocations = [
        2 * max(m * k for m, _, k in physical),
        2 * max(n * k for _, n, k in physical),
        *(
            (4 * math.prod(item.logical[:2]) for item in geometries)
            if workspace is None
            else (workspace.allocated_bytes,)
        ),
        8 * len(geometries),
        4,
    ]
    return sum((size + 127) // 128 * 128 for size in allocations)


def allocate_tmem_resources(
    max_columns: int,
    barrier_count: int,
    threads: int,
    shared_allocations: Sequence[str] = (),
) -> list[str]:
    """Establish the existing CTA-wide TMEM allocator and completion protocol.

    Call before any role branches, with the full launch participant count.
    TmemAllocator retains its default allocating warp; named barrier 1 remains
    reserved for its CTA-wide retrieval protocol. ``barrier_count`` spans the
    original stage IDs, including stages that do not issue a TCgen05 operation.
    Caller-owned shared allocations are inserted before protocol allocations;
    this helper neither creates stage C views nor appends their final CTA sync.
    Release through ``free_stages`` only after all roles have joined.
    """
    return [
        *begin_tmem_resources(max_columns, barrier_count, threads, shared_allocations),
        *finish_tmem_allocation(early_release=True),
    ]


def begin_tmem_resources(
    max_columns: int,
    barrier_count: int,
    threads: int,
    shared_allocations: Sequence[str] = (),
    *,
    early_release: bool = True,
) -> list[str]:
    """Start the CTA allocation; operand transfers may precede its retrieval.

    The allocating warp must stay fully active through permit release. The
    caller must execute ``finish_tmem_allocation`` before any TMEM access.
    """
    return [
        "from cutlass.cute.nvgpu import tcgen05",
        "from cutlass.utils import blackwell_helpers as chain_sm100",
        "from cutlass.utils import TmemAllocator",
        "import cutlass.pipeline as chain_pipeline",
        "chain_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
        *shared_allocations,
        *tmem_resource_setup(
            max_columns, barrier_count, threads, early_release=early_release
        ),
    ]


def tmem_resource_setup(
    max_columns: int, barrier_count: int, threads: int, *, early_release: bool
) -> list[str]:
    """Allocate completion objects and start TMEM after caller-owned SMEM."""
    if type(early_release) is not bool:
        raise ValueError("early_release must be a boolean")
    if type(max_columns) is not int or not 0 < max_columns <= VALID_TMEM_COLUMNS[-1]:
        raise ValueError("TMEM columns must be a positive integer within capacity")
    if type(barrier_count) is not int or barrier_count <= 0:
        raise ValueError("TMEM completion barrier count must be a positive integer")
    if (
        type(threads) is not int
        or not 128 <= threads <= MAX_THREADS_PER_BLOCK
        or threads % WARP_SIZE
    ):
        raise ValueError("TMEM setup requires 128..1024 whole-warp CTA participants")
    tmem_columns = max(32, 1 << (max_columns - 1).bit_length())
    return [
        f"chain_bars = cute.arch.alloc_smem(cutlass.Int64, {barrier_count}, alignment=16)",
        "chain_holding = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)",
        "if chain_thread == 0:",
        *(
            f"    cute.arch.mbarrier_init(chain_bars + {i}, 1)"
            for i in range(barrier_count)
        ),
        "cute.arch.mbarrier_init_fence()",
        f"chain_allocation_barrier = chain_pipeline.NamedBarrier(barrier_id=1, num_threads={threads})",
        "chain_allocator = TmemAllocator(chain_holding, barrier_for_retrieve=chain_allocation_barrier)",
        f"chain_allocator.allocate({tmem_columns})",
        *(["chain_allocator.relinquish_alloc_permit()"] if early_release else []),
    ]


def finish_tmem_allocation(*, early_release: bool) -> list[str]:
    """Publish the allocated pointer at the original CTA-wide retrieve join."""
    if type(early_release) is not bool:
        raise ValueError("early_release must be a boolean")
    return [
        "chain_allocator.wait_for_alloc()",
        "chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)",
        *([] if early_release else ["chain_allocator.relinquish_alloc_permit()"]),
    ]


def allocate_stages(
    geometries: tuple[StageGeometry, ...],
    groups: tuple[ContractionGroup, ...] | None = None,
    workspace: SharedMemoryLayoutPlan | None = None,
    scratch: ScratchLayouts | None = None,
    threads: int = 128,
    *,
    workspace_allocated: bool = False,
) -> list[str]:
    """Allocate once outside a lexical loop; each stage publishes logical C."""
    physical = tuple(
        item.physical for item in (geometries if groups is None else groups)
    )
    a_size = max(m * k for m, _, k in physical)
    b_size = max(n * k for _, n, k in physical)
    columns = max(n for _, n, _ in physical)
    lines = allocate_tmem_resources(
        columns,
        len(geometries),
        threads,
        (
            f"chain_a_workspace = cute.arch.alloc_smem(cutlass.BFloat16, {a_size}, alignment=128)",
            f"chain_b_workspace = cute.arch.alloc_smem(cutlass.BFloat16, {b_size}, alignment=128)",
        ),
    )
    if workspace is not None and not workspace_allocated:
        lines.append(
            f"chain_c_workspace = cute.arch.alloc_smem(cutlass.Float32, {workspace.allocated_bytes // 4}, alignment=128)"
        )
    regions = (
        {} if workspace is None else {item.name: item for item in workspace.regions}
    )
    scratch = ScratchLayouts() if scratch is None else scratch
    for stage, geometry in enumerate(geometries):
        m, n, _ = geometry.logical
        pointer = (
            f"cute.arch.alloc_smem(cutlass.Float32, {m * n}, alignment=128)"
            if workspace is None
            else f"chain_c_workspace + {regions[f'chain_{stage}_c'].byte_offset // 4}"
        )
        lines.append(
            f"chain_{stage}_c = cute.make_tensor({pointer}, {scratch.layout(f'chain_{stage}_c', (m, n))})"
        )
    lines.append("cute.arch.sync_threads()")
    return lines


def emit_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    stage: int,
    geometry: StageGeometry,
    phase: str,
    group: ContractionGroup | None = None,
    vector_staging: VectorStaging | None = None,
    producer_unroll: BoundedProducerUnroll | None = None,
    *,
    execution: ChainedExecution | None = None,
    prepared_shape: tuple[int, int, int] | None = None,
    residency: TmemAccumulatorResidency | None = None,
    prepared_operand: PreparedOperand | None = None,
    prepared_group: PreparedGroupBinding | None = None,
    tmem_input: TmemOperandBinding | None = None,
    tmem_output: LoopTmemTransport | None = None,
    tmem_accumulator: str | None = None,
    tmem_carry: LoopTmemCarryTransport | None = None,
    seed_tiling: SeedTiling | None = None,
    operand_producer: Callable[[tuple[VectorStageOperand, ...]], list[str] | None]
    | None = None,
    direct_operands: DirectStageOperands | None = None,
    pending_allocation: bool | None = None,
    terminal_fragment: bool = False,
    root_actions: RootStageAction | None = None,
    root_body: RootBodyStage | None = None,
    root_action_body: RootActionBody | None = None,
    output_lease: BoundOutputLease | None = None,
    completed_store: CompletedStoreAction | None = None,
    drain_tiling: DrainTiling | None = None,
    preparation_completion: PreparationWarpCompletion | None = None,
    member_fragment: PreparationFragment | None = None,
    island_input: IslandConsumerPublication | None = None,
    broadcast_capture: BroadcastRetentionAttempt | None = None,
    prepared_edge: PreparedTmemEdge | None = None,
) -> list[str]:
    """Execute one contraction or common-left group; publish each logical C.

    A separately planned warp-stage frame may omit unused physical M padding.
    Its prepared shape must exactly match the selected warp computation; this
    does not change operand expressions, grouped B offsets, or logical C views.

    An optional complete operand producer receives those same original operand
    descriptors before emission. It owns only the fill, not the native views,
    async-proxy fence, role barrier, MMA or result publication. Returning None
    rejects the attempt before any result boundaries are installed; the caller
    must retain its ordinary preceding producers in that case.
    """
    members = (
        ((stage, geometry, 0),)
        if group is None
        else tuple(zip(group.stages, group.geometries, group.offsets, strict=True))
    )
    prefix = f"chain_{stage}"
    m, n, k = geometry.physical if group is None else group.physical
    use_warp = stage in plan.warp_mma_stages
    if output_lease is not None and (
        use_warp
        or root_actions is not None
        or group is None
        or phase != "chain_iteration & 1"
    ):
        raise chain._UnsupportedChain(
            "output lease requires its ordered final TCgen group"
        )
    if any(
        member_geometry.native_rows != 128 for _, member_geometry, _ in members
    ) and (
        root_actions is None
        or root_actions.sequence.independent is None
        or group is not None
        or plan.loop is not None
        or len(plan.dots) != 1
        or stage != 0
        or plan.dots[0].args[2] is not None
        or plan.initialized_accumulator is not None
    ):
        raise chain._UnsupportedChain(
            "native M64 requires an independent root input action"
        )
    resident_source = residency is not None and stage == residency.source_stage
    resident_destination = (
        residency is not None and stage == residency.destination_group
    )
    if tmem_carry is not None and (
        use_warp
        or tmem_carry.candidate.final_group.stages
        != tuple(member for member, _, _ in members)
        or tmem_carry.candidate.final_group.geometries
        != tuple(member_geometry for _, member_geometry, _ in members)
        or tmem_carry.candidate.residency != residency
        or tmem_accumulator != tmem_carry.arena
    ):
        raise chain._UnsupportedChain("resident carry does not match this final group")
    if tmem_input is not None and (
        use_warp
        or tmem_input.group.stages != tuple(member for member, _, _ in members)
        or tmem_input.group.geometries
        != tuple(member_geometry for _, member_geometry, _ in members)
        or tmem_input.physical_shape != (m, k)
        or tmem_input.dtype != plan.operand_dtype(stage)
    ):
        raise chain._UnsupportedChain("packed TMEM input does not match this group")
    if tmem_output is not None and (
        use_warp
        or resident_source
        or len(members) != 1
        or tmem_output.slot.candidate.source_stage != stage
        or tmem_output.slot.candidate.source is not plan.dots[stage]
        or tmem_output.slot.candidate.source_geometry != geometry
    ):
        raise chain._UnsupportedChain("packed TMEM source does not match this stage")
    if resident_source or resident_destination:
        # The proof module imports StageGeometry; keep this import local to
        # avoid a cycle while revalidating retained graph facts before emission.
        from .chained_tmem_accumulator import validate_tmem_accumulator_residency

        assert residency is not None
        if use_warp or not validate_tmem_accumulator_residency(plan, residency):
            raise chain._UnsupportedChain(
                "invalid explicit accumulator residency proof"
            )
        if resident_source and (
            tuple(member for member, _, _ in members) != (residency.source_stage,)
            or geometry != residency.source_geometry
        ):
            raise chain._UnsupportedChain("resident producer geometry does not match")
        if resident_destination and not any(
            member == residency.destination_member
            and member_geometry == residency.destination_geometry
            and offset == residency.offset
            for member, member_geometry, offset in members
        ):
            raise chain._UnsupportedChain("resident accumulator segment does not match")
    if prepared_shape is not None:
        if (
            not use_warp
            or type(prepared_shape) is not tuple
            or len(prepared_shape) != 3
            or any(type(size) is not int or size <= 0 for size in prepared_shape)
            or prepared_shape != warp_mma_shape(geometry, group)
        ):
            raise ValueError(
                "prepared_shape requires the exact selected warp MMA shape"
            )
        m, n, k = prepared_shape
    dtype = CompileEnvironment.current().backend.dtype_str(plan.operand_dtype(stage))
    if prepared_operand is not None and (
        use_warp
        or prepared_operand.role != "b"
        or prepared_operand.stage != stage
        or prepared_operand.geometry != geometry
        or len(members) != 1
        or prepared_operand.physical_shape != (n, k)
        or prepared_operand.buffer.dtype != plan.operand_dtype(stage)
        or prepared_operand.buffer.node is None
        or plan.dots[stage].args[prepared_operand.operand_index]
        is not prepared_operand.buffer.node
        or boundaries.get(prepared_operand.buffer.node) != prepared_operand.buffer.name
    ):
        raise chain._UnsupportedChain("prepared operand does not match the stage")
    if execution is not None and not use_warp and execution.threads < 128:
        raise ValueError("TCgen05 stages require at least 128 execution participants")
    if prepared_group is not None:
        candidate = prepared_group.candidate
        if (
            use_warp
            or prepared_operand is not None
            or candidate.group != group
            or len(members) < 2
            or candidate.physical_shape != (n, k)
            or candidate.dtype != plan.operand_dtype(stage)
            or tuple(
                (member.stage, member.geometry, member.row_offset)
                for member in candidate.members
            )
            != members
            or any(
                member.buffer.node is None
                or plan.dots[member.stage].args[member.operand_index]
                is not member.buffer.node
                or boundaries.get(member.buffer.node) != member.buffer.name
                for member in candidate.members
            )
        ):
            raise chain._UnsupportedChain("prepared group does not match the stage")
    execution = execution or ChainedExecution(plan.threads)
    if root_action_body is not None:
        if root_actions is None or root_body is not None:
            raise chain._UnsupportedChain(
                "root action body requires its original action"
            )
        root_action_body.validate_stage(
            cg, plan, boundaries, root_actions, stage, geometry, execution
        )
    if root_body is not None:
        if (
            use_warp
            or group is not None
            or root_actions is not None
            or prepared_operand is not None
            or prepared_group is not None
            or operand_producer is not None
            or residency is not None
            or tmem_input is not None
            or tmem_output is not None
            or tmem_carry is not None
            or seed_tiling is not None
            or prepared_shape is not None
            or output_lease is not None
            or completed_store is not None
            or vector_staging is not None
            or producer_unroll is not None
            or phase != "0"
            or direct_operands is not root_body.body.direct
            or terminal_fragment != (root_body.result == "terminal")
            or pending_allocation
            != (root_body.body.early_release if stage == 0 else None)
            or tmem_accumulator != "chain_tptr + 0"
        ):
            raise chain._UnsupportedChain(
                "ordered root body conflicts with stage policy"
            )
        root_body.validate(cg, plan, boundaries, stage, geometry, execution)
    if root_actions is not None:
        if (
            use_warp
            or group is not None
            or direct_operands is not None
            or prepared_operand is not None
            or prepared_group is not None
            or operand_producer is not None
            or residency is not None
            or tmem_output is not None
            or tmem_carry is not None
            or vector_staging is not None
            or producer_unroll is not None
            or seed_tiling is not None
            or phase != "0"
            or completed_store is not None
        ):
            raise chain._UnsupportedChain(
                "root input actions conflict with another stage policy"
            )
        root_actions.validate(
            cg,
            plan,
            stage,
            geometry,
            execution,
            tmem_input,
            tmem_accumulator,
            pending_allocation,
            terminal_fragment,
            boundaries,
        )
    if direct_operands is not None and (
        use_warp
        or group is not None
        or plan.threads != 128
        or geometry.transpose
        or geometry.logical != plan.shapes[stage]
        or geometry.logical != geometry.physical
        or direct_operands.stage != stage
        or direct_operands.shape != geometry.physical
        or direct_operands.nodes != tuple(plan.dots[stage].args[:2])
        or direct_operands.dtype != plan.operand_dtype(stage)
        or not direct_operands.a
        or not direct_operands.b
        or execution.threads != 128
        or execution.thread != "chain_thread"
        or tmem_input is not None
        or prepared_operand is not None
        or prepared_group is not None
        or operand_producer is not None
    ):
        raise chain._UnsupportedChain("direct operand transfers do not match the stage")
    if pending_allocation is not None and (
        type(pending_allocation) is not bool
        or (direct_operands is None and root_actions is None and root_body is None)
        or execution.tmem != "chain_tptr"
    ):
        raise chain._UnsupportedChain(
            "deferred allocation requires direct CTA operands"
        )
    if terminal_fragment and (
        use_warp
        or group is not None
        or geometry.transpose
        or stage != len(plan.dots) - 1
        or residency is not None
        or tmem_output is not None
        or tmem_carry is not None
        or execution.threads != 128
    ):
        raise chain._UnsupportedChain("terminal fragment requires one complete result")
    if operand_producer is not None and (
        not use_warp
        or prepared_shape is None
        or tmem_input is not None
        or tmem_accumulator is not None
        or prepared_operand is not None
        or prepared_group is not None
    ):
        raise chain._UnsupportedChain(
            "complete operand producers require an ordinary prepared warp stage"
        )
    if island_input is not None:
        if (
            not use_warp
            or prepared_shape is None
            or len(members) != 1
            or phase != "0"
            or plan.loop is None
            or any(
                item is not None
                for item in (
                    residency,
                    prepared_operand,
                    prepared_group,
                    tmem_input,
                    tmem_output,
                    tmem_accumulator,
                    tmem_carry,
                    seed_tiling,
                    operand_producer,
                    direct_operands,
                    pending_allocation,
                    root_actions,
                    root_body,
                    root_action_body,
                    output_lease,
                    completed_store,
                )
            )
            or terminal_fragment
        ):
            raise chain._UnsupportedChain(
                "island publication requires its singleton preparation warp stage"
            )
        island_input.validate_stage(
            cg, plan, boundaries, stage, geometry, execution, vector_staging
        )
    source = "SMEM" if tmem_input is None else "TMEM"
    if prepared_edge is not None:
        if (
            use_warp
            or len(members) != 1
            or tmem_output is None
            or tmem_input is not None
            or residency is not None
            or root_actions is not None
            or root_body is not None
            or root_action_body is not None
            or completed_store is not None
            or output_lease is not None
            or terminal_fragment
            or direct_operands is not None
            or tmem_carry is not None
            or tmem_accumulator is not None
            or pending_allocation is not None
        ):
            raise chain._UnsupportedChain("unsupported explicit prepared TCgen edge")
        prepared_edge.validate(
            cg, plan, boundaries, stage, geometry, phase, execution, tmem_output
        )
    major_a, major_b = ("K", "K") if root_actions is None else root_actions.major_modes
    if root_body is not None:
        major_a, major_b = root_body.major_modes
    if completed_store is not None:
        from .chained_pipeline_storage import StageTransports

        if (
            use_warp
            or group is None
            or root_actions is not None
            or terminal_fragment
            or resident_source
            or tmem_output is not None
            or direct_operands is not None
            or pending_allocation is not None
            or prepared_shape is not None
            or operand_producer is not None
            or output_lease is not completed_store.output_lease
            or execution.threads != 128
        ):
            raise chain._UnsupportedChain(
                "completed store requires a full common stage"
            )
        completed_store.begin_stage(
            plan,
            StageTransports(
                group,
                tmem_input,
                prepared_operand,
                prepared_group,
                tmem_output,
                residency,
                tmem_accumulator,
                tmem_carry,
            ),
            execution,
            boundaries,
            phase,
        )
    if root_action_body is not None:
        assert root_actions is not None
        root_action_body.begin_stage(root_actions)
    if broadcast_capture is None and vector_staging is not None:
        broadcast_capture = vector_staging.broadcast
    broadcast_first = 0 if broadcast_capture is None else broadcast_capture.checkpoint()
    broadcast_placements: list[BroadcastPlacement | None] = []
    lines = (
        []
        if use_warp
        else [
            f"{prefix}_mma = chain_sm100.make_trivial_tiled_mma({dtype}, {dtype}, cute.nvgpu.OperandMajorMode.{major_a}, cute.nvgpu.OperandMajorMode.{major_b}, cutlass.Float32, tcgen05.CtaGroup.ONE, ({m}, {n}), tcgen05.OperandSource.{source})"
        ]
    )
    grouped_producer = None
    grouped_roles: tuple[str, ...] = ()
    if (
        (root_actions is not None and root_actions.grouped_operands)
        or operand_producer is not None
        or (
            direct_operands is None
            and vector_staging is not None
            and vector_staging.group_enabled
            and tmem_input is None
            and prepared_operand is None
            and prepared_group is None
        )
    ):
        operands = []
        for role, shape in (("a", (m, k)), ("b", (n, k))):
            for member, member_geometry, offset in (
                members[:1] if role == "a" else members
            ):
                operand_index, _coords = member_geometry.operand(role, "row", "column")
                operands.append(
                    VectorStageOperand(
                        cast("Node", plan.dots[member].args[operand_index]),
                        member_geometry,
                        role,
                        shape if role == "a" else (member_geometry.physical[1], k),
                        f"{prefix}_{role}",
                        offset if role == "b" else 0,
                    )
                )
        selected_operands = tuple(operands)
        producer_first = (
            0 if broadcast_capture is None else broadcast_capture.checkpoint()
        )
        if root_actions is not None:
            grouped_producer = root_actions.grouped_operand_lines(
                cg, boundaries, tuple(operands)
            )
        elif operand_producer is not None:
            grouped_producer = operand_producer(tuple(operands))
            if not grouped_producer:
                raise chain._UnsupportedChain("complete operand producer rejected")
        else:
            assert vector_staging is not None
            if island_input is not None:
                selected_operands = island_input.grouped_operands(selected_operands)
            grouped_producer = emit_vector_stage_group(
                cg,
                plan,
                boundaries,
                selected_operands,
                vector_staging,
                tag=f"{prefix}_vector_group",
                producer_unroll=producer_unroll,
                execution=execution,
            )
        if grouped_producer is not None:
            grouped_roles = (
                ("a", "b")
                if island_input is None
                else tuple(operand.role for operand in selected_operands)
            )
            if root_actions is not None:
                root_actions.check_grouped_body(cg, boundaries, grouped_producer)
            for role, shape in (("a", (m, k)), ("b", (n, k))):
                if role not in grouped_roles:
                    continue
                lines.extend(
                    [
                        f"{prefix}_{role}_ptr = {execution.a_workspace if role == 'a' else execution.b_workspace}",
                        *_layout(f"{prefix}_{role}", shape, 1, dtype),
                    ]
                )
            if broadcast_capture is not None:
                broadcast_placements.append(
                    broadcast_capture.place(
                        producer_first, grouped_producer, len(lines)
                    )
                )
            lines.extend(grouped_producer)
    for role, shape in (("a", (m, k)), ("b", (n, k))):
        if role in grouped_roles:
            continue
        if root_actions is not None:
            lines.extend(root_actions.operand_lines(cg, boundaries, role))
            continue
        if root_body is not None and direct_operands is None:
            lines.extend(root_body.operand_lines(cg, boundaries, role, dtype))
            continue
        if direct_operands is not None:
            lines.extend(
                [
                    f"{prefix}_{role}_ptr = {execution.a_workspace if role == 'a' else execution.b_workspace}",
                    *_layout(f"{prefix}_{role}", shape, 1, dtype),
                    *(direct_operands.a if role == "a" else direct_operands.b),
                ]
            )
            continue
        if role == "a" and tmem_input is not None:
            continue
        if role == "b" and prepared_operand is not None:
            lines.append(f"{prefix}_b = {prepared_operand.buffer.name}")
            continue
        if role == "b" and prepared_group is not None:
            if producer_unroll is not None:
                # These exact boundary-only operands would use scalar fills:
                # vector admission requires a host leaf. Account for removing
                # those loops without changing any surviving loop's factor.
                for member in prepared_group.candidate.members:
                    size = math.prod(member.physical_shape)
                    producer_unroll.eliminate_loop(
                        (size + execution.threads - 1) // execution.threads
                    )
            lines.append(f"{prefix}_b = {prepared_group.candidate.name}")
            continue
        lines.extend(
            [
                f"{prefix}_{role}_ptr = {execution.a_workspace if role == 'a' else execution.b_workspace}",
                *_layout(f"{prefix}_{role}", shape, 1, dtype),
            ]
        )
        if island_input is not None and role == island_input.candidate.role:
            island_input.consume(
                cg,
                plan,
                boundaries,
                stage,
                geometry,
                execution,
                vector_staging,
                role,
                shape,
                dtype,
            )
            if producer_unroll is not None:
                producer_unroll.eliminate_loop(
                    (math.prod(shape) + execution.threads - 1) // execution.threads
                )
            continue
        for member, member_geometry, offset in members[:1] if role == "a" else members:
            member_shape = shape if role == "a" else (member_geometry.physical[1], k)
            index = f"{prefix}_{role}_{member}_index"
            row, column = f"{index} // {k}", f"{index} % {k}"
            operand_index, coords = member_geometry.operand(role, row, column)
            operand = cast("Node", plan.dots[member].args[operand_index])
            if vector_staging is not None and vector_staging.enabled:
                options: BroadcastOptions = {}
                if vector_staging.broadcast is not None:
                    options["broadcast"] = vector_staging.broadcast
                producer_first = (
                    0 if broadcast_capture is None else broadcast_capture.checkpoint()
                )
                vector_lines = emit_vector_stage(
                    cg,
                    plan,
                    boundaries,
                    operand,
                    member_geometry,
                    role=role,
                    shape=member_shape,
                    offset=offset if role == "b" else 0,
                    tag=f"{prefix}_{role}_{member}_vector",
                    target=f"{prefix}_{role}",
                    producer_unroll=producer_unroll,
                    execution=execution,
                    **options,
                )
                if vector_lines is not None:
                    vector_staging.activated = True
                    if broadcast_capture is not None:
                        broadcast_placements.append(
                            broadcast_capture.place(
                                producer_first, vector_lines, len(lines)
                            )
                        )
                    lines.extend(vector_lines)
                    continue
            logical_shape = chain._shape(operand)
            predicate = " & ".join(
                f"(({coord}) < {extent})"
                for coord, extent in zip(coords, logical_shape, strict=True)
            )
            expression = chain._Expression(cg, plan, boundaries)
            expression.coordinate_names.add(index)
            value = expression.value(operand, coords)
            domain = chain._operand_domain(cg, operand, coords, plan)
            destination_row = (
                row if role == "a" or offset == 0 else f"({row}) + {offset}"
            )
            size = math.prod(member_shape)
            trips = (size + execution.threads - 1) // execution.threads
            unroll = (
                1 if producer_unroll is None else producer_unroll.loop_factor(trips)
            )
            producer = [
                f"if {predicate}:",
                chain._indent(expression.lines),
                f"    {prefix}_{role}[{destination_row}, {column}] = {chain._masked_operand(value, dtype, domain)}",
                "else:",
                f"    {prefix}_{role}[{destination_row}, {column}] = {dtype}(0)",
            ]
            if size % execution.threads:
                # The logical padding branch may only write inside this
                # member's physical allocation, never past a short B tile.
                producer = [f"if {index} < {size}:", chain._indent(producer)]
            lines.extend(
                [
                    f"for {prefix}_{role}_{member}_step in cutlass.range({trips}, unroll={unroll}):",
                    f"    {index} = {execution.thread} + {prefix}_{role}_{member}_step * {execution.threads}",
                    chain._indent(producer),
                ]
            )
    split_k = root_actions is not None and root_actions.k_schedule is not None
    if root_actions is not None and not split_k:
        lines.extend(
            [
                "cute.arch.cp_async_commit_group()",
                *root_actions.after_commit(),
                "cute.arch.cp_async_wait_group(0)",
            ]
        )
    elif direct_operands is not None or root_body is not None:
        lines.extend(
            ["cute.arch.cp_async_commit_group()", "cute.arch.cp_async_wait_group(0)"]
        )
    warp_completion = None
    if use_warp:
        from .chained_warp_stage import join_warp_inputs

        joined, warp_completion = join_warp_inputs(
            cg,
            plan,
            boundaries,
            execution,
            lines,
            proxy_fence=True,
        )
        lines.extend(joined)
    elif not split_k:
        lines.extend(["cute.arch.fence_view_async_shared()", execution.sync])
    if pending_allocation is not None:
        lines.extend(finish_tmem_allocation(early_release=pending_allocation))
    if root_actions is not None:
        lines.extend(root_actions.after_wait(cg, boundaries))
    if use_warp:
        from .chained_warp_stage import WarpMemberResult
        from .chained_warp_stage import emit_prepared_warp_stage
        from .chained_warp_stage import prepare_warp_stage

        assert warp_completion is not None
        warp_shape = warp_mma_shape(geometry, group)
        rows, columns, reduction = warp_shape
        # The warp tile must divide the prepared N extent. Using only its
        # magnitude would over-tile widths such as 48 or grouped width 96.
        column_atoms = columns // 8
        warps = min(execution.threads // 32, column_atoms & -column_atoms)
        # A role can contain a non-power-of-two number of complete warps.
        # The active MMA team still has to divide the prepared column extent.
        threads = 32 * (1 << (warps.bit_length() - 1))
        warp = f"{prefix}_warp"
        prepared_warp = prepare_warp_stage(
            warp_completion,
            warp,
            dtype,
            warp_shape,
            threads,
            (1, 1),
            WarpMemberResult(prefix, tuple(members), member_fragment),
        )
        lines.extend(
            emit_prepared_warp_stage(cg, plan, boundaries, prepared_warp, lines)
        )
        if preparation_completion is not None:
            publication = warp_completion._state.publication
            if publication is None:
                raise chain._UnsupportedChain("missing original warp publication")
            preparation_completion.record(publication, cg, plan, boundaries, lines)
        if broadcast_capture is not None:
            broadcast_capture.enclose(broadcast_first, lines, broadcast_placements)
        return lines
    if member_fragment is not None:
        raise chain._UnsupportedChain(
            "member fragment requires original warp completion"
        )
    lines.extend(
        [
            f"{prefix}_layout = {prefix}_mma.make_fragment_C({prefix}_mma.partition_shape_C(({m}, {n}))).layout",
            f"{prefix}_acc = cute.make_tensor({execution.tmem if tmem_accumulator is None else tmem_accumulator}, {prefix}_layout)",
            f"{prefix}_slice = {prefix}_mma.get_slice(0)",
        ]
    )
    initialized = any(plan.dots[member].args[2] is not None for member, _, _ in members)
    tiled_seed = seed_tiling is not None and seed_tiling.max_columns != 0
    if resident_destination or tmem_carry is not None or initialized and tiled_seed:
        lines.extend(
            _seed_nonresident_accumulators(
                cg,
                plan,
                boundaries,
                prefix,
                (m, n),
                members,
                residency if resident_destination else None,
                execution,
                tmem_carry=tmem_carry,
                seed_tiling=seed_tiling,
            )
        )
    if (
        initialized
        and not resident_destination
        and tmem_carry is None
        and not tiled_seed
    ):
        seed_begin = len(lines)
        seed = f"{prefix}_seed"
        row, column = f"{seed}_row", f"{seed}_column"
        lines.extend(
            [
                f"{seed}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition({min(32, n & -n)})), cutlass.Float32), {prefix}_acc)",
                f"{seed}_thread = {seed}_copy.get_slice({execution.thread})",
                f"{seed}_target = {seed}_thread.partition_D({prefix}_acc)",
                f"{seed}_coords = {seed}_thread.partition_S({prefix}_slice.partition_C(cute.make_identity_tensor(({m}, {n}))))",
                f"{seed}_values = cute.make_rmem_tensor({seed}_coords.shape, cutlass.Float32)",
                f"{seed}_values.fill(0.0)",
                f"for {seed}_index in cutlass.range_constexpr(cute.size({seed}_values)):",
                f"    {row}, {column} = {seed}_coords[{seed}_index]",
            ]
        )
        for member, member_geometry, offset in members:
            accumulator = plan.dots[member].args[2]
            if accumulator is None:
                continue
            logical_m, logical_n, _ = member_geometry.logical
            coords = member_geometry.result_coordinates(row, f"({column} - {offset})")
            expression = chain._Expression(cg, plan, boundaries)
            expression.coordinate_names.update((row, column))
            value = expression.value(cast("Node", accumulator), coords)
            lines.extend(
                [
                    f"    if ({column} >= {offset}) & ({column} < {offset + member_geometry.physical[1]}) & ({coords[0]} < {logical_m}) & ({coords[1]} < {logical_n}):",
                    chain._indent(expression.lines, 8),
                    f"        {seed}_values[{seed}_index] = cutlass.Float32({value})",
                ]
            )
        lines.extend(
            [
                f"cute.copy({seed}_copy, {seed}_values, {seed}_target)",
                "cute.arch.fence_view_async_tmem_store()",
            ]
        )
        if execution.threads > 128:
            seed_lines = lines[seed_begin:]
            lines[seed_begin:] = [
                f"if {execution.thread} < 128:",
                chain._indent(seed_lines),
            ]
        lines.append(execution.sync)
    if tmem_input is None:
        lines.append(
            f"{prefix}_ra = {prefix}_mma.make_fragment_A({prefix}_slice.partition_A({prefix}_a))"
        )
        a_slice = f"{prefix}_ra[None, None, {prefix}_kk]"
    else:
        from .chained_tmem_transport import emit_tmem_operand_view

        lines.extend(
            emit_tmem_operand_view(
                prefix,
                (m, n, k),
                dtype,
                base=tmem_input.base
                if root_actions is not None and tmem_input.column_offset == 0
                else f"({tmem_input.base} + {tmem_input.column_offset})",
            )
        )
        a_slice = f"{prefix}_ra[None, None, {prefix}_kk, 0]"
    if split_k:
        from .chained_k_issue import emit_k_half_issues

        assert root_actions is not None and root_actions.k_schedule is not None
        lines.extend(
            emit_k_half_issues(
                prefix,
                stage,
                root_actions.k_schedule.mode,
                root_actions.half_lines(),
                retire_each_half=root_actions.retire_each_half,
                **(
                    {"continuation": root_action_body}
                    if root_action_body is not None
                    and root_action_body.continuation is not None
                    else {}
                ),
            )
        )
    elif prepared_edge is not None:
        prepared_edge.capture_stage_setup(lines)
        lines.extend(prepared_edge.issue(prefix, initialized, False))
    elif root_action_body is not None and root_action_body.continuation is not None:
        from .prepared_continuation import emit_root_continuation

        assert root_actions is not None
        lines.extend(emit_root_continuation(root_action_body, root_actions, prefix))
    else:
        lines.extend(
            emit_full_k_issue(
                prefix,
                stage,
                phase,
                a_slice,
                initialized
                or root_actions is not None
                and root_actions.seeded_accumulator,
                execution,
            )
        )
    if root_actions is not None:
        lines.extend(root_actions.post_issue_lines())
    if output_lease is not None:
        from .chained_pipeline_storage import StageTransports

        assert group is not None
        lines.extend(
            output_lease.post_issue_lines(
                plan,
                StageTransports(
                    group,
                    tmem_input,
                    prepared_operand,
                    prepared_group,
                    tmem_output,
                    residency,
                    tmem_accumulator,
                    tmem_carry,
                ),
                execution,
                boundaries,
            )
        )
    if resident_source:
        # The next issue consumes the completed FP32 cells directly. No SMEM
        # boundary is published and this role must retain the same TMEM arena.
        lines.append(execution.sync)
        from .native_matmul_metadata import record_native_stage

        record_native_stage(
            cg, plan, tuple(plan.dots[i] for i, _, _ in members), "tcgen05", lines
        )
        return lines
    result_begin = len(lines)
    initialized_result = (
        root_actions.sequence.initialized
        if root_actions is not None and stage == 0
        else None
    )
    shared_state = (
        initialized_result is not None
        and root_action_body is not None
        and root_action_body.continuation is not None
    )
    retained_result = (
        root_actions is not None
        and root_actions.retains_completed_result(cg, boundaries)
    )
    from .chained_tmem_drains import plan_fp32_drain

    drain_panels = None
    if drain_tiling is not None:
        drain_tiling.check(
            cg.device_function.config.config.get("cute_chained_drain_tile_columns", 0)
        )
        if (
            root_actions is None
            and not terminal_fragment
            and completed_store is None
            and tmem_output is None
            and tmem_carry is None
            and (root_body is None or root_body.result == "allocated")
        ):
            drain_panels = plan_fp32_drain((m, n), drain_tiling.columns)
    if (
        not retained_result
        and not shared_state
        and (initialized_result is None or not initialized_result.max_columns)
        and drain_panels is None
    ):
        if prepared_edge is None:
            lines.extend(_load_result(prefix, (m, n), execution=execution))
        else:
            lines.extend(
                _load_result(
                    prefix, (m, n), execution=execution, prepared_edge=prepared_edge
                )
            )
    if root_actions is not None:
        # Either a terminal epilogue or the next original, exclusive bridge
        # consumes these FP32 registers. No SMEM C image is invented.
        if shared_state:
            from .prepared_state_body import emit_initialized_state

            assert root_action_body is not None and initialized_result is not None
            lines.extend(
                emit_initialized_state(
                    root_action_body,
                    root_actions,
                    initialized_result,
                    geometry,
                    execution,
                )
            )
        elif initialized_result is not None:
            lines.extend(initialized_result.lines)
        lines.append(execution.sync)
        root_actions.complete()
        from .native_matmul_metadata import record_native_stage

        record_native_stage(
            cg, plan, tuple(plan.dots[i] for i, _, _ in members), "tcgen05", lines
        )
        if root_action_body is not None:
            root_action_body.record_completion(
                cg, plan, boundaries, root_actions, stage, geometry, execution, lines
            )
        return lines
    if root_body is not None:
        from .chained_tcgen05 import _materialize_result
        from .native_matmul_metadata import record_native_stage

        root_body.validate(cg, plan, boundaries, stage, geometry, execution)
        if root_body.result == "allocated":
            lines.extend(
                _materialize_result(
                    prefix,
                    (m, n),
                    root_body.body.scratch,
                    drain_tiling=drain_tiling if drain_panels is not None else None,
                )
            )
        lines.append(execution.sync)
        record_native_stage(
            cg, plan, tuple(plan.dots[i] for i, _, _ in members), "tcgen05", lines
        )
        root_body.record_completion(lines)
        return lines
    if terminal_fragment:
        # The caller consumes the completed original FP32 fragment through its
        # existing epilogue. Do not invent a logical SMEM result boundary.
        lines.append(execution.sync)
        from .native_matmul_metadata import record_native_stage

        record_native_stage(
            cg, plan, tuple(plan.dots[i] for i, _, _ in members), "tcgen05", lines
        )
        return lines
    if completed_store is not None:
        # Compile-time lowering only: its scalar statements still execute at
        # the original store dispatch, after this stage's publication join.
        completed_store.prepare_point(cg, plan, boundaries)
    if tmem_output is None:
        published_members = (
            members
            if tmem_carry is None
            else tuple(
                item for item in members if item[0] != tmem_carry.candidate.member
            )
        )
        if completed_store is not None:
            published_members = tuple(
                item
                for item in published_members
                if not completed_store.suppresses(item[0])
            )
        if published_members:
            if drain_panels is not None:
                assert drain_tiling is not None
                lines.extend(
                    drain_tiling.emit(
                        drain_panels,
                        f"{prefix}_drain",
                        prefix,
                        _scalar_publication(prefix, published_members),
                        execution=ChainedExecution(
                            128, thread=execution.thread, warp=execution.warp
                        ),
                    )
                )
                for member, _, _ in published_members:
                    boundaries[plan.dots[member]] = f"chain_{member}_c"
            else:
                lines.extend(
                    _publish_result(plan, boundaries, prefix, published_members)
                )
    else:
        # This is a separate 128-thread TMEM participant barrier, never the
        # wider recurrence-role barrier hidden under the predicate below.
        transport_execution = ChainedExecution(
            128,
            thread=execution.thread,
            warp=execution.warp,
            sync="chain_tmem_barrier.arrive_and_wait()",
            tmem=execution.tmem,
        )
        lines.extend(
            tmem_output.emit(transport_execution)
            if prepared_edge is None
            else prepared_edge.publish(
                tmem_output.prefix, tmem_output, transport_execution
            )
        )
    if execution.threads > 128:
        result_lines = lines[result_begin:]
        lines[result_begin:] = [
            f"if {execution.thread} < 128:",
            chain._indent(result_lines),
        ]
    lines.append(execution.sync)
    if prepared_edge is not None:
        prepared_edge.finish(lines)
    if completed_store is not None:
        completed_store.complete_stage(plan, boundaries, lines)
    from .native_matmul_metadata import record_native_stage

    record_native_stage(
        cg, plan, tuple(plan.dots[i] for i, _, _ in members), "tcgen05", lines
    )
    if broadcast_capture is not None:
        broadcast_capture.enclose(broadcast_first, lines, broadcast_placements)
    return lines


def emit_full_k_issue(
    prefix: str,
    stage: int,
    phase: str,
    a_slice: str,
    initialized: bool,
    execution: ChainedExecution,
) -> list[str]:
    """Issue every original K atom in order, then complete the same stage barrier.

    Operand readiness, accumulator initialization, and resource lifetimes are
    caller obligations. Both straight-line and lexical-loop stages use this
    implementation; K-split schedules keep their separately proved protocol.
    """
    return [
        f"{prefix}_rb = {prefix}_mma.make_fragment_B({prefix}_slice.partition_B({prefix}_b))",
        f"if {execution.warp} == 0:",
        f"    {prefix}_mma.set(tcgen05.Field.ACCUMULATE, {initialized})",
        f"    for {prefix}_kk in cutlass.range_constexpr(cute.size({prefix}_ra, mode=[2])):",
        f"        cute.gemm({prefix}_mma, {prefix}_acc, {a_slice}, {prefix}_rb[None, None, {prefix}_kk], {prefix}_acc)",
        f"        {prefix}_mma.set(tcgen05.Field.ACCUMULATE, True)",
        "    with cute.arch.elect_one():",
        f"        tcgen05.commit({execution.barriers} + {stage})",
        f"cute.arch.mbarrier_wait({execution.barriers} + {stage}, {phase})",
    ]


def _seed_nonresident_accumulators(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    prefix: str,
    shape: tuple[int, int],
    members: tuple[tuple[int, StageGeometry, int], ...],
    residency: TmemAccumulatorResidency | None,
    execution: ChainedExecution,
    *,
    tmem_carry: LoopTmemCarryTransport | None = None,
    seed_tiling: SeedTiling | None = None,
) -> list[str]:
    """Seed disjoint member subviews without reading or overwriting residency.

    Full-M128 TCgen fragments have nested shape ((128,N),1,1). Restrict both
    the TMEM target and its coordinate tensor to the exact nonresident member;
    a whole-group zero fill/copy would destroy the preserved accumulator.
    """
    from .chained_seed_tiles import plan_seed_panels

    lines: list[str] = []
    for member, geometry, offset in members:
        if residency is not None and member == residency.destination_member:
            continue
        width = geometry.physical[1]
        seed = f"{prefix}_seed_{member}"
        row, column = f"{seed}_row", f"{seed}_column"
        panels = (
            plan_seed_panels(shape, offset, width)
            if seed_tiling is None
            else seed_tiling.panels(shape, offset, width)
        )
        if (
            tmem_carry is not None
            and member == tmem_carry.candidate.member
            and tmem_carry.snapshot_seed is not None
        ):
            # The same disjoint N32 seed panels were published by the snapshot.
            # Keep the original final fence/join and every other member seed.
            continue
        accumulator = plan.dots[member].args[2]
        for panel in panels:
            lines.extend(
                [
                    *panel.store_views(seed, prefix, execution.thread),
                    f"{seed}_values = cute.make_rmem_tensor({seed}_coords.shape, cutlass.Float32)",
                    f"{seed}_values.fill(0.0)",
                ]
            )
            if tmem_carry is not None and member == tmem_carry.candidate.member:
                lines.extend(tmem_carry.seed_values(execution, width=panel.width))
            elif accumulator is not None:
                logical_m, logical_n, _ = geometry.logical
                coords = geometry.result_coordinates(row, f"({column} - {offset})")
                expression = chain._Expression(cg, plan, boundaries)
                expression.coordinate_names.update((row, column))
                value = expression.value(cast("Node", accumulator), coords)
                lines.extend(
                    [
                        f"for {seed}_index in cutlass.range_constexpr(cute.size({seed}_values)):",
                        f"    {row}, {column} = {seed}_coords[{seed}_index]",
                        f"    if ({column} >= {offset}) & ({column} < {offset + width}) & ({coords[0]} < {logical_m}) & ({coords[1]} < {logical_n}):",
                        chain._indent(expression.lines, 8),
                        f"        {seed}_values[{seed}_index] = cutlass.Float32({value})",
                    ]
                )
            lines.append(f"cute.copy({seed}_copy, {seed}_values, {seed}_target)")
    lines.append("cute.arch.fence_view_async_tmem_store()")
    if execution.threads > 128:
        lines = [f"if {execution.thread} < 128:", chain._indent(lines)]
    return [*lines, execution.sync]


def _publish_result(
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    prefix: str,
    members: tuple[tuple[int, StageGeometry, int], ...],
) -> list[str]:
    """Both instruction families publish the same logical member boundaries."""
    lines = _scalar_publication(prefix, members).lines(
        f"{prefix}_values", f"{prefix}_coords"
    )
    for member, _, _ in members:
        boundaries[plan.dots[member]] = f"chain_{member}_c"
    return lines


def _scalar_publication(
    prefix: str, members: tuple[tuple[int, StageGeometry, int], ...]
) -> ScalarPublication:
    from .chained_tmem_drains import FP32SharedTarget
    from .chained_tmem_drains import ScalarPublication

    return ScalarPublication(
        f"{prefix}_result_index",
        f"{prefix}_result_row",
        f"{prefix}_result_column",
        tuple(
            FP32SharedTarget(
                f"chain_{member}_c",
                geometry.logical[:2],
                geometry.transpose,
                offset,
                geometry.physical[1],
            )
            for member, geometry, offset in members
        ),
        guarded=True,
    )


def free_stages() -> list[str]:
    """Release the CTA-owned allocator after all stage/role paths have joined."""
    return ["cute.arch.sync_threads()", "chain_allocator.free(chain_tptr)"]
