"""Native MMA search for independently launched materialized regions."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from .active_blocks import active_block_ids
from .materialized_fission_codegen import _region_graph_ids
from .pipeline_smem import Tcgen05PipelineSmemFacts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from ...language.matmul_ops import CuteTcgen05SearchPlan
    from ...language.matmul_ops import _CuteTcgen05SearchPlanningResult
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .cute_mma import _CuteMmaNode

    _Placed: TypeAlias = dict[int, tuple[_CuteMmaNode, CuteTcgen05SearchPlan]]


def _mma_operand_cone(node: torch.fx.Node) -> frozenset[torch.fx.Node]:
    """The nodes ``node`` consumes, transitively, within its graph."""
    cone: set[torch.fx.Node] = set()
    stack = [node]
    while stack:
        for operand in stack.pop().all_input_nodes:
            if operand not in cone:
                cone.add(operand)
                stack.append(operand)
    return frozenset(cone)


def _plain_gemm_region_loops(
    graphs: Sequence[GraphInfo],
    node: torch.fx.Node,
    stores: set[torch.fx.Node],
    *,
    admitted: frozenset[torch.fx.Node] = frozenset(),
) -> int | None:
    """Count the loops of a region holding nothing beside one GEMM's work.

    Admits the matmul ``node``, the stores in ``stores`` (its proved epilogue
    chains), the ``admitted`` nodes (the matmul's operand cone, whose layout
    ops the candidate analysis proved), tile and loop plumbing, loads and
    pure pointwise math; any other op -- a store the matmul does not reach,
    an atomic, a reduction, a collective, an unknown lowering -- returns
    ``None``.  The tcgen05 region codegen runs only the matmul's fused
    epilogue per element, so anything else in the region would fail there
    (or borrow the pipeline's budget).
    """
    from ...language import _tracing_ops
    from ...language import creation_ops
    from ...language import memory_ops
    from ...language import tile_ops

    scalar_ops = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._for_loop,
        creation_ops.full,
        tile_ops.tile_index,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_id,
        operator.getitem,
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        torch.ops.aten.sym_size.int,
        memory_ops.load,
    }
    loops = 0
    for info in graphs:
        for other in info.graph.nodes:
            if other.op in ("placeholder", "output") or other is node:
                continue
            if other.op != "call_function":
                return None
            if other.target is memory_ops.store:
                if other not in stores:
                    return None
            elif other.target in scalar_ops:
                loops += other.target is _tracing_ops._for_loop
            elif other in admitted:
                continue
            elif not (
                isinstance(other.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in other.target.tags
                and not other.target._schema.is_mutable
                and torch.Tag.nondeterministic_seeded not in other.target.tags
                and not other.is_impure()
            ):
                return None
    return loops


def _paired_region_smem_facts(
    candidate: _CuteMmaNode,
    node: torch.fx.Node,
    graphs: Sequence[GraphInfo],
    *,
    capacity_bytes: int,
) -> Tcgen05PipelineSmemFacts | None:
    """Prove the complete region has only AB, ACC and two output rings.

    The later typed fanout proof establishes shared arithmetic and ownership.
    This separate whole-region check excludes allocations unrelated to that
    pair: an extra store, reduction, collective or unknown lowering cannot
    borrow the pair's shared-memory budget.
    """
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .cute_mma import _tcgen05_tma_matrix_major

    operands = candidate.operands
    if (
        capacity_bytes <= 0
        or operands.has_leading_passthrough
        or operands.lhs.source_fake.ndim != 2
        or operands.rhs.source_fake.ndim != 2
        or operands.lhs.source_fake.dtype not in (torch.float16, torch.bfloat16)
        or operands.rhs.source_fake.dtype != operands.lhs.source_fake.dtype
        or operands.lhs.matrix_major != "row"
        or operands.rhs.matrix_major not in ("row", "col")
        or candidate.requires_accumulator_seed
    ):
        return None
    chains = analyze_tcgen05_matmul_store_chains(graphs, node)
    if chains is None or len(chains) != 2:
        return None
    stores = {store for store, _chain in chains}
    if _plain_gemm_region_loops(graphs, node, stores) != 1:
        return None
    for store in stores:
        output_node = store.args[0]
        if not isinstance(output_node, torch.fx.Node):
            return None
        output = output_node.meta["val"]
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.dtype not in (torch.float16, torch.bfloat16)
            or _tcgen05_tma_matrix_major(output) != "row"
        ):
            return None
    return Tcgen05PipelineSmemFacts(
        operands.lhs.source_fake.dtype.itemsize, 2, capacity_bytes
    )


def _place_region_candidates(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    regions: Sequence[frozenset[int]],
    candidates: Sequence[tuple[_CuteMmaNode, torch.Tensor, CuteTcgen05SearchPlan]],
) -> tuple[_Placed, set[int]] | None:
    """Assign each proved candidate to the one separately launched root it lives in.

    Every admitted contraction is a plain, static, 2-D fp16/bf16 GEMM whose
    output tile is exactly its root's grid and whose three axes are its own;
    two candidates in one root, a shared axis, or a small-N / batched operand
    declines the whole surface.
    """
    graph_ids = {id(info.graph): info.graph_id for info in device_ir.graphs}
    placed: _Placed = {}
    occupied: set[int] = set()
    for candidate, lhs, plan in candidates:
        operands = candidate.operands
        axes = (operands.m_block_id, operands.n_block_id, operands.k_block_id)
        graph_id = graph_ids.get(id(candidate.lhs.graph))
        owners = [i for i, region in enumerate(regions) if graph_id in region]
        if (
            len(owners) != 1
            or owners[0] in placed
            or operands.has_leading_passthrough
            or lhs.ndim != 2
            or operands.lhs.source_fake.dtype not in (torch.float16, torch.bfloat16)
            or operands.rhs.source_fake.dtype != operands.lhs.source_fake.dtype
            or tuple(device_ir.grid_block_ids[owners[0]]) != operands.output_block_ids
            or len(set(axes)) != 3
            or occupied.intersection(axes)
            or plan.is_small_n
            or any(
                value is None for value in (plan.static_m, plan.static_n, plan.static_k)
            )
        ):
            return None
        placed[owners[0]] = (candidate, plan)
        occupied.update(axes)
    return placed, occupied


def _register_region_mma_search(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    regions: Sequence[frozenset[int]],
    placed: _Placed,
    *,
    mma_nodes: dict[int, torch.fx.Node],
    allow_persistent: bool,
    reason: str,
    region_pid_types: bool = False,
) -> None:
    """Publish one tile triple, shape and pipeline budget per GEMM launch.

    ``region_pid_types`` marks an ordered barrier chain: its launches are
    flat, and ``tcgen05_region_pid_types`` may run the GEMM launches as the
    static persistent two-CTA pipeline (``CuteTcgen05Config``).
    """
    from ...language.matmul_ops import _cuda_num_sms_or_zero
    from .pipeline_smem import analyze_pipeline_smem_facts
    from .tcgen05_config import CuteTcgen05Config

    spec = env.config_spec
    state = spec._cute_tcgen05_config
    state.region_pid_type_search_enabled = region_pid_types
    state.device_sm_count = _cuda_num_sms_or_zero(
        placed[min(placed)][0].operands.lhs.source_fake.device
    )
    indices = sorted(placed)
    state.materialized_matmul_shapes = tuple(
        (
            cast("int", plan.static_m),
            cast("int", plan.static_n),
            cast("int", plan.static_k),
        )
        for _candidate, plan in (placed[index] for index in indices)
    )
    # Preserve ordinary single-output facts; complete shared-fanout validation
    # later accounts for its second ring from the typed paired-store proof.
    facts = []
    pair_budget_facts = []
    for index in indices:
        candidate, _plan = placed[index]
        capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(
            candidate.operands.lhs.source_fake.device
        )
        region_graphs = [
            info for info in device_ir.graphs if info.graph_id in regions[index]
        ]
        fact = analyze_pipeline_smem_facts(
            candidate, mma_nodes[id(candidate)], region_graphs, capacity_bytes=capacity
        )
        facts.append(fact)
        pair_budget_facts.append(
            _paired_region_smem_facts(
                candidate,
                mma_nodes[id(candidate)],
                region_graphs,
                capacity_bytes=capacity,
            )
        )
    state.materialized_pipeline_facts = tuple(facts)
    state.materialized_pair_budget_facts = tuple(pair_budget_facts)
    spec.cute_tcgen05_search_enabled = True
    state.materialized_matmul_block_ids = tuple(
        (
            candidate.operands.m_block_id,
            candidate.operands.n_block_id,
            candidate.operands.k_block_id,
        )
        for candidate, _plan in (placed[index] for index in indices)
    )
    state.materialized_matmul_region_indices = tuple(indices)
    # Existing one-CTA seeds remain unchanged. Optional paired fanout schedules
    # use the complete independent region budgets and strict full-tile guards.
    spec.narrow_tcgen05_autotune_to_validated_configs(
        allow_persistent_pid_types=allow_persistent,
        reason=reason,
    )
    for candidate, plan in placed.values():
        operands = candidate.operands
        for block_id, minimum, maximum in (
            (operands.m_block_id, plan.min_search_m, min(plan.max_search_m, 256)),
            (operands.n_block_id, plan.min_search_n, plan.max_search_n),
            (operands.k_block_id, plan.mma_k, plan.max_search_k),
        ):
            env.block_sizes[block_id].update_min_block(minimum, allow_flattened=True)
            env.block_sizes[block_id].update_max_block(maximum)


def enable_materialized_mma_search(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    planning_results: Sequence[_CuteTcgen05SearchPlanningResult],
    candidates: Sequence[tuple[_CuteMmaNode, torch.Tensor, CuteTcgen05SearchPlan]],
    *,
    mma_nodes: dict[int, torch.fx.Node],
) -> bool:
    """Admit independent tile axes and proven pipeline budgets per launch.

    The ordinary native-MMA planner has already proved each candidate. A
    fission plan additionally proves that these roots execute as separate
    device functions, so their distinct matrix axes do not need to agree.
    Single-matrix projection rules stay disabled: no one region's shape may
    silently determine another region's cluster or shared-memory layout.
    """
    fission = env.cute_fission_plan
    if (
        fission is None
        or len(candidates) != fission.region_count
        or fission.region_count < 2
        or not all(result.guards_complete for result in planning_results)
    ):
        return False
    regions = [
        _region_graph_ids(device_ir.graphs, root_id) for root_id in device_ir.root_ids
    ]
    placement = _place_region_candidates(env, device_ir, regions, candidates)
    if placement is None:
        return False
    placed, occupied = placement
    if len(placed) != len(regions):
        return False
    spec = env.config_spec
    if set(spec.block_sizes.valid_block_ids()) != occupied:
        return False
    allow_persistent = all(
        plan.static_m is not None
        and plan.static_n is not None
        and plan.static_k is not None
        and plan.static_m % min(plan.max_search_m, 128) == 0
        and plan.static_n % plan.max_search_n == 0
        and plan.static_k % plan.max_search_k == 0
        for _candidate, plan in placed.values()
    )
    _register_region_mma_search(
        env,
        device_ir,
        regions,
        placed,
        mma_nodes=mma_nodes,
        allow_persistent=allow_persistent,
        reason="independent materialized CuTe MMA regions",
    )
    return True


def enable_phase_mma_search(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    planning_results: Sequence[_CuteTcgen05SearchPlanningResult],
    candidates: Sequence[tuple[_CuteMmaNode, torch.Tensor, CuteTcgen05SearchPlan]],
    *,
    mma_nodes: dict[int, torch.fx.Node],
) -> bool:
    """Admit one native contraction per ``hl.barrier()`` phase launched in order.

    Explicit barrier phases lower to ordered per-phase launches under
    ``pid_type="flat"`` (``generate_ordered_phases``): the same separate
    device functions a materialized fission plan produces, so every GEMM phase
    may own its own tile triple, pipeline budget and seeds.  Phases without a
    matmul keep the ordinary SIMT lowering and its layout knobs
    (``cute_pointwise_region_block_ids``).  The single persistent launch
    cannot host the tcgen05 pipeline; enabling this surface makes the
    multi-root planning veto the persistent pid types, leaving the ordered
    launches.  Within the flat chain, ``tcgen05_region_pid_types`` may run
    individual GEMM launches as the static persistent two-CTA pipeline of
    the standalone single-GEMM surface (``CuteTcgen05Config``), each with
    its own complete allocation proof and ring depth.  Anything unproven
    declines and the SIMT lowering stays.
    """
    from .aux_tensor import _TCGEN05_AUX_DETECTOR_MMA_TARGETS
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .ordered_phases import ordered_phases_supported

    host = device_ir.host_function
    spec = env.config_spec
    if (
        env.cute_fission_plan is not None
        or host is None
        or not env.has_barrier
        or len(device_ir.root_ids) < 2
        or len(device_ir.phases) != len(device_ir.root_ids)
        or "flat" not in spec.allowed_pid_types
        or len(candidates) < 2
        or not all(result.guards_complete for result in planning_results)
        or not ordered_phases_supported(host, device_ir)
    ):
        return False
    regions = [
        _region_graph_ids(device_ir.graphs, root_id) for root_id in device_ir.root_ids
    ]
    placement = _place_region_candidates(env, device_ir, regions, candidates)
    if placement is None:
        return False
    placed, occupied = placement
    if len(placed) < 2:
        return False
    # A GEMM phase holds exactly its admitted matmul and its three axes, every
    # store its result reaches is a chain the tcgen05 epilogue splices, and
    # nothing else runs beside the matmul (the single-GEMM surface fails
    # loudly at codegen instead; here the SIMT lowering is the live
    # alternative, so an unsupported epilogue or side store declines).
    # A plain phase holds no matmul at all (an unadmitted one would take the
    # SIMT fallback whose lane-loop suppression the pointwise layout knobs
    # cannot see) and none of the GEMM axes.
    matmul_targets = (
        *_TCGEN05_AUX_DETECTOR_MMA_TARGETS,
        torch.ops.aten.mm.dtype,
        torch.ops.aten.addmm.dtype,
        torch.ops.aten.bmm.dtype,
        torch.ops.aten.baddbmm.dtype,
    )
    plain_block_ids: set[int] = set()
    plain_grids: list[tuple[int, ...]] = []
    for index, region in enumerate(regions):
        region_graphs = [info for info in device_ir.graphs if info.graph_id in region]
        matmuls = [
            node
            for info in region_graphs
            for node in info.graph.nodes
            if node.op == "call_function" and node.target in matmul_targets
        ]
        active = active_block_ids(region_graphs, device_ir.grid_block_ids[index], env)
        if index in placed:
            candidate, _plan = placed[index]
            operands = candidate.operands
            mma_node = mma_nodes[id(candidate)]
            if matmuls != [mma_node] or active != {
                operands.m_block_id,
                operands.n_block_id,
                operands.k_block_id,
            }:
                return False
            chains = analyze_tcgen05_matmul_store_chains(region_graphs, mma_node)
            # Every store the matmul reaches is a spliceable chain, and the
            # phase holds nothing else: an independent store (or any other
            # op) beside a collective matmul has no tcgen05 lowering.
            if chains is None or (
                _plain_gemm_region_loops(
                    region_graphs,
                    mma_node,
                    {store for store, _chain in chains},
                    admitted=_mma_operand_cone(mma_node),
                )
                is None
            ):
                return False
        else:
            if matmuls or active & occupied:
                return False
            plain_block_ids.update(active)
            plain_grids.append(tuple(device_ir.grid_block_ids[index]))
    if set(spec.block_sizes.valid_block_ids()) != occupied | plain_block_ids:
        return False
    _register_region_mma_search(
        env,
        device_ir,
        regions,
        placed,
        mma_nodes=mma_nodes,
        allow_persistent=False,
        reason="independent CuTe MMA barrier phases",
        region_pid_types=True,
    )
    spec.cute_pointwise_region_block_ids = frozenset(plain_block_ids)
    # An axis reused by another phase keeps the shared search floor.
    spec.cute_pointwise_region_grid_groups = tuple(
        grid
        for grid in plain_grids
        if all(
            sum(block_id in other for other in device_ir.grid_block_ids) == 1
            for block_id in grid
        )
    )
    return True
