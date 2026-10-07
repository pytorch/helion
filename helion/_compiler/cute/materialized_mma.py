"""Native MMA search for independently launched materialized regions."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from .materialized_fission_codegen import _region_graph_ids
from .pipeline_smem import Tcgen05PipelineSmemFacts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...language.matmul_ops import CuteTcgen05SearchPlan
    from ...language.matmul_ops import _CuteTcgen05SearchPlanningResult
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .cute_mma import _CuteMmaNode


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
    from ...language import _tracing_ops
    from ...language import creation_ops
    from ...language import memory_ops
    from ...language import tile_ops
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
            elif not (
                isinstance(other.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in other.target.tags
                and not other.target._schema.is_mutable
                and torch.Tag.nondeterministic_seeded not in other.target.tags
                and not other.is_impure()
            ):
                return None
    if loops != 1:
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
    graph_ids = {id(info.graph): info.graph_id for info in device_ir.graphs}
    regions = [
        _region_graph_ids(device_ir.graphs, root_id) for root_id in device_ir.root_ids
    ]
    placed: dict[int, tuple[_CuteMmaNode, CuteTcgen05SearchPlan]] = {}
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
            return False
        placed[owners[0]] = (candidate, plan)
        occupied.update(axes)
    if len(placed) != len(regions):
        return False
    spec = env.config_spec
    if set(spec.block_sizes.valid_block_ids()) != occupied:
        return False

    from .pipeline_smem import analyze_pipeline_smem_facts
    from .tcgen05_config import CuteTcgen05Config

    state = spec._cute_tcgen05_config
    state.materialized_matmul_shapes = tuple(
        (
            cast("int", plan.static_m),
            cast("int", plan.static_n),
            cast("int", plan.static_k),
        )
        for _candidate, plan in (placed[index] for index in range(len(regions)))
    )
    # Preserve ordinary single-output facts; complete shared-fanout validation
    # later accounts for its second ring from the typed paired-store proof.
    facts = []
    pair_budget_facts = []
    for index in range(len(regions)):
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
    spec._cute_tcgen05_config.materialized_matmul_block_ids = tuple(
        (
            candidate.operands.m_block_id,
            candidate.operands.n_block_id,
            candidate.operands.k_block_id,
        )
        for candidate, _plan in (placed[index] for index in range(len(regions)))
    )
    # Existing one-CTA seeds remain unchanged. Optional paired fanout schedules
    # use the complete independent region budgets and strict full-tile guards.
    allow_persistent = all(
        plan.static_m is not None
        and plan.static_n is not None
        and plan.static_k is not None
        and plan.static_m % min(plan.max_search_m, 128) == 0
        and plan.static_n % plan.max_search_n == 0
        and plan.static_k % plan.max_search_k == 0
        for _candidate, plan in placed.values()
    )
    spec.narrow_tcgen05_autotune_to_validated_configs(
        allow_persistent_pid_types=allow_persistent,
        reason="independent materialized CuTe MMA regions",
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
    return True
