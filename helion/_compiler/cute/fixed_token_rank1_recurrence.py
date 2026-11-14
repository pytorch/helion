"""Structural lowering for fixed-token split-input rank-1 recurrence.

This lowering recognizes the complete unrolled recurrence used by speculative
decode and replaces it with a grouped two-phase CuTe schedule.  The matcher is
deliberately fail closed: it proves every load, reduction, state transition,
output store, and checkpoint store before installing the plan.  Tensor and
function names are suggestions for generated code only; they are never
matching keys.

The initial admitted family has two through six tokens, BF16 Q/K/V/gate/state,
FP32 decay parameters, equal query/value head counts, and ``D == V == 128``.
Each CTA handles 32 value rows and uses the existing ``num_warps`` config to
select two or four key partitions (64 or 128 threads).  It stages Q/K/decay for
all tokens, then keeps its state fragment in FP32 registers across the
recurrence.
"""

from __future__ import annotations

import dataclasses
import operator
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from .analysis.rank1_recurrence import _MUL_TARGETS
from .analysis.rank1_recurrence import Rank1DecayMatch
from .analysis.rank1_recurrence import Rank1StepMatch
from .analysis.rank1_recurrence import Rank1UpdateMatch
from .analysis.rank1_recurrence import _affine
from .analysis.rank1_recurrence import _binary_pair
from .analysis.rank1_recurrence import _broadcast_source
from .analysis.rank1_recurrence import _cast_from
from .analysis.rank1_recurrence import _has_unmatched_side_effect
from .analysis.rank1_recurrence import _indices
from .analysis.rank1_recurrence import _is_direct_float_load
from .analysis.rank1_recurrence import _match_decay
from .analysis.rank1_recurrence import _match_factored_output
from .analysis.rank1_recurrence import _match_normalized_vector
from .analysis.rank1_recurrence import _same_scalar
from .analysis.rank1_recurrence import _strip_casts
from .analysis.rank1_recurrence import _strip_state_index_casts
from .analysis.rank1_recurrence import _sum_input
from .analysis.rank1_recurrence import match_rank1_step
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _is_call
from .fx_matcher import _linear_offsets_fit_i32
from .fx_matcher import _load_ref
from .fx_matcher import _protected_arguments
from .fx_matcher import _same_ref
from .fx_matcher import _store_ref
from .fx_matcher import _TensorRef
from .fx_matcher import _xyz_grid_fits
from .single_token_rank1_recurrence import _RANK1_HELPER_ABI_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


_KEY_SIZE = 128
_VALUE_SIZE = 128
_MIN_TOKENS = 2
_MAX_TOKENS = 6
_VALUE_COLUMNS_PER_CTA = 32
_SUPPORTED_KEY_SPLITS = (2, 4)
_FIXED_RANK1_CODEGEN_ABI_VERSION = 3


@dataclasses.dataclass(frozen=True)
class CuteFixedTokenRank1Plan:
    """Values consumed by the grouped fixed-token code generator."""

    root_graph_id: int
    q: _TensorRef
    k: _TensorRef
    v: _TensorRef
    gate: _TensorRef
    beta: _TensorRef
    a_log: _TensorRef
    bias: _TensorRef
    state_indices: _TensorRef
    num_accepted_tokens: _TensorRef
    state: _TensorRef
    output: _TensorRef
    scale: object
    lower_bound: object
    log2_e: object
    epsilon: object
    num_tokens: int
    num_sequences: int
    num_heads: int
    state_size: int
    sequence_block_id: int
    head_block_id: int
    value_block_id: int
    launch_block_ids: tuple[int, int, int]
    key_split: int
    value_split: int


def _bounded_decay(node: torch.fx.Node) -> Rank1DecayMatch | None:
    """The bounded sigmoid decay ``exp2(sigmoid(a * (gate + bias)) * lb * log2_e)``."""

    matched = _match_decay(node)
    if matched is None:
        return None
    gate, bias, a_log, lower_bound, log2_e = matched
    return Rank1DecayMatch(gate, bias, a_log, log2_e, lower_bound=lower_bound)


def _match_projection(
    output_store: torch.fx.Node,
    update: Rank1UpdateMatch,
) -> tuple[torch.fx.Node, object, object] | None:
    """Match ``sum(state * normalized_q) * scale`` and return Q load/scale."""

    stored = output_store.args[2] if len(output_store.args) > 2 else None
    output_value = _strip_casts(stored)
    if output_value is None or not _cast_from(stored, output_value, torch.bfloat16):
        return None
    scaled = _binary_pair(output_value, _MUL_TARGETS)
    if scaled is None:
        return None
    for projection, scale_node in (scaled, reversed(scaled)):
        normalized_query: torch.fx.Node | None = None
        product = _sum_input(projection, (-1,))
        pair = _binary_pair(product, _MUL_TARGETS)
        if pair is not None:
            query_broadcast: torch.fx.Node | None = None
            for maybe_state, maybe_query in (pair, reversed(pair)):
                if maybe_state is update.state_update:
                    query_broadcast = maybe_query
                    break
            if query_broadcast is not None:
                normalized_query = _broadcast_source(
                    query_broadcast, (None, slice(None))
                )
        if normalized_query is None:
            normalized_query = _match_factored_output(
                projection, update.decayed, update.residual, update.key_broadcast
            )
        if normalized_query is None:
            continue
        query_match = _match_normalized_vector(normalized_query)
        if query_match is None:
            continue
        _, query_fp32, epsilon = query_match
        query_load = _is_direct_float_load(query_fp32)
        scale = scale_node.meta.get("val")
        if query_load is not None and isinstance(scale, (int, float, torch.SymFloat)):
            return query_load, scale, epsilon
    return None


def _match_accepted_index(
    node: object,
    num_tokens: int,
) -> torch.fx.Node | None:
    """Match clamp(accepted_count - 1, 0, T - 1), returning its load."""

    if (
        not _is_call(node, torch.ops.aten.clamp_max.default)
        or not isinstance(node, torch.fx.Node)
        or len(node.args) != 2
        or node.args[1] != num_tokens - 1
        or node.kwargs
    ):
        return None
    clamp_min = node.args[0]
    if (
        not _is_call(clamp_min, torch.ops.aten.clamp_min.default)
        or not isinstance(clamp_min, torch.fx.Node)
        or len(clamp_min.args) != 2
        or clamp_min.args[1] != 0
        or clamp_min.kwargs
    ):
        return None
    subtraction = clamp_min.args[0]
    if (
        not isinstance(subtraction, torch.fx.Node)
        or subtraction.op != "call_function"
        or subtraction.target not in (operator.sub, torch.ops.aten.sub.Tensor)
        or len(subtraction.args) != 2
        or subtraction.args[1] != 1
        or subtraction.kwargs
    ):
        return None
    load = _strip_state_index_casts(subtraction.args[0])
    if not isinstance(load, torch.fx.Node) or _load_ref(load) is None:
        return None
    return load


def _tensor_shapes_match(
    refs: dict[str, _TensorRef],
    num_tokens: int,
) -> tuple[int, int] | None:
    """Validate fake/runtime tensor metadata and return ``(N, H)``."""

    env = CompileEnvironment.current()
    q = refs["q"].fake
    k = refs["k"].fake
    v = refs["v"].fake
    gate = refs["gate"].fake
    beta = refs["beta"].fake
    a_log = refs["a_log"].fake
    bias = refs["bias"].fake
    state_indices = refs["state_indices"].fake
    accepted = refs["num_accepted_tokens"].fake
    state = refs["state"].fake
    output = refs["output"].fake
    tensors = (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        bias,
        state_indices,
        accepted,
        state,
        output,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        return None
    if (
        any(
            tensor.dtype is not torch.bfloat16
            for tensor in (q, k, v, gate, beta, state, output)
        )
        or a_log.dtype is not torch.float32
        or bias.dtype is not torch.float32
        or state_indices.dtype is not torch.int32
        or accepted.dtype is not torch.int32
        or any(tensor.ndim != 3 for tensor in (q, k, v, gate, output))
        or beta.ndim != 2
        or a_log.ndim != 1
        or bias.ndim != 1
        or state_indices.ndim != 2
        or accepted.ndim != 1
        or state.ndim != 4
    ):
        return None

    if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
        return None
    runtime_values = tuple(env.runtime_value_for_tensor(tensor) for tensor in tensors)
    if not all(isinstance(tensor, torch.Tensor) for tensor in runtime_values):
        return None
    runtime_tensors = cast("tuple[torch.Tensor, ...]", runtime_values)
    (
        runtime_q,
        runtime_k,
        runtime_v,
        runtime_gate,
        runtime_beta,
        runtime_a_log,
        runtime_bias,
        runtime_indices,
        runtime_accepted,
        runtime_state,
        runtime_output,
    ) = runtime_tensors
    runtime_n = runtime_indices.shape[0]
    runtime_h = runtime_q.shape[1]
    expected_vectors = (runtime_n * num_tokens, runtime_h, _KEY_SIZE)
    if (
        runtime_n < 1
        or runtime_h < 1
        or tuple(runtime_q.shape) != expected_vectors
        or tuple(runtime_k.shape) != expected_vectors
        or tuple(runtime_v.shape) != expected_vectors
        or tuple(runtime_gate.shape) != expected_vectors
        or tuple(runtime_output.shape) != expected_vectors
        or tuple(runtime_beta.shape) != expected_vectors[:2]
        or tuple(runtime_a_log.shape) != (runtime_h,)
        or tuple(runtime_bias.shape) != (runtime_h * _KEY_SIZE,)
        or tuple(runtime_indices.shape) != (runtime_n, num_tokens)
        or tuple(runtime_accepted.shape) != (runtime_n,)
        or runtime_state.shape[0] < 1
        or tuple(runtime_state.shape[1:]) != (runtime_h, _VALUE_SIZE, _KEY_SIZE)
        or any(not tensor.is_contiguous() for tensor in runtime_tensors)
    ):
        return None
    work_items = runtime_n * runtime_h * (_VALUE_SIZE // _VALUE_COLUMNS_PER_CTA)
    if not _linear_offsets_fit_i32(
        work_items,
        tuple(tensor.numel() for tensor in runtime_tensors),
    ):
        return None

    from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
    from .memory_ops import runtime_tensor_has_specialized_alignment

    if not all(
        runtime_tensor_has_specialized_alignment(env, tensor, alignment)
        for tensor, alignment in (
            (q, 16),
            (k, 16),
            (gate, 16),
            (bias, 16),
            (state, 16),
        )
    ):
        return None
    read_tensors = (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        bias,
        state_indices,
        accepted,
    )
    if not all(
        runtime_tensors_are_proven_disjoint(env, written, read)
        for written in (state, output)
        for read in read_tensors
    ) or not runtime_tensors_are_proven_disjoint(env, state, output):
        return None
    return runtime_n, runtime_h


def _plan_fixed_token_rank1_recurrence(
    graphs: Sequence[GraphInfo],
    _tile_strategy: TileStrategyDispatch,
) -> CuteFixedTokenRank1Plan | None:
    from ...language import memory_ops
    from ..device_function import DeviceFunction
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    capability = env.config_spec.target_device_capability
    key_split = df.config.num_warps
    if (
        not env.settings.fast_math
        or df.config.pid_type != "flat"
        or key_split not in _SUPPORTED_KEY_SPLITS
        or capability is None
        or capability < (10, 0)
    ):
        return None
    roots = [info for info in graphs if isinstance(info, RootGraphInfo)]
    if len(graphs) != 1 or len(roots) != 1:
        return None
    root = roots[0]
    nodes = list(root.graph.nodes)
    loads = [node for node in nodes if _is_call(node, memory_ops.load)]
    stores = [node for node in nodes if _is_call(node, memory_ops.store)]
    if (
        _has_unmatched_side_effect(nodes, set(stores))
        or any(
            len(node.args) != 4
            or node.args[2] is not None
            or node.args[3] is not None
            or node.kwargs
            for node in loads
        )
        or any(
            len(node.args) != 4 or node.args[3] is not None or node.kwargs
            for node in stores
        )
    ):
        return None

    state_loads = [
        load
        for load in loads
        if any(_same_ref(_load_ref(load), _store_ref(store)) for store in stores)
    ]
    if len(state_loads) != 1:
        return None
    state_load = state_loads[0]
    state_ref = _load_ref(state_load)
    if state_ref is None:
        return None
    state_stores = [
        store for store in stores if _same_ref(_store_ref(store), state_ref)
    ]
    num_tokens = len(state_stores)
    if not _MIN_TOKENS <= num_tokens <= _MAX_TOKENS:
        return None
    output_stores = [store for store in stores if store not in state_stores]
    if (
        len(output_stores) != num_tokens
        or not output_stores
        or any(
            not _same_ref(_store_ref(store), _store_ref(output_stores[0]))
            for store in output_stores
        )
        or _same_ref(_store_ref(output_stores[0]), state_ref)
    ):
        return None

    initial_indices = _indices(state_load)
    if initial_indices is None or len(initial_indices) != 4:
        return None
    initial_slot = _strip_state_index_casts(initial_indices[0])
    if not isinstance(initial_slot, torch.fx.Node) or _load_ref(initial_slot) is None:
        return None
    initial_slot_indices = _indices(initial_slot)
    if initial_slot_indices is None or len(initial_slot_indices) != 2:
        return None
    sequence_coord, accepted_index = initial_slot_indices
    if not isinstance(sequence_coord, torch.fx.Node):
        return None
    accepted_load = _match_accepted_index(accepted_index, num_tokens)
    if accepted_load is None or _indices(accepted_load) != (sequence_coord,):
        return None
    state_fp32 = next(
        (
            user
            for user in state_load.users
            if _cast_from(user, state_load, torch.float32)
        ),
        None,
    )
    if state_fp32 is None:
        return None

    matched_steps: list[Rank1StepMatch] = []
    incoming_state = state_fp32
    for state_store, output_store in zip(state_stores, output_stores, strict=True):
        matched = match_rank1_step(
            state_store,
            output_store,
            incoming_state,
            beta=_is_direct_float_load,
            decay=_bounded_decay,
            projection=_match_projection,
        )
        if matched is None:
            return None
        matched_steps.append(matched)
        incoming_state = matched.state_update
    first = matched_steps[0]
    if not all(
        isinstance(value, (int, float, torch.SymFloat))
        for value in (
            first.scale,
            first.lower_bound,
            first.log2_e,
            first.epsilon,
        )
    ):
        return None
    if initial_indices[1:] != (
        first.head_coord,
        first.value_coord,
        first.key_coord,
    ):
        return None

    for token_index, step in enumerate(matched_steps):
        token_affine = _affine(step.token_coord)
        checkpoint_indices = _indices(step.checkpoint_load)
        if (
            token_affine != (token_index, {sequence_coord: num_tokens})
            or checkpoint_indices != (sequence_coord, token_index)
            or not _same_ref(_load_ref(step.checkpoint_load), _load_ref(initial_slot))
            or step.head_coord is not first.head_coord
            or step.value_coord is not first.value_coord
            or step.key_coord is not first.key_coord
            or not _same_scalar(step.scale, first.scale)
            or not _same_scalar(step.lower_bound, first.lower_bound)
            or not _same_scalar(step.log2_e, first.log2_e)
            or not _same_scalar(step.epsilon, first.epsilon)
        ):
            return None

    role_loads: dict[str, list[torch.fx.Node]] = {
        "q": [step.q_load for step in matched_steps],
        "k": [step.k_load for step in matched_steps],
        "v": [step.v_load for step in matched_steps],
        "gate": [step.gate_load for step in matched_steps],
        "beta": [step.beta_load for step in matched_steps],
        "a_log": [step.a_log_load for step in matched_steps],
        "bias": [step.bias_load for step in matched_steps],
    }
    refs: dict[str, _TensorRef] = {}
    for role, role_nodes in role_loads.items():
        ref = _load_ref(role_nodes[0])
        if ref is None or any(
            not _same_ref(_load_ref(node), ref) for node in role_nodes
        ):
            return None
        refs[role] = ref
    output_ref = _store_ref(output_stores[0])
    indices_ref = _load_ref(initial_slot)
    accepted_ref = _load_ref(accepted_load)
    if output_ref is None or indices_ref is None or accepted_ref is None:
        return None
    refs.update(
        {
            "state_indices": indices_ref,
            "num_accepted_tokens": accepted_ref,
            "state": state_ref,
            "output": output_ref,
        }
    )
    expected_loads = {state_load, initial_slot, accepted_load}
    for step in matched_steps:
        expected_loads.update(step.loads)
    if set(loads) != expected_loads or set(stores) != set(state_stores + output_stores):
        return None

    shape_match = _tensor_shapes_match(refs, num_tokens)
    if shape_match is None:
        return None
    num_sequences, num_heads = shape_match

    device_ir = HostFunction.current().device_ir
    grid_coords = (sequence_coord, first.head_coord, first.value_coord)
    grid_coord_ids = _canonical_root_axis_ids(
        device_ir,
        root_phase_index=root.phase_index,
        coordinates=grid_coords,
        expected_extents=(num_sequences, num_heads, _VALUE_SIZE),
    )
    if grid_coord_ids is None:
        return None
    resolved_grid_block_sizes = [
        df.resolved_block_size(block_id) for block_id in grid_coord_ids
    ]
    value_columns_per_cta = resolved_grid_block_sizes[2]
    if (
        resolved_grid_block_sizes[:2] != [1, 1]
        or value_columns_per_cta != _VALUE_COLUMNS_PER_CTA
    ):
        return None
    value_split = _VALUE_SIZE // value_columns_per_cta
    if df.config.loop_orders:
        loop_order = env.config_spec.loop_orders.config_get(
            df.config.loop_orders, cast("int", grid_coord_ids[0])
        )
        if loop_order is None:
            return None
        launch_block_ids = tuple(grid_coord_ids[index] for index in loop_order)
    else:
        launch_block_ids = (
            grid_coord_ids[1],
            grid_coord_ids[0],
            grid_coord_ids[2],
        )
    extent_by_block_id = dict(
        zip(
            grid_coord_ids,
            (num_sequences, num_heads, value_split),
            strict=True,
        )
    )
    if not _xyz_grid_fits(
        tuple(extent_by_block_id[block_id] for block_id in launch_block_ids)
    ):
        return None
    return CuteFixedTokenRank1Plan(
        root_graph_id=root.graph_id,
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        gate=refs["gate"],
        beta=refs["beta"],
        a_log=refs["a_log"],
        bias=refs["bias"],
        state_indices=refs["state_indices"],
        num_accepted_tokens=refs["num_accepted_tokens"],
        state=refs["state"],
        output=refs["output"],
        scale=first.scale,
        lower_bound=first.lower_bound,
        log2_e=first.log2_e,
        epsilon=first.epsilon,
        num_tokens=num_tokens,
        num_sequences=num_sequences,
        num_heads=num_heads,
        state_size=cast(
            "torch.Tensor", env.runtime_value_for_tensor(state_ref.fake)
        ).shape[0],
        sequence_block_id=cast("int", grid_coord_ids[0]),
        head_block_id=cast("int", grid_coord_ids[1]),
        value_block_id=cast("int", grid_coord_ids[2]),
        launch_block_ids=cast("tuple[int, int, int]", launch_block_ids),
        key_split=key_split,
        value_split=value_split,
    )


def plan_fixed_token_rank1_recurrence(
    graphs: Sequence[GraphInfo],
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Install the grouped plan, or leave ordinary CuTe lowering untouched."""

    from ..device_function import DeviceFunction

    device_function = DeviceFunction.current()
    plan = _plan_fixed_token_rank1_recurrence(graphs, tile_strategy)
    device_function.cute_state.root_plans.fixed_token_rank1_plan = plan
    if plan is not None:
        device_function.promote_expr_arg_to_constexpr(plan.scale)
        device_function.promote_expr_arg_to_constexpr(plan.lower_bound)
        device_function.promote_expr_arg_to_constexpr(plan.log2_e)
        device_function.promote_expr_arg_to_constexpr(plan.epsilon)


def codegen_fixed_token_rank1_recurrence(cg: GenerateAST) -> bool:
    """Replace the matched root with the grouped two-phase schedule."""

    df = cg.device_function
    plan = df.cute_state.root_plans.fixed_token_rank1_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.root_graph_id:
        return False

    arguments = _protected_arguments(
        cg,
        "fixed_rank1",
        (
            plan.q,
            plan.k,
            plan.v,
            plan.gate,
            plan.beta,
            plan.a_log,
            plan.bias,
            plan.state_indices,
            plan.num_accepted_tokens,
            plan.state,
            plan.output,
        ),
        (plan.scale, plan.lower_bound, plan.log2_e, plan.epsilon),
    )
    q, k, v, gate, beta, a_log, bias, state_indices, accepted, state, output = (
        arguments.tensors
    )
    scale, lower_bound, log2_e, epsilon = arguments.scalars
    num_tokens = plan.num_tokens
    value_columns_per_cta = _VALUE_SIZE // plan.value_split
    threads = _KEY_SIZE * plan.key_split // plan.value_split
    elements_per_thread = _KEY_SIZE // threads
    state_granules_per_thread = (_KEY_SIZE // plan.key_split) // 8
    norm_warps = min(_KEY_SIZE, threads) // 32

    from ..program_id import XYZProgramIDs

    if df.pid is None:
        return False
    logical_block_ids = (
        plan.sequence_block_id,
        plan.head_block_id,
        plan.value_block_id,
    )
    configured_pid_info = list(df.pid.pid_info)
    if len(configured_pid_info) != 3 or {
        info.block_id for info in configured_pid_info
    } != set(logical_block_ids):
        return False
    # Planning records and validates the exact physical order, including the
    # tighter CUDA Y/Z limits, before this whole-body lowering is selected.
    pid_info_by_block = {info.block_id: info for info in configured_pid_info}
    launch_pid_info = [
        pid_info_by_block[block_id] for block_id in plan.launch_block_ids
    ]
    axis_by_block = {info.block_id: axis for axis, info in enumerate(launch_pid_info)}
    sequence_axis = axis_by_block[plan.sequence_block_id]
    head_axis = axis_by_block[plan.head_block_id]
    value_axis = axis_by_block[plan.value_block_id]
    df.pid = XYZProgramIDs(pid_info=launch_pid_info)

    shuffle_offsets = tuple(1 << bit for bit in range(plan.key_split.bit_length() - 1))

    def shuffle_reduction_source(accumulator: str) -> str:
        return "\n".join(
            textwrap.indent(
                f"""{accumulator} += cute.arch.shuffle_sync_bfly(
    {accumulator}, {offset}
)""",
                "    ",
            )
            for offset in shuffle_offsets
        )

    prediction_shuffle = shuffle_reduction_source("fixed_rank1_prediction")
    projection_shuffle = shuffle_reduction_source("fixed_rank1_projected")

    decay_elements = num_tokens * _KEY_SIZE
    norm_elements = num_tokens * 16
    preamble = f"""
fixed_rank1_codegen_abi_version = {_FIXED_RANK1_CODEGEN_ABI_VERSION}
fixed_rank1_rank1_helper_abi_version = {_RANK1_HELPER_ABI_VERSION}
fixed_rank1_key_split = {plan.key_split}
fixed_rank1_value_split = {plan.value_split}
fixed_rank1_decay_smem_ptr = cute.arch.alloc_smem(
    cutlass.Float32, {decay_elements}, alignment=16
)
fixed_rank1_decay_smem = cute.make_tensor(
    fixed_rank1_decay_smem_ptr, cute.make_layout({decay_elements})
)
fixed_rank1_key_smem_ptr = cute.arch.alloc_smem(
    cutlass.Float32, {decay_elements}, alignment=16
)
fixed_rank1_key_smem = cute.make_tensor(
    fixed_rank1_key_smem_ptr, cute.make_layout({decay_elements})
)
fixed_rank1_query_smem_ptr = cute.arch.alloc_smem(
    cutlass.Float32, {decay_elements}, alignment=16
)
fixed_rank1_query_smem = cute.make_tensor(
    fixed_rank1_query_smem_ptr, cute.make_layout({decay_elements})
)
fixed_rank1_norm_smem_ptr = cute.arch.alloc_smem(
    cutlass.Float32, {norm_elements}, alignment=16
)
fixed_rank1_norm_smem = cute.make_tensor(
    fixed_rank1_norm_smem_ptr, cute.make_layout({norm_elements})
)
"""
    body = f"""
fixed_rank1_value_head = cutlass.Int32(cute.arch.block_idx()[{head_axis}])
fixed_rank1_sequence = cutlass.Int32(cute.arch.block_idx()[{sequence_axis}])
fixed_rank1_value_split_id = cutlass.Int32(cute.arch.block_idx()[{value_axis}])
fixed_rank1_tid = cutlass.Int32(cute.arch.thread_idx()[0])
fixed_rank1_value_index = (
    fixed_rank1_value_split_id * cutlass.Int32({value_columns_per_cta})
    + fixed_rank1_tid // cutlass.Int32({plan.key_split})
)
fixed_rank1_key_part = fixed_rank1_tid % cutlass.Int32({plan.key_split})

fixed_rank1_decay_fragment = cute.make_tensor(
    fixed_rank1_decay_smem.iterator,
    cute.make_layout(
        (8, {state_granules_per_thread}, {plan.key_split}, {num_tokens}),
        stride=(1, {plan.key_split * 8}, 8, {_KEY_SIZE}),
    ),
)[None, None, fixed_rank1_key_part, None]
fixed_rank1_key_fragment = cute.make_tensor(
    fixed_rank1_key_smem.iterator,
    cute.make_layout(
        (8, {state_granules_per_thread}, {plan.key_split}, {num_tokens}),
        stride=(1, {plan.key_split * 8}, 8, {_KEY_SIZE}),
    ),
)[None, None, fixed_rank1_key_part, None]
fixed_rank1_query_fragment = cute.make_tensor(
    fixed_rank1_query_smem.iterator,
    cute.make_layout(
        (8, {state_granules_per_thread}, {plan.key_split}, {num_tokens}),
        stride=(1, {plan.key_split * 8}, 8, {_KEY_SIZE}),
    ),
)[None, None, fixed_rank1_key_part, None]

fixed_rank1_accepted_index = cutlass.Int64(
    {accepted}[fixed_rank1_sequence]
) - cutlass.Int64(1)
if fixed_rank1_accepted_index < 0:
    fixed_rank1_accepted_index = cutlass.Int64(0)
if fixed_rank1_accepted_index > {num_tokens - 1}:
    fixed_rank1_accepted_index = cutlass.Int64({num_tokens - 1})
fixed_rank1_initial_slot = cutlass.Int32(
    {state_indices}[
        fixed_rank1_sequence,
        cutlass.Int32(fixed_rank1_accepted_index),
    ]
)
fixed_rank1_state_fp32 = cute.make_rmem_tensor(
    (8, {state_granules_per_thread}), cutlass.Float32
)
fixed_rank1_initial_state_offset = (
    cutlass.Uint32(fixed_rank1_initial_slot)
    * cutlass.Uint32({state}.layout.stride[0])
    + cutlass.Uint32(fixed_rank1_value_head)
    * cutlass.Uint32({state}.layout.stride[1])
    + cutlass.Uint32(fixed_rank1_value_index)
    * cutlass.Uint32({state}.layout.stride[2])
    + cutlass.Uint32(fixed_rank1_key_part * cutlass.Int32(8))
)
for fixed_rank1_granule in cutlass.range_constexpr(
    {state_granules_per_thread}
):
    fixed_rank1_initial_state_ptr = cute.recast_ptr(
        {state}.iterator
        + fixed_rank1_initial_state_offset
        + cutlass.Uint32(
            fixed_rank1_granule * cutlass.Int32({plan.key_split * 8})
        ),
        dtype=cutlass.Uint32,
    )
    fixed_rank1_initial_words = _cute_rank1_load_u32x4_if_valid(
        fixed_rank1_initial_state_ptr,
        fixed_rank1_initial_slot,
        cutlass.Int64({plan.state_size}),
    )
    for fixed_rank1_pair in cutlass.range_constexpr(4):
        fixed_rank1_initial_bits = cutlass.Uint32(
            fixed_rank1_initial_words[fixed_rank1_pair]
        )
        fixed_rank1_state_fp32[
            fixed_rank1_pair * 2, fixed_rank1_granule
        ] = cutlass.Float32(
            cutlass.Uint16(fixed_rank1_initial_bits).bitcast(
                cutlass.BFloat16
            )
        )
        fixed_rank1_state_fp32[
            fixed_rank1_pair * 2 + 1, fixed_rank1_granule
        ] = cutlass.Float32(
            cutlass.Uint16(
                fixed_rank1_initial_bits >> cutlass.Uint32(16)
            ).bitcast(cutlass.BFloat16)
        )

fixed_rank1_key_index = fixed_rank1_tid % cutlass.Int32({_KEY_SIZE})
fixed_rank1_decay_parameter = cute.exp2(
    cutlass.Float32({a_log}[fixed_rank1_value_head])
    * cutlass.Float32({log2_e}),
    fastmath=True,
)
fixed_rank1_bias_values = []
for fixed_rank1_element in cutlass.range_constexpr({elements_per_thread}):
    fixed_rank1_bias_values.append(
        cutlass.Float32(
            {bias}[
                fixed_rank1_value_head * cutlass.Int32({_KEY_SIZE})
                + fixed_rank1_key_index
                + fixed_rank1_element * cutlass.Int32({threads})
            ]
        )
    )

fixed_rank1_checkpoint_slots = []
fixed_rank1_values = []
fixed_rank1_betas = []
for fixed_rank1_token in cutlass.range_constexpr({num_tokens}):
    fixed_rank1_checkpoint_slots.append(
        cutlass.Int32(
            {state_indices}[fixed_rank1_sequence, fixed_rank1_token]
        )
    )
    fixed_rank1_packed_token = (
        fixed_rank1_sequence * cutlass.Int32({num_tokens})
        + fixed_rank1_token
    )
    fixed_rank1_values.append(
        cutlass.Float32(
            {v}[
                fixed_rank1_packed_token,
                fixed_rank1_value_head,
                fixed_rank1_value_index,
            ]
        )
    )
    fixed_rank1_betas.append(
        cutlass.Float32(
            {beta}[fixed_rank1_packed_token, fixed_rank1_value_head]
        )
    )
    fixed_rank1_query_sum = cutlass.Float32(0.0)
    fixed_rank1_key_sum = cutlass.Float32(0.0)
    for fixed_rank1_element in cutlass.range_constexpr({elements_per_thread}):
        fixed_rank1_dimension = (
            fixed_rank1_key_index
            + fixed_rank1_element * cutlass.Int32({threads})
        )
        fixed_rank1_query_value = cutlass.Float32(
            {q}[
                fixed_rank1_packed_token,
                fixed_rank1_value_head,
                fixed_rank1_dimension,
            ]
        )
        fixed_rank1_key_value = cutlass.Float32(
            {k}[
                fixed_rank1_packed_token,
                fixed_rank1_value_head,
                fixed_rank1_dimension,
            ]
        )
        fixed_rank1_gate_value = cutlass.Float32(
            {gate}[
                fixed_rank1_packed_token,
                fixed_rank1_value_head,
                fixed_rank1_dimension,
            ]
        )
        fixed_rank1_query_sum += (
            fixed_rank1_query_value * fixed_rank1_query_value
        )
        fixed_rank1_key_sum += fixed_rank1_key_value * fixed_rank1_key_value
        fixed_rank1_gate_input = (
            fixed_rank1_gate_value
            + fixed_rank1_bias_values[fixed_rank1_element]
        )
        fixed_rank1_sigmoid = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0)
            + cute.exp2(
                -(fixed_rank1_decay_parameter * fixed_rank1_gate_input)
                * cutlass.Float32({log2_e}),
                fastmath=True,
            )
        )
        fixed_rank1_decay = cute.exp2(
            cutlass.Float32({lower_bound})
            * fixed_rank1_sigmoid
            * cutlass.Float32({log2_e}),
            fastmath=True,
        )
        fixed_rank1_shared_offset = (
            fixed_rank1_token * cutlass.Int32({_KEY_SIZE})
            + fixed_rank1_dimension
        )
        fixed_rank1_decay_smem[fixed_rank1_shared_offset] = fixed_rank1_decay
        fixed_rank1_key_smem[fixed_rank1_shared_offset] = fixed_rank1_key_value
        fixed_rank1_query_smem[fixed_rank1_shared_offset] = fixed_rank1_query_value

    fixed_rank1_query_partial = cute.arch.warp_reduction_sum(
        fixed_rank1_query_sum
    )
    fixed_rank1_key_partial = cute.arch.warp_reduction_sum(fixed_rank1_key_sum)
    fixed_rank1_lane = fixed_rank1_tid % cutlass.Int32(32)
    fixed_rank1_warp = fixed_rank1_tid // cutlass.Int32(32)
    if (fixed_rank1_lane == 0) & (fixed_rank1_warp < {norm_warps}):
        fixed_rank1_norm_smem[
            fixed_rank1_token * cutlass.Int32(16) + fixed_rank1_warp
        ] = fixed_rank1_query_partial
        fixed_rank1_norm_smem[
            fixed_rank1_token * cutlass.Int32(16)
            + cutlass.Int32(8)
            + fixed_rank1_warp
        ] = fixed_rank1_key_partial
cute.arch.sync_threads()

fixed_rank1_partials = cute.make_rmem_tensor((8,), cutlass.Float32)
for fixed_rank1_token in cutlass.range_constexpr({num_tokens}):
    fixed_rank1_checkpoint_slot = fixed_rank1_checkpoint_slots[
        fixed_rank1_token
    ]
    fixed_rank1_packed_token = (
        fixed_rank1_sequence * cutlass.Int32({num_tokens})
        + fixed_rank1_token
    )
    fixed_rank1_query_norm = cutlass.Float32(0.0)
    fixed_rank1_key_norm = cutlass.Float32(0.0)
    for fixed_rank1_warp in cutlass.range_constexpr({norm_warps}):
        fixed_rank1_query_norm += fixed_rank1_norm_smem[
            fixed_rank1_token * cutlass.Int32(16) + fixed_rank1_warp
        ]
        fixed_rank1_key_norm += fixed_rank1_norm_smem[
            fixed_rank1_token * cutlass.Int32(16)
            + cutlass.Int32(8)
            + fixed_rank1_warp
        ]
    fixed_rank1_query_scale = cute.rsqrt(
        fixed_rank1_query_norm + cutlass.Float32({epsilon}),
        fastmath=True,
    ) * cutlass.Float32({scale})
    fixed_rank1_key_scale = cute.rsqrt(
        fixed_rank1_key_norm + cutlass.Float32({epsilon}),
        fastmath=True,
    )

    fixed_rank1_state_vector = (
        fixed_rank1_state_fp32[None, 0].load()
        * fixed_rank1_decay_fragment[None, 0, fixed_rank1_token].load()
    )
    fixed_rank1_state_fp32[None, 0].store(fixed_rank1_state_vector)
    fixed_rank1_prediction_vector = (
        fixed_rank1_key_fragment[None, 0, fixed_rank1_token].load()
        * fixed_rank1_state_vector
    )
    for fixed_rank1_granule in cutlass.range_constexpr(
        1, {state_granules_per_thread}, 1
    ):
        fixed_rank1_state_vector = (
            fixed_rank1_state_fp32[None, fixed_rank1_granule].load()
            * fixed_rank1_decay_fragment[
                None, fixed_rank1_granule, fixed_rank1_token
            ].load()
        )
        fixed_rank1_state_fp32[None, fixed_rank1_granule].store(
            fixed_rank1_state_vector
        )
        fixed_rank1_prediction_vector += (
            fixed_rank1_key_fragment[
                None, fixed_rank1_granule, fixed_rank1_token
            ].load()
            * fixed_rank1_state_vector
        )
    fixed_rank1_partials.store(fixed_rank1_prediction_vector)
    fixed_rank1_prediction_01 = (
        fixed_rank1_partials[0] + fixed_rank1_partials[1]
    )
    fixed_rank1_prediction_23 = (
        fixed_rank1_partials[2] + fixed_rank1_partials[3]
    )
    fixed_rank1_prediction_45 = (
        fixed_rank1_partials[4] + fixed_rank1_partials[5]
    )
    fixed_rank1_prediction_67 = (
        fixed_rank1_partials[6] + fixed_rank1_partials[7]
    )
    fixed_rank1_prediction_left = (
        fixed_rank1_prediction_01 + fixed_rank1_prediction_23
    )
    fixed_rank1_prediction_right = (
        fixed_rank1_prediction_45 + fixed_rank1_prediction_67
    )
    fixed_rank1_prediction = (
        fixed_rank1_prediction_left + fixed_rank1_prediction_right
    )
{prediction_shuffle}
    fixed_rank1_delta_key = (
        fixed_rank1_key_scale
        * fixed_rank1_betas[fixed_rank1_token]
        * (
            fixed_rank1_values[fixed_rank1_token]
            - fixed_rank1_key_scale * fixed_rank1_prediction
        )
    )
    fixed_rank1_state_vector = (
        fixed_rank1_state_fp32[None, 0].load()
        + fixed_rank1_key_fragment[None, 0, fixed_rank1_token].load()
        * fixed_rank1_delta_key
    )
    fixed_rank1_state_fp32[None, 0].store(fixed_rank1_state_vector)
    fixed_rank1_output_vector = (
        fixed_rank1_query_fragment[None, 0, fixed_rank1_token].load()
        * fixed_rank1_state_vector
    )
    for fixed_rank1_granule in cutlass.range_constexpr(
        1, {state_granules_per_thread}, 1
    ):
        fixed_rank1_state_vector = (
            fixed_rank1_state_fp32[None, fixed_rank1_granule].load()
            + fixed_rank1_key_fragment[
                None, fixed_rank1_granule, fixed_rank1_token
            ].load()
            * fixed_rank1_delta_key
        )
        fixed_rank1_state_fp32[None, fixed_rank1_granule].store(
            fixed_rank1_state_vector
        )
        fixed_rank1_output_vector += (
            fixed_rank1_query_fragment[
                None, fixed_rank1_granule, fixed_rank1_token
            ].load()
            * fixed_rank1_state_vector
        )
    fixed_rank1_partials.store(fixed_rank1_output_vector)
    fixed_rank1_projected_01 = (
        fixed_rank1_partials[0] + fixed_rank1_partials[1]
    )
    fixed_rank1_projected_23 = (
        fixed_rank1_partials[2] + fixed_rank1_partials[3]
    )
    fixed_rank1_projected_45 = (
        fixed_rank1_partials[4] + fixed_rank1_partials[5]
    )
    fixed_rank1_projected_67 = (
        fixed_rank1_partials[6] + fixed_rank1_partials[7]
    )
    fixed_rank1_projected_left = (
        fixed_rank1_projected_01 + fixed_rank1_projected_23
    )
    fixed_rank1_projected_right = (
        fixed_rank1_projected_45 + fixed_rank1_projected_67
    )
    fixed_rank1_projected = (
        fixed_rank1_projected_left + fixed_rank1_projected_right
    )
{projection_shuffle}
    if fixed_rank1_key_part == 0:
        {output}[
            fixed_rank1_packed_token,
            fixed_rank1_value_head,
            fixed_rank1_value_index,
        ] = (fixed_rank1_query_scale * fixed_rank1_projected).to(
            cutlass.BFloat16
        )

    fixed_rank1_checkpoint_offset = (
        cutlass.Uint32(fixed_rank1_checkpoint_slot)
        * cutlass.Uint32({state}.layout.stride[0])
        + cutlass.Uint32(fixed_rank1_value_head)
        * cutlass.Uint32({state}.layout.stride[1])
        + cutlass.Uint32(fixed_rank1_value_index)
        * cutlass.Uint32({state}.layout.stride[2])
        + cutlass.Uint32(fixed_rank1_key_part * cutlass.Int32(8))
    )
    fixed_rank1_checkpoint_words = cute.make_rmem_tensor(
        (4,), cutlass.Uint32
    )
    for fixed_rank1_granule in cutlass.range_constexpr(
        {state_granules_per_thread}
    ):
        for fixed_rank1_pair in cutlass.range_constexpr(4):
            fixed_rank1_checkpoint_words[fixed_rank1_pair] = (
                _cute_rank1_pack_bf16x2(
                    fixed_rank1_state_fp32[
                        fixed_rank1_pair * 2, fixed_rank1_granule
                    ],
                    fixed_rank1_state_fp32[
                        fixed_rank1_pair * 2 + 1,
                        fixed_rank1_granule,
                    ],
                )
            )
        fixed_rank1_checkpoint_ptr = cute.recast_ptr(
            {state}.iterator
            + fixed_rank1_checkpoint_offset
            + cutlass.Uint32(
                fixed_rank1_granule
                * cutlass.Int32({plan.key_split * 8})
            ),
            dtype=cutlass.Uint32,
        )
        _cute_rank1_store_u32x4_if_valid(
            fixed_rank1_checkpoint_ptr,
            fixed_rank1_checkpoint_words[0],
            fixed_rank1_checkpoint_words[1],
            fixed_rank1_checkpoint_words[2],
            fixed_rank1_checkpoint_words[3],
            fixed_rank1_checkpoint_slot,
            cutlass.Int64({plan.state_size}),
        )
"""
    arguments.install(df, preamble, body)
    return True
