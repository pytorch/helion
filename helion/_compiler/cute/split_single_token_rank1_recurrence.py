"""Packed one-warp lowering for split-input single-token rank-1 recurrence.

This module recognizes the complete T=1 recurrent KDA graph over separate
Q/K/V/gate tensors and emits a native BF16x2 one-warp schedule.  The matcher
is deliberately fail closed: it proves the recurrence, overflow-safe
unbounded-softplus gate, beta-logit sigmoid, normalization, tensor indexing,
stores, logical grid, runtime shapes, dtypes, contiguity, alignment, and
aliasing before returning a plan.

The packed schedule rounds normalized Q/K, decay, state updates, and reduction
partials to BF16.  It is therefore an approximate lowering and is admitted
only under the user-facing ``Settings.fast_math`` policy, which explicitly
permits reduced precision and changed NaN/Inf behavior.

Integration hooks intentionally live elsewhere.  Keeping planning and codegen
self-contained lets callers install this path only after all other structural
lowering hooks have been ordered explicitly.
"""

from __future__ import annotations

import ast
import dataclasses
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from .fixed_token_rank1_recurrence import _is_direct_float_load
from .fixed_token_rank1_recurrence import _match_accepted_index
from .fixed_token_rank1_recurrence import _match_normalized_vector
from .fixed_token_rank1_recurrence import _matches_bias_index
from .fixed_token_rank1_recurrence import _same_scalar
from .fixed_token_rank1_recurrence import _tensor_shapes_match
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _GeneratedCodeTemplate
from .fx_matcher import _linear_offsets_fit_i32
from .single_token_rank1_recurrence import _ADD_TARGETS
from .single_token_rank1_recurrence import _MUL_TARGETS
from .single_token_rank1_recurrence import _RANK1_HELPER_ABI_VERSION
from .single_token_rank1_recurrence import _binary_other
from .single_token_rank1_recurrence import _binary_pair
from .single_token_rank1_recurrence import _broadcast_source
from .single_token_rank1_recurrence import _cast_from
from .single_token_rank1_recurrence import _has_unmatched_side_effect
from .single_token_rank1_recurrence import _indices
from .single_token_rank1_recurrence import _is_call
from .single_token_rank1_recurrence import _is_standard_key_iota
from .single_token_rank1_recurrence import _load_ref
from .single_token_rank1_recurrence import _match_factored_output
from .single_token_rank1_recurrence import _same_ref
from .single_token_rank1_recurrence import _store_ref
from .single_token_rank1_recurrence import _strip_casts
from .single_token_rank1_recurrence import _strip_state_index_casts
from .single_token_rank1_recurrence import _sum_input
from .single_token_rank1_recurrence import _TensorRef

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


_KEY_SIZE = 128
_VALUE_SIZE = 128
_VALUE_TILE = 16
_THREADS = 32
_K_LANES = 16
_V_LANES = 2
_VALUES_PER_THREAD = 4
_NORM_EPSILON = 1.0e-6
_LOG2_E = 1.4426950408889634
_SOFTPLUS_THRESHOLD = 20.0
_SPLIT_T1_CODEGEN_ABI_VERSION = 1


@dataclasses.dataclass(frozen=True)
class CuteSplitSingleTokenRank1Plan:
    """Values consumed by the split-input packed code generator."""

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
    log2_e: object
    epsilon: object
    softplus_threshold: object
    num_sequences: int
    num_heads: int
    state_size: int
    state_stride0: int
    tile_rows: int = _VALUE_TILE
    threads: int = _THREADS


@dataclasses.dataclass(frozen=True)
class _MatchedT1Step:
    state_update: torch.fx.Node
    output_store: torch.fx.Node
    state_store: torch.fx.Node
    q_load: torch.fx.Node
    k_load: torch.fx.Node
    v_load: torch.fx.Node
    gate_load: torch.fx.Node
    beta_load: torch.fx.Node
    a_log_load: torch.fx.Node
    bias_load: torch.fx.Node
    checkpoint_load: torch.fx.Node
    token_coord: torch.fx.Node
    head_coord: torch.fx.Node
    value_coord: torch.fx.Node
    key_coord: torch.fx.Node
    scale: object
    log2_e: object
    epsilon: object
    softplus_threshold: object

    @property
    def loads(self) -> set[torch.fx.Node]:
        return {
            self.q_load,
            self.k_load,
            self.v_load,
            self.gate_load,
            self.beta_load,
            self.a_log_load,
            self.bias_load,
            self.checkpoint_load,
        }


def _scalar_value(value: object) -> object:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _static_float(value: object) -> float | None:
    """Resolve a specialized scalar without accepting a live symbolic value."""

    import sympy

    from ..host_function import HostFunction
    from ..variable_origin import AttributeOrigin
    from ..variable_origin import NameOrigin

    def replay_origin(origin: object) -> object | None:
        host_function = HostFunction.current()
        if isinstance(origin, AttributeOrigin):
            base = replay_origin(origin.value)
            return getattr(base, origin.key) if base is not None else None
        if isinstance(origin, NameOrigin):
            if origin.name in host_function.params.arguments:
                return host_function.params.arguments[origin.name]
            imported = host_function.global_imports.get(origin.name)
            return imported.value if imported is not None else None
        return None

    value = _scalar_value(value)
    if isinstance(value, (int, float)):
        return float(cast("int | float", value))
    if not isinstance(value, (torch.SymFloat, sympy.Expr)):
        return None
    expr = value._sympy_() if isinstance(value, torch.SymFloat) else value
    if not isinstance(expr, sympy.Expr):
        return None
    expr = CompileEnvironment.current().specialize_expr(expr)
    if not expr.free_symbols:
        try:
            return float(expr)
        except TypeError:
            return None
    origin = HostFunction.current().expr_to_origin.get(expr)
    resolved = replay_origin(origin.origin) if origin is not None else None
    return (
        float(cast("int | float", resolved))
        if isinstance(resolved, (int, float))
        else None
    )


def _is_direct_scalar_source(value: object) -> bool:
    """Accept a literal or one scalar kernel argument, never an expression."""

    from ...language._tracing_ops import _get_symnode

    return type(value) in (int, float) or (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target is _get_symnode
        and len(value.args) == 1
        and isinstance(value.args[0], str)
        and not value.kwargs
    )


def _match_negation(node: object) -> torch.fx.Node | None:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.neg.default
        and len(node.args) == 1
        and isinstance(node.args[0], torch.fx.Node)
        and not node.kwargs
    ):
        return node.args[0]
    return None


def _match_exp2_scaled(
    node: object,
    scale_node: torch.fx.Node | None = None,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """Match ``exp2(value * scale)`` and return both operands."""

    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not torch.ops.aten.exp2.default
        or len(node.args) != 1
        or node.kwargs
    ):
        return None
    pair = _binary_pair(node.args[0], _MUL_TARGETS)
    if pair is None:
        return None
    for value, maybe_scale in (pair, reversed(pair)):
        if scale_node is None or maybe_scale is scale_node:
            return value, maybe_scale
    return None


def _match_unbounded_decay(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, object, object] | None:
    """Match the exact overflow-safe unbounded-softplus decay graph."""

    outer = _match_exp2_scaled(node)
    if outer is None:
        return None
    log_decay, log2_e_node = outer
    if not isinstance(log2_e_node, torch.fx.Node):
        return None
    log_pair = _binary_pair(log_decay, _MUL_TARGETS)
    if log_pair is None:
        return None
    decay_parameter: torch.fx.Node | None = None
    softplus: torch.fx.Node | None = None
    for maybe_negative_a, maybe_softplus in (log_pair, reversed(log_pair)):
        maybe_a = _match_negation(maybe_negative_a)
        if maybe_a is not None:
            decay_parameter = maybe_a
            softplus = maybe_softplus
            break
    if decay_parameter is None or softplus is None:
        return None

    a_match = _match_exp2_scaled(decay_parameter, log2_e_node)
    if a_match is None:
        return None
    a_log, _ = a_match
    if (
        softplus.op != "call_function"
        or softplus.target is not torch.ops.aten.where.self
        or len(softplus.args) != 3
        or softplus.kwargs
    ):
        return None
    condition, log_branch, linear_branch = softplus.args
    if (
        not isinstance(condition, torch.fx.Node)
        or condition.op != "call_function"
        or condition.target is not torch.ops.aten.le.Scalar
        or len(condition.args) != 2
        or condition.kwargs
        or condition.args[0] is not linear_branch
    ):
        return None
    threshold = _scalar_value(condition.args[1])
    if (
        not isinstance(log_branch, torch.fx.Node)
        or log_branch.op != "call_function"
        or log_branch.target is not torch.ops.aten.log.default
        or len(log_branch.args) != 1
        or log_branch.kwargs
    ):
        return None
    plus_one_node = log_branch.args[0]
    if (
        not isinstance(plus_one_node, torch.fx.Node)
        or plus_one_node.op != "call_function"
        or plus_one_node.target not in _ADD_TARGETS
        or len(plus_one_node.args) < 2
        or plus_one_node.kwargs
    ):
        return None
    gate_exp: torch.fx.Node | None = None
    plus_one_args = plus_one_node.args[:2]
    for maybe_exp, maybe_one in (plus_one_args, reversed(plus_one_args)):
        if (
            isinstance(maybe_exp, torch.fx.Node)
            and isinstance(maybe_one, (int, float))
            and float(cast("int | float", maybe_one)) == 1.0
        ):
            gate_exp = maybe_exp
            break
    if gate_exp is None:
        return None
    gate_exp_match = _match_exp2_scaled(gate_exp, log2_e_node)
    if gate_exp_match is None or gate_exp_match[0] is not linear_branch:
        return None
    gate_plus_bias = linear_branch
    gate_pair = _binary_pair(gate_plus_bias, _ADD_TARGETS)
    if gate_pair is None:
        return None
    gate, bias = gate_pair
    return (
        gate,
        bias,
        a_log,
        _scalar_value(log2_e_node),
        threshold,
    )


def _match_beta_logit(node: object) -> torch.fx.Node | None:
    """Return the BF16 load under an exact FP32 sigmoid(beta-logit)."""

    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not torch.ops.aten.sigmoid.default
        or len(node.args) != 1
        or node.kwargs
    ):
        return None
    return _is_direct_float_load(node.args[0])


def _match_scaled_factored_projection(
    output_store: torch.fx.Node,
    decayed: torch.fx.Node,
    residual: torch.fx.Node,
    key_broadcast: torch.fx.Node,
) -> tuple[torch.fx.Node, object, object] | None:
    """Match the factored output graph and return Q load/scale/epsilon."""

    stored = output_store.args[2] if len(output_store.args) > 2 else None
    output_value = _strip_casts(stored)
    if output_value is None or not _cast_from(stored, output_value, torch.bfloat16):
        return None
    scaled = _binary_pair(output_value, _MUL_TARGETS)
    if scaled is None:
        return None
    for factored, scale_source in (scaled, reversed(scaled)):
        if not _is_direct_scalar_source(scale_source):
            continue
        normalized_query = _match_factored_output(
            factored,
            decayed,
            residual,
            key_broadcast,
        )
        if normalized_query is None:
            continue
        query_match = _match_normalized_vector(normalized_query)
        scale = _scalar_value(scale_source)
        if query_match is None or not isinstance(scale, (int, float, torch.SymFloat)):
            continue
        _, query_fp32, epsilon = query_match
        query_load = _is_direct_float_load(query_fp32)
        if query_load is not None:
            return query_load, scale, epsilon
    return None


def _match_t1_step(
    state_store: torch.fx.Node,
    output_store: torch.fx.Node,
    incoming_state: torch.fx.Node,
) -> _MatchedT1Step | None:
    """Match the complete T1 update, projection, and both stores."""

    stored_state = state_store.args[2] if len(state_store.args) > 2 else None
    state_update = _strip_casts(stored_state)
    update_pair = _binary_pair(state_update, _ADD_TARGETS)
    if (
        state_update is None
        or update_pair is None
        or not _cast_from(stored_state, state_update, torch.bfloat16)
    ):
        return None

    decayed: torch.fx.Node | None = None
    decay: torch.fx.Node | None = None
    residual: torch.fx.Node | None = None
    normalized_key: torch.fx.Node | None = None
    key_broadcast: torch.fx.Node | None = None
    for maybe_decayed, maybe_outer in (update_pair, reversed(update_pair)):
        decay_broadcast = _binary_other(maybe_decayed, incoming_state, _MUL_TARGETS)
        outer_pair = _binary_pair(maybe_outer, _MUL_TARGETS)
        if decay_broadcast is None or outer_pair is None:
            continue
        maybe_decay = _broadcast_source(decay_broadcast, (None, slice(None)))
        if maybe_decay is None:
            continue
        for residual_broadcast, maybe_key_broadcast in (
            outer_pair,
            reversed(outer_pair),
        ):
            maybe_residual = _broadcast_source(residual_broadcast, (slice(None), None))
            maybe_key = _broadcast_source(maybe_key_broadcast, (None, slice(None)))
            if maybe_residual is None or maybe_key is None:
                continue
            decayed = maybe_decayed
            decay = maybe_decay
            residual = maybe_residual
            normalized_key = maybe_key
            key_broadcast = maybe_key_broadcast
            break
        if decayed is not None:
            break
    if any(
        value is None
        for value in (decayed, decay, residual, normalized_key, key_broadcast)
    ):
        return None
    assert decayed is not None
    assert decay is not None
    assert residual is not None
    assert normalized_key is not None
    assert key_broadcast is not None

    residual_pair = _binary_pair(residual, _MUL_TARGETS)
    if residual_pair is None:
        return None
    difference: torch.fx.Node | None = None
    beta_load: torch.fx.Node | None = None
    for maybe_difference, maybe_beta in (residual_pair, reversed(residual_pair)):
        if (
            maybe_difference.op == "call_function"
            and maybe_difference.target is torch.ops.aten.sub.Tensor
            and len(maybe_difference.args) == 2
            and not maybe_difference.kwargs
        ):
            matched_beta = _match_beta_logit(maybe_beta)
            if matched_beta is not None:
                difference = maybe_difference
                beta_load = matched_beta
                break
    if difference is None or beta_load is None:
        return None
    value_fp32, prediction = difference.args
    if not isinstance(value_fp32, torch.fx.Node) or not isinstance(
        prediction, torch.fx.Node
    ):
        return None
    value_load = _is_direct_float_load(value_fp32)
    prediction_product = _sum_input(prediction, (-1,))
    prediction_pair = _binary_pair(prediction_product, _MUL_TARGETS)
    if value_load is None or prediction_pair is None:
        return None
    if not any(
        maybe_decayed is decayed
        and _broadcast_source(maybe_key, (None, slice(None))) is normalized_key
        for maybe_decayed, maybe_key in (prediction_pair, reversed(prediction_pair))
    ):
        return None

    key_match = _match_normalized_vector(normalized_key)
    if key_match is None:
        return None
    _, key_fp32, key_epsilon = key_match
    key_load = _is_direct_float_load(key_fp32)
    if key_load is None:
        return None

    decay_match = _match_unbounded_decay(decay)
    if decay_match is None:
        return None
    gate_fp32, bias_value, a_log_value, log2_e, threshold = decay_match
    gate_load = _is_direct_float_load(gate_fp32)
    bias_load = _strip_casts(bias_value)
    a_log_load = _strip_casts(a_log_value)
    if (
        gate_load is None
        or not isinstance(bias_load, torch.fx.Node)
        or not isinstance(a_log_load, torch.fx.Node)
        or _load_ref(bias_load) is None
        or _load_ref(a_log_load) is None
        or bias_value is not bias_load
        or a_log_value is not a_log_load
    ):
        return None

    projection_match = _match_scaled_factored_projection(
        output_store,
        decayed,
        residual,
        key_broadcast,
    )
    if projection_match is None:
        return None
    query_load, scale, query_epsilon = projection_match
    if not _same_scalar(query_epsilon, key_epsilon):
        return None

    state_indices = _indices(state_store)
    output_indices = _indices(output_store)
    query_indices = _indices(query_load)
    key_indices = _indices(key_load)
    value_indices = _indices(value_load)
    gate_indices = _indices(gate_load)
    beta_indices = _indices(beta_load)
    a_log_indices = _indices(a_log_load)
    bias_indices = _indices(bias_load)
    all_indices = (
        state_indices,
        output_indices,
        query_indices,
        key_indices,
        value_indices,
        gate_indices,
        beta_indices,
        a_log_indices,
        bias_indices,
    )
    if any(indices is None for indices in all_indices):
        return None
    assert state_indices is not None
    assert output_indices is not None
    assert query_indices is not None
    assert key_indices is not None
    assert value_indices is not None
    assert gate_indices is not None
    assert beta_indices is not None
    assert a_log_indices is not None
    assert bias_indices is not None
    if (
        len(state_indices) != 4
        or len(output_indices) != 3
        or len(query_indices) != 3
        or len(key_indices) != 3
        or len(value_indices) != 3
        or len(gate_indices) != 3
        or len(beta_indices) != 2
        or len(a_log_indices) != 1
        or len(bias_indices) != 1
    ):
        return None
    token_coord, head_coord, value_coord = output_indices
    if not all(
        isinstance(coord, torch.fx.Node)
        for coord in (token_coord, head_coord, value_coord)
    ):
        return None
    assert isinstance(token_coord, torch.fx.Node)
    assert isinstance(head_coord, torch.fx.Node)
    assert isinstance(value_coord, torch.fx.Node)
    key_coord = query_indices[2]
    if not isinstance(key_coord, torch.fx.Node) or not _is_standard_key_iota(key_coord):
        return None
    if (
        query_indices != (token_coord, head_coord, key_coord)
        or key_indices != (token_coord, head_coord, key_coord)
        or value_indices != (token_coord, head_coord, value_coord)
        or gate_indices != (token_coord, head_coord, key_coord)
        or beta_indices != (token_coord, head_coord)
        or a_log_indices != (head_coord,)
        or state_indices[1:] != (head_coord, value_coord, key_coord)
        or not _matches_bias_index(bias_indices[0], head_coord, key_coord)
    ):
        return None
    checkpoint_load = _strip_state_index_casts(state_indices[0])
    if (
        not isinstance(checkpoint_load, torch.fx.Node)
        or _load_ref(checkpoint_load) is None
    ):
        return None

    return _MatchedT1Step(
        state_update=state_update,
        output_store=output_store,
        state_store=state_store,
        q_load=query_load,
        k_load=key_load,
        v_load=value_load,
        gate_load=gate_load,
        beta_load=beta_load,
        a_log_load=a_log_load,
        bias_load=bias_load,
        checkpoint_load=checkpoint_load,
        token_coord=token_coord,
        head_coord=head_coord,
        value_coord=value_coord,
        key_coord=key_coord,
        scale=scale,
        log2_e=log2_e,
        epsilon=query_epsilon,
        softplus_threshold=threshold,
    )


def _plan_split_single_token_rank1_recurrence(
    graphs: Sequence[GraphInfo],
    _tile_strategy: TileStrategyDispatch,
) -> CuteSplitSingleTokenRank1Plan | None:
    """Return a plan only after proving the complete split-input T1 graph."""

    from ...language import memory_ops
    from ..device_function import DeviceFunction
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    capability = env.config_spec.target_device_capability
    # This schedule intentionally demotes FP32 intermediates and reassociates
    # reductions.  Settings.fast_math is the user opt-in for those changes.
    if (
        not env.settings.fast_math
        or df.config.pid_type != "flat"
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
    output_stores = [store for store in stores if store not in state_stores]
    if (
        len(state_stores) != 1
        or len(output_stores) != 1
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
    accepted_load = _match_accepted_index(accepted_index, 1)
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

    matched = _match_t1_step(state_stores[0], output_stores[0], state_fp32)
    if matched is None:
        return None
    if (
        initial_indices[1:]
        != (matched.head_coord, matched.value_coord, matched.key_coord)
        or matched.token_coord is not sequence_coord
    ):
        return None
    checkpoint_indices = _indices(matched.checkpoint_load)
    if checkpoint_indices != (sequence_coord, 0) or not _same_ref(
        _load_ref(matched.checkpoint_load), _load_ref(initial_slot)
    ):
        return None

    refs = {
        "q": cast("_TensorRef", _load_ref(matched.q_load)),
        "k": cast("_TensorRef", _load_ref(matched.k_load)),
        "v": cast("_TensorRef", _load_ref(matched.v_load)),
        "gate": cast("_TensorRef", _load_ref(matched.gate_load)),
        "beta": cast("_TensorRef", _load_ref(matched.beta_load)),
        "a_log": cast("_TensorRef", _load_ref(matched.a_log_load)),
        "bias": cast("_TensorRef", _load_ref(matched.bias_load)),
        "state_indices": cast("_TensorRef", _load_ref(initial_slot)),
        "num_accepted_tokens": cast("_TensorRef", _load_ref(accepted_load)),
        "state": state_ref,
        "output": cast("_TensorRef", _store_ref(output_stores[0])),
    }
    if any(ref is None for ref in refs.values()):
        return None
    expected_loads = {state_load, initial_slot, accepted_load}
    expected_loads.update(matched.loads)
    if set(loads) != expected_loads or set(stores) != {
        state_stores[0],
        output_stores[0],
    }:
        return None

    shape_match = _tensor_shapes_match(refs, 1)
    if shape_match is None:
        return None
    num_sequences, num_heads = shape_match
    if not all(
        isinstance(_scalar_value(value), (int, float, torch.SymFloat))
        for value in (matched.scale, matched.epsilon)
    ):
        return None
    if (
        _static_float(matched.epsilon) != _NORM_EPSILON
        or _static_float(matched.log2_e) != _LOG2_E
        or _static_float(matched.softplus_threshold) != _SOFTPLUS_THRESHOLD
    ):
        return None

    linear_refs = tuple(
        refs[name]
        for name in (
            "q",
            "k",
            "v",
            "gate",
            "beta",
            "a_log",
            "bias",
            "state_indices",
            "num_accepted_tokens",
            "output",
        )
    )
    runtime_linear_values = tuple(
        env.runtime_value_for_tensor(ref.fake) for ref in linear_refs
    )
    if not all(isinstance(value, torch.Tensor) for value in runtime_linear_values):
        return None
    runtime_linear_tensors = cast("tuple[torch.Tensor, ...]", runtime_linear_values)
    if not _linear_offsets_fit_i32(
        num_sequences * num_heads * (_VALUE_SIZE // _VALUE_TILE),
        tuple(tensor.numel() for tensor in runtime_linear_tensors),
    ):
        return None

    device_ir = HostFunction.current().device_ir
    grid_coords = (sequence_coord, matched.head_coord, matched.value_coord)
    grid_coord_ids = _canonical_root_axis_ids(
        device_ir,
        root_phase_index=root.phase_index,
        coordinates=grid_coords,
        expected_extents=(num_sequences, num_heads, _VALUE_SIZE),
    )
    if grid_coord_ids is None or [
        df.resolved_block_size(block_id) for block_id in grid_coord_ids
    ] != [1, 1, _VALUE_TILE]:
        return None

    runtime_state = env.runtime_value_for_tensor(state_ref.fake)
    if not isinstance(runtime_state, torch.Tensor):
        return None
    return CuteSplitSingleTokenRank1Plan(
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
        scale=matched.scale,
        log2_e=matched.log2_e,
        epsilon=matched.epsilon,
        softplus_threshold=matched.softplus_threshold,
        num_sequences=num_sequences,
        num_heads=num_heads,
        state_size=runtime_state.shape[0],
        state_stride0=runtime_state.stride(0),
    )


def plan_split_single_token_rank1_recurrence(
    graphs: Sequence[GraphInfo],
    tile_strategy: TileStrategyDispatch,
) -> CuteSplitSingleTokenRank1Plan | None:
    """Plan the split T1 lowering and promote its semantic scalar arguments."""

    from ..device_function import DeviceFunction

    plan = _plan_split_single_token_rank1_recurrence(graphs, tile_strategy)
    if plan is not None:
        device_function = DeviceFunction.current()
        device_function.promote_expr_arg_to_constexpr(plan.scale)
        device_function.promote_expr_arg_to_constexpr(plan.log2_e)
        device_function.promote_expr_arg_to_constexpr(plan.epsilon)
        device_function.promote_expr_arg_to_constexpr(plan.softplus_threshold)
    return plan


def _tensor_arg(cg: GenerateAST, ref: _TensorRef) -> str:
    return cg.device_function.tensor_arg(ref.fake, prefer_name=ref.suggested_name).name


def codegen_split_single_token_rank1_recurrence(
    cg: GenerateAST,
    plan: CuteSplitSingleTokenRank1Plan,
) -> bool:
    """Replace the proven root with the packed one-warp hand schedule."""

    root = cg.current_root_graph_info
    if root is None or root.graph_id != plan.root_graph_id:
        return False
    df = cg.device_function
    q = _tensor_arg(cg, plan.q)
    k = _tensor_arg(cg, plan.k)
    v = _tensor_arg(cg, plan.v)
    gate = _tensor_arg(cg, plan.gate)
    beta = _tensor_arg(cg, plan.beta)
    a_log = _tensor_arg(cg, plan.a_log)
    bias = _tensor_arg(cg, plan.bias)
    state_indices = _tensor_arg(cg, plan.state_indices)
    accepted = _tensor_arg(cg, plan.num_accepted_tokens)
    state = _tensor_arg(cg, plan.state)
    output = _tensor_arg(cg, plan.output)
    scale = df.literal_expr(plan.scale)
    log2_e = df.literal_expr(plan.log2_e)
    epsilon = df.literal_expr(plan.epsilon)
    threshold = _static_float(plan.softplus_threshold)
    if threshold is None:
        return False
    placeholder_args = {
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
    }
    template = _GeneratedCodeTemplate(
        "split_t1",
        (*tuple(argument.name for argument in df.arguments), *cg._extra_params),
        df.new_var,
    )
    q = template.protect(q)
    k = template.protect(k)
    v = template.protect(v)
    gate = template.protect(gate)
    beta = template.protect(beta)
    a_log = template.protect(a_log)
    bias = template.protect(bias)
    state_indices = template.protect(state_indices)
    accepted = template.protect(accepted)
    state = template.protect(state)
    output = template.protect(output)
    scale = template.protect(scale)
    log2_e = template.protect(log2_e)
    epsilon = template.protect(epsilon)

    preamble = f"""
split_t1_codegen_abi_version = {_SPLIT_T1_CODEGEN_ABI_VERSION}
split_t1_rank1_helper_abi_version = {_RANK1_HELPER_ABI_VERSION}
split_t1_state_bits = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
split_t1_query_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
split_t1_key_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
split_t1_decay_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
split_t1_decayed_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
"""
    body = f"""
split_t1_work = cutlass.Int32(cute.arch.block_idx()[0])
split_t1_value_tile = split_t1_work % cutlass.Int32(8)
split_t1_sequence_head = split_t1_work // cutlass.Int32(8)
split_t1_head = split_t1_sequence_head % cutlass.Int32({plan.num_heads})
split_t1_sequence = split_t1_sequence_head // cutlass.Int32({plan.num_heads})
split_t1_value_offset = split_t1_value_tile * cutlass.Int32({_VALUE_TILE})
split_t1_thread = cutlass.Int32(cute.arch.thread_idx()[0])
split_t1_key_lane = split_t1_thread % cutlass.Int32({_K_LANES})
split_t1_value_lane = split_t1_thread // cutlass.Int32({_K_LANES})
split_t1_source_base = split_t1_thread * cutlass.Int32({_VALUES_PER_THREAD})

# T=1 makes clamp(num_accepted_tokens[n] - 1, 0, 0) identically zero.
split_t1_state_index = cutlass.Int64(
    _cute_rank1_load_i32_nc(
        {state_indices}.iterator + split_t1_sequence
    )
)
split_t1_query_head_offset = (
    split_t1_sequence * cutlass.Int32({plan.num_heads}) + split_t1_head
) * cutlass.Int32({_KEY_SIZE})
split_t1_value_head_offset = split_t1_query_head_offset
split_t1_query_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {q}.iterator + split_t1_query_head_offset + split_t1_source_base,
        dtype=cutlass.Uint32,
    )
)
split_t1_key_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {k}.iterator + split_t1_query_head_offset + split_t1_source_base,
        dtype=cutlass.Uint32,
    )
)
split_t1_gate_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {gate}.iterator + split_t1_value_head_offset + split_t1_source_base,
        dtype=cutlass.Uint32,
    )
)
split_t1_bias_raw = _cute_rank1_load_f32x4_nc(
    cute.recast_ptr(
        {bias}.iterator
        + split_t1_head * cutlass.Int32({_KEY_SIZE})
        + split_t1_source_base,
        dtype=cutlass.Float32,
    )
)
split_t1_a_log = _cute_rank1_load_f32_nc(
    {a_log}.iterator + split_t1_head
)

# The complete 16x128 tile is loaded before recurrence and remains packed in
# registers. The shared unsigned bound helper maps negative and upper-invalid
# slots to zero without dereferencing the computed pointer.
for split_t1_row in cutlass.range_constexpr(8):
    split_t1_value_index = (
        split_t1_value_offset
        + split_t1_value_lane
        + cutlass.Int32({_V_LANES}) * split_t1_row
    )
    split_t1_state_element_offset = (
        cutlass.Uint32(split_t1_state_index)
        * cutlass.Uint32({plan.state_stride0})
        + cutlass.Uint32(split_t1_head * cutlass.Int32({_VALUE_SIZE * _KEY_SIZE}))
        + cutlass.Uint32(
            split_t1_value_index * cutlass.Int32({_KEY_SIZE})
            + split_t1_key_lane * cutlass.Int32(8)
        )
    )
    split_t1_state_pointer = cute.recast_ptr(
        {state}.iterator + split_t1_state_element_offset,
        dtype=cutlass.Uint32,
    )
    split_t1_state_vector = _cute_rank1_load_u32x4_if_valid(
        split_t1_state_pointer,
        split_t1_state_index,
        cutlass.Int64({plan.state_size}),
    )
    for split_t1_pair in cutlass.range_constexpr(4):
        split_t1_state_bits[split_t1_row, split_t1_pair] = (
            split_t1_state_vector[split_t1_pair]
        )

split_t1_decay_parameter = cute.exp(
    split_t1_a_log, fastmath=True
)
split_t1_decay_source_packed = cute.make_rmem_tensor(2, cutlass.Uint32)
split_t1_query_sum_lane = cutlass.Float32(
    {epsilon}
) * cutlass.Float32(
    {1.0 / _THREADS!r}
)
split_t1_key_sum_lane = cutlass.Float32(
    {epsilon}
) * cutlass.Float32(
    {1.0 / _THREADS!r}
)
for split_t1_source_pair in cutlass.range_constexpr(2):
    split_t1_low = split_t1_source_pair * 2
    split_t1_query_low = cutlass.Float32(
        cutlass.Uint16(
            split_t1_query_raw[split_t1_source_pair]
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_query_high = cutlass.Float32(
        cutlass.Uint16(
            split_t1_query_raw[split_t1_source_pair] >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_key_low = cutlass.Float32(
        cutlass.Uint16(
            split_t1_key_raw[split_t1_source_pair]
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_key_high = cutlass.Float32(
        cutlass.Uint16(
            split_t1_key_raw[split_t1_source_pair] >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_gate_low = cutlass.Float32(
        cutlass.Uint16(
            split_t1_gate_raw[split_t1_source_pair]
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_gate_high = cutlass.Float32(
        cutlass.Uint16(
            split_t1_gate_raw[split_t1_source_pair] >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_gate_input_low = (
        split_t1_gate_low
        + cutlass.Float32(split_t1_bias_raw[split_t1_low])
    )
    split_t1_gate_input_high = (
        split_t1_gate_high
        + cutlass.Float32(split_t1_bias_raw[split_t1_low + 1])
    )
    split_t1_gate_exp_low = cute.exp(split_t1_gate_input_low, fastmath=True)
    split_t1_gate_exp_high = cute.exp(split_t1_gate_input_high, fastmath=True)
    split_t1_softplus_log2_low = cute.log2(
        cutlass.Float32(1.0) + split_t1_gate_exp_low,
        fastmath=True,
    )
    split_t1_softplus_log2_high = cute.log2(
        cutlass.Float32(1.0) + split_t1_gate_exp_high,
        fastmath=True,
    )
    if split_t1_gate_input_low > {threshold!r}:
        split_t1_softplus_log2_low = (
            split_t1_gate_input_low * cutlass.Float32({log2_e})
        )
    if split_t1_gate_input_high > {threshold!r}:
        split_t1_softplus_log2_high = (
            split_t1_gate_input_high * cutlass.Float32({log2_e})
        )
    split_t1_decay_low = cute.exp2(
        -split_t1_decay_parameter * split_t1_softplus_log2_low,
        fastmath=True,
    )
    split_t1_decay_high = cute.exp2(
        -split_t1_decay_parameter * split_t1_softplus_log2_high,
        fastmath=True,
    )
    split_t1_decay_source_packed[split_t1_source_pair] = (
        _cute_rank1_pack_bf16x2(
            split_t1_decay_low,
            split_t1_decay_high,
        )
    )
    split_t1_query_sum_lane = (
        split_t1_query_sum_lane
        + split_t1_query_low * split_t1_query_low
        + split_t1_query_high * split_t1_query_high
    )
    split_t1_key_sum_lane = (
        split_t1_key_sum_lane
        + split_t1_key_low * split_t1_key_low
        + split_t1_key_high * split_t1_key_high
    )

split_t1_query_sum = cutlass.Float32(
    cute.arch.warp_reduction_sum(
        split_t1_query_sum_lane, threads_in_group=32
    )
)
split_t1_key_sum = cutlass.Float32(
    cute.arch.warp_reduction_sum(
        split_t1_key_sum_lane, threads_in_group=32
    )
)
split_t1_query_scale = (
    cute.rsqrt(split_t1_query_sum, fastmath=True)
    * cutlass.Float32({scale})
)
split_t1_key_scale = cute.rsqrt(split_t1_key_sum, fastmath=True)
split_t1_query_scale_packed = _cute_rank1_pack_bf16x2(
    split_t1_query_scale, split_t1_query_scale
)
split_t1_key_scale_packed = _cute_rank1_pack_bf16x2(
    split_t1_key_scale, split_t1_key_scale
)
split_t1_query_source_packed = cute.make_rmem_tensor(2, cutlass.Uint32)
split_t1_key_source_packed = cute.make_rmem_tensor(2, cutlass.Uint32)
for split_t1_source_pair in cutlass.range_constexpr(2):
    split_t1_query_source_packed[split_t1_source_pair] = (
        _cute_rank1_mul_bf16x2(
            split_t1_query_raw[split_t1_source_pair],
            split_t1_query_scale_packed,
        )
    )
    split_t1_key_source_packed[split_t1_source_pair] = (
        _cute_rank1_mul_bf16x2(
            split_t1_key_raw[split_t1_source_pair],
            split_t1_key_scale_packed,
        )
    )

for split_t1_pair in cutlass.range_constexpr(4):
    split_t1_source_lane = (
        split_t1_key_lane * cutlass.Int32({_V_LANES})
        + split_t1_pair // 2
    )
    split_t1_source_pair = split_t1_pair % 2
    split_t1_query_cache[split_t1_pair] = cutlass.Uint32(
        cute.arch.shuffle_sync(
            split_t1_query_source_packed[split_t1_source_pair],
            split_t1_source_lane,
            mask=0xFFFFFFFF,
        )
    )
    split_t1_key_cache[split_t1_pair] = cutlass.Uint32(
        cute.arch.shuffle_sync(
            split_t1_key_source_packed[split_t1_source_pair],
            split_t1_source_lane,
            mask=0xFFFFFFFF,
        )
    )
    split_t1_decay_cache[split_t1_pair] = cutlass.Uint32(
        cute.arch.shuffle_sync(
            split_t1_decay_source_packed[split_t1_source_pair],
            split_t1_source_lane,
            mask=0xFFFFFFFF,
        )
    )

split_t1_beta_raw = cutlass.Float32(
    _cute_rank1_load_u16_nc(
        {beta}.iterator
        + split_t1_sequence * cutlass.Int32({plan.num_heads})
        + split_t1_head
    ).bitcast(cutlass.BFloat16)
)
split_t1_beta = cutlass.Float32(1.0) / (
    cutlass.Float32(1.0)
    + cute.exp(-split_t1_beta_raw, fastmath=True)
)
split_t1_value_loaded = cutlass.Float32(0.0)
if split_t1_thread < cutlass.Int32({_VALUE_TILE}):
    split_t1_value_loaded = cutlass.Float32(
        _cute_rank1_load_u16_nc(
            {v}.iterator
            + split_t1_value_head_offset
            + split_t1_value_offset
            + split_t1_thread
        ).bitcast(cutlass.BFloat16)
    )

for split_t1_row in cutlass.range_constexpr(8):
    split_t1_prediction_pair = cutlass.Uint32(0)
    for split_t1_pair in cutlass.range_constexpr(4):
        split_t1_decayed_pair = _cute_rank1_mul_bf16x2(
            split_t1_state_bits[split_t1_row, split_t1_pair],
            split_t1_decay_cache[split_t1_pair],
        )
        split_t1_decayed_cache[split_t1_pair] = split_t1_decayed_pair
        split_t1_prediction_pair = _cute_rank1_fma_bf16x2(
            split_t1_decayed_pair,
            split_t1_key_cache[split_t1_pair],
            split_t1_prediction_pair,
        )
    for split_t1_step in cutlass.range_constexpr(4):
        split_t1_prediction_other = cutlass.Uint32(
            cute.arch.shuffle_sync_bfly(
                split_t1_prediction_pair,
                cutlass.Int32(8 >> split_t1_step),
                mask=0xFFFFFFFF,
            )
        )
        split_t1_prediction_pair = _cute_rank1_add_bf16x2(
            split_t1_prediction_pair,
            split_t1_prediction_other,
        )
    split_t1_prediction = cutlass.Float32(
        cutlass.Uint16(split_t1_prediction_pair).bitcast(cutlass.BFloat16)
    ) + cutlass.Float32(
        cutlass.Uint16(
            split_t1_prediction_pair >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    split_t1_value_index = (
        split_t1_value_offset
        + split_t1_value_lane
        + cutlass.Int32({_V_LANES}) * split_t1_row
    )
    split_t1_value = cutlass.Float32(
        cute.arch.shuffle_sync(
            split_t1_value_loaded,
            split_t1_value_lane + cutlass.Int32({_V_LANES}) * split_t1_row,
            mask=0xFFFFFFFF,
        )
    )
    split_t1_delta = (
        split_t1_value - split_t1_prediction
    ) * split_t1_beta
    split_t1_delta_packed = _cute_rank1_pack_bf16x2(
        split_t1_delta, split_t1_delta
    )
    split_t1_output_pair = cutlass.Uint32(0)
    for split_t1_pair in cutlass.range_constexpr(4):
        split_t1_updated_pair = _cute_rank1_fma_bf16x2(
            split_t1_key_cache[split_t1_pair],
            split_t1_delta_packed,
            split_t1_decayed_cache[split_t1_pair],
        )
        split_t1_state_bits[split_t1_row, split_t1_pair] = (
            split_t1_updated_pair
        )
        split_t1_output_pair = _cute_rank1_fma_bf16x2(
            split_t1_updated_pair,
            split_t1_query_cache[split_t1_pair],
            split_t1_output_pair,
        )
    for split_t1_step in cutlass.range_constexpr(4):
        split_t1_output_other = cutlass.Uint32(
            cute.arch.shuffle_sync_bfly(
                split_t1_output_pair,
                cutlass.Int32(8 >> split_t1_step),
                mask=0xFFFFFFFF,
            )
        )
        split_t1_output_pair = _cute_rank1_add_bf16x2(
            split_t1_output_pair,
            split_t1_output_other,
        )
    split_t1_output_pair = _cute_rank1_add_bf16x2(
        split_t1_output_pair,
        split_t1_output_pair >> cutlass.Uint32(16),
    )
    # The matched graph has an unconditional output store.  An invalid state
    # slot supplies zero state above, but does not suppress or zero the output.
    if split_t1_key_lane == split_t1_row:
        split_t1_output_element_offset = (
            split_t1_value_head_offset + split_t1_value_index
        )
        ({output}.iterator + cutlass.Uint32(
            split_t1_output_element_offset
        )).store(
            cutlass.Uint16(split_t1_output_pair).bitcast(cutlass.BFloat16)
        )

for split_t1_row in cutlass.range_constexpr(8):
    split_t1_value_index = (
        split_t1_value_offset
        + split_t1_value_lane
        + cutlass.Int32({_V_LANES}) * split_t1_row
    )
    split_t1_state_element_offset = (
        cutlass.Uint32(split_t1_state_index)
        * cutlass.Uint32({plan.state_stride0})
        + cutlass.Uint32(split_t1_head * cutlass.Int32({_VALUE_SIZE * _KEY_SIZE}))
        + cutlass.Uint32(
            split_t1_value_index * cutlass.Int32({_KEY_SIZE})
            + split_t1_key_lane * cutlass.Int32(8)
        )
    )
    split_t1_state_pointer = cute.recast_ptr(
        {state}.iterator + split_t1_state_element_offset,
        dtype=cutlass.Uint32,
    )
    _cute_rank1_store_u32x4_if_valid(
        split_t1_state_pointer,
        split_t1_state_bits[split_t1_row, 0],
        split_t1_state_bits[split_t1_row, 1],
        split_t1_state_bits[split_t1_row, 2],
        split_t1_state_bits[split_t1_row, 3],
        split_t1_state_index,
        cutlass.Int64({plan.state_size}),
    )
"""
    df.preamble = cast(
        "list[ast.AST]",
        ast.parse(textwrap.dedent(template.render(preamble))).body,
    )
    df.body = cast(
        "list[ast.AST]",
        ast.parse(textwrap.dedent(template.render(body))).body,
    )
    df.placeholder_args.update(placeholder_args)
    return True
