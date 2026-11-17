"""FX analysis shared by the three rank-1 recurrence emitters.

Role: analysis.  The single-token, fixed-token and split-T1 rank-1 emitters
(``*_rank1_recurrence.py``) recognise the same KDA recurrent update over
BF16 Q/K/V/gate tensors; this module holds the exact-match predicates they
share (cast stripping, binary-operand pairing, broadcast and reduction
sources, the bounded decay and factored-output graphs) and the one step
matcher, ``match_rank1_step``.  The step matcher proves the recurrent state
update, the residual, the normalized key, the decay and the projection, and
takes the three sub-matchers that differ between the fixed-token and split-T1
graphs as arguments: how the beta scalar is loaded, which decay graph is
admitted, and how the output projection is formed.  Everything here runs at
match time only; renders never change when this module does.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...fx_side_effects import is_impure_node
from ..fx_matcher import _is_call
from ..fx_matcher import _load_ref
from ..fx_matcher import _memory_indices

if TYPE_CHECKING:
    from collections.abc import Sequence


_KEY_SIZE = 128

# The rank-1 modules' historical name for the index list of a load or store.
_indices = _memory_indices


def _has_unmatched_side_effect(
    nodes: Sequence[torch.fx.Node], allowed: set[torch.fx.Node]
) -> bool:
    return any(
        node.op == "call_function" and is_impure_node(node) and node not in allowed
        for node in nodes
    )


def _strip_cast_chain(
    node: object,
    accepts_cast: Callable[[torch.fx.Node, torch.fx.Node], bool],
) -> torch.fx.Node | None:
    """Walk through clones, detaches and the element-type casts ``accepts_cast`` approves.

    Returns the first node that is not such a cast, or ``None`` when ``node`` is
    not an fx ``Node``, for a malformed or rejected cast (``accepts_cast``
    receives the cast node and its source), or after eight hops.
    """

    if not isinstance(node, torch.fx.Node):
        return None
    current = node
    for _ in range(8):
        if current.op != "call_function":
            return current
        if current.target in (
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
        ):
            if (
                len(current.args) != 1
                or current.kwargs
                or not isinstance(current.args[0], torch.fx.Node)
            ):
                return None
            current = current.args[0]
            continue
        if current.target is not torch.ops.prims.convert_element_type.default:
            return current
        if (
            len(current.args) != 2
            or current.kwargs
            or not isinstance(current.args[0], torch.fx.Node)
        ):
            return None
        source = current.args[0]
        if not accepts_cast(current, source):
            return None
        current = source
    return None


def _is_bf16_fp32_cast(cast_node: torch.fx.Node, source: torch.fx.Node) -> bool:
    source_value = source.meta.get("val")
    output_value = cast_node.meta.get("val")
    return (
        isinstance(source_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and (source_value.dtype, output_value.dtype)
        in (
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
        )
    )


def _is_state_index_widening(cast_node: torch.fx.Node, source: torch.fx.Node) -> bool:
    """The exact signed-int widening used by ``Tensor.long``."""

    source_value = source.meta.get("val")
    return (
        cast_node.args[1] is torch.int64
        and isinstance(source_value, torch.Tensor)
        and source_value.dtype in (torch.int32, torch.int64)
    )


def _strip_casts(node: object) -> torch.fx.Node | None:
    return _strip_cast_chain(node, _is_bf16_fp32_cast)


def _strip_state_index_casts(node: object) -> torch.fx.Node | None:
    return _strip_cast_chain(node, _is_state_index_widening)


def _cast_from(node: object, source: torch.fx.Node, dtype: torch.dtype) -> bool:
    return bool(
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.prims.convert_element_type.default
        and len(node.args) >= 2
        and node.args[0] is source
        and node.args[1] is dtype
        and not node.kwargs
    )


_ADD_TARGETS: set[object] = {operator.add, torch.ops.aten.add.Tensor}


_MUL_TARGETS: set[object] = {operator.mul, torch.ops.aten.mul.Tensor}


def _binary_other(
    node: object,
    known: torch.fx.Node,
    targets: set[object],
) -> torch.fx.Node | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) < 2
        or node.kwargs
    ):
        return None
    left, right = node.args[:2]
    if left is known and isinstance(right, torch.fx.Node):
        return right
    if right is known and isinstance(left, torch.fx.Node):
        return left
    return None


def _binary_pair(
    node: object,
    targets: set[object],
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) < 2
        or node.kwargs
        or not isinstance(node.args[0], torch.fx.Node)
        or not isinstance(node.args[1], torch.fx.Node)
    ):
        return None
    return node.args[0], node.args[1]


def _broadcast_source(
    node: object,
    indices: tuple[object, ...],
) -> torch.fx.Node | None:
    from ....language import view_ops

    current = _strip_casts(node)
    if (
        current is None
        or current.op != "call_function"
        or current.target is not view_ops.subscript
        or len(current.args) < 2
        or not isinstance(current.args[0], torch.fx.Node)
        or not isinstance(current.args[1], (list, tuple))
        or len(current.args[1]) != len(indices)
    ):
        return None
    for actual, expected in zip(current.args[1], indices, strict=True):
        if expected == slice(None):
            if actual != slice(None):
                return None
        elif actual is not expected:
            return None
    return _strip_casts(current.args[0])


def _sum_input(
    node: object,
    dimensions: tuple[int, ...] | None,
) -> torch.fx.Node | None:
    from ....language._tracing_ops import _mask_to

    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or not node.args
        or not isinstance(node.args[0], torch.fx.Node)
        or node.kwargs
    ):
        return None
    if dimensions is None:
        if node.target is not torch.ops.aten.sum.default or len(node.args) != 1:
            return None
    elif (
        node.target is not torch.ops.aten.sum.dim_IntList
        or len(node.args) != 2
        or not isinstance(node.args[1], (list, tuple))
        or tuple(node.args[1]) != dimensions
    ):
        return None
    value = node.args[0]
    if _is_call(value, _mask_to):
        if len(value.args) != 2 or value.args[1] != 0 or value.kwargs:
            return None
        value = value.args[0]
    return _strip_casts(value)


def _affine(
    value: object,
) -> tuple[int, dict[torch.fx.Node, int]] | None:
    """Parse integer add/constant-multiply index expressions."""

    if isinstance(value, int):
        return value, {}
    if not isinstance(value, torch.fx.Node):
        return None
    if value.op != "call_function":
        return 0, {value: 1}
    if value.kwargs:
        return None
    if value.target in _ADD_TARGETS and len(value.args) >= 2:
        left = _affine(value.args[0])
        right = _affine(value.args[1])
        if left is None or right is None:
            return None
        constant = left[0] + right[0]
        terms = dict(left[1])
        for leaf, coefficient in right[1].items():
            terms[leaf] = terms.get(leaf, 0) + coefficient
        return constant, {leaf: value for leaf, value in terms.items() if value}
    if value.target in _MUL_TARGETS and len(value.args) >= 2:
        left, right = value.args[:2]
        if isinstance(left, int):
            parsed = _affine(right)
            scale = left
        elif isinstance(right, int):
            parsed = _affine(left)
            scale = right
        else:
            return 0, {value: 1}
        if parsed is None:
            return None
        return (
            parsed[0] * scale,
            {leaf: coefficient * scale for leaf, coefficient in parsed[1].items()},
        )
    return 0, {value: 1}


def _is_standard_key_iota(node: object) -> bool:
    if (
        not _is_call(node, torch.ops.prims.iota.default)
        or not isinstance(node, torch.fx.Node)
        or node.args != (_KEY_SIZE,)
        or set(node.kwargs) != {"start", "step", "dtype", "device", "requires_grad"}
        or node.kwargs["start"] != 0
        or node.kwargs["step"] != 1
        or node.kwargs["dtype"] not in (torch.int32, torch.int64)
        or node.kwargs["requires_grad"] is not False
    ):
        return False
    device = node.kwargs["device"]
    return isinstance(device, torch.device) and device.type == "cuda"


def _match_decay(
    node: torch.fx.Node,
) -> (
    tuple[
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        object,
        object,
    ]
    | None
):
    """Match the bounded sigmoid decay expression."""

    if node.op != "call_function" or node.target is not torch.ops.aten.exp2.default:
        return None
    outer_mul = _binary_pair(node.args[0] if node.args else None, _MUL_TARGETS)
    if outer_mul is None:
        return None
    for log_decay, log2_e in (outer_mul, reversed(outer_mul)):
        log_pair = _binary_pair(log_decay, _MUL_TARGETS)
        if log_pair is None:
            continue
        for sigmoid, lower_bound in (log_pair, reversed(log_pair)):
            if (
                sigmoid.op != "call_function"
                or sigmoid.target is not torch.ops.aten.sigmoid.default
                or not sigmoid.args
                or not isinstance(sigmoid.args[0], torch.fx.Node)
            ):
                continue
            gate_product = _binary_pair(sigmoid.args[0], _MUL_TARGETS)
            if gate_product is None:
                continue
            for a_value, gate_plus_bias in (gate_product, reversed(gate_product)):
                if (
                    a_value.op != "call_function"
                    or a_value.target is not torch.ops.aten.exp2.default
                    or not a_value.args
                ):
                    continue
                a_mul = _binary_pair(a_value.args[0], _MUL_TARGETS)
                gate_pair = _binary_pair(gate_plus_bias, _ADD_TARGETS)
                if a_mul is None or gate_pair is None:
                    continue
                a_log: torch.fx.Node | None = None
                for maybe_a_log, maybe_log2_e in (a_mul, reversed(a_mul)):
                    if maybe_log2_e is log2_e:
                        a_log = maybe_a_log
                        break
                if a_log is None:
                    continue
                gate, bias = gate_pair
                return (
                    gate,
                    bias,
                    a_log,
                    lower_bound.meta.get("val"),
                    log2_e.meta.get("val"),
                )
    return None


def _match_factored_output(
    output: torch.fx.Node,
    decayed: torch.fx.Node,
    residual: torch.fx.Node,
    key_broadcast: torch.fx.Node,
) -> torch.fx.Node | None:
    """Return the scaled normalized-Q node from an equivalent output graph."""

    outer = _binary_pair(output, _ADD_TARGETS)
    if outer is None:
        return None
    for base_sum, scaled_sum in (outer, reversed(outer)):
        base_product = _sum_input(base_sum, (1,))
        scaled_product = _sum_input(scaled_sum, (1,))
        if base_product is None or scaled_product is None:
            continue
        base_pair = _binary_pair(base_product, _MUL_TARGETS)
        scaled_pair = _binary_pair(scaled_product, _MUL_TARGETS)
        if base_pair is None or scaled_pair is None:
            continue
        q_broadcast: torch.fx.Node | None = None
        for maybe_decayed, maybe_q in (base_pair, reversed(base_pair)):
            if maybe_decayed is decayed:
                q_broadcast = maybe_q
                break
        if q_broadcast is None:
            continue
        q = _broadcast_source(q_broadcast, (None, slice(None)))
        if q is None:
            continue
        dot_sum_broadcast: torch.fx.Node | None = None
        for maybe_residual, maybe_dot in (scaled_pair, reversed(scaled_pair)):
            if _broadcast_source(maybe_residual, (slice(None), None)) is residual:
                dot_sum_broadcast = maybe_dot
                break
        if dot_sum_broadcast is None:
            continue
        dot_sum = _broadcast_source(dot_sum_broadcast, (slice(None), None))
        if dot_sum is None:
            continue
        dot_input = _sum_input(dot_sum, (1,))
        if dot_input is None:
            continue
        if (
            dot_input.op == "call_function"
            and dot_input.target is torch.ops.aten.expand.default
            and len(dot_input.args) == 2
            and isinstance(dot_input.args[0], torch.fx.Node)
            and isinstance(dot_input.args[1], (list, tuple))
            and len(dot_input.args[1]) == 2
            and dot_input.args[1][1] == _KEY_SIZE
            and not dot_input.kwargs
        ):
            dot_input = dot_input.args[0]
        dot_pair = _binary_pair(dot_input, _MUL_TARGETS)
        if dot_pair is None:
            continue
        if any(
            maybe_key is key_broadcast and maybe_q is q_broadcast
            for maybe_key, maybe_q in (dot_pair, reversed(dot_pair))
        ):
            return q
    return None


def _is_direct_float_load(node: object) -> torch.fx.Node | None:
    """Return the BF16 load underlying one exact BF16-to-FP32 cast."""

    load = _strip_casts(node)
    if (
        not isinstance(node, torch.fx.Node)
        or not isinstance(load, torch.fx.Node)
        or _load_ref(load) is None
        or not _cast_from(node, load, torch.float32)
    ):
        return None
    return load


def _match_normalized_vector(
    node: object,
) -> tuple[torch.fx.Node, torch.fx.Node, object] | None:
    """Match L2 normalization and return normalized/raw/epsilon.

    Fixed-token kernels receive the epsilon through a module attribute, so it
    appears as an FX scalar node rather than a Python literal.  Capture that
    scalar and later promote it to a CuTe constexpr instead of keying on its
    source name.
    """

    pair = _binary_pair(node, _MUL_TARGETS)
    if pair is None:
        return None
    for raw, scale in (pair, reversed(pair)):
        if (
            scale.op != "call_function"
            or scale.target is not torch.ops.aten.rsqrt.default
            or len(scale.args) != 1
            or scale.kwargs
            or not isinstance(scale.args[0], torch.fx.Node)
        ):
            continue
        norm = scale.args[0]
        if (
            norm.op != "call_function"
            or norm.target not in _ADD_TARGETS
            or len(norm.args) < 2
            or norm.kwargs
        ):
            continue
        norm_pair = norm.args[:2]
        for maybe_sum, epsilon_source in (norm_pair, reversed(norm_pair)):
            square = _sum_input(maybe_sum, None)
            if square is None or not _match_product(square, raw, raw):
                continue
            epsilon = (
                epsilon_source.meta.get("val")
                if isinstance(epsilon_source, torch.fx.Node)
                else epsilon_source
            )
            if isinstance(epsilon, (int, float, torch.SymFloat)):
                return cast("torch.fx.Node", node), raw, epsilon
    return None


def _match_product(
    node: object,
    first: torch.fx.Node,
    second: torch.fx.Node,
) -> bool:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in _MUL_TARGETS
        or len(node.args) < 2
        or node.kwargs
    ):
        return False
    return node.args[:2] == (first, second)


def _matches_bias_index(
    node: object,
    head_coord: torch.fx.Node,
    key_coord: torch.fx.Node,
) -> bool:
    """Match exactly ``head * 128 + key`` without parsing iota kwargs."""

    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in _ADD_TARGETS
        or len(node.args) < 2
        or node.kwargs
    ):
        return False
    for maybe_key, maybe_head_term in (
        node.args[:2],
        reversed(node.args[:2]),
    ):
        if maybe_key is not key_coord or not isinstance(maybe_head_term, torch.fx.Node):
            continue
        if (
            maybe_head_term.op == "call_function"
            and maybe_head_term.target in _MUL_TARGETS
            and len(maybe_head_term.args) >= 2
            and not maybe_head_term.kwargs
            and maybe_head_term.args[:2]
            in ((head_coord, _KEY_SIZE), (_KEY_SIZE, head_coord))
        ):
            return True
    return False


def _same_scalar(left: object, right: object) -> bool:
    if left is right:
        return True
    return type(left) in (int, float) and type(right) in (int, float) and left == right


@dataclasses.dataclass(frozen=True)
class Rank1DecayMatch:
    """The operands of an admitted decay graph ``exp2(log_decay * log2_e)``."""

    gate: torch.fx.Node
    bias: torch.fx.Node
    a_log: torch.fx.Node
    log2_e: object
    lower_bound: object = None
    softplus_threshold: object = None


@dataclasses.dataclass(frozen=True)
class Rank1UpdateMatch:
    """The matched ``decay * state + residual (x) normalized_key`` update."""

    state_update: torch.fx.Node
    decayed: torch.fx.Node
    decay: torch.fx.Node
    residual: torch.fx.Node
    normalized_key: torch.fx.Node
    key_broadcast: torch.fx.Node


@dataclasses.dataclass(frozen=True)
class Rank1StepMatch:
    """One complete recurrent step: its loads, stores, coordinates and scalars.

    ``lower_bound`` is set by the bounded-sigmoid decay family (fixed token)
    and ``softplus_threshold`` by the unbounded-softplus family (split T1).
    """

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
    lower_bound: object = None
    softplus_threshold: object = None

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


# The beta operand of the residual: returns the BF16 load it is formed from.
Rank1BetaMatcher = Callable[[object], "torch.fx.Node | None"]
# The decay operand of the state update.
Rank1DecayMatcher = Callable[[torch.fx.Node], "Rank1DecayMatch | None"]
# The stored output: returns ``(q_load, scale, epsilon)``.
Rank1ProjectionMatcher = Callable[
    [torch.fx.Node, Rank1UpdateMatch], "tuple[torch.fx.Node, object, object] | None"
]


def _match_state_update(
    state_store: torch.fx.Node,
    incoming_state: torch.fx.Node,
) -> Rank1UpdateMatch | None:
    """Match ``bf16(decay[None, :] * state + residual[:, None] * key[None, :])``."""

    stored_state = state_store.args[2] if len(state_store.args) > 2 else None
    state_update = _strip_casts(stored_state)
    update_pair = _binary_pair(state_update, _ADD_TARGETS)
    if (
        state_update is None
        or update_pair is None
        or not _cast_from(stored_state, state_update, torch.bfloat16)
    ):
        return None
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
            return Rank1UpdateMatch(
                state_update=state_update,
                decayed=maybe_decayed,
                decay=maybe_decay,
                residual=maybe_residual,
                normalized_key=maybe_key,
                key_broadcast=maybe_key_broadcast,
            )
    return None


def match_rank1_step(
    state_store: torch.fx.Node,
    output_store: torch.fx.Node,
    incoming_state: torch.fx.Node,
    *,
    beta: Rank1BetaMatcher,
    decay: Rank1DecayMatcher,
    projection: Rank1ProjectionMatcher,
) -> Rank1StepMatch | None:
    """Match one complete recurrent update, its projection and both stores.

    The topology (state update, residual ``beta * (v - sum(decayed * key))``,
    normalized key, decay operands, index contracts) is common to the
    fixed-token and split-T1 families; ``beta``, ``decay`` and ``projection``
    are the three sub-matchers in which they differ.  Fail closed: every load,
    cast and coordinate is proved before a match is returned.
    """

    update = _match_state_update(state_store, incoming_state)
    if update is None:
        return None
    state_update = update.state_update
    decayed = update.decayed
    residual = update.residual
    normalized_key = update.normalized_key

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
            matched_beta = beta(maybe_beta)
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

    decay_match = decay(update.decay)
    if decay_match is None:
        return None
    gate_load = _is_direct_float_load(decay_match.gate)
    bias_load = _strip_casts(decay_match.bias)
    a_log_load = _strip_casts(decay_match.a_log)
    if (
        gate_load is None
        or not isinstance(bias_load, torch.fx.Node)
        or not isinstance(a_log_load, torch.fx.Node)
        or _load_ref(bias_load) is None
        or _load_ref(a_log_load) is None
        or decay_match.bias is not bias_load
        or decay_match.a_log is not a_log_load
    ):
        return None

    projection_match = projection(output_store, update)
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

    return Rank1StepMatch(
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
        log2_e=decay_match.log2_e,
        epsilon=query_epsilon,
        lower_bound=decay_match.lower_bound,
        softplus_threshold=decay_match.softplus_threshold,
    )
