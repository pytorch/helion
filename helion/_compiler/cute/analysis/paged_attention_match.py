"""Semantic recognition for paged, block-scaled online attention.

This proof retains the source accesses and logical mask separately. A transport
planner must still validate its descriptor layouts, scalar expression support,
and tile sizes; matching does not select a device implementation.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import TypeGuard

import sympy
import torch

from ....language import matmul_ops
from ....language import tile_ops
from ....language import view_ops
from ....language._tracing_ops import _get_symnode
from ....language._tracing_ops import _mask_to
from ....language._tracing_ops import _new_var
from ...utils import is_full_slice
from .match_common import scalar_value
from .match_common import symnode_sympy

if TYPE_CHECKING:
    from torch.fx import Node

    from ..cute_mma_dag import Access
    from ..cute_mma_dag import Carry
    from ..cute_mma_dag import Loop
    from ..cute_mma_dag import MmaDagProgram
    from ..cute_mma_dag import VecCarry


@dataclasses.dataclass(frozen=True)
class BlockScaledOperand:
    """An exact ``(int8.float() * exp2(uint8.float() - 127)).to(dtype)``.

    Both accesses retain the runtime tensor name, physical strides,
    scalar page/head subscripts, and logical axes. Only the scale's final axis
    has a group divisor; neither access is replaced by a dense view. The named runtime tensor pointer
    includes its storage offset; the fake tensor's offset is normalized.
    """

    raw: Access
    scale: Access
    group_size: int
    value: Node


@dataclasses.dataclass(frozen=True)
class LogicalAttentionMask:
    predicate: Node
    key_position: Node
    query_position: Node
    window_left: int | None


@dataclasses.dataclass(frozen=True)
class ReflectedAttentionIteration:
    """A typed ``(begin + end) - 1 - iteration`` source index.

    The transport separately proves that its page lookup uses this same value.
    Integer additions/subtractions and the later key-width product retain dtype.
    """

    value: Node
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class AttentionCoordinates:
    """The proved logical positions, with each tensor arithmetic width.

    ``key = iota(width) + token * width`` uses either the original iteration
    or the separately proved typed reflection retained in ``reflected``;
    ``query = (lengths[request] - query_len) + row_index // group``.
    Tensor operations keep their recorded dtype: in particular an int32 length
    subtraction wraps before the final query addition, even in an int64 loop.
    Loop bounds remain the original FX expressions in ``PagedAttentionPlan.loop``.
    """

    width: int
    lengths: Access
    query_len: int
    group: int
    key_dtype: torch.dtype
    row_dtype: torch.dtype
    length_base_dtype: torch.dtype
    query_dtype: torch.dtype
    reflected: ReflectedAttentionIteration | None


@dataclasses.dataclass(frozen=True)
class PagedAttentionPlan:
    """Source semantics, before choosing QK/PV physical MMA orientations."""

    program: MmaDagProgram
    loop: Loop
    query: Access
    key: BlockScaledOperand
    value: BlockScaledOperand
    page_table: Access
    page_symbol: sympy.Symbol
    page_dimension: int
    mask: LogicalAttentionMask
    coordinates: AttentionCoordinates
    score_scale: float
    score: Node
    probability: Node
    maximum: VecCarry
    denominator: VecCarry | Carry
    accumulator: Carry
    output: Access
    lse: Access | None
    # A matrix recurrence delays the key-lane sum until after the loop.
    # Keep its distinct rounding order instead of replacing it with a
    # per-iteration reduced vector recurrence.
    denominator_per_key: bool
    denominator_final: Node
    # A separate source recurrence centers QK at a nonnegative finite old max,
    # then reuses those relative scores. Its FP32 rounding differs from the
    # ordinary recurrence and must never be selected for an ordinary source.
    score_reference: Node | None
    # Explicit exp2 probability/alpha and a natural-log LSE conversion.
    # This is a source recurrence choice, never an exp->exp2 rewrite.
    score_base2: bool

    @property
    def io_dtype(self) -> torch.dtype:
        return self.query.dtype


def _call(value: object, target: object) -> TypeGuard[Node]:
    if not (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target is target
    ):
        return False
    if target in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor):
        alpha = value.kwargs.get("alpha", 1)
        return isinstance(alpha, (int, float)) and alpha == 1
    return True


def _copy(value: object) -> object:
    while _call(value, _new_var):
        assert isinstance(value, torch.fx.Node)
        value = value.args[0]
    return value


def _zero_mask(value: object) -> object:
    value = _copy(value)
    if _call(value, _mask_to):
        assert isinstance(value, torch.fx.Node)
        if len(value.args) >= 2 and value.args[1] == 0:
            return _copy(value.args[0])
    return value


def _other(value: object, target: object, source: object) -> object:
    if not _call(value, target):
        return None
    assert isinstance(value, torch.fx.Node)
    if len(value.args) != 2:
        return None
    left, right = value.args
    if _copy(left) is _copy(source):
        return right
    return left if _copy(right) is _copy(source) else None


def _scalar(value: object, expected: float) -> bool:
    if _call(value, torch.ops.aten.scalar_tensor.default):
        tensor = value.meta.get("val")
        # The constants read here (-inf, 0, 1, 127) are exact in these
        # floating dtypes. Integer/bool/FP8 constructors can change them.
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return False
    return scalar_value(value) == expected


def _cast(value: object, dtype: torch.dtype) -> object:
    if _call(value, torch.ops.prims.convert_element_type.default):
        assert isinstance(value, torch.fx.Node)
        if len(value.args) == 2 and value.args[1] == dtype:
            return value.args[0]
    return None


def _broadcast(value: object, *, row: bool) -> object:
    if not _call(value, view_ops.subscript):
        return None
    assert isinstance(value, torch.fx.Node)
    if len(value.args) != 2:
        return None
    indices = value.args[1]
    if not isinstance(indices, (tuple, list)) or len(indices) != 2:
        return None
    full, added = (indices[0], indices[1]) if row else (indices[1], indices[0])
    return _copy(value.args[0]) if is_full_slice(full) and added is None else None


def _row_reduction(value: object, target: object, source: object, fill: float) -> bool:
    if not _call(value, target):
        return False
    assert isinstance(value, torch.fx.Node)
    if len(value.args) not in (2, 3) or value.args[1] not in ([1], [-1]):
        return False
    if len(value.args) == 3 and value.args[2] is not False:
        return False
    masked = value.args[0]
    return (
        _call(masked, _mask_to)
        and len(masked.args) >= 2
        and masked.args[0] is source
        and _scalar(masked.args[1], fill)
    )


def _safe_value(value: object, source: object, empty: float, substitute: float) -> bool:
    if not _call(value, torch.ops.aten.where.self):
        return False
    assert isinstance(value, torch.fx.Node)
    if len(value.args) != 3:
        return False
    condition, replacement, unchanged = value.args
    return (
        _copy(unchanged) is _copy(source)
        and _scalar(replacement, substitute)
        and _call(condition, torch.ops.aten.eq.Scalar)
        and len(condition.args) == 2
        and _copy(condition.args[0]) is _copy(source)
        and condition.args[1:] == (empty,)
    )


def _masked_scores(
    value: object, qk: Node
) -> tuple[Node, float, LogicalAttentionMask, Node | None] | None:
    if not _call(value, torch.ops.aten.where.self):
        return None
    assert isinstance(value, torch.fx.Node)
    if len(value.args) != 3 or not _scalar(value.args[2], -math.inf):
        return None
    predicate, scaled, _ = value.args
    reference = None
    if _call(scaled, torch.ops.aten.sub.Tensor) and len(scaled.args) == 2:
        reference = _broadcast(scaled.args[1], row=True)
        if not isinstance(reference, torch.fx.Node):
            return None
        scaled = scaled.args[0]
    scale = _other(scaled, torch.ops.aten.mul.Tensor, qk)
    # Tensor constants have already rounded through their constructor dtype;
    # leave those recipes on the ordinary route rather than reusing a literal.
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        return None
    causal = predicate
    lower = None
    if _call(predicate, torch.ops.aten.bitwise_and.Tensor):
        if len(predicate.args) != 2:
            return None
        causal, lower = predicate.args
        if _call(lower, torch.ops.aten.le.Tensor):
            causal, lower = lower, causal
    if not _call(causal, torch.ops.aten.le.Tensor) or len(causal.args) != 2:
        return None
    key = _broadcast(causal.args[0], row=False)
    query = _broadcast(causal.args[1], row=True)
    if not isinstance(key, torch.fx.Node) or not isinstance(query, torch.fx.Node):
        return None
    window = None
    if lower is not None:
        if not _call(lower, torch.ops.aten.ge.Tensor) or len(lower.args) != 2:
            return None
        if _broadcast(lower.args[0], row=False) is not key:
            return None
        bound = lower.args[1]
        if not _call(bound, torch.ops.aten.sub.Tensor) or len(bound.args) != 2:
            return None
        if _broadcast(bound.args[0], row=True) is not query:
            return None
        window = bound.args[1]
        if type(window) is not int or window < 0:
            return None
    assert isinstance(predicate, torch.fx.Node)
    return (
        value,
        float(scale),
        LogicalAttentionMask(predicate, key, query, window),
        reference,
    )


def _nonnegative_reference(value: Node, maximum: Node) -> bool:
    """Prove where((maximum >= 0) & (maximum < inf), maximum, 0)."""
    if not _call(value, torch.ops.aten.where.self) or len(value.args) != 3:
        return False
    condition, unchanged, zero = value.args
    if _copy(unchanged) is not maximum or not _scalar(zero, 0):
        return False
    if (
        not _call(condition, torch.ops.aten.bitwise_and.Tensor)
        or len(condition.args) != 2
    ):
        return False
    lower, upper = condition.args
    if _call(lower, torch.ops.aten.lt.Scalar):
        lower, upper = upper, lower
    return (
        _call(lower, torch.ops.aten.ge.Scalar)
        and len(lower.args) == 2
        and _copy(lower.args[0]) is maximum
        and _scalar(lower.args[1], 0)
        and _call(upper, torch.ops.aten.lt.Scalar)
        and len(upper.args) == 2
        and _copy(upper.args[0]) is maximum
        and _scalar(upper.args[1], math.inf)
    )


def _block_scaled_operand(
    value: object, accesses: dict[Node, Access], dtype: torch.dtype
) -> BlockScaledOperand | None:
    value = _zero_mask(value)
    product = _cast(value, dtype)
    if not _call(product, torch.ops.aten.mul.Tensor) or len(product.args) != 2:
        return None
    for integer_cast, exponent in (product.args, tuple(reversed(product.args))):
        raw = _cast(integer_cast, torch.float32)
        if (
            not isinstance(raw, torch.fx.Node)
            or raw not in accesses
            or accesses[raw].dtype != torch.int8
        ):
            continue
        if not _call(exponent, torch.ops.aten.exp2.default):
            continue
        bias = exponent.args[0]
        if not _call(bias, torch.ops.aten.sub.Tensor) or len(bias.args) != 2:
            continue
        if not _scalar(bias.args[1], 127):
            continue
        scale = _cast(bias.args[0], torch.float32)
        if (
            not isinstance(scale, torch.fx.Node)
            or scale not in accesses
            or accesses[scale].dtype != torch.uint8
        ):
            continue
        raw_access, scale_access = accesses[raw], accesses[scale]
        if raw_access.grouped or len(raw_access.subscripts) != len(
            scale_access.subscripts
        ):
            continue
        divisors = [s.divisor for s in scale_access.subscripts if s.divisor is not None]
        if len(divisors) != 1 or raw_access.dims != scale_access.dims:
            continue
        group = divisors[0]
        assert group is not None
        if group <= 0 or scale_access.subscripts[-1].divisor != group:
            continue
        if any(
            left != dataclasses.replace(right, divisor=None)
            for left, right in zip(
                raw_access.subscripts, scale_access.subscripts, strict=True
            )
        ):
            continue
        assert isinstance(value, torch.fx.Node)
        return BlockScaledOperand(raw_access, scale_access, group, value)
    return None


def _integer_dtype(node: object) -> torch.dtype | None:
    if isinstance(node, torch.fx.Node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and value.dtype in (
            torch.int32,
            torch.int64,
        ):
            return value.dtype
    return None


def _unit_product(program: MmaDagProgram, value: object) -> object:
    if isinstance(value, torch.fx.Node):
        value = program.resolve(value)
    if _call(value, torch.ops.aten.mul.Tensor) and len(value.args) == 2:
        child, factor = value.args
        if type(child) is int:
            child, factor = factor, child
        if (
            type(factor) is int
            and factor == 1
            and _integer_dtype(child) == _integer_dtype(value)
        ):
            return program.resolve(child) if isinstance(child, torch.fx.Node) else child
    return value


def _reflected_iteration(
    program: MmaDagProgram, loop: Loop, value: object
) -> ReflectedAttentionIteration | None:
    """Prove reflection without rewriting typed integer expressions as algebra."""
    if not _call(value, torch.ops.aten.sub.Tensor) or len(value.args) != 2:
        return None
    prefix, iteration = value.args
    dtype = _integer_dtype(value)
    if (
        dtype is None
        or not _call(iteration, _get_symnode)
        or symnode_sympy(iteration) != loop.symbol
    ):
        return None
    if not _call(prefix, torch.ops.aten.sub.Tensor) or len(prefix.args) != 2:
        return None
    total, one = prefix.args
    if type(one) is not int or one != 1 or _integer_dtype(prefix) != dtype:
        return None
    if _integer_dtype(total) != dtype:
        return None
    total = _unit_product(program, total)
    if not _call(total, torch.ops.aten.add.Tensor) or len(total.args) != 2:
        return None
    begin, end = (_unit_product(program, arg) for arg in total.args)
    expected_begin = _unit_product(program, loop.begin)
    expected_end = _unit_product(program, loop.end)
    if not (
        (begin is expected_begin and end is expected_end)
        or (begin is expected_end and end is expected_begin)
    ):
        return None
    if any(
        _integer_dtype(node) != dtype
        for node in (begin, end, total, loop.begin, loop.end)
    ):
        return None
    return ReflectedAttentionIteration(value, dtype)


def _coordinate_contract(
    program: MmaDagProgram,
    query: Access,
    key: BlockScaledOperand,
    loop: Loop,
    mask: LogicalAttentionMask,
    accesses: dict[Node, Access],
) -> AttentionCoordinates | None:
    key_position = mask.key_position
    if (
        not _call(key_position, torch.ops.aten.add.Tensor)
        or len(key_position.args) != 2
    ):
        return None
    iota, tile_offset = key_position.args
    if _call(tile_offset, torch.ops.prims.iota.default):
        iota, tile_offset = tile_offset, iota
    if not _call(iota, torch.ops.prims.iota.default) or len(iota.args) != 1:
        return None
    width = iota.args[0]
    if type(width) is not int or width != key.raw.tile_dims[0].extent:
        return None
    if iota.kwargs.get("start", 0) != 0 or iota.kwargs.get("step", 1) != 1:
        return None
    symbolic_product = _call(tile_offset, operator.mul)
    if (
        not symbolic_product and not _call(tile_offset, torch.ops.aten.mul.Tensor)
    ) or len(tile_offset.args) != 2:
        return None
    iteration, multiplier = tile_offset.args
    if type(iteration) is int:
        iteration, multiplier = multiplier, iteration
    if type(multiplier) is not int or multiplier != width:
        return None
    reflected = None
    if symbolic_product:
        if (
            not _call(iteration, _get_symnode)
            or symnode_sympy(iteration) != loop.symbol
        ):
            return None
    else:
        reflected = _reflected_iteration(program, loop, iteration)
        if reflected is None or _integer_dtype(tile_offset) != reflected.dtype:
            return None
    position = program.resolve(mask.query_position)
    if not _call(position, torch.ops.aten.add.Tensor) or len(position.args) != 2:
        return None
    base, grouped_row = position.args
    if _call(base, torch.ops.aten.div.Tensor_mode):
        base, grouped_row = grouped_row, base
    if (
        not _call(grouped_row, torch.ops.aten.div.Tensor_mode)
        or len(grouped_row.args) != 2
    ):
        return None
    row, group = grouped_row.args
    if (
        grouped_row.kwargs != {"rounding_mode": "floor"}
        or type(group) is not int
        or group <= 0
    ):
        return None
    if not _call(row, tile_ops.tile_index) or len(row.args) != 1:
        return None
    # Compare the actual root tile identity, not a symbolic size hint: unrelated
    # axes may share an extent (e.g. both query and key tiles are 64 rows).
    if query.index_map is not None:
        query_row_origin = query.index_map.axes[0].origin
    else:
        query_subscripts = query.node.args[1]
        if not isinstance(query_subscripts, (tuple, list)):
            return None
        query_row_origin = query_subscripts[query.subscript_index(0)]
    if row.args[0] is not query_row_origin:
        return None
    if not _call(base, torch.ops.aten.sub.Tensor) or len(base.args) != 2:
        return None
    length, query_len = base.args
    if type(query_len) is not int or query_len < 0:
        return None
    if not isinstance(length, torch.fx.Node) or length not in accesses:
        return None
    lengths = accesses[length]
    if lengths.dims or lengths.loops or lengths.dtype not in (torch.int32, torch.int64):
        return None
    key_dtype = _integer_dtype(key_position)
    row_dtype = _integer_dtype(grouped_row)
    length_base_dtype = _integer_dtype(base)
    query_dtype = _integer_dtype(position)
    if None in (key_dtype, row_dtype, length_base_dtype, query_dtype):
        return None
    assert key_dtype is not None and row_dtype is not None
    assert length_base_dtype is not None and query_dtype is not None
    return AttentionCoordinates(
        width,
        lengths,
        query_len,
        group,
        key_dtype,
        row_dtype,
        length_base_dtype,
        query_dtype,
        reflected,
    )


def match_paged_attention(program: MmaDagProgram) -> PagedAttentionPlan | None:
    """Recognize a single paged two-dot loop with exact empty-row semantics.

    The MMA matcher has already proved axis binding, scalar index uniformity,
    and supported expression types. This matcher additionally proves the
    recurrence, byte conversion, output normalization, and page-address sharing.
    Layout/architecture restrictions belong to the subsequent transport plan.
    """
    if len(program.loops) != 1 or len(program.dots) != 2:
        return None
    loop = program.loops[0]
    if (
        loop.axes
        or loop.symbol is None
        or (len(loop.carries), len(loop.vec_carries)) not in ((1, 2), (2, 1))
    ):
        return None
    qk, pv = (dot.node for dot in program.dots)
    if any(dot.loop != loop.index for dot in program.dots):
        return None
    if not all(
        _call(node, matmul_ops.dot) and len(node.args) >= 3 for node in (qk, pv)
    ):
        return None
    if qk.args[2] is not None:
        return None
    accesses = {access.node: access for access in program.accesses}
    if not isinstance(qk.args[0], torch.fx.Node):
        return None
    query_node = program.resolve(qk.args[0])
    query = accesses.get(query_node)
    if query is None or query.dtype not in (torch.bfloat16, torch.float16):
        return None
    if len(query.tile_dims) != 2 or query.loops or query.grouped or query.windows:
        return None
    transposed_key = qk.args[1]
    if not _call(transposed_key, torch.ops.aten.permute.default) or transposed_key.args[
        1
    ] != [1, 0]:
        return None
    key = _block_scaled_operand(transposed_key.args[0], accesses, query.dtype)
    value = _block_scaled_operand(pv.args[1], accesses, query.dtype)
    if key is None or value is None or key.raw.subscripts != value.raw.subscripts:
        return None
    if key.raw.tile_dims != (
        program.dots[0].cols,
        program.dots[0].k,
    ):
        return None
    # A page number is one shared scalar table read, not the logical token
    # coordinate. Retain the complete Access rather than assuming table strides.
    page_indices = [
        (dimension, subscript.expr)
        for dimension, subscript in enumerate(key.raw.subscripts)
        if subscript.kind == "scalar" and subscript.expr in program.index_loads
    ]
    if len(page_indices) != 1:
        return None
    page_dimension, page_symbol = page_indices[0]
    assert isinstance(page_symbol, sympy.Symbol)
    page_table = program.index_loads[page_symbol]
    if page_table.loops != frozenset({loop.index}):
        return None
    probability = _cast(_zero_mask(pv.args[0]), query.dtype)
    score_base2 = _call(probability, torch.ops.aten.exp2.default)
    if not score_base2 and not _call(probability, torch.ops.aten.exp.default):
        return None
    centered = probability.args[0]
    relative_scores = _call(centered, torch.ops.aten.add.Tensor)
    if (not relative_scores and not _call(centered, torch.ops.aten.sub.Tensor)) or len(
        centered.args
    ) != 2:
        return None
    score_operand, shift_operand = centered.args
    score_match = _masked_scores(score_operand, qk)
    if score_match is None and relative_scores:
        score_operand, shift_operand = shift_operand, score_operand
        score_match = _masked_scores(score_operand, qk)
    if score_match is None:
        return None
    score, score_scale, mask, reference = score_match
    if (score_base2 and reference is None) or relative_scores != (
        reference is not None
    ):
        return None
    coordinates = _coordinate_contract(program, query, key, loop, mask, accesses)
    if coordinates is None:
        return None
    safe_max = _broadcast(shift_operand, row=True)
    if reference is not None:
        if (
            not _call(safe_max, torch.ops.aten.sub.Tensor)
            or len(safe_max.args) != 2
            or _copy(safe_max.args[0]) is not reference
        ):
            return None
        safe_max = safe_max.args[1]
    maximums = [
        carry
        for carry in loop.vec_carries
        if carry.init_vec is None
        and carry.init_fill == -math.inf
        and not carry.identity
    ]
    denominators = [
        carry
        for carry in loop.vec_carries
        if carry.init_vec is None and carry.init_fill == 0 and not carry.identity
    ]
    per_key = len(loop.carries) == 2
    if len(maximums) != 1 or len(denominators) != (0 if per_key else 1):
        return None
    maximum = maximums[0]
    if reference is not None and (
        not per_key or not _nonnegative_reference(reference, maximum.placeholder)
    ):
        return None
    max_value = _copy(maximum.output)
    row_max = _other(max_value, torch.ops.aten.maximum.default, maximum.placeholder)
    if reference is not None:
        row_max = _other(row_max, torch.ops.aten.add.Tensor, reference)
    if not _row_reduction(row_max, torch.ops.aten.amax.default, score, -math.inf):
        return None
    if not _safe_value(safe_max, max_value, -math.inf, 0):
        return None
    accumulators = [carry for carry in loop.carries if _copy(carry.output) is pv]
    if len(accumulators) != 1:
        return None
    accumulator = accumulators[0]
    if (
        accumulator.init is not None
        or accumulator.init_frag is not None
        or accumulator.init_value is not None
    ):
        return None
    rescale = pv.args[2]
    alpha_view = _other(rescale, torch.ops.aten.mul.Tensor, accumulator.placeholder)
    alpha = _broadcast(alpha_view, row=True)
    if not _call(
        alpha,
        torch.ops.aten.exp2.default if score_base2 else torch.ops.aten.exp.default,
    ):
        return None
    difference = alpha.args[0]
    if not _call(difference, torch.ops.aten.sub.Tensor) or len(difference.args) != 2:
        return None
    if (
        _copy(difference.args[0]) is not maximum.placeholder
        or difference.args[1] is not safe_max
    ):
        return None
    if per_key:
        denominator = next(carry for carry in loop.carries if carry is not accumulator)
        if (
            denominator.init is not None
            or denominator.init_frag is not None
            or denominator.init_value is not None
            or (denominator.rows, denominator.cols)
            != (program.dots[0].rows, program.dots[0].cols)
        ):
            return None
        rescaled_denominator = _other(
            _copy(denominator.output), torch.ops.aten.add.Tensor, probability
        )
        denominator_alpha = _other(
            rescaled_denominator, torch.ops.aten.mul.Tensor, denominator.placeholder
        )
        if _broadcast(denominator_alpha, row=True) is not alpha:
            return None
    else:
        denominator = denominators[0]
        denominator_value = _copy(denominator.output)
        if not _call(denominator_value, torch.ops.aten.add.Tensor):
            return None
        sums = [
            term
            for term in denominator_value.args
            if _row_reduction(term, torch.ops.aten.sum.dim_IntList, probability, 0)
        ]
        if len(sums) != 1:
            return None
        rescaled_denominator = _other(
            denominator_value, torch.ops.aten.add.Tensor, sums[0]
        )
        if (
            _copy(_other(rescaled_denominator, torch.ops.aten.mul.Tensor, alpha))
            is not denominator.placeholder
        ):
            return None
    # Exactly one normalized matrix store and an optional natural-log LSE
    # store; no other side effects may be dropped by a fused replacement.
    if len(program.stores) != 1 or len(program.vec_stores) > 1:
        return None
    output = program.stores[0]
    lse = program.vec_stores[0] if program.vec_stores else None
    if (
        output.region != "epilogue"
        or (lse is not None and lse.loop != -1)
        or output.access.dtype != query.dtype
    ):
        return None
    normalized = _cast(output.value, query.dtype)
    if not _call(normalized, torch.ops.aten.div.Tensor) or len(normalized.args) != 2:
        return None
    if _copy(normalized.args[0]) is not accumulator.phi:
        return None
    safe_denominator = _broadcast(normalized.args[1], row=True)
    if (
        not _call(safe_denominator, torch.ops.aten.where.self)
        or len(safe_denominator.args) != 3
    ):
        return None
    denominator_final = _copy(safe_denominator.args[2])
    if not isinstance(denominator_final, torch.fx.Node):
        return None
    if per_key:
        if not _row_reduction(
            denominator_final, torch.ops.aten.sum.dim_IntList, denominator.phi, 0
        ):
            return None
    elif denominator_final is not denominator.phi:
        return None
    if not _safe_value(safe_denominator, denominator_final, 0, 1):
        return None
    if lse is not None:
        lse_maximum = maximum.phi
        if score_base2:
            if not _call(lse.value, torch.ops.aten.add.Tensor):
                return None
            maxima = []
            for term in lse.value.args:
                factor = _other(_copy(term), torch.ops.aten.mul.Tensor, maximum.phi)
                # Scalar tensors have already rounded ln2 through their
                # constructor dtype; their recorded Python literal has not.
                if type(factor) is float and factor == math.log(2):
                    maxima.append(term)
            if len(maxima) != 1:
                return None
            lse_maximum = maxima[0]
        lse_log = _other(lse.value, torch.ops.aten.add.Tensor, lse_maximum)
        if (
            not _call(lse_log, torch.ops.aten.log.default)
            or len(lse_log.args) != 1
            or _copy(lse_log.args[0]) is not denominator_final
            or lse.access.dtype != torch.float32
        ):
            return None
    if query.index_map is not None:
        query_axes = query.index_map.axes
        if output.access.index_map is None:
            return None
        expected_axes = [(output.access.index_map.axes, query_axes)]
        if lse is not None:
            if lse.access.index_map is None:
                return None
            expected_axes.append((lse.access.index_map.axes, query_axes[:-1]))
        for actual, expected in expected_axes:
            if len(actual) != len(expected) or any(
                left.axis != right.axis or left.origin is not right.origin
                for left, right in zip(actual, expected, strict=True)
            ):
                return None
        # The physical layout planner separately proves that these logical
        # rows address the matching request/head and disjoint output elements.
    elif (
        output.access.index_map is not None
        or output.access.subscripts != query.subscripts
        or (
            lse is not None
            and (
                lse.access.index_map is not None
                or lse.access.subscripts != query.subscripts[:-1]
            )
        )
    ):
        return None
    return PagedAttentionPlan(
        program,
        loop,
        query,
        key,
        value,
        page_table,
        page_symbol,
        page_dimension,
        mask,
        coordinates,
        score_scale,
        score,
        probability,
        maximum,
        denominator,
        accumulator,
        output.access,
        lse.access if lse is not None else None,
        per_key,
        denominator_final,
        reference,
        score_base2,
    )
