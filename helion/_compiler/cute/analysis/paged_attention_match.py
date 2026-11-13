"""Semantic recognition for paged, block-scaled online attention.

This proof retains the source accesses and logical mask separately. A transport
planner must still validate its descriptor layouts, scalar expression support,
and tile sizes; matching does not select a device implementation.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import struct
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeGuard

import sympy
import torch

from .... import exc
from ....autotuner.config_fragment import EnumFragment
from ....autotuner.config_fragment import IntegerFragment
from ....autotuner.config_fragment import PowerOfTwoFragment
from ....language import matmul_ops
from ....language import tile_ops
from ....language import view_ops
from ....language._tracing_ops import _get_symnode
from ....language._tracing_ops import _mask_to
from ....language._tracing_ops import _new_var
from ...compile_environment import ConfigValueExpression
from ...utils import is_full_slice
from ..cute_mma_dag import _storage
from .match_common import scalar_value
from .match_common import symnode_sympy
from .native_s2f6_operand import match_native_s2f6_operand

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx import Node

    from ....autotuner.config_fragment import ConfigSpecFragment
    from ....runtime.config import Config
    from ..cute_mma_dag import Access
    from ..cute_mma_dag import Carry
    from ..cute_mma_dag import Loop
    from ..cute_mma_dag import MmaDagProgram
    from ..cute_mma_dag import VecCarry
    from .native_s2f6_match import NativeS2F6Decode


@dataclasses.dataclass(frozen=True)
class BlockScaledOperand:
    """A typed byte decode with optional explicit BF16 scaling and V +0.

    The base expression is ``(int8.float() * exp2(uint8.float() - 127)).to(dtype)``.
    A post-multiply by a power of two retains that first rounding boundary.
    Value operands may also add positive zero after the final BF16 rounding.

    Both accesses retain the runtime tensor name, physical strides,
    scalar page/head subscripts, and logical axes. Only the scale's final axis
    has a group divisor; neither access is replaced by a dense view. The named runtime tensor pointer
    includes its storage offset; the fake tensor's offset is normalized.
    """

    raw: Access
    scale: Access
    group_size: int
    value: Node
    # Optional source multiplication after the first BF16 rounding.
    post_scale_log2: int = 0
    # Explicit BF16 addition of positive zero after the decode/postscale.
    positive_zero: bool = False
    native_proof: NativeS2F6Decode | None = None

    @property
    def decode_kind(self) -> Literal["float_exp2", "native_s2f6_bf16"]:
        return "native_s2f6_bf16" if self.native_proof is not None else "float_exp2"


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
class RowwiseRescale:
    """An explicit predicate selecting proposed maxima per row or whole head."""

    threshold: float | ConfigValueExpression
    headwide: bool = False

    def threshold_for(self, config: Config) -> float:
        """Resolve the selected threshold without substituting a search default."""
        if not isinstance(self.threshold, ConfigValueExpression):
            return self.threshold
        try:
            value = self.threshold.evaluate(config)
        except TypeError as error:
            raise exc.InvalidConfig(
                "rowwise rescale threshold must be an integer"
            ) from error
        threshold = _exact_config_threshold(value)
        if threshold is None:
            raise exc.InvalidConfig(
                "rowwise rescale threshold must be nonnegative and exactly "
                "representable as finite float32"
            )
        return threshold


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
    # Source explicitly computes a row reciprocal then multiplies each output.
    # Preserve this rounding order separately from per-element division.
    output_reciprocal: bool
    # An explicit FP32 multiplication after the rounded row reciprocal.
    output_scale_log2: int = 0
    # Explicit FP32 clamp_max(exp2(...), 1) on both recurrence coefficients.
    coefficients_capped: bool = False
    # Raw FP32 accumulator output with separate exact maximum/denominator stores.
    # Current admission couples this epilogue to capped coefficients. Keep the
    # math and storage facts separate: neither implies the other for a source.
    partial_statistics: tuple[Access, Access] | None = None
    rowwise_rescale: RowwiseRescale | None = None

    @property
    def row_outputs(self) -> tuple[Access, ...]:
        """All epilogue row stores, in their proved logical roles."""
        return (() if self.lse is None else (self.lse,)) + (
            self.partial_statistics or ()
        )

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
        # Structural constants (-inf, 0, 1, 127) are exact in these floating
        # dtypes. _power_of_two also checks its wider constants against the
        # constructor dtype. Integer/bool/FP8 constructors can change them.
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


def _unit_coefficient_input(value: object) -> object:
    """Strip an explicit FP32 unit cap; callers prove the returned operation."""
    if not _call(value, torch.ops.aten.clamp_max.default) or len(value.args) != 2:
        return None
    if not _scalar(value.args[1], 1.0):
        return None
    for node in (value, value.args[0]):
        tensor = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            return None
    return value.args[0]


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


def _power_of_two(value: object, low: int, high: int) -> int | None:
    """Match binary constants only when their source dtype represents them exactly."""
    for exponent in range(low, high + 1):
        factor = 2.0**exponent
        if _scalar(value, factor):
            if _call(value, torch.ops.aten.scalar_tensor.default):
                tensor = value.meta.get("val")
                assert isinstance(tensor, torch.Tensor)  # Proved by _scalar.
                limits = torch.finfo(tensor.dtype)
                # A binary power inside this interval is exact, including
                # subnormals. Outside it the constructor rounds to zero/Inf.
                if not limits.tiny * limits.eps <= factor <= limits.max:
                    return None
            return exponent
    return None


def _reciprocal_denominator(value: object) -> tuple[object, int]:
    """The FP32 row reciprocal, including Torch's ``1.0 / tensor`` lowering."""
    value = _copy(value)
    output_scale = 0
    # One explicit power-of-two multiplier, with optional identity multiplies
    # introduced by Torch's 1.0 / tensor lowering. Do not combine two nonunit
    # factors: their intermediate rounding/overflow is part of the source.
    # The bounded output range mirrors the supported V representation scales;
    # other factors retain ordinary source lowering.
    while _call(value, torch.ops.aten.mul.Tensor) and len(value.args) == 2:
        tensor = value.meta.get("val")
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            return None, 0
        left, right = value.args
        scale = _power_of_two(right, 0, 22)
        if scale is None:
            left, right = right, left
            scale = _power_of_two(right, 0, 22)
        if scale is None or (scale != 0 and output_scale != 0):
            return None, 0
        output_scale += scale
        value = _copy(left)
    if not _call(value, torch.ops.aten.reciprocal.default) or len(value.args) != 1:
        return None, 0
    tensor = value.meta.get("val")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        return None, 0
    return _copy(value.args[0]), output_scale


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
    return _nonnegative_finite(condition, maximum)


def _nonnegative_finite(condition: object, maximum: Node) -> bool:
    if (
        not _call(condition, torch.ops.aten.bitwise_and.Tensor)
        or len(condition.args) != 2
    ):
        return False
    assert isinstance(condition, torch.fx.Node)
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


def _exact_fp32_threshold(value: object) -> float | None:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return None
    if not 0 <= value <= 3.4028234663852886e38:
        return None
    result = float(value)
    # Preserve the existing literal conversion through Python float before
    # checking that the emitted FP32 comparison needs no further rounding.
    return result if struct.unpack("f", struct.pack("f", result))[0] == result else None


def _exact_config_threshold(value: object) -> float | None:
    """A declared integer choice must survive both conversions without rounding."""
    if type(value) is not int:
        return None
    result = _exact_fp32_threshold(value)
    return result if result == value else None


def _rowwise_threshold_domain(fragment: ConfigSpecFragment | None) -> bool:
    """Admit only a complete known domain; other kernels keep ordinary lowering.

    A search subset is not the domain: users can select any declared Enum choice.
    Custom fragments may generate values outside these built-in rules.
    """
    if fragment is None:
        return False
    if type(fragment) is EnumFragment:
        return bool(fragment.choices) and all(
            _exact_config_threshold(value) is not None for value in fragment.choices
        )
    # Integer defaults are clamped, not rounded onto the generated domain.
    # In particular, a power-of-two fragment can have an inexact interior default.
    if type(fragment) is IntegerFragment:
        return (
            _exact_config_threshold(fragment.default()) is not None
            and type(fragment.low) is int
            and type(fragment.high) is int
            and (
                0 <= fragment.low <= fragment.high <= 2**24
                or fragment.low == fragment.high
                and _exact_config_threshold(fragment.low) is not None
            )
        )
    if type(fragment) is PowerOfTwoFragment:
        return (
            _exact_config_threshold(fragment.default()) is not None
            and type(fragment.low) is int
            and type(fragment.high) is int
            and 0 < fragment.low <= fragment.high <= 2**127
            and fragment.low & (fragment.low - 1) == 0
            and fragment.high & (fragment.high - 1) == 0
        )
    return False


def _rowwise_rescale(
    predicate: Node, tile_maximum: Node, maximum: Node, program: MmaDagProgram
) -> RowwiseRescale | None:
    """Prove row_max > T | ~valid(old) | isnan(row_max), with no extra terms."""
    terms: list[Node] = []
    pending: list[object] = [predicate]
    while pending:
        term = _copy(pending.pop())
        if not isinstance(term, torch.fx.Node):
            return None
        if _call(term, torch.ops.aten.bitwise_or.Tensor) and len(term.args) == 2:
            pending.extend(term.args)
        else:
            terms.append(term)
    if len(terms) != 3:
        return None
    greater = [t for t in terms if _call(t, torch.ops.aten.gt.Scalar)]
    invalid = [t for t in terms if _call(t, torch.ops.aten.bitwise_not.default)]
    nan = [t for t in terms if _call(t, torch.ops.aten.isnan.default)]
    if len(greater) != 1 or len(invalid) != 1 or len(nan) != 1:
        return None
    (comparison,) = greater
    (invalid_old,) = invalid
    (nan_maximum,) = nan
    if (
        len(comparison.args) != 2
        or _copy(comparison.args[0]) is not tile_maximum
        or len(invalid_old.args) != 1
        or not _nonnegative_finite(invalid_old.args[0], maximum)
        or len(nan_maximum.args) != 1
        or _copy(nan_maximum.args[0]) is not tile_maximum
    ):
        return None
    threshold = _copy(comparison.args[1])
    if _call(threshold, _get_symnode) and isinstance(
        threshold.meta.get("val"), torch.SymInt
    ):
        expression = symnode_sympy(threshold)
        if not isinstance(expression, sympy.Symbol):
            return None
        config_value = program.config_scalars.get(expression)
        if config_value is None or config_value.operation != "config":
            return None
        (name,) = config_value.arguments
        if not isinstance(name, str) or not _rowwise_threshold_domain(
            program.config_fragments.get(name)
        ):
            return None
        return RowwiseRescale(config_value)
    literal = _exact_fp32_threshold(threshold)
    if literal is None:
        return None
    return RowwiseRescale(literal)


def _headwide_predicate(
    predicate: Node, query: Access, maximum: Node, coordinates: AttentionCoordinates
) -> Node | None:
    """Prove max(float(bool_rows)) > 0 over exactly one complete logical head.

    The explicit negative-infinity mask excludes padded root rows, including invalid old
    maxima in those rows. Float32 represents every Boolean value exactly.
    """
    if (
        not _call(predicate, torch.ops.aten.gt.Scalar)
        or len(predicate.args) != 2
        or predicate.args[1] != 0.0
    ):
        return None
    reduction = predicate.args[0]
    if (
        not _call(reduction, torch.ops.aten.amax.default)
        or len(reduction.args) not in (2, 3)
        or reduction.args[1] not in ([0], [-1])
        or reduction.kwargs
        or (len(reduction.args) == 3 and reduction.args[2] is not False)
    ):
        return None
    masked = reduction.args[0]
    if (
        not _call(masked, _mask_to)
        or len(masked.args) != 2
        or not _scalar(masked.args[1], -math.inf)
        or masked.kwargs
    ):
        return None
    converted = masked.args[0]
    if (
        not _call(converted, torch.ops.prims.convert_element_type.default)
        or len(converted.args) != 2
        or converted.args[1] is not torch.float32
        or converted.kwargs
    ):
        return None
    rows = converted.args[0]
    if not isinstance(rows, torch.fx.Node):
        return None
    value = rows.meta.get("val")
    old = maximum.meta.get("val")
    result = predicate.meta.get("val")
    axis = query.tile_dims[0]
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.ndim != 1
        or not isinstance(old, torch.Tensor)
        or value.shape != old.shape
        or not isinstance(result, torch.Tensor)
        or result.dtype != torch.bool
        or result.ndim != 0
        or axis.kind != "root"
        or axis.size is None
        or axis.size < axis.extent
        or axis.extent != coordinates.query_len * coordinates.group
    ):
        return None
    return rows


def _block_scaled_access_pair(
    raw: Node, scale: Node, accesses: dict[Node, Access]
) -> tuple[Access, Access, int] | None:
    raw_access, scale_access = accesses.get(raw), accesses.get(scale)
    if (
        raw_access is None
        or scale_access is None
        or raw_access.dtype != torch.int8
        or scale_access.dtype != torch.uint8
        or raw_access.grouped
        or len(raw_access.subscripts) != len(scale_access.subscripts)
        or raw_access.dims != scale_access.dims
    ):
        return None
    divisors = [s.divisor for s in scale_access.subscripts if s.divisor is not None]
    if len(divisors) != 1:
        return None
    group = divisors[0]
    assert group is not None
    if group <= 0 or scale_access.subscripts[-1].divisor != group:
        return None
    if any(
        left != dataclasses.replace(right, divisor=None)
        for left, right in zip(
            raw_access.subscripts, scale_access.subscripts, strict=True
        )
    ):
        return None
    return raw_access, scale_access, group


def _block_scaled_operand(
    value: object,
    accesses: dict[Node, Access],
    dtype: torch.dtype,
    *,
    allow_positive_zero: bool = False,
    resolve: Callable[[Node], Node] | None = None,
) -> BlockScaledOperand | None:
    value = _zero_mask(value)
    if dtype == torch.bfloat16 and isinstance(value, torch.fx.Node):
        native = match_native_s2f6_operand(value, resolve=resolve)
        if native is not None:
            pair = _block_scaled_access_pair(native.raw, native.scale, accesses)
            if pair is None or pair[2] != 32:
                return None
            raw_access, scale_access, group = pair
            return BlockScaledOperand(
                raw_access,
                scale_access,
                group,
                value,
                native_proof=native,
            )
    decoded = value
    positive_zero = False
    if allow_positive_zero and _call(value, torch.ops.aten.add.Tensor):
        if len(value.args) != 2 or dtype != torch.bfloat16:
            return None
        left, right = value.args
        if type(left) is float:
            left, right = right, left
        if type(right) is not float or right != 0.0 or math.copysign(1.0, right) != 1.0:
            return None
        for node in (value, left):
            tensor = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
                return None
        decoded = left
        positive_zero = True
    post_scale = 0
    if _call(decoded, torch.ops.aten.mul.Tensor) and len(decoded.args) == 2:
        tensor = decoded.meta.get("val")
        if (
            dtype != torch.bfloat16
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != dtype
        ):
            return None
        decoded, factor = decoded.args
        exponent = _power_of_two(factor, -22, 0)
        if exponent is None:
            decoded, factor = factor, decoded
            exponent = _power_of_two(factor, -22, 0)
        if exponent is None:
            return None
        post_scale = exponent
    product = _cast(decoded, dtype)
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
        pair = _block_scaled_access_pair(raw, scale, accesses)
        if pair is None:
            continue
        raw_access, scale_access, group = pair
        assert isinstance(value, torch.fx.Node)
        return BlockScaledOperand(
            raw_access, scale_access, group, value, post_scale, positive_zero
        )
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
    key = _block_scaled_operand(
        transposed_key.args[0], accesses, query.dtype, resolve=program.resolve
    )
    value = _block_scaled_operand(
        pv.args[1],
        accesses,
        query.dtype,
        allow_positive_zero=True,
        resolve=program.resolve,
    )
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
    if not isinstance(probability, torch.fx.Node):
        return None
    coefficients_capped = _call(probability, torch.ops.aten.clamp_max.default)
    probability_exp = (
        _unit_coefficient_input(probability) if coefficients_capped else probability
    )
    score_base2 = _call(probability_exp, torch.ops.aten.exp2.default)
    if not score_base2 and not _call(probability_exp, torch.ops.aten.exp.default):
        return None
    assert isinstance(probability_exp, torch.fx.Node)
    centered = probability_exp.args[0]
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
    if (
        score_base2 and reference is None and not coefficients_capped
    ) or relative_scores != (reference is not None):
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
    if coefficients_capped and (
        not score_base2 or reference is not None or not per_key
    ):
        return None
    if len(maximums) != 1 or len(denominators) != (0 if per_key else 1):
        return None
    maximum = maximums[0]
    if reference is not None and (
        not per_key or not _nonnegative_reference(reference, maximum.placeholder)
    ):
        return None
    max_value = _copy(maximum.output)
    proposed_max = max_value
    rowwise_predicate = None
    if _call(max_value, torch.ops.aten.where.self):
        assert isinstance(max_value, torch.fx.Node)
        if (
            reference is None
            or not score_base2
            or len(max_value.args) != 3
            or _copy(max_value.args[2]) is not maximum.placeholder
            or not isinstance(max_value.args[0], torch.fx.Node)
        ):
            return None
        rowwise_predicate = max_value.args[0]
        proposed_max = _copy(max_value.args[1])
    row_max = _other(proposed_max, torch.ops.aten.maximum.default, maximum.placeholder)
    if reference is not None:
        row_max = _other(row_max, torch.ops.aten.add.Tensor, reference)
    if not _row_reduction(row_max, torch.ops.aten.amax.default, score, -math.inf):
        return None
    rowwise_rescale = None
    if rowwise_predicate is not None:
        assert isinstance(row_max, torch.fx.Node)
        head_rows = _headwide_predicate(
            rowwise_predicate, query, maximum.placeholder, coordinates
        )
        rowwise_rescale = _rowwise_rescale(
            head_rows if head_rows is not None else rowwise_predicate,
            row_max,
            maximum.placeholder,
            program,
        )
        if rowwise_rescale is None:
            return None
        if head_rows is not None:
            rowwise_rescale = dataclasses.replace(rowwise_rescale, headwide=True)
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
    alpha_exp = _unit_coefficient_input(alpha) if coefficients_capped else alpha
    if not _call(
        alpha_exp,
        torch.ops.aten.exp2.default if score_base2 else torch.ops.aten.exp.default,
    ):
        return None
    difference = alpha_exp.args[0]
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
    # All stores must belong to one of the two explicit epilogue contracts.
    # Neither route may discard an unrelated side effect.
    if len(program.stores) != 1:
        return None
    output = program.stores[0]
    if output.region != "epilogue" or any(
        store.loop != -1 for store in program.vec_stores
    ):
        return None
    partial_statistics = None
    lse = None
    output_reciprocal = False
    output_scale = 0
    if coefficients_capped:
        if (
            query.dtype != torch.bfloat16
            or output.access.dtype != torch.float32
            or _copy(output.value) is not accumulator.phi
            or len(program.vec_stores) != 2
        ):
            return None
        # This schedule groups raw/m/l stores by kind rather than source
        # order. Same-storage views therefore need the ordinary emitter.
        # The runtime alias guard separately protects distinct arguments
        # that acquire overlapping storage on a later invocation.
        if (
            len({_storage(store.access) for store in (output, *program.vec_stores)})
            != 3
        ):
            return None
        maxima = [
            store for store in program.vec_stores if _copy(store.value) is maximum.phi
        ]
        totals = [
            store
            for store in program.vec_stores
            if _row_reduction(
                _copy(store.value), torch.ops.aten.sum.dim_IntList, denominator.phi, 0
            )
        ]
        if (
            len(maxima) != 1
            or len(totals) != 1
            or any(store.access.dtype != torch.float32 for store in program.vec_stores)
        ):
            return None
        partial_statistics = (maxima[0].access, totals[0].access)
        denominator_final = _copy(totals[0].value)
        assert isinstance(denominator_final, torch.fx.Node)
    else:
        if len(program.vec_stores) > 1 or output.access.dtype != query.dtype:
            return None
        lse = program.vec_stores[0] if program.vec_stores else None
        normalized = _cast(output.value, query.dtype)
        output_reciprocal = not _call(normalized, torch.ops.aten.div.Tensor)
        if output_reciprocal:
            factor = _other(normalized, torch.ops.aten.mul.Tensor, accumulator.phi)
            safe_denominator, output_scale = _reciprocal_denominator(
                _broadcast(factor, row=True)
            )
        else:
            assert isinstance(normalized, torch.fx.Node)
            if (
                len(normalized.args) != 2
                or _copy(normalized.args[0]) is not accumulator.phi
            ):
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
        for store in program.vec_stores:
            if store.access.index_map is None:
                return None
            expected_axes.append((store.access.index_map.axes, query_axes[:-1]))
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
            any(
                store.access.index_map is not None
                or store.access.subscripts != query.subscripts[:-1]
                for store in program.vec_stores
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
        output_reciprocal,
        output_scale,
        coefficients_capped,
        partial_statistics,
        rowwise_rescale,
    )
