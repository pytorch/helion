"""Structural lowering for a single-token rank-1 state recurrence.

The family recognized here has the algebraic form::

    d = exp(lower_bound * sigmoid(exp(A_log) * (gate + bias)))
    k = normalize(k)
    q = scale * normalize(q)
    s0 = state * d
    residual = sigmoid(beta) * (v - sum(s0 * k))
    state = s0 + residual * k
    output = sum(state * q)

The first admitted member is the BF16 ``K == V == 128`` decode shape.  The
matcher is intentionally fail closed: it proves the complete load, update,
store, normalization, and output graph before replacing the root body.  Tensor
and function names are used only as generated-code suggestions, never as
matching keys.

This lowering changes reduction association and performs the recurrent update
with native packed BF16 arithmetic.  It is therefore available only under the
user-facing ``Settings.fast_math`` numerical policy.  For ``lower_bound=-5``
it maps the unbounded gate input through ``tanh(x / 2)`` and evaluates a
degree-7 decay polynomial over the resulting compact interval.  The fit was
audited on ``[-20, 20]`` plus extreme inputs ``+/-100`` and remains bounded,
avoiding the narrow-domain failure mode of the earlier direct polynomial.
"""

from __future__ import annotations

import ast
import dataclasses
import operator
import textwrap
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import dsl_user_op
import torch

from ..compile_environment import CompileEnvironment
from ._mlir_compat import ir
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _GeneratedCodeTemplate
from .fx_matcher import _xyz_grid_fits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch
_KEY_SIZE = 128
_VALUE_SIZE = 128
_VALUE_TILE = 16
_THREADS = 256
_ONE_WARP_VALUE_TILES = (8, 16)
_ONE_WARP_THREADS = 32
_SINGLE_TOKEN_RANK1_CODEGEN_ABI_VERSION = 2
_LOG2_E = 1.4426950408889634
_NORM_EPSILON = 1.0e-6
_DECAY_TANH_POLY_COEFFICIENTS = (
    0.08205670446506756,
    -0.20514968979728407,
    0.2574133207524337,
    -0.21444737369177794,
    0.12918135990752894,
    -0.06471848297367808,
    0.03468821743131916,
    -0.012315480037717208,
)
# Bump whenever an imported rank-1 ``dsl_user_op`` changes.  CuTe's compiled
# launcher hashes generated source, not the Python bodies behind its imports.
_RANK1_HELPER_ABI_VERSION = 4


@dsl_user_op
def rank1_pack_bf16x2(
    low: object,
    high: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint32:
    """Round two FP32 values and pack them as one BF16x2 register."""

    result = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [
            cutlass.Float32(low).ir_value(loc=loc, ip=ip),
            cutlass.Float32(high).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


def _rank1_binary_bf16x2(
    instruction: str,
    left: object,
    right: object,
    *,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> cutlass.Uint32:
    result = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [
            cutlass.Uint32(left).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(right).ir_value(loc=loc, ip=ip),
        ],
        instruction,
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def rank1_mul_bf16x2(
    left: object,
    right: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint32:
    return _rank1_binary_bf16x2(
        "mul.rn.bf16x2 $0, $1, $2;", left, right, loc=loc, ip=ip
    )


@dsl_user_op
def rank1_add_bf16x2(
    left: object,
    right: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint32:
    return _rank1_binary_bf16x2(
        "add.rn.bf16x2 $0, $1, $2;", left, right, loc=loc, ip=ip
    )


@dsl_user_op
def rank1_fma_bf16x2(
    left: object,
    right: object,
    accumulator: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint32:
    result = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [
            cutlass.Uint32(left).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(right).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(accumulator).ir_value(loc=loc, ip=ip),
        ],
        "fma.rn.bf16x2 $0, $1, $2, $3;",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


def _rank1_inline_nc_load(
    ptr: object,
    count: int,
    *,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> ir.Value:
    """Load one or more 32-bit words through the read-only cache path."""

    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    u32 = cutlass.Uint32.mlir_type
    if count == 1:
        return llvm.inline_asm(
            u32,
            [address],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    result_type = cast("Any", llvm).StructType.get_literal([u32] * count)
    outputs = ", ".join(f"${index}" for index in range(count))
    result = llvm.inline_asm(
        result_type,
        [address],
        f"ld.global.nc.v{count}.b32 {{{outputs}}}, [${count}];",
        ",".join(["=r"] * count + ["l"]),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = [
        llvm.extractvalue(u32, result, [index], loc=loc, ip=ip)
        for index in range(count)
    ]
    return vector.from_elements(ir.VectorType.get([count], u32), values, loc=loc, ip=ip)


@dsl_user_op
def rank1_load_u32x2_nc(
    ptr: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Load two immutable packed words through the read-only cache path."""

    return _rank1_inline_nc_load(ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def rank1_load_f32x4_nc(
    ptr: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Load four immutable FP32 values through the read-only cache path."""

    packed = _rank1_inline_nc_load(ptr, 4, loc=loc, ip=ip)
    return vector.bitcast(
        ir.VectorType.get([4], cutlass.Float32.mlir_type),
        packed,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def rank1_load_f32_nc(
    ptr: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    """Load one immutable FP32 value through the read-only cache path."""

    return cutlass.Float32(
        cutlass.Uint32(_rank1_inline_nc_load(ptr, 1, loc=loc, ip=ip)).bitcast(
            cutlass.Float32
        )
    )


@dsl_user_op
def rank1_load_u16_nc(
    ptr: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint16:
    """Load one immutable 16-bit value through the read-only cache path."""

    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        cutlass.Uint16.mlir_type,
        [address],
        "ld.global.nc.b16 $0, [$1];",
        "=h,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint16(result)


@dsl_user_op
def rank1_load_i32_nc(
    ptr: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Int32:
    """Load one immutable signed 32-bit value through read-only cache."""

    return cutlass.Int32(_rank1_inline_nc_load(ptr, 1, loc=loc, ip=ip))


@dsl_user_op
def rank1_load_i64_nc(
    ptr: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Int64:
    """Load one immutable signed 64-bit value through read-only cache."""

    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        cutlass.Int64.mlir_type,
        [address],
        "ld.global.nc.b64 $0, [$1];",
        "=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(result)


@dsl_user_op
def rank1_load_u32x4_if_valid(
    ptr: object,
    state_index: object,
    state_size: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Load a state vector for an in-range slot and return zero otherwise."""

    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    index = cutlass.Int64(state_index).ir_value(loc=loc, ip=ip)
    size = cutlass.Int64(state_size).ir_value(loc=loc, ip=ip)
    u32 = cutlass.Uint32.mlir_type
    result_type = cast("Any", llvm).StructType.get_literal([u32] * 4)
    result = llvm.inline_asm(
        result_type,
        [index, size, address],
        (
            "{ .reg .pred valid; "
            "mov.u32 $0, 0; mov.u32 $1, 0; mov.u32 $2, 0; mov.u32 $3, 0; "
            "setp.lt.u64 valid, $4, $5; "
            "@valid ld.global.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$6]; }"
        ),
        "=r,=r,=r,=r,l,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = [
        llvm.extractvalue(u32, result, [index], loc=loc, ip=ip) for index in range(4)
    ]
    return vector.from_elements(ir.VectorType.get([4], u32), values, loc=loc, ip=ip)


@dsl_user_op
def rank1_store_u32x4_if_valid(
    ptr: object,
    value0: object,
    value1: object,
    value2: object,
    value3: object,
    state_index: object,
    state_size: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store a state vector only when its slot is in range."""

    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    values = [
        cutlass.Uint32(value).ir_value(loc=loc, ip=ip)
        for value in (value0, value1, value2, value3)
    ]
    index = cutlass.Int64(state_index).ir_value(loc=loc, ip=ip)
    size = cutlass.Int64(state_size).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address, *values, index, size],
        (
            "{ .reg .pred valid; "
            "setp.lt.u64 valid, $5, $6; "
            "@valid st.global.L1::no_allocate.v4.u32 [$0], {$1, $2, $3, $4}; }"
        ),
        "l,r,r,r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def rank1_store_u16_or_zero(
    ptr: object,
    element_offset: object,
    value: object,
    state_index: object,
    state_size: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store ``value`` for an in-range slot and zero otherwise."""

    base = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    offset = cutlass.Uint32(element_offset).ir_value(loc=loc, ip=ip)
    bits = cutlass.Uint16(value).ir_value(loc=loc, ip=ip)
    index = cutlass.Int64(state_index).ir_value(loc=loc, ip=ip)
    size = cutlass.Int64(state_size).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [base, offset, bits, index, size],
        (
            "{ .reg .pred valid; .reg .u16 selected; "
            ".reg .u64 address; "
            "setp.lt.u64 valid, $3, $4; "
            "selp.u16 selected, $2, 0, valid; "
            "mad.wide.u32 address, $1, 2, $0; "
            "st.global.u16 [address], selected; }"
        ),
        "l,r,h,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dataclasses.dataclass(frozen=True)
class _TensorRef:
    fake: torch.Tensor
    suggested_name: str


@dataclasses.dataclass(frozen=True)
class CuteSingleTokenRank1Plan:
    """Values consumed by the dedicated rank-1 code generator."""

    root_graph_id: int
    mixed: _TensorRef
    gate: _TensorRef
    beta: _TensorRef
    a_log: _TensorRef
    bias: _TensorRef
    state: _TensorRef
    output: _TensorRef
    state_indices: _TensorRef
    scale: object
    lower_bound: object
    log2_e: object
    batch_size: int
    state_size: int
    state_stride0: int
    mixed_stride0: int
    value_heads: int
    tile_rows: int
    threads: int
    topology: str
    batch_block_id: int
    head_block_id: int
    value_block_id: int


def _is_call(node: object, target: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
    )


def _host_tensor_ref(node: object) -> _TensorRef | None:
    from ...language._tracing_ops import _host_tensor

    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not _host_tensor
        or not node.args
        or not isinstance(node.args[0], str)
    ):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return _TensorRef(value, node.args[0])


def _load_ref(node: object) -> _TensorRef | None:
    from ...language import memory_ops

    if not _is_call(node, memory_ops.load):
        return None
    assert isinstance(node, torch.fx.Node)
    return _host_tensor_ref(node.args[0] if node.args else None)


def _store_ref(node: object) -> _TensorRef | None:
    from ...language import memory_ops

    if not _is_call(node, memory_ops.store):
        return None
    assert isinstance(node, torch.fx.Node)
    return _host_tensor_ref(node.args[0] if node.args else None)


def _same_ref(left: _TensorRef | None, right: _TensorRef | None) -> bool:
    return (
        left is not None
        and right is not None
        and left.fake is right.fake
        and left.suggested_name == right.suggested_name
    )


def _has_unmatched_side_effect(
    nodes: Sequence[torch.fx.Node], allowed: set[torch.fx.Node]
) -> bool:
    from torch.fx.node import _side_effectful_functions

    return any(
        node.op == "call_function"
        and node.target in _side_effectful_functions
        and node not in allowed
        for node in nodes
    )


def _strip_casts(node: object) -> torch.fx.Node | None:
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
        source_value = source.meta.get("val")
        output_value = current.meta.get("val")
        if (
            not isinstance(source_value, torch.Tensor)
            or not isinstance(output_value, torch.Tensor)
            or (source_value.dtype, output_value.dtype)
            not in (
                (torch.bfloat16, torch.float32),
                (torch.float32, torch.bfloat16),
            )
        ):
            return None
        current = source
    return None


def _strip_state_index_casts(node: object) -> torch.fx.Node | None:
    """Strip only the exact signed-int widening used by ``Tensor.long``."""

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
            or current.args[1] is not torch.int64
            or current.kwargs
            or not isinstance(current.args[0], torch.fx.Node)
        ):
            return None
        source = current.args[0]
        source_value = source.meta.get("val")
        if not isinstance(source_value, torch.Tensor) or source_value.dtype not in (
            torch.int32,
            torch.int64,
        ):
            return None
        current = source
    return None


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
    from ...language import view_ops

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
    from ...language._tracing_ops import _mask_to

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


def _match_normalized_vector(
    node: object,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node] | None:
    """Return ``(normalized, raw_fp32, norm_sum)`` for L2+epsilon+rsqrt."""

    pair = _binary_pair(node, _MUL_TARGETS)
    if pair is None:
        return None
    for raw, scale in (pair, reversed(pair)):
        if (
            scale.op != "call_function"
            or scale.target is not torch.ops.aten.rsqrt.default
            or not scale.args
            or not isinstance(scale.args[0], torch.fx.Node)
        ):
            continue
        norm = scale.args[0]
        if (
            not isinstance(norm, torch.fx.Node)
            or norm.op != "call_function"
            or norm.target not in _ADD_TARGETS
            or len(norm.args) < 2
            or norm.kwargs
        ):
            continue
        norm_sum: torch.fx.Node | None = None
        norm_args = norm.args[:2]
        for maybe_sum, epsilon in (norm_args, reversed(norm_args)):
            if (
                isinstance(maybe_sum, torch.fx.Node)
                and isinstance(epsilon, (int, float))
                and float(epsilon) == _NORM_EPSILON
                and _sum_input(maybe_sum, None) is not None
            ):
                norm_sum = maybe_sum
                break
        if norm_sum is None:
            continue
        square = _sum_input(norm_sum, None)
        if square is None or square.op != "call_function":
            continue
        if square.target not in _MUL_TARGETS or len(square.args) < 2:
            continue
        # Device-IR reduction extraction replaces the repeated second operand
        # of ``x * x`` by ``None``.  Before extraction both operands may still
        # be present, so accept exactly those two equivalent encodings.
        square_args = square.args[:2]
        if square_args not in ((raw, raw), (raw, None), (None, raw)):
            continue
        return cast("torch.fx.Node", node), raw, norm_sum
    return None


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


def _indices(node: torch.fx.Node) -> tuple[object, ...] | None:
    if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
        return None
    return tuple(node.args[1])


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


def _static_int(value: object) -> int | None:
    import sympy

    if isinstance(value, int):
        return value
    if not isinstance(value, (torch.SymInt, sympy.Expr)):
        return None
    expr = value._sympy_() if isinstance(value, torch.SymInt) else value
    if not isinstance(expr, sympy.Expr):
        return None
    expr = CompileEnvironment.current().specialize_expr(expr)
    if expr.free_symbols:
        return None
    try:
        return int(expr)
    except TypeError:
        return None


def _contiguous_tail(tensor: torch.Tensor, rank: int) -> bool:
    expected = 1
    for dim in range(tensor.ndim - 1, tensor.ndim - rank - 1, -1):
        if _static_int(tensor.stride(dim)) != expected:
            return False
        size = _static_int(tensor.shape[dim])
        if size is None:
            return False
        expected *= size
    return True


def _reduction_signature(info: GraphInfo) -> tuple[tuple[int, ...] | None, ...] | None:
    calls = [node for node in info.graph.nodes if node.op == "call_function"]
    result: list[tuple[int, ...] | None] = []
    for node in calls:
        if node.kwargs:
            return None
        if node.target is torch.ops.aten.sum.default and len(node.args) == 1:
            result.append(None)
        elif (
            node.target is torch.ops.aten.sum.dim_IntList
            and len(node.args) == 2
            and isinstance(node.args[1], (list, tuple))
        ):
            result.append(tuple(node.args[1]))
        else:
            return None
    return tuple(result)


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


def _match_rank1_else(
    info: GraphInfo,
) -> (
    tuple[
        dict[str, _TensorRef],
        object,
        object,
        object,
        torch.fx.Node,
        torch.fx.Node,
        tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node],
    ]
    | None
):
    """Match the branch that performs the recurrence and both stores."""

    from ...language import memory_ops
    from ...language._tracing_ops import _new_var
    from ...language.tile_ops import tile_index
    from ..device_ir import ElseGraphInfo

    if not isinstance(info, ElseGraphInfo):
        return None
    nodes = list(info.graph.nodes)
    loads = [node for node in nodes if _is_call(node, memory_ops.load)]
    stores = [node for node in nodes if _is_call(node, memory_ops.store)]
    if (
        len(loads) != 8
        or len(stores) != 2
        or _has_unmatched_side_effect(nodes, set(stores))
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

    state_load: torch.fx.Node | None = None
    state_store: torch.fx.Node | None = None
    for load in loads:
        ref = _load_ref(load)
        matching_stores = [
            store for store in stores if _same_ref(_store_ref(store), ref)
        ]
        if len(matching_stores) == 1:
            if state_load is not None:
                return None
            state_load = load
            state_store = matching_stores[0]
    if state_load is None or state_store is None:
        return None
    output_store = stores[1] if stores[0] is state_store else stores[0]

    state_indices = _indices(state_load)
    state_store_indices = _indices(state_store)
    output_indices = _indices(output_store)
    if (
        state_indices is None
        or state_store_indices != state_indices
        or output_indices is None
        or len(state_indices) != 4
        or len(output_indices) != 4
        or output_indices[1] != 0
        or output_indices[2] is not state_indices[1]
        or output_indices[3] is not state_indices[2]
    ):
        return None

    placeholders = list(info.graph.find_nodes(op="placeholder"))
    if len(placeholders) != 2:
        return None
    iota_placeholder, state_index_placeholder = placeholders
    k_coord = _strip_state_index_casts(state_indices[3])
    state_index = _strip_state_index_casts(state_indices[0])
    if not isinstance(k_coord, torch.fx.Node) or not isinstance(
        state_index, torch.fx.Node
    ):
        return None
    if (
        not _is_call(k_coord, _new_var)
        or not k_coord.args
        or k_coord.args[0] is not iota_placeholder
        or not _is_call(state_index, _new_var)
        or not state_index.args
        or state_index.args[0] is not state_index_placeholder
    ):
        return None
    head_coord = state_indices[1]
    value_coord = state_indices[2]
    batch_coord = output_indices[0]
    if not all(
        isinstance(value, torch.fx.Node)
        for value in (head_coord, value_coord, batch_coord)
    ):
        return None
    assert isinstance(head_coord, torch.fx.Node)
    assert isinstance(value_coord, torch.fx.Node)
    assert isinstance(batch_coord, torch.fx.Node)

    state_fp32 = next(
        (
            node
            for node in state_load.users
            if _cast_from(node, state_load, torch.float32)
        ),
        None,
    )
    state_update = _strip_casts(
        state_store.args[2] if len(state_store.args) > 2 else None
    )
    update_pair = _binary_pair(state_update, _ADD_TARGETS)
    if (
        state_fp32 is None
        or state_update is None
        or update_pair is None
        or not _cast_from(state_store.args[2], state_update, torch.bfloat16)
    ):
        return None

    decayed: torch.fx.Node | None = None
    residual: torch.fx.Node | None = None
    normalized_key: torch.fx.Node | None = None
    key_broadcast: torch.fx.Node | None = None
    decay: torch.fx.Node | None = None
    for maybe_decayed, maybe_outer in (update_pair, reversed(update_pair)):
        decay_broadcast = _binary_other(maybe_decayed, state_fp32, _MUL_TARGETS)
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
            residual = maybe_residual
            normalized_key = maybe_key
            key_broadcast = maybe_key_broadcast
            decay = maybe_decay
            break
        if decayed is not None:
            break
    if any(
        value is None
        for value in (decayed, residual, normalized_key, key_broadcast, decay)
    ):
        return None
    assert decayed is not None
    assert residual is not None
    assert normalized_key is not None
    assert key_broadcast is not None
    assert decay is not None

    residual_pair = _binary_pair(residual, _MUL_TARGETS)
    if residual_pair is None:
        return None
    difference: torch.fx.Node | None = None
    beta_value: torch.fx.Node | None = None
    for maybe_difference, maybe_beta in (residual_pair, reversed(residual_pair)):
        if (
            maybe_difference.op == "call_function"
            and maybe_difference.target is torch.ops.aten.sub.Tensor
            and len(maybe_difference.args) == 2
            and not maybe_difference.kwargs
            and maybe_beta.op == "call_function"
            and maybe_beta.target is torch.ops.aten.sigmoid.default
            and len(maybe_beta.args) == 1
            and not maybe_beta.kwargs
        ):
            difference = maybe_difference
            beta_value = maybe_beta
            break
    if difference is None or beta_value is None or len(difference.args) < 2:
        return None
    value_fp32, prediction = difference.args[:2]
    if not isinstance(value_fp32, torch.fx.Node) or not isinstance(
        prediction, torch.fx.Node
    ):
        return None
    prediction_product = _sum_input(prediction, (-1,))
    prediction_pair = _binary_pair(prediction_product, _MUL_TARGETS)
    if prediction_pair is None or not any(
        maybe_decayed is decayed
        and _broadcast_source(maybe_key, (None, slice(None))) is normalized_key
        for maybe_decayed, maybe_key in (prediction_pair, reversed(prediction_pair))
    ):
        return None

    key_match = _match_normalized_vector(normalized_key)
    if key_match is None:
        return None
    _, key_fp32, key_norm_sum = key_match
    key_load = _strip_casts(key_fp32)
    value_load = _strip_casts(value_fp32)
    if (
        _load_ref(key_load) is None
        or _load_ref(value_load) is None
        or not isinstance(key_load, torch.fx.Node)
        or not isinstance(value_load, torch.fx.Node)
        or not _cast_from(key_fp32, key_load, torch.float32)
        or not _cast_from(value_fp32, value_load, torch.float32)
    ):
        return None
    assert isinstance(key_load, torch.fx.Node)
    assert isinstance(value_load, torch.fx.Node)

    beta_source = beta_value.args[0] if beta_value.args else None
    beta_load = _strip_casts(beta_source)
    if (
        _load_ref(beta_load) is None
        or not isinstance(beta_load, torch.fx.Node)
        or not _cast_from(beta_source, beta_load, torch.float32)
    ):
        return None

    decay_match = _match_decay(decay)
    if decay_match is None:
        return None
    gate_fp32, bias_value, a_log_value, lower_bound, log2_e = decay_match
    gate_load = _strip_casts(gate_fp32)
    bias_load = _strip_casts(bias_value)
    a_log_load = _strip_casts(a_log_value)
    if not all(
        isinstance(node, torch.fx.Node) and _load_ref(node) is not None
        for node in (gate_load, bias_load, a_log_load)
    ):
        return None
    assert isinstance(gate_load, torch.fx.Node)
    assert isinstance(bias_load, torch.fx.Node)
    assert isinstance(a_log_load, torch.fx.Node)
    if (
        not _cast_from(gate_fp32, gate_load, torch.float32)
        or bias_value is not bias_load
        or a_log_value is not a_log_load
    ):
        return None

    output_value = _strip_casts(
        output_store.args[2] if len(output_store.args) > 2 else None
    )
    if output_value is None:
        return None
    if not _cast_from(output_store.args[2], output_value, torch.bfloat16):
        return None
    scaled_query = _match_factored_output(
        output_value, decayed, residual, key_broadcast
    )
    if scaled_query is None:
        return None
    query_pair = _binary_pair(scaled_query, _MUL_TARGETS)
    if query_pair is None:
        return None
    query_normalized: torch.fx.Node | None = None
    scale: object | None = None
    for maybe_query, maybe_scale in (query_pair, reversed(query_pair)):
        if _match_normalized_vector(maybe_query) is not None:
            query_normalized = maybe_query
            scale = maybe_scale.meta.get("val")
            break
    if query_normalized is None or scale is None:
        return None
    query_match = _match_normalized_vector(query_normalized)
    assert query_match is not None
    _, query_fp32, query_norm_sum = query_match
    query_load = _strip_casts(query_fp32)
    if (
        _load_ref(query_load) is None
        or not isinstance(query_load, torch.fx.Node)
        or not _cast_from(query_fp32, query_load, torch.float32)
    ):
        return None

    mixed_ref = _load_ref(query_load)
    if not _same_ref(_load_ref(key_load), mixed_ref) or not _same_ref(
        _load_ref(value_load), mixed_ref
    ):
        return None

    expected_loads = {
        state_load,
        key_load,
        value_load,
        query_load,
        beta_load,
        gate_load,
        bias_load,
        a_log_load,
    }
    if set(loads) != expected_loads:
        return None

    # Prove the packed Q/K/V and gate address algebra.  This prevents a
    # source-compatible-looking recurrence with a shifted head or tensor slice
    # from entering the shape-specialized code generator.
    query_indices = _indices(query_load)
    key_indices = _indices(key_load)
    value_indices = _indices(value_load)
    gate_indices = _indices(gate_load)
    bias_indices = _indices(bias_load)
    beta_indices = _indices(beta_load)
    a_log_indices = _indices(a_log_load)
    if any(
        value is None
        for value in (
            query_indices,
            key_indices,
            value_indices,
            gate_indices,
            bias_indices,
            beta_indices,
            a_log_indices,
        )
    ):
        return None
    assert query_indices is not None
    assert key_indices is not None
    assert value_indices is not None
    assert gate_indices is not None
    assert bias_indices is not None
    assert beta_indices is not None
    assert a_log_indices is not None
    if (
        len(query_indices) != 2
        or len(key_indices) != 2
        or len(value_indices) != 2
        or len(gate_indices) != 2
        or len(bias_indices) != 1
        or len(beta_indices) != 2
        or len(a_log_indices) != 1
        or query_indices[0] is not batch_coord
        or key_indices[0] is not batch_coord
        or value_indices[0] is not batch_coord
        or gate_indices[0] is not batch_coord
        or beta_indices != (batch_coord, head_coord)
        or a_log_indices != (head_coord,)
    ):
        return None

    query_affine = _affine(query_indices[1])
    key_affine = _affine(key_indices[1])
    value_affine = _affine(value_indices[1])
    gate_affine = _affine(gate_indices[1])
    bias_affine = _affine(bias_indices[0])
    if any(
        value is None
        for value in (
            query_affine,
            key_affine,
            value_affine,
            gate_affine,
            bias_affine,
        )
    ):
        return None
    assert query_affine is not None
    assert key_affine is not None
    assert value_affine is not None
    assert gate_affine is not None
    assert bias_affine is not None
    if query_affine[0] or len(query_affine[1]) != 2:
        return None
    if query_affine[1].get(k_coord) != 1:
        return None
    query_head_nodes = [
        leaf
        for leaf, coefficient in query_affine[1].items()
        if leaf is not k_coord and coefficient == _KEY_SIZE
    ]
    if len(query_head_nodes) != 1:
        return None
    query_head = query_head_nodes[0]
    query_heads_nodes = [
        leaf
        for leaf, coefficient in key_affine[1].items()
        if leaf not in (k_coord, query_head) and coefficient == _KEY_SIZE
    ]
    if (
        key_affine[0]
        or key_affine[1].get(k_coord) != 1
        or key_affine[1].get(query_head) != _KEY_SIZE
        or len(query_heads_nodes) != 1
    ):
        return None
    query_heads = query_heads_nodes[0]
    expected_gate = (0, {k_coord: 1, head_coord: _KEY_SIZE})
    if gate_affine != expected_gate or bias_affine != expected_gate:
        return None
    # The free V coordinate reaches the packed offset through tile_index.
    value_leaves = value_affine[1]
    value_tile = next(
        (
            leaf
            for leaf in value_leaves
            if _is_call(leaf, tile_index) and value_leaves[leaf] == 1
        ),
        None,
    )
    if (
        value_tile is None
        or not value_tile.args
        or value_tile.args[0] is not value_coord
    ):
        return None
    if value_affine != (
        0,
        {
            value_tile: 1,
            head_coord: _VALUE_SIZE,
            query_heads: 2 * _KEY_SIZE,
        },
    ):
        return None

    state_ref = _load_ref(state_load)
    if state_ref is None or state_ref.fake.ndim < 3:
        return None
    value_heads = _static_int(state_ref.fake.shape[-3])
    if value_heads is None or value_heads <= 0:
        return None
    mixed_fake = cast("_TensorRef", mixed_ref).fake
    query_heads_value = query_heads.meta.get("val")
    query_head_value = query_head.meta.get("val")
    head_value = head_coord.meta.get("val")
    if not all(
        isinstance(value, (int, torch.SymInt))
        for value in (query_heads_value, query_head_value, head_value)
    ):
        return None
    env = CompileEnvironment.current()
    inferred_query_heads = (mixed_fake.shape[1] - value_heads * _VALUE_SIZE) // (
        2 * _KEY_SIZE
    )
    if not env.known_equal(
        cast("int | torch.SymInt", query_heads_value), inferred_query_heads
    ):
        return None
    expected_query_head = cast("int | torch.SymInt", head_value) // (
        value_heads // cast("int | torch.SymInt", query_heads_value)
    )
    if not env.known_equal(
        cast("int | torch.SymInt", query_head_value),
        expected_query_head,
    ):
        return None

    refs = {
        "mixed": cast("_TensorRef", mixed_ref),
        "gate": cast("_TensorRef", _load_ref(gate_load)),
        "beta": cast("_TensorRef", _load_ref(beta_load)),
        "a_log": cast("_TensorRef", _load_ref(a_log_load)),
        "bias": cast("_TensorRef", _load_ref(bias_load)),
        "state": state_ref,
        "output": cast("_TensorRef", _store_ref(output_store)),
    }
    return (
        refs,
        scale,
        lower_bound,
        log2_e,
        iota_placeholder,
        state_index_placeholder,
        (batch_coord, head_coord, value_coord),
    )


def _plan_single_token_rank1_recurrence(
    graphs: Sequence[GraphInfo],
    _tile_strategy: TileStrategyDispatch,
) -> CuteSingleTokenRank1Plan | None:
    from ...language import memory_ops
    from ...language._tracing_ops import _if
    from ...language.tile_ops import tile_id
    from ..device_function import DeviceFunction
    from ..device_ir import ElseGraphInfo
    from ..device_ir import IfGraphInfo
    from ..device_ir import ReductionLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    if not env.settings.fast_math or df.config.pid_type != "flat":
        return None
    capability = env.config_spec.target_device_capability
    # The packed FP32 Horner path uses Blackwell's ``fma_packed_f32x2``.
    if capability is None or capability < (10, 0):
        return None

    roots = [info for info in graphs if isinstance(info, RootGraphInfo)]
    if_graphs = [info for info in graphs if isinstance(info, IfGraphInfo)]
    else_graphs = [info for info in graphs if isinstance(info, ElseGraphInfo)]
    reductions = [info for info in graphs if isinstance(info, ReductionLoopGraphInfo)]
    if (
        len(graphs) != 7
        or len(roots) != 1
        or len(if_graphs) != 1
        or len(else_graphs) != 1
        or len(reductions) != 4
    ):
        return None
    reduction_signatures = sorted(
        (_reduction_signature(info) for info in reductions), key=repr
    )
    if reduction_signatures != sorted(
        ((None,), (None,), ((-1,),), ((1,), (1,))), key=repr
    ):
        return None
    root = roots[0]
    if_info = if_graphs[0]
    else_info = else_graphs[0]

    root_calls = [node for node in root.graph.nodes if _is_call(node, _if)]
    root_loads = [node for node in root.graph.nodes if _is_call(node, memory_ops.load)]
    if (
        len(root_calls) != 1
        or len(root_loads) != 1
        or len(root_loads[0].args) != 4
        or root_loads[0].args[2] is not None
        or root_loads[0].args[3] is not None
        or root_loads[0].kwargs
    ):
        return None
    branch = root_calls[0]
    if _has_unmatched_side_effect(list(root.graph.nodes), {branch}):
        return None
    if (
        len(branch.args) < 5
        or branch.args[1] != if_info.graph_id
        or branch.args[2] != else_info.graph_id
        or branch.args[3] != []
        or not isinstance(branch.args[4], (list, tuple))
        or len(branch.args[4]) != 2
    ):
        return None
    predicate = branch.args[0]
    if (
        not isinstance(predicate, torch.fx.Node)
        or predicate.op != "call_function"
        or predicate.target is not torch.ops.aten.lt.Scalar
        or len(predicate.args) < 2
        or predicate.args[1] != 0
    ):
        return None
    state_index_value = _strip_state_index_casts(predicate.args[0])
    if state_index_value is not root_loads[0]:
        return None
    root_index_ref = _load_ref(root_loads[0])
    root_indices = _indices(root_loads[0])
    if (
        root_index_ref is None
        or root_indices is None
        or len(root_indices) != 1
        or not _is_call(root_indices[0], tile_id)
        or root_index_ref.fake.ndim != 1
        or root_index_ref.fake.dtype not in (torch.int32, torch.int64)
    ):
        return None

    branch_args = branch.args[4]
    iota = branch_args[0]
    state_index_arg = branch_args[1]
    if not _is_standard_key_iota(iota) or state_index_arg is not predicate.args[0]:
        return None

    matched = _match_rank1_else(else_info)
    if matched is None:
        return None
    (
        refs,
        scale,
        lower_bound,
        log2_e,
        iota_placeholder,
        state_index_placeholder,
        grid_coords,
    ) = matched
    if not all(
        isinstance(value, (int, float, torch.SymFloat))
        for value in (scale, lower_bound, log2_e)
    ):
        return None
    # Device-IR's codegen graph copy intentionally retains the branch
    # ``node_args`` from the source graph (see NodeArgsGraphInfo.copy), so
    # object identity cannot be compared across that boundary.  Positional
    # placeholder order is the branch ABI and was proven in the branch matcher.
    if list(else_info.graph.find_nodes(op="placeholder")) != [
        iota_placeholder,
        state_index_placeholder,
    ]:
        return None

    if_nodes = list(if_info.graph.nodes)
    if_loads = [node for node in if_nodes if _is_call(node, memory_ops.load)]
    if_stores = [node for node in if_nodes if _is_call(node, memory_ops.store)]
    if (
        if_loads
        or len(if_stores) != 1
        or _has_unmatched_side_effect(if_nodes, set(if_stores))
        or not _same_ref(_store_ref(if_stores[0]), refs["output"])
        or len(if_stores[0].args) < 3
        or type(if_stores[0].args[2]) not in (int, float)
        or float(if_stores[0].args[2]) != 0.0
        or len(if_stores[0].args) != 4
        or if_stores[0].args[3] is not None
        or if_stores[0].kwargs
    ):
        return None
    zero_indices = _indices(if_stores[0])
    if (
        zero_indices is None
        or len(zero_indices) != 4
        or zero_indices[1] != 0
        or not all(
            isinstance(zero_indices[index], torch.fx.Node) for index in (0, 2, 3)
        )
    ):
        return None

    mixed = refs["mixed"].fake
    gate = refs["gate"].fake
    beta = refs["beta"].fake
    a_log = refs["a_log"].fake
    bias = refs["bias"].fake
    state = refs["state"].fake
    output = refs["output"].fake
    value_heads = _static_int(state.shape[-3]) if state.ndim >= 3 else None
    tensors = (mixed, gate, beta, a_log, bias, state, output, root_index_ref.fake)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        return None
    if (
        mixed.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or beta.dtype is not torch.bfloat16
        or state.dtype is not torch.bfloat16
        or output.dtype is not torch.bfloat16
        or a_log.dtype is not torch.float32
        or bias.dtype is not torch.float32
        or mixed.ndim != 2
        or gate.ndim != 2
        or beta.ndim != 2
        or a_log.ndim != 1
        or bias.ndim != 1
        or state.ndim != 4
        or output.ndim != 4
        or value_heads is None
        or value_heads <= 0
        or tuple(_static_int(size) for size in state.shape[-3:])
        != (value_heads, _VALUE_SIZE, _KEY_SIZE)
        or not _contiguous_tail(state, 3)
        or not all(_static_int(tensor.stride(-1)) == 1 for tensor in tensors)
    ):
        return None
    assert value_heads is not None
    # Both dedicated schedules index Q and K with the value-head coordinate.
    # Keep that topology explicit: grouped-query layouts continue through the
    # generic lowering instead of receiving incorrect packed-input offsets.
    mixed_width = value_heads * (2 * _KEY_SIZE + _VALUE_SIZE)
    gate_width = value_heads * _KEY_SIZE
    state_head_stride = _VALUE_SIZE * _KEY_SIZE
    state_span = value_heads * state_head_stride

    if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
        return None
    runtime_tensors = tuple(env.runtime_value_for_tensor(tensor) for tensor in tensors)
    if not all(isinstance(tensor, torch.Tensor) for tensor in runtime_tensors):
        return None
    (
        runtime_mixed,
        runtime_gate,
        runtime_beta,
        runtime_a_log,
        runtime_bias,
        runtime_state,
        runtime_output,
        runtime_indices,
    ) = cast("tuple[torch.Tensor, ...]", runtime_tensors)
    runtime_batch_size = runtime_mixed.shape[0]
    if (
        runtime_batch_size < 1
        or tuple(runtime_mixed.shape) != (runtime_batch_size, mixed_width)
        or tuple(runtime_gate.shape) != (runtime_batch_size, gate_width)
        or tuple(runtime_beta.shape) != (runtime_batch_size, value_heads)
        or tuple(runtime_a_log.shape) != (value_heads,)
        or tuple(runtime_bias.shape) != (gate_width,)
        or runtime_state.ndim != 4
        or runtime_state.shape[0] < 1
        or tuple(runtime_state.shape[1:]) != (value_heads, _VALUE_SIZE, _KEY_SIZE)
        or tuple(runtime_output.shape)
        != (runtime_batch_size, 1, value_heads, _VALUE_SIZE)
        or tuple(runtime_indices.shape) != (runtime_batch_size,)
        or runtime_mixed.stride(1) != 1
        or runtime_mixed.stride(0) < mixed_width
        or runtime_mixed.stride(0) * runtime_mixed.element_size() % 8
        or not runtime_gate.is_contiguous()
        or not runtime_beta.is_contiguous()
        or not runtime_a_log.is_contiguous()
        or not runtime_bias.is_contiguous()
        or tuple(runtime_state.stride()[1:]) != (state_head_stride, _KEY_SIZE, 1)
        or runtime_state.stride(0) < state_span
        or runtime_state.stride(0) * runtime_state.element_size() % 16
        or not runtime_output.is_contiguous()
        or not runtime_indices.is_contiguous()
    ):
        return None
    runtime_state_size = runtime_state.shape[0]
    runtime_state_stride0 = runtime_state.stride(0)
    max_state_element = (
        (runtime_state_size - 1) * runtime_state_stride0 + state_span - 1
    )
    if max_state_element > torch.iinfo(torch.uint32).max:
        return None

    max_i32 = torch.iinfo(torch.int32).max
    if any(
        maximum_offset > max_i32
        for maximum_offset in (
            (runtime_batch_size - 1) * runtime_mixed.stride(0) + mixed_width - 1,
            runtime_gate.numel() - 1,
            runtime_beta.numel() - 1,
            runtime_output.numel() - 1,
            runtime_indices.numel() - 1,
        )
    ):
        return None

    from .memory_ops import runtime_tensor_has_specialized_alignment
    from .memory_ops import runtime_tensors_are_proven_disjoint

    if not all(
        runtime_tensor_has_specialized_alignment(env, tensor, alignment)
        for tensor, alignment in (
            (mixed, 8),
            (gate, 8),
            (bias, 16),
            (state, 16),
        )
    ):
        return None
    read_tensors = (mixed, gate, beta, a_log, bias, root_index_ref.fake)
    if not all(
        runtime_tensors_are_proven_disjoint(env, written, read)
        for written in (state, output)
        for read in read_tensors
    ) or not runtime_tensors_are_proven_disjoint(env, state, output):
        return None

    device_ir = HostFunction.current().device_ir
    grid_coord_ids = _canonical_root_axis_ids(
        device_ir,
        root_phase_index=root.phase_index,
        coordinates=grid_coords,
        expected_extents=(mixed.shape[0], value_heads, _VALUE_SIZE),
    )
    if grid_coord_ids is None:
        return None
    resolved_grid_block_sizes = [
        df.resolved_block_size(block_id) for block_id in grid_coord_ids
    ]
    if resolved_grid_block_sizes[:2] != [1, 1]:
        return None
    tile_rows = resolved_grid_block_sizes[2]
    if runtime_batch_size == 1:
        # Preserve the existing B1 specialization and every config that used
        # it.  The legacy body owns exactly sixteen value rows per 256-thread
        # CTA and intentionally ignores the generic ``num_warps`` setting.
        if tile_rows != _VALUE_TILE:
            return None
        topology = "wide_b1"
        threads = _THREADS
    else:
        # For batched decode, one warp owns a complete value-row tile.  The
        # autotuner expresses that topology through CuTe's per-axis
        # ``num_threads`` surface.  Directly pinned configs written before that
        # surface existed still use ``num_warps=1``; accept that legacy signal
        # only when it is not contradicted by explicit per-axis threads.
        reduction_block_ids = [
            block.block_id for block in env.block_sizes if block.reduction
        ]
        explicit_threads = any(thread > 0 for thread in df.config.num_threads)
        explicit_one_warp = False
        if len(reduction_block_ids) == 1:
            explicit_one_warp = (
                env.config_spec.num_threads.config_get(
                    df.config.num_threads,
                    cast("int", grid_coord_ids[2]),
                    0,
                )
                == 1
                and env.config_spec.num_threads.config_get(
                    df.config.num_threads,
                    reduction_block_ids[0],
                    0,
                )
                == _ONE_WARP_THREADS
            )
        pinned_one_warp = not explicit_threads and df.config.num_warps == 1
        if (
            tile_rows not in _ONE_WARP_VALUE_TILES
            or not (explicit_one_warp or pinned_one_warp)
            or df.config.get("cute_async_load_stages", 0) != 0
        ):
            return None
        topology = "one_warp"
        threads = _ONE_WARP_THREADS
    root_tile = cast("torch.fx.Node", root_indices[0])
    if not root_tile.args:
        return None
    root_tile_block = root_tile.args[0]
    if isinstance(root_tile_block, torch.fx.Node):
        root_tile_block = root_tile_block.meta.get("val")
    if env.resolve_block_id(root_tile_block) != grid_coord_ids[0]:
        return None
    zero_coord_ids = tuple(
        env.resolve_block_id(cast("torch.fx.Node", zero_indices[index]).meta.get("val"))
        for index in (0, 2, 3)
    )
    if zero_coord_ids != grid_coord_ids:
        return None
    if topology == "one_warp" and not _xyz_grid_fits(
        (
            value_heads * (_VALUE_SIZE // tile_rows),
            runtime_batch_size,
            1,
        )
    ):
        return None

    return CuteSingleTokenRank1Plan(
        root_graph_id=root.graph_id,
        mixed=refs["mixed"],
        gate=refs["gate"],
        beta=refs["beta"],
        a_log=refs["a_log"],
        bias=refs["bias"],
        state=refs["state"],
        output=refs["output"],
        state_indices=root_index_ref,
        scale=scale,
        lower_bound=lower_bound,
        log2_e=log2_e,
        batch_size=runtime_batch_size,
        state_size=runtime_state_size,
        state_stride0=runtime_state_stride0,
        mixed_stride0=runtime_mixed.stride(0),
        value_heads=value_heads,
        tile_rows=tile_rows,
        threads=threads,
        topology=topology,
        batch_block_id=grid_coord_ids[0],
        head_block_id=grid_coord_ids[1],
        value_block_id=grid_coord_ids[2],
    )


def plan_single_token_rank1_recurrence(
    graphs: Sequence[GraphInfo],
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Install the rank-1 plan, or leave ordinary CuTe lowering untouched."""

    from ..device_function import DeviceFunction

    device_function = DeviceFunction.current()
    plan = _plan_single_token_rank1_recurrence(graphs, tile_strategy)
    device_function.cute_state.single_token_rank1_plan = plan
    if plan is not None:
        # These scalars control compile-time specialization below.  CuTe's
        # launcher keys Constexpr arguments by value, so the fast polynomial
        # is selected only for the exact lower-bound/log-base pair and a later
        # call with different values compiles the generic fast-math fallback
        # instead.
        device_function.promote_expr_arg_to_constexpr(plan.scale)
        device_function.promote_expr_arg_to_constexpr(plan.lower_bound)
        device_function.promote_expr_arg_to_constexpr(plan.log2_e)


def _tensor_arg(cg: GenerateAST, ref: _TensorRef) -> str:
    return cg.device_function.tensor_arg(ref.fake, prefer_name=ref.suggested_name).name


def _codegen_one_warp_single_token_rank1_recurrence(
    cg: GenerateAST,
    plan: CuteSingleTokenRank1Plan,
) -> bool:
    """Emit the packed one-warp schedule for batched single-token decode."""

    from ..program_id import XYZProgramIDs

    df = cg.device_function
    if df.pid is None:
        return False
    logical_block_ids = (
        plan.batch_block_id,
        plan.head_block_id,
        plan.value_block_id,
    )
    configured_pid_info = list(df.pid.pid_info)
    if len(configured_pid_info) != 3 or {
        info.block_id for info in configured_pid_info
    } != set(logical_block_ids):
        return False
    pid_info_by_block = {info.block_id: info for info in configured_pid_info}
    value_info = pid_info_by_block[plan.value_block_id]
    batch_info = pid_info_by_block[plan.batch_block_id]
    head_info = pid_info_by_block[plan.head_block_id]
    value_tiles = _VALUE_SIZE // plan.tile_rows
    # Reproduce the hand schedule's physical grid exactly: X combines head and
    # value tile, Y is batch, and Z is one.  Keeping this as a late physical-ID
    # rewrite preserves the already-proven logical B/H/V grid and avoids any
    # benchmark-name or fixed-B dispatch.
    launch_pid_info = [
        value_info._replace(
            block_size_var="1",
            numel=str(plan.value_heads * value_tiles),
        ),
        batch_info,
        head_info._replace(block_size_var="1", numel="1"),
    ]
    df.pid = XYZProgramIDs(pid_info=launch_pid_info)

    state_indices = _tensor_arg(cg, plan.state_indices)
    output = _tensor_arg(cg, plan.output)
    gate = _tensor_arg(cg, plan.gate)
    bias = _tensor_arg(cg, plan.bias)
    a_log = _tensor_arg(cg, plan.a_log)
    beta = _tensor_arg(cg, plan.beta)
    state = _tensor_arg(cg, plan.state)
    mixed = _tensor_arg(cg, plan.mixed)
    scale = df.literal_expr(plan.scale)
    lower_bound = df.literal_expr(plan.lower_bound)
    log2_e = df.literal_expr(plan.log2_e)
    state_index_loader = (
        "_cute_rank1_load_i32_nc"
        if plan.state_indices.fake.dtype is torch.int32
        else "_cute_rank1_load_i64_nc"
    )
    placeholder_args = {
        mixed,
        gate,
        beta,
        a_log,
        bias,
        state,
        output,
        state_indices,
    }
    template = _GeneratedCodeTemplate(
        "packed_rank1",
        (*tuple(argument.name for argument in df.arguments), *cg._extra_params),
        df.new_var,
    )
    state_indices = template.protect(state_indices)
    output = template.protect(output)
    gate = template.protect(gate)
    bias = template.protect(bias)
    a_log = template.protect(a_log)
    beta = template.protect(beta)
    state = template.protect(state)
    mixed = template.protect(mixed)
    scale = template.protect(scale)
    lower_bound = template.protect(lower_bound)
    log2_e = template.protect(log2_e)
    rows_per_lane_group = plan.tile_rows // 2

    preamble = f"""
packed_rank1_codegen_abi_version = {_SINGLE_TOKEN_RANK1_CODEGEN_ABI_VERSION}
packed_rank1_helper_abi_version = {_RANK1_HELPER_ABI_VERSION}
packed_rank1_state_bits = cute.make_rmem_tensor(
    ({rows_per_lane_group}, 4), cutlass.Uint32
)
packed_rank1_decayed_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
packed_rank1_query_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
packed_rank1_key_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
packed_rank1_decay_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
packed_rank1_decay_source = cute.make_rmem_tensor(2, cutlass.Uint32)
"""
    body = f"""
packed_rank1_work = cutlass.Int32(cute.arch.block_idx()[0])
packed_rank1_batch = cutlass.Int32(cute.arch.block_idx()[1])
packed_rank1_value_tile = packed_rank1_work % cutlass.Int32({value_tiles})
packed_rank1_head = packed_rank1_work // cutlass.Int32({value_tiles})
packed_rank1_thread = cutlass.Int32(cute.arch.thread_idx()[0])
packed_rank1_value_offset = packed_rank1_value_tile * cutlass.Int32({plan.tile_rows})
packed_rank1_key_lane = packed_rank1_thread % cutlass.Int32(16)
packed_rank1_value_lane = packed_rank1_thread // cutlass.Int32(16)
packed_rank1_source_base = packed_rank1_thread * cutlass.Int32(4)

packed_rank1_state_index = cutlass.Int64(
    {state_index_loader}({state_indices}.iterator + packed_rank1_batch)
)
packed_rank1_mixed_row = packed_rank1_batch * cutlass.Int32({plan.mixed_stride0})
packed_rank1_query_base = (
    packed_rank1_mixed_row
    + packed_rank1_head * cutlass.Int32({_KEY_SIZE})
)
packed_rank1_key_base = (
    packed_rank1_mixed_row
    + cutlass.Int32({plan.value_heads * _KEY_SIZE})
    + packed_rank1_head * cutlass.Int32({_KEY_SIZE})
)
packed_rank1_value_base = (
    packed_rank1_mixed_row
    + cutlass.Int32({2 * plan.value_heads * _KEY_SIZE})
    + packed_rank1_head * cutlass.Int32({_VALUE_SIZE})
)
packed_rank1_gate_base = (
    (packed_rank1_batch * cutlass.Int32({plan.value_heads}) + packed_rank1_head)
    * cutlass.Int32({_KEY_SIZE})
)

# Issue immutable input loads before the data-dependent state-slot accesses.
packed_rank1_query_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {mixed}.iterator
        + packed_rank1_query_base
        + packed_rank1_source_base,
        dtype=cutlass.Uint32,
    )
)
packed_rank1_key_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {mixed}.iterator
        + packed_rank1_key_base
        + packed_rank1_source_base,
        dtype=cutlass.Uint32,
    )
)
packed_rank1_gate_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {gate}.iterator
        + packed_rank1_gate_base
        + packed_rank1_source_base,
        dtype=cutlass.Uint32,
    )
)
packed_rank1_bias_raw = _cute_rank1_load_f32x4_nc(
    cute.recast_ptr(
        {bias}.iterator
        + packed_rank1_head * cutlass.Int32({_KEY_SIZE})
        + packed_rank1_source_base,
        dtype=cutlass.Float32,
    )
)
packed_rank1_a_log = _cute_rank1_load_f32_nc(
    {a_log}.iterator + packed_rank1_head
)

for packed_rank1_row in cutlass.range_constexpr({rows_per_lane_group}):
    packed_rank1_value_index = (
        packed_rank1_value_offset
        + packed_rank1_value_lane
        + cutlass.Int32(2) * packed_rank1_row
    )
    packed_rank1_state_offset = (
        cutlass.Uint32(packed_rank1_state_index)
        * cutlass.Uint32({plan.state_stride0})
        + cutlass.Uint32(packed_rank1_head)
        * cutlass.Uint32({_VALUE_SIZE * _KEY_SIZE})
        + cutlass.Uint32(packed_rank1_value_index)
        * cutlass.Uint32({_KEY_SIZE})
        + cutlass.Uint32(packed_rank1_key_lane) * cutlass.Uint32(8)
    )
    packed_rank1_state_ptr = cute.recast_ptr(
        {state}.iterator + packed_rank1_state_offset,
        dtype=cutlass.Uint32,
    )
    packed_rank1_state_vector = _cute_rank1_load_u32x4_if_valid(
        packed_rank1_state_ptr,
        packed_rank1_state_index,
        cutlass.Int64({plan.state_size}),
    )
    for packed_rank1_pair in cutlass.range_constexpr(4):
        packed_rank1_state_bits[packed_rank1_row, packed_rank1_pair] = (
            packed_rank1_state_vector[packed_rank1_pair]
        )

packed_rank1_value_loaded = cutlass.Float32(0.0)
if packed_rank1_thread < cutlass.Int32({plan.tile_rows}):
    packed_rank1_value_loaded = cutlass.Float32(
        _cute_rank1_load_u16_nc(
            {mixed}.iterator
            + packed_rank1_value_base
            + packed_rank1_value_offset
            + packed_rank1_thread
        ).bitcast(cutlass.BFloat16)
    )
packed_rank1_beta_raw = cutlass.Float32(
    _cute_rank1_load_u16_nc(
        {beta}.iterator
        + packed_rank1_batch * cutlass.Int32({plan.value_heads})
        + packed_rank1_head
    ).bitcast(cutlass.BFloat16)
)

packed_rank1_query_sum_lane = cutlass.Float32({_NORM_EPSILON / 32.0!r})
packed_rank1_key_sum_lane = cutlass.Float32({_NORM_EPSILON / 32.0!r})
packed_rank1_a = cute.math.exp2(
    packed_rank1_a_log * cutlass.Float32({log2_e}), fastmath=True
)
for packed_rank1_source_pair in cutlass.range_constexpr(2):
    packed_rank1_low_index = packed_rank1_source_pair * 2
    packed_rank1_query_low = cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_query_raw[packed_rank1_source_pair]
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_query_high = cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_query_raw[packed_rank1_source_pair]
            >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_key_low = cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_key_raw[packed_rank1_source_pair]
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_key_high = cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_key_raw[packed_rank1_source_pair]
            >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_gate_low = cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_gate_raw[packed_rank1_source_pair]
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_gate_high = cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_gate_raw[packed_rank1_source_pair]
            >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_gate_input_low = packed_rank1_a * (
        packed_rank1_gate_low
        + cutlass.Float32(packed_rank1_bias_raw[packed_rank1_low_index])
    )
    packed_rank1_gate_input_high = packed_rank1_a * (
        packed_rank1_gate_high
        + cutlass.Float32(packed_rank1_bias_raw[packed_rank1_low_index + 1])
    )
    packed_rank1_decay_low = cutlass.Float32(0.0)
    packed_rank1_decay_high = cutlass.Float32(0.0)
    if ({lower_bound} == -5.0) and ({log2_e} == {_LOG2_E!r}):
        packed_rank1_gate_tanh_low = cute.math.tanh(
            cutlass.Float32(0.5) * packed_rank1_gate_input_low,
            approx=True,
        )
        packed_rank1_gate_tanh_high = cute.math.tanh(
            cutlass.Float32(0.5) * packed_rank1_gate_input_high,
            approx=True,
        )
        packed_rank1_coefficients = {_DECAY_TANH_POLY_COEFFICIENTS!r}
        packed_rank1_decay_low = cutlass.Float32(packed_rank1_coefficients[7])
        packed_rank1_decay_high = cutlass.Float32(packed_rank1_coefficients[7])
        for packed_rank1_coefficient in cutlass.range_constexpr(6, -1, -1):
            packed_rank1_coefficient_value = cutlass.Float32(
                packed_rank1_coefficients[packed_rank1_coefficient]
            )
            packed_rank1_decay_low, packed_rank1_decay_high = (
                cute.arch.fma_packed_f32x2(
                    (packed_rank1_decay_low, packed_rank1_decay_high),
                    (
                        packed_rank1_gate_tanh_low,
                        packed_rank1_gate_tanh_high,
                    ),
                    (
                        packed_rank1_coefficient_value,
                        packed_rank1_coefficient_value,
                    ),
                    ftz=True,
                )
            )
    else:
        packed_rank1_sigmoid_low = _cute_sigmoid_approx_ftz_f32(
            packed_rank1_gate_input_low
        )
        packed_rank1_sigmoid_high = _cute_sigmoid_approx_ftz_f32(
            packed_rank1_gate_input_high
        )
        packed_rank1_decay_low = cute.math.exp2(
            cutlass.Float32({lower_bound})
            * packed_rank1_sigmoid_low
            * cutlass.Float32({log2_e}),
            fastmath=True,
        )
        packed_rank1_decay_high = cute.math.exp2(
            cutlass.Float32({lower_bound})
            * packed_rank1_sigmoid_high
            * cutlass.Float32({log2_e}),
            fastmath=True,
        )
    packed_rank1_decay_source[packed_rank1_source_pair] = (
        _cute_rank1_pack_bf16x2(
            packed_rank1_decay_low,
            packed_rank1_decay_high,
        )
    )
    packed_rank1_query_sum_lane = (
        packed_rank1_query_sum_lane
        + packed_rank1_query_low * packed_rank1_query_low
        + packed_rank1_query_high * packed_rank1_query_high
    )
    packed_rank1_key_sum_lane = (
        packed_rank1_key_sum_lane
        + packed_rank1_key_low * packed_rank1_key_low
        + packed_rank1_key_high * packed_rank1_key_high
    )

packed_rank1_query_sum = cutlass.Float32(
    cute.arch.warp_reduction_sum(
        packed_rank1_query_sum_lane, threads_in_group=32
    )
)
packed_rank1_key_sum = cutlass.Float32(
    cute.arch.warp_reduction_sum(
        packed_rank1_key_sum_lane, threads_in_group=32
    )
)
packed_rank1_query_scale = (
    cute.math.rsqrt(packed_rank1_query_sum, fastmath=True)
    * cutlass.Float32({scale})
)
packed_rank1_key_scale = cute.math.rsqrt(
    packed_rank1_key_sum, fastmath=True
)
packed_rank1_query_scale_packed = _cute_rank1_pack_bf16x2(
    packed_rank1_query_scale, packed_rank1_query_scale
)
packed_rank1_key_scale_packed = _cute_rank1_pack_bf16x2(
    packed_rank1_key_scale, packed_rank1_key_scale
)
packed_rank1_query_source = cute.make_rmem_tensor(2, cutlass.Uint32)
packed_rank1_key_source = cute.make_rmem_tensor(2, cutlass.Uint32)
for packed_rank1_source_pair in cutlass.range_constexpr(2):
    packed_rank1_query_source[packed_rank1_source_pair] = (
        _cute_rank1_mul_bf16x2(
            packed_rank1_query_raw[packed_rank1_source_pair],
            packed_rank1_query_scale_packed,
        )
    )
    packed_rank1_key_source[packed_rank1_source_pair] = (
        _cute_rank1_mul_bf16x2(
            packed_rank1_key_raw[packed_rank1_source_pair],
            packed_rank1_key_scale_packed,
        )
    )

for packed_rank1_pair in cutlass.range_constexpr(4):
    packed_rank1_source_lane = (
        packed_rank1_key_lane * cutlass.Int32(2)
        + cutlass.Int32(packed_rank1_pair // 2)
    )
    packed_rank1_source_pair = packed_rank1_pair % 2
    packed_rank1_query_cache[packed_rank1_pair] = cutlass.Uint32(
        cute.arch.shuffle_sync(
            packed_rank1_query_source[packed_rank1_source_pair],
            packed_rank1_source_lane,
            mask=0xFFFFFFFF,
        )
    )
    packed_rank1_key_cache[packed_rank1_pair] = cutlass.Uint32(
        cute.arch.shuffle_sync(
            packed_rank1_key_source[packed_rank1_source_pair],
            packed_rank1_source_lane,
            mask=0xFFFFFFFF,
        )
    )
    packed_rank1_decay_cache[packed_rank1_pair] = cutlass.Uint32(
        cute.arch.shuffle_sync(
            packed_rank1_decay_source[packed_rank1_source_pair],
            packed_rank1_source_lane,
            mask=0xFFFFFFFF,
        )
    )

packed_rank1_beta = cutlass.Float32(0.5) * (
    cute.math.tanh(
        cutlass.Float32(0.5) * packed_rank1_beta_raw,
        approx=True,
    )
    + cutlass.Float32(1.0)
)

for packed_rank1_row in cutlass.range_constexpr({rows_per_lane_group}):
    packed_rank1_prediction_pair = cutlass.Uint32(0)
    for packed_rank1_pair in cutlass.range_constexpr(4):
        packed_rank1_decayed_pair = _cute_rank1_mul_bf16x2(
            packed_rank1_state_bits[packed_rank1_row, packed_rank1_pair],
            packed_rank1_decay_cache[packed_rank1_pair],
        )
        packed_rank1_decayed_cache[packed_rank1_pair] = packed_rank1_decayed_pair
        packed_rank1_prediction_pair = _cute_rank1_fma_bf16x2(
            packed_rank1_decayed_pair,
            packed_rank1_key_cache[packed_rank1_pair],
            packed_rank1_prediction_pair,
        )
    for packed_rank1_step in cutlass.range_constexpr(4):
        packed_rank1_prediction_other = cutlass.Uint32(
            cute.arch.shuffle_sync_bfly(
                packed_rank1_prediction_pair,
                cutlass.Int32(8 >> packed_rank1_step),
                mask=0xFFFFFFFF,
            )
        )
        packed_rank1_prediction_pair = _cute_rank1_add_bf16x2(
            packed_rank1_prediction_pair,
            packed_rank1_prediction_other,
        )
    packed_rank1_prediction = cutlass.Float32(
        cutlass.Uint16(packed_rank1_prediction_pair).bitcast(cutlass.BFloat16)
    ) + cutlass.Float32(
        cutlass.Uint16(
            packed_rank1_prediction_pair >> cutlass.Uint32(16)
        ).bitcast(cutlass.BFloat16)
    )
    packed_rank1_value = cutlass.Float32(
        cute.arch.shuffle_sync(
            packed_rank1_value_loaded,
            packed_rank1_value_lane
            + cutlass.Int32(2) * packed_rank1_row,
            mask=0xFFFFFFFF,
        )
    )
    packed_rank1_delta = (
        packed_rank1_value - packed_rank1_prediction
    ) * packed_rank1_beta
    packed_rank1_delta_packed = _cute_rank1_pack_bf16x2(
        packed_rank1_delta, packed_rank1_delta
    )
    packed_rank1_output_pair = cutlass.Uint32(0)
    for packed_rank1_pair in cutlass.range_constexpr(4):
        packed_rank1_updated_pair = _cute_rank1_fma_bf16x2(
            packed_rank1_key_cache[packed_rank1_pair],
            packed_rank1_delta_packed,
            packed_rank1_decayed_cache[packed_rank1_pair],
        )
        packed_rank1_state_bits[packed_rank1_row, packed_rank1_pair] = (
            packed_rank1_updated_pair
        )
        packed_rank1_output_pair = _cute_rank1_fma_bf16x2(
            packed_rank1_updated_pair,
            packed_rank1_query_cache[packed_rank1_pair],
            packed_rank1_output_pair,
        )
    for packed_rank1_step in cutlass.range_constexpr(4):
        packed_rank1_output_other = cutlass.Uint32(
            cute.arch.shuffle_sync_bfly(
                packed_rank1_output_pair,
                cutlass.Int32(8 >> packed_rank1_step),
                mask=0xFFFFFFFF,
            )
        )
        packed_rank1_output_pair = _cute_rank1_add_bf16x2(
            packed_rank1_output_pair,
            packed_rank1_output_other,
        )
    packed_rank1_output_pair = _cute_rank1_add_bf16x2(
        packed_rank1_output_pair,
        packed_rank1_output_pair >> cutlass.Uint32(16),
    )
    if packed_rank1_key_lane == packed_rank1_row:
        packed_rank1_value_index = (
            packed_rank1_value_offset
            + packed_rank1_value_lane
            + cutlass.Int32(2) * packed_rank1_row
        )
        packed_rank1_output_offset = (
            (packed_rank1_batch * cutlass.Int32({plan.value_heads})
             + packed_rank1_head)
            * cutlass.Int32({_VALUE_SIZE})
            + packed_rank1_value_index
        )
        _cute_rank1_store_u16_or_zero(
            {output}.iterator,
            cutlass.Uint32(packed_rank1_output_offset),
            cutlass.Uint16(packed_rank1_output_pair),
            packed_rank1_state_index,
            cutlass.Int64({plan.state_size}),
        )

for packed_rank1_row in cutlass.range_constexpr({rows_per_lane_group}):
    packed_rank1_value_index = (
        packed_rank1_value_offset
        + packed_rank1_value_lane
        + cutlass.Int32(2) * packed_rank1_row
    )
    packed_rank1_state_offset = (
        cutlass.Uint32(packed_rank1_state_index)
        * cutlass.Uint32({plan.state_stride0})
        + cutlass.Uint32(packed_rank1_head)
        * cutlass.Uint32({_VALUE_SIZE * _KEY_SIZE})
        + cutlass.Uint32(packed_rank1_value_index)
        * cutlass.Uint32({_KEY_SIZE})
        + cutlass.Uint32(packed_rank1_key_lane) * cutlass.Uint32(8)
    )
    packed_rank1_state_ptr = cute.recast_ptr(
        {state}.iterator + packed_rank1_state_offset,
        dtype=cutlass.Uint32,
    )
    _cute_rank1_store_u32x4_if_valid(
        packed_rank1_state_ptr,
        packed_rank1_state_bits[packed_rank1_row, 0],
        packed_rank1_state_bits[packed_rank1_row, 1],
        packed_rank1_state_bits[packed_rank1_row, 2],
        packed_rank1_state_bits[packed_rank1_row, 3],
        packed_rank1_state_index,
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


def codegen_single_token_rank1_recurrence(cg: GenerateAST) -> bool:
    """Replace the matched root with the packed 16-lane rank-1 schedule."""

    df = cg.device_function
    plan = df.cute_state.single_token_rank1_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.root_graph_id:
        return False
    if plan.topology == "one_warp":
        return _codegen_one_warp_single_token_rank1_recurrence(cg, plan)

    # Keep this specialization's ABI aligned with the compact reference
    # schedule.  Tensor arguments retain insertion order in ``sorted_args``;
    # this order reduces address-setup live ranges in ptxas.
    state_indices = _tensor_arg(cg, plan.state_indices)
    output = _tensor_arg(cg, plan.output)
    gate = _tensor_arg(cg, plan.gate)
    bias = _tensor_arg(cg, plan.bias)
    a_log = _tensor_arg(cg, plan.a_log)
    beta = _tensor_arg(cg, plan.beta)
    state = _tensor_arg(cg, plan.state)
    mixed = _tensor_arg(cg, plan.mixed)
    state_size = plan.state_size
    state_stride0 = plan.state_stride0
    scale = df.literal_expr(plan.scale)
    lower_bound = df.literal_expr(plan.lower_bound)
    log2_e = df.literal_expr(plan.log2_e)
    state_index_loader = (
        "_cute_rank1_load_i32_nc"
        if plan.state_indices.fake.dtype is torch.int32
        else "_cute_rank1_load_i64_nc"
    )
    placeholder_args = {
        mixed,
        gate,
        beta,
        a_log,
        bias,
        state,
        output,
        state_indices,
    }
    template = _GeneratedCodeTemplate(
        "rank1",
        (*tuple(argument.name for argument in df.arguments), *cg._extra_params),
        df.new_var,
    )
    state_indices = template.protect(state_indices)
    output = template.protect(output)
    gate = template.protect(gate)
    bias = template.protect(bias)
    a_log = template.protect(a_log)
    beta = template.protect(beta)
    state = template.protect(state)
    mixed = template.protect(mixed)
    scale = template.protect(scale)
    lower_bound = template.protect(lower_bound)
    log2_e = template.protect(log2_e)

    preamble = f"""
rank1_helper_abi_version = {_RANK1_HELPER_ABI_VERSION}
rank1_k_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
rank1_q_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
rank1_decay_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
rank1_decayed_cache = cute.make_rmem_tensor(4, cutlass.Uint32)
rank1_state_bits = cute.make_rmem_tensor(4, cutlass.Uint32)
rank1_gate_source = cute.make_rmem_tensor(4, cutlass.Float32)
rank1_bias_source = cute.make_rmem_tensor(4, cutlass.Float32)
rank1_decay_source = cute.make_rmem_tensor(4, cutlass.Float32)
"""
    body = f"""
rank1_work = cutlass.Int32(cute.arch.block_idx()[0])
rank1_head = rank1_work // cutlass.Int32(8)
rank1_linear_thread = cutlass.Int32(cute.arch.thread_idx()[0])
rank1_key_lane = rank1_linear_thread % cutlass.Int32(16)
rank1_group = rank1_linear_thread // cutlass.Int32(16)
rank1_global_row = rank1_work * cutlass.Int32({_VALUE_TILE}) + rank1_group
rank1_state_index = cutlass.Int64(
    {state_index_loader}({state_indices}.iterator)
)

# These tensors are immutable during the kernel.  Issue their loads after the
# slot lookup but before branching on its value so a data-dependent state-slot
# guard cannot serialize the independent input-memory pipeline.  State itself
# remains strictly below the bounds check.
rank1_warp_lane = rank1_linear_thread % cutlass.Int32(32)
rank1_source_base = rank1_warp_lane * cutlass.Int32(4)
rank1_query_base = rank1_head * cutlass.Int32({_KEY_SIZE})
rank1_key_input_base = cutlass.Int32({plan.value_heads * _KEY_SIZE}) + rank1_query_base
rank1_value_input = cutlass.Int32({2 * plan.value_heads * _KEY_SIZE}) + rank1_global_row

rank1_query_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {mixed}.iterator
        + rank1_query_base
        + rank1_source_base,
        dtype=cutlass.Uint32,
    )
)
rank1_key_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {mixed}.iterator
        + rank1_key_input_base
        + rank1_source_base,
        dtype=cutlass.Uint32,
    )
)
rank1_gate_raw = _cute_rank1_load_u32x2_nc(
    cute.recast_ptr(
        {gate}.iterator
        + rank1_head * cutlass.Int32({_KEY_SIZE})
        + rank1_source_base,
        dtype=cutlass.Uint32,
    )
)
rank1_bias_raw = _cute_rank1_load_f32x4_nc(
    cute.recast_ptr(
        {bias}.iterator
        + rank1_head * cutlass.Int32({_KEY_SIZE})
        + rank1_source_base,
        dtype=cutlass.Float32,
    )
)
rank1_a_log = _cute_rank1_load_f32_nc({a_log}.iterator + rank1_head)
rank1_value = cutlass.Float32(0.0)
if rank1_key_lane == 0:
    rank1_value = cutlass.Float32(
        _cute_rank1_load_u16_nc(
            {mixed}.iterator
            + rank1_value_input
        ).bitcast(cutlass.BFloat16)
    )
rank1_value_source_lane = (rank1_group % cutlass.Int32(2)) * cutlass.Int32(16)
rank1_value = cutlass.Float32(
    cute.arch.shuffle_sync(rank1_value, rank1_value_source_lane)
)
if True:
    rank1_state_offset = (
        cutlass.Uint32(rank1_state_index) * cutlass.Uint32({state_stride0})
        + cutlass.Uint32(rank1_work) * cutlass.Uint32(2048)
        + cutlass.Uint32(rank1_linear_thread) * cutlass.Uint32(8)
    )
    rank1_state_ptr = cute.recast_ptr(
        {state}.iterator + rank1_state_offset, dtype=cutlass.Uint32
    )
    rank1_state_vector = _cute_rank1_load_u32x4_if_valid(
        rank1_state_ptr, rank1_state_index, cutlass.Int64({state_size})
    )
    for rank1_pair in cutlass.range_constexpr(4):
        rank1_state_bits[rank1_pair] = rank1_state_vector[rank1_pair]

    rank1_query_sum_lane = cutlass.Float32({_NORM_EPSILON / 32.0!r})
    rank1_key_sum_lane = cutlass.Float32({_NORM_EPSILON / 32.0!r})
    for rank1_element in cutlass.range_constexpr(4):
        rank1_source_pair = rank1_element // 2
        rank1_shift = cutlass.Uint32((rank1_element % 2) * 16)
        rank1_query_value = cutlass.Float32(
            cutlass.Uint16(rank1_query_raw[rank1_source_pair] >> rank1_shift).bitcast(cutlass.BFloat16)
        )
        rank1_key_value = cutlass.Float32(
            cutlass.Uint16(rank1_key_raw[rank1_source_pair] >> rank1_shift).bitcast(cutlass.BFloat16)
        )
        rank1_gate_source[rank1_element] = cutlass.Float32(
            cutlass.Uint16(rank1_gate_raw[rank1_source_pair] >> rank1_shift).bitcast(cutlass.BFloat16)
        )
        rank1_bias_source[rank1_element] = cutlass.Float32(rank1_bias_raw[rank1_element])
        rank1_query_sum_lane = rank1_query_sum_lane + rank1_query_value * rank1_query_value
        rank1_key_sum_lane = rank1_key_sum_lane + rank1_key_value * rank1_key_value

    rank1_query_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(rank1_query_sum_lane, threads_in_group=32)
    )
    rank1_key_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(rank1_key_sum_lane, threads_in_group=32)
    )
    rank1_query_scale = cute.math.rsqrt(rank1_query_sum, fastmath=True) * cutlass.Float32({scale})
    rank1_key_scale = cute.math.rsqrt(rank1_key_sum, fastmath=True)
    rank1_query_scale_packed = _cute_rank1_pack_bf16x2(
        rank1_query_scale, rank1_query_scale
    )
    rank1_key_scale_packed = _cute_rank1_pack_bf16x2(
        rank1_key_scale, rank1_key_scale
    )

    rank1_a = cute.math.exp2(rank1_a_log * cutlass.Float32({log2_e}), fastmath=True)
    if ({lower_bound} == -5.0) and ({log2_e} == {_LOG2_E!r}):
        for rank1_gate_pair in cutlass.range_constexpr(2):
            rank1_gate_low = rank1_gate_pair * 2
            rank1_gate_high = rank1_gate_low + 1
            rank1_gate_input_low = rank1_a * (
                rank1_gate_source[rank1_gate_low]
                + rank1_bias_source[rank1_gate_low]
            )
            rank1_gate_input_high = rank1_a * (
                rank1_gate_source[rank1_gate_high]
                + rank1_bias_source[rank1_gate_high]
            )
            rank1_gate_tanh_low = cute.math.tanh(
                cutlass.Float32(0.5) * cutlass.Float32(rank1_gate_input_low),
                approx=True,
            )
            rank1_gate_tanh_high = cute.math.tanh(
                cutlass.Float32(0.5) * cutlass.Float32(rank1_gate_input_high),
                approx=True,
            )
            rank1_coefficients = {_DECAY_TANH_POLY_COEFFICIENTS!r}
            rank1_poly_low = cutlass.Float32(rank1_coefficients[7])
            rank1_poly_high = cutlass.Float32(rank1_coefficients[7])
            for rank1_coefficient in cutlass.range_constexpr(6, -1, -1):
                rank1_coefficient_value = cutlass.Float32(
                    rank1_coefficients[rank1_coefficient]
                )
                rank1_poly_low, rank1_poly_high = cute.arch.fma_packed_f32x2(
                    (rank1_poly_low, rank1_poly_high),
                    (
                        rank1_gate_tanh_low,
                        rank1_gate_tanh_high,
                    ),
                    (rank1_coefficient_value, rank1_coefficient_value),
                    ftz=True,
                )
            rank1_decay_source[rank1_gate_low] = rank1_poly_low
            rank1_decay_source[rank1_gate_high] = rank1_poly_high
    else:
        for rank1_element in cutlass.range_constexpr(4):
            rank1_gate_input = rank1_a * (
                rank1_gate_source[rank1_element]
                + rank1_bias_source[rank1_element]
            )
            rank1_sigmoid = _cute_sigmoid_approx_ftz_f32(
                rank1_gate_input
            )
            rank1_decay_source[rank1_element] = cute.math.exp2(
                cutlass.Float32({lower_bound})
                * rank1_sigmoid
                * cutlass.Float32({log2_e}),
                fastmath=True,
            )

    rank1_query_source_packed = cute.make_rmem_tensor(2, cutlass.Uint32)
    rank1_key_source_packed = cute.make_rmem_tensor(2, cutlass.Uint32)
    rank1_decay_source_packed = cute.make_rmem_tensor(2, cutlass.Uint32)
    for rank1_source_pair in cutlass.range_constexpr(2):
        rank1_query_source_packed[rank1_source_pair] = _cute_rank1_mul_bf16x2(
            rank1_query_raw[rank1_source_pair], rank1_query_scale_packed
        )
        rank1_key_source_packed[rank1_source_pair] = _cute_rank1_mul_bf16x2(
            rank1_key_raw[rank1_source_pair], rank1_key_scale_packed
        )
        rank1_low = rank1_source_pair * 2
        rank1_decay_source_packed[rank1_source_pair] = _cute_rank1_pack_bf16x2(
            rank1_decay_source[rank1_low], rank1_decay_source[rank1_low + 1]
        )

    for rank1_pair in cutlass.range_constexpr(4):
        rank1_source_lane = rank1_key_lane * 2 + rank1_pair // 2
        rank1_source_pair = rank1_pair % 2
        rank1_q_cache[rank1_pair] = cutlass.Uint32(
            cute.arch.shuffle_sync(
                rank1_query_source_packed[rank1_source_pair], rank1_source_lane
            )
        )
        rank1_k_cache[rank1_pair] = cutlass.Uint32(
            cute.arch.shuffle_sync(
                rank1_key_source_packed[rank1_source_pair], rank1_source_lane
            )
        )
        rank1_decay_cache[rank1_pair] = cutlass.Uint32(
            cute.arch.shuffle_sync(
                rank1_decay_source_packed[rank1_source_pair], rank1_source_lane
            )
        )

    rank1_beta_raw = cutlass.Float32(
        _cute_rank1_load_u16_nc(
            {beta}.iterator
            + rank1_head
        ).bitcast(cutlass.BFloat16)
    )
    rank1_beta = cutlass.Float32(0.5) * (
        cute.math.tanh(cutlass.Float32(0.5) * rank1_beta_raw, approx=True)
        + cutlass.Float32(1.0)
    )

    rank1_prediction_pair = cutlass.Uint32(0)
    for rank1_pair in cutlass.range_constexpr(4):
        rank1_decayed_pair = _cute_rank1_mul_bf16x2(
            rank1_state_bits[rank1_pair], rank1_decay_cache[rank1_pair]
        )
        rank1_decayed_cache[rank1_pair] = rank1_decayed_pair
        rank1_prediction_pair = _cute_rank1_fma_bf16x2(
            rank1_decayed_pair,
            rank1_k_cache[rank1_pair],
            rank1_prediction_pair,
        )
    for rank1_step in cutlass.range_constexpr(4):
        rank1_prediction_other = cutlass.Uint32(
            cute.arch.shuffle_sync_bfly(
                rank1_prediction_pair, cutlass.Int32(8 >> rank1_step)
            )
        )
        rank1_prediction_pair = _cute_rank1_add_bf16x2(
            rank1_prediction_pair, rank1_prediction_other
        )
    rank1_prediction = cutlass.Float32(
        cutlass.Uint16(rank1_prediction_pair).bitcast(cutlass.BFloat16)
    ) + cutlass.Float32(
        cutlass.Uint16(rank1_prediction_pair >> cutlass.Uint32(16)).bitcast(cutlass.BFloat16)
    )
    rank1_delta = (rank1_value - rank1_prediction) * rank1_beta
    rank1_delta_packed = _cute_rank1_pack_bf16x2(rank1_delta, rank1_delta)
    rank1_output_pair = cutlass.Uint32(0)
    rank1_updated = cute.make_rmem_tensor(4, cutlass.Uint32)
    for rank1_pair in cutlass.range_constexpr(4):
        rank1_updated_pair = _cute_rank1_fma_bf16x2(
            rank1_k_cache[rank1_pair],
            rank1_delta_packed,
            rank1_decayed_cache[rank1_pair],
        )
        rank1_updated[rank1_pair] = rank1_updated_pair
        rank1_output_pair = _cute_rank1_fma_bf16x2(
            rank1_updated_pair,
            rank1_q_cache[rank1_pair],
            rank1_output_pair,
        )

    _cute_rank1_store_u32x4_if_valid(
        rank1_state_ptr,
        rank1_updated[0],
        rank1_updated[1],
        rank1_updated[2],
        rank1_updated[3],
        rank1_state_index,
        cutlass.Int64({state_size}),
    )
    for rank1_step in cutlass.range_constexpr(4):
        rank1_output_other = cutlass.Uint32(
            cute.arch.shuffle_sync_bfly(
                rank1_output_pair, cutlass.Int32(8 >> rank1_step)
            )
        )
        rank1_output_pair = _cute_rank1_add_bf16x2(
            rank1_output_pair, rank1_output_other
        )
    rank1_output_pair = _cute_rank1_add_bf16x2(
        rank1_output_pair, rank1_output_pair >> cutlass.Uint32(16)
    )
    if rank1_key_lane == 0:
        _cute_rank1_store_u16_or_zero(
            {output}.iterator,
            cutlass.Uint32(rank1_global_row),
            cutlass.Uint16(rank1_output_pair),
            rank1_state_index,
            cutlass.Int64({state_size}),
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
