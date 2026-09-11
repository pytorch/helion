"""Whole-root lowering for the five-factor BT16 chunk-prepare graph.

The schedule is intentionally narrow and fail closed.  It recognizes the
semantic graph (including all rounding boundaries and physical output
permutations), then replaces the complete root with the independently verified
split-alias CuTe implementation in :mod:`chunk_prepare_split_alias_device`.
That schedule deliberately overlaps the dead raw Q/K buffers with Qd/Ki; its
extra release barriers are part of the lifetime proof and are not optional
variants of the removed unaliased schedule.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...autotuner.config_spec import CUTE_CHUNK_PREPARE_SCHEDULE_KEY
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _linear_offsets_fit_i32
from .fx_matcher import _xyz_grid_fits

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    import sympy
    from torch._guards import Source

    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


_BT = 16
_DK = 128
_SEMANTIC_CPC = 4
_SCHEDULE_CPC_CHOICES = (1, 2, 3, 4, 5)
_DEFAULT_SPLIT_ALIAS_CPC = 4
_TARGET_SPLIT_ALIAS_WAVES = 8
_THREADS = 128
_SPLIT_ALIAS_SMEM_BYTES = 21_968
_SPLIT_ALIAS_SCHEDULES = tuple(
    f"split_alias_cpc{chunks_per_cta}" for chunks_per_cta in _SCHEDULE_CPC_CHOICES
)
_VALID_PREPARE_SCHEDULES = _SPLIT_ALIAS_SCHEDULES
_BF16_MMA_COUNT = 168
_FP16_MMA_COUNT = 24
_SCALED_DEVICE_ABI = 3
_UNSCALED_DEVICE_ABI = 4
_LOG2_E = 1.4426950408889634
_PACKED_WORKSPACE_SPECIALIZATION_KEY = "cute_kda_packed_workspace_v1"


@dataclasses.dataclass(frozen=True)
class _TensorRef:
    fake: torch.Tensor
    name: str


@dataclasses.dataclass(frozen=True)
class CuteChunkPreparePlan:
    """Compile-time contract for one exact BT16/K128 prepare root.

    ``device_abi`` versions the callable schedule. Workspace compatibility is
    described independently by ``outputs_scaled`` and ``factor_key_xor``.
    """

    root_graph_id: int
    chunk_size: int
    key_size: int
    chunks_per_cta: int
    schedule: str
    heads: int
    total_tokens: int
    total_chunks: int
    gate_is_fp32: bool
    outputs_scaled: bool
    factor_key_xor: int
    q: _TensorRef
    k: _TensorRef
    gate: _TensorRef
    beta: _TensorRef
    a_log: _TensorRef
    dt_bias: _TensorRef
    cu_seqlens: _TensorRef
    cu_chunks: _TensorRef
    chunk_to_seq: _TensorRef
    kd: _TensorRef
    qd: _TensorRef
    ak: _TensorRef
    aq: _TensorRef
    g_total: _TensorRef
    scale: sympy.Expr
    gate_scale_log2: sympy.Expr
    threads: int = _THREADS
    smem_bytes: int = _SPLIT_ALIAS_SMEM_BYTES
    bf16_mma_count: int = _BF16_MMA_COUNT
    fp16_mma_count: int = _FP16_MMA_COUNT
    device_abi: int = _SCALED_DEVICE_ABI


@dataclasses.dataclass(frozen=True)
class _CuteChunkPrepareMatch:
    """Exact semantic/storage match shared by planning and autotune gating."""

    root_graph_id: int
    root_phase_index: int
    heads: int
    total_tokens: int
    total_chunks: int
    gate_is_fp32: bool
    outputs_scaled: bool
    factor_key_xor: int
    q: _TensorRef
    k: _TensorRef
    gate: _TensorRef
    beta: _TensorRef
    a_log: _TensorRef
    dt_bias: _TensorRef
    cu_seqlens: _TensorRef
    cu_chunks: _TensorRef
    chunk_to_seq: _TensorRef
    kd: _TensorRef
    qd: _TensorRef
    ak: _TensorRef
    aq: _TensorRef
    g_total: _TensorRef
    chunk_group_coord: torch.fx.Node
    head_coord: torch.fx.Node
    scale: sympy.Expr
    gate_scale_log2: sympy.Expr


def _preferred_split_alias_chunks_per_cta(
    *, total_chunks: int, heads: int, num_sm: int
) -> int:
    """Seed enough chunk grouping to target eight CTA waves.

    This geometry-only seed produces CPC1 for underfilled launches and grows
    through CPC5 as aggregate work increases. Full autotuning still evaluates
    every legal split-alias CPC schedule.
    """

    if total_chunks <= 0 or heads <= 0 or num_sm <= 0:
        return _DEFAULT_SPLIT_ALIAS_CPC
    work_items = total_chunks * heads
    target_capacity = _TARGET_SPLIT_ALIAS_WAVES * num_sm
    chunks_per_cta = (work_items + target_capacity - 1) // target_capacity
    return max(
        _SCHEDULE_CPC_CHOICES[0],
        min(_SCHEDULE_CPC_CHOICES[-1], chunks_per_cta),
    )


def _static_extent(value: object) -> int | None:
    import sympy

    from ..host_function import HostFunction
    from ..variable_origin import AttributeOrigin
    from ..variable_origin import GlobalOrigin

    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if not isinstance(value, (int, sympy.Expr)):
        return None
    if isinstance(value, int):
        return value
    env = CompileEnvironment.current()
    expr = env.specialize_expr(sympy.sympify(value))
    replacements: dict[sympy.Basic, int] = {}
    for symbol in expr.free_symbols:
        origin_info = HostFunction.current().expr_to_origin.get(symbol)
        if origin_info is None:
            continue
        origin = origin_info.origin
        if isinstance(origin, AttributeOrigin) and isinstance(
            origin.value, GlobalOrigin
        ):
            constant = HostFunction.current().fn.__globals__.get(origin.key)
            if type(constant) is int:
                replacements[symbol] = constant
    if replacements:
        expr = expr.xreplace(replacements)
    if getattr(expr, "free_symbols", None):
        return None
    try:
        return int(expr)
    except TypeError:
        return None


def _static_float(value: object) -> float | None:
    """Resolve a specialized scalar without accepting a live symbolic value."""

    import sympy

    from ..host_function import HostFunction
    from ..variable_origin import AttributeOrigin
    from ..variable_origin import GlobalOrigin

    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.SymFloat):
        value = value._sympy_()
    if not isinstance(value, sympy.Expr):
        return None
    env = CompileEnvironment.current()
    expr = env.specialize_expr(value)
    replacements: dict[sympy.Basic, float] = {}
    for symbol in expr.free_symbols:
        origin_info = HostFunction.current().expr_to_origin.get(symbol)
        if origin_info is None:
            continue
        origin = origin_info.origin
        if isinstance(origin, AttributeOrigin) and isinstance(
            origin.value, GlobalOrigin
        ):
            constant = HostFunction.current().fn.__globals__.get(origin.key)
            if isinstance(constant, (int, float)):
                replacements[symbol] = float(constant)
    if replacements:
        expr = expr.xreplace(replacements)
    if isinstance(expr, (int, float)):
        return float(expr)
    if expr.free_symbols:
        return None
    try:
        return float(expr)
    except TypeError:
        return None


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


def _ref_key(ref: _TensorRef) -> tuple[int, str]:
    return id(ref.fake), ref.name


def _unique_refs(refs: Iterable[_TensorRef]) -> list[_TensorRef]:
    result: dict[tuple[int, str], _TensorRef] = {}
    for ref in refs:
        result.setdefault(_ref_key(ref), ref)
    return list(result.values())


def _shape(value: torch.Tensor) -> tuple[int, ...] | None:
    result = tuple(_static_extent(dim) for dim in value.shape)
    if any(dim is None for dim in result):
        return None
    return cast("tuple[int, ...]", result)


def _tensor_meta(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _ancestors(node: object) -> set[torch.fx.Node]:
    pending = [node]
    seen: set[torch.fx.Node] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        if not isinstance(current, torch.fx.Node):
            continue
        if current in seen:
            continue
        seen.add(current)
        pending.extend(current.all_input_nodes)
    return seen


def _ancestor_ref_keys(node: object) -> set[tuple[int, str]]:
    return {
        _ref_key(ref)
        for ancestor in _ancestors(node)
        if (ref := _load_ref(ancestor)) is not None
    }


def _strip_value_casts(node: object) -> torch.fx.Node | None:
    if not isinstance(node, torch.fx.Node):
        return None
    current = node
    passthroughs = {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.prims.convert_element_type.default,
    }
    for _ in range(16):
        if current.op != "call_function" or current.target not in passthroughs:
            return current
        if not current.args or not isinstance(current.args[0], torch.fx.Node):
            return None
        current = current.args[0]
    return None


def _strip_index_broadcast(node: object) -> torch.fx.Node | None:
    from ...language import view_ops

    if not isinstance(node, torch.fx.Node):
        return None
    current = node
    for _ in range(4):
        if current.op != "call_function" or current.target is not view_ops.subscript:
            return current
        if len(current.args) < 2 or not isinstance(current.args[1], (list, tuple)):
            return None
        if any(index is not None and index != slice(None) for index in current.args[1]):
            return None
        if not isinstance(current.args[0], torch.fx.Node):
            return None
        current = current.args[0]
    return None


def _is_row_broadcast_of(node: object, source: torch.fx.Node) -> bool:
    from ...language import view_ops

    return bool(
        isinstance(node, torch.fx.Node)
        and _is_call(node, view_ops.subscript)
        and len(node.args) == 2
        and node.args[0] is source
        and isinstance(node.args[1], (list, tuple))
        and tuple(node.args[1]) == (slice(None), None)
    )


def _is_column_broadcast_of(node: object, source: torch.fx.Node) -> bool:
    from ...language import view_ops

    return bool(
        isinstance(node, torch.fx.Node)
        and _is_call(node, view_ops.subscript)
        and len(node.args) == 2
        and node.args[0] is source
        and isinstance(node.args[1], (list, tuple))
        and tuple(node.args[1]) == (None, slice(None))
    )


def _memory_indices(node: torch.fx.Node) -> tuple[object, ...] | None:
    if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
        return None
    return tuple(node.args[1])


def _binary_args(
    node: object,
    targets: tuple[object, ...],
) -> tuple[object, object] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) < 2
    ):
        return None
    return node.args[0], node.args[1]


def _scaled_term(node: object, source: torch.fx.Node, scale: int) -> bool:
    args = _binary_args(node, (operator.mul, torch.ops.aten.mul.Tensor))
    if args is None:
        return False
    for value, constant in (args, reversed(args)):
        if value is source and _static_extent(constant) == scale:
            return True
    return False


def _is_transpose_of(node: object, source: torch.fx.Node) -> bool:
    current = _strip_value_casts(node)
    if not isinstance(current, torch.fx.Node) or current.op != "call_function":
        return False
    if current.target is not torch.ops.aten.permute.default or len(current.args) < 2:
        return False
    permutation = current.args[1]
    return bool(
        current.args[0] is source
        and isinstance(permutation, (list, tuple))
        and tuple(permutation) == (1, 0)
    )


def _has_scalar_xor(node: object, value: int) -> bool:
    return any(
        ancestor.op == "call_function"
        and ancestor.target is torch.ops.aten.bitwise_xor.Scalar
        and len(ancestor.args) >= 2
        and ancestor.args[1] == value
        for ancestor in _ancestors(node)
    )


def _tensor_is_exact_contiguous(ref: _TensorRef, shape: tuple[int, ...]) -> bool:
    return (
        _shape(ref.fake) == shape
        and ref.fake.device.type == "cuda"
        and ref.fake.is_contiguous()
    )


def _store_value(store: torch.fx.Node) -> torch.fx.Node | None:
    if len(store.args) < 4 or store.args[3] is not None:
        return None
    value = store.args[2]
    return value if isinstance(value, torch.fx.Node) else None


def _is_convert(node: object, source: object, dtype: torch.dtype) -> bool:
    return bool(
        isinstance(node, torch.fx.Node)
        and _is_call(node, torch.ops.prims.convert_element_type.default)
        and node.args == (source, dtype)
    )


def _roundtrip_source(node: object, storage_dtype: torch.dtype) -> torch.fx.Node | None:
    if (
        not isinstance(node, torch.fx.Node)
        or not _is_call(node, torch.ops.prims.convert_element_type.default)
        or len(node.args) != 2
        or node.args[1] is not torch.float32
    ):
        return None
    rounded = node.args[0]
    if (
        not isinstance(rounded, torch.fx.Node)
        or not _is_call(rounded, torch.ops.prims.convert_element_type.default)
        or len(rounded.args) != 2
        or rounded.args[1] is not storage_dtype
        or not isinstance(rounded.args[0], torch.fx.Node)
    ):
        return None
    return rounded.args[0]


def _bf16_roundtrip_source(node: object) -> torch.fx.Node | None:
    return _roundtrip_source(node, torch.bfloat16)


def _single_user(
    source: torch.fx.Node,
    predicate: Callable[[torch.fx.Node], bool],
) -> torch.fx.Node | None:
    matches = [user for user in source.users if predicate(user)]
    return matches[0] if len(matches) == 1 else None


def _zero_scalar(node: object) -> bool:
    return bool(
        isinstance(node, torch.fx.Node)
        and _is_call(node, torch.ops.aten.scalar_tensor.default)
        and node.args == (0.0,)
        and node.kwargs.get("dtype") is torch.float32
    )


def _single_bf16_roundtrip(
    nodes: Sequence[torch.fx.Node], source: torch.fx.Node
) -> torch.fx.Node | None:
    matches = [node for node in nodes if _bf16_roundtrip_source(node) is source]
    return matches[0] if len(matches) == 1 else None


def _single_load_for_ref(
    loads: Sequence[torch.fx.Node], ref: _TensorRef
) -> torch.fx.Node | None:
    matches = [
        node
        for node in loads
        if (loaded := _load_ref(node)) is not None and _ref_key(loaded) == _ref_key(ref)
    ]
    return matches[0] if len(matches) == 1 else None


def _dot_signature(node: torch.fx.Node) -> tuple[object, ...] | None:
    if len(node.args) < 4 or node.args[2] is not None:
        return None
    lhs = _tensor_meta(node.args[0])
    rhs = _tensor_meta(node.args[1])
    out = _tensor_meta(node)
    if lhs is None or rhs is None or out is None:
        return None
    return (_shape(lhs), lhs.dtype, _shape(rhs), rhs.dtype, _shape(out), out.dtype)


def _body_target_counts(
    gate_is_fp32: bool, *, outputs_scaled: bool = True
) -> Counter[object]:
    from ...language import memory_ops
    from ...language import scan_ops
    from ...language import view_ops
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _mask_to
    from ...language.tile_ops import tile_id

    counts: Counter[object] = Counter(
        {
            _host_tensor: 14,
            _get_symnode: 5,
            memory_ops.load: 10,
            torch.ops.prims.convert_element_type.default: (58 if gate_is_fp32 else 59),
            torch.ops.aten.add.Tensor: 11,
            torch.ops.aten.sub.Tensor: 2,
            torch.ops.aten.mul.Tensor: 20,
            torch.ops.prims.iota.default: 3,
            torch.ops.aten.lt.Tensor: 1,
            tile_id: 1,
            operator.mul: 2,
            operator.add: 1,
            view_ops.subscript: 30,
            torch.ops.aten.tanh.default: 1,
            torch.ops.aten.scalar_tensor.default: 6,
            torch.ops.aten.where.self: 7,
            scan_ops._associative_scan: 1,
            torch.ops.aten.clamp_min.default: 3,
            torch.ops.aten.exp2.default: 2,
            torch.ops.aten.eq.Scalar: 1,
            _mask_to: 1,
            torch.ops.aten.sum.dim_IntList: 3,
            torch.ops.aten.rsqrt.default: 2,
            torch.ops.aten.full.default: 1,
            torch.ops.aten.reciprocal.default: 1,
            torch.ops.aten.sigmoid.default: 1,
            torch.ops.aten.permute.default: 9,
            dot: 10,
            torch.ops.aten.ge.Tensor: 1,
            torch.ops.aten.gt.Tensor: 1,
            torch.ops.aten.unsqueeze.default: 6,
            torch.ops.aten.lt.Scalar: 3,
            torch.ops.aten.eq.Tensor: 2,
            torch.ops.aten.ge.Scalar: 1,
            torch.ops.aten.bitwise_and.Tensor: 1,
            torch.ops.aten.neg.default: 1,
            memory_ops.store: 5,
            torch.ops.aten.bitwise_xor.Scalar: 2,
            torch.ops.aten.__rshift__.Scalar: 1,
            torch.ops.aten.bitwise_and.Scalar: 1,
            torch.ops.aten.__lshift__.Scalar: 1,
            torch.ops.aten.bitwise_xor.Tensor: 1,
            torch.ops.aten.div.Tensor_mode: 1,
        }
    )
    if not outputs_scaled:
        counts.subtract(
            {
                _get_symnode: 1,
                view_ops.subscript: 1,
                torch.ops.aten.full.default: 1,
                torch.ops.aten.mul.Tensor: 1,
                torch.ops.prims.convert_element_type.default: 4,
            }
        )
        counts[torch.ops.aten.bitwise_xor.Scalar] += 2
    return counts


def _single(items: Iterable[object]) -> object | None:
    values = list(items)
    return values[0] if len(values) == 1 else None


def _aq_pair_index_matches(node: object, token_lane: torch.fx.Node) -> bool:
    pair_args = _binary_args(node, (torch.ops.aten.div.Tensor_mode,))
    if (
        pair_args is None
        or pair_args[1] != 2
        or not isinstance(node, torch.fx.Node)
        or node.kwargs != {"rounding_mode": "floor"}
    ):
        return False
    xor_args = _binary_args(pair_args[0], (torch.ops.aten.bitwise_xor.Tensor,))
    if xor_args is None:
        return False
    byte_offset, shifted = xor_args
    byte_args = _binary_args(byte_offset, (torch.ops.aten.mul.Tensor,))
    shifted_args = _binary_args(shifted, (torch.ops.aten.__lshift__.Scalar,))
    if (
        byte_args is None
        or byte_args[1] != 2
        or shifted_args is None
        or shifted_args[1] != 4
    ):
        return False
    offset_args = _binary_args(byte_args[0], (torch.ops.aten.add.Tensor,))
    masked_args = _binary_args(shifted_args[0], (torch.ops.aten.bitwise_and.Scalar,))
    if offset_args is None or masked_args is None or masked_args[1] != 1:
        return False
    row_args = _binary_args(offset_args[0], (torch.ops.aten.mul.Tensor,))
    storage_col_args = _binary_args(
        offset_args[1], (torch.ops.aten.bitwise_xor.Scalar,)
    )
    shifted_byte_args = _binary_args(
        masked_args[0], (torch.ops.aten.__rshift__.Scalar,)
    )
    return bool(
        row_args is not None
        and _is_row_broadcast_of(row_args[0], token_lane)
        and _static_extent(row_args[1]) == _BT
        and storage_col_args is not None
        and _is_column_broadcast_of(storage_col_args[0], token_lane)
        and storage_col_args[1] == 8
        and shifted_byte_args == (byte_offset, 7)
    )


def _match_chunk_group_coordinate(
    loop: ForLoopGraphInfo,
    loop_predicate: object,
    loop_extent: torch.fx.Node,
    total_chunks: int,
) -> torch.fx.Node | None:
    """Match ``chunk = group * CPC + local`` and prove both coordinates.

    The replacement schedule reconstructs this affine coordinate instead of
    executing the carrier loop. Its group and local terms must therefore be
    the canonical root and inner-loop coordinates, with no hidden offset.
    """

    from ...language._tracing_ops import _get_symnode
    from ...language.tile_ops import tile_id
    from ..host_function import HostFunction
    from ..variable_origin import BlockSizeOrigin
    from ..variable_origin import GridOrigin
    from ..variable_origin import TileIdOrigin

    def exact_binary(
        node: object,
        targets: tuple[object, ...],
    ) -> tuple[object, object] | None:
        if (
            not isinstance(node, torch.fx.Node)
            or node.op != "call_function"
            or node.target not in targets
            or len(node.args) != 2
            or node.kwargs
        ):
            return None
        return node.args[0], node.args[1]

    if len(loop.block_ids) != 1:
        return None
    predicate_args = exact_binary(
        loop_predicate,
        (operator.lt, torch.ops.aten.lt.Tensor),
    )
    if predicate_args is None or _static_extent(predicate_args[1]) != total_chunks:
        return None
    chunk_args = exact_binary(
        predicate_args[0],
        (operator.add, torch.ops.aten.add.Tensor),
    )
    if chunk_args is None:
        return None

    def origin(node: torch.fx.Node) -> object | None:
        value = node.meta.get("val")
        if not isinstance(value, torch.SymInt):
            return None
        info = HostFunction.current().expr_to_origin.get(value._sympy_())
        return info.origin if info is not None else None

    loop_extent_value = loop_extent.meta.get("val")
    if not isinstance(loop_extent_value, torch.SymInt):
        return None
    matches: list[torch.fx.Node] = []
    for product, local in (chunk_args, reversed(chunk_args)):
        local_origin = origin(local) if isinstance(local, torch.fx.Node) else None
        if (
            not isinstance(local, torch.fx.Node)
            or not _is_call(local, _get_symnode)
            or len(local.args) != 1
            or local.kwargs
            or type(local_origin) is not GridOrigin
            or local_origin.block_id != loop.block_ids[0]
        ):
            continue
        product_args = exact_binary(
            product,
            (operator.mul, torch.ops.aten.mul.Tensor),
        )
        if product_args is None:
            continue
        for group, chunks_per_group in (product_args, reversed(product_args)):
            group_origin = origin(group) if isinstance(group, torch.fx.Node) else None
            block_size_origin = (
                origin(group.args[0])
                if isinstance(group, torch.fx.Node)
                and group.args
                and isinstance(group.args[0], torch.fx.Node)
                else None
            )
            chunks_per_group_value = (
                chunks_per_group.meta.get("val")
                if isinstance(chunks_per_group, torch.fx.Node)
                else None
            )
            if (
                not isinstance(group, torch.fx.Node)
                or not _is_call(group, tile_id)
                or len(group.args) != 1
                or group.kwargs
                or not isinstance(group.args[0], torch.fx.Node)
                or not isinstance(group_origin, TileIdOrigin)
                or not isinstance(block_size_origin, BlockSizeOrigin)
                or block_size_origin.block_id != group_origin.block_id
                or group_origin.block_id == loop.block_ids[0]
                or not isinstance(chunks_per_group, torch.fx.Node)
                or not _is_call(chunks_per_group, _get_symnode)
                or len(chunks_per_group.args) != 1
                or chunks_per_group.kwargs
                or not isinstance(chunks_per_group_value, torch.SymInt)
                or chunks_per_group_value._sympy_() != loop_extent_value._sympy_()
                or _static_extent(chunks_per_group_value) != _SEMANTIC_CPC
            ):
                continue
            matches.append(group)
    return matches[0] if len(matches) == 1 else None


def _validate_memory_index_topology(
    *,
    loads: Sequence[torch.fx.Node],
    kd_store: torch.fx.Node,
    qd_store: torch.fx.Node,
    ak_store: torch.fx.Node,
    aq_store: torch.fx.Node,
    gt_store: torch.fx.Node,
    q: _TensorRef,
    k: _TensorRef,
    gate: _TensorRef,
    beta: _TensorRef,
    a_log: _TensorRef,
    dt_bias: _TensorRef,
    cu_seqlens: _TensorRef,
    cu_chunks: _TensorRef,
    chunk_to_seq: _TensorRef,
    total_chunks: int,
    heads: int,
    loop_predicate: object,
    factor_key_xor: int,
) -> bool:
    if any(len(node.args) != 4 or node.args[3] is not None for node in loads):
        return False
    q_load = _single_load_for_ref(loads, q)
    k_load = _single_load_for_ref(loads, k)
    gate_load = _single_load_for_ref(loads, gate)
    beta_load = _single_load_for_ref(loads, beta)
    dt_load = _single_load_for_ref(loads, dt_bias)
    a_log_load = _single_load_for_ref(loads, a_log)
    chunk_to_seq_load = _single_load_for_ref(loads, chunk_to_seq)
    cu_chunks_load = _single_load_for_ref(loads, cu_chunks)
    if any(
        node is None
        for node in (
            q_load,
            k_load,
            gate_load,
            beta_load,
            dt_load,
            a_log_load,
            chunk_to_seq_load,
            cu_chunks_load,
        )
    ):
        return False
    assert q_load is not None
    assert k_load is not None
    assert gate_load is not None
    assert beta_load is not None
    assert dt_load is not None
    assert a_log_load is not None
    assert chunk_to_seq_load is not None
    assert cu_chunks_load is not None

    q_indices = _memory_indices(q_load)
    k_indices = _memory_indices(k_load)
    gate_indices = _memory_indices(gate_load)
    beta_indices = _memory_indices(beta_load)
    dt_indices = _memory_indices(dt_load)
    a_log_indices = _memory_indices(a_log_load)
    if any(
        indices is None
        for indices in (
            q_indices,
            k_indices,
            gate_indices,
            beta_indices,
            dt_indices,
            a_log_indices,
        )
    ):
        return False
    assert q_indices is not None
    assert k_indices is not None
    assert gate_indices is not None
    assert beta_indices is not None
    assert dt_indices is not None
    assert a_log_indices is not None
    if not (
        len(q_indices) == len(k_indices) == len(gate_indices) == 2
        and len(beta_indices) == 1
        and len(dt_indices) == 2
        and len(a_log_indices) == 1
    ):
        return False
    input_row = _strip_index_broadcast(q_indices[0])
    feature = _strip_index_broadcast(q_indices[1])
    if input_row is None or feature is None:
        return False
    if any(
        _strip_index_broadcast(indices[0]) is not input_row
        or _strip_index_broadcast(indices[1]) is not feature
        for indices in (k_indices, gate_indices)
    ):
        return False
    valid = _strip_index_broadcast(q_load.args[2])
    if valid is None or any(
        _strip_index_broadcast(node.args[2]) is not valid
        for node in (k_load, gate_load)
    ):
        return False
    if (
        _strip_index_broadcast(beta_indices[0]) is not input_row
        or _strip_index_broadcast(beta_load.args[2]) is not valid
        or any(node.args[2] is not None for node in (dt_load, a_log_load))
    ):
        return False

    head = _strip_index_broadcast(dt_indices[0])
    if (
        head is None
        or _strip_index_broadcast(dt_indices[1]) is not feature
        or _strip_index_broadcast(a_log_indices[0]) is not head
    ):
        return False
    input_args = _binary_args(input_row, (operator.add, torch.ops.aten.add.Tensor))
    if input_args is None:
        return False
    # Refuse to infer through a different head coordinate: recover the token
    # only from the exact ``token * H + head`` form.
    token = None
    for product, minor in (input_args, reversed(input_args)):
        if minor is not head or not isinstance(product, torch.fx.Node):
            continue
        product_args = _binary_args(product, (operator.mul, torch.ops.aten.mul.Tensor))
        if product_args is None:
            continue
        for candidate, constant in (product_args, reversed(product_args)):
            if (
                isinstance(candidate, torch.fx.Node)
                and _static_extent(constant) == heads
            ):
                token = candidate
    if token is None:
        return False
    valid_args = _binary_args(valid, (operator.lt, torch.ops.aten.lt.Tensor))
    if valid_args is None or valid_args[0] is not token:
        return False
    sequence_end = valid_args[1]
    token_args = _binary_args(token, (operator.add, torch.ops.aten.add.Tensor))
    if token_args is None:
        return False
    token_lane = next(
        (
            value
            for value in token_args
            if isinstance(value, torch.fx.Node)
            and _is_call(value, torch.ops.prims.iota.default)
            and _static_extent(value.args[0]) == _BT
        ),
        None,
    )
    if not isinstance(token_lane, torch.fx.Node):
        return False
    chunk_begin = token_args[0] if token_args[1] is token_lane else token_args[1]

    kd_indices = _memory_indices(kd_store)
    qd_indices = _memory_indices(qd_store)
    ak_indices = _memory_indices(ak_store)
    aq_indices = _memory_indices(aq_store)
    gt_indices = _memory_indices(gt_store)
    if any(
        indices is None
        for indices in (kd_indices, qd_indices, ak_indices, aq_indices, gt_indices)
    ):
        return False
    assert kd_indices is not None
    assert qd_indices is not None
    assert ak_indices is not None
    assert aq_indices is not None
    assert gt_indices is not None
    if not all(
        len(indices) == 2
        for indices in (kd_indices, qd_indices, ak_indices, aq_indices, gt_indices)
    ):
        return False
    factor_row = _strip_index_broadcast(kd_indices[0])
    kd_feature = _strip_index_broadcast(kd_indices[1])
    qd_feature = _strip_index_broadcast(qd_indices[1])
    if factor_key_xor:
        feature_indices_match = all(
            _binary_args(index, (torch.ops.aten.bitwise_xor.Scalar,))
            == (feature, factor_key_xor)
            for index in (kd_feature, qd_feature)
        )
    else:
        feature_indices_match = kd_feature is feature and qd_feature is feature
    if (
        factor_row is None
        or _strip_index_broadcast(qd_indices[0]) is not factor_row
        or not feature_indices_match
        or _strip_index_broadcast(ak_indices[1]) is not feature
        or _strip_index_broadcast(gt_indices[1]) is not feature
    ):
        return False
    chunk_head = _strip_index_broadcast(aq_indices[0])
    if chunk_head is None or _strip_index_broadcast(gt_indices[0]) is not chunk_head:
        return False
    factor_args = _binary_args(factor_row, (operator.add, torch.ops.aten.add.Tensor))
    if factor_args is None:
        return False
    factor_base = next(
        (
            value
            for value in factor_args
            if isinstance(value, torch.fx.Node) and _scaled_term(value, chunk_head, _BT)
        ),
        None,
    )
    if factor_base is None or token_lane not in factor_args:
        return False
    ak_row = _strip_index_broadcast(ak_indices[0])
    ak_args = _binary_args(ak_row, (operator.add, torch.ops.aten.add.Tensor))
    if ak_args is None or factor_base not in ak_args:
        return False
    ak_lane = ak_args[0] if ak_args[1] is factor_base else ak_args[1]
    ak_lane_args = _binary_args(ak_lane, (torch.ops.aten.bitwise_xor.Scalar,))
    if ak_lane_args != (token_lane, 8):
        return False
    if not _aq_pair_index_matches(aq_indices[1], token_lane):
        return False

    chunk_head_args = _binary_args(
        chunk_head, (operator.add, torch.ops.aten.add.Tensor)
    )
    if chunk_head_args is None:
        return False
    chunk = None
    for product, minor in (chunk_head_args, reversed(chunk_head_args)):
        if not isinstance(minor, torch.fx.Node) or not isinstance(
            product, torch.fx.Node
        ):
            continue
        if _scaled_term(product, head, total_chunks):
            chunk = minor
    if chunk is None:
        return False
    chunk_indices = _memory_indices(chunk_to_seq_load)
    if (
        chunk_indices is None
        or len(chunk_indices) != 1
        or chunk_indices[0] is not chunk
    ):
        return False
    sequence = _strip_value_casts(chunk_to_seq_load)
    # ``_strip_value_casts`` starts at the load. Recover its sole widening user.
    sequence_users = [
        user
        for user in chunk_to_seq_load.users
        if user.target is torch.ops.prims.convert_element_type.default
    ]
    if sequence is not chunk_to_seq_load or len(sequence_users) != 1:
        return False
    sequence = sequence_users[0]
    seq_loads = [
        node
        for node in loads
        if (ref := _load_ref(node)) is not None
        and _ref_key(ref) == _ref_key(cu_seqlens)
    ]
    if len(seq_loads) != 2:
        return False
    seq_indices = [_memory_indices(node) for node in seq_loads]
    if any(indices is None or len(indices) != 1 for indices in seq_indices):
        return False
    checked_seq_indices = cast("list[tuple[object, ...]]", seq_indices)
    sequence_begin_load = next(
        (
            node
            for node, indices in zip(seq_loads, checked_seq_indices, strict=True)
            if indices[0] is sequence
        ),
        None,
    )
    sequence_end_load = next(
        (
            node
            for node, indices in zip(seq_loads, checked_seq_indices, strict=True)
            if (
                (
                    index_args := _binary_args(
                        indices[0], (operator.add, torch.ops.aten.add.Tensor)
                    )
                )
                is not None
                and sequence in index_args
                and 1 in index_args
            )
        ),
        None,
    )
    cu_chunk_indices = _memory_indices(cu_chunks_load)
    if (
        sequence_begin_load is None
        or sequence_end_load is None
        or cu_chunk_indices != (sequence,)
        or _strip_value_casts(sequence_end) is not sequence_end_load
    ):
        return False
    sequence_begin_users = [
        user
        for user in sequence_begin_load.users
        if user.target is torch.ops.prims.convert_element_type.default
    ]
    if len(sequence_begin_users) != 1:
        return False
    chunk_begin_args = _binary_args(
        chunk_begin, (operator.add, torch.ops.aten.add.Tensor)
    )
    if chunk_begin_args is None or sequence_begin_users[0] not in chunk_begin_args:
        return False
    chunk_delta = (
        chunk_begin_args[0]
        if chunk_begin_args[1] is sequence_begin_users[0]
        else chunk_begin_args[1]
    )
    delta_args = _binary_args(chunk_delta, (operator.mul, torch.ops.aten.mul.Tensor))
    if delta_args is None:
        return False
    difference = next(
        (
            value
            for value, constant in (delta_args, reversed(delta_args))
            if isinstance(value, torch.fx.Node) and _static_extent(constant) == _BT
        ),
        None,
    )
    difference_args = _binary_args(
        difference, (operator.sub, torch.ops.aten.sub.Tensor)
    )
    if difference_args is None or difference_args != (chunk, cu_chunks_load):
        return False

    predicate_args = _binary_args(
        loop_predicate, (operator.lt, torch.ops.aten.lt.Tensor)
    )
    if predicate_args is None or _static_extent(predicate_args[1]) != total_chunks:
        return False
    outer_chunk = (
        predicate_args[0].meta.get("val")
        if isinstance(predicate_args[0], torch.fx.Node)
        else None
    )
    inner_chunk = chunk.meta.get("val")
    return bool(
        isinstance(outer_chunk, torch.SymInt)
        and isinstance(inner_chunk, torch.SymInt)
        and outer_chunk._sympy_() == inner_chunk._sympy_()
    )


def _packed_workspace_is_exact(
    kd: _TensorRef,
    qd: _TensorRef,
    ak: _TensorRef,
    aq: _TensorRef,
    g_total: _TensorRef,
) -> bool:
    refs = (kd, qd, ak, aq, g_total)
    storages = [ref.fake.untyped_storage() for ref in refs]
    if any(storage is not storages[0] for storage in storages[1:]):
        return False
    offsets = tuple(ref.fake.storage_offset() * ref.fake.element_size() for ref in refs)
    vector_bytes = kd.fake.numel() * kd.fake.element_size()
    aq_bytes = aq.fake.numel() * aq.fake.element_size()
    workspace_bytes = (
        3 * vector_bytes + aq_bytes + g_total.fake.numel() * g_total.fake.element_size()
    )
    return offsets == (
        0,
        vector_bytes,
        2 * vector_bytes,
        3 * vector_bytes,
        3 * vector_bytes + aq_bytes,
    ) and all(storage.nbytes() >= workspace_bytes for storage in storages)


def _packed_workspace_runtime_signature(values: Sequence[object]) -> bool:
    if len(values) != 5 or any(not isinstance(value, torch.Tensor) for value in values):
        return False
    kd, qd, ak, aq, g_total = cast("tuple[torch.Tensor, ...]", tuple(values))
    refs = tuple(_TensorRef(value, "") for value in (kd, qd, ak, aq, g_total))
    if not _packed_workspace_is_exact(*refs):
        return False
    # Real launch addresses must also satisfy the descriptor's 256-byte base
    # alignment. FakeTensor has no meaningful device address, so its structural
    # storage proof is the strongest fact available to compile-only tests.
    from torch._subclasses import FakeTensor

    return isinstance(kd, FakeTensor) or kd.data_ptr() % 256 == 0


def _packed_workspace_sources(
    env: CompileEnvironment,
    tensors: Sequence[torch.Tensor],
) -> tuple[Source, ...] | None:
    sources: list[Source] = []
    for fake in tensors:
        storage_key = fake.untyped_storage()._cdata
        matches = [
            source
            for input_tensor, source in env.input_sources.items()
            if input_tensor.untyped_storage()._cdata == storage_key
        ]
        if len(matches) != 1:
            return None
        sources.append(matches[0])
    return tuple(sources)


def _register_packed_workspace_specialization(
    env: CompileEnvironment,
    tensors: Sequence[torch.Tensor],
) -> bool:
    from ..compile_environment import RuntimeInputSpecialization

    sources = _packed_workspace_sources(env, tensors)
    if sources is None:
        return False
    env.register_runtime_input_specialization(
        _PACKED_WORKSPACE_SPECIALIZATION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=("kda_packed_workspace_v1", tuple(map(repr, sources))),
            classifier=_packed_workspace_runtime_signature,
            reusable_tensor_properties=frozenset(("data_ptr", "storage_span")),
        ),
    )
    return True


def _packed_workspace_is_proven(env: CompileEnvironment) -> bool:
    return env.runtime_input_specialization_matches_bound(
        _PACKED_WORKSPACE_SPECIALIZATION_KEY,
        True,
    )


def _validate_block_inverse_semantics(
    lower: torch.fx.Node,
    inverse: torch.fx.Node,
    fp16_dots: Sequence[torch.fx.Node],
) -> bool:
    """Match the exact two-block FP16 inverse and all six contractions."""

    if len(fp16_dots) != 6 or not _is_convert(
        inverse,
        inverse.args[0] if inverse.args else None,
        torch.bfloat16,
    ):
        return False
    diagonal2, diagonal4, correction2, correction4, first, lower_left = fp16_dots

    def f16(node: object, source: object) -> bool:
        return _is_convert(node, source, torch.float16)

    def f16_transpose(node: object, source: torch.fx.Node) -> bool:
        if not isinstance(node, torch.fx.Node) or not _is_convert(
            node, node.args[0] if node.args else None, torch.float16
        ):
            return False
        transpose = node.args[0]
        return bool(
            isinstance(transpose, torch.fx.Node)
            and _is_call(transpose, torch.ops.aten.permute.default)
            and transpose.args == (source, [1, 0])
        )

    diagonal = (
        diagonal2.args[0].args[0]
        if isinstance(diagonal2.args[0], torch.fx.Node) and diagonal2.args[0].args
        else None
    )
    if (
        not isinstance(diagonal, torch.fx.Node)
        or not f16(diagonal2.args[0], diagonal)
        or not f16_transpose(diagonal2.args[1], diagonal)
        or diagonal2.args[2] is not None
        or not f16(diagonal4.args[0], diagonal2)
        or not f16_transpose(diagonal4.args[1], diagonal2)
        or diagonal4.args[2] is not None
    ):
        return False
    if (
        not _is_call(diagonal, torch.ops.aten.where.self)
        or diagonal.args[1] is not lower
        or not _zero_scalar(diagonal.args[2])
    ):
        return False
    same_half = diagonal.args[0]
    same_half_args = _binary_args(same_half, (torch.ops.aten.eq.Tensor,))
    if same_half_args is None:
        return False

    final_where = inverse.args[0]
    if not isinstance(final_where, torch.fx.Node) or not _is_call(
        final_where, torch.ops.aten.where.self
    ):
        return False
    lower_half = final_where.args[0]
    negated_lower_left = final_where.args[1]
    inverse2 = final_where.args[2]
    if (
        not isinstance(negated_lower_left, torch.fx.Node)
        or not _is_call(negated_lower_left, torch.ops.aten.neg.default)
        or negated_lower_left.args != (lower_left,)
        or not isinstance(inverse2, torch.fx.Node)
    ):
        return False
    lower_half_args = _binary_args(lower_half, (torch.ops.aten.bitwise_and.Tensor,))
    if lower_half_args is None:
        return False

    def unsqueezed_iota(node: object, dim: int) -> torch.fx.Node | None:
        if (
            not isinstance(node, torch.fx.Node)
            or not _is_call(node, torch.ops.aten.unsqueeze.default)
            or node.args[1] != dim
            or not isinstance(node.args[0], torch.fx.Node)
            or not _is_call(node.args[0], torch.ops.prims.iota.default)
            or _static_extent(node.args[0].args[0]) != _BT
        ):
            return None
        return node.args[0]

    same_row_args = _binary_args(same_half_args[0], (torch.ops.aten.lt.Scalar,))
    same_col_args = _binary_args(same_half_args[1], (torch.ops.aten.lt.Scalar,))
    lower_row_args = _binary_args(lower_half_args[0], (torch.ops.aten.ge.Scalar,))
    lower_col_args = _binary_args(lower_half_args[1], (torch.ops.aten.lt.Scalar,))
    if any(
        args is None or args[1] != 8
        for args in (same_row_args, same_col_args, lower_row_args, lower_col_args)
    ):
        return False
    assert same_row_args is not None
    assert same_col_args is not None
    assert lower_row_args is not None
    assert lower_col_args is not None
    inverse_iotas = (
        unsqueezed_iota(same_row_args[0], 1),
        unsqueezed_iota(same_col_args[0], 0),
        unsqueezed_iota(lower_row_args[0], 1),
        unsqueezed_iota(lower_col_args[0], 0),
    )
    if inverse_iotas[0] is None or any(
        iota is not inverse_iotas[0] for iota in inverse_iotas[1:]
    ):
        return False
    inverse_iota = inverse_iotas[0]
    assert inverse_iota is not None

    coupling = (
        first.args[1].args[0].args[0]
        if isinstance(first.args[1], torch.fx.Node)
        and first.args[1].args
        and isinstance(first.args[1].args[0], torch.fx.Node)
        and first.args[1].args[0].args
        else None
    )
    if (
        not isinstance(coupling, torch.fx.Node)
        or not _is_call(coupling, torch.ops.aten.where.self)
        or coupling.args[0] is not lower_half
        or coupling.args[1] is not lower
        or not _zero_scalar(coupling.args[2])
        or not f16_transpose(first.args[1], coupling)
    ):
        return False

    inverse2_args = _binary_args(inverse2, (torch.ops.aten.add.Tensor,))
    if inverse2_args is None or inverse2_args[1] is not correction4:
        return False
    inverse1_f32 = inverse2_args[0]
    inverse1 = _roundtrip_source(inverse1_f32, torch.float16)
    if inverse1 is None:
        return False
    inverse1_args = _binary_args(inverse1, (torch.ops.aten.add.Tensor,))
    if inverse1_args is None or inverse1_args[1] is not correction2:
        return False
    inverse0_f32 = inverse1_args[0]
    inverse0 = _roundtrip_source(inverse0_f32, torch.float16)
    inverse0_args = _binary_args(inverse0, (torch.ops.aten.sub.Tensor,))
    if inverse0_args is None:
        return False
    eye = inverse0_args[0]
    diagonal_f32 = inverse0_args[1]
    if (
        not isinstance(eye, torch.fx.Node)
        or not _is_convert(
            eye,
            eye.args[0] if eye.args else None,
            torch.float32,
        )
        or _roundtrip_source(diagonal_f32, torch.float16) is not diagonal
    ):
        return False
    eye_args = _binary_args(eye.args[0], (torch.ops.aten.eq.Tensor,))
    if eye_args is None:
        return False
    if (
        unsqueezed_iota(eye_args[0], 1) is not inverse_iota
        or unsqueezed_iota(eye_args[1], 0) is not inverse_iota
    ):
        return False

    if not (
        f16(correction2.args[0], inverse0)
        and f16_transpose(correction2.args[1], diagonal2)
        and correction2.args[2] is None
        and f16(correction4.args[0], inverse1)
        and f16_transpose(correction4.args[1], diagonal4)
        and correction4.args[2] is None
        and f16(first.args[0], inverse2)
        and first.args[2] is None
        and f16(lower_left.args[0], first)
        and f16_transpose(lower_left.args[1], inverse2)
        and lower_left.args[2] is None
    ):
        return False
    return all(
        len(node.args) == 4 and node.args[3] is torch.float32 for node in fp16_dots
    )


def _validate_prepare_semantics(
    *,
    body_nodes: Sequence[torch.fx.Node],
    loads: Sequence[torch.fx.Node],
    kd_store: torch.fx.Node,
    qd_store: torch.fx.Node,
    ak_store: torch.fx.Node,
    aq_store: torch.fx.Node,
    gt_store: torch.fx.Node,
    kk_dot: torch.fx.Node,
    qk_dot: torch.fx.Node,
    aq_dot: torch.fx.Node,
    ak_dot: torch.fx.Node,
    fp16_dots: Sequence[torch.fx.Node],
    q: _TensorRef,
    k: _TensorRef,
    gate: _TensorRef,
    beta: _TensorRef,
    a_log: _TensorRef,
    dt_bias: _TensorRef,
    outputs_scaled: bool,
) -> bool:
    """Prove every numerically sensitive edge in the prepare carrier."""

    from ...language import scan_ops
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _mask_to

    q_load = _single_load_for_ref(loads, q)
    k_load = _single_load_for_ref(loads, k)
    gate_load = _single_load_for_ref(loads, gate)
    beta_load = _single_load_for_ref(loads, beta)
    dt_load = _single_load_for_ref(loads, dt_bias)
    a_log_load = _single_load_for_ref(loads, a_log)
    if any(
        node is None
        for node in (q_load, k_load, gate_load, beta_load, dt_load, a_log_load)
    ):
        return False
    assert q_load is not None
    assert k_load is not None
    assert gate_load is not None
    assert beta_load is not None
    assert dt_load is not None
    assert a_log_load is not None

    q_raw = _single_user(q_load, lambda node: _is_convert(node, q_load, torch.float32))
    k_raw = _single_user(k_load, lambda node: _is_convert(node, k_load, torch.float32))
    gate_raw = (
        gate_load
        if gate.fake.dtype is torch.float32
        else _single_user(
            gate_load, lambda node: _is_convert(node, gate_load, torch.float32)
        )
    )
    beta_f32 = _single_user(
        beta_load, lambda node: _is_convert(node, beta_load, torch.float32)
    )
    if any(value is None for value in (q_raw, k_raw, gate_raw, beta_f32)):
        return False
    assert q_raw is not None
    assert k_raw is not None
    assert gate_raw is not None
    assert beta_f32 is not None

    # Pin token validity to every masked activation load and to the gate scan.
    valid = _strip_index_broadcast(q_load.args[2])
    if (
        valid is None
        or any(
            _strip_index_broadcast(node.args[2]) is not valid
            for node in (k_load, gate_load)
        )
        or beta_load.args[2] is not valid
    ):
        return False
    valid_args = _binary_args(valid, (torch.ops.aten.lt.Tensor,))
    if valid_args is None:
        return False
    token = valid_args[0]
    token_args = _binary_args(token, (torch.ops.aten.add.Tensor,))
    if token_args is None:
        return False
    token_lane = next(
        (
            value
            for value in token_args
            if isinstance(value, torch.fx.Node)
            and _is_call(value, torch.ops.prims.iota.default)
            and _static_extent(value.args[0]) == _BT
        ),
        None,
    )
    if token_lane is None:
        return False

    gamma = _store_value(gt_store)
    if (
        not isinstance(gamma, torch.fx.Node)
        or not _is_call(gamma, torch.ops.aten.sum.dim_IntList)
        or gamma.args[1] != [0]
    ):
        return False
    masked_last = gamma.args[0]
    if (
        not isinstance(masked_last, torch.fx.Node)
        or not _is_call(masked_last, _mask_to)
        or masked_last.args[1] != 0
    ):
        return False
    last_where = masked_last.args[0]
    if (
        not isinstance(last_where, torch.fx.Node)
        or not _is_call(last_where, torch.ops.aten.where.self)
        or not _zero_scalar(last_where.args[2])
    ):
        return False
    last_condition = _strip_index_broadcast(last_where.args[0])
    last_args = _binary_args(last_condition, (torch.ops.aten.eq.Scalar,))
    exp_gate = last_where.args[1]
    if last_args != (token_lane, _BT - 1) or not _is_call(
        exp_gate, torch.ops.aten.exp2.default
    ):
        return False
    assert isinstance(exp_gate, torch.fx.Node)
    clamp = exp_gate.args[0]
    if (
        not isinstance(clamp, torch.fx.Node)
        or not _is_call(clamp, torch.ops.aten.clamp_min.default)
        or clamp.args[1] != -126.0
    ):
        return False
    gate_prefix = clamp.args[0]
    if not isinstance(gate_prefix, torch.fx.Node) or not _is_call(
        gate_prefix, scan_ops._associative_scan
    ):
        return False
    masked_increment = gate_prefix.args[1]
    if (
        not isinstance(masked_increment, torch.fx.Node)
        or not _is_call(masked_increment, torch.ops.aten.where.self)
        or _strip_index_broadcast(masked_increment.args[0]) is not valid
        or not _zero_scalar(masked_increment.args[2])
    ):
        return False
    gate_increment = masked_increment.args[1]
    gate_scale_args = _binary_args(gate_increment, (torch.ops.aten.mul.Tensor,))
    if gate_scale_args is None or not _is_call(gate_scale_args[1], _get_symnode):
        return False
    shifted_sigmoid = gate_scale_args[0]
    shifted_args = _binary_args(shifted_sigmoid, (torch.ops.aten.add.Tensor,))
    if shifted_args is None or shifted_args[1] != 0.5:
        return False
    tanh_half_args = _binary_args(shifted_args[0], (torch.ops.aten.mul.Tensor,))
    if tanh_half_args is None or tanh_half_args[1] != 0.5:
        return False
    tanh = tanh_half_args[0]
    if not _is_call(tanh, torch.ops.aten.tanh.default):
        return False
    assert isinstance(tanh, torch.fx.Node)
    tanh_input_args = _binary_args(tanh.args[0], (torch.ops.aten.mul.Tensor,))
    if tanh_input_args is None or tanh_input_args[1] != 0.5:
        return False
    alog_gate_args = _binary_args(tanh_input_args[0], (torch.ops.aten.mul.Tensor,))
    if alog_gate_args is None:
        return False
    a_log_scaled = _single_user(
        a_log_load,
        lambda node: (
            (args := _binary_args(node, (torch.ops.aten.mul.Tensor,))) is not None
            and any(
                value is a_log_load and _static_float(scale) == _LOG2_E
                for value, scale in (args, reversed(args))
            )
        ),
    )
    if a_log_scaled is None:
        return False
    log_decay_scale = _single_user(
        a_log_scaled,
        lambda node: (
            _is_call(node, torch.ops.aten.exp2.default) and node.args == (a_log_scaled,)
        ),
    )
    if log_decay_scale is None:
        return False
    gate_bias = next(
        (
            maybe_gate_bias
            for maybe_scale, maybe_gate_bias in (
                alog_gate_args,
                tuple(reversed(alog_gate_args)),
            )
            if maybe_scale is log_decay_scale
        ),
        None,
    )
    gate_bias_args = _binary_args(gate_bias, (torch.ops.aten.add.Tensor,))
    if gate_bias_args is None or gate_bias_args[0] is not gate_raw:
        return False
    if _strip_index_broadcast(gate_bias_args[1]) is not dt_load:
        return False

    def normalized(raw: torch.fx.Node) -> torch.fx.Node | None:
        square = _single_user(
            raw,
            lambda node: (
                _binary_args(node, (torch.ops.aten.mul.Tensor,))
                in ((raw, raw), (raw, None))
            ),
        )
        if square is None:
            return None
        sum_squares = _single_user(
            square,
            lambda node: (
                _is_call(node, torch.ops.aten.sum.dim_IntList)
                and node.args == (square, [-1])
            ),
        )
        if sum_squares is None:
            return None
        floor = _single_user(
            sum_squares,
            lambda node: (
                _is_call(node, torch.ops.aten.clamp_min.default)
                and node.args == (sum_squares, 1.0e-24)
            ),
        )
        if floor is None:
            return None
        inverse = _single_user(
            floor,
            lambda node: (
                _is_call(node, torch.ops.aten.rsqrt.default) and node.args == (floor,)
            ),
        )
        if inverse is None:
            return None
        return _single_user(
            raw,
            lambda node: (
                (args := _binary_args(node, (torch.ops.aten.mul.Tensor,))) is not None
                and args[0] is raw
                and _strip_index_broadcast(args[1]) is inverse
            ),
        )

    q_norm_product = normalized(q_raw)
    k_norm = normalized(k_raw)
    if q_norm_product is None or k_norm is None:
        return False
    q_norm = _single_bf16_roundtrip(body_nodes, q_norm_product)
    exp_gate_bf16 = _single_bf16_roundtrip(body_nodes, exp_gate)
    k_norm_bf16 = _single_bf16_roundtrip(body_nodes, k_norm)
    if q_norm is None or exp_gate_bf16 is None or k_norm_bf16 is None:
        return False

    kd_product = _single(
        node
        for node in body_nodes
        if _binary_args(node, (torch.ops.aten.mul.Tensor,))
        == (k_norm_bf16, exp_gate_bf16)
    )
    kd_value = (
        _single_bf16_roundtrip(body_nodes, kd_product)
        if isinstance(kd_product, torch.fx.Node)
        else None
    )
    reciprocal = _single_user(
        exp_gate,
        lambda node: (
            _is_call(node, torch.ops.aten.reciprocal.default)
            and node.args == (exp_gate,)
        ),
    )
    ki_product = (
        _single(
            node
            for node in body_nodes
            if _binary_args(node, (torch.ops.aten.mul.Tensor,)) == (k_norm, reciprocal)
        )
        if reciprocal is not None
        else None
    )
    ki_value = (
        _single_bf16_roundtrip(body_nodes, ki_product)
        if isinstance(ki_product, torch.fx.Node)
        else None
    )
    qd_product = _single(
        node
        for node in body_nodes
        if _binary_args(node, (torch.ops.aten.mul.Tensor,)) == (q_norm, exp_gate_bf16)
    )
    qd_unscaled = (
        _single_bf16_roundtrip(body_nodes, qd_product)
        if isinstance(qd_product, torch.fx.Node)
        else None
    )
    if any(value is None for value in (kd_value, ki_value, qd_unscaled)):
        return False

    qd_value = qd_unscaled
    if outputs_scaled:
        full_nodes = [
            node for node in body_nodes if _is_call(node, torch.ops.aten.full.default)
        ]
        if (
            len(full_nodes) != 1
            or full_nodes[0].args[0] != [_BT]
            or not _is_call(full_nodes[0].args[1], _get_symnode)
        ):
            return False
        scale_bf16 = _single_bf16_roundtrip(body_nodes, full_nodes[0])
        scaled_qd = (
            _single(
                node
                for node in body_nodes
                if scale_bf16 is not None
                and (args := _binary_args(node, (torch.ops.aten.mul.Tensor,)))
                is not None
                and args[0] is qd_unscaled
                and _strip_index_broadcast(args[1]) is scale_bf16
            )
            if scale_bf16 is not None
            else None
        )
        qd_value = (
            _single_bf16_roundtrip(body_nodes, scaled_qd)
            if isinstance(scaled_qd, torch.fx.Node)
            else None
        )
    if not isinstance(qd_value, torch.fx.Node):
        return False

    kd_stored = _store_value(kd_store)
    qd_stored = _store_value(qd_store)
    ak_stored = _store_value(ak_store)
    aq_stored = _store_value(aq_store)
    if not _is_convert(kd_stored, kd_value, torch.bfloat16) or not _is_convert(
        qd_stored, qd_value, torch.bfloat16
    ):
        return False

    def bf16_operand(node: object, source: object) -> bool:
        return _is_convert(node, source, torch.bfloat16)

    def bf16_transpose(node: object, source: object) -> bool:
        if not isinstance(node, torch.fx.Node) or not _is_convert(
            node, node.args[0] if node.args else None, torch.bfloat16
        ):
            return False
        transpose = node.args[0]
        return bool(
            isinstance(transpose, torch.fx.Node)
            and _is_call(transpose, torch.ops.aten.permute.default)
            and transpose.args == (source, [1, 0])
        )

    if not (
        bf16_operand(kk_dot.args[0], kd_value)
        and bf16_transpose(kk_dot.args[1], ki_value)
        and bf16_operand(qk_dot.args[0], qd_value)
        and bf16_transpose(qk_dot.args[1], ki_value)
        and kk_dot.args[2] is qk_dot.args[2] is None
    ):
        return False

    beta_value = _single_user(
        beta_f32,
        lambda node: (
            _is_call(node, torch.ops.aten.sigmoid.default) and node.args == (beta_f32,)
        ),
    )
    if beta_value is None:
        return False
    kk_scaled = _single(
        node
        for node in body_nodes
        if (args := _binary_args(node, (torch.ops.aten.mul.Tensor,))) is not None
        and args[0] is kk_dot
        and _strip_index_broadcast(args[1]) is beta_value
    )
    lower = (
        _single(
            node
            for node in body_nodes
            if isinstance(kk_scaled, torch.fx.Node)
            and _is_call(node, torch.ops.aten.where.self)
            and node.args[1] is kk_scaled
            and _zero_scalar(node.args[2])
        )
        if kk_scaled is not None
        else None
    )
    if not isinstance(lower, torch.fx.Node):
        return False
    strict = lower.args[0]
    strict_args = _binary_args(strict, (torch.ops.aten.gt.Tensor,))
    if (
        strict_args is None
        or not _is_row_broadcast_of(strict_args[0], token_lane)
        or not _is_column_broadcast_of(strict_args[1], token_lane)
    ):
        return False

    inverse_beta = aq_dot.args[1]
    if not isinstance(inverse_beta, torch.fx.Node) or not _is_convert(
        inverse_beta,
        inverse_beta.args[0] if inverse_beta.args else None,
        torch.bfloat16,
    ):
        return False
    inverse_beta_product = inverse_beta.args[0]
    inverse_beta_args = _binary_args(inverse_beta_product, (torch.ops.aten.mul.Tensor,))
    if inverse_beta_args is None:
        return False
    inverse_f32 = inverse_beta_args[0]
    if (
        not isinstance(inverse_f32, torch.fx.Node)
        or not _is_call(inverse_f32, torch.ops.prims.convert_element_type.default)
        or inverse_f32.args[1] is not torch.float32
    ):
        return False
    inverse = inverse_f32.args[0]
    if not _is_convert(
        inverse,
        inverse.args[0] if isinstance(inverse, torch.fx.Node) else None,
        torch.bfloat16,
    ):
        return False
    if not isinstance(inverse, torch.fx.Node) or not _validate_block_inverse_semantics(
        lower,
        inverse,
        fp16_dots,
    ):
        return False
    rounded_beta = inverse_beta_args[1]
    rounded_beta_source = _bf16_roundtrip_source(rounded_beta)
    if (
        rounded_beta_source is None
        or _strip_index_broadcast(rounded_beta_source) is not beta_value
    ):
        return False

    qk_masked = aq_dot.args[0]
    if not isinstance(qk_masked, torch.fx.Node) or not _is_convert(
        qk_masked, qk_masked.args[0] if qk_masked.args else None, torch.bfloat16
    ):
        return False
    causal_where = qk_masked.args[0]
    if (
        not isinstance(causal_where, torch.fx.Node)
        or not _is_call(causal_where, torch.ops.aten.where.self)
        or causal_where.args[1] is not qk_dot
        or not _zero_scalar(causal_where.args[2])
    ):
        return False
    causal_args = _binary_args(causal_where.args[0], (torch.ops.aten.ge.Tensor,))
    if (
        causal_args is None
        or not _is_row_broadcast_of(causal_args[0], token_lane)
        or not _is_column_broadcast_of(causal_args[1], token_lane)
    ):
        return False
    if aq_dot.args[2] is not None or not _is_convert(aq_stored, aq_dot, torch.bfloat16):
        return False

    gamma_broadcast = _single(
        node
        for node in body_nodes
        if _strip_index_broadcast(node) is gamma and node is not gamma
    )
    gamma_bf16 = (
        _single_bf16_roundtrip(body_nodes, gamma_broadcast)
        if isinstance(gamma_broadcast, torch.fx.Node)
        else None
    )
    kg_product = _single(
        node
        for node in body_nodes
        if gamma_bf16 is not None
        and _binary_args(node, (torch.ops.aten.mul.Tensor,)) == (ki_value, gamma_bf16)
    )
    kg = (
        _single_user(
            cast("torch.fx.Node", kg_product),
            lambda node: _is_convert(node, kg_product, torch.bfloat16),
        )
        if isinstance(kg_product, torch.fx.Node)
        else None
    )
    if (
        kg is None
        or not _is_transpose_of(ak_dot.args[0], inverse_beta)
        or ak_dot.args[1] is not kg
        or ak_dot.args[2] is not None
        or not _is_convert(ak_stored, ak_dot, torch.bfloat16)
    ):
        return False
    return all(
        len(node.args) == 4 and node.args[2] is None and node.args[3] is torch.float32
        for node in (kk_dot, qk_dot, aq_dot, ak_dot, *fp16_dots)
    )


def _match_chunk_prepare_graphs(
    graphs: Sequence[GraphInfo],
) -> _CuteChunkPrepareMatch | None:
    import sympy

    from ...language import scan_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _if
    from ..device_ir import ElseGraphInfo
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import HelperFunctionGraphInfo
    from ..device_ir import IfGraphInfo
    from ..device_ir import RootGraphInfo

    roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
    loops = [graph for graph in graphs if isinstance(graph, ForLoopGraphInfo)]
    ifs = [graph for graph in graphs if isinstance(graph, IfGraphInfo)]
    elses = [graph for graph in graphs if isinstance(graph, ElseGraphInfo)]
    helpers = [graph for graph in graphs if isinstance(graph, HelperFunctionGraphInfo)]
    if not (
        len(graphs) == 5
        and len(roots) == len(loops) == len(ifs) == len(elses) == len(helpers) == 1
    ):
        return None
    root_info, loop_info, if_info, else_info, helper_info = (
        roots[0],
        loops[0],
        ifs[0],
        elses[0],
        helpers[0],
    )
    root_calls = [node for node in root_info.graph.nodes if node.op == "call_function"]
    if len(root_calls) != 2 or not _is_call(root_calls[1], _for_loop):
        return None
    loop_call = root_calls[1]
    if (
        not loop_call.args
        or loop_call.args[0] != loop_info.graph_id
        or len(loop_call.args) < 4
        or loop_call.args[1] != [0]
        or not isinstance(loop_call.args[2], list)
        or len(loop_call.args[2]) != 1
        or not isinstance(loop_call.args[2][0], torch.fx.Node)
        or _static_extent(loop_call.args[2][0].meta.get("val")) != _SEMANTIC_CPC
        or loop_call.args[3] != []
        or len(loop_info.block_ids) != 1
    ):
        return None
    loop_calls = [node for node in loop_info.graph.nodes if node.op == "call_function"]
    if_calls = [node for node in loop_calls if node.target is _if]
    if len(if_calls) != 1:
        return None
    if_call = if_calls[0]
    if (
        len(if_call.args) < 5
        or if_call.args[1] != if_info.graph_id
        or if_call.args[2] != else_info.graph_id
        or if_call.args[3] != []
        or if_call.args[4] != []
        or list(else_info.graph.nodes)[-1].op != "output"
        or len(list(else_info.graph.nodes)) != 1
    ):
        return None
    helper_nodes = list(helper_info.graph.nodes)
    helper_calls = [node for node in helper_nodes if node.op == "call_function"]
    helper_placeholders = [node for node in helper_nodes if node.op == "placeholder"]
    helper_outputs = [node for node in helper_nodes if node.op == "output"]
    if (
        len(helper_nodes) != 4
        or len(helper_placeholders) != 2
        or len(helper_calls) != 1
        or helper_calls[0].target is not torch.ops.aten.add.Tensor
        or helper_calls[0].args != tuple(helper_placeholders)
        or len(helper_outputs) != 1
        or helper_outputs[0].args != (helper_calls[0],)
    ):
        return None

    body_nodes = list(if_info.graph.nodes)
    stores = [node for node in body_nodes if _store_ref(node) is not None]
    loads = [node for node in body_nodes if _load_ref(node) is not None]
    dots = [node for node in body_nodes if _is_call(node, dot)]
    if len(stores) != 5 or len(loads) != 10 or len(dots) != 10:
        return None
    if any(_store_value(store) is None for store in stores):
        return None

    # First classify outputs by their write value and physical index image.
    gt_store = _single(
        store
        for store in stores
        if (value := _store_value(store)) is not None
        and (meta := _tensor_meta(value)) is not None
        and _shape(meta) == (_DK,)
        and meta.dtype is torch.float32
    )
    aq_store = _single(
        store
        for store in stores
        if (ref := _store_ref(store)) is not None
        and ref.fake.dtype is torch.bfloat16
        and (shape := _shape(ref.fake)) is not None
        and len(shape) == 2
        and shape[-1] == _BT * _BT
        and (value := _store_value(store)) is not None
        and (meta := _tensor_meta(value)) is not None
        and _shape(meta) == (_BT, _BT)
        and _has_scalar_xor(store.args[1], 8)
    )
    vector_stores = [
        store
        for store in stores
        if (ref := _store_ref(store)) is not None
        and ref.fake.dtype is torch.bfloat16
        and (shape := _shape(ref.fake)) is not None
        and len(shape) == 2
        and shape[-1] == _DK
        and (value := _store_value(store)) is not None
        and (meta := _tensor_meta(value)) is not None
        and _shape(meta) == (_BT, _DK)
    ]
    if not isinstance(gt_store, torch.fx.Node) or not isinstance(
        aq_store, torch.fx.Node
    ):
        return None
    if len(vector_stores) != 3:
        return None
    ak_store = _single(
        store
        for store in vector_stores
        if (indices := _memory_indices(store)) is not None
        and _has_scalar_xor(indices[0], 8)
    )
    regular_vector_stores = [store for store in vector_stores if store is not ak_store]
    if not isinstance(ak_store, torch.fx.Node) or len(regular_vector_stores) != 2:
        return None

    long_dots = [
        node
        for node in dots
        if _dot_signature(node)
        == (
            (_BT, _DK),
            torch.bfloat16,
            (_DK, _BT),
            torch.bfloat16,
            (_BT, _BT),
            torch.float32,
        )
    ]
    fp16_dots = [
        node
        for node in dots
        if _dot_signature(node)
        == (
            (_BT, _BT),
            torch.float16,
            (_BT, _BT),
            torch.float16,
            (_BT, _BT),
            torch.float32,
        )
    ]
    aq_dots = [
        node
        for node in dots
        if _dot_signature(node)
        == (
            (_BT, _BT),
            torch.bfloat16,
            (_BT, _BT),
            torch.bfloat16,
            (_BT, _BT),
            torch.float32,
        )
    ]
    ak_dots = [
        node
        for node in dots
        if _dot_signature(node)
        == (
            (_BT, _BT),
            torch.bfloat16,
            (_BT, _DK),
            torch.bfloat16,
            (_BT, _DK),
            torch.float32,
        )
    ]
    if not (
        len(long_dots) == 2
        and len(fp16_dots) == 6
        and len(aq_dots) == 1
        and len(ak_dots) == 1
    ):
        return None
    if any(len(node.args) < 4 or node.args[3] is not torch.float32 for node in dots):
        return None
    aq_dot, ak_dot = aq_dots[0], ak_dots[0]
    aq_value = _store_value(aq_store)
    ak_value = _store_value(ak_store)
    if (
        _strip_value_casts(aq_value) is not aq_dot
        or _strip_value_casts(ak_value) is not ak_dot
        or not isinstance(aq_dot.args[1], torch.fx.Node)
        or not _is_transpose_of(ak_dot.args[0], aq_dot.args[1])
    ):
        return None

    store_to_dot: dict[torch.fx.Node, torch.fx.Node] = {}
    for store in regular_vector_stores:
        value_root = _strip_value_casts(_store_value(store))
        matched = [
            candidate
            for candidate in long_dots
            if _strip_value_casts(candidate.args[0]) is value_root
        ]
        if len(matched) != 1:
            return None
        store_to_dot[store] = matched[0]
    kd_store = _single(
        store
        for store, candidate in store_to_dot.items()
        if any(user.target is torch.ops.aten.mul.Tensor for user in candidate.users)
    )
    qd_store = _single(
        store
        for store, candidate in store_to_dot.items()
        if any(user.target is torch.ops.aten.where.self for user in candidate.users)
    )
    if not isinstance(kd_store, torch.fx.Node) or not isinstance(
        qd_store, torch.fx.Node
    ):
        return None
    kk_dot, qk_dot = store_to_dot[kd_store], store_to_dot[qd_store]
    kk_mul = _single(
        user for user in kk_dot.users if user.target is torch.ops.aten.mul.Tensor
    )
    qk_where = _single(
        user for user in qk_dot.users if user.target is torch.ops.aten.where.self
    )
    if not isinstance(kk_mul, torch.fx.Node) or not isinstance(qk_where, torch.fx.Node):
        return None
    kk_lower = _single(
        user for user in kk_mul.users if user.target is torch.ops.aten.where.self
    )
    if (
        not isinstance(kk_lower, torch.fx.Node)
        or not _is_call(kk_lower.args[0], torch.ops.aten.gt.Tensor)
        or not _is_call(qk_where.args[0], torch.ops.aten.ge.Tensor)
        or _strip_value_casts(aq_dot.args[0]) is not qk_where
    ):
        return None

    kd = cast("_TensorRef", _store_ref(kd_store))
    qd = cast("_TensorRef", _store_ref(qd_store))
    ak = cast("_TensorRef", _store_ref(ak_store))
    aq = cast("_TensorRef", _store_ref(aq_store))
    g_total = cast("_TensorRef", _store_ref(gt_store))

    load_refs = _unique_refs(cast("_TensorRef", _load_ref(node)) for node in loads)
    a_log = cast(
        "_TensorRef | None",
        _single(
            ref
            for ref in load_refs
            if ref.fake.dtype is torch.float32
            and (shape := _shape(ref.fake)) is not None
            and len(shape) == 1
        ),
    )
    if a_log is None or (a_shape := _shape(a_log.fake)) is None:
        return None
    heads = a_shape[0]
    dt_bias = cast(
        "_TensorRef | None",
        _single(
            ref
            for ref in load_refs
            if ref.fake.dtype is torch.float32 and _shape(ref.fake) == (heads, _DK)
        ),
    )
    beta = cast(
        "_TensorRef | None",
        _single(
            ref
            for ref in load_refs
            if ref.fake.dtype is torch.bfloat16
            and (shape := _shape(ref.fake)) is not None
            and len(shape) == 1
        ),
    )
    if dt_bias is None or beta is None or (beta_shape := _shape(beta.fake)) is None:
        return None
    if heads <= 0 or beta_shape[0] % heads:
        return None
    total_tokens = beta_shape[0] // heads
    activation_refs = [
        ref
        for ref in load_refs
        if ref.fake.dtype in (torch.bfloat16, torch.float32)
        and _shape(ref.fake) == (total_tokens * heads, _DK)
        and _ref_key(ref) != _ref_key(dt_bias)
    ]
    if len(activation_refs) != 3:
        return None
    activation_keys = {_ref_key(ref): ref for ref in activation_refs}
    gt_deps = _ancestor_ref_keys(_store_value(gt_store)) & activation_keys.keys()
    kd_deps = _ancestor_ref_keys(_store_value(kd_store)) & activation_keys.keys()
    qd_deps = _ancestor_ref_keys(_store_value(qd_store)) & activation_keys.keys()
    if len(gt_deps) != 1 or len(kd_deps) != 2 or len(qd_deps) != 2:
        return None
    gate_key = next(iter(gt_deps))
    if gate_key not in kd_deps or gate_key not in qd_deps:
        return None
    k_keys = kd_deps - {gate_key}
    q_keys = qd_deps - {gate_key}
    if len(k_keys) != 1 or len(q_keys) != 1 or k_keys == q_keys:
        return None
    gate = activation_keys[gate_key]
    k = activation_keys[next(iter(k_keys))]
    q = activation_keys[next(iter(q_keys))]
    if q.fake.dtype is not torch.bfloat16 or k.fake.dtype is not torch.bfloat16:
        return None
    gate_is_fp32 = gate.fake.dtype is torch.float32
    if gate.fake.dtype not in (torch.bfloat16, torch.float32):
        return None

    # The full target multiset makes every rounding/cast operation part of the
    # accepted contract while remaining independent of source/node names.
    target_counts = Counter(
        node.target for node in body_nodes if node.op == "call_function"
    )
    if target_counts == _body_target_counts(gate_is_fp32):
        outputs_scaled = True
    elif target_counts == _body_target_counts(gate_is_fp32, outputs_scaled=False):
        outputs_scaled = False
    else:
        return None
    factor_key_xor = 0 if outputs_scaled else 8
    iota_extents = [
        _static_extent(node.args[0])
        for node in body_nodes
        if _is_call(node, torch.ops.prims.iota.default) and node.args
    ]
    if any(extent is None for extent in iota_extents) or sorted(
        cast("list[int]", iota_extents)
    ) != [_BT, _BT, _DK]:
        return None
    scans = [node for node in body_nodes if _is_call(node, scan_ops._associative_scan)]
    if (
        len(scans) != 1
        or len(scans[0].args) < 5
        or scans[0].args[0] != helper_info.graph_id
        or scans[0].args[2:] != (0, False, False)
    ):
        return None

    gt_shape = _shape(g_total.fake)
    if (
        gt_shape is None
        or len(gt_shape) != 2
        or gt_shape[1] != _DK
        or gt_shape[0] % heads
    ):
        return None
    total_chunks = gt_shape[0] // heads
    chunk_group_coord = _match_chunk_group_coordinate(
        loop_info,
        if_call.args[0],
        loop_call.args[2][0],
        total_chunks,
    )
    if chunk_group_coord is None:
        return None
    rows = total_chunks * _BT
    if not all(
        (
            _tensor_is_exact_contiguous(ref, (heads * rows, _DK))
            and ref.fake.dtype is torch.bfloat16
        )
        for ref in (kd, qd, ak)
    ):
        return None
    if not (
        _tensor_is_exact_contiguous(aq, (heads * total_chunks, _BT * _BT))
        and aq.fake.dtype is torch.bfloat16
        and _tensor_is_exact_contiguous(g_total, (heads * total_chunks, _DK))
        and g_total.fake.dtype is torch.float32
    ):
        return None

    int_refs = [ref for ref in load_refs if ref.fake.dtype is torch.int32]
    if len(int_refs) != 3 or any(
        ref.fake.ndim != 1 or not ref.fake.is_contiguous() for ref in int_refs
    ):
        return None
    chunk_to_seq = cast(
        "_TensorRef | None",
        _single(ref for ref in int_refs if _shape(ref.fake) == (total_chunks,)),
    )
    if chunk_to_seq is None:
        return None
    remaining_int = [ref for ref in int_refs if ref is not chunk_to_seq]
    load_counts = Counter(
        _ref_key(cast("_TensorRef", _load_ref(node))) for node in loads
    )
    cu_seqlens = cast(
        "_TensorRef | None",
        _single(ref for ref in remaining_int if load_counts[_ref_key(ref)] == 2),
    )
    cu_chunks = cast(
        "_TensorRef | None",
        _single(ref for ref in remaining_int if load_counts[_ref_key(ref)] == 1),
    )
    if (
        cu_seqlens is None
        or cu_chunks is None
        or _shape(cu_seqlens.fake) != _shape(cu_chunks.fake)
    ):
        return None
    if not _validate_memory_index_topology(
        loads=loads,
        kd_store=kd_store,
        qd_store=qd_store,
        ak_store=ak_store,
        aq_store=aq_store,
        gt_store=gt_store,
        q=q,
        k=k,
        gate=gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        cu_chunks=cu_chunks,
        chunk_to_seq=chunk_to_seq,
        total_chunks=total_chunks,
        heads=heads,
        loop_predicate=if_call.args[0],
        factor_key_xor=factor_key_xor,
    ):
        return None
    if not _validate_prepare_semantics(
        body_nodes=body_nodes,
        loads=loads,
        kd_store=kd_store,
        qd_store=qd_store,
        ak_store=ak_store,
        aq_store=aq_store,
        gt_store=gt_store,
        kk_dot=kk_dot,
        qk_dot=qk_dot,
        aq_dot=aq_dot,
        ak_dot=ak_dot,
        fp16_dots=fp16_dots,
        q=q,
        k=k,
        gate=gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        outputs_scaled=outputs_scaled,
    ):
        return None

    dt_load = _single_load_for_ref(loads, dt_bias)
    dt_indices = _memory_indices(dt_load) if dt_load is not None else None
    head_coord = (
        _strip_index_broadcast(dt_indices[0])
        if dt_indices is not None and len(dt_indices) == 2
        else None
    )
    if not isinstance(head_coord, torch.fx.Node):
        return None

    if any(
        not ref.fake.is_contiguous() or ref.fake.device.type != "cuda"
        for ref in (
            q,
            k,
            gate,
            beta,
            a_log,
            dt_bias,
            cu_seqlens,
            cu_chunks,
            chunk_to_seq,
        )
    ):
        return None
    output_storage = kd.fake.untyped_storage()
    if any(
        ref.fake.untyped_storage() is output_storage
        for ref in (
            q,
            k,
            gate,
            beta,
            a_log,
            dt_bias,
            cu_seqlens,
            cu_chunks,
            chunk_to_seq,
        )
    ):
        return None

    symfloat_nodes = [
        node
        for node in body_nodes
        if _is_call(node, _get_symnode)
        and isinstance(node.meta.get("val"), torch.SymFloat)
    ]
    log2_e_nodes = [node for node in symfloat_nodes if _static_float(node) == _LOG2_E]
    if len(log2_e_nodes) != 1:
        return None
    symfloat_nodes.remove(log2_e_nodes[0])
    expected_symfloat_nodes = 2 if outputs_scaled else 1
    if len(symfloat_nodes) != expected_symfloat_nodes:
        return None
    gt_ancestors = _ancestors(_store_value(gt_store))
    qd_ancestors = _ancestors(_store_value(qd_store))
    gate_scale_node = cast(
        "torch.fx.Node | None",
        _single(node for node in symfloat_nodes if node in gt_ancestors),
    )
    scale_node = cast(
        "torch.fx.Node | None",
        _single(
            node
            for node in symfloat_nodes
            if node in qd_ancestors and node not in gt_ancestors
        ),
    )
    if gate_scale_node is None or (outputs_scaled and scale_node is None):
        return None
    scale_value = (
        cast("torch.SymFloat", scale_node.meta["val"])._sympy_()
        if scale_node is not None
        else sympy.Float(1.0)
    )
    gate_scale_value = cast("torch.SymFloat", gate_scale_node.meta["val"])._sympy_()
    if not isinstance(scale_value, sympy.Expr) or not isinstance(
        gate_scale_value, sympy.Expr
    ):
        return None

    return _CuteChunkPrepareMatch(
        root_graph_id=root_info.graph_id,
        root_phase_index=root_info.phase_index,
        heads=heads,
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        gate_is_fp32=gate_is_fp32,
        outputs_scaled=outputs_scaled,
        factor_key_xor=factor_key_xor,
        q=q,
        k=k,
        gate=gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        cu_chunks=cu_chunks,
        chunk_to_seq=chunk_to_seq,
        kd=kd,
        qd=qd,
        ak=ak,
        aq=aq,
        g_total=g_total,
        chunk_group_coord=chunk_group_coord,
        head_coord=head_coord,
        scale=scale_value,
        gate_scale_log2=gate_scale_value,
    )


def preferred_chunk_prepare_schedule(
    graphs: Sequence[GraphInfo],
    *,
    env: CompileEnvironment | None = None,
    num_sm: int,
) -> str | None:
    """Choose a geometry-derived seed while leaving all schedules searchable."""

    if env is None:
        env = CompileEnvironment.current()
    if not env.settings.fast_math:
        return None
    match = _match_chunk_prepare_graphs(graphs)
    if match is None:
        return None
    if not _register_packed_workspace_specialization(
        env,
        tuple(
            ref.fake for ref in (match.kd, match.qd, match.ak, match.aq, match.g_total)
        ),
    ):
        return None
    chunks_per_cta = _preferred_split_alias_chunks_per_cta(
        total_chunks=match.total_chunks,
        heads=match.heads,
        num_sm=num_sm,
    )
    return f"split_alias_cpc{chunks_per_cta}"


def _plan_chunk_prepare(
    graphs: Sequence[GraphInfo],
    _tile_strategy: TileStrategyDispatch,
) -> CuteChunkPreparePlan | None:
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction

    device_function = DeviceFunction.current()
    env = CompileEnvironment.current()
    if not env.settings.fast_math or device_function.config.pid_type != "flat":
        return None
    match = _match_chunk_prepare_graphs(graphs)
    if match is None:
        return None

    device_ir = HostFunction.current().device_ir
    if not _packed_workspace_is_proven(env):
        return None
    grid_ids = _canonical_root_axis_ids(
        device_ir,
        root_phase_index=match.root_phase_index,
        coordinates=(match.chunk_group_coord, match.head_coord),
        expected_extents=(
            (match.total_chunks + _SEMANTIC_CPC - 1) // _SEMANTIC_CPC,
            match.heads,
        ),
    )
    if grid_ids is None:
        return None
    if any(device_function.resolved_block_size(index) != 1 for index in grid_ids):
        return None
    target = env.config_spec.target_device_capability
    if target is None or target < (9, 0):
        return None

    schedule = device_function.config.get(
        CUTE_CHUNK_PREPARE_SCHEDULE_KEY,
        preferred_chunk_prepare_schedule(
            graphs,
            env=env,
            num_sm=env.config_spec.num_sm,
        ),
    )
    if type(schedule) is not str or schedule not in _VALID_PREPARE_SCHEDULES:
        return None
    chunks_per_cta = int(schedule.removeprefix("split_alias_cpc"))
    refs = (
        match.q,
        match.k,
        match.gate,
        match.beta,
        match.a_log,
        match.dt_bias,
        match.cu_seqlens,
        match.cu_chunks,
        match.chunk_to_seq,
        match.kd,
        match.qd,
        match.ak,
        match.aq,
        match.g_total,
    )
    grid = (
        (match.total_chunks + chunks_per_cta - 1) // chunks_per_cta,
        match.heads,
        1,
    )
    if not _linear_offsets_fit_i32(
        match.total_chunks * match.heads,
        tuple(ref.fake.numel() for ref in refs),
    ) or not _xyz_grid_fits(grid):
        return None
    return CuteChunkPreparePlan(
        root_graph_id=match.root_graph_id,
        chunk_size=_BT,
        key_size=_DK,
        chunks_per_cta=chunks_per_cta,
        schedule=schedule,
        heads=match.heads,
        total_tokens=match.total_tokens,
        total_chunks=match.total_chunks,
        gate_is_fp32=match.gate_is_fp32,
        outputs_scaled=match.outputs_scaled,
        factor_key_xor=match.factor_key_xor,
        q=match.q,
        k=match.k,
        gate=match.gate,
        beta=match.beta,
        a_log=match.a_log,
        dt_bias=match.dt_bias,
        cu_seqlens=match.cu_seqlens,
        cu_chunks=match.cu_chunks,
        chunk_to_seq=match.chunk_to_seq,
        kd=match.kd,
        qd=match.qd,
        ak=match.ak,
        aq=match.aq,
        g_total=match.g_total,
        scale=match.scale,
        gate_scale_log2=match.gate_scale_log2,
        smem_bytes=_SPLIT_ALIAS_SMEM_BYTES,
        device_abi=(
            _SCALED_DEVICE_ABI if match.outputs_scaled else _UNSCALED_DEVICE_ABI
        ),
    )


def plan_chunk_prepare(
    graphs: Sequence[GraphInfo],
    tile_strategy: TileStrategyDispatch,
) -> None:
    """Install a complete whole-root prepare plan, or leave no partial state."""

    from ..device_function import DeviceFunction

    DeviceFunction.current().cute_state.chunk_prepare_plan = _plan_chunk_prepare(
        graphs, tile_strategy
    )


def _tensor_arg(cg: GenerateAST, ref: _TensorRef) -> str:
    return cg.device_function.tensor_arg(ref.fake, prefer_name=ref.name).name


def _flat_tensor(name: str, elements: int) -> str:
    return (
        f"cute.make_tensor({name}.iterator, "
        f"cute.make_layout(({elements},), stride=(1,)))"
    )


def _emit_module_imports(cg: GenerateAST, helper_name: str) -> None:
    if getattr(cg, "_helion_chunk_prepare_module_emitted", False):
        return
    cg._helion_chunk_prepare_module_emitted = True  # type: ignore[attr-defined]
    alias = "" if helper_name == "emit_bt16_prepare" else f" as {helper_name}"
    source = (
        "from helion._compiler.cute.chunk_prepare_split_alias_device "
        f"import emit_bt16_prepare{alias}"
    )
    cg.module_statements.extend(ast.parse(source).body)


def codegen_chunk_prepare(cg: GenerateAST) -> bool:
    """Replace the matched root with the exact imported device schedule."""

    df = cg.device_function
    plan = df.cute_state.chunk_prepare_plan
    root = cg.current_root_graph_info
    if (
        plan is None
        or root is None
        or root.graph_id != plan.root_graph_id
        or plan.chunk_size != _BT
        or plan.key_size != _DK
        or plan.chunks_per_cta not in _SCHEDULE_CPC_CHOICES
        or plan.schedule not in _VALID_PREPARE_SCHEDULES
        or (
            plan.schedule in _SPLIT_ALIAS_SCHEDULES
            and plan.schedule != f"split_alias_cpc{plan.chunks_per_cta}"
        )
        or plan.threads != _THREADS
        or plan.smem_bytes != _SPLIT_ALIAS_SMEM_BYTES
        or plan.bf16_mma_count + plan.fp16_mma_count != 192
        or plan.device_abi
        != (_SCALED_DEVICE_ABI if plan.outputs_scaled else _UNSCALED_DEVICE_ABI)
        or plan.factor_key_xor != (0 if plan.outputs_scaled else 8)
    ):
        return False

    refs = (
        plan.q,
        plan.k,
        plan.gate,
        plan.beta,
        plan.a_log,
        plan.dt_bias,
        plan.cu_seqlens,
        plan.cu_chunks,
        plan.chunk_to_seq,
        plan.kd,
        plan.qd,
        plan.ak,
        plan.aq,
        plan.g_total,
    )
    names = tuple(_tensor_arg(cg, ref) for ref in refs)
    (
        q,
        k,
        gate,
        beta,
        a_log,
        dt_bias,
        cu_seqlens,
        cu_chunks,
        chunk_to_seq,
        kd,
        qd,
        ak,
        aq,
        g_total,
    ) = names
    scale = df.literal_expr(plan.scale)
    gate_scale_log2 = df.literal_expr(plan.gate_scale_log2)
    desc_names = tuple(
        df.new_var(name)
        for name in (
            "_chunk_prepare_desc_q",
            "_chunk_prepare_desc_k",
            "_chunk_prepare_desc_g",
            "_chunk_prepare_desc_factor",
        )
    )
    helper_name = df.new_var("emit_bt16_prepare")

    _emit_module_imports(cg, helper_name)
    wrapper_plan: dict[str, object] = {
        "kind": "chunk_prepare_tma",
        "q_name": q,
        "k_name": k,
        "g_name": gate,
        "beta_name": beta,
        "a_log_name": a_log,
        "dt_name": dt_bias,
        "cu_seqlens_name": cu_seqlens,
        "cu_chunks_name": cu_chunks,
        "chunk_to_seq_name": chunk_to_seq,
        "kd_name": kd,
        "qd_name": qd,
        "ak_name": ak,
        "aq_name": aq,
        "gt_name": g_total,
        "desc_args": desc_names,
        "total_tokens": plan.total_tokens,
        "total_chunks": plan.total_chunks,
        "heads": plan.heads,
        "chunk_size": plan.chunk_size,
        "key_size": plan.key_size,
        "chunks_per_cta": plan.chunks_per_cta,
        "schedule": plan.schedule,
        "gate_is_fp32": plan.gate_is_fp32,
        "outputs_scaled": plan.outputs_scaled,
        "factor_key_xor": plan.factor_key_xor,
        "device_abi": plan.device_abi,
        "smem_bytes": plan.smem_bytes,
        "bf16_mma_count": plan.bf16_mma_count,
        "fp16_mma_count": plan.fp16_mma_count,
    }
    cg.cute_wrapper_plans.append(wrapper_plan)
    df.wrapper_only_params.extend(desc_names)
    df.placeholder_args.update(names)

    flat_names = tuple(
        df.new_var(f"_chunk_prepare_arg_{index}") for index in range(len(names))
    )
    numels = tuple(ref.fake.numel() for ref in refs)
    preamble = "\n".join(
        f"{flat} = {_flat_tensor(name, numel)}"
        for flat, name, numel in zip(flat_names, names, numels, strict=True)
    )
    args = ",\n    ".join(
        [
            *flat_names,
            *desc_names,
            f"cutlass.Float32({scale})",
            f"cutlass.Float32({gate_scale_log2})",
            f"cutlass.Int32({plan.total_chunks})",
            f"cutlass.Int32({plan.heads})",
            str(plan.outputs_scaled),
            str(plan.factor_key_xor),
            str(plan.chunks_per_cta),
            str(plan.gate_is_fp32),
        ]
    )
    df.preamble = cast("list[ast.AST]", ast.parse(preamble).body)
    df.body = cast(
        "list[ast.AST]",
        ast.parse(f"{helper_name}(\n    {args},\n)").body,
    )
    cg.cute_uses_matmul = True
    return True
