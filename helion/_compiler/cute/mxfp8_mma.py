"""Prove complete E4M3/E8M0 contractions for native Blackwell block scaling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import view_ops
from ...language.matmul_ops import dot_scaled
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..host_function import HostFunction
from ..indexing_strategy import subscript_tile_info
from .block_scaled_config import mxfp8_epilogue_parameters
from .block_scaled_config import mxfp8_epilogue_supported
from .block_scaled_config import mxfp8_shared_memory_bytes
from .block_scaled_mma import BlockScaledMmaPlan
from .block_scaled_mma import PreparedOperand
from .block_scaled_mma import _metadata_resolver
from .block_scaled_mma import _nonoverlapping_output
from .block_scaled_provenance import PackedOperand
from .promote_output_axis import _fresh_tensors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo


@dataclass(frozen=True)
class Mxfp8Match:
    graph_id: int
    lhs: torch.Tensor
    rhs: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    output: torch.Tensor
    row_axes: tuple[int, int]
    round_dtypes: tuple[torch.dtype, ...]
    rhs_transposed: bool
    m: int
    n: int
    k: int
    partition_key: str | None = None


def _host_tensor(node: object) -> torch.Tensor | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not _tracing_ops._host_tensor
    ):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) and value.ndim == 2 else None


def _tile_load(node: object) -> tuple[torch.Tensor, list[object]] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not memory_ops.load
        or len(node.args) != 4
        or node.args[2:] != (None, None)
        or node.kwargs
        or not isinstance(node.args[1], (list, tuple))
        or len(node.args[1]) != 2
    ):
        return None
    tensor = _host_tensor(node.args[0])
    return (tensor, list(node.args[1])) if tensor is not None else None


def _packed_group_expression(
    node: object, groups: int, *, budget: int = 128
) -> sympy.Expr | None:
    """Prove a broadcasted full group iota and its bounded integer arithmetic."""
    remaining = budget
    coordinate = sympy.Symbol("group", integer=True, nonnegative=True)
    if groups <= 0 or groups >= 1 << 26:
        return None
    quotient = sympy.floor(sympy.Mul(coordinate, sympy.Rational(1, 4)))
    remainder = sympy.Mod(coordinate, 4)
    if not isinstance(remainder, sympy.Expr):
        return None

    def visit(value: object) -> sympy.Expr | None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            return None
        if type(value) is int:
            return sympy.Integer(value)
        if not isinstance(value, torch.fx.Node) or value.op != "call_function":
            return None
        if value.target in (memory_ops.load, view_ops.subscript):
            if len(value.args) not in (2, 4) or value.kwargs:
                return None
            if len(value.args) == 4 and value.args[2:] != (None, None):
                return None
            if not isinstance(value.args[1], (list, tuple)) or any(
                item is not None and item != slice(None) for item in value.args[1]
            ):
                return None
            return visit(value.args[0])
        if value.target is torch.ops.aten.unsqueeze.default and len(value.args) == 2:
            return visit(value.args[0])
        if value.target is torch.ops.prims.convert_element_type.default:
            if len(value.args) != 2 or value.args[1] not in (torch.int32, torch.int64):
                return None
            return visit(value.args[0])
        if value.target in (
            torch.ops.aten.arange.default,
            torch.ops.aten.arange.start,
            torch.ops.aten.arange.start_step,
        ):
            if value.kwargs.get("dtype", torch.int64) not in (torch.int32, torch.int64):
                return None
            values = tuple(value.args)
            if values not in ((groups,), (0, groups), (0, groups, 1)):
                return None
            return coordinate
        if value.target is torch.ops.prims.iota.default:
            if (
                value.args != (groups,)
                or value.kwargs.get("start", 0) != 0
                or value.kwargs.get("step", 1) != 1
                or value.kwargs.get("dtype", torch.int64)
                not in (torch.int32, torch.int64)
            ):
                return None
            return coordinate
        if len(value.args) != 2:
            return None
        left, right = (visit(item) for item in value.args)
        if left is None or right is None:
            return None
        # Match the bounded byte recipe without algebraic cancellation. A
        # wider integer expression can overflow before simplifying to this
        # formula, so arbitrary arithmetic must not become native provenance.
        if value.target in (
            operator.floordiv,
            torch.ops.aten.div.Tensor_mode,
        ):
            if value.target is torch.ops.aten.div.Tensor_mode:
                if value.kwargs != {"rounding_mode": "floor"}:
                    return None
            elif value.kwargs:
                return None
            return quotient if left == coordinate and right == 4 else None
        if value.kwargs:
            return None
        if value.target in (
            operator.mod,
            torch.ops.aten.remainder.Scalar,
            torch.ops.aten.remainder.Tensor,
        ):
            return remainder if left == coordinate and right == 4 else None
        if value.target in (operator.mul, torch.ops.aten.mul.Tensor):
            if (left == remainder and right == 8) or (right == remainder and left == 8):
                return sympy.Mul(remainder, 8)
        return None

    return visit(node)


def _scale(node: object, row: object, groups: int) -> torch.Tensor | None:
    host = _host_tensor(node)
    if host is not None and host.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return host
    loaded = _tile_load(node)
    if (
        loaded is not None
        and loaded[1] == [row, slice(None)]
        and loaded[0].dtype
        in (
            torch.uint8,
            torch.float8_e8m0fnu,
        )
    ):
        return loaded[0]
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not torch.ops.prims.convert_element_type.default
        or node.args[1:] != (torch.uint8,)
        or node.kwargs
    ):
        return None
    masked = node.args[0]
    if (
        not isinstance(masked, torch.fx.Node)
        or masked.target not in (operator.and_, torch.ops.aten.bitwise_and.Scalar)
        or len(masked.args) != 2
        or masked.args[1] != 255
        or masked.kwargs
    ):
        return None
    shifted = masked.args[0]
    if (
        not isinstance(shifted, torch.fx.Node)
        or shifted.target
        not in (
            operator.rshift,
            torch.ops.aten.bitwise_right_shift.Tensor,
            torch.ops.aten.__rshift__.Tensor,
        )
        or len(shifted.args) != 2
        or shifted.kwargs
    ):
        return None
    words = _tile_load(shifted.args[0])
    if words is None or words[0].dtype != torch.int32 or words[1][0] != row:
        return None
    index_value = (
        words[1][1].meta.get("val") if isinstance(words[1][1], torch.fx.Node) else None
    )
    shift_value = (
        shifted.args[1].meta.get("val")
        if isinstance(shifted.args[1], torch.fx.Node)
        else None
    )
    if (
        not isinstance(index_value, torch.Tensor)
        or index_value.ndim != 1
        or not isinstance(shift_value, torch.Tensor)
        or not (
            shift_value.ndim == 1
            or (
                shift_value.ndim == 2
                and type(shift_value.shape[0]) is int
                and shift_value.shape[0] == 1
            )
        )
    ):
        return None
    # Full iota provenance below proves the exact group extent. A broadcasted
    # view's metadata may use a symbolic full-slice axis before config binding;
    # requiring a literal shape here would lose the same source's native seeds.
    group = sympy.Symbol("group", integer=True, nonnegative=True)
    remainder = sympy.Mod(group, 4)
    if not isinstance(remainder, sympy.Expr):
        return None
    if _packed_group_expression(words[1][1], groups) != sympy.floor(
        sympy.Mul(group, sympy.Rational(1, 4))
    ) or _packed_group_expression(shifted.args[1], groups) != sympy.Mul(remainder, 8):
        return None
    return words[0]


def _unmask_zero_operand(node: object) -> object:
    """Native padding reproduces a full contraction's zero-filled tile lanes."""
    while (
        isinstance(node, torch.fx.Node)
        and node.target is _tracing_ops._mask_to
        and len(node.args) == 2
        and not node.kwargs
        and isinstance(node.args[1], (int, float))
        and node.args[1] == 0
        and math.copysign(1, node.args[1]) > 0
    ):
        node = node.args[0]
    return node


def _match_mxfp8(graphs: Sequence[GraphInfo], ir: DeviceIR) -> Mxfp8Match | None:
    """Accept plain full-K loads, a zero FP32 carry, and a rounding-only store."""
    from .registered_mxfp8 import match_registered_mxfp8

    registered = match_registered_mxfp8(graphs, ir)
    if registered is not None:
        return registered
    env = CompileEnvironment.current()
    if len(ir.root_ids) != 1 or len(ir.phases) != 1:
        return None
    root = graphs[ir.root_ids[0]]
    dots = list(root.graph.find_nodes(op="call_function", target=dot_scaled))
    stores = list(root.graph.find_nodes(op="call_function", target=memory_ops.store))
    if len(dots) != 1 or len(stores) != 1:
        return None
    dot, store = dots[0], stores[0]
    if (
        len(dot.args) != 8
        or dot.args[2] != "e4m3"
        or dot.args[5] != "e4m3"
        or dot.args[7] not in (None, torch.float32)
        or dot.kwargs
        or dot.meta["val"].dtype != torch.float32
    ):
        return None
    lhs = _tile_load(_unmask_zero_operand(dot.args[0]))
    rhs_node = _unmask_zero_operand(dot.args[3])
    permuted = (
        isinstance(rhs_node, torch.fx.Node)
        and rhs_node.target is torch.ops.aten.permute.default
        and len(rhs_node.args) == 2
        and rhs_node.args[1] == [1, 0]
        and not rhs_node.kwargs
    )
    if permuted:
        rhs_node = cast("torch.fx.Node", rhs_node).args[0]
    rhs = _tile_load(_unmask_zero_operand(rhs_node))
    if lhs is None or rhs is None or lhs[1][1] != slice(None):
        return None
    rhs_row = rhs[1][0] if permuted else rhs[1][1]
    if (rhs[1][1] if permuted else rhs[1][0]) != slice(None):
        return None
    groups = env.size_hint(lhs[0].shape[1]) // 32
    sa, sb = (
        _scale(dot.args[1], lhs[1][0], groups),
        _scale(dot.args[4], rhs_row, groups),
    )
    output = _host_tensor(store.args[0])
    if (
        sa is None
        or sb is None
        or output is None
        or len(store.args) != 4
        or store.args[3] is not None
        or store.kwargs
        or list(store.args[1]) != [lhs[1][0], rhs_row]
        or lhs[0].dtype != torch.float8_e4m3fn
        or rhs[0].dtype != torch.float8_e4m3fn
        or sa.dtype not in (torch.uint8, torch.float8_e8m0fnu, torch.int32)
        or sb.dtype not in (torch.uint8, torch.float8_e8m0fnu, torch.int32)
    ):
        return None
    acc = dot.args[6]
    if acc is not None and (
        not isinstance(acc, torch.fx.Node)
        or acc.target is not creation_ops.full
        or not isinstance(acc.args[1], (int, float))
        or acc.args[1] != 0
        or math.copysign(1, acc.args[1]) < 0
        or acc.meta["val"].dtype != torch.float32
    ):
        return None
    value = store.args[2]
    round_dtypes = []
    while (
        isinstance(value, torch.fx.Node)
        and value.target is torch.ops.prims.convert_element_type.default
        and len(value.args) == 2
        and not value.kwargs
        and value.args[1] in (torch.float16, torch.bfloat16, torch.float32)
    ):
        round_dtypes.append(cast("torch.dtype", value.args[1]))
        value = value.args[0]
    if value is not dot or output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return None
    allowed = {store}
    pending = [store]
    while pending:
        for node in pending.pop().all_input_nodes:
            if node not in allowed:
                allowed.add(node)
                pending.append(node)
    if any(node.op != "output" and node not in allowed for node in root.graph.nodes):
        return None
    axes = []
    for row in (lhs[1][0], rhs_row):
        tile = subscript_tile_info(env, row)
        if (
            tile is None
            or not env.known_equal(tile.offset, 0)
            or tile.block_size is not None
        ):
            return None
        axes.append(env.canonical_block_id(tile.block_id))
    if tuple(ir.grid_block_ids[0]) != tuple(axes):
        return None
    resolve = _metadata_resolver(env)
    m, k = [resolve(s) for s in lhs[0].shape]
    n = resolve(rhs[0].shape[0 if permuted else 1])
    rk = resolve(rhs[0].shape[1 if permuted else 0])
    if m is None or n is None or k is None:
        return None
    if any(x <= 0 or x >= 1 << 31 for x in (m, n, k)) or rk != k or k % 32:
        return None
    if tuple(resolve(x) for x in output.shape) != (m, n):
        return None
    if any(
        resolve(env.block_sizes[axis].numel) != extent
        for axis, extent in zip(axes, (m, n), strict=True)
    ):
        return None
    for tensor, rows in ((sa, m), (sb, n)):
        columns = (k + 127) // 128 if tensor.dtype == torch.int32 else k // 32
        if tuple(resolve(x) for x in tensor.shape) != (rows, columns):
            return None
        if any(value is None or value < 0 for value in map(resolve, tensor.stride())):
            return None
        if tensor.dtype == torch.int32:
            rs, cs = tuple(resolve(x) for x in tensor.stride())
            if rs != 1 or cs is None or cs < rows or cs % 4:
                return None
    for tensor, transposed in ((lhs[0], False), (rhs[0], not permuted)):
        strides = tuple(resolve(x) for x in tensor.stride())
        row_stride, k_stride = strides[::-1] if transposed else strides
        if k_stride != 1 or row_stride is None or row_stride < k or row_stride % 16:
            return None
    if not _nonoverlapping_output(output, resolve):
        return None
    return Mxfp8Match(
        root.graph_id,
        lhs[0],
        rhs[0],
        sa,
        sb,
        output,
        (axes[0], axes[1]),
        tuple(reversed(round_dtypes)),
        not permuted,
        m,
        n,
        k,
    )


def _sf_tma_eligible(match: Mxfp8Match) -> bool:
    """Tensor-map legality for the canonical packed scale operands.

    Unit row coordinates are restored independently of descriptor bounds.
    The packed word axis remains non-unit; runtime staging preserves strides
    while aligning replacement allocations.
    """
    if match.scale_a.dtype != torch.int32 or match.scale_b.dtype != torch.int32:
        return False
    if (match.k + 127) // 128 <= 1:
        return False
    resolve = _metadata_resolver(CompileEnvironment.current())
    for scale in (match.scale_a, match.scale_b):
        row_stride, column_stride = tuple(resolve(value) for value in scale.stride())
        if (
            row_stride != 1
            or column_stride is None
            or not 0 < column_stride * 4 < 1 << 40
            or column_stride % 4
        ):
            return False
    return True


def _tma_output_eligible(match: Mxfp8Match) -> bool:
    """Prove the two coordinate axes and the BF16 row-major tensor map."""
    output = match.output
    if output.ndim != 2:
        return False
    resolve = _metadata_resolver(CompileEnvironment.current())
    row_stride, column_stride = map(resolve, output.stride())
    return (
        output.dtype == torch.bfloat16
        and match.scale_a.dtype == match.scale_b.dtype == torch.int32
        and match.m > 1
        and match.n > 1
        and column_stride == 1
        and row_stride is not None
        and row_stride >= match.n
        and row_stride % 8 == 0
    )


def expose_mxfp8(device_ir: DeviceIR) -> bool:
    match = _match_mxfp8(device_ir.graphs, device_ir)
    if match is None:
        return False
    spec = CompileEnvironment.current().config_spec
    spec.cute_scaled_mma_available = True
    spec.cute_mxfp8_mma_available = True
    spec.cute_mxfp8_partition_key = match.partition_key
    spec.cute_mxfp8_fused_scales_available = (
        match.scale_a.dtype == match.scale_b.dtype == torch.int32
    )
    spec.cute_mxfp8_sf_tma_available = _sf_tma_eligible(match)
    spec.cute_mxfp8_problem_shape = (match.m, match.n, match.k)
    spec.cute_mxfp8_output_element_size = match.output.element_size()
    spec.cute_mxfp8_tma_output_available = _tma_output_eligible(match)
    return True


def plan_mxfp8(graphs: Sequence[GraphInfo]) -> BlockScaledMmaPlan | None:
    env, df, host = (
        CompileEnvironment.current(),
        DeviceFunction.current(),
        HostFunction.current(),
    )
    if (
        env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] != 10
        or df.config.pid_type != "flat"
    ):
        return None
    match = _match_mxfp8(graphs, host.device_ir)
    if (
        match is None
        or "input_tensor_metadata" not in env.compiler_fact_specialization_facts
    ):
        return None
    if match.partition_key is not None and (
        type(df.config.get(match.partition_key)) is not int
        or df.config.get(match.partition_key) != 1
    ):
        return None
    refs = (match.lhs, match.rhs, match.scale_a, match.scale_b, match.output)
    if match.partition_key is None:
        fresh = _fresh_tensors(host)
    else:
        from .registered_mxfp8 import registered_host_freshness

        symbol = next(
            symbol
            for symbol, expression in env.config_value_expressions.items()
            if expression.operation == "config"
            and expression.arguments == (match.partition_key,)
        )
        proven = registered_host_freshness(host, symbol)
        if proven is None:
            return None
        fresh = proven[0]
    if any(env.tensor_input_source(t) is None and t not in fresh for t in refs):
        return None
    names = [df.tensor_arg(t).name for t in refs]
    fresh_registered_output = (
        match.partition_key is not None
        and match.output in fresh
        and all(env.tensor_input_source(tensor) is not None for tensor in refs[:-1])
    )
    if not fresh_registered_output and any(
        frozenset((name, names[-1])) not in df.proven_disjoint_tensor_pairs()
        for name in names[:-1]
    ):
        return None
    m, n, k = match.m, match.n, match.k
    fused_scales = bool(df.config.get("cute_scaled_fused_scales", False))
    logical_m_per_cta = df.config.get("cute_scaled_tile_m", 128)
    swap_ab = df.config.get("cute_scaled_swap_ab", False)
    compact_m = df.config.get("cute_scaled_compact_m", False)
    sf_tma = df.config.get("cute_scaled_sf_tma", False)
    sf_inplace = df.config.get("cute_scaled_sf_inplace", False)
    epilogue_mode = df.config.get("cute_scaled_epilogue_mode", "default")
    if not isinstance(epilogue_mode, str):
        return None
    if fused_scales and not (match.scale_a.dtype == match.scale_b.dtype == torch.int32):
        return None
    if type(sf_tma) is not bool or (
        sf_tma and (not fused_scales or not _sf_tma_eligible(match))
    ):
        return None
    if type(sf_inplace) is not bool or (sf_inplace and not sf_tma):
        return None
    bn, bk, stages, cluster, persistent = list(
        starmap(
            df.config.get,
            (
                ("cute_scaled_tile_n", 256),
                ("cute_scaled_tile_k", 256),
                ("cute_scaled_stages", 4),
                ("cute_scaled_cluster_m", 2),
                ("cute_scaled_persistent", True),
            ),
        )
    )
    if (
        bn not in (8, 16, 32, 64, 128, 224, 256)
        or bk not in (128, 256)
        or stages not in (2, 3, 4, 5, 6, 8, 10, 12)
        or cluster not in (1, 2)
        or (bn == 8 and cluster != 1)
        or not isinstance(persistent, bool)
        or logical_m_per_cta not in (32, 64, 128)
        or not isinstance(swap_ab, bool)
        or not isinstance(compact_m, bool)
    ):
        return None
    runtime_m = n if swap_ab else m
    compact = compact_m and logical_m_per_cta < 128 and runtime_m > 1
    if not mxfp8_epilogue_supported(
        epilogue_mode,
        bn,
        bk,
        logical_m_per_cta,
        swap_ab=swap_ab,
        fused_scales=fused_scales,
        tma_output_available=_tma_output_eligible(match),
    ):
        return None
    accumulator_stages, prefetch_output_registers, tma_output = (
        mxfp8_epilogue_parameters(epilogue_mode)
    )
    if (
        mxfp8_shared_memory_bytes(
            runtime_m,
            logical_m_per_cta,
            bn,
            bk,
            stages,
            cluster,
            match.output.element_size(),
            compact_m=compact,
            sf_tma=sf_tma,
            sf_inplace=sf_inplace,
            epilogue_mode=epilogue_mode,
        )
        > 232448
    ):
        return None
    operands = []
    for tensor, scale, axis in (
        (match.lhs, match.scale_a, match.row_axes[0]),
        (match.rhs, match.scale_b, match.row_axes[1]),
    ):
        ss = tuple(env.size_hint(x) for x in scale.stride())
        span = 1 + sum(
            (env.size_hint(size) - 1) * env.size_hint(stride)
            for size, stride in zip(scale.shape, scale.stride(), strict=True)
        )
        operands.append(
            PreparedOperand(
                PackedOperand(tensor, scale, (), (), axis),
                (),
                (
                    f"row * {ss[0] * 4} + (group // 4) * {ss[1] * 4} + group % 4"
                    if scale.dtype == torch.int32
                    else f"row * {ss[0]} + group * {ss[1]}"
                ),
                env.size_hint(tensor.numel()),
                span * (4 if scale.dtype == torch.int32 else 1),
                False,
            )
        )
    return BlockScaledMmaPlan(
        match.graph_id,
        operands[0],
        operands[1],
        match.output,
        1.0,
        match.round_dtypes,
        m,
        n,
        k,
        bn,
        bk,
        stages,
        cluster,
        persistent,
        env.config_spec.num_sm or 1,
        "mxfp8",
        match.rhs_transposed,
        fused_scales,
        logical_m_per_cta,
        swap_ab,
        compact_m,
        sf_tma,
        sf_inplace,
        accumulator_stages,
        prefetch_output_registers,
        tma_output,
    )
