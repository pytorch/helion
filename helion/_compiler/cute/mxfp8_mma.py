"""Prove complete E4M3/E8M0 contractions for native Blackwell block scaling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language.matmul_ops import dot_scaled
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..host_function import HostFunction
from ..indexing_strategy import subscript_tile_info
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


def _scale(node: object, row: object) -> torch.Tensor | None:
    host = _host_tensor(node)
    if host is not None:
        return host
    loaded = _tile_load(node)
    if loaded is None or loaded[1] != [row, slice(None)]:
        return None
    return loaded[0]


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
    sa, sb = _scale(dot.args[1], lhs[1][0]), _scale(dot.args[4], rhs_row)
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
        or sa.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or sb.dtype not in (torch.uint8, torch.float8_e8m0fnu)
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
        if tuple(resolve(x) for x in tensor.shape) != (rows, k // 32):
            return None
        if any(value is None or value < 0 for value in map(resolve, tensor.stride())):
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


def expose_mxfp8(device_ir: DeviceIR) -> bool:
    match = _match_mxfp8(device_ir.graphs, device_ir)
    if match is None:
        return False
    spec = CompileEnvironment.current().config_spec
    spec.cute_scaled_mma_available = True
    spec.cute_mxfp8_mma_available = True
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
    refs = (match.lhs, match.rhs, match.scale_a, match.scale_b, match.output)
    fresh = _fresh_tensors(host)
    if any(env.tensor_input_source(t) is None and t not in fresh for t in refs):
        return None
    names = [df.tensor_arg(t).name for t in refs]
    if any(
        frozenset((name, names[-1])) not in df.proven_disjoint_tensor_pairs()
        for name in names[:-1]
    ):
        return None
    m, n, k = match.m, match.n, match.k
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
        bn not in (64, 128, 256)
        or bk not in (128, 256)
        or stages not in (2, 3, 4, 5)
        or cluster not in (1, 2)
        or not isinstance(persistent, bool)
    ):
        return None
    stage_bytes = (128 + bn // cluster) * bk + (128 + max(bn, 128)) * bk // 32
    if stage_bytes * stages + 128 * 64 * match.output.element_size() + 8192 > 232448:
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
                f"row * {ss[0]} + group * {ss[1]}",
                env.size_hint(tensor.numel()),
                span,
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
    )
