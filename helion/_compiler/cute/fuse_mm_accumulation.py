"""Expose a canonical FP32 tiled matmul accumulator to CuTe MMA lowering.

``acc = acc + mm(lhs, rhs)`` and ``addmm(acc, lhs, rhs)`` have the same
FP32 input/output contract, as do their rank-three batched forms. The explicit
accumulator lets the existing backend
choose its normal tensor-core or scalar lowering. Input conversions stay in
the graph, so only the existing operand proof can recover a narrower source
dtype; an arbitrary FP32 matrix is never narrowed by this pass.

This runs before lowering/masking metadata is prepared. It accepts one carried
matrix in one contraction loop, together with independent captured arguments,
and leaves other graph structures untouched.
As with an ordinary fused GEMM, FP32 accumulation order can change.
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language.matmul_ops import MATMUL_DIM_BLOCK_IDS_META
from ...language.matmul_ops import MATMUL_FACT_ID_META
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_free_symbols
from ..device_ir import ForLoopGraphInfo
from ..indexing_strategy import exact_tile_block_ids
from .cute_mma import _direct_load_tensor
from .cute_mma import _is_unmasked_load
from .cute_mma import _unwrap_lossless_mma_upcast
from .fold_noop_stores import _is_read_only

if TYPE_CHECKING:
    from ..device_ir import GraphInfo


def _fp32_matrix(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return (
        value
        if isinstance(value, torch.Tensor)
        and value.ndim in (2, 3)
        and value.dtype is torch.float32
        else None
    )


def _carry_input(node: torch.fx.Node) -> torch.fx.Node | None:
    if (
        node.op == "call_function"
        and node.target is _tracing_ops._new_var
        and len(node.args) == 1
        and not node.kwargs
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    return node if node.op == "placeholder" else None


def _specialize_input_extents(operands: tuple[torch.fx.Node, torch.fx.Node]) -> None:
    """Key dynamic direct inputs before using concrete MMA problem extents.

    This is the same backed-symbol specialization used by ``hl.specialize``.
    Both independent K sizes are keyed, including unequal bindings that retain
    scalar lowering. M/N are included because collective codegen bakes them.
    """
    env = CompileEnvironment.current()
    if env.settings.static_shapes:
        return
    inputs: list[torch.Tensor] = []
    for operand in operands:
        info = _direct_load_tensor(_unwrap_lossless_mma_upcast(operand))
        if info is None:
            return
        load, _, source = info
        subscripts = load.args[1]
        if (
            source.ndim != 2
            or not _is_unmasked_load(load)
            or not isinstance(subscripts, (tuple, list))
            or exact_tile_block_ids(env, subscripts) is None
            or env.tensor_input_source(source) is None
        ):
            return
        inputs.append(source)
    if inputs[0].dtype != inputs[1].dtype:
        return
    dynamic_sizes = [
        size
        for source in inputs
        for size in source.shape
        if isinstance(size, torch.SymInt)
    ]
    symbols = {
        symbol for size in dynamic_sizes for symbol in _symint_free_symbols(size)
    }
    source_symbols = {
        symbol for symbol, sources in env.shape_env.var_to_sources.items() if sources
    }
    if not symbols.issubset(source_symbols):
        return
    env.specialized_vars.update(symbols)
    for size in dynamic_sizes:
        # Materializing now also makes the keyed equality available to the
        # ordinary operand proof; a coincidentally equal hint alone is not used.
        int(size)


def fuse_mm_accumulation(info: GraphInfo) -> int:
    """Fuse one proved matrix carry, preserving all source operand nodes."""
    env = CompileEnvironment.current()
    if (
        env.backend_name != "cute"
        or not isinstance(info, ForLoopGraphInfo)
        or len(info.block_ids) != 1
    ):
        return 0
    graph = info.graph
    placeholders = list(graph.find_nodes(op="placeholder"))
    output_nodes = list(graph.find_nodes(op="output"))
    if len(placeholders) != len(info.node_args) or len(output_nodes) != 1:
        return 0
    outputs = output_nodes[0].args[0]
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
        return 0
    addition = outputs[0]
    if (
        not isinstance(addition, torch.fx.Node)
        or addition.op != "call_function"
        or addition.target is not torch.ops.aten.add.Tensor
        or len(addition.args) != 2
        or set(addition.kwargs) - {"alpha"}
        or addition.kwargs.get("alpha", 1) != 1
        or _fp32_matrix(addition) is None
    ):
        return 0
    for product, accumulator in (addition.args, addition.args[::-1]):
        if (
            not isinstance(product, torch.fx.Node)
            or product.op != "call_function"
            or product.target
            not in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default)
            or len(product.args) != 2
            or product.kwargs
            or tuple(product.users) != (addition,)
            or not isinstance(accumulator, torch.fx.Node)
            or _carry_input(accumulator) not in placeholders
        ):
            continue
        acc_value = _fp32_matrix(accumulator)
        product_value = _fp32_matrix(product)
        lhs_value, rhs_value = (_fp32_matrix(arg) for arg in product.args)
        if (
            acc_value is None
            or product_value is None
            or lhs_value is None
            or rhs_value is None
            or len({acc_value.ndim, product_value.ndim, lhs_value.ndim, rhs_value.ndim})
            != 1
            or not all(
                starmap(
                    env.known_equal,
                    zip(acc_value.shape, product_value.shape, strict=True),
                )
            )
        ):
            continue
        k_block_id = env.canonical_block_id(info.block_ids[0])
        if (
            env.resolve_block_id(lhs_value.shape[-1]) != k_block_id
            or env.resolve_block_id(rhs_value.shape[-2]) != k_block_id
            or k_block_id
            in {env.resolve_block_id(size) for size in product_value.shape}
        ):
            continue
        nodes = list(graph.nodes)
        if not all(
            _is_read_only(node)
            for node in nodes[nodes.index(product) + 1 : nodes.index(addition)]
        ):
            continue
        lhs, rhs = product.args
        assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
        _specialize_input_extents((lhs, rhs))
        addition.target = (
            torch.ops.aten.addmm.default
            if product.target is torch.ops.aten.mm.default
            else torch.ops.aten.baddbmm.default
        )
        addition.args = (accumulator, *product.args)
        addition.kwargs = {}
        for key in (MATMUL_FACT_ID_META, MATMUL_DIM_BLOCK_IDS_META):
            if key in product.meta:
                addition.meta[key] = product.meta[key]
        graph.erase_node(product)
        graph.lint()
        return 1
    return 0
