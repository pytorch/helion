"""Typed ownership of the N sum emitted by the collapsed baddbmm fallback."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

from ...language import _tracing_ops
from ...language.creation_ops import full
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo

if TYPE_CHECKING:
    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST
    from ..helper_function import CodegenInterface


@dataclass(frozen=True)
class MatmulSumPath:
    producer: Node
    consumer: Node
    body: ForLoopGraphInfo
    placeholder: Node
    accumulator_copy: Node
    loop_call: Node
    initial: Node
    result_item: Node
    phi: Node
    masked: Node | None
    n_block_id: int
    n_extent: int


@dataclass(frozen=True)
class CompletedMatmulSum:
    codegen: GenerateAST
    path: MatmulSumPath
    expression: ast.AST


def _one_user(node: Node) -> Node | None:
    return next(iter(node.users)) if len(node.users) == 1 else None


def _only_value_user(node: Node, expected: Node) -> bool:
    # Shape queries do not observe the matrix's elements. The typed shape is
    # unchanged by the physical N fold, including through the loop carry.
    return {
        user
        for user in node.users
        if not (
            user.op == "call_function"
            and user.target == torch.ops.aten.sym_size.int
            and len(user.args) == 2
            and user.args[0] is node
            and type(user.args[1]) is int
            and not user.kwargs
        )
    } == {expected}


def _positive_zero(value: object) -> bool:
    if type(value) is int or type(value) is float:
        return value == 0 and math.copysign(1.0, value) == 1.0
    return False


def _fp32_shape(node: Node) -> tuple[object, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        return None
    return tuple(
        size._sympy_() if isinstance(size, torch.SymInt) else size
        for size in value.shape
    )


def match_matmul_sum_path(
    cg: GenerateAST, producer: Node, n_block_id: int, n_extent: int
) -> MatmulSumPath | None:
    """Prove one zero-initialized FP32 carry reaches only its final N sum.

    GraphInfo.node_args can belong to the pre-codegen graph. Follow the current
    loop call's operands, current body placeholders and current output instead.
    Each edge carrying the matrix is exclusive; casts, other axes, nonzero
    initial values and escaping matrix users keep the ordinary reduction path.
    """
    from ..inductor_lowering import ReductionLowering

    if (
        producer.op != "call_function"
        or producer.target != torch.ops.aten.baddbmm.default
        or len(producer.args) != 3
        or producer.kwargs
        or n_extent <= 0
    ):
        return None
    shape = _fp32_shape(producer)
    if shape is None or len(shape) != 3 or shape[-2:] != (n_extent, n_extent):
        return None
    acc, lhs, rhs = producer.args
    if not all(isinstance(value, Node) for value in (acc, lhs, rhs)):
        return None
    assert isinstance(acc, Node) and isinstance(lhs, Node) and isinstance(rhs, Node)
    if (
        _fp32_shape(acc) != shape
        or _fp32_shape(lhs) is None
        or _fp32_shape(rhs) is None
        or acc.graph is not producer.graph
        or acc.op != "call_function"
        or acc.target is not _tracing_ops._new_var
        or len(acc.args) != 1
        or acc.kwargs
        or _one_user(acc) is not producer
    ):
        return None
    placeholder = acc.args[0]
    if (
        not isinstance(placeholder, Node)
        or placeholder.op != "placeholder"
        or placeholder.graph is not producer.graph
        or not _only_value_user(placeholder, acc)
        or _fp32_shape(placeholder) != shape
    ):
        return None
    bodies = [info for info in cg.codegen_graphs if info.graph is producer.graph]
    if len(bodies) != 1 or not isinstance(bodies[0], ForLoopGraphInfo):
        return None
    body = bodies[0]
    # A single carried tensor gives an unambiguous output/argument slot. Other
    # captured inputs may be added here only with a corresponding slot proof.
    if list(body.graph.find_nodes(op="placeholder")) != [placeholder]:
        return None
    output = _one_user(producer)
    if (
        output is None
        or output.op != "output"
        or len(output.args) != 1
        or not isinstance(output.args[0], (list, tuple))
        or tuple(output.args[0]) != (producer,)
    ):
        return None
    callers = [
        node
        for info in cg.codegen_graphs
        for node in info.graph.nodes
        if node.op == "call_function"
        and _tracing_ops.is_for_loop_target(node.target)
        and len(node.args) == 4
        and node.args[0] == body.graph_id
    ]
    if len(callers) != 1:
        return None
    loop_call = callers[0]
    initial_args = loop_call.args[3]
    if (
        loop_call.graph is producer.graph
        or loop_call.kwargs
        or not isinstance(initial_args, (list, tuple))
        or len(initial_args) != 1
        or not isinstance(initial_args[0], Node)
    ):
        return None
    initial = initial_args[0]
    if (
        initial.graph is not loop_call.graph
        or initial.op != "call_function"
        or initial.target is not full
        or len(initial.args) != 4
        or initial.kwargs
        or not _positive_zero(initial.args[1])
        or initial.args[2] != torch.float32
        or _fp32_shape(initial) != shape
    ):
        return None
    result_item = _one_user(loop_call)
    if (
        result_item is None
        or result_item.graph is not loop_call.graph
        or result_item.op != "call_function"
        or result_item.target is not operator.getitem
        or result_item.args != (loop_call, 0)
        or result_item.kwargs
        or _fp32_shape(result_item) != shape
    ):
        return None
    phi = _one_user(result_item)
    if (
        phi is None
        or phi.graph is not loop_call.graph
        or phi.op != "call_function"
        or phi.target is not _tracing_ops._phi
        or phi.args != (initial, result_item)
        or phi.kwargs
        or _fp32_shape(phi) != shape
        or set(initial.users) != {loop_call, phi}
    ):
        return None
    consumer = _one_user(phi)
    masked = None
    if consumer is not None and consumer.target is _tracing_ops._mask_to:
        if (
            consumer.op != "call_function"
            or consumer.graph is not phi.graph
            or len(consumer.args) != 2
            or consumer.args[0] is not phi
            or not _positive_zero(consumer.args[1])
            or consumer.kwargs
            or _fp32_shape(consumer) != shape
        ):
            return None
        masked = consumer
        consumer = _one_user(consumer)
    if (
        consumer is None
        or consumer.graph is not phi.graph
        or consumer.op != "call_function"
        or consumer.target != torch.ops.aten.sum.dim_IntList
        or len(consumer.args) != 2
        or consumer.args[0] is not (masked if masked is not None else phi)
        or not isinstance(consumer.args[1], (list, tuple))
        or tuple(consumer.args[1]) not in ((-1,), (2,))
        or consumer.kwargs
        or _fp32_shape(consumer) != shape[:-1]
    ):
        return None
    lowering = consumer.meta.get("lowering")
    env = CompileEnvironment.current()
    if (
        not isinstance(lowering, ReductionLowering)
        or lowering.reduction_type != "sum"
        or not env.block_sizes[n_block_id].reduction
        or env.canonical_block_id(lowering.block_index)
        != env.canonical_block_id(n_block_id)
    ):
        return None
    return MatmulSumPath(
        producer,
        consumer,
        body,
        placeholder,
        acc,
        loop_call,
        initial,
        result_item,
        phi,
        masked,
        n_block_id,
        n_extent,
    )


def record_completed_matmul_sum(
    cg: CodegenInterface,
    producer: Node,
    n_block_id: int,
    n_extent: int,
    expression: ast.AST,
) -> None:
    """Called only after the ordinary full-N fallback emitter succeeds."""
    from ..generate_ast import GenerateAST

    assert isinstance(cg, GenerateAST)
    path = match_matmul_sum_path(cg, producer, n_block_id, n_extent)
    if path is not None:
        cg.device_function.cute_state.completed_matmul_sums[path.consumer] = (
            CompletedMatmulSum(cg, path, expression)
        )


def completed_matmul_sum_input(ctx: LoweringContext, node: Node) -> ast.AST | None:
    """Return only the typed carry whose N sum this codegen instance emitted."""
    from ..generate_ast import GenerateAST

    cg = ctx.cg
    if not isinstance(cg, GenerateAST):
        return None
    completed = cg.device_function.cute_state.completed_matmul_sums.get(node)
    if completed is None or completed.codegen is not cg:
        return None
    path = completed.path
    if (
        path.consumer is not node
        or path.producer.meta.get("codegen") is not completed.expression
        or match_matmul_sum_path(cg, path.producer, path.n_block_id, path.n_extent)
        != path
    ):
        return None
    value = ctx.env.get(path.masked if path.masked is not None else path.phi)
    phi_value = ctx.env.get(path.phi)
    # Keep nontrivial masking on the normal path. These names are already
    # bound to the proven phi and mask nodes, not found by textual matching.
    if (
        not isinstance(value, ast.Name)
        or not isinstance(phi_value, ast.Name)
        or value.id != phi_value.id
    ):
        return None
    return value
