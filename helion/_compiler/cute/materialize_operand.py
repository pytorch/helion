"""Materialize pure reused matrix operands into an ordered CuTe launch.

The first envelope owns a two-axis output grid, one exclusive FP32 contraction
loop, and one complete fresh output. Pointwise RHS recipes may interleave a
fixed number of physical rows. Their operations and casts are copied unchanged;
codegen preserves separate FP32 products unless fast math is enabled. The
consumer sees a fresh dense tensor and an ordinary logical K tile loop.

A temporary compiler proves the original program. Shape specialization inserted
in that proof is also inserted into the owning compiler, so sample sizes never
stand in for runtime guards. Rejected programs retain their original source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import logging
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import language as hl
from ...language import _tracing_ops
from ...language import memory_ops
from ...language.matmul_ops import dot
from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..kernel_compiler import KernelCompiler
from ..type_info import TensorType
from ..type_info import TileIndexType
from ..variable_origin import ArgumentOrigin
from .collective_matmul import _is_direct_zero_seed
from .cute_mma import _mma_loop_is_exclusive
from .full_slice_matmul import _bound_names
from .full_slice_matmul import _global_reference
from .full_slice_matmul import _global_value
from .full_slice_matmul import _pure_host_prelude
from .materialized_fission import _ANALYSIS_CALLS
from .materialized_fission import _ANALYSIS_METHODS
from .materialized_fission import MaterializedFissionPlan
from .materialized_fission import _clone_untyped
from .materialized_fission import _exact_tile
from .materialized_fission import _Ineligible
from .materialized_fission import _is_contiguous_2d
from .materialized_fission import _require
from .packed_matmul import interleaved_matrix_terms
from .packed_matmul import packed_matmul_axis
from .promote_output_axis import _access_tensor
from .promote_output_axis import _fresh_tensors
from .promote_output_axis import _indices
from .promote_output_axis import _ordinary_operation
from .signed_bitfield import _CASTS
from .signed_bitfield import _cast_input
from .signed_bitfield import _integer_field

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from torch.fx import Node

    from ...runtime.kernel import Kernel
    from ..host_function import HostFunction


log = logging.getLogger(__name__)
_PURE_CALLS = (*_ANALYSIS_CALLS, hl.register_block_size, torch.stack)


@dataclass
class _Source:
    index: int
    root: ast.For
    loop: ast.For
    accumulator: str
    m: str
    n: str
    k: str
    output: str
    lhs: ast.expr
    rhs: ast.expr
    definitions: dict[str, ast.Assign]
    call: ast.Call


def _callee(node: ast.AST, host: HostFunction) -> object:
    return _global_value(node, host, _bound_names(host))


def _reference(value: object, host: HostFunction) -> ast.expr:
    reference = _global_reference(value, host, _bound_names(host))
    _require(reference is not None, "required stable language global unavailable")
    assert reference is not None
    # Fill in ExtendedAST context nodes as well as the Name/Attribute nodes.
    return cast("ast.expr", expr_from_string(ast.unparse(reference)))


def _safe_source(host: HostFunction) -> bool:
    """Do not retrace opaque Python helpers or host-side tensor effects."""
    locals_ = _bound_names(host)
    if not _pure_host_prelude(host, locals_, allow_block_size_registration=True):
        return False

    def global_value(node: ast.AST) -> object:
        return _global_value(node, host, locals_)

    def known(value: object) -> bool:
        return (
            value is torch
            or value is hl
            or type(value) in (int, float, bool, str, type(None), torch.dtype)
            or any(value is candidate for candidate in _PURE_CALLS)
        )

    def local(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in locals_
        if isinstance(node, ast.Subscript):
            return local(node.value)
        if isinstance(node, ast.BinOp):
            return local(node.left) or local(node.right)
        if isinstance(node, ast.UnaryOp):
            return local(node.operand)
        if isinstance(node, ast.Call):
            return any(global_value(node.func) is call for call in _PURE_CALLS) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ANALYSIS_METHODS
                and local(node.func.value)
            )
        return False

    for statement in host.body:
        for node in ast.walk(statement):
            if isinstance(
                node,
                (
                    ast.AugAssign,
                    ast.AnnAssign,
                    ast.Delete,
                    ast.If,
                    ast.While,
                    ast.With,
                    ast.Try,
                    ast.FunctionDef,
                    ast.Lambda,
                    ast.ListComp,
                    ast.GeneratorExp,
                    ast.NamedExpr,
                ),
            ):
                return False
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) for target in node.targets
            ):
                return False
            if isinstance(node, ast.Call):
                if any(global_value(node.func) is call for call in _PURE_CALLS):
                    continue
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in _ANALYSIS_METHODS
                    and local(node.func.value)
                ):
                    continue
                return False
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in locals_
                and not known(global_value(node))
            ):
                return False
            if (
                isinstance(node, ast.Attribute)
                and not local(node.value)
                and not known(global_value(node))
            ):
                return False
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id in host.params.arguments
            ):
                return False
    return True


def _source(host: HostFunction) -> _Source:
    roots = [
        (index, node)
        for index, node in enumerate(host.body)
        if isinstance(node, ast.For)
    ]
    _require(len(roots) == 1, "requires one output grid")
    index, root = roots[0]
    _require(
        isinstance(root.iter, ast.Call)
        and _callee(root.iter.func, host) is hl.tile
        and len(root.iter.args) == 1
        and isinstance(root.iter.args[0], (ast.Tuple, ast.List))
        and len(root.iter.args[0].elts) == 2
        and not root.iter.keywords
        and isinstance(root.target, (ast.Tuple, ast.List))
        and len(root.target.elts) == 2
        and all(isinstance(item, ast.Name) for item in root.target.elts)
        and not root.orelse
        and len(root.body) == 3,
        "requires a canonical M,N tile and one contraction",
    )
    assert isinstance(root.target, (ast.Tuple, ast.List))
    m, n = (cast("ast.Name", item).id for item in root.target.elts)
    initial, loop, store = root.body
    _require(
        isinstance(initial, ast.Assign)
        and len(initial.targets) == 1
        and isinstance(initial.targets[0], ast.Name)
        and isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and isinstance(loop.iter, ast.Call)
        and _callee(loop.iter.func, host) is hl.tile
        and len(loop.iter.args) == 1
        and all(keyword.arg == "block_size" for keyword in loop.iter.keywords)
        and not loop.orelse
        and bool(loop.body)
        and isinstance(store, ast.Assign)
        and len(store.targets) == 1
        and isinstance(store.targets[0], ast.Subscript)
        and isinstance(store.targets[0].value, ast.Name),
        "requires a local accumulator, one K loop and one output store",
    )
    assert isinstance(initial, ast.Assign) and isinstance(initial.targets[0], ast.Name)
    assert isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
    assert isinstance(store, ast.Assign) and isinstance(store.targets[0], ast.Subscript)
    assert isinstance(store.targets[0].value, ast.Name)
    accumulator = initial.targets[0].id
    output = store.targets[0].value.id
    tail = host.body[index + 1 :]
    _require(
        len(tail) == 1
        and isinstance(tail[0], ast.Return)
        and isinstance(tail[0].value, ast.Name)
        and tail[0].value.id == output,
        "requires a single materialized result",
    )
    definitions: dict[str, ast.Assign] = {}
    host_names = (
        {
            node.id
            for statement in host.body[:index]
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        | set(host.params.arguments)
        | {m, n, loop.target.id, accumulator}
    )
    for statement in loop.body[:-1]:
        _require(
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name),
            "K loop has a nonlocal effect",
        )
        assert isinstance(statement, ast.Assign)
        target = cast("ast.Name", statement.targets[0]).id
        _require(
            target not in host_names | definitions.keys(),
            "K recipe rebinds an existing value",
        )
        definitions[target] = statement
    update = loop.body[-1]
    _require(
        isinstance(update, ast.Assign)
        and len(update.targets) == 1
        and isinstance(update.targets[0], ast.Name)
        and update.targets[0].id == accumulator
        and isinstance(update.value, ast.Call),
        "requires one final accumulator update",
    )
    assert isinstance(update, ast.Assign) and isinstance(update.value, ast.Call)
    call = update.value
    callee = _callee(call.func, host)
    if callee is hl.dot:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        _require(
            len(call.args) == 2
            and isinstance(keywords.get("acc"), ast.Name)
            and cast("ast.Name", keywords["acc"]).id == accumulator,
            "dot must carry the local accumulator",
        )
        lhs, rhs = call.args
    else:
        _require(
            callee is torch.addmm
            and len(call.args) == 3
            and not call.keywords
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == accumulator,
            "requires a unit-scale addmm or explicit-accumulator dot",
        )
        lhs, rhs = call.args[1:]
    return _Source(
        index,
        root,
        loop,
        accumulator,
        m,
        n,
        loop.target.id,
        output,
        lhs,
        rhs,
        definitions,
        call,
    )


def _resolve(value: ast.expr, source: _Source) -> ast.expr:
    seen: set[str] = set()
    while isinstance(value, ast.Name) and value.id in source.definitions:
        _require(value.id not in seen, "cyclic recipe")
        seen.add(value.id)
        value = source.definitions[value.id].value
    return value


def _needed_statements(values: Sequence[ast.expr], source: _Source) -> list[ast.Assign]:
    pending = [
        node.id
        for value in values
        for node in ast.walk(value)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    ]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        statement = source.definitions.get(name)
        if statement is not None:
            pending.extend(
                node.id
                for node in ast.walk(statement.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
    _require(
        not seen.intersection((source.m, source.accumulator)),
        "operand depends on output rows or the accumulator",
    )
    return [statement for name, statement in source.definitions.items() if name in seen]


def _static(value: int | torch.SymInt, env: CompileEnvironment) -> int | None:
    expression = env.specialize_expr(env.shape_env.replace(sympy.sympify(value)))
    return int(expression) if expression.is_Integer else None


def _exact_integer_cast(node: Node) -> bool:
    """Keep wide integral recipes on their original FP32 accumulation path.

    Materialization makes the result eligible for native half/BF16 MMA. Admit
    integral conversions only when every value in a proved range is exact in
    that format; a sample tensor or copied node annotation is not range proof.
    """
    if node.target not in _CASTS:
        return True
    source = _cast_input(node)
    value = node.meta.get("val")
    original = source.meta.get("val") if source is not None else None
    if (
        source is None
        or source.graph is not node.graph
        or not isinstance(value, torch.Tensor)
        or not isinstance(original, torch.Tensor)
    ):
        return False
    if (
        value.dtype not in (torch.float16, torch.bfloat16)
        or original.is_floating_point()
        or original.is_complex()
        or original.dtype is torch.bool
    ):
        return True
    if original.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return False
    bounds = torch.iinfo(original.dtype)
    low, high = bounds.min, bounds.max
    field = _integer_field(source)
    if field is not None:
        if any(item.graph is not node.graph for item in field.nodes):
            return False
        low, high = field.bounds
    # A p-bit significand represents every integer in [-2**p, 2**p].
    limit = 256 if value.dtype is torch.bfloat16 else 2048
    return -limit <= low <= high <= limit


def _prove_recipe(
    nodes: Sequence[Node],
    *,
    k: int,
    n: int,
    inputs: set[torch.Tensor],
    env: CompileEnvironment,
) -> set[Node]:
    shape = (env.block_sizes[k].var, env.block_sizes[n].var)
    visited: set[Node] = set()
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            _require(
                tuple(map(sympy.sympify, value.shape)) == shape,
                "recipe changes coordinates or tile shape",
            )
        if node.target is memory_ops.load:
            tensor = _access_tensor(node)
            indices = node.args[1]
            _require(
                tensor in inputs
                and isinstance(indices, (tuple, list))
                and len(indices) == 2
                and _exact_tile(indices[0], k, env)
                and _exact_tile(indices[1], n, env)
                and len(node.args) == 4
                and node.args[2] is None
                and node.args[3] is None,
                "recipe load is not a canonical read-only K,N tile",
            )
            continue
        target = node.target
        _require(
            node.op == "call_function"
            and isinstance(target, torch._ops.OpOverload)
            and torch.Tag.pointwise in target.tags
            and _ordinary_operation(node),
            "recipe contains a reduction, gather or opaque operation",
        )
        _require(
            _exact_integer_cast(node),
            "integer cast lacks an exactly representable range",
        )
        pending.extend(node.all_input_nodes)
    return visited


def _strip_zero_padding(node: Node) -> tuple[Node, set[Node]]:
    """A dense materialization reload supplies the same out-of-domain zeros."""
    masks: set[Node] = set()
    while node.target is _tracing_ops._mask_to:
        _require(
            len(node.args) == 2
            and isinstance(node.args[1], (int, float))
            and node.args[1] == 0
            and not node.kwargs
            and isinstance(node.args[0], torch.fx.Node),
            "unrecognized contraction padding",
        )
        masks.add(node)
        assert isinstance(node.args[0], torch.fx.Node)
        node = node.args[0]
    return node, masks


def _prove(
    host: HostFunction, source: _Source, env: CompileEnvironment
) -> tuple[int, torch.dtype, str, tuple[ast.expr, ...], tuple[str, ...]]:
    ir = host.device_ir
    _require(len(ir.root_ids) == 1 and len(ir.graphs) == 2, "unexpected nested graphs")
    root_axes = _indices(source.root)
    k_axes = _indices(source.loop)
    _require(
        root_axes is not None
        and len(root_axes) == 2
        and all(isinstance(axis, TileIndexType) for axis in root_axes)
        and k_axes is not None
        and len(k_axes) == 1
        and isinstance(k_axes[0], TileIndexType)
        and not ir.noncanonical_task_origin_block_ids,
        "requires canonical tile axes",
    )
    assert root_axes is not None and k_axes is not None
    m_id, n_id = (axis.block_id for axis in root_axes)
    k_id = k_axes[0].block_id
    contractions = [
        node
        for graph in ir.graphs
        for node in graph.graph.nodes
        if node.target in (dot, torch.ops.aten.addmm.default)
    ]
    _require(len(contractions) == 1, "requires one matrix contraction")
    node = contractions[0]
    if node.target is dot:
        lhs, rhs, acc = node.args[:3]
    else:
        acc, lhs, rhs = node.args[:3]
    _require(
        all(isinstance(value, torch.fx.Node) for value in (lhs, rhs, acc)),
        "unrecognized matrix operands",
    )
    assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
    assert isinstance(acc, torch.fx.Node)
    _require(
        _is_direct_zero_seed(ir.graphs, node, acc)
        and _mma_loop_is_exclusive(node)
        and acc.meta["val"].dtype == torch.float32,
        "requires one exclusive FP32 accumulator with a local zero seed",
    )
    loop_graph = next(graph for graph in ir.graphs if graph.graph is node.graph)
    _require(
        isinstance(loop_graph, ForLoopGraphInfo) and loop_graph.block_ids == [k_id],
        "contraction is not owned by the canonical K loop",
    )
    a, b = lhs.meta["val"], rhs.meta["val"]
    _require(
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.ndim == b.ndim == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and lhs.target is memory_ops.load,
        "requires direct half/BF16 A and a matching RHS recipe",
    )
    left = _resolve(source.lhs, source)
    _require(
        isinstance(left, ast.Subscript)
        and isinstance(left.value, ast.Name)
        and left.value.id in host.params.arguments,
        "A must load directly from an unchanged input binding",
    )
    assert isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name)
    a_name = left.value.id
    a_tensor = _access_tensor(lhs)
    _require(
        a_tensor is host.params.arguments[a_name] and _is_contiguous_2d(a_tensor, env),
        "A has an unsupported alias or layout",
    )
    assert a_tensor is not None
    recipe, padding = _strip_zero_padding(rhs)
    packed = packed_matmul_axis(env, lhs, recipe)
    factor = packed.factor if packed is not None else 1
    if packed is not None:
        _require(packed.block_id == k_id, "packed K does not own the contraction")
        terms = interleaved_matrix_terms(recipe, lhs=False)
        _require(terms is not None, "unrecognized interleaved RHS")
        value = _resolve(source.rhs, source)
        _require(
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in ("reshape", "view"),
            "interleaved source is not a reshape",
        )
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
        stack = _resolve(value.func.value, source)
        _require(
            isinstance(stack, ast.Call)
            and _callee(stack.func, host) is torch.stack
            and len(stack.args) >= 1
            and isinstance(stack.args[0], (ast.Tuple, ast.List))
            and len(stack.args[0].elts) == factor,
            "source and FX interleave order disagree",
        )
        assert isinstance(stack, ast.Call) and isinstance(
            stack.args[0], (ast.Tuple, ast.List)
        )
        expressions = tuple(stack.args[0].elts)
    else:
        indices = lhs.args[1]
        _require(
            isinstance(indices, (list, tuple))
            and len(indices) == 2
            and _exact_tile(indices[1], k_id, env),
            "direct A has a noncanonical K mapping",
        )
        terms = (recipe,)
        expressions = (source.rhs,)
    indices = lhs.args[1]
    _require(
        isinstance(indices, (list, tuple))
        and len(indices) == 2
        and _exact_tile(indices[0], m_id, env),
        "A changes output-row ownership",
    )
    extents = [
        _static(cast("int | torch.SymInt", env.block_sizes[axis].size), env)
        for axis in (m_id, n_id, k_id)
    ]
    _require(
        all(extent is not None and extent > 0 for extent in extents)
        and _static(a_tensor.shape[0], env) == extents[0]
        and _static(a_tensor.shape[1], env) == cast("int", extents[2]) * factor,
        "guarded matrix extents do not cover the complete logical K domain",
    )
    inputs = {
        value
        for value in host.params.arguments.values()
        if isinstance(value, torch.Tensor)
    }
    assert terms is not None
    recipe_nodes = _prove_recipe(terms, k=k_id, n=n_id, inputs=inputs, env=env)
    _require(
        factor > 1 or any(item.target is not memory_ops.load for item in recipe_nodes),
        "direct RHS is already materialized",
    )
    _needed_statements(expressions, source)
    recipe_nodes.update(padding)
    stores = [
        item
        for graph in ir.graphs
        for item in graph.graph.nodes
        if item.target is memory_ops.store
    ]
    _require(len(stores) == 1, "requires one complete output store")
    store = stores[0]
    output = _access_tensor(store)
    fresh = _fresh_tensors(host)
    _require(
        output is not None
        and output in fresh
        and _is_contiguous_2d(output, env)
        and store.graph is not node.graph
        and len(store.args) == 4
        and store.args[3] is None,
        "output is not one unconditional fresh row-major allocation",
    )
    assert output is not None
    store_indices = store.args[1]
    _require(
        isinstance(store_indices, (tuple, list))
        and len(store_indices) == 2
        and _exact_tile(store_indices[0], m_id, env)
        and _exact_tile(store_indices[1], n_id, env)
        and _static(output.shape[0], env) == extents[0]
        and _static(output.shape[1], env) == extents[1],
        "output grid does not overwrite every element exactly once",
    )
    for graph in ir.graphs:
        for item in graph.graph.nodes:
            if item.target is memory_ops.load:
                _require(item is lhs or item in recipe_nodes, "unexpected memory read")
            _require(
                item is node or item in recipe_nodes or _ordinary_operation(item),
                "unexpected side effect or control flow",
            )
    captures: list[str] = []
    for statement in host.body:
        if isinstance(statement, ast.For):
            break
        if isinstance(statement, ast.Assign) and isinstance(
            value_type := cast("ExtendedAST", statement.value)._type_info, TensorType
        ):
            _require(
                value_type.proxy() in fresh
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name),
                "unrecognized shared allocation",
            )
            captures.append(cast("ast.Name", statement.targets[0]).id)
    _require(source.output in captures, "output is not captured by its original name")
    return factor, b.dtype, a_name, expressions, tuple(captures)


def _replacement(
    host: HostFunction,
    source: _Source,
    original_body: list[ast.stmt],
    specializations: list[ast.stmt],
    proof: tuple[int, torch.dtype, str, tuple[ast.expr, ...], tuple[str, ...]],
) -> MaterializedFissionPlan:
    factor, dtype, a_name, expressions, captures = proof
    used = _bound_names(host) | set(host.fn.__globals__)

    def fresh(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    matrix = fresh("_helion_materialized_operand")
    row = fresh("_helion_materialize_k")
    column = fresh("_helion_materialize_n")
    index = fresh("_helion_materialize_index")
    logical_k = fresh("_helion_materialize_logical_k")
    assert isinstance(source.root, ExtendedAST)
    with source.root:
        tile = _reference(hl.tile, host)
        empty = _reference(torch.empty, host)
        arange = _reference(hl.arange, host)
        element_type = _reference(dtype, host)
        assert isinstance(source.loop.iter, ast.Call)
        assert isinstance(source.root.iter, ast.Call)
        assert isinstance(source.root.iter.args[0], (ast.Tuple, ast.List))
        n_bound = source.root.iter.args[0].elts[1]
        allocation = statement_from_string(
            f"{matrix} = {{empty}}(({a_name}.size(1), {{n}}), "
            f"dtype={{dtype}}, device={a_name}.device)",
            empty=empty,
            n=n_bound,
            dtype=element_type,
        )
        producer = statement_from_string(
            f"for {row}, {column} in {{tile}}([{{k}}, {{n}}]):\n    pass",
            tile=tile,
            k=source.loop.iter.args[0],
            n=n_bound,
        )
        assert isinstance(producer, ast.For) and isinstance(producer.iter, ast.Call)
        if source.loop.iter.keywords:
            block = source.loop.iter.keywords[0].value
            block_call = expr_from_string("f(block_size=[{block}, None])", block=block)
            assert isinstance(block_call, ast.Call)
            producer.iter.keywords = block_call.keywords
        body: list[ast.stmt] = list(_needed_statements(expressions, source))
        if factor > 1:
            body.append(
                statement_from_string(
                    f"{index} = {{arange}}({source.k}.begin, "
                    f"{source.k}.begin + {source.k}.block_size)",
                    arange=arange,
                )
            )
        for position, expression in enumerate(expressions):
            row_expression = (
                f"{index} * {factor} + {position}" if factor > 1 else source.k
            )
            body.append(
                statement_from_string(
                    f"{matrix}[{row_expression}, {source.n}] = {{value}}",
                    value=expression,
                )
            )
        producer.body = [cast("ast.stmt", _clone_untyped(stmt)) for stmt in body]
        for node in ast.walk(producer):
            if isinstance(node, ast.Name):
                if node.id == source.k:
                    node.id = row
                elif node.id == source.n:
                    node.id = column
        consumer = cast("ast.For", _clone_untyped(source.root))
        original_update = source.loop.body[-1]
        assert isinstance(original_update, ast.Assign)
        update = cast("ast.Assign", _clone_untyped(original_update))
        assert isinstance(update.value, ast.Call)
        load_a = cast(
            "ast.expr", expr_from_string(f"{a_name}[{source.m}, {logical_k}]")
        )
        load_b = cast(
            "ast.expr", expr_from_string(f"{matrix}[{logical_k}, {source.n}]")
        )
        if _callee(source.call.func, host) is hl.dot:
            update.value.args = [load_a, load_b]
        else:
            update.value.args[1:] = [load_a, load_b]
        contraction = statement_from_string(
            f"for {logical_k} in {{tile}}({a_name}.size(1)):\n    pass", tile=tile
        )
        assert isinstance(contraction, ast.For)
        contraction.body = [update]
        consumer.body[1] = contraction
        prefix = [
            cast("ast.stmt", _clone_untyped(stmt))
            for stmt in original_body[: source.index]
        ]
        for statement in prefix:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == source.output
                and isinstance(statement.value, ast.Call)
            ):
                factory = _callee(statement.value.func, host)
                replacements: dict[object, object] = {
                    torch.zeros: torch.empty,
                    torch.ones: torch.empty,
                    torch.zeros_like: torch.empty_like,
                    torch.ones_like: torch.empty_like,
                }
                replacement = replacements.get(factory)
                if replacement is not None:
                    reference = _reference(replacement, host)
                    statement.value.func = cast("ast.expr", reference)
        replacement_body = (
            *specializations,
            *prefix,
            allocation,
            producer,
            consumer,
            *original_body[source.index + 1 :],
        )
    return MaterializedFissionPlan(
        root_index=len(specializations) + source.index + 1,
        source_root_key=ast.dump(original_body[source.index]),
        materialized_names=(*captures, matrix),
        region_count=2,
        source_root_index=source.index,
        replacement_body=tuple(
            cast("ast.stmt", _clone_untyped(statement))
            for statement in replacement_body
        ),
        pointwise_region_indices=(0,),
        operand_interleave_factor=factor,
    )


def plan_operand_materialization(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    owning_env: CompileEnvironment,
) -> MaterializedFissionPlan | None:
    if (
        not owning_env.settings.cute_materialize_transformed_operands
        or owning_env.backend_name != "cute"
        or owning_env._is_distributed
        or any(type(value) not in (torch.Tensor, torch.nn.Parameter) for value in args)
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in kernel.signature.parameters.values()
        )
    ):
        return None
    env = CompileEnvironment(
        owning_env.device,
        owning_env.settings,
        index_dtype=owning_env.index_dtype,
        is_distributed=False,
    )
    with env:
        fake_args = [
            env.to_fake(value, ArgumentOrigin(name))
            for name, value in zip(kernel.signature.parameters, args, strict=True)
        ]
        compiler = KernelCompiler(env)
        host = compiler.parse(kernel.fn, fake_args, {})
        with host, compiler._compilation_context():
            if not _safe_source(host):
                return None
            try:
                _source(host)
                compiler.unroll(host)
                compiler.customize_ast(host)
                source = _source(host)
                _needed_statements((source.rhs,), source)
                _require(
                    not host.args.posonlyargs
                    and not host.args.kwonlyargs
                    and host.args.vararg is None
                    and host.args.kwarg is None
                    and not host.constexpr_args,
                    "requires ordinary positional tensor inputs",
                )
                original_body = [
                    cast("ast.stmt", _clone_untyped(statement))
                    for statement in host.body
                ]
                specializations: list[ast.stmt] = []
                assert isinstance(source.root, ExtendedAST)
                with source.root:
                    specialize = _reference(hl.specialize, host)
                    used = _bound_names(host)
                    for name in host.params.arguments:
                        temporary = f"_helion_materialize_shape_{name}"
                        while temporary in used:
                            temporary += "_"
                        used.add(temporary)
                        specializations.append(
                            statement_from_string(
                                f"{temporary} = {{specialize}}({name}.shape)",
                                specialize=specialize,
                            )
                        )
                host.body[:0] = specializations
                compiler.propagate_types(host)
                compiler.finalize_config()
                compiler.lower(host)
                proof = _prove(host, source, env)
                return _replacement(host, source, original_body, specializations, proof)
            except _Ineligible as error:
                log.debug("CuTe operand materialization rejected: %s", error)
                return None
