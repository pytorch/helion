"""Typed, row-rematerializable values for ordinary FP32 additive scans."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.fx.node import map_arg

from ...language import memory_ops
from ...language.scan_ops import _associative_scan
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from .indexing import CuteSortableLoad

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


_POINTWISE = frozenset(
    {
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.clamp_max.default,
    }
)


def _typed_pointwise(node: Node) -> bool:
    value = node.meta.get("val")
    return (
        node.op == "call_function"
        and node.target in _POINTWISE
        and isinstance(value, torch.Tensor)
        and value.dtype is torch.float32
    )


def feeds_computed_scan(load: Node) -> bool:
    """Discover actual load leaves, without relaxing sort/topk discovery."""
    seen: set[Node] = set()
    pending = list(load.users)
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.target is _associative_scan:
            return True
        if _typed_pointwise(node):
            pending.extend(node.users)
    return False


@dataclasses.dataclass(frozen=True)
class ScanValuePlan:
    result: Node
    nodes: tuple[Node, ...]
    inputs: dict[Node, tuple[Node, ...]]
    leaves: tuple[Node, ...]


def plan_scan_value(result: object) -> ScanValuePlan | None:
    """Admit a pure typed DAG, never represent an expression as a raw load."""
    if not isinstance(result, Node):
        return None
    nodes: list[Node] = []
    inputs: dict[Node, tuple[Node, ...]] = {}
    leaves: list[Node] = []
    seen: set[Node] = set()
    computed = False

    def visit(node: Node) -> bool:
        nonlocal computed
        if node in seen:
            return True
        seen.add(node)
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return False
        if node.target is memory_ops.load:
            if (
                value.dtype not in (torch.bfloat16, torch.float32)
                or not isinstance(node.meta.get("cute_sortable_load"), CuteSortableLoad)
                or any(arg is not None for arg in node.args[2:])
                or node.kwargs
            ):
                return False
            leaves.append(node)
            nodes.append(node)
            return True
        if not _typed_pointwise(node):
            return False
        lowering = node.meta.get("lowering")
        if not isinstance(lowering, PointwiseLowering):
            return False
        if node.target is torch.ops.prims.convert_element_type.default:
            if len(node.args) != 2 or node.args[1] is not torch.float32:
                return False
        else:
            computed = True
        children: list[Node] = []

        def add(child: Node) -> Node:
            children.append(child)
            return child

        map_arg((node.args, {**node.kwargs, "_extra_deps": None}), add)
        if len(children) != len(lowering.input_names) or not all(map(visit, children)):
            return False
        # Arithmetic consumes FP32 values; only the explicit widening cast may
        # read BF16. Never erase a narrow/re-widen boundary or implicit cast.
        if node.target is not torch.ops.prims.convert_element_type.default and any(
            child.meta["val"].dtype is not torch.float32 for child in children
        ):
            return False
        inputs[node] = tuple(children)
        nodes.append(node)
        return True

    if not visit(result) or not computed or not leaves:
        return None
    return ScanValuePlan(result, tuple(nodes), inputs, tuple(leaves))


def scan_loop_ranges(
    state: CodegenState,
) -> dict[int, tuple[int | torch.SymInt, int | torch.SymInt]] | None:
    """Read the actual owning tile range, not source sizes or trip-count hints."""
    from ...language._decorators import is_api_func
    from ...language.loops import tile
    from ..ast_extension import ExtendedAST
    from ..generate_ast import _flatten_starred_args
    from ..host_function import HostFunction

    grid = state.codegen.current_grid_state
    host = HostFunction.current()
    if grid is None or len(host.device_ir.root_ids) != 1:
        return None
    roots = [
        node
        for statement in host.body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node, ExtendedAST)
        and node._root_id == 0
    ]
    if len(roots) != 1 or not isinstance(call := roots[0].iter, ast.Call):
        return None

    def proxy(node: ast.AST) -> object:
        assert isinstance(node, ExtendedAST) and node._type_info is not None
        return node._type_info.proxy()

    # Same typed signature/proxies used to emit this root in GenerateAST.
    # Do not infer begin/end from LoopDimInfo: for explicit root starts its
    # size hints can represent a trip count, not the global end coordinate.
    fn = proxy(call.func)
    if fn is not tile or not is_api_func(fn):
        return None
    if any(keyword.arg is None for keyword in call.keywords):
        return None
    kwargs = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        kwargs[keyword.arg] = proxy(keyword.value)
    args = [proxy(node) for node in _flatten_starred_args(call.args)]
    bound = fn._signature.bind(*args, **kwargs)
    bound.apply_defaults()
    start, end = bound.arguments["begin_or_end"], bound.arguments["end_or_none"]
    if end is None:
        end = start
        start = [0] * len(grid.block_ids)
    starts = list(start) if isinstance(start, (list, tuple)) else [start]
    ends = list(end) if isinstance(end, (list, tuple)) else [end]
    if len(starts) != len(grid.block_ids) or len(ends) != len(grid.block_ids):
        return None
    result = {}
    for block, begin, stop in zip(grid.block_ids, starts, ends, strict=True):
        if not isinstance(begin, (int, torch.SymInt)) or not isinstance(
            stop, (int, torch.SymInt)
        ):
            return None
        result[block] = (begin, stop)
    return result


def original_load_is_in_bounds(
    state: CodegenState,
    leaf: Node,
    ranges: dict[int, tuple[int | torch.SymInt, int | torch.SymInt]],
) -> bool:
    """Do not depend on DCE removing the original, now-unused scalar loads.

    Existing scalar memory lowering may mask by the loop rather than a shorter
    source's extent. The new route therefore requires source coverage of each
    directly indexed owner axis (or a proven in-bounds constant index).
    """
    from ..compile_environment import _to_sympy

    env = CompileEnvironment.current()
    grid = state.codegen.current_grid_state
    assert grid is not None
    load = leaf.meta["cute_sortable_load"]
    assert isinstance(load, CuteSortableLoad)
    source = leaf.args[0]
    assert isinstance(source, Node)
    tensor = source.meta["val"]
    owners = {
        name: bounds
        for block, bounds in ranges.items()
        for name in (grid.strategy.index_var(block), grid.strategy.offset_var(block))
    }

    def known(expr: sympy.Basic) -> bool:
        return env.shape_env._maybe_evaluate_static(expr) is sympy.true

    for expression, size in zip(load.index_exprs, tensor.shape, strict=True):
        index = ast.parse(expression, mode="eval").body
        stop = _to_sympy(size)
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            if not known(
                sympy.And(sympy.Ge(index.value, 0), sympy.Lt(index.value, stop))
            ):
                return False
        elif isinstance(index, ast.Name) and index.id in owners:
            lower, upper = owners[index.id]
            if not known(
                sympy.And(
                    sympy.Ge(_to_sympy(lower), 0), sympy.Le(_to_sympy(upper), stop)
                )
            ):
                return False
        else:
            return False
    return True


def emit_scan_value(
    state: CodegenState,
    plan: ScanValuePlan,
    positions: dict[Node, int | None],
    scan_row: str,
    value_name: str,
    domain: str,
) -> list[str]:
    """Rebuild each typed operation at a requested row, then mask its result."""
    from ...language.memory_ops import _cute_tensor_dim_size_expr
    from ..aten_lowering import LoweringContext

    env = CompileEnvironment.current()
    values: dict[Node, ast.AST] = {}
    lines: list[str] = []
    for node in plan.nodes:
        if node in positions:
            load = node.meta["cute_sortable_load"]
            assert isinstance(load, CuteSortableLoad)
            source = node.args[0]
            assert isinstance(source, Node)
            tensor = source.meta["val"]
            indices = list(load.index_exprs)
            pos = positions[node]
            if pos is not None:
                indices[pos] = scan_row
            bounds = [
                f"0 <= ({index}) < cutlass.Int32({_cute_tensor_dim_size_expr(state, tensor, dim)})"
                for dim, index in enumerate(indices)
            ]
            # The plan permits only ordinary indexed loads with automatic
            # bounds. Rebuild every bound at this row, including uniform axes.
            mask = " and ".join(bounds) or "True"
            name = state.device_function.new_var("scan_expr_load")
            dtype = env.backend.dtype_str(load.dtype)
            lines.append(
                f"{name} = {load.tensor_name}[{', '.join(indices)}] if {mask} else {dtype}(0)"
            )
            values[node] = expr_from_string(name)
            continue
        lowering = node.meta["lowering"]
        assert isinstance(lowering, PointwiseLowering)
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = state.codegen
        ctx.env = {}
        statements: list[ast.AST] = []
        with state.codegen.set_statements(statements), V.set_current_node(node):
            result = lowering.codegen_from_input_asts(
                ctx, node, [values[child] for child in plan.inputs[node]]
            )
        assert isinstance(result, ast.AST)
        lines.extend(ast.unparse(statement) for statement in statements)
        name = state.device_function.new_var("scan_expr_value")
        lines.append(f"{name} = {ast.unparse(result)}")
        values[node] = expr_from_string(name)
    lines.append(
        f"{value_name} = cutlass.Float32({ast.unparse(values[plan.result])}) if {domain} else cutlass.Float32(0)"
    )
    return ["    " + line.replace("\n", "\n    ") for line in lines]
