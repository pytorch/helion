"""Factor affine terms out of floating-point sum reductions under fast math.

For a reduction over ``reduction_dims``, this pass rewrites::

    sum((base + scalar[..., None] * direction[None, ...]) * weight[None, ...])

to::

    sum(base * weight[None, ...])
      + scalar * sum(direction[None, ...] * weight[None, ...])

when ``scalar`` varies only over the retained dimensions and ``direction`` and
``weight`` vary only over the reduced dimensions.  The latter reduction is
therefore invariant across the retained tile lanes and can be hoisted by the
CuTe lane-invariant reduction pass.

The identity distributes and reassociates floating-point arithmetic.  Callers
must only run this pass when the user opted into ``Settings.fast_math``.
Matching is deliberately narrow and fails closed: only explicit broadcast
views, ``aten.sum.dim_IntList`` reductions, and fp32 tensors are accepted.  It
runs before lowering metadata is prepared; the ordinary reduction lowering
therefore adds the standard zero masks to both replacement sums.

On Blackwell, the late AST phase can also pair independent lanes of eligible
even-width fp32 reduction and list-building update loops into packed f32x2
instructions.  That transform is separately guarded by fast math and target
capability, and rejects unknown effects, ambiguous fragment aliases, and
cross-lane or escaping dependencies.
"""

from __future__ import annotations

import ast
from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import view_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .licm_profitability import repeated_work_is_profitable

if TYPE_CHECKING:
    from collections.abc import Iterable

_ADD = torch.ops.aten.add.Tensor
_MUL = torch.ops.aten.mul.Tensor
_SUM = torch.ops.aten.sum.dim_IntList


def _call_node(node: object, target: object) -> torch.fx.Node | None:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
    ):
        return node
    return None


def _binary_args(
    node: torch.fx.Node, target: object
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    if _call_node(node, target) is None or len(node.args) != 2 or node.kwargs:
        return None
    lhs, rhs = node.args
    if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
        return None
    return lhs, rhs


def _is_full_slice(value: object) -> bool:
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


def _shape(node: torch.fx.Node) -> tuple[object, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(value.shape)


def _same_size(left: object, right: object) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, torch.SymInt) and isinstance(right, torch.SymInt):
        return left.node is right.node
    return False


def _is_fp32_tensor(node: torch.fx.Node) -> bool:
    value = node.meta.get("val")
    return isinstance(value, torch.Tensor) and value.dtype is torch.float32


def _normalized_dims(node: torch.fx.Node) -> tuple[int, ...] | None:
    if len(node.args) < 2 or len(node.args) > 3:
        return None
    if set(node.kwargs) - {"keepdim", "dtype"}:
        return None
    if len(node.args) == 3 and "keepdim" in node.kwargs:
        return None
    if node.kwargs.get("dtype") is not None:
        return None
    source = node.args[0]
    dims = node.args[1]
    keepdim = node.args[2] if len(node.args) == 3 else node.kwargs.get("keepdim", False)
    if keepdim is not False or not isinstance(source, torch.fx.Node):
        return None
    shape = _shape(source)
    if shape is None or not shape or not isinstance(dims, (list, tuple)):
        return None
    if not dims or not all(isinstance(dim, int) for dim in dims):
        return None
    normalized = tuple(sorted({cast("int", dim) % len(shape) for dim in dims}))
    if len(normalized) != len(dims):
        return None
    return normalized


def _broadcast_source(
    node: torch.fx.Node,
    *,
    rank: int,
    reduced_dims: frozenset[int],
    source_spans_reduced_dims: bool,
) -> torch.fx.Node | None:
    """Unwrap a pure ``None``/full-slice broadcast with an exact axis role.

    ``source_spans_reduced_dims=False`` recognizes ``scalar[..., None]``;
    ``True`` recognizes ``vector[None, ...]``.  Requiring an explicit view
    avoids guessing whether a size-one symbolic dimension is semantically
    invariant.
    """
    view = _call_node(node, view_ops.subscript)
    if view is None or len(view.args) != 2 or view.kwargs:
        return None
    source, indices = view.args
    if not isinstance(source, torch.fx.Node) or not isinstance(indices, (list, tuple)):
        return None
    if len(indices) != rank:
        return None
    output_shape = _shape(node)
    source_shape = _shape(source)
    if output_shape is None or source_shape is None:
        return None
    source_dim = 0
    for dim, index in enumerate(indices):
        should_span = (dim in reduced_dims) == source_spans_reduced_dims
        if should_span:
            if (
                not _is_full_slice(index)
                or source_dim >= len(source_shape)
                or not _same_size(output_shape[dim], source_shape[source_dim])
            ):
                return None
            source_dim += 1
        elif index is not None or not _same_size(output_shape[dim], 1):
            return None
    expected_rank = (
        len(reduced_dims) if source_spans_reduced_dims else rank - len(reduced_dims)
    )
    if len(source_shape) != expected_rank:
        return None
    return source


def _copy_meta(
    node: torch.fx.Node, template: torch.fx.Node, value: torch.Tensor
) -> None:
    node.meta = {**template.meta, "val": value}


def _new_call(
    graph: torch.fx.Graph,
    target: object,
    args: tuple[object, ...],
    *,
    template: torch.fx.Node,
    value: torch.Tensor,
    name: str | None = None,
) -> torch.fx.Node:
    # pyrefly: ignore [bad-argument-type]
    node = graph.call_function(cast("object", target), args, {}, name=name)
    _copy_meta(node, template, value)
    return node


def _materialized_shape(
    graph: torch.fx.Graph, shape: tuple[object, ...]
) -> list[int | torch.fx.Node] | None:
    result: list[int | torch.fx.Node] = []
    for size in shape:
        if isinstance(size, int):
            result.append(size)
            continue
        if not isinstance(size, torch.SymInt):
            return None
        source = next(
            (
                node
                for node in graph.nodes
                if isinstance(node.meta.get("val"), torch.SymInt)
                and cast("torch.SymInt", node.meta["val"]).node is size.node
            ),
            None,
        )
        if source is None:
            return None
        result.append(source)
    return result


def _match_sum(node: torch.fx.Node) -> tuple[torch.fx.Node, tuple[int, ...]] | None:
    if _call_node(node, _SUM) is None:
        return None
    dims = _normalized_dims(node)
    if dims is None:
        return None
    product = node.args[0]
    if not isinstance(product, torch.fx.Node):
        return None
    return product, dims


def _match_affine_product(
    product: torch.fx.Node,
    dims: tuple[int, ...],
) -> (
    tuple[
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
    ]
    | None
):
    """Return sources and broadcasts for the matched affine expression."""
    product_args = _binary_args(product, _MUL)
    if product_args is None:
        return None
    product_shape = _shape(product)
    if product_shape is None:
        return None
    rank = len(product_shape)
    reduced = frozenset(dims)

    for update, weight in (product_args, product_args[::-1]):
        weight_source = _broadcast_source(
            weight,
            rank=rank,
            reduced_dims=reduced,
            source_spans_reduced_dims=True,
        )
        if weight_source is None:
            continue
        add_args = _binary_args(update, _ADD)
        if add_args is None:
            continue
        for base, affine in (add_args, add_args[::-1]):
            affine_args = _binary_args(affine, _MUL)
            if affine_args is None:
                continue
            for scalar_view, direction in (affine_args, affine_args[::-1]):
                scalar = _broadcast_source(
                    scalar_view,
                    rank=rank,
                    reduced_dims=reduced,
                    source_spans_reduced_dims=False,
                )
                if scalar is None:
                    continue
                direction_source = _broadcast_source(
                    direction,
                    rank=rank,
                    reduced_dims=reduced,
                    source_spans_reduced_dims=True,
                )
                if direction_source is None:
                    continue
                nodes = (base, scalar, direction_source, weight_source)
                if all(_is_fp32_tensor(candidate) for candidate in nodes):
                    return (
                        base,
                        scalar,
                        scalar_view,
                        direction_source,
                        weight_source,
                        direction,
                        weight,
                    )
    return None


def _rewrite_sum(graph: torch.fx.Graph, reduction: torch.fx.Node) -> bool:
    if not _is_fp32_tensor(reduction):
        return False
    matched_sum = _match_sum(reduction)
    if matched_sum is None:
        return False
    product, dims = matched_sum
    matched_product = _match_affine_product(product, dims)
    if matched_product is None:
        return False
    (
        base,
        scalar,
        scalar_view,
        direction,
        weight,
        direction_broadcast,
        weight_broadcast,
    ) = matched_product

    reduction_value = reduction.meta.get("val")
    product_value = product.meta.get("val")
    direction_value = direction.meta.get("val")
    weight_value = weight.meta.get("val")
    direction_broadcast_value = direction_broadcast.meta.get("val")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            reduction_value,
            product_value,
            direction_value,
            weight_value,
            direction_broadcast_value,
        )
    ):
        return False
    reduction_value = cast("torch.Tensor", reduction_value)
    product_value = cast("torch.Tensor", product_value)
    direction_value = cast("torch.Tensor", direction_value)
    weight_value = cast("torch.Tensor", weight_value)
    direction_broadcast_value = cast("torch.Tensor", direction_broadcast_value)

    try:
        dot_product_shape = tuple(
            torch.broadcast_shapes(direction_value.shape, weight_value.shape)
        )
    except RuntimeError:
        return False
    reduced_input_shape = tuple(product_value.shape[dim] for dim in dims)
    if len(dot_product_shape) != len(dims) or not all(
        starmap(_same_size, zip(dot_product_shape, reduced_input_shape, strict=True))
    ):
        return False
    broadcast_dot_shape = _shape(direction_broadcast)
    scalar_shape = _shape(scalar)
    if broadcast_dot_shape is None or scalar_shape is None:
        return False
    weight_broadcast_shape = _shape(weight_broadcast)
    if (
        weight_broadcast_shape is None
        or len(broadcast_dot_shape) != len(weight_broadcast_shape)
        or not all(
            starmap(
                _same_size,
                zip(broadcast_dot_shape, weight_broadcast_shape, strict=True),
            )
        )
    ):
        return False
    reduced_output_shape = tuple(
        size for dim, size in enumerate(product_value.shape) if dim not in dims
    )
    if len(scalar_shape) != len(reduced_output_shape) or not all(
        starmap(_same_size, zip(scalar_shape, reduced_output_shape, strict=True))
    ):
        return False
    expanded_shape = _materialized_shape(graph, tuple(product_value.shape))
    if expanded_shape is None:
        return False

    with graph.inserting_before(reduction):
        base_product = _new_call(
            graph,
            _MUL,
            (base, weight_broadcast),
            template=product,
            value=product_value.new_empty(product_value.shape),
        )
        base_sum = _new_call(
            graph,
            _SUM,
            (base_product, list(dims)),
            template=reduction,
            value=reduction_value.new_empty(reduction_value.shape),
            name="_helion_factored_base_sum",
        )
        dot_product = _new_call(
            graph,
            _MUL,
            (direction_broadcast, weight_broadcast),
            template=product,
            value=direction_broadcast_value.new_empty(
                cast("tuple[int | torch.SymInt, ...]", broadcast_dot_shape)
            ),
        )
        expanded_dot_product = _new_call(
            graph,
            torch.ops.aten.expand.default,
            (dot_product, expanded_shape),
            template=product,
            value=product_value.new_empty(product_value.shape),
        )
        dot_sum = _new_call(
            graph,
            _SUM,
            (expanded_dot_product, list(dims)),
            template=reduction,
            value=reduction_value.new_empty(reduction_value.shape),
            name="_helion_factored_dot_sum",
        )
        scalar_output_view = _new_call(
            graph,
            view_ops.subscript,
            (scalar, scalar_view.args[1]),
            template=scalar_view,
            value=cast("torch.Tensor", scalar_view.meta["val"]).new_empty(
                cast("torch.Tensor", scalar_view.meta["val"]).shape
            ),
        )
        dot_output_view = _new_call(
            graph,
            view_ops.subscript,
            (dot_sum, scalar_view.args[1]),
            template=scalar_view,
            value=cast("torch.Tensor", scalar_view.meta["val"]).new_empty(
                cast("torch.Tensor", scalar_view.meta["val"]).shape
            ),
        )
        scaled_dot_view = _new_call(
            graph,
            _MUL,
            (scalar_output_view, dot_output_view),
            template=scalar_view,
            value=cast("torch.Tensor", scalar_view.meta["val"]).new_empty(
                cast("torch.Tensor", scalar_view.meta["val"]).shape
            ),
        )
        scaled_dot = _new_call(
            graph,
            _SUM,
            (scaled_dot_view, list(dims)),
            template=reduction,
            value=reduction_value.new_empty(reduction_value.shape),
            name="_helion_factored_scaled_dot",
        )
        factored = _new_call(
            graph,
            _ADD,
            (base_sum, scaled_dot),
            template=reduction,
            value=reduction_value.new_empty(reduction_value.shape),
            name="_helion_factored_result",
        )

    reduction.replace_all_uses_with(factored)
    return True


def factor_affine_reductions(graph: torch.fx.Graph, *, fast_math: bool) -> int:
    """Factor eligible affine reductions in ``graph`` and return the count.

    ``fast_math`` is an explicit argument so the numeric-policy gate remains
    visible and directly testable at the only public entry point.
    """
    if not fast_math:
        return 0
    changed = 0
    for node in list(graph.nodes):
        if _rewrite_sum(graph, node):
            changed += 1
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


def _ast_call_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _assigned_name(stmt: ast.stmt) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def _names_read(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _names_written(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _mutation_roots(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript) or not isinstance(child.ctx, ast.Store):
            continue
        root = child.value
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name):
            result.add(root.id)
    return result


def _statement_effects(stmt: ast.stmt) -> tuple[set[str], set[str]]:
    if isinstance(stmt, ast.Assign):
        reads = _names_read(stmt.value)
        writes = {target.id for target in stmt.targets if isinstance(target, ast.Name)}
        for target in stmt.targets:
            if not isinstance(target, ast.Name):
                reads.update(_names_read(target))
        return reads, writes
    if isinstance(stmt, ast.For):
        bound = {stmt.target.id} if isinstance(stmt.target, ast.Name) else set()
        live, writes = _block_effects(stmt.body, bound)
        return _names_read(stmt.iter) | live, bound | writes
    if isinstance(stmt, ast.If):
        body_live, body_writes = _block_effects(stmt.body, set())
        else_live, else_writes = _block_effects(stmt.orelse, set())
        return (
            _names_read(stmt.test) | body_live | else_live,
            body_writes | else_writes,
        )
    return _names_read(stmt), _names_written(stmt)


def _block_effects(
    body: list[ast.stmt], initially_written: set[str]
) -> tuple[set[str], set[str]]:
    live: set[str] = set()
    written = set(initially_written)
    all_writes: set[str] = set()
    for statement in body:
        reads, writes = _statement_effects(statement)
        live.update(reads - written)
        written.update(writes)
        all_writes.update(writes)
    return live, all_writes


def _definitely_written_block(body: list[ast.stmt]) -> set[str]:
    result: set[str] = set()
    for statement in body:
        if isinstance(statement, ast.Assign):
            result.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(statement, ast.If) and statement.orelse:
            result.update(
                _definitely_written_block(statement.body)
                & _definitely_written_block(statement.orelse)
            )
    return result


def _reads_before_def_statement(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.Assign):
        reads = _names_read(statement.value)
        for target in statement.targets:
            if not isinstance(target, ast.Name):
                reads.update(_names_read(target))
        return reads
    if isinstance(statement, ast.If):
        return (
            _names_read(statement.test)
            | _live_before_block(statement.body)
            | _live_before_block(statement.orelse)
        )
    if isinstance(statement, ast.For):
        bound = (
            {statement.target.id} if isinstance(statement.target, ast.Name) else set()
        )
        return (
            _names_read(statement.iter)
            | _live_before_block(statement.body, initially_defined=bound)
            | _live_before_block(statement.orelse)
        )
    return _names_read(statement)


def _live_before_block(
    body: list[ast.stmt],
    initially_defined: set[str] | None = None,
    live_out: set[str] | None = None,
) -> set[str]:
    """Names that a block may read before a guaranteed local definition."""
    defined = set() if initially_defined is None else set(initially_defined)
    live: set[str] = set()
    for statement in body:
        reads = _reads_before_def_statement(statement)
        live.update(reads - defined)
        if isinstance(statement, ast.Assign):
            defined.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(statement, ast.If) and statement.orelse:
            defined.update(
                _definitely_written_block(statement.body)
                & _definitely_written_block(statement.orelse)
            )
    if live_out is not None:
        live.update(live_out - defined)
    return live


def _nested_live_out(
    statement: ast.stmt,
    field: str,
    child: list[ast.stmt],
    statement_live_out: set[str],
) -> set[str]:
    result = set(statement_live_out)
    if field == "body" and isinstance(statement, ast.For):
        bound = (
            {statement.target.id} if isinstance(statement.target, ast.Name) else set()
        )
        result.update(_live_before_block(child, initially_defined=bound))
    elif field == "body" and isinstance(statement, ast.While):
        result.update(_names_read(statement.test))
        result.update(_live_before_block(child))
    return result


def _zero_fp32_assignment(stmt: ast.stmt) -> str | None:
    name = _assigned_name(stmt)
    if name is None or not isinstance(cast("ast.Assign", stmt).value, ast.Call):
        return None
    call = cast("ast.Call", cast("ast.Assign", stmt).value)
    if (
        _ast_call_path(call.func) != "cutlass.Float32"
        or len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Constant)
        or call.args[0].value != 0
    ):
        return None
    return name


def _constexpr_loop(node: ast.stmt) -> tuple[ast.For, str, int] | None:
    if (
        not isinstance(node, ast.For)
        or not isinstance(node.target, ast.Name)
        or not isinstance(node.iter, ast.Call)
        or _ast_call_path(node.iter.func) != "cutlass.range_constexpr"
        or len(node.iter.args) != 1
        or node.iter.keywords
        or not isinstance(node.iter.args[0], ast.Constant)
        or not isinstance(node.iter.args[0].value, int)
        or node.iter.args[0].value <= 0
        or node.orelse
    ):
        return None
    return node, node.target.id, node.iter.args[0].value


def _static_repeated_loop(node: ast.stmt) -> tuple[ast.For, str, int] | None:
    """Match either generated spelling of a compile-time lane loop."""
    constexpr = _constexpr_loop(node)
    if constexpr is not None:
        return constexpr
    if (
        not isinstance(node, ast.For)
        or not isinstance(node.target, ast.Name)
        or not isinstance(node.iter, ast.Call)
        or _ast_call_path(node.iter.func) != "range"
        or len(node.iter.args) != 1
        or node.iter.keywords
        or not isinstance(node.iter.args[0], ast.Constant)
        or not isinstance(node.iter.args[0].value, int)
        or isinstance(node.iter.args[0].value, bool)
        or node.iter.args[0].value <= 0
        or node.orelse
    ):
        return None
    return node, node.target.id, node.iter.args[0].value


def _reduction_result(stmt: ast.stmt, accumulator: str) -> tuple[str, ast.Call] | None:
    result = _assigned_name(stmt)
    if result is None:
        return None
    value = cast("ast.Assign", stmt).value
    if (
        not isinstance(value, ast.Call)
        or _ast_call_path(value.func) != "cutlass.Float32"
        or len(value.args) != 1
        or value.keywords
        or not isinstance(value.args[0], ast.Call)
    ):
        return None
    reduction = value.args[0]
    if (
        _ast_call_path(reduction.func) != "cute.arch.warp_reduction_sum"
        or len(reduction.args) != 1
        or any(
            keyword.arg != "threads_in_group"
            or not isinstance(keyword.value, ast.Constant)
            or not isinstance(keyword.value.value, int)
            or keyword.value.value <= 0
            for keyword in reduction.keywords
        )
    ):
        return None
    argument = reduction.args[0]
    if not isinstance(argument, ast.Name) or argument.id != accumulator:
        return None
    return result, reduction


def _warp_reduction_assignment(
    stmt: ast.stmt,
) -> tuple[str, str, ast.Call] | None:
    result = _assigned_name(stmt)
    if result is None:
        return None
    value = cast("ast.Assign", stmt).value
    if (
        not isinstance(value, ast.Call)
        or _ast_call_path(value.func) != "cutlass.Float32"
        or len(value.args) != 1
        or value.keywords
        or not isinstance(value.args[0], ast.Call)
    ):
        return None
    reduction = value.args[0]
    if (
        _ast_call_path(reduction.func) != "cute.arch.warp_reduction_sum"
        or len(reduction.args) != 1
        or any(
            keyword.arg != "threads_in_group"
            or not isinstance(keyword.value, ast.Constant)
            or not isinstance(keyword.value.value, int)
            or keyword.value.value <= 0
            for keyword in reduction.keywords
        )
    ):
        return None
    argument = reduction.args[0]
    if not isinstance(argument, ast.Name):
        return None
    return result, argument.id, reduction


def _accumulated_term(stmt: ast.stmt, accumulator: str) -> ast.expr | None:
    if _assigned_name(stmt) != accumulator:
        return None
    value = cast("ast.Assign", stmt).value
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return None
    if isinstance(value.left, ast.Name) and value.left.id == accumulator:
        return value.right
    if isinstance(value.right, ast.Name) and value.right.id == accumulator:
        return value.left
    return None


class _InlineInnerExpression(ast.NodeTransformer):
    def __init__(
        self,
        definitions: dict[str, ast.expr],
        lane_var: str,
        replacement_lane: str,
    ) -> None:
        self.definitions = definitions
        self.lane_var = lane_var
        self.replacement_lane = replacement_lane
        self.resolving: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        if node.id == self.lane_var:
            return ast.copy_location(
                ast.Name(id=self.replacement_lane, ctx=ast.Load()), node
            )
        expression = self.definitions.get(node.id)
        if expression is None:
            return node
        if node.id in self.resolving:
            raise ValueError("cyclic generated definition")
        self.resolving.add(node.id)
        try:
            return self.visit(expr_from_string(ast.unparse(expression)))
        finally:
            self.resolving.remove(node.id)


_AST_PURE_CALLS = {
    "cutlass.BFloat16",
    "cutlass.Float16",
    "cutlass.Float32",
    "cutlass.Int32",
    "cutlass.Int64",
    "cutlass.Uint16",
    "cutlass.Uint32",
    "cute.arch.load",
}
_AST_PURE_OPERATOR_CALLS = {
    "operator.add",
    "operator.and_",
    "operator.eq",
    "operator.floordiv",
    "operator.ge",
    "operator.gt",
    "operator.index",
    "operator.invert",
    "operator.le",
    "operator.lshift",
    "operator.lt",
    "operator.mod",
    "operator.mul",
    "operator.ne",
    "operator.neg",
    "operator.or_",
    "operator.pos",
    "operator.pow",
    "operator.rshift",
    "operator.sub",
    "operator.truediv",
    "operator.xor",
}
_AST_GLOBAL_NAMES = {"cutlass", "cute", "operator"}


def _clone_extended_ast(value: object) -> object:
    """Clone Python/ExtendedAST nodes without losing Helion metadata."""
    if isinstance(value, list):
        return [_clone_extended_ast(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_extended_ast(item) for item in value)
    if isinstance(value, ast.AST):
        fields = {
            field: _clone_extended_ast(getattr(value, field)) for field in value._fields
        }
        if isinstance(value, ExtendedAST):
            return value.copy(**fields)
        return ast.copy_location(type(value)(**fields), value)
    return value


def _safe_invariant_expression(
    expression: ast.expr,
    *,
    outer_lane: str,
    outer_writes: set[str],
    outer_mutations: set[str],
    rmem_names: set[str],
) -> bool:
    if outer_lane in _names_read(expression):
        return False
    free_names = _names_read(expression) - _AST_GLOBAL_NAMES
    if free_names & (outer_writes | outer_mutations):
        return False
    for child in ast.walk(expression):
        if isinstance(
            child,
            (
                ast.Await,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.ListComp,
                ast.NamedExpr,
                ast.SetComp,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            return False
        if isinstance(child, ast.Call):
            path = _ast_call_path(child.func)
            if path == "cute.arch.load" or path not in _AST_PURE_CALLS:
                return False
        if isinstance(child, ast.Subscript):
            if isinstance(child.ctx, ast.Store):
                return False
            if (
                not isinstance(child.value, ast.Name)
                or child.value.id not in rmem_names
            ):
                return False
    return True


def _fresh_ast_name(occupied: set[str], prefix: str) -> str:
    index = 0
    while f"{prefix}_{index}" in occupied:
        index += 1
    result = f"{prefix}_{index}"
    occupied.add(result)
    return result


class _RenameAstName(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id != self.old:
            return node
        return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)


def _unwrap_fp32(expression: ast.expr) -> ast.expr:
    if (
        isinstance(expression, ast.Call)
        and _ast_call_path(expression.func) == "cutlass.Float32"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return expression.args[0]
    return expression


def _as_fp32(expression: ast.expr) -> ast.expr:
    if (
        isinstance(expression, ast.Call)
        and _ast_call_path(expression.func) == "cutlass.Float32"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return expression
    return create(
        ast.Call,
        func=expr_from_string("cutlass.Float32"),
        args=[expression],
        keywords=[],
    )


def _packed_pair_assignment(
    lo_name: str,
    hi_name: str,
    function: str,
    lo_lhs: ast.expr,
    hi_lhs: ast.expr,
    lo_rhs: ast.expr,
    hi_rhs: ast.expr,
    lo_acc: ast.expr | None = None,
    hi_acc: ast.expr | None = None,
) -> ast.Assign:
    arguments = [
        create(
            ast.Tuple,
            elts=[_as_fp32(lo_lhs), _as_fp32(hi_lhs)],
            ctx=ast.Load(),
        ),
        create(
            ast.Tuple,
            elts=[_as_fp32(lo_rhs), _as_fp32(hi_rhs)],
            ctx=ast.Load(),
        ),
    ]
    if lo_acc is not None and hi_acc is not None:
        arguments.append(
            create(
                ast.Tuple,
                elts=[_as_fp32(lo_acc), _as_fp32(hi_acc)],
                ctx=ast.Load(),
            )
        )
    return create(
        ast.Assign,
        targets=[
            create(
                ast.Tuple,
                elts=[
                    create(ast.Name, id=lo_name, ctx=ast.Store()),
                    create(ast.Name, id=hi_name, ctx=ast.Store()),
                ],
                ctx=ast.Store(),
            )
        ],
        value=create(
            ast.Call,
            func=expr_from_string(function),
            args=arguments,
            keywords=[],
        ),
    )


def _pair_lane_expression(pair_name: str, high: bool) -> ast.expr:
    expression = create(
        ast.BinOp,
        left=create(ast.Name, id=pair_name, ctx=ast.Load()),
        op=create(ast.Mult),
        right=create(ast.Constant, value=2),
    )
    if high:
        expression = create(
            ast.BinOp,
            left=expression,
            op=create(ast.Add),
            right=create(ast.Constant, value=1),
        )
    return expression


def _subscript_accesses(
    body: list[ast.stmt], context: type[ast.expr_context]
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for statement in body:
        for child in ast.walk(statement):
            if not isinstance(child, ast.Subscript) or not isinstance(
                child.ctx, context
            ):
                continue
            if isinstance(child.value, ast.Name):
                result.setdefault(child.value.id, set()).add(ast.unparse(child.slice))
    return result


def _has_effectful_call(
    body: list[ast.stmt], *, ignored_calls: frozenset[int] = frozenset()
) -> bool:
    for statement in body:
        for child in ast.walk(statement):
            if not isinstance(child, ast.Call):
                continue
            if id(child) in ignored_calls:
                continue
            path = _ast_call_path(child.func)
            if path is None:
                if isinstance(child.func, ast.Attribute):
                    receiver = child.func.value
                    if (
                        child.func.attr == "bitcast"
                        and isinstance(receiver, ast.Call)
                        and _ast_call_path(receiver.func) in _AST_PURE_CALLS
                    ):
                        continue
                    if child.func.attr == "load" and any(
                        isinstance(node, ast.Attribute) and node.attr == "iterator"
                        for node in ast.walk(receiver)
                    ):
                        continue
                return True
            if path in _AST_PURE_CALLS or path.startswith("cute.math."):
                continue
            if path in _AST_PURE_OPERATOR_CALLS:
                continue
            return True
    return False


def _direct_subscript_key(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    return node.value.id, ast.dump(node.slice, include_attributes=False)


def _deduplicate_fused_body(
    context: list[ast.stmt],
    first: list[ast.stmt],
    second: list[ast.stmt],
    rmem_names: set[str],
) -> list[ast.stmt]:
    """Remove second-sweep scalar recomputation made redundant by fusion."""
    versions: dict[str, int] = {}
    memory_versions: dict[str, int] = {}
    definitions: dict[str, tuple[str, dict[str, int], dict[str, int]]] = {}
    cache_values: dict[tuple[str, str], tuple[str, int, int]] = {}

    def process(statement: ast.stmt, *, may_drop: bool) -> bool:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return False
        target = statement.targets[0]
        if isinstance(target, ast.Subscript):
            key = _direct_subscript_key(target)
            if key is not None and key[0] in rmem_names:
                root = key[0]
                memory_versions[root] = memory_versions.get(root, 0) + 1
                for stale_key in [
                    cached_key for cached_key in cache_values if cached_key[0] == root
                ]:
                    del cache_values[stale_key]
                if isinstance(statement.value, ast.Name):
                    cache_values[key] = (
                        statement.value.id,
                        versions.get(statement.value.id, 0),
                        memory_versions[root],
                    )
            return False
        if not isinstance(target, ast.Name):
            return False

        name = target.id
        rhs = statement.value
        rhs_names = _names_read(rhs) - _AST_GLOBAL_NAMES
        rhs_versions = {
            dependency: versions.get(dependency, 0) for dependency in rhs_names
        }
        rhs_roots: set[str] = set()
        for child in ast.walk(rhs):
            key = _direct_subscript_key(child)
            if key is not None and key[0] in rmem_names:
                rhs_roots.add(key[0])
        rhs_memory_versions = {root: memory_versions.get(root, 0) for root in rhs_roots}
        redundant = False
        previous = definitions.get(name)
        if (
            may_drop
            and previous is not None
            and not _has_effectful_call([statement])
            and previous
            == (
                ast.dump(rhs, include_attributes=False),
                rhs_versions,
                rhs_memory_versions,
            )
        ):
            redundant = True
        cache_key = _direct_subscript_key(rhs)
        cached = cache_values.get(cache_key) if cache_key is not None else None
        if (
            may_drop
            and cache_key is not None
            and cached is not None
            and cached[0] == name
            and cached[1] == versions.get(name, 0)
            and cached[2] == memory_versions.get(cache_key[0], 0)
        ):
            redundant = True
        if redundant:
            return True

        versions[name] = versions.get(name, 0) + 1
        definitions[name] = (
            ast.dump(rhs, include_attributes=False),
            {dependency: versions.get(dependency, 0) for dependency in rhs_names},
            rhs_memory_versions,
        )
        return False

    for statement in (*context, *first):
        process(statement, may_drop=False)
    kept_second = [
        statement for statement in second if not process(statement, may_drop=True)
    ]
    return [*first, *kept_second]


class _ClonePairLane(ast.NodeTransformer):
    def __init__(
        self,
        lane_name: str,
        lane_expression: ast.expr,
        renames: dict[str, str],
    ) -> None:
        self.lane_name = lane_name
        self.lane_expression = lane_expression
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.lane_name and isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                cast("ast.expr", _clone_extended_ast(self.lane_expression)),
                node,
            )
        replacement = self.renames.get(node.id)
        if replacement is None:
            return node
        cloned = cast("ast.Name", _clone_extended_ast(node))
        cloned.id = replacement
        return cloned


def _clone_pair_expression(
    expression: ast.expr,
    *,
    lane_name: str,
    lane_expression: ast.expr,
    renames: dict[str, str],
) -> ast.expr:
    return cast(
        "ast.expr",
        _ClonePairLane(lane_name, lane_expression, renames).visit(
            cast("ast.expr", _clone_extended_ast(expression))
        ),
    )


def _definitions(body: list[ast.stmt]) -> dict[str, ast.expr] | None:
    result: dict[str, ast.expr] = {}
    for statement in body:
        name = _assigned_name(statement)
        if name is None:
            continue
        if name in result:
            return None
        result[name] = cast("ast.Assign", statement).value
    return result


def _read_counts(body: list[ast.stmt]) -> dict[str, int]:
    result: dict[str, int] = {}
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                result[node.id] = result.get(node.id, 0) + 1
    return result


def _dependency_closure(names: set[str], definitions: dict[str, ast.expr]) -> set[str]:
    result = set(names)
    worklist = list(names)
    while worklist:
        name = worklist.pop()
        expression = definitions.get(name)
        if expression is None:
            continue
        for dependency in _names_read(expression):
            if dependency not in result:
                result.add(dependency)
                worklist.append(dependency)
    return result


def _index_dependencies(
    body: list[ast.stmt], definitions: dict[str, ast.expr]
) -> set[str]:
    names = {
        name
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Subscript)
        for name in _names_read(node.slice)
    }
    return _dependency_closure(names, definitions)


def _strip_int_cast(expression: ast.expr) -> ast.expr:
    while (
        isinstance(expression, ast.Call)
        and _ast_call_path(expression.func) in {"cutlass.Int32", "cutlass.Int64"}
        and len(expression.args) == 1
        and not expression.keywords
    ):
        expression = expression.args[0]
    return expression


def _lane_affine_coefficient(expression: ast.expr, lane_name: str) -> int | None:
    """Return the integer lane coefficient for a simple affine expression."""
    expression = _strip_int_cast(expression)
    if isinstance(expression, ast.Name):
        return 1 if expression.id == lane_name else 0
    if isinstance(expression, ast.Constant):
        return 0 if isinstance(expression.value, (int, bool)) else None
    if isinstance(expression, ast.UnaryOp) and isinstance(
        expression.op, (ast.UAdd, ast.USub)
    ):
        coefficient = _lane_affine_coefficient(expression.operand, lane_name)
        if coefficient is None:
            return None
        return -coefficient if isinstance(expression.op, ast.USub) else coefficient
    if not isinstance(expression, ast.BinOp):
        return 0 if lane_name not in _names_read(expression) else None
    if isinstance(expression.op, (ast.Add, ast.Sub)):
        left = _lane_affine_coefficient(expression.left, lane_name)
        right = _lane_affine_coefficient(expression.right, lane_name)
        if left is None or right is None:
            return None
        return left - right if isinstance(expression.op, ast.Sub) else left + right
    if isinstance(expression.op, ast.Mult):
        left = _strip_int_cast(expression.left)
        right = _strip_int_cast(expression.right)
        if isinstance(left, ast.Constant) and isinstance(left.value, int):
            coefficient = _lane_affine_coefficient(right, lane_name)
            return None if coefficient is None else left.value * coefficient
        if isinstance(right, ast.Constant) and isinstance(right.value, int):
            coefficient = _lane_affine_coefficient(left, lane_name)
            return None if coefficient is None else right.value * coefficient
        return None if lane_name in _names_read(expression) else 0
    if (
        isinstance(expression.op, ast.FloorDiv)
        and isinstance(_strip_int_cast(expression.right), ast.Constant)
        and cast("ast.Constant", _strip_int_cast(expression.right)).value in {1, -1}
    ):
        coefficient = _lane_affine_coefficient(expression.left, lane_name)
        if coefficient is None:
            return None
        divisor = cast(
            "int", cast("ast.Constant", _strip_int_cast(expression.right)).value
        )
        return coefficient // divisor
    return None if lane_name in _names_read(expression) else 0


def _safe_lane_local_stores(
    loop: ast.For, lane_name: str, rmem_names: set[str]
) -> bool:
    mutated = _mutation_roots(loop)
    loop_locals = _names_written(loop) - {lane_name}
    store_mappings: dict[str, set[str]] = {}
    if not mutated <= rmem_names:
        return False
    for node in ast.walk(loop):
        if not isinstance(node, ast.Subscript):
            continue
        if isinstance(node.ctx, ast.Store) and not isinstance(node.value, ast.Name):
            return False
        if not isinstance(node.value, ast.Name):
            continue
        root = node.value.id
        if root not in mutated:
            continue
        if isinstance(node.ctx, ast.Load):
            # A read from a fragment written by the loop can carry values
            # between iterations.  Pairing changes that statement order.
            return False
        if not isinstance(node.ctx, ast.Store):
            return False
        if _names_read(node.slice) & loop_locals:
            return False
        coefficient = _lane_affine_coefficient(node.slice, lane_name)
        if coefficient in {None, 0}:
            return False
        store_mappings.setdefault(root, set()).add(
            ast.dump(node.slice, include_attributes=False)
        )
        if len(store_mappings[root]) > 1:
            return False
    return True


_FP32_MATH_CALLS = {
    "cute.math.absf",
    "cute.math.atan2",
    "cute.math.div",
    "cute.math.erf",
    "cute.math.exp",
    "cute.math.exp2",
    "cute.math.fma",
    "cute.math.log",
    "cute.math.log2",
    "cute.math.maximum",
    "cute.math.minimum",
    "cute.math.rcp",
    "cute.math.rsqrt",
    "cute.math.sqrt",
    "cute.math.tanh",
}


def _numeric_expression_kind(
    expression: ast.expr,
    definitions: dict[str, ast.expr],
    fp32_rmem: set[str],
    fp32_names: set[str],
    resolving: set[str] | None = None,
) -> str | None:
    if (
        isinstance(expression, ast.Call)
        and _ast_call_path(expression.func) == "cutlass.Float32"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return "fp32"
    if (
        isinstance(expression, ast.Call)
        and _ast_call_path(expression.func) == "cutlass.Int32"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return "int"
    if isinstance(expression, ast.Constant):
        if isinstance(expression.value, float):
            return "fp32"
        if isinstance(expression.value, (bool, int)):
            return "int"
        return None
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name):
        return "fp32" if expression.value.id in fp32_rmem else None
    if isinstance(expression, ast.Name):
        if expression.id in fp32_names:
            return "fp32"
        definition = definitions.get(expression.id)
        if definition is None:
            return None
        resolving = set() if resolving is None else resolving
        if expression.id in resolving:
            return None
        resolving.add(expression.id)
        try:
            return _numeric_expression_kind(
                definition,
                definitions,
                fp32_rmem,
                fp32_names,
                resolving,
            )
        finally:
            resolving.remove(expression.id)
    if isinstance(expression, ast.UnaryOp) and isinstance(
        expression.op, (ast.UAdd, ast.USub)
    ):
        return _numeric_expression_kind(
            expression.operand, definitions, fp32_rmem, fp32_names, resolving
        )
    if isinstance(expression, ast.BinOp) and isinstance(
        expression.op, (ast.Add, ast.Mult, ast.Sub)
    ):
        left = _numeric_expression_kind(
            expression.left, definitions, fp32_rmem, fp32_names, resolving
        )
        right = _numeric_expression_kind(
            expression.right, definitions, fp32_rmem, fp32_names, resolving
        )
        if left is None or right is None:
            return None
        return "fp32" if "fp32" in {left, right} else "int"
    if isinstance(expression, ast.IfExp):
        body_kind = _numeric_expression_kind(
            expression.body, definitions, fp32_rmem, fp32_names, resolving
        )
        else_kind = _numeric_expression_kind(
            expression.orelse, definitions, fp32_rmem, fp32_names, resolving
        )
        if body_kind is None or else_kind is None:
            return None
        return "fp32" if "fp32" in {body_kind, else_kind} else "int"
    if isinstance(expression, ast.Call):
        path = _ast_call_path(expression.func)
        if path not in _FP32_MATH_CALLS or not expression.args:
            return None
        argument_kinds = [
            _numeric_expression_kind(
                argument, definitions, fp32_rmem, fp32_names, resolving
            )
            for argument in expression.args
        ]
        if any(kind is None for kind in argument_kinds):
            return None
        return "fp32" if "fp32" in argument_kinds else None
    return None


def _is_fp32_expression(
    expression: ast.expr,
    definitions: dict[str, ast.expr],
    fp32_rmem: set[str],
    fp32_names: set[str],
) -> bool:
    return (
        _numeric_expression_kind(
            expression,
            definitions,
            fp32_rmem,
            fp32_names,
        )
        == "fp32"
    )


def _single_use_product(
    expression: ast.expr,
    definitions: dict[str, ast.expr],
    reads: dict[str, int],
) -> tuple[str | None, ast.BinOp] | None:
    expression = _unwrap_fp32(expression)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        return None, expression
    if not isinstance(expression, ast.Name) or reads.get(expression.id) != 1:
        return None
    definition = _unwrap_fp32(definitions.get(expression.id, expression))
    if not isinstance(definition, ast.BinOp) or not isinstance(definition.op, ast.Mult):
        return None
    return expression.id, definition


class _InlineSingleUseProducts(ast.NodeTransformer):
    def __init__(
        self,
        definitions: dict[str, ast.expr],
        reads: dict[str, int],
        blocked: set[str],
        fp32_rmem: set[str],
        fp32_names: set[str],
    ) -> None:
        self.definitions = definitions
        self.reads = reads
        self.blocked = blocked
        self.fp32_rmem = fp32_rmem
        self.fp32_names = fp32_names
        self.inlined: set[str] = set()
        self.resolving: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            not isinstance(node.ctx, ast.Load)
            or node.id in self.blocked
            or node.id in self.resolving
            or self.reads.get(node.id) != 1
        ):
            return node
        definition = self.definitions.get(node.id)
        if definition is None:
            return node
        product = _unwrap_fp32(definition)
        if (
            not isinstance(product, ast.BinOp)
            or not isinstance(product.op, ast.Mult)
            or not _is_fp32_expression(
                product,
                self.definitions,
                self.fp32_rmem,
                self.fp32_names,
            )
        ):
            return node
        self.inlined.add(node.id)
        self.resolving.add(node.id)
        try:
            return self.visit(cast("ast.expr", _clone_extended_ast(product)))
        finally:
            self.resolving.remove(node.id)


def _prepare_packed_product(
    matched: tuple[str | None, ast.BinOp],
    *,
    definitions: dict[str, ast.expr],
    reads: dict[str, int],
    blocked: set[str],
    fp32_rmem: set[str],
    fp32_names: set[str],
) -> tuple[ast.BinOp, set[str]]:
    name, product = matched
    inliner = _InlineSingleUseProducts(
        definitions,
        reads,
        blocked,
        fp32_rmem,
        fp32_names,
    )
    prepared = cast(
        "ast.BinOp",
        inliner.visit(cast("ast.BinOp", _clone_extended_ast(product))),
    )
    skipped = set(inliner.inlined)
    if name is not None:
        skipped.add(name)
    return prepared, skipped


def _pair_assignment_names(
    local_names: set[str], occupied: set[str]
) -> tuple[dict[str, str], dict[str, str]]:
    lo = {
        name: _fresh_ast_name(occupied, f"_factored_{name}_lo")
        for name in sorted(local_names)
    }
    hi = {
        name: _fresh_ast_name(occupied, f"_factored_{name}_hi")
        for name in sorted(local_names)
    }
    return lo, hi


def _clone_statement_for_pair(
    statement: ast.stmt,
    *,
    lane_name: str,
    lane_expression: ast.expr,
    renames: dict[str, str],
) -> ast.stmt:
    return cast(
        "ast.stmt",
        _ClonePairLane(lane_name, lane_expression, renames).visit(
            cast("ast.stmt", _clone_extended_ast(statement))
        ),
    )


def _pair_values(
    expression: ast.expr,
    *,
    lane_name: str,
    lane_lo: ast.expr,
    lane_hi: ast.expr,
    renames_lo: dict[str, str],
    renames_hi: dict[str, str],
) -> tuple[ast.expr, ast.expr]:
    return (
        _clone_pair_expression(
            expression,
            lane_name=lane_name,
            lane_expression=lane_lo,
            renames=renames_lo,
        ),
        _clone_pair_expression(
            expression,
            lane_name=lane_name,
            lane_expression=lane_hi,
            renames=renames_hi,
        ),
    )


def _pair_arithmetic_assignment(
    target: str,
    expression: ast.BinOp,
    *,
    lane_name: str,
    lane_lo: ast.expr,
    lane_hi: ast.expr,
    renames_lo: dict[str, str],
    renames_hi: dict[str, str],
    product: ast.BinOp | None = None,
    addend: ast.expr | None = None,
) -> ast.Assign:
    lhs_expression = expression.left if product is None else product.left
    rhs_expression = expression.right if product is None else product.right
    lhs_lo, lhs_hi = _pair_values(
        lhs_expression,
        lane_name=lane_name,
        lane_lo=lane_lo,
        lane_hi=lane_hi,
        renames_lo=renames_lo,
        renames_hi=renames_hi,
    )
    rhs_lo, rhs_hi = _pair_values(
        rhs_expression,
        lane_name=lane_name,
        lane_lo=lane_lo,
        lane_hi=lane_hi,
        renames_lo=renames_lo,
        renames_hi=renames_hi,
    )
    if product is not None and addend is not None:
        acc_lo, acc_hi = _pair_values(
            addend,
            lane_name=lane_name,
            lane_lo=lane_lo,
            lane_hi=lane_hi,
            renames_lo=renames_lo,
            renames_hi=renames_hi,
        )
        return _packed_pair_assignment(
            renames_lo[target],
            renames_hi[target],
            "cute.arch.fma_packed_f32x2",
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            acc_lo,
            acc_hi,
        )
    function = (
        "cute.arch.mul_packed_f32x2"
        if isinstance(expression.op, ast.Mult)
        else "cute.arch.add_packed_f32x2"
    )
    return _packed_pair_assignment(
        renames_lo[target],
        renames_hi[target],
        function,
        lhs_lo,
        lhs_hi,
        rhs_lo,
        rhs_hi,
    )


def _append_call(statement: ast.stmt) -> ast.Call | None:
    if (
        not isinstance(statement, ast.Expr)
        or not isinstance(statement.value, ast.Call)
        or not isinstance(statement.value.func, ast.Attribute)
        or statement.value.func.attr != "append"
        or len(statement.value.args) != 1
        or statement.value.keywords
    ):
        return None
    return statement.value


def _pair_constexpr_loop(
    loop: ast.For,
    accumulators: tuple[str, ...],
    accumulator_pairs: dict[str, tuple[str, str]],
    *,
    occupied: set[str],
    rmem_names: set[str],
    fp32_rmem: set[str],
    fp32_names: set[str],
    live_after: set[str],
) -> ast.For | None:
    matched = _constexpr_loop(loop)
    if matched is None:
        return None
    _loop, lane_name, extent = matched
    if extent % 2 != 0 or extent < 2:
        return None
    append_calls = [
        call for statement in loop.body if (call := _append_call(statement)) is not None
    ]
    if not accumulators and len(append_calls) != 1:
        return None
    if len(append_calls) > 1:
        return None
    if append_calls and _append_call(loop.body[-1]) is None:
        return None
    if any(
        isinstance(
            node,
            (
                ast.Await,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.ListComp,
                ast.NamedExpr,
                ast.SetComp,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for node in ast.walk(loop)
    ):
        return None
    for statement in loop.body:
        if _append_call(statement) is not None:
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        if not isinstance(target, (ast.Name, ast.Subscript)):
            return None
        if isinstance(target, ast.Subscript) and not isinstance(target.value, ast.Name):
            return None
    if _has_effectful_call(
        loop.body,
        ignored_calls=frozenset(id(call) for call in append_calls),
    ):
        return None
    if not _safe_lane_local_stores(loop, lane_name, rmem_names):
        return None
    if len(accumulators) == 1 and _mutation_roots(loop):
        # Pairing one reduction while duplicating its cache-population path
        # lengthens live ranges for little arithmetic savings.  Multiple
        # reductions amortize those paired lane values; a pure reduction has
        # no store path to duplicate.
        return None
    definitions = _definitions(loop.body)
    if definitions is None:
        return None
    local_names = set(definitions) - set(accumulators)
    if lane_name in local_names:
        return None
    if (local_names | {lane_name}) & live_after:
        return None

    seen: set[str] = set()
    accumulator_set = set(accumulators)
    for statement in loop.body:
        assigned = _assigned_name(statement)
        reads = _names_read(statement)
        if assigned in accumulator_set:
            assert assigned is not None
            term = _accumulated_term(statement, assigned)
            if term is None or _names_read(term) & accumulator_set:
                return None
            reads.discard(assigned)
        elif reads & accumulator_set:
            return None
        if reads & (local_names - seen):
            # Reject forward and loop-carried scalar dependencies.
            return None
        if assigned is not None:
            seen.add(assigned)

    reads = _read_counts(loop.body)
    seeds: set[str] = set()
    accumulator_terms: dict[str, ast.expr] = {}
    for accumulator in accumulators:
        matches = [
            term
            for statement in loop.body
            if (term := _accumulated_term(statement, accumulator)) is not None
        ]
        if len(matches) != 1:
            return None
        accumulator_terms[accumulator] = matches[0]
        seeds.update(_names_read(matches[0]))
    append_statements = [
        statement for statement in loop.body if _append_call(statement) is not None
    ]
    for statement in append_statements:
        seeds.update(_names_read(statement))
    value_dependencies = _dependency_closure(seeds, definitions)
    index_dependencies = _index_dependencies(loop.body, definitions)
    blocked_inline = index_dependencies | accumulator_set

    skipped_products: set[str] = set()
    accumulator_products: dict[str, ast.BinOp | None] = {}
    for accumulator, term in accumulator_terms.items():
        product = _single_use_product(term, definitions, reads)
        if product is not None and _is_fp32_expression(
            product[1], definitions, fp32_rmem, fp32_names
        ):
            prepared, skipped = _prepare_packed_product(
                product,
                definitions=definitions,
                reads=reads,
                blocked=blocked_inline,
                fp32_rmem=fp32_rmem,
                fp32_names=fp32_names,
            )
            accumulator_products[accumulator] = prepared
            skipped_products.update(skipped)
        elif _is_fp32_expression(term, definitions, fp32_rmem, fp32_names):
            accumulator_products[accumulator] = None
        else:
            return None

    fused_assignments: dict[str, tuple[ast.BinOp, ast.expr]] = {}
    for name in value_dependencies - index_dependencies:
        expression = _unwrap_fp32(definitions.get(name, ast.Name(id=name)))
        if (
            not isinstance(expression, ast.BinOp)
            or not isinstance(expression.op, ast.Add)
            or not _is_fp32_expression(expression, definitions, fp32_rmem, fp32_names)
        ):
            continue
        candidates: list[tuple[int, ast.BinOp, ast.expr, set[str]]] = []
        for operand, other in (
            (expression.left, expression.right),
            (expression.right, expression.left),
        ):
            product = _single_use_product(operand, definitions, reads)
            if (
                product is not None
                and product[0] is not None
                and _is_fp32_expression(product[1], definitions, fp32_rmem, fp32_names)
            ):
                prepared, skipped = _prepare_packed_product(
                    product,
                    definitions=definitions,
                    reads=reads,
                    blocked=blocked_inline,
                    fp32_rmem=fp32_rmem,
                    fp32_names=fp32_names,
                )
                candidates.append((len(skipped), prepared, other, skipped))
        if candidates:
            best = candidates[0]
            for candidate in candidates[1:]:
                if candidate[0] > best[0]:
                    best = candidate
            _score, prepared, other, skipped = best
            skipped_products.update(skipped)
            fused_assignments[name] = prepared, other

    renames_lo, renames_hi = _pair_assignment_names(local_names, occupied)
    for accumulator, (accumulator_lo, accumulator_hi) in accumulator_pairs.items():
        renames_lo[accumulator] = accumulator_lo
        renames_hi[accumulator] = accumulator_hi
    pair_lane = _fresh_ast_name(occupied, "_factored_pair_lane")
    lane_lo = _pair_lane_expression(pair_lane, False)
    lane_hi = _pair_lane_expression(pair_lane, True)
    result: list[ast.stmt] = []
    packed_operations = 0

    for statement in loop.body:
        assigned = _assigned_name(statement)
        if assigned in accumulators:
            assert assigned is not None
            term = accumulator_terms[assigned]
            product = accumulator_products[assigned]
            accumulator_lo, accumulator_hi = accumulator_pairs[assigned]
            if product is not None:
                lhs_lo, lhs_hi = _pair_values(
                    product.left,
                    lane_name=lane_name,
                    lane_lo=lane_lo,
                    lane_hi=lane_hi,
                    renames_lo=renames_lo,
                    renames_hi=renames_hi,
                )
                rhs_lo, rhs_hi = _pair_values(
                    product.right,
                    lane_name=lane_name,
                    lane_lo=lane_lo,
                    lane_hi=lane_hi,
                    renames_lo=renames_lo,
                    renames_hi=renames_hi,
                )
                result.append(
                    _packed_pair_assignment(
                        accumulator_lo,
                        accumulator_hi,
                        "cute.arch.fma_packed_f32x2",
                        lhs_lo,
                        lhs_hi,
                        rhs_lo,
                        rhs_hi,
                        create(ast.Name, id=accumulator_lo, ctx=ast.Load()),
                        create(ast.Name, id=accumulator_hi, ctx=ast.Load()),
                    )
                )
                packed_operations += 1
            else:
                term_lo, term_hi = _pair_values(
                    _unwrap_fp32(term),
                    lane_name=lane_name,
                    lane_lo=lane_lo,
                    lane_hi=lane_hi,
                    renames_lo=renames_lo,
                    renames_hi=renames_hi,
                )
                result.append(
                    _packed_pair_assignment(
                        accumulator_lo,
                        accumulator_hi,
                        "cute.arch.add_packed_f32x2",
                        create(ast.Name, id=accumulator_lo, ctx=ast.Load()),
                        create(ast.Name, id=accumulator_hi, ctx=ast.Load()),
                        term_lo,
                        term_hi,
                    )
                )
                packed_operations += 1
            continue
        if assigned is not None and assigned in skipped_products:
            continue
        if assigned is not None:
            expression = _unwrap_fp32(cast("ast.Assign", statement).value)
            fused = fused_assignments.get(assigned)
            if fused is not None:
                result.append(
                    _pair_arithmetic_assignment(
                        assigned,
                        cast("ast.BinOp", expression),
                        lane_name=lane_name,
                        lane_lo=lane_lo,
                        lane_hi=lane_hi,
                        renames_lo=renames_lo,
                        renames_hi=renames_hi,
                        product=fused[0],
                        addend=fused[1],
                    )
                )
                packed_operations += 1
                continue
            if (
                assigned in value_dependencies - index_dependencies
                and isinstance(expression, ast.BinOp)
                and isinstance(expression.op, (ast.Add, ast.Mult))
                and _is_fp32_expression(expression, definitions, fp32_rmem, fp32_names)
            ):
                result.append(
                    _pair_arithmetic_assignment(
                        assigned,
                        expression,
                        lane_name=lane_name,
                        lane_lo=lane_lo,
                        lane_hi=lane_hi,
                        renames_lo=renames_lo,
                        renames_hi=renames_hi,
                    )
                )
                packed_operations += 1
                continue
        result.extend(
            (
                _clone_statement_for_pair(
                    statement,
                    lane_name=lane_name,
                    lane_expression=lane_lo,
                    renames=renames_lo,
                ),
                _clone_statement_for_pair(
                    statement,
                    lane_name=lane_name,
                    lane_expression=lane_hi,
                    renames=renames_hi,
                ),
            )
        )

    if packed_operations == 0:
        return None
    return create(
        ast.For,
        target=create(ast.Name, id=pair_lane, ctx=ast.Store()),
        iter=expr_from_string(f"cutlass.range_constexpr({extent // 2})"),
        body=result,
        orelse=[],
        type_comment=None,
    )


def _rmem_storage_names(body: list[ast.stmt]) -> tuple[set[str], set[str]]:
    module = ast.Module(body=body, type_ignores=[])
    assignments: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(module):
        name = _assigned_name(node) if isinstance(node, ast.stmt) else None
        if name is not None:
            assignments.setdefault(name, []).append(cast("ast.Assign", node))
    write_counts: dict[str, int] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            write_counts[node.id] = write_counts.get(node.id, 0) + 1

    rmem_names: set[str] = set()
    fp32_rmem: set[str] = set()
    for name, definitions in assignments.items():
        if len(definitions) != 1 or write_counts.get(name) != 1:
            continue
        value = definitions[0].value
        if (
            not isinstance(value, ast.Call)
            or _ast_call_path(value.func) != "cute.make_rmem_tensor"
            or len(value.args) != 2
            or value.keywords
        ):
            continue
        rmem_names.add(name)
        if _ast_call_path(cast("ast.expr", value.args[1])) == "cutlass.Float32":
            fp32_rmem.add(name)

    # A register fragment is considered non-aliasing only when every load of
    # its Python name is the base of a direct subscript.  Passing the fragment
    # to a helper or assigning it to another name could create an alias that a
    # lane-local store analysis cannot see.
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    aliased: set[str] = set()
    for node in ast.walk(module):
        if (
            not isinstance(node, ast.Name)
            or not isinstance(node.ctx, ast.Load)
            or node.id not in rmem_names
        ):
            continue
        parent = parents.get(id(node))
        if not (isinstance(parent, ast.Subscript) and parent.value is node):
            aliased.add(node.id)
    rmem_names -= aliased
    return rmem_names, fp32_rmem


def _collect_fp32_names(
    body: list[ast.stmt],
    fp32_rmem: set[str],
    initial: set[str],
) -> set[str]:
    module = ast.Module(body=body, type_ignores=[])
    assignments: dict[str, list[ast.expr]] = {}
    write_counts: dict[str, int] = {}
    for node in ast.walk(module):
        name = _assigned_name(node) if isinstance(node, ast.stmt) else None
        if name is not None:
            assignments.setdefault(name, []).append(cast("ast.Assign", node).value)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            write_counts[node.id] = write_counts.get(node.id, 0) + 1
    result = {name for name in initial if write_counts.get(name, 0) == 0}
    changed = True
    while changed:
        changed = False
        for name, expressions in assignments.items():
            if name in result:
                continue
            if write_counts.get(name) == len(expressions) and all(
                _is_fp32_expression(expression, {}, fp32_rmem, result)
                for expression in expressions
            ):
                result.add(name)
                changed = True
    return result


def _empty_list_assignment(statement: ast.stmt) -> str | None:
    name = _assigned_name(statement)
    if name is None:
        return None
    value = cast("ast.Assign", statement).value
    if not isinstance(value, ast.List) or value.elts:
        return None
    return name


def _append_target(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute) or not isinstance(
        call.func.value, ast.Name
    ):
        return None
    return call.func.value.id


def _valid_append_output(body: list[ast.stmt], loop_index: int, loop: ast.For) -> bool:
    calls = [
        call for statement in loop.body if (call := _append_call(statement)) is not None
    ]
    if not calls:
        return True
    if len(calls) != 1 or _append_call(loop.body[-1]) is None:
        return False
    target = _append_target(calls[0])
    if target is None:
        return False
    initializer = next(
        (
            index
            for index in range(loop_index - 1, -1, -1)
            if target in _statement_effects(body[index])[1]
        ),
        None,
    )
    if initializer is None or _empty_list_assignment(body[initializer]) != target:
        return False
    if any(
        target in (_statement_effects(statement)[0] | _statement_effects(statement)[1])
        for statement in body[initializer + 1 : loop_index]
    ):
        return False
    for node in ast.walk(loop):
        if not isinstance(node, ast.Name) or node.id != target:
            continue
        if not isinstance(node.ctx, ast.Load):
            return False
        parent_calls = [
            call for call in calls if cast("ast.Attribute", call.func).value is node
        ]
        if not parent_calls:
            return False
    return True


def _loop_accumulator_initializers(
    body: list[ast.stmt], loop_index: int, loop: ast.For
) -> dict[str, int] | None:
    accumulators = {
        name
        for statement in loop.body
        if (name := _assigned_name(statement)) is not None
        and _accumulated_term(statement, name) is not None
    }
    if not accumulators:
        return {}
    result: dict[str, int] = {}
    for accumulator in accumulators:
        initializer = next(
            (
                index
                for index in range(loop_index - 1, -1, -1)
                if accumulator in _statement_effects(body[index])[1]
            ),
            None,
        )
        if (
            initializer is None
            or _zero_fp32_assignment(body[initializer]) != accumulator
        ):
            return None
        if any(
            accumulator
            in (_statement_effects(statement)[0] | _statement_effects(statement)[1])
            for statement in body[initializer + 1 : loop_index]
        ):
            return None
        result[accumulator] = initializer
    return result


def _pack_fp32_block(
    body: list[ast.stmt],
    *,
    occupied: set[str],
    rmem_names: set[str],
    fp32_rmem: set[str],
    fp32_names: set[str],
    live_out: set[str],
) -> list[ast.stmt]:
    for statement_index, statement in enumerate(body):
        statement_live_out = _live_before_block(
            body[statement_index + 1 :], live_out=live_out
        )
        for field in ("body", "orelse", "finalbody"):
            child = getattr(statement, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(
                    statement,
                    field,
                    _pack_fp32_block(
                        child,
                        occupied=occupied,
                        rmem_names=rmem_names,
                        fp32_rmem=fp32_rmem,
                        fp32_names=fp32_names,
                        live_out=_nested_live_out(
                            statement,
                            field,
                            child,
                            statement_live_out,
                        ),
                    ),
                )

    index = 0
    while index < len(body):
        statement = body[index]
        if not isinstance(statement, ast.For) or _constexpr_loop(statement) is None:
            index += 1
            continue
        initializer_indices = _loop_accumulator_initializers(body, index, statement)
        if initializer_indices is None or not _valid_append_output(
            body, index, statement
        ):
            index += 1
            continue
        accumulators = tuple(
            name
            for candidate in statement.body
            if (name := _assigned_name(candidate)) in initializer_indices
        )
        if not accumulators and not any(
            _append_call(candidate) is not None for candidate in statement.body
        ):
            index += 1
            continue

        trial_occupied = set(occupied)
        accumulator_pairs = {
            accumulator: (
                _fresh_ast_name(trial_occupied, f"_{accumulator}_lo"),
                _fresh_ast_name(trial_occupied, f"_{accumulator}_hi"),
            )
            for accumulator in accumulators
        }
        live_after = _live_before_block(body[index + 1 :], live_out=live_out)
        paired = _pair_constexpr_loop(
            statement,
            accumulators,
            accumulator_pairs,
            occupied=trial_occupied,
            rmem_names=rmem_names,
            fp32_rmem=fp32_rmem,
            fp32_names=fp32_names,
            live_after=live_after,
        )
        if paired is None:
            index += 1
            continue

        occupied.update(trial_occupied)
        for initializer_index in sorted(
            set(initializer_indices.values()), reverse=True
        ):
            accumulator = _zero_fp32_assignment(body[initializer_index])
            assert accumulator is not None
            lo_name, hi_name = accumulator_pairs[accumulator]
            body[initializer_index : initializer_index + 1] = [
                create(
                    ast.Assign,
                    targets=[create(ast.Name, id=lo_name, ctx=ast.Store())],
                    value=expr_from_string("cutlass.Float32(0)"),
                ),
                create(
                    ast.Assign,
                    targets=[create(ast.Name, id=hi_name, ctx=ast.Store())],
                    value=expr_from_string("cutlass.Float32(0)"),
                ),
            ]
            if initializer_index < index:
                index += 1

        reductions = [
            create(
                ast.Assign,
                targets=[create(ast.Name, id=accumulator, ctx=ast.Store())],
                value=expr_from_string(
                    f"cutlass.Float32({accumulator_pairs[accumulator][0]}) + "
                    f"cutlass.Float32({accumulator_pairs[accumulator][1]})"
                ),
            )
            for accumulator in accumulators
        ]
        body[index : index + 1] = [paired, *reductions]
        index += 1 + len(reductions)
    return body


def pack_fp32_constexpr_loops(
    body: list[ast.stmt],
    *,
    fast_math: bool,
    target_device_capability: tuple[int, int] | None,
    float_scalar_names: set[str] | frozenset[str] = frozenset(),
) -> list[ast.stmt]:
    """Pair independent fp32 loop lanes into Blackwell packed instructions.

    Packed f32x2 arithmetic changes evaluation order and is therefore enabled
    only for fast-math kernels.  The FFMA2 path is validated on Blackwell; all
    older or unknown targets retain their scalar generated AST.
    """
    if (
        not fast_math
        or target_device_capability is None
        or target_device_capability < (10, 0)
    ):
        return body
    occupied = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    rmem_names, fp32_rmem = _rmem_storage_names(body)
    fp32_names = _collect_fp32_names(body, fp32_rmem, set(float_scalar_names))
    return [
        ast.fix_missing_locations(statement)
        for statement in _pack_fp32_block(
            body,
            occupied=occupied,
            rmem_names=rmem_names,
            fp32_rmem=fp32_rmem,
            fp32_names=fp32_names,
            live_out=set(),
        )
    ]


_PACKED_LICM_CALLS = {
    *_AST_PURE_CALLS,
    "cutlass.range_constexpr",
    "cute.arch.fma_packed_f32x2",
    "cute.arch.warp_reduction_sum",
}
_PACKED_LICM_UNSUPPORTED_CONTROL = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.ClassDef,
    ast.FunctionDef,
    ast.Global,
    ast.Match,
    ast.Nonlocal,
    ast.Try,
    ast.With,
)


def _fp32_name(expression: ast.expr, name: str) -> bool:
    return (
        isinstance(expression, ast.Call)
        and _ast_call_path(expression.func) == "cutlass.Float32"
        and len(expression.args) == 1
        and not expression.keywords
        and isinstance(expression.args[0], ast.Name)
        and expression.args[0].id == name
    )


def _packed_accumulator_update(
    statement: ast.stmt, lo_accumulator: str, hi_accumulator: str
) -> bool:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Tuple)
    ):
        return False
    target = statement.targets[0]
    if (
        len(target.elts) != 2
        or not all(isinstance(elt, ast.Name) for elt in target.elts)
        or cast("ast.Name", target.elts[0]).id != lo_accumulator
        or cast("ast.Name", target.elts[1]).id != hi_accumulator
        or not isinstance(statement.value, ast.Call)
        or _ast_call_path(statement.value.func) != "cute.arch.fma_packed_f32x2"
        or len(statement.value.args) != 3
        or statement.value.keywords
        or any(
            not isinstance(argument, ast.Tuple) or len(argument.elts) != 2
            for argument in statement.value.args
        )
    ):
        return False
    accumulator_pair = statement.value.args[2]
    return (
        isinstance(accumulator_pair, ast.Tuple)
        and len(accumulator_pair.elts) == 2
        and _fp32_name(accumulator_pair.elts[0], lo_accumulator)
        and _fp32_name(accumulator_pair.elts[1], hi_accumulator)
    )


def _packed_pair_sum(
    statement: ast.stmt, lo_accumulator: str, hi_accumulator: str
) -> str | None:
    result = _assigned_name(statement)
    if result is None:
        return None
    value = cast("ast.Assign", statement).value
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return None
    if not (
        _fp32_name(value.left, lo_accumulator)
        and _fp32_name(value.right, hi_accumulator)
    ):
        return None
    return result


def _packed_licm_call_is_pure(call: ast.Call) -> bool:
    path = _ast_call_path(call.func)
    if path == "cute.math.div":
        return (
            len(call.args) == 2
            and not call.keywords
            and isinstance(call.args[1], ast.Constant)
            and isinstance(call.args[1].value, (int, float))
            and not isinstance(call.args[1].value, bool)
            and call.args[1].value != 0
        )
    if path == "cute.arch.load":
        return False
    if path in _PACKED_LICM_CALLS or path in _FP32_MATH_CALLS:
        return True
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "bitcast"
        and isinstance(call.func.value, ast.Call)
    ):
        return False
    return _ast_call_path(call.func.value.func) in _AST_PURE_CALLS


def _packed_licm_statement_is_pure(statement: ast.stmt, rmem_names: set[str]) -> bool:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], (ast.Name, ast.Tuple)
        ):
            return False
        if isinstance(statement.targets[0], ast.Tuple) and not all(
            isinstance(elt, ast.Name) for elt in statement.targets[0].elts
        ):
            return False
    elif isinstance(statement, ast.For):
        if _constexpr_loop(statement) is None:
            return False
        if not all(
            _packed_licm_statement_is_pure(child, rmem_names)
            for child in statement.body
        ):
            return False
    else:
        return False

    parents = {
        id(child): parent
        for parent in ast.walk(statement)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(statement):
        if isinstance(
            node,
            (
                ast.Await,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.ListComp,
                ast.NamedExpr,
                ast.SetComp,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            return False
        if isinstance(node, ast.Call) and not _packed_licm_call_is_pure(node):
            return False
        if isinstance(node, ast.Attribute):
            call_path_root: ast.AST = node
            while (
                isinstance(parent := parents.get(id(call_path_root)), ast.Attribute)
                and parent.value is call_path_root
            ):
                call_path_root = parent
            parent = parents.get(id(call_path_root))
            if not (isinstance(parent, ast.Call) and parent.func is call_path_root):
                return False
        if isinstance(node, (ast.BoolOp, ast.Compare, ast.IfExp)):
            return False
        if isinstance(node, ast.UnaryOp) and not isinstance(
            node.op, (ast.Invert, ast.UAdd, ast.USub)
        ):
            return False
        if isinstance(node, ast.Subscript):
            if (
                not isinstance(node.ctx, ast.Load)
                or not isinstance(node.value, ast.Name)
                or node.value.id not in rmem_names
            ):
                return False
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Mult, ast.Sub)):
                continue
            if not (
                isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
                and not isinstance(node.right.value, bool)
                and node.right.value != 0
            ):
                return False
    return True


def _dead_packed_assignment_is_discardable(statement: ast.stmt) -> bool:
    """Return whether deleting an unused generated scalar assignment is safe."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return False

    def safe_expression(expression: ast.expr) -> bool:
        if isinstance(expression, (ast.Constant, ast.Name)):
            return True
        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, (ast.Invert, ast.UAdd, ast.USub)
        ):
            return safe_expression(expression.operand)
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op, (ast.Add, ast.Mult, ast.Sub)
        ):
            return safe_expression(expression.left) and safe_expression(
                expression.right
            )
        if not isinstance(expression, ast.Call) or expression.keywords:
            return False
        path = _ast_call_path(expression.func)
        return path in (_AST_PURE_CALLS - {"cute.arch.load"}) and all(
            safe_expression(argument) for argument in expression.args
        )

    return safe_expression(statement.value)


def _packed_block_effects(
    body: list[ast.stmt], initially_defined: set[str] | None = None
) -> tuple[set[str], set[str]]:
    live: set[str] = set()
    written = set() if initially_defined is None else set(initially_defined)
    all_writes: set[str] = set()
    for statement in body:
        if isinstance(statement, ast.Assign):
            reads = _names_read(statement.value)
            for target in statement.targets:
                if not isinstance(target, (ast.Name, ast.Tuple)):
                    reads.update(_names_read(target))
            writes = {
                node.id
                for target in statement.targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
        elif isinstance(statement, ast.For) and isinstance(statement.target, ast.Name):
            nested_live, nested_writes = _packed_block_effects(
                statement.body, {statement.target.id}
            )
            reads = _names_read(statement.iter) | nested_live
            writes = {statement.target.id} | nested_writes
        else:
            reads = _names_read(statement)
            writes = _names_written(statement)
        live.update(reads - written)
        written.update(writes)
        all_writes.update(writes)
    return live, all_writes


def _store_name_counts(nodes: Iterable[ast.AST]) -> dict[str, int]:
    result: dict[str, int] = {}
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                result[node.id] = result.get(node.id, 0) + 1
    return result


def _load_name_counts(nodes: Iterable[ast.AST]) -> dict[str, int]:
    result: dict[str, int] = {}
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                result[node.id] = result.get(node.id, 0) + 1
    return result


def _canonical_ast_name(name: str, rename_groups: dict[str, str]) -> str:
    seen: set[str] = set()
    while name in rename_groups and name not in seen:
        seen.add(name)
        name = rename_groups[name]
    return name


def _canonical_counts(
    counts: dict[str, int], rename_groups: dict[str, str]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, count in counts.items():
        canonical = _canonical_ast_name(name, rename_groups)
        result[canonical] = result.get(canonical, 0) + count
    return result


def _packed_inner_dependency_slice(
    loop: ast.For,
    lo_accumulator: str,
    hi_accumulator: str,
    rmem_names: set[str],
) -> list[ast.stmt] | None:
    if not loop.body or not _packed_accumulator_update(
        loop.body[-1], lo_accumulator, hi_accumulator
    ):
        return None
    definitions: dict[str, ast.expr] = {}
    definition_indices: dict[str, int] = {}
    for index, statement in enumerate(loop.body[:-1]):
        name = _assigned_name(statement)
        if (
            name is None
            or name in definitions
            or name in {lo_accumulator, hi_accumulator}
        ):
            return None
        assert isinstance(statement, ast.Assign)
        definitions[name] = statement.value
        definition_indices[name] = index

    local_names = set(definitions)
    seen = {cast("ast.Name", loop.target).id}
    for statement in loop.body[:-1]:
        name = _assigned_name(statement)
        assert name is not None
        if _names_read(statement) & (local_names - seen):
            return None
        seen.add(name)

    needed = _dependency_closure(
        _names_read(cast("ast.Assign", loop.body[-1]).value)
        - {lo_accumulator, hi_accumulator},
        definitions,
    )
    selected_indices = {
        definition_indices[name] for name in needed if name in definition_indices
    }
    for index, statement in enumerate(loop.body[:-1]):
        if index not in selected_indices and not _dead_packed_assignment_is_discardable(
            statement
        ):
            return None
    selected = [
        statement
        for index, statement in enumerate(loop.body[:-1])
        if index in selected_indices
    ]
    selected.append(loop.body[-1])
    if not all(
        _packed_licm_statement_is_pure(statement, rmem_names) for statement in selected
    ):
        return None
    return selected


def _try_hoist_one_packed_reduction(
    loop: ast.For,
    *,
    row_extent: int,
    rmem_names: set[str],
    rename_groups: dict[str, str],
    scope_stores: dict[str, int],
    scope_loads: dict[str, int],
    live_after_scope: set[str],
) -> tuple[list[ast.stmt], set[str], str] | None:
    body = loop.body
    if any(
        isinstance(
            node,
            (
                ast.Break,
                ast.Continue,
                ast.Delete,
                ast.Raise,
                ast.Return,
                *_PACKED_LICM_UNSUPPORTED_CONTROL,
            ),
        )
        or type(node).__name__ == "TryStar"
        for node in ast.walk(loop)
    ):
        return None
    for index in range(len(body) - 4):
        lo_accumulator = _zero_fp32_assignment(body[index])
        hi_accumulator = _zero_fp32_assignment(body[index + 1])
        inner_match = _constexpr_loop(body[index + 2])
        if (
            lo_accumulator is None
            or hi_accumulator is None
            or lo_accumulator == hi_accumulator
            or inner_match is None
        ):
            continue
        inner_loop, _inner_lane, _inner_extent = inner_match
        selected_inner = _packed_inner_dependency_slice(
            inner_loop, lo_accumulator, hi_accumulator, rmem_names
        )
        if selected_inner is None:
            continue
        combined = _packed_pair_sum(body[index + 3], lo_accumulator, hi_accumulator)
        reduction = _warp_reduction_assignment(body[index + 4])
        if combined is None or reduction is None or reduction[1] != combined:
            continue
        result_name = reduction[0]
        original_pattern = body[index : index + 5]
        call_paths = [
            _ast_call_path(node.func)
            for statement in original_pattern
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ]
        if (
            call_paths.count("cute.arch.fma_packed_f32x2") != 1
            or call_paths.count("cute.arch.warp_reduction_sum") != 1
        ):
            continue
        if not all(
            _packed_licm_statement_is_pure(statement, rmem_names)
            for statement in original_pattern
        ):
            continue

        inner_copy = cast("ast.For", _clone_extended_ast(inner_loop))
        inner_copy.body = [
            cast("ast.stmt", _clone_extended_ast(statement))
            for statement in selected_inner
        ]
        hoisted = [
            body[index],
            body[index + 1],
            inner_copy,
            body[index + 3],
            body[index + 4],
        ]
        external_reads, _hoisted_writes = _packed_block_effects(hoisted)
        external_reads -= _AST_GLOBAL_NAMES
        lane_name = cast("ast.Name", loop.target).id
        if lane_name in external_reads:
            continue

        original_stores = _store_name_counts(original_pattern)
        canonical_original_stores = _canonical_counts(original_stores, rename_groups)
        if len(canonical_original_stores) != len(original_stores):
            continue
        if any(
            scope_stores.get(name) != count
            for name, count in canonical_original_stores.items()
        ):
            continue

        internal_names = set(original_stores) - {result_name}
        original_loads = _canonical_counts(
            _load_name_counts(original_pattern), rename_groups
        )
        if any(
            scope_loads.get(_canonical_ast_name(name, rename_groups), 0)
            != original_loads.get(_canonical_ast_name(name, rename_groups), 0)
            for name in internal_names
        ):
            continue
        canonical_live_after = {
            _canonical_ast_name(name, rename_groups) for name in live_after_scope
        }
        if {
            _canonical_ast_name(name, rename_groups) for name in internal_names
        } & canonical_live_after:
            continue

        row_prefix = body[:index]
        canonical_result = _canonical_ast_name(result_name, rename_groups)
        if canonical_result in {
            _canonical_ast_name(name, rename_groups)
            for name in _names_read(ast.Module(body=row_prefix, type_ignores=[]))
        }:
            continue
        row_suffix = body[index + 5 :]
        if canonical_result not in {
            _canonical_ast_name(name, rename_groups)
            for name in _names_read(ast.Module(body=row_suffix, type_ignores=[]))
        }:
            continue

        unselected = [*row_prefix, *row_suffix]
        unselected_writes = {
            _canonical_ast_name(name, rename_groups)
            for name in _names_written(ast.Module(body=unselected, type_ignores=[]))
        }
        unselected_mutations = {
            _canonical_ast_name(name, rename_groups)
            for name in _mutation_roots(ast.Module(body=unselected, type_ignores=[]))
        }
        canonical_external_reads = {
            _canonical_ast_name(name, rename_groups) for name in external_reads
        }
        if _canonical_ast_name(lane_name, rename_groups) in canonical_external_reads:
            continue
        if canonical_external_reads & (
            set(canonical_original_stores) | unselected_writes | unselected_mutations
        ):
            continue
        if not repeated_work_is_profitable(
            hoisted,
            repeat_extent=row_extent,
            live_scalars=1,
        ):
            continue

        loop.body = unselected
        return hoisted, canonical_external_reads, canonical_result
    return None


def _earliest_packed_hoist_index(
    prefix: list[ast.stmt],
    *,
    canonical_dependencies: set[str],
    canonical_result: str,
    rename_groups: dict[str, str],
) -> int:
    """Place a reduction after its last dependency or control-flow barrier."""
    insertion_index = 0
    for index, statement in enumerate(prefix):
        written_or_mutated = {
            _canonical_ast_name(name, rename_groups)
            for name in _names_written(statement) | _mutation_roots(statement)
        }
        read = {
            _canonical_ast_name(name, rename_groups) for name in _names_read(statement)
        }
        has_control_exit = any(
            isinstance(
                node,
                (
                    ast.Break,
                    ast.Continue,
                    ast.Delete,
                    ast.Raise,
                    ast.Return,
                    *_PACKED_LICM_UNSUPPORTED_CONTROL,
                ),
            )
            or type(node).__name__ == "TryStar"
            for node in ast.walk(statement)
        )
        if (
            canonical_dependencies & written_or_mutated
            or canonical_result in read
            or has_control_exit
        ):
            insertion_index = index + 1
    return insertion_index


def _hoist_packed_reduction_block(
    body: list[ast.stmt],
    *,
    rmem_names: set[str],
    rename_groups: dict[str, str],
    live_after: set[str],
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    scope_stores = _canonical_counts(_store_name_counts(body), rename_groups)
    scope_loads = _canonical_counts(_load_name_counts(body), rename_groups)
    for statement_index, statement in enumerate(body):
        statement_live_after = live_after | _names_read(
            ast.Module(body=body[statement_index + 1 :], type_ignores=[])
        )
        if isinstance(statement, (ast.For, ast.If, ast.While)):
            child_fields = ("body", "orelse")
        else:
            child_fields = ()
        for field in child_fields:
            child = getattr(statement, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(
                    statement,
                    field,
                    _hoist_packed_reduction_block(
                        child,
                        rmem_names=rmem_names,
                        rename_groups=rename_groups,
                        live_after=_nested_live_out(
                            statement,
                            field,
                            child,
                            statement_live_after,
                        ),
                    ),
                )
        lane_name = getattr(statement, HELION_LANE_LOOP_VAR_ATTR, None)
        outer_match = _static_repeated_loop(statement)
        if (
            not isinstance(statement, ast.For)
            or not isinstance(lane_name, str)
            or outer_match is None
            or outer_match[1] != lane_name
            or outer_match[2] <= 1
        ):
            result.append(statement)
            continue
        last_hoist_end = 0
        while replacement := _try_hoist_one_packed_reduction(
            statement,
            row_extent=outer_match[2],
            rmem_names=rmem_names,
            rename_groups=rename_groups,
            scope_stores=scope_stores,
            scope_loads=scope_loads,
            live_after_scope=live_after,
        ):
            hoisted, dependencies, reduction_result = replacement
            insertion_index = max(
                last_hoist_end,
                _earliest_packed_hoist_index(
                    result,
                    canonical_dependencies=dependencies,
                    canonical_result=reduction_result,
                    rename_groups=rename_groups,
                ),
            )
            result[insertion_index:insertion_index] = hoisted
            last_hoist_end = insertion_index + len(hoisted)
        result.append(statement)
    return result


def hoist_packed_factored_reductions(
    body: list[ast.stmt],
    *,
    fast_math: bool,
    rename_groups: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """Hoist pure row-invariant reductions exposed by fp32 lane packing.

    Packing replaces a scalar accumulator with a two-name tuple update, which
    deliberately falls outside the earlier general LICM.  Recognize only the
    generated zero/constexpr-loop/pair-sum/warp-reduction sequence and retain
    the exact ordered dependency slice feeding its packed FMA.
    """
    if not fast_math:
        return body
    renames = rename_groups or {}
    rmem_names, _fp32_rmem = _rmem_storage_names(body)
    return [
        ast.fix_missing_locations(statement)
        for statement in _hoist_packed_reduction_block(
            body,
            rmem_names=rmem_names,
            rename_groups=renames,
            live_after=set(),
        )
    ]


def _try_fuse_adjacent_reduction_loops(loop: ast.For, rmem_names: set[str]) -> bool:
    """Fuse two independent neighboring constexpr reduction sweeps."""
    body = loop.body
    for first_index in range(len(body) - 4):
        first_loop_match = _constexpr_loop(body[first_index])
        if first_loop_match is None:
            continue
        first_loop, first_lane, first_extent = first_loop_match
        first_reduction = _warp_reduction_assignment(body[first_index + 1])
        if first_reduction is None:
            continue
        _first_result, first_acc, _first_call = first_reduction
        init_candidates = [
            index
            for index in range(first_index)
            if _zero_fp32_assignment(body[index]) == first_acc
        ]
        if not init_candidates:
            continue
        first_init_index = init_candidates[-1]
        if any(
            first_acc in _statement_effects(statement)[1]
            for statement in body[first_init_index + 1 : first_index]
        ):
            continue
        second_acc = _zero_fp32_assignment(body[first_index + 2])
        second_loop_match = _constexpr_loop(body[first_index + 3])
        if second_acc is None or second_loop_match is None:
            continue
        second_loop, second_lane, second_extent = second_loop_match
        second_reduction = _reduction_result(body[first_index + 4], second_acc)
        if (
            first_extent != second_extent
            or second_reduction is None
            or not second_reduction[0].startswith("_helion_factored_base_sum")
            or first_acc in _statement_effects(second_loop)[0]
        ):
            continue
        if not all(
            isinstance(statement, ast.Assign)
            for statement in (*first_loop.body, *second_loop.body)
        ):
            continue
        first_live = _live_before_block(first_loop.body, {first_lane, first_acc})
        first_writes = _names_written(first_loop)
        second_live = _live_before_block(second_loop.body, {second_lane, second_acc})
        second_writes = _names_written(second_loop)
        if (first_live & first_writes) - {first_acc} or (
            second_live & second_writes
        ) - {second_acc}:
            continue
        if first_writes & second_live or second_writes & first_live:
            continue
        first_reduction_reads, first_reduction_writes = _statement_effects(
            body[first_index + 1]
        )
        if (
            second_writes & first_reduction_reads
            or first_reduction_writes & second_live
            or first_reduction_writes & second_writes
            or second_acc
            in (
                first_live
                | first_writes
                | first_reduction_reads
                | first_reduction_writes
            )
        ):
            continue
        if (
            not _safe_lane_local_stores(first_loop, first_lane, rmem_names)
            or _mutation_roots(second_loop)
            or _has_effectful_call(first_loop.body)
            or _has_effectful_call(second_loop.body)
        ):
            continue

        second_loop_copy = cast("ast.For", _clone_extended_ast(second_loop))
        if second_lane != first_lane:
            rename = _RenameAstName(second_lane, first_lane)
            for statement in second_loop_copy.body:
                rename.visit(statement)
        stores = _subscript_accesses(first_loop.body, ast.Store)
        loads = _subscript_accesses(second_loop_copy.body, ast.Load)
        if any(
            root in loads and not loads[root] <= slices
            for root, slices in stores.items()
        ):
            continue

        first_loop.body = _deduplicate_fused_body(
            body[:first_index], first_loop.body, second_loop_copy.body, rmem_names
        )
        loop.body = [
            *body[: first_init_index + 1],
            *body[first_init_index + 1 : first_index],
            body[first_index + 2],
            first_loop,
            body[first_index + 1],
            body[first_index + 4],
            *body[first_index + 5 :],
        ]
        return True
    return False


def _hoist_one_invariant_reduction(
    loop: ast.For,
    occupied: set[str],
    rmem_names: set[str],
    live_after: set[str],
) -> list[ast.stmt] | None:
    if not isinstance(loop.target, ast.Name):
        return None
    body = loop.body
    outer_lane = loop.target.id
    outer_writes = _names_written(loop)
    outer_mutations = _mutation_roots(loop)
    for index in range(len(body) - 2):
        accumulator = _zero_fp32_assignment(body[index])
        inner_match = _constexpr_loop(body[index + 1])
        if accumulator is None or inner_match is None:
            continue
        inner_loop, inner_lane, extent = inner_match
        reduced = _reduction_result(body[index + 2], accumulator)
        if reduced is None or not inner_loop.body:
            continue
        result_name, _reduction_call = reduced
        if not result_name.startswith("_helion_factored_dot_sum"):
            continue
        term = _accumulated_term(inner_loop.body[-1], accumulator)
        if term is None:
            continue
        definitions: dict[str, ast.expr] = {}
        valid = True
        for statement in inner_loop.body[:-1]:
            name = _assigned_name(statement)
            if name is None:
                valid = False
                break
            assert isinstance(statement, ast.Assign)
            if name in definitions or name == accumulator:
                valid = False
                break
            definitions[name] = statement.value
        if not valid or _has_effectful_call(inner_loop.body[:-1]):
            continue
        seen: set[str] = set()
        for statement in inner_loop.body[:-1]:
            name = _assigned_name(statement)
            assert name is not None
            if _names_read(statement) & (set(definitions) - seen):
                valid = False
                break
            seen.add(name)
        if not valid:
            continue
        term_dependencies = _dependency_closure(_names_read(term), definitions)
        if set(definitions) - term_dependencies:
            continue
        new_lane = _fresh_ast_name(occupied, "_factored_lane")
        try:
            normalized = cast(
                "ast.expr",
                _InlineInnerExpression(definitions, inner_lane, new_lane).visit(
                    expr_from_string(ast.unparse(term))
                ),
            )
        except ValueError:
            continue
        candidate_writes = (
            _names_written(body[index])
            | _names_written(body[index + 1])
            | _names_written(body[index + 2])
        )
        if not _safe_invariant_expression(
            normalized,
            outer_lane=outer_lane,
            outer_writes=outer_writes - candidate_writes,
            outer_mutations=outer_mutations,
            rmem_names=rmem_names,
        ):
            continue
        deleted_names = _names_written(inner_loop) | {inner_lane, accumulator}
        outside_slice = [*body[:index], *body[index + 3 :]]
        if deleted_names & (
            live_after
            | {name for statement in outside_slice for name in _names_read(statement)}
        ):
            continue
        new_accumulator = _fresh_ast_name(occupied, "_factored_acc")
        init = create(
            ast.Assign,
            targets=[create(ast.Name, id=new_accumulator, ctx=ast.Store())],
            value=expr_from_string("cutlass.Float32(0)"),
        )
        accumulation = create(
            ast.Assign,
            targets=[create(ast.Name, id=new_accumulator, ctx=ast.Store())],
            value=create(
                ast.BinOp,
                left=create(ast.Name, id=new_accumulator, ctx=ast.Load()),
                op=create(ast.Add),
                right=normalized,
            ),
        )
        precompute_loop = create(
            ast.For,
            target=create(ast.Name, id=new_lane, ctx=ast.Store()),
            iter=expr_from_string(f"cutlass.range_constexpr({extent})"),
            body=[accumulation],
            orelse=[],
            type_comment=None,
        )
        result_assignment = create(
            ast.Assign,
            targets=[create(ast.Name, id=result_name, ctx=ast.Store())],
            value=expr_from_string(
                ast.unparse(cast("ast.Assign", body[index + 2]).value)
            ),
        )
        replacement_call = next(
            child
            for child in ast.walk(result_assignment.value)
            if isinstance(child, ast.Call)
            and _ast_call_path(child.func) == "cute.arch.warp_reduction_sum"
        )
        replacement_call.args[0] = create(ast.Name, id=new_accumulator, ctx=ast.Load())
        del body[index : index + 3]
        return [init, precompute_loop, result_assignment, loop]
    return None


def _hoist_factored_block(
    body: list[ast.stmt],
    occupied: set[str],
    rmem_names: set[str],
    live_out: set[str],
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement_index, statement in enumerate(body):
        statement_live_out = _live_before_block(
            body[statement_index + 1 :], live_out=live_out
        )
        for field in ("body", "orelse", "finalbody"):
            child = getattr(statement, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(
                    statement,
                    field,
                    _hoist_factored_block(
                        child,
                        occupied,
                        rmem_names,
                        _nested_live_out(
                            statement,
                            field,
                            child,
                            statement_live_out,
                        ),
                    ),
                )
        if isinstance(statement, ast.For) and isinstance(
            getattr(statement, HELION_LANE_LOOP_VAR_ATTR, None), str
        ):
            replacement = _hoist_one_invariant_reduction(
                statement,
                occupied,
                rmem_names,
                statement_live_out,
            )
            if replacement is not None:
                remaining_loop = cast("ast.For", replacement[-1])
                _try_fuse_adjacent_reduction_loops(remaining_loop, rmem_names)
                result.extend(replacement)
                continue
            _try_fuse_adjacent_reduction_loops(statement, rmem_names)
        result.append(statement)
    return result


def hoist_factored_reductions(
    body: list[ast.stmt],
    *,
    fast_math: bool,
) -> list[ast.stmt]:
    """Hoist standalone reductions exposed by :func:`factor_affine_reductions`."""
    if not fast_math:
        return body
    occupied = {
        node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    rmem_names, _fp32_rmem = _rmem_storage_names(body)
    return [
        ast.fix_missing_locations(stmt)
        for stmt in _hoist_factored_block(body, occupied, rmem_names, set())
    ]
