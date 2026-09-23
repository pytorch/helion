"""Expose independent offset-delimited matrix rows to the CuTe scheduler.

The initial normalization accepts a direct, zero-seeded contraction whose only
effect is one store to a fresh matrix. The offset loads and accumulator's exact
conversion are retained. Group-specific M bounds become explicit load/store
masks; group, M and N are then available to the existing grouped scheduler.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import language as hl
from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .full_slice_matmul import _bound_names
from .full_slice_matmul import _global_reference
from .full_slice_matmul import _global_value
from .full_slice_matmul import _input_binding_is_stable
from .full_slice_matmul import _pure_host_prelude

if TYPE_CHECKING:
    from ..host_function import HostFunction


def _name(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant) and type(node.value) is int and node.value == 0
    )


def _same(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _assignment(node: ast.AST) -> tuple[str, ast.expr] | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id, node.value
    return None


def _matrix_subscript(node: ast.AST) -> tuple[str, ast.expr, ast.expr] | None:
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
    ):
        return node.value.id, node.slice.elts[0], node.slice.elts[1]
    return None


def _metadata_aliases(host: HostFunction) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for statement in host.body:
        if isinstance(statement, ast.For):
            break
        if assigned := _assignment(statement):
            result[assigned[0]] = assigned[1]
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], (ast.Tuple, ast.List))
            and isinstance(statement.value, ast.Attribute)
            and statement.value.attr == "shape"
        ):
            for dim, target in enumerate(statement.targets[0].elts):
                if isinstance(target, ast.Name):
                    result[target.id] = ast.Subscript(
                        value=statement.value,
                        slice=ast.Constant(dim),
                        ctx=ast.Load(),
                    )
    return result


def _resolve(node: ast.expr, aliases: dict[str, ast.expr]) -> ast.expr:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in aliases and node.id not in seen:
        seen.add(node.id)
        node = aliases[node.id]
    return node


def _dimension(
    node: ast.expr, tensor: str, axis: int, aliases: dict[str, ast.expr]
) -> bool:
    node = _resolve(node, aliases)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _name(node.func.value, tensor)
        and node.func.attr == "size"
        and len(node.args) == 1
        and not node.keywords
    ):
        index = node.args[0]
    elif (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and _name(node.value.value, tensor)
        and node.value.attr == "shape"
    ):
        index = node.slice
    else:
        return False
    return isinstance(index, ast.Constant) and index.value == axis


@dataclasses.dataclass(frozen=True)
class _SegmentedMatmul:
    root: ast.For
    group: str
    row_tile: str
    column_tile: str
    contraction_tile: str
    start: str
    extent: str
    lhs: str
    rhs: str
    offsets: str
    output: str
    prefix: tuple[ast.stmt, ...]
    initialize: ast.stmt
    reduction: ast.For
    lhs_load: ast.Subscript
    store: ast.Assign


def _match(
    host: HostFunction, root: ast.For, local_names: set[str]
) -> _SegmentedMatmul | None:
    if (
        not isinstance(root.target, ast.Name)
        or not isinstance(root.iter, ast.Call)
        or _global_value(root.iter.func, host, local_names) is not hl.grid
        or len(root.iter.args) != 1
        or root.iter.keywords
        or root.orelse
        or len(root.body) != 4
        or not isinstance(condition := root.body[-1], ast.If)
        or condition.orelse
        or len(condition.body) != 1
        or not isinstance(rows := condition.body[0], ast.For)
        or rows.orelse
        or not isinstance(rows.target, (ast.Tuple, ast.List))
        or len(rows.target.elts) != 2
        or not all(isinstance(item, ast.Name) for item in rows.target.elts)
        or not isinstance(rows.iter, ast.Call)
        or _global_value(rows.iter.func, host, local_names) is not hl.tile
        or rows.iter.keywords
        or len(rows.iter.args) != 1
        or not isinstance(row_extents := rows.iter.args[0], (ast.List, ast.Tuple))
        or len(row_extents.elts) != 2
        or not isinstance(row_extent := row_extents.elts[0], ast.Name)
        or len(rows.body) != 3
    ):
        return None
    group = root.target.id
    row, column = (cast("ast.Name", item).id for item in rows.target.elts)
    extent = row_extent.id
    if not (
        isinstance(predicate := condition.test, ast.Compare)
        and _name(predicate.left, extent)
        and len(predicate.ops) == len(predicate.comparators) == 1
        and isinstance(predicate.ops[0], (ast.NotEq, ast.Gt))
        and _zero(predicate.comparators[0])
    ):
        return None
    assignments = [_assignment(statement) for statement in root.body[:-1]]
    if any(item is None for item in assignments):
        return None
    prefix = dict(cast("list[tuple[str, ast.expr]]", assignments))
    if len(prefix) != 3:
        return None
    difference = prefix.get(extent)
    if not (
        isinstance(difference, ast.BinOp)
        and isinstance(difference.op, ast.Sub)
        and isinstance(difference.left, ast.Name)
        and isinstance(difference.right, ast.Name)
    ):
        return None
    start = difference.right.id
    start_load = prefix.get(start)
    end_load = prefix.get(difference.left.id)
    if not (
        isinstance(start_load, ast.Subscript)
        and isinstance(start_load.value, ast.Name)
        and _name(start_load.slice, group)
        and isinstance(end_load, ast.Subscript)
        and _same(end_load.value, start_load.value)
        and isinstance(next_group := end_load.slice, ast.BinOp)
        and isinstance(next_group.op, ast.Add)
        and _name(next_group.left, group)
        and isinstance(next_group.right, ast.Constant)
        and type(next_group.right.value) is int
        and next_group.right.value == 1
    ):
        return None
    offsets = start_load.value.id
    offset_tensor = host.params.arguments.get(offsets)
    aliases = _metadata_aliases(host)
    groups = _resolve(root.iter.args[0], aliases)
    if not (
        isinstance(offset_tensor, torch.Tensor)
        and offset_tensor.ndim == 1
        and offset_tensor.dtype in (torch.int32, torch.int64)
        and isinstance(groups, ast.BinOp)
        and isinstance(groups.op, ast.Sub)
        and _dimension(groups.left, offsets, 0, aliases)
        and isinstance(groups.right, ast.Constant)
        and type(groups.right.value) is int
        and groups.right.value == 1
    ):
        return None
    initialize, reduction, store = rows.body
    initialized = _assignment(initialize)
    if not (
        initialized is not None
        and isinstance(zeros := initialized[1], ast.Call)
        and _global_value(zeros.func, host, local_names) is hl.zeros
        and len(zeros.args) == 1
        and isinstance(zeros.args[0], (ast.List, ast.Tuple))
        and len(zeros.args[0].elts) == 2
        and _name(zeros.args[0].elts[0], row)
        and _name(zeros.args[0].elts[1], column)
        and len(zeros.keywords) == 1
        and zeros.keywords[0].arg == "dtype"
        and _global_value(zeros.keywords[0].value, host, local_names) is torch.float32
        and isinstance(reduction, ast.For)
        and isinstance(reduction.target, ast.Name)
        and isinstance(reduction.iter, ast.Call)
        and _global_value(reduction.iter.func, host, local_names) is hl.tile
        and len(reduction.iter.args) == 1
        and not reduction.orelse
        and isinstance(store, ast.Assign)
        and len(store.targets) == 1
    ):
        return None
    acc = initialized[0]
    contraction = reduction.target.id
    inner = [_assignment(statement) for statement in reduction.body]
    if any(item is None for item in inner) or not inner:
        return None
    inner_defs = dict(cast("list[tuple[str, ast.expr]]", inner))
    if len(inner_defs) != len(inner):
        return None
    update = inner_defs.get(acc)
    if not (
        inner[-1] is not None
        and inner[-1][0] == acc
        and isinstance(update, ast.Call)
        and _global_value(update.func, host, local_names) is torch.addmm
        and len(update.args) == 3
        and not update.keywords
        and _name(update.args[0], acc)
    ):
        return None
    lhs_expr = _resolve(update.args[1], inner_defs)
    rhs_expr = _resolve(update.args[2], inner_defs)
    lhs_access = _matrix_subscript(lhs_expr)
    rhs_access = _matrix_subscript(rhs_expr)
    output_access = _matrix_subscript(store.targets[0])
    if lhs_access is None or rhs_access is None or output_access is None:
        return None
    lhs, row_index, k_index = lhs_access
    rhs, rhs_k_index, column_index = rhs_access
    output, output_row, output_column = output_access
    if not (
        isinstance(row_index, ast.BinOp)
        and isinstance(row_index.op, ast.Add)
        and _name(row_index.left, start)
        and isinstance(row_index.right, ast.Attribute)
        and _name(row_index.right.value, row)
        and row_index.right.attr == "index"
        and _name(k_index, contraction)
        and _name(rhs_k_index, contraction)
        and _name(column_index, column)
        and _same(output_row, row_index)
        and _name(output_column, column)
        and isinstance(convert := store.value, ast.Call)
        and isinstance(convert.func, ast.Attribute)
        and _name(convert.func.value, acc)
        and convert.func.attr == "to"
        and len(convert.args) == 1
        and not convert.keywords
        and isinstance(convert.args[0], ast.Attribute)
        and _name(convert.args[0].value, output)
        and convert.args[0].attr == "dtype"
    ):
        return None
    allowed_inner = {acc}
    for operand in update.args[1:]:
        if isinstance(operand, ast.Name):
            allowed_inner.add(operand.id)
    if set(inner_defs) != allowed_inner:
        return None
    if allowed_inner & {group, row, column, contraction, *prefix}:
        return None
    prefix_names = list(prefix)
    if any(
        prefix_names.index(name) >= prefix_names.index(extent)
        for name in (start, difference.left.id)
    ):
        return None
    lhs_tensor = host.params.arguments.get(lhs)
    rhs_tensor = host.params.arguments.get(rhs)
    if not (
        isinstance(lhs_tensor, torch.Tensor)
        and isinstance(rhs_tensor, torch.Tensor)
        and lhs_tensor.ndim == rhs_tensor.ndim == 2
        and lhs_tensor.dtype == rhs_tensor.dtype
        and lhs_tensor.dtype in (torch.float16, torch.bfloat16)
        and _dimension(row_extents.elts[1], rhs, 1, aliases)
        and _dimension(reduction.iter.args[0], lhs, 1, aliases)
        and all(
            _input_binding_is_stable(name, host, local_names)
            for name in (lhs, rhs, offsets)
        )
    ):
        return None
    allocation = aliases.get(output)
    if not (
        isinstance(allocation, ast.Call)
        and _global_value(allocation.func, host, local_names)
        in (torch.empty, torch.zeros)
        and all(keyword.arg in ("dtype", "device") for keyword in allocation.keywords)
    ):
        return None
    shape = allocation.args
    if len(shape) == 1 and isinstance(shape[0], (ast.Tuple, ast.List)):
        shape = shape[0].elts
    if not (
        len(shape) == 2
        and _dimension(shape[0], lhs, 0, aliases)
        and _dimension(shape[1], rhs, 1, aliases)
    ):
        return None
    return _SegmentedMatmul(
        root,
        group,
        row,
        column,
        contraction,
        start,
        extent,
        lhs,
        rhs,
        offsets,
        output,
        tuple(root.body[:-1]),
        initialize,
        reduction,
        cast("ast.Subscript", lhs_expr),
        store,
    )


def normalize_segmented_matmuls(host: HostFunction) -> bool:
    env = CompileEnvironment.current()
    if env.backend_name != "cute" or not env.settings.cute_segmented_matmul_tiling:
        return False
    local_names = _bound_names(host)
    if not _pure_host_prelude(host, local_names):
        return False
    roots = [statement for statement in host.body if isinstance(statement, ast.For)]
    if len(roots) != 1:
        return False
    root = roots[0]
    matched = _match(host, root, local_names)
    if matched is None:
        return False
    following = host.body[host.body.index(root) + 1 :]
    if not (
        len(following) == 1
        and isinstance(following[0], ast.Return)
        and following[0].value is not None
        and _name(following[0].value, matched.output)
    ):
        return False
    assert isinstance(root, ExtendedAST)
    with root:
        tile = _global_reference(hl.tile, host, local_names)
        load = _global_reference(hl.load, host, local_names)
        store = _global_reference(hl.store, host, local_names)
        clamp_min = _global_reference(torch.clamp_min, host, local_names)
        clamp_max = _global_reference(torch.clamp_max, host, local_names)
        if (
            tile is None
            or load is None
            or store is None
            or clamp_min is None
            or clamp_max is None
        ):
            return False
        used = local_names | set(host.fn.__globals__)

        def fresh(name: str) -> str:
            while name in used:
                name += "_"
            used.add(name)
            return name

        group_tile = fresh("_helion_segment_group")
        clipped_start = fresh("_helion_segment_start")
        clipped_extent = fresh("_helion_segment_extent")
        row_index = fresh("_helion_segment_row")
        valid = fresh("_helion_segment_valid")
        replacement = statement_from_string(
            f"for {group_tile}, {matched.row_tile}, {matched.column_tile} in "
            f"{{tile}}([{matched.offsets}.size(0) - 1, "
            f"{matched.lhs}.size(0), {matched.rhs}.size(1)], "
            "block_size=[1, None, None]):\n"
            f"    {matched.group} = {group_tile}.index.sum()\n"
            "    pass",
            tile=tile,
        )
        assert isinstance(replacement, ast.For)
        start_statement = statement_from_string(
            f"{clipped_start} = {{clamp_min}}({matched.start}, 0)",
            clamp_min=clamp_min,
        )
        # Drop the negative-index prefix before limiting local M to A.size(0).
        # Clamp the original signed extent first, preserving its integer
        # subtraction and empty-range semantics even when that subtraction
        # wraps. The remaining addition combines nonnegative and nonpositive
        # values, so it cannot overflow.
        extent_statement = statement_from_string(
            f"{clipped_extent} = {{clamp_min}}("
            f"{{clamp_min}}({matched.extent}, 0) + "
            f"{{clamp_max}}({matched.start}, 0), 0)",
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        row_statement = statement_from_string(
            f"{row_index} = {clipped_start} + {matched.row_tile}.index"
        )
        mask_statement = statement_from_string(
            f"{valid} = {matched.row_tile}.index < {clipped_extent}"
        )
        load_statement = statement_from_string(
            f"_ = {{load}}({matched.lhs}, [{row_index}, {matched.contraction_tile}], "
            f"extra_mask={valid}[:, None])",
            load=load,
        )
        assert isinstance(load_statement, ast.Assign)

        class ReplaceLoad(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                return load_statement.value if node is matched.lhs_load else node

        ReplaceLoad().visit(matched.reduction)
        finish = statement_from_string(
            f"{{store}}({matched.output}, [{row_index}, {matched.column_tile}], "
            f"{{value}}, extra_mask={valid}[:, None])",
            store=store,
            value=matched.store.value,
        )
        replacement.body = [
            replacement.body[0],
            *matched.prefix,
            start_statement,
            extent_statement,
            row_statement,
            mask_statement,
            matched.initialize,
            matched.reduction,
            finish,
        ]
    host.body[host.body.index(root)] = replacement
    return True
