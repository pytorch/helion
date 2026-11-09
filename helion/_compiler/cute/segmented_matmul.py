"""Expose independent offset-delimited matrix rows to the CuTe scheduler.

The initial normalization accepts a direct, zero-seeded contraction whose only
effect is one store to a fresh matrix. The offset loads and accumulator's exact
conversion are retained. Group-specific M bounds become explicit load/store
masks; group, M and N are then available to the existing grouped scheduler.

Two source shapes reach that segment tile space:

* the per-group tile loop: ``for g in hl.grid(G)`` loading the group's start
  and end offsets, and under ``if M_g != 0`` a ``for tile_m, tile_n in
  hl.tile([M_g, N])`` with the contraction over ``lhs[start + tile_m.index,
  tile_k]`` and the store to the same rows;
* the hand-rolled persistent worklist (TritonBench's variant): ``for w in
  hl.grid(num_sms)`` around that group loop, whose branch counts the group's
  ``BLOCK_M x BLOCK_N`` tiles, walks them with ``for t in hl.grid(num_tiles)``
  strided by the worker count under ``if t * num_sms + w < num_tiles``, and
  builds the tile's rows and columns as ``start + m_idx * BLOCK_M +
  hl.arange(BLOCK_M)`` and ``n_idx * BLOCK_N + hl.arange(BLOCK_N)`` with the
  masks ``rows < end`` / ``cols < N`` on both loads and the store. With at
  least one worker every tile of every group is visited exactly once, and a
  tile's masked rows and columns are loaded as zero and never stored, so the
  program is the per-group tile loop with its schedule spelled out. The
  normalization keeps the group statements and the ``acc.to(out.dtype)``
  conversion, drops the schedule and the two ``register_block_size`` tiles
  it indexed with (the grouped scheduler brings its own tiles), and emits the
  same segment tile space as for the first shape.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import language as hl
from ... import runtime
from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ._ast_pass_utils import name_assignment
from .full_slice_matmul import _bound_names
from .full_slice_matmul import _global_reference
from .full_slice_matmul import _global_value
from .full_slice_matmul import _host_prelude
from .full_slice_matmul import _input_binding_is_stable
from .full_slice_matmul import _tensor_dimension

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

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
        if assigned := name_assignment(statement):
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


def _assignments(statements: Iterable[ast.stmt]) -> dict[str, ast.expr] | None:
    """``NAME = value`` statements binding distinct names, in order."""
    result: dict[str, ast.expr] = {}
    for statement in statements:
        assigned = name_assignment(statement)
        if assigned is None or assigned[0] in result:
            return None
        result[assigned[0]] = assigned[1]
    return result


def _grid_loop(
    node: ast.AST, host: HostFunction, local_names: set[str]
) -> ast.For | None:
    """``for NAME in hl.grid(extent):`` with no ``else``."""
    if (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and _global_value(node.iter.func, host, local_names) is hl.grid
        and len(node.iter.args) == 1
        and not node.iter.keywords
        and not node.orelse
    ):
        return node
    return None


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
    store_value: ast.expr
    # Prelude statements the rewrite deletes: the worklist shape's block-size
    # registrations, which only its schedule indexed with.
    removed: tuple[ast.stmt, ...] = ()


@dataclasses.dataclass(frozen=True)
class _Segments:
    """The group loop both shapes share: ``for g in hl.grid(G)`` loading the
    offsets ``start = offsets[g]``, ``end = offsets[g + 1]`` and the extent
    ``end - start``, then one ``if extent != 0`` (or ``> 0``) branch."""

    group: str
    start: str
    extent: str
    offsets: str
    prefix: tuple[ast.stmt, ...]
    condition: ast.If


def _match_segments(
    host: HostFunction,
    loop: ast.AST,
    local_names: set[str],
    aliases: dict[str, ast.expr],
) -> _Segments | None:
    grid = _grid_loop(loop, host, local_names)
    if (
        grid is None
        or len(grid.body) != 4
        or not isinstance(condition := grid.body[-1], ast.If)
        or condition.orelse
    ):
        return None
    group = cast("ast.Name", grid.target).id
    predicate = condition.test
    if not (
        isinstance(predicate, ast.Compare)
        and isinstance(predicate.left, ast.Name)
        and len(predicate.ops) == len(predicate.comparators) == 1
        and isinstance(predicate.ops[0], (ast.NotEq, ast.Gt))
        and _zero(predicate.comparators[0])
    ):
        return None
    extent = predicate.left.id
    assignments = [name_assignment(statement) for statement in grid.body[:-1]]
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
    end = difference.left.id
    start_load = prefix.get(start)
    end_load = prefix.get(end)
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
    groups = _resolve(cast("ast.Call", grid.iter).args[0], aliases)
    prefix_names = list(prefix)
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
        and prefix_names.index(start) < prefix_names.index(extent)
        and prefix_names.index(end) < prefix_names.index(extent)
    ):
        return None
    return _Segments(group, start, extent, offsets, tuple(grid.body[:-1]), condition)


def _operands_admit(
    host: HostFunction, local_names: set[str], lhs: str, rhs: str, offsets: str
) -> bool:
    lhs_tensor = host.params.arguments.get(lhs)
    rhs_tensor = host.params.arguments.get(rhs)
    return (
        isinstance(lhs_tensor, torch.Tensor)
        and isinstance(rhs_tensor, torch.Tensor)
        and lhs_tensor.ndim == rhs_tensor.ndim == 2
        and lhs_tensor.dtype == rhs_tensor.dtype
        and lhs_tensor.dtype in (torch.float16, torch.bfloat16)
        and all(
            _input_binding_is_stable(name, host, local_names)
            for name in (lhs, rhs, offsets)
        )
    )


def _fresh_output(
    host: HostFunction,
    local_names: set[str],
    aliases: dict[str, ast.expr],
    output: str,
    lhs: str,
    rhs: str,
) -> bool:
    """``output`` is a fresh ``[lhs.size(0), rhs.size(1)]`` allocation."""
    allocation = aliases.get(output)
    if not (
        isinstance(allocation, ast.Call)
        and _global_value(allocation.func, host, local_names)
        in (torch.empty, torch.zeros)
        and all(keyword.arg in ("dtype", "device") for keyword in allocation.keywords)
    ):
        return False
    shape = allocation.args
    if len(shape) == 1 and isinstance(shape[0], (ast.Tuple, ast.List)):
        shape = shape[0].elts
    return (
        len(shape) == 2
        and _dimension(shape[0], lhs, 0, aliases)
        and _dimension(shape[1], rhs, 1, aliases)
    )


def _match(
    host: HostFunction,
    root: ast.For,
    local_names: set[str],
    aliases: dict[str, ast.expr],
) -> _SegmentedMatmul | None:
    segments = _match_segments(host, root, local_names, aliases)
    if (
        segments is None
        or len(segments.condition.body) != 1
        or not isinstance(rows := segments.condition.body[0], ast.For)
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
        or not _name(row_extents.elts[0], segments.extent)
        or len(rows.body) != 3
    ):
        return None
    group, start, extent = segments.group, segments.start, segments.extent
    row, column = (cast("ast.Name", item).id for item in rows.target.elts)
    initialize, reduction, store = rows.body
    initialized = name_assignment(initialize)
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
    inner = [name_assignment(statement) for statement in reduction.body]
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
    prefix = dict(
        cast(
            "list[tuple[str, ast.expr]]",
            [name_assignment(statement) for statement in segments.prefix],
        )
    )
    if allowed_inner & {group, row, column, contraction, *prefix}:
        return None
    if not (
        _operands_admit(host, local_names, lhs, rhs, segments.offsets)
        and _dimension(row_extents.elts[1], rhs, 1, aliases)
        and _dimension(reduction.iter.args[0], lhs, 1, aliases)
        and _fresh_output(host, local_names, aliases, output, lhs, rhs)
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
        segments.offsets,
        output,
        segments.prefix,
        initialize,
        reduction,
        cast("ast.Subscript", lhs_expr),
        store.value,
    )


# Canonical names for the calls and the dtype the worklist proof compares
# expressions by; the user's own aliases of ``hl`` / ``torch`` resolve to them.
_MARKERS: tuple[tuple[object, str], ...] = (
    (hl.arange, "_helion_arange"),
    (hl.load, "_helion_load"),
    (hl.zeros, "_helion_zeros"),
    (torch.float32, "_torch_float32"),
)
_COMMUTATIVE = (ast.Add, ast.Mult, ast.BitAnd)


def _plain(node: ast.AST) -> ast.expr:
    """A fresh plain-``ast`` copy of an expression (no source metadata)."""
    return ast.parse(ast.unparse(node), mode="eval").body


class _Expand(ast.NodeTransformer):
    """Substitute a loop body's local definitions into an expression.

    A name bound in ``scope`` is replaced by its (expanded) definition unless
    it is ``opaque`` (the tensors, the registered block sizes, the worker
    count and the accumulator); ``t.size(i)`` / ``t.shape[i]`` of a tensor
    becomes ``_dim_t_i``, the contraction tile's ``.index`` its tile, and the
    recognised ``hl`` / ``torch`` references their markers. Two spellings of
    one value then share a ``_key``. ``used`` records the names expanded.
    """

    def __init__(
        self,
        scope: dict[str, ast.expr],
        opaque: set[str],
        tensors: set[str],
        contraction: str,
        host: HostFunction,
        local_names: set[str],
    ) -> None:
        self.scope = scope
        self.opaque = opaque
        self.tensors = tensors
        self.contraction = contraction
        self.host = host
        self.local_names = local_names
        self.used: set[str] = set()
        self.cyclic = False
        self._stack: list[str] = []

    def expand(self, node: ast.AST) -> ast.expr:
        return cast("ast.expr", self.visit(_plain(node)))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        name = node.id
        if name not in self.scope or name in self.opaque:
            return node
        if name in self._stack:
            self.cyclic = True
            return node
        self.used.add(name)
        self._stack.append(name)
        result = self.expand(self.scope[name])
        self._stack.pop()
        return result

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr == "index" and _name(node.value, self.contraction):
            return ast.Name(id=self.contraction, ctx=ast.Load())
        return self._marked(node) or self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        return self._dimension(node) or self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        dimension = self._dimension(node)
        if dimension is not None:
            return dimension
        marker = self._marked(node.func)
        self.generic_visit(node)
        if marker is not None:
            node.func = marker
        return node

    def _dimension(self, node: ast.expr) -> ast.Name | None:
        dimension = _tensor_dimension(node, self.tensors)
        if dimension is None or dimension[1] is None:
            return None
        return ast.Name(id=f"_dim_{dimension[0]}_{dimension[1]}", ctx=ast.Load())

    def _marked(self, node: ast.expr) -> ast.Name | None:
        value = _global_value(node, self.host, self.local_names)
        for reference, marker in _MARKERS:
            if value is reference:
                return ast.Name(id=marker, ctx=ast.Load())
        return None


def _key(node: object) -> object:
    """Structural identity modulo the order of ``+``, ``*`` and ``&`` operands
    and the direction of ``<`` / ``>``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, _COMMUTATIVE):
        terms: list[str] = []
        pending: list[ast.expr] = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, ast.BinOp) and type(current.op) is type(node.op):
                pending.extend((current.right, current.left))
            else:
                terms.append(repr(_key(current)))
        return (type(node.op).__name__, *sorted(terms))
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
    ):
        return _key(
            ast.Compare(
                left=node.comparators[0], ops=[ast.Lt()], comparators=[node.left]
            )
        )
    if isinstance(node, ast.AST):
        return (
            type(node).__name__,
            *(
                (field, _key(getattr(node, field)))
                for field in node._fields
                if field != "ctx"
            ),
        )
    if isinstance(node, list):
        return tuple(_key(item) for item in node)
    return (type(node).__name__, repr(node))


def _equal(node: ast.expr, template: str) -> bool:
    return _key(node) == _key(ast.parse(template, mode="eval").body)


def _worker_count(node: ast.expr, host: HostFunction, local_names: set[str]) -> bool:
    """At least one worker: the device's SM count, or a positive constant."""
    if isinstance(node, ast.Call):
        return (
            _global_value(node.func, host, local_names) is runtime.get_num_sm
            and len(node.args) == 1
            and not node.keywords
        )
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0


def _masked_load(
    node: ast.expr, inputs: set[str]
) -> tuple[str, list[ast.expr], ast.expr] | None:
    """``hl.load(input, [i, j], extra_mask=mask)`` -> (input, [i, j], mask)."""
    if (
        isinstance(node, ast.Call)
        and _name(node.func, "_helion_load")
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in inputs
        and isinstance(node.args[1], (ast.List, ast.Tuple))
        and len(node.args[1].elts) == 2
        and len(node.keywords) == 1
        and node.keywords[0].arg == "extra_mask"
    ):
        return node.args[0].id, list(node.args[1].elts), node.keywords[0].value
    return None


def _zero_tile(node: ast.expr) -> tuple[str, str] | None:
    """``hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)`` -> the two names."""
    if (
        isinstance(node, ast.Call)
        and _name(node.func, "_helion_zeros")
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
        and len(node.args[0].elts) == 2
        and all(isinstance(item, ast.Name) for item in node.args[0].elts)
        and len(node.keywords) == 1
        and node.keywords[0].arg == "dtype"
        and _name(node.keywords[0].value, "_torch_float32")
    ):
        first, second = cast("list[ast.Name]", node.args[0].elts)
        return first.id, second.id
    return None


def _match_worklist(
    host: HostFunction,
    root: ast.For,
    local_names: set[str],
    aliases: dict[str, ast.expr],
    tensors: set[str],
    registered: tuple[str, ...],
    fresh: Callable[[str], str],
) -> _SegmentedMatmul | None:
    """The hand-rolled persistent worklist over the segments (module docstring).

    The proof expands every local name of the worker, group, tile and
    contraction bodies into the loop targets, the offsets loads, the
    registered block sizes, the worker count and the tensor dimensions, and
    compares the expansions (``_key``: modulo the order of ``+`` / ``*`` /
    ``&`` operands and the direction of ``<``) with the admitted schedule,
    either linearization of the group's tile grid; every local statement
    must take part, so nothing else happens in the loops.
    """
    if (
        _grid_loop(root, host, local_names) is None
        or len(root.body) != 1
        or not isinstance(workers := cast("ast.Call", root.iter).args[0], ast.Name)
        or not _worker_count(_resolve(workers, aliases), host, local_names)
        or (segments := _match_segments(host, root.body[0], local_names, aliases))
        is None
        or len(registered) != 2
    ):
        return None
    worker = cast("ast.Name", root.target).id
    *count_statements, tile_loop = segments.condition.body
    counts = _assignments(count_statements)
    tiles = _grid_loop(tile_loop, host, local_names)
    if (
        counts is None
        or tiles is None
        or len(tiles.body) != 2
        or (linear := name_assignment(tiles.body[0])) is None
        or not isinstance(guard := tiles.body[1], ast.If)
        or guard.orelse
        or len(guard.body) < 3
    ):
        return None
    local_tile = cast("ast.Name", tiles.target).id
    tile_count = cast("ast.Call", tiles.iter).args[0]
    # The tile body: definitions around one contraction loop, then the store.
    loops = [
        index for index, item in enumerate(guard.body) if isinstance(item, ast.For)
    ]
    if len(loops) != 1:
        return None
    reduction = cast("ast.For", guard.body[loops[0]])
    store = guard.body[-1]
    definitions = _assignments(
        statement for statement in guard.body[:-1] if statement is not reduction
    )
    if (
        definitions is None
        or not isinstance(reduction.target, ast.Name)
        or not isinstance(reduction.iter, ast.Call)
        or _global_value(reduction.iter.func, host, local_names) is not hl.tile
        or len(reduction.iter.args) != 1
        or reduction.iter.keywords
        or reduction.orelse
        or (inner := _assignments(reduction.body)) is None
        or not inner
    ):
        return None
    contraction = reduction.target.id
    acc, update = next(reversed(inner.items()))
    if not (
        isinstance(update, ast.Call)
        and _global_value(update.func, host, local_names) is torch.addmm
        and len(update.args) == 3
        and not update.keywords
        and _name(update.args[0], acc)
        and acc in definitions
    ):
        return None
    # The store: ``hl.store(out, [rows, cols], acc.to(out.dtype), extra_mask=m)``.
    if not (
        isinstance(store, ast.Expr)
        and isinstance(call := store.value, ast.Call)
        and _global_value(call.func, host, local_names) is hl.store
        and len(call.args) == 3
        and isinstance(call.args[0], ast.Name)
        and isinstance(indices := call.args[1], (ast.List, ast.Tuple))
        and len(indices.elts) == 2
        and len(call.keywords) == 1
        and call.keywords[0].arg == "extra_mask"
        and isinstance(convert := call.args[2], ast.Call)
        and isinstance(convert.func, ast.Attribute)
        and _name(convert.func.value, acc)
        and convert.func.attr == "to"
        and len(convert.args) == 1
        and not convert.keywords
        and isinstance(convert.args[0], ast.Attribute)
        and _name(convert.args[0].value, call.args[0].id)
        and convert.args[0].attr == "dtype"
    ):
        return None
    output = call.args[0].id
    prefix = dict(
        cast(
            "list[tuple[str, ast.expr]]",
            [name_assignment(statement) for statement in segments.prefix],
        )
    )
    # The schedule's local names, bound once each and shadowing nothing.
    local = [
        *counts,
        linear[0],
        *(name for name in definitions if name != acc),
        *(name for name in inner if name != acc),
    ]
    targets = {worker, segments.group, local_tile, contraction}
    reserved = tensors | set(registered) | targets | {workers.id, *prefix}
    if len(set(local)) != len(local) or len(targets) != 4 or set(local) & reserved:
        return None
    expand = _Expand(
        {**aliases, **prefix, **counts, linear[0]: linear[1], **definitions, **inner},
        tensors | set(registered) | {workers.id, acc},
        tensors,
        contraction,
        host,
        local_names,
    )
    inputs = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    lhs_access = _masked_load(expand.expand(update.args[1]), inputs)
    rhs_access = _masked_load(expand.expand(update.args[2]), inputs)
    blocks = _zero_tile(expand.expand(definitions[acc]))
    if (
        lhs_access is None
        or rhs_access is None
        or blocks is None
        or set(blocks) != set(registered)
        or blocks[0] == blocks[1]
    ):
        return None
    lhs, (rows, k_index), rows_mask = lhs_access
    rhs, (rhs_k_index, columns), columns_mask = rhs_access
    block_m, block_n = blocks
    n = f"_dim_{rhs}_1"
    tile = f"({local_tile} * {workers.id} + {worker})"
    start = f"{segments.offsets}[{segments.group}]"
    end = f"{segments.offsets}[{segments.group} + 1]"
    m_tiles = f"(({end} - {start} + {block_m} - 1) // {block_m})"
    n_tiles = f"(({n} + {block_n} - 1) // {block_n})"
    # Row-fastest or column-fastest linearization of the group's tile grid.
    for m_index, n_index in (
        (f"{tile} % {m_tiles}", f"{tile} // {m_tiles}"),
        (f"{tile} // {n_tiles}", f"{tile} % {n_tiles}"),
    ):
        row_template = f"{start} + {m_index} * {block_m} + _helion_arange({block_m})"
        column_template = f"{n_index} * {block_n} + _helion_arange({block_n})"
        if _equal(rows, row_template) and _equal(columns, column_template):
            break
    else:
        return None
    rows_valid = f"({row_template}) < {end}"
    columns_valid = f"({column_template}) < {n}"
    store_rows, store_columns = (expand.expand(item) for item in indices.elts)
    if not (
        _equal(k_index, contraction)
        and _equal(rhs_k_index, contraction)
        and _equal(rows_mask, f"({rows_valid})[:, None]")
        and _equal(columns_mask, f"({columns_valid})[None, :]")
        and _equal(expand.expand(tile_count), f"{m_tiles} * {n_tiles}")
        and _equal(expand.expand(guard.test), f"{tile} < {m_tiles} * {n_tiles}")
        and _equal(store_rows, row_template)
        and _equal(store_columns, column_template)
        and _equal(
            expand.expand(call.keywords[0].value),
            f"({rows_valid})[:, None] & ({columns_valid})[None, :]",
        )
        and not expand.cyclic
        # Every local statement fed the schedule or the contraction.
        and expand.used >= set(local)
        and _operands_admit(host, local_names, lhs, rhs, segments.offsets)
        and _dimension(reduction.iter.args[0], lhs, 1, aliases)
        and _fresh_output(host, local_names, aliases, output, lhs, rhs)
    ):
        return None
    removed = tuple(
        statement
        for statement in host.body
        if (assigned := name_assignment(statement)) is not None
        and assigned[0] in registered
    )
    # Nothing but the replaced loop reads the registered block sizes.
    for statement in host.body:
        if statement is root or statement in removed:
            continue
        if any(
            isinstance(node, ast.Name) and node.id in registered
            for node in ast.walk(statement)
        ):
            return None
    assert isinstance(root, ExtendedAST)
    with root:
        tile_reference = _global_reference(hl.tile, host, local_names)
        zeros = _global_reference(hl.zeros, host, local_names)
        addmm = _global_reference(torch.addmm, host, local_names)
        float32 = _global_reference(torch.float32, host, local_names)
        if tile_reference is None or zeros is None or addmm is None or float32 is None:
            return None
        row_tile = fresh("_helion_segment_tile_m")
        column_tile = fresh("_helion_segment_tile_n")
        lhs_name, rhs_name = (
            operand.id if isinstance(operand, ast.Name) else fresh(default)
            for operand, default in zip(
                update.args[1:],
                ("_helion_segment_lhs", "_helion_segment_rhs"),
                strict=True,
            )
        )
        initialize = statement_from_string(
            f"{acc} = {{zeros}}([{row_tile}, {column_tile}], dtype={{dtype}})",
            zeros=zeros,
            dtype=float32,
        )
        loop = statement_from_string(
            f"for {contraction} in {{tile}}({{extent}}):\n    pass",
            tile=tile_reference,
            extent=reduction.iter.args[0],
        )
        assert isinstance(loop, ast.For)
        # The lhs subscript is the placeholder the masked segment load replaces.
        lhs_statement = statement_from_string(
            f"{lhs_name} = {lhs}[{row_tile}, {contraction}]"
        )
        assert isinstance(lhs_statement, ast.Assign)
        loop.body = [
            lhs_statement,
            statement_from_string(f"{rhs_name} = {rhs}[{contraction}, {column_tile}]"),
            statement_from_string(
                f"{acc} = {{addmm}}({acc}, {lhs_name}, {rhs_name})", addmm=addmm
            ),
        ]
    return _SegmentedMatmul(
        root,
        segments.group,
        row_tile,
        column_tile,
        contraction,
        segments.start,
        segments.extent,
        lhs,
        rhs,
        segments.offsets,
        output,
        segments.prefix,
        initialize,
        loop,
        cast("ast.Subscript", lhs_statement.value),
        convert,
        removed,
    )


def normalize_segmented_matmuls(host: HostFunction) -> bool:
    env = CompileEnvironment.current()
    if env.backend_name != "cute" or not env.settings.cute_segmented_matmul_tiling:
        return False
    local_names = _bound_names(host)
    prelude = _host_prelude(host, local_names, allow_block_size_registration=True)
    if prelude is None:
        return False
    roots = [statement for statement in host.body if isinstance(statement, ast.For)]
    if len(roots) != 1:
        return False
    root = roots[0]
    registered = tuple(
        assigned[0]
        for statement in host.body[: host.body.index(root)]
        if (assigned := name_assignment(statement)) is not None
        and isinstance(assigned[1], ast.Call)
        and _global_value(assigned[1].func, host, local_names) is hl.register_block_size
    )
    used = local_names | set(host.fn.__globals__)

    def fresh(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    aliases = _metadata_aliases(host)
    matched = None if registered else _match(host, root, local_names, aliases)
    if matched is None:
        matched = _match_worklist(
            host, root, local_names, aliases, prelude.tensors, registered, fresh
        )
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
            value=matched.store_value,
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
    for statement in matched.removed:
        host.body.remove(statement)
    return True
