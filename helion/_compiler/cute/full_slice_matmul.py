"""Give direct full-slice CuTe matmuls an ordinary tunable contraction loop.

The envelope accepts two-dimensional half/BF16 input tensors whose contraction
is a full slice. A private FP32 accumulator spans the entire contraction; the
original result dtype is restored once, before any following epilogue. The
contraction extents must already be provably equal; sample values cannot
establish equality for distinct dynamic shape symbols.

Two root shapes are rewritten:

* ``for a, b in hl.tile([m, n])`` with ``lhs[a, :] @ rhs[:, b]``: only the
  matmul expression changes. It may be the whole right-hand side of an
  assignment, nested inside a pure expression (``torch.relu(lhs[a, :] @
  rhs[:, b])``) whose loop-carried evaluation is hoisted in front of the
  statement, or read an operand through a local alias (``t = lhs[a, :]``)
  that is bound exactly once, consumed only by rewritten contractions, and
  whose source tensor is only read inside the root. No memory operation
  outside the replaced expression moves across the loop.
* ``for a in hl.tile(m)`` with ``lhs[a, :] @ rhs[:, :]``: the root gains an
  N tile, ``for a, n in hl.tile([m, rhs.size(1)])``, and every other
  ``t[a, :]`` load or store of the body is re-sliced to ``t[a, n]``. That is
  only meaning-preserving when the body treats the sliced axis
  element-wise, so the body is admitted by an explicit whitelist: local
  assignments and ``t[a, :]`` stores whose values are pointwise arithmetic,
  comparisons, pointwise ``torch`` functions and dtype casts over such row
  tiles; every re-sliced tensor must have a dim-1 extent provably equal to
  N; the matmul operands are never stored in the root (an N tile would
  otherwise race on the shared contraction rows); statements before the
  matmul's own statement may only bind its operand aliases, and alias loads
  that survive are re-issued after the contraction loop (the tcgen05 K-loop
  family refuses per-element reads placed ahead of it). Anything outside the
  whitelist keeps the original source.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import language as hl
from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from ..host_function import HostFunction


_FACTORIES = (
    torch.empty,
    torch.empty_like,
    torch.zeros,
    torch.zeros_like,
    torch.ones,
    torch.ones_like,
    torch.full,
    torch.full_like,
)
_LIKE_FACTORIES = (
    torch.empty_like,
    torch.zeros_like,
    torch.ones_like,
    torch.full_like,
)
_MATMULS = (torch.mm, torch.matmul)
_METADATA_ATTRIBUTES = frozenset({"shape", "dtype", "device", "ndim"})
_METADATA_METHODS = frozenset({"size", "stride", "dim", "ndimension"})
_SCALAR_TYPES = (int, float, bool)
_ELEMENTWISE_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_ELEMENTWISE_UNARYOPS = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
# Pointwise ``torch`` functions whose result keeps the shape of (the
# broadcast of) their tensor arguments; re-slicing their inputs along one
# axis re-slices their outputs identically.
_POINTWISE_TORCH = frozenset(
    {
        "abs",
        "add",
        "ceil",
        "clamp",
        "clamp_max",
        "clamp_min",
        "cos",
        "div",
        "erf",
        "erfc",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "isinf",
        "isnan",
        "log",
        "log1p",
        "log2",
        "logical_and",
        "logical_not",
        "logical_or",
        "maximum",
        "minimum",
        "mul",
        "neg",
        "pow",
        "reciprocal",
        "relu",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "sin",
        "sqrt",
        "square",
        "sub",
        "tanh",
        "true_divide",
        "trunc",
        "where",
    }
)
_POINTWISE_METHODS = frozenset(
    {
        "abs",
        "bfloat16",
        "clamp",
        "clamp_max",
        "clamp_min",
        "exp",
        "exp2",
        "float",
        "half",
        "log",
        "log2",
        "neg",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sqrt",
        "square",
        "tanh",
        "to",
    }
)
_PURE_HL = ("zeros", "full", "arange", "dot")


def _bound_names(host: HostFunction) -> set[str]:
    return {
        *host.params.arguments,
        *(
            node.id
            for statement in host.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        ),
    }


def _global_value(node: ast.AST, host: HostFunction, local_names: set[str]) -> object:
    if isinstance(node, ast.Name) and node.id not in local_names:
        return host.fn.__globals__.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _global_value(node.value, host, local_names)
        if parent is torch or parent is hl:
            return vars(parent).get(node.attr)
    return None


def _global_reference(
    value: object, host: HostFunction, local_names: set[str]
) -> ast.expr | None:
    """Resolve real bindings without changing the user's globals or aliases."""
    for name, candidate in host.fn.__globals__.items():
        if name in local_names or not name.isidentifier():
            continue
        if candidate is value:
            return create(ast.Name, id=name, ctx=ast.Load())
        if candidate is torch or candidate is hl:
            for member, item in vars(candidate).items():
                if item is value and member.isidentifier():
                    return create(
                        ast.Attribute,
                        value=create(ast.Name, id=name, ctx=ast.Load()),
                        attr=member,
                        ctx=ast.Load(),
                    )
    return None


def _metadata_expression(
    node: ast.AST,
    tensors: set[str],
    metadata: set[str],
    host: HostFunction,
    local_names: set[str],
) -> bool:
    def recurse(value: ast.AST) -> bool:
        return _metadata_expression(value, tensors, metadata, host, local_names)

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name) and node.id in metadata:
        return True
    if isinstance(node, (ast.Name, ast.Attribute)):
        value = _global_value(node, host, local_names)
        if type(value) in (
            int,
            float,
            bool,
            str,
            torch.dtype,
            torch.device,
            torch.memory_format,
        ):
            return True
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in tensors
            and node.attr in _METADATA_ATTRIBUTES
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(recurse(item) for item in node.elts)
    if isinstance(node, ast.Subscript):
        return recurse(node.value) and recurse(node.slice)
    if isinstance(node, ast.Call):
        if _global_value(node.func, host, local_names) is torch.promote_types:
            return not node.keywords and all(recurse(arg) for arg in node.args)
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in tensors
            and node.func.attr in _METADATA_METHODS
            and not node.keywords
            and all(recurse(arg) for arg in node.args)
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
            and recurse(node.left)
            and recurse(node.right)
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)) and recurse(
            node.operand
        )
    if isinstance(node, ast.Compare):
        return recurse(node.left) and all(recurse(item) for item in node.comparators)
    if isinstance(node, ast.BoolOp):
        return all(recurse(item) for item in node.values)
    return False


# A resolved allocation extent: a constant, the (tensor, axis) a dimension
# name read from when the allocation ran, or unknown.
_Extent = tuple[str, int] | int | None


@dataclasses.dataclass
class _Prelude:
    """What the pure host prelude binds before the first device loop.

    ``dims`` maps a metadata name to the tensor dimension it currently
    reads from (``m, n = x.size()``; ``n = x.size(1)``; ``n = x.shape[1]``)
    and ``shapes`` records each fresh allocation's extents (or the tensor it
    was allocated like), so a later proof can compare the extent of an
    allocated output with an input dimension. Extents are resolved at the
    allocation statement, against the bindings live there, to a constant or
    the ``(tensor, axis)`` a dimension name read from: tensors are never
    rebound in the prelude, so the record stays valid when a dimension name
    is rebound after the allocation (``n = y.size(1)`` following
    ``aux = torch.empty([m, n])`` must not re-size ``aux``).
    """

    tensors: set[str]
    metadata: set[str]
    dims: dict[str, tuple[str, int]] = dataclasses.field(default_factory=dict)
    shapes: dict[str, list[_Extent] | str | None] = dataclasses.field(
        default_factory=dict
    )


def _tensor_dimension(
    node: ast.AST, tensors: set[str]
) -> tuple[str, int | None] | None:
    """``t.size()`` / ``t.shape`` -> (t, None); ``t.size(i)`` / ``t.shape[i]`` -> (t, i)."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in tensors
        and node.func.attr == "size"
        and not node.keywords
        and len(node.args) <= 1
        and all(isinstance(arg, ast.Constant) for arg in node.args)
    ):
        if not node.args:
            return node.func.value.id, None
        axis = cast("ast.Constant", node.args[0]).value
        return (node.func.value.id, axis) if type(axis) is int else None
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in tensors
        and node.attr == "shape"
    ):
        return node.value.id, None
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
    ):
        parent = _tensor_dimension(node.value, tensors)
        if parent is not None and parent[1] is None:
            return parent[0], node.slice.value
    return None


def _host_prelude(
    host: HostFunction,
    local_names: set[str],
    *,
    allow_block_size_registration: bool = False,
) -> _Prelude | None:
    """Do not infer input metadata across arbitrary host code or aliases."""
    prelude = _Prelude(
        tensors={
            name
            for name, value in host.params.arguments.items()
            if isinstance(value, torch.Tensor)
        },
        metadata=set(),
    )
    tensors, metadata = prelude.tensors, prelude.metadata
    metadata.update(set(host.params.arguments) - tensors)
    for statement in host.body:
        if isinstance(statement, ast.For):
            return prelude
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue
        if isinstance(statement, ast.Assert):
            if not _metadata_expression(
                statement.test, tensors, metadata, host, local_names
            ):
                return None
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        names = (
            [target.id]
            if isinstance(target, ast.Name)
            else [item.id for item in target.elts if isinstance(item, ast.Name)]
            if isinstance(target, (ast.Tuple, ast.List))
            else []
        )
        if (
            not names
            or isinstance(target, (ast.Tuple, ast.List))
            and len(names) != len(target.elts)
        ):
            return None
        if set(names) & (set(host.params.arguments) | tensors):
            return None
        for name in names:
            prelude.dims.pop(name, None)
        value = statement.value
        if isinstance(value, ast.Call) and any(
            _global_value(value.func, host, local_names) is factory
            for factory in _FACTORIES
        ):
            if len(names) != 1 or any(
                keyword.arg in (None, "out") for keyword in value.keywords
            ):
                return None
            if not all(
                isinstance(arg, ast.Name)
                and arg.id in tensors
                or _metadata_expression(arg, tensors, metadata, host, local_names)
                for arg in (*value.args, *(keyword.value for keyword in value.keywords))
            ):
                return None
            prelude.shapes[names[0]] = _allocation_shape(
                prelude, value, host, local_names
            )
            tensors.add(names[0])
        elif (
            allow_block_size_registration
            and isinstance(value, ast.Call)
            and _global_value(value.func, host, local_names) is hl.register_block_size
            and not value.keywords
            and all(
                _metadata_expression(arg, tensors, metadata, host, local_names)
                for arg in value.args
            )
        ) or _metadata_expression(value, tensors, metadata, host, local_names):
            metadata.update(names)
            _record_dimensions(prelude, host, target, value, names)
        else:
            return None
    return None


def _record_dimensions(
    prelude: _Prelude,
    host: HostFunction,
    target: ast.expr,
    value: ast.expr,
    names: list[str],
) -> None:
    """``n = t.size(1)``; ``m, n = t.size()``; ``m, n = t.size(0), u.size(1)``."""
    tensors = prelude.tensors
    if isinstance(target, ast.Name):
        dimension = _tensor_dimension(value, tensors)
        if dimension is not None and dimension[1] is not None:
            prelude.dims[names[0]] = (dimension[0], dimension[1])
        return
    if isinstance(value, (ast.Tuple, ast.List)):
        if len(value.elts) != len(names):
            return
        for name, item in zip(names, value.elts, strict=True):
            dimension = _tensor_dimension(item, tensors)
            if dimension is not None and dimension[1] is not None:
                prelude.dims[name] = (dimension[0], dimension[1])
        return
    dimension = _tensor_dimension(value, tensors)
    if (
        dimension is not None
        and dimension[1] is None
        and _prelude_ndim(prelude, host, dimension[0]) == len(names)
    ):
        for index, name in enumerate(names):
            prelude.dims[name] = (dimension[0], index)


def _prelude_ndim(prelude: _Prelude, host: HostFunction, name: str) -> int | None:
    value = host.params.arguments.get(name)
    if isinstance(value, torch.Tensor):
        return value.ndim
    shape = prelude.shapes.get(name)
    if isinstance(shape, str):
        return _prelude_ndim(prelude, host, shape)
    return len(shape) if shape is not None else None


def _allocation_shape(
    prelude: _Prelude, call: ast.Call, host: HostFunction, local_names: set[str]
) -> list[_Extent] | str | None:
    """The allocation's extents as the prelude binds them at this statement."""
    factory = _global_value(call.func, host, local_names)
    if factory in _LIKE_FACTORIES:
        return (
            call.args[0].id
            if call.args and isinstance(call.args[0], ast.Name)
            else None
        )
    size = [keyword.value for keyword in call.keywords if keyword.arg == "size"]
    if size:
        shape: list[ast.expr] | None = (
            list(size[0].elts) if isinstance(size[0], (ast.Tuple, ast.List)) else None
        )
    elif call.args and isinstance(call.args[0], (ast.Tuple, ast.List)):
        shape = list(call.args[0].elts)
    elif factory is torch.full:
        shape = None
    else:
        shape = list(call.args) or None
    if shape is None:
        return None
    return [_resolve_extent(prelude, item) for item in shape]


def _resolve_extent(prelude: _Prelude, node: ast.AST) -> _Extent:
    """A shape expression under the bindings live where it is evaluated.

    A constant stays a constant; a dimension name becomes the ``(tensor,
    axis)`` it reads from right now; a direct ``t.size(i)`` / ``t.shape[i]``
    names the dimension; anything else is unknown (fail closed).
    """
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is int else None
    if isinstance(node, ast.Name):
        return prelude.dims.get(node.id)
    dimension = _tensor_dimension(node, prelude.tensors)
    if dimension is not None and dimension[1] is not None:
        return dimension[0], dimension[1]
    return None


def _pure_host_prelude(
    host: HostFunction,
    local_names: set[str],
    *,
    allow_block_size_registration: bool = False,
) -> bool:
    """Do not infer input metadata across arbitrary host code or aliases."""
    return (
        _host_prelude(
            host,
            local_names,
            allow_block_size_registration=allow_block_size_registration,
        )
        is not None
    )


def _input_binding_is_stable(
    name: str, host: HostFunction, local_names: set[str]
) -> bool:
    """Reject rebindings and escapes that could mutate input shape or dtype."""
    for statement in host.body:
        parents = {
            child: parent
            for parent in ast.walk(statement)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or node.id != name:
                continue
            if not isinstance(node.ctx, ast.Load):
                return False
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                continue
            if isinstance(parent, ast.Attribute) and parent.value is node:
                if (
                    isinstance(parent.ctx, ast.Load)
                    and parent.attr in _METADATA_ATTRIBUTES | _METADATA_METHODS
                ):
                    continue
                return False
            if isinstance(parent, ast.Call) and any(
                _global_value(parent.func, host, local_names) is function
                for function in (*_FACTORIES, *_MATMULS)
            ):
                continue
            return False
    return True


def _full_slice(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _operand(node: ast.AST, free_name: str, contraction_dimension: int) -> str | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
    ):
        return None
    contraction = node.slice.elts[contraction_dimension]
    free = node.slice.elts[1 - contraction_dimension]
    return (
        node.value.id
        if _full_slice(contraction)
        and isinstance(free, ast.Name)
        and free.id == free_name
        else None
    )


def _row_subscript(node: ast.AST, row_name: str) -> str | None:
    """``t[row, :]`` -> ``t``."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and isinstance(node.slice.elts[0], ast.Name)
        and node.slice.elts[0].id == row_name
        and _full_slice(node.slice.elts[1])
    ):
        return node.value.id
    return None


def _whole_subscript(node: ast.AST) -> str | None:
    """``t[:, :]`` -> ``t``."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and all(_full_slice(item) for item in node.slice.elts)
    ):
        return node.value.id
    return None


def _only_read_in(statements: list[ast.stmt], name: str) -> bool:
    """``name`` appears only as a subscript load or metadata read here."""
    for statement in statements:
        parents = {
            child: parent
            for parent in ast.walk(statement)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or node.id != name:
                continue
            parent = parents.get(node)
            if (
                isinstance(node.ctx, ast.Load)
                and isinstance(parent, ast.Subscript)
                and parent.value is node
                and isinstance(parent.ctx, ast.Load)
            ):
                continue
            if (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and isinstance(parent.ctx, ast.Load)
                and parent.attr in _METADATA_ATTRIBUTES | _METADATA_METHODS
            ):
                continue
            return False
    return True


def _count_names(
    statements: list[ast.stmt],
    name: str,
    ctx: type[ast.expr_context] | tuple[type[ast.expr_context], ...],
) -> int:
    return sum(
        1
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ctx)
    )


def _replace_child(parent: ast.AST, old: ast.AST, new: ast.AST) -> None:
    for field, value in ast.iter_fields(parent):
        if value is old:
            setattr(parent, field, new)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                if item is old:
                    value[index] = new
                    return
    raise AssertionError("child not found")


@dataclasses.dataclass
class _Rewriter:
    host: HostFunction
    env: CompileEnvironment
    local_names: set[str]
    prelude: _Prelude
    input_tensors: dict[str, torch.Tensor]
    used_names: set[str]
    torch_members: frozenset[int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.torch_members = frozenset(id(value) for value in vars(torch).values())

    # -- shared helpers ---------------------------------------------------

    def fresh(self, prefix: str) -> str:
        while prefix in self.used_names:
            prefix += "_"
        self.used_names.add(prefix)
        return prefix

    def global_value(self, node: ast.AST) -> object:
        return _global_value(node, self.host, self.local_names)

    def matmul_operands(self, node: ast.AST) -> tuple[ast.expr, ast.expr] | None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return node.left, node.right
        if (
            isinstance(node, ast.Call)
            and len(node.args) == 2
            and not node.keywords
            and any(self.global_value(node.func) is function for function in _MATMULS)
        ):
            return node.args[0], node.args[1]
        return None

    def find_matmuls(self, node: ast.AST) -> list[ast.expr]:
        return [
            cast("ast.expr", child)
            for child in ast.walk(node)
            if self.matmul_operands(child) is not None
        ]

    def alias_definitions(self, root: ast.For) -> dict[str, ast.Assign]:
        """Body-level ``name = tensor[...]`` bindings that are never rebound."""
        result: dict[str, ast.Assign] = {}
        for statement in root.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Subscript)
                and isinstance(statement.value.value, ast.Name)
            ):
                continue
            name = statement.targets[0].id
            if (
                name in self.host.params.arguments
                or name in self.prelude.tensors
                or name in self.prelude.metadata
                or _count_names(self.host.body, name, (ast.Store, ast.Del)) != 1
            ):
                continue
            result[name] = statement
        return result

    def tensor_ndim(self, name: str) -> int | None:
        value = self.host.params.arguments.get(name)
        if isinstance(value, torch.Tensor):
            return value.ndim
        shape = self.prelude.shapes.get(name)
        if isinstance(shape, str):
            return self.tensor_ndim(shape)
        return len(shape) if shape is not None else None

    def tensor_extent(self, name: str, axis: int) -> int | torch.SymInt | None:
        value = self.host.params.arguments.get(name)
        if isinstance(value, torch.Tensor):
            return value.size(axis) if 0 <= axis < value.ndim else None
        shape = self.prelude.shapes.get(name)
        if isinstance(shape, str):
            return self.tensor_extent(shape, axis)
        if shape is None or not 0 <= axis < len(shape):
            return None
        extent = shape[axis]
        if isinstance(extent, tuple):
            return self.tensor_extent(*extent)
        return extent

    def known_equal(
        self, a: int | torch.SymInt | None, b: int | torch.SymInt | None
    ) -> bool:
        return a is not None and b is not None and self.env.known_equal(a, b)

    def pure_expression(self, node: ast.AST) -> bool:
        """Hoisting the contraction out of this expression reorders nothing.

        The hoisted matmul reads only input tensors the root never stores,
        so the other sub-expressions may be any functional ``torch`` /
        ``hl`` call or tensor method of a value computed here: whatever they
        compute, they cannot change what the contraction reads. What is
        refused is anything that could write through a name -- an ``out=``
        keyword, an in-place (trailing underscore) method, a receiver that is
        a module member or other global -- and any syntax not listed.
        """
        recurse = self.pure_expression
        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        if isinstance(node, ast.Attribute):
            return isinstance(node.ctx, ast.Load) and recurse(node.value)
        if isinstance(node, ast.Subscript):
            return (
                isinstance(node.ctx, ast.Load)
                and recurse(node.value)
                and recurse(node.slice)
            )
        if isinstance(node, ast.Slice):
            return all(
                part is None or recurse(part)
                for part in (node.lower, node.upper, node.step)
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(recurse(item) for item in node.elts)
        if isinstance(node, ast.BinOp):
            return recurse(node.left) and recurse(node.right)
        if isinstance(node, ast.UnaryOp):
            return recurse(node.operand)
        if isinstance(node, ast.Compare):
            return recurse(node.left) and all(
                recurse(item) for item in node.comparators
            )
        if isinstance(node, ast.BoolOp):
            return all(recurse(item) for item in node.values)
        if isinstance(node, ast.IfExp):
            return recurse(node.test) and recurse(node.body) and recurse(node.orelse)
        if isinstance(node, ast.Call):
            if any(keyword.arg in (None, "out") for keyword in node.keywords):
                return False
            if not all(
                recurse(item)
                for item in (*node.args, *(keyword.value for keyword in node.keywords))
            ):
                return False
            value = self.global_value(node.func)
            if value is not None:
                return id(value) in self.torch_members or any(
                    value is getattr(hl, name) for name in _PURE_HL
                )
            # A functional tensor method on a value computed here. Module
            # members and other globals are not receivers (fail closed).
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.isidentifier()
                and not node.func.attr.startswith("_")
                and not node.func.attr.endswith("_")
                and self.local_receiver(node.func.value)
                and recurse(node.func.value)
            )
        return False

    def local_receiver(self, node: ast.AST) -> bool:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id in self.local_names
        return isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call))

    def resolve_operands(
        self,
        node: ast.expr,
        aliases: dict[str, ast.Assign],
        lhs_free: str,
        rhs_free: str | None,
    ) -> tuple[str, str, ast.Assign | None, ast.Assign | None] | None:
        """Operand tensor names and the alias statements they were read through."""
        operands = self.matmul_operands(node)
        if operands is None:
            return None
        lhs, rhs = operands
        lhs_alias = rhs_alias = None
        if isinstance(lhs, ast.Name) and lhs.id in aliases:
            lhs_alias = aliases[lhs.id]
            lhs = lhs_alias.value
        if isinstance(rhs, ast.Name) and rhs.id in aliases:
            rhs_alias = aliases[rhs.id]
            rhs = rhs_alias.value
        lhs_name = _operand(lhs, lhs_free, 1)
        rhs_name = (
            _operand(rhs, rhs_free, 0)
            if rhs_free is not None
            else _whole_subscript(rhs)
        )
        if lhs_name not in self.input_tensors or rhs_name not in self.input_tensors:
            return None
        lhs_value, rhs_value = (
            self.input_tensors[lhs_name],
            self.input_tensors[rhs_name],
        )
        if lhs_value.dtype != rhs_value.dtype or not self.env.known_equal(
            lhs_value.size(1), rhs_value.size(0)
        ):
            return None
        return lhs_name, rhs_name, lhs_alias, rhs_alias

    def contraction(
        self,
        statement: ast.stmt,
        root: ast.For,
        lhs_name: str,
        rhs_name: str,
        m_name: str,
        n_name: str,
        target: ast.expr,
    ) -> list[ast.stmt] | None:
        """``[accumulator init, K loop, target = accumulator.to(dtype)]``."""
        assert isinstance(statement, ExtendedAST)
        with statement:
            zeros = _global_reference(hl.zeros, self.host, self.local_names)
            addmm = _global_reference(torch.addmm, self.host, self.local_names)
            float32 = _global_reference(torch.float32, self.host, self.local_names)
            if zeros is None or addmm is None or float32 is None:
                return None
            accumulator = self.fresh("_helion_fullslice_acc")
            contraction = self.fresh("_helion_fullslice_k")
            initial = statement_from_string(
                f"{accumulator} = {{zeros}}([{m_name}, {n_name}], dtype={{dtype}})",
                zeros=zeros,
                dtype=float32,
            )
            loop = statement_from_string(
                f"for {contraction} in {{tile}}({lhs_name}.size(1)):\n"
                f"    {accumulator} = {{addmm}}({accumulator}, "
                f"{lhs_name}[{m_name}, {contraction}], {rhs_name}[{contraction}, {n_name}])",
                tile=cast("ast.Call", root.iter).func,
                addmm=addmm,
            )
            finish = statement_from_string(
                f"{{target}} = {accumulator}.to({lhs_name}.dtype)",
                target=target,
            )
            return [initial, loop, finish]

    def splice(
        self,
        statement: ast.stmt,
        node: ast.expr,
        root: ast.For,
        lhs_name: str,
        rhs_name: str,
        m_name: str,
        n_name: str,
    ) -> tuple[list[ast.stmt], bool] | None:
        """Hoist ``node``'s contraction in front of ``statement``.

        Returns the statements to insert and whether ``statement`` itself
        was consumed (a plain ``name = matmul`` assignment).
        """
        assert isinstance(statement, ast.Assign)
        if node is statement.value and isinstance(statement.targets[0], ast.Name):
            emitted = self.contraction(
                statement,
                root,
                lhs_name,
                rhs_name,
                m_name,
                n_name,
                statement.targets[0],
            )
            return (emitted, True) if emitted is not None else None
        assert isinstance(statement, ExtendedAST)
        with statement:
            result = self.fresh("_helion_fullslice_mm")
            emitted = self.contraction(
                statement,
                root,
                lhs_name,
                rhs_name,
                m_name,
                n_name,
                create(ast.Name, id=result, ctx=ast.Store()),
            )
            if emitted is None:
                return None
            parents = {
                child: parent
                for parent in ast.walk(statement)
                for child in ast.iter_child_nodes(parent)
            }
            _replace_child(
                parents[node], node, create(ast.Name, id=result, ctx=ast.Load())
            )
        return emitted, False

    # -- two-axis roots ---------------------------------------------------

    def rewrite_two_axis_root(self, root: ast.For) -> int:
        m_name, n_name = (
            cast("ast.Name", item).id for item in cast("ast.Tuple", root.target).elts
        )
        aliases = self.alias_definitions(root)
        # Pass 1: collect provable contractions without touching the tree.
        candidates: list[tuple[ast.Assign, ast.expr, str, str, list[ast.Assign]]] = []
        for statement in root.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], (ast.Name, ast.Subscript))
                and statement not in aliases.values()
            ):
                continue
            target = statement.targets[0]
            if not self.pure_expression(statement.value) or (
                isinstance(target, ast.Subscript)
                and not self.pure_expression(target.slice)
            ):
                continue
            for node in self.find_matmuls(statement.value):
                resolved = self.resolve_operands(node, aliases, m_name, n_name)
                if resolved is None:
                    continue
                lhs_name, rhs_name, lhs_alias, rhs_alias = resolved
                used = [alias for alias in (lhs_alias, rhs_alias) if alias is not None]
                if any(
                    alias is not None and not _only_read_in(root.body, name)
                    for alias, name in ((lhs_alias, lhs_name), (rhs_alias, rhs_name))
                ):
                    continue
                candidates.append((statement, node, lhs_name, rhs_name, used))
        # An alias must be consumed entirely by the rewritten contractions:
        # a surviving full-slice value beside the new K loop would keep a
        # whole-row axis in the thread plan that no default config fits.
        operand_reads: dict[str, int] = {}
        for _, _, _, _, used in candidates:
            for alias in used:
                name = cast("ast.Name", alias.targets[0]).id
                operand_reads[name] = operand_reads.get(name, 0) + 1
        live = {
            name
            for name, reads in operand_reads.items()
            if _count_names(self.host.body, name, ast.Load) != reads
        }
        candidates = [
            candidate
            for candidate in candidates
            if not any(
                cast("ast.Name", alias.targets[0]).id in live for alias in candidate[4]
            )
        ]
        # Pass 2: splice in body order; an alias disappears with its reads.
        changed = 0
        body: list[ast.stmt] = []
        consumed_aliases: list[ast.Assign] = []
        for statement in root.body:
            body.append(statement)
            for candidate in candidates:
                if candidate[0] is not statement:
                    continue
                _, node, lhs_name, rhs_name, used = candidate
                spliced = self.splice(
                    statement, node, root, lhs_name, rhs_name, m_name, n_name
                )
                if spliced is None:
                    continue
                emitted, consumed = spliced
                if consumed:
                    body[-1:] = emitted
                else:
                    body[-1:-1] = emitted
                consumed_aliases.extend(used)
                changed += 1
                if consumed:
                    break
        root.body = body
        for alias in consumed_aliases:
            name = cast("ast.Name", alias.targets[0]).id
            if alias in root.body and _count_names(self.host.body, name, ast.Load) == 0:
                root.body.remove(alias)
        return changed

    # -- single-axis roots ------------------------------------------------

    def rewrite_single_axis_root(self, root: ast.For) -> int:
        m_name = cast("ast.Name", root.target).id
        iterator = cast("ast.Call", root.iter)
        if iterator.keywords or isinstance(iterator.args[0], (ast.Tuple, ast.List)):
            return 0
        body = root.body
        tensor_names = self.prelude.tensors
        body_locals: set[str] = set()
        for statement in body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1):
                return 0
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                if (
                    target.id in self.host.params.arguments
                    or target.id in tensor_names
                    or target.id in self.prelude.metadata
                ):
                    return 0
                body_locals.add(target.id)
            elif _row_subscript(target, m_name) not in tensor_names:
                return 0
        matmuls = [
            (statement, node)
            for statement in body
            for node in self.find_matmuls(cast("ast.Assign", statement).value)
        ]
        if len(matmuls) != 1:
            return 0
        statement, node = matmuls[0]
        aliases = self.alias_definitions(root)
        resolved = self.resolve_operands(node, aliases, m_name, None)
        if resolved is None:
            return 0
        lhs_name, rhs_name, lhs_alias, rhs_alias = resolved
        if not (_only_read_in(body, lhs_name) and _only_read_in(body, rhs_name)):
            return 0
        operand_aliases = [
            alias for alias in (lhs_alias, rhs_alias) if alias is not None
        ]
        position = body.index(statement)
        if any(item not in operand_aliases for item in body[:position]):
            return 0
        if any(item not in body[:position] for item in operand_aliases):
            return 0
        rhs_value = self.input_tensors[rhs_name]
        n_extent = rhs_value.size(1)
        if rhs_alias is not None and (
            _count_names(
                self.host.body, cast("ast.Name", rhs_alias.targets[0]).id, ast.Load
            )
            != 1
        ):
            return 0
        lhs_alias_live = lhs_alias is not None and (
            _count_names(
                self.host.body, cast("ast.Name", lhs_alias.targets[0]).id, ast.Load
            )
            != 1
        )
        if lhs_alias_live and not self.known_equal(
            self.input_tensors[lhs_name].size(1), n_extent
        ):
            return 0
        sites: list[ast.Subscript] = []

        def safe(value: ast.AST) -> bool:
            if value is node:
                return True
            if isinstance(value, ast.Name):
                if value.id in body_locals or value.id in self.prelude.metadata:
                    return True
                return (
                    value.id not in self.local_names
                    and type(self.global_value(value)) in _SCALAR_TYPES
                )
            if isinstance(value, ast.Constant):
                return type(value.value) in _SCALAR_TYPES
            if isinstance(value, ast.Subscript):
                tensor = _row_subscript(value, m_name)
                if tensor is None or tensor not in tensor_names:
                    return False
                sites.append(value)
                return True
            if isinstance(value, ast.BinOp):
                return (
                    isinstance(value.op, _ELEMENTWISE_BINOPS)
                    and safe(value.left)
                    and safe(value.right)
                )
            if isinstance(value, ast.UnaryOp):
                return isinstance(value.op, _ELEMENTWISE_UNARYOPS) and safe(
                    value.operand
                )
            if isinstance(value, ast.Compare):
                return safe(value.left) and all(
                    safe(item) for item in value.comparators
                )
            if isinstance(value, ast.Attribute):
                return self.metadata(value)
            if isinstance(value, ast.Call):
                if not all(self.metadata(keyword.value) for keyword in value.keywords):
                    return False
                func = value.func
                if not isinstance(func, ast.Attribute):
                    return False
                resolved = self.global_value(func)
                if resolved is not None:
                    return (
                        func.attr in _POINTWISE_TORCH
                        and resolved is getattr(torch, func.attr, None)
                        and all(safe(arg) for arg in value.args)
                    )
                return (
                    func.attr in _POINTWISE_METHODS
                    and safe(func.value)
                    and all(self.metadata(arg) for arg in value.args)
                )
            return False

        for item in body:
            if item in operand_aliases:
                continue
            assert isinstance(item, ast.Assign)
            if not safe(item.value):
                return 0
            target = item.targets[0]
            if isinstance(target, ast.Subscript):
                sites.append(target)
        for site in sites:
            tensor = cast("ast.Name", site.value).id
            if self.tensor_ndim(tensor) != 2 or not self.known_equal(
                self.tensor_extent(tensor, 1), n_extent
            ):
                return 0
        assert isinstance(root, ExtendedAST)
        with root:
            n_name = self.fresh("_helion_fullslice_n")
            spliced = self.splice(
                statement, node, root, lhs_name, rhs_name, m_name, n_name
            )
            if spliced is None:
                return 0
            emitted, consumed = spliced
            for site in sites:
                cast("ast.Tuple", site.slice).elts[1] = create(
                    ast.Name, id=n_name, ctx=ast.Load()
                )
            if lhs_alias_live:
                assert lhs_alias is not None
                cast("ast.Tuple", cast("ast.Subscript", lhs_alias.value).slice).elts[
                    1
                ] = create(ast.Name, id=n_name, ctx=ast.Load())
                emitted[2:2] = [lhs_alias]
            root.target = create(
                ast.Tuple,
                elts=[root.target, create(ast.Name, id=n_name, ctx=ast.Store())],
                ctx=ast.Store(),
            )
            iterator.args[0] = create(
                ast.List,
                elts=[iterator.args[0], expr_from_string(f"{rhs_name}.size(1)")],
                ctx=ast.Load(),
            )
            root.body = [
                *emitted,
                *([] if consumed else [statement]),
                *body[position + 1 :],
            ]
        return 1

    def metadata(self, node: ast.AST) -> bool:
        return _metadata_expression(
            node,
            self.prelude.tensors,
            self.prelude.metadata,
            self.host,
            self.local_names,
        )


def _two_axis_root(root: ast.stmt, host: HostFunction, local_names: set[str]) -> bool:
    return (
        isinstance(root, ast.For)
        and isinstance(root.iter, ast.Call)
        and _global_value(root.iter.func, host, local_names) is hl.tile
        and len(root.iter.args) == 1
        and isinstance(root.iter.args[0], (ast.Tuple, ast.List))
        and len(root.iter.args[0].elts) == 2
        and isinstance(root.target, (ast.Tuple, ast.List))
        and len(root.target.elts) == 2
        and all(isinstance(item, ast.Name) for item in root.target.elts)
        and not root.orelse
    )


def _single_axis_root(
    root: ast.stmt, host: HostFunction, local_names: set[str]
) -> bool:
    return (
        isinstance(root, ast.For)
        and isinstance(root.iter, ast.Call)
        and _global_value(root.iter.func, host, local_names) is hl.tile
        and len(root.iter.args) == 1
        and isinstance(root.target, ast.Name)
        and not root.orelse
    )


def stripmine_full_slice_matmuls(host: HostFunction) -> int:
    """Replace eligible statements, leaving rejected source completely unchanged."""
    env = CompileEnvironment.current()
    if env.backend_name != "cute" or not env.settings.cute_full_slice_matmul_tiling:
        return 0
    local_names = _bound_names(host)
    prelude = _host_prelude(host, local_names)
    if prelude is None:
        return 0
    input_tensors = {
        name: value
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.dtype in (torch.float16, torch.bfloat16)
        and _input_binding_is_stable(name, host, local_names)
    }
    if not input_tensors:
        return 0
    rewriter = _Rewriter(
        host,
        env,
        local_names,
        prelude,
        input_tensors,
        local_names | set(host.fn.__globals__),
    )
    changed = 0
    for root in host.body:
        if _two_axis_root(root, host, local_names):
            changed += rewriter.rewrite_two_axis_root(cast("ast.For", root))
        elif _single_axis_root(root, host, local_names):
            changed += rewriter.rewrite_single_axis_root(cast("ast.For", root))
    return changed
