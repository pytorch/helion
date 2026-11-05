"""Reduction blocks that are only the power-of-two padding of another block.

A constant-shape factory in a device loop is padded to a power of two
(``tensor_utils._pad_dim``), so with ``n = hl.specialize(x.size(1)) == 40``
the carry ``acc = x.new_zeros(n)`` is 64 long.  ``acc += xt.sum(0)``, with
``xt = x[t, :]`` on the 40-long slice block A, makes the shape env guard
``A.var == 64``, and from then on every fake shape shows that dim as 64.  A
later reduction over it sees a range of 64 and allocates block B of 64 for
the same tensor dim.  Triton reduces by position, so A and B coincide.  A
Metal thread holds one index per block, so B is a *twin*: it owns no
positions, and its reductions run on A's threads and lanes, masked by A's
extent (``MetalPaddedTwinStrategy``).

The fake shapes still say 64, and anything the compiler evaluates at trace
time reads that: ``xt.size(-1)`` folds into the IR as the constant 64,
``xt.mean(-1)`` divides by 64, and a Python helper given ``xt`` sees 64.  So
a kernel with a twin is admitted only if every use of a value carrying the
padded dim is one that cannot depend on the dim's length (``_check_uses``),
and every value a twin reduction reads along that dim comes from A's real
elements (``cone_violation``).

Not covered: Python code that builds a factory of ``n`` itself and reads
its extent (a helper returning ``torch.zeros(n).numel()``) sees the padded
64 whether or not the kernel has a twin, since the factory padding is
shared code.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sympy
import torch

from ... import exc
from ..._utils import next_power_of_2
from ...language._tracing_ops import _for_loop
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import ExtendedAST
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..type_info import CallableType
from ..type_info import CollectionType
from ..type_info import NestedFunctionType
from ..type_info import TensorAttributeType
from ..type_info import TensorType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..device_ir import GraphInfo
    from ..type_info import TypeInfo

#: Torch functions and tensor methods whose result does not depend on how
#: many elements a reduced or kept dim has, besides the pointwise ops.
#: Mean, var, std and friends divide by the count; Inductor lowers them as a
#: sum over the range, so they would divide by the padded extent.
_EXTENT_FREE_CALLS = frozenset(
    {
        "amax",
        "amin",
        "argmax",
        "argmin",
        "bfloat16",
        "double",
        "float",
        "half",
        "log_softmax",
        "max",
        "min",
        "prod",
        "softmax",
        "sum",
        "to",
        "type_as",
        "unsqueeze",
    }
)
_METADATA_ATTRS = frozenset({"dtype", "device"})
#: Nodes that only pass a value on, or apply an elementwise operator to it.
_PASS_THROUGH = (
    ast.Tuple,
    ast.List,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Return,
    ast.Expr,
    ast.UnaryOp,
    ast.Compare,
)


def reduction_dims_padding_to(size: int) -> list[int]:
    """The reduction blocks whose extent is *size* or pads to it.

    A *size*-long fake dim can stand for any of them: a constant-shape
    factory pads each dim to a power of two (``tensor_utils._pad_dim``).
    """
    env = CompileEnvironment.current()
    return [
        info.block_id
        for info in env.block_sizes
        if info.reduction
        and isinstance(info.size, (int, torch.SymInt))
        and (
            info.size_matches(sympy.Integer(size))
            or (
                (hint := env.size_hint(info.size)) > 0 and next_power_of_2(hint) == size
            )
        )
    ]


def padded_twin_owner(block_id: int) -> int | None:
    """The block whose padding reduction block *block_id* is, or ``None``.

    The owner's var must be known equal to the padded size: that is the
    evidence that typing merged the two dims.  Without it *block_id* is a
    genuine dim of that size.  Raises when the kernel uses the padded dim in
    a way the twin can't cover.
    """
    env = CompileEnvironment.current()
    info = env.block_sizes[block_id]
    padded = info.size
    if not info.reduction or not isinstance(padded, int):
        return None
    owners = [
        other
        for other in reduction_dims_padding_to(padded)
        if other != block_id
        and isinstance(real := env.block_sizes[other].size, int)
        and real != padded
        and env.known_equal(env.block_sizes[other].var, padded)
    ]
    if not owners:
        return None
    if len(owners) > 1:
        raise exc.BackendUnsupported(
            "metal",
            f"reduction dim {block_id} of {padded} elements pads dims {owners}; "
            "one block cannot carry both extents' masks.  Give the `:` slices "
            "lengths that pad to different powers of two",
        )
    (owner,) = owners
    _check_memory(block_id, padded, owner)
    _check_uses(block_id, padded, owner)
    return owner


def padded_twins(owner: int) -> list[int]:
    """The reduction blocks whose padding owner is *owner*."""
    return [
        info.block_id
        for info in CompileEnvironment.current().block_sizes
        if info.block_id != owner and padded_twin_owner(info.block_id) == owner
    ]


def _describe(block_id: int, padded: int, owner: int) -> str:
    size = CompileEnvironment.current().block_sizes[owner].size
    return (
        f"a value whose dim {block_id} is dim {owner} ({size} elements) padded "
        f"to {padded}"
    )


def _slice_extents(node: torch.fx.Node) -> Iterator[tuple[int, slice, int]]:
    """``(host dim size, slice, dim)`` for each slice of a load or store."""
    host = node.args[0]
    assert isinstance(host, torch.fx.Node)
    fake = host.meta["val"]
    assert isinstance(fake, torch.Tensor)
    subscript = node.args[1]
    assert isinstance(subscript, (list, tuple))
    dim = 0
    for entry in subscript:
        if entry is None:
            continue
        if isinstance(entry, slice):
            yield fake.size(dim), entry, dim
        dim += 1


def _slice_length(size: int | torch.SymInt, entry: slice) -> int | None:
    start, stop, step = entry.start or 0, entry.stop, entry.step or 1
    if stop is None:
        stop = size
    if not all(isinstance(v, int) for v in (start, stop, step, size)):
        return None
    return len(range(start, min(stop, size), step))


def _check_memory(block_id: int, padded: int, owner: int) -> None:
    """Where memory has a dim of the padded length, the twin's positions
    are real elements, not padding."""
    for graph in HostFunction.current().device_ir.graphs:
        for node in graph.graph.nodes:
            if node.op != "call_function" or node.target not in (load, store):
                continue
            for size, entry, dim in _slice_extents(node):
                if _slice_length(size, entry) == padded:
                    host = node.args[0]
                    assert isinstance(host, torch.fx.Node)
                    raise exc.BackendUnsupported(
                        "metal",
                        f"reduction dim {block_id} pads dim {owner} to {padded} "
                        f"elements, but {host.name} dim {dim} has {padded} real "
                        "elements; give the `:` slices lengths that don't pad "
                        "to another slice's length",
                    )


def _is_pointwise(name: str) -> bool:
    packet = getattr(torch.ops.aten, name, None)
    return isinstance(packet, torch._ops.OpOverloadPacket) and any(
        torch.Tag.pointwise in getattr(packet, overload).tags
        for overload in packet.overloads()
    )


def _extent_free(name: str) -> bool:
    return name in _EXTENT_FREE_CALLS or _is_pointwise(name)


def _callee_admits(func: ast.expr) -> bool:
    """Whether the callee may take a padded value as an argument: a nested
    function, whose body the walk checks too, or a C-implemented torch
    function or tensor method that is extent-free.

    The identity check matters: a module-level helper named ``amax`` runs
    as Python on the fake tensors.
    """
    assert isinstance(func, ExtendedAST)
    type_info = func._type_info
    if isinstance(type_info, NestedFunctionType):
        return True
    if isinstance(type_info, TensorAttributeType):
        name = type_info.attr()
        fn = getattr(torch.Tensor, name)
    elif isinstance(type_info, CallableType):
        fn = type_info.value
        name = getattr(fn, "__name__", None)
    else:
        return False
    if name is None:
        return False
    genuine = fn is getattr(torch, name, None) or fn is getattr(
        torch.Tensor, name, None
    )
    return genuine and _extent_free(name)


def _padded_tensor(type_info: TypeInfo | None, padded: int) -> bool:
    """Host tensors keep their real sizes; only device values are padded."""
    env = CompileEnvironment.current()
    return (
        isinstance(type_info, TensorType)
        and type_info.origin.is_device()
        and any(env.known_equal(d, padded) for d in type_info.fake_value.shape)
    )


def _holds_padded(type_info: TypeInfo | None, padded: int) -> bool:
    if _padded_tensor(type_info, padded):
        return True
    if not isinstance(type_info, CollectionType):
        return False
    elements = type_info.element_types
    if isinstance(elements, dict):
        values = list(elements.values())
    elif isinstance(elements, slice):
        values = [elements.start, elements.stop, elements.step]
    else:
        values = list(elements)
    return any(_holds_padded(value, padded) for value in values)


def _carries_padded(node: ast.AST, padded: int) -> bool:
    return isinstance(node, ExtendedAST) and _holds_padded(node._type_info, padded)


def _use_problem(parent: ast.AST, child: ast.expr, callees: set[int]) -> str | None:
    """``None`` if *parent* uses the padded value *child* in a way that can't
    read its extent, else what it does with it."""
    if isinstance(parent, (ast.Call, ast.keyword, ast.Starred)):
        # Call arguments are judged per call, with their callee.
        return None
    if isinstance(parent, ast.Attribute):
        if id(parent) in callees:
            return None if _extent_free(parent.attr) else f"calls .{parent.attr}()"
        return None if parent.attr in _METADATA_ATTRS else f"reads .{parent.attr}"
    if isinstance(parent, ast.Subscript):
        # Subscripting it can't read the extent: a kernel tensor takes only
        # ``:`` and ``None`` (``Backend.fake_subscript_shape``).
        return "indexes with it" if child is not parent.value else None
    if isinstance(parent, ast.BinOp):
        return "contracts it with @" if isinstance(parent.op, ast.MatMult) else None
    if isinstance(parent, _PASS_THROUGH):
        return None
    return f"uses it in a {type(parent).__name__}"


def _check_uses(block_id: int, padded: int, owner: int) -> None:
    """Every use of a device value carrying the padded dim must be one that
    can't read the dim's length: an elementwise operator, an extent-free
    torch call or method, a nested function, a ``.dtype``/``.device`` read,
    a subscript of it, or a pass-through.

    Size queries run at trace time and fold into constants, so no device-IR
    scan could find them; this walks the kernel's source instead.
    """
    nodes = list(
        ast.walk(ast.Module(body=HostFunction.current().body, type_ignores=[]))
    )
    callees = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    for parent in nodes:
        if isinstance(parent, ast.Call):
            values = [
                arg.value if isinstance(arg, ast.Starred) else arg
                for arg in parent.args
            ]
            values += [keyword.value for keyword in parent.keywords]
            if any(_carries_padded(value, padded) for value in values) and not (
                _callee_admits(parent.func)
            ):
                raise exc.BackendUnsupported(
                    "metal",
                    f"line {parent.lineno} passes "
                    f"{_describe(block_id, padded, owner)} to "
                    f"{ast.unparse(parent.func)}(); only pointwise torch ops, "
                    f"{', '.join(sorted(_EXTENT_FREE_CALLS))} and nested "
                    "functions may take it",
                )
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, ast.expr) or not _carries_padded(child, padded):
                continue
            problem = _use_problem(parent, child, callees)
            if problem is not None:
                raise exc.BackendUnsupported(
                    "metal",
                    f"line {child.lineno} {problem} on "
                    f"{_describe(block_id, padded, owner)}, so it would see "
                    f"{padded} elements; only elementwise ops, extent-free "
                    "reductions and dtype casts may take it",
                )


def _padded_dims(node: torch.fx.Node, padded: int) -> int:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return 0
    return sum(1 for size in value.shape if isinstance(size, int) and size == padded)


def _loop_input(
    placeholder: torch.fx.Node, graphs: list[GraphInfo]
) -> torch.fx.Node | None:
    """The argument of the single ``_for_loop`` that feeds *placeholder*."""
    (graph,) = [g for g in graphs if g.graph is placeholder.graph]
    placeholders = [n for n in placeholder.graph.nodes if n.op == "placeholder"]
    index = placeholders.index(placeholder)
    callers = [
        node
        for g in graphs
        for node in g.graph.nodes
        if node.op == "call_function"
        and node.target is _for_loop
        and node.args[0] == graph.graph_id
    ]
    if len(callers) != 1:
        return None
    args = callers[0].args[3]
    assert isinstance(args, (list, tuple))
    source = args[index] if index < len(args) else None
    return source if isinstance(source, torch.fx.Node) else None


def cone_violation(
    node: torch.fx.Node, padded: int, real: int, graphs: list[GraphInfo]
) -> str | None:
    """``None`` if every value under *node*'s inputs that carries the padded
    dim comes from a full slice of a host dim of *real* elements, else the
    first value that doesn't.

    Only those positions past *real* are padding that the owner's mask may
    drop.  A factory of the padded size (``hl.full([t, 64], 1.0)``) holds
    real elements there.  A value with no padded dim broadcasts along it.
    """
    seen: set[torch.fx.Node] = set()
    stack = list(reversed(node.all_input_nodes))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        count = _padded_dims(current, padded)
        if current.op == "placeholder":
            if count == 0:
                continue
            source = _loop_input(current, graphs)
            if source is None:
                return f"{current.name} enters from outside its loop or branch"
            stack.append(source)
            continue
        if count == 0:
            continue
        if count > 1:
            return f"{current.name} has the padded dim twice"
        if current.target is load:
            subscript = current.args[1]
            assert isinstance(subscript, (list, tuple))
            if any(
                isinstance(entry, torch.fx.Node)
                and isinstance(entry.meta.get("val"), torch.Tensor)
                for entry in subscript
            ):
                return f"{current.name} is an indexed load"
            if not any(
                size == real and _slice_length(size, entry) == real
                for size, entry, _ in _slice_extents(current)
            ):
                return f"{current.name} loads no full slice of {real} elements"
            continue
        inputs = [
            i
            for i in current.all_input_nodes
            if i.op == "placeholder" or _padded_dims(i, padded)
        ]
        if not inputs:
            return f"{current.name} creates the padded dim"
        stack.extend(reversed(inputs))
    return None
