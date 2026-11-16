"""Dims of length ``c * block`` held as ``c`` per-thread parts.

A Metal thread holds one element of a value per block coordinate, so a dim
of length ``c * u`` (``c`` a literal int >= 2, ``u`` a block) has no thread
layout of its own.  When it is the row-major merge of ``(u, c)``, element
``j = c * k + r`` sits at the thread's index ``k`` of ``u``, so the thread
holds it as ``c`` parts: part ``r`` is element ``c * k + r``.  Each lowering
hands the parts to its consumers' lowerings as a :class:`FactoredValue`, not
an AST (``GraphInterpreter.run_node`` passes a non-AST result through), the
way CuTe passes ``CuteAffineRangeIndex`` to its loads and stores.

Sources:

- an iota widening a tile, ``iota(c * u, start=c * tile_begin(u))``
  (:func:`factored_iota`), CuTe's ``match_cute_affine_range_iota`` form;
- a load at such an iota, one load per part (:func:`factored_load`);
- ``stack(ts, dim=d + 1)`` merged by a view into dim ``d`` of block ``u``
  (:func:`stacked`, :func:`merged_stack`), CuTe's
  ``match_cute_stack_reshape_rhs`` form.

The consumers are a store at a factored iota over the same block and
factor, one store per part (:func:`factored_store`), and a matmul the MPP
planner lowers (``mpp_matmul``), which reads a widened slice as a device
view or stages the parts.  Every producer rejects any other consumer; any
other matmul gets its own message, since lowering one over parts belongs to
the MPP lowering.
"""

from __future__ import annotations

import ast
import dataclasses
from itertools import starmap
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx.node import Node

from ... import exc
from ...language import _tracing_ops
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.tile_ops import tile_begin
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..generate_ast import GenerateAST
from ..indexing_strategy import SubscriptIndexing
from .free_arange import MATMUL_LHS
from .free_arange import check_lane_carries
from .free_arange import live_graphs
from .positional_rebind import canonical_block

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..aten_lowering import LoweringContext
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState

_VIEW_TARGETS = (torch.ops.aten.view.default, torch.ops.aten.reshape.default)

_MATMUL_OVER_PARTS = (
    "a matmul over a dim held as per-thread parts (a widened hl.arange() slice "
    "or a merged torch.stack) that the cooperative MPP lowering doesn't admit"
)


def _unsupported(message: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("metal", message)


@dataclasses.dataclass(frozen=True)
class FactoredValue:
    """A value whose dim ``dim`` has length ``factor * block`` and is held as
    ``factor`` parts: part ``r`` is element ``factor * k + r`` at the
    thread's index ``k`` of ``block_id``.  ``part_fake`` is one part's fake
    value, the value with that dim of the block's length."""

    dim: int
    block_id: int
    factor: int
    parts: tuple[ast.AST, ...]
    part_fake: torch.Tensor
    #: For an iota, the end of the loop over ``block_id``; the parts cover
    #: ``[factor * begin, factor * end)``.  None for a load or a merged stack.
    end: int | torch.SymInt | None = None


@dataclasses.dataclass(frozen=True)
class StackedValue:
    """``torch.stack(parts, dim)``, waiting for the view that merges it."""

    dim: int
    parts: tuple[ast.AST, ...]
    part_fake: torch.Tensor


def factored_size(size: object) -> tuple[int, int] | None:
    """``(u, c)`` for a dim of length ``c * u``, ``c`` a literal int >= 2 and
    ``u`` a block, else None."""
    if not isinstance(size, torch.SymInt):
        return None
    expr = size._sympy_()
    if not (isinstance(expr, sympy.Mul) and len(expr.args) == 2):
        return None
    c, sym = expr.args
    if not (isinstance(c, sympy.Integer) and int(c) >= 2):
        return None
    block_id = CompileEnvironment.current().get_block_id(sym)
    return None if block_id is None else (block_id, int(c))


def _check_value_users(node: Node, what: str) -> None:
    """Only a store that takes *node* as its value, or a matmul the MPP
    planner lowers, may consume parts."""
    from .mpp_matmul import planned

    for user in node.users:
        if (
            user.target is store
            and user.args[2] is node
            and node not in user.args[3:]
            and node not in _flat(user.args[1])
        ):
            continue
        if planned(user):
            continue
        shape = list(node.meta["val"].shape)
        if user.target in MATMUL_LHS:
            raise _unsupported(
                f"{_MATMUL_OVER_PARTS}; {what} {node.name} of shape {shape} feeds "
                f"{user.name}"
            )
        raise _unsupported(
            f"{what} {node.name} of shape {shape}, which Metal holds as per-thread "
            f"parts, feeds {user.name}; only a store at a widened hl.arange() "
            "index can consume it"
        )


def _flat(indices: object) -> list[object]:
    return [*indices] if isinstance(indices, (list, tuple)) else [indices]


def _constant_multiple(value: object) -> tuple[object, int]:
    """``(base, c)`` for ``base * c`` with a literal int ``c > 0``, else
    ``(value, 1)``."""
    if (
        isinstance(value, Node)
        and value.op == "call_function"
        and value.target is operator.mul
    ):
        lhs, rhs = value.args
        if isinstance(lhs, int) and lhs > 0:
            return rhs, lhs
        if isinstance(rhs, int) and rhs > 0:
            return lhs, rhs
    return value, 1


def _loop_end(
    graphs: list[GraphInfo], node: Node, block_id: int
) -> int | torch.SymInt | None:
    """The end of *block_id*'s device loop when *node* is in that loop's own
    body graph, else None."""
    from ..device_ir import ForLoopGraphInfo

    info = next((g for g in graphs if g.graph is node.graph), None)
    if not (isinstance(info, ForLoopGraphInfo) and block_id in info.block_ids):
        return None
    position = info.block_ids.index(block_id)
    for graph in graphs:
        for loop in graph.graph.nodes:
            if (
                _tracing_ops.is_for_loop_target(loop.target)
                and loop.args[0] == info.graph_id
            ):
                end = loop.args[2][position]
                return end.meta["val"] if isinstance(end, Node) else end
    return None


def widened_iota(node: Node) -> tuple[int, int] | None:
    """``(u, c)`` for ``iota(c * u, start=c * tile_begin(u), step=1)``, the
    affine range of tile ``u`` widened by ``c`` (CuTe's
    ``match_cute_affine_range_iota``), else None."""
    if node.target is not torch.ops.prims.iota.default:
        return None
    if node.kwargs.get("step", 1) != 1:
        return None
    (length,) = node.args
    base, factor = _constant_multiple(length)
    start_base, start_factor = _constant_multiple(node.kwargs.get("start", 0))
    if not (
        factor >= 2
        and start_factor == factor
        and isinstance(start_base, Node)
        and start_base.target is tile_begin
        and start_base.args[0] is base
    ):
        return None
    assert isinstance(base, Node)
    block_id = CompileEnvironment.current().get_block_id(base.meta["val"])
    assert block_id is not None
    return block_id, factor


def factored_iota(ctx: LoweringContext, node: Node) -> FactoredValue | None:
    """``iota(c * u, start=c * tile_begin(u), step=1)`` as ``c`` parts
    ``c * index(u) + r``, or None for any other iota.

    Element ``j = c * k + r`` is ``c * begin + c * k + r``, which is
    ``c * (begin + k) + r``, and ``begin + k`` is the thread's index of ``u``.
    """
    widened = widened_iota(node)
    if widened is None:
        return None
    block_id, factor = widened
    env = CompileEnvironment.current()
    base, _ = _constant_multiple(node.args[0])
    assert isinstance(base, Node)
    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    end = _loop_end(cg.codegen_graphs, node, block_id)
    if end is None:
        raise _unsupported(
            f"hl.arange() {node.name} widens tile {block_id} by {factor}, but "
            "not in the body of that tile's own device loop"
        )
    for user in node.users:
        if not (
            user.target in (load, store)
            and node in _flat(user.args[1])
            and node not in user.args[2:]
        ):
            raise _unsupported(
                f"hl.arange() {node.name} widens tile {block_id} by {factor} and "
                f"feeds {user.name}; Metal holds it as per-thread parts, which "
                "only a load's or a store's index can consume"
            )
    backend = env.backend
    dtype = node.kwargs.get("dtype")
    dtype_str = backend.dtype_str(
        dtype if isinstance(dtype, torch.dtype) else env.index_dtype
    )
    index = cg.index_var(block_id)
    parts = tuple(
        cg.lift(
            expr_from_string(backend.cast_expr(f"{factor} * {index} + {r}", dtype_str)),
            dce=True,
            prefix="iota_part",
        )
        for r in range(factor)
    )
    part_fake = node.meta["val"].new_empty([base.meta["val"]])
    return FactoredValue(0, block_id, factor, parts, part_fake, end)


def _factored_index(
    subscript: list[object], ast_subscript: list[object], access: str
) -> tuple[int, FactoredValue]:
    """The position and value of the one factored iota in a subscript; any
    other tensor index raises."""
    pos = next(i for i, k in enumerate(ast_subscript) if isinstance(k, FactoredValue))
    index = ast_subscript[pos]
    assert isinstance(index, FactoredValue)
    if index.end is None:
        raise _unsupported(
            f"a {access} indexed by a value Metal holds as per-thread parts; only "
            "a widened hl.arange() can index"
        )
    if any(isinstance(k, torch.Tensor) for i, k in enumerate(subscript) if i != pos):
        raise _unsupported(
            f"a {access} indexed by a widened hl.arange() and another tensor"
        )
    return pos, index


def _check_extent(
    tensor: torch.Tensor,
    subscript: list[object],
    pos: int,
    index: FactoredValue,
    access: str,
) -> None:
    """Eager clips a widened slice at the tensor's end, while the parts cover
    ``[c * begin, c * end)`` masked by the loop's mask; they agree only when
    the indexed dim is ``c * end``."""
    env = CompileEnvironment.current()
    assert index.end is not None
    tensor_dim = sum(1 for k in subscript[:pos] if k is not None)
    size = tensor.size(tensor_dim)
    if env.known_equal(size, index.factor * index.end):
        return
    hint = (
        ""
        if isinstance(size, int) and isinstance(index.end, int)
        else (
            ", and can't prove it of symbolic sizes; pass static_shapes=True and "
            "hl.specialize() any int argument the sizes come from"
        )
    )
    raise _unsupported(
        f"a {access} of dim {tensor_dim} (size {size}) at an hl.arange() widening "
        f"tile {index.block_id} by {index.factor}, whose loop ends at "
        f"{index.end}; Metal needs the dim to be exactly {index.factor} x the "
        f"loop's end{hint}"
    )


def _part_shape(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object],
    pos: int,
    index: FactoredValue,
    access: str,
) -> tuple[list[object], list[int | torch.SymInt], int]:
    """The subscript one part accesses, its shape, and the shape's factored
    dim.  A thread holds one element per coordinate of a block, so no other
    dim of the part shape may be over the factored block (the rule
    ``plan_repeated_blocks`` applies to traced shapes, which never show this
    one)."""
    env = CompileEnvironment.current()
    part_subscript = [*subscript]
    part_subscript[pos] = index.part_fake
    full_shape = SubscriptIndexing.compute_shape(tensor, subscript, state)
    part_shape = SubscriptIndexing.compute_shape(tensor, part_subscript, state)
    (dim,) = [
        i
        for i, (full, part) in enumerate(zip(full_shape, part_shape, strict=True))
        if not env.known_equal(full, part)
    ]
    others = [
        i
        for i, size in enumerate(part_shape)
        if i != dim and env.get_block_id(size) == index.block_id
    ]
    if others:
        raise _unsupported(
            f"a {access} at an hl.arange() widening tile {index.block_id} by "
            f"{index.factor} that tile {index.block_id} also indexes in another "
            f"dim (dims {[dim, *others]} of the part shape {list(part_shape)}); a "
            "thread holds one element per coordinate of a tile"
        )
    return part_subscript, part_shape, dim


def _with_part(
    state: CodegenState,
    part_subscript: list[object],
    ast_subscript: list[object],
    pos: int,
    part: ast.AST,
    value: ast.AST | None = None,
) -> CodegenState:
    part_ast_subscript = [*ast_subscript]
    part_ast_subscript[pos] = part
    proxy_args = [state.proxy_args[0], part_subscript, *state.proxy_args[2:]]
    ast_args = [state.ast_args[0], part_ast_subscript, *state.ast_args[2:]]
    if value is not None:
        ast_args[2] = value
    return state._replace(proxy_args=proxy_args, ast_args=ast_args)


def factored_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object],
    ast_subscript: list[object],
    load_part: Callable[[CodegenState, list[object]], ast.AST],
) -> FactoredValue:
    """One load per part of a factored iota index.

    Each part's load is *load_part*, the ordinary load, with the part as a
    ``[u]`` tensor index, so it is masked by ``u``'s mask like any index
    tensor over ``u``.
    """
    pos, index = _factored_index(subscript, ast_subscript, "load")
    _check_extent(tensor, subscript, pos, index, "load")
    part_subscript, part_shape, dim = _part_shape(
        state, tensor, subscript, pos, index, "load"
    )
    assert state.fx_node is not None
    _check_value_users(state.fx_node, "the load")
    parts = tuple(
        state.codegen.lift(
            load_part(
                _with_part(state, part_subscript, ast_subscript, pos, part),
                part_subscript,
            ),
            dce=True,
            prefix="load_part",
        )
        for part in index.parts
    )
    fake = state.fake_value
    assert isinstance(fake, torch.Tensor)
    return FactoredValue(
        dim, index.block_id, index.factor, parts, fake.new_empty(part_shape)
    )


def factored_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object],
    ast_subscript: list[object],
    value: object,
    store_part: Callable[[CodegenState, list[object], ast.AST], ast.AST],
) -> None:
    """One store per part: value part ``r`` at index part ``r``, each the
    ordinary store (*store_part*) with the part as a ``[u]`` tensor index, so
    every block's mask applies."""
    if not any(isinstance(k, FactoredValue) for k in ast_subscript):
        raise _unsupported(
            "a store of a value Metal holds as per-thread parts at an index "
            "without a widened hl.arange()"
        )
    _check_kernel(state)
    pos, index = _factored_index(subscript, ast_subscript, "store")
    _check_extent(tensor, subscript, pos, index, "store")
    if not isinstance(value, FactoredValue):
        raise _unsupported(
            "a store at a widened hl.arange() index of a value Metal doesn't hold "
            "as per-thread parts (only a widened load or a merged stack)"
        )
    if (value.block_id, value.factor) != (index.block_id, index.factor):
        raise _unsupported(
            f"a store of a value widening tile {value.block_id} by {value.factor} "
            f"at an hl.arange() widening tile {index.block_id} by {index.factor}; "
            "Metal pairs their parts only over the same tile and factor"
        )
    part_subscript, part_shape, dim = _part_shape(
        state, tensor, subscript, pos, index, "store"
    )
    _check_aligned(value, part_shape, dim)
    for value_part, index_part in zip(value.parts, index.parts, strict=True):
        part_state = _with_part(
            state, part_subscript, ast_subscript, pos, index_part, value_part
        )
        state.add_statement(
            create(ast.Expr, value=store_part(part_state, part_subscript, value_part))
        )


def _same_layout(size: int | torch.SymInt, other: int | torch.SymInt) -> bool:
    """Whether two dims lay their elements out over the same threads: the
    same canonical block when either has one (two tiles with equal literal
    block sizes have equal sizes but different threads), else equal sizes."""
    block, other_block = canonical_block(size), canonical_block(other)
    if block is not None or other_block is not None:
        return block == other_block
    return CompileEnvironment.current().known_equal(size, other)


def _check_aligned(
    value: FactoredValue, part_shape: list[int | torch.SymInt], dim: int
) -> None:
    """The value broadcasts to the store's part shape, right-aligned as any
    broadcast is, each dim of the same layout as the store dim it faces, and
    its factored dim facing the index's.  Helion's store tracing checks only
    ranks, so without this a value dim could face a ``:`` dim, another tile,
    or the factored dim of the index."""
    env = CompileEnvironment.current()
    value_shape = [*value.part_fake.shape]
    offset = len(part_shape) - len(value_shape)
    if (
        offset < 0
        or value.dim + offset != dim
        or not all(
            env.known_equal(size, 1) or _same_layout(size, part_shape[i + offset])
            for i, size in enumerate(value_shape)
        )
    ):
        raise _unsupported(
            f"a store of a value of part shape {value_shape} widened in dim "
            f"{value.dim} at a part shape {list(part_shape)} widened in dim {dim}; "
            "Metal needs each value dim to be 1 or over the same block as the "
            "store's, and the widened dims to face each other"
        )


def _check_kernel(state: CodegenState) -> None:
    """``plan_free_aranges``' admission rule, for the same reason: a kernel
    with a lane-looped carry that isn't a matmul over the loop's block is
    wrong at some layouts beside per-thread code."""
    graphs = state.codegen.codegen_graphs
    check_lane_carries(
        live_graphs(graphs),
        state.tile_strategy,
        state.config,
        lambda why: _unsupported(f"a store at a widened hl.arange() index: {why}"),
    )


def stacked(ctx: LoweringContext, node: Node) -> StackedValue:
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    assert isinstance(tensors, (list, tuple)) and isinstance(dim, int)
    out = node.meta["val"]
    if not all(user.target in _VIEW_TARGETS for user in node.users):
        raise _unsupported(
            f"aten.stack {node.name} whose result isn't merged by a view into the "
            "dim before the stacked one"
        )
    if len(tensors) < 2:
        raise _unsupported("aten.stack of a single tensor")
    parts = []
    for tensor in tensors:
        assert isinstance(tensor, Node)
        part = ctx.env[tensor]
        assert isinstance(part, ast.AST)
        parts.append(part)
    first = tensors[0]
    assert isinstance(first, Node)
    # The view and the store see only input 0's shape, so every other input
    # must hold its elements on the same threads.
    first_shape = [*first.meta["val"].shape]
    for tensor in tensors[1:]:
        assert isinstance(tensor, Node)
        shape = [*tensor.meta["val"].shape]
        if not all(starmap(_same_layout, zip(first_shape, shape, strict=True))):
            raise _unsupported(
                f"aten.stack {node.name} of inputs of shapes {first_shape} and "
                f"{shape} over different blocks; Metal stacks parts only of one "
                "thread layout"
            )
    # No promotion cast: the parts' one consumer is a store, which converts
    # to the target's dtype, and a target of another dtype than the stack's
    # traces a convert_element_type, which _check_value_users rejects.
    return StackedValue(dim % out.ndim, tuple(parts), first.meta["val"].to(out.dtype))


def merged_stack(node: Node, value: StackedValue) -> FactoredValue:
    """A view of ``stack(ts, dim=d)`` merging dims ``d - 1`` (a block ``u``)
    and ``d`` into one of length ``c * u``, row-major: index ``c * k + r`` is
    ``ts[r]`` at ``k``."""
    env = CompileEnvironment.current()
    source = node.args[0]
    assert isinstance(source, Node)
    in_shape = [*source.meta["val"].shape]
    out_shape = [*node.meta["val"].shape]
    d = value.dim
    c = len(value.parts)
    block_id = env.get_block_id(in_shape[d - 1]) if d >= 1 else None
    merged = [*in_shape[: d - 1], c * in_shape[d - 1], *in_shape[d + 1 :]]
    if not (
        block_id is not None
        and len(out_shape) == len(merged)
        and all(starmap(env.known_equal, zip(out_shape, merged, strict=True)))
    ):
        raise _unsupported(
            f"a view of a stack from {in_shape} to {out_shape}; Metal only merges "
            "the stacked dim into a block dim right before it"
        )
    _check_value_users(node, "the view")
    return FactoredValue(d - 1, block_id, c, value.parts, value.part_fake)
