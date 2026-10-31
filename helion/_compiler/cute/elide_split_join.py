"""Elide a split/join round trip over a register tile: ``f(y)`` in place of
``reshape(permute(hl.join(f(lo), f(hi))))`` where ``(lo, hi) = hl.split(permute(reshape(y)))``.

Kernels subtile a register tile for Triton's benefit: reshape one dim of the
tile into a pair, move the pair dim last, ``hl.split`` it, apply the same
elementwise op to both halves, ``hl.join`` them and view the result back into
the tile's shape (``blackwell_attention``'s accumulator scaled by ``alpha``;
``torch.chunk``/``unbind`` lower the same way).  On CuTe the split of a
non-load tile stages the whole tile through shared memory behind a barrier,
and is refused when its lane-loop proof cannot settle the exchange, while
the round trip is the identity: every element ends up where it started, so
the pair never has to leave its thread.

For an elementwise ``f`` whose other operands are shared by both halves and
broadcast along the split dim (size one there, or absent), the whole chain is
``f(y, z...)`` on the source tile: the pass replaces the final view with that
node, and the chain, the split, its ``getitem`` nodes and the two ``f`` nodes
go away when nothing else uses them.  Only pure elementwise ops qualify
(``inline_asm_elementwise`` with ``is_pure``, and ATen pointwise ops), and
operands count as shared when they are the same node or the same pure view
of the same node (``alpha[:, None]`` written twice).  Half-specific operands,
a different op per half, the halves joined in the other order, or a join
consumed by anything but the inverse views leave the graph alone.
"""

from __future__ import annotations

import operator
from typing import Any
from typing import cast

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map

from ...language import view_ops
from ...language.inline_asm_ops import inline_asm_elementwise

# Views the shared-operand comparison may look through.
_PURE_VIEW_TARGETS = frozenset(
    {
        view_ops.subscript,
        operator.getitem,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
    }
)
_EQUIVALENCE_DEPTH = 4


def _is_elementwise(node: torch.fx.Node) -> bool:
    target = node.target
    if target is inline_asm_elementwise:
        # (asm, constraints, args, dtype, is_pure, pack)
        return len(node.args) >= 5 and node.args[4] is True
    tags = getattr(target, "tags", ())
    return torch.Tag.pointwise in tags


def _equivalent(left: object, right: object, depth: int = _EQUIVALENCE_DEPTH) -> bool:
    """Whether two argument leaves denote the same value: the same node, the
    same constant, or the same pure view applied to equivalent arguments."""
    if left is right:
        return True
    if isinstance(left, torch.fx.Node) or isinstance(right, torch.fx.Node):
        if not (isinstance(left, torch.fx.Node) and isinstance(right, torch.fx.Node)):
            return False
        if (
            depth <= 0
            or left.op != "call_function"
            or right.op != "call_function"
            or left.target is not right.target
            or left.target not in _PURE_VIEW_TARGETS
            or len(left.args) != len(right.args)
            or left.kwargs.keys() != right.kwargs.keys()
        ):
            return False
        return all(
            _equivalent(a, b, depth - 1)
            for a, b in zip(left.args, right.args, strict=True)
        ) and all(
            _equivalent(left.kwargs[k], right.kwargs[k], depth - 1) for k in left.kwargs
        )
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except Exception:
        return False


def _tensor_val(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


def _half_of(node: object) -> tuple[torch.fx.Node, int] | None:
    """``(split_node, index)`` when ``node`` is ``getitem(hl.split(y), index)``."""
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is operator.getitem
        and len(node.args) == 2
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].op == "call_function"
        and node.args[0].target is view_ops.split
        and node.args[1] in (0, 1)
    ):
        return node.args[0], int(node.args[1])  # pyrefly: ignore[bad-argument-type]
    return None


_PERMUTE = torch.ops.aten.permute.default
_CLONE = torch.ops.aten.clone.default
_VIEWS = frozenset({torch.ops.aten.view.default, torch.ops.aten.reshape.default})


def _only_user(node: torch.fx.Node) -> torch.fx.Node | None:
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _shape(node: object) -> tuple[object, ...] | None:
    val = _tensor_val(node)
    return tuple(val.shape) if val is not None else None


def _same_shape(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    if len(left) != len(right):
        return False
    try:
        return all(bool(a == b) for a, b in zip(left, right, strict=True))
    except Exception:
        return False


def _split_dim(source: tuple[object, ...], viewed: tuple[object, ...]) -> int | None:
    """The index of the one source dim the view splits into a factor of two
    and its half (either order); ``None`` for any other reshape."""
    if len(viewed) != len(source) + 1:
        return None
    for i in range(len(source)):
        head = _same_shape(source[:i], viewed[:i])
        tail = _same_shape(source[i + 1 :], viewed[i + 2 :])
        if not (head and tail):
            continue
        a, b = cast("Any", viewed[i]), cast("Any", viewed[i + 1])
        try:
            if (a == 2 and b * 2 == source[i]) or (b == 2 and a * 2 == source[i]):
                return i
        except Exception:
            return None
    return None


def _match(
    join: torch.fx.Node,
) -> (
    tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]
    | None
):
    """``(f_lo, f_hi, split, source, final)`` for the round trip
    ``final = view(clone?(permute(join(f(lo, z), f(hi, z)))))`` over
    ``(lo, hi) = split(permute(view(source)))``."""
    if len(join.args) != 2:
        return None
    f_lo, f_hi = join.args
    if not (isinstance(f_lo, torch.fx.Node) and isinstance(f_hi, torch.fx.Node)):
        return None
    if (
        f_lo is f_hi
        or f_lo.op != "call_function"
        or f_hi.op != "call_function"
        or f_lo.target is not f_hi.target
        or not _is_elementwise(f_lo)
        or f_lo.kwargs.keys() != f_hi.kwargs.keys()
    ):
        return None
    lo_leaves, lo_spec = tree_flatten((f_lo.args, f_lo.kwargs))
    hi_leaves, hi_spec = tree_flatten((f_hi.args, f_hi.kwargs))
    if lo_spec != hi_spec:
        return None
    split: torch.fx.Node | None = None
    shared: list[torch.fx.Node] = []
    for a, b in zip(lo_leaves, hi_leaves, strict=True):
        if _equivalent(a, b):
            if _tensor_val(a) is not None:
                shared.append(a)  # pyrefly: ignore[bad-argument-type]
            continue
        half_a, half_b = _half_of(a), _half_of(b)
        if (
            split is not None
            or half_a is None
            or half_b is None
            or half_a[0] is not half_b[0]
            or half_a[1] != 0
            or half_b[1] != 1
        ):
            return None
        split = half_a[0]
    if split is None:
        return None
    # Forward chain: source -> view (one dim split into a pair) -> permute
    # (the pair factor moved last) -> split.
    permute_fwd = split.args[0]
    if not (
        isinstance(permute_fwd, torch.fx.Node)
        and permute_fwd.op == "call_function"
        and permute_fwd.target is _PERMUTE
        and isinstance(permute_fwd.args[0], torch.fx.Node)
    ):
        return None
    view_fwd = permute_fwd.args[0]
    if not (
        view_fwd.op == "call_function"
        and view_fwd.target in _VIEWS
        and isinstance(view_fwd.args[0], torch.fx.Node)
    ):
        return None
    source = view_fwd.args[0]
    source_shape, viewed_shape = _shape(source), _shape(view_fwd)
    pair_shape, out_shape, half_shape = _shape(permute_fwd), _shape(join), _shape(f_lo)
    if None in (source_shape, viewed_shape, pair_shape, out_shape, half_shape):
        return None
    assert source_shape is not None and viewed_shape is not None
    assert pair_shape is not None and out_shape is not None and half_shape is not None
    split_dim = _split_dim(source_shape, viewed_shape)
    if split_dim is None:
        return None
    perm = permute_fwd.args[1]
    if not isinstance(perm, (list, tuple)):
        return None
    perm = [int(p) for p in perm]  # pyrefly: ignore[bad-argument-type]
    rank = len(viewed_shape)
    pair_factor = split_dim if viewed_shape[split_dim] == 2 else split_dim + 1
    others = [i for i in range(rank) if i != pair_factor]
    if perm != [*others, pair_factor]:
        return None
    if not (
        _same_shape(out_shape, pair_shape)
        and _same_shape(half_shape, pair_shape[:-1])
        and pair_shape[-1] == 2
    ):
        return None
    # Shared operands must broadcast along the split dim: size one there, or
    # too low a rank to reach it (right-aligned broadcasting).
    half_rank = len(half_shape)
    for operand in shared:
        shape = _shape(operand)
        assert shape is not None
        offset = half_rank - len(shape)
        if offset < 0:
            return None
        position = split_dim - offset
        if position >= 0:
            try:
                if not bool(shape[position] == 1):
                    return None
            except Exception:
                return None
    # Inverse chain: join -> permute (inverse) -> clone? -> view to the source shape.
    permute_inv = _only_user(join)
    if not (
        permute_inv is not None
        and permute_inv.op == "call_function"
        and permute_inv.target is _PERMUTE
        and isinstance(permute_inv.args[1], (list, tuple))
    ):
        return None
    inverse = [0] * rank
    for position, axis in enumerate(perm):
        inverse[axis] = position
    if [int(p) for p in permute_inv.args[1]] != inverse:  # pyrefly: ignore[bad-argument-type]
        return None
    final = _only_user(permute_inv)
    if final is not None and final.op == "call_function" and final.target is _CLONE:
        final = _only_user(final)
    if not (
        final is not None
        and final.op == "call_function"
        and final.target in _VIEWS
        and (final_shape := _shape(final)) is not None
        and _same_shape(final_shape, source_shape)
    ):
        return None
    return f_lo, f_hi, split, source, final


def _rewrite(
    graph: torch.fx.Graph,
    match: tuple[
        torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node
    ],
) -> None:
    f_lo, f_hi, split, source, final = match
    source_val = _tensor_val(source)
    f_val = _tensor_val(f_lo)
    assert source_val is not None and f_val is not None

    def operand(leaf: object) -> object:
        return source if _half_of(leaf) is not None else leaf

    with graph.inserting_before(final):
        args, kwargs = tree_map(operand, (f_lo.args, f_lo.kwargs))
        fused = graph.call_function(f_lo.target, args, kwargs)  # pyrefly: ignore[bad-argument-type]
        fused.meta = {
            **{k: v for k, v in f_lo.meta.items() if k in ("location", "stack_trace")},
            "val": source_val.new_empty(source_val.shape, dtype=f_val.dtype),
        }
    final.replace_all_uses_with(fused)
    # Erase the dead chain bottom-up.
    dead = [final]
    while dead:
        node = dead.pop()
        if node.users or node is source or node is fused:
            continue
        inputs = list(node.all_input_nodes)
        graph.erase_node(node)
        dead.extend(inputs)


def elide_split_join_roundtrips(graph: torch.fx.Graph) -> int:
    """Rewrite every ``join(f(lo), f(hi))`` over one split; returns the count."""
    changed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not view_ops.join:
            continue
        match = _match(node)
        if match is None:
            continue
        _rewrite(graph, match)
        changed += 1
    if changed:
        graph.lint()
    return changed
