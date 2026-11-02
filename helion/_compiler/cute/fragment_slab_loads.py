"""Slab staging of repeated static windows (``cute_fragment_slab_loads``).

A fragment root that reads the same host tensor through several static
windows -- ``q[tile, 16 + hl.arange(16), :]`` per tier, ``gc[tile, 48, :]``
anchor rows, ``k[tile, hl.arange(48), :]`` growing column windows -- stages
every window as its own phase: one global round trip and one barrier each,
and overlapping windows are fetched again.  With the knob on, the windows of
one tensor that differ only along one dimension by static, in-bounds offsets
are covered by a single slab (their union along that dimension), staged
once at the start of the root; every window reads a view of the slab.  The
slab never exceeds the members' summed extents, so it never moves more data
than the windows did, while it collapses their round trips into one.

The plan is static (FX graph + specialized shapes) and conservative: a
member is a mask-free load of a readonly host tensor whose indices are the
same block indices / full slices / integer constants / CTA-uniform integer
expressions (``tile.begin``, ``tile.begin // H``) in every member, plus
exactly one dimension indexed by an ``hl.arange`` window (``start + arange``
with static integer bounds) or an integer constant.

A varlen kernel reads its token rows through the same static windows shifted
by a runtime row offset, each under the mask that ends the sequence::

    base = token_base[j]; vlen = valid_len[j]            # uniform 0-d loads
    hl.load(q, [base + 16 + hl.arange(16), h, :], extra_mask=(16 + hl.arange(16) < vlen)[:, None])

Such windows join a slab when every member shifts by the same offset load and
masks by the same limit load over its own window (``row < vlen`` in the
window's coordinates): the slab's row ``i`` (relative to the union's start)
is loaded at ``base + start + i`` when ``start + i < vlen`` and the row is
inside the tensor, else zero -- exactly what each masked member read.  The
offset and the limit are read once per CTA ahead of the slab (readonly
tensors at uniform subscripts, so the body's own loads of them see the same
values); a member without the mask, or with a mask over another window or
another limit, keeps its own walk.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import GridOrigin

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from .computed_fragment import Fragment

KEY = "cute_fragment_slab_loads"

Captures = Mapping[Node, Node]


def capture_map(graphs: list[GraphInfo]) -> dict[Node, Node]:
    """Placeholder of an ``_if`` branch graph -> the outer node it captures."""
    by_id = {info.graph_id: info for info in graphs}
    captures: dict[Node, Node] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is not _tracing_ops._if:
                continue
            for side in range(2):
                branch = by_id.get(cast("int", node.args[1 + side]))
                outer = node.args[3 + side]
                if branch is None or not isinstance(outer, (list, tuple)):
                    continue
                placeholders = list(branch.graph.find_nodes(op="placeholder"))
                if len(placeholders) != len(outer):
                    continue
                captures.update(
                    (placeholder, source)
                    for placeholder, source in zip(placeholders, outer, strict=True)
                    if isinstance(source, Node)
                )
    return captures


def resolve_capture(node: object, captures: Captures) -> object:
    """``node`` through ``_new_var`` copies and ``_if`` branch placeholders
    to the node that defines its value."""
    while isinstance(node, Node):
        if node.target is _tracing_ops._new_var:
            node = node.args[0]
            continue
        if node.op == "placeholder" and node in captures:
            node = captures[node]
            continue
        break
    return node


def static_window(
    node: object, captures: Captures | None = None
) -> tuple[int, int] | None:
    """``(start, extent)`` of a one-dimensional ``hl.arange`` window index:
    ``prims.iota`` (unit step, static integer start) optionally shifted by
    an integer constant through ``aten.add``."""
    captures = captures or {}
    node = resolve_capture(node, captures)
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        return None
    extent = value.shape[0]
    if not isinstance(extent, int):
        return None
    offset = 0
    current: object = node
    while True:
        current = resolve_capture(current, captures)
        if not isinstance(current, Node):
            return None
        if current.target is torch.ops.prims.iota.default:
            start = current.kwargs.get("start", 0)
            step = current.kwargs.get("step", 1)
            if (
                isinstance(start, int)
                and isinstance(step, int)
                and not isinstance(start, bool)
                and step == 1
            ):
                return start + offset, extent
            return None
        if (
            current.target
            in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add.Scalar,
            )
            and len(current.args) == 2
        ):
            a, b = current.args
            if isinstance(a, Node) and isinstance(b, int) and not isinstance(b, bool):
                offset += b
                current = a
                continue
            if isinstance(b, Node) and isinstance(a, int) and not isinstance(a, bool):
                offset += a
                current = b
                continue
        return None


@dataclass(frozen=True)
class ScalarLoad:
    """A CTA-uniform 0-d integer load of a host tensor (``token_base[j]``):
    the tensor argument and one uniform integer expression per dim.  Two
    loads of one tensor at equal expressions read the same value (the
    tensor is proved readonly for the root), so they compare equal here;
    ``node`` (the first seen ``_host_tensor`` node) carries the fake tensor
    the staging names the argument by."""

    tensor: str
    indices: tuple[sympy.Expr, ...]
    dtype: torch.dtype
    node: Node = field(compare=False, hash=False, repr=False)


@dataclass
class SlabGroup:
    tensor: Node
    """The ``_host_tensor`` node."""
    descriptors: tuple[tuple[str, object], ...]
    """Per tensor dimension: ``("block", block_id)``, ``("slice", None)``,
    ``("const", int)``, ``("uniform", sympy.Expr)`` or ``("window", None)``
    for the varying dimension."""
    window_dim: int
    start: int
    extent: int
    members: dict[Node, tuple[int, int]] = field(default_factory=dict)
    """Member load -> ``(start, extent)`` along the window dimension
    (``extent`` 0 for an integer constant index, which drops the dimension)."""
    offset: ScalarLoad | None = None
    """The runtime row offset the window dimension's static windows shift
    by (None: static windows at their own offsets)."""
    limit: ScalarLoad | None = None
    """The runtime limit of the window dimension: rows at or past it load
    zero (every member masked ``window < limit``); None: no limit."""
    fragment: Fragment | None = None
    """The staged slab, once the root materialized it."""


def _uniform_expr(env: CompileEnvironment, proxy: object) -> sympy.Expr | None:
    """``proxy`` as a CTA-uniform integer expression: an int, or a SymInt
    whose symbols are grid tile begins (``tile.begin``), block sizes or host
    values (None: a data-dependent or otherwise unplaceable symbol)."""
    from ..host_function import HostFunction

    if isinstance(proxy, bool):
        return None
    if isinstance(proxy, int):
        return sympy.Integer(proxy)
    if not isinstance(proxy, torch.SymInt):
        return None
    expr = proxy._sympy_()
    if not isinstance(expr, sympy.Expr):
        return None
    origins = HostFunction.current().expr_to_origin
    for symbol in expr.free_symbols:
        info = origins.get(symbol)
        if info is None:
            return None
        origin = info.origin
        if isinstance(origin, (GridOrigin, BlockSizeOrigin)) or origin.is_host():
            continue
        return None
    return expr


def scalar_load(
    env: CompileEnvironment,
    node: object,
    captures: Captures,
    graphs: list[GraphInfo],
) -> ScalarLoad | None:
    """``node`` as a uniform 0-d integer load of a readonly host tensor
    (None for anything else: a masked load, a tensor subscript, a computed
    value, a tensor the root also stores to -- the slab reads the value
    ahead of the body, which must see the same one)."""
    from .register_loads import host_load_is_readonly

    node = resolve_capture(node, captures)
    if not isinstance(node, Node) or node.target is not memory_ops.load:
        return None
    if len(node.args) < 3 or node.args[2] is not None:
        return None
    if not host_load_is_readonly(node, env, graphs, allow_unbound=True):
        return None
    source = node.args[0]
    if not isinstance(source, Node) or source.target is not _tracing_ops._host_tensor:
        return None
    fake = source.meta.get("val")
    value = node.meta.get("val")
    if (
        not isinstance(fake, torch.Tensor)
        or not isinstance(value, torch.Tensor)
        or value.ndim != 0
        or value.dtype not in (torch.int32, torch.int64)
    ):
        return None
    indices = cast("list[object]", node.args[1])
    if len(indices) != fake.ndim:
        return None
    exprs = []
    for index in indices:
        proxy = index.meta.get("val") if isinstance(index, Node) else index
        expr = _uniform_expr(env, proxy)
        if expr is None:
            return None
        exprs.append(expr)
    return ScalarLoad(cast("str", source.args[0]), tuple(exprs), value.dtype, source)


def runtime_window(
    env: CompileEnvironment,
    node: object,
    captures: Captures,
    graphs: list[GraphInfo],
) -> tuple[ScalarLoad, int, int] | None:
    """``(offset, start, extent)`` of a window index at a runtime row offset:
    ``scalar + (start + hl.arange(extent))`` with ``scalar`` a uniform 0-d
    load."""
    node = resolve_capture(node, captures)
    if (
        not isinstance(node, Node)
        or node.target is not torch.ops.aten.add.Tensor
        or len(node.args) != 2
    ):
        return None
    for first, second in (node.args, node.args[::-1]):
        window = static_window(second, captures)
        if window is None:
            continue
        offset = scalar_load(env, first, captures, graphs)
        if offset is not None:
            return offset, *window
    return None


def window_limit(
    env: CompileEnvironment,
    mask: object,
    captures: Captures,
    graphs: list[GraphInfo],
    position: int,
    ndim: int,
) -> tuple[tuple[int, int], ScalarLoad] | None:
    """The row limit a load's ``extra_mask`` states: ``static_window <
    limit`` (``limit`` a uniform 0-d load) over value dim ``position`` of an
    ``ndim``-dim value -- the vector itself for a 1-D value, broadcast by one
    ``[:, None]`` / ``[None, :]`` subscript over a 2-D value: ``((start,
    extent), limit)``; None for any other mask."""
    node = resolve_capture(mask, captures)
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        return None
    if ndim == 2:
        index = node.args[1] if len(node.args) > 1 else None
        if (
            node.target is not view_ops.subscript
            or not isinstance(index, (list, tuple))
            or len(index) != 2
            or index.count(None) != 1
            or index.count(slice(None)) != 1
            or list(index).index(slice(None)) != position
        ):
            return None
        node = resolve_capture(node.args[0], captures)
    elif ndim != 1 or position != 0:
        return None
    if (
        not isinstance(node, Node)
        or node.target is not torch.ops.aten.lt.Tensor
        or len(node.args) != 2
    ):
        return None
    window = static_window(node.args[0], captures)
    limit = scalar_load(env, node.args[1], captures, graphs)
    if window is None or limit is None:
        return None
    return window, limit


def _describe(
    env: CompileEnvironment,
    index: object,
    captures: Captures | None = None,
    graphs: list[GraphInfo] | None = None,
) -> tuple[str, object] | None:
    captures = captures or {}
    graphs = graphs or []
    if index is None:
        return None
    if isinstance(index, slice):
        return ("slice", None) if index == slice(None) else None
    if isinstance(index, bool):
        return None
    if isinstance(index, int):
        return ("const", index)
    if isinstance(index, Node):
        proxy = index.meta.get("val")
        if isinstance(proxy, torch.SymInt):
            block = env.resolve_block_id(proxy)
            if (
                block is not None
                and proxy._sympy_() == env.block_sizes[block].var._sympy_()
            ):
                return ("block", block)
            expr = _uniform_expr(env, proxy)
            if expr is not None:
                return ("uniform", expr)
            return None
        window = static_window(index, captures)
        if window is not None:
            return ("window", window)
        runtime = runtime_window(env, index, captures, graphs)
        if runtime is not None:
            return ("rwindow", runtime)
    return None


def _static_size(env: CompileEnvironment, size: object) -> int | None:
    if isinstance(size, int):
        return size
    if isinstance(size, torch.SymInt):
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if isinstance(expr, sympy.Expr) and expr.is_number:
            return int(expr)
    return None


def _hashable(value: object) -> object:
    """``value`` with every symbolic scalar (a non-nested ``SymInt`` has no
    hash) spelled as its expression, so it can key the candidate groups; a
    group keyed by a symbolic size is dropped later (``_static_size``)."""
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return ("sym", str(value))
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    return value


def plan_slab_loads(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> dict[Node, SlabGroup]:
    """Member load -> its slab group, for every group of two or more
    readonly window loads of one tensor that vary along one dimension only
    (mask-free static windows, or windows at one runtime offset under one
    runtime limit)."""
    from .register_loads import host_load_is_readonly

    captures = capture_map(graphs)
    candidates: dict[
        tuple[object, ...], list[tuple[Node, Node, list[tuple[str, object]]]]
    ] = {}
    scalars: dict[tuple[object, ...], ScalarLoad] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is not memory_ops.load:
                continue
            source = node.args[0]
            if (
                not isinstance(source, Node)
                or source.target is not _tracing_ops._host_tensor
            ):
                continue
            fake = source.meta.get("val")
            if not isinstance(fake, torch.Tensor) or not fake.ndim:
                continue
            indices = [index for index in node.args[1] if index is not None]  # type: ignore[union-attr]
            if len(indices) != fake.ndim:
                continue
            described = [_describe(env, index, captures, graphs) for index in indices]
            if any(item is None for item in described):
                continue
            items = [item for item in described if item is not None]
            runtime_dims = [
                dim for dim, (kind, _) in enumerate(items) if kind == "rwindow"
            ]
            if len(runtime_dims) > 1:
                continue
            mask = node.args[2]
            offset: ScalarLoad | None = None
            limit: ScalarLoad | None = None
            if runtime_dims:
                # The window's value dim position: the dims a scalar
                # subscript drops come before it.
                (dim,) = runtime_dims
                offset, start, extent = cast(
                    "tuple[ScalarLoad, int, int]", items[dim][1]
                )
                position = sum(
                    1 for kind, _ in items[:dim] if kind not in ("const", "uniform")
                )
                ndim = sum(1 for kind, _ in items if kind not in ("const", "uniform"))
                if mask is not None:
                    decoded = window_limit(env, mask, captures, graphs, position, ndim)
                    if decoded is None or decoded[0] != (start, extent):
                        continue
                    limit = decoded[1]
                items[dim] = ("rwindow", (start, extent))
            elif mask is not None:
                continue
            if not host_load_is_readonly(node, env, graphs, allow_unbound=True):
                continue
            # Group by the tensor and the shape of the index pattern with
            # windows/constants erased: one varying dimension is chosen below.
            for dim, (kind, _value) in enumerate(items):
                if kind not in ("window", "const", "rwindow"):
                    continue
                if runtime_dims and kind != "rwindow":
                    continue
                scalar_key = (
                    None
                    if offset is None
                    else (
                        (offset.tensor, offset.indices, offset.dtype),
                        None
                        if limit is None
                        else (limit.tensor, limit.indices, limit.dtype),
                    )
                )
                key = cast(
                    "tuple[object, ...]",
                    _hashable(
                        (
                            source.args[0],
                            tuple(fake.shape),
                            tuple(fake.stride()),
                            fake.dtype,
                            dim,
                            tuple(
                                ("vary", scalar_key) if d == dim else item
                                for d, item in enumerate(items)
                            ),
                        )
                    ),
                )
                candidates.setdefault(key, []).append((node, source, items))
                if offset is not None:
                    scalars.setdefault((*key, "offset"), offset)
                    if limit is not None:
                        scalars.setdefault((*key, "limit"), limit)
    plan: dict[Node, SlabGroup] = {}
    claimed: set[Node] = set()
    # Prefer the grouping that covers the most members.
    for key, members in sorted(candidates.items(), key=lambda kv: -len(kv[1])):
        dim = cast("int", key[-2])
        pattern = cast("tuple[tuple[str, object], ...]", key[-1])
        fresh = [
            (node, items) for node, _source, items in members if node not in claimed
        ]
        if len(fresh) < 2:
            continue
        # The first member's placeholder names the kernel argument; every
        # graph's placeholder for the tensor resolves to that same argument.
        source = members[0][1]
        fake = source.meta["val"]
        size = _static_size(env, fake.shape[dim])
        if size is None:
            continue
        offset = scalars.get((*key, "offset"))
        limit = scalars.get((*key, "limit"))
        ranges: dict[Node, tuple[int, int]] = {}
        total = 0
        for node, items in fresh:
            kind, value = items[dim]
            if kind in ("window", "rwindow"):
                start, extent = cast("tuple[int, int]", value)
                span = extent
            else:
                start, extent, span = cast("int", value), 0, 1
            if start < 0 or (offset is None and start + span > size):
                # A window past the tensor (a tier guarded off by the chunk
                # size) keeps its own masked walk.  (A runtime-offset window
                # is bounded by the tensor's end at run time.)
                continue
            ranges[node] = (start, extent)
            total += span
        if len(ranges) < 2:
            continue
        slab_start = min(start for start, _ in ranges.values())
        slab_end = max(start + max(extent, 1) for start, extent in ranges.values())
        if slab_end - slab_start > total:
            # The union would move more than the windows do: keep them.
            continue
        group = SlabGroup(
            tensor=source,
            descriptors=tuple(
                ("window", None) if d == dim else item for d, item in enumerate(pattern)
            ),
            window_dim=dim,
            start=slab_start,
            extent=slab_end - slab_start,
            members=ranges,
            offset=offset,
            limit=limit,
        )
        for node in ranges:
            plan[node] = group
            claimed.add(node)
    return plan
