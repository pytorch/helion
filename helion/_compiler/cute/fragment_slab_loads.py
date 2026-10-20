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
same block indices / full slices / integer constants in every member, plus
exactly one dimension indexed by an ``hl.arange`` window (``start + arange``
with static integer bounds) or an integer constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from .computed_fragment import Fragment

KEY = "cute_fragment_slab_loads"


def static_window(node: object) -> tuple[int, int] | None:
    """``(start, extent)`` of a one-dimensional ``hl.arange`` window index:
    ``prims.iota`` (unit step, static integer start) optionally shifted by
    an integer constant through ``aten.add``."""
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        return None
    extent = value.shape[0]
    if not isinstance(extent, int):
        return None
    offset = 0
    current = node
    while True:
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


@dataclass
class SlabGroup:
    tensor: Node
    """The ``_host_tensor`` node."""
    descriptors: tuple[tuple[str, object], ...]
    """Per tensor dimension: ``("block", block_id)``, ``("slice", None)``,
    ``("const", int)`` or ``("window", None)`` for the varying dimension."""
    window_dim: int
    start: int
    extent: int
    members: dict[Node, tuple[int, int]] = field(default_factory=dict)
    """Member load -> ``(start, extent)`` along the window dimension
    (``extent`` 0 for an integer constant index, which drops the dimension)."""
    fragment: Fragment | None = None
    """The staged slab, once the root materialized it."""


def _describe(env: CompileEnvironment, index: object) -> tuple[str, object] | None:
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
            if block is None or proxy._sympy_() != env.block_sizes[block].var._sympy_():
                return None
            return ("block", block)
        window = static_window(index)
        if window is not None:
            return ("window", window)
    return None


def _static_size(env: CompileEnvironment, size: object) -> int | None:
    if isinstance(size, int):
        return size
    if isinstance(size, torch.SymInt):
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if isinstance(expr, sympy.Expr) and expr.is_number:
            return int(expr)
    return None


def plan_slab_loads(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> dict[Node, SlabGroup]:
    """Member load -> its slab group, for every group of two or more
    mask-free readonly window loads of one tensor that vary along one
    dimension only."""
    from .register_loads import host_load_is_readonly

    candidates: dict[
        tuple[object, ...], list[tuple[Node, Node, list[tuple[str, object]]]]
    ] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is not memory_ops.load:
                continue
            source = node.args[0]
            if (
                not isinstance(source, Node)
                or source.target is not _tracing_ops._host_tensor
                or node.args[2] is not None
            ):
                continue
            fake = source.meta.get("val")
            if not isinstance(fake, torch.Tensor) or not fake.ndim:
                continue
            indices = [index for index in node.args[1] if index is not None]  # type: ignore[union-attr]
            if len(indices) != fake.ndim:
                continue
            described = [_describe(env, index) for index in indices]
            if any(item is None for item in described):
                continue
            items = [item for item in described if item is not None]
            if not host_load_is_readonly(node, env, graphs, allow_unbound=True):
                continue
            # Group by the tensor and the shape of the index pattern with
            # windows/constants erased: one varying dimension is chosen below.
            for dim, (kind, _value) in enumerate(items):
                if kind not in ("window", "const"):
                    continue
                key = (
                    source.args[0],
                    tuple(fake.shape),
                    tuple(fake.stride()),
                    fake.dtype,
                    dim,
                    tuple(
                        ("vary", None) if d == dim else item
                        for d, item in enumerate(items)
                    ),
                )
                candidates.setdefault(key, []).append((node, source, items))
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
        ranges: dict[Node, tuple[int, int]] = {}
        total = 0
        for node, items in fresh:
            kind, value = items[dim]
            if kind == "window":
                start, extent = cast("tuple[int, int]", value)
                span = extent
            else:
                start, extent, span = cast("int", value), 0, 1
            if start < 0 or start + span > size:
                # A window past the tensor (a tier guarded off by the chunk
                # size) keeps its own masked walk.
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
        )
        for node in ranges:
            plan[node] = group
            claimed.add(node)
    return plan
