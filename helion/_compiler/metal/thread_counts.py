"""How a Metal launch's tiles take their threads from the threadgroup budget
(``threadgroup_budget``: ``num_warps`` SIMD groups, at most 1024) and its
three thread axes, when no reduction or free ``hl.arange`` axis reserves any.

``MetalBackend._config_with_auto_threads_shrunk`` and
``_serialize_nested_auto_tiles`` run the budget rules on each device loop
while planning a kernel, and the shared planner then fits each loop's tiles
to the axes (``demote_auto_thread_axes``); :func:`launch_threads` runs them
over the whole kernel, to tell which threadgroup a config launches before
any codegen.
"""

from __future__ import annotations

from itertools import starmap
import math
from typing import TYPE_CHECKING

from ..backend import _largest_divisor_at_most
from ..per_thread_planning import demote_auto_thread_axes
from ._constants import MAX_THREAD_AXES

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Sequence


def requested_threads(size: int, configured: int) -> int:
    """A tile's thread request: what its ``num_threads`` launches, the
    largest divisor of its block at most the count, or its block for an auto
    (0) count."""
    return _largest_divisor_at_most(size, configured) if configured > 0 else size


def shrink_tile_threads(
    sizes: Sequence[int],
    requested: Sequence[int],
    budget: int,
    *,
    fixed: int = 1,
    to_one: Collection[int] = (),
) -> list[int]:
    """A device loop's tiles' threads: their ``requested`` counts, the
    largest shrunk to the largest divisor of its block below it (ties to the
    first) until, times the loop's ``fixed`` literal tiles, they fit
    ``budget``.  A tile in ``to_one`` (reduced after its loop, which needs
    one element per thread or one thread: ``MetalBlockReductionStrategy``)
    goes straight to one."""
    threads = list(requested)
    while threads and fixed * math.prod(threads) > budget:
        widest = max(range(len(threads)), key=threads.__getitem__)
        if threads[widest] <= 1:
            break
        threads[widest] = (
            1
            if widest in to_one
            else _largest_divisor_at_most(sizes[widest], threads[widest] - 1)
        )
    return threads


def serialize_auto_threads(
    sizes: Sequence[int],
    threads: Sequence[int],
    serial: Sequence[int],
    total: int,
    budget: int,
    *,
    to_one: Collection[int] = (),
) -> tuple[list[int], int]:
    """While the kernel's loop blocks together (``total`` threads) pass
    ``budget``, shrink a nested loop's auto tiles (positions ``serial``, in
    order) to the largest divisor of their block that fits what the others
    leave; lanes walk the rest.  A tile in ``to_one`` goes to one.  Returns
    the loop's threads and the new total."""
    threads = list(threads)
    for i in serial:
        if total <= budget:
            break
        rest = total // threads[i]
        threads[i] = (
            1
            if i in to_one
            else _largest_divisor_at_most(
                sizes[i], min(threads[i], max(1, budget // rest))
            )
        )
        total = rest * threads[i]
    return threads, total


def launch_threads(
    loops: Sequence[Sequence[int]],
    sizes: Sequence[int],
    configured: Sequence[int],
    budget: int,
    *,
    to_one: Collection[int] = (),
) -> list[int]:
    """Each tile's threads in the launch, as the backend plans the loops:
    ``loops`` holds the positions (into ``sizes`` and ``configured``) of
    each device loop's tiles, the grid's first.  Each loop's tiles shrink to
    the budget (:func:`shrink_tile_threads`); a nested loop's auto tiles
    then shrink to what the other loops' requests leave
    (:func:`serialize_auto_threads`).  A nested explicit count stays as
    given, and nothing fills a threadgroup short of the budget.  Last, the
    tiles fit the three thread axes (:func:`demote_auto_tiles`); a config
    whose explicit counts still take a fourth fails in codegen
    (``MetalBackend.thread_index``)."""
    threads = list(starmap(requested_threads, zip(sizes, configured, strict=True)))
    requested = list(threads)
    for depth, loop in enumerate(loops):
        loop_sizes = [sizes[i] for i in loop]
        loop_to_one = {j for j, i in enumerate(loop) if i in to_one}
        shrunk = shrink_tile_threads(
            loop_sizes, [requested[i] for i in loop], budget, to_one=loop_to_one
        )
        outside = math.prod(
            requested[i] for other in loops if other is not loop for i in other
        )
        serial = [j for j, i in enumerate(loop) if configured[i] <= 0] if depth else []
        shrunk, _ = serialize_auto_threads(
            loop_sizes,
            shrunk,
            serial,
            outside * math.prod(shrunk),
            budget,
            to_one=loop_to_one,
        )
        for j, i in enumerate(loop):
            threads[i] = shrunk[j]
    # Below a nested loop, the planner counts the axes the grid's tiles take
    # as it plans them (``plan_per_thread_loop_strategy``).
    grid_axes = sum(threads[i] > 1 for i in loops[0]) if loops else 0
    for depth, loop in enumerate(loops):
        # A tile the passes above left at its block is still on auto there.
        counts = [
            0 if configured[i] <= 0 and threads[i] == sizes[i] else threads[i]
            for i in loop
        ]
        demote_auto_tiles(
            [sizes[i] for i in loop],
            counts,
            MAX_THREAD_AXES - grid_axes if depth else MAX_THREAD_AXES,
        )
        for j, i in enumerate(loop):
            threads[i] = counts[j] or threads[i]
    return threads


def demote_auto_tiles(sizes: Sequence[int], counts: list[int], axes: int) -> None:
    """Run a loop's latest auto tiles (``counts`` 0: a thread per element of
    their ``sizes``) on one thread while more of its tiles take a thread
    axis than the ``axes`` left to it, as the shared planner does
    (``demote_auto_thread_axes``)."""
    demote_auto_thread_axes(
        counts, axes, lambda i, count: (count or sizes[i]) > 1, lambda i: True
    )
