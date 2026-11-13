"""How a Metal launch's tiles take their threads from the threadgroup budget
(``threadgroup_budget``: ``num_warps`` SIMD groups, at most 1024), when no
reduction or free ``hl.arange`` axis reserves any.

``MetalBackend._config_with_auto_threads_shrunk`` and
``_serialize_nested_auto_tiles`` run these rules on each device loop while
planning a kernel; the search surface's ``fit_threads`` runs them over the
whole kernel (:func:`launch_threads`) to tell which threadgroup a config
launches before any codegen.  One copy, so the two can't disagree.
"""

from __future__ import annotations

from itertools import starmap
import math
from typing import TYPE_CHECKING

from ..backend import largest_divisor_at_most

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Sequence


def requested_threads(size: int, configured: int) -> int:
    """A tile's thread request: its ``num_threads``, at most its block, or
    its block for an auto (0) count."""
    return min(configured, size) if configured > 0 else size


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
            else largest_divisor_at_most(sizes[widest], threads[widest] - 1)
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
            else largest_divisor_at_most(
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
    fixed: Sequence[int] = (),
    to_one: Collection[int] = (),
) -> list[int]:
    """Each tile's threads in the launch, as the backend plans the loops:
    ``loops`` holds the positions (into ``sizes`` and ``configured``) of
    each device loop's tiles, the grid's first, and ``fixed`` (when given)
    the product of each loop's literal-size tiles, which have no count and
    run one thread per element.  Each loop's tiles shrink to the budget
    (:func:`shrink_tile_threads`); a nested loop's auto tiles then shrink to
    what the other loops' requests leave (:func:`serialize_auto_threads`).
    A nested explicit count stays as given, and nothing fills a threadgroup
    short of the budget.  The launch is the product of the returned threads
    and ``fixed``."""
    threads = list(starmap(requested_threads, zip(sizes, configured, strict=True)))
    requested = list(threads)
    literal = list(fixed) or [1] * len(loops)
    for depth, loop in enumerate(loops):
        loop_sizes = [sizes[i] for i in loop]
        loop_to_one = {j for j, i in enumerate(loop) if i in to_one}
        shrunk = shrink_tile_threads(
            loop_sizes,
            [requested[i] for i in loop],
            budget,
            fixed=literal[depth],
            to_one=loop_to_one,
        )
        outside = math.prod(literal) * math.prod(
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
    return threads
