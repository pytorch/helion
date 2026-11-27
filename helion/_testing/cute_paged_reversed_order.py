"""Reference model of the paired schedule's reversed page order.

``reversed_walk`` states which logical page each step of a work item's walk
takes, straight from the option's definition: the pages newest first, and
with a lead of the clipped page the oldest page moved ahead in that list.
``template_walk`` runs the template's own mapping
(``attention_page_coordinates``) on the host for the same steps, so CPU
tests compare the two for every step. ``window_layouts`` places zero and
random K pages under a request's window for the device tests: a window
whose scores are all equal, one whose newest pages alone hold data, and
one that holds data everywhere.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

from .cute_paged_page_order import host_args
from .cute_paged_page_order import unit_page_range
from .cute_paged_stream import jit_host_body
from helion._compiler.cute import _paged_attention_partial_pipeline as partial
from helion._compiler.cute.paged_attention_page_order import PAGE_KEYS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.paged_attention_page_order import PageWalk


def reversed_walk(first: int, last: int, lead: int = 0) -> list[int]:
    """The logical pages a reversed walk over ``[first, last)`` takes, step
    by step: newest first, and with a lead of more than one page the oldest
    page taken when that many pages are left, itself included. A walk of
    fewer pages than the lead takes it first."""
    pages = list(range(last - 1, first - 1, -1))
    if lead > 1 and pages:
        oldest = pages.pop()
        pages.insert(max(len(pages) + 1 - lead, 0), oldest)
    return pages


def template_walk(
    first: int, last: int, walk: PageWalk, steps: Sequence[int] | None = None
) -> list[tuple[int, int]]:
    """The page and key base the template's mapping gives each step of a
    walk over ``[first, last)`` (``steps``, or every step of the range), run
    from its original body on the host. ``walk`` is what the body hands it,
    its ``page_walk``: ``None`` for the source's loop, ``int`` for a source
    reflection (the dtype's wraparound is not modelled) or a
    ``ReversedWalk``."""
    typed = SimpleNamespace(Int64=int, const_expr=lambda value: value)
    mapping = jit_host_body(partial.attention_page_coordinates)
    with patch.object(partial, "cutlass", typed):
        return [
            cast("tuple[int, int]", mapping(first, last, step, walk))
            for step in (range(max(last - first, 0)) if steps is None else steps)
        ]


def split_pool(args: tuple[object, ...]) -> int:
    """Zero the K of the pool's lower half and return its first page with
    random K: scores of a zero page are equal for every key, whatever Q."""
    k = cast("torch.Tensor", args[1])
    data = k.size(0) // 2
    assert data > 0
    k[:data].zero_()
    return data


def window_layouts(
    args: tuple[object, ...], data: int, fresh: int, seed: int
) -> dict[str, torch.Tensor]:
    """Page tables for the requests' current lengths over a pool whose pages
    below ``data`` have zero K (``split_pool``): every page of a request's
    range a zero page, its newest ``fresh`` pages random and the older ones
    zero, and every page random. Logical pages outside a request's range
    keep random ids of the whole pool."""
    table = cast("torch.Tensor", args[5])
    pool = cast("torch.Tensor", args[1]).size(0)
    generator = torch.Generator().manual_seed(seed)
    zero = torch.randint(0, data, table.shape, generator=generator)
    random = torch.randint(data, pool, table.shape, generator=generator)
    host = host_args(args)
    part = zero.clone()
    for request in range(table.size(0)):
        first, last = unit_page_range(host, request)
        newest = slice(max(last - fresh, first, 0), max(last, 0))
        part[request, newest] = random[request, newest]
    return {
        "zero": zero.to(table),
        "part": part.to(table),
        "all": random.to(table),
    }


def clipped_keys(args: tuple[object, ...], request: int) -> int:
    """Keys of a request's oldest page that its window hides from the first
    row: zero for a request no longer than its window and for a window that
    begins at a page boundary."""
    host = host_args(args)
    length = int(cast("torch.Tensor", host[6])[request].item())
    query_len, window = cast("int", args[7]), cast("int", args[9])
    assert window >= 0
    return max(length - query_len - window, 0) % PAGE_KEYS
