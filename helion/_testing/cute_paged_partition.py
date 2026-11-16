"""Shared fixtures for the paired attention schedule's work partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .cute_optional_lse import optional_lse_config
from .cute_paged_attention import paged_online_attention
from .cute_paged_attention_schedule import paged_attention_args
import helion
from helion.runtime import default_cute_launcher

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import Kernel

PAGE = 128
# The fixture's page table has seventeen pages per request.
PAGES = 17
# A window of this many keys visits at most four pages, and a request longer
# than it starts at a later page: its first tile is a window edge.
SHORT_WINDOW = 300


def partition_attention_args(
    *,
    units: int = 17,
    rows: int = 40,
    heads: int = 2,
    reference_scores: bool = False,
    reverse_pages: bool = False,
    window: int = PAGES * PAGE - 1,
    device: torch.device | str = "cpu",
) -> tuple[object, ...]:
    """Per-key paged attention over ``units`` head pairs with an LSE output.

    The default window admits every page of the table, so a request's length
    alone selects how many pages its unit streams, from page zero.
    ``SHORT_WINDOW`` instead moves a long request's first page with its
    length, which is where a boundary key clip applies.
    """
    assert units * 2 % heads == 0
    values = list(
        paged_attention_args(
            rows=rows, heads=heads, requests=units * 2 // heads, device=device
        )
    )
    values[7:10] = [rows // 8, 8, window]
    return (*values, None, True, True, reference_scores, reverse_pages)


def partition_attention_kernel(
    *,
    fast_math: bool = True,
    reserved_sms: int = 0,
    reverse_pages: bool = False,
    input_banks: int = 1,
) -> Kernel[tuple[torch.Tensor, torch.Tensor | None]]:
    """The fixture's kernel; ``reserved_sms`` shrinks the persistent capacity.

    Reversed pages reflect the iteration in the index dtype, which that
    source form needs to be Int64. More than one input bank is the benchmark
    that times a search's candidates back to back in one captured graph.
    """
    return helion.kernel(
        paged_online_attention.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
        persistent_reserved_sms=reserved_sms,
        index_dtype=torch.int64 if reverse_pages else torch.int32,
        autotune_benchmark_input_banks=input_banks,
    )


def partition_config(
    partition: str | None = None,
    *,
    default_policies: bool = False,
    **overrides: object,
) -> helion.Config:
    """A paired query-TMA config; ``partition`` None leaves the knob absent.

    The config names ``cute_pdl=False``, which declines the paired schedule's
    guarded capture overlap. ``default_policies`` leaves the key out, so the
    kernel gets every default of the schedule: that overlap with its
    wave-aware release, early TMEM retirement and, for a windowed kernel
    under fast math, the leading key clip.
    """
    config = optional_lse_config()
    config.config.update(
        block_sizes=[64],
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_query_tma=True,
    )
    if default_policies:
        del config.config["cute_pdl"]
    config.config.update(overrides)
    if partition is not None:
        config.config["cute_mma_dag_partition"] = partition
    return config


def randomize_partition_inputs(args: tuple[object, ...], seed: int) -> None:
    """Fill the tensors with seeded values that keep both heads' rows distinct."""
    q, k, v, ks, vs, pages, _lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q.copy_(torch.randn(q.shape, generator=generator).to(q))
    for raw, scale, low in ((k, ks, 118), (v, vs, 121)):
        raw.copy_(torch.randint(-127, 128, raw.shape, generator=generator).to(raw))
        scale.copy_(
            torch.randint(low, low + 6, scale.shape, generator=generator).to(scale)
        )
    pages.copy_(torch.randint(0, k.size(0), pages.shape, generator=generator).to(pages))


def set_request_pages(args: tuple[object, ...], pages: list[int]) -> None:
    """Give request ``i`` a length that streams exactly ``pages[i]`` pages."""
    lengths = cast("torch.Tensor", args[6])
    assert len(pages) == lengths.numel() and max(pages) <= PAGES
    lengths.copy_(
        lengths.new_tensor(
            [
                count * PAGE - (37 * index % PAGE if count else 0)
                for index, count in enumerate(pages)
            ]
        )
    )


def set_request_lengths(args: tuple[object, ...], lengths: list[int]) -> None:
    """Give request ``i`` the length ``lengths[i]``, at most the page table's keys."""
    tensor = cast("torch.Tensor", args[6])
    assert len(lengths) == tensor.numel() and max(lengths) <= PAGES * PAGE
    tensor.copy_(tensor.new_tensor(lengths))


def launch_with_grid(
    compiled: Callable[..., object], args: tuple[object, ...], grid: int
) -> object:
    """Call the generated host function, launching ``grid`` CTAs instead of its
    own grid. A persistent partition reads the grid it was launched with."""

    def launcher(
        kernel: object, _grid: object, *arguments: object, **keywords: object
    ) -> object:
        return default_cute_launcher(kernel, (grid,), *arguments, **keywords)

    return compiled(*args, _launcher=launcher)
