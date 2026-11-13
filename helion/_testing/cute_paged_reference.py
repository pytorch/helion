"""Shared source fixtures for the explicit reference-score recurrence."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import cast

from . import cpu_render_target
from .cute_codegen import cpu_bind
from .cute_optional_lse import check_optional_lse_replay
from .cute_optional_lse import initialize_paged_attention_inputs
from .cute_optional_lse import optional_lse_config
from .cute_optional_lse import optional_lse_fixture
import helion
from helion._compiler.cute.cute_mma_dag import match_mma_dag

if TYPE_CHECKING:
    import torch

    from helion._compiler.cute.cute_mma_dag import MmaDagProgram
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


def reference_attention_fixture(
    form: str,
    query_len: int,
    return_lse: bool,
    device: torch.device | str = "cpu",
    *,
    window: int | None = 31,
) -> tuple[Kernel[tuple[torch.Tensor, torch.Tensor | None]], tuple[object, ...]]:
    kernel, args = optional_lse_fixture(
        form, return_lse, device, query_len=query_len, window=window
    )
    return kernel, (*args, True)


def reverse_attention_fixture(
    form: str,
    query_len: int,
    return_lse: bool,
    device: torch.device | str = "cpu",
    *,
    reference_scores: bool = True,
) -> tuple[Kernel[tuple[torch.Tensor, torch.Tensor | None]], tuple[object, ...]]:
    """Select explicit reverse traversal with either proved score recurrence."""
    kernel, args = optional_lse_fixture(form, return_lse, device, query_len=query_len)
    return kernel, (*args, reference_scores, True)


@functools.cache
def reference_attention_bound(
    form: str, query_len: int, *, reverse_pages: bool = False
) -> tuple[BoundKernel[object], MmaDagProgram]:
    """Share CPU-bound source graphs; mutation tests must copy the program."""
    fixture = (
        reverse_attention_fixture if reverse_pages else reference_attention_fixture
    )
    kernel, args = fixture(form, query_len, True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", kernel), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return bound, program


REFERENCE_ATTENTION_CASES = [
    ("mapped", 1, False, True, False, 1, False),
    ("mapped", 4, True, False, True, 2, True),
    ("mapped", 5, False, True, True, 2, False),
    ("per_key", 1, True, False, False, 2, False),
    ("per_key", 4, False, True, False, 1, True),
    ("per_key", 5, True, True, True, 2, False),
]


def check_reference_attention_fixture(
    kernel: Kernel[tuple[torch.Tensor, torch.Tensor | None]],
    args: tuple[object, ...],
    form: str,
    query_len: int,
    fast_math: bool,
    lse: bool,
    scoped: bool,
    occupancy: int,
    pdl: bool,
    *,
    base2_scores: bool = False,
    reference_scores: bool = True,
    reverse_pages: bool = False,
    index_dtype: torch.dtype | None = None,
) -> None:
    """Compile each source choice and replay it against matching source math."""
    initialize_paged_attention_inputs(args)
    settings = helion.Settings(backend="cute", static_shapes=True, fast_math=fast_math)
    if index_dtype is not None:
        settings.index_dtype = index_dtype
    bound = helion.kernel(kernel.fn, settings=settings).bind(args)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_scoped" if scoped else "attention",
        cute_min_blocks_per_mp=occupancy,
        cute_pdl=pdl,
    )
    source = bound.to_code(config)
    assert ("reference_scores=True" in source) == reference_scores
    assert ("base2_scores=True" in source) == base2_scores
    assert ("reflected_dtype=" in source) == reverse_pages
    if reverse_pages:
        # These source fixtures use INT32 bounds even with INT64 row indices.
        assert "reflected_dtype=cutlass.Int32" in source
    check_optional_lse_replay(
        bound,
        config,
        args,
        form,
        lse,
        reference_scores=reference_scores,
        base2_scores=base2_scores,
        reverse_pages=reverse_pages,
    )
