"""Shared fixtures for paged attention with and without an LSE output."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .cute_folded_attention import folded_attention_args
from .cute_folded_attention import folded_attention_reference
from .cute_folded_attention import folded_paged_attention
from .cute_paged_attention import paged_online_attention
from .cute_paged_attention_schedule import paged_attention_args
import helion

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


def optional_lse_fixture(
    form: str,
    return_lse: bool,
    device: torch.device | str = "cpu",
    *,
    query_len: int = 4,
    window: int | None = 31,
) -> tuple[Kernel[tuple[torch.Tensor, torch.Tensor | None]], tuple[object, ...]]:
    if form == "mapped":
        args = list(folded_attention_args(query_len, device=device))
        args[9] = return_lse
        args[7] = window
        return folded_paged_attention, tuple(args)
    assert form in ("reduced", "per_key")
    args = list(paged_attention_args(rows=query_len * 8, device=device))
    args[7] = query_len
    args[9] = window if window is not None else -1
    return paged_online_attention, (*args, None, form == "per_key", return_lse)


def optional_lse_config() -> helion.Config:
    return helion.Config(
        block_sizes=[32],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention",
        cute_min_blocks_per_mp=1,
        cute_pdl=False,
    )


def initialize_paged_attention_inputs(args: tuple[object, ...]) -> None:
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    for tensor, period, center in ((q, 19, 9), (k, 15, 7), (v, 17, 8)):
        values = (
            torch.arange(tensor.numel(), device=tensor.device).reshape(tensor.shape)
            % period
            - center
        )
        tensor.copy_(values / 32 if tensor is q else values)
    for scale in (ks, vs):
        scale.copy_(
            (torch.arange(scale.numel(), device=scale.device) % 4 + 125).reshape(
                scale.shape
            )
        )
    pages.copy_(
        (torch.arange(pages.numel(), device=pages.device) * 3 % k.size(0)).reshape(
            pages.shape
        )
    )
    lengths.copy_(torch.tensor([0, 257], device=lengths.device))


def optional_lse_reference(
    args: tuple[object, ...],
    form: str,
    *,
    reference_scores: bool = False,
    reverse_pages: bool = False,
    base2_scores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if form == "mapped":
        return folded_attention_reference(
            args,
            reference_scores=reference_scores,
            reverse_pages=reverse_pages,
            base2_scores=base2_scores,
        )
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    query_len, group, window = (cast("int", arg) for arg in args[7:10])
    items, rows, dim = q.shape
    heads = k.size(3)
    batch = items // heads
    original = (
        q.reshape(batch, heads, query_len, group, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_len, heads * group, dim)
    )
    mapped_args = (
        original,
        k[:, 0],
        v[:, 0],
        ks[:, 0],
        vs[:, 0],
        pages,
        lengths,
        window if window >= 0 else None,
    )
    output, lse = folded_attention_reference(
        mapped_args,
        reference_scores=reference_scores,
        reverse_pages=reverse_pages,
        base2_scores=base2_scores,
    )
    return (
        output.reshape(batch, query_len, heads, group, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(items, rows, dim),
        lse.reshape(batch, query_len, heads, group)
        .permute(0, 2, 1, 3)
        .reshape(items, rows),
    )


def check_optional_lse_replay(
    bound: BoundKernel,
    config: helion.Config,
    args: tuple[object, ...],
    form: str,
    return_lse: bool,
    *,
    reference_scores: bool = False,
    reverse_pages: bool = False,
    base2_scores: bool = False,
) -> None:
    """Compare eager and graph calls after independent input/output mutations."""
    compiled = bound.compile_config(config)
    compiled(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    original_keys = k.clone()
    for changed in range(5 if reference_scores else 4):
        if changed == 1:
            q.neg_()
            pages.add_(2).remainder_(k.size(0))
            lengths.copy_(lengths.new_tensor([129, 0]))
        elif changed == 2:
            # Later finite tiles retain their zero maximum, exercising the
            # uniform branch which skips accumulator and denominator rescaling.
            k.zero_()
        elif changed == 3:
            # Preserve nonuniform-weight coverage of exceptional V scales.
            k.copy_(original_keys)
            if form == "mapped":
                v[:, :, 0, :32] = 0
                vs[:, :, 0, 0] = 255
            else:
                v[:, :, :, 0, :32] = 0
                vs[:, :, :, 0, 0] = 255
            q[(0,) * q.ndim] = float("nan")
        elif changed == 4:
            # A negative running maximum must keep a zero reference and force
            # the full update path, even on later tiles with the same maximum.
            initialize_paged_attention_inputs(args)
            q.fill_(0.5)
            k.fill_(-1)
            lengths.copy_(lengths.new_tensor([129, 257]))
        expected = optional_lse_reference(
            args,
            form,
            reference_scores=reference_scores,
            reverse_pages=reverse_pages,
            base2_scores=base2_scores,
        )
        for replay in (False, True):
            if replay:
                for value in captured:
                    if value is not None:
                        value.fill_(float("nan"))
                graph.replay()
                actual = captured
            else:
                actual = compiled(*args)
            torch.testing.assert_close(
                actual[0], expected[0], rtol=0.002, atol=0.002, equal_nan=True
            )
            if return_lse:
                assert actual[1] is not None
                torch.testing.assert_close(
                    actual[1], expected[1], rtol=0.002, atol=0.002, equal_nan=True
                )
            else:
                assert actual[1] is None
