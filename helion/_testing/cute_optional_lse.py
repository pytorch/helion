"""Shared fixtures for paged attention with and without an LSE output."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

from .cuda_graph import cuda_graph_for_replay
from .cute_folded_attention import folded_attention_args
from .cute_folded_attention import folded_attention_reference
from .cute_folded_attention import folded_paged_attention
from .cute_paged_attention import paged_online_attention
from .cute_paged_attention_schedule import paged_attention_args
import helion
from helion.runtime.cute import capture_pdl_launch
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from helion.runtime.cute.capture_pdl import CaptureGuard
    from helion.runtime.cute.capture_pdl import LaunchEffects
    from helion.runtime.cute.capture_pdl_cuda import CaptureDriver
    from helion.runtime.cute.capture_pdl_cuda import CaptureSnapshot
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
    output_reciprocal: bool = False,
    key_post_scale_log2: int = 0,
    value_post_scale_log2: int = 0,
    output_scale_log2: int = 0,
    value_positive_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if form == "mapped":
        return folded_attention_reference(
            args,
            reference_scores=reference_scores,
            reverse_pages=reverse_pages,
            base2_scores=base2_scores,
            output_reciprocal=output_reciprocal,
            key_post_scale_log2=key_post_scale_log2,
            value_post_scale_log2=value_post_scale_log2,
            output_scale_log2=output_scale_log2,
            value_positive_zero=value_positive_zero,
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
        output_reciprocal=output_reciprocal,
        key_post_scale_log2=key_post_scale_log2,
        value_post_scale_log2=value_post_scale_log2,
        output_scale_log2=output_scale_log2,
        value_positive_zero=value_positive_zero,
    )
    return (
        output.reshape(batch, query_len, heads, group, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(items, rows, dim),
        lse.reshape(batch, query_len, heads, group)
        .permute(0, 2, 1, 3)
        .reshape(items, rows),
    )


def _assert_attention_outputs(
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor],
    return_lse: bool,
) -> None:
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
    output_reciprocal: bool = False,
    key_post_scale_log2: int = 0,
    value_post_scale_log2: int = 0,
    output_scale_log2: int = 0,
    value_positive_zero: bool = False,
) -> None:
    """Compare eager and graph calls after independent input/output mutations."""
    compiled = bound.compile_config(config)
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        q, k, v, ks, vs, pages, lengths = (
            cast("torch.Tensor", arg) for arg in args[:7]
        )
        original_keys = k.clone()
        ordinary_cases = 5 if reference_scores else 4
        replay_cases = ordinary_cases + (3 if value_positive_zero else 0)
        rounding_case = output_reciprocal and not bound.env.settings.fast_math
        for changed in range(replay_cases + int(rounding_case)):
            if changed == replay_cases:
                # Seven unit probabilities sum to exactly 7. Their decoded V sum
                # is 10.52734375: division rounds to the BF16 midpoint 1.50390625
                # (then 1.5), while one FP32 reciprocal followed by multiplication
                # lands one FP32 ulp higher (then 1.5078125).
                q.zero_()
                k.zero_()
                v.zero_()
                ks.fill_(127)
                vs.fill_(127)
                token_axis = 1 if form == "mapped" else 2
                v.narrow(token_axis, 0, 5).fill_(2)
                v.select(token_axis, 5).fill_(67)
                v.select(token_axis, 6).fill_(68)
                vs.narrow(token_axis, 5, 2).fill_(119)
                pages.zero_()
                query_len = q.size(1) if form == "mapped" else cast("int", args[7])
                lengths.fill_(query_len + 6)
            elif value_positive_zero and changed == ordinary_cases:
                # Negative raw bytes underflow to -0 before the source +0.
                # Replaying this tile catches stale/missing TMEM columns;
                # test_positive_zero_decode_bits_and_changed_replay checks
                # the decoder's V-level zero signs over its full domain.
                initialize_paged_attention_inputs(args)
                q.zero_()
                v.fill_(-128 if value_post_scale_log2 <= -14 else 0)
                vs.zero_()
                pages.zero_()
                lengths.copy_(lengths.new_tensor([129, 257]))
            elif value_positive_zero and changed == ordinary_cases + 1:
                # One visible key/group makes this replay a mixed tile.
                if form == "mapped":
                    v[0, 127, 0, 32:64] = 1
                    vs[0, 127, 0, 1] = 127
                else:
                    v[0, 0, 127, 0, 32:64] = 1
                    vs[0, 0, 127, 0, 1] = 127
            elif value_positive_zero and changed == ordinary_cases + 2:
                # Restoring zero V must not suppress the original NaN * 0 PV.
                vs.zero_()
                v.fill_(-128 if value_post_scale_log2 <= -14 else 0)
                q[(0,) * q.ndim] = float("nan")
            elif changed == 1:
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
                output_reciprocal=output_reciprocal,
                key_post_scale_log2=key_post_scale_log2,
                value_post_scale_log2=value_post_scale_log2,
                output_scale_log2=output_scale_log2,
                value_positive_zero=value_positive_zero,
            )
            for replay in (False, True):
                if replay:
                    for value, reference in zip(captured, expected, strict=True):
                        if value is not None:
                            # Every expected NaN gets a finite sentinel; every
                            # other result gets NaN, so all missing writes fail.
                            value.fill_(float("nan"))
                            value.masked_fill_(torch.isnan(reference), 0.0)
                    graph.replay()
                    actual = captured
                else:
                    actual = compiled(*args)
                _assert_attention_outputs(actual, expected, return_lse)
                if value_positive_zero and changed == ordinary_cases:
                    assert (actual[0].view(torch.int16) == 0).all().item()
                if changed == replay_cases:
                    first_row = (
                        actual[0][0, 0, 0] if form == "mapped" else actual[0][0, 0]
                    )
                    torch.testing.assert_close(
                        first_row, torch.full_like(first_row, 1.5078125), rtol=0, atol=0
                    )


def bind_paged_attention_length_replay(
    kernel: Kernel,
    args: tuple[object, ...],
    form: str,
    *,
    fast_math: bool,
    reverse_pages: bool,
) -> tuple[BoundKernel, tuple[object, ...]]:
    """Bind initialized inputs with room for all seventeen-page replay cases."""
    values = list(args)
    pages = cast("torch.Tensor", values[5])
    values[5] = pages.new_empty((pages.size(0), 17))
    # A wide window permits seventeen-page replay; per-key uses -1 for no window.
    values[7 if form == "mapped" else 9] = 2047 if form == "mapped" else -1
    inputs = tuple(values)
    initialize_paged_attention_inputs(inputs)
    bound = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
        index_dtype=torch.int64 if reverse_pages else torch.int32,
    ).bind(inputs)
    return bound, inputs


def check_paged_attention_length_replay(
    bound: BoundKernel,
    config: helion.Config,
    args: tuple[object, ...],
    form: str,
    return_lse: bool,
    *,
    reference_scores: bool = False,
    reverse_pages: bool = False,
    base2_scores: bool = False,
    output_reciprocal: bool = False,
    key_post_scale_log2: int = 0,
    value_post_scale_log2: int = 0,
    output_scale_log2: int = 0,
    value_positive_zero: bool = False,
) -> None:
    """Replay ranges from bind_paged_attention_length_replay, empty through 17 pages."""
    compiled = bound.compile_config(config)
    initialize_paged_attention_inputs(args)
    lengths = cast("torch.Tensor", args[6])
    with cuda_graph_for_replay() as graph:
        capture_overlap = config.config.get("cute_pdl_capture_overlap", False)
        modes: list[WaitMode] = []
        finish = capture_pdl_launch.finish_capture_launch

        def observed(
            driver: CaptureDriver,
            guard: CaptureGuard,
            before: CaptureSnapshot,
            effects: LaunchEffects,
            parameter_sizes: tuple[int, ...],
        ) -> WaitMode:
            mode = finish(driver, guard, before, effects, parameter_sizes)
            modes.append(mode)
            return mode

        with (
            patch.object(capture_pdl_launch, "finish_capture_launch", observed),
            torch.cuda.graph(graph),
        ):
            captured = [compiled(*args)]
            if capture_overlap:
                # Keep both outputs alive so the graph pool cannot reuse their storage.
                captured.append(compiled(*args))
        if capture_overlap:
            assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
        for counts in ((0, 0), (1, 128), (129, 256), (513, 640), (2049, 2175)):
            lengths.copy_(lengths.new_tensor(counts))
            expected = optional_lse_reference(
                args,
                form,
                reference_scores=reference_scores,
                reverse_pages=reverse_pages,
                base2_scores=base2_scores,
                output_reciprocal=output_reciprocal,
                key_post_scale_log2=key_post_scale_log2,
                value_post_scale_log2=value_post_scale_log2,
                output_scale_log2=output_scale_log2,
                value_positive_zero=value_positive_zero,
            )
            # Poison and check both launches independently, including empty intervals.
            for output in captured:
                for value in output:
                    if value is not None:
                        value.fill_(float("nan"))
            graph.replay()
            for output in captured:
                _assert_attention_outputs(output, expected, return_lse)
