"""Variable-length top-k indices with the measured GB300 ordinary-kernel config.

The supported recipe is contiguous FP32[16,8192], Int32[16] mixed lengths and
K=1024. Short rows return their original indices followed by -1 padding; long
rows return an unordered top-k value multiset. No selected-NaN validation is
claimed.

The adjacent AOT module contains the measured configuration and policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


STRUCTURAL_POLICY = helion.CuteStructuralPolicy(
    cute_region_fission=True,
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_flatten_nested_reductions=True,
    cute_materialize_transformed_operands=True,
)


@helion.aot_kernel(
    backend="cute", static_shapes=True, cute_structural_policy=STRUCTURAL_POLICY
)
def varlen_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: hl.constexpr,
    next_n: hl.constexpr = 1,  # pyrefly: ignore[bad-function-definition]
    compress_ratio: hl.constexpr = 1,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    key_bits = hl.specialize(32 if logits.dtype == torch.float32 else 16)
    infinity_bits = hl.specialize(
        0x7F800000
        if logits.dtype == torch.float32
        else (0x7C00 if logits.dtype == torch.float16 else 0x7F80)
    )
    rows, width = logits.shape
    capacity = helion.next_power_of_2(width)
    compact_capacity = min(helion.next_power_of_2(k), 32)  # pyrefly: ignore[bad-argument-type]
    histogram_bins = 2048
    rank_capacity = min(compact_capacity, 32)
    result = torch.empty((rows, k), dtype=torch.int32, device=logits.device)  # pyrefly: ignore[no-matching-overload]
    for row in hl.grid(rows):
        count = torch.clamp(
            (lengths[row // next_n] - next_n + row % next_n + 1) // compress_ratio,  # pyrefly: ignore[unsupported-operation]
            min=0,
            max=width,
        )
        columns = hl.arange(capacity)
        output_columns = hl.arange(k)  # pyrefly: ignore[bad-argument-type]
        hl.store(
            result,
            [row, output_columns],
            torch.where((count <= k) & (output_columns < count), output_columns, -1).to(  # pyrefly: ignore[unsupported-operation]
                torch.int32
            ),
        )
        valid = (columns < count) & (count > k)  # pyrefly: ignore[unsupported-operation]
        values = hl.load(logits, [row, columns], extra_mask=valid).to(torch.float32)
        # Sample-only load: the long-row input no longer produces a full
        # ordered-key tensor or the unused integer classifier before the bin.
        sample_columns = hl.arange(helion.next_power_of_2(k))  # pyrefly: ignore[bad-argument-type]
        sample_valid = (sample_columns < k) & (sample_columns < count) & (count > k)  # pyrefly: ignore[unsupported-operation]
        sample_values = hl.load(
            logits, [row, sample_columns], extra_mask=sample_valid
        ).to(torch.float32)
        sample_low = torch.where(sample_valid, sample_values, float("inf")).min()
        sample_high = torch.where(sample_valid, sample_values, -float("inf")).max()
        sample_span = sample_high - sample_low
        usable_float_span = (
            (sample_low > -float("inf"))
            & (sample_high < float("inf"))
            & (sample_span > 0)
            & (sample_span < float("inf"))
        )
        float_low = torch.where(usable_float_span, sample_low, 0.0)
        float_high = torch.where(usable_float_span, sample_high, 1.0)
        float_span = torch.clamp(
            torch.where(usable_float_span, sample_span, 1.0), min=1e-30
        )
        inv_span = 1.0 / float_span
        # Clipping before subtraction prevents overflow outside the sampled
        # interval. NaNs share the highest bin and retain their exact key.
        finite_values = torch.where(values == values, values, float_high)
        fraction = torch.clamp(
            (torch.clamp(finite_values, min=float_low, max=float_high) - float_low)
            * inv_span,
            min=0.0,
            max=1.0,
        )
        float_buckets = (fraction * (histogram_bins - 1)).to(torch.int32)
        buckets = torch.where(usable_float_span, float_buckets, 0)
        histogram = hl.zeros([histogram_bins], dtype=torch.int32)
        hl.atomic_add(histogram, [histogram_bins - 1 - buckets], valid.to(torch.int32))
        bins = hl.arange(histogram_bins)
        cumulative = hl.cumsum(histogram, dim=0)
        descending_cut = torch.where(cumulative >= k, bins, histogram_bins - 1).min()  # pyrefly: ignore[unsupported-operation]
        selected = histogram_bins - 1 - descending_cut
        cutoff_buffer = hl.zeros([1], dtype=torch.int32)
        hl.atomic_add(cutoff_buffer, [0], selected)
        crossing = valid & (buckets == selected)
        crossing_count = hl.zeros([1], dtype=torch.int32)
        compact_keys = hl.zeros([compact_capacity], dtype=torch.int32)
        compact_ids = hl.zeros([compact_capacity], dtype=torch.int32)
        fast_higher_count = hl.zeros([1], dtype=torch.int32)
        fast_cut = cutoff_buffer.sum()
        tickets = hl.atomic_add(
            crossing_count, [torch.where(crossing, 0, 1)], crossing.to(torch.int32)
        )
        safe_slot = torch.where(
            tickets < compact_capacity, tickets, compact_capacity - 1
        )
        within = crossing & (tickets < compact_capacity)
        if key_bits == 32:
            raw_values = values.view(torch.int32)
        else:
            raw_values = values.to(logits.dtype).view(torch.int16).to(torch.int32)
        hl.atomic_add(
            compact_keys,
            [torch.where(within, safe_slot, compact_capacity)],
            torch.where(within, raw_values, 0),
        )
        hl.atomic_add(
            compact_ids,
            [torch.where(within, safe_slot, compact_capacity)],
            torch.where(within, columns + 1, 0).to(torch.int32),
        )
        fast_above = valid & (buckets > fast_cut)
        fast_higher_ticket = hl.atomic_add(
            fast_higher_count,
            [torch.where(fast_above, 0, 1)],
            fast_above.to(torch.int32),
        )
        hl.store(
            result,
            [row, fast_higher_ticket],
            columns.to(torch.int32),
            extra_mask=fast_above,
        )
        population = crossing_count.sum()
        if population <= rank_capacity:
            fast_need = k - fast_higher_count.sum()  # pyrefly: ignore[unsupported-operation]
            small_columns = hl.arange(rank_capacity)
            small_valid = small_columns < population
            small_word = compact_keys.to(torch.int64)
            small_raw = small_word & ((1 << key_bits) - 1)
            small_raw = torch.where(
                (small_raw & ((1 << (key_bits - 1)) - 1)) == 0, 0, small_raw
            )
            small_keys = torch.where(
                (small_raw & (1 << (key_bits - 1))) != 0,
                ((1 << key_bits) - 1) - small_raw,
                small_raw ^ (1 << (key_bits - 1)),
            )
            small_keys = torch.where(
                (small_raw & ((1 << (key_bits - 1)) - 1)) > infinity_bits,
                (1 << key_bits) - 1,
                small_keys,
            )
            small_ids = compact_ids - 1
            precedes = small_valid[None, :] & (
                (small_keys[None, :] > small_keys[:, None])
                | (
                    (small_keys[None, :] == small_keys[:, None])
                    & (small_columns[None, :] < small_columns[:, None])
                )
            )
            small_rank = precedes.to(torch.int32).sum(dim=1, dtype=torch.int32)
            hl.store(
                result,
                [row, k - fast_need + small_rank],  # pyrefly: ignore[unsupported-operation]
                small_ids,
                extra_mask=small_valid & (small_rank < fast_need),
            )
        else:
            # A complete fallback constructs full keys only in this arm.
            # Failed/collapsed float brackets and compact overflow never
            # consume a truncated crossing buffer.
            if key_bits == 32:
                signed = values.view(torch.int32)
            else:
                signed = values.to(logits.dtype).view(torch.int16).to(torch.int32)
            signed = torch.where(values == 0, 0, signed)
            raw = signed.to(torch.int64) & ((1 << key_bits) - 1)
            keys = torch.where(
                signed < 0, ((1 << key_bits) - 1) - raw, raw ^ (1 << (key_bits - 1))
            )
            keys = torch.where(
                (raw & ((1 << (key_bits - 1)) - 1)) > infinity_bits,
                (1 << key_bits) - 1,
                keys,
            )
            full_prefix = hl.full([], 0, dtype=torch.int64)
            full_remaining = torch.where(count > k, k, 0).to(torch.int32)  # pyrefly: ignore[unsupported-operation, no-matching-overload]
            for digit in hl.static_range(key_bits // 8):
                full_shift = key_bits - 8 - digit * 8
                full_hist = hl.zeros([256], dtype=torch.int32)
                full_active = valid & (
                    ((keys >> (full_shift + 8)) << (full_shift + 8)) == full_prefix
                )
                full_digit = (255 - ((keys >> full_shift) & 255)).to(torch.int32)
                hl.atomic_add(full_hist, [full_digit], full_active.to(torch.int32))
                full_bins = hl.arange(256)
                full_cdf = hl.cumsum(full_hist, dim=0)
                full_selected = torch.where(
                    full_cdf >= full_remaining, full_bins, 256
                ).min()
                full_before = (
                    full_hist * (full_bins < full_selected).to(torch.int32)
                ).sum(dtype=torch.int32)
                full_prefix = full_prefix | (
                    (255 - full_selected).to(torch.int64) << full_shift
                )
                full_remaining = full_remaining - full_before
            full_tie = valid & (keys == full_prefix)
            full_above = valid & (keys > full_prefix)
            full_slots = hl.zeros([2], dtype=torch.int32)
            full_ticket = hl.atomic_add(
                full_slots,
                [full_tie.to(torch.int32)],
                (full_above | full_tie).to(torch.int32),
            )
            full_position = torch.where(
                full_above, full_ticket, k - full_remaining + full_ticket
            )
            hl.store(
                result,
                [row, full_position],
                columns.to(torch.int32),
                extra_mask=full_above | (full_tie & (full_ticket < full_remaining)),
            )
    return result


SHAPES = [(16, 8192, 1024)]  # rows, maximum length, K


def make_inputs(
    shape: tuple[int, int, int], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Reproduce the measured mixed-length distribution and fixed ABI."""
    if shape not in SHAPES:
        raise ValueError(f"No measured variable-length top-k recipe for {shape}")
    rows, width, k = shape
    generator = torch.Generator(device=device).manual_seed(73)
    logits = torch.randn(rows, width, device=device, generator=generator)
    lengths = torch.randint(
        0, width + 1, (rows,), device=device, dtype=torch.int32, generator=generator
    )
    lengths[:4] = torch.tensor([0, k - 1, k, width], device=device, dtype=torch.int32)
    return logits, lengths, k, 1, 1


def _torch_reference(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> torch.Tensor:
    rows, width = logits.shape
    row = torch.arange(rows, device=logits.device)
    count = (
        (lengths[row // next_n] - next_n + row % next_n + 1) // compress_ratio
    ).clamp(0, width)
    columns = torch.arange(width, device=logits.device)
    output_columns = torch.arange(k, device=logits.device)
    masked = torch.where(columns[None, :] < count[:, None], logits, -torch.inf)
    # Stability keeps valid -Inf entries ahead of the invalid row tail.
    selected = masked.argsort(dim=-1, descending=True, stable=True)[:, :k]
    selected = torch.where(count[:, None] <= k, output_columns[None, :], selected)
    return torch.where(output_columns[None, :] < count[:, None], selected, -1).to(
        torch.int32
    )


def check_output(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    output: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> None:
    """Validate every selected value, index and padding slot, allowing ties."""
    rows, width = logits.shape
    assert output.shape == (rows, k)
    assert output.dtype == torch.int32 and output.device == logits.device
    for row in range(rows):
        count = max(
            0,
            min(
                width,
                (int(lengths[row // next_n].item()) - next_n + row % next_n + 1)
                // compress_ratio,
            ),
        )
        indices = output[row]
        assert bool((indices >= -1).all())
        selected = indices[indices >= 0].long()
        assert selected.numel() == min(k, count)
        assert selected.unique().numel() == selected.numel()
        assert bool((selected < count).all())
        assert int((indices == -1).sum().item()) == max(k - count, 0)
        if count <= k:
            expected = torch.arange(k, device=logits.device, dtype=torch.int32)
            expected[count:] = -1
            torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        if selected.numel():
            actual_values = logits[row, selected].sort().values
            expected_values = (
                logits[row, :count].topk(min(k, count)).values.sort().values
            )
            # The original numerical checker does not admit selected NaNs.
            torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


@dataclass
class _CapturedCall:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    logits: torch.Tensor
    lengths: torch.Tensor

    def __call__(self) -> torch.Tensor:
        self.graph.replay()
        return self.output


def check_case(shape: tuple[int, int, int]) -> tuple:
    """Check direct and poisoned graph calls, then return them for benchmarking."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import capture_cuda_graph  # pyrefly: ignore[missing-import]

    logits, lengths, k, next_n, compress_ratio = make_inputs(shape)
    original_logits, original_lengths = logits.clone(), lengths.clone()

    def helion_call() -> torch.Tensor:
        return varlen_topk(logits, lengths, k, next_n, compress_ratio)

    def torch_call() -> torch.Tensor:
        return _torch_reference(logits, lengths, k, next_n, compress_ratio)

    def check(output: torch.Tensor) -> None:
        torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
        torch.testing.assert_close(lengths, original_lengths, rtol=0, atol=0)
        check_output(
            original_logits, original_lengths, k, output, next_n, compress_ratio
        )

    def capture_checked(call: Callable[[], torch.Tensor]) -> _CapturedCall:
        check(call())
        graph, output = capture_cuda_graph(call)
        # -2 detects missing writes to both selected indices and -1 padding.
        output.fill_(-2)
        graph.replay()
        check(output)
        return _CapturedCall(graph, output, logits, lengths)

    return (
        capture_checked(helion_call),
        [("torch", capture_checked(torch_call))],
        f"{shape[0]:>5d}  {shape[1]:>5d}  {shape[2]:>4d}",
    )


def use_cudagraph() -> bool:
    return True


def correctness_check() -> None:
    for shape in SHAPES:
        check_case(shape)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    return run_sweep(
        SHAPES,
        check_case,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        rep=100,
        thermal_warmup_ms=1000,
        verbose=verbose,
        shape_header=f"{'rows':>5s}  {'N':>5s}  {'K':>4s}",
    )


if __name__ == "__main__":
    main()
