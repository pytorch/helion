"""GPU tests: ``cute_tail_peel`` computes the online softmax statistics of rows
with a partial last tile bit for bit as the knob-off kernel does, and the
unmasked complete tiles of a sunk split reduction sum the same rows."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.split_reduction_kernels import SPLIT_CONFIG
from helion._testing.split_reduction_kernels import split_column_sums
from helion._testing.split_reduction_kernels import split_column_sums_reference
from helion._testing.tail_peel_kernels import PHASE_CONFIG
from helion._testing.tail_peel_kernels import STATS_CONFIG
from helion._testing.tail_peel_kernels import online_softmax_stats
from helion._testing.tail_peel_kernels import padded_rows
from helion._testing.tail_peel_kernels import phased_tail_rows


@pytest.mark.parametrize("pid_type", ["flat", "persistent_blocked"])
def test_peeled_rows_are_bitwise_the_masked_rows(pid_type: str) -> None:
    torch.manual_seed(0)
    # 5003 columns: nineteen complete 256-column tiles and 139 more, in rows
    # padded to 5008 elements, so each row starts 16-byte aligned.  Scaled in
    # place, the rows keep that stride (a product would be a contiguous copy,
    # whose rows no packet can load).
    logits = padded_rows(64, 5003, DEVICE).mul_(4)
    assert logits.stride() == (5008, 1)
    kernels = [
        helion.kernel(
            online_softmax_stats.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                **STATS_CONFIG, pid_type=pid_type, cute_tail_peel=peel
            ),
        )
        for peel in (False, True)
    ]
    bound = kernels[1].bind((logits,))
    peeled = helion.Config(**STATS_CONFIG, pid_type=pid_type, cute_tail_peel=True)
    code = bound.to_code(bound.config_spec.normalized_config(peeled))
    assert re.search(r"^\s*tail_packet\w* = cute\.arch\.load\(", code, re.MULTILINE)
    results = [kernel(logits) for kernel in kernels]
    for on, off in zip(results[1], results[0], strict=True):
        assert torch.equal(on.view(torch.int32), off.view(torch.int32))
    row_max, row_sum = results[1]
    x = logits.float()
    torch.testing.assert_close(row_max, x.amax(-1), rtol=0, atol=0)
    torch.testing.assert_close(
        row_sum, torch.exp(x - x.amax(-1, keepdim=True)).sum(-1), rtol=1e-5, atol=0
    )


@pytest.mark.parametrize(
    ("rows", "rows_per_split", "knobs"),
    [
        # Every row of the complete tiles proven: all packets unpredicated.
        (4800, 600, {}),
        # A partial last split: the eighth lane keeps its predicate.
        (4700, 600, {}),
        # Tiles that divide a split: unmasked in place.
        (5120, 640, {}),
        # Sixteen warps (128 row threads beside 4 column threads), their
        # combine deferred past the loops.
        (
            4800,
            600,
            {
                "block_sizes": [1, 32, 512],
                "num_threads": [1, 4, 128],
                "cute_lane_unroll": 4,
                "cute_defer_tile_reduce": True,
            },
        ),
    ],
    ids=["proven", "partly_proven", "dividing_tiles", "deferred_sixteen_warps"],
)
def test_unmasked_sunk_tiles_sum_the_same_rows(
    rows: int, rows_per_split: int, knobs: dict[str, object]
) -> None:
    kernels = [
        helion.kernel(
            split_column_sums.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict(
                {**SPLIT_CONFIG, **knobs, "cute_tail_peel": peel}
            ),
        )
        for peel in (False, True)
    ]
    shape = (rows, 64)
    empty = torch.empty(shape, dtype=torch.bfloat16, device=DEVICE)
    bound = kernels[1].bind((empty, empty, rows_per_split))
    code = bound.to_code(bound.config_spec.normalized_config(kernels[1].configs[0]))
    complete = code.split("for tile_offset_2 in range(", 2)[1]
    assert "cute.arch.load(" in complete
    torch.manual_seed(0)
    # Small integers: every product and partial sum is exact in fp32, so
    # neither a fused multiply-add nor the order of the adds can show.
    x = torch.randint(-4, 5, shape, device=DEVICE).to(torch.bfloat16)
    y = torch.randint(-4, 5, shape, device=DEVICE).to(torch.bfloat16)
    expected = split_column_sums_reference(x, y, rows_per_split)
    for kernel in kernels:
        for actual, reference in zip(
            kernel(x, y, rows_per_split), expected, strict=True
        ):
            assert torch.equal(actual.double(), reference)
    # Random values: the sums of ``x`` have no product to fuse and match bit
    # for bit; the products' sums may differ by a fused multiply-add.
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    (total_off, dot_off), (total_on, dot_on) = (
        kernel(x, y, rows_per_split) for kernel in kernels
    )
    assert torch.equal(total_on.view(torch.int32), total_off.view(torch.int32))
    torch.testing.assert_close(dot_on, dot_off, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("pid_type", ["persistent_interleaved", "persistent_blocked"])
def test_phased_tail_reduces_partial_rows_before_the_barrier(pid_type: str) -> None:
    # Signed integer BF16 values have an exact FP32 sum. Poison the backing
    # padding so an unmasked final packet cannot silently add padded values.
    backing = torch.randint(-4, 5, (8, 1008), device=DEVICE).to(torch.bfloat16)
    backing[:, 1003:] = float("nan")
    rows = backing[:, :1003]
    config = helion.Config(**PHASE_CONFIG, pid_type=pid_type, cute_tail_peel=True)
    bound = phased_tail_rows.bind((rows,))
    assert "tail_packet =" in bound.to_code(bound.config_spec.normalized_config(config))
    actual = bound.compile_config(config)(rows)
    torch.testing.assert_close(actual, rows.float().sum(1) + 1, atol=0, rtol=0)
