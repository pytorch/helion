"""GPU tests: ``cute_tail_peel`` computes the online softmax statistics of rows
with a partial last tile bit for bit as the knob-off kernel does."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.tail_peel_kernels import STATS_CONFIG
from helion._testing.tail_peel_kernels import online_softmax_stats
from helion._testing.tail_peel_kernels import padded_rows


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
