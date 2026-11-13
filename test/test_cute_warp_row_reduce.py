from __future__ import annotations

import pytest
import torch

from helion._compiler.cute import _paged_attention_paired_pipeline
from helion._compiler.cute.device import warp_reduce
from helion._testing.cute_warp_row_reduce import assert_same_float32_bits
from helion._testing.cute_warp_row_reduce import reference_row_sums
from helion.runtime.cute.host_wrappers.mma_dag import EMITTERS
from helion.runtime.cute.source_dependencies import referenced_helper_sources

pytestmark = pytest.mark.cpu_only


def test_warp_row_sum_preserves_the_rounded_tree(monkeypatch):
    # Exercise the actual Python staging algorithm on CPU vectors. This
    # independent reference reduces each row without transposing ownership.
    lanes = torch.arange(32)
    monkeypatch.setattr(warp_reduce, "_select_float32", torch.where)
    monkeypatch.setattr(
        warp_reduce.cute.arch,
        "shuffle_sync_bfly",
        lambda value, mask: value[lanes ^ mask],
    )
    generator = torch.Generator().manual_seed(1731)
    values = torch.ldexp(
        torch.randn((32, 32), generator=generator),
        torch.randint(-100, 100, (32, 32), generator=generator),
    )
    values[0] = -0.0
    values[1, :4] = torch.tensor([float("inf"), -float("inf"), 1, -1])
    values[2, 0] = float("nan")
    for count in (1, 2, 4, 8, 16, 32):
        expected = reference_row_sums(values[:count])
        rows, total = warp_reduce.warp_row_sum(list(values[:count]), lanes)
        assert sorted(rows[:count].tolist()) == list(range(count))
        actual = torch.empty_like(expected)
        actual[rows[:count]] = total[:count]
        assert_same_float32_bits(actual, expected)


@pytest.mark.parametrize("count", [0, 3, 33])
def test_warp_row_sum_rejects_unsupported_row_lists(count):
    with pytest.raises(AssertionError):
        warp_reduce.warp_row_sum([None] * count, None)


def test_paired_epilogue_fingerprints_the_row_helper():
    sources = referenced_helper_sources({"paired": _paged_attention_paired_pipeline})
    assert sources is not None
    assert "_compiler/cute/device/warp_reduce.py" in dict(sources)
    paired = next(
        emitter for emitter in EMITTERS if emitter.kind == "helion_paged_attention"
    )
    assert "_compiler/cute/device/warp_reduce.py" in paired.source_dependencies
