"""Ordinary matrix recurrences must not be mistaken for attention state."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.analysis.conditional_attention_recurrence import (
    match_conditional_attention_recurrence,
)
from helion._compiler.cute.analysis.conditional_regions import prove_conditional_regions
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none", static_shapes=True)
def two_dot_recurrence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    for tile in hl.grid(x.shape[0] // 32):
        rows = tile * 32 + hl.arange(32)[:, None]
        columns = hl.arange(32)[None, :]
        weight = y[hl.arange(32)[:, None], columns]
        total = x[rows, columns].float()
        for _ in range(2):
            scores = hl.dot(total.to(torch.bfloat16), weight)
            total = hl.dot(scores.to(torch.bfloat16), weight, acc=total)
        out[rows, columns] = total
    return out


def test_two_dot_single_carry_declines_attention_match(caplog):
    with cpu_target():
        args = (torch.empty((32, 32), dtype=torch.bfloat16),) * 2
        bound = cpu_bind(two_dot_recurrence, args)
        ir = bound.host_function.device_ir
        regions = prove_conditional_regions(ir.graphs, ir.root_ids[0])
        assert regions is not None
        assert len(regions.carries) == 1
        assert match_conditional_attention_recurrence(ir, bound.env) is None
        with caplog.at_level(
            "DEBUG", logger="helion._compiler.cute.conditional_attention_codegen"
        ):
            source = bound.to_code(bound.config_spec.default_config())
        assert "_conditional_attention_page(" not in source
        assert "source, recurrence, or memory-effect proof did not match" in caplog.text
