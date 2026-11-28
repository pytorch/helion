"""Numerical coverage for persistent 3/4 row stripes and their first-tile writer."""

from __future__ import annotations

from types import FunctionType

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cuda_graph import cuda_graph_for_replay
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _projection_and_publication(
    x: torch.Tensor,
    w: torch.Tensor,
    temporary: torch.Tensor,
    rows_out: torch.Tensor,
    first_out: torch.Tensor,
) -> None:
    rows, k = w.shape
    block_k = hl.register_block_size(k)
    for producer in hl.tile(k):
        temporary[producer] = x[producer] * 2
    for row in hl.tile(rows):
        carry = hl.zeros([row, block_k], dtype=torch.float32)
        for col in hl.tile(k, block_size=block_k):
            carry += w[row, col].float() * temporary[col].float()[None, :]
        rows_out[row] = torch.sum(carry, dim=-1)
        if row.begin == 0:
            first_out[0] = temporary[0].float()


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[256, 32, 64],
        num_threads=[1, 32, 1],
        cross_loop_pipeline="barrier",
        pid_type="persistent_interleaved",
        num_sm_multiplier=1,
        cute_fragment_row_carry=True,
        cute_fragment_reduction="warp",
        cute_fragment_compact_rows=True,
    )


def test_compact_carry_publishes_768_rows_and_first_tile_twice() -> None:
    if torch.cuda.get_device_capability(DEVICE) < (8, 0):
        pytest.skip("BF16 persistent warp row reduction requires SM80+")
    rows, k = 768, 512
    rng = torch.Generator(device=DEVICE).manual_seed(23)
    # Signed integer-valued inputs give an exact FP32 oracle. Offset both
    # outputs so a wrong stripe stride or wrong first-tile store hits a canary.
    x = torch.randint(-4, 5, (k,), device=DEVICE, generator=rng).bfloat16()
    w = torch.randint(-4, 5, (rows, k), device=DEVICE, generator=rng).bfloat16()
    sentinel = 1234567.0
    output = torch.full((rows + 2,), sentinel, device=DEVICE)
    scalar = torch.full((3,), sentinel, device=DEVICE)
    temp = torch.empty_like(x)
    args = x, w, temp, output[1:-1], scalar[1:2]
    bound = _projection_and_publication.bind(args)
    config = _config()
    call = bound.compile_config(config)
    call(*args)
    # Inspect the actual loaded host program: a regular 64-row scalar path
    # would produce the same values, but would miss the intended schedule.
    assert isinstance(call, FunctionType)
    (axis,) = bound.config_spec.cute_fragment_compact_row_extents
    assert call.__globals__[f"_BLOCK_SIZE_{axis}"] == 48
    assert call.__globals__[f"_helion_{call.__name__}"]._helion_cute_cooperative
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            call(*args)
        for first in (3, -7):
            x.neg_()
            x[0] = first
            w[:, 0].neg_()
            output.fill_(sentinel)
            scalar.fill_(sentinel)
            graph.replay()
            torch.testing.assert_close(
                output[1:-1], w.float() @ (x * 2).float(), atol=0, rtol=0
            )
            assert scalar.tolist() == [sentinel, float(first * 2), sentinel]
            assert output[0].item() == output[-1].item() == sentinel
