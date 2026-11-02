from __future__ import annotations

from examples.nvfp4_gemv import _nvfp4_gemv_fp4in_body
import pytest
import torch

from test.test_cute_tile_packing import _gemv_args

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _config(groups: int, threads: int, unroll: int) -> helion.Config:
    return helion.Config(
        block_sizes=[1, groups],
        num_threads=[1, threads],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["blocked", "strided"],
        cute_lane_unroll=unroll,
    )


def test_device_loop_lane_loop_keeps_the_unroll_knob() -> None:
    """A lane loop of an ``hl.tile`` axis inside the kernel (not a grid axis)
    forms a lane loop too, so ``cute_lane_unroll`` survives normalization."""
    rows, kbytes = 64, 2048
    args, _expected = _gemv_args(rows, kbytes)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        config=_config(256, 32, 8),
    )
    spec = kernel.bind(args).config_spec
    assert spec.normalized_config(_config(256, 32, 8)).config["cute_lane_unroll"] == 8
    # No lane loop at all (one element per thread): the knob is inert.
    assert spec.normalized_config(_config(256, 256, 8)).config["cute_lane_unroll"] == 1


@pytest.mark.parametrize("unroll", [4, 8])
def test_device_loop_lane_loop_unrolls_loads_first(unroll: int) -> None:
    rows, kbytes = 64, 2048
    args, expected = _gemv_args(rows, kbytes)
    config = _config(256, 32, unroll)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True, config=config
    )
    source = kernel.bind(args).to_triton_code(config)
    # The load phase and the compute phase: two constexpr copies of the lane
    # body, inside a rolled block loop when the unroll is partial.
    lane = "lane_1" if unroll == 8 else "lane_1_u"
    assert source.count(f"for {lane} in cutlass.range_constexpr({unroll}):") == 2
    assert "_lane_unroll_loads_" in source
    if unroll < 8:
        assert f"for lane_1_blk in range({8 // unroll}):" in source
    kernel(*args)
    torch.testing.assert_close(args[4].float(), expected.float(), rtol=0, atol=0)


def test_unrolled_lane_loop_is_off_without_the_knob() -> None:
    rows, kbytes = 64, 2048
    args, _expected = _gemv_args(rows, kbytes)
    config = _config(256, 32, 1)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True, config=config
    )
    source = kernel.bind(args).to_triton_code(config)
    assert "for lane_1 in range(8):" in source
    assert "_lane_unroll_loads_" not in source
