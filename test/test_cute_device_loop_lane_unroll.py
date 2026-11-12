from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import import_example
from helion._testing.cute_chunked_rows import row_shift
from helion._testing.nvfp4_gemv_args import nvfp4_gemv_args

_nvfp4_gemv_fp4in_body = import_example("nvfp4_gemv")._nvfp4_gemv_fp4in_body


def _config(groups: int, threads: int, unroll: int) -> helion.Config:
    return helion.Config(
        block_sizes=[1, groups],
        num_threads=[1, threads],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["blocked", "strided"],
        cute_lane_unroll=unroll,
    )


@pytest.mark.parametrize("unroll", [4, 8])
def test_device_loop_lane_loop_unrolls_loads_first(unroll: int) -> None:
    rows, kbytes = 64, 2048
    args, expected = nvfp4_gemv_args(rows, kbytes)
    config = _config(256, 32, unroll)
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body, backend="cute", static_shapes=True, config=config
    )
    source = kernel.bind(args).to_code(config)
    # The load phase and the compute phase: two constexpr copies of the lane
    # body, inside a rolled block loop when the unroll is partial.
    lane = "lane_1" if unroll == 8 else "lane_1_u"
    assert source.count(f"for {lane} in cutlass.range_constexpr({unroll}):") == 2
    assert "_lane_unroll_loads_" in source
    if unroll < 8:
        assert f"for lane_1_blk in range({8 // unroll}):" in source
    kernel(*args)
    torch.testing.assert_close(args[4].float(), expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("hint", [None, "streaming"])
def test_reduction_lane_unroll_is_bitwise_neutral(hint: str | None) -> None:
    """A looped reduction's lanes unrolled loads-first compute what the
    rolled lanes compute, bit for bit: the reduce sweep still adds its
    elements lane by lane and the consume sweep's loads only move ahead of
    stores to another tensor (``test_cute_reduction_lane_unroll.py`` checks
    the code).  With ``hint`` the fp32 ``y`` is read through hinted scalar
    loads, which the unroll moves too."""
    rows, n = 8, 4099
    # Odd rows: ``y`` and ``out`` are accessed per element, ``x`` by packets.
    x = torch.randn(rows, n + 5, device=DEVICE, dtype=torch.bfloat16)[:, :n]
    dtype = torch.bfloat16 if hint is None else torch.float32
    y = torch.randn(rows, n, device=DEVICE, dtype=dtype)
    outputs = []
    for unroll in (1, 8):
        config = helion.Config(
            block_sizes=[1],
            reduction_loops=[1024],
            num_threads=[1, 64],
            cute_vector_widths=[2, 1],
            cute_lane_unroll=unroll,
            load_eviction_policies=["", hint or ""],
        )
        kernel = helion.kernel(
            row_shift, backend="cute", static_shapes=True, config=config
        )
        if unroll > 1:
            code = kernel.bind((x, y)).to_code(config)
            # Were the split refused here (``LaneUnrollNotApplied``), the
            # rolled code would run and match itself.
            assert "_lane_unroll_loads_" in code
            if hint is not None:
                # ... and ``y``'s loads carry the hint.
                assert "_cute_scalar_policy_load(y.iterator" in code
        outputs.append(kernel(x, y))
    bits = torch.int16 if hint is None else torch.int32
    assert torch.equal(outputs[0].view(bits), outputs[1].view(bits))
    expected = (y.float() - x.float().sum(-1, keepdim=True)).to(dtype)
    torch.testing.assert_close(outputs[1], expected, rtol=2e-2, atol=2e-1)
