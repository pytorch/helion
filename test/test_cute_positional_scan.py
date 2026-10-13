from __future__ import annotations

import pytest
import torch

from test import _positional_scan_kernels as S

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion.runtime.ref_mode import RefMode

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _eager(kernel, args):
    return helion.kernel(kernel.fn, ref_mode=RefMode.EAGER)(*args)


def _check(kernel, args, **config):
    code, result = code_and_output(kernel, args, **config)
    assert "ptp_thread" in code  # the positional route owns the root
    results = result if isinstance(result, tuple) else (result,)
    expected = _eager(kernel, args)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for actual, reference in zip(results, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["masked_cumsum", "masked_reverse_cumsum"])
def test_cumsum_directions(name: str) -> None:
    kernel = getattr(S, name)
    _check(kernel, (torch.randn(4, 32, 32, device=DEVICE),))


def test_tuple_noncommutative_scan_forward_and_reverse() -> None:
    a = torch.rand(4, 16, 16, device=DEVICE) + 0.5
    x = torch.randn(4, 16, 16, device=DEVICE)
    _check(S.affine_scan, (a, x))


def test_partial_tile_reverse_scan_ignores_tail_storage() -> None:
    x = torch.randn(3, 40, device=DEVICE)[:, :20]
    k = torch.randn(3, 16, 16, device=DEVICE)
    _check(S.tail_reverse_cumsum, (x, k))


@pytest.mark.parametrize("num_warps", [1, 4])
def test_bf16_state_rounds_each_step(num_warps: int) -> None:
    x = torch.randn(2, 32, 32, device=DEVICE).bfloat16()
    code, out = code_and_output(S.bf16_cumsum, (x,), num_warps=num_warps)
    assert "ptp_thread" in code
    torch.testing.assert_close(out, _eager(S.bf16_cumsum, (x,)), rtol=0, atol=0)


@pytest.mark.parametrize("diag_anchored", [False, True])
def test_engine_dqkg_scan_stage(diag_anchored: bool) -> None:
    from examples.linear import linear_attention_engine as engine

    torch.manual_seed(0)
    b, c, d, dv = 8, 64, 64 if not diag_anchored else 32, 64
    dtype = torch.bfloat16 if not diag_anchored else torch.float32
    q, k = (torch.randn(b, c, d, device=DEVICE, dtype=dtype) for _ in range(2))
    v = torch.randn(b, c, dv, device=DEVICE, dtype=dtype)
    do = torch.randn(b, c, dv, device=DEVICE, dtype=dtype)
    h = torch.randn(b, d, dv, device=DEVICE, dtype=dtype)
    dh = torch.randn(b, d, dv, device=DEVICE)
    if diag_anchored:
        g_cs = -torch.rand(b, c, d, device=DEVICE).cumsum(1) * 0.05
        g_last = None
    else:
        g_cs = -torch.rand(b, c, device=DEVICE).cumsum(-1) * 0.05
        g_last = g_cs[:, -1].contiguous()
    args = (q, k, v, g_cs, h, do, dh, g_last, diag_anchored, True, 0.125)
    kernel = helion.kernel(
        engine.chunk_bwd_dqkg_scalar_helion.fn, backend="cute", static_shapes=True
    )
    bound = kernel.bind(args)
    result = bound.compile_config(bound.config_spec.default_config())(*args)
    reference = _eager(engine.chunk_bwd_dqkg_scalar_helion, args)
    for actual, expected in zip(result, reference, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-2
        )


def test_padded_reverse_scan_matches_eager() -> None:
    _check(S.padded_reverse_exp_cumsum, (torch.randn(3, 40, 40, device=DEVICE) / 4,))
