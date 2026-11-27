"""A 16-bit ``torch.mm`` / ``torch.bmm`` stored as it is by ``cute_mma_dag``.

The store writes the fp32 product to the 16-bit tensor and so rounds it once:
bit for bit what a pointwise reader that rounds it, ``.to(torch.float32)``,
writes.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_mma_dag_stored import bmm_read
from helion._testing.cute_mma_dag_stored import bmm_stored
from helion._testing.cute_mma_dag_stored import mm_read
from helion._testing.cute_mma_dag_stored import mm_stored
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _kernel(fn: Callable[..., None]) -> helion.Kernel:
    # Autotuning off, as in helion.inductor's tests: the stored mm is then
    # cute_mma_dag's (an autotuned bind may offer other matmul owners).
    return helion.kernel(fn, backend="cute", static_shapes=True, autotune_effort="none")


def _run(
    fn: Callable[..., None],
    a: torch.Tensor,
    b: torch.Tensor,
    out_shape: tuple[int, ...],
    config: dict[str, object],
) -> torch.Tensor:
    # A fresh output of NaNs, so that a store that skips elements shows.
    out = torch.full(out_shape, float("nan"), device=DEVICE, dtype=torch.bfloat16)
    bound = _kernel(fn).bind((a, b, out))
    bound.set_config(helion.Config(**config))
    bound(a, b, out)
    return out


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("batched", [False, True])
def test_a_stored_16_bit_matmul_is_the_rounded_product(batched: bool) -> None:
    torch.manual_seed(0)
    if batched:
        a = torch.randn(8, 128, 64, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
        out_shape = (8, 128, 128)
        stored, read = bmm_stored, bmm_read
    else:
        a = torch.randn(512, 64, device=DEVICE, dtype=torch.bfloat16)
        # A column-major b, as nn.Linear's weight.t() reaches a matmul.
        b = torch.randn(256, 64, device=DEVICE, dtype=torch.bfloat16).t()
        out_shape = (512, 256)
        stored, read = mm_stored, mm_read
    # One config for both, in cute_mma_dag="warp": the default on sm100 and
    # sm103, a seed elsewhere.
    out = torch.empty(out_shape, device=DEVICE, dtype=torch.bfloat16)
    spec = _kernel(stored).bind((a, b, out))
    spec = spec.config_spec
    assert spec.cute_mma_dag is not None
    default = spec.default_config()
    warp = next(
        c
        for c in (default, *spec.compiler_seed_configs)
        if c.config.get("cute_mma_dag") == "warp"
    )
    config = {**default.config, **warp.config}
    expected = _run(read, a, b, out_shape, config)
    torch.testing.assert_close(
        expected.float(), torch.matmul(a.float(), b.float()), atol=0.15, rtol=0.02
    )
    assert torch.equal(_run(stored, a, b, out_shape, config), expected)


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("m", [56, 49])
def test_one_cta_covers_a_small_ragged_matrix(m: int) -> None:
    # A 64 x 64 tile over each 56 x 56 matrix (N ragged, whole 16-byte
    # packets) and each 49 x 49 one (98-byte rows: element by element). 4100
    # matrices put the last matrix's last rows at the end of the tensors,
    # where a packet read past N would fault.
    torch.manual_seed(0)
    batch = 4100
    a = torch.randn(batch, m, 32, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(batch, 32, m, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty((batch, m, m), device=DEVICE, dtype=torch.bfloat16)
    spec = _kernel(bmm_stored).bind((a, b, out)).config_spec
    assert spec.cute_mma_dag is not None
    config = {
        **spec.default_config().config,
        "block_sizes": [1, 64, 64],
        "cute_mma_dag": "warp",
        "cute_mma_dag_warps": 4,
    }
    stored = _run(bmm_stored, a, b, (batch, m, m), config)
    torch.testing.assert_close(
        stored.float(), torch.bmm(a.float(), b.float()), atol=0.15, rtol=0.02
    )
    assert torch.equal(stored, _run(bmm_read, a, b, (batch, m, m), config))


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize(("k", "a_contiguous"), [(49, "m"), (49, "k"), (8, "m")])
def test_a_k_row_that_is_no_power_of_two_runs_padded(k: int, a_contiguous: str) -> None:
    # Window attention's backward attn.mT @ grad over 7x7 windows: K = 49 on
    # a 64-wide row, the elements past 49 zero-filled; a column-major A (M
    # contiguous) and a row-major one, both 98-byte rows. K = 8, under one
    # 16-element MMA K step, runs on a 16-wide row. 4100 matrices put the
    # last matrix's rows at the tensors' ends, where a K row loaded past the
    # extent would read past them.
    torch.manual_seed(0)
    batch = 4100
    a = torch.randn(batch, 49, k, device=DEVICE, dtype=torch.bfloat16)
    if a_contiguous == "m":
        a = torch.randn(batch, k, 49, device=DEVICE, dtype=torch.bfloat16).mT
    b = torch.randn(batch, k, 32, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty((batch, 49, 32), device=DEVICE, dtype=torch.bfloat16)
    spec = _kernel(bmm_stored).bind((a, b, out)).config_spec
    assert spec.cute_mma_dag is not None
    config = {
        **spec.default_config().config,
        "block_sizes": [1, 64, 32],
        "cute_mma_dag": "warp",
        "cute_mma_dag_warps": 1,
    }
    stored = _run(bmm_stored, a, b, (batch, 49, 32), config)
    torch.testing.assert_close(
        stored.float(), torch.bmm(a.float(), b.float()), atol=0.15, rtol=0.02
    )
    assert torch.equal(stored, _run(bmm_read, a, b, (batch, 49, 32), config))


def _mm_over_n_tiles(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_m in hl.tile(a.size(0)):
        for tile_n in hl.tile(b.size(1)):
            out[tile_m, tile_n] = torch.mm(torch.exp(a[tile_m, :]), b[:, tile_n])


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("k", [49, 80])
def test_a_padded_k_row_in_a_tile_loop(k: int) -> None:
    # K = 49 (98-byte rows of A: the synchronous pass) and K = 80 (whole
    # 16-byte packets) on their padded rows inside a loop over N tiles, on
    # the warp MMA; exp(0) = 1 past the extent takes the frontend's zero
    # fill.
    torch.manual_seed(0)
    a = torch.randn(1024, k, device=DEVICE, dtype=torch.bfloat16) * 0.1
    b = torch.randn(k, 512, device=DEVICE, dtype=torch.bfloat16)
    out = torch.full((1024, 512), float("nan"), device=DEVICE, dtype=torch.bfloat16)
    bound = helion.kernel(
        _mm_over_n_tiles,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.FastMathUnused],
    ).bind((a, b, out))
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    bound.set_config(
        helion.Config(**{**spec.default_config().config, "cute_mma_dag": "warp"})
    )
    bound(a, b, out)
    torch.testing.assert_close(
        out.float(), torch.exp(a.float()) @ b.float(), atol=0.15, rtol=0.02
    )
