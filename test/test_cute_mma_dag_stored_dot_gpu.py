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
