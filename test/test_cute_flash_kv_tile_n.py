"""A non-default KV tile width must reach the per-tile (local TMA) partition.

The local TMA partition tiled K/V with a hard-coded 128 while the smem stages
were ``kv_tile_n`` wide, so every ``fa4_local_tma*`` config with
``cute_flash_kv_tile_n=160`` failed to compile ("expects smem and gmem have the
same size in the first rank").
"""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


def _attention_kernel() -> helion.Kernel:
    from examples.attention import attention

    return helion.kernel(
        attention.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


_LOCAL_TMA_160 = {
    "block_sizes": [1, 128, 128],
    "cute_flash_pipeline_family": "fa4_local_tma",
    "cute_flash_persistent": True,
    "cute_flash_kv_tile_n": 160,
    "cute_flash_kv_order": "descending",
    "cute_flash_softmax_disc": False,
}


@onlyBackends(["cute"])
def test_local_tma_partition_tiles_kv_with_the_configured_width() -> None:
    q, k, v = (
        torch.empty(1, 4, 256, 64, dtype=torch.float16, device=DEVICE) for _ in range(3)
    )
    bound = _attention_kernel().bind((q, k, v))
    code = bound.to_code(bound._normalized_config_copy(helion.Config(**_LOCAL_TMA_160)))
    assert "cute.local_tile(flash_mK_cur, (160, 64), (None, 0))" in code
    assert "cute.local_tile(flash_mV_cur, (64, 160), (0, None))" in code


@onlyBackends(["cute"])
def test_local_tma_partial_kv_tile_matches_reference() -> None:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 4, 256, 64, dtype=torch.float16, device=DEVICE) for _ in range(3)
    )
    expected_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
    expected_lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    code, (out, lse) = code_and_output(_attention_kernel(), (q, k, v), **_LOCAL_TMA_160)
    assert "(160, 64), (None, 0))" in code
    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)
