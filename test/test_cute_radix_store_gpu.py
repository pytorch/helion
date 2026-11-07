"""Fused head-split stores through the hierarchical TMA destination (sm_100)."""

from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_radix_store import ONE_CTA_CONFIG
from helion._testing.cute_radix_store import PAIR_CONFIG
from helion._testing.cute_radix_store import head_split_args
from helion._testing.cute_radix_store import head_split_mm
from helion._testing.cute_radix_store import head_split_reference
from helion._testing.cute_radix_store import seq384_head_split_mm
from helion._testing.cute_radix_store import transposed_store_mm

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="tcgen05 needs sm_100",
    ),
]


@pytest.mark.parametrize(
    ("batch", "heads", "seq"),
    ((2, 4, 512), (3, 12, 512), (2, 4, 384)),
    ids=("pair_pow2", "pair_bert_base_heads", "one_cta_seq384"),
)
def test_head_split_store_matches_permuted_matmul(
    batch: int, heads: int, seq: int
) -> None:
    torch.manual_seed(0)
    a, b, out = head_split_args(batch, heads, seq=seq, device=DEVICE)
    # 256-row pair tiles divide 512-row runs; 384-row runs need 128-row tiles.
    kernel, config = (
        (head_split_mm, PAIR_CONFIG)
        if seq == 512
        else (seq384_head_split_mm, ONE_CTA_CONFIG)
    )
    bound = kernel.bind((a, b, out))
    assert "d_hier_layout" in bound.to_code(config)
    bound.set_config(config)
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(
        out, head_split_reference(a, b, out.shape), rtol=2e-2, atol=1e-1
    )


def test_transposed_destination_matches_the_matmul() -> None:
    """``out[x, y] = acc[y, x]`` into the transpose of a row-major [M, N]:
    the TMA store writes the plain [M, N] view the proof found."""
    torch.manual_seed(0)
    a = torch.randn((1024, 256), dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn((256, 256), dtype=torch.bfloat16, device=DEVICE).T
    storage = torch.empty((1024, 256), dtype=torch.bfloat16, device=DEVICE)
    out = storage.T
    bound = transposed_store_mm.bind((a, b, out))
    assert "d_hier_layout" in bound.to_code(PAIR_CONFIG)
    bound.set_config(PAIR_CONFIG)
    storage.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(
        storage, (a.float() @ b.float()).to(torch.bfloat16), rtol=2e-2, atol=1e-1
    )
