"""Fused head-split stores through the hierarchical TMA destination (sm_100)."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_radix_store import HEAD_MERGE_CONFIG
from helion._testing.cute_radix_store import ONE_CTA_CONFIG
from helion._testing.cute_radix_store import PAIR_CONFIG
from helion._testing.cute_radix_store import PERSISTENT_ONE_CTA_VALUES
from helion._testing.cute_radix_store import TRANSPOSED_HEAD_MERGE_CONFIG
from helion._testing.cute_radix_store import head_merge_args
from helion._testing.cute_radix_store import head_merge_bmm
from helion._testing.cute_radix_store import head_merge_reference
from helion._testing.cute_radix_store import head_split_args
from helion._testing.cute_radix_store import head_split_mm
from helion._testing.cute_radix_store import head_split_reference
from helion._testing.cute_radix_store import scaled_head_split_mm
from helion._testing.cute_radix_store import seq384_head_split_mm
from helion._testing.cute_radix_store import short_head_split_mm
from helion._testing.cute_radix_store import transposed_head_merge_bmm
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


@pytest.mark.parametrize(
    ("kernel", "shape", "config", "scale"),
    (
        (scaled_head_split_mm, (16, 16, 512), PAIR_CONFIG, 0.5),
        (
            short_head_split_mm,
            (16, 4, 64),
            helion.Config(block_sizes=[64, 128, 64], **PERSISTENT_ONE_CTA_VALUES),
            1.0,
        ),
        (
            head_split_mm,
            (2, 4, 512),
            helion.Config(block_sizes=[128, 256, 64], **PERSISTENT_ONE_CTA_VALUES),
            1.0,
        ),
    ),
    ids=("scaled_epilogue", "short_runs", "radix_subtile"),
)
def test_rank2_head_splits_match_the_matmul(
    kernel: helion.Kernel,
    shape: tuple[int, int, int],
    config: helion.Config,
    scale: float,
) -> None:
    """A scaled chain ahead of the TMA store on the pair, 64-row runs on
    64-row tiles and the automatic radix subtile of a one-CTA tile without
    an explicit epilogue tile all store through the layout."""
    torch.manual_seed(0)
    batch, heads, seq = shape
    a, b, out = head_split_args(batch, heads, seq=seq, device=DEVICE)
    bound = kernel.bind((a, b, out))
    assert "d_hier_layout" in bound.to_code(config)
    bound.set_config(config)
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(
        out, head_split_reference(a, b * scale, out.shape), rtol=2e-2, atol=1e-1
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


@pytest.mark.parametrize("default", (False, True), ids=("pinned", "default"))
@pytest.mark.parametrize("transposed", (False, True), ids=("rows", "transposed"))
def test_batched_head_merge_matches_the_bmm(transposed: bool, default: bool) -> None:
    """A batched bmm's head merge (scaled), row-major and transposed, lands
    every element where the permuted torch result has it, at a pinned config
    and at the default."""
    torch.manual_seed(0)
    kernel, config, shape = (
        (transposed_head_merge_bmm, TRANSPOSED_HEAD_MERGE_CONFIG, {"m": 64, "n": 256})
        if transposed
        else (head_merge_bmm, HEAD_MERGE_CONFIG, {})
    )
    a, b, out = head_merge_args(**shape, transposed=transposed, device=DEVICE)
    bound = kernel.bind((a, b, out))
    if default:
        config = bound.config_spec.default_config()
    assert "d_hier_layout" in bound.to_code(config)
    bound.set_config(config)
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(
        out, head_merge_reference(a, b, transposed=transposed), rtol=2e-2, atol=1e-1
    )


# The defaults the store used to refuse: the wide heads' small grid
# (the 128-row pair before), K and M not whole in 64-deep k-blocks or
# 128-row tiles (the 64x16 base tile before).
@pytest.mark.parametrize(
    "shape",
    ({"batch": 2, "n": 128, "k": 64}, {"batch": 16, "k": 96}, {"batch": 16, "m": 192}),
    ids=("wide_small_grid", "k96", "m192"),
)
def test_batched_head_merge_default_tiles_match_the_bmm(
    shape: dict[str, int],
) -> None:
    torch.manual_seed(0)
    a, b, out = head_merge_args(**shape, device=DEVICE)
    bound = head_merge_bmm.bind((a, b, out))
    config = bound.config_spec.default_config()
    assert "d_hier_layout" in bound.to_code(config)
    bound.set_config(config)
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, head_merge_reference(a, b), rtol=2e-2, atol=1e-1)
