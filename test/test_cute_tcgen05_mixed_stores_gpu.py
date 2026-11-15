"""Runtime numerics of tcgen05 epilogues whose stores differ in element type
or read epilogue inputs alone (sm_100 GPU).

The CPU side, which pins the rendered subtiles, is
``test_cute_tcgen05_mixed_stores``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_mixed_stores import bmm_three_stores
from helion._testing.cute_mixed_stores import bmm_three_stores_args
from helion._testing.cute_mixed_stores import mm_and_copy
from helion._testing.cute_mixed_stores import mm_and_copy_args
from helion._testing.cute_mixed_stores import mm_mixed_stores
from helion._testing.cute_mixed_stores import mm_mixed_stores_args

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]


# The default flat tile, and the persistent 128x64 tile whose BF16 store walks
# the FP32 stores' (128, 16) subtiles.
@pytest.mark.parametrize(
    "overrides",
    ({}, {"block_sizes": [1, 128, 64, 64], "pid_type": "persistent_blocked"}),
    ids=("default", "persistent_128x64"),
)
def test_three_stores_match_torch(overrides: dict[str, object]) -> None:
    args = bmm_three_stores_args(4, 512, 64, 256, device=DEVICE, random=True)
    a, b, aux, acc_out, aux_out, sum_out = args
    bound = bmm_three_stores.bind(args)
    config = bound.config_spec.default_config().config
    if overrides:
        config = {
            key: value
            for key, value in config.items()
            if key != "tcgen05_persistence_model"
        }
    bound.set_config(helion.Config(**{**config, **overrides}))
    bound(*args)
    acc = torch.bmm(a.float(), b.float())
    torch.testing.assert_close(acc_out, acc, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(aux_out, aux.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        sum_out, (acc + aux.float()).to(torch.bfloat16), atol=0.25, rtol=1.6e-2
    )


# The epilogue-issued aux ring (``epi_tma``), which holds the plan's FP32-sized
# subtile the BF16 store walks too, and the two-CTA 256x64 tile.
_MIXED_CONFIGS = {
    "epi_tma_128x64": (
        (1024, 512, 256),
        {
            "block_sizes": [128, 64, 64],
            "pid_type": "persistent_blocked",
            "tcgen05_aux_load_mode": "epi_tma",
        },
    ),
    "two_cta_256x64": (
        (2048, 1024, 512),
        {
            "block_sizes": [256, 64, 64],
            "tcgen05_cluster_m": 2,
            "pid_type": "persistent_interleaved",
        },
    ),
}


@pytest.mark.parametrize("config_name", sorted(_MIXED_CONFIGS))
def test_mixed_stores_match_torch(config_name: str) -> None:
    (m, n, k), overrides = _MIXED_CONFIGS[config_name]
    args = mm_mixed_stores_args(m, n, k, device=DEVICE, random=True)
    a, b, aux, acc_out, sum_out = args
    bound = mm_mixed_stores.bind(args)
    config = {
        key: value
        for key, value in bound.config_spec.default_config().config.items()
        if key != "tcgen05_persistence_model"
    }
    bound.set_config(helion.Config(**{**config, **overrides}))
    bound(*args)
    acc = a.float() @ b.float()
    torch.testing.assert_close(acc_out, acc, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(
        sum_out, (acc + aux.float()).to(torch.bfloat16), atol=0.25, rtol=1.6e-2
    )


@pytest.mark.parametrize("copy_first", (True, False), ids=("first", "last"))
def test_aux_only_store_first_or_last_matches_torch(copy_first: bool) -> None:
    """The copy that waits for the accumulator, or releases it, without a TMEM
    load of its own."""
    args = mm_and_copy_args(1024, 512, 256, device=DEVICE, random=True)
    a, b, aux, acc_out, aux_out = args
    mm_and_copy(*args, copy_first)
    torch.testing.assert_close(acc_out, a.float() @ b.float(), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(aux_out, aux.float() * 2.0, atol=0, rtol=0)


# Partial M and N tiles (1000 x 520), whose default config stores both outputs
# from the SIMT store body, and a collective-MMA config.
@pytest.mark.parametrize("config_kind", ("edge_tiles", "collective_mma"))
def test_aux_only_store_on_other_epilogue_paths_matches_torch(config_kind: str) -> None:
    args = mm_and_copy_args(1000, 520, 256, device=DEVICE, random=True)
    a, b, aux, acc_out, aux_out = args
    bound = mm_and_copy.bind((*args, False))
    config = {
        "edge_tiles": bound.config_spec.default_config(),
        "collective_mma": helion.Config(
            block_sizes=[64, 64, 32],
            cute_collective_mma=True,
            cute_collective_compute="tcgen05",
            cute_collective_copy="async_cached",
            cute_collective_stages=2,
        ),
    }[config_kind]
    bound.set_config(config)
    bound(*args, False)
    torch.testing.assert_close(acc_out, a.float() @ b.float(), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(aux_out, aux.float() * 2.0, atol=0, rtol=0)
