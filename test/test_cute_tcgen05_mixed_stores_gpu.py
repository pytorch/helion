"""Runtime numerics of tcgen05 epilogues whose stores differ in element type
(sm_100 GPU).

The CPU side, which pins the rendered subtiles, is
``test_cute_tcgen05_mixed_stores``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_mixed_stores import mm_mixed_stores
from helion._testing.cute_mixed_stores import mm_mixed_stores_args

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]


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
