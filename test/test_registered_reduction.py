"""Registered reduction extents remain correct across workspace sizes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.registered_reduction import partitioned_copy_sum
from helion._testing.registered_reduction import registered_chunk_max


# The Triton allowlist includes TileIR, which needs its own barrier skip.
@skipUnlessBackends(["triton", "cute"])
@skipIfRefEager("requires compiled configurations")
@skipIfTileIR("TileIR does not support barrier operations")
def test_materialized_registered_reduction_sizes():
    x = (torch.arange(7 * 11, device=DEVICE).reshape(7, 11) % 13 - 6).float()
    bound = partitioned_copy_sum.bind((x,))
    default = dict(bound.config_spec.default_config())
    # CuTe uses separate materialized phases for the explicit barrier.
    if bound.env.backend_name == "cute":
        default["pid_type"] = "flat"
    for parts in (1, 2, 4, 8):
        config = default | {"block_sizes": [parts]}
        compiled = bound.compile_config(helion.Config.from_dict(config))
        for delta in (0, 1):
            changed = x + delta
            torch.testing.assert_close(compiled(changed), changed, rtol=0, atol=0)


@skipUnlessBackends(["triton", "cute"])
@skipIfRefEager("requires compiled configurations")
def test_masked_registered_arange_maximum():
    x = -(torch.arange(3 * 97, device=DEVICE).reshape(3, 97) + 1).float()
    bound = registered_chunk_max.bind((x,))
    for chunk in (16, 32, 64):
        config = dict(bound.config_spec.default_config()) | {"block_sizes": [chunk]}
        compiled = bound.compile_config(helion.Config.from_dict(config))
        parts = helion.cdiv(x.size(1), chunk)
        for delta in (0, 0.5):
            changed = x + delta
            expected = (
                F.pad(changed, (0, parts * chunk - x.size(1)), value=float("-inf"))
                .reshape(x.size(0), parts, chunk)
                .amax(dim=-1)
            )
            torch.testing.assert_close(compiled(changed), expected, rtol=0, atol=0)
