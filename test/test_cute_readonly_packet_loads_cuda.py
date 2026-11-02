"""GPU numerics for read-only noncoherent packet loads and L2 evict-last
packet stores (see test_cute_readonly_packet_loads.py and
test_cute_store_policy.py for the CPU codegen).

A cache policy changes only the opcode of a proven packet: every hinted
render must match the reference bit for bit.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.cache_policy_loads import _CUTE_NC_L2_FIRST_HELPERS
from helion._compiler.cute.cutedsl_compat import (
    fixed_l2_evict_last_store_policy_supported,
)
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        out[tile] = (x[tile] * 2).to(out.dtype)
    return out


@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0), reason="ld.global.nc.v8 needs SM100+")
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.bfloat16, 8),
        (torch.bfloat16, 16),
        (torch.float32, 4),
        (torch.float32, 8),
    ],
)
def test_readonly_packets_load_every_bit(
    dtype: torch.dtype, width: int, pdl: bool
) -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 1024, device=DEVICE).to(dtype)
    out = torch.empty_like(x)
    # Isolated bindings keep each pinned configuration independent.
    bound = _scale_into._bind_isolated((x, out))
    config = helion.Config(
        block_sizes=[4, 256],
        num_threads=[1, 256 // width],
        cute_vector_widths=[1, width],
        load_eviction_policies=["nc_l2_first"],
        cute_pdl=pdl,
    )
    helper = _CUTE_NC_L2_FIRST_HELPERS[width * dtype.itemsize]
    assert f"{helper}(x.iterator" in bound.to_code(config)
    fn = bound.compile_config(config)
    for _ in range(2):
        # A producer rewrites every source element ahead of each launch: the
        # read-only packets must observe it after the ``cute_pdl`` wait.
        x.add_(1)
        fn(x, out)
    torch.testing.assert_close(out, x * 2, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        column = hl.arange(x.size(1))
        data = x[row, column].to(torch.float32)
        maximum = data.abs().amax(-1)
        out[row, column] = (data / (maximum[:, None] + 1)).to(out.dtype)
    return out


@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
@pytest.mark.parametrize(
    "out_dtype,width",
    [(torch.bfloat16, 8), (torch.float32, 4), (torch.float8_e4m3fn, 16)],
)
def test_l2_evict_last_packets_store_every_bit(
    out_dtype: torch.dtype, width: int
) -> None:
    if not fixed_l2_evict_last_store_policy_supported(
        torch.cuda.get_device_capability(), torch.version.cuda
    ):
        pytest.skip("the fixed L2 store descriptor is validated for SM103/CUDA13")
    torch.manual_seed(0)
    x = torch.randn(128, 32, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(128, 32, device=DEVICE, dtype=out_dtype)
    bound = _rows_into._bind_isolated((x, out))
    config = helion.Config(
        block_sizes=[64],
        num_threads=[32, 32 // width],
        cute_vector_widths=[width, 1],
        cute_lane_layouts=["blocked", "blocked"],
        cute_reduction_reloads=["register"],
        load_eviction_policies="nc_l2_first",
        cute_store_policy="l2_evict_last",
    )
    assert "_l2_evict_last(out.iterator" in bound.to_code(config)
    bound.compile_config(config)(x, out)
    data = x.float()
    reference = (data / (data.abs().amax(-1, keepdim=True) + 1)).to(out_dtype)
    assert torch.equal(out.view(torch.uint8), reference.view(torch.uint8))
