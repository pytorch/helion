"""Compile checks for the fp32-row geometries the SMEM model admits (sm_100 GPU).

Each case is a ``pre_acc_wait`` row-vector GEMM whose 192 KiB AB ring, 32 KiB
c=4 ring and 2 KiB fp32 row stage fill the per-CTA opt-in to within a KiB
(231 588-231 716 B of 232 448 modelled; the cubins report 232 612-232 740 B
``SHARED`` with the 1 KiB reservation). The 3 KiB seed headroom demoted every
one of them to its ``post_acc_wait`` / c=2 twin; the arena model keeps them,
and they must compile, launch and agree with the reference.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

import helion
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES,
)
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]

LAUNCHES = 8
B200_OPTIN_BYTES = 232_448


def _f16_scale_gemm(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc * scale[tile_n]).to(x.dtype)
    return out


def _fp8_scale_gemm(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc * scale[tile_n]).to(torch.bfloat16)
    return out


class _Case(NamedTuple):
    operands: str
    bm: int
    bn: int
    bk: int
    cluster_m: int
    ab_stages: int
    values: dict[str, object]
    # Modelled arena bytes (``default_layout_smem_bytes``), exact for these
    # kernels: the compiled cubins report these plus the 1 KiB reservation.
    arena_bytes: int


def _two_cta(operands: str, bk: int, ab_stages: int, cluster_n: int) -> _Case:
    return _Case(
        operands,
        256,
        128,
        bk,
        2,
        ab_stages,
        {
            "block_sizes": [256, 128, bk],
            "pid_type": "persistent_interleaved",
            "l2_groupings": [4],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": cluster_n,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_c_stages": 4,
            "tcgen05_aux_load_placement": "pre_acc_wait",
        },
        231_596,
    )


def _one_cta(operands: str, bk: int, ab_stages: int) -> _Case:
    return _Case(
        operands,
        128,
        128,
        bk,
        1,
        ab_stages,
        {
            "block_sizes": [128, 128, bk],
            "pid_type": "persistent_interleaved",
            "tcgen05_cta_group": "auto",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_c_stages": 4,
            "tcgen05_acc_stages": 2,
            "l2_groupings": [1],
            "tcgen05_l2_swizzle_size": 1,
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_monolithic",
            "tcgen05_layout_strategy": "default",
            "tcgen05_aux_load_placement": "pre_acc_wait",
        },
        # A ring deeper than eight stages takes a second 128 B mbarrier chunk.
        231_716 if ab_stages > 8 else 231_588,
    )


_CASES = {
    "f16_256x128x64_cm2cn1_ab8": _two_cta("f16", 64, 8, 1),
    "f16_256x128x64_cm2cn2_ab8": _two_cta("f16", 64, 8, 2),
    "f16_256x128x128_cm2cn1_ab4": _two_cta("f16", 128, 4, 1),
    "f16_256x128x128_cm2cn2_ab4": _two_cta("f16", 128, 4, 2),
    "f16_128x128x32_ab12": _one_cta("f16", 32, 12),
    "f16_128x128x64_ab6": _one_cta("f16", 64, 6),
    "f16_128x128x128_ab3": _one_cta("f16", 128, 3),
    "fp8_256x128x128_cm2cn1_ab8": _two_cta("fp8", 128, 8, 1),
    "fp8_256x128x128_cm2cn2_ab8": _two_cta("fp8", 128, 8, 2),
    "fp8_128x128x64_ab12": _one_cta("fp8", 64, 12),
    "fp8_128x128x128_ab6": _one_cta("fp8", 128, 6),
}


def _inputs(operands: str) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    torch.manual_seed(0)
    if operands == "f16":
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
    else:
        x = (torch.randn(1024, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(1024, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
    scale = torch.rand(1024, device=DEVICE, dtype=torch.float32) + 0.5
    expected = (x.float() @ y.float()) * scale
    return (x, y, scale), expected


@pytest.mark.parametrize("case", sorted(_CASES), ids=sorted(_CASES))
def test_admitted_fp32_row_geometry_compiles_and_runs(case: str) -> None:
    c = _CASES[case]
    args, expected = _inputs(c.operands)
    kernel_fn = _f16_scale_gemm if c.operands == "f16" else _fp8_scale_gemm
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    spec = bound.config_spec
    # The search projection keeps the sampled knobs ...
    projected = dict(c.values)
    spec.normalize(projected, _fix_invalid=True)
    assert projected["tcgen05_ab_stages"] == c.ab_stages
    assert projected["tcgen05_c_stages"] == 4
    assert projected["tcgen05_aux_load_placement"] == "pre_acc_wait"
    # ... because the modelled arena (exact for these kernels) fits the opt-in.
    modelled = spec._cute_tcgen05_config.default_layout_smem_bytes(
        bm=c.bm,
        bn=c.bn,
        bk=c.bk,
        cluster_m=c.cluster_m,
        ab_stages=c.ab_stages,
        c_stages=4,
        stage_rows=True,
    )
    assert modelled == c.arena_bytes
    assert modelled + TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES <= B200_OPTIN_BYTES
    bound.set_config(helion.Config(**c.values))
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for out in outputs:
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=1e-1)
