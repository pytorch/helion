"""CPU codegen of tcgen05 epilogues whose stores differ in element type.

One matmul's output stores walk the epilogue tile of the widest store's
element type, each staging and storing it in its own. The GPU side is
``test_cute_tcgen05_mixed_stores_gpu``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.cute_mma import _tcgen05_epilogue_tile_dtype
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion._testing.cute_mixed_stores import mm_mixed_stores
from helion._testing.cute_mixed_stores import mm_mixed_stores_args

if TYPE_CHECKING:
    from collections.abc import Iterator

CPU_DEVICE = "cpu"

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    # B200's 148 SMs, which the default tile heuristics size waves from
    with (
        cpu_codegen_target(),
        forbid_cuda_work(),
        patch(
            "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
            return_value=148,
        ),
    ):
        yield


# A store's epilogue tile sized by FP32, whatever the store's own dtype.
_F32_TILE = re.compile(
    r"(tcgen05_store_epi_tile(?:_\d+)?) = cutlass\.utils\.blackwell_helpers\."
    r"compute_epilogue_tile_shape\(\(\d+, \d+\), False, [\w.]+, cutlass\.Float32,"
)


def test_epilogue_tile_dtype_is_the_widest_store() -> None:
    supported = (torch.float16, torch.bfloat16, torch.float32)
    bf16, f16, f32 = torch.bfloat16, torch.float16, torch.float32
    pick = _tcgen05_epilogue_tile_dtype
    assert pick(frozenset({bf16}), bf16, supported) is bf16
    assert pick(frozenset({bf16, f32}), bf16, supported) is f32
    # Equally wide outputs: the input dtype, else the first by name.
    assert pick(frozenset({bf16, f16}), f16, supported) is f16
    assert pick(frozenset({bf16, f16}), f32, supported) is bf16
    # No traced store, or one the epilogue cannot store.
    assert pick(None, bf16, supported) is None
    assert pick(frozenset({bf16, torch.int32}), bf16, supported) is None


# A one-CTA 128x64 tile, and the two-CTA (``tcgen05_cluster_m=2``) 256x64 one.
_MIXED_TILES = {
    "one_cta_128x64": ((1024, 512, 256), {"block_sizes": [128, 64, 64]}),
    "two_cta_256x64": (
        (2048, 1024, 512),
        {
            "block_sizes": [256, 64, 64],
            "tcgen05_cluster_m": 2,
            "pid_type": "persistent_interleaved",
        },
    ),
}


@pytest.mark.parametrize("tile", sorted(_MIXED_TILES))
def test_mixed_dtype_stores_share_the_fp32_subtile(tile: str) -> None:
    """The BF16 store walks the FP32 store's subtiles in its own element
    type, and its host TMA box is built from the same tile."""
    (m, n, k), overrides = _MIXED_TILES[tile]
    bound = mm_mixed_stores._bind_isolated(
        mm_mixed_stores_args(m, n, k, device=CPU_DEVICE, random=False)
    )
    default = {
        key: value
        for key, value in bound.config_spec.default_config().config.items()
        if key != "tcgen05_persistence_model"
    }
    code = bound.to_code(helion.Config(**{**default, **overrides}))
    tiles = _F32_TILE.findall(code)
    assert len(tiles) == 2, tiles
    assert re.search(
        rf"make_smem_layout_epi\(cutlass\.BFloat16, [\w.]+, {tiles[1]},", code
    )
    d_plans = [plan for plan in wrapper_plans(code) if plan["kind"] == "tcgen05_d_tma"]
    assert len(d_plans) == 2
    # The FP32 store's box is sized by its own dtype; the BF16 one names the
    # FP32 tile its device loop stages.
    assert "epi_tile_raw_expr" not in d_plans[0]
    assert "cutlass.Float32" in str(d_plans[1]["epi_tile_raw_expr"])
