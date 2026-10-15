from __future__ import annotations

import ast

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_cute_grouped_fp32_output import _grouped_fp32_output
from .test_cute_grouped_grid_multiplier import _config
import helion
from helion._compiler.cute.tcgen05_constants import (
    resolve_tcgen05_grouped_worklist_mma_shape,
)
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_store_shape
from helion.runtime.cute.launcher import _tcgen05_grouped_device_source_m_tile_supported


@pytest.mark.parametrize("source_rows", [8, 16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_narrow_grouped_source_keeps_its_extent_through_store(
    source_rows: int, offset_dtype: torch.dtype
) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=offset_dtype),
    )
    config = dict(_config(2).config)
    config["block_sizes"] = [256, 128, 128]
    config["tcgen05_ab_stages"] = 3
    config["tcgen05_grouped_worklist_source_m_tile"] = source_rows
    config["tcgen05_warp_register_reallocation"] = False
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        normalized = bound.config_spec.normalized_config(
            helion.Config.from_dict(config)
        )
        assert normalized["tcgen05_cluster_m"] == 1
        code = bound.to_code(normalized)
    ast.parse(code)
    assert f"CtaGroup.ONE, (128, {source_rows})" in code
    assert f"Repetition.x{source_rows // 8}" in code
    assert f"'cta_tile_shape_mnk': (128, {source_rows}, 128)" in code
    assert f"cute.make_layout({source_rows})" in code
    assert "setmaxregister_" not in code
    assert code.count(".allocate(") == 1
    assert code.count(".relinquish_alloc_permit()") == 1
    assert code.index(".relinquish_alloc_permit()") < code.index(
        "StaticPersistentGroupTileScheduler.create"
    )


def test_narrow_source_smem_accounts_for_ab_and_output_width() -> None:
    def footprint(source_rows: int) -> int:
        return tcgen05_grouped_worklist_smem_bytes(
            group_count=128,
            device_split_sizes=True,
            sched_stage_count=1,
            bm=128,
            bn=source_rows,
            bk=128,
            dtype_bytes=2,
            output_dtype_bytes=4,
            ab_stages=3,
            acc_stages=2,
            c_stages=2,
            cluster_m=1,
        )

    assert footprint(8) < footprint(16) < footprint(32)
    assert footprint(16) - footprint(8) == 8 * (128 * 2 * 3 + 128 * 4 * 2)
    assert tcgen05_grouped_worklist_store_shape(128, 8) == (128, 8, 8)
    assert tcgen05_grouped_worklist_store_shape(128, 16) == (128, 16, 16)
    assert tcgen05_grouped_worklist_store_shape(128, 32) == (128, 32, 32)
    assert tcgen05_grouped_worklist_store_shape(256, 224) == (128, 32, 32)


@pytest.mark.parametrize("source_rows", [8, 16])
def test_narrow_source_does_not_expand_other_cta_profiles(source_rows: int) -> None:
    assert (
        resolve_tcgen05_grouped_worklist_mma_shape(
            cluster_m=2, block_k=128, source_m_tile=source_rows
        )
        is None
    )
    assert (
        resolve_tcgen05_grouped_worklist_mma_shape(
            cluster_m=1, block_k=128, source_m_tile=source_rows, mma_m=64
        )
        is None
    )
    plan: dict[str, object] = {
        "kind": "tcgen05_grouped_static_persistent",
        "scheduler_mode": "device_group_search",
        "orientation": "nm",
        "source_m_tile": source_rows,
        "bm": 128,
        "bn": source_rows,
        "cluster_m": 1,
        "cluster_n": 1,
    }
    assert _tcgen05_grouped_device_source_m_tile_supported(plan)
    assert not _tcgen05_grouped_device_source_m_tile_supported({**plan, "cluster_m": 2})
    assert not _tcgen05_grouped_device_source_m_tile_supported({**plan, "bn": 32})
