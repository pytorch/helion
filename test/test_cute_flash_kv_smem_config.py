from __future__ import annotations

import ast
import dataclasses
import importlib
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _mock_cuda_unavailable
import helion
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.flash_schedule import FlashScheduleError
from helion._compiler.cute.flash_schedule import FlashScheduleSpec
from helion._compiler.cute.flash_schedule import build_fa4_schedule
from helion._compiler.cute.flash_schedule import max_fa4_kv_depth
from helion._compiler.cute.flash_schedule import verify_flash_schedule
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import EnumFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
flash_fa4_shared_storage = importlib.import_module(
    "helion._compiler.cute._flash_runtime"
).flash_fa4_shared_storage


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True),
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only test")
        ),
        patch(
            "helion.autotuner.config_spec.get_target_device_capability",
            return_value=(10, 3),
        ),
    ):
        yield


def _config_spec(head_dim: int, dtype: torch.dtype, num_kv: int = 48) -> ConfigSpec:
    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=64,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        standard_dense_output=True,
    )
    return spec


@pytest.mark.parametrize(
    ("head_dim", "stage_output", "expected_cap"),
    ((64, False, 24), (64, True, 20), (128, False, 10), (128, True, 6)),
)
def test_cooperative_capacity_matches_runtime_and_rejects_overflow(
    head_dim: int, stage_output: bool, expected_cap: int
) -> None:
    spec = FlashScheduleSpec(
        head_dim,
        expected_cap,
        cta_count=2,
        cooperative_mma=True,
        multicast_kv=True,
        stage_output=stage_output,
    )
    assert max_fa4_kv_depth(spec) == expected_cap
    schedule = verify_flash_schedule(build_fa4_schedule(spec)).schedule
    storage = flash_fa4_shared_storage(
        head_dim, expected_cap, epi_tma=stage_output, kv_cta_group_size=2
    )
    assert schedule.shared_memory_bytes == storage.size_in_bytes() == 232448
    with pytest.raises(FlashScheduleError, match="shared-memory capacity"):
        verify_flash_schedule(
            build_fa4_schedule(dataclasses.replace(spec, kv_depth=expected_cap + 1))
        )


@pytest.mark.parametrize(
    ("cooperative", "separate_kv"), ((False, False), (True, False), (False, True))
)
def test_schedule_regions_use_configured_width_and_cooperative_extent(
    cooperative: bool, separate_kv: bool
) -> None:
    spec = FlashScheduleSpec(
        64,
        3,
        cta_count=2,
        cooperative_mma=cooperative,
        multicast_kv=True,
        separate_kv=separate_kv,
        kv_tile_n=160,
    )
    schedule = verify_flash_schedule(build_fa4_schedule(spec)).schedule
    storage = flash_fa4_shared_storage(
        64,
        3,
        epi_tma=True,
        separate_kv=separate_kv,
        kv_tile_n=160,
        kv_cta_group_size=2 if cooperative else 1,
    )
    assert schedule.shared_memory_bytes == storage.size_in_bytes()
    regions = {region.name: region for region in schedule.memory_regions}
    expected_ring_bytes = 30720 if cooperative else 61440
    for rank in (0, 1):
        assert regions[f"K_r{rank}"].extent == expected_ring_bytes
        assert regions[f"V_r{rank}"].extent == expected_ring_bytes
        assert regions[f"V_r{rank}"].offset - regions[f"K_r{rank}"].offset == (
            expected_ring_bytes if separate_kv else 0
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("output_requires_tma", (False, True))
def test_fragment_depth_domain_tracks_available_instruction_groups(
    dtype: torch.dtype, head_dim: int, output_requires_tma: bool
) -> None:
    local_cap = (
        (10 if output_requires_tma else 12)
        if head_dim == 64
        else (3 if output_requires_tma else 5)
    )
    for family, multiplier in (("fa4", 1), ("fa4_2cta", 2), (None, 2)):
        fragments = cute_flash.flash_autotune_fragments(
            head_dim,
            48,
            dtype=dtype,
            num_bh=64,
            standard_dense_output=True,
            output_requires_tma=output_requires_tma,
            pipeline_family_override=family,
        )
        fragment = cast("EnumFragment", fragments[cute_flash.FLASH_KV_STAGE_KEY])
        assert set(fragment.search_choices or ()) == set(
            range(2, multiplier * local_cap + 1)
        )
        assert fragment.default() == 3


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_new_depths_remain_distinct_after_full_config_normalization(
    dtype: torch.dtype,
) -> None:
    spec = _config_spec(128, dtype)
    configs = []
    for stage in (3, 4, 5, 6):
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_2cta",
            cute_flash_kv_stage=stage,
            cute_flash_softmax_disc=True,
            cute_flash_epi_tma=True,
        )
        spec.normalize(config)
        assert config.config[cute_flash.FLASH_KV_STAGE_KEY] == stage
        configs.append(config)
    assert len({repr(config) for config in configs}) == 4


def test_whole_row_ring2_keeps_independent_depth_two_limit() -> None:
    for family in ("fa4", "fa4_2cta"):
        config = cute_flash.resolve_flash_config(
            128,
            48,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                cute_flash.FLASH_KV_STAGE_KEY: 6,
                cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
                cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
            },
            standard_dense_output=True,
        )
        assert config.stat_transport == "ring2"
        assert config.kv_stage == 2


@pytest.mark.parametrize(
    ("head_dim", "family", "is_causal", "stage", "cta_group_size"),
    (
        (128, "fa4_2cta", False, 3, 2),
        (128, "fa4_2cta", False, 6, 2),
        (64, "fa4_2cta_causal", True, 13, 2),
        (64, "fa4_cga2_local", False, 10, 1),
        (128, "fa4", False, 3, 1),
    ),
)
def test_codegen_storage_group_preserves_cluster_transaction_bytes(
    head_dim: int,
    family: str,
    is_causal: bool,
    stage: int,
    cta_group_size: int,
) -> None:
    config = cute_flash.resolve_flash_config(
        head_dim,
        48,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_KV_STAGE_KEY: stage,
            cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
            cute_flash.FLASH_EPI_TMA_KEY: True,
        },
        num_bh=64,
        is_causal=is_causal,
        standard_dense_output=not is_causal,
        standard_causal_output=is_causal,
    )
    assert config.pipeline_family == family
    assert config.kv_stage == stage
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=48,
        sequence_extent=6144,
        num_bh=64,
        total_tiles=64 * 48 // (4 if config.use_2cta_instrs else 2),
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16",
        score_plan=causal_score_plan(head_dim)
        if is_causal
        else dense_score_plan(head_dim),
    )
    module = ast.Module(body=body, type_ignores=[])
    storage_call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "flash_fa4_shared_storage"
    )
    assert ast.literal_eval(storage_call.args[-1]) == cta_group_size
    source = ast.unparse(module)
    byte_scale = " * 2" if cta_group_size == 2 else ""
    for operand in ("q", "k"):
        assert (
            f"flash_{operand}_bytes = cute.size_in_bytes(cutlass.Float16, "
            f"cute.select(_flash_{operand}sl, mode=[0, 1, 2])){byte_scale}\n"
        ) in source
    assert f"num_stages={stage}" in source
    assert "tx_count=flash_k_bytes" in source
    assert "flash_kv_prod.tail()" in source


def test_structural_pipeline_catalog_retains_new_depth_witnesses() -> None:
    spec = _config_spec(128, torch.float16)
    generation = spec.create_config_generation()
    catalog = generation.flash_pipeline_lane_catalog()
    witnesses = generation.flash_pipeline_lane_witnesses()
    chunked = cute_flash.FlashStructuralLeaf("fa4_2cta", None, True)
    whole_row = cute_flash.FlashStructuralLeaf("fa4_2cta", None, False)
    for stage in (3, 4, 5, 6, 7, 8, 9, 10):
        key = (cute_flash.FLASH_KV_STAGE_KEY, stage)
        assert key in catalog[chunked]
        assert key in catalog[whole_row]
        for leaf in (chunked, whole_row):
            config = witnesses[(leaf, *key)]
            assert config.config[cute_flash.FLASH_KV_STAGE_KEY] == stage
            assert generation.canonicalize_flat(generation.flatten(config))[1] == config
        whole_config = witnesses[(whole_row, *key)]
        assert whole_config.config[cute_flash.FLASH_STAT_TRANSPORT_KEY] == "single"
        # The new acknowledged transport makes deeper whole-row witnesses
        # legal; the independent two-slot restriction must still hold.
        ring_config = helion.Config.from_dict(
            {**whole_config.config, cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2"}
        )
        spec.normalize(ring_config)
        assert ring_config.config[cute_flash.FLASH_STAT_TRANSPORT_KEY] == "ring2"
        assert ring_config.config[cute_flash.FLASH_KV_STAGE_KEY] == 2
    assert generation.flash_structural_coverage_uncovered_values() == []


if __name__ == "__main__":
    pytest.main([__file__])
