from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel

import helion
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
import helion.language as hl

SOURCE = "tcgen05_grouped_worklist_source_m_tile"
IMMUTABLE = "tcgen05_grouped_full_allocation_a"
PHYSICAL_K = "tcgen05_grouped_ab_tile_k"
SHIFTED = "tcgen05_grouped_d_shift"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grouped_offsets_k64(
    a: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    rows, reduction = a.shape
    groups, columns, _ = weights.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty((rows, columns), device=a.device, dtype=a.dtype)
    for group_tile, row_tile, column_tile in hl.tile(
        [groups, rows, columns], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        extent = offsets[group + 1] - offsets[group]
        row_indices = start + row_tile.index
        valid = row_tile.index < extent
        acc = hl.zeros([row_tile, column_tile], dtype=torch.float32)
        for inner in hl.tile(reduction, block_size=block_k):
            left = hl.load(a, [row_indices, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, weights[group, column_tile, inner].T)
        hl.store(out, [row_indices, column_tile], acc, extra_mask=valid[:, None])
    return out


def _kernel():
    # A fresh bind is needed when comparing seed-generation policies.
    return helion.kernel(
        _grouped_offsets_k64.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


def _matching(seeds, source=256):
    return [
        seed
        for seed in seeds
        if seed.get(SOURCE) == source
        and seed.block_sizes == [256, 128, 64]
        and seed.get(PHYSICAL_K) is None
        and seed.get("tcgen05_cluster_m") == 2
        and seed.get("tcgen05_c_stages") == 2
        and seed.get("tcgen05_consumer_regs") == 256
        and seed.get(IMMUTABLE) is True
        and seed.get(SHIFTED) is not True
    ]


def test_actual_k1_metadata_adds_deep_immutable_seed_without_changing_the_prefix():
    args = (
        torch.empty((8192, 2048), dtype=torch.bfloat16),
        torch.empty((16, 1536, 2048), dtype=torch.bfloat16),
        torch.arange(17, dtype=torch.int32) * 512,
    )
    original = heuristics._tcgen05_grouped_device_tile_seeds

    def legacy(*arguments, **keywords):
        if (
            keywords["source_m_tile"] in (224, 256)
            and keywords.get("physical_k") is None
        ):
            return []
        return original(*arguments, **keywords)

    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        with patch.object(
            heuristics, "_tcgen05_grouped_device_tile_seeds", side_effect=legacy
        ):
            before = _cpu_bind(_kernel(), args)
            old_seeds = [
                seed
                for seed in before.config_spec.compiler_seed_configs
                if seed.get(SHIFTED) is not True
            ]
            old_default = before.config_spec.compiler_default_config
        bound = _cpu_bind(_kernel(), args)
        spec = bound.config_spec
        seeds = spec.compiler_seed_configs
        unshifted = [seed for seed in seeds if seed.get(SHIFTED) is not True]
        shifted = [seed for seed in seeds if seed.get(SHIFTED) is True]
        assert unshifted[: len(old_seeds)] == old_seeds
        assert seeds == [*unshifted, *shifted]
        assert spec.compiler_default_config == old_default
        assert spec.block_sizes[2].max_size == 64
        expected = _matching(seeds)
        assert len(expected) == 1 and expected[0]["tcgen05_ab_stages"] == 6
        normalized = spec.normalized_config(expected[0])
        assert normalized.block_sizes == [256, 128, 64]
        assert normalized.get(PHYSICAL_K) is None
        code = bound.to_code(normalized)
        assert "'ab_stage_count': 6" in code
        assert "'full_allocation_a': True" in code
        assert "CtaGroup.TWO, (256, 256)" in code
        assert _matching(seeds, source=224)
        generation = spec.create_config_generation()
        transferred = [config for _, config in generation.seed_flat_config_pairs()]
        transferred_match = _matching(transferred)
        assert len(transferred_match) == 1
        assert transferred_match[0]["tcgen05_ab_stages"] == 6
        assert bound.to_code(transferred_match[0]) == code


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_split_sizes_receive_the_same_large_source_device_pipeline(offset_dtype):
    args = (
        torch.empty((2048, 512), dtype=torch.bfloat16),
        torch.empty((8, 512, 512), dtype=torch.bfloat16),
        torch.full((8,), 256, dtype=offset_dtype),
    )
    kernel = helion.kernel(
        _device_split_sizes_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        assert bound.config_spec.block_sizes[2].max_size == 64
        seeds = _matching(bound.config_spec.compiler_seed_configs)
        assert seeds
        for seed in seeds:
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(PHYSICAL_K) is None
            assert "'device_layout_kind': 'split_sizes'" in bound.to_code(normalized)


@pytest.mark.parametrize(
    ("groups", "columns", "reduction", "major", "immutable"),
    [
        (16, 1536, 2048, "k", True),
        (16, 160, 2048, "k", False),
        (2048, 256, 4096, "k", False),
        (16, 1536, 2048, "n", False),
    ],
)
def test_large_source_helper_keeps_feature_signed_domain_and_resource_guards(
    groups, columns, reduction, major, immutable
):
    args = (
        torch.empty((512, 256), dtype=torch.bfloat16),
        torch.empty((2, 256, 256), dtype=torch.bfloat16),
        torch.tensor([0, 256, 512], dtype=torch.int32),
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), args)
        seeds = heuristics._tcgen05_grouped_device_tile_seeds(
            bound.config_spec,
            groups=groups,
            n=columns,
            k=reduction,
            b_major=major,
            dtype_bytes=2,
            output_dtype_bytes=2,
            capacity_bytes=232448,
            source_m_tile=256,
        )
        assert seeds
        assert any(seed.get(IMMUTABLE) is True for seed in seeds) == immutable
        for seed in seeds:
            assert seed.block_sizes == [256, 128, 64]
            assert seed.get(PHYSICAL_K) is None
            assert (
                tcgen05_grouped_worklist_smem_bytes(
                    group_count=groups,
                    device_split_sizes=True,
                    sched_stage_count=1,
                    bm=256,
                    bn=256,
                    bk=64,
                    dtype_bytes=2,
                    output_dtype_bytes=2,
                    ab_stages=seed["tcgen05_ab_stages"],
                    acc_stages=2,
                    c_stages=2,
                    cluster_m=2,
                )
                <= 232448
            )
