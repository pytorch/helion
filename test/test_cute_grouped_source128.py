from __future__ import annotations

import ast
from copy import deepcopy
import random
from unittest.mock import patch

from packaging.version import Version
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population
from test.test_autotuner_heuristics import _grouped_worklist_kernel_body
from test.test_cute_grouped_fp32_output import _grouped_fp32_output
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel

import helion
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedWorklistHeuristic,
)
from helion._compiler.cute.cutedsl_compat import _CuteDSLVersionProbe
from helion._compiler.cute.cutedsl_compat import tcgen05_runtime_n_ptx_compatible
from helion._compiler.cute.tcgen05_constants import (
    resolve_tcgen05_grouped_worklist_mma_shape,
)
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.cute.launcher import _tcgen05_grouped_device_source_m_tile_supported
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes

SOURCE = "tcgen05_grouped_worklist_source_m_tile"
MODE = "tcgen05_grouped_mode"
SHIFTED = "tcgen05_grouped_d_shift"


def _config(cluster_m: int, block_k: int) -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, block_k],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=cluster_m,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=128,
    )


def _plans(source: str) -> list[dict[str, object]]:
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_helion_cute_wrapper_plans"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("missing CuTe wrapper plans")


@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("block_k", [64, 128])
def test_source128_resolves_both_cta_groups(cluster_m: int, block_k: int) -> None:
    assert resolve_tcgen05_grouped_worklist_mma_shape(
        cluster_m=cluster_m, block_k=block_k, source_m_tile=128
    ) == (128 * cluster_m, 128)
    assert (
        resolve_tcgen05_grouped_worklist_mma_shape(
            cluster_m=cluster_m,
            block_k=block_k,
            source_m_tile=128,
            mma_m=64,
        )
        is None
    )


@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("block_k", [64, 128])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_source128_codegen_retains_live_offsets_and_feature_tail(
    cluster_m: int, block_k: int, offset_dtype: torch.dtype
) -> None:
    args = (
        torch.empty((640, 256), dtype=torch.bfloat16),
        torch.empty((8, 160, 256), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 257, 512, 639, 640, 640], dtype=offset_dtype),
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        config = bound.config_spec.normalized_config(_config(cluster_m, block_k))
        source = bound.to_code(config)
    plan = next(
        plan
        for plan in _plans(source)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert plan["source_m_tile"] == 128
    assert (plan["bm"], plan["bn"], plan["bk"]) == (
        128 * cluster_m,
        128,
        block_k,
    )
    assert plan["scheduler_mode"] == "device_group_search"
    assert plan["device_split_sizes"] is True
    assert plan["device_layout_kind"] == "offsets"
    assert plan["dynamic_d_tensormap"] is True
    assert _tcgen05_grouped_device_source_m_tile_supported(plan)
    _validate_tcgen05_grouped_device_split_sizes(plan, args[2])
    cta_group = "ONE" if cluster_m == 1 else "TWO"
    assert f"CtaGroup.{cta_group}, ({128 * cluster_m}, 128)" in source
    assert source.count(".allocate(") == 1
    assert source.count(".relinquish_alloc_permit()") == 1
    assert source.count(".free(") == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"device_split_sizes": False},
        {"device_layout_kind": "external_worklist"},
        {"cluster_n": 2},
        {"cluster_m": 3},
        {"bm": 64},
        {"bn": 64},
        {"scheduler_mode": "runtime_direct"},
    ],
)
def test_source128_launcher_rejects_other_metadata_and_geometry(
    updates: dict[str, object],
) -> None:
    plan = {
        "kind": "tcgen05_grouped_static_persistent",
        "scheduler_mode": "device_group_search",
        "orientation": "nm",
        "source_m_tile": 128,
        "bm": 128,
        "bn": 128,
        "cluster_m": 1,
        "cluster_n": 1,
        "device_split_sizes": True,
        "device_layout_kind": "offsets",
    }
    assert _tcgen05_grouped_device_source_m_tile_supported(plan)
    assert not _tcgen05_grouped_device_source_m_tile_supported(plan | updates)


@pytest.mark.parametrize("metadata_kind", ["offsets", "split_sizes"])
def test_typed_device_seeds_and_promotion_survive_newer_dsl(
    metadata_kind: str,
) -> None:
    metadata = torch.full((8,), 256, dtype=torch.int64)
    kernel = _device_split_sizes_kernel
    if metadata_kind == "offsets":
        metadata = torch.arange(9, dtype=torch.int64) * 256
        kernel = _device_offsets_kernel
    args = (
        torch.empty((2048, 256), dtype=torch.bfloat16),
        torch.empty((8, 256, 256), dtype=torch.bfloat16),
        metadata,
    )
    with (
        _forbid_cuda_work(),
        _target(),
        _mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.cutedsl_compat._installed_cute_dsl_version",
            return_value=_CuteDSLVersionProbe(version=Version("4.8.0")),
        ),
    ):
        assert not tcgen05_runtime_n_ptx_compatible()
        bound = _cpu_bind(kernel, args)
        seeds = bound.config_spec.compiler_seed_configs
        assert seeds
        assert CuteTcgen05GroupedWorklistHeuristic.should_promote(bound.env)
        assert bound.config_spec.compiler_default_config[MODE] == "worklist_nm"
        assert {seed[SOURCE] for seed in seeds} >= {32, 128, 224, 256}
        for seed in seeds:
            assert seed.get("tcgen05_grouped_runtime_direct", False) is False
            assert seed.get("tcgen05_persistence_model") != "clc_persistent"
        source128 = [seed for seed in seeds if seed.get(SOURCE) == 128]
        assert {seed["tcgen05_cluster_m"] for seed in source128} == {1, 2}
        assert bound.config_spec._cute_tcgen05_config.grouped_source128_search_enabled
        fields = bound.config_spec._flat_fields()
        assert "worklist_nm" in fields[MODE].search_choices
        assert 128 in fields[SOURCE].search_choices
        for seed in source128:
            ast.parse(bound.to_code(seed))


def test_source128_seeds_reach_a_cold_full_initial_population() -> None:
    args = (
        torch.empty((2048, 512), dtype=torch.bfloat16),
        torch.empty((8, 256, 512), dtype=torch.bfloat16),
        torch.arange(9, dtype=torch.int32) * 256,
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        state = random.getstate()
        try:
            random.seed(281)
            search = LFBOTreeSearch(
                bound,
                args,
                **LFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile("full"), bound.settings
                ),
            )
            population = checked_initial_population(search)
            configs = [search.config_gen.unflatten(row) for row in population]
            source128 = [
                config
                for _, config in search.config_gen.seed_flat_config_pairs()
                if config.get(SOURCE) == 128
            ]
            assert source128 and all(config in configs for config in source128)
        finally:
            random.setstate(state)


def test_external_worklist_retains_the_runtime_n_dsl_gate() -> None:
    kernel = helion.kernel(
        _grouped_worklist_kernel_body,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    args = (
        torch.empty((512, 128), dtype=torch.bfloat16),
        torch.empty((2, 256, 128), dtype=torch.bfloat16),
        torch.tensor([[0, 0, 128, 256], [1, 256, 128, 256]], dtype=torch.int32),
    )
    with (
        _forbid_cuda_work(),
        _target(),
        _mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.cutedsl_compat._installed_cute_dsl_version",
            return_value=_CuteDSLVersionProbe(version=Version("4.8.0")),
        ),
    ):
        bound = _cpu_bind(kernel, args)
        assert bound.host_function is not None
        assert not tcgen05_runtime_n_ptx_compatible()
        assert (
            CuteTcgen05GroupedWorklistHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            == []
        )
        assert not CuteTcgen05GroupedWorklistHeuristic.should_promote(bound.env)
        assert (
            not bound.config_spec._cute_tcgen05_config.grouped_source128_search_enabled
        )


def test_source128_seed_depths_use_physical_smem_and_preserve_old_prefix() -> None:
    args = (
        torch.empty((2048, 512), dtype=torch.bfloat16),
        torch.empty((8, 256, 512), dtype=torch.bfloat16),
        torch.arange(9, dtype=torch.int32) * 256,
    )
    device_tiles = heuristics._tcgen05_grouped_device_tile_seeds

    def source128_era_tiles(*arguments, **keywords):
        # Later larger-source additions are independently covered; this test
        # compares the source128 addition against its original seed prefix.
        if (
            keywords["source_m_tile"] in (224, 256)
            and keywords.get("physical_k") is None
        ):
            return []
        return device_tiles(*arguments, **keywords)

    with (
        _forbid_cuda_work(),
        _target(),
        _mock_cuda_unavailable(),
        patch(
            "helion._compiler.autotuner_heuristics.cute._tcgen05_grouped_physical_k_seeds",
            return_value=[],
        ),
        patch.object(
            heuristics,
            "_tcgen05_grouped_device_tile_seeds",
            side_effect=source128_era_tiles,
        ),
    ):
        with patch(
            "helion._compiler.autotuner_heuristics.cute._tcgen05_grouped_source128_seeds",
            return_value=[],
        ):
            previous = _cpu_bind(_grouped_fp32_output, args)
            old_seeds = deepcopy(
                [
                    seed
                    for seed in previous.config_spec.compiler_seed_configs
                    if seed.get(SHIFTED) is not True
                ]
            )
            old_default = deepcopy(previous.config_spec.compiler_default_config)
        bound = _cpu_bind(_grouped_fp32_output, args)
        all_seeds = bound.config_spec.compiler_seed_configs
        seeds = [seed for seed in all_seeds if seed.get(SHIFTED) is not True]
        shifted = [seed for seed in all_seeds if seed.get(SHIFTED) is True]
        assert all_seeds == [*seeds, *shifted]
        assert seeds[: len(old_seeds)] == old_seeds
        assert bound.config_spec.compiler_default_config == old_default
        for seed in seeds[len(old_seeds) :]:
            assert seed[SOURCE] == 128
            cluster_m = seed["tcgen05_cluster_m"]
            required = tcgen05_grouped_worklist_smem_bytes(
                group_count=8,
                device_split_sizes=True,
                sched_stage_count=1,
                bm=128 * cluster_m,
                bn=128,
                bk=seed["block_sizes"][2],
                dtype_bytes=2,
                output_dtype_bytes=4,
                ab_stages=seed["tcgen05_ab_stages"],
                acc_stages=2,
                c_stages=2,
                cluster_m=cluster_m,
            )
            assert required < 232448
            assert seed.get("num_sm_multiplier", 1) * required < 232448


@pytest.mark.cuda
@pytest.mark.parametrize("metadata_kind", ["offsets", "split_sizes"])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("block_k", [64, 128])
@skipUnlessBackends(["cute"])
def test_source128_cuda_live_routing_rows_and_feature_tails(
    metadata_kind: str,
    offset_dtype: torch.dtype,
    cluster_m: int,
    block_k: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family grouped MMA")
    generator = torch.Generator(device=DEVICE).manual_seed(271)
    a = torch.randn((640, 256), generator=generator, device=DEVICE).bfloat16()
    b = torch.randn((8, 160, 256), generator=generator, device=DEVICE).bfloat16()
    lengths = [0, 1, 128, 128, 254, 128, 1, 0]
    metadata = torch.tensor(lengths, device=DEVICE, dtype=offset_dtype)
    kernel = _device_split_sizes_kernel
    if metadata_kind == "offsets":
        metadata = torch.cat([metadata.new_zeros(1), metadata.cumsum(0)]).to(
            offset_dtype
        )
        kernel = _device_offsets_kernel
    bound = kernel._bind_isolated((a, b, metadata))
    bound.set_config(_config(cluster_m, block_k))

    def reference() -> torch.Tensor:
        expected = torch.empty((640, 160), device=DEVICE, dtype=torch.bfloat16)
        start = 0
        for group, length in enumerate(lengths):
            end = start + length
            expected[start:end] = (a[start:end].float() @ b[group].float().T).bfloat16()
            start = end
        return expected

    torch.testing.assert_close(bound(a, b, metadata), reference(), rtol=1e-2, atol=1e-2)
    # Keep tensor identity, storage and total rows fixed while a formerly empty
    # group acquires a row and its neighbor becomes empty.
    lengths[0], lengths[1] = 1, 0
    metadata[0 if metadata_kind == "split_sizes" else 1] = 1
    if metadata_kind == "split_sizes":
        metadata[1] = 0
    torch.testing.assert_close(bound(a, b, metadata), reference(), rtol=1e-2, atol=1e-2)
