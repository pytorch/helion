from __future__ import annotations

import ast
from contextlib import ExitStack
import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.attention import attention_output
import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
import helion
from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import SOFTCAP_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_fragment import EnumFragment


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


def _spec(
    head_dim: int = 128,
    num_kv: int = 64,
    *,
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    capability: tuple[int, int] | None = (10, 3),
    standard_output: bool = True,
    score_compatible: bool | None = None,
) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=capability,
        device=torch.device("cpu"),
        num_sm=152,
    )
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=64,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=causal,
        standard_dense_output=not causal and standard_output,
        standard_causal_output=causal and standard_output,
        tmem_rowmax_compatible=score_compatible,
    )
    return spec


def _source(
    *,
    head_dim: int = 128,
    num_kv: int = 64,
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    rowmax: str = "tmem",
    has_lse: bool = False,
    overrides: Mapping[str, object] | None = None,
) -> tuple[cute_flash.FlashAttentionConfig, str]:
    requested: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_ROWMAX_KEY: rowmax,
        cute_flash.FLASH_PERSISTENT_KEY: False,
        **(overrides or {}),
    }
    config = cute_flash.resolve_flash_config(
        head_dim,
        num_kv,
        requested,
        dtype=dtype,
        num_bh=1,
        is_causal=causal,
        standard_dense_output=not causal and not has_lse,
        standard_causal_output=causal and not has_lse,
        tmem_rowmax_compatible=True,
        target_device_capability=(10, 3),
    )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=1,
        total_tiles=num_kv // (4 if config.causal_two_cta else 2),
        cfg=config,
        has_lse=has_lse,
        io_dtype="cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16",
        score_plan=(causal_score_plan if causal else dense_score_plan)(head_dim),
        target_device_capability=(10, 3),
    )
    return config, ast.unparse(ast.Module(body=body, type_ignores=[]))


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("causal", (False, True))
def test_rowmax_choices_are_length_invariant_and_round_trip(
    head_dim: int, dtype: torch.dtype, causal: bool
) -> None:
    for num_kv in (4, 48, 256, 512):
        spec = _spec(head_dim, num_kv, dtype=dtype, causal=causal)
        fragment = cast(
            "EnumFragment", spec._flat_fields()[cute_flash.FLASH_ROWMAX_KEY]
        )
        assert fragment.choices == ("software", "tmem")
        assert fragment.default() == "software"
        generation = spec.create_config_generation()
        for value in fragment.choices:
            config = spec.default_config()
            config.config[cute_flash.FLASH_ROWMAX_KEY] = value
            spec.normalize(config)
            assert config.config[cute_flash.FLASH_ROWMAX_KEY] == value
            flat, normalized = generation.canonicalize_flat(generation.flatten(config))
            assert normalized == config
            assert generation.unflatten(flat) == config


@pytest.mark.parametrize("capability", (None, (9, 0), (10, 0), (12, 0)))
def test_rowmax_unsupported_target_is_canonical_and_not_pinnable(
    capability: tuple[int, int] | None,
) -> None:
    spec = _spec(capability=capability)
    config = helion.Config(block_sizes=[1, 128, 128], cute_flash_rowmax="tmem")
    spec.normalize(config)
    assert config.config[cute_flash.FLASH_ROWMAX_KEY] == "software"
    with pytest.raises(InvalidConfig, match="not (legal|effective)"):
        generation = spec.create_config_generation(
            overrides={cute_flash.FLASH_ROWMAX_KEY: "tmem"}
        )
        generation.unflatten(generation.default_flat())


@pytest.mark.parametrize(
    "overrides",
    (
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
        {cute_flash.FLASH_S_LOAD_REP_KEY: 16},
    ),
)
def test_rowmax_inactive_layout_is_canonical(overrides: dict[str, object]) -> None:
    spec = _spec()
    config = helion.Config.from_dict(
        {"block_sizes": [1, 128, 128], cute_flash.FLASH_ROWMAX_KEY: "tmem", **overrides}
    )
    spec.normalize(config)
    assert config.config[cute_flash.FLASH_ROWMAX_KEY] == "software"


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("disc", (False, True))
@pytest.mark.parametrize("setup", ("shared", "stage_local"))
def test_dense_rowmax_changes_consumed_reduction(
    head_dim: int, dtype: torch.dtype, disc: bool, setup: str
) -> None:
    overrides = {
        cute_flash.FLASH_SOFTMAX_DISC_KEY: disc,
        cute_flash.FLASH_SOFTMAX_SETUP_KEY: setup,
    }
    config, hardware = _source(
        head_dim=head_dim, dtype=dtype, num_kv=4, overrides=overrides
    )
    _, software = _source(
        head_dim=head_dim, dtype=dtype, num_kv=4, rowmax="software", overrides=overrides
    )
    assert config.rowmax == "tmem"
    assert "LdRed32x32bOp" in hardware
    assert "LdRed32x32bOp" not in software
    assert (
        "disc_rowmax_ldred" if disc else "flash_row_max, flash_hw_row_max"
    ) in hardware


@pytest.mark.parametrize("family", ("fa4", "fa4_2cta_causal"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_causal_hardware_maximum_only_updates_unmasked_rows(
    family: str, dtype: torch.dtype
) -> None:
    config, source = _source(
        dtype=dtype,
        causal=True,
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
        },
    )
    assert config.rowmax == "tmem"
    # The unacknowledged whole-row causal protocol remains canonically unavailable.
    assert config.softmax_disc
    assert "fa4_disc_rowmax_causal_balanced" in source
    assert "disc_rowmax_ldred" in source


def test_partial_resident_causal_scores_retain_hardware_maxima() -> None:
    packet = cute_flash._FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET
    config, source = _source(
        dtype=torch.bfloat16,
        causal=True,
        overrides={cute_flash.FLASH_EXP2_PACKET_KEY: packet},
    )
    assert config.exp2_packet == packet
    assert "flash_res_frg0, flash_res_red" in source
    assert "flash_res_frg1, flash_res_red" in source
    assert "flash_res_frg3, flash_res_red" in source
    assert "flash_res_rowmax_tmp, flash_res_red" in source
    assert "flash_row_max, flash_res_red[flash_red_i]" in source
    masked = source[
        source.index("for flash_kv_mask_iter") : source.index(
            "for flash_kv_unmask_iter"
        )
    ]
    assert "fa4_disc_rowmax_causal_balanced" in masked
    assert "flash_res_red" not in masked


def test_dense_kv_tail_reduces_after_masking() -> None:
    config, source = _source(
        head_dim=64,
        num_kv=4,
        overrides={
            cute_flash.FLASH_SOFTMAX_DISC_KEY: False,
            cute_flash.FLASH_KV_TILE_N_KEY: 160,
            cute_flash.FLASH_KV_ORDER_KEY: "descending",
        },
    )
    assert config.kv_tile_n == 160
    tail_start = source.index("for flash_kv_tail_iter")
    tail = source[tail_start : source.index("for flash_kv_iter", tail_start)]
    assert tail.index("mask_r2p_sm100_rank1") < tail.index("fmax_reduce_packed")
    assert "flash_row_max, flash_hw_row_max" not in tail


@pytest.mark.parametrize("causal", (False, True))
def test_rowmax_is_covered_and_present_in_terminal_coordinates(causal: bool) -> None:
    spec = _spec(dtype=torch.bfloat16, causal=causal)
    generation = spec.create_config_generation()
    configs = generation.flash_deterministic_population_configs()
    assert {config.config[cute_flash.FLASH_ROWMAX_KEY] for config in configs} == {
        "software",
        "tmem",
    }
    assert generation.flash_structural_coverage_uncovered_values() == []
    assert generation.flash_structural_coverage_underqualified_leaves() == []
    assert generation.flash_structural_coverage_uncovered_interactions() == []
    leaves = cast(
        "list[dict[str, object]]",
        generation.flash_terminal_coordinate_surface_catalog()["leaves"],
    )
    for leaf in leaves:
        assert any(
            coordinate["key"] == cute_flash.FLASH_ROWMAX_KEY
            for coordinate in cast("list[dict[str, object]]", leaf["coordinates"])
        )
    seeds = spec.autotune_seed_configs()
    assert any(seed.config[cute_flash.FLASH_ROWMAX_KEY] == "tmem" for seed in seeds)


@pytest.mark.parametrize("causal", (False, True))
def test_rowmax_capability_does_not_depend_on_lse_output(causal: bool) -> None:
    config, source = _source(causal=causal, has_lse=True)
    assert config.rowmax == "tmem"
    assert "disc_rowmax_ldred" in source


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("standard_output", (False, True))
def test_compiler_seed_routes_preserve_score_proof(
    causal: bool, standard_output: bool
) -> None:
    spec = _spec(causal=causal, standard_output=standard_output, score_compatible=True)
    heuristic = CuteFlashAttentionHeuristic.get_seed_configs(
        cast("CompileEnvironment", SimpleNamespace(config_spec=spec)),
        cast("DeviceIR", None),
    )
    canonical = spec.autotune_seed_configs()
    assert heuristic == canonical
    assert any(seed.config[cute_flash.FLASH_ROWMAX_KEY] == "tmem" for seed in canonical)


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("score_compatible", (False, True))
def test_target_seed_respects_explicit_score_proof(
    causal: bool, score_compatible: bool
) -> None:
    spec = _spec(64, 512, causal=causal, score_compatible=score_compatible)
    seeds = spec.autotune_seed_configs()
    assert seeds[0].config[cute_flash.FLASH_ROWMAX_KEY] == (
        "tmem" if score_compatible else "software"
    )
    if not score_compatible:
        assert all(
            seed.config[cute_flash.FLASH_ROWMAX_KEY] == "software" for seed in seeds
        )


def test_score_transform_proof_rejects_pre_transform_maximum() -> None:
    dense = dense_score_plan(128)
    causal = causal_score_plan(128)
    modified = dataclasses.replace(
        dense,
        modifiers=(AttentionScoreModifier(SOFTCAP_KIND, value_log2=1.0),),
    )
    assert cute_flash.flash_tmem_rowmax_score_plan_supported(dense)
    assert cute_flash.flash_tmem_rowmax_score_plan_supported(causal)
    assert not cute_flash.flash_tmem_rowmax_score_plan_supported(modified)
    config = cute_flash.resolve_flash_config(
        128,
        4,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_ROWMAX_KEY: "tmem",
        },
        target_device_capability=(10, 3),
        standard_dense_output=True,
        tmem_rowmax_compatible=False,
    )
    assert config.rowmax == "software"


def test_native_causal_lowering_remains_available_and_software_is_independent() -> None:
    seed = cute_flash.flash_attention_seed_config(
        64,
        512,
        dtype=torch.float16,
        is_causal=True,
        standard_causal_output=True,
        target_device_capability=(10, 3),
    )
    assert seed is not None
    for mode in ("software", "tmem"):
        _, source = _source(
            head_dim=64,
            num_kv=512,
            causal=True,
            overrides={**seed.config, cute_flash.FLASH_ROWMAX_KEY: mode},
        )
        if mode == "software":
            assert "LdRed32x32bOp" not in source
            assert "ResidentSoftmaxState.create" not in source
        else:
            assert "ResidentSoftmaxState.create" in source
            masked_start = source.index("for flash_kv_mask_iter")
            unmasked_start = source.index("for flash_kv_unmask_iter")
            masked = source[masked_start:unmasked_start]
            unmasked = source[unmasked_start:]
            assert "update_row_max_masked" in masked
            assert "flash_tiled_ldred0" not in masked
            assert "update_row_max_precomputed" in unmasked
            assert "flash_tiled_ldred0" in unmasked


@pytest.mark.parametrize(("sequence", "head_dim"), ((512, 128), (127, 128), (256, 256)))
@skipUnlessCuteAvailable("requires the supported CuTe runtime")
def test_frontend_bind_carries_rowmax_proof_without_widening_fused_scope(
    sequence: int, head_dim: int
) -> None:
    support = CuteMmaSupport(
        universal=True,
        warp_f16bf16=True,
        warpgroup_f16bf16=True,
        tcgen05_f16bf16=True,
        tcgen05_f8=True,
        tcgen05_tf32=True,
    )
    with ExitStack() as stack:
        for target, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 3)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 3)),
            ("helion.runtime.get_num_sm", 152),
            ("helion._compiler.cute.mma_support.get_cute_mma_support", support),
        ):
            stack.enter_context(patch(target, return_value=value))
        kernel = helion.kernel(
            attention_output.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        inputs = tuple(
            torch.empty(
                1,
                2,
                sequence,
                head_dim,
                dtype=torch.float16,
                device="meta",  # @ignore-device-lint
            )
            for _ in range(3)
        )
        bound = kernel._bind_isolated(inputs)
        spec = bound.config_spec
        if sequence % 128 or head_dim == 256:
            assert not spec.cute_flash_search_enabled
            assert cute_flash.FLASH_ROWMAX_KEY not in spec._flat_fields()
            return
        assert spec.cute_flash_search_enabled
        assert spec._cute_flash_tmem_rowmax_compatible
        config = spec.default_config()
        config.config[cute_flash.FLASH_ROWMAX_KEY] = "tmem"
        spec.normalize(config)
        source = bound.to_code(config)
        assert "LdRed32x32bOp" in source
        assert "disc_rowmax_ldred" in source
