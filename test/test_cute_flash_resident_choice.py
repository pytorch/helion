from __future__ import annotations

import ast
import inspect
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from .test_cute_flash_exp2 import _emit_causal_resident_native_source
from .test_cute_flash_exp2 import _emit_dense_resident_value_graph_source
from .test_cute_flash_exp2 import _flash_runtime
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import dense_score_plan
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_function import DeviceFunction


def _resolve(
    head_dim: int = 128,
    num_kv: int = 96,
    *,
    overrides: dict[str, object] | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    standard_dense_output: bool = True,
) -> cute_flash.FlashAttentionConfig:
    values = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
        cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph",
        **({} if overrides is None else overrides),
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            values,
            dtype=dtype,
            is_causal=is_causal,
            standard_dense_output=standard_dense_output,
            target_device_capability=(10, 3),
        )


def _emit(
    head_dim: int = 128,
    num_kv: int = 96,
    *,
    dtype: torch.dtype = torch.float16,
    overrides: dict[str, object] | None = None,
) -> tuple[cute_flash.FlashAttentionConfig, str]:
    config = _resolve(head_dim, num_kv, overrides=overrides, dtype=dtype)
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=64,
        total_tiles=num_kv * (16 if config.use_2cta_instrs else 32),
        cfg=config,
        has_lse=False,
        io_dtype=("cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16"),
        score_plan=dense_score_plan(head_dim),
        target_device_capability=(10, 3),
    )
    return config, ast.unparse(ast.Module(body=body, type_ignores=[]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("num_kv", (4, 96, 100, 260))
def test_resident_choice_is_general(
    head_dim: int, num_kv: int, dtype: torch.dtype
) -> None:
    config, source = _emit(head_dim, num_kv, dtype=dtype)
    assert config.softmax_lowering == "resident_value_graph"
    assert "resident_softmax_value_graph" in source
    assert "fa4_sp_exp_convert_store_whole_rowsum" not in source
    assert "cutlass.Float32(7.0) - flash_row_max_safe" not in source
    # D128 uses all 512 TMEM columns at N=128; a requested wider score tile
    # must still obey the existing allocation bound.
    if head_dim == 128:
        wide = _resolve(
            head_dim,
            num_kv,
            dtype=dtype,
            overrides={cute_flash.FLASH_KV_TILE_N_KEY: 160},
        )
        assert wide.kv_tile_n == 128


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("head_dim", "family"),
    (
        (128, "fa4"),
        (128, "fa4_2cta"),
        (128, "fa4_deep_1cta"),
        (128, "fa4_local_tma"),
        (64, "fa4_tma_4d"),
        (64, "fa4_2cta_tma_4d"),
        (64, "fa4_cga2_local"),
        (64, "fa4_clc"),
        (64, "fa4_clc_local_tma_4d"),
    ),
)
def test_resident_choice_preserves_family_and_persistent_credit(
    head_dim: int, family: str, dtype: torch.dtype
) -> None:
    config, source = _emit(
        head_dim,
        dtype=dtype,
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
            cute_flash.FLASH_PERSISTENT_KEY: True,
        },
    )
    # The pre-existing BF16 family resolver drops unsupported 4-D TMA views.
    expected_family = family.removesuffix("_4d") if dtype is torch.bfloat16 else family
    if expected_family == "fa4_tma":
        expected_family = "fa4"
    elif expected_family == "fa4_2cta_tma":
        expected_family = "fa4_2cta"
    assert config.pipeline_family == expected_family
    assert config.persistent
    module = ast.parse(source)
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resident_softmax_value_graph"
    ]
    assert len(calls) == 2
    parameters = tuple(
        inspect.signature(_flash_runtime.resident_softmax_value_graph).parameters
    )
    for call in calls:
        arguments = dict(zip(parameters, call.args, strict=False))
        assert ast.unparse(arguments["stats_empty_phase"]) == "flash_s_corr_prod_phase"
        assert ast.unparse(arguments["row_sum_init"]) == "flash_row_sum * flash_alpha"
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        if dtype is torch.bfloat16:
            assert keywords.pop("io_dtype") == "cutlass.BFloat16"
        assert keywords == (
            {
                "pfor_peer_cta_rank": "cutlass.Int32(0)",
                "pfor_self_cta_rank": (
                    "flash_mma_tile_coord_v" if head_dim == 128 else "None"
                ),
            }
            if config.use_2cta_instrs
            else {}
        )
    # Persistent wrappers must carry the producer phase across work items.
    # Each role initializes it once outside the outer work loop, then acquires
    # entry credit and helper credit before the final row-sum/tail handshake.
    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(1)") == 2
    start = source.index("flash_s_corr_prod_phase = cutlass.Int32(1)")
    end = source.index("flash_s_corr_prod_phase = cutlass.Int32(1)", start + 1)
    softmax0 = source[start:end]
    assert softmax0.count("flash_s_corr_prod_phase ^= 1") == 2
    helper = softmax0.index("resident_softmax_value_graph")
    advance = softmax0.index("flash_s_corr_prod_phase ^= 1", helper)
    rowsum = softmax0.index("= flash_row_sum", advance)
    tail = softmax0.index("mbar_spin_wait(flash_s0_corr_empty_ptr", rowsum)
    assert helper < advance < rowsum < tail
    assert "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum" not in softmax0


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rowmax", ("software", "tmem"))
@pytest.mark.parametrize("persistent", (False, True))
@pytest.mark.parametrize("threshold", (0.0, 8.0))
def test_resident_alpha_precedes_value_graph(
    rowmax: str, persistent: bool, threshold: float, dtype: torch.dtype
) -> None:
    config, source = _emit(
        dtype=dtype,
        overrides={
            cute_flash.FLASH_ROWMAX_KEY: rowmax,
            cute_flash.FLASH_PERSISTENT_KEY: persistent,
            cute_flash.FLASH_RESCALE_THRESHOLD_KEY: threshold,
        },
    )
    assert config.rowmax == rowmax
    assert config.rescale_threshold == threshold
    start = source.index("flash_s_corr_prod_phase = cutlass.Int32(1)")
    loop = source.index("for flash_kv", start)
    alpha = source.index("flash_alpha =", loop)
    publish = source.index("= flash_alpha", alpha)
    helper = source.index("resident_softmax_value_graph", publish)
    assert loop < alpha < publish < helper


@pytest.mark.parametrize(
    ("dtype", "is_causal", "standard_dense_output"),
    (
        (torch.float32, False, True),
        (torch.float16, True, True),
        (torch.float16, False, False),
    ),
)
def test_resident_rejects_unsupported_workload(
    dtype: torch.dtype, is_causal: bool, standard_dense_output: bool
) -> None:
    with pytest.raises(
        InvalidConfig, match="FP16/BF16 dense output on an FA4 pipeline"
    ):
        _resolve(
            dtype=dtype,
            is_causal=is_causal,
            standard_dense_output=standard_dense_output,
        )


def test_resident_rejects_ws_overlap_and_invalid_choice() -> None:
    with pytest.raises(InvalidConfig, match="FA4 pipeline"):
        _resolve(overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"})
    with pytest.raises(InvalidConfig, match="FA4 pipeline"):
        _resolve(num_kv=95)
    with pytest.raises(ValueError, match="invalid flash softmax lowering"):
        _resolve(overrides={cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "unknown"})


@pytest.mark.parametrize(
    ("emit", "marker"),
    (
        (_emit_dense_resident_value_graph_source, "resident_softmax_value_graph"),
        (_emit_causal_resident_native_source, "ResidentSoftmaxState.create"),
    ),
)
def test_auto_preserves_existing_resident_lowerings(
    emit: Callable[..., str], marker: str
) -> None:
    automatic = emit()
    assert automatic == emit(
        config_overrides={cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "auto"}
    )
    assert marker in automatic
    standard = emit(
        config_overrides={cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"}
    )
    assert marker not in standard


def _spec(head_dim: int, num_kv: int, dtype: torch.dtype = torch.float16) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=152,
    )
    for block_id, size_hint in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=size_hint))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        standard_dense_output=True,
    )
    return spec


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(("head_dim", "num_kv"), ((64, 100), (128, 96), (128, 260)))
def test_resident_seed_survives_normalization_and_full_population(
    head_dim: int, num_kv: int, dtype: torch.dtype
) -> None:
    spec = _spec(head_dim, num_kv, dtype)
    seeds = cute_flash.flash_attention_seed_configs(
        head_dim,
        num_kv,
        dtype=dtype,
        standard_dense_output=True,
        target_device_capability=(10, 3),
    )
    spec.compiler_seed_configs = list(seeds)
    resident = [
        seed
        for seed in seeds
        if seed.config.get(cute_flash.FLASH_SOFTMAX_LOWERING_KEY)
        == "resident_value_graph"
    ]
    assert len(resident) == 3
    joint = resident[1:]
    assert [seed.config[cute_flash.FLASH_KV_STAGE_KEY] for seed in joint] == [3, 6]
    for seed in joint:
        assert seed.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4_2cta"
        assert seed.config[cute_flash.FLASH_ROWMAX_KEY] == "tmem"
        assert (
            seed.config[cute_flash.FLASH_SOFTMAX_REGS_KEY],
            seed.config[cute_flash.FLASH_CORR_REGS_KEY],
            seed.config[cute_flash.FLASH_OTHER_REGS_KEY],
        ) == (200, 64, 48)
    generation = ConfigGeneration(spec)
    for seed in resident:
        normalized = generation.unflatten(generation.flatten(seed))
        assert generation.unflatten(generation.flatten(normalized)) == normalized
        assert any(
            config == normalized for _, config in generation.seed_flat_config_pairs()
        )
        effective = _resolve(head_dim, num_kv, dtype=dtype, overrides=normalized.config)
        assert effective.softmax_lowering == "resident_value_graph"
        assert effective.exp2_impl == "xu"
        assert effective.e2e_schedule == "xu"
        assert effective.exp2_packet == "1x1"
        assert effective.e2e_offset == effective.e2e_offset0 == 0
        assert effective.sp_row_sum == "whole"
        assert effective.stat_transport == "single"


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_resident_override_owns_cadence_and_rejects_conflicting_request(
    dtype: torch.dtype,
) -> None:
    spec = _spec(128, 96, dtype)
    config = {
        cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph",
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta",
        cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
        cute_flash.FLASH_E2E_OFFSET_KEY: 4,
        cute_flash.FLASH_EXP2_PACKET_KEY: "deg2_16x6",
    }
    spec.prepare_override_normalization(
        config, {cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph"}
    )
    assert config[cute_flash.FLASH_E2E_SCHEDULE_KEY] == "xu"
    assert config[cute_flash.FLASH_EXP2_PACKET_KEY] == "1x1"
    assert config[cute_flash.FLASH_E2E_OFFSET_KEY] == 0
    with pytest.raises(InvalidConfig, match="requires cute_flash_e2e_schedule"):
        spec.prepare_override_normalization(
            config,
            {
                cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph",
                cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
            },
        )


def test_resident_runtime_keeps_fp32_math_and_selects_only_p_dtype() -> None:
    helper = _flash_runtime.resident_softmax_value_graph
    parameter = inspect.signature(helper).parameters["io_dtype"]
    assert str(parameter.default) == "Float16"
    source = inspect.getsource(helper)
    assert "assert tLDrS.element_type is cutlass.Float32" in source
    assert "cute.recast_ptr(tSTrS.iterator, dtype=io_dtype)" in source
    assert "src[None, ci]).load().to(io_dtype)" in source
    assert "fadd_reduce_packed(tLDrS, row_sum_init)" in source
    assert source.index("fence_view_async_tmem_store") < source.index("mbar_spin_wait")
    assert source.index("mbar_spin_wait") < source.index("fadd_reduce_packed")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("key", "value"),
    (
        (cute_flash.FLASH_EXP2_PACKET_KEY, "deg2_16x6"),
        (cute_flash.FLASH_SOFTMAX_DISC_KEY, True),
        (cute_flash.FLASH_STAT_TRANSPORT_KEY, "ring2"),
        (cute_flash.FLASH_S_LOAD_REP_KEY, 16),
        (cute_flash.FLASH_P_STORE_REP_KEY, 32),
    ),
)
def test_sampled_resident_cannot_overwrite_fixed_controls(
    dtype: torch.dtype, key: str, value: object
) -> None:
    spec = _spec(128, 96, dtype)
    config = {cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph", key: value}
    before = dict(config)
    with pytest.raises(InvalidConfig, match=f"requires {key}="):
        spec.prepare_override_normalization(config, {key: value})
    assert config == before


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sampled_resident_keeps_compatible_fixed_controls(dtype: torch.dtype) -> None:
    spec = _spec(128, 96, dtype)
    config = {
        cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph",
        cute_flash.FLASH_EXP2_PACKET_KEY: "1x1",
        cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
    }
    spec.prepare_override_normalization(
        config,
        {
            cute_flash.FLASH_EXP2_PACKET_KEY: "1x1",
            cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
        },
    )
    assert config[cute_flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_value_graph"
