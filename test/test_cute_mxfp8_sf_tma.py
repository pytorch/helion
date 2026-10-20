"""Packed scale TMA legality, cache identity and replacement-buffer staging."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test.test_cute_mxfp8_mma import _config
from test.test_cute_mxfp8_mma import _cpu_inputs
from test.test_cute_mxfp8_mma import _cpu_target
from test.test_cute_mxfp8_mma import _mxfp8
from test.test_cute_mxfp8_mma import _wrapper_plan
from test.test_cute_mxfp8_packed_provenance import _packed_mxfp8

import helion
from helion._compiler.autotuner_heuristics.cute_block_scaled import (
    CuteBlockScaledMmaHeuristic,
)
from helion._compiler.cute.block_scaled_config import BLOCK_SCALED_CHOICES
from helion._compiler.cute.block_scaled_config import block_scaled_choices
from helion._compiler.cute.block_scaled_config import normalize_block_scaled_config
from helion._testing import skipUnlessBackends
from helion.runtime.cute.launcher import _cute_stage_under_aligned_operands

pytestmark = skipUnlessBackends(["cute"])


def _packed_inputs(
    m: int = 32, n: int = 5120, k: int = 2048
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=torch.float8_e4m3fn),
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.empty_strided(
            (m, (k + 127) // 128), (1, (m + 3) // 4 * 4), dtype=torch.int32
        ),
        torch.empty_strided(
            (n, (k + 127) // 128), (1, (n + 3) // 4 * 4), dtype=torch.int32
        ),
        torch.empty((m, n), dtype=torch.bfloat16),
    )


def test_sf_tma_vocabulary_keeps_the_existing_default_and_nvfp4_domain() -> None:
    assert block_scaled_choices() == BLOCK_SCALED_CHOICES
    assert block_scaled_choices(mxfp8=True)["cute_scaled_sf_tma"] == (False, True)
    config: dict[str, object] = {"cute_scaled_mma": True}
    normalize_block_scaled_config(
        config,
        available=True,
        capability=(10, 3),
        fix_invalid=False,
        mxfp8=True,
        fused_scales_available=True,
        sf_tma_available=True,
    )
    assert config["cute_scaled_sf_tma"] is False


@pytest.mark.parametrize("sf_tma", [1, "true"])
def test_sf_tma_requires_a_boolean(sf_tma: object) -> None:
    with pytest.raises(helion.exc.InvalidConfig):
        normalize_block_scaled_config(
            {"cute_scaled_sf_tma": sf_tma},
            available=True,
            capability=(10, 3),
            fix_invalid=False,
            mxfp8=True,
            fused_scales_available=True,
            sf_tma_available=True,
        )


@pytest.mark.parametrize("shape", [(1, 129, 32), (7, 257, 128)])
def test_unit_word_tensor_map_axes_keep_global_scale_loading(
    shape: tuple[int, int, int],
) -> None:
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(_packed_inputs(*shape))
        assert bound.config_spec.cute_mxfp8_fused_scales_available
        assert not bound.config_spec.cute_mxfp8_sf_tma_available
        with pytest.raises(helion.exc.InvalidConfig, match="SF TMA"):
            bound.to_code(
                _config(bound, cute_scaled_fused_scales=True, cute_scaled_sf_tma=True)
            )


@pytest.mark.parametrize("shape", [(1, 129, 160), (73, 1, 160), (1, 1, 160)])
def test_unit_row_tensor_map_axes_preserve_descriptor_bounds(
    shape: tuple[int, int, int],
) -> None:
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(_packed_inputs(*shape))
        assert bound.config_spec.cute_mxfp8_sf_tma_available
        config = _config(
            bound,
            cute_scaled_fused_scales=True,
            cute_scaled_sf_tma=True,
            cute_scaled_sf_inplace=True,
            cute_scaled_cluster_m=1,
            cute_scaled_tile_n=32,
            cute_scaled_tile_k=128,
            cute_scaled_stages=2,
        )
        plan = _wrapper_plan(bound.to_code(config))
        assert (plan["m"], plan["n"], plan["k"]) == shape
        assert plan["sf_tma"] is True and plan["sf_inplace"] is True


def test_byte_scales_cannot_select_the_packed_scale_tma() -> None:
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((73, 273, 288)))
        assert not bound.config_spec.cute_mxfp8_sf_tma_available
        with pytest.raises(helion.exc.InvalidConfig, match="SF TMA"):
            bound.to_code(_config(bound, cute_scaled_sf_tma=True))


def test_native_tma_config_preserves_json_flattening_and_source_identity() -> None:
    initialized = torch.cuda.is_initialized()
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(_packed_inputs())
        assert bound.config_spec.cute_mxfp8_sf_tma_available
        base = _config(
            bound,
            cute_scaled_fused_scales=True,
            cute_scaled_tile_n=32,
            cute_scaled_tile_k=256,
            cute_scaled_stages=2,
            cute_scaled_tile_m=64,
            cute_scaled_swap_ab=True,
            cute_scaled_compact_m=True,
        )
        varied = helion.Config.from_dict(base.config | {"cute_scaled_sf_tma": True})
        normalized = bound.config_spec.normalized_config(varied)
        serialized = helion.Config.from_dict(
            __import__("json").loads(normalized.to_json())
        )
        assert bound.config_spec.normalized_config(serialized) == normalized
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            restored = generation.unflatten(generation.flatten(varied))
        source = bound.to_code(restored)
        assert source == bound.to_code(varied)
        assert _wrapper_plan(source)["sf_tma"] is True
        assert _wrapper_plan(bound.to_code(base))["sf_tma"] is False
        assert source != bound.to_code(base)
    assert torch.cuda.is_initialized() is initialized


def test_raw_sf_buffers_are_included_in_the_native_shared_memory_limit() -> None:
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(_packed_inputs())
        geometry = {
            "cute_scaled_fused_scales": True,
            "cute_scaled_tile_m": 128,
            "cute_scaled_tile_n": 16,
            "cute_scaled_tile_k": 128,
            "cute_scaled_stages": 12,
            "cute_scaled_cluster_m": 2,
            "cute_scaled_swap_ab": True,
        }
        assert (
            _wrapper_plan(bound.to_code(_config(bound, **geometry)))["sf_tma"] is False
        )
        with pytest.raises(helion.exc.BackendUnsupported):
            bound.to_code(_config(bound, **geometry, cute_scaled_sf_tma=True))
        restored = _wrapper_plan(
            bound.to_code(
                _config(
                    bound,
                    **geometry,
                    cute_scaled_sf_tma=True,
                    cute_scaled_sf_inplace=True,
                )
            )
        )
        assert restored["sf_tma"] is True and restored["sf_inplace"] is True


def test_inplace_requires_the_scale_tma_path() -> None:
    with pytest.raises(helion.exc.InvalidConfig, match="requires SF TMA"):
        normalize_block_scaled_config(
            {"cute_scaled_sf_inplace": True, "cute_scaled_fused_scales": True},
            available=True,
            capability=(10, 3),
            fix_invalid=False,
            mxfp8=True,
            fused_scales_available=True,
            sf_tma_available=True,
        )


def test_joint_native_inplace_profile_preserves_both_persistent_choices() -> None:
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(_packed_inputs())
        with bound.env:
            seeds = CuteBlockScaledMmaHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            choices = [bound.config_spec.normalized_config(seed) for seed in seeds]
        matching = [
            config
            for config in choices
            if config.get("cute_scaled_sf_inplace") is True
            and config["cute_scaled_cluster_m"] == 2
            and config["cute_scaled_tile_m"] == 128
            and config["cute_scaled_tile_n"] == 16
            and config["cute_scaled_tile_k"] == 128
            and config["cute_scaled_stages"] == 12
            and config.get("cute_scaled_swap_ab") is True
            and config.get("cute_scaled_compact_m") is False
        ]
        assert {config["cute_scaled_persistent"] for config in matching} == {
            False,
            True,
        }
        for config in matching:
            assert _wrapper_plan(bound.to_code(config))["sf_inplace"] is True


def test_legal_old_seeds_remain_and_the_proved_sf_pipeline_is_discoverable() -> None:
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(_packed_inputs())
        with bound.env:
            seeds = CuteBlockScaledMmaHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            normalized = [bound.config_spec.normalized_config(seed) for seed in seeds]
        assert (
            len(
                [
                    seed
                    for seed in normalized
                    if seed.get("cute_scaled_sf_tma") is not True
                ]
            )
            == 49
        )
        assert any(
            seed.get("cute_scaled_sf_tma") is True
            and seed["cute_scaled_tile_n"] == 32
            and seed["cute_scaled_tile_k"] == 256
            and seed["cute_scaled_stages"] == 2
            and seed["cute_scaled_tile_m"] == 64
            for seed in normalized
        )


def test_aligned_replacement_of_sf_words_preserves_padding_stride_and_values() -> None:
    backing = torch.arange(80, dtype=torch.int32)
    words = backing[1:].as_strided((7, 10), (1, 8))
    assert words.data_ptr() % 16 == 4
    snapshots = words.clone()
    inputs = _packed_inputs(7, 257, 1280)
    arguments = (*inputs[:2], words, inputs[3])
    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=[
            {
                "kind": "block_scaled_mma",
                "format": "mxfp8",
                "sf_tma": True,
                "lhs_idx": 0,
                "rhs_idx": 1,
                "lhs_scale_idx": 2,
                "rhs_scale_idx": 3,
            }
        ]
    )
    staged = _cute_stage_under_aligned_operands(kernel, arguments)
    assert staged is not None
    values, copy_backs = staged
    assert not copy_backs
    assert values[2].data_ptr() % 16 == 0
    assert values[2].stride() == words.stride()
    assert torch.equal(values[2], words) and torch.equal(words, snapshots)
    assert values[0] is inputs[0] and values[1] is inputs[1] and values[3] is inputs[3]
