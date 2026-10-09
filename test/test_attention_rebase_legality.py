"""Keep the rebased row/alternating families separate from resident FA4 modes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import cute_flash as flash
from helion._compiler.device_ir import DeviceIR
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


def _spec(
    *,
    num_kv: int = 4,
    score_modifiers: bool = False,
    row_epilogue: bool = False,
    causal: bool = False,
) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=152,
    )
    for index, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=index, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=128,
        num_kv=num_kv,
        num_bh=4,
        dtype=torch.float16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
        causal_resident_compatible=causal,
        plain_row_body=not score_modifiers and not row_epilogue,
        has_row_epilogue=row_epilogue,
        has_score_modifiers=score_modifiers,
        device_sm_count=152,
    )
    return spec


@pytest.mark.parametrize("num_kv", (3, 4))
@pytest.mark.parametrize(
    "selector", (flash.FLASH_PIPELINE_FAMILY_KEY, flash.FLASH_TOPOLOGY_KEY)
)
def test_row_topology_precedes_fa4_paired_query_rule(
    num_kv: int, selector: str
) -> None:
    spec = _spec(num_kv=num_kv)
    parent = {selector: "row_mma"}
    assert spec._resolve_cute_flash_config(parent).topology == "row_mma"
    assert (
        spec._cute_flash_config_topology(
            {
                **parent,
                flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
                flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire",
            }
        )
        == "row_mma"
    )


@pytest.mark.parametrize("num_kv", (3, 4))
def test_row_topology_respects_workload_admission(num_kv: int) -> None:
    spec = _spec(num_kv=num_kv, score_modifiers=True)
    parent = {flash.FLASH_PIPELINE_FAMILY_KEY: "row_mma"}
    expected = "ws_overlap" if num_kv % 2 else "fa4"
    assert spec._resolve_cute_flash_config(parent).topology == expected
    assert spec._cute_flash_config_topology(parent) == expected


@pytest.mark.parametrize("num_kv", (3, 4))
@pytest.mark.parametrize(
    "env_key", ("HELION_CUTE_FLASH_PIPELINE_FAMILY", "HELION_CUTE_FLASH_TOPOLOGY")
)
def test_environment_row_topology_retains_odd_kv_and_explicit_precedence(
    monkeypatch: pytest.MonkeyPatch, num_kv: int, env_key: str
) -> None:
    monkeypatch.setenv(env_key, "row_mma")
    spec = _spec(num_kv=num_kv)
    assert spec._resolve_cute_flash_config({}).topology == "row_mma"
    assert (
        spec._cute_flash_config_topology(
            {flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful"}
        )
        == "row_mma"
    )
    explicit = {flash.FLASH_TOPOLOGY_KEY: "fa4"}
    expected = "ws_overlap" if num_kv % 2 else "fa4"
    assert spec._resolve_cute_flash_config(explicit).topology == expected
    assert spec._cute_flash_config_topology(explicit) == expected


@pytest.mark.parametrize("family", ("row_mma", "fa4_alt"))
@pytest.mark.parametrize("lowering", ("resident_value_graph", "resident_stateful"))
def test_separate_emitters_reject_explicit_resident_lowerings(
    family: str, lowering: str
) -> None:
    spec = _spec()
    config = {
        "block_sizes": [1, 128, 128],
        flash.FLASH_PIPELINE_FAMILY_KEY: family,
        flash.FLASH_SOFTMAX_LOWERING_KEY: lowering,
    }
    with pytest.raises(InvalidConfig):
        spec._resolve_cute_flash_config(config)
    with pytest.raises(InvalidConfig):
        spec.normalized_config(config)


@pytest.mark.parametrize("score_modifiers", (False, True))
def test_detector_heuristic_and_spec_share_row_admission(
    score_modifiers: bool,
) -> None:
    spec = _spec(score_modifiers=score_modifiers)
    env = Mock(spec=CompileEnvironment)
    env.config_spec = spec
    heuristic_seeds = CuteFlashAttentionHeuristic.get_seed_configs(
        env, Mock(spec=DeviceIR)
    )
    assert heuristic_seeds is not None
    direct_seeds = spec.autotune_seed_configs()
    for seeds in (heuristic_seeds, direct_seeds):
        row_seeds = [
            seed
            for seed in seeds
            if seed.config.get(flash.FLASH_PIPELINE_FAMILY_KEY) == "row_mma"
        ]
        assert bool(row_seeds) is not score_modifiers
        for seed in row_seeds:
            normalized = spec.normalized_config(seed.config)
            assert normalized.config[flash.FLASH_PIPELINE_FAMILY_KEY] == "row_mma"


@pytest.mark.parametrize("family", ("row_mma", "fa4_alt"))
def test_separate_emitters_canonicalize_inactive_resident_coordinates(
    family: str,
) -> None:
    spec = _spec()
    base = {
        "block_sizes": [1, 128, 128],
        flash.FLASH_PIPELINE_FAMILY_KEY: family,
    }
    canonical = spec.normalized_config(base)
    alias = spec.normalized_config(
        {
            **base,
            flash.FLASH_SOFTMAX_LOWERING_KEY: "standard",
            flash.FLASH_ROWMAX_KEY: "tmem",
            flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire",
            flash.FLASH_P_CHUNK_ARRIVE_KEY: True,
        }
    )
    assert alias == canonical
    assert canonical.config[flash.FLASH_SOFTMAX_LOWERING_KEY] == "auto"
    assert canonical.config[flash.FLASH_ROWMAX_KEY] == "software"
    assert canonical.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == "post_acquire"
    assert canonical.config[flash.FLASH_P_CHUNK_ARRIVE_KEY] is False


def test_resident_p_chunk_ownership_preserves_ordinary_choice() -> None:
    spec = _spec()
    family = {flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    resident = {**family, flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_value_graph"}
    with pytest.raises(InvalidConfig, match="requires cute_flash_p_chunk_arrive"):
        contradictory = spec.create_config_generation(
            overrides={**resident, flash.FLASH_P_CHUNK_ARRIVE_KEY: True}
        )
        contradictory.canonicalize_flat(contradictory.default_flat())

    sampled = spec.create_config_generation(
        overrides={**family, flash.FLASH_P_CHUNK_ARRIVE_KEY: True}
    )
    fragment = sampled._flat_fields()[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(fragment, EnumFragment)
    assert "resident_value_graph" not in fragment._active_choices()

    ordinary_generation = spec.create_config_generation(
        overrides={
            **family,
            flash.FLASH_SOFTMAX_LOWERING_KEY: "standard",
            flash.FLASH_SOFTMAX_DISC_KEY: True,
            flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
            flash.FLASH_P_CHUNK_ARRIVE_KEY: True,
        }
    )
    ordinary = ordinary_generation.canonicalize_flat(
        ordinary_generation.default_flat()
    )[1]
    assert ordinary.config[flash.FLASH_P_CHUNK_ARRIVE_KEY] is True

    generation = ConfigGeneration(spec)
    config = spec.normalized_config({"block_sizes": [1, 128, 128], **resident})
    assert config.config[flash.FLASH_P_CHUNK_ARRIVE_KEY] is False
    identities = generation._flat_coordinate_identities()
    owned = generation.flash_owned_coordinate_indices(config)
    ordinary_owned = generation.flash_owned_coordinate_indices(ordinary)
    assert flash.FLASH_P_CHUNK_ARRIVE_KEY in {identities[i][0] for i in owned}
    assert flash.FLASH_P_CHUNK_ARRIVE_KEY not in {
        identities[i][0] for i in ordinary_owned
    }


@pytest.mark.parametrize(
    ("causal", "lowering"),
    ((False, "resident_value_graph"), (True, "resident_stateful")),
)
def test_row_epilogue_keeps_chunked_safety_and_excludes_resident_seeds(
    causal: bool, lowering: str
) -> None:
    spec = _spec(causal=causal, row_epilogue=True)
    parent = {flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    assert spec._resolve_cute_flash_config(parent).softmax_disc is True
    request = {**parent, flash.FLASH_SOFTMAX_LOWERING_KEY: lowering}
    with pytest.raises(InvalidConfig):
        spec._resolve_cute_flash_config(request)
    with pytest.raises(InvalidConfig):
        spec.normalized_config({"block_sizes": [1, 128, 128], **request})
    fragments = spec._cute_flash_autotune_fragments("fa4", "fa4")
    fragment = fragments[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(fragment, EnumFragment)
    assert lowering not in fragment._active_choices()
    for seed in spec.autotune_seed_configs():
        assert seed.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) not in (
            "resident_value_graph",
            "resident_stateful",
        )
