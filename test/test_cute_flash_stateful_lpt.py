from __future__ import annotations

import ast
import dataclasses
import random
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
import helion
from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash as flash
from helion._compiler.cute.attention_plan import causal_score_plan
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.device_ir import DeviceIR


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


def _spec(heads: int, *, proof: bool = True) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=152,
    )
    for index, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=index, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=64,
        num_kv=32,
        num_bh=heads,
        dtype=torch.float16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=True,
        standard_causal_output=True,
        causal_resident_compatible=proof,
    )
    return spec


def _resolve(
    heads: int, width: int, *, lowering: str = "resident_stateful"
) -> flash.FlashAttentionConfig:
    return flash.resolve_flash_config(
        64,
        32,
        {
            flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            flash.FLASH_SOFTMAX_LOWERING_KEY: lowering,
            flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: width,
        },
        dtype=torch.float16,
        num_bh=heads,
        is_causal=True,
        standard_causal_output=True,
        causal_resident_compatible=True,
        target_device_capability=(10, 3),
    )


@pytest.mark.parametrize("heads", (1, 3, 8, 64, 65, 127, 129))
def test_fixed_stateful_width_survives_canonical_round_trip(heads: int) -> None:
    spec = _spec(heads)
    generation = ConfigGeneration(spec)
    for width in (0, 1, 2, 3, 4, 8, 16, 32, 64):
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_softmax_lowering="resident_stateful",
            cute_flash_causal_lpt_swizzle=width,
        )
        spec.normalize(config)
        assert config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == max(
            1, min(width, heads)
        )
        assert generation.unflatten(generation.flatten(config)) == config
        assert (
            _resolve(heads, width).causal_lpt_swizzle
            == config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
        )


@pytest.mark.parametrize("lowering", ("auto", "standard"))
def test_historical_routes_keep_serial_mapping(lowering: str) -> None:
    spec = _spec(64)
    config = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family="fa4",
        cute_flash_softmax_lowering=lowering,
        cute_flash_causal_lpt_swizzle=64,
    )
    spec.normalize(config)
    assert config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == 1
    assert _resolve(64, 64, lowering=lowering).causal_lpt_swizzle == 1


def test_unqualified_stateful_does_not_gain_a_mapping() -> None:
    config = helion.Config(
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family="fa4",
        cute_flash_softmax_lowering="resident_stateful",
        cute_flash_causal_lpt_swizzle=64,
    )
    with pytest.raises(InvalidConfig):
        _spec(64, proof=False).normalize(config)


@pytest.mark.parametrize("heads", (1, 3, 64, 129))
def test_active_widths_are_effective_and_compiler_seeded(heads: int) -> None:
    spec = _spec(heads)
    fragment = spec._cute_flash_autotune_fragments()[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
    assert 64 in fragment.choices
    assert fragment._active_choices() == flash._flash_stateful_lpt_candidates(heads)
    seeds = spec.autotune_seed_configs()
    assert seeds
    assert all(
        seed.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
        in (0, *fragment._active_choices())
        for seed in seeds
    )


@pytest.mark.parametrize("head_dim,dtype", ((64, torch.float16), (128, torch.bfloat16)))
@pytest.mark.parametrize(
    "heads,num_kv", ((1, 2), (3, 6), (65, 6), (127, 6), (129, 6), (64, 512))
)
def test_wider_source_changes_only_bijective_cta_decoder(
    head_dim: int, dtype: torch.dtype, heads: int, num_kv: int
) -> None:
    def emit(width: int) -> str:
        config = flash.resolve_flash_config(
            head_dim,
            num_kv,
            {
                flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
                flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: width,
            },
            dtype=dtype,
            num_bh=heads,
            is_causal=True,
            standard_causal_output=True,
            causal_resident_compatible=True,
            target_device_capability=(10, 3),
        )
        body = flash.emit_flash_fa4_device_body(
            cast("DeviceFunction", None),
            head_dim=head_dim,
            num_kv=num_kv,
            sequence_extent=num_kv * 128,
            num_bh=heads,
            total_tiles=heads * num_kv // 2,
            cfg=config,
            has_lse=False,
            io_dtype="cutlass.Float16"
            if dtype is torch.float16
            else "cutlass.BFloat16",
            score_plan=causal_score_plan(head_dim),
            target_device_capability=(10, 3),
        )
        return ast.unparse(ast.Module(body=body, type_ignores=[]))

    decoder = re.compile(
        r"(?m)^flash_pid = cutlass.Int32\(cute.arch.block_idx\(\)\[0\]\)\n"
        r"[\s\S]*?^flash_q_mma_tile1 = flash_m_tile1$"
    )
    serial = emit(1)
    old = decoder.search(serial)
    assert old is not None
    for width in (2, 4, 8, 16, 32, 64):
        source = emit(width)
        new = decoder.search(source)
        assert new is not None
        assert source[: new.start()] + old[0] + source[new.end() :] == serial
        # Execute precisely the emitted integer decoder, replacing only device
        # scalar constructors and the block-index read with host integers.
        program = (
            new[0]
            .replace("cutlass.Int32", "int")
            .replace("cute.arch.block_idx()[0]", "pid")
        )
        code = compile(program, "<actual CTA decoder>", "exec")
        visited = set()
        for pid in range(heads * num_kv // 2):
            values = {"pid": pid}
            exec(code, {"int": int}, values)
            pair = (values["flash_bh"], values["flash_m_pair"])
            assert pair not in visited
            visited.add(pair)
        assert visited == {
            (head, pair) for head in range(heads) for pair in range(num_kv // 2)
        }


@pytest.mark.parametrize("schedule", ("post_acquire", "pre_acquire"))
def test_both_schedules_reach_every_active_width(schedule: str) -> None:
    spec = _spec(64)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    )
    widths = flash._flash_stateful_lpt_candidates(64)
    for width in widths:
        base, config = generation.canonicalize_flat(
            generation.flatten(
                helion.Config(
                    cute_flash_softmax_lowering="resident_stateful",
                    cute_flash_row_sum_schedule=schedule,
                    cute_flash_causal_lpt_swizzle=width,
                )
            )
        )
        assert config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == width
        assert config.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == schedule
        owned = generation.flash_owned_coordinate_indices(config)
        identities = generation._flat_coordinate_identities()
        assert all(
            identities[i][0]
            not in (
                flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY,
                flash.FLASH_ROW_SUM_SCHEDULE_KEY,
            )
            for i in owned
        )
        neighbors = generation.coordinate_neighbor_projections(
            base, radius=64, frozen_indices=owned
        )
        assert {
            item.config.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
            for item in neighbors
            if item.key == flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
            and item.config is not None
        } == set(widths) - {width}
        switches = [
            item.config
            for item in neighbors
            if item.key == flash.FLASH_SOFTMAX_LOWERING_KEY and item.config is not None
        ]
        assert {c.config[flash.FLASH_SOFTMAX_LOWERING_KEY] for c in switches} >= {
            "auto",
            "standard",
        }
        assert all(c.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == 1 for c in switches)
        assert all(
            c.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == "post_acquire"
            for c in switches
        )


@pytest.mark.parametrize("causal", (False, True))
def test_inactive_owned_mapping_uses_workload_context(causal: bool) -> None:
    for lowering in ("auto", "standard", "resident_value_graph"):
        owned = flash._flash_resident_softmax_overrides(
            lowering, is_causal=causal, topology="fa4"
        )
        assert owned[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == int(causal)
    assert (
        flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
        not in flash._flash_resident_softmax_overrides(
            "resident_stateful", is_causal=causal, topology="fa4"
        )
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {flash.FLASH_P_STORE_REP_KEY: 32},
        {flash.FLASH_SOFTMAX_LOWERING_KEY: "auto"},
        {flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"},
    ),
)
def test_fixed_parent_excludes_unreachable_wide_mapping(
    overrides: dict[str, object],
) -> None:
    generation = _spec(64).create_config_generation(overrides=overrides)
    fragment = generation._flat_fields()[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment._active_choices() == (1,)
    assert all(
        key != flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY or value == 1
        for key, value in generation.flash_structural_coverage_active_values()
    )


def test_fixed_wide_mapping_limits_lowering_without_widening_user_domain() -> None:
    spec = _spec(64)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 32}
    )
    mode = generation._flat_fields()[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(mode, EnumFragment)
    assert mode._active_choices() == ("resident_stateful",)
    fields = dict(spec._flat_fields())
    width = fields[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]
    assert isinstance(width, EnumFragment)
    fields[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] = dataclasses.replace(
        width, search_choices=(8, 32), coverage_choices=(8, 32)
    )
    narrowed = ConfigGeneration(spec, _field_view=fields)
    assert narrowed._flat_fields()[
        flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY
    ]._active_choices() == (8, 32)
    with pytest.raises(ValueError, match="search_choices must not be empty"):
        ConfigGeneration(
            spec,
            overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"},
            _field_view=fields,
        )
    assert fields[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY]._active_choices() == (8, 32)


@pytest.mark.parametrize(
    "heads,tiles,sms,expected",
    (
        (64, 32, 152, (1, 8, 32, 64)),
        (64, 32, 32, (1, 2, 32, 64)),
        (3, 6, 152, (1, 3)),
        (64, 512, 152, (1, 2, 64)),
        (1, 32, 152, (1,)),
        (None, 32, 152, (1,)),
        (64, 32, None, (1,)),
    ),
)
def test_generic_seed_widths_use_captured_geometry(
    heads: int | None, tiles: int, sms: int | None, expected: tuple[int, ...]
) -> None:
    assert flash._FLASH_LPT_SEED_KV_BUDGET_BYTES == 50 * 1024 * 1024
    assert (
        flash._flash_stateful_lpt_seed_widths(
            64, tiles, num_bh=heads, num_sm=sms, dtype=torch.float16
        )
        == expected
    )


def test_compiler_seed_routes_share_sm_count_and_preserve_old_prefix() -> None:
    spec = _spec(64)
    spec.num_sm = 32
    with patch.object(flash, "_flash_stateful_lpt_seed_widths", return_value=(1,)):
        old = spec.autotune_seed_configs()
    seeds = spec.autotune_seed_configs()
    assert seeds[: len(old)] == old
    assert (
        CuteFlashAttentionHeuristic.get_seed_configs(
            SimpleNamespace(config_spec=spec), cast("DeviceIR", None)
        )
        == seeds
    )
    stateful = [
        c
        for c in seeds
        if c.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
    ]
    assert {c.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] for c in stateful} == {
        1,
        2,
        32,
        64,
    }
    for width in (1, 2, 32, 64):
        at_width = [
            c for c in stateful if c.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] == width
        ]
        assert {
            c.config.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire")
            for c in at_width
        } == {"post_acquire", "pre_acquire"}
        assert {c.config[flash.FLASH_ROWMAX_KEY] for c in at_width} == {
            "software",
            "tmem",
        }


def test_normal100_retains_all_seeds_and_both_schedules() -> None:
    spec = _spec(64)
    spec.compiler_seed_configs = spec.autotune_seed_configs()
    generation = ConfigGeneration(spec)
    state = random.getstate()
    try:
        random.seed(613)
        population = generation.random_population_flat(100, user_seed_configs=())
    finally:
        random.setstate(state)
    configs = [generation.canonicalize_flat(flat)[1] for flat in population]
    seeds = {config for _, config in generation.seed_flat_config_pairs()}
    coverage = set(generation.flash_deterministic_population_configs())
    # The complete design fits the nominal budget, so retain every coverage
    # row as well as every seed. Their union may exceed the nominal 100 rows.
    assert len(coverage) <= 100
    required = seeds | coverage
    assert len(configs) == len(set(configs)) == max(100, len(required))
    assert required <= set(configs)
    stateful = [
        c
        for c in configs
        if c.config[flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_stateful"
    ]
    assert {c.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] for c in stateful} == {
        "post_acquire",
        "pre_acquire",
    }
    assert {c.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY] for c in stateful} == set(
        flash._flash_stateful_lpt_candidates(64)
    )
