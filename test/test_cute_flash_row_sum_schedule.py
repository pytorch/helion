"""Private causal row-sum scheduling and shared-state ordering; no native compile."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .test_cute_flash_exp2 import _emit_causal_resident_native_source
from .test_cute_flash_exp2 import _emit_dense_resident_value_graph_source
from .test_cute_flash_resident_choice import _emit as _emit_dense
from .test_cute_flash_stateful_choice import _emit
from .test_cute_flash_stateful_choice import _resolve
from .test_cute_flash_stateful_choice import _spec
import helion
from helion._compiler.cute import cute_flash as flash
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("num_kv", "rowmax", "threshold"),
    (
        (2, "software", 0.0),
        (2, "tmem", 8.0),
        (6, "software", 0.0),
        (6, "software", 8.0),
        (6, "tmem", 0.0),
        (6, "tmem", 8.0),
    ),
)
def test_explicit_stateful_rowsum_precedes_stat_acquire(
    head_dim: int, dtype: torch.dtype, num_kv: int, rowmax: str, threshold: float
) -> None:
    config, source = _emit(
        head_dim,
        num_kv,
        dtype=dtype,
        overrides={
            flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire",
            flash.FLASH_ROWMAX_KEY: rowmax,
            flash.FLASH_RESCALE_THRESHOLD_KEY: threshold,
        },
    )
    assert config.softmax_lowering == "resident_stateful"
    assert config.row_sum_schedule == "pre_acquire"
    assert config.stat_transport == "single"
    assert config.rowmax == rowmax
    assert config.rescale_threshold == threshold
    module = ast.parse(source)
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    counts = {
        ast.unparse(call.args[0]): ast.literal_eval(call.args[1])
        for call in calls
        if ast.unparse(call.func) == "cute.arch.mbarrier_init"
    }
    assert counts["flash_pfor2_ptr + flash_st"] == 128
    assert counts["flash_pfor_ptr + flash_st"] == 256
    assert counts["flash_s0_corr_empty_ptr + flash_st"] == 128
    assert counts["flash_s1_corr_empty_ptr + flash_st"] == 128
    final_calls = [
        call
        for call in calls
        if ast.unparse(call.func) == "_helion_flash_rt.mbarrier_arrive"
        and ast.unparse(call.args[0]).startswith("flash_pfor2_ptr + ")
    ]
    assert len(final_calls) == 6
    for stage in (0, 1):
        arrival = f"_helion_flash_rt.mbarrier_arrive(flash_pfor2_ptr + {stage})"
        matched = 0
        for node in ast.walk(module):
            for _, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                for index, statement in enumerate(value):
                    if ast.unparse(statement) != arrival:
                        continue
                    matched += 1
                    assert ast.unparse(value[index - 1]) == (
                        "cute.arch.fence_view_async_tmem_store()"
                    )
                    # Both P stores and publications precede private Rmem reduction.
                    earlier = [ast.unparse(child) for child in value[:index]]
                    assert (
                        f"_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + {stage})"
                    ) in earlier
                    assert ast.unparse(value[index + 1]).startswith(
                        "flash_softmax.update_row_sum(tLDrS.load(), flash_alpha"
                    )
                    assert ast.unparse(value[index + 2]).startswith(
                        "_helion_flash_rt.mbar_spin_wait("
                        f"flash_s{stage}_corr_empty_ptr + 0, flash_s_corr_prod_phase,"
                    )
                    assert (
                        ast.unparse(value[index + 3]) == "flash_s_corr_prod_phase ^= 1"
                    )
        # One first-step, one masked-loop and one unmasked-loop source site.
        # The second slot's masked loop remains legitimately zero-trip.
        assert matched == 3


@pytest.mark.parametrize("mode", ("auto", "standard"))
def test_automatic_and_standard_final_p_retain_thread_counts(mode: str) -> None:
    for emit in (
        _emit_causal_resident_native_source,
        _emit_dense_resident_value_graph_source,
    ):
        source = emit(config_overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: mode})
        counts = [
            ast.literal_eval(node.args[1])
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.mbarrier_init"
            and ast.unparse(node.args[0]).startswith("flash_pfor2_ptr + ")
        ]
        assert len(counts) == 1
        assert counts[0] in (128, 256)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_explicit_dense_final_p_retains_thread_count(dtype: torch.dtype) -> None:
    config, source = _emit_dense(dtype=dtype)
    assert config.softmax_lowering == "resident_value_graph"
    assert "resident_softmax_value_graph" in source
    counts = [
        ast.literal_eval(node.args[1])
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.mbarrier_init"
        and ast.unparse(node.args[0]).startswith("flash_pfor2_ptr + ")
    ]
    assert counts == [256]


def test_historical_automatic_stateful_retains_acquire_before_rowsum() -> None:
    source = _emit_causal_resident_native_source()
    matched = 0
    for node in ast.walk(ast.parse(source)):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value):
                if not ast.unparse(statement).startswith(
                    "flash_softmax.update_row_sum(tLDrS.load(), flash_alpha"
                ):
                    continue
                matched += 1
                assert ast.unparse(value[index - 1]).startswith(
                    "_helion_flash_rt.mbar_spin_wait(flash_s"
                )
                assert ast.unparse(value[index + 1]) == "flash_s_corr_prod_phase ^= 1"
    assert matched == 6


@pytest.mark.parametrize("mode", ("auto", "standard", "resident_value_graph"))
def test_other_lowerings_own_post_acquire_schedule(mode: str) -> None:
    assert (
        flash._flash_resident_softmax_overrides(mode)[flash.FLASH_ROW_SUM_SCHEDULE_KEY]
        == "post_acquire"
    )
    spec = _spec(64, 32, torch.float16)
    with pytest.raises(InvalidConfig, match="requires cute_flash_row_sum_schedule"):
        spec.prepare_override_normalization(
            {flash.FLASH_SOFTMAX_LOWERING_KEY: mode},
            {flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire"},
        )


@pytest.mark.parametrize("value", ("unknown", "", False, 1))
def test_invalid_row_sum_schedule_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="invalid flash row-sum schedule"):
        _resolve(overrides={flash.FLASH_ROW_SUM_SCHEDULE_KEY: value})


@pytest.mark.parametrize("mode", ("auto", "standard"))
def test_nonstateful_samples_canonicalize_inactive_schedule(mode: str) -> None:
    config = _resolve(
        overrides={
            flash.FLASH_SOFTMAX_LOWERING_KEY: mode,
            flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire",
        }
    )
    assert config.row_sum_schedule == "post_acquire"


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_both_schedules_normalize_idempotently_and_are_searchable(
    head_dim: int, dtype: torch.dtype
) -> None:
    spec = _spec(head_dim, 32, dtype)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful"}
    )
    base, parent = generation.canonicalize_flat(
        generation.flatten(helion.Config(cute_flash_pipeline_family="fa4"))
    )
    assert parent.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == "post_acquire"
    owned = generation.flash_owned_coordinate_indices(parent)
    projections = generation.coordinate_neighbor_projections(base, frozen_indices=owned)
    choices = [p for p in projections if p.key == flash.FLASH_ROW_SUM_SCHEDULE_KEY]
    assert len(choices) == 1 and choices[0].outcome == "candidate"
    pre = choices[0].config
    assert pre is not None
    assert pre.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == "pre_acquire"
    assert pre.config == {
        **parent.config,
        flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire",
    }
    expected = dict(pre.config)
    spec.normalize(pre)
    assert pre.config == expected
    assert (
        flash.FLASH_ROW_SUM_SCHEDULE_KEY
        not in flash._flash_resident_softmax_overrides("resident_stateful")
    )


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("num_kv", (2, 6, 32, 100))
def test_general_stateful_seed_pairs_preserve_post_alternative(
    head_dim: int, dtype: torch.dtype, num_kv: int
) -> None:
    spec = _spec(head_dim, num_kv, dtype)
    raw = spec.autotune_seed_configs()
    stateful = [
        seed
        for seed in raw
        if seed.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
    ]
    post = [
        seed for seed in stateful if flash.FLASH_ROW_SUM_SCHEDULE_KEY not in seed.config
    ]
    pre = [
        seed
        for seed in stateful
        if seed.config.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY) == "pre_acquire"
    ]
    assert post and len(post) == len(pre)
    assert [seed.config for seed in pre] == [
        {**seed.config, flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire"}
        for seed in post
    ]
    for seed in stateful:
        requested = seed.config.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire")
        spec.normalize(seed)
        assert seed.config[flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_stateful"
        assert seed.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == requested


def test_pre_acquire_seed_and_search_choice_require_actual_detector_proof() -> None:
    spec = _spec(64, 32, torch.float16, proof=False)
    fragment = spec._cute_flash_autotune_fragments()[flash.FLASH_ROW_SUM_SCHEDULE_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment._active_choices() == ("post_acquire",)
    assert all(
        seed.config.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire")
        == "post_acquire"
        for seed in spec.autotune_seed_configs()
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {flash.FLASH_P_STORE_REP_KEY: 32},
        {flash.FLASH_SOFTMAX_LOWERING_KEY: "auto"},
        {flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"},
    ),
)
def test_fixed_parent_domains_do_not_advertise_inactive_pre_acquire(
    overrides: dict[str, object],
) -> None:
    spec = _spec(64, 32, torch.float16)
    generation = spec.create_config_generation(overrides=overrides)
    fragment = generation._flat_fields()[flash.FLASH_ROW_SUM_SCHEDULE_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment._active_choices() == ("post_acquire",)
    assert (flash.FLASH_ROW_SUM_SCHEDULE_KEY, "pre_acquire") not in (
        generation.flash_structural_coverage_active_values()
    )


def test_fixed_pre_acquire_limits_lowering_domain() -> None:
    spec = _spec(64, 32, torch.float16)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire"}
    )
    fragment = generation._flat_fields()[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment._active_choices() == ("resident_stateful",)


def test_incompatible_pre_only_subset_is_rejected_without_widening() -> None:
    spec = _spec(64, 32, torch.float16)
    fields = dict(spec._flat_fields())
    fragment = fields[flash.FLASH_ROW_SUM_SCHEDULE_KEY]
    assert isinstance(fragment, EnumFragment)
    narrowed = dataclasses.replace(fragment, search_choices=("pre_acquire",))
    fields[flash.FLASH_ROW_SUM_SCHEDULE_KEY] = narrowed
    with pytest.raises(ValueError, match="search_choices must not be empty"):
        ConfigGeneration(
            spec,
            overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"},
            _field_view=fields,
        )
    assert narrowed._active_choices() == ("pre_acquire",)


def test_pre_acquire_parent_can_switch_to_ordinary_lowerings() -> None:
    spec = _spec(64, 32, torch.float16)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    )
    base, parent = generation.canonicalize_flat(
        generation.flatten(
            helion.Config(
                cute_flash_softmax_lowering="resident_stateful",
                cute_flash_row_sum_schedule="pre_acquire",
            )
        )
    )
    projections = generation.coordinate_neighbor_projections(
        base, frozen_indices=generation.flash_owned_coordinate_indices(parent)
    )
    switches = [
        p.config for p in projections if p.key == flash.FLASH_SOFTMAX_LOWERING_KEY
    ]
    assert {
        c.config[flash.FLASH_SOFTMAX_LOWERING_KEY] for c in switches if c is not None
    } == {"auto", "standard"}
    assert all(
        c is not None and c.config[flash.FLASH_ROW_SUM_SCHEDULE_KEY] == "post_acquire"
        for c in switches
    )
