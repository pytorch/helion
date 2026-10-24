"""Joint output-transfer coverage under real packed contraction metadata."""

from __future__ import annotations

import copy
import random
from unittest.mock import patch

import pytest

from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.test_compiler_coverage import make_search
from test.test_cute_mxfp8_mma import _cpu_inputs
from test.test_cute_mxfp8_mma import _mxfp8
from test.test_cute_mxfp8_mma import _wrapper_plan
from test.test_cute_mxfp8_registered_grid import _bound

import helion
from helion._compiler.autotuner_heuristics.cute_mxfp8_epilogue_coverage import KEY
from helion._compiler.autotuner_heuristics.cute_mxfp8_epilogue_coverage import (
    register_mxfp8_epilogue_coverage,
)
from helion._compiler.cute.block_scaled_config import MXFP8_EPILOGUE_MODES
from helion._compiler.cute.block_scaled_config import mxfp8_shared_memory_bytes
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion.autotuner.metrics import AutotuneMetrics


@pytest.fixture(autouse=True)
def _cpu_only():
    with (
        _forbid_cuda_work(),
        _mock_cuda_unavailable(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=228 * 1024
        ),
    ):
        yield


def _group(bound):
    groups = [
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == KEY
    ]
    assert len(groups) == 1
    return groups[0]


def _clear_groups(spec) -> None:
    # The ordinary field domain and seed list remain present; recreate only
    # the registration to exercise unavailable or invalid seed carriers.
    spec._compiler_coverage_groups = ()
    spec._compiler_coverage_fields = ()


def test_joint_witnesses_keep_registered_owners_and_strict_geometry() -> None:
    bound = _bound(8, 264, 256)
    group = _group(bound)
    assert group.domain == tuple(MXFP8_EPILOGUE_MODES)
    assert [w.value for w in group.witnesses] == list(group.domain[1:])
    assert len(group.supplemental_witnesses) == 1
    assert group.supplemental_witnesses[0].value == "tma_prefetch_2"
    for witness in (*group.witnesses, *group.supplemental_witnesses):
        carrier = witness.carrier
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            _, effective = generation.strict_config_pair(
                helion.Config.from_dict(carrier.config | {KEY: witness.value})
            )
        assert {
            key: value for key, value in effective.config.items() if key != KEY
        } == {key: value for key, value in carrier.config.items() if key != KEY}
        assert len(effective["block_sizes"]) == 3
        assert effective["partition_count"] == 1
        assert effective["independent_choice"] == 2
        assert effective["cute_scaled_fused_scales"]
        assert effective["cute_scaled_sf_tma"]
        assert effective["cute_scaled_sf_inplace"]
        plan = _wrapper_plan(bound.to_code(effective))
        assert plan["tma_output"]
        assert plan["sf_tma"] and plan["sf_inplace"]
        assert (
            mxfp8_shared_memory_bytes(
                8,
                128,
                224,
                128,
                6,
                2,
                2,
                compact_m=False,
                sf_tma=True,
                sf_inplace=True,
                epilogue_mode=witness.value,
            )
            <= 232448
        )
    full2 = [
        witness.carrier["cute_scaled_persistent"]
        for witness in (*group.witnesses, *group.supplemental_witnesses)
        if witness.value == "tma_prefetch_2"
    ]
    assert set(full2) == {False, True}


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 264, 256), (8, 1, 256), (8, 264, 32), (8, 264, 128), (8, 263, 256)],
)
def test_unsupported_metadata_has_no_output_coverage(m: int, n: int, k: int) -> None:
    bound = _bound(m, n, k)
    assert not any(
        group.key == KEY for group in bound.config_spec.compiler_coverage_groups
    )
    seeds = copy.deepcopy(bound.config_spec.compiler_seed_configs)
    with bound.env:
        register_mxfp8_epilogue_coverage(bound.env)
    assert bound.config_spec.compiler_seed_configs == seeds
    assert not any(
        group.key == KEY for group in bound.config_spec.compiler_coverage_groups
    )


def test_unpacked_scale_family_has_no_output_coverage() -> None:
    from test._cute_binding import _cpu_bind

    bound = _cpu_bind(_mxfp8, _cpu_inputs((8, 264, 256)))
    assert bound.config_spec.cute_mxfp8_mma_available
    assert not bound.config_spec.cute_mxfp8_tma_output_available
    assert not any(
        group.key == KEY for group in bound.config_spec.compiler_coverage_groups
    )


def test_no_legal_seed_carrier_retains_an_empty_registry() -> None:
    bound = _bound(8, 264, 256)
    group = _group(bound)
    spec = bound.config_spec
    # These otherwise exact carriers violate the recovered partition contract.
    spec.compiler_seed_configs = [
        helion.Config.from_dict(witness.carrier.config | {"partition_count": 2})
        for witness in (group.witnesses[0], group.supplemental_witnesses[0])
    ]
    _clear_groups(spec)
    before = copy.deepcopy(spec.compiler_seed_configs)
    default = copy.deepcopy(spec.default_config())
    with bound.env:
        register_mxfp8_epilogue_coverage(bound.env)
    assert spec.compiler_coverage_groups == ()
    assert spec.compiler_seed_configs == before
    assert spec.default_config() == default
    assert spec.create_config_generation().initial_population_view().config_spec is spec


def test_missing_joint_seed_family_does_not_manufacture_a_carrier() -> None:
    bound = _bound(8, 264, 256)
    spec = bound.config_spec
    spec.compiler_seed_configs = [spec.default_config()]
    _clear_groups(spec)
    with bound.env:
        register_mxfp8_epilogue_coverage(bound.env)
    assert spec.compiler_coverage_groups == ()


def test_only_one_legal_persistence_keeps_three_modes() -> None:
    bound = _bound(8, 264, 256)
    carrier = _group(bound).supplemental_witnesses[0].carrier
    assert carrier["cute_scaled_persistent"] is False
    spec = bound.config_spec
    spec.compiler_seed_configs = [carrier]
    _clear_groups(spec)
    with bound.env:
        register_mxfp8_epilogue_coverage(bound.env)
    group = _group(bound)
    assert [w.value for w in group.witnesses] == list(group.domain[1:])
    assert all(w.carrier["cute_scaled_persistent"] is False for w in group.witnesses)
    assert group.supplemental_witnesses == ()


def test_malformed_persistence_is_not_an_implicit_boolean() -> None:
    bound = _bound(8, 264, 256)
    carrier = _group(bound).witnesses[0].carrier
    spec = bound.config_spec
    spec.compiler_seed_configs = [
        helion.Config.from_dict(carrier.config | {"cute_scaled_persistent": 1})
    ]
    _clear_groups(spec)
    with bound.env:
        register_mxfp8_epilogue_coverage(bound.env)
    assert spec.compiler_coverage_groups == ()


@pytest.mark.parametrize("disabled", (False, True))
def test_user_default_override_or_disabled_heuristics_does_not_force_modes(
    disabled: bool,
) -> None:
    bound = _bound(8, 264, 256)
    search = make_search(
        bound.config_spec,
        disabled=disabled,
        overrides=None if disabled else {KEY: "default"},
    )
    random.seed(6102966)
    configs = [
        search.config_gen.unflatten(copy.deepcopy(row))
        for row in search._generate_initial_population_flat()
    ]
    assert len(configs) == 100
    assert all(config[KEY] == "default" for config in configs)
    assert not any(
        entry.outcome == "added"
        for entry in getattr(search, "compiler_coverage_outcomes", ())
    )


def test_actual_full_loop_measures_joint_profiles_after_the_nominal_population() -> (
    None
):
    bound = _bound(8, 264, 256)
    search = make_search(bound.config_spec)
    search._autotune_metrics = AutotuneMetrics()
    delivered = []

    class HeldBenchmark(Exception):
        pass

    def hold(members, *, desc, raise_if_no_viable_config=True):
        assert desc == "Initial population"
        delivered.extend(copy.deepcopy(member.config) for member in members)
        raise HeldBenchmark

    random.seed(6102966)
    with (
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        search._autotune()
    assert len(delivered) == 104
    added = [
        entry.effective
        for entry in search.compiler_coverage_outcomes
        if entry.outcome == "added"
    ]
    assert len(added) == 4
    assert all(config in delivered[100:] for config in added)
    assert all(config in search._pinned_finalist_configs for config in added)
    assert all(config[KEY] == "default" for config in delivered[:100])
