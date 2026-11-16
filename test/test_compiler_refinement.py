from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.compiler_refinement import compiler_refinement_candidates
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.search_space_logger import canonical_config_id
import helion.language as hl
from helion.runtime.config_cache import ConfigCache
from helion.runtime.config_cache import ConfigSet
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only
KEYS = ("first_option", "second_option", "third_option")
CHOICES = ((False, True), (False, True), (0, 1, 2, 3))
# These remain four distinct layouts after every backend's normalization.
PARENT_BLOCK_SIZES = (128, 256, 512, 1024)


def _pointwise(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@pytest.fixture(scope="module")
def pointwise_bound():
    with cpu_render_target():
        return helion.kernel(_pointwise).bind((torch.empty(2048),))


@pytest.fixture
def bound(pointwise_bound):
    with (
        cpu_render_target(),
        pointwise_bound.env,
        patch.dict(
            pointwise_bound.config_spec.user_defined_tunables,
            {
                key: EnumFragment(values)
                for key, values in zip(KEYS, CHOICES, strict=True)
            },
        ),
    ):
        yield pointwise_bound


def _member(search, size, values, perf=1.0, status="ok"):
    config = helion.Config(block_sizes=[size], **dict(zip(KEYS, values, strict=True)))
    flat, config = search.config_gen.strict_config_pair(config)
    return PopulationMember(lambda: None, [perf], flat, config, status=status)


def _search(bound, *, initial=(False, False, 0), keys=KEYS):
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.settings = Settings(
        backend=bound.config_spec.backend_name, autotune_effort="full"
    )
    search.config_spec = bound.config_spec
    search.config_gen = ConfigGeneration(bound.config_spec)
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.args = ()
    search.benchmark_provider = SimpleNamespace()
    search.final_selection = FinalSelectionConfig()
    search._compiler_refinement_keys = keys
    search._autotune_metrics = AutotuneMetrics()
    search._autotune_budget_start = None
    search._fallback_excluded_configs = ConfigSet()
    search._benchmarked_members = ConfigCache()
    search._pinned_finalist_members = ConfigCache()
    search._pinned_finalist_configs = ConfigSet()
    search._terminal_refinement_members = ConfigCache()
    search.log = Mock()
    history = [
        _member(search, size, initial, 1.0 - index * 0.1)
        for index, size in enumerate(PARENT_BLOCK_SIZES)
    ]
    assert (
        tuple(member.config.block_sizes[0] for member in history) == PARENT_BLOCK_SIZES
    )
    # Older optimistic timings must not replace the actual current incumbent.
    search.population = [history[0]]
    for member in history:
        search._record_benchmarked_member(member)
    return search


def _plan(search):
    history = sorted(search._terminal_refinement_members.values(), key=lambda m: m.perf)
    return compiler_refinement_candidates(
        search.config_gen,
        [search.best.config, *(m.config for m in history)],
        search._compiler_refinement_keys,
    )


def _measure(search, calls, *, winner=None):
    def benchmark(members, **kwargs):
        assert kwargs["raise_if_no_viable_config"] is False
        for member in members:
            config = member.config
            calls.append(copy.deepcopy(config))
            values = tuple(config[key] for key in KEYS)
            member.perfs.append(
                0.5 if (config.block_sizes[0], values) == winner else 2.0
            )
            member.status = "ok"
            search._record_benchmarked_member(member)
        return members

    return benchmark


@pytest.mark.parametrize("initial", [(False, False, 0), (True, True, 3)])
def test_complete_parent_neighborhoods_find_later_joint_winner(bound, initial):
    search = _search(bound, initial=initial)
    assert search.best.config.block_sizes == [PARENT_BLOCK_SIZES[0]]
    assert len(search._terminal_refinement_members) == 4
    plan = _plan(search)
    winner_values = (not initial[0], not initial[1], 3 - initial[2])
    calls = []
    with patch.object(
        search,
        "benchmark_population",
        side_effect=_measure(
            search, calls, winner=(PARENT_BLOCK_SIZES[1], winner_values)
        ),
    ):
        winner = search.run_compiler_refinement(search.best)
    assert tuple(winner.config[key] for key in KEYS) == winner_values
    assert winner.config.block_sizes == [PARENT_BLOCK_SIZES[1]]
    assert calls == [candidate.config for candidate in plan]
    assert len(calls) == 60
    for size in PARENT_BLOCK_SIZES:
        assert {
            tuple(config[key] for key in KEYS)
            for config in calls
            if config.block_sizes == [size]
        } == set(itertools.product(*CHOICES)) - {initial}
    trace = search._autotune_metrics.compiler_refinement
    assert trace["new_trials"] == 60
    assert {row["outcome"] for row in trace["trials"]} == {"ok"}
    assert search._budget_reserve_released is False


def test_complete_small_plan_keeps_original_fair_order(bound):
    search = _search(bound, keys=KEYS[:2])
    plan = _plan(search)
    assert len(plan) == 12
    calls = []
    with patch.object(
        search, "benchmark_population", side_effect=_measure(search, calls)
    ):
        search.run_compiler_refinement(search.best)
    assert calls == [candidate.config for candidate in plan]
    assert "incumbent_neighborhood" not in search._autotune_metrics.compiler_refinement


def test_refinement_transcript_snapshots_containers_without_opaque_copy(bound):
    class CopyHook(str):
        __slots__ = ()

        def __deepcopy__(self, memo):
            raise AssertionError("opaque config copy hook")

    search = _search(bound)
    candidate = _plan(search)[0]
    opaque = CopyHook("opaque")
    payload = {"owners": [opaque]}
    candidate.config.config["diagnostic"] = payload
    # Supply a candidate at the planner boundary to check transcript capture
    # independently of the current planner's narrower immutable-value policy.
    with (
        patch(
            "helion.autotuner.base_search.compiler_refinement_candidates",
            return_value=[candidate],
        ),
        patch.object(search, "_backend_config_is_viable", return_value=False),
    ):
        assert search.run_compiler_refinement(search.best) is search.best
    trace = search._autotune_metrics.compiler_refinement
    row = trace["trials"][0]
    assert row["outcome"] == "invalid"
    snapshot = row["config"]["diagnostic"]
    assert snapshot is not payload and snapshot["owners"] is not payload["owners"]
    assert snapshot["owners"][0] is opaque
    payload["owners"].append("changed")
    assert snapshot["owners"] == [opaque]


def test_known_outcomes_and_backend_rejections_do_not_consume_trials(bound):
    search = _search(bound)
    plan = _plan(search)
    assert len(plan) == 60
    invalid = ConfigSet()
    # Exactly 16 unseen admissions remain; all four prior-outcome classes
    # avoid a new trial while preserving the candidate order.
    for index, candidate in enumerate(plan[16:]):
        kind = index % 4
        if kind == 2:
            search._fallback_excluded_configs.add(candidate.config)
        elif kind == 3:
            invalid.add(candidate.config)
        else:
            member = PopulationMember(
                lambda: None,
                [5.0 if kind == 0 else float("inf")],
                candidate.flat_values,
                candidate.config,
                status="ok" if kind == 0 else "accuracy_error",
            )
            search._record_benchmarked_member(member)
    calls = []
    with (
        patch.object(
            search, "_backend_config_is_viable", side_effect=lambda c: c not in invalid
        ),
        patch.object(
            search, "benchmark_population", side_effect=_measure(search, calls)
        ),
    ):
        search.run_compiler_refinement(search.best)
    assert calls == [candidate.config for candidate in plan[:16]]
    trace = search._autotune_metrics.compiler_refinement
    assert "incumbent_neighborhood" not in trace
    assert trace["new_trials"] == 16
    assert {row["outcome"] for row in trace["trials"]} == {
        "ok",
        "reused",
        "prior_failure",
        "quarantined",
        "invalid",
    }


def test_deadline_preserves_final_selection_budget(bound):
    search = _search(bound)
    plan = _plan(search)
    for candidate, status in zip(plan[-3:-1], ("ok", "accuracy_error"), strict=True):
        member = PopulationMember(
            lambda: None,
            [5.0 if status == "ok" else float("inf")],
            candidate.flat_values,
            candidate.config,
            status=status,
        )
        search._record_benchmarked_member(member)
    search._fallback_excluded_configs.add(plan[-1].config)
    search.settings.autotune_budget_seconds = 100
    search._autotune_budget_start = 100.0
    clock = [174.99]

    def viable(config):
        clock[0] += 0.1
        return True

    with (
        patch(
            "helion.autotuner.base_search.time.perf_counter",
            side_effect=lambda: clock[0],
        ),
        patch.object(search, "_backend_config_is_viable", side_effect=viable),
        patch.object(search, "_budget_reserve_seconds", return_value=25),
        patch.object(search, "_in_flight_hold_back_seconds", return_value=0),
        patch.object(search, "benchmark_population") as benchmark,
    ):
        assert search.run_compiler_refinement(search.best) is search.best
    benchmark.assert_not_called()
    trace = search._autotune_metrics.compiler_refinement
    assert "incumbent_neighborhood" not in trace
    assert trace["new_trials"] == 0 and trace["budget_exhausted"]
    assert {row["outcome"] for row in trace["trials"]} == {
        "budget_exhausted",
        "reused",
        "prior_failure",
        "quarantined",
    }
    assert 175.0 <= clock[0] < 175.1
    assert search._budget_reserve_released is False


def test_deadline_rechecked_after_candidate_construction(bound):
    search = _search(bound, keys=KEYS[:1])
    search.settings.autotune_budget_seconds = 100
    search._autotune_budget_start = 100.0
    clock = [174.0]
    checks = []

    def viable(config):
        checks.append(config)
        # The actual make_unbenchmarked check follows admission viability
        # and consumes the remaining search time.
        if len(checks) == 2:
            clock[0] = 175.1
        return True

    with (
        patch(
            "helion.autotuner.base_search.time.perf_counter",
            side_effect=lambda: clock[0],
        ),
        patch.object(search, "_backend_config_is_viable", side_effect=viable),
        patch.object(search, "_budget_reserve_seconds", return_value=25),
        patch.object(search, "_in_flight_hold_back_seconds", return_value=0),
        patch.object(search, "benchmark_population") as benchmark,
    ):
        assert search.run_compiler_refinement(search.best) is search.best
    benchmark.assert_not_called()
    assert len(checks) == 2
    trace = search._autotune_metrics.compiler_refinement
    assert trace["new_trials"] == 0 and trace["budget_exhausted"]
    assert search._budget_reserve_released is False


def test_large_product_keeps_single_coordinate_fallback_and_trial_cap(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables,
        third_option=EnumFragment(tuple(range(16))),
    ):
        search = _search(bound)
        plan = _plan(search)
        assert len(plan) == 68
        calls = []
        with patch.object(
            search, "benchmark_population", side_effect=_measure(search, calls)
        ):
            search.run_compiler_refinement(search.best)
    assert calls == [candidate.config for candidate in plan[:64]]
    assert "incumbent_neighborhood" not in search._autotune_metrics.compiler_refinement

    trace = search._autotune_metrics.compiler_refinement
    assert trace["new_trials"] == 64
    assert sum(row["outcome"] == "trial_limit" for row in trace["trials"]) == 4


def test_invalid_incumbent_keeps_valid_history_neighborhoods(bound):
    search = _search(bound)
    search.best.config.config["unknown_key"] = 1
    plan = _plan(search)
    calls = []
    with patch.object(
        search, "benchmark_population", side_effect=_measure(search, calls)
    ):
        search.run_compiler_refinement(search.best)
    assert calls == [candidate.config for candidate in plan]
    assert len(calls) == 60
    assert "incumbent_neighborhood" not in search._autotune_metrics.compiler_refinement


def test_refinement_rechecks_quarantine_after_each_trial(bound):
    search = _search(bound)
    plan = _plan(search)
    incumbent = [
        candidate
        for candidate in plan
        if candidate.parent.block_sizes == [PARENT_BLOCK_SIZES[0]]
    ]
    blocked = incumbent[1].config
    calls = []
    measure = _measure(search, calls)

    def benchmark(members, **kwargs):
        result = measure(members, **kwargs)
        search._fallback_excluded_configs.add(blocked)
        return result

    with patch.object(search, "benchmark_population", side_effect=benchmark):
        search.run_compiler_refinement(search.best)
    assert len(calls) == 59 and blocked not in calls
    trace = search._autotune_metrics.compiler_refinement
    rows = [
        row
        for row in trace["trials"]
        if row["config_id"] == canonical_config_id(blocked)
    ]
    assert len(rows) == 1 and rows[0]["outcome"] == "quarantined"


def test_exact_duplicate_parents_collapse_but_signed_zero_structures_do_not(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables, sentinel=EnumFragment((0.0, -0.0))
    ):
        search = _search(bound)
        positive = copy.deepcopy(search.best.config)
        negative = copy.deepcopy(positive)
        negative.config["sentinel"] = -0.0
        parents = [positive, copy.deepcopy(positive), negative, copy.deepcopy(negative)]
        candidates = compiler_refinement_candidates(search.config_gen, parents, KEYS)
    assert len(candidates) == 30
    assert (
        len({canonical_config_id(candidate.config) for candidate in candidates}) == 30
    )
    assert {candidate.parent["sentinel"].hex() for candidate in candidates} == {
        "0x0.0p+0",
        "-0x0.0p+0",
    }


@pytest.mark.parametrize("budget", [None, 1000])
def test_refinement_budget_checks_preserve_generation_history(bound, budget):
    search = _search(bound)
    search.settings.autotune_budget_seconds = budget
    search._autotune_budget_start = 1.0
    search._last_generation_boundary = 2.0
    search._longest_generation_seconds = 3.0
    calls = []
    with (
        patch("helion.autotuner.base_search.time.perf_counter", return_value=10.0),
        patch.object(
            search, "benchmark_population", side_effect=_measure(search, calls)
        ),
    ):
        search.run_compiler_refinement(search.best)
    assert calls
    assert search._last_generation_boundary == 2.0
    assert search._longest_generation_seconds == 3.0
    assert not search._budget_reserve_released
