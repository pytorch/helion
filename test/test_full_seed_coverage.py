from __future__ import annotations

import copy
import math
import random
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from test.test_compiler_coverage import make_search
from test.test_compiler_coverage import make_spec

from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.seed_coverage import CompilerSeedWitness
from helion.autotuner.seed_coverage import append_seed_coverage
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.config import Config


def coverage_search(*, count=100, many=True, cls=LFBOTreeSearch, **kwargs):
    spec = make_spec(expanded=False, many_seeds=many)
    spec.compiler_seed_witnesses = (
        CompilerSeedWitness("extra", Config(block_sizes=[32], old_choice=245)),
    )
    search = make_search(spec, count=count, cls=cls, **kwargs)
    search.settings.autotune_effort = "full"
    search.settings.autotune_compiler_seed_coverage = True
    return search


@pytest.mark.parametrize("cls", (PatternSearch, LFBOPatternSearch, LFBOTreeSearch))
@pytest.mark.parametrize("count", (1, 100, 121, 150))
@pytest.mark.parametrize("strategy", tuple(InitialPopulationStrategy))
def test_overflow_keeps_base_rng_and_covers_every_seed(cls, count, strategy):
    search = coverage_search(count=count, cls=cls, strategy=strategy)
    original_seeds = copy.deepcopy(search.config_spec.compiler_seed_configs)
    default = search.config_spec.default_config()
    random.seed(193)
    expected = search._generate_initial_population_with_coverage()
    rng = random.getstate()
    random.seed(193)
    actual = search._generate_initial_population_flat()
    assert actual[: len(expected)] == expected
    assert random.getstate() == rng
    present = {search.config_gen.unflatten(row) for row in actual}
    assert all(
        config in present for _, config in search.config_gen.seed_flat_config_pairs()
    )
    _, witness = search.config_gen.strict_config_pair(
        search.config_spec.compiler_seed_witnesses[0].config
    )
    assert witness in present
    assert search.config_spec.compiler_seed_configs == original_seeds
    assert search.config_spec.default_config() == default
    assert witness in search._pinned_finalist_configs


@pytest.mark.parametrize(
    "change", ("quick", "none", "optout", "disabled", "zero", "other_backend")
)
def test_policy_scope_is_exact(change):
    search = coverage_search()
    if change in ("quick", "none"):
        search.settings.autotune_effort = change
    elif change == "optout":
        search.settings.autotune_compiler_seed_coverage = False
    elif change == "disabled":
        search.settings.disable_autotuner_heuristics = True
    elif change == "zero":
        search.initial_population = 0
    else:
        search.config_spec.backend_name = "triton"
        assert not search._full_seed_coverage_enabled()
        assert "full_compiler_seed_coverage" not in search._algorithm_cache_policy()
        return
    random.seed(194)
    expected = search._generate_initial_population_with_coverage()
    rng = random.getstate()
    random.seed(194)
    actual = search._generate_initial_population_flat()
    assert actual == expected and random.getstate() == rng
    assert not search.seed_coverage_outcomes
    assert "full_compiler_seed_coverage" not in search._algorithm_cache_policy()


def test_witness_is_immutable_and_cache_policy_binds_recipe():
    search = coverage_search()
    old = search._algorithm_cache_policy()
    witness = search.config_spec.compiler_seed_witnesses[0]
    snapshot = witness.config
    sizes = snapshot.config["block_sizes"]
    assert isinstance(sizes, list)
    sizes[0] = 64
    assert witness.config.config["block_sizes"] == [32]
    search.config_spec.compiler_seed_witnesses = (
        CompilerSeedWitness("extra", snapshot),
    )
    assert old != search._algorithm_cache_policy()


def test_overrides_and_invalid_witness_do_not_get_silently_repaired():
    search = coverage_search(overrides={"old_choice": 10})
    search.config_spec.compiler_seed_witnesses += (
        CompilerSeedWitness("invalid", Config(block_sizes=[32], old_choice="foreign")),
    )
    rows, outcomes = append_seed_coverage([], search.config_gen)
    assert len(rows) == 1
    assert {x.status for x in outcomes[-2:]} == {"overridden"}
    search = coverage_search()
    search.config_spec.compiler_seed_witnesses = (
        CompilerSeedWitness("invalid", Config(block_sizes=[32], old_choice="foreign")),
    )
    _, outcomes = append_seed_coverage([], search.config_gen)
    assert outcomes[-1].status.startswith("invalid:")


def setup_benchmark(*, budget_after=None, provider_cancels=False, fail_base=False):
    search = coverage_search(count=1, many=False)
    search.settings.autotune_budget_seconds = (
        10 if budget_after is not None or provider_cancels else None
    )
    rows = search._generate_initial_population_flat()
    members = []
    for row in rows:
        member = search.make_unbenchmarked(row)
        assert member is not None
        members.append(member)
    search._autotune_metrics = AutotuneMetrics()
    search.benchmark_provider = Mock()
    search.benchmark_provider.take_no_viable_config_error.return_value = None
    search.benchmark_provider.take_effective_source_repairs.return_value = []
    batches = []
    search._autotune_budget_exceeded_across_ranks = lambda: (
        budget_after is not None and len(batches) >= budget_after
    )

    def benchmark(configs, **kwargs):
        batches.append(copy.deepcopy(configs))
        if provider_cancels and len(batches) == 2:
            search._autotune_budget_exceeded_across_ranks = lambda: True
        failed = (fail_base and len(batches) == 1) or (
            provider_cancels and len(batches) == 2
        )
        return [
            BenchmarkResult(
                c,
                lambda: None,
                math.inf if failed else float(c.config["old_choice"]),
                "error" if failed else "ok",
                0.0,
            )
            for c in configs
        ]

    search.benchmark_batch = benchmark
    return search, members, batches


def test_real_member_update_and_finalist_path_precedes_pruning():
    search, members, batches = setup_benchmark()
    with patch.object(
        PopulationBasedSearch, "rebenchmark_population", autospec=True
    ) as final:
        assert search.benchmark_initial_population(members) is members
        # Both PatternSearch and LFBO invoke this original method immediately
        # after the initial call. All added members now carry ordinary results.
        search.rebenchmark_population(members)
        assert final.call_args.args[1] is members
    assert len(batches) == 2
    assert sum(map(len, batches)) == len(members)
    assert all(member.perfs and member.status == "ok" for member in members)
    assert all(member.config in search._pinned_finalist_configs for member in members)
    assert all(
        row.status == "ok"
        for row in search.seed_coverage_outcomes
        if row.status != "already_present"
    )


def test_uncapped_overflow_batches_all_admitted_configs_in_order():
    search, members, batches = setup_benchmark()
    requested = [member.config for member in members]
    search.benchmark_initial_population(members)
    assert batches == [requested[:1], requested[1:]]
    assert [member.config for member in members] == requested


def test_uncapped_batch_provider_exception_marks_submitted_requests():
    search, members, batches = setup_benchmark()
    original = search.benchmark_batch

    def interrupted(configs, **kwargs):
        if batches:
            raise RuntimeError("uncapped coverage interrupted")
        return original(configs, **kwargs)

    search.benchmark_batch = interrupted
    with pytest.raises(RuntimeError, match="uncapped coverage interrupted"):
        search.benchmark_initial_population(members)
    assert len(batches) == 1
    assert all(
        row.status == "provider_interrupted" for row in search.seed_coverage_outcomes
    )


def test_failed_base_does_not_hide_valid_supplemental_candidate():
    search, members, batches = setup_benchmark(fail_base=True)
    search.benchmark_initial_population(members)
    assert not math.isfinite(members[0].perf)
    assert any(math.isfinite(member.perf) for member in members[1:])
    assert len(batches) == 2
    assert sum(map(len, batches)) == len(members)


def test_budget_marks_exact_unsubmitted_tail():
    search, members, batches = setup_benchmark(budget_after=2)
    search.benchmark_initial_population(members)
    assert len(batches) == 2
    states = [row.status for row in search.seed_coverage_outcomes]
    assert states == [
        "ok",
        "unattempted_budget",
        "unattempted_budget",
        "unattempted_budget",
    ]
    assert all(not math.isfinite(member.perf) for member in members[2:])


def test_provider_cancellation_is_not_counted_as_success_or_forced_work():
    search, members, batches = setup_benchmark(provider_cancels=True)
    search.benchmark_initial_population(members)
    assert len(batches) == 2
    assert [row.status for row in search.seed_coverage_outcomes] == [
        "provider_interrupted_or_failed",
        "unattempted_budget",
        "unattempted_budget",
        "unattempted_budget",
    ]


def test_base_provider_exception_records_unattempted_coverage():
    search, members, batches = setup_benchmark()
    error = RuntimeError("provider stopped")
    search.benchmark_batch = Mock(side_effect=error)
    with pytest.raises(RuntimeError, match="provider stopped"):
        search.benchmark_initial_population(members)
    assert not batches
    assert all(
        row.status == "unattempted_interrupted" for row in search.seed_coverage_outcomes
    )
    phase = search._autotune_metrics.search_phase_metrics
    assert phase is not None and phase["compiler_seed_coverage"]


def test_source_filter_and_accuracy_failure_are_never_pre_scored():
    search, members, batches = setup_benchmark()
    original = search.benchmark_batch

    def rejected(configs, **kwargs):
        results = original(configs, **kwargs)
        if len(batches) > 1:
            return [
                result._replace(perf=math.inf, status="accuracy_error")
                for result in results
            ]
        return results

    search.benchmark_batch = rejected
    search.benchmark_initial_population(members)
    assert all(
        member.status == "accuracy_error" and not math.isfinite(member.perf)
        for member in members[1:]
    )
    assert all(row.status == "accuracy_error" for row in search.seed_coverage_outcomes)


def test_non_gdn_graph_without_structural_carrier_keeps_original_codegen():
    import torch

    from test.cute_population_contracts import _pointwise
    from test.cute_population_contracts import _target

    import helion

    with _target():
        args = (torch.empty(37),)
        bound = helion.kernel(_pointwise, backend="cute", autotune_effort="full").bind(
            args
        )
        assert not bound.config_spec.compiler_seed_witnesses
        seeds = copy.deepcopy(bound.config_spec.compiler_seed_configs)
        default = bound.config_spec.default_config()
        bound.settings.autotune_compiler_seed_coverage = False
        control = bound.to_code(default)
        bound.settings.autotune_compiler_seed_coverage = True
        assert bound.to_code(default) == control
        assert bound.config_spec.compiler_seed_configs == seeds
        assert bound.config_spec.default_config() == default


def test_ordinary_cache_key_distinguishes_full_coverage_and_witness_policy():
    from helion.autotuner.local_cache import _cute_flash_search_policy_hash

    search = coverage_search()
    first = _cute_flash_search_policy_hash(search, cute_flash_search_enabled=False)
    assert first
    search.config_spec.compiler_seed_witnesses = (
        CompilerSeedWitness("changed", Config(block_sizes=[32], old_choice=244)),
    )
    second = _cute_flash_search_policy_hash(search, cute_flash_search_enabled=False)
    assert second != first
    search.settings.autotune_compiler_seed_coverage = False
    assert _cute_flash_search_policy_hash(search, cute_flash_search_enabled=False) == ""


def test_existing_coordinate_coverage_prefix_and_sampling_remain_intact():
    search = make_search(make_spec(expanded=True, many_seeds=True), count=3)
    search.settings.autotune_effort = "full"
    search.settings.autotune_compiler_seed_coverage = True
    random.seed(196)
    base = search._generate_initial_population_with_coverage()
    state = random.getstate()
    random.seed(196)
    full = search._generate_initial_population_flat()
    assert full[: len(base)] == base
    assert random.getstate() == state
    assert all(
        outcome.status != "pending" or outcome.effective is not None
        for outcome in search.seed_coverage_outcomes
    )


@pytest.mark.parametrize("named_group", (False, True))
def test_distributed_context_keeps_old_path_despite_different_local_budgets(
    named_group,
):
    rank_rows = []
    for exhausted in (False, True):
        search = coverage_search(count=3)
        search.kernel.env.process_group_name = "original_group" if named_group else None
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=exhausted)
        with (
            patch("torch.distributed.is_initialized", return_value=not named_group),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=AssertionError("new collective"),
            ),
        ):
            random.seed(197)
            expected = search._generate_initial_population_with_coverage()
            random.seed(197)
            actual = search._generate_initial_population_flat()
            assert actual == expected and not search.seed_coverage_outcomes
            assert not search.requires_search_policy_cache_key()
            with patch.object(
                PopulationBasedSearch, "benchmark_initial_population", return_value=[]
            ) as original:
                search.benchmark_initial_population([])
                original.assert_called_once()
            search._autotune_budget_exceeded_across_ranks.assert_not_called()
            rank_rows.append(actual)
    assert rank_rows[0] == rank_rows[1]


@pytest.mark.parametrize(
    "error", (AssertionError("invariant"), KeyError("field"), TypeError("contract"))
)
def test_internal_transfer_errors_are_not_swallowed(error):
    search = coverage_search()
    with (
        patch.object(search.config_gen, "strict_config_pair", side_effect=error),
        pytest.raises(type(error)),
    ):
        append_seed_coverage([], search.config_gen)


@pytest.mark.parametrize("kind", ("already_present", "invalid", "overridden"))
def test_no_pending_requests_are_recorded_with_population_status(kind):
    search = coverage_search(
        count=10,
        many=False,
        overrides={"old_choice": 10} if kind == "overridden" else {},
    )
    if kind == "already_present":
        config = search.config_spec.compiler_seed_configs[0]
    else:
        config = Config(
            block_sizes=[32], old_choice="foreign" if kind == "invalid" else 245
        )
    search.config_spec.compiler_seed_witnesses = (
        CompilerSeedWitness("request", config),
    )
    rows = search._generate_initial_population_flat()
    assert not any(row.status == "pending" for row in search.seed_coverage_outcomes)
    members = []
    for flat in rows:
        member = search.make_unbenchmarked(flat)
        assert member is not None
        member.status = "ok"
        members.append(member)
    search._autotune_metrics = AutotuneMetrics()
    with patch.object(
        PopulationBasedSearch, "benchmark_initial_population", return_value=members
    ):
        search.benchmark_initial_population(members)
    phase = search._autotune_metrics.search_phase_metrics
    assert phase is not None
    records = phase["compiler_seed_coverage"]
    assert isinstance(records, list)
    last = records[-1]
    assert last["status"].startswith(kind)
    assert last["population_status"] == (
        "ok" if kind in ("already_present", "overridden") else None
    )
