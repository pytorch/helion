from __future__ import annotations

from contextlib import ExitStack
import copy
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_config
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.compiler_refinement import compiler_refinement_candidates
from helion.autotuner.compiler_refinement import same_refinement_value
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.search_surface import BackendSearchSurface
import helion.language as hl
from helion.runtime.config_cache import ConfigCache
from helion.runtime.config_cache import ConfigSet
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only
KEY = "cute_compiler_pipeline"


def _pointwise(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@pytest.fixture(scope="module")
def pointwise_bound():
    with cpu_target():
        return cpu_bind(helion.kernel(_pointwise, backend="cute"), (torch.empty(2048),))


@pytest.fixture
def bound(pointwise_bound):
    # Reuse the bind, but leave no active compiler environment between tests
    # that bind a different kernel, such as the paired-attention surface check.
    with cpu_target(), pointwise_bound.env:
        yield pointwise_bound


def _config(size=32, pipeline="default"):
    return helion.Config(block_sizes=[size], num_threads=32, **{KEY: pipeline})


def _member(generation, size, perf, pipeline="default", status="ok"):
    flat, config = generation.strict_config_pair(_config(size, pipeline))
    return PopulationMember(lambda: None, [perf], flat, config, status=status)


def _search(bound, sizes=(32, 64)):
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.settings = Settings(backend="cute", autotune_effort="full")
    search.config_spec = bound.config_spec
    search.config_gen = ConfigGeneration(bound.config_spec)
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.args = ()
    search.benchmark_provider = SimpleNamespace()
    search.final_selection = FinalSelectionConfig()
    search._compiler_refinement_keys = (KEY,)
    search._autotune_metrics = AutotuneMetrics()
    search._autotune_budget_start = None
    search._fallback_excluded_configs = ConfigSet()
    search._benchmarked_members = ConfigCache()
    search._pinned_finalist_members = ConfigCache()
    search._pinned_finalist_configs = ConfigSet()
    search._terminal_refinement_members = ConfigCache()
    search.log = Mock()
    search.population = [
        _member(search.config_gen, size, 1.0 + i * 0.1) for i, size in enumerate(sizes)
    ]
    search.best = search.population[0]
    for member in search.population:
        search._record_benchmarked_member(member)
    return search


def _measure(search, calls, perf=0.5, status="ok"):
    def benchmark(members, **kwargs):
        assert kwargs["raise_if_no_viable_config"] is False
        for member in members:
            calls.append(copy.deepcopy(member.config))
            member.perfs.append(perf)
            member.status = status
            search._record_benchmarked_member(member)
        return members

    return benchmark


def test_refine_discovered_structures_without_changing_layout(bound):
    generation = ConfigGeneration(bound.config_spec)
    parents = [_config(32), _config(32, "opt2"), _config(64)]
    original = copy.deepcopy(parents)
    candidates = compiler_refinement_candidates(generation, parents, (KEY,))
    assert len(candidates) == 2
    assert [candidate.config.block_sizes for candidate in candidates] == [[32], [64]]
    for candidate in candidates:
        assert candidate.config.cute_compiler_pipeline == "opt2"
        expected = dict(candidate.parent.config, **{KEY: "opt2"})
        assert candidate.config.config == expected
    assert parents == original


def test_joint_compiler_options_can_win_when_each_change_alone_loses(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables,
        second_option=EnumFragment((False, True)),
    ):
        search = _search(bound, (32,))
        search._compiler_refinement_keys = (KEY, "second_option")
        calls = []

        def benchmark(members, **kwargs):
            for member in members:
                config = member.config.config
                joint = config[KEY] == "opt2" and config["second_option"] is True
                calls.append((config[KEY], config["second_option"]))
                member.perfs.append(0.5 if joint else 2.0)
                member.status = "ok"
                search._record_benchmarked_member(member)

        with patch.object(search, "benchmark_population", side_effect=benchmark):
            winner = search.run_compiler_refinement(search.best)
    assert set(calls) == {("default", True), ("opt2", False), ("opt2", True)}
    assert winner.perf == 0.5
    assert winner.config.config[KEY] == "opt2"
    assert winner.config.config["second_option"] is True
    assert winner.config.block_sizes == [32]


def test_joint_refinement_keeps_pinned_options(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables,
        second_option=EnumFragment((False, True)),
    ):
        generation = ConfigGeneration(
            bound.config_spec, overrides={"second_option": True}
        )
        parent = _config()
        parent.config["second_option"] = True
        candidates = compiler_refinement_candidates(
            generation, [parent], (KEY, "second_option")
        )
    assert len(candidates) == 1
    assert candidates[0].config.config[KEY] == "opt2"
    assert candidates[0].config.config["second_option"] is True


def test_nearest_candidates_visit_each_parent_before_joint_changes(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables,
        second_option=EnumFragment((False, True)),
    ):
        generation = ConfigGeneration(bound.config_spec)
        parents = [_config(size, "opt2") for size in (32, 64)]
        for parent in parents:
            parent.config["second_option"] = True
        candidates = compiler_refinement_candidates(
            generation, parents, (KEY, "second_option")
        )
    distances = [
        int(candidate.config.config[KEY] != "opt2")
        + int(candidate.config.config["second_option"] is not True)
        for candidate in candidates
    ]
    assert distances == [1, 1, 1, 1, 2, 2]
    assert all(
        candidate.config.config[KEY] == "default" for candidate in candidates[:2]
    )
    assert [candidate.parent.block_sizes for candidate in candidates] == [
        [32],
        [64],
        [32],
        [64],
        [32],
        [64],
    ]


def test_real_paired_surface_crosses_retirement_and_v_lag_only_on_paired_parents():
    bound = paired_v_lag_bound()
    retirement = "cute_mma_dag_early_tmem_retirement"
    v_lag = "cute_mma_dag_v_lag"
    keys = bound.config_spec.search_surface.compiler_refinement_keys
    assert keys == (KEY, retirement, v_lag)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        parent = paired_v_lag_config(0)
        candidates = compiler_refinement_candidates(generation, [parent], keys)
        assert {
            (
                candidate.config.config[KEY],
                candidate.config.config[retirement],
                candidate.config.config[v_lag],
            )
            for candidate in candidates
        } == {
            (pipeline, retire, lag)
            for pipeline in ("default", "opt2")
            for retire in (False, True)
            for lag in range(4)
        } - {("default", False, 0)}
        selected = next(
            candidate
            for candidate in candidates
            if candidate.config[KEY] == "opt2"
            and candidate.config[retirement]
            and candidate.config[v_lag] == 2
        )
        parent.config["cute_mma_dag_pass_warps"] = "attention"
        candidates = compiler_refinement_candidates(generation, [parent], keys)
        assert len(candidates) == 1
        assert candidates[0].config.config[KEY] == "opt2"
        assert candidates[0].config.config[retirement] is False
        assert candidates[0].config.config[v_lag] == 0
        assert candidates[0].config.config["cute_mma_dag_pass_warps"] == "attention"
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(selected.config)
    assert "v_lag=2" in source and "early_tmem_retirement=True" in source


def test_large_option_product_keeps_single_coordinate_candidates(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables,
        second_option=EnumFragment(tuple(range(16))),
    ):
        generation = ConfigGeneration(bound.config_spec)
        candidates = compiler_refinement_candidates(
            generation, [_config()], (KEY, "second_option")
        )
    assert len(candidates) == 16
    assert all(
        (candidate.config.config[KEY] == "opt2")
        != (candidate.config.config["second_option"] != 0)
        for candidate in candidates
    )


def test_full_effort_constructor_defers_history_until_prepare(bound):
    with (
        patch.object(bound.settings, "autotune_effort", "full"),
        patch.dict(os.environ),
    ):
        os.environ.pop("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", None)
        search = DifferentialEvolutionSearch(bound, (torch.empty(2048),))
    assert search._terminal_refinement_members is None
    assert search._prepared is False
    assert "benchmark_provider" not in vars(search)


@pytest.mark.parametrize("initial, prepared", [(0, 8), (8, 1), (8, 8)])
def test_history_uses_preparation_policy(bound, monkeypatch, initial, prepared):
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", raising=False)
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", str(initial))
    with patch.object(bound.settings, "autotune_effort", "full"):
        search = DifferentialEvolutionSearch(bound, (torch.empty(2048),))
        assert search._terminal_refinement_members is None
        # Keep the real preparation/metadata lifecycle while replacing only
        # the measurement provider, whose constructor would run a GPU baseline.
        provider = Mock(spec=LocalBenchmarkProvider)
        search._benchmark_provider_cls = Mock(return_value=provider)
        monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", str(prepared))
        search._prepare()
        assert search._prepared and search.benchmark_provider is provider
        assert search._terminal_refinement_members == ({} if prepared > 1 else None)
        registry = search._terminal_refinement_members
        search._prepare()
        assert search._terminal_refinement_members is registry


def test_incumbent_precedes_optimistic_history(bound):
    search = _search(bound, (32,))
    for size, early, recent in (
        (64, 0.8, 1.7),
        (128, 0.7, 1.4),
        (256, 0.6, 1.5),
        (512, 0.5, 1.6),
    ):
        member = _member(search.config_gen, size, recent)
        member.perfs.insert(0, early)
        search._record_benchmarked_member(member)
    calls = []
    with patch.object(
        search, "benchmark_population", side_effect=_measure(search, calls)
    ):
        search.run_compiler_refinement(search.best)
    assert [c.block_sizes for c in calls] == [[32], [128], [256], [512]]
    assert search._autotune_metrics.compiler_refinement["new_trials"] == 4


@pytest.mark.parametrize("top_k", [0, 1])
def test_top_k_opt_out_skips_refinement(bound, monkeypatch, top_k):
    search = _search(bound)
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", str(top_k))
    with patch.object(search, "benchmark_population") as benchmark:
        assert search.run_compiler_refinement(search.best) is search.best
    benchmark.assert_not_called()
    assert search._autotune_metrics.compiler_refinement is None


@pytest.mark.parametrize(
    "mode", ["heuristics", "distributed", "custom", "multi", "top0", "top1", "rounds0"]
)
def test_disabled_constructor_does_not_retain_history(bound, monkeypatch, mode):
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", raising=False)
    args = (torch.empty(2048),)
    with ExitStack() as stack:
        stack.enter_context(patch.object(bound.settings, "autotune_effort", "full"))
        if mode == "heuristics":
            stack.enter_context(
                patch.object(bound.settings, "disable_autotuner_heuristics", True)
            )
        elif mode == "distributed":
            stack.enter_context(patch.object(bound.env, "process_group_name", "group"))
        elif mode == "custom":
            stack.enter_context(
                patch.object(bound.settings, "autotune_benchmark_fn", lambda *args: 1.0)
            )
        elif mode == "multi":
            args = _MultiShapeAutotuneArgs(
                cases=((bound, args),),
                aggregation="geomean",
                relative_to=None,
                cache_tag=None,
                workload_key=(),
            )
        elif mode == "rounds0":
            monkeypatch.setenv("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", "0")
        else:
            monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", mode[-1])
        search = DifferentialEvolutionSearch(bound, args)
        assert search._terminal_refinement_members is None
        assert search._prepared is False
        assert "benchmark_provider" not in vars(search)
        search._record_benchmarked_member(_member(search.config_gen, 32, 1.0))
        assert search._terminal_refinement_members is None


def test_budget_is_rechecked_after_each_trial(bound):
    search = _search(bound)
    calls = []
    with (
        patch.object(
            search, "benchmark_population", side_effect=_measure(search, calls)
        ),
        patch.object(
            search,
            "_autotune_budget_exceeded",
            side_effect=lambda: bool(calls),
        ),
    ):
        search.run_compiler_refinement(search.best)
    assert len(calls) == 1
    assert search._autotune_metrics.compiler_refinement["new_trials"] == 1
    assert [
        r["outcome"] for r in search._autotune_metrics.compiler_refinement["trials"]
    ] == ["ok", "budget_exhausted"]
    assert search._budget_reserve_released is False


def test_explicit_pin_and_restricted_domain_are_respected(bound):
    pinned = ConfigGeneration(bound.config_spec, overrides={KEY: "default"})
    assert compiler_refinement_candidates(pinned, [_config()], (KEY,)) == []
    generation = ConfigGeneration(bound.config_spec)
    index = generation._key_to_flat_indices[KEY][0][0]
    generation.flat_spec[index] = EnumFragment(
        choices=("default", "opt2"), search_choices=("default",)
    )
    assert compiler_refinement_candidates(generation, [_config()], (KEY,)) == []
    assert compiler_refinement_candidates(generation, [_config()], ()) == []


def test_scalar_types_and_signed_zero_remain_distinct_structures(bound):
    values = (0.0, -0.0, 1, 1.0)
    with patch.dict(
        bound.config_spec.user_defined_tunables, sentinel=EnumFragment(values)
    ):
        generation = ConfigGeneration(bound.config_spec)
        parents = []
        for value in values:
            config = _config()
            config.config["sentinel"] = value
            parents.append(config)
        candidates = compiler_refinement_candidates(generation, parents, (KEY,))
    assert len(candidates) == 4
    assert [type(candidate.config.config["sentinel"]) for candidate in candidates] == [
        float,
        float,
        int,
        float,
    ]
    assert [
        candidate.config.config["sentinel"].hex() for candidate in candidates[:2]
    ] == ["0x0.0p+0", "-0x0.0p+0"]
    assert not same_refinement_value({"opaque": object()}, {"opaque": object()})


def test_typed_alias_in_history_is_not_a_measured_trial(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables, sentinel=EnumFragment((0.0, -0.0))
    ):
        search = _search(bound, (32,))
        config = _config(pipeline="opt2")
        config.config["sentinel"] = -0.0
        flat, config = search.config_gen.strict_config_pair(config)
        known = PopulationMember(lambda: None, [0.75], flat, config, status="ok")
        search._record_benchmarked_member(known)
        calls = []
        with patch.object(
            search, "benchmark_population", side_effect=_measure(search, calls)
        ):
            search.run_compiler_refinement(search.best)
    positive_opt2 = [
        config
        for config in calls
        if config.cute_compiler_pipeline == "opt2"
        and config.config["sentinel"].hex() == "0x0.0p+0"
    ]
    assert len(positive_opt2) == 1
    assert all(
        row["outcome"] != "reused"
        for row in search._autotune_metrics.compiler_refinement["trials"]
    )


def test_refinement_rejects_normalization_that_changes_structure(bound):
    generation = ConfigGeneration(bound.config_spec)
    original = generation.strict_config_pair

    def normalize(config):
        flat, normalized = original(config)
        if normalized.cute_compiler_pipeline == "opt2":
            normalized.config["block_sizes"] = [64]
            flat = generation.flatten(normalized)
        return flat, normalized

    with patch.object(generation, "strict_config_pair", side_effect=normalize):
        assert compiler_refinement_candidates(generation, [_config()], (KEY,)) == []


def test_trial_bound_and_reuse_of_successes_and_failures(bound):
    search = _search(bound, (32, 64, 128, 256, 512, 1024))
    known = _member(search.config_gen, 32, 0.75, "opt2")
    failed = _member(search.config_gen, 64, float("inf"), "opt2", "error")
    for member in (known, failed):
        search._record_benchmarked_member(member)
    calls = []
    with patch.object(
        search, "benchmark_population", side_effect=_measure(search, calls)
    ):
        winner = search.run_compiler_refinement(search.best)
    assert [config.block_sizes for config in calls] == [[128], [256]]
    assert winner.perf == 0.5
    trace = search._autotune_metrics.compiler_refinement
    assert trace["new_trials"] == 2
    assert [row["outcome"] for row in trace["trials"]] == [
        "reused",
        "prior_failure",
        "ok",
        "ok",
    ]
    assert trace["completed"] is True
    assert search._autotune_metrics.to_dict()["compiler_refinement"] == trace


def test_single_option_refinement_visits_only_four_parent_structures(bound):
    search = _search(bound, (32, 64, 128, 256, 512, 1024))
    calls = []
    with patch.object(
        search, "benchmark_population", side_effect=_measure(search, calls)
    ):
        search.run_compiler_refinement(search.best)
    assert len(calls) == 4
    assert [config.block_sizes for config in calls] == [[32], [64], [128], [256]]


def test_multiple_compiler_coordinates_complete_each_parent(bound):
    with patch.dict(
        bound.config_spec.user_defined_tunables,
        second_option=EnumFragment((0, 1, 2, 3, 4)),
    ):
        search = _search(bound, (32, 64, 128))
        search._compiler_refinement_keys = (KEY, "second_option")
        calls = []
        with patch.object(
            search, "benchmark_population", side_effect=_measure(search, calls)
        ):
            search.run_compiler_refinement(search.best)
    assert len(calls) == 27
    assert [config.block_sizes for config in calls[:3]] == [[32], [64], [128]]
    for size in (32, 64, 128):
        assert {
            (config.config[KEY], config.config["second_option"])
            for config in calls
            if config.block_sizes == [size]
        } == {
            (pipeline, second)
            for pipeline in ("default", "opt2")
            for second in range(5)
        } - {("default", 0)}
    assert {
        row["outcome"] for row in search._autotune_metrics.compiler_refinement["trials"]
    } == {"ok"}


@pytest.mark.parametrize(
    "disabled", ["effort", "heuristics", "custom", "multi", "distributed"]
)
def test_unsupported_search_modes_do_not_refine(bound, disabled):
    search = _search(bound)
    if disabled == "effort":
        search.final_selection = None
    elif disabled == "heuristics":
        search.settings.disable_autotuner_heuristics = True
    elif disabled == "custom":
        search.settings.autotune_benchmark_fn = lambda *args: 1.0
    elif disabled == "multi":
        search.benchmark_provider = Mock(spec=MultiShapeBenchmarkProvider)
    else:
        search.kernel.env.process_group_name = "group"
    with patch.object(search, "benchmark_population") as benchmark:
        assert search.run_compiler_refinement(search.best) is search.best
    benchmark.assert_not_called()
    assert search._autotune_metrics.compiler_refinement is None


def test_budget_reserve_stays_closed(bound):
    search = _search(bound)
    search.settings.autotune_budget_seconds = 100
    search._autotune_budget_start = time.perf_counter() - 80
    with (
        patch.object(search, "_budget_reserve_seconds", return_value=25),
        patch.object(search, "_in_flight_hold_back_seconds", return_value=0),
        patch.object(search, "benchmark_population") as benchmark,
    ):
        assert search.run_compiler_refinement(search.best) is search.best
    benchmark.assert_not_called()
    assert search._budget_reserve_released is False
    assert search._autotune_metrics.compiler_refinement["budget_exhausted"] is True


def test_failed_trial_keeps_incumbent(bound):
    search = _search(bound)
    calls = []
    with patch.object(
        search,
        "benchmark_population",
        side_effect=_measure(search, calls, float("inf"), "accuracy_error"),
    ):
        assert search.run_compiler_refinement(search.best) is search.best
    assert len(calls) == 2


def test_unkillable_trial_stops_and_is_quarantined(bound):
    search = _search(bound)
    error = BenchmarkWorkerUnkillable("worker did not exit")
    error.fn_index = 0
    with (
        patch.object(search, "benchmark_population", side_effect=error) as benchmark,
        pytest.raises(BenchmarkWorkerUnkillable),
    ):
        search.run_compiler_refinement(search.best)
    assert benchmark.call_count == 1
    assert benchmark.call_args.args[0][0].config in search._fallback_excluded_configs
    assert search._autotune_metrics.compiler_refinement["completed"] is False


def test_cache_identity_includes_refinement_policy(bound):
    search = _search(bound)
    policy = search._final_rebenchmark_cache_policy()
    assert policy["compiler_refinement"]["version"] == 6
    assert policy["compiler_refinement"]["ordering"] == "distance_key_parent"
    assert policy["compiler_refinement"]["max_new_trials"] == 64
    assert policy["compiler_refinement"]["max_joint_product"] == 16
    with patch.object(
        search.config_spec.search_surface, "compiler_refinement_keys", ()
    ):
        old = search.config_spec.cache_fingerprint_hash()
    assert search.config_spec.cache_fingerprint_hash() != old
    assert BackendSearchSurface.compiler_refinement_keys == ()


# One axis covers the distinct enabled-phase exits and the disabled no-event path.
@pytest.mark.parametrize(
    "ending",
    [
        "success",
        "empty",
        "budget",
        "failed",
        "known",
        "quarantine",
        "limit",
        "invalid",
        "unkillable",
        "error",
        "discovery_error",
        "disabled",
    ],
)
def test_refinement_transcript_is_saved_once_on_every_exit(bound, tmp_path, ending):
    with ExitStack() as stack:
        if ending == "limit":
            stack.enter_context(
                patch.dict(
                    bound.config_spec.user_defined_tunables,
                    second_option=EnumFragment(tuple(range(16))),
                    third_option=EnumFragment(tuple(range(16))),
                )
            )
        search = _search(bound, (32, 64, 128))
        if ending == "limit":
            search._compiler_refinement_keys = (KEY, "second_option", "third_option")
        elif ending == "empty":
            search._compiler_refinement_keys = ("absent",)
        elif ending == "budget":
            stack.enter_context(
                patch.object(search, "_autotune_budget_exceeded", return_value=True)
            )
        elif ending == "known":
            search._record_benchmarked_member(
                _member(search.config_gen, 32, 0.75, "opt2")
            )
            search._record_benchmarked_member(
                _member(search.config_gen, 64, float("inf"), "opt2", "error")
            )
        elif ending == "quarantine":
            search._fallback_excluded_configs.add(
                _member(search.config_gen, 32, 0.75, "opt2").config
            )
        elif ending == "disabled":
            search.final_selection = None
        elif ending == "invalid":
            stack.enter_context(
                patch.object(search, "make_unbenchmarked", return_value=None)
            )
        calls = []
        effect = _measure(
            search,
            calls,
            float("inf") if ending == "failed" else 0.5,
            "accuracy_error" if ending == "failed" else "ok",
        )
        error = None
        if ending == "unkillable":
            error = BenchmarkWorkerUnkillable("worker did not exit")
            error.fn_index = 0
            effect = error
        elif ending == "error":
            error = RuntimeError("measurement failed")
            effect = error
        elif ending == "discovery_error":
            error = RuntimeError("candidate discovery failed")
            stack.enter_context(
                patch(
                    "helion.autotuner.base_search.compiler_refinement_candidates",
                    side_effect=error,
                )
            )
        stack.enter_context(
            patch.object(search, "benchmark_population", side_effect=effect)
        )
        search.settings.autotune_log = str(tmp_path / "run")
        search.settings.autotune_log_details = True
        search.log = AutotuningLogger(search.settings)
        with search.log.autotune_tracing("LFBOTreeSearch"):
            if error is not None:
                with pytest.raises(type(error), match=str(error)):
                    search.run_compiler_refinement(search.best)
            else:
                search.run_compiler_refinement(search.best)
        events = [
            json.loads(line)
            for line in (tmp_path / "run.trace.jsonl").read_text().splitlines()
        ]
        saved = [event for event in events if event["event"] == "compiler_refinement"]
        if ending == "disabled":
            assert saved == [] and search._autotune_metrics.compiler_refinement is None
            return
        assert len(saved) == 1
        record = saved[0]
        assert record["algorithm"] == "LFBOTreeSearch"
        assert record["transcript"] == json.loads(
            json.dumps(search._autotune_metrics.compiler_refinement)
        )
        transcript = record["transcript"]
        assert transcript["completed"] is (error is None)
        assert transcript["new_trials"] <= 64
        assert search._budget_reserve_released is False
        outcomes = {trial["outcome"] for trial in transcript["trials"]}
        if ending in {"empty", "discovery_error"}:
            assert transcript["trials"] == []
        elif ending == "budget":
            assert transcript["budget_exhausted"] and outcomes == {"budget_exhausted"}
        elif ending == "known":
            assert {"reused", "prior_failure"} <= outcomes
        elif ending == "quarantine":
            assert "quarantined" in outcomes
        elif ending == "limit":
            assert transcript["new_trials"] == 64 and "trial_limit" in outcomes
        elif ending == "unkillable":
            assert outcomes == {"unkillable_worker"}
        elif ending == "error":
            assert outcomes == {"benchmarking"}
        elif ending == "failed":
            assert outcomes == {"accuracy_error"}
        elif ending == "invalid":
            assert outcomes == {"invalid"}
        else:
            assert outcomes == {"ok"}
