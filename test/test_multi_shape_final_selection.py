from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_autotuner_final_selection import _make_search
from test.test_autotuner_final_selection import _member
from test.test_cute_grouped_fp32_output import _grouped_fp32_output
from test.test_cute_matmul_benchmark_policy import _CpuKernel
from test.test_cute_matmul_benchmark_policy import _provider

import helion
from helion.autotuner import benchmark_provider as provider_module
from helion.autotuner.base_search import _MultiShapeFinalSelectionMeasurer
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics


@pytest.fixture
def bound():
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        yield _cpu_bind(_grouped_fp32_output, args)


def _configs():
    return [
        helion.Config(block_sizes=[128, 16, 16]),
        helion.Config(block_sizes=[128, 32, 16]),
    ]


def _functions(bound, case):
    functions = [_CpuKernel(), _CpuKernel()]
    for index, function in enumerate(functions):
        bound.config_spec.backend.annotate_compiled_module(
            SimpleNamespace(kernel=function), f"case={case}; source={index}", "kernel"
        )
    return functions


def _multi(bound, first=(2.0, 6.0), second=(9.0, 7.0)):
    children = [_provider(bound), _provider(bound)]
    functions = [_functions(bound, case) for case in range(2)]
    for case, child in enumerate(children):
        child.kernel.settings = child.settings
        child.kernel.config_spec = child.config_spec
        child.args = (case,)
        child.kernel.compile_config.side_effect = (
            lambda config, allow_print=False, case=case: functions[case][
                int(config.block_sizes[1] == 32)
            ]
        )
    args = _MultiShapeAutotuneArgs(
        cases=tuple((child.kernel, tuple(child.args)) for child in children),
        aggregation="max",
        relative_to=None,
        cache_tag=None,
        workload_key=("cpu-source-retiming",),
    )
    with patch.object(provider_module, "LocalBenchmarkProvider", side_effect=children):
        provider = MultiShapeBenchmarkProvider(
            children[0].kernel,
            children[0].settings,
            bound.config_spec,
            args,
            AutotuningLogger(children[0].settings),
            AutotuneMetrics(),
        )
    children[0]._benchmark_function = Mock(side_effect=list(first))
    children[1]._benchmark_function = Mock(side_effect=list(second))
    results = provider.benchmark(_configs())
    return provider, children, functions, results


def test_real_source_aliases_are_measured_again_then_use_the_fresh_cache(bound):
    provider, children, _, initial = _multi(
        bound, first=(2.0, 6.0, 3.0, 8.0), second=(9.0, 7.0, 4.0, 6.0)
    )
    assert [result.perf for result in initial] == [9.0, 7.0]
    children[0]._invalid_effective_source_hashes.add("unrelated-rejection")
    timings = provider.rebenchmark(_configs(), [9.0, 7.0], desc="fresh")
    assert timings == [4.0, 8.0]
    assert [child._benchmark_function.call_count for child in children] == [4, 4]
    assert "unrelated-rejection" in children[0]._invalid_effective_source_hashes
    repeated = provider.benchmark(_configs())
    assert [result.per_shape for result in repeated] == [(3.0, 4.0), (8.0, 6.0)]
    assert all(
        result.per_shape_statuses == ("deduplicated", "deduplicated")
        for result in repeated
    )
    assert [child._benchmark_function.call_count for child in children] == [4, 4]


def test_successful_steady_reading_refreshes_aliases_without_reviving_rejection(bound):
    provider, children, functions, _ = _multi(bound)
    assert provider.record_steady_measurement(_configs()[0], (3.0, 4.0)) == 4.0
    assert provider.benchmark([_configs()[0]])[0].per_shape == (3.0, 4.0)
    child = children[0]
    function = functions[0][0]
    source_hash = bound.config_spec.backend.generated_source_hash(function)
    child.invalidate_effective_source_hash(source_hash)
    child.update_effective_source_measurement(_configs()[0], function, 1.0)
    assert source_hash not in child._effective_source_results
    with patch.object(provider_module.dist, "is_initialized", return_value=True):
        children[1].update_effective_source_measurement(
            _configs()[0], functions[1][0], 1.0
        )
    assert provider.benchmark([_configs()[0]])[0].perf == math.inf


class _CaseTimer:
    def __init__(self, case, model, events):
        self.case = case
        self.model = model
        self.events = events
        self.counts = [0, 0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.events.append(("exit", self.case))

    def warm(self, index, duration_ms):
        self.events.append(("warm", self.case, index, duration_ms))

    def measure(self, index, warmup_ms, rep_ms):
        self.events.append(("measure", self.case, index, warmup_ms, rep_ms))
        number = self.counts[index]
        self.counts[index] += 1
        return self.model[index][number]


def _steady_search(provider, first, second, *, rounds=4):
    members = [_member(16, 9.0), _member(32, 7.0)]
    for member, config in zip(members, _configs(), strict=True):
        member.config = config
        member.fn = provider.functions_for_config(config)[0]
    policy = FinalSelectionConfig(
        top_k=2,
        rounds=rounds,
        warmup_ms=20,
        rep_ms=50,
        thermal_warmup_ms=100,
        screen_rep_ms=0,
    )
    search = _make_search(members, policy=policy, pinned=[members[0]])
    search.benchmark_provider = provider
    search.args = provider.args
    search.config_spec = provider.config_spec
    events = []
    timers = [_CaseTimer(0, first, events), _CaseTimer(1, second, events)]
    measurer = _MultiShapeFinalSelectionMeasurer(provider, _configs(), timers, [0, 1])
    return search, members, policy, measurer, events


def test_multi_case_sustained_selection_changes_winner_and_aggregates_case_medians(
    bound,
):
    provider, children, _, _ = _multi(bound)
    search, members, _, measurer, events = _steady_search(
        provider,
        first=([1.0, 5.0, 1.0, 5.0], [4.0] * 4),
        second=([5.0, 1.0, 5.0, 1.0], [4.0] * 4),
    )
    with patch.object(search, "_final_selection_measurer", return_value=measurer):
        selected = search.final_rebenchmark_best(members[1])
    assert selected is members[0]
    assert selected.perf == 3.0  # max(median(case0), median(case1)), not median(max).
    measurement = provider.benchmark([_configs()[0]])[0]
    assert measurement.per_shape == (3.0, 3.0)
    assert measurement.perf == 3.0
    assert [child._benchmark_function.call_count for child in children] == [2, 2]
    measures = [event[1:3] for event in events if event[0] == "measure"]
    assert (
        measures
        == [
            (0, 1),
            (1, 1),
            (0, 0),
            (1, 0),
            (1, 0),
            (0, 0),
            (1, 1),
            (0, 1),
        ]
        * 2
    )
    assert [event for event in events if event[0] == "warm"] == [
        ("warm", 0, 1, 100),
        ("warm", 1, 1, 100),
    ]
    metrics = search._autotune_metrics.final_selection
    assert metrics["case_orders"] == [[0, 1], [1, 0], [0, 1], [1, 0]]
    assert metrics["objective_reduction"] == "aggregate of per-shape round medians"


@pytest.mark.parametrize("missing", [None, IsolatedBenchmarkFailure("timeout")])
def test_incomplete_case_cannot_win_or_reintroduce_its_old_source_reading(
    bound, missing
):
    provider, children, functions, _ = _multi(bound)
    search, members, _, measurer, events = _steady_search(
        provider,
        first=([1.0], [4.0] * 4),
        second=([missing], [4.0] * 4),
    )
    with patch.object(search, "_final_selection_measurer", return_value=measurer):
        selected = search.final_rebenchmark_best(members[1])
    assert selected is members[1]
    assert not provider.has_valid_measurement(_configs()[0])
    source_hash = provider.config_spec.backend.generated_source_hash(functions[1][0])
    assert source_hash not in children[1]._effective_source_results
    assert (source_hash in children[1]._invalid_effective_source_hashes) == (
        missing is not None
    )
    assert (
        len([event for event in events if event[0] == "measure" and event[2] == 0]) == 2
    )


@pytest.mark.parametrize("worker", [True, False])
def test_factory_uses_each_cases_validated_functions_and_argument_set(bound, worker):
    provider, children, functions, _ = _multi(bound)
    search, members, policy, _, _ = _steady_search(
        provider, first=([1.0] * 4, [4.0] * 4), second=([1.0] * 4, [4.0] * 4)
    )
    for child in children:
        child.private_argument_copies_affordable = Mock(return_value=worker)
        child.steady_window_benchmark_available = Mock(return_value=worker)
    target = (
        "helion.autotuner.base_search._WorkerFinalSelectionMeasurer"
        if worker
        else "helion.autotuner.base_search._InProcessFinalSelectionMeasurer"
    )
    with patch(target) as factory:
        measurer = search._final_selection_measurer(members, policy)
    assert isinstance(measurer, _MultiShapeFinalSelectionMeasurer)
    assert factory.call_count == 2
    for case, call in enumerate(factory.call_args_list):
        if worker:
            assert call.args[0] is children[case]
            assert call.args[1] == functions[case]
            assert call.kwargs["private_copies"] is True
        else:
            assert call.args[0] == functions[case]
            assert call.args[1] == children[case].args
        if worker or case == 0:
            children[case].private_argument_copies_affordable.assert_called_once_with(4)
        else:
            children[case].private_argument_copies_affordable.assert_not_called()


def test_all_unavailable_windows_keep_only_the_prior_incumbent_ledger(bound):
    provider, children, _, _ = _multi(bound)
    search, members, _, measurer, _ = _steady_search(
        provider, first=([None], [None]), second=([None], [None])
    )
    with patch.object(search, "_final_selection_measurer", return_value=measurer):
        selected = search.final_rebenchmark_best(members[1])
    assert selected is members[1]
    assert selected.perf == 7.0
    assert provider.has_valid_measurement(selected.config)
    assert not provider.has_valid_measurement(members[0].config)
    assert all(not child._effective_source_results for child in children)


def test_multi_case_final_selection_scales_the_budget_reserve_and_fit(bound):
    provider, _, _, _ = _multi(bound)
    search, members, policy, _, _ = _steady_search(
        provider, first=([1.0] * 4, [4.0] * 4), second=([1.0] * 4, [4.0] * 4)
    )
    single = _make_search(
        members, policy=policy, pinned=[members[0]], budget_seconds=10000
    )
    search.settings = single.settings
    assert (
        search._minimum_final_selection_seconds()
        == 2 * single._minimum_final_selection_seconds()
    )
    assert search._budget_reserve_seconds() == 2 * single._budget_reserve_seconds()
    search._autotune_budget_start = 0.0
    with (
        patch("helion.autotuner.base_search.time.perf_counter", return_value=100.0),
        patch.object(
            FinalSelectionConfig, "fit_to_seconds", return_value=policy
        ) as fit,
    ):
        assert search._final_selection_plan(2) is policy
    fit.assert_called_once_with(4950.0, 2)
