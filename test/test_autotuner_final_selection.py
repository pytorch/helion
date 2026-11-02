from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import os
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner import PatternSearch
from helion.autotuner import search_algorithms
import helion.autotuner.base_search as base_search_module
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.base_search import _InProcessFinalSelectionMeasurer
from helion.autotuner.base_search import _WorkerFinalSelectionMeasurer
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkSubprocessError
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorkerDied
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.de_surrogate_hybrid import DESurrogateHybrid
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.kernel_args import load_private_kernel_args
from helion.autotuner.kernel_args import release_private_kernel_args
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.search_space_logger import canonical_config_id
import helion.language as hl
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing_extensions import Self


class _FakeMeasurer:
    """Replays a per-finalist reading model and records every window taken.

    ``model(member, window_number)`` returns the reading (ms) for the
    finalist ``member`` at the ``window_number``-th steady window of the whole
    phase, or an ``IsolatedBenchmarkFailure``. Windows are recorded as
    ``(kind, member, warmup_ms, rep_ms)``; a window taken in a named L2
    regime (the second column) is answered by ``second_model(member,
    window_number)`` and recorded in ``second_windows`` as ``(regime,
    member, warmup_ms, rep_ms)`` as well. ``regime_label`` names the regime
    the measurer stands for; unset, the selection labels the phase from the
    environment as it does for a measurer that does not say.
    """

    def __init__(
        self,
        model: Callable[[PopulationMember, int], object],
        second_model: Callable[[PopulationMember, int], object] | None = None,
        regime_label: str | None = None,
    ) -> None:
        self.model = model
        self.second_model = second_model
        if regime_label is not None:
            self.regime_label = regime_label
        self.second_windows: list[tuple[str, PopulationMember, int, int]] = []
        self.finalists: list[PopulationMember] = []
        self.policies: list[FinalSelectionConfig] = []
        self.windows: list[tuple[str, PopulationMember, int, int]] = []
        self.entered = 0
        self.exited = 0

    def __call__(
        self, finalists: list[PopulationMember], policy: FinalSelectionConfig
    ) -> Self:
        # stands in for ``PopulationBasedSearch._final_selection_measurer``
        self.finalists = list(finalists)
        self.policies.append(policy)
        return self

    def __enter__(self) -> Self:
        self.entered += 1
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.exited += 1

    def warm(self, index: int, duration_ms: int) -> None:
        self.windows.append(("warm", self.finalists[index], duration_ms, 0))

    def measure(
        self,
        index: int,
        warmup_ms: int,
        rep_ms: int,
        l2_flush_mode: str | None = None,
    ) -> object:
        member = self.finalists[index]
        self.windows.append(("measure", member, warmup_ms, rep_ms))
        window_number = sum(1 for kind, *_ in self.windows if kind == "measure")
        if l2_flush_mode is not None:
            self.second_windows.append((l2_flush_mode, member, warmup_ms, rep_ms))
            assert self.second_model is not None, "no second column expected"
            return self.second_model(member, window_number)
        return self.model(member, window_number)

    def measured(self) -> list[PopulationMember]:
        return [window[1] for window in self.windows if window[0] == "measure"]


def _make_search(
    members: list[PopulationMember],
    *,
    policy: FinalSelectionConfig | None,
    pinned: list[PopulationMember] = (),  # type: ignore[assignment]
    budget_seconds: int | None = None,
    budget_elapsed: float = 0.0,
    process_group_name: str | None = None,
    cls: type[PopulationBasedSearch] = PopulationBasedSearch,
) -> PopulationBasedSearch:
    settings = Settings(
        autotune_log_level=logging.CRITICAL,
        autotune_budget_seconds=budget_seconds,
    )
    search = cls.__new__(cls)
    search.settings = settings
    search.log = AutotuningLogger(settings)
    search.args = ()
    search.kernel = SimpleNamespace(
        env=SimpleNamespace(process_group_name=process_group_name)
    )
    search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
    # no family knobs: the family quota admits nothing (``_family_search``
    # names some)
    search.config_spec = SimpleNamespace(final_selection_family_keys=())  # type: ignore[misc]
    search._autotune_metrics = AutotuneMetrics()
    search._autotune_budget_start = (
        None if budget_seconds is None else time.perf_counter() - budget_elapsed
    )
    search.final_selection = policy
    search.best_perf_so_far = min(member.perf for member in members)
    search._benchmarked_members = {member.config: member for member in members}
    search._pinned_finalist_configs = {member.config for member in pinned}
    search._pinned_finalist_members = {member.config: member for member in pinned}
    search._unkillable_configs = set()
    search.population = list(members)
    return search


# test/conftest.py keeps the suite's shortlist at two; the cold tie-break
# tests shortlist their three finalists
_SHORTLIST_ALL = {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "4"}


def _clear_regime_envs() -> None:
    """The default regimes (a per-call flushed search and selection, the
    ``auto`` target); call inside a ``patch.dict(os.environ, ...)``."""
    for name in (
        "HELION_AUTOTUNE_L2_FLUSH",
        "HELION_AUTOTUNE_FINAL_L2_FLUSH",
        "HELION_AUTOTUNE_TARGET_REGIME",
    ):
        os.environ.pop(name, None)


def _rows(search: PopulationBasedSearch) -> list[dict[str, object]]:
    metrics = search._autotune_metrics.final_selection
    assert metrics is not None
    return cast("list[dict[str, object]]", metrics["rows"])


def _member(block: int, perf: float) -> PopulationMember:
    return PopulationMember(
        lambda: None,
        [perf],
        (block,),
        helion.Config(block_sizes=[block]),
        status="ok",
    )


def _family_member(block: int, family: str, perf: float) -> PopulationMember:
    """A member whose config carries the family knob ``fam``."""
    return PopulationMember(
        lambda: None,
        [perf],
        (block,),
        helion.Config(block_sizes=[block], fam=family),
        status="ok",
    )


def _family_search(
    members: list[PopulationMember],
    *,
    policy: FinalSelectionConfig | None,
    pinned: list[PopulationMember] = (),  # type: ignore[assignment]
    keys: tuple[str, ...] = ("fam",),
    budget_seconds: int | None = None,
) -> PopulationBasedSearch:
    """A search whose config spec names ``keys`` as the family knobs."""
    search = _make_search(
        members, policy=policy, pinned=pinned, budget_seconds=budget_seconds
    )
    search.config_spec = SimpleNamespace(final_selection_family_keys=keys)  # type: ignore[misc]
    return search


def _mock_kernel(settings: Settings) -> SimpleNamespace:
    config_spec = SimpleNamespace(
        default_config=lambda: helion.Config(block_sizes=[1]),
        cute_flash_search_enabled=False,
        compiler_seed_timeout_retry_repetitions=None,
        create_config_generation=lambda **kwargs: SimpleNamespace(
            invalid_config_count=0
        ),
        backend=SimpleNamespace(
            autotune_config_is_viable=lambda _config_spec, _config: True,
            should_deduplicate_generated_sources=lambda config_spec: False,
            get_do_bench=lambda: None,
            classify_autotune_exception=lambda error: None,
            probe_long_autotune_kernels=lambda _config_spec: False,
            supports_precompile=lambda: True,
        ),
    )
    return SimpleNamespace(
        settings=settings,
        config_spec=config_spec,
        format_kernel_decorator=lambda config, s: "decorator",
        to_triton_code=lambda config: "code",
        maybe_log_repro=lambda log_func, args, config=None: None,
        supports_subprocess_benchmark=lambda: False,
        env=SimpleNamespace(process_group_name=None),
    )


class TestFinalSelection(TestCase):
    def test_median_of_rounds_ignores_a_bimodal_flip(self) -> None:
        # The search's single reading favours B (a lucky 0.95 ms); the steady
        # windows say A is 3% faster, except for one bimodal +8% flip on A.
        fast = _member(32, 1.00)
        lucky = _member(64, 0.95)
        truth = {id(fast): 1.00, id(lucky): 1.03}
        policy = FinalSelectionConfig(
            top_k=4, rounds=3, screen_rep_ms=0, thermal_warmup_ms=1000
        )
        search = _make_search([lucky, fast], policy=policy)

        def model(member: PopulationMember, window_number: int) -> float:
            reading = truth[id(member)]
            if member is fast and window_number == 3:
                reading *= 1.08
            return reading

        measurer = _FakeMeasurer(model)
        with patch.object(search, "_final_selection_measurer", new=measurer):
            selected = search.final_rebenchmark_best(lucky)

        self.assertIs(selected, fast)
        self.assertEqual(selected.perf, 1.00)
        self.assertEqual(lucky.perf, 1.03)
        # The phase settled the clocks on the incumbent before any window.
        self.assertEqual(measurer.windows[0], ("warm", lucky, 1000, 0))
        rounds = [window for window in measurer.windows if window[0] == "measure"]
        self.assertEqual(len(rounds), 6)
        # finalists are ordered by the search's numbers; rounds alternate
        self.assertEqual(measurer.measured(), [lucky, fast, fast, lucky, lucky, fast])
        self.assertTrue(
            all(window[2:] == (policy.warmup_ms, policy.rep_ms) for window in rounds)
        )
        self.assertEqual(measurer.entered, 1)
        self.assertEqual(measurer.exited, 1)
        self.assertEqual(measurer.policies, [policy])
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["finalists"], 2)
        rows = _rows(search)
        self.assertEqual(rows[0]["median_ms"], 1.00)
        self.assertEqual(rows[0]["rounds_ms"], [1.00, 1.08, 1.00])
        self.assertIsNone(rows[0]["disqualified"])

    def test_screen_cuts_shortlist_and_alternates_round_order(self) -> None:
        # Twelve candidates whose search readings are inverted relative to the
        # truth, under a clock that slows 0.1% per window. The screen keeps
        # the best four plus the best pinned seed and the incumbent (the
        # search's best, which screens last); alternating round order keeps
        # the drift from mis-ranking near-tied finalists.
        members = [_member(16 * (i + 1), 1.5 - 0.02 * i) for i in range(12)]
        truth = {id(member): 1.00 + 0.01 * i for i, member in enumerate(members)}
        seed = members[6]  # true 1.06 ms: outside the top four
        incumbent = members[11]  # the search's best, true 1.11 ms: the worst
        policy = FinalSelectionConfig(
            top_k=4, rounds=3, screen_rep_ms=200, screen_warmup_ms=50
        )
        search = _make_search(members, policy=policy, pinned=[seed])

        def model(member: PopulationMember, window_number: int) -> float:
            return truth[id(member)] * (1.0 + 0.001 * window_number)

        measurer = _FakeMeasurer(model)
        with (
            patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "12"}),
            patch.object(search, "_final_selection_measurer", new=measurer),
        ):
            selected = search.final_rebenchmark_best(incumbent)

        self.assertIs(selected, members[0])
        # finalists are [pinned seed, *remaining sorted by the search's perf]
        self.assertEqual(measurer.finalists[0], seed)
        self.assertEqual(
            measurer.finalists[1:], [m for m in members[::-1] if m is not seed]
        )
        windows = measurer.windows
        self.assertEqual(windows[0], ("warm", incumbent, 5000, 0))
        screen = windows[1:13]
        self.assertEqual([w[1] for w in screen], measurer.finalists)
        self.assertTrue(all(w[2:] == (50, 200) for w in screen))
        rounds = windows[13:]
        self.assertEqual(len(rounds), 3 * 6)
        order_0 = [w[1] for w in rounds[0:6]]
        order_1 = [w[1] for w in rounds[6:12]]
        order_2 = [w[1] for w in rounds[12:18]]
        self.assertEqual(order_1, order_0[::-1])
        self.assertEqual(order_2, order_0)
        # the four best by the screen (in screen rank) plus the rescued seed
        # and the incumbent
        self.assertEqual(
            order_0,
            [members[0], members[1], members[2], members[3], seed, incumbent],
        )
        for member in order_0:
            self.assertEqual(len(member.perfs), 2)
        for member in members[4:11]:
            if member is not seed:
                self.assertEqual(len(member.perfs), 1)
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["shortlisted"], 12)
        self.assertEqual(metrics["finalists"], 6)
        pinned_rows = [row for row in _rows(search) if row["pinned"]]
        self.assertEqual(len(pinned_rows), 1)

    def test_screen_runs_only_when_the_shortlist_exceeds_the_finalists(
        self,
    ) -> None:
        # The run and the budget estimate agree: a shortlist of top_k + 1
        # (the finalists the rounds take at the least) is not screened, and
        # the estimate counts up to top_k + 2 finalists (the best pinned seed
        # and the incumbent both survive outside the top_k).
        # explicit windows tell the screen's windows apart from the rounds'
        policy = FinalSelectionConfig(
            top_k=4,
            rounds=2,
            thermal_warmup_ms=0,
            warmup_ms=100,
            rep_ms=500,
            screen_warmup_ms=50,
            screen_rep_ms=200,
        )
        self.assertFalse(policy.screens(5))
        self.assertTrue(policy.screens(6))
        self.assertEqual(policy.finalists(5), 5)
        self.assertEqual(policy.finalists(6), 6)
        # plus the family quota's representatives (two by default)
        self.assertEqual(policy.finalists(7), 7)
        self.assertEqual(policy.finalists(9), 8)
        no_quota = dataclasses.replace(policy, family_finalists=0)
        self.assertEqual(no_quota.finalists(7), 6)
        self.assertEqual(policy.finalists(3), 3)
        # Without a screen the search's numbers rank the shortlist; the best
        # top_k go on, plus the best pinned seed when it is not among them.
        for count, screened, pinned_last in (
            (5, False, False),
            (5, False, True),
            (6, True, True),
        ):
            members = [_member(16 * (i + 1), 1.0 + 0.1 * i) for i in range(count)]
            pinned = [members[-1]] if pinned_last else []
            search = _make_search(members, policy=policy, pinned=pinned)
            measurer = _FakeMeasurer(lambda member, _n: member.perf)
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "12"}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
            ):
                search.final_rebenchmark_best(members[0])
            screen_windows = [
                w
                for w in measurer.windows
                if w[2:] == (policy.screen_warmup_ms, policy.screen_rep_ms)
            ]
            self.assertEqual(len(screen_windows), count if screened else 0)
            round_windows = [
                w
                for w in measurer.windows
                if w[2:] == (policy.warmup_ms, policy.rep_ms)
            ]
            self.assertEqual(len(round_windows), 2 * (5 if pinned_last else 4))
            survivors = {w[1].config for w in round_windows}
            self.assertEqual(survivors, {m.config for m in (*members[:4], *pinned)})

    def test_even_rounds_cancel_linear_drift(self) -> None:
        # Under a 0.1%/window slowdown the median of three alternating rounds
        # samples position k at window 2n-1-k: the last finalist is measured
        # earliest and a 0.6%-slower config wins. With four rounds the two
        # middle samples of every position sum to the same window, so the
        # true best wins. The default is therefore even.
        self.assertEqual(FinalSelectionConfig().rounds % 2, 0)
        count = 9
        members = [_member(16 * (i + 1), 1.0 + 0.01 * i) for i in range(count)]
        truth = {id(member): 1.01 for member in members}
        truth[id(members[0])] = 1.000  # the true best, first by the search
        truth[id(members[-1])] = 1.006  # 0.6% slower, last by the search

        def run(rounds: int) -> PopulationMember:
            policy = FinalSelectionConfig(
                top_k=count, rounds=rounds, screen_rep_ms=0, thermal_warmup_ms=0
            )
            fresh = [copy.deepcopy(member) for member in members]
            fresh_truth = {
                id(new): truth[id(old)] for new, old in zip(fresh, members, strict=True)
            }
            search = _make_search(fresh, policy=policy)
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "12"}
                ),
                patch.object(
                    search,
                    "_final_selection_measurer",
                    new=_FakeMeasurer(
                        lambda member, n: fresh_truth[id(member)] * (1.0 + 0.001 * n)
                    ),
                ),
            ):
                return search.final_rebenchmark_best(fresh[0])

        self.assertEqual(run(3).config, members[-1].config)
        self.assertEqual(run(4).config, members[0].config)

    def test_pinned_seed_within_tolerance_wins_the_tie(self) -> None:
        seed = _member(32, 1.2)
        rival = _member(64, 1.0)
        policy = FinalSelectionConfig(top_k=4, rounds=1, screen_rep_ms=0)
        search = _make_search([rival, seed], policy=policy, pinned=[seed])
        readings = {id(seed): 1.000, id(rival): 0.999}
        with (
            patch.object(
                search,
                "_final_selection_measurer",
                new=_FakeMeasurer(lambda member, _n: readings[id(member)]),
            ),
            patch.dict(
                os.environ,
                {"HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE": "0.01"},
            ),
        ):
            self.assertIs(search.final_rebenchmark_best(rival), seed)

    def test_finalist_without_readings_cannot_win(self) -> None:
        stale = _member(32, 0.5)  # search reading nobody could reproduce
        steady = _member(64, 1.0)
        policy = FinalSelectionConfig(top_k=4, rounds=2, screen_rep_ms=0)
        search = _make_search([stale, steady], policy=policy)

        def model(member: PopulationMember, _window_number: int) -> object:
            if member is stale:
                return IsolatedBenchmarkFailure("timeout")
            return 1.0

        with patch.object(
            search, "_final_selection_measurer", new=_FakeMeasurer(model)
        ):
            selected = search.final_rebenchmark_best(stale)

        self.assertIs(selected, steady)
        failed = [row for row in _rows(search) if row["disqualified"] is not None]
        self.assertEqual([row["disqualified"] for row in failed], ["timeout"])
        self.assertIsNone(failed[0]["median_ms"])

    def test_a_failed_or_missing_round_disqualifies_the_finalist(self) -> None:
        # One lucky reading must not win: a finalist that timed out in later
        # rounds, or produced no reading in one, is out even though the
        # readings it has are the lowest.
        lucky = _member(32, 0.6)
        flaky = _member(48, 0.7)
        steady = _member(64, 1.0)
        policy = FinalSelectionConfig(top_k=4, rounds=3, screen_rep_ms=0)
        search = _make_search([lucky, flaky, steady], policy=policy)
        seen: dict[int, int] = {}

        def model(member: PopulationMember, _window_number: int) -> object:
            seen[id(member)] = seen.get(id(member), 0) + 1
            if member is lucky:
                return (
                    0.5
                    if seen[id(member)] == 1
                    else IsolatedBenchmarkFailure("timeout")
                )
            if member is flaky:
                return None if seen[id(member)] == 2 else 0.4
            return 1.0

        with (
            patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "12"}),
            patch.object(search, "_final_selection_measurer", new=_FakeMeasurer(model)),
        ):
            selected = search.final_rebenchmark_best(lucky)

        self.assertIs(selected, steady)
        rows = {row["search_ms"]: row for row in _rows(search)}
        self.assertEqual(set(rows), {0.6, 0.7, 1.0})
        self.assertEqual(rows[0.6]["disqualified"], "timeout")
        self.assertEqual(rows[0.6]["rounds_ms"], [0.5])
        self.assertEqual(rows[0.7]["disqualified"], "incomplete")
        self.assertEqual(rows[0.7]["rounds_ms"], [0.4])
        self.assertIsNone(rows[0.7]["median_ms"])
        self.assertEqual(rows[1.0]["median_ms"], 1.0)
        self.assertEqual(rows[1.0]["rounds_ms"], [1.0] * 3)

    def test_disqualified_finalists_are_not_timed_again(self) -> None:
        # A finalist that timed out, errored or missed a reading cannot
        # compete any more, so the later rounds skip it (a hung finalist
        # would otherwise cost each of them a timeout); its rivals keep
        # their full rounds and the verdicts are unchanged.
        hung = _member(32, 0.6)
        flaky = _member(48, 0.7)
        steady = _member(64, 1.0)
        policy = FinalSelectionConfig(top_k=4, rounds=4, screen_rep_ms=0)
        search = _make_search([hung, flaky, steady], policy=policy)
        windows: dict[int, int] = {}

        def model(member: PopulationMember, _window_number: int) -> object:
            windows[id(member)] = windows.get(id(member), 0) + 1
            if member is hung:
                return IsolatedBenchmarkFailure("timeout")
            if member is flaky:
                return None if windows[id(member)] == 2 else 0.4
            return 1.0

        with (
            patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "12"}),
            patch.object(search, "_final_selection_measurer", new=_FakeMeasurer(model)),
        ):
            selected = search.final_rebenchmark_best(hung)

        self.assertIs(selected, steady)
        self.assertEqual(
            (windows[id(hung)], windows[id(flaky)], windows[id(steady)]), (1, 2, 4)
        )
        rows = {row["search_ms"]: row for row in _rows(search)}
        self.assertEqual(rows[0.6]["disqualified"], "timeout")
        self.assertEqual(rows[0.6]["rounds_ms"], [])
        self.assertEqual(rows[0.7]["disqualified"], "incomplete")
        self.assertEqual(rows[0.7]["rounds_ms"], [0.4])
        self.assertIsNone(rows[1.0]["disqualified"])
        self.assertEqual(rows[1.0]["rounds_ms"], [1.0] * 4)

    def test_no_readings_at_all_keeps_the_search_best(self) -> None:
        best = _member(32, 0.9)
        other = _member(64, 1.0)
        policy = FinalSelectionConfig(top_k=4, rounds=2, screen_rep_ms=0)
        search = _make_search([best, other], policy=policy)
        with patch.object(
            search,
            "_final_selection_measurer",
            new=_FakeMeasurer(lambda _member, _n: None),
        ):
            self.assertIs(search.final_rebenchmark_best(best), best)

    def test_incumbent_is_always_a_finalist(self) -> None:
        # A later phase re-timed the incumbent above the shortlist's search
        # numbers; it still defends its title in the shootout: shortlisted,
        # timed in every round whether the shortlist is screened (and it
        # screens last) or ranked by those numbers, and it wins when the
        # steady windows side with it.
        incumbent = _member(16, 2.0)
        cases = [
            # the default policy: eight shortlisted plus the incumbent is not
            # more than top_k + 1, so there is no screen
            (
                FinalSelectionConfig(),
                "8",
                [_member(32 * (i + 1), 1.0 + 0.01 * i) for i in range(8)],
                False,
            ),
            (
                FinalSelectionConfig(top_k=1, rounds=2, screen_rep_ms=0),
                "2",
                [_member(32, 1.0), _member(64, 1.1), _member(96, 1.2)],
                False,
            ),
            (
                # a screen window shorter than the rounds' so the count below
                # sees the rounds alone
                FinalSelectionConfig(top_k=1, rounds=2, screen_rep_ms=200),
                "2",
                [_member(32, 1.0), _member(64, 1.1), _member(96, 1.2)],
                True,
            ),
        ]
        for policy, top_k, others, screened in cases:
            search = _make_search([*others, incumbent], policy=policy)
            measurer = _FakeMeasurer(
                lambda member, _n: 0.5 if member is incumbent else 1.0
            )
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": top_k}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
            ):
                selected = search.final_rebenchmark_best(incumbent)
            shortlist = [*others[: int(top_k)], incumbent]
            self.assertEqual(measurer.finalists, shortlist)
            self.assertEqual(policy.screens(len(shortlist)), screened, policy)
            rounds_of_incumbent = [
                window
                for window in measurer.windows
                if window[0] == "measure"
                and window[1] is incumbent
                and window[3] == policy.rep_ms
            ]
            self.assertEqual(len(rounds_of_incumbent), policy.rounds, policy)
            self.assertIs(selected, incumbent, policy)
            self.assertEqual(incumbent.perf, 0.5)
            incumbent.perfs[:] = [2.0]

    def test_distributed_ranks_follow_rank_zero(self) -> None:
        # Screen readings are rank-local; rank 0's survivors (with the
        # incumbent, ``members[0]``), their order, each round's skip list and
        # the final timings are broadcast so every rank takes the same
        # windows together and applies the same numbers.
        count = 6
        members = [_member(16 * (i + 1), 1.0 + 0.1 * i) for i in range(count)]
        rank0_truth = {id(m): 1.0 + 0.1 * i for i, m in enumerate(members)}
        rank0_truth[id(members[4])] = 0.90
        rank0_truth[id(members[5])] = 0.91
        rank1_truth = {id(m): 1.0 + 0.1 * i for i, m in enumerate(members)}
        rank1_truth[id(members[0])] = 0.80
        rank1_truth[id(members[1])] = 0.81
        # explicit windows so the round windows below are told apart from
        # the screen's
        policy = FinalSelectionConfig(
            top_k=2,
            rounds=2,
            thermal_warmup_ms=0,
            warmup_ms=100,
            rep_ms=500,
            screen_warmup_ms=50,
            screen_rep_ms=200,
        )
        broadcasts: list[object] = []

        def rank0_sync(value: object, process_group_name: str | None = None) -> object:
            self.assertEqual(process_group_name, "autotune")
            broadcasts.append(copy.deepcopy(value))
            return value

        def rank1_sync(value: object, process_group_name: str | None = None) -> object:
            self.assertEqual(process_group_name, "autotune")
            return broadcasts.pop(0)

        def run(
            truth: dict[int, float], sync: Callable[..., object]
        ) -> tuple[PopulationMember, _FakeMeasurer, list[PopulationMember]]:
            fresh = [copy.deepcopy(member) for member in members]
            fresh_truth = {
                id(new): truth[id(old)] for new, old in zip(fresh, members, strict=True)
            }
            search = _make_search(fresh, policy=policy, process_group_name="autotune")
            measurer = _FakeMeasurer(lambda member, _n: fresh_truth[id(member)])
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "12"}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
                patch.object(base_search_module, "sync_object", side_effect=sync),
            ):
                return search.final_rebenchmark_best(fresh[0]), measurer, fresh

        selected0, measurer0, fresh0 = run(rank0_truth, rank0_sync)
        # plan, survivors, one skip list per round, results
        self.assertEqual(len(broadcasts), 3 + policy.rounds)
        selected1, measurer1, fresh1 = run(rank1_truth, rank1_sync)
        self.assertEqual(broadcasts, [])

        self.assertEqual(selected0.config, members[4].config)
        self.assertEqual(selected1.config, selected0.config)
        rounds0 = [w[1].config for w in measurer0.windows if w[2:] == (100, 500)]
        rounds1 = [w[1].config for w in measurer1.windows if w[2:] == (100, 500)]
        self.assertEqual(
            rounds0,
            [
                members[4].config,
                members[5].config,
                members[0].config,
                members[0].config,
                members[5].config,
                members[4].config,
            ],
        )
        self.assertEqual(rounds1, rounds0)
        # rank 1 applies rank 0's medians to its own members
        self.assertEqual(fresh1[4].perf, fresh0[4].perf)
        self.assertEqual(fresh1[5].perf, fresh0[5].perf)
        self.assertEqual(fresh1[4].perf, 0.90)

    def test_budget_sizes_the_phase_then_skips_it(self) -> None:
        noisy = _member(32, 0.9)
        stable = _member(64, 1.0)
        readings = {id(noisy): 1.05, id(stable): 1.00}
        policy = FinalSelectionConfig(top_k=4)

        # 15 s left of a 100 s budget: no room for four rounds with a thermal
        # warmup, so the plan shrinks to two rounds without one.
        search = _make_search(
            [noisy, stable], policy=policy, budget_seconds=100, budget_elapsed=85.0
        )
        measurer = _FakeMeasurer(lambda member, _n: readings[id(member)])
        with patch.object(search, "_final_selection_measurer", new=measurer):
            self.assertIs(search.final_rebenchmark_best(noisy), stable)
        self.assertEqual(
            measurer.policies,
            [
                dataclasses.replace(
                    policy,
                    rounds=2,
                    top_k=2,
                    thermal_warmup_ms=0,
                    screen_rep_ms=0,
                    tiebreak_rounds=0,
                    family_finalists=0,
                )
            ],
        )
        self.assertEqual([w[0] for w in measurer.windows], ["measure"] * 4)
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(cast("dict[str, object]", metrics["policy"])["rounds"], 2)

        # The budget is spent: not even the smallest plan fits, so the phase
        # is skipped and the search's best is returned untouched.
        search = _make_search(
            [noisy, stable], policy=policy, budget_seconds=1, budget_elapsed=2.0
        )
        measurer = _FakeMeasurer(lambda member, _n: readings[id(member)])
        perfs_before = list(noisy.perfs)
        with patch.object(search, "_final_selection_measurer", new=measurer):
            self.assertIs(search.final_rebenchmark_best(noisy), noisy)
        self.assertEqual(measurer.windows, [])
        self.assertIsNone(search._autotune_metrics.final_selection)
        self.assertEqual(noisy.perfs, perfs_before)

        # Without a budget the profile's policy runs as is.
        search = _make_search([noisy, stable], policy=policy)
        measurer = _FakeMeasurer(lambda member, _n: readings[id(member)])
        with patch.object(search, "_final_selection_measurer", new=measurer):
            search.final_rebenchmark_best(noisy)
        self.assertEqual(measurer.policies, [policy])

        legacy = _make_search(
            [noisy, stable], policy=None, budget_seconds=1, budget_elapsed=2.0
        )
        with patch.object(legacy, "rebenchmark") as rebenchmark:
            self.assertIs(legacy.final_rebenchmark_best(noisy), noisy)
        rebenchmark.assert_not_called()

    def test_budget_counts_the_second_column_only_where_it_can_run(self) -> None:
        """A per-call flushed selection (the default) takes no second
        column, so the reserve the search keeps for the phase and the plan
        the phase is sized to leave the column out; a warm selection
        (HELION_AUTOTUNE_L2_FLUSH=replay) or an explicit target regime
        reserves it for every finalist, its upper bound. The cache policy
        keeps the profile's fields in either regime."""
        members = [_member(32, 1.0), _member(64, 1.1)]
        # the spec of these searches names no family knob: the sizing
        # policy drops the family quota as well (see
        # ``test_budget_counts_the_family_quota_only_with_family_keys``)
        policy = FinalSelectionConfig(family_finalists=0)
        one_column = dataclasses.replace(policy, tiebreak_rounds=0)
        self.assertGreater(policy.estimated_seconds(4), one_column.estimated_seconds(4))
        search = _make_search(
            members, policy=policy, pinned=members[:1], budget_seconds=1000
        )
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}):
            _clear_regime_envs()
            self.assertEqual(search._final_selection_sizing_policy(), one_column)
            self.assertAlmostEqual(
                search._budget_reserve_seconds(), one_column.estimated_seconds(4)
            )
            # the smallest plan (4 finalists x 2 rounds) has no column in
            # either regime
            self.assertEqual(search._minimum_final_selection_seconds(), 17.0)
            default_key = search._final_rebenchmark_cache_policy()
            for env in (
                {"HELION_AUTOTUNE_L2_FLUSH": "replay"},
                {"HELION_AUTOTUNE_L2_FLUSH": "0"},
                {"HELION_AUTOTUNE_FINAL_L2_FLUSH": "replay"},
                {"HELION_AUTOTUNE_TARGET_REGIME": "replay"},
            ):
                with patch.dict(os.environ, env):
                    self.assertEqual(search._final_selection_sizing_policy(), policy)
                    self.assertAlmostEqual(
                        search._budget_reserve_seconds(), policy.estimated_seconds(4)
                    )
                    self.assertEqual(search._minimum_final_selection_seconds(), 17.0)
                    self.assertEqual(
                        search._final_rebenchmark_cache_policy(), default_key
                    )
            with patch.dict(
                os.environ,
                {
                    "HELION_AUTOTUNE_L2_FLUSH": "replay",
                    "HELION_AUTOTUNE_TARGET_REGIME": "replay",
                },
            ):
                # the target is the ranking regime already: one column
                self.assertEqual(search._final_selection_sizing_policy(), one_column)

        # The phase is sized to the time left by the same policy: with room
        # for the rounds but not the column, the default regime runs the
        # profile's policy as is (its column cannot run) where a replay
        # selection sheds the column.
        noisy = _member(32, 0.9)
        stable = _member(64, 1.0)
        readings = {id(noisy): 1.05, id(stable): 1.00}
        policy = FinalSelectionConfig(top_k=4, family_finalists=0)
        fits_one_column = dataclasses.replace(
            policy, screen_rep_ms=0, tiebreak_rounds=0
        )
        remaining = (
            fits_one_column.estimated_seconds(2) + policy.estimated_seconds(2)
        ) / 2
        for regime, expected, sized in (
            ("call", policy, False),
            ("replay", fits_one_column, True),
        ):
            search = _make_search(
                [noisy, stable],
                policy=policy,
                budget_seconds=1000,
                budget_elapsed=1000.0 - remaining,
            )
            measurer = _FakeMeasurer(
                lambda member, _n: readings[id(member)],
                lambda member, _n: readings[id(member)],
                regime_label=regime,
            )
            with (
                patch.dict(
                    os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": regime}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
                self.assertLogs(search.log._logger, level=logging.INFO) as logs,
            ):
                os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
                os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
                search.log.level = logging.INFO
                self.assertIs(search.final_rebenchmark_best(noisy), stable, regime)
            self.assertEqual(measurer.policies, [expected], regime)
            sizing_lines = [
                line for line in logs.output if "Final selection sized to" in line
            ]
            self.assertEqual(len(sizing_lines), 1 if sized else 0, regime)
            if sized:
                self.assertIn("one regime", sizing_lines[0])
            self.assertEqual(measurer.second_windows, [], regime)

    def test_fit_to_seconds_sheds_cost_in_order(self) -> None:
        policy = FinalSelectionConfig()
        ladder = policy.budget_ladder()
        self.assertEqual(ladder[0], policy)
        self.assertEqual(ladder[1], dataclasses.replace(policy, screen_rep_ms=0))
        self.assertEqual(
            ladder[2],
            dataclasses.replace(policy, screen_rep_ms=0, tiebreak_rounds=0),
        )
        self.assertEqual(
            ladder[3],
            dataclasses.replace(
                policy, screen_rep_ms=0, tiebreak_rounds=0, family_finalists=0
            ),
        )
        self.assertEqual(
            ladder[4],
            dataclasses.replace(
                policy,
                screen_rep_ms=0,
                tiebreak_rounds=0,
                family_finalists=0,
                rounds=2,
            ),
        )
        self.assertEqual(
            ladder[-1],
            dataclasses.replace(
                policy,
                screen_rep_ms=0,
                tiebreak_rounds=0,
                family_finalists=0,
                rounds=2,
                top_k=2,
                thermal_warmup_ms=0,
            ),
        )
        estimates = [plan.estimated_seconds(33) for plan in ladder]
        self.assertEqual(estimates, sorted(estimates, reverse=True))
        self.assertTrue(all(plan.rounds % 2 == 0 for plan in ladder))
        self.assertIs(policy.fit_to_seconds(1e9, 33), policy)
        self.assertEqual(policy.fit_to_seconds(estimates[1], 33), ladder[1])
        self.assertEqual(policy.fit_to_seconds(estimates[-1], 33), ladder[-1])
        self.assertIsNone(policy.fit_to_seconds(estimates[-1] - 0.01, 33))
        # an odd override keeps its round count rather than dropping to one
        odd = FinalSelectionConfig(rounds=3)
        self.assertEqual({plan.rounds for plan in odd.budget_ladder()}, {3})

    def test_search_phase_reserves_budget_for_the_final_selection(self) -> None:
        members = [_member(32, 1.0), _member(64, 1.1)]
        policy = FinalSelectionConfig()
        search = _make_search(
            members, policy=policy, pinned=members[:1], budget_seconds=100
        )
        # test/conftest.py shrinks the retained shortlist to 2; the default
        # shortlist of 8 + 1 pinned seed + the incumbent (10 finalists x 4
        # rounds) estimates ~86 s with the screen (71 s without) in the
        # default per-call flushed regime, capped at a quarter of the budget.
        one_column = dataclasses.replace(policy, tiebreak_rounds=0)
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "8"}):
            _clear_regime_envs()
            self.assertEqual(search._final_selection_shortlist_estimate(), 10)
            self.assertEqual(search._budget_reserve_seconds(), 25.0)
            self.assertGreater(one_column.estimated_seconds(10), 25.0)
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}):
            _clear_regime_envs()
            # 2 + 1 pinned + the incumbent shortlisted: the estimate (~35 s)
            # is below the cap of a 200 s budget
            self.assertEqual(search._final_selection_shortlist_estimate(), 4)
            roomy = _make_search(
                members, policy=policy, pinned=members[:1], budget_seconds=200
            )
            self.assertAlmostEqual(
                roomy._budget_reserve_seconds(), one_column.estimated_seconds(4)
            )
            self.assertLess(roomy._budget_reserve_seconds(), 50.0)
        search._budget_reserve_seconds = lambda: 25.0  # type: ignore[method-assign]

        search._autotune_budget_start = time.perf_counter() - 70.0
        self.assertFalse(search._autotune_budget_exceeded())
        search._autotune_budget_start = time.perf_counter() - 80.0
        self.assertTrue(search._autotune_budget_exceeded())
        # Once the search hands over, the final selection may use the reserve.
        search._budget_reserve_released = True
        self.assertFalse(search._autotune_budget_exceeded())
        search._autotune_budget_start = time.perf_counter() - 101.0
        self.assertTrue(search._autotune_budget_exceeded())

        legacy = _make_search(members, policy=None, budget_seconds=100)
        self.assertEqual(legacy._budget_reserve_seconds(), 0.0)
        legacy._autotune_budget_start = time.perf_counter() - 80.0
        self.assertFalse(legacy._autotune_budget_exceeded())

    def test_search_phase_also_reserves_the_generation_in_flight(self) -> None:
        # A budget check only stops new compiles; the generation in flight
        # finishes its benchmark pass. The search phase therefore stops one
        # generation's wall time (the longest seen between two of its
        # generation-boundary checks) before the final selection's reserve.
        members = [_member(32, 1.0), _member(64, 1.1)]
        search = _make_search(
            members, policy=FinalSelectionConfig(), budget_seconds=200
        )
        search.log = Mock()
        search._budget_reserve_seconds = lambda: 25.0  # type: ignore[method-assign]
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False, final_selection_family_keys=()
        )
        self.assertEqual(search._longest_generation_seconds, 0.0)

        generations: list[int] = []
        for generation in search._budgeted_range(3):
            generations.append(generation)
            time.sleep(0.02)
        self.assertEqual(generations, [0, 1, 2])
        self.assertGreaterEqual(search._longest_generation_seconds, 0.02)
        # a shorter generation does not shrink the allowance
        search._longest_generation_seconds = 1.0
        self.assertFalse(search._autotune_budget_exceeded_across_ranks())
        self.assertEqual(search._longest_generation_seconds, 1.0)

        search._longest_generation_seconds = 20.0
        search._autotune_budget_start = time.perf_counter() - 150.0
        self.assertFalse(search._autotune_budget_exceeded())
        search._autotune_budget_start = time.perf_counter() - 156.0
        self.assertTrue(search._autotune_budget_exceeded())
        message = str(search.log.call_args_list[-1].args[0])
        self.assertIn("25s reserved for the final selection", message)
        self.assertIn("20s for the generation in flight", message)
        # the final selection itself is not held back by it, and its checks
        # are not generations
        search._budget_reserve_released = True
        search._autotune_budget_start = time.perf_counter() - 199.0
        self.assertFalse(search._autotune_budget_exceeded_across_ranks())
        self.assertEqual(search._longest_generation_seconds, 20.0)
        # nor is a search without a final selection
        legacy = _make_search(members, policy=None, budget_seconds=100)
        legacy._longest_generation_seconds = 30.0
        legacy._autotune_budget_start = time.perf_counter() - 80.0
        self.assertFalse(legacy._autotune_budget_exceeded())

    def test_in_flight_hold_back_shares_the_reserve_cap(self) -> None:
        # The reserve and the hold-back for the generation in flight together
        # take at most the reserve's quarter of the budget: an 1800 s budget
        # with 400 s generations and a 52 s reserve stops the search at
        # 1800 - 52 - 398 = 1350 s rather than 1348 s, and the selection sizes
        # itself to what is left (a 400 s generation leaves it 52 s at the
        # least, well above its 14 s smallest plan, so the cap binds).
        members = [_member(32, 1.0), _member(64, 1.1)]
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}):
            search = _make_search(
                members, policy=FinalSelectionConfig(), budget_seconds=1800
            )
            search.log = Mock()
            self.assertEqual(search._minimum_final_selection_seconds(), 14.0)
            search._budget_reserve_seconds = lambda: 52.0  # type: ignore[method-assign]
            search._longest_generation_seconds = 400.0
            self.assertEqual(search._in_flight_hold_back_seconds(1800, 52.0), 398.0)
            search._autotune_budget_start = time.perf_counter() - 1349.0
            self.assertFalse(search._autotune_budget_exceeded())
            search._autotune_budget_start = time.perf_counter() - 1351.0
            self.assertTrue(search._autotune_budget_exceeded())
            message = str(search.log.call_args_list[-1].args[0])
            self.assertIn("52s reserved for the final selection", message)
            self.assertIn("398s for the generation in flight", message)
            # a reserve at the cap leaves the hold-back nothing when the
            # generation in flight cannot run the selection out of its
            # smallest plan
            search._budget_reserve_seconds = lambda: 450.0  # type: ignore[method-assign]
            search._longest_generation_seconds = 118.0
            self.assertEqual(search._in_flight_hold_back_seconds(1800, 450.0), 0.0)
            search._autotune_budget_start = time.perf_counter() - 1349.0
            self.assertFalse(search._autotune_budget_exceeded())
            search._autotune_budget_start = time.perf_counter() - 1351.0
            # The cutoff line is logged once per phase; a new scenario here
            # stands for a new search.
            search._budget_exceeded_logged = ()
            self.assertTrue(search._autotune_budget_exceeded())
            self.assertIn(
                "0s for the generation in flight",
                str(search.log.call_args_list[-1].args[0]),
            )
            # a generation that fits under the cap is held back in full
            search._budget_reserve_seconds = lambda: 52.0  # type: ignore[method-assign]
            self.assertEqual(search._in_flight_hold_back_seconds(1800, 52.0), 118.0)
            search._autotune_budget_start = time.perf_counter() - 1629.0
            self.assertFalse(search._autotune_budget_exceeded())
            search._autotune_budget_start = time.perf_counter() - 1631.0
            self.assertTrue(search._autotune_budget_exceeded())

    def test_hold_back_keeps_the_smallest_plan_on_a_short_budget(self) -> None:
        # A 300 s budget with 118 s generations: the cap leaves the hold-back
        # 9 s next to a 66 s reserve and nothing next to a 75 s reserve at
        # the cap, so the generation in flight would run the selection out
        # of time (or the run past the budget). The hold-back then grows so
        # the generation ends with the selection's smallest plan (14 s here)
        # still reserved: the search stops at 300 - 118 - 14 = 168 s with
        # either reserve.
        members = [_member(32, 1.0), _member(64, 1.1)]
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}):
            search = _make_search(
                members, policy=FinalSelectionConfig(), budget_seconds=300
            )
            search.log = Mock()
            self.assertEqual(search._minimum_final_selection_seconds(), 14.0)
            search._longest_generation_seconds = 118.0
            for reserve, hold_back in ((66.0, 66.0), (75.0, 57.0)):
                search._budget_reserve_seconds = (  # type: ignore[method-assign]
                    lambda reserve=reserve: reserve
                )
                self.assertEqual(
                    search._in_flight_hold_back_seconds(300, reserve), hold_back
                )
                search._autotune_budget_start = time.perf_counter() - 167.0
                self.assertFalse(search._autotune_budget_exceeded())
                search._autotune_budget_start = time.perf_counter() - 169.0
                # The cutoff line is logged once per phase; each scenario
                # stands for a new search.
                search._budget_exceeded_logged = ()
                self.assertTrue(search._autotune_budget_exceeded())
                message = str(search.log.call_args_list[-1].args[0])
                self.assertIn(
                    f"{reserve:.0f}s reserved for the final selection", message
                )
                self.assertIn(f"{hold_back:.0f}s for the generation in flight", message)
            # a generation the reserve absorbs with the smallest plan to spare
            # is held back only within the cap: 60 s generations next to the
            # 66 s reserve stop the search at 300 - 66 - 9 = 225 s
            search._budget_reserve_seconds = lambda: 66.0  # type: ignore[method-assign]
            search._longest_generation_seconds = 60.0
            self.assertEqual(search._in_flight_hold_back_seconds(300, 66.0), 9.0)
            search._autotune_budget_start = time.perf_counter() - 224.0
            self.assertFalse(search._autotune_budget_exceeded())
            search._autotune_budget_start = time.perf_counter() - 226.0
            self.assertTrue(search._autotune_budget_exceeded())
            # without a final selection nothing is held back
            legacy = _make_search(members, policy=None, budget_seconds=300)
            self.assertEqual(legacy._minimum_final_selection_seconds(), 0.0)
            legacy._longest_generation_seconds = 118.0
            legacy._autotune_budget_start = time.perf_counter() - 299.0
            self.assertFalse(legacy._autotune_budget_exceeded())

    def test_reserve_is_zero_without_a_budget_or_with_a_custom_timer(self) -> None:
        members = [_member(32, 1.0), _member(64, 1.1)]
        search = _make_search(members, policy=FinalSelectionConfig())
        self.assertEqual(search._budget_reserve_seconds(), 0.0)
        search = _make_search(
            members, policy=FinalSelectionConfig(), budget_seconds=100
        )
        search.settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_budget_seconds=100,
            autotune_benchmark_fn=lambda fns, repeat: [1.0] * len(fns),
        )
        self.assertFalse(search._final_selection_enabled())
        self.assertEqual(search._budget_reserve_seconds(), 0.0)

    def test_effort_profiles_enable_the_selection_for_full_only(self) -> None:
        self.assertIsNotNone(get_effort_profile("full").final_selection)
        self.assertIsNone(get_effort_profile("quick").final_selection)
        self.assertIsNone(get_effort_profile("none").final_selection)
        full = get_effort_profile("full")
        rounds_env = "HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS"
        with patch.dict(os.environ, {rounds_env: ""}):
            os.environ.pop(rounds_env)
            policy = PopulationBasedSearch._final_selection_from_profile(full)
            self.assertEqual(policy, FinalSelectionConfig())
            for effort, expected in (
                ("full", FinalSelectionConfig()),
                ("quick", None),
                ("none", None),
            ):
                settings = Settings(
                    autotune_log_level=logging.CRITICAL, autotune_effort=effort
                )
                self.assertEqual(
                    PopulationBasedSearch._final_selection_from_settings(settings),
                    expected,
                    effort,
                )
            os.environ[rounds_env] = "6"
            policy = PopulationBasedSearch._final_selection_from_profile(full)
            self.assertEqual(policy, FinalSelectionConfig(rounds=6))
            os.environ[rounds_env] = "0"
            self.assertIsNone(PopulationBasedSearch._final_selection_from_profile(full))
        # the policy is not a constructor kwarg the profile has to forward
        kwargs = PopulationBasedSearch.get_kwargs_from_profile(
            full, Settings(autotune_log_level=logging.CRITICAL)
        )
        self.assertNotIn("final_selection", kwargs)

    def test_policy_is_resolved_from_settings_not_the_constructor(self) -> None:
        # A benchmarking policy, not a search-algorithm knob: no registered
        # search names it in its signature (so ``_algorithm_cache_policy`` does
        # not classify it; ``_final_rebenchmark_cache_policy`` carries it) and
        # every population search resolves it from ``autotune_effort``.
        for cls in search_algorithms.values():
            self.assertNotIn(
                "final_selection",
                inspect.signature(cls.__init__).parameters,
                cls.__name__,
            )
        for effort, expected in (("full", FinalSelectionConfig()), ("quick", None)):
            settings = Settings(
                autotune_log_level=logging.CRITICAL, autotune_effort=effort
            )
            with patch.dict(os.environ):
                # test/conftest.py keeps the suite on the single-pass path
                os.environ.pop("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", None)
                search = DifferentialEvolutionSearch(_mock_kernel(settings), ())
            self.assertEqual(search.final_selection, expected, effort)
            self.assertEqual(
                search._final_rebenchmark_cache_policy()["final_selection"],
                None if expected is None else dataclasses.asdict(expected),
            )
            self.assertNotIn("final_selection", search._algorithm_cache_policy())

    def test_single_pass_knobs_warn_once_when_the_selection_runs(self) -> None:
        members = [_member(32, 1.0), _member(64, 1.1)]
        policy = FinalSelectionConfig(top_k=4, rounds=1, screen_rep_ms=0)
        env = {
            "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS": "5000",
            "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED": "1",
        }

        def run_once() -> list[str]:
            search = _make_search(members, policy=policy)
            search.log = Mock()
            with (
                patch.dict(os.environ, env),
                patch.object(
                    search,
                    "_final_selection_measurer",
                    new=_FakeMeasurer(lambda member, _n: member.perf),
                ),
            ):
                search.final_rebenchmark_best(members[0])
            return [str(call.args[0]) for call in search.log.warning.call_args_list]

        base_search_module._final_selection_ignored_envs_warned = False
        try:
            first = run_once()
            self.assertEqual(len(first), 1)
            self.assertIn("HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS", first[0])
            self.assertIn("HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED", first[0])
            self.assertIn("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS=0", first[0])
            # said once per process
            self.assertEqual(run_once(), [])
        finally:
            base_search_module._final_selection_ignored_envs_warned = False

    def test_final_selection_runs_after_the_finishing_and_terminal_phases(
        self,
    ) -> None:
        # The later phases compare with the search's short windows; their picks
        # join the shootout instead of replacing its winner, and they spend
        # the search phase's budget share: the reserve opens for the
        # selection only.
        members = [_member(32, 1.0), _member(64, 1.1)]
        refined = _member(96, 0.95)
        verified = _member(128, 0.9)
        order: list[str] = []
        released: list[bool] = []

        def finishing(best: PopulationMember, rounds: int) -> PopulationMember:
            order.append("finishing")
            released.append(search._budget_reserve_released)
            return best

        def terminal(best: PopulationMember) -> PopulationMember:
            order.append("terminal")
            released.append(search._budget_reserve_released)
            return refined

        def final(best: PopulationMember) -> PopulationMember:
            order.append("final")
            released.append(search._budget_reserve_released)
            return verified

        for policy, expected_order, expected, expected_released in (
            (
                FinalSelectionConfig(),
                ["finishing", "terminal", "final"],
                verified,
                [False, False, True],
            ),
            (None, ["final", "finishing", "terminal"], refined, [True, True, True]),
        ):
            search = _make_search(members, policy=policy)
            search.finishing_rounds = 0
            order.clear()
            released.clear()
            with (
                patch.object(search, "run_finishing_phase", side_effect=finishing),
                patch.object(search, "run_terminal_refinement", side_effect=terminal),
                patch.object(search, "final_rebenchmark_best", side_effect=final),
                patch.object(search, "_final_pick_supported", return_value=False),
            ):
                config = search._finalize()
            self.assertEqual(order, expected_order)
            self.assertEqual(released, expected_released)
            self.assertEqual(config, expected.config)
            self.assertIs(search.best, expected)
            if policy is not None:
                self.assertIs(search._selected_member, verified)

    def test_every_search_ends_in_the_shared_tail(self) -> None:
        # Differential evolution, its surrogate hybrid and the LLM-guided
        # search ran the finishing phase after the selection; every search
        # now ends in ``_finalize``, so the selection is last and the
        # reserve opens for it alone.
        members = [_member(32, 1.0), _member(64, 1.1)]
        verified = _member(128, 0.9)

        def run(search: PopulationBasedSearch) -> helion.Config:
            if isinstance(search, LLMGuidedSearch):
                search.max_rounds = 1
                with (
                    patch.object(search, "_initialize_prompt_state"),
                    patch.object(search, "_run_initial_round"),
                ):
                    return search._autotune_inner()
            search.min_improvement_delta = None
            search.patience = None
            search.initial_population_strategy = SimpleNamespace(name="random")
            search.population_size = 2
            search.max_generations = 2
            search.crossover_rate = 0.8
            search.compile_timeout_lower_bound = 0.0
            search.compile_timeout_quantile = 0.0
            if isinstance(search, DESurrogateHybrid):
                search.surrogate_threshold = 10
                search.candidate_ratio = 2
                search.all_observations = []
            with (
                patch.object(search, "initial_two_generations"),
                patch.object(search, "set_adaptive_compile_timeout"),
                patch.object(search, "rebenchmark_population"),
            ):
                return search._autotune()

        def phases(
            search: PopulationBasedSearch, order: list[tuple[str, bool]]
        ) -> tuple[Callable[..., PopulationMember], ...]:
            def finishing(best: PopulationMember, rounds: int) -> PopulationMember:
                order.append(("finishing", search._budget_reserve_released))
                return best

            def terminal(best: PopulationMember) -> PopulationMember:
                order.append(("terminal", search._budget_reserve_released))
                return best

            def final(best: PopulationMember) -> PopulationMember:
                order.append(("final", search._budget_reserve_released))
                return verified

            return finishing, terminal, final

        for cls in (DifferentialEvolutionSearch, DESurrogateHybrid, LLMGuidedSearch):
            search = _make_search(members, policy=FinalSelectionConfig(), cls=cls)
            search.finishing_rounds = 1
            order: list[tuple[str, bool]] = []
            finishing, terminal, final = phases(search, order)
            with (
                patch.object(search, "_budgeted_range", return_value=iter(())),
                patch.object(search, "run_finishing_phase", side_effect=finishing),
                patch.object(search, "run_terminal_refinement", side_effect=terminal),
                patch.object(search, "final_rebenchmark_best", side_effect=final),
                patch.object(search, "_final_pick_supported", return_value=False),
            ):
                config = run(search)
            self.assertEqual(
                order,
                [("finishing", False), ("terminal", False), ("final", True)],
                cls.__name__,
            )
            self.assertEqual(config, verified.config, cls.__name__)
            self.assertIs(search.best, verified, cls.__name__)
            self.assertIs(search._selected_member, verified, cls.__name__)

    def test_estimated_seconds_scales_with_windows(self) -> None:
        policy = FinalSelectionConfig(
            top_k=8,
            rounds=4,
            warmup_ms=100,
            rep_ms=500,
            thermal_warmup_ms=5000,
            screen_warmup_ms=50,
            screen_rep_ms=200,
            tiebreak_rounds=0,
            family_finalists=0,
        )
        # 0.2 s of worker/loader overhead per window; up to ten finalists
        # (top_k, the best pinned seed and the incumbent)
        expected = 1.25 * (5.0 + 40 * 0.45 + 10 * 4 * 0.8) + 5.0
        self.assertAlmostEqual(policy.estimated_seconds(40), expected)
        # the family quota's representatives are counted too (its upper
        # bound: every one of them admitted)
        self.assertAlmostEqual(
            dataclasses.replace(policy, family_finalists=2).estimated_seconds(40),
            1.25 * (5.0 + 40 * 0.45 + 12 * 4 * 0.8) + 5.0,
        )
        # no screen when the shortlist is the finalists already
        self.assertAlmostEqual(
            policy.estimated_seconds(9), 1.25 * (5.0 + 9 * 4 * 0.8) + 5.0
        )
        self.assertAlmostEqual(
            policy.estimated_seconds(10),
            1.25 * (5.0 + 10 * 0.45 + 10 * 4 * 0.8) + 5.0,
        )
        self.assertAlmostEqual(
            policy.estimated_seconds(11),
            1.25 * (5.0 + 11 * 0.45 + 10 * 4 * 0.8) + 5.0,
        )
        # the second regime's column is counted for every finalist (its
        # upper bound: all of them tied, or an explicit target regime)
        self.assertAlmostEqual(
            dataclasses.replace(policy, tiebreak_rounds=2).estimated_seconds(9),
            1.25 * (5.0 + 9 * 4 * 0.8 + 9 * 2 * 0.8) + 5.0,
        )

    def test_metrics_report_the_phase(self) -> None:
        metrics = AutotuneMetrics()
        self.assertNotIn("final_selection", metrics.to_dict())
        metrics.final_selection = {"finalists": 2}
        self.assertEqual(metrics.to_dict()["final_selection"], {"finalists": 2})

    def test_finalists_are_timed_in_the_final_selection_regime(self) -> None:
        """A resident search (HELION_AUTOTUNE_L2_FLUSH=0) ranks its finalists
        in the replay regime, where configs that tie L2-resident separate;
        the default per-call flush search keeps its regime. Both measurers
        carry the regime to their timer; a backend's own steady bench does
        not know the regimes and is called as before."""
        member = _member(32, 1.0)
        policy = FinalSelectionConfig(top_k=2, rounds=2, screen_rep_ms=0)
        search = _make_search([member], policy=policy)
        provider = Mock(spec=LocalBenchmarkProvider)
        provider.mutated_arg_indices = []
        provider.private_argument_copies_affordable.return_value = False
        provider.steady_window_benchmark_available.return_value = True
        provider._subprocess_benchmark_uses_wall_clock.return_value = False
        provider.benchmark_isolated.return_value = [1.0]
        search.benchmark_provider = provider
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(get_do_bench=lambda: None)
        )
        for env, regime in [
            ({"HELION_AUTOTUNE_L2_FLUSH": "0"}, "replay"),
            ({}, "call"),
            (
                {
                    "HELION_AUTOTUNE_L2_FLUSH": "0",
                    "HELION_AUTOTUNE_FINAL_L2_FLUSH": "0",
                },
                "off",
            ),
        ]:
            with patch.dict(os.environ, env, clear=False):
                for name in (
                    "HELION_AUTOTUNE_L2_FLUSH",
                    "HELION_AUTOTUNE_FINAL_L2_FLUSH",
                ):
                    if name not in env:
                        os.environ.pop(name, None)
                provider.benchmark_isolated.reset_mock()
                with search._final_selection_measurer([member], policy) as measurer:
                    self.assertIsInstance(measurer, _WorkerFinalSelectionMeasurer)
                    self.assertEqual(measurer.regime_label, regime)
                    self.assertEqual(measurer.measure(0, 3, 4), 1.0)
                kwargs = provider.benchmark_isolated.call_args.kwargs
                self.assertEqual(kwargs["l2_flush_mode"], regime)
                self.assertEqual((kwargs["warmup"], kwargs["rep"]), (3, 4))
        # a wall-clock worker always flushes, whatever the regime says
        provider._subprocess_benchmark_uses_wall_clock.return_value = True
        with patch.dict(os.environ, {"HELION_AUTOTUNE_L2_FLUSH": "0"}):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            with search._final_selection_measurer([member], policy) as measurer:
                self.assertEqual(
                    measurer.regime_label, "call (wall clock, always flushed)"
                )
        provider._subprocess_benchmark_uses_wall_clock.return_value = False

        provider.steady_window_benchmark_available.return_value = False
        seen: list[dict[str, object]] = []

        def fake_do_bench(fn: Callable[[], object], **kwargs: object) -> float:
            seen.append(kwargs)
            return 2.0

        with (
            patch.dict(os.environ, {"HELION_AUTOTUNE_L2_FLUSH": "0"}),
            patch.object(base_search_module, "do_bench", new=fake_do_bench),
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            with search._final_selection_measurer([member], policy) as measurer:
                self.assertIsInstance(measurer, _InProcessFinalSelectionMeasurer)
                self.assertEqual(measurer.regime_label, "replay")
                self.assertEqual(measurer.measure(0, 3, 4), 2.0)
            self.assertEqual(seen[-1]["l2_flush_mode"], "replay")
            self.assertEqual((seen[-1]["warmup"], seen[-1]["rep"]), (3, 4))
            # a backend's own steady bench is called without the regime and
            # the selection says so
            search.config_spec.backend.get_do_bench = lambda: fake_do_bench
            with (
                patch.object(base_search_module, "do_bench", new=lambda: None),
                search._final_selection_measurer([member], policy) as measurer,
            ):
                self.assertEqual(measurer.regime_label, "backend bench")
                self.assertEqual(measurer.measure(0, 3, 4), 2.0)
            self.assertNotIn("l2_flush_mode", seen[-1])
            # the wall-clock bench takes the regime but always flushes
            with (
                patch.object(base_search_module, "do_bench", new=lambda: None),
                patch.object(base_search_module, "do_bench_generic", new=fake_do_bench),
                search._final_selection_measurer([member], policy) as measurer,
            ):
                self.assertEqual(
                    measurer.regime_label, "call (wall clock, always flushed)"
                )
                self.assertEqual(measurer.measure(0, 3, 4), 2.0)
            self.assertEqual(seen[-1]["l2_flush_mode"], "replay")

    def test_final_selection_logs_and_reports_its_regime(self) -> None:
        fast = _member(32, 1.00)
        slow = _member(64, 0.95)
        policy = FinalSelectionConfig(top_k=4, rounds=2, screen_rep_ms=0)
        search = _make_search([slow, fast], policy=policy)
        truth = {id(fast): 1.00, id(slow): 1.05}
        measurer = _FakeMeasurer(lambda member, window: truth[id(member)])
        with (
            patch.dict(os.environ, {"HELION_AUTOTUNE_L2_FLUSH": "0"}),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(slow), fast)
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["l2_flush_mode"], "replay")
        self.assertTrue(any("(L2 replay)" in line for line in logs.output))
        # a measurer that times in another regime labels the phase itself
        search = _make_search([slow, fast], policy=policy)
        measurer = _FakeMeasurer(lambda member, window: truth[id(member)])
        measurer.regime_label = "backend bench"  # type: ignore[attr-defined]
        with (
            patch.dict(os.environ, {"HELION_AUTOTUNE_L2_FLUSH": "0"}),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(slow), fast)
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["l2_flush_mode"], "backend bench")
        self.assertTrue(any("(L2 backend bench)" in line for line in logs.output))

    def _tied_warm_search(
        self,
        *,
        cold: dict[str, float],
        warm: dict[str, float] | None = None,
        tiebreak_rounds: int = 2,
    ) -> tuple[PopulationBasedSearch, _FakeMeasurer, dict[str, PopulationMember]]:
        """Three finalists: ``a`` and ``b`` tie in the warm (replay) regime
        at 3.055 ms, ``c`` trails by 10%; ``cold`` gives their per-call
        flushed readings by name."""
        members: dict[str, PopulationMember] = {
            name: _member(32 * (i + 1), 3.2) for i, name in enumerate("abc")
        }
        warm = warm or {"a": 3.055, "b": 3.0551, "c": 3.36}
        names = {id(member): name for name, member in members.items()}
        policy = FinalSelectionConfig(
            top_k=4, rounds=2, screen_rep_ms=0, tiebreak_rounds=tiebreak_rounds
        )
        search = _make_search(list(members.values()), policy=policy)
        measurer = _FakeMeasurer(
            lambda member, _n: warm[names[id(member)]],
            lambda member, _n: cold[names[id(member)]],
            regime_label="replay",
        )
        return search, measurer, members

    def test_cold_column_breaks_a_warm_tie(self) -> None:
        """Finalists tied in the replay regime (one flush, then a burst of
        back-to-back calls) are timed again per call flushed, and the cold
        medians break the tie: the pick carries the knob the burst cannot
        see (the recurrent-step state load's L2 eviction hint read 3.90 vs
        5.41 us cold behind a 3.055 us tie). Only the tied group pays for
        the cold rounds, and the finalists keep their replay medians."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 5.41, "b": 3.90, "c": 3.80}
        )
        with (
            patch.dict(
                os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["b"])
        # two cold rounds over the tied pair only, alternating order
        self.assertEqual(
            [(w[0], w[1]) for w in measurer.second_windows],
            [
                ("call", members["a"]),
                ("call", members["b"]),
                ("call", members["b"]),
                ("call", members["a"]),
            ],
        )
        self.assertEqual(measurer.second_windows[0][2:], (500, 500))
        # the ranking regime's medians stay the finalists' perf
        self.assertAlmostEqual(members["b"].perf, 3.0551)
        self.assertAlmostEqual(members["a"].perf, 3.055)
        rows = {cast("str", row["config_id"]): row for row in _rows(search)}
        by_name = {
            name: rows[canonical_config_id(member.config)]
            for name, member in members.items()
        }
        self.assertEqual(by_name["b"]["second_median_ms"], 3.90)
        self.assertEqual(by_name["b"]["second_rounds_ms"], [3.90, 3.90])
        self.assertEqual(by_name["a"]["second_median_ms"], 5.41)
        self.assertNotIn("second_median_ms", by_name["c"])
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["l2_flush_mode"], "replay")
        self.assertEqual(metrics["second_l2_flush_mode"], "call")
        self.assertEqual(metrics["selected_by"], "replay")
        self.assertEqual(metrics["target_regime"], "auto")
        self.assertEqual(
            metrics["selected_config_id"], canonical_config_id(members["b"].config)
        )
        line = next(line for line in logs.output if "Final selection:" in line)
        self.assertIn("(L2 replay)", line)
        self.assertIn("the tie within 1.0% broken by 2 rounds of L2 call", line)
        self.assertIn("call=3.9[3.9 3.9]", line)
        self.assertIn("call=5.41[5.41 5.41]", line)

    def test_cold_column_keeps_the_warm_pick_when_cold_ties_too(self) -> None:
        """The replay ranking stands unless a tied rival is faster cold by
        more than the tolerance; without a second column (tiebreak_rounds 0)
        nothing is timed cold and the ranking regime decides alone."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 3.90, "b": 3.905, "c": 3.80}
        )
        with (
            patch.dict(
                os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        self.assertEqual(len(measurer.second_windows), 4)

        search, measurer, members = self._tied_warm_search(
            cold={"a": 5.41, "b": 3.90, "c": 3.80}, tiebreak_rounds=0
        )
        with (
            patch.dict(
                os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        self.assertEqual(measurer.second_windows, [])
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertIsNone(metrics["second_l2_flush_mode"])

    def test_no_tie_means_no_cold_column(self) -> None:
        """Finalists more than the tolerance apart in the ranking regime are
        not timed again, and the default per-call flushed selection never
        takes a second column (a cold reading carries the steady state)."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 5.41, "b": 3.90, "c": 3.80},
            warm={"a": 3.055, "b": 3.20, "c": 3.36},
        )
        with (
            patch.dict(
                os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        self.assertEqual(measurer.second_windows, [])
        line = next(line for line in logs.output if "Final selection:" in line)
        self.assertIn("no tie within 1.0% for L2 call windows to break", line)

        search, measurer, members = self._tied_warm_search(
            cold={"a": 5.41, "b": 3.90, "c": 3.80}
        )
        measurer.regime_label = "call"
        with patch.dict(os.environ, _SHORTLIST_ALL):
            for name in (
                "HELION_AUTOTUNE_L2_FLUSH",
                "HELION_AUTOTUNE_FINAL_L2_FLUSH",
                "HELION_AUTOTUNE_TARGET_REGIME",
            ):
                os.environ.pop(name, None)
            with patch.object(search, "_final_selection_measurer", new=measurer):
                self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        self.assertEqual(measurer.second_windows, [])

    def test_target_regime_selects_by_its_column(self) -> None:
        """HELION_AUTOTUNE_TARGET_REGIME names the regime the kernel is
        scored in: every complete finalist is timed in it as well and that
        column selects outright, even against the ranking regime (dh: 10.4
        vs 15.5 us cold behind a warm ranking the other way); the finalists'
        perf becomes the target regime's median."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 15.5, "b": 10.4, "c": 12.0},
            warm={"a": 8.0, "b": 9.0, "c": 9.5},
        )
        with (
            patch.dict(
                os.environ,
                {
                    **_SHORTLIST_ALL,
                    "HELION_AUTOTUNE_L2_FLUSH": "replay",
                    "HELION_AUTOTUNE_TARGET_REGIME": "call",
                },
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["b"])
        self.assertEqual(
            sorted({id(w[1]) for w in measurer.second_windows}),
            sorted(id(member) for member in members.values()),
        )
        self.assertEqual(len(measurer.second_windows), 6)
        self.assertEqual(members["b"].perf, 10.4)
        self.assertEqual(members["a"].perf, 15.5)
        self.assertEqual(members["b"].perfs, [3.2, 9.0, 10.4])
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["selected_by"], "call")
        self.assertEqual(metrics["target_regime"], "call")
        line = next(line for line in logs.output if "Final selection:" in line)
        self.assertIn("selected 10.4", line)
        self.assertIn("by 2 rounds of L2 call windows", line)
        # the table is ordered by the selecting column
        table = line.split("\n")[1:]
        self.assertIn(canonical_config_id(members["b"].config), table[0])
        self.assertIn(canonical_config_id(members["a"].config), table[-1])

        # a target that is the ranking regime already adds nothing; the
        # per-call default search can be told to select by the burst
        search, measurer, members = self._tied_warm_search(
            cold={"a": 15.5, "b": 10.4, "c": 12.0},
            warm={"a": 8.0, "b": 9.0, "c": 9.5},
        )
        with (
            patch.dict(
                os.environ,
                {
                    **_SHORTLIST_ALL,
                    "HELION_AUTOTUNE_L2_FLUSH": "replay",
                    "HELION_AUTOTUNE_TARGET_REGIME": "replay",
                },
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        self.assertEqual(measurer.second_windows, [])
        search, measurer, members = self._tied_warm_search(
            cold={"a": 15.5, "b": 10.4, "c": 12.0},
            warm={"a": 8.0, "b": 9.0, "c": 9.5},
        )
        measurer.regime_label = "call"
        with (
            patch.dict(
                os.environ,
                {**_SHORTLIST_ALL, "HELION_AUTOTUNE_TARGET_REGIME": "replay"},
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
        ):
            for name in ("HELION_AUTOTUNE_L2_FLUSH", "HELION_AUTOTUNE_FINAL_L2_FLUSH"):
                os.environ.pop(name, None)
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["b"])
        self.assertEqual({w[0] for w in measurer.second_windows}, {"replay"})

    def test_second_regime_follows_the_target(self) -> None:
        from helion.autotuner.benchmarking import final_selection_second_regime
        from helion.autotuner.benchmarking import final_selection_target_regime

        cases = [
            # (target env, ranking regime) -> (target, second regime)
            (None, "replay", "auto", "call"),
            (None, "off", "auto", "call"),
            (None, "call", "auto", None),
            ("auto", "replay", "auto", "call"),
            ("call", "replay", "call", "call"),
            ("call", "call", "call", None),
            ("replay", "call", "replay", "replay"),
            ("replay", "replay", "replay", None),
            ("0", "replay", "off", "off"),
            # timers that cannot honour a regime take no second column
            (None, "backend bench", "auto", None),
            ("call", "call (wall clock, always flushed)", "call", None),
        ]
        for target, regime, expected_target, expected in cases:
            with patch.dict(os.environ, {}):
                os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
                if target is not None:
                    os.environ["HELION_AUTOTUNE_TARGET_REGIME"] = target
                self.assertEqual(
                    final_selection_second_regime(regime), expected, (target, regime)
                )
                self.assertEqual(final_selection_target_regime(), expected_target)

    def test_cold_failure_keeps_the_warm_ranking(self) -> None:
        """A finalist that fails only in the cold column keeps its replay
        median and is not timed cold again; the tie-break runs on the rest,
        and the pinned preference reaches into the cold column."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 5.41, "b": 3.90, "c": 3.80}
        )
        failing = members["a"]
        measurer.second_model = lambda member, _n: (
            IsolatedBenchmarkFailure("timeout")
            if member is failing
            else {"b": 3.90, "c": 3.80}[
                next(name for name, m in members.items() if m is member)
            ]
        )
        with (
            patch.dict(
                os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
            # a's cold column failed: no cold evidence against a, so the
            # replay pick stands
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        self.assertEqual(
            [w[1] for w in measurer.second_windows],
            [members["a"], members["b"], members["b"]],
        )
        self.assertAlmostEqual(members["a"].perf, 3.055)
        rows = {cast("str", row["config_id"]): row for row in _rows(search)}
        self.assertEqual(
            rows[canonical_config_id(members["a"].config)]["second_disqualified"],
            "timeout",
        )

        # The pinned preference reaches into the cold column: with every
        # finalist tied warm, the cold-best is c, and the pinned b within
        # the pinned tolerance of it cold keeps the preference; without a
        # tolerance c wins.
        for tolerance, winner in (("0.02", "b"), (None, "c")):
            search, measurer, members = self._tied_warm_search(
                cold={"a": 5.41, "b": 3.95, "c": 3.90},
                warm={"a": 3.055, "b": 3.0551, "c": 3.056},
            )
            search._pinned_finalist_configs = {members["b"].config}
            search._pinned_finalist_members = {members["b"].config: members["b"]}
            env = {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            if tolerance is not None:
                env["HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE"] = tolerance
            with (
                patch.dict(os.environ, env),
                patch.object(search, "_final_selection_measurer", new=measurer),
            ):
                if tolerance is None:
                    os.environ.pop(
                        "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE", None
                    )
                os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
                os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
                self.assertIs(
                    search.final_rebenchmark_best(members["a"]),
                    members[winner],
                    tolerance,
                )
            self.assertEqual(len(measurer.second_windows), 6)

    def test_target_column_without_readings_falls_back_to_the_ranking(self) -> None:
        """An explicit target regime whose column takes no complete reading
        (every finalist failed or missed a round there) leaves the ranking
        regime's pick, and the report says which column selected."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 15.5, "b": 10.4, "c": 12.0},
            warm={"a": 8.0, "b": 9.0, "c": 9.5},
        )
        measurer.second_model = lambda member, _n: IsolatedBenchmarkFailure("timeout")
        with (
            patch.dict(
                os.environ,
                {
                    **_SHORTLIST_ALL,
                    "HELION_AUTOTUNE_L2_FLUSH": "replay",
                    "HELION_AUTOTUNE_TARGET_REGIME": "call",
                },
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        # every finalist was tried cold once and not again after failing
        self.assertEqual(len(measurer.second_windows), 3)
        self.assertEqual(members["a"].perf, 8.0)
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["selected_by"], "replay")
        self.assertEqual(metrics["second_l2_flush_mode"], "call")
        self.assertEqual(metrics["target_regime"], "call")
        self.assertTrue(
            any(
                "no complete readings in the L2 call regime" in message
                for message in logs.output
            )
        )
        line = next(line for line in logs.output if "Final selection:" in line)
        self.assertIn("selected 8", line)
        self.assertIn(
            "by the L2 replay column (no complete reading in 2 rounds of L2 call "
            "windows)",
            line,
        )
        # the table is ordered by the column that selected
        table = line.split("\n")[1:]
        self.assertIn(canonical_config_id(members["a"].config), table[0])
        self.assertIn(canonical_config_id(members["c"].config), table[-1])
        rows = {cast("str", row["config_id"]): row for row in _rows(search)}
        self.assertEqual(
            {cast("str", row["second_disqualified"]) for row in rows.values()},
            {"timeout"},
        )

    def test_tied_group_without_cold_readings_is_reported_as_timed(self) -> None:
        """When every member of the tied group produced no cold reading
        (the windows returned nothing, no failure), the windows were still
        spent: the rows carry the empty column and the header does not
        claim there was no tie to break; the replay pick stands."""
        search, measurer, members = self._tied_warm_search(
            cold={"a": 5.41, "b": 3.90, "c": 3.80}
        )
        measurer.second_model = lambda member, _n: None
        with (
            patch.dict(
                os.environ, {**_SHORTLIST_ALL, "HELION_AUTOTUNE_L2_FLUSH": "replay"}
            ),
            patch.object(search, "_final_selection_measurer", new=measurer),
            self.assertLogs(search.log._logger, level=logging.INFO) as logs,
        ):
            os.environ.pop("HELION_AUTOTUNE_FINAL_L2_FLUSH", None)
            os.environ.pop("HELION_AUTOTUNE_TARGET_REGIME", None)
            search.log.level = logging.INFO
            self.assertIs(search.final_rebenchmark_best(members["a"]), members["a"])
        # a and b were tried once each; a finalist without a reading in a
        # round is not timed again
        self.assertEqual(
            sorted(id(w[1]) for w in measurer.second_windows),
            sorted(id(members[name]) for name in "ab"),
        )
        rows = {cast("str", row["config_id"]): row for row in _rows(search)}
        by_name = {
            name: rows[canonical_config_id(member.config)]
            for name, member in members.items()
        }
        self.assertEqual(by_name["a"]["second_disqualified"], "incomplete")
        self.assertEqual(by_name["a"]["second_rounds_ms"], [])
        self.assertNotIn("second_disqualified", by_name["c"])
        line = next(line for line in logs.output if "Final selection:" in line)
        self.assertIn(
            "the tie within 1.0% unresolved: no complete reading in 2 rounds of "
            "L2 call windows",
            line,
        )
        self.assertIn("call=incomplete[]", line)
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["selected_by"], "replay")

    def test_family_quota_admits_the_other_family_at_the_screen(self) -> None:
        # Six configs of family A and one of family B, shortlisted whole.
        # The screen (the search's clock regime) ranks A's best 1% ahead of
        # B and B below the top_k cut; the rounds (the sustained regime)
        # read B 3% faster. With the quota B runs the rounds and wins; with
        # the quota off, or without family knobs, A wins and B is never
        # timed past the screen.
        policy = FinalSelectionConfig(
            top_k=2,
            rounds=2,
            thermal_warmup_ms=0,
            warmup_ms=100,
            rep_ms=500,
            screen_warmup_ms=50,
            screen_rep_ms=200,
            tiebreak_rounds=0,
        )

        def run(
            search_policy: FinalSelectionConfig, *, keys: bool
        ) -> tuple[PopulationBasedSearch, _FakeMeasurer, PopulationMember]:
            a = [_family_member(16 * (i + 1), "a", 1.00 + 0.002 * i) for i in range(6)]
            b = _family_member(256, "b", 1.03)
            members = [*a, b]
            screen = {id(member): member.perf for member in a}
            screen[id(b)] = 1.01
            rounds = {id(member): member.perf for member in a}
            rounds[id(b)] = 0.97
            measurer = _FakeMeasurer(lambda member, _n: 0.0)

            def model(member: PopulationMember, _n: int) -> float:
                window = measurer.windows[-1]
                if window[2:] == (policy.screen_warmup_ms, policy.screen_rep_ms):
                    return screen[id(member)]
                return rounds[id(member)]

            measurer.model = model
            search = (
                _family_search(members, policy=search_policy)
                if keys
                else _make_search(members, policy=search_policy)
            )
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "8"}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
                self.assertLogs(search.log._logger, level=logging.INFO) as logs,
            ):
                search.log.level = logging.INFO
                selected = search.final_rebenchmark_best(a[0])
            search._quota_logs = [  # type: ignore[attr-defined]
                line for line in logs.output if "Final selection family quota" in line
            ]
            return search, measurer, selected

        search, measurer, selected = run(policy, keys=True)
        self.assertEqual(selected.config["fam"], "b")
        round_windows = [
            w for w in measurer.windows if w[2:] == (policy.warmup_ms, policy.rep_ms)
        ]
        # the two best of A by the screen, plus B (the incumbent A0 is among them)
        self.assertEqual(len(round_windows), 2 * 3)
        self.assertEqual({w[1].config["fam"] for w in round_windows}, {"a", "b"})
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None
        self.assertEqual(metrics["shortlisted"], 7)
        self.assertEqual(metrics["finalists"], 3)
        self.assertEqual(metrics["family_keys"], ["fam"])
        rows = {row["family"]: row for row in _rows(search)}
        self.assertEqual(set(rows), {"a", "b"})
        self.assertTrue(rows["b"]["family_quota"])
        self.assertFalse(rows["a"]["family_quota"])
        quota_logs = search._quota_logs  # type: ignore[attr-defined]
        self.assertEqual(len(quota_logs), 1)
        self.assertIn("family quota (screen)", quota_logs[0])
        self.assertIn("(fam) within 1.25x", quota_logs[0])
        self.assertIn("b at", quota_logs[0])

        for search_policy, keys in (
            (dataclasses.replace(policy, family_finalists=0), True),
            (policy, False),
        ):
            search, measurer, selected = run(search_policy, keys=keys)
            self.assertEqual(selected.config["fam"], "a")
            self.assertEqual(selected.perf, 1.00)
            round_windows = [
                w
                for w in measurer.windows
                if w[2:] == (policy.warmup_ms, policy.rep_ms)
            ]
            self.assertEqual(len(round_windows), 2 * 2)
            self.assertEqual({w[1].config["fam"] for w in round_windows}, {"a"})
            self.assertEqual(search._quota_logs, [])  # type: ignore[attr-defined]
            metrics = search._autotune_metrics.final_selection
            assert metrics is not None
            self.assertEqual(metrics["family_keys"], ["fam"] if keys else [])
            for row in _rows(search):
                self.assertEqual("family" in row, keys)

    def test_family_quota_counts_the_seed_and_the_incumbent_as_represented(
        self,
    ) -> None:
        # At the screen cut the best pinned seed and the incumbent survive
        # outside the top_k on their own right; the quota is computed after
        # them, so its (single) slot goes to a family neither of them
        # belongs to (C), not to a second config of their family (B).
        policy = FinalSelectionConfig(
            top_k=2,
            rounds=2,
            family_finalists=1,
            thermal_warmup_ms=0,
            warmup_ms=100,
            rep_ms=500,
            screen_warmup_ms=50,
            screen_rep_ms=200,
            tiebreak_rounds=0,
        )

        def check(role: str) -> None:
            a = [_family_member(16 * (i + 1), "a", 1.00 + 0.002 * i) for i in range(3)]
            b = [_family_member(128 + 16 * i, "b", 1.05 + 0.01 * i) for i in range(2)]
            c = _family_member(512, "c", 1.08)
            members = [*a, *b, c]
            rounds = {id(member): member.perf for member in members}
            rounds[id(c)] = 0.97
            measurer = _FakeMeasurer(lambda member, _n: 0.0)

            def model(member: PopulationMember, _n: int) -> float:
                window = measurer.windows[-1]
                if window[2:] == (policy.screen_warmup_ms, policy.screen_rep_ms):
                    return member.perf
                return rounds[id(member)]

            measurer.model = model
            search = _family_search(
                members, policy=policy, pinned=[b[1]] if role == "pinned" else []
            )
            incumbent = b[1] if role == "incumbent" else a[0]
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "8"}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
                self.assertLogs(search.log._logger, level=logging.INFO) as logs,
            ):
                search.log.level = logging.INFO
                selected = search.final_rebenchmark_best(incumbent)
            self.assertIs(selected, c, role)
            round_members = {
                id(w[1])
                for w in measurer.windows
                if w[2:] == (policy.warmup_ms, policy.rep_ms)
            }
            # the top_k of A, B's representative (the seed / the incumbent,
            # not its better-screened sibling) and the quota's C
            self.assertEqual(
                round_members, {id(m) for m in (a[0], a[1], b[1], c)}, role
            )
            rows = {row["family"]: row for row in _rows(search)}
            self.assertEqual(set(rows), {"a", "b", "c"}, role)
            self.assertTrue(rows["c"]["family_quota"], role)
            self.assertFalse(rows["b"]["family_quota"], role)
            quota_logs = [
                line for line in logs.output if "Final selection family quota" in line
            ]
            self.assertEqual(len(quota_logs), 1, role)
            self.assertIn("1 more family (fam)", quota_logs[0])
            self.assertIn("c at", quota_logs[0])

        check("incumbent")
        check("pinned")

    def test_family_quota_reaches_the_shortlist(self) -> None:
        # The shortlist is cut by the search's own readings
        # (HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K); B's best, ranked below
        # the cut, is shortlisted by the quota and screened like the rest,
        # while a family beyond the spread (C at 1.4x) is not.
        a = [_family_member(16 * (i + 1), "a", 1.00 + 0.01 * i) for i in range(3)]
        b = [_family_member(128 + 16 * i, "b", 1.05 + 0.01 * i) for i in range(2)]
        c = _family_member(512, "c", 1.40)
        members = [*a, *b, c]
        policy = FinalSelectionConfig(
            top_k=2,
            rounds=2,
            thermal_warmup_ms=0,
            screen_warmup_ms=50,
            screen_rep_ms=200,
            tiebreak_rounds=0,
        )
        for search_policy, expected in (
            (policy, [a[0], a[1], b[0]]),
            (dataclasses.replace(policy, family_finalists=0), [a[0], a[1]]),
        ):
            search = _family_search(members, policy=search_policy)
            measurer = _FakeMeasurer(lambda member, _n: member.perf)
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
                self.assertLogs(search.log._logger, level=logging.INFO) as logs,
            ):
                search.log.level = logging.INFO
                self.assertIs(search.final_rebenchmark_best(a[0]), a[0])
            self.assertEqual(measurer.finalists, expected)
            quota_logs = [
                line for line in logs.output if "Final selection family quota" in line
            ]
            if search_policy.family_finalists:
                # admitted at the shortlist cut, then again at the rounds'
                # cut (three shortlisted: no screen, the search's readings
                # rank them and B is third)
                self.assertEqual(len(quota_logs), 2)
                self.assertIn("family quota (shortlist)", quota_logs[0])
                self.assertIn("family quota (rounds)", quota_logs[1])
                for line in quota_logs:
                    self.assertIn("b at", line)
                    self.assertNotIn("c at", line)
            else:
                self.assertEqual(quota_logs, [])
            metrics = search._autotune_metrics.final_selection
            assert metrics is not None
            self.assertEqual(metrics["shortlisted"], len(expected))

    def test_family_quota_is_capped_and_bounded(self) -> None:
        # Without a screen the search's readings rank the shortlist; of the
        # families behind the cut the closest ``family_finalists`` within
        # the spread join the rounds, one config each, and a family beyond
        # the spread never does.
        a = [_family_member(16, "a", 1.00), _family_member(32, "a", 1.005)]
        b, c, d = (
            _family_member(64, "b", 1.01),
            _family_member(80, "c", 1.02),
            _family_member(96, "d", 1.03),
        )
        e = _family_member(112, "e", 1.30)
        members = [*a, b, c, d, e]
        base = FinalSelectionConfig(
            top_k=1, rounds=2, thermal_warmup_ms=0, screen_rep_ms=0, tiebreak_rounds=0
        )
        for policy, expected in (
            (base, [a[0], b, c]),
            (dataclasses.replace(base, family_finalists=3), [a[0], b, c, d]),
            (
                dataclasses.replace(base, family_finalists=3, family_spread=1.015),
                [a[0], b],
            ),
            (dataclasses.replace(base, family_finalists=0), [a[0]]),
        ):
            search = _family_search(members, policy=policy)
            measurer = _FakeMeasurer(lambda member, _n: member.perf)
            with (
                patch.dict(
                    os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "8"}
                ),
                patch.object(search, "_final_selection_measurer", new=measurer),
            ):
                self.assertIs(search.final_rebenchmark_best(a[0]), a[0])
            timed = []
            for window in measurer.windows:
                if window[0] == "measure" and window[1] not in timed:
                    timed.append(window[1])
            self.assertEqual(timed, expected, policy)
            metrics = search._autotune_metrics.final_selection
            assert metrics is not None
            self.assertEqual(
                [row["family_quota"] for row in _rows(search)],
                [member is not a[0] for member in expected],
            )

    def test_budget_counts_the_family_quota_only_with_family_keys(self) -> None:
        # The quota's representatives are reserved for (and counted in the
        # shortlist estimate) only where the spec names family knobs; the
        # ladder sheds them right after the second regime's column.
        members = [_member(32, 1.0), _member(64, 1.1)]
        policy = FinalSelectionConfig()
        one_column = dataclasses.replace(policy, tiebreak_rounds=0)
        self.assertEqual(policy.budget_ladder()[3].family_finalists, 0)
        self.assertEqual(policy.budget_ladder()[3].rounds, policy.rounds)
        with patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "8"}):
            _clear_regime_envs()
            plain = _make_search(
                members, policy=policy, pinned=members[:1], budget_seconds=1000
            )
            self.assertEqual(
                plain._final_selection_sizing_policy(),
                dataclasses.replace(one_column, family_finalists=0),
            )
            self.assertEqual(plain._final_selection_shortlist_estimate(), 10)
            self.assertAlmostEqual(
                plain._budget_reserve_seconds(),
                dataclasses.replace(one_column, family_finalists=0).estimated_seconds(
                    10
                ),
            )
            family = _family_search(
                members, policy=policy, pinned=members[:1], budget_seconds=1000
            )
            self.assertEqual(family._final_selection_sizing_policy(), one_column)
            self.assertEqual(family._final_selection_shortlist_estimate(), 12)
            self.assertAlmostEqual(
                family._budget_reserve_seconds(), one_column.estimated_seconds(12)
            )
            self.assertGreater(
                family._budget_reserve_seconds(), plain._budget_reserve_seconds()
            )
            # the cache key keeps the profile's fields either way
            self.assertEqual(
                family._final_rebenchmark_cache_policy(),
                plain._final_rebenchmark_cache_policy(),
            )


class TestPrivateKernelArgs(TestCase):
    def test_private_slots_own_distinct_storage_and_keep_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "args.pt")
            base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            view = base[:, 1:3]
            torch.save((base, view, 7, "x"), path)
            slot0 = load_private_kernel_args(path, 0)
            slot1 = load_private_kernel_args(path, 1)
            self.assertIs(load_private_kernel_args(path, 0), slot0)
            base0, view0, seven, text = slot0
            base1, view1, _, _ = slot1
            assert isinstance(base0, torch.Tensor)
            assert isinstance(view0, torch.Tensor)
            assert isinstance(base1, torch.Tensor)
            assert isinstance(view1, torch.Tensor)
            self.assertEqual(seven, 7)
            self.assertEqual(text, "x")
            torch.testing.assert_close(base0, base)
            torch.testing.assert_close(base1, base)
            self.assertNotEqual(base0.data_ptr(), base1.data_ptr())
            self.assertNotEqual(base0.data_ptr(), base.data_ptr())
            # the view still aliases its base inside the slot
            self.assertEqual(
                view0.untyped_storage().data_ptr(), base0.untyped_storage().data_ptr()
            )
            self.assertEqual(view0.stride(), view.stride())
            self.assertEqual(view0.storage_offset(), view.storage_offset())
            base0.fill_(1.0)
            self.assertTrue(bool((view0 == 1.0).all()))
            self.assertFalse(bool((base1 == 1.0).all()))
            release_private_kernel_args()

    def test_release_empties_the_private_slot_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "args.pt")
            torch.save((torch.zeros(4),), path)
            release_private_kernel_args()
            slot0 = load_private_kernel_args(path, 0)
            load_private_kernel_args(path, 1)
            self.assertEqual(load_private_kernel_args.cache_info().currsize, 2)
            release_private_kernel_args()
            self.assertEqual(load_private_kernel_args.cache_info().currsize, 0)
            self.assertIsNot(load_private_kernel_args(path, 0), slot0)
            release_private_kernel_args()

    def test_worker_measurer_releases_private_copies_on_exit(self) -> None:
        provider = Mock()
        with _WorkerFinalSelectionMeasurer(
            provider, [lambda: None], private_copies=True, timeout_s=1.0
        ) as measurer:
            self.assertEqual(measurer.slots, [0])
        provider.release_private_argument_copies.assert_called_once_with()

        provider = Mock()
        with _WorkerFinalSelectionMeasurer(
            provider, [lambda: None], private_copies=False, timeout_s=1.0
        ) as measurer:
            self.assertEqual(measurer.slots, [None])
        provider.release_private_argument_copies.assert_not_called()

        # a worker that raised on the way out is not asked to release
        provider = Mock()
        with (
            self.assertRaises(BenchmarkWorkerUnkillable),
            _WorkerFinalSelectionMeasurer(
                provider, [lambda: None], private_copies=True, timeout_s=1.0
            ),
        ):
            raise BenchmarkWorkerUnkillable("worker remained alive")
        provider.release_private_argument_copies.assert_not_called()

    def test_provider_release_runs_the_job_in_a_live_worker(self) -> None:
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.settings = Settings(autotune_benchmark_timeout=30)
        provider._benchmark_worker = None
        provider.release_private_argument_copies()
        worker = Mock()
        worker.alive.return_value = False
        provider._benchmark_worker = worker
        provider.release_private_argument_copies()
        worker.run.assert_not_called()
        worker.alive.return_value = True
        provider.release_private_argument_copies()
        worker.run.assert_called_once_with(release_private_kernel_args, timeout=30.0)
        # Best-effort: the selection is complete, so a worker that times out
        # or dies on the release (its copies die with it), or a release job
        # that raises in the worker (a sticky CUDA error from a finalist,
        # surfaced by the cache release), is logged rather than failing the
        # autotune; an unkillable worker still raises.
        provider.log = Mock()
        errors = (
            BenchmarkTimeout("benchmark timeout after 30.0s"),
            BenchmarkWorkerDied("worker pipe closed before sending result"),
            BenchmarkSubprocessError("worker raised unpickleable error"),
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            torch.AcceleratorError("CUDA error: unspecified launch failure"),
        )
        for error in errors:
            worker.run.side_effect = error
            provider.release_private_argument_copies()
            self.assertIn(str(error), str(provider.log.warning.call_args.args[0]))
        self.assertEqual(provider.log.warning.call_count, len(errors))
        worker.run.side_effect = BenchmarkWorkerUnkillable("worker remained alive")
        with self.assertRaises(BenchmarkWorkerUnkillable):
            provider.release_private_argument_copies()
        # a programming error in the job is not a release failure
        worker.run.side_effect = TypeError("release_private_kernel_args() takes")
        with self.assertRaises(TypeError):
            provider.release_private_argument_copies()
        self.assertEqual(provider.log.warning.call_count, len(errors))

    def test_benchmark_job_slot_is_a_trailing_optional_field(self) -> None:
        job = BenchmarkJob(object(), "args.pt", 2, 3, True, 7, False)  # type: ignore[arg-type]
        self.assertIsNone(job.args_slot)
        self.assertEqual(job.fixed_repetitions, 7)
        job = BenchmarkJob(object(), "args.pt", args_slot=3)  # type: ignore[arg-type]
        self.assertEqual(job.args_slot, 3)

    def test_worker_measurer_names_the_regime_of_every_window(self) -> None:
        provider = Mock()
        provider.benchmark_isolated.return_value = [0.5]
        fn = lambda: None  # noqa: E731
        with _WorkerFinalSelectionMeasurer(
            provider, [fn], private_copies=False, timeout_s=1.0, l2_flush_mode="replay"
        ) as measurer:
            measurer.warm(0, 100)
            self.assertEqual(measurer.measure(0, 3, 4), 0.5)
        calls = provider.benchmark_isolated.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[1].kwargs,
            {
                "warmup": 3,
                "rep": 4,
                "desc": "Final selection",
                "args_slots": [None],
                "timeout": 1.0,
                "l2_flush_mode": "replay",
            },
        )
        self.assertEqual(calls[0].kwargs["l2_flush_mode"], "replay")

    def test_benchmark_job_times_in_the_regime_it_names(self) -> None:
        seen: list[dict[str, object]] = []

        def fake_bench(fn: Callable[[], object], **kwargs: object) -> float:
            seen.append(kwargs)
            return 1.5

        with (
            patch(
                "helion.autotuner.benchmark_job._load_compiled_fn_for_worker",
                return_value=lambda *args: args,
            ),
            patch("helion.autotuner.benchmark_job._unload_compiled_fn"),
            patch(
                "helion.autotuner.benchmark_job.load_trusted_kernel_args",
                return_value=("shared",),
            ),
            patch("helion.autotuner.benchmark_job.do_bench", side_effect=fake_bench),
        ):
            self.assertEqual(BenchmarkJob(object(), "args.pt")(), 1.5)  # type: ignore[arg-type]
            self.assertEqual(
                BenchmarkJob(object(), "args.pt", l2_flush_mode="replay")(),  # type: ignore[arg-type]
                1.5,
            )
        # the search's jobs keep the worker's HELION_AUTOTUNE_L2_FLUSH regime
        self.assertIsNone(seen[0]["l2_flush_mode"])
        self.assertEqual(seen[1]["l2_flush_mode"], "replay")

    def test_benchmark_job_reads_the_private_slot(self) -> None:
        seen: dict[str, object] = {}

        def fake_bench(fn: Callable[[], object], **kwargs: object) -> float:
            seen["args"] = fn()
            return 1.5

        with (
            patch(
                "helion.autotuner.benchmark_job._load_compiled_fn_for_worker",
                return_value=lambda *args: args,
            ),
            patch("helion.autotuner.benchmark_job._unload_compiled_fn"),
            patch(
                "helion.autotuner.benchmark_job.load_private_kernel_args",
                return_value=("private", 2),
            ) as private,
            patch(
                "helion.autotuner.benchmark_job.load_trusted_kernel_args",
                return_value=("shared",),
            ) as shared,
            patch("helion.autotuner.benchmark_job.do_bench", side_effect=fake_bench),
        ):
            job = BenchmarkJob(object(), "args.pt", args_slot=2)  # type: ignore[arg-type]
            self.assertEqual(job(), 1.5)
        private.assert_called_once_with("args.pt", 2)
        shared.assert_not_called()
        self.assertEqual(seen["args"], ("private", 2))


if __name__ == "__main__":
    unittest.main()


@helion.kernel(autotune_effort="full", static_shapes=True)
def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(out.size()):
        out[tile] = a[tile] + b[tile]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "runs the selection on a CUDA device")
@onlyBackends(["triton"])
class TestFinalSelectionOnDevice(TestCase):
    """test/conftest.py keeps every other test on the single-pass verification
    (``HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS=0``); this smoke test runs the
    real full-effort selection on the device, in the benchmark worker.

    Triton only: a directly constructed search over a CuTe kernel fails in the
    fork precompile path before any config is timed (``_prepare_precompiler_for_fork``
    hands the CuTe host function a launcher stub that ends up as the logger:
    ``'function' object has no attribute 'debug'``), on this branch and on dev
    alike; the selection itself is backend independent and the CPU tests cover
    it."""

    @skipIfRefEager("compiles and times configs on the device")
    def test_full_effort_selection_runs(self) -> None:
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound = _add.bind(args)
        env = {
            "HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS": "2",
            "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2",
        }
        with patch.dict(os.environ, env):
            search = PatternSearch(
                bound, args, initial_population=3, max_generations=1, copies=1
            )
            policy = search.final_selection
            assert policy is not None
            self.assertEqual(policy.rounds, 2)
            best = search.autotune()
        metrics = search._autotune_metrics.final_selection
        assert metrics is not None, "the final selection did not run"
        self.assertEqual(cast("dict[str, object]", metrics["policy"])["rounds"], 2)
        self.assertGreaterEqual(cast("int", metrics["finalists"]), 2)
        self.assertEqual(metrics["selected_config_id"], canonical_config_id(best))
        rows = cast("list[dict[str, object]]", metrics["rows"])
        selected = [
            row for row in rows if row["config_id"] == metrics["selected_config_id"]
        ]
        self.assertEqual(len(selected), 1)
        self.assertIsNone(selected[0]["disqualified"])
        self.assertEqual(len(cast("list[float]", selected[0]["rounds_ms"])), 2)
        # ``search.best`` ranks the population by its search-window timings,
        # which may still favour a member the harness-grade selection re-ranked
        # below the pick; the selection's member is what autotune returned.
        selected_member = search._selected_member
        assert selected_member is not None
        self.assertEqual(selected_member.config, best)
        fn = bound.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] + args[1])
