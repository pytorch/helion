"""Source aliases in the population searches.

A source alias is a candidate whose generated source an earlier candidate of
the run already produced (``status == "deduplicated"``): another config of the
same kernel.  On a backend that collapses them
(``Backend.collapses_generated_source_aliases``) a search path and the polish
descent do not move to one, a generation does not keep one in its population,
the final selection's history does not record one, no two search paths start
from one kernel, and one member holds each kernel
(``PopulationBasedSearch._kernel_holder``): it is the one a re-timing times,
once however many configs of the kernel a population holds, and every other
config of the kernel has its reading.
"""

from __future__ import annotations

from contextlib import nullcontext
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import helion
from helion._compiler.backend import Backend
from helion._compiler.cute.backend import CuteBackend
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import BenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

pytestmark = pytest.mark.cpu_only


def _spec(*, collapse: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cute_flash_search_enabled=False,
        backend=SimpleNamespace(
            collapses_generated_source_aliases=lambda config_spec: collapse
        ),
    )


def _provider(*, settles: bool = True, **attrs: object) -> SimpleNamespace:
    """A stand-in benchmark provider: one that reports source aliases
    (``BenchmarkProvider.settles_source_aliases``) unless told otherwise."""
    return SimpleNamespace(settles_source_aliases=lambda: settles, **attrs)


def _member(perf: float | None, num_warps: int, status: str = "ok") -> PopulationMember:
    return PopulationMember(
        lambda: None,
        [] if perf is None else [perf],
        (num_warps,),
        helion.Config(num_warps=num_warps),
        status=status,  # type: ignore[arg-type]
    )


def _holding_search(
    *, collapse: bool, search_type: type[PopulationBasedSearch] = PopulationBasedSearch
) -> PopulationBasedSearch:
    """A bare search with what settling and re-timing members read: an
    empty population, final selection history and pins."""
    search = object.__new__(search_type)
    search.config_spec = _spec(collapse=collapse)  # type: ignore[assignment]
    search.benchmark_provider = _provider()  # type: ignore[assignment]
    search.population = []
    search._kernels = {}
    search._benchmarked_members = {}
    search._pinned_finalist_configs = set()  # type: ignore[assignment]
    search._pinned_finalist_members = {}
    return search


def _alias(kernel: PopulationMember, num_warps: int) -> PopulationMember:
    """Another config of ``kernel``'s code, with the reading it has now."""
    alias = _member(kernel.perf, num_warps, "deduplicated")
    alias.fn = kernel.fn
    return alias


def _benchmark(outcomes: dict[int, tuple[float, str]]) -> Mock:
    """A ``benchmark_population`` that gives each member its (perf, status)."""

    def run(members: list[PopulationMember], desc: str = "") -> None:
        for member in members:
            perf, status = outcomes[id(member)]
            member.perfs.append(perf)
            member.status = status  # type: ignore[assignment]

    return Mock(side_effect=run)


def test_cute_deduplicates_every_search_and_collapses_outside_flash() -> None:
    backend = CuteBackend()
    plain = SimpleNamespace(cute_flash_search_enabled=False)
    flash = SimpleNamespace(cute_flash_search_enabled=True)
    assert backend.should_deduplicate_generated_sources(plain)  # type: ignore[arg-type]
    assert backend.should_deduplicate_generated_sources(flash)  # type: ignore[arg-type]
    assert backend.collapses_generated_source_aliases(plain)  # type: ignore[arg-type]
    # The flash search keeps its own alias policy.
    assert not backend.collapses_generated_source_aliases(flash)  # type: ignore[arg-type]
    # Other backends keep every alias a candidate of its own.
    assert not Backend.collapses_generated_source_aliases(backend, plain)  # type: ignore[arg-type]


@pytest.mark.parametrize("collapse", [True, False])
def test_without_source_aliases_drops_only_new_aliases(collapse: bool) -> None:
    search = object.__new__(PopulationBasedSearch)
    search.config_spec = _spec(collapse=collapse)  # type: ignore[assignment]
    search.benchmark_provider = _provider()  # type: ignore[assignment]
    incumbent = _member(10.0, 1, "deduplicated")  # an earlier batch's alias
    fresh = _member(9.0, 2)
    alias = _member(8.0, 3, "deduplicated")
    failed = _member(float("inf"), 4, "error")
    members = [incumbent, fresh, alias, failed]

    kept = search._without_source_aliases(members, members[1:])

    assert kept == ([incumbent, fresh, failed] if collapse else members)
    assert search._is_source_alias(alias) is collapse
    assert not search._is_source_alias(fresh)


@pytest.mark.parametrize("backend_collapses", [True, False])
@pytest.mark.parametrize("provider_settles", [True, False])
def test_a_search_drops_aliases_only_where_its_provider_reports_them(
    provider_settles: bool, backend_collapses: bool
) -> None:
    """The backend asks for the collapse; whether a search has aliases to
    drop is its provider's to say.  A multi-shape provider reports none
    (its ``benchmark`` results are joint readings of every shape) and a
    local one none once torch.distributed is initialized, so there a
    ``"deduplicated"`` member (a source repair of a multi-shape flash
    search, say) is a candidate of its own and every member of a re-timed
    list is timed."""
    search = _holding_search(collapse=backend_collapses)
    search.benchmark_provider = _provider(settles=provider_settles)  # type: ignore[assignment]
    drops = backend_collapses and provider_settles
    kernel = _member(10.0, 1)
    alias = _alias(kernel, 2)
    timed: list[list[PopulationMember]] = []
    search.rebenchmark = lambda members, **kwargs: timed.append(members)  # type: ignore[method-assign]

    assert search._collapses_source_aliases() is drops
    assert search._is_source_alias(alias) is drops
    search._settle_benchmarked_members([kernel, alias], [kernel, alias])
    assert (search._kernel_holder(alias) is kernel) is drops
    search._rebenchmark_kernels([kernel, alias])
    assert timed == [[kernel] if drops else [kernel, alias]]


def test_multi_shape_provider_reports_no_aliases() -> None:
    provider = MultiShapeBenchmarkProvider.__new__(MultiShapeBenchmarkProvider)
    assert not provider.settles_source_aliases()
    # The default of a provider that is none of the built-in ones.
    assert not BenchmarkProvider.settles_source_aliases(provider)


def _polish_search(*, collapse: bool, settles: bool = True) -> LFBOTreeSearch:
    search = object.__new__(LFBOTreeSearch)
    search.polish_rounds = 10
    search.config_spec = _spec(collapse=collapse)  # type: ignore[assignment]
    search.benchmark_provider = _provider(settles=settles)  # type: ignore[assignment]
    search.min_improvement_delta = 0.001
    search.log = Mock()
    search._budgeted_range = lambda *args: iter(range(*args))  # type: ignore[method-assign]
    search.set_generation = Mock()  # type: ignore[method-assign]
    search._autotune_metrics = SimpleNamespace(num_generations=5)  # type: ignore[assignment]
    search.format_performance = lambda perf: f"{perf:.4f}"  # type: ignore[method-assign]
    search.rebenchmark_population = Mock()  # type: ignore[method-assign]
    return search


@pytest.mark.parametrize("collapse", [True, False])
def test_polish_does_not_move_to_a_source_alias(collapse: bool) -> None:
    """Two neighbors repeat earlier code and read below the incumbent (the
    cached reading of that code); the only new code is slower."""
    search = _polish_search(collapse=collapse)
    start = _member(10.0, 1)
    alias_a = _member(None, 2)
    alias_b = _member(None, 3)
    slower = _member(None, 4)
    search.population = [start]
    visited = {start.config}
    members = {m.flat_values: m for m in (alias_a, alias_b, slower)}
    search.benchmark_population = _benchmark(  # type: ignore[method-assign]
        {
            id(alias_a): (9.0, "deduplicated"),
            id(alias_b): (9.5, "deduplicated"),
            id(slower): (11.0, "ok"),
        }
    )
    with (
        patch.object(
            PatternSearch,
            "_generate_neighbors",
            side_effect=lambda self_, base: (
                [*members] if base == start.flat_values else []
            ),
        ),
        patch.object(search, "make_unbenchmarked", side_effect=members.get),
    ):
        search._polish_descent(visited)  # type: ignore[arg-type]

    (verified,), _kwargs = search.rebenchmark_population.call_args_list[0]  # type: ignore[attr-defined]
    messages = [call.args[0] for call in search.log.call_args_list]  # type: ignore[attr-defined]
    if collapse:
        # One round: the incumbent stands, the aliases are neither verified
        # nor population members, and the log says how many were dropped.
        assert search.best is start
        assert search.population == [start, slower]
        assert verified == [start, slower]
        assert messages == [
            "Polish round 1: 3 neighbors (2 repeat earlier code), best 10.0000"
        ]
    else:
        # Every alias is a candidate of its own: the descent moves to one.
        assert search.best is alias_a
        assert verified == [start, alias_a, alias_b, slower]
        assert messages[0] == "Polish round 1: 3 neighbors, best 9.0000"
    assert search.benchmark_population.call_count == 1  # type: ignore[attr-defined]


_Rounds = list[tuple[PopulationMember, list[tuple[PopulationMember, float]]]]


def _descend(
    search: LFBOTreeSearch,
    start: PopulationMember,
    rounds: _Rounds,
    rereads: list[tuple[PopulationMember, float]] | None = None,
) -> tuple[list[PopulationMember], list[str]]:
    """Run the polish descent from ``start`` over a scripted neighborhood.

    ``rounds`` gives, per incumbent, the new kernels of its round with their
    first readings, and ``rereads`` the reading every later verification
    gives a kernel it re-times.  Returns the incumbent of every round and
    the log.
    """
    by_flat = {start.flat_values: start}
    by_base = {}
    first = {}
    for base, new in rounds:
        by_base[base.flat_values] = [member.flat_values for member, _perf in new]
        for member, perf in new:
            by_flat[member.flat_values] = member
            first[id(member)] = (perf, "ok")
    again = {id(member): perf for member, perf in rereads or []}

    def verify(candidates: list[PopulationMember], desc: str = "") -> None:
        # The round's own new kernels keep their first reading.
        if id(candidates[0]) in again:
            candidates[0].perfs.append(again[id(candidates[0])])

    search.population = [start]
    search.benchmark_population = _benchmark(first)  # type: ignore[method-assign]
    search.rebenchmark_population = Mock(side_effect=verify)  # type: ignore[method-assign]
    with (
        patch.object(
            PatternSearch,
            "_generate_neighbors",
            side_effect=lambda self_, base: by_base.get(base, []),
        ) as generate,
        patch.object(search, "make_unbenchmarked", side_effect=by_flat.get),
    ):
        search._polish_descent({start.config})  # type: ignore[arg-type]
    incumbents = [by_flat[call.args[1]] for call in generate.call_args_list]
    messages = [call.args[0] for call in search.log.call_args_list]  # type: ignore[attr-defined]
    return incumbents, messages


@pytest.mark.parametrize("collapse", [True, False])
def test_polish_crosses_a_plateau_of_new_code(collapse: bool) -> None:
    """No neighbor reads below the incumbent and two new kernels read exactly
    the same (the timer's step): the descent moves to one of them."""
    search = _polish_search(collapse=collapse)
    start = _member(10.0, 1)
    slower = _member(None, 2)
    tie = _member(None, 3)
    second_tie = _member(None, 4)
    beyond = _member(None, 5)

    incumbents, messages = _descend(
        search,
        start,
        [
            (start, [(slower, 11.0), (tie, 10.0), (second_tie, 10.0)]),
            (tie, [(beyond, 12.0)]),
        ],
    )

    if collapse:
        # Round 2 starts from the tie and ends the descent: its only
        # neighbor is slower and nothing reads level.
        assert incumbents == [start, tie]
        assert messages == [
            (
                "Polish round 1: 3 neighbors, best 10.0000"
                " (level with the incumbent: new code)"
            ),
            "Polish round 2: 1 neighbors, best 10.0000",
        ]
        assert beyond in search.population
    else:
        # A tie is no improvement: the descent ends on the incumbent.
        assert incumbents == [start]
        assert messages == ["Polish round 1: 3 neighbors, best 10.0000"]


def test_polish_keeps_its_stop_rule_where_the_provider_reports_no_aliases() -> None:
    """A multi-shape search, or one in a process with torch.distributed
    initialized, on a backend that asks for the collapse: no alias is ever
    dropped, so a neighbor that reads level may be a copy of the incumbent,
    timed on its own.  The descent ends on its first round without an
    improvement, as it does on a backend that keeps its aliases, and takes
    no level move and no patience."""
    search = _polish_search(collapse=True, settles=False)
    start = _member(10.0, 1)
    copy = _member(None, 2)
    beyond = _member(None, 3)

    incumbents, messages = _descend(
        search, start, [(start, [(copy, 10.0)]), (copy, [(beyond, 5.0)])]
    )

    assert incumbents == [start]
    assert messages == ["Polish round 1: 1 neighbors, best 10.0000"]
    assert beyond not in search.population


# One 32 ns timer step at a 40 us incumbent is 0.08%, below the 0.1%
# min_improvement_delta: a step better, a tie and a step worse are one case.
@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("reading", [39.968, 40.0, 40.032])
def test_polish_treats_a_step_below_the_delta_as_the_plateau(
    reading: float, collapse: bool
) -> None:
    search = _polish_search(collapse=collapse)
    start = _member(40.0, 1)
    near = _member(None, 2)
    slower = _member(None, 3)
    beyond = _member(None, 4)

    incumbents, _messages = _descend(
        search,
        start,
        [(start, [(slower, 41.0), (near, reading)]), (near, [(beyond, 30.0)])],
    )

    # With the aliases dropped the descent reaches the faster kernel behind
    # the neighbor in all three cases; without, it ends in all three.
    assert incumbents == ([start, near, beyond] if collapse else [start])
    assert (beyond in search.population) is collapse


def test_polish_does_not_cross_to_code_that_reads_worse_than_the_delta() -> None:
    search = _polish_search(collapse=True)
    start = _member(40.0, 1)
    worse = _member(None, 2)
    beyond = _member(None, 3)

    incumbents, messages = _descend(
        search, start, [(start, [(worse, 40.1)]), (worse, [(beyond, 30.0)])]
    )

    assert incumbents == [start]
    assert messages == ["Polish round 1: 1 neighbors, best 40.0000"]


def test_polish_moves_to_the_lowest_reading_level_kernel() -> None:
    search = _polish_search(collapse=True)
    start = _member(40.0, 1)
    above = _member(None, 2)
    below = _member(None, 3)
    tie = _member(None, 4)

    # ``below`` is the round's best but less than the delta under the
    # incumbent; ``above`` comes first in neighbor order.
    incumbents, _messages = _descend(
        search, start, [(start, [(above, 40.02), (below, 39.99), (tie, 40.0)])]
    )
    assert incumbents == [start, below]

    # With the incumbent the round's best, the lowest level reading above it.
    search = _polish_search(collapse=True)
    start = _member(40.0, 1)
    higher = _member(None, 2)
    lower = _member(None, 3)
    incumbents, _messages = _descend(
        search, start, [(start, [(higher, 40.03), (lower, 40.01)])]
    )
    assert incumbents == [start, lower]


def _chain(readings: list[float]) -> tuple[list[PopulationMember], _Rounds]:
    """A start kernel and one new kernel per round, each the next's base."""
    members = [_member(readings[0], 1)]
    members += [_member(None, index + 2) for index in range(len(readings) - 1)]
    rounds = [
        (base, [(new, perf)])
        for base, new, perf in zip(members, members[1:], readings[1:], strict=False)
    ]
    return members, rounds


@pytest.mark.parametrize(
    ("patience", "rounds_run"), [(0, 1), (1, 1), (2, 2), (3, 3), (10, 7)]
)
def test_polish_patience_bounds_the_rounds_on_a_plateau(
    patience: int, rounds_run: int
) -> None:
    """Eight kernels that all read the same: the descent crosses the plateau
    for ``polish_patience`` rounds (seven rounds exhaust the chain)."""
    search = _polish_search(collapse=True)
    search.polish_patience = patience
    members, rounds = _chain([10.0] * 8)

    incumbents, messages = _descend(search, members[0], rounds)

    if rounds_run < 7:
        assert incumbents == members[:rounds_run]
        assert messages[-1] == (
            f"Polish stops: {rounds_run} rounds without a reading below 10.0000"
        )
    else:
        assert incumbents == members
        assert messages[-1] == "Polish round 8: no unvisited neighbors"


def test_polish_patience_restarts_at_a_new_lowest_reading() -> None:
    search = _polish_search(collapse=True)
    # Two level rounds, an improvement, then three more level rounds.
    members, rounds = _chain([10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0])

    incumbents, messages = _descend(search, members[0], rounds)

    assert incumbents == members[:6]
    assert messages[-1] == "Polish stops: 3 rounds without a reading below 9.0000"


def test_polish_patience_counts_a_level_step_at_a_lower_reading() -> None:
    """The verification re-reads the incumbent 1% down and new code ties
    that reading: the incumbent has read as low itself, so the round is no
    new lowest reading."""
    search = _polish_search(collapse=True)
    search.polish_patience = 1
    members, rounds = _chain([10.0, 9.9, 9.9])

    incumbents, messages = _descend(
        search, members[0], rounds, rereads=[(members[0], 9.9)]
    )

    assert incumbents == members[:1]
    assert messages == [
        "Polish round 1: 1 neighbors, best 9.9000 (level with the incumbent: new code)",
        "Polish stops: 1 rounds without a reading below 9.9000",
    ]


@pytest.mark.parametrize("collapse", [True, False])
def test_polish_patience_counts_a_step_from_a_noisy_incumbent(collapse: bool) -> None:
    """Every verification re-reads the incumbent 10% up, so each round's new
    kernel is an improvement on the incumbent's latest reading without ever
    reading below the descent's lowest.  With the aliases dropped these
    rounds use up the patience; without, the descent keeps its old rule."""
    search = _polish_search(collapse=collapse)
    search.polish_patience = 2
    members, rounds = _chain([10.0, 9.0, 9.0, 9.0, 9.0, 9.0])

    incumbents, messages = _descend(
        search, members[0], rounds, rereads=[(member, 9.9) for member in members[1:]]
    )

    if collapse:
        # Round 1 is the new lowest reading; rounds 2 and 3 are not.
        assert incumbents == members[:3]
        assert messages[-1] == "Polish stops: 2 rounds without a reading below 9.0000"
    else:
        assert incumbents == members
        assert not any(message.startswith("Polish stops") for message in messages)


def test_level_rule_and_stop_rule_share_one_tolerance() -> None:
    """New code reads level with the incumbent exactly where the stop rule
    counts the step to it as no improvement: one check
    (``PatternSearch._reads_level``) at the one delta, for the paths of a
    pattern search and for the polish descent.  Checked every 0.01% from 3%
    below the incumbent to 3% above it, across the 0.1% delta."""
    for search in (_polish_search(collapse=True), object.__new__(PatternSearch)):
        search.min_improvement_delta = 0.001
        search.log = Mock()
        current = _member(0.0032, 1)
        level = []
        for step in range(-300, 301):
            candidate = _member(0.0032 * (1.0 + step / 10000), 2)
            if candidate.perf != current.perf:
                reads_level = search._reads_level(candidate.perf, current.perf)
                assert reads_level is search._check_early_stopping(candidate, current)
                level.append(reads_level)
        # Level within 0.1% on either side, and nowhere else.
        assert 0 < sum(level) < 30
        assert not level[0] and not level[-1]
    # A search without a delta takes every step: only equal readings tie.
    search.min_improvement_delta = 0.0
    assert search._reads_level(0.0032, 0.0032)
    assert not search._reads_level(0.0032 * 0.99999, 0.0032)
    assert not search._check_early_stopping(_member(0.0032, 3), _member(0.0032, 4))


def _advance(steps: Iterator[list[PopulationMember]]) -> list[PopulationMember] | None:
    return next(steps, None)


@pytest.mark.parametrize("collapse", [True, False])
def test_pattern_search_path_does_not_move_to_a_source_alias(collapse: bool) -> None:
    search = object.__new__(PatternSearch)
    search.config_spec = _spec(collapse=collapse)  # type: ignore[assignment]
    search.benchmark_provider = _provider()  # type: ignore[assignment]
    search.max_generations = 5
    search.min_improvement_delta = 0.001
    start = _member(10.0, 1)
    alias = _member(None, 2)
    slower = _member(None, 3)
    beyond = _member(None, 4)
    by_base = {
        start.flat_values: [alias.flat_values, slower.flat_values],
        alias.flat_values: [beyond.flat_values],
    }
    members = {m.flat_values: m for m in (alias, slower, beyond)}
    visited = {start.config}
    with (
        patch.object(
            search, "_generate_neighbors", side_effect=lambda base: by_base[base]
        ),
        patch.object(search, "make_unbenchmarked", side_effect=members.get),
    ):
        steps = search._pattern_search_from(start, visited)  # type: ignore[arg-type]
        assert _advance(steps) == [start, alias, slower]
        _benchmark({id(alias): (9.0, "deduplicated"), id(slower): (11.0, "ok")})(
            [alias, slower]
        )
        following = _advance(steps)

    # The path ends where no new code beats its incumbent; without the
    # collapse it continues from the alias.
    assert following == (None if collapse else [alias, beyond])


@pytest.mark.parametrize("collapse", [True, False])
def test_lfbo_path_does_not_move_to_a_source_alias(collapse: bool) -> None:
    search = object.__new__(LFBOTreeSearch)
    search.config_spec = _spec(collapse=collapse)  # type: ignore[assignment]
    search.benchmark_provider = _provider()  # type: ignore[assignment]
    search.max_generations = 5
    search.min_improvement_delta = 0.001
    search.patience = 0
    search.frac_selected = 1.0
    search.log = Mock()
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))  # type: ignore[assignment]
    start = _member(10.0, 1)
    alias = _member(None, 2)
    slower = _member(None, 3)
    beyond = _member(None, 4)
    by_base = {
        start.flat_values: [alias.flat_values, slower.flat_values],
        alias.flat_values: [beyond.flat_values],
    }
    members = {m.flat_values: m for m in (alias, slower, beyond)}
    with (
        patch.object(
            search, "_generate_neighbors", side_effect=lambda base: by_base[base]
        ),
        patch.object(search, "make_unbenchmarked", side_effect=members.get),
        patch.object(
            search, "_surrogate_select", side_effect=lambda candidates, n: candidates
        ),
    ):
        steps = search._pruned_pattern_search_from(0, start, set())
        assert _advance(steps) == [start, alias, slower]
        _benchmark({id(alias): (9.0, "deduplicated"), id(slower): (11.0, "ok")})(
            [alias, slower]
        )
        following = _advance(steps)

    assert following == (None if collapse else [alias, beyond])


def _pinning_search(
    *, collapse: bool, search_type: type[PopulationBasedSearch] = PopulationBasedSearch
) -> PopulationBasedSearch:
    """A bare search that ``benchmark_population`` runs on."""
    search = _holding_search(collapse=collapse, search_type=search_type)
    search.config_gen = SimpleNamespace(flatten=lambda config: [config.num_warps])  # type: ignore[assignment]
    search._benchmarking_initial_population = False
    search.benchmark_provider = _provider(take_effective_source_repairs=list)  # type: ignore[assignment]
    return search


@pytest.mark.parametrize("collapse", [True, False])
def test_final_selection_history_skips_source_aliases(collapse: bool) -> None:
    search = _pinning_search(collapse=collapse)
    fresh = _member(None, 2)
    alias = _member(None, 3)
    results = [
        SimpleNamespace(
            config=fresh.config, perf=9.0, fn=fresh.fn, status="ok", compile_time=1.0
        ),
        SimpleNamespace(
            config=alias.config,
            perf=8.0,
            fn=fresh.fn,
            status="deduplicated",
            compile_time=None,
        ),
    ]
    with (
        patch.object(search, "benchmark_batch", return_value=results),
        patch.object(search, "_record_benchmarked_member") as record,
    ):
        search.benchmark_population([fresh, alias])

    recorded = [call.args[0] for call in record.call_args_list]
    assert recorded == ([fresh] if collapse else [fresh, alias])
    # The alias still carries a reading (the surrogate trains on it): its
    # kernel's, where the search counts it as that kernel.
    assert alias.perf == (9.0 if collapse else 8.0)
    assert alias.status == "deduplicated"


@pytest.mark.parametrize("collapse", [True, False])
def test_a_pinned_source_alias_pins_the_config_that_holds_its_kernel(
    collapse: bool,
) -> None:
    """A pinned seed repeats the code an unpinned config produced.  The
    alias takes no entry in the final selection, so the pin goes to the
    config that holds the kernel: the kernel reaches the final selection
    whichever of its configs was benchmarked first."""
    search = _pinning_search(collapse=collapse)
    first = _member(None, 2)
    seed = _member(None, 3)
    other = _member(None, 4)
    search._pinned_finalist_configs.add(seed.config)
    results = [
        SimpleNamespace(
            config=seed.config,
            perf=9.0,
            fn=first.fn,
            status="deduplicated",
            compile_time=None,
        ),
        SimpleNamespace(
            config=first.config, perf=9.0, fn=first.fn, status="ok", compile_time=1.0
        ),
        SimpleNamespace(
            config=other.config, perf=8.0, fn=other.fn, status="ok", compile_time=1.0
        ),
    ]
    with (
        patch.object(search, "benchmark_batch", return_value=results),
        patch.object(search, "_final_rebenchmark_top_k", return_value=8),
    ):
        search.benchmark_population([seed, first, other])

    pinned = {config.num_warps for config in search._pinned_finalist_members}
    assert pinned == ({2} if collapse else {3})
    assert (first.config in search._pinned_finalist_configs) is collapse
    assert other.config not in search._pinned_finalist_configs


@pytest.mark.parametrize("collapse", [True, False])
def test_a_reverified_pinned_alias_takes_no_final_selection_entry(
    collapse: bool,
) -> None:
    """Two pinned seeds are one kernel.  A re-timing that reaches the alias
    itself records nothing under the alias's config: the final selection
    would screen the kernel twice."""
    search = _pinning_search(collapse=collapse)
    kernel = _member(10.0, 1)
    alias = _member(10.0, 2, "deduplicated")
    alias.fn = kernel.fn
    search._pinned_finalist_configs.update({kernel.config, alias.config})
    with patch.object(search, "_final_rebenchmark_top_k", return_value=8):
        search._record_benchmarked_member(kernel)
        alias.perfs.append(9.9)
        search._refresh_benchmarked_members_after_rebenchmark([alias])

    pinned = {config.num_warps for config in search._pinned_finalist_members}
    assert pinned == ({1} if collapse else {1, 2})


@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("pinned", [True, False])
def test_a_repaired_member_follows_the_alias_rules(
    pinned: bool, collapse: bool
) -> None:
    """A config fails (a timeout) and a later config with the same generated
    source is timed: the provider repairs the failed member with that
    reading, which makes it a source alias.  Where aliases are dropped it
    takes no entry in the final selection's history beside its kernel, and a
    pinned one (a seed) passes its pin to the config that was timed, as an
    alias of the batch does."""
    search = _pinning_search(collapse=collapse)
    search._compiler_seed_members = []
    failed = _member(float("inf"), 2, "timeout")
    search.population = [failed]
    if pinned:
        search._pinned_finalist_configs.add(failed.config)
    kernel = _member(None, 3)
    result = SimpleNamespace(
        config=kernel.config, perf=9.0, fn=kernel.fn, status="ok", compile_time=1.0
    )
    # The provider's repair of the failed config, once the batch measured
    # its generated source under ``kernel``'s config.
    repairs = {
        failed.config: SimpleNamespace(
            config=failed.config,
            perf=9.0,
            fn=kernel.fn,
            status="deduplicated",
            compile_time=None,
        )
    }
    search.benchmark_provider = _provider(  # type: ignore[assignment]
        take_effective_source_repairs=lambda: repairs
    )
    with (
        patch.object(search, "benchmark_batch", return_value=[result]),
        patch.object(search, "_final_rebenchmark_top_k", return_value=8),
    ):
        search.benchmark_population([kernel])

    # The failed member holds the kernel now, under its own config.
    assert (failed.perf, failed.status, failed.fn) == (9.0, "deduplicated", kernel.fn)
    history = {config.num_warps for config in search._benchmarked_members}
    assert history == ({3} if collapse else {2, 3})
    pins = {config.num_warps for config in search._pinned_finalist_members}
    if not pinned:
        assert pins == set()
    else:
        # One finalist entry for the kernel, under the config that was timed.
        assert pins == ({3} if collapse else {2})
        assert (kernel.config in search._pinned_finalist_configs) is collapse


@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("search_type", [PatternSearch, LFBOTreeSearch])
def test_generation_drops_new_source_aliases_from_the_population(
    search_type: type[PatternSearch], collapse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One generation of one path: [incumbent, alias, new code].  The alias
    leaves the population before the rebenchmark; the surrogate still trains
    on it."""
    start = _member(10.0, 1)
    alias = _member(None, 2)
    slower = _member(None, 3)
    search = _holding_search(collapse=collapse, search_type=search_type)
    search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
    search.copies = 1
    # The path ends after its one generation; LFBO trains between generations.
    search.max_generations = 2
    search.similarity_penalty = 1.0  # type: ignore[attr-defined]
    search.compile_timeout_lower_bound = 0.0
    search.compile_timeout_quantile = 0.0
    search.log = Mock()
    search.log.autotune_tracing.return_value = nullcontext()
    search.kernel = Mock(env=SimpleNamespace(process_group_name=None))
    search.config_gen = Mock(encode_config=lambda flat: flat)
    search._autotune_metrics = AutotuneMetrics()
    path = iter([[start, alias, slower]])
    verified: list[list[PopulationMember]] = []
    stubs = {
        "_generate_initial_population_flat": Mock(return_value=[start.flat_values]),
        "make_unbenchmarked": Mock(return_value=start),
        "benchmark_initial_population": Mock(),
        "_random_fallback_population_target": Mock(return_value=None),
        "set_adaptive_compile_timeout": Mock(),
        "capture_compiler_seed_members": Mock(),
        "rebenchmark_population": Mock(
            side_effect=lambda members, desc="": verified.append(list(members))
        ),
        "_pattern_search_from": Mock(return_value=path),
        "_pruned_pattern_search_from": Mock(return_value=path),
        "_budgeted_range": lambda *args: iter(range(*args)),
        "set_generation": Mock(),
        "benchmark_population": _benchmark(
            {id(alias): (9.0, "deduplicated"), id(slower): (11.0, "ok")}
        ),
        "_finalize": Mock(return_value=start.config),
        # LFBO only:
        "_append_training_sample": Mock(),
        "_fit_surrogate": Mock(),
        "_run_flash_structural_qualification": Mock(return_value=0),
        "_autotune_budget_exceeded_across_ranks": Mock(return_value=False),
        "_select_starting_paths": Mock(return_value=[(start, ())]),
        "_path_exhausts_generation_budget": Mock(return_value=False),
        "_polish_descent": Mock(),
    }
    for name, stub in stubs.items():
        setattr(search, name, stub)
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.check_population_consistency", Mock()
    )

    search._autotune()

    kept = [start, slower] if collapse else [start, alias, slower]
    assert search.population == kept
    # verified[0] is the initial population; the generation's follows.
    assert verified[1] == kept
    if search_type is LFBOTreeSearch:
        trained = [
            call.kwargs["member"]
            for call in search._append_training_sample.call_args_list  # type: ignore[attr-defined]
        ]
        assert alias in trained


@pytest.mark.parametrize("collapse", [True, False])
def test_a_new_alias_takes_the_reading_its_kernel_has_now(collapse: bool) -> None:
    """The provider settles an alias with its record of the kernel's first
    reading: 16.4 us for kernels the search has since re-timed at 11.8, 12.0
    and 12.2 us.  The alias takes the reading of the member that holds its
    kernel, the config the search first timed it under, wherever that member
    is now: in the population, behind an earlier alias of the kernel that
    still carries an older reading, or in neither the population nor the
    final selection's history.  Only for a kernel the search never timed
    itself does the provider's record stand; the first alias of it then
    stands in for the kernel, and the next one takes its reading."""
    search = _pinning_search(collapse=collapse)
    kernels = [_member(None, index) for index in (1, 2, 3)]
    in_population, behind_an_alias, gone = kernels
    unseen = _member(0.0164, 4)  # never benchmarked as a member of this search

    def result(config: helion.Config, fn: object, status: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=config, perf=0.0164, fn=fn, status=status, compile_time=None
        )

    def benchmark(members: list[PopulationMember], results: list[object]) -> None:
        with (
            patch.object(search, "benchmark_batch", return_value=results),
            patch.object(search, "_record_benchmarked_member"),
        ):
            search.benchmark_population(members)

    benchmark(kernels, [result(k.config, k.fn, "ok") for k in kernels])
    earlier_alias = _member(None, 7)
    benchmark(
        [earlier_alias],
        [result(earlier_alias.config, behind_an_alias.fn, "deduplicated")],
    )
    # Verifications since: each kernel was re-timed, the earlier alias was
    # not in the lists.
    for kernel, reading in zip(kernels, (0.0118, 0.0120, 0.0122), strict=True):
        kernel.perfs.append(reading)
    search.population = [in_population, earlier_alias, behind_an_alias]
    sources = (in_population, behind_an_alias, gone, unseen, unseen)
    aliases = [_member(None, 10 + index) for index in range(len(sources))]
    benchmark(
        aliases,
        [
            result(alias.config, kernel.fn, "deduplicated")
            for alias, kernel in zip(aliases, sources, strict=True)
        ],
    )

    assert [alias.perfs for alias in aliases] == (
        [[0.0118], [0.0120], [0.0122], [0.0164], [0.0164]]
        if collapse
        else [[0.0164]] * 5
    )
    if collapse:
        assert search._kernel_holder(aliases[2]) is gone
        assert search._kernel_holder(aliases[4]) is aliases[3]


def _retiming_search(
    *, collapse: bool
) -> tuple[PopulationBasedSearch, list[list[PopulationMember]]]:
    """A bare search whose ``rebenchmark`` halves the reading of every
    member it is given, with the lists it timed.  Like the real one it
    leaves a single member as it is: a re-timing compares members."""
    search = _pinning_search(collapse=collapse)
    search.settings = SimpleNamespace(get_rebenchmark_threshold=lambda: 1.5)  # type: ignore[assignment]
    search.best_perf_so_far = 10.0
    timed: list[list[PopulationMember]] = []

    def rebenchmark(members: list[PopulationMember], **kwargs: object) -> None:
        if len(members) < 2:
            return
        timed.append(list(members))
        for member in members:
            member.perfs.append(member.perf * 0.5)
        # as ``rebenchmark`` does for the members it timed
        search._refresh_benchmarked_members_after_rebenchmark(members)

    search.rebenchmark = rebenchmark  # type: ignore[method-assign]
    search._final_rebenchmark_top_k = lambda: 8  # type: ignore[method-assign]
    return search, timed


def _kernel(
    search: PopulationBasedSearch, perf: float, num_warps: int
) -> PopulationMember:
    """A member that ``search`` timed under a config of its own, settled as
    ``benchmark_population`` settles it: it holds its kernel and has its
    entry in the final selection's history."""
    kernel = _member(perf, num_warps)
    search._settle_benchmarked_members([kernel], [kernel])
    return kernel


@pytest.mark.parametrize("collapse", [True, False])
def test_rebenchmark_times_a_kernel_once_and_its_aliases_follow(
    collapse: bool,
) -> None:
    """A population holds a kernel under two configs (an initial population
    keeps its aliases).  The verification re-times the kernel once and the
    alias takes that reading: no second timing of code already read."""
    search, timed = _retiming_search(collapse=collapse)
    kernel = _member(10.0, 1)
    alias = _alias(kernel, 2)
    other = _member(11.0, 3)
    failing = _member(12.0, 4)
    failing_alias = _alias(failing, 5)
    far = _member(20.0, 6)  # beyond the re-timing threshold

    def rebenchmark(members: list[PopulationMember], **kwargs: object) -> None:
        timed.append(list(members))
        for member in members:
            # A re-timing that fails keeps the member's readings as they are.
            if member is not failing:
                member.perfs.append(member.perf * 0.5)

    search.rebenchmark = rebenchmark  # type: ignore[method-assign]
    members = [kernel, alias, other, failing, failing_alias, far]
    search.rebenchmark_population(members, desc="verifying")

    assert kernel.perfs == alias.perfs == [10.0, 5.0]
    assert failing.perfs == [12.0]
    assert far.perfs == [20.0]
    if collapse:
        assert timed == [[kernel, other, failing]]
        # No new reading of the kernel, none for its alias.
        assert failing_alias.perfs == [12.0]
    else:
        # Every alias is a candidate of its own, with its own timing.
        assert timed == [[kernel, alias, other, failing, failing_alias]]
        assert failing_alias.perfs == [12.0, 6.0]


def test_members_that_share_a_function_without_an_alias_are_kernels() -> None:
    """Only a source alias stands behind another member: two members that
    hold one function for another reason (stand-ins, a failed compile's
    placeholder) are each re-timed."""
    search, timed = _retiming_search(collapse=True)
    first = _member(10.0, 1)
    second = _member(10.0, 2)
    second.fn = first.fn
    alias = _alias(first, 3)

    search.rebenchmark_population([first, second, alias])

    assert timed == [[first, second]]
    assert alias.perf == first.perf == 5.0


@pytest.mark.parametrize("collapse", [True, False])
def test_an_alias_ahead_of_its_kernel_does_not_stand_for_it(collapse: bool) -> None:
    """A population lists an alias before the member that holds its kernel
    (an initial population keeps its aliases, in the order it was built).
    The member that is no alias is the one timed: only a member that is
    timed refreshes its entry in the final selection's history, which holds
    the kernel under that member's config."""
    search, timed = _retiming_search(collapse=collapse)
    kernel = _member(10.0, 1)
    alias = _alias(kernel, 2)
    second_alias = _alias(kernel, 3)
    other = _member(11.0, 4)
    search._record_benchmarked_member(kernel)

    search.rebenchmark_population([alias, second_alias, kernel, other])

    assert kernel.perfs == alias.perfs == second_alias.perfs == [10.0, 5.0]
    entry = search._benchmarked_members[kernel.config]
    if collapse:
        assert timed == [[kernel, other]]
        # The history holds the kernel's new reading.
        assert entry.perfs == [10.0, 5.0]
    else:
        assert timed == [[alias, second_alias, kernel, other]]


def test_the_member_that_holds_a_kernel_is_timed_after_it_left_the_population() -> None:
    """The member that holds a kernel has left the population; two aliases of
    it remain (the best of an initial population can be one).  A
    verification still times that member for the kernel, once, which
    refreshes the kernel's entry in the final selection's history; the
    aliases follow it and take no entry of their own.  Where the search
    never timed the kernel under a config of its own, the first alias
    stands in and is the one timed."""
    search, timed = _retiming_search(collapse=True)
    kernel = _kernel(search, 10.0, 1)
    alias = _alias(kernel, 2)
    second_alias = _alias(kernel, 3)
    other = _kernel(search, 11.0, 7)
    search.population = [alias, second_alias, other]

    search.rebenchmark_population()

    assert timed == [[kernel, other]]
    assert kernel.perfs == alias.perfs == second_alias.perfs == [10.0, 5.0]
    entry = search._benchmarked_members[kernel.config]
    assert entry.perfs == [10.0, 5.0]

    unseen = _member(8.0, 4)
    first = _alias(unseen, 5)
    second = _alias(unseen, 6)
    search.population = [first, second, other]
    search.rebenchmark_population()
    assert timed[1:] == [[first, other]]
    assert first.perfs == second.perfs == [8.0, 4.0]
    assert {config.num_warps for config in search._benchmarked_members} == {1, 7}


def test_a_list_that_is_one_kernel_is_not_retimed() -> None:
    """A re-timing compares kernels, and the real ``rebenchmark`` leaves a
    single member as it is.  A list that holds one kernel under several
    configs is that case: it takes no timing, and its configs keep the one
    reading they share."""
    search = _holding_search(collapse=True)
    search.settings = SimpleNamespace(get_rebenchmark_threshold=lambda: 1.5)  # type: ignore[assignment]
    search.best_perf_so_far = 10.0
    kernel = _kernel(search, 10.0, 1)
    alias = _alias(kernel, 2)
    search.population = [kernel, alias]
    timings = Mock()
    search._isolated_rebenchmark_results = timings  # type: ignore[method-assign]
    search._apply_rebenchmark_timings = timings  # type: ignore[method-assign]

    # (the real ``rebenchmark``, which returns before it times one member)
    search.rebenchmark_population([kernel, alias])

    timings.assert_not_called()
    assert kernel.perfs == alias.perfs == [10.0]


def test_an_alias_left_out_of_a_retiming_follows_its_kernel() -> None:
    """An initial population keeps an alias of the kernel a polish round
    then verifies without it (the round re-times its candidates only).  The
    alias takes the kernel's new reading all the same: no member of the
    population keeps a reading its kernel no longer has, so the best of the
    population is never a config of a kernel at a reading of the past."""
    search, timed = _retiming_search(collapse=True)
    kernel = _kernel(search, 10.0, 1)
    alias = _alias(kernel, 2)
    new = _member(10.2, 3)
    search.population = [alias, kernel]

    search.rebenchmark_population([kernel, new], desc="Polish round 1: verifying")

    assert timed == [[kernel, new]]
    assert alias.perfs == [10.0, 5.0]
    assert search.best.perf == kernel.perf == 5.0


@pytest.mark.parametrize("collapse", [True, False])
def test_compare_times_two_configs_of_one_kernel_once(collapse: bool) -> None:
    """``compare`` (the differential-evolution searches' replacement rule)
    re-times a pair near the best.  A child that is its parent's kernel
    under another config takes no timing where aliases are dropped: the two
    are equal, and the parent stays."""
    search = _holding_search(collapse=collapse)
    search.settings = SimpleNamespace(get_rebenchmark_threshold=lambda: 1.5)  # type: ignore[assignment]
    search.best_perf_so_far = 10.0
    parent = _member(10.0, 1)
    child = _alias(parent, 2)
    other = _member(10.5, 3)
    timed: list[list[PopulationMember]] = []

    def rebenchmark(members: list[PopulationMember], **kwargs: object) -> None:
        if len(members) < 2:
            return
        timed.append(list(members))
        for index, member in enumerate(members):
            # noise: the second of a pair reads lower
            member.perfs.append(member.perf * (1.0 - 0.01 * index))

    with patch.object(search, "rebenchmark", side_effect=rebenchmark):
        same_kernel = search.compare(child, parent)
        different = search.compare(other, parent)

    assert different == 1
    if collapse:
        assert timed == [[other, parent]]
        assert same_kernel == 0
        assert child.perfs == [10.0]
    else:
        # every alias is a candidate of its own: timed against its parent
        assert timed == [[child, parent], [other, parent]]
        assert same_kernel == 1


def test_compare_of_one_kernel_is_equal_whatever_its_configs_read() -> None:
    """A differential-evolution population keeps its aliases, and ``compare``
    re-times only the pair it is given: the parent was re-timed against
    another member (12.0) while a child that repeats its code still carries
    the reading it was settled with (10.0).  The two are one kernel, so they
    are equal and the parent stays, in either order, and so are two aliases
    of one kernel whose holder is a third member."""
    search = _holding_search(collapse=True)
    search.settings = SimpleNamespace(get_rebenchmark_threshold=lambda: 1.5)  # type: ignore[assignment]
    search.best_perf_so_far = 10.0
    search.rebenchmark = Mock()  # type: ignore[method-assign]
    parent = _member(10.0, 1)
    child = _alias(parent, 2)
    sibling = _alias(parent, 3)
    parent.perfs.append(12.0)
    sibling.perfs.append(11.0)
    assert child.perf < parent.perf

    assert search.compare(child, parent) == 0
    assert search.compare(parent, child) == 0
    search.population = [parent]
    assert search.compare(child, sibling) == 0
    search.rebenchmark.assert_not_called()


def test_compare_reads_a_pending_alias_as_its_kernel_reads_now() -> None:
    """A differential-evolution generation that benchmarks its candidates
    first and compares them afterwards (``immediate_update=False``) settles
    every candidate with the reading its kernel has then.  An earlier
    comparison of the batch re-times a kernel (10 -> 20); a later candidate
    that repeats that kernel's code is no member of the population yet, so
    the re-timing did not reach it and it still reads 10.  ``compare`` reads
    it as its kernel reads now: it does not replace a parent that reads 12
    for a reading its kernel no longer has."""
    search = _holding_search(collapse=True)
    # nothing is near enough to the best to be re-timed
    search.settings = SimpleNamespace(get_rebenchmark_threshold=lambda: 0.9)  # type: ignore[assignment]
    search.best_perf_so_far = 10.0
    search.rebenchmark = Mock()  # type: ignore[method-assign]
    kernel = _kernel(search, 10.0, 1)
    candidate = _alias(kernel, 2)  # settled before its kernel was re-timed
    parent = _kernel(search, 12.0, 3)
    search.population = [kernel, parent]
    kernel.perfs.append(20.0)

    assert search.compare(candidate, parent) == 1
    assert candidate.perfs == [10.0, 20.0]
    # ... and in the other order
    stale = _alias(kernel, 4)
    stale.perfs[-1] = 10.0
    assert search.compare(parent, stale) == -1
    search.rebenchmark.assert_not_called()


@pytest.mark.parametrize("collapse", [True, False])
def test_one_member_per_kernel_keeps_the_config_that_holds_it(collapse: bool) -> None:
    kernel = _member(10.0, 1)
    ahead = _alias(kernel, 2)
    behind = _alias(kernel, 3)
    other = _member(10.5, 4)
    lone = _member(11.0, 5, "deduplicated")  # its kernel is in no collection
    lone_again = _alias(lone, 6)
    search = _holding_search(collapse=collapse)
    members = [ahead, kernel, behind, other, lone, lone_again]
    search.population = members

    kept = search._one_member_per_kernel(members)

    assert kept == ([kernel, other, lone] if collapse else members)


@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("search_type", [PatternSearch, LFBOTreeSearch])
def test_no_two_paths_start_from_one_kernel(
    search_type: type[PatternSearch], collapse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seed and the default config differ in a setting the code does not
    depend on: after the verification the alias reads what its kernel reads
    and sorts beside it.  Each of the two copies starts from a kernel of its
    own: a path from the alias would generate the first path's neighborhood
    again, every neighbor a repeat."""
    default = _member(10.0, 1)
    seed = _alias(default, 2)
    third = _member(10.5, 3)
    search = _holding_search(collapse=collapse, search_type=search_type)
    search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
    search.copies = 2
    search.max_generations = 1
    search.similarity_penalty = 1.0  # type: ignore[attr-defined]
    search.compile_timeout_lower_bound = 0.0
    search.compile_timeout_quantile = 0.0
    search.log = Mock()
    search.log.autotune_tracing.return_value = nullcontext()
    search.kernel = Mock(env=SimpleNamespace(process_group_name=None))
    search.config_gen = Mock(encode_config=lambda flat: flat)
    search._autotune_metrics = AutotuneMetrics()
    members = {m.flat_values: m for m in (seed, default, third)}
    starts: list[PopulationMember] = []

    def path(*args: object, **kwargs: object) -> Iterator[list[PopulationMember]]:
        starts.append(next(arg for arg in args if isinstance(arg, PopulationMember)))
        return iter(())

    stubs = {
        "_generate_initial_population_flat": Mock(return_value=[*members]),
        "make_unbenchmarked": Mock(side_effect=members.get),
        "benchmark_initial_population": Mock(),
        "_random_fallback_population_target": Mock(return_value=None),
        "set_adaptive_compile_timeout": Mock(),
        "capture_compiler_seed_members": Mock(),
        "rebenchmark_population": Mock(),
        "_pattern_search_from": Mock(side_effect=path),
        "_pruned_pattern_search_from": Mock(side_effect=path),
        "_budgeted_range": lambda *args: iter(range(*args)),
        "_finalize": Mock(return_value=default.config),
        # LFBO only:
        "_append_training_sample": Mock(),
        "_fit_surrogate": Mock(),
        "_run_flash_structural_qualification": Mock(return_value=0),
        "_autotune_budget_exceeded_across_ranks": Mock(return_value=False),
        "_polish_descent": Mock(),
    }
    for name, stub in stubs.items():
        setattr(search, name, stub)
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.check_population_consistency", Mock()
    )

    search._autotune()

    assert starts == ([default, third] if collapse else [seed, default])


@pytest.mark.parametrize("collapse", [True, False])
def test_an_alias_trains_the_surrogate_with_its_kernels_reading(collapse: bool) -> None:
    """A generation drops its aliases from the population, so no re-timing
    reaches them, but the surrogate keeps their rows.  The row of an alias
    takes the reading of the member that holds its kernel: one kernel does
    not train the surrogate with a label of each verification."""
    search = cast(
        "LFBOTreeSearch", _holding_search(collapse=collapse, search_type=LFBOTreeSearch)
    )
    search._final_rebenchmark_top_k = lambda: 8  # type: ignore[method-assign]
    kernel = _kernel(search, 16.4, 1)
    alias = _alias(kernel, 2)
    search.train_configs = None
    search.train_y = [16.4, 16.4, 20.0]
    search._train_members = [kernel, alias, None]
    kernel.perfs.append(14.3)  # a verification since; the alias is no member

    search._sync_training_labels()

    assert search.train_y == [14.3, 14.3 if collapse else 16.4, 20.0]


@pytest.mark.parametrize("in_population", [True, False])
def test_a_repaired_failure_trains_the_surrogate_with_its_kernels_reading(
    in_population: bool,
) -> None:
    """A config fails (a timeout) and its row trains the surrogate as a
    failure.  A later config with the same generated source is timed, which
    repairs the failed config: its code works and reads what that kernel
    reads.  The row follows, also when the failed member left the population
    generations ago (the surrogate's rows are all that still hold it)."""
    search = cast(
        "LFBOTreeSearch", _pinning_search(collapse=True, search_type=LFBOTreeSearch)
    )
    search._compiler_seed_members = []
    search.train_configs = None
    failed = _member(float("inf"), 2, "timeout")
    search.population = [failed] if in_population else []
    search.train_y = [float("inf"), float("inf"), float("inf")]
    other_failure = _member(float("inf"), 4, "error")
    search._train_members = [failed, other_failure, None]
    kernel = _member(None, 3)
    result = SimpleNamespace(
        config=kernel.config, perf=9.0, fn=kernel.fn, status="ok", compile_time=1.0
    )
    repairs = {
        failed.config: SimpleNamespace(
            config=failed.config,
            perf=9.0,
            fn=kernel.fn,
            status="deduplicated",
            compile_time=None,
        )
    }
    search.benchmark_provider = _provider(  # type: ignore[assignment]
        take_effective_source_repairs=lambda: repairs
    )
    with (
        patch.object(search, "benchmark_batch", return_value=[result]),
        patch.object(search, "_final_rebenchmark_top_k", return_value=8),
    ):
        search.benchmark_population([kernel])
    kernel.perfs.append(8.0)  # a verification since

    search._sync_training_labels()

    assert (failed.status, failed.fn) == ("deduplicated", kernel.fn)
    # The repaired row has its kernel's reading; a config that failed and
    # was not repaired, and a seeded row without a member, stay as they are.
    assert search.train_y == [8.0, float("inf"), float("inf")]
    # The repaired config is an alias: no entry of its own in the history.
    assert {config.num_warps for config in search._benchmarked_members} == {3}


@pytest.mark.parametrize("search_type", [PatternSearch, LFBOTreeSearch])
def test_a_search_keeps_one_holder_and_one_reading_per_kernel(
    search_type: type[PatternSearch], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole search over a stand-in space, with the real generation loop,
    path generators, polish descent, settling and re-timing.  A config is
    ``(code, dead)`` and its generated code depends on ``code`` alone, so
    two thirds of the space repeat earlier code; every timing reads a
    kernel a little differently.  Whatever the search does:

    - a re-timing times a kernel at most once;
    - after every batch and every re-timing, each member of the population
      has the reading of the member that holds its kernel, and so has each
      row the surrogate trains on;
    - the final selection's history holds a kernel once;
    - no two paths start from one kernel.
    """
    noise = random.Random(0)
    fns: dict[int, Callable[[], None]] = {}
    first_readings: dict[int, float] = {}
    members: dict[tuple[int, int], PopulationMember] = {}
    # the member each kernel was timed under: the one that holds it
    timed_under: dict[int, PopulationMember] = {}

    def code_of(fn: object) -> int:
        return next(code for code, held in fns.items() if held is fn)

    def reading(fn: object) -> float:
        return (10.0 + abs(code_of(fn) - 6)) * noise.uniform(0.97, 1.03)

    def make(flat: tuple[int, int]) -> PopulationMember:
        if flat not in members:
            config = helion.Config(block_sizes=[flat[0]], num_warps=flat[1])
            members[flat] = PopulationMember(lambda: None, [], flat, config)
        return members[flat]

    def neighbors(base: tuple[int, int]) -> list[tuple[int, int]]:
        code, dead = base
        steps = [(code - 1, dead), (code + 1, dead), (code, dead - 1), (code, dead + 1)]
        return [(c, d) for c, d in steps if 0 <= c < 12 and 0 <= d < 3]

    def benchmark_batch(configs: list[helion.Config], **kwargs: object) -> list[object]:
        results = []
        for config in configs:
            code = config.block_sizes[0]
            alias = code in fns
            if not alias:
                fns[code] = lambda: None
                timed_under[code] = members[code, config.num_warps]
                first_readings[code] = reading(fns[code])
            results.append(
                SimpleNamespace(
                    config=config,
                    perf=first_readings[code],
                    fn=fns[code],
                    status="deduplicated" if alias else "ok",
                    compile_time=None if alias else 1.0,
                )
            )
        return results

    search = cast(
        "PatternSearch", _holding_search(collapse=True, search_type=search_type)
    )

    def consistent() -> None:
        for member in search.population:
            if member.perfs:
                assert member.perf == timed_under[code_of(member.fn)].perf, member
        history = [*search._benchmarked_members.values()]
        assert len({id(member.fn) for member in history}) == len(history)

    def rebenchmark(timed: list[PopulationMember], **kwargs: object) -> None:
        # (as the real one: a single member is left as it is)
        if len(timed) < 2:
            return
        assert len({id(member.fn) for member in timed}) == len(timed)
        assert all(member is timed_under[code_of(member.fn)] for member in timed)
        for member in timed:
            member.perfs.append(reading(member.fn))
            search.best_perf_so_far = min(search.best_perf_so_far, member.perf)
        search._refresh_benchmarked_members_after_rebenchmark(timed)

    def checked(method: Callable[..., object]) -> Callable[..., object]:
        def run(*args: object, **kwargs: object) -> object:
            result = method(*args, **kwargs)
            consistent()
            return result

        return run

    starts: list[PopulationMember] = []

    def recording(method: Callable[..., object]) -> Callable[..., object]:
        def run(*args: object, **kwargs: object) -> object:
            starts.append(
                next(arg for arg in args if isinstance(arg, PopulationMember))
            )
            return method(*args, **kwargs)

        return run

    def fit_surrogate() -> None:
        search._sync_training_labels()  # type: ignore[attr-defined]
        rows = zip(search._train_members, search.train_y, strict=True)  # type: ignore[attr-defined]
        for member, label in rows:
            assert label == timed_under[code_of(member.fn)].perf

    search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
    search.copies = 3
    search.max_generations = 4
    search.min_improvement_delta = 0.001
    search.compile_timeout_lower_bound = 0.0
    search.compile_timeout_quantile = 0.0
    search.settings = SimpleNamespace(get_rebenchmark_threshold=lambda: 1.5)  # type: ignore[assignment]
    search.best_perf_so_far = float("inf")
    search.log = Mock()
    search.log.autotune_tracing.return_value = nullcontext()
    search.kernel = Mock(env=SimpleNamespace(process_group_name=None))
    search.config_gen = Mock(
        encode_config=lambda flat: flat,
        flatten=lambda config: (
            config.block_sizes[0],
            config.num_warps,
        ),
    )
    search._autotune_metrics = AutotuneMetrics()
    search._benchmarking_initial_population = False
    search.benchmark_provider = _provider(take_effective_source_repairs=list)  # type: ignore[assignment]
    # LFBO only:
    search.similarity_penalty = 1.0  # type: ignore[attr-defined]
    search.patience = 0  # type: ignore[attr-defined]
    search.frac_selected = 1.0  # type: ignore[attr-defined]
    search.polish_rounds = 4  # type: ignore[attr-defined]
    search.train_x, search.train_y, search._train_members = [], [], []  # type: ignore[attr-defined]
    search.train_configs = None  # type: ignore[attr-defined]
    stubs = {
        # the seed (2, 0), the default (2, 1) that repeats its code, others
        "_generate_initial_population_flat": lambda: [(2, 0), (2, 1), (9, 0), (2, 2)],
        "make_unbenchmarked": make,
        "_generate_neighbors": lambda base, **kwargs: neighbors(base),
        "benchmark_batch": benchmark_batch,
        "benchmark_initial_population": lambda population, **kwargs: (
            search.benchmark_population(population)
        ),
        "benchmark_population": checked(search.benchmark_population),
        "rebenchmark": rebenchmark,
        "_rebenchmark_kernels": checked(search._rebenchmark_kernels),
        "_final_rebenchmark_top_k": lambda: 8,
        "_random_fallback_population_target": lambda: None,
        "set_adaptive_compile_timeout": Mock(),
        "capture_compiler_seed_members": Mock(),
        "_budgeted_range": lambda *args: iter(range(*args)),
        "set_generation": Mock(),
        "format_performance": lambda perf: f"{perf:.4f}",
        "_finalize": lambda: search.best.config,
        "_pattern_search_from": recording(search._pattern_search_from),
        # LFBO only:
        "_fit_surrogate": fit_surrogate,
        "_surrogate_select": lambda candidates, n: candidates,
        "_run_flash_structural_qualification": Mock(return_value=0),
        "_autotune_budget_exceeded_across_ranks": Mock(return_value=False),
        "_path_exhausts_generation_budget": Mock(return_value=False),
    }
    if search_type is LFBOTreeSearch:
        stubs["_pruned_pattern_search_from"] = recording(
            search._pruned_pattern_search_from  # type: ignore[attr-defined]
        )
    for name, stub in stubs.items():
        setattr(search, name, stub)
    # (the polish descent asks the class for the whole neighborhood)
    monkeypatch.setattr(
        PatternSearch, "_generate_neighbors", lambda self, base: neighbors(base)
    )
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.check_population_consistency", Mock()
    )

    search._autotune()

    # Two paths, not three: the seed and the default are one kernel.
    assert sorted(start.flat_values for start in starts) == [(2, 0), (9, 0)]
    # The search timed 12 kernels at most and met repeats of them.
    assert len(members) > len(fns) > 4


@pytest.mark.parametrize("collapse", [True, False])
def test_llm_guided_results_follow_the_alias_rules(collapse: bool) -> None:
    """The LLM-guided search builds its members from results itself.  They
    take the step every benchmarked member takes
    (``_settle_benchmarked_members``): an alias has its kernel's reading
    and no entry in the final selection's history, and a pinned one passes
    its pin on."""
    search = _holding_search(collapse=collapse, search_type=LLMGuidedSearch)
    search.config_gen = SimpleNamespace(flatten=lambda config: [config.num_warps])  # type: ignore[assignment]
    search.configs_per_round = 8
    search._latest_results_by_config_key = {}
    search._config_key = lambda config: repr(config)  # type: ignore[method-assign]
    search._final_rebenchmark_top_k = lambda: 8  # type: ignore[method-assign]
    kernel = _kernel(search, 10.0, 1)
    kernel.perfs.append(8.0)  # re-timed since the provider recorded it
    search.population = [kernel]
    seed = helion.Config(num_warps=2)
    search._pinned_finalist_configs.add(seed)
    results = [
        SimpleNamespace(
            config=seed,
            perf=10.0,
            fn=kernel.fn,
            status="deduplicated",
            compile_time=None,
        ),
        SimpleNamespace(
            config=helion.Config(num_warps=3),
            perf=9.0,
            fn=lambda: None,
            status="ok",
            compile_time=1.0,
        ),
    ]

    search._ingest_results(results)  # type: ignore[arg-type]

    alias, fresh = search.population[1:]
    assert fresh.perf == 9.0
    history = {config.num_warps for config in search._benchmarked_members}
    pins = {config.num_warps for config in search._pinned_finalist_members}
    if collapse:
        assert alias.perf == 8.0
        assert history == {1, 3}
        assert pins == {1} and kernel.config in search._pinned_finalist_configs
    else:
        assert alias.perf == 10.0
        assert history == {1, 2, 3}
        assert pins == {2}


def test_finishing_round_takes_dead_resets_without_a_timing() -> None:
    """``run_finishing_phase`` resets settings to their defaults one at a
    time, then all the resets that kept the performance at once.  A reset
    the generated code does not depend on is the incumbent's kernel again:
    the kernel is timed once in the round, such a reset takes its reading
    and counts as one that kept the performance, and the combined reset of
    two of them takes no timing either."""
    search = _holding_search(collapse=True)
    search.args = ()
    search.log = Mock()
    search._budgeted_range = lambda *args: iter(range(*args))  # type: ignore[method-assign]
    search._autotune_metrics = SimpleNamespace(num_generations=3)  # type: ignore[assignment]
    search.set_generation = Mock()  # type: ignore[method-assign]

    def config(flat: list[int]) -> helion.Config:
        return helion.Config(
            block_sizes=[flat[0]], num_warps=flat[1], num_stages=flat[2]
        )

    current = PopulationMember(
        lambda: None, [10.0], [2, 5, 7], config([2, 5, 7]), status="ok"
    )

    def make(flat: list[int]) -> PopulationMember:
        return PopulationMember(lambda: None, [], list(flat), config(flat))

    def benchmark(members: list[PopulationMember], desc: str = "") -> None:
        for member in members:
            if member.flat_values[0] == 2:
                # num_warps and num_stages are dead: the incumbent's kernel
                member.perfs.append(current.perf)
                member.status = "deduplicated"
                member.fn = current.fn
            else:
                member.perfs.append(12.0)
                member.status = "ok"

    timed: list[list[list[int]]] = []

    def rebenchmark(members: list[PopulationMember], **kwargs: object) -> None:
        if len(members) < 2:
            return
        timed.append([list(member.flat_values) for member in members])
        for member in members:
            member.perfs.append(member.perf * 1.01)

    search.config_gen = SimpleNamespace(  # type: ignore[assignment]
        default_flat=lambda: [1, 4, 6],
        flatten=lambda config: [
            config.block_sizes[0],
            config.num_warps,
            config.num_stages,
        ],
    )
    search.make_unbenchmarked = make  # type: ignore[method-assign]
    search.benchmark_population = benchmark  # type: ignore[method-assign]
    search.rebenchmark = rebenchmark  # type: ignore[method-assign]
    with patch.object(helion.Config, "minimize", lambda self, spec: self):
        result = search.run_finishing_phase(current, 1)

    # one timing of the incumbent's kernel beside the reset that is other code
    assert timed == [[[2, 5, 7], [1, 5, 7]]]
    assert result.flat_values == [2, 4, 6]
    assert result.perfs == [10.0 * 1.01]
