from __future__ import annotations

import ast
import dataclasses
import functools
import random
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.analysis.separable_index_map import prove_separable_index_map
from helion._compiler.cute.autotune import config_generation_ext
from helion._compiler.cute.cute_mma_dag import _Matcher
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_paged_attention
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import without_paired_head
from helion._testing.cute_paged_v_lag import without_retirement_seeds
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.effort_profile import RANDOM_SEARCH_DEFAULTS
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.random_search import RandomSearch
import helion.language as hl
from helion.language import tile_ops
from helion.language.memory_ops import load

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _shifted_folded_gather(
    storage: torch.Tensor,
    weight: torch.Tensor,
    row_begin: int,
    item_begin: int,
) -> torch.Tensor:
    row_begin = hl.specialize(row_begin)
    item_begin = hl.specialize(item_begin)
    output = torch.empty((2, 40, 128), dtype=storage.dtype, device=storage.device)
    for item, rows in hl.tile(
        [item_begin, row_begin], [item_begin + 2, row_begin + 40], block_size=[1, None]
    ):
        channel = hl.arange(128)
        offset = (
            item.begin * 16384
            + (rows.index // 8)[:, None] * 2048
            + (rows.index % 8)[:, None] * 128
            + channel[None, :]
        )
        value = storage[offset]
        output[item.id - item_begin, rows.index - row_begin, :] = hl.dot(
            value, weight[:, :]
        )
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _shifted_folded_iota_gather(
    storage: torch.Tensor, weight: torch.Tensor, item_begin: int
) -> torch.Tensor:
    item_begin = hl.specialize(item_begin)
    output = torch.empty((2, 64, 128), dtype=storage.dtype, device=storage.device)
    for item in hl.tile(item_begin, item_begin + 2, block_size=1):
        rows = hl.arange(64)
        channel = hl.arange(128)
        offset = (
            item.begin * 16384
            + (rows // 8)[:, None] * 2048
            + (rows % 8)[:, None] * 128
            + channel[None, :]
        )
        output[item.id - item_begin, :, :] = hl.dot(storage[offset], weight[:, :])
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _nonlead_folded_gather(storage: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty((64, 128), dtype=storage.dtype, device=storage.device)
    for rows in hl.tile(64):
        channel = hl.arange(128)
        offset = (
            rows.begin * 256
            + (rows.index // 8)[:, None] * 2048
            + (rows.index % 8)[:, None] * 128
            + channel[None, :]
        )
        output[rows, :] = hl.dot(storage[offset], weight[:, :])
    return output


@functools.lru_cache(None)
def _args(query_len: int = 5, variant: str = "plain") -> tuple[object, ...]:
    batch, heads, dim, pages = 3, 16, 128, 7
    column_stride = 2 if variant == "strided_columns" else 1
    head_stride = dim * column_stride
    query_stride = 4608
    query_batch = query_len * query_stride
    offset = 8 if variant == "padded_output" else 0
    q = torch.empty(query_batch * batch + offset, dtype=torch.bfloat16).as_strided(
        (batch, query_len, heads, dim),
        (query_batch, query_stride, head_stride, column_stride),
        offset,
    )
    out_query = 1024 if variant == "overlap" else 2048
    if variant == "padded_output":
        out_query += 128
    out_batch = query_len * out_query + (256 if variant == "padded_output" else 0)
    out = torch.empty(out_batch * batch + 2048 + offset, dtype=q.dtype).as_strided(
        q.shape, (out_batch, out_query, 128, 1), offset
    )
    storage = [torch.empty(pages * 33792, dtype=torch.int8) for _ in range(2)]
    k, v = [x.as_strided((pages, 128, 2, 128), (33792, 256, 128, 1)) for x in storage]
    ks, vs = [
        x.view(torch.uint8).as_strided((pages, 128, 2, 4), (33792, 4, 512, 1), 32768)
        for x in storage
    ]
    return (
        q,
        k,
        v,
        ks,
        vs,
        torch.empty((batch, 17), dtype=torch.int32),
        torch.empty(batch, dtype=torch.int32),
        31,
        128,
        True,
        out,
    )


@functools.lru_cache(None)
def _bound(query_len: int = 5, variant: str = "plain") -> BoundKernel:
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(folded_paged_attention, _args(query_len, variant))


def _config(
    query_len: int = 5, mode: str = "tcgen05", schedule: str = "attention"
) -> helion.Config:
    return helion.Config(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag=mode,
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps=schedule,
        cute_min_blocks_per_mp=1,
        cute_pdl=False,
    )


@pytest.mark.parametrize("query_len,variant", [(5, "plain"), (1, "padded_output")])
def test_folded_attention_emits_one_direct_kernel(query_len: int, variant: str) -> None:
    bound = _bound(query_len, variant)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(_config(query_len))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paged_attention_partial_body"
    ]
    assert len(calls) == 1
    layouts = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "io_layouts"
    )
    query, output, lse = ast.literal_eval(layouts)
    assert query == (query_len * 4608, 4608, 128, 1)
    out_query = 2176 if variant == "padded_output" else 2048
    assert output == (
        query_len * out_query + (256 if variant == "padded_output" else 0),
        out_query,
        128,
        1,
    )
    assert lse == (query_len * 16, 16, 1)
    assert source.count("@cute.kernel") == 1
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.mapped_io


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_folded_attention_rejects_ordinary_mma_routes(mode: str) -> None:
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(
            (helion.exc.InvalidConfig, helion.exc.BackendUnsupported),
            match="folded storage",
        ),
    ):
        _bound(5, "plain").to_code(_config(mode=mode, schedule="off"))


@pytest.mark.parametrize("variant", ["overlap", "strided_columns"])
def test_folded_attention_declines_unimplemented_physical_layouts(variant: str) -> None:
    assert _bound(5, variant).config_spec.cute_mma_dag_facts is None


def test_folded_attention_rejects_shifted_output_request_head_map() -> None:
    bound = _bound(5, "plain")
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None and semantic.output.index_map is not None
        output_map = semantic.output.index_map
        shifted = dataclasses.replace(output_map, base=output_map.base + 128)
        semantic = dataclasses.replace(
            semantic, output=dataclasses.replace(semantic.output, index_map=shifted)
        )
        assert plan_paged_attention_transport(bound.env, semantic) is None


def test_folded_attention_exposes_only_implemented_mma_seeds() -> None:
    bound = _bound(5, "plain")
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.mapped_io
    assert facts.attention is not None
    with bound.env, bound.host_function:
        seeds = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
        seeds = without_paired_head(without_retirement_seeds(seeds), facts)
        # Only the paired schedule has the guarded capture default.
        unpaired = (False, False, "auto")
        paired = (True, True, "loader_tail")
        assert [
            (
                seed.get("cute_mma_dag_pass_warps"),
                seed.get("cute_min_blocks_per_mp"),
                seed.get("cute_mma_dag_v_lag", 0),
                seed.get("cute_mma_dag_sleep_waits", False),
                seed.get("cute_mma_dag_query_tma", False),
                tuple(
                    seed.get("load_eviction_policies")[slot]
                    for slot in facts.attention.tma_load_slots
                ),
                seed.get("cute_pdl", False),
                seed.get("cute_pdl_capture_overlap", False),
                seed.get("cute_pdl_release", "auto"),
            )
            for seed in seeds
        ] == [
            (*schedule, False, False, (policy,) * 4, *pdl)
            for schedule, pdl in [
                (("attention", 1, 0), unpaired),
                (("attention", 2, 0), unpaired),
                (("attention_scoped", 2, 0), unpaired),
                *((("attention_paired", 1, lag), paired) for lag in range(4)),
            ]
            for policy in ("", "streaming", "l2_last")
        ] + [
            ("attention_paired", 1, lag, True, False, ("",) * 4, *paired)
            for lag in range(4)
        ] + [
            ("attention_paired", 1, lag, False, True, ("",) * 4, *paired)
            for lag in range(4)
        ]
        for seed in seeds:
            config = bound.config_spec.normalized_config(seed)
            assert config.get("cute_pdl_capture_overlap", False) is seed.get(
                "cute_pdl_capture_overlap", False
            )
            assert config.get("cute_pdl_release", "auto") == seed.get(
                "cute_pdl_release", "auto"
            )
            assert config.get("cute_mma_dag_query_tma", False) == seed.get(
                "cute_mma_dag_query_tma", False
            )
            assert config.get("cute_mma_dag_sleep_waits", False) == seed.get(
                "cute_mma_dag_sleep_waits", False
            )
            assert config.get("cute_mma_dag_pass_warps") == seed.get(
                "cute_mma_dag_pass_warps"
            )
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, config
            )


def test_folded_attention_without_a_config_runs_the_paired_family() -> None:
    # Folded storage admits the attention schedules alone, and the attention
    # block is the whole seed list: its paired head is the default.
    bound = _bound(5, "plain")
    spec = bound.config_spec
    promoted = spec.compiler_default_config
    assert promoted is not None and promoted == spec.compiler_seed_configs[0]
    assert promoted.get("cute_mma_dag_pass_warps") == "attention_paired"
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148):
        with bound.env, bound.host_function:
            default = spec.default_config()
        source = bound.to_code(default)
        assert source == bound.to_code(helion.Config.from_dict(dict(promoted.config)))
    assert default.get("cute_mma_dag_pass_warps") == "attention_paired"
    assert default.get("cute_pdl_capture_overlap") is True
    assert "paged_attention_paired_body(" in source
    assert "paged_attention_partial_body(" not in source
    assert "capture_release_tail=True" in source


def test_folded_attention_population_leaves_out_generic_random_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Folded storage leaves the attention schedules as the only lowering
    # besides the generic one, which a random draw almost always renders and
    # which cannot compete with the seeds.
    bound = _bound(5, "plain")
    spec = bound.config_spec
    assert spec.cute_mma_dag_facts is not None and spec.cute_mma_dag_facts.mapped_io
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        generation = ConfigGeneration(spec)
        seeds = [config for _flat, config in generation.seed_flat_config_pairs()]
        assert len(seeds) >= config_generation_ext.MIN_SEEDS_WITHOUT_RANDOM_DRAWS
        random.seed(0)
        population = [
            generation.unflatten(flat)
            for flat in generation.random_population_flat(100)
        ]
        # The fragment default (the search's reference) leads, then the seeds;
        # the few draws that named an attention schedule follow.
        assert population[0].get("cute_mma_dag") == "off"
        assert population[1 : 1 + len(seeds)] == seeds
        assert len(population) < 100
        assert all(config.get("cute_mma_dag") == "tcgen05" for config in population[1:])
        # Too few seeds to stand in for a population: the draws stay.
        monkeypatch.setattr(
            config_generation_ext, "MIN_SEEDS_WITHOUT_RANDOM_DRAWS", len(seeds) + 1
        )
        random.seed(0)
        padded = ConfigGeneration(spec).random_population_flat(100)
        assert len(padded) == 100


@pytest.mark.parametrize(
    "overrides,scheduled",
    [
        # Overrides that exclude the attention schedules, or narrow them to a
        # few seeds, leave the generic lowering as the surface to search:
        ({"cute_mma_dag": "off"}, False),
        ({"cute_min_blocks_per_mp": 0}, False),
        ({"cute_mma_dag_pass_warps": "attention_scoped"}, False),
        # one that keeps the seeds in the search keeps the short population.
        ({"cute_pdl_spread": False}, True),
    ],
)
def test_folded_attention_population_follows_the_seeds_overrides_admit(
    overrides: dict[str, object], scheduled: bool
) -> None:
    # The seeds stand in for the random draws only where they start the
    # search: they are counted as the generation transfers them.
    bound = _bound(5, "plain")
    spec = bound.config_spec
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        generation = ConfigGeneration(spec, overrides=overrides)
        seeds = [
            config
            for _flat, config in generation.seed_flat_config_pairs()
            if config.get("cute_mma_dag") != "off"
        ]
        enough = len(seeds) >= config_generation_ext.MIN_SEEDS_WITHOUT_RANDOM_DRAWS
        assert enough is scheduled
        random.seed(0)
        population = generation.random_population_flat(100)
        if scheduled:
            assert len(seeds) < len(population) < 100
        else:
            # A search needs these starting points: differential evolution
            # draws four members to mutate one.
            assert len(population) == 100


class _OpaqueChoice:
    """A choice of a user's tunable that config handling leaves as it is: it
    compares by identity and refuses to be copied."""

    def __deepcopy__(self, memo: dict[int, object]) -> _OpaqueChoice:
        raise AssertionError("a copy hook of an opaque config value ran")


def test_folded_attention_population_copies_no_opaque_choice() -> None:
    # A user's tunable may hold choices that are opaque to config handling,
    # which copies the config containers and leaves such a value as it is
    # (``_copy_config_structure``). The rule that declines a draw decodes a
    # copy of it. That copy holds the choices themselves: no copy hook runs,
    # and the generic draws are left out as they are without the tunable.
    choices = (_OpaqueChoice(), _OpaqueChoice())
    with cpu_render_target(fake_device_sm_count=148):
        # A bind of its own: the tunable joins its config spec.
        bound = cpu_bind(folded_paged_attention, _args(5, "plain"))
        spec = bound.config_spec
        spec.user_defined_tunables["selected"] = EnumFragment(choices)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            generation = ConfigGeneration(spec)
            random.seed(0)
            population = [
                generation.unflatten(flat)
                for flat in generation.random_population_flat(100)
            ]
            assert len(population) < 100
            assert all(config.get("cute_mma_dag") != "off" for config in population[1:])
            assert all(
                any(config["selected"] is choice for choice in choices)
                for config in population
            )


def _drops_the_seeds(config: helion.Config) -> helion.Config | None:
    # Every seed of the fixture has a block of 64.
    return config if max(config.block_sizes) <= 16 else None


class _PassesASeedOnce:
    """A filter that keeps state: it passes a config on an attention
    schedule the first time it is asked about it, and drops every other
    config and every repeat."""

    def __init__(self) -> None:
        self.passed: set[helion.Config] = set()

    def __call__(self, config: helion.Config) -> helion.Config | None:
        if config.get("cute_mma_dag") == "off" or config in self.passed:
            return None
        self.passed.add(config)
        return config


@pytest.mark.parametrize(
    "new_filter,passes_the_seeds",
    [
        # Passes no seed, no witness and not the default, only draws.
        (lambda: _drops_the_seeds, False),
        # Leaves every candidate as it is.
        (lambda: lambda config: config, True),
        # Asked about a seed a second time, it drops the seed.
        (_PassesASeedOnce, True),
    ],
    ids=["drops_the_seeds", "passes_every_config", "passes_a_seed_once"],
)
def test_folded_attention_population_keeps_the_draws_behind_a_filter(
    monkeypatch: pytest.MonkeyPatch,
    new_filter: Callable[[], Callable[[helion.Config], helion.Config | None]],
    passes_the_seeds: bool,
) -> None:
    # A seed that ``autotune_config_filter`` drops or rewrites starts no
    # search, so it stands in for no draw. Which seeds those are is the
    # filter's answer when the population is benchmarked, and the filter
    # may keep state, so a search with a filter keeps its draws and asks
    # the filter nothing while it builds its population. Draws declined
    # behind seeds that the filter drops leave a search nothing to time,
    # and seeds shown to a filter that passes each config once are dropped
    # when they are benchmarked.
    bound = _bound(5, "plain")
    args = _args(5, "plain")
    assert bound.host_function is not None

    def start(
        cls: type[PatternSearch | DifferentialEvolutionSearch], **kwargs: int
    ) -> tuple[list[list[object]], list[helion.Config]]:
        config_filter = Mock(side_effect=new_filter())
        monkeypatch.setattr(bound.settings, "autotune_config_filter", config_filter)
        search = cls(bound, args, **kwargs)
        random.seed(0)
        population = search._generate_initial_population_flat()
        config_filter.assert_not_called()
        configs = [search.config_gen.unflatten(flat) for flat in population]
        passing, _indices = search._apply_config_filter(configs, record_trace=False)
        return population, passing

    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        seeds = [
            config
            for _flat, config in ConfigGeneration(
                bound.config_spec
            ).seed_flat_config_pairs()
        ]
        assert len(seeds) >= config_generation_ext.MIN_SEEDS_WITHOUT_RANDOM_DRAWS
        # Without a filter the seeds stand in for the draws.
        unfiltered = PatternSearch(bound, args, initial_population=100)
        random.seed(0)
        assert len(unfiltered._generate_initial_population_flat()) < 100
        # Pattern search draws through a view of its generation
        # (``initial_population_view``), differential evolution through the
        # generation itself.
        population, passing = start(PatternSearch, initial_population=100)
        evolution, evolving = start(DifferentialEvolutionSearch, population_size=50)
        assert len(population) >= 100
        assert len(evolution) == 100
        assert passing and evolving
        if passes_the_seeds:
            assert all(seed in passing and seed in evolving for seed in seeds)


def test_refills_of_a_folded_population_bring_no_generic_draws_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A search pads its starting points in more places than
    # ``random_population_flat``: the refill after the backend's pre-screen
    # rejected a config, and the random pad of a population built from the
    # best available configs. Each asks the same rule, so none of them
    # brings back the generic draws the population left out.
    bound = _bound(5, "plain")
    args = _args(5, "plain")
    assert bound.host_function is not None

    def generic(search: PatternSearch | DifferentialEvolutionSearch) -> int:
        random.seed(0)
        population = search._generate_initial_population_flat()
        assert len(population) < 100
        return sum(
            search.config_gen.unflatten(flat).get("cute_mma_dag") == "off"
            for flat in population
        )

    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        search = PatternSearch(bound, args, initial_population=100)
        assert (
            search.initial_population_strategy is InitialPopulationStrategy.FROM_RANDOM
        )
        # The fragment default and the compiler-coverage witnesses, which
        # are not draws.
        witnesses = generic(search)
        assert 0 < witnesses < 20
        seeds = [config for _flat, config in search.config_gen.seed_flat_config_pairs()]
        with monkeypatch.context() as screen:
            # The pre-screen rejects one seed, and the population is refilled.
            screen.setattr(
                PatternSearch,
                "_backend_config_is_viable",
                lambda self, config: config != seeds[1],
            )
            assert generic(search) == witnesses
        best_available = {
            "initial_population_strategy": InitialPopulationStrategy.FROM_BEST_AVAILABLE,
            "best_available_pad_random": True,
        }
        for cls in (PatternSearch, DifferentialEvolutionSearch):
            monkeypatch.setattr(
                cls, "_find_similar_cached_configs", lambda self, n, **kwargs: []
            )
        padded = PatternSearch(bound, args, initial_population=100, **best_available)
        assert generic(padded) == witnesses
        # Differential evolution adds no coverage witnesses: the default alone.
        evolution = DifferentialEvolutionSearch(
            bound, args, population_size=50, **best_available
        )
        assert generic(evolution) == 1
        # The fallback for a search whose seeds all failed keeps every draw:
        # no seed is left for the generic lowering to stand behind.
        random.seed(0)
        fallback = search._pad_initial_population_with_unique_random(
            [search.config_gen.default_flat()], 100, keep_every_draw=True
        )
        assert len(fallback) == 100


def test_random_search_gets_the_candidates_it_asks_for() -> None:
    # A random search times its population and nothing else: the draws that
    # the starting points of an iterative search leave out are its
    # candidates. It asks for 1,000 by default.
    bound = _bound(5, "plain")
    spec = bound.config_spec
    count = RANDOM_SEARCH_DEFAULTS.count
    assert count == 1000
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        generation = ConfigGeneration(spec)
        random.seed(0)
        assert len(generation.random_population_flat(count)) < 100
        random.seed(0)
        population = generation.random_population(count)
        assert len(population) == count
        assert generation.invalid_config_count == 0
        # Nearly all of them render the generic lowering.
        generic = sum(config.get("cute_mma_dag") == "off" for config in population)
        assert generic > count // 2
        random.seed(0)
        search = RandomSearch(bound, _args(5, "plain"))
        assert len(search.configs) == count


def test_unfolded_attention_population_keeps_its_random_draws() -> None:
    # Direct storage also admits the ordinary MMA lowerings, which random
    # draws can reach: the generic population is unchanged.
    bound = paired_v_lag_bound()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None and not facts.mapped_io
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        random.seed(0)
        population = ConfigGeneration(bound.config_spec).random_population_flat(100)
    assert len(population) == 100


def test_folded_index_proof_checks_intermediate_integer_overflow() -> None:
    bound = _bound(5, "plain")
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        matcher = _Matcher(bound.host_function.device_ir.graphs)
        program = matcher.match()
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None
        index = semantic.query.node.args[1][0]

        def rewritten_index(inject_overflow: bool) -> torch.fx.Node:
            graph = torch.fx.Graph()
            copied: dict[torch.fx.Node, torch.fx.Node] = {}

            def clone(node: torch.fx.Node) -> torch.fx.Node:
                if node not in copied:
                    result = graph.node_copy(node, clone)
                    result.meta = dict(node.meta)
                    if inject_overflow and node.target is tile_ops.tile_index:
                        product = graph.call_function(
                            torch.ops.aten.mul.Tensor, (result, 2**30)
                        )
                        product.meta = dict(node.meta)
                        result = graph.call_function(
                            torch.ops.aten.div.Tensor_mode,
                            (product, 2**30),
                            {"rounding_mode": "floor"},
                        )
                        result.meta = dict(node.meta)
                    copied[node] = result
                return copied[node]

            return clone(index)

        assert (
            prove_separable_index_map(
                matcher, rewritten_index(False), semantic.query.fake.numel()
            )
            is not None
        )
        # Algebraically unchanged coordinates still overflow in the original
        # Int32 intermediate. Simplifying the full expression first is unsafe.
        assert (
            prove_separable_index_map(
                matcher, rewritten_index(True), semantic.query.fake.numel()
            )
            is None
        )


@pytest.mark.parametrize("row_begin,item_begin", [(8, 0), (0, 1)])
def test_folded_index_proof_rejects_nonzero_task_origins(
    row_begin: int, item_begin: int
) -> None:
    # These are real shifted source loops, with enough backing storage that
    # every original address remains valid. Extent alone is not a coordinate.
    args = (
        torch.empty(65536, dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        row_begin,
        item_begin,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_shifted_folded_gather, args)
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            assert ir.noncanonical_task_origin_block_ids
            matcher = _Matcher(ir.graphs)
            # Initialize matcher shape/origin tables even when the complete
            # non-attention GEMM does not match the specialized consumer.
            matcher.match()
            indices = [
                node.args[1][0]
                for graph in ir.graphs
                for node in graph.graph.nodes
                if node.target is load
                and isinstance(node.args[1], (tuple, list))
                and len(node.args[1]) == 1
                and isinstance(node.args[1][0], torch.fx.Node)
                and node.args[1][0].meta["val"].ndim == 2
            ]
            assert len(indices) == 1
            assert prove_separable_index_map(matcher, indices[0], 65536) is None


def test_folded_index_proof_rejects_shifted_lead_with_canonical_iotas() -> None:
    args = (
        torch.empty(65536, dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        1,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_shifted_folded_iota_gather, args)
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            matcher = _Matcher(ir.graphs)
            # The complete matcher initializes the coordinate tables used by
            # the focused address proof; full GEMM admission is not required.
            matcher.match()
            indices = [
                node.args[1][0]
                for graph in ir.graphs
                for node in graph.graph.nodes
                if node.target is load
                and isinstance(node.args[1], (tuple, list))
                and len(node.args[1]) == 1
                and isinstance(node.args[1][0], torch.fx.Node)
                and node.args[1][0].meta["val"].ndim == 2
            ]
            assert len(indices) == 1
            iotas = [
                node
                for graph in ir.graphs
                for node in graph.graph.nodes
                if node.target is torch.ops.prims.iota.default
            ]
            assert len(iotas) == 2
            for node in iotas:
                (axis,) = matcher.shape_axes(node)
                assert axis is not None
                assert axis.block_id not in ir.noncanonical_task_origin_block_ids
            # Only the uniform leading item has a shifted origin, so this
            # specifically exercises bounds(), not axis_expression().
            assert ir.noncanonical_task_origin_block_ids
            assert prove_separable_index_map(matcher, indices[0], 65536) is None


def test_folded_index_proof_declines_nonleading_uniform_base() -> None:
    args = (
        torch.empty(65536, dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_nonlead_folded_gather, args)
        with bound.env, bound.host_function:
            matcher = _Matcher(bound.host_function.device_ir.graphs)
            # Nonleading tile.begin is intentionally outside the map contract.
            matcher.match()
            indices = [
                node.args[1][0]
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.target is load
                and isinstance(node.args[1], (tuple, list))
                and len(node.args[1]) == 1
                and isinstance(node.args[1][0], torch.fx.Node)
                and node.args[1][0].meta["val"].ndim == 2
            ]
            assert len(indices) == 1
            assert prove_separable_index_map(matcher, indices[0], 65536) is None
