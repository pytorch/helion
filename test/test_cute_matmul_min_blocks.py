from __future__ import annotations

import ast
from copy import deepcopy
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.matmul import matmul
from examples.matmul_split_k import matmul_split_k
from examples.softmax import softmax
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population
from test.test_cute_flash_exp2 import _dense_attention_output

import helion
from helion._compiler.autotuner_heuristics.cute_launch_bounds import MIN_BLOCKS_KEY
from helion._compiler.autotuner_heuristics.cute_launch_bounds import (
    register_matmul_min_blocks_coverage,
)
from helion.autotuner import LFBOTreeSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.metrics import AutotuneMetrics
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.settings import default_autotuner_fn

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence

    from helion.autotuner.base_search import PopulationMember
    from helion.autotuner.compiler_coverage import CompilerCoverageGroup
    from helion.autotuner.config_generation import FlatConfig
    from helion.runtime.kernel import BoundKernel


def _explicit_offset_mm(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    groups = offsets.size(0) - 1
    for group_tile, row, column in hl.tile([groups, m, n], block_size=[1, None, None]):
        group = group_tile.index.sum()
        start = offsets[group]
        count = offsets[group + 1] - start
        clipped_start = torch.clamp_min(start, 0)
        clipped_count = torch.clamp_min(
            torch.clamp_min(count, 0) + torch.clamp_max(start, 0), 0
        )
        row_index = clipped_start + row.index
        valid = row.index < clipped_count
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            x = hl.load(a, [row_index, reduction], extra_mask=valid[:, None])
            y = b[reduction, column]
            acc = torch.addmm(acc, x, y)
        hl.store(out, [row_index, column], acc.to(out.dtype), extra_mask=valid[:, None])
    return out


def _grouped_bound(
    shape: int = 0, **settings: object
) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    m, n, k, groups = ((640, 128, 128, 4), (2560, 512, 512, 8))[shape]
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=torch.int32) * (m // groups),
    )
    kernel = helion.kernel(
        _explicit_offset_mm,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        **settings,
    )
    return kernel.bind(args), args


def _search(
    bound: BoundKernel, args: tuple[torch.Tensor, ...], **options: object
) -> LFBOTreeSearch:
    # CPU tensor inputs have no production CUDA cache key. Hold only that query;
    # the ordinary full-effort factory and search constructor remain real.
    with bound.env, patch.object(LocalAutotuneCache, "_generate_key"):
        cache = default_autotuner_fn(bound, args, **options)
    assert isinstance(cache, LocalAutotuneCache)
    search = cache.autotuner
    assert type(search) is LFBOTreeSearch and search.initial_population == 100
    return search


STRUCTURAL_SETTINGS = (
    "cute_full_slice_matmul_tiling",
    "cute_segmented_matmul_tiling",
    "cute_flatten_nested_reductions",
)


REGISTRATION = (
    "helion._compiler.autotuner_heuristics.register_matmul_min_blocks_coverage"
)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for name in (
        "HELION_AUTOTUNE_EFFORT",
        "HELION_AUTOTUNER",
        "HELION_AUTOTUNER_INITIAL_POPULATION",
        "HELION_AUTOTUNE_CONFIG_OVERRIDES",
        "HELION_CUTE_FULL_SLICE_MATMUL_TILING",
        "HELION_CUTE_SEGMENTED_MATMUL_TILING",
        "HELION_CUTE_FLATTEN_NESTED_REDUCTIONS",
    ):
        monkeypatch.delenv(name, raising=False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


def _bind(
    kind: str = "matmul", dtype: torch.dtype = torch.float16, **settings: object
) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    if kind == "se":
        function = squeeze_and_excitation_net_fwd.fn
        args = tuple(
            torch.empty(shape, dtype=dtype, requires_grad=True)
            for shape in ((256, 256), (256, 64), (64, 256))
        )
    elif kind == "collective":
        function = matmul_split_k.fn
        args = tuple(
            torch.empty(shape, dtype=dtype) for shape in ((32, 4096), (4096, 64))
        )
    elif kind == "softmax":
        function = softmax.fn
        args = (torch.empty((64, 2048), dtype=dtype),)
    else:
        assert kind == "matmul"
        function = matmul.fn
        args = (
            torch.empty((128, 128), dtype=dtype),
            torch.empty((128, 128), dtype=dtype),
        )
    settings = dict.fromkeys(STRUCTURAL_SETTINGS, True) | settings
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        **settings,
    )
    return kernel.bind(args), args


def _group(bound: BoundKernel) -> CompilerCoverageGroup:
    groups = [
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == MIN_BLOCKS_KEY
    ]
    assert len(groups) == 1
    return groups[0]


@pytest.mark.parametrize("kind", ("matmul", "se"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_existing_matmul_option_has_a_complete_owned_pair(
    kind: str, dtype: torch.dtype
) -> None:
    with patch(REGISTRATION):
        previous, _ = _bind(kind, dtype)
    bound, _ = _bind(kind, dtype)
    spec = bound.config_spec
    group = _group(bound)
    assert group.domain == (0, 1) and group.legacy == 0
    assert [witness.value for witness in group.witnesses] == [0, 1]
    assert group.witnesses[0].carrier == group.witnesses[1].carrier
    assert MIN_BLOCKS_KEY not in group.witnesses[0].carrier
    assert spec.compiler_seed_configs == previous.config_spec.compiler_seed_configs
    assert spec.compiler_default_config == previous.config_spec.compiler_default_config
    assert spec.default_config() == previous.config_spec.default_config()
    assert (
        spec.projected_cache_fingerprint_hash()
        == previous.config_spec.projected_cache_fingerprint_hash()
    )
    assert (
        spec.cache_fingerprint_hash() != previous.config_spec.cache_fingerprint_hash()
    )
    generation = spec.create_config_generation()
    for value in (0, 1, 2, 3, 4, 6):
        requested = group.witnesses[0].carrier
        requested.config[MIN_BLOCKS_KEY] = value
        normalized = bound._normalize_config(requested)
        assert normalized.config[MIN_BLOCKS_KEY] == value
        round_trip = generation.unflatten(generation.flatten(normalized))
        assert round_trip.config[MIN_BLOCKS_KEY] == value
        source = bound.to_code(normalized)
        assert ast.parse(source)
        attribute = "_helion_cute_min_blocks_per_mp = " + str(value)
        assert (attribute in source) is (value > 0)


@pytest.mark.parametrize(
    "strategy,pad",
    (
        (None, True),
        ("from_random", True),
        ("from_best_available", False),
        ("from_best_available", True),
    ),
)
@pytest.mark.parametrize("kind", ("matmul", "grouped"))
def test_real_full_factory_preserves_base_rows_and_rng(
    strategy: str | None, pad: bool, kind: str
) -> None:
    settings = {"autotune_initial_population_strategy": strategy}
    bind = _grouped_bound if kind == "grouped" else _bind
    with patch(REGISTRATION):
        previous, previous_args = bind(**settings)
    bound, args = bind(**settings)
    assert (
        tuple(
            group
            for group in bound.config_spec.compiler_coverage_groups
            if group.key != MIN_BLOCKS_KEY
        )
        == previous.config_spec.compiler_coverage_groups
    )
    # The grouped default is now representable too; its automatic zero keeps
    # the original emitted source while one changes only launch metadata.
    for witness in _group(bound).witnesses:
        requested = witness.carrier
        requested.config[MIN_BLOCKS_KEY] = witness.value
        source = bound.to_code(requested)
        if witness.value == 0:
            assert source == previous.to_code(witness.carrier)
        else:
            assert "_helion_cute_min_blocks_per_mp = 1" in source
    old = _search(previous, previous_args, best_available_pad_random=pad)
    search = _search(bound, args, best_available_pad_random=pad)

    def population(
        search: LFBOTreeSearch,
    ) -> tuple[list[FlatConfig], list[FlatConfig]]:
        base_rows = []
        build_base = search._generate_initial_population_base

        def observe_base(**kwargs):
            result = build_base(**kwargs)
            base_rows.extend(deepcopy(result))
            return result

        with patch.object(search, "_generate_initial_population_base", observe_base):
            rows = deepcopy(search._generate_initial_population_flat())
        assert rows[: len(base_rows)] == base_rows
        return rows, base_rows

    with (
        previous.env,
        patch.object(old, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091468)
        old_rows, old_base = population(old)
        old_rng = random.getstate()
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091468)
        rows, base = population(search)
        assert random.getstate() == old_rng
    indices, sequence = search.config_gen._key_to_flat_indices[MIN_BLOCKS_KEY]
    assert not sequence and len(indices) == 1
    assert [
        [value for index, value in enumerate(row) if index != indices[0]]
        for row in base
    ] == old_base
    outcomes = [
        entry
        for entry in search.compiler_coverage_outcomes
        if entry.mechanism == "cute.matmul_min_blocks"
    ]
    # Later coverage groups may follow this pair. Their declarations and
    # effective candidates stay exact after projecting the new public zero.
    other_outcomes = [
        entry
        for entry in search.compiler_coverage_outcomes
        if entry.mechanism != "cute.matmul_min_blocks"
    ]
    old_outcomes = (
        old.compiler_coverage_outcomes
        if previous.config_spec.compiler_coverage_groups
        else []
    )
    if not previous.config_spec.compiler_coverage_groups:
        assert "compiler_coverage_outcomes" not in vars(old)
    assert len(other_outcomes) == len(old_outcomes)
    for before, after in zip(old_outcomes, other_outcomes, strict=True):
        assert (before.mechanism, before.origin, before.requested, before.outcome) == (
            after.mechanism,
            after.origin,
            after.requested,
            after.outcome,
        )
        effective = deepcopy(after.effective)
        if effective is not None:
            effective.config.pop(MIN_BLOCKS_KEY, None)
        assert effective == before.effective
    assert len(outcomes) == 2
    assert all(entry.outcome in ("added", "already_present") for entry in outcomes)
    assert {entry.effective.get(MIN_BLOCKS_KEY) for entry in outcomes} == {0, 1}
    assert len(rows) == len(old_rows) + sum(
        entry.outcome == "added" for entry in outcomes
    )


@pytest.mark.parametrize("disabled", (False, True))
def test_global_zero_and_disabled_heuristics_veto_automatic_additions(
    disabled: bool,
) -> None:
    settings = (
        {"disable_autotuner_heuristics": True}
        if disabled
        else {"autotune_config_overrides": {MIN_BLOCKS_KEY: 0}}
    )
    bound, args = _bind(**settings)
    group = _group(bound)
    search = _search(bound, args)
    with bound.env:
        rows = search._generate_initial_population_flat()
    assert all(
        search.config_gen.unflatten(row).get(MIN_BLOCKS_KEY, 0) == 0 for row in rows
    )
    if disabled:
        assert "compiler_coverage_outcomes" not in vars(search)
    else:
        assert not any(
            entry.requested.get(MIN_BLOCKS_KEY) == 1
            and entry.outcome in ("added", "already_present")
            for entry in search.compiler_coverage_outcomes
        )
    requested = group.witnesses[0].carrier
    requested.config[MIN_BLOCKS_KEY] = 1
    assert "_helion_cute_min_blocks_per_mp = 1" in bound.to_code(requested)


def test_option_reaches_the_real_first_benchmark() -> None:
    bound, args = _bind()
    search = _search(bound, args)
    search._autotune_metrics = AutotuneMetrics()
    delivered = []

    class HeldBenchmark(Exception):
        pass

    def hold(members: Sequence[PopulationMember], *, desc: str) -> None:
        assert desc == "Initial population"
        delivered.extend(deepcopy(member.config) for member in members)
        raise HeldBenchmark

    with (
        bound.env,
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        random.seed(2026091469)
        search._autotune()
    candidates = [
        entry.effective
        for entry in search.compiler_coverage_outcomes
        if entry.mechanism == "cute.matmul_min_blocks"
        and entry.requested.get(MIN_BLOCKS_KEY) == 1
    ]
    assert len(candidates) == 1 and candidates[0] is not None
    assert (
        candidates[0] in delivered and candidates[0] in search._pinned_finalist_configs
    )


@pytest.mark.parametrize("value", (False, True, "1", 1.0))
def test_malformed_coverage_requests_cannot_count(value: object) -> None:
    bound, _ = _bind()
    requested = _group(bound).witnesses[0].carrier
    requested.config[MIN_BLOCKS_KEY] = value
    with pytest.raises(InvalidConfig, match="compiler coverage mode"):
        bound.config_spec.create_config_generation().strict_config_pair(requested)


def test_existing_simt_domain_and_default_are_unchanged() -> None:
    with patch(REGISTRATION):
        previous, _ = _bind("softmax")
    bound, _ = _bind("softmax")
    spec = bound.config_spec
    assert not spec.cute_matmul_min_blocks_search_enabled
    assert spec._flat_fields()[MIN_BLOCKS_KEY].choices == (0, 1, 2, 3, 4, 6)
    assert (
        spec.structural_fingerprint() == previous.config_spec.structural_fingerprint()
    )
    assert (
        spec.cache_fingerprint_hash() == previous.config_spec.cache_fingerprint_hash()
    )
    assert spec.compiler_seed_configs == previous.config_spec.compiler_seed_configs
    assert spec.default_config() == previous.config_spec.default_config()


def test_flat_zero_preserves_public_config_and_generated_source() -> None:
    bound, _ = _bind()
    public = bound.config_spec.default_config()
    saved = deepcopy(public)
    assert MIN_BLOCKS_KEY not in public.config
    explicit_zero = deepcopy(public)
    explicit_zero.config[MIN_BLOCKS_KEY] = 0
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        flat = generation.flatten(public)
        assert generation.unflatten(flat) == explicit_zero
    assert public == saved
    implicit_source = bound.to_code(public)
    assert bound.to_code(explicit_zero) == implicit_source
    positive = deepcopy(public)
    positive.config[MIN_BLOCKS_KEY] = 1
    assert "_helion_cute_min_blocks_per_mp = 1" in bound.to_code(positive)
    assert "_helion_cute_min_blocks_per_mp = 1" not in implicit_source


@pytest.mark.parametrize("family", ("fa4", "ws_overlap"))
def test_flash_families_retain_old_defaults_and_sources(family: str) -> None:
    args = tuple(torch.empty((1, 1, 1024, 64), dtype=torch.bfloat16) for _ in range(3))

    def bind() -> BoundKernel:
        return helion.kernel(
            _dense_attention_output.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="full",
        ).bind(args)

    with patch(REGISTRATION):
        previous = bind()
    bound = bind()
    spec = bound.config_spec
    assert spec.matmul_facts and spec.cute_flash_search_enabled
    assert not spec.cute_matmul_min_blocks_search_enabled
    assert MIN_BLOCKS_KEY not in spec._flat_fields()
    assert (
        spec.cache_fingerprint_hash() == previous.config_spec.cache_fingerprint_hash()
    )
    assert spec.default_config() == previous.config_spec.default_config()
    assert spec.compiler_seed_configs == previous.config_spec.compiler_seed_configs
    generated = []
    for current in (previous, bound):
        with current.env:
            generation = current.config_spec.create_config_generation(
                overrides={"cute_flash_pipeline_family": family}
            )
            config = generation.unflatten(generation.default_flat())
            assert config["cute_flash_pipeline_family"] == family
        generated.append(current.to_code(config))
    assert generated[0] == generated[1]


def test_unrepresentable_grouped_default_keeps_old_registry_and_population() -> None:
    with patch(REGISTRATION):
        previous, previous_args = _grouped_bound()
        bound, args = _grouped_bound()
    # Do not rely on a historical default geometry being unrepresentable.
    # Exercise the failed initial strict-transfer boundary directly.
    with patch.object(
        ConfigGeneration,
        "strict_config_pair",
        side_effect=InvalidConfig("held unrepresentable grouped default"),
    ) as rejected:
        assert bound.host_function is not None
        register_matmul_min_blocks_coverage(bound.env, bound.host_function.device_ir)
    assert rejected.call_count == 1
    spec = bound.config_spec
    assert (
        spec.compiler_coverage_groups == previous.config_spec.compiler_coverage_groups
    )
    assert not spec.cute_matmul_min_blocks_search_enabled
    assert MIN_BLOCKS_KEY not in spec._flat_fields()
    assert (
        spec.cache_fingerprint_hash() == previous.config_spec.cache_fingerprint_hash()
    )
    assert spec.compiler_seed_configs == previous.config_spec.compiler_seed_configs
    assert spec.default_config() == previous.config_spec.default_config()
    old, search = _search(previous, previous_args), _search(bound, args)
    with previous.env:
        random.seed(2026091470)
        old_rows = old._generate_initial_population_flat()
        old_rng = random.getstate()
    with bound.env:
        random.seed(2026091470)
        rows = search._generate_initial_population_flat()
        assert random.getstate() == old_rng
    assert rows == old_rows
    assert search.compiler_coverage_outcomes == old.compiler_coverage_outcomes


@pytest.mark.parametrize("disabled", (False, True))
def test_explicit_seed_modes_survive_without_becoming_global_veto(
    disabled: bool,
) -> None:
    base, _ = _bind(disable_autotuner_heuristics=disabled)
    carrier = _group(base).witnesses[0].carrier
    seeds = [
        helion.Config.from_dict(carrier.config | {MIN_BLOCKS_KEY: value})
        for value in (0, 1)
    ]
    bound, args = _bind(
        disable_autotuner_heuristics=disabled, autotune_seed_configs=seeds
    )
    search = _search(bound, args)
    with bound.env:
        rows = search._generate_initial_population_flat()
        configs = [search.config_gen.unflatten(row) for row in rows]
    assert {config[MIN_BLOCKS_KEY] for config in configs} == {0, 1}
    for seed in seeds:
        assert bound._normalize_config(seed) in configs
    if disabled:
        assert len(rows) == 100
    else:
        assert any(
            entry.requested.get(MIN_BLOCKS_KEY) == 1
            and entry.outcome == "already_present"
            for entry in search.compiler_coverage_outcomes
        )


def test_later_random_and_coordinate_neighbors_reach_one() -> None:
    bound, _ = _bind()
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        projections = generation.coordinate_neighbor_projections(
            generation.default_flat()
        )
        assert any(
            item.key == MIN_BLOCKS_KEY
            and item.outcome == "candidate"
            and item.config is not None
            and item.config[MIN_BLOCKS_KEY] == 1
            for item in projections
        )
        random.seed(2026091471)
        assert {generation.random_config()[MIN_BLOCKS_KEY] for _ in range(20)} == {0, 1}


def test_failed_carrier_transfer_keeps_old_registry_and_domain() -> None:
    with patch(REGISTRATION):
        bound, _ = _bind()
    spec = bound.config_spec
    original = spec.cache_fingerprint_hash()
    groups = spec.compiler_coverage_groups
    strict = ConfigGeneration.strict_config_pair
    rejected = []

    def reject_new(
        generation: ConfigGeneration, config: helion.Config
    ) -> tuple[FlatConfig, helion.Config]:
        if MIN_BLOCKS_KEY in config:
            rejected.append(config)
            raise InvalidConfig("held unrepresentable carrier")
        return strict(generation, config)

    with patch.object(ConfigGeneration, "strict_config_pair", reject_new):
        assert bound.host_function is not None
        register_matmul_min_blocks_coverage(bound.env, bound.host_function.device_ir)
    assert len(rejected) == 1
    assert not spec.cute_matmul_min_blocks_search_enabled
    assert spec.compiler_coverage_groups == groups
    spec.validate_compiler_coverage_groups()
    assert spec.cache_fingerprint_hash() == original


@pytest.mark.parametrize("value", (2, 3, 4, 6))
@pytest.mark.parametrize("disabled", (False, True))
def test_global_positive_override_preserves_other_coverage_groups(
    value: int, disabled: bool
) -> None:
    settings = {
        "autotune_config_overrides": {MIN_BLOCKS_KEY: value},
        "disable_autotuner_heuristics": disabled,
    }
    with patch(REGISTRATION):
        previous, previous_args = _bind("collective", **settings)
    bound, args = _bind("collective", **settings)
    spec = bound.config_spec
    assert not spec.cute_matmul_min_blocks_search_enabled
    assert MIN_BLOCKS_KEY not in spec._flat_fields()
    assert (
        spec.compiler_coverage_groups == previous.config_spec.compiler_coverage_groups
    )
    # Ordinary warp collectives supply a real unrelated group even when
    # automatic heuristic use is disabled. No materialized bundle is needed.
    assert [group.mechanism for group in spec.compiler_coverage_groups] == [
        "cute.collective_static_layouts"
    ]
    assert all(not group.dependencies for group in spec.compiler_coverage_groups)
    assert (
        spec.structural_fingerprint() == previous.config_spec.structural_fingerprint()
    )
    assert (
        spec.cache_fingerprint_hash() == previous.config_spec.cache_fingerprint_hash()
    )
    assert spec.default_config() == previous.config_spec.default_config()
    assert spec.compiler_seed_configs == previous.config_spec.compiler_seed_configs
    old, search = _search(previous, previous_args), _search(bound, args)
    with (
        previous.env,
        patch.object(old, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091481)
        old_rows = deepcopy(checked_initial_population(old))
        old_rng = random.getstate()
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091481)
        rows = checked_initial_population(search)
        assert random.getstate() == old_rng
        assert rows == old_rows
        assert all(
            search.config_gen.unflatten(row)[MIN_BLOCKS_KEY] == value for row in rows
        )
    assert vars(search).get("compiler_coverage_outcomes") == vars(old).get(
        "compiler_coverage_outcomes"
    )
    # Each unchanged registry's actual additions were independently checked.
    for group in spec.compiler_coverage_groups:
        for witness in group.witnesses:
            requested = witness.carrier
            requested.config[group.key] = witness.value
            requested.config[MIN_BLOCKS_KEY] = value
            assert bound._normalize_config(requested) == previous._normalize_config(
                requested
            )


@pytest.mark.parametrize("value", (2, 3, 4, 6))
@pytest.mark.parametrize("disabled", (False, True))
def test_seed_local_positive_keeps_automatic_pair(value: int, disabled: bool) -> None:
    base, _ = _bind("matmul")
    seed = _group(base).witnesses[0].carrier
    seed.config[MIN_BLOCKS_KEY] = value
    settings = {
        "autotune_seed_configs": [seed],
        "disable_autotuner_heuristics": disabled,
    }
    with patch(REGISTRATION):
        previous, previous_args = _bind("matmul", **settings)
    bound, args = _bind("matmul", **settings)
    assert _group(bound).domain == (0, 1)
    old, search = _search(previous, previous_args), _search(bound, args)
    with (
        previous.env,
        patch.object(old, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091482)
        old_rows = deepcopy(checked_initial_population(old))
        old_rng = random.getstate()
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        random.seed(2026091482)
        rows = checked_initial_population(search)
        # The old schema drops this explicit value and deduplicates its
        # ordinary carrier with the default. Retaining the value adds one
        # explicit seed at row 1, so the existing builder draws one fewer
        # random padding row. This is the intentional v1 seed behavior;
        # the v2 global-override repair must not veto local positive seeds.
        assert random.getstate() != old_rng
        indices, sequence = search.config_gen._key_to_flat_indices[MIN_BLOCKS_KEY]
        assert not sequence and len(indices) == 1
        assert [
            [entry for index, entry in enumerate(row) if index != indices[0]]
            for row_index, row in enumerate(rows[:100])
            if row_index != 1
        ] == old_rows[:99]
        configs = [search.config_gen.unflatten(row) for row in rows]
    assert configs[1] == bound._normalize_config(seed)
    assert bound._normalize_config(seed) in configs
    if disabled:
        assert len(rows) == 100
        assert "compiler_coverage_outcomes" not in vars(search)
    else:
        outcomes = search.compiler_coverage_outcomes
        assert any(
            entry.mechanism == "cute.matmul_min_blocks"
            and entry.requested[MIN_BLOCKS_KEY] == 1
            and entry.outcome in ("added", "already_present")
            for entry in outcomes
        )


@pytest.mark.parametrize("disabled", (False, True))
def test_inactive_fanout_default_placement_seed_roundtrip(disabled: bool) -> None:
    base, _ = _bind("matmul")
    seed = _group(base).witnesses[0].carrier
    seed.config[MIN_BLOCKS_KEY] = 0
    seed.config.pop("tcgen05_c_acquire_placement", None)
    bound, _ = _bind("matmul", disable_autotuner_heuristics=disabled)
    implicit = bound._normalized_config_copy(seed)
    seed.config["tcgen05_c_acquire_placement"] = "pre_loop"
    explicit = bound._normalized_config_copy(seed)
    assert explicit == implicit
    assert "tcgen05_c_acquire_placement" not in explicit.config
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        flat, selected = generation.strict_config_pair(seed)
        assert selected == explicit
        assert generation.unflatten(flat) == explicit


@pytest.mark.parametrize("disabled", (False, True))
@pytest.mark.parametrize("placement", ("first_in_loop", "later_before_barrier"))
def test_inactive_fanout_nondefault_placement_is_preserved(
    disabled: bool, placement: str
) -> None:
    base, _ = _bind("matmul")
    seed = _group(base).witnesses[0].carrier
    seed.config["tcgen05_c_acquire_placement"] = placement
    bound, _ = _bind("matmul", disable_autotuner_heuristics=disabled)
    selected = bound._normalized_config_copy(seed)
    assert selected["tcgen05_c_acquire_placement"] == placement
    assert bound._normalized_config_copy(selected) == selected
