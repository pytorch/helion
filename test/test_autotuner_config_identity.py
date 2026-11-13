"""Compiler-distinct scalar selections survive search and final verification."""

from __future__ import annotations

import math
import os
import pickle
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import helion
from helion._compiler.backend_registry import get_backend
from helion._testing.cute_codegen import cpu_target
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.handoff import HandoffPolicy
from helion.autotuner.handoff import _HandoffSession
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.config_cache import ConfigCache
from helion.runtime.config_cache import ConfigSet
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.autotuner.base_search import _AutotunableKernel

pytestmark = pytest.mark.cpu_only


class _OpaqueList(list[object]):
    pass


def _tag(value: object) -> tuple[type, str]:
    return type(value), value.hex() if type(value) is float else repr(value)


def _search(values: tuple[object, ...]) -> PatternSearch:
    backend = get_backend(_get_backend())
    spec = ConfigSpec(
        backend=backend,
        target_device_capability=(10, 0),
        user_defined_tunables={"selected": EnumFragment(values)},
    )
    settings = helion.Settings(
        backend=backend.name,
        autotune_log_level=0,
        autotune_log_details=False,
        autotune_log_search_space=False,
    )
    kernel = SimpleNamespace(
        settings=settings,
        config_spec=spec,
        env=SimpleNamespace(process_group_name=None),
        to_code=lambda config: repr(_tag(config["selected"])),
    )
    result = PatternSearch(
        cast("_AutotunableKernel", kernel), (), initial_population=2, copies=1
    )
    result.final_selection = None
    return result


def _member(value: object, perf: float) -> PopulationMember:
    return PopulationMember(
        lambda: value, [perf], [value], helion.Config(selected=value), status="ok"
    )


def test_backend_defaults_without_environment(monkeypatch):
    monkeypatch.delenv("HELION_BACKEND", raising=False)
    with cpu_target():
        assert _search((1, 2)).config_spec.backend.name == "triton"


@pytest.mark.parametrize("search_type", [LFBOPatternSearch, LFBOTreeSearch])
@pytest.mark.parametrize("values", [(0.0, -0.0), (1, True), (1, 1.0)])
def test_lfbo_neighbors_preserve_scalar_type_and_zero_sign(
    search_type: type[LFBOPatternSearch], values: tuple[object, object]
) -> None:
    with cpu_target():
        generator = _search(values).config_gen
        base = generator.flatten(helion.Config(selected=values[0]))
        alternative = generator.flatten(helion.Config(selected=values[1]))
        (selected,) = [
            index
            for index, (left, right) in enumerate(zip(base, alternative, strict=True))
            if _tag(left) != _tag(right)
        ]
        assert base == alternative
        assert _tag(generator.unflatten(alternative)["selected"]) == _tag(values[1])
        search = search_type.__new__(search_type)
        search.config_gen = generator
        search.num_neighbors = 1
        search.num_neighbors_cap = -1
        search.radius = 1
        frozen = {index: value for index, value in enumerate(base) if index != selected}

        if isinstance(search, LFBOTreeSearch):
            search._encoded_to_flat_mapping = None
            start, end, _ = search._get_encoded_to_flat_mapping()[selected]
            target = np.asarray(generator.flat_spec[selected].encode(values[1]))

            class Estimator:
                tree_ = SimpleNamespace(feature=np.array([start]))

                def decision_path(self, _encoded: np.ndarray) -> SimpleNamespace:
                    return SimpleNamespace(indices=np.array([0]))

                def predict_proba(self, encoded: np.ndarray) -> np.ndarray:
                    scores = np.all(encoded[:, start:end] == target, axis=1).astype(
                        float
                    )
                    return np.column_stack((1 - scores, scores))

            search.surrogate = SimpleNamespace(estimators_=[Estimator()])
            search._autotune_metrics = SimpleNamespace(num_generations=2)
            random_index = 0
        else:
            random_index = 1

        with patch(
            "helion.autotuner.surrogate_pattern_search.random.randint",
            return_value=random_index,
        ):
            neighbors = search._generate_neighbors(base, fixed_flat_values=frozen)
            assert len(neighbors) == 1
            assert _tag(generator.unflatten(neighbors[0])["selected"]) == _tag(
                values[1]
            )
            # Pins still suppress a transition; a no-op proposal is not a neighbor.
            assert (
                search._generate_neighbors(
                    base, fixed_flat_values=dict(enumerate(base))
                )
                == []
            )
            with patch.object(generator, "conditional_neighbor", return_value=base):
                assert search._generate_neighbors(base, fixed_flat_values=frozen) == []


def test_lfbo_tree_reencodes_typed_conditional_neighbor() -> None:
    fields = [EnumFragment(("a", "b")), EnumFragment((0.0, -0.0))]

    def encode(flat: list[object]) -> list[float]:
        return [
            part
            for field, value in zip(fields, flat, strict=True)
            for part in field.encode(value)
        ]

    def project(
        base: list[object], requested: list[object], **kwargs: object
    ) -> list[object]:
        return ["b", -0.0] if requested[0] == "b" else requested

    class Estimator:
        tree_ = SimpleNamespace(feature=np.array([0]))

        def decision_path(self, _encoded: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(indices=np.array([0]))

        def predict_proba(self, encoded: np.ndarray) -> np.ndarray:
            np.testing.assert_array_equal(
                encoded, [encode(["a", 0.0]), encode(["b", -0.0])]
            )
            return np.array([[1.0, 0.0], [0.0, 1.0]])

    search = LFBOTreeSearch.__new__(LFBOTreeSearch)
    search.num_neighbors = search.radius = 1
    search.num_neighbors_cap = -1
    search._encoded_to_flat_mapping = None
    search._autotune_metrics = SimpleNamespace(num_generations=2)
    search.surrogate = SimpleNamespace(estimators_=[Estimator()])
    search.config_gen = SimpleNamespace(
        flat_spec=fields,
        block_size_indices=[],
        num_warps_index=-1,
        overridden_flat_indices=set(),
        conditional_neighbor=project,
        encode_config=encode,
    )
    (neighbor,) = search._generate_neighbors(["a", 0.0])
    assert neighbor[0] == "b" and _tag(neighbor[1]) == _tag(-0.0)


def test_seed_transfer_retains_scalar_type_and_zero_sign() -> None:
    values = (True, 1, 1.0, -0.0, 0.0)
    with cpu_target():
        search = _search(values)
        for ordered in (values, tuple(reversed(values))):
            configs = [helion.Config(selected=value) for value in ordered]
            # Establish that transfer preserves each value before testing dedup.
            assert [
                _tag(
                    search.config_gen.canonicalize_flat(search.config_gen.flatten(c))[
                        1
                    ]["selected"]
                )
                for c in configs
            ] == [_tag(value) for value in ordered]
            search.config_spec.compiler_seed_configs = configs
            compiler = search.config_gen.seed_flat_config_pairs()
            user = search.config_gen.user_seed_flat_config_pairs(configs)
            for pairs in (compiler, user):
                assert [_tag(config["selected"]) for _, config in pairs] == [
                    _tag(value) for value in ordered
                ]


@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_finalist_registries_keep_each_compilable_config(
    values: tuple[object, object],
) -> None:
    with (
        cpu_target(),
        patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}),
    ):
        search = _search(values)
        members = [
            _member(value, 1.0 + index / 100) for index, value in enumerate(values)
        ]
        assert members[0].config == members[1].config
        search.pin_finalist_configs([member.config for member in members])
        for member in members:
            search._record_benchmarked_member(member)
        assert len(search._pinned_finalist_configs) == 2
        assert len(search._pinned_finalist_members) == 2
        assert len(search._benchmarked_members) == 2
        assert [
            _tag(member.fn()) for member in search._benchmarked_members.values()
        ] == [_tag(value) for value in values]
        search.population = members
        targets, sources = search._rebenchmark_failure_targets(members, [None, "error"])
        assert any(member is members[1] for member, _ in targets)
        # The failed choice's retained history copies are also invalidated.
        assert all(
            _tag(member.config["selected"]) == _tag(values[1]) for member, _ in targets
        )
        assert not sources


@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_final_verification_can_select_the_python_equal_alternative(
    values: tuple[object, object],
) -> None:
    with (
        cpu_target(),
        patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}),
    ):
        search = _search(values)
        members = [
            _member(value, 1.0 + index / 100) for index, value in enumerate(values)
        ]
        search.population = members
        measured: list[tuple[type, str]] = []

        def measure(finalists: Sequence[PopulationMember], **kwargs: object) -> None:
            for member in finalists:
                tag = _tag(member.config["selected"])
                measured.append(tag)
                member.perfs.append(0.5 if tag == _tag(values[1]) else 10.0)

        with patch.object(search, "rebenchmark", side_effect=measure):
            winner = search.final_rebenchmark_best(members[0])
        assert measured == [_tag(value) for value in values]
        assert _tag(winner.fn()) == _tag(values[1])


@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_handoff_failure_does_not_remove_python_equal_success(
    values: tuple[object, object],
) -> None:
    with cpu_target():
        search = _search(values)
        members = [
            _member(value, 1.0 + index / 100) for index, value in enumerate(values)
        ]
        session = _HandoffSession(search, HandoffPolicy())
        session.observe(
            search,
            [BenchmarkResult(m.config, m.fn, m.perf, "ok", None) for m in members],
        )
        assert len(session.candidates) == 2
        members[1].perfs.append(math.inf)
        session.refresh([members[1]])
        assert len(session.candidates) == 1
        (retained,) = session.candidates.values()
        assert _tag(retained.config["selected"]) == _tag(values[0])


def test_config_set_operations_snapshot_nested_values() -> None:
    original = helion.Config(selected={"parts": [-0.0, True]})
    signed = helion.Config(selected={"parts": [0.0, True]})
    integer = helion.Config(selected={"parts": [-0.0, 1]})
    configs = ConfigSet([original, signed, integer])
    original["selected"]["parts"][0] = 7.0
    saved = helion.Config(selected={"parts": [-0.0, True]})
    assert len(configs) == 3 and saved in configs and original not in configs
    # Returned values are private copies, and process handoff keeps the keys.
    next(iter(configs))["selected"]["parts"].clear()
    copied = pickle.loads(pickle.dumps(configs.copy()))
    assert len(copied) == 3 and saved in copied
    copied.difference_update([signed])
    assert len(copied) == 2 and signed not in copied
    copied.update(ConfigSet([signed]), [integer])
    assert len(copied) == 3
    assert len(copied - ConfigSet([integer])) == 2
    assert len(ConfigSet([saved]) | ConfigSet([signed])) == 2
    copied.difference_update(copied)
    assert not copied and len(configs) == 3


def test_handoff_confirmation_preserves_extra_typed_config_and_source_dedup() -> None:
    with cpu_target():
        search = _search((-0.0, 0.0))
        members = [_member(-0.0, 1.0), _member(0.0, 2.0)]
        # Both configs have one source here: ordinary source dedup must survive.
        search.kernel.to_code = Mock(return_value="same generated source")
        session = _HandoffSession(search, HandoffPolicy(finalists=2, repetitions=3))
        session.observe(
            search,
            [BenchmarkResult(m.config, m.fn, m.perf, "ok", None) for m in members],
        )
        provider = Mock()
        measured = []

        def benchmark(
            configs: list[helion.Config], **kwargs: object
        ) -> list[BenchmarkResult]:
            measured.append([_tag(c["selected"]) for c in configs])
            return [BenchmarkResult(c, lambda: None, 1.0, "ok", None) for c in configs]

        provider.benchmark.side_effect = benchmark
        search._benchmark_provider_cls = Mock(return_value=provider)
        assert len(session.confirm()) == 1
        assert measured == [[_tag(-0.0)]] * 3
        measured.clear()
        confirmed = session.confirm(extra=members[1].config)
        pair = [_tag(-0.0), _tag(0.0)]
        assert measured == [pair, pair[::-1], pair]
        assert [_tag(c.config["selected"]) for c in confirmed] == pair
        assert all(len(c.samples) == 3 for c in confirmed)
        assert provider.cleanup.call_count == 6


def test_config_cache_constructor_does_not_collapse_mapping_keys() -> None:
    configs = [helion.Config(selected=value) for value in (True, 1, 1.0)]
    cache = ConfigCache((config, index) for index, config in enumerate(configs))
    copied = ConfigCache(cache)
    assert len(copied) == 3
    assert [copied[config] for config in configs] == [0, 1, 2]
    opaque = _OpaqueList([1])
    mutable = helion.Config(selected=opaque)
    original = ConfigCache([(mutable, 7)])
    opaque.append(2)
    copied = ConfigCache(original)
    assert mutable not in copied and len(copied) == 1
    saved, value = copied.popitem()
    assert saved["selected"] is opaque and value == 7


def test_handoff_refresh_retains_opaque_config_leaf_identity() -> None:
    value = _OpaqueList([1])
    with cpu_target():
        search = _search((value,))
        config = helion.Config(selected=value)
        flat = search.config_gen.flatten(config)
        config = search.config_gen.unflatten(flat)
        value = config["selected"]
        assert type(value) is _OpaqueList
        session = _HandoffSession(search, HandoffPolicy())
        session.observe(
            search, [BenchmarkResult(config, lambda: None, 1.0, "ok", None)]
        )
        session.refresh(
            [PopulationMember(lambda: None, [math.inf], flat, config, status="ok")]
        )
        assert not session.candidates


@pytest.mark.parametrize("failure", ["local", "remote", "remote_clone"])
def test_handoff_confirmation_shares_positions_not_opaque_identities(
    failure: str,
) -> None:
    with cpu_target():
        search = _search((_OpaqueList([1]), _OpaqueList([2])))
        configs = [
            search.config_gen.unflatten(
                search.config_gen.flatten(helion.Config(selected=value))
            )
            for value in search.config_spec.user_defined_tunables["selected"].choices
        ]
        session = _HandoffSession(search, HandoffPolicy(finalists=2, repetitions=3))
        session.observe(
            search,
            [
                BenchmarkResult(config, lambda: None, float(index + 1), "ok", None)
                for index, config in enumerate(configs)
            ],
        )
        original_measurements = tuple(session.measurements)
        provider = Mock()
        measured = []

        def benchmark(
            ordered: list[helion.Config], **kwargs: object
        ) -> list[BenchmarkResult]:
            measured.extend(ordered)
            return [
                BenchmarkResult(
                    config,
                    lambda: None,
                    math.inf
                    if failure == "local" and config["selected"] == [2]
                    else float(config["selected"][0]),
                    "ok",
                    None,
                )
                for config in ordered
            ]

        def gather(invalid: set[int], group: str | None) -> list[set[int]]:
            # A peer's report is transported by value, never with local object IDs.
            assert invalid == ({1} if failure == "local" else set())
            return [invalid, pickle.loads(pickle.dumps({1}))]

        def synchronize(value: object, group: str | None) -> object:
            # A non-source rank receives newly unpickled config leaves.
            return (
                pickle.loads(pickle.dumps(value))
                if failure == "remote_clone"
                else value
            )

        provider.benchmark.side_effect = benchmark
        search._benchmark_provider_cls = Mock(return_value=provider)
        with (
            patch("helion.autotuner.handoff.all_gather_object", side_effect=gather),
            patch("helion.autotuner.handoff.sync_object", side_effect=synchronize),
        ):
            confirmed = session.confirm()
        assert len(confirmed) == len(session.candidates) == 1
        (winner,) = confirmed
        assert winner.config["selected"] is measured[0]["selected"]
        assert winner.config["selected"] == [1]
        assert winner.perf == 1.0 and winner.samples == (1.0, 1.0, 1.0)
        assert all(
            actual is original
            for actual, original in zip(
                session.measurements, original_measurements, strict=False
            )
        )
        assert len(session.measurements) == 8
        assert provider.cleanup.call_count == 3
        member = PopulationMember(
            lambda: None, [math.inf], [], winner.config, status="ok"
        )
        session.refresh([member])
        session.refresh([member])
        assert not session.candidates


def test_finalist_history_preserves_opaque_selection_identity() -> None:
    with (
        cpu_target(),
        patch.dict(os.environ, {"HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K": "2"}),
    ):
        search = _search((_OpaqueList([1]),))
        flat = search.config_gen.default_flat()
        config = search.config_gen.unflatten(flat)
        assert type(config["selected"]) is _OpaqueList
        member = PopulationMember(lambda: None, [1.0], flat, config, status="ok")
        search.pin_finalist_config(config)
        search._record_benchmarked_member(member)
        assert config in search._pinned_finalist_configs
        assert (
            search._pinned_finalist_members[config].config["selected"]
            is config["selected"]
        )
        assert (
            search._benchmarked_members[config].config["selected"] is config["selected"]
        )
        targets, _ = search._rebenchmark_failure_targets([member], ["error"])
        assert len(targets) == 3
        # A later source-equivalent success repairs the same selected config.
        provider = object.__new__(LocalBenchmarkProvider)
        provider._pending_effective_source_failures = {}
        provider._effective_source_repairs = ConfigCache()
        provider._autotune_metrics = AutotuneMetrics()
        provider._remember_effective_source_failure("same source", config, None, None)
        result = BenchmarkResult(config, member.fn, 1.0, "ok", None)
        repairs = provider._resolve_effective_source_failures("same source", result)
        assert repairs[config].config["selected"] is config["selected"]
        assert provider.take_effective_source_repairs()[config].fn is member.fn


def test_canonicalization_keeps_opaque_enum_owner_and_deduplicates():
    value = _OpaqueList([1])
    with cpu_target():
        generator = _search((value, _OpaqueList([2]))).config_gen
        initial = generator.initial_population_view()
        assert (
            initial.unflatten(initial.flatten(helion.Config(selected=value)))[
                "selected"
            ]
            is value
        )
        flat = generator.flatten(helion.Config(selected=value))
        first_flat, first = generator.canonicalize_flat(flat)
        second_flat, second = generator.canonicalize_flat(flat)
    assert first["selected"] is value
    assert second["selected"] is value
    assert second in ConfigSet((first,))
    assert first_flat is not second_flat and first_flat is not flat
    selected = generator._key_to_flat_indices["selected"][0][0]
    assert first_flat[selected] is value
    assert second_flat[selected] is value


@pytest.mark.parametrize("values", [(1, True), (0.0, -0.0)])
def test_conditional_neighbor_requires_exact_searchable_prerequisite(values):
    with cpu_target():
        generator = _search(values).config_gen
        base = generator.flatten(helion.Config(selected=values[0]))
        selected = generator._key_to_flat_indices["selected"][0][0]
        requested = list(base)
        generator.flat_spec[selected] = EnumFragment(
            values, search_choices=(values[0],)
        )
        with patch.object(
            generator.extension,
            "conditional_neighbor_updates",
            return_value={selected: values[1]},
        ):
            assert generator.conditional_neighbor(base, requested) is requested
            generator.flat_spec[selected] = EnumFragment(
                values, search_choices=(values[1],)
            )
            result = generator.conditional_neighbor(base, requested)
            assert _tag(result[selected]) == _tag(values[1])
            assert _tag(requested[selected]) == _tag(values[0])
