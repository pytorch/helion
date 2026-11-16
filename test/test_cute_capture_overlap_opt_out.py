"""A declined capture overlap stays declined in a config no search produced.

The proved paired attention kernel defaults to guarded capture overlap, whose
condition the compiler cannot check: the application leaves its captured
graphs as captured. An application that edits them opts out with
``cute_pdl=False`` or ``cute_pdl_capture_overlap=False`` in
``autotune_config_overrides``. Overrides act on a search and are not part of
the autotune cache identity, so the kernel also selects configs that no
search under its overrides produced: a cache hit, AOT tuning data, the
default of a kernel that does not autotune. The opt-out has to reach those.
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._hardware import HardwareInfo
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion.autotuner import handoff
from helion.autotuner import handoff_pipeline
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_compile import _standalone_call_key
from helion.autotuner.aot_structural_policy import attach_model_policy
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.remote_cache import RemoteAutotuneCache
from helion.autotuner.remote_cache import RemoteCacheBackend
from helion.autotuner.remote_cache import StrictRemoteAutotuneCache
from helion.runtime.cute.structural_config import CuteStructuralConfig
from helion.runtime.cute.structural_policy import CuteStructuralPolicy
from helion.runtime.settings import default_autotuner_fn
from helion.runtime.structural_policy import StructuralPolicyError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion.autotuner.base_cache import AutotuneCacheBase
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

PDL = "cute_pdl"
DAG = "cute_mma_dag"
PASS_WARPS = "cute_mma_dag_pass_warps"
CAPTURE = "cute_pdl_capture_overlap"
RELEASE = "cute_pdl_release"
SPREAD = "cute_pdl_spread"
NAMES = (PDL, CAPTURE, RELEASE, SPREAD)
# The guarded default, and what each documented opt-out leaves of it.
POLICY = (True, True, "loader_tail", False)
OFF = (False, False, "auto", False)
OPT_OUTS = [{PDL: False}, {CAPTURE: False}]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot"))
    # No test here may search: a miss is an error, and a hit is what the
    # variable lets through.
    monkeypatch.setenv("HELION_ASSERT_CACHE_HIT", "1")
    for name in ("HELION_SKIP_CACHE", "HELION_FORCE_AUTOTUNE", "HELION_AOT_MODE"):
        monkeypatch.delenv(name, raising=False)
    AOTAutotuneCache.clear_caches()
    try:
        with (
            cpu_render_target(fake_device_sm_count=148),
            patch(
                "helion.autotuner.aot_cache.get_hardware_info",
                return_value=HardwareInfo(
                    device_kind="cuda",
                    hardware_name="CPU SM100 proof",
                    runtime_version="13.0",
                    compute_capability="sm100",
                ),
            ),
        ):
            yield
    finally:
        AOTAutotuneCache.clear_caches()


class _MemoryRemote(RemoteCacheBackend):
    def __init__(self) -> None:
        self.records: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.records.get(key)

    def put(self, key: str, data: str) -> None:
        self.records[key] = data


_REMOTE = _MemoryRemote()


@contextmanager
def _cache_identity() -> Iterator[None]:
    """The device fields of an autotune cache key, which a kernel bound for
    the CPU cannot supply, and one remote backend for every cache."""
    with (
        patch(
            "helion.autotuner.local_cache.extract_device",
            return_value=torch.device("cuda:0"),
        ),
        patch(
            "helion.autotuner.local_cache.get_device_name",
            return_value="cpu-sm100-proof",
        ),
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "helion.autotuner.remote_cache._load_remote_backend", return_value=_REMOTE
        ),
    ):
        yield


def _autotuner(
    bound: BoundKernel[object], args: Sequence[object], **kwargs: object
) -> AutotuneCacheBase:
    """The cache-wrapped search a kernel builds for itself, keyed as on a
    CUDA device."""
    with _cache_identity():
        return cast("AutotuneCacheBase", default_autotuner_fn(bound, args, **kwargs))


@functools.cache
def _args(heads: int = 2) -> tuple[object, ...]:
    return (*paged_attention_args(rows=8, requests=2, heads=heads), None, True)


def _kernel(overrides: dict[str, object], **settings: object) -> Kernel[object]:
    # ``autotuner_fn`` keeps the kernel in the config schema every kernel
    # with an override, a seed or tuning data is in (no automatic structural
    # policy), where the cache key is the same with and without overrides.
    return cast(
        "Kernel[object]",
        helion.kernel(
            paged_online_attention.fn,
            backend="cute",
            static_shapes=True,
            autotuner_fn=_autotuner,
            autotune_config_overrides=overrides,
            **settings,
        ),
    )


@functools.cache
def _bound_cached(overrides: tuple[tuple[str, object], ...]) -> BoundKernel[object]:
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(_kernel(dict(overrides)), _args())


def _bound(overrides: dict[str, object] | None = None) -> BoundKernel[object]:
    return _bound_cached(tuple(sorted((overrides or {}).items())))


def _policy(config: helion.Config) -> tuple[object, object, object, object]:
    return (
        config.get(PDL, False),
        config.get(CAPTURE, False),
        config.get(RELEASE, "auto"),
        config.get(SPREAD, False),
    )


def _paired(config: helion.Config) -> bool:
    return config.get("cute_mma_dag_pass_warps") == "attention_paired"


def _resolved(
    bound: BoundKernel[object], config: helion.Config
) -> tuple[object, object, object, object]:
    """The policy ``config`` compiles with."""
    return _policy(bound.config_spec.normalized_config(config))


def _tuned(bound: BoundKernel[object]) -> helion.Config:
    """What a search without overrides returns: a paired seed, which carries
    the guarded default."""
    seed = next(filter(_paired, bound.config_spec.compiler_seed_configs))
    config = bound.config_spec.normalized_config(seed)
    assert _policy(config) == POLICY
    return config


def _carries_contract(bound: BoundKernel[object], config: helion.Config) -> bool:
    source = bound.to_code(config)
    contract = "capture_pdl_contract" in source
    assert ("capture_pdl_wait_mode" in source) is contract
    return contract


@pytest.mark.parametrize(
    "overrides,policy",
    [
        # The two documented opt-outs,
        ({PDL: False}, OFF),
        ({CAPTURE: False}, OFF),
        # the overlap opted out with ordinary dependent launch kept, by name
        # or by naming the release it runs on,
        ({PDL: True, CAPTURE: False}, (True, False, "auto", False)),
        ({CAPTURE: False, RELEASE: "auto"}, (True, False, "auto", False)),
        ({CAPTURE: False, RELEASE: "entry"}, (True, False, "entry", False)),
        ({CAPTURE: False, RELEASE: "last_wave"}, (True, False, "last_wave", False)),
        # an opt-out beside another default restated, which changes nothing,
        ({CAPTURE: False, SPREAD: False}, OFF),
        ({PDL: False, RELEASE: "auto"}, OFF),
        # and every other override the policy cannot keep.
        ({RELEASE: "exit"}, (False, False, "exit", False)),
        ({SPREAD: True}, (False, False, "auto", True)),
    ],
)
def test_unsearched_config_is_declined_as_a_search_declines_its_seeds(
    overrides: dict[str, object], policy: tuple[object, ...]
) -> None:
    # One rule for both: what the override hook leaves a seed that carried
    # the policy is what a config the kernel selects without a search gets.
    bound = _bound()
    spec = bound.config_spec
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        seeded = [
            config
            for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
            if _paired(config)
        ]
        searched = [
            config
            for _flat, config in spec.create_config_generation(
                overrides=overrides
            ).seed_flat_config_pairs()
            if _paired(config)
        ]
        assert seeded and len(searched) == len(seeded)
        for seed, candidate in zip(seeded, searched, strict=True):
            assert _policy(seed) == POLICY
            before = seed.to_json()
            selected = spec.unsearched_config(seed, overrides)
            # The config a cache holds is never edited.
            assert selected is not seed and seed.to_json() == before
            assert _policy(candidate) == policy
            assert spec.normalized_config(selected) == candidate
            # Asked again, the hook leaves its own result alone: a cache hit
            # passes through it in the cache and once more in the backend.
            assert spec.unsearched_config(selected, overrides) is selected
            # Only the dependent-launch options differ from the seed.
            assert {k: v for k, v in selected.items() if k not in NAMES} == {
                k: v for k, v in seed.items() if k not in NAMES
            }


@pytest.mark.parametrize(
    "overrides",
    [
        # Overrides the policy keeps, also where a search would move the
        # release or restate the default,
        {PDL: True},
        {RELEASE: "entry"},
        {RELEASE: "auto"},
        {RELEASE: "loader_tail"},
        {RELEASE: "last_wave"},
        {SPREAD: False},
        {CAPTURE: True},
        # and overrides of other options, which a search alone applies.
        {"cute_mma_dag_v_lag": 3},
        {"cute_min_blocks_per_mp": 1},
    ],
)
def test_other_overrides_act_on_a_search_only(overrides: dict[str, object]) -> None:
    tuned = _tuned(_bound())
    bound = _bound(overrides)
    assert bound.config_spec.unsearched_config(tuned, overrides) is tuned
    # Nothing to hold an export to: the kernel generates what a kernel
    # without overrides generates.
    assert bound.config_spec.unsearched_overrides_identity(overrides) is None
    # Through the kernel's autotuner: the hit is the object the cache read.
    with patch.object(LocalAutotuneCache, "get", return_value=tuned):
        assert bound.env.backend.autotune(bound, _args(), force=False) is tuned


def test_the_held_overrides_are_the_policys_own_options() -> None:
    # What is generated from an unsearched config (an AOT standalone export)
    # is identified by the overrides that can decline the policy, and by no
    # other: a search-only override leaves an export in the entry it is in.
    spec = _bound().config_spec
    assert spec.unsearched_overrides_identity(None) is None
    assert spec.unsearched_overrides_identity({}) is None
    assert spec.unsearched_overrides_identity({"cute_mma_dag_v_lag": 3}) is None
    identities = [
        spec.unsearched_overrides_identity(overrides)
        for overrides in (
            {PDL: False},
            {CAPTURE: False},
            {PDL: True, CAPTURE: False},
            {RELEASE: "exit"},
            {SPREAD: True},
        )
    ]
    assert None not in identities and len(set(identities)) == len(identities)
    assert spec.unsearched_overrides_identity(
        {PDL: False, "cute_mma_dag_v_lag": 3}
    ) == spec.unsearched_overrides_identity({PDL: False})
    envelope = CuteStructuralConfig({PDL: False}, CuteStructuralPolicy())
    assert spec.unsearched_overrides_identity(envelope) == identities[0]


# Schedule overrides: the ones that leave the paired schedule, and the ones
# that restate it.
OFF_SCHEDULE = [
    {DAG: "off"},
    {DAG: "warp"},
    {PASS_WARPS: "attention"},
    {DAG: "warp", PASS_WARPS: "attention"},
    {DAG: "tcgen05", PASS_WARPS: "attention"},
]
ON_SCHEDULE = [
    {DAG: "tcgen05"},
    {PASS_WARPS: "attention_paired"},
    {DAG: "tcgen05", PASS_WARPS: "attention_paired"},
]


def test_schedule_overrides_act_on_a_search_only() -> None:
    # A schedule override is no opt-out. In a search it moves every
    # candidate, and one that leaves the paired schedule leaves no candidate
    # with the policy. A config no search produced keeps the schedule it
    # selects, on which its policy holds: declining there would compile a
    # paired config without the overlap, which is neither what its search
    # selected nor what a search under the override returns. So the config
    # is compiled as it is, and the override holds nothing: its kernel
    # shares the AOT standalone export of a kernel without overrides.
    bound = _bound()
    spec = bound.config_spec
    tuned = _tuned(bound)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        # The seeds of a search under an override to another schedule.
        searched = [
            config
            for _flat, config in spec.create_config_generation(
                overrides=OFF_SCHEDULE[2]
            ).seed_flat_config_pairs()
        ]
    assert searched and not any(map(_paired, searched))
    for overrides in (*OFF_SCHEDULE, *ON_SCHEDULE):
        assert spec.unsearched_overrides_identity(overrides) is None
        assert spec.unsearched_config(tuned, overrides) is tuned
    # Beside an option of the policy, that option alone is held and declines:
    # the config loses the overlap and keeps the schedule it selects.
    held = json.dumps({CAPTURE: False})
    declined = cast("helion.Config", spec.unsearched_config(tuned, {CAPTURE: False}))
    assert _paired(declined) and _resolved(bound, declined) == OFF
    for overrides in (*OFF_SCHEDULE, *ON_SCHEDULE):
        both = {CAPTURE: False, **overrides}
        assert spec.unsearched_overrides_identity(both) == held
        assert spec.unsearched_config(tuned, both) == declined


def _carrying(bound: BoundKernel[object]) -> list[helion.Config]:
    """A config with the overlap in every form one can have it: the default
    policy left out, and ``cute_pdl`` with the overlap on each release a
    capture launch can use, the automatic one named and left out, with
    spread padding named off and left out."""
    tuned = _tuned(bound)
    rest = {k: v for k, v in tuned.items() if k not in NAMES}
    configs = [helion.Config.from_dict(rest)]
    for release in (
        {},
        {RELEASE: "auto"},
        {RELEASE: "entry"},
        {RELEASE: "loader_tail"},
    ):
        for spread in ({}, {SPREAD: False}):
            configs.append(
                helion.Config.from_dict(
                    {**rest, PDL: True, CAPTURE: True, **release, **spread}
                )
            )
    assert all(_resolved(bound, config)[:2] == (True, True) for config in configs)
    return configs


def test_overrides_hold_something_exactly_where_they_decline_a_config() -> None:
    # The held overrides are empty for exactly the overrides that leave
    # every config with the overlap as it is. A kernel under those shares
    # the export of a kernel without overrides, whose code it generates;
    # a kernel under any other set exports from an entry of its own, because
    # some config compiles to other code under it.
    #
    # Every combination, because the decline reads the entries together
    # (``cute_pdl`` without a value declines beside a named release and not
    # alone): each of the four options left out, without a value and at
    # values the policy keeps and cannot keep, beside a schedule override
    # that restates the paired schedule, leaves it or is absent.
    bound = _bound()
    spec = bound.config_spec
    configs = _carrying(bound)
    absent = object()
    choices: dict[str, tuple[object, ...]] = {
        PDL: (absent, None, True, False),
        CAPTURE: (absent, None, True, False),
        RELEASE: (absent, None, "auto", "entry", "loader_tail", "last_wave", "exit"),
        SPREAD: (absent, None, False, True),
        PASS_WARPS: (absent, "attention_paired", "attention"),
    }
    declining = 0
    for values in itertools.product(*choices.values()):
        overrides = {
            name: value
            for name, value in zip(choices, values, strict=True)
            if value is not absent
        }
        declines = any(
            spec.unsearched_config(config, overrides) is not config
            for config in configs
        )
        held = spec.unsearched_overrides_identity(overrides) is not None
        assert held is declines, overrides
        declining += declines
    # Both kinds occur: most sets decline, and the rest are the ones the two
    # tests above list and their combinations.
    assert 0 < declining < math.prod(map(len, choices.values()))


def test_only_a_config_with_the_overlap_is_declined() -> None:
    bound = _bound()
    spec = bound.config_spec
    tuned = _tuned(bound)
    for options in (
        # Declined already, ordinary dependent launch, the unpaired schedule.
        {PDL: False},
        {PDL: True},
        {PDL: True, "cute_mma_dag_pass_warps": "attention"},
    ):
        config = helion.Config.from_dict(
            {**{k: v for k, v in tuned.items() if k not in NAMES}, **options}
        )
        assert spec.normalized_config(config).get(CAPTURE, False) is False
        for overrides in OPT_OUTS:
            assert spec.unsearched_config(config, overrides) is config
    # The overlap on an entry release holds without the overlap: its opt-out
    # keeps the dependent launch and the release, as for a search candidate.
    # So does the overlap on the release a config names ``last_wave``.
    for release in ("entry", "last_wave"):
        named = helion.Config.from_dict({**tuned.config, RELEASE: release})
        assert _resolved(bound, named) == (True, True, release, False)
        kept = spec.unsearched_config(named, {CAPTURE: False})
        assert _resolved(bound, kept) == (True, False, release, False)
        assert _resolved(bound, spec.unsearched_config(named, {PDL: False})) == OFF
    # So does the overlap on an automatic release, named or left out.
    rest = {k: v for k, v in tuned.items() if k not in NAMES}
    for release in ({RELEASE: "auto"}, {}):
        automatic = helion.Config.from_dict(
            {**rest, PDL: True, CAPTURE: True, **release}
        )
        assert _resolved(bound, automatic) == (True, True, "auto", False)
        kept = spec.unsearched_config(automatic, {CAPTURE: False})
        assert _resolved(bound, kept) == (True, False, "auto", False)
        assert _resolved(bound, spec.unsearched_config(automatic, {PDL: False})) == OFF
    # A config that leaves the policy to the default carries it too.
    implicit = helion.Config.from_dict(
        {k: v for k, v in tuned.items() if k not in NAMES}
    )
    assert _resolved(bound, implicit) == POLICY
    for overrides in OPT_OUTS:
        assert _resolved(bound, spec.unsearched_config(implicit, overrides)) == OFF
    # Overrides inconsistent in themselves are rejected, as a search rejects
    # the candidate.
    broken = spec.unsearched_config(tuned, {PDL: False, RELEASE: "loader_tail"})
    with pytest.raises(helion.exc.InvalidConfig, match="loader_tail"):
        spec.normalized_config(broken)
    # A policy envelope set as a late override names the same options.
    envelope = CuteStructuralConfig({PDL: False}, CuteStructuralPolicy())
    assert _resolved(bound, spec.unsearched_config(tuned, envelope)) == OFF


def test_overrides_that_decline_nothing_rewrite_nothing() -> None:
    # What the kernel would compile decides. It keeps the config's schedule
    # and names an option only where it gives a value, so these overrides
    # leave the overlap where it was: the config is not replaced by another
    # one that has it (the default policy in place of a stored release).
    bound = _bound()
    spec = bound.config_spec
    tuned = _tuned(bound)
    entry = helion.Config.from_dict({**tuned.config, RELEASE: "entry"})
    implicit = helion.Config.from_dict(
        {k: v for k, v in tuned.items() if k not in NAMES}
    )
    configs = (tuned, entry, implicit)
    for overrides in (
        # No value for ``cute_pdl``: no option of the policy is left, and
        # the config would take the default back.
        {PDL: None},
        {PDL: None, SPREAD: None},
        # Capture overlap forced on beside another schedule: the config
        # keeps its own schedule, where the overlap holds.
        {PDL: True, CAPTURE: True, "cute_mma_dag_pass_warps": "attention"},
    ):
        for config in configs:
            assert spec.unsearched_config(config, overrides) is config
        # They decline no config, so they hold nothing.
        assert spec.unsearched_overrides_identity(overrides) is None
    # Through the kernel: the stored release, not the default's.
    unset = _bound({PDL: None})
    with patch.object(LocalAutotuneCache, "get", return_value=entry):
        assert unset.env.backend.autotune(unset, _args(), force=False) is entry
    # An override that does decline still does beside an option without a
    # value, and the config it leaves names none.
    for overrides in (
        {PDL: False, CAPTURE: None},
        {CAPTURE: False, PDL: None},
        {CAPTURE: None},
        {RELEASE: None},
    ):
        for config in configs:
            selected = spec.unsearched_config(config, overrides)
            assert selected is not config and None not in selected.values()
            assert _resolved(bound, selected)[1] is False


@pytest.mark.parametrize("kind", ["dict", "envelope"])
def test_opt_out_reaches_whatever_an_autotuner_hands_back(
    kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``set_config`` reads a ``Config``, a plain dict of its options and a
    # policy envelope alike, so a custom autotuner may hand back any of them.
    tuned = _tuned(_bound())
    policy = CuteStructuralPolicy()

    def declared(returned: object, overrides: dict[str, object]) -> BoundKernel[object]:
        class Stored:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def autotune(self, *, skip_cache: bool = False) -> object:
                return returned

        settings: dict[str, object] = (
            {"cute_structural_policy": policy} if kind == "envelope" else {}
        )
        kernel = helion.kernel(
            paged_online_attention.fn,
            backend="cute",
            static_shapes=True,
            autotuner_fn=Stored,
            autotune_config_overrides=overrides,
            **settings,
        )
        return cpu_bind(cast("Kernel[object]", kernel), _args())

    returned = (
        dict(tuned.config) if kind == "dict" else CuteStructuralConfig(tuned, policy)
    )
    for overrides, expected in (
        ({}, POLICY),
        # An override that acts on a search only: handed back as it is.
        ({"cute_mma_dag_v_lag": 3}, POLICY),
        *((opt_out, OFF) for opt_out in OPT_OUTS),
    ):
        bound = declared(returned, overrides)
        capsys.readouterr()
        selected = bound.autotune(_args(), force=False)
        assert (selected is returned) is (expected == POLICY)
        # The kernel says so when it compiles another config than the one
        # an autotuner of the application's own returned.
        notice = "autotune_config_overrides change the config the autotuner returned"
        assert (notice in capsys.readouterr().err) is (expected == OFF)
        assert bound._config is not None
        assert _resolved(bound, bound._config) == expected
        assert _carries_contract(bound, bound._config) is (expected == POLICY)
    # An envelope answers for its policy where the kernel reads it; so it
    # does where an opt-out replaces its config, which carries no policy.
    foreign = CuteStructuralConfig(
        tuned, CuteStructuralPolicy(cute_region_fission=True)
    )
    if kind == "envelope":
        for overrides in ({}, *OPT_OUTS):
            with pytest.raises(StructuralPolicyError, match="structural policy"):
                declared(foreign, overrides).autotune(_args(), force=False)


@pytest.mark.parametrize("overrides", OPT_OUTS)
def test_opt_out_reaches_a_result_cached_without_it(
    overrides: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    # The application tuned once; the result carries the guarded default.
    tuned_bound = _bound()
    tuned = _tuned(tuned_bound)
    stored = _autotuner(tuned_bound, _args())
    assert isinstance(stored, LocalAutotuneCache)
    stored.put(tuned)
    path = stored._get_local_cache_path()
    entry = path.read_text()
    assert _carries_contract(tuned_bound, tuned)

    # It then declares that it edits its captured graphs. The cache identity
    # does not hold the overrides, so the kernel reads the same entry back
    # (a miss would raise: HELION_ASSERT_CACHE_HIT).
    bound = cpu_bind(_kernel(overrides), _args())
    cache = _autotuner(bound, _args())
    assert isinstance(cache, LocalAutotuneCache)
    assert cache.key == stored.key
    assert cache.get() == tuned
    capsys.readouterr()
    selected = bound.autotune(_args(), force=False)
    assert bound._config is selected
    assert _resolved(bound, selected) == OFF
    # The cache reports the config the kernel compiles, and that the
    # overrides changed its entry.
    reported = capsys.readouterr().err
    assert "Using cached config, as autotune_config_overrides change it" in reported
    assert "cute_pdl=False" in reported
    assert "cute_pdl_capture_overlap=True" not in reported
    assert not _carries_contract(bound, selected)
    assert "._helion_cute_use_pdl = True" not in bound.to_code(selected)
    # Nothing but the dependent-launch policy changed, and the entry is as
    # the search wrote it.
    assert {k: v for k, v in selected.items() if k not in NAMES} == {
        k: v for k, v in tuned.items() if k not in NAMES
    }
    assert path.read_text() == entry

    # An application may call the kernel's autotuner itself and pass the
    # result on. The hit holds the opt-out as a search under it would.
    driven = cpu_bind(_kernel(overrides), _args())
    driven.set_config(_autotuner(driven, _args()).autotune())
    assert driven._config is not None
    assert _resolved(driven, driven._config) == OFF
    assert not _carries_contract(driven, driven._config)
    assert path.read_text() == entry


@pytest.mark.parametrize(
    "cls",
    [
        LocalAutotuneCache,
        StrictLocalAutotuneCache,
        RemoteAutotuneCache,
        StrictRemoteAutotuneCache,
    ],
)
def test_opt_out_reaches_every_cache(
    cls: type[LocalAutotuneCache], monkeypatch: pytest.MonkeyPatch
) -> None:
    def autotuner(bound: BoundKernel[object]) -> LocalAutotuneCache:
        monkeypatch.setattr(bound.settings, "autotune_cache", cls.__name__)
        cache = _autotuner(bound, _args())
        assert type(cache) is cls
        return cache

    tuned_bound = _bound()
    tuned = _tuned(tuned_bound)
    _REMOTE.records.clear()
    stored = autotuner(tuned_bound)
    stored.put(tuned)
    remote = issubclass(cls, (RemoteAutotuneCache, StrictRemoteAutotuneCache))
    if remote:
        # A remote hit alone: it is read through to the local cache as stored.
        stored._get_local_cache_path().unlink()
    for overrides in OPT_OUTS:
        bound = _bound(overrides)
        cache = autotuner(bound)
        assert cache.key == stored.key
        # ``get`` returns the entry as stored. ``autotune`` returns what the
        # kernel compiles, on a hit as after a search, also to an
        # application that calls it itself.
        assert cache.get() == tuned
        assert _resolved(bound, cache.autotune()) == OFF
        selected = bound.env.backend.autotune(bound, _args(), force=False)
        assert _resolved(bound, selected) == OFF
        written = json.loads(cache._get_local_cache_path().read_text())
        assert helion.Config.from_json(written["config"]) == tuned
    assert bool(_REMOTE.records) is remote


@pytest.mark.parametrize("overrides", OPT_OUTS)
def test_opt_out_reaches_the_default_of_a_kernel_that_does_not_autotune(
    overrides: dict[str, object],
) -> None:
    for declared in ({}, overrides):
        bound = cpu_bind(_kernel(declared, autotune_effort="none"), _args())
        spec = bound.config_spec
        # A promoted paired default carries the policy.
        spec.compiler_default_config = _tuned(bound)
        default = spec.default_config()
        assert _paired(default) and _policy(default) == POLICY
        expected = OFF if declared else POLICY
        # The call path, and an explicit ``autotune`` of such a kernel.
        implicit = bound._require_implicit_config()
        assert _resolved(bound, implicit) == expected
        selected = bound.autotune(_args())
        assert _resolved(bound, selected) == expected
        assert _carries_contract(bound, selected) is (not declared)
        assert {k: v for k, v in selected.items() if k not in NAMES} == {
            k: v for k, v in default.items() if k not in NAMES
        }


@pytest.mark.parametrize("overrides", OPT_OUTS)
def test_opt_out_reaches_the_source_a_handoff_exports(
    overrides: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    # With ``autotune_handoff`` the native-source bundle is built from the
    # config the autotuner returns, before the kernel compiles anything. On
    # a cache hit that has to be the config the kernel compiles.
    tuned_bound = _bound()
    tuned = _tuned(tuned_bound)
    _autotuner(tuned_bound, _args()).put(tuned)

    def agent(request: object) -> None:
        raise AssertionError("the stub bundle below runs no round")

    bound = cpu_bind(
        _kernel(overrides, autotune_handoff=True, autotune_handoff_agent=agent),
        _args(),
    )
    exported: list[helion.Config] = []

    def confirm(
        session: object, extra: helion.Config | None = None
    ) -> tuple[handoff.HandoffCandidate, ...]:
        # Nothing is timed without a GPU: the returned config is confirmed.
        assert extra is not None
        return (handoff.HandoffCandidate(extra, None, (1.0,), 1.0, 0.0),)

    def build(
        autotuner: object, point: handoff.HandoffPoint, directory: Path
    ) -> SimpleNamespace:
        exported.append(point.config)
        return SimpleNamespace(
            manifest={"baseline_evaluation": None},
            run_agent_rounds=lambda source_agent, **kwargs: SimpleNamespace(
                stop_reason="stub", directory=directory
            ),
        )

    monkeypatch.setattr(handoff._HandoffSession, "confirm", confirm)
    monkeypatch.setattr(handoff_pipeline, "build_handoff", build)
    # A hand-off that starts says that it is experimental.
    with pytest.warns(UserWarning, match="hand-off .* is experimental"):
        selected = bound.env.backend.autotune(bound, _args(), force=False)
    (source_config,) = exported
    assert _resolved(bound, source_config) == OFF
    assert not _carries_contract(bound, source_config)
    assert _resolved(bound, selected) == OFF


def test_a_config_the_application_passes_is_compiled_as_written() -> None:
    # The overrides bind what the kernel selects. A config the application
    # writes itself opts out in the config.
    tuned = _tuned(_bound())
    for config in (
        tuned,
        helion.Config.from_dict({k: v for k, v in tuned.items() if k not in NAMES}),
    ):
        kernel = cast(
            "Kernel[object]",
            helion.kernel(
                paged_online_attention.fn,
                backend="cute",
                static_shapes=True,
                config=config,
                autotune_config_overrides={PDL: False},
            ),
        )
        bound = cpu_bind(kernel, _args())
        selected = bound._require_implicit_config()
        assert selected is bound.kernel.configs[0]
        assert _resolved(bound, selected) == POLICY
        bound.set_config(selected)
        assert bound._config is not None and _carries_contract(bound, bound._config)


def _aot_autotuner(
    bound: BoundKernel[object], monkeypatch: pytest.MonkeyPatch
) -> AOTAutotuneCache:
    monkeypatch.setattr(bound.settings, "autotune_cache", "AOTAutotuneCache")
    cache = _autotuner(bound, _args())
    assert type(cache) is AOTAutotuneCache
    return cache


def _export_path(written: dict[str, Any]) -> Path:
    """A path for the writer these tests replace to return, beside the
    kernel's source as the file it would have written is: the cache only
    reports it."""
    source = Path(written["kernel_source_file"])
    return source.with_name(f"{source.stem}_{written['kernel_name']}_standalone.py")


def _export_entries() -> dict[str, int]:
    """The entries the AOT cache keeps the variants of a static export
    without a structural policy in, by the overrides their kernels are held
    to ("" for none), which end such an entry's key: the number of variants
    each holds. These tests export one function from one heuristic, so
    nothing else tells two entries apart."""
    entries = AOTAutotuneCache._compiled_kernel_variants
    held = [cast("str", key[-1]) for key in entries]
    assert len(set(held)) == len(held)
    return dict(zip(held, map(len, entries.values()), strict=True))


def _export_owner() -> str:
    """The held overrides of the entry (``_export_entries``) that wrote
    the standalone file last, for tests whose kernels export to one file."""
    (owner,) = AOTAutotuneCache._standalone_export_owners.values()
    assert owner in AOTAutotuneCache._compiled_kernel_variants
    return cast("str", owner[-1])


def _export_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What the AOT cache logged, of what ``caplog`` holds."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "helion.autotuner.aot_cache"
    ]


@contextmanager
def _aot_tuning_data(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[helion.Config, AOTAutotuneCache, list[list[str]]]]:
    """Tuning data collected without the opt-out and the heuristic made of
    it, read in ``mode``: the tuned config, the cache that collected it and
    the code of every standalone export written meanwhile."""
    tuned_bound = _bound()
    tuned = _tuned(tuned_bound)
    name = tuned_bound.kernel.name
    monkeypatch.setenv("HELION_AOT_MODE", "collect")
    collected = _aot_autotuner(tuned_bound, monkeypatch)
    collected._add_tuned_config(name, tuned, collected.shape_key)
    collected._save_tuned_configs()
    heuristic = tmp_path / "heuristic.py"
    heuristic.write_text(
        f"CONFIGS = [{tuned.config!r}]\n"
        f"def autotune_{name}(*args):\n    return CONFIGS[0]\n"
    )
    monkeypatch.setenv("HELION_AOT_MODE", mode)
    exported: list[list[str]] = []

    def export(**kwargs: Any) -> Path:
        exported.append(kwargs["triton_codes"])
        return _export_path(kwargs)

    with (
        patch.object(AOTAutotuneCache, "_find_heuristic_file", return_value=heuristic),
        patch.object(
            AOTAutotuneCache, "measure_all_configs", return_value=[(tuned, 1.0)]
        ),
        patch("helion.autotuner.aot_compile.generate_standalone_file", export),
    ):
        yield tuned, collected, exported


@pytest.mark.parametrize("mode", ["collect", "measure", "evaluate", "compile"])
def test_opt_out_reaches_aot_tuning_data(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _aot_tuning_data(mode, monkeypatch, tmp_path) as (tuned, collected, exported):
        # One process, three kernels over the same function: the standalone
        # export of each override set is generated for it, with nothing
        # cleared in between.
        for overrides, expected in (({}, POLICY), *((o, OFF) for o in OPT_OUTS)):
            bound = _bound(overrides)
            assert _aot_autotuner(bound, monkeypatch).mode == mode
            selected = bound.env.backend.autotune(bound, _args(), force=False)
            if mode != "compile":
                # The autotuner returns the same to a caller of its own.
                driven = _aot_autotuner(bound, monkeypatch).autotune()
                assert _resolved(bound, driven) == expected
            assert _resolved(bound, selected) == expected
            if mode == "compile":
                # The standalone file is generated from the same tuning data.
                (code,) = exported.pop()
                assert ("capture_pdl_contract" in code) is (expected == POLICY)
            assert not exported
    # The tuning data itself is as it was collected, and so is the
    # heuristic's answer, which every kernel of the process shares: the one
    # ``get()`` keeps for the heuristic's path and, in compile mode, the one
    # the exports keep for the bytes they read.
    assert _policy(tuned) == POLICY
    shared = list(AOTAutotuneCache._heuristic_results.values())
    assert len(shared) == {"evaluate": 1, "compile": 2}.get(mode, 0)
    assert all(_policy(cast("helion.Config", config)) == POLICY for config in shared)
    name = _bound().kernel.name
    assert [_policy(c) for c in collected._get_all_configs_for_kernel(name)] == [POLICY]


def test_aot_export_follows_an_opt_out_set_on_an_exported_kernel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An application that starts editing its captured graphs sets the opt-out
    # on the kernel it already exported and resets it. The same call then
    # compiles to other code, which is exported from an entry of the
    # opt-out: it does not collide with the first variant, and the file is
    # written with it alone. Taking the opt-out away again does not find the
    # first export still in place: the file is written from the first entry
    # again, which kept its variant.
    kernel = _kernel({})
    with _aot_tuning_data("compile", monkeypatch, tmp_path) as (_tuned, _, exported):
        previous: dict[str, object] | None = None
        for overrides, contract in (
            ({}, True),
            ({PDL: False}, False),
            ({PDL: False}, False),
            ({}, True),
            ({CAPTURE: False}, False),
        ):
            kernel.settings.autotune_config_overrides = overrides
            kernel.reset()
            bound = cpu_bind(kernel, _args())
            _aot_autotuner(bound, monkeypatch)
            selected = bound.env.backend.autotune(bound, _args(), force=False)
            assert _carries_contract(bound, selected) is contract
            if overrides == previous:
                # Unchanged overrides: the variant is known, nothing to write.
                assert not exported
            else:
                (code,) = exported.pop()
                assert ("capture_pdl_contract" in code) is contract
            previous = overrides
            # The file is of the entry the kernel's overrides hold it to.
            assert _export_owner() == (json.dumps(overrides) if overrides else "")
        # An entry per set of overrides, each with the one call.
        assert _export_entries() == {
            "": 1,
            json.dumps({PDL: False}): 1,
            json.dumps({CAPTURE: False}): 1,
        }


def test_aot_export_keeps_the_shapes_of_one_kernel_in_one_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The capture proof belongs to a binding: this kernel has it with two
    # heads per request and not with one. A standalone export is held to
    # the kernel's overrides, whichever shape is bound, so one kernel under
    # an unchanged opt-out exports both shapes from one entry into one file.
    # An identity that followed the proof would put the two shapes into two
    # entries: each change of shape would write the file from the other
    # entry, and leave it with the variant of the last call alone.
    kernel = _kernel({PDL: False})
    name = kernel.name
    shapes = {heads: _args(heads) for heads in (2, 1)}
    bound = {heads: cpu_bind(kernel, args) for heads, args in shapes.items()}
    spec = {heads: binding.config_spec for heads, binding in bound.items()}
    assert spec[2].cute_pdl_capture_overlap_available
    assert not spec[1].cute_pdl_capture_overlap_available
    identity = bound[2]._unsearched_overrides_identity()
    assert identity == json.dumps({PDL: False})
    assert bound[1]._unsearched_overrides_identity() == identity
    # Without the proof there is still nothing to decline.
    single = spec[1].default_config()
    assert spec[1].unsearched_config(single, {PDL: False}) is single

    written: list[dict[str, object]] = []

    def export(**kwargs: Any) -> Path:
        written.append(kwargs)
        return _export_path(kwargs)

    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    with (
        _aot_tuning_data("compile", monkeypatch, tmp_path) as (tuned, _, _exported),
        patch("helion.autotuner.aot_compile.generate_standalone_file", export),
    ):
        # Tuning data for both shapes: the tuned paired config, and the
        # default of the shape that has no paired schedule. No ``CONFIGS``
        # list: a static export normalizes every entry of one for the
        # binding at hand, and neither config is valid for the other shape.
        rows = cast("torch.Tensor", shapes[2][0]).shape[0]
        (tmp_path / "heuristic.py").write_text(
            f"def autotune_{name}(*args):\n"
            f"    if args[0].shape[0] == {rows}:\n"
            f"        return {tuned.config!r}\n"
            f"    return {single.config!r}\n"
        )
        monkeypatch.setattr(kernel.settings, "autotune_cache", "AOTAutotuneCache")
        for heads in (2, 1, 2):
            selected = bound[heads].env.backend.autotune(
                bound[heads], shapes[heads], force=False
            )
            assert not _carries_contract(bound[heads], selected)
        entries, owner = _export_entries(), _export_owner()
    assert not _export_warnings(caplog)
    # One entry, of the kernel's overrides, which the file is written from:
    # a variant per shape, and a call already exported writes nothing.
    assert entries == {identity: 2} and owner == identity
    assert [len(cast("list[str]", call["triton_codes"])) for call in written] == [1, 2]
    # The file dispatches a call of either shape to its variant.
    guards = cast("bool", written[-1]["tensor_descriptor_guards"])
    assert set(cast("list[object]", written[-1]["dispatch_keys"])) == {
        _standalone_call_key(
            kernel.normalize_args(*args), tensor_descriptor_guards=guards
        )
        for args in shapes.values()
    }


def test_aot_export_is_shared_with_overrides_that_decline_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Live kernels over one function, one without overrides and the others
    # under overrides the decline acts on for no config, schedule overrides
    # among them. All compile the tuned config as it is, to the same code,
    # so they export from one entry: none writes the file again after the
    # first, and it keeps its variant.
    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    with _aot_tuning_data("compile", monkeypatch, tmp_path) as (_tuned, _, exported):
        for turn, overrides in enumerate(
            (
                {},
                {PDL: True},
                {},
                {RELEASE: "entry"},
                {CAPTURE: True},
                {PDL: None},
                OFF_SCHEDULE[0],
                OFF_SCHEDULE[2],
            )
        ):
            bound = _bound(overrides)
            _aot_autotuner(bound, monkeypatch)
            selected = bound.env.backend.autotune(bound, _args(), force=False)
            assert _carries_contract(bound, selected)
            if turn == 0:
                (code,) = exported.pop()
                assert "capture_pdl_contract" in code
            # Afterwards the variant is known, whichever kernel asks.
            assert not exported
        entries, owner = _export_entries(), _export_owner()
    assert not _export_warnings(caplog)
    assert entries == {"": 1} and owner == ""


def test_aot_export_of_a_kernel_without_the_proof_under_held_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Live kernels over one function, one of them under an opt-out, on a
    # shape without the capture proof (one head per request): there is no
    # policy to decline, and every kernel compiles the same config.
    #
    # The held overrides are read off the overrides alone
    # (``test_aot_export_keeps_the_shapes_of_one_kernel_in_one_file`` says
    # why), so they cannot see that. What they can see: a schedule override
    # holds nothing, and its kernel shares the export of a kernel without
    # overrides. An override that declines the policy is held for every
    # kernel: its kernel exports from an entry of its own, and each time it
    # and a kernel that holds nothing take turns the file is written again
    # from the entry of the one that exports, without a warning, with the
    # same code.
    # Only the binding's proof could tell this kernel from one that has a
    # policy to decline.
    args = _args(1)
    kernels = {
        name: _kernel(overrides)
        for name, overrides in (
            ("plain", {}),
            ("schedule", {DAG: "off"}),
            ("away", {PDL: False}),
        )
    }
    bound = {name: cpu_bind(kernel, args) for name, kernel in kernels.items()}
    spec = bound["plain"].config_spec
    assert not spec.cute_pdl_capture_overlap_available
    single = spec.default_config()
    for binding in bound.values():
        overrides = binding.settings.autotune_config_overrides
        assert binding.config_spec.unsearched_config(single, overrides) is single
    assert bound["plain"]._unsearched_overrides_identity() is None
    assert bound["schedule"]._unsearched_overrides_identity() is None
    away = json.dumps({PDL: False})
    assert bound["away"]._unsearched_overrides_identity() == away

    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    held: list[str] = []
    with _aot_tuning_data("compile", monkeypatch, tmp_path) as (_tuned, _, exported):
        name = kernels["plain"].name
        (tmp_path / "heuristic.py").write_text(
            f"def autotune_{name}(*args):\n    return {single.config!r}\n"
        )
        codes: list[str] = []
        for turn, (who, written) in enumerate(
            (
                ("plain", True),
                # One entry for the two kernels that hold nothing.
                ("schedule", False),
                # Each change of hands with the opt-out writes the file from
                # the entry of the kernel that exports: the first one again
                # for either kernel that holds nothing.
                ("away", True),
                ("schedule", True),
                ("away", True),
            )
        ):
            kernel = kernels[who]
            monkeypatch.setattr(kernel.settings, "autotune_cache", "AOTAutotuneCache")
            bound[who].env.backend.autotune(bound[who], args, force=False)
            assert bool(exported) is written, (turn, who)
            if written:
                (code,) = exported.pop()
                codes.append(code)
            held.append(_export_owner())
        entries = _export_entries()
    assert held == ["", "", away, "", away]
    # Two entries, which keep their variant between their turns.
    assert entries == {"": 1, away: 1}
    assert not _export_warnings(caplog)
    # Nothing was declined: every export wrote the same variant.
    assert len(codes) == 4 and len(set(codes)) == 1
    assert "capture_pdl_contract" not in codes[0]


def test_a_structural_aot_export_keeps_the_opt_out_it_was_written_under(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Under a structural policy a standalone file keeps the entry that wrote
    # it first, and what the kernel's overrides decline is part of that
    # entry. The opt-out reaches the export, and the same call under another
    # opt-out, or under none, is refused and writes nothing until the AOT
    # cache is cleared. Overrides that decline nothing else leave a kernel
    # in the entry: its call is known there.
    policy = CuteStructuralPolicy()

    def bind(overrides: dict[str, object]) -> BoundKernel[object]:
        bound = cpu_bind(_kernel(overrides, cute_structural_policy=policy), _args())
        _aot_autotuner(bound, monkeypatch)
        return bound

    plain = bind({})
    name = plain.kernel.name
    model = tmp_path / "model.py"
    model.write_text(
        attach_model_policy(
            f"def key_{name}(*args):\n    return 0\n"
            f"def autotune_{name}(*args):\n    return {{}}\n",
            {name: [_tuned(plain)]},
            policy,
        )
    )
    monkeypatch.setenv("HELION_AOT_MODE", "compile")
    exported: list[list[str]] = []

    def export(**kwargs: Any) -> Path:
        exported.append(kwargs["triton_codes"])
        return _export_path(kwargs)

    def autotune(overrides: dict[str, object]) -> bool:
        """Whether a kernel under ``overrides`` selects a config with the
        capture contract, after its export."""
        bound = bind(overrides)
        selected = bound.env.backend.autotune(bound, _args(), force=False)
        return _carries_contract(bound, selected)

    with (
        patch.object(AOTAutotuneCache, "_find_heuristic_file", return_value=model),
        patch("helion.autotuner.aot_compile.generate_standalone_file", export),
    ):
        for first, same, others in (
            ({}, {RELEASE: "entry"}, OPT_OUTS),
            (
                {PDL: False},
                {PDL: False, "cute_mma_dag_v_lag": 3},
                [{}, {CAPTURE: False}],
            ),
        ):
            contract = not first
            assert autotune(first) is contract
            (code,) = exported.pop()
            assert ("capture_pdl_contract" in code) is contract
            for overrides in others:
                with pytest.raises(StructuralPolicyError, match="already owns"):
                    autotune(overrides)
                assert not exported
            assert autotune(same) is contract
            assert not exported
            AOTAutotuneCache.clear_caches()
