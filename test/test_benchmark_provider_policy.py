from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.benchmark_job import AccuracyCheckResult
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import _search_policy_hash
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from helion.autotuner.base_search import BaseSearch
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel()
def _pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        out[i] = x[i] + 1
    return out


class _AdditionalAccuracyProvider(LocalBenchmarkProvider):
    def _run_subprocess_accuracy_check_job(self, fn: object) -> AccuracyCheckResult:
        return AccuracyCheckResult(False, "additional validation rejects candidate")


def _provider_factory(**kwargs: object) -> LocalBenchmarkProvider:
    return _AdditionalAccuracyProvider(**kwargs)


class _ProviderFactory:
    def __call__(self, **kwargs: object) -> LocalBenchmarkProvider:
        return _provider_factory(**kwargs)


@pytest.fixture
def inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[BoundKernel, tuple[torch.Tensor]]]:
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    args = (torch.empty(8),)
    with cpu_render_target():
        yield cpu_bind(_pointwise, args), args


def _cache(search: BaseSearch) -> LocalAutotuneCache:
    # Keep the real bound arguments, config schema and cache construction.
    # Only the hardware identity is modeled; no candidate executes.
    with (
        patch(
            "helion.autotuner.local_cache.extract_device",
            return_value=torch.device("cuda:0"),
        ),
        patch("helion.autotuner.local_cache.get_device_name", return_value="cpu-proof"),
        patch("torch.cuda.is_available", return_value=True),
    ):
        return LocalAutotuneCache(search)


def _configs() -> list[helion.Config]:
    return [helion.Config(block_sizes=[1]), helion.Config(block_sizes=[2])]


def _check_no_cache_io(cache: LocalAutotuneCache) -> None:
    stale, validated = _configs()
    with (
        patch.object(cache, "get", return_value=stale) as get,
        patch.object(cache, "put") as put,
        patch.object(cache, "_run_autotune_trials", return_value=validated) as run,
    ):
        assert cache.autotune() == validated
    get.assert_not_called()
    put.assert_not_called()
    run.assert_called_once_with(skip_cache=True)


@pytest.mark.parametrize(
    "provider", [_AdditionalAccuracyProvider, _provider_factory, _ProviderFactory()]
)
@pytest.mark.parametrize("flash", [False, True])
def test_custom_provider_cannot_reuse_or_publish_cached_winner(inputs, provider, flash):
    bound, args = inputs
    search = FiniteSearch(bound, args, _configs(), benchmark_provider_cls=provider)
    _search_policy_hash(search, cute_flash_search_enabled=flash)
    _check_no_cache_io(_cache(search))


@pytest.mark.parametrize("flash", [False, True])
def test_lfbo_provider_installed_before_cache_creation_is_uncacheable(inputs, flash):
    bound, args = inputs
    search = LFBOTreeSearch(bound, args)
    # A factory can install its provider before constructing the cache.
    search._benchmark_provider_cls = _AdditionalAccuracyProvider
    assert search.cache_policy() is None
    _search_policy_hash(search, cute_flash_search_enabled=flash, explicit_aot=True)
    assert not search._search_policy_cacheable
    _search_policy_hash(search, cute_flash_search_enabled=flash)
    _check_no_cache_io(_cache(search))


def test_custom_provider_is_uncacheable_before_local_key_construction(inputs):
    bound, args = inputs
    search = FiniteSearch(
        bound, args, _configs(), benchmark_provider_cls=_AdditionalAccuracyProvider
    )
    wrapper = Mock(autotuner=search)
    assert not AutotuneCacheBase._search_policy_allows_cache_io(wrapper)


def test_custom_provider_policy_nonce_is_stable_only_within_one_search(inputs):
    bound, args = inputs
    searches = [
        FiniteSearch(
            bound, args, _configs(), benchmark_provider_cls=_AdditionalAccuracyProvider
        )
        for _ in range(2)
    ]
    keys = [
        _search_policy_hash(search, cute_flash_search_enabled=False)
        for search in searches
    ]
    assert keys[0] and keys[0] != keys[1]
    assert keys[0] == _search_policy_hash(searches[0], cute_flash_search_enabled=False)


@pytest.mark.parametrize("bypass", [None, "force", "skip"])
def test_custom_provider_honors_cache_hit_assertion(inputs, monkeypatch, bypass):
    bound, args = inputs
    monkeypatch.setenv("HELION_ASSERT_CACHE_HIT", "1")
    if bypass == "skip":
        monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    search = FiniteSearch(
        bound, args, _configs(), benchmark_provider_cls=_AdditionalAccuracyProvider
    )
    cache = _cache(search)
    with (
        patch.object(cache, "get") as get,
        patch.object(cache, "put") as put,
        patch.object(cache, "_get_cache_key") as key,
        patch.object(cache, "_list_cache_entries") as entries,
        patch.object(cache, "_run_autotune_trials", return_value=_configs()[1]) as run,
    ):
        if bypass is None:
            with pytest.raises(helion.exc.CacheAssertionError):
                cache.autotune()
            run.assert_not_called()
        else:
            assert cache.autotune(skip_cache=bypass == "force") == _configs()[1]
            run.assert_called_once_with(skip_cache=True)
    get.assert_not_called()
    put.assert_not_called()
    key.assert_not_called()
    entries.assert_not_called()


def test_default_provider_keeps_versioned_key_and_cache_hit(inputs):
    bound, args = inputs
    implicit = FiniteSearch(bound, args, _configs())
    explicit = FiniteSearch(
        bound, args, _configs(), benchmark_provider_cls=LocalBenchmarkProvider
    )
    first, second = _cache(implicit), _cache(explicit)
    assert first.key == second.key
    assert first.key.search_policy_hash
    cached = _configs()[0]
    with (
        patch.object(first, "get", return_value=cached) as get,
        patch.object(first, "put") as put,
        patch.object(first, "_run_autotune_trials") as run,
    ):
        assert first.autotune() == cached
    get.assert_called_once_with()
    put.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("flash", [False, True])
def test_explicit_aot_policy_is_stable_without_enabling_winner_cache(
    inputs, flash, monkeypatch
):
    bound, args = inputs
    searches = [
        FiniteSearch(
            bound, args, _configs(), benchmark_provider_cls=_AdditionalAccuracyProvider
        )
        for _ in range(2)
    ]
    keys = [
        _search_policy_hash(search, cute_flash_search_enabled=flash, explicit_aot=True)
        for search in searches
    ]
    assert keys[0] == keys[1]
    assert all(not search._search_policy_cacheable for search in searches)
    for search in searches:
        _check_no_cache_io(_cache(search))
    monkeypatch.setattr(bound.settings, "autotune_benchmark_input_banks", 3)
    assert (
        _search_policy_hash(
            searches[0], cute_flash_search_enabled=flash, explicit_aot=True
        )
        != keys[0]
    )
