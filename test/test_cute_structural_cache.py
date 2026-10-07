from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_full_slice_matmul import _matmul
from test.test_cute_shared_rhs_grouped import _target
from test.test_cute_structural_compatibility import _pointwise_bound

import helion
from helion._hardware import HardwareInfo
from helion._testing import skipUnlessBackends
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import parse_cache_entry
from helion.autotuner.metrics import KernelMetadata
from helion.autotuner.metrics import _codegen_signature
from helion.autotuner.remote_cache import RemoteAutotuneCache
from helion.autotuner.remote_cache import RemoteCacheBackend
from helion.autotuner.remote_cache import StrictRemoteAutotuneCache
from helion.runtime.cute_structural_config import CuteStructuralConfig
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute_structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in CUTE_STRUCTURAL_SETTINGS:
        monkeypatch.delenv(f"HELION_{name.upper()}", raising=False)
    monkeypatch.delenv("HELION_BACKEND", raising=False)
    monkeypatch.setenv("HELION_AUTOTUNE_RANDOM_SEED", "0")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Share one undo stack with test bodies so their changes cannot restore
    # fixture-local AOT settings after this fixture has finished.
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot"))
    monkeypatch.setenv("HELION_AOT_MODE", "collect")
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    AOTAutotuneCache.clear_caches()
    try:
        with (
            _target(),
            _mock_cuda_unavailable(),
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
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
        torch.set_num_threads(previous_threads)


def _search(policy: CuteStructuralPolicy | None) -> BaseSearch:
    bound = _pointwise_bound(policy)
    return BaseSearch(bound, (torch.empty(37),))


def _cache(
    search: BaseSearch, cls: type[LocalAutotuneCache] = LocalAutotuneCache
) -> LocalAutotuneCache:
    # Only hardware metadata is synthetic. The actual BoundKernel, argument
    # schema, source hash, config fingerprint and cache constructor are used.
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
    ):
        return cls(search)


@skipUnlessBackends(["cute"])
def test_actual_local_cache_policy_hit_miss_and_legacy_payload(tmp_path: Path) -> None:
    off = CuteStructuralPolicy()
    on = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    searches = [_search(policy) for policy in (off, off, on, None)]
    caches = [_cache(search) for search in searches]
    config = searches[0].config_spec.default_config()
    assert (
        len({search.config_spec.cache_fingerprint_hash() for search in searches}) == 1
    )
    assert caches[0].key == caches[1].key
    assert (
        len({cache.key.stable_hash() for cache in (caches[0], caches[2], caches[3])})
        == 3
    )
    caches[0].put(config)
    assert caches[1].get() == config
    assert caches[2].get() is None
    assert caches[3].get() is None
    payload = json.loads(caches[0]._get_local_cache_path().read_text())
    assert payload["cute_structural_policy"] == off.to_dict()
    assert parse_cache_entry(json.dumps(payload)).cute_structural_policy == off
    caches[3].put(config)
    legacy = json.loads(caches[3]._get_local_cache_path().read_text())
    assert "cute_structural_policy" not in legacy
    assert legacy["config"] == config.to_json()
    assert caches[3].get() == config
    assert caches[3].key.extra_cache_key == ""
    (tmp_path / "policy-cache-evidence.json").write_text(
        json.dumps(
            {
                "keys": [cache.key.stable_hash() for cache in caches],
                "source_hashes": [cache.key.kernel_source_hash for cache in caches],
                "config_fingerprints": [cache.key.config_spec_hash for cache in caches],
                "legacy_payload": legacy,
                "policy_payload": payload,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("kind", ["wrong", "missing", "bad_version"])
@skipUnlessBackends(["cute"])
def test_local_payload_checks_policy_before_reading_config(kind: str) -> None:
    search = _search(CuteStructuralPolicy())
    cache = _cache(search)
    cache.put(search.config_spec.default_config())
    path = cache._get_local_cache_path()
    payload = json.loads(path.read_text())
    if kind == "missing":
        payload.pop("cute_structural_policy")
    elif kind == "wrong":
        payload["cute_structural_policy"] = CuteStructuralPolicy(
            cute_region_fission=True
        ).to_dict()
    else:
        payload["cute_structural_policy"]["version"] = 2
    path.write_text(json.dumps(payload))
    with (
        patch.object(
            helion.Config,
            "from_json",
            side_effect=AssertionError("config must not be interpreted"),
        ),
        pytest.raises(StructuralPolicyError),
    ):
        cache.get()


@skipUnlessBackends(["cute"])
def test_similar_config_scan_filters_local_and_remote_policy_records() -> None:
    search = _search(CuteStructuralPolicy())
    cache = _cache(search)
    cache.put(search.config_spec.default_config())
    path = cache._get_local_cache_path()
    payload = json.loads(path.read_text())
    good = parse_cache_entry(json.dumps(payload))
    bad = copy.deepcopy(payload)
    bad["cute_structural_policy"] = CuteStructuralPolicy(
        cute_region_fission=True
    ).to_dict()
    # Deliberately retain the matching source/spec key and flat vector: policy
    # validation must be independent of a config's length or generated code.
    path.write_text(json.dumps(bad))
    remote = _MemoryRemote()
    remote.records["bad"] = json.dumps(bad)
    remote.records["good"] = json.dumps(payload)
    with (
        patch.object(
            search,
            "_get_current_hardware_and_specialization",
            return_value=(good.hardware, good.specialization_key),
        ),
        patch(
            "helion.autotuner.remote_cache._load_remote_backend_if_configured",
            return_value=remote,
        ),
    ):
        matched = search._find_similar_cached_configs(8)
    assert len(matched) == 1
    assert matched[0].cute_structural_policy == CuteStructuralPolicy()


class _MemoryRemote(RemoteCacheBackend):
    def __init__(self) -> None:
        self.records: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.records.get(key)

    def put(self, key: str, data: str) -> None:
        self.records[key] = data

    def list(self, max_results: int | None = None) -> list[str]:
        return list(self.records.values())[:max_results]


@pytest.mark.parametrize("cls", [RemoteAutotuneCache, StrictRemoteAutotuneCache])
@skipUnlessBackends(["cute"])
def test_actual_remote_read_through_checks_and_preserves_policy(
    cls: type[LocalAutotuneCache],
) -> None:
    remote = _MemoryRemote()
    search = _search(CuteStructuralPolicy())
    with patch(
        "helion.autotuner.remote_cache._load_remote_backend", return_value=remote
    ):
        cache = _cache(search, cls)
    config = search.config_spec.default_config()
    cache.put(config)
    path = cache._get_local_cache_path()
    assert remote.records[cache.key.stable_hash()] == path.read_text()
    path.unlink()
    assert cache.get() == config
    assert (
        json.loads(path.read_text())["cute_structural_policy"]
        == CuteStructuralPolicy().to_dict()
    )
    foreign = json.loads(remote.records[cache.key.stable_hash()])
    foreign["cute_structural_policy"] = CuteStructuralPolicy(
        cute_region_fission=True
    ).to_dict()
    remote.records[cache.key.stable_hash()] = json.dumps(foreign)
    path.unlink()
    with (
        patch.object(
            helion.Config,
            "from_json",
            side_effect=AssertionError("config must not be interpreted"),
        ),
        pytest.raises(StructuralPolicyError, match="Cached config"),
    ):
        cache.get()
    assert not path.exists()


@skipUnlessBackends(["cute"])
def test_actual_aot_collection_save_load_dedup_and_policy_filtering(
    tmp_path: Path,
) -> None:
    policies = [
        CuteStructuralPolicy(),
        CuteStructuralPolicy(cute_region_fission=True),
        None,
    ]
    searches = [_search(policy) for policy in policies]
    for search, policy in zip(searches, policies, strict=True):
        cache = AOTAutotuneCache(search)
        assert cache.get() is None
        config = search.config_spec.default_config()
        cache._add_tuned_config(search.kernel.kernel.name, config, cache.shape_key)
        cache._add_tuned_config(
            search.kernel.kernel.name, copy.deepcopy(config), cache.shape_key
        )
        assert len(cache._get_all_configs_for_kernel(search.kernel.kernel.name)) == 1
        cache._save_tuned_configs()
        restored = AOTAutotuneCache(search)
        assert restored.get() == config
        if policy is not None:
            config.block_sizes[0] *= 2
            assert restored.get() != config
            selected = restored.get()
            assert selected is not None
            selected.block_sizes[0] *= 2
            assert restored.get() != selected
    final = AOTAutotuneCache(searches[0])
    rows = json.loads(final._configs_file.read_text())[searches[0].kernel.kernel.name]
    assert len(rows) == 3
    assert [row.get("cute_structural_policy") for row in rows] == [
        policy.to_dict() if policy is not None else None for policy in policies
    ]
    for search in searches:
        restored = AOTAutotuneCache(search)
        assert restored.get() == search.config_spec.default_config()
    (tmp_path / "aot-roundtrip-evidence.json").write_text(json.dumps(rows, indent=2))


@pytest.mark.parametrize("via_envelope", [False, True])
@skipUnlessBackends(["cute"])
def test_public_aot_decorator_resolves_policy_before_shape_binding(
    via_envelope: bool,
) -> None:
    policy = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    config = helion.Config(block_sizes=[16, 32, 16])
    declaration = CuteStructuralConfig(config, policy) if via_envelope else config
    kernel = helion.aot_kernel(
        _matmul,
        config=declaration,
        backend="cute",
        static_shapes=True,
        cute_structural_policy=None if via_envelope else policy,
    )
    args = (
        torch.empty((32, 16), dtype=torch.float16),
        torch.empty((16, 64), dtype=torch.float16),
    )
    bound = kernel._bind_isolated(args)
    assert kernel.cute_structural_policy == policy
    assert len(bound.config_spec.default_config().block_sizes) == 3
    search = BaseSearch(bound, args)
    cache = AOTAutotuneCache(search)
    assert cache.get() is None
    cache._add_tuned_config(kernel.name, config, cache.shape_key)
    cache._save_tuned_configs()
    restored = AOTAutotuneCache(search).get()
    assert restored == config
    assert bound.to_code(restored) == bound.to_code(config)


@skipUnlessBackends(["cute"])
def test_invalid_aot_policy_version_precedes_config_construction() -> None:
    search = _search(CuteStructuralPolicy())
    cache = AOTAutotuneCache(search)
    cache._add_tuned_config(
        search.kernel.kernel.name, search.config_spec.default_config(), cache.shape_key
    )
    cache._save_tuned_configs()
    payload = json.loads(cache._configs_file.read_text())
    payload[search.kernel.kernel.name][0]["cute_structural_policy"]["version"] = 2
    cache._configs_file.write_text(json.dumps(payload))
    with (
        patch.object(
            helion.Config,
            "__init__",
            side_effect=AssertionError("config must not be interpreted"),
        ),
        pytest.raises(StructuralPolicyError, match="version"),
    ):
        AOTAutotuneCache(search)


@pytest.mark.parametrize("new_policy", [False, True])
@skipUnlessBackends(["cute"])
def test_actual_aot_heuristic_import_and_cache_round_trip(
    new_policy: bool, tmp_path: Path
) -> None:
    policy = CuteStructuralPolicy() if new_policy else None
    search = _search(policy)
    cache = AOTAutotuneCache(search)
    config = search.config_spec.default_config()
    value = (
        repr(CuteStructuralConfig(config, policy))
        if policy is not None
        else repr(config.config)
    )
    heuristic = tmp_path / "selected_heuristic.py"
    heuristic.write_text(
        "import helion\ncalls = []\n"
        f"def autotune_{search.kernel.kernel.name}(x):\n"
        f"    calls.append(tuple(x.shape))\n    return {value}\n"
    )
    with patch.object(cache, "_find_heuristic_file", return_value=heuristic):
        assert cache._get_heuristic_config() == config
        assert cache._get_heuristic_config() == config
        module = AOTAutotuneCache._heuristic_modules[heuristic]
        assert module.calls == [(37,)]
        if new_policy:
            selected = cache._get_heuristic_config()
            assert selected is not None
            selected.block_sizes[0] *= 2
            assert cache._get_heuristic_config() == config
    other = AOTAutotuneCache(_search(CuteStructuralPolicy(cute_region_fission=True)))
    with (
        patch.object(other, "_find_heuristic_file", return_value=heuristic),
        pytest.raises(StructuralPolicyError, match="AOT heuristic"),
    ):
        other._get_heuristic_config()
    last_module = AOTAutotuneCache._heuristic_modules[heuristic]
    if new_policy:
        assert last_module is module
        assert module.calls == [(37,), (37,)]
    else:
        # The new content-keyed loader does not reuse an unverified legacy
        # bytecode module. Both actual callbacks still run exactly once.
        assert last_module is not module
        assert module.calls == last_module.calls == [(37,)]


@pytest.mark.parametrize(
    "operation",
    ["measure_all_configs", "_maybe_run_compile", "_run_measure_fn_workflow"],
)
@skipUnlessBackends(["cute"])
def test_policy_aot_workflows_accept_the_resolved_policy(operation: str) -> None:
    cache = AOTAutotuneCache(_search(CuteStructuralPolicy()))
    callback = Mock(return_value=iter(()))
    cache._measure_fn = callback
    with (
        patch.object(cache.autotuner, "_prepare") as prepare,
        patch.object(cache, "_find_heuristic_file", return_value=None),
    ):
        getattr(cache, operation)()
    assert prepare.call_count == int(operation == "measure_all_configs")
    callback.assert_not_called()  # No collected configs exist in this fixture.


@pytest.mark.parametrize("mode", ["measure", "compile"])
@skipUnlessBackends(["cute"])
def test_policy_aot_modes_keep_selected_policy_and_partition(mode: str) -> None:
    search = _search(CuteStructuralPolicy())
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setenv("HELION_AOT_MODE", mode)
        cache = AOTAutotuneCache(search)
        assert cache.mode == mode
        assert CuteStructuralPolicy().identity() in cache._measurements_file.name


def test_metadata_preserves_legacy_id_and_records_new_policy() -> None:
    settings = {"backend": "cute", "static_shapes": True}
    legacy = KernelMetadata(
        kernel_source="same source", settings=settings, hardware="cpu-proof"
    )
    old_payload = f"same source\x00{_codegen_signature(settings)}\x00\x00\x00cpu-proof"
    assert legacy.run_id == hashlib.sha256(old_payload.encode()).hexdigest()
    assert "cute_structural_policy" not in legacy.to_dict()
    first = dataclasses.replace(legacy, cute_structural_policy=CuteStructuralPolicy())
    again = dataclasses.replace(legacy, cute_structural_policy=CuteStructuralPolicy())
    other = dataclasses.replace(
        legacy, cute_structural_policy=CuteStructuralPolicy(cute_region_fission=True)
    )
    assert first.run_id == again.run_id
    assert len({legacy.run_id, first.run_id, other.run_id}) == 3
    assert first.to_dict()["cute_structural_policy"] == CuteStructuralPolicy().to_dict()


@skipUnlessBackends(["cute"])
def test_actual_search_metadata_contains_selected_policy() -> None:
    search = _search(CuteStructuralPolicy())
    # Keep the actual metadata path. The benchmark provider normally executes a
    # GPU baseline in its constructor, which is outside this CPU-only proof.
    with (
        patch("helion.autotuner.base_search.get_device_name", return_value="cpu-proof"),
        patch.object(search, "_benchmark_provider_cls", return_value=Mock()),
    ):
        search._prepare()
    assert search._kernel_metadata.cute_structural_policy == CuteStructuralPolicy()
    assert (
        search._kernel_metadata.to_dict()["cute_structural_policy"]
        == CuteStructuralPolicy().to_dict()
    )
