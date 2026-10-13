from __future__ import annotations

import dataclasses
import functools
import hashlib
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from test.test_cute_source_batch import _bound
from test.test_cute_source_batch import _compile_prefix

from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="source admission")


@pytest.fixture(autouse=True)
def forbid_real_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("model-only retry test entered compilation or source workers")

    for name in ("to_code", "to_triton_code", "compile_config", "_compile_config"):
        monkeypatch.setattr(BoundKernel, name, forbidden)
    monkeypatch.setattr(sb, "_run_workers", forbidden)
    monkeypatch.setattr(os, "fork", forbidden)


class _CompileModel:
    """Model cache publication; run real admission and source receipt validation."""

    def __init__(self, workers: int, after_serial: object) -> None:
        self.bound, self.tensor = _bound()
        self.bound.settings.autotune_cute_source_workers = workers
        self.after_serial = after_serial
        self.calls = []
        self.serial_misses = []
        self.consumed = []
        self.prepared = []
        self.expired = False
        self.expire_during_prepare = False
        self.corrupt_receipts = False
        # _bound supplies a source fixture; it must not run in this test either.
        self.bound.to_triton_code = lambda *args, **kwargs: pytest.fail(
            "model-only test called the source fixture"
        )
        self.bound.compile_config = self.compile_config
        self.bound._compile_config = self.compile_config
        self.provider = SimpleNamespace(
            kernel=self.bound,
            args=[self.tensor],
            settings=self.bound.settings,
            config_spec=self.bound.config_spec,
            _validate_timing_policy=lambda: None,
            _effective_source_dedup_enabled=lambda: False,
            _effective_source_hashes=set(),
            _budget_exceeded_synced=lambda: self.expired,
            _autotune_metrics=SimpleNamespace(num_successful_candidate_measurements=1),
        )

    def compile_config(
        self, config: Config, *, allow_print: bool, prepared_source: object = None
    ) -> object:
        assert allow_print is False
        self.calls.append(config)
        if config in self.bound._compile_cache:
            return self.bound._compile_cache[config]
        source = (
            prepared_source.consume(self.bound, config)
            if prepared_source is not None
            else None
        )
        if source is None:
            self.serial_misses.append(config)
            self.after_serial()
        else:
            self.consumed.append(config)
        function = functools.partial(int, config.config["block_sizes"][0])
        self.bound._compile_cache[config] = function
        self.bound._cache_path_map[config] = "model://" + sb._config_key(config)
        return function

    def prepare(self, configs: list[Config], budget: object) -> dict:
        assert budget is self.provider._budget_exceeded_synced
        assert not budget()
        self.prepared.append(list(configs))
        result = {}
        for i, config in enumerate(configs):
            if config in self.bound._compile_cache:
                continue
            source = sb._config_key(config)
            receipt = sb.PreparedSource(
                self.bound,
                source,
                source,
                hashlib.sha256(source.encode()).hexdigest(),
                sb._snapshot(self.bound, (self.tensor,)),
                (self.tensor,),
                (self.tensor.untyped_storage(),),
                "",
            )
            if self.corrupt_receipts:
                receipt = dataclasses.replace(receipt, source="corrupted")
            result[i] = receipt
        if self.expire_during_prepare:
            self.expired = True
        return result


@pytest.fixture
def foreign_thread():
    finished = threading.Event()
    foreign = threading.Thread(target=finished.wait, name="foreign-retry-test")
    foreign.start()

    def quiesce() -> None:
        # Test-owned work finishes during the serial model. The provider never
        # signals or joins an unrelated thread, and unknown threads stay rejected.
        finished.set()
        foreign.join(2)
        assert not foreign.is_alive()

    try:
        yield foreign, quiesce
    finally:
        quiesce()


def _observe_admission(monkeypatch: pytest.MonkeyPatch, model: _CompileModel):
    original_create = sb.SourceBatch.create
    attempts = []
    monkeypatch.setattr(sb, "_guard_targets", list)

    def create(kernel: object, args: object) -> object:
        assert model.bound.settings.autotune_cute_source_workers > 1
        batch = original_create(kernel, args)
        attempts.append((len(model.calls), batch is not None))
        if batch is not None:
            batch.prepare = model.prepare
        return batch

    monkeypatch.setattr(sb.SourceBatch, "create", create)
    return attempts


@pytest.mark.parametrize("workers", [0, 1, 32])
@pytest.mark.parametrize("lifetime", ["transient", "persistent"])
def test_short_batch_retries_after_serial_config(
    workers: int, lifetime: str, monkeypatch: pytest.MonkeyPatch, foreign_thread
) -> None:
    foreign, quiesce = foreign_thread
    model = _CompileModel(workers, quiesce if lifetime == "transient" else lambda: None)
    attempts = _observe_admission(monkeypatch, model)
    configs = [Config(block_sizes=[n]) for n in (8, 4, 8, 2, 1, 16)]
    compiled, results = _compile_prefix()(model.provider, configs)
    assert list(compiled) == list(range(len(configs)))
    assert [fn() for fn in compiled.values()] == [8, 4, 8, 2, 1, 16]
    assert compiled[0] is compiled[2]
    assert model.calls == configs
    assert [row.config for row in results] == configs
    assert list(model.bound._compile_cache) == list(dict.fromkeys(configs))
    assert model.bound._cache_path_map == {
        config: "model://" + sb._config_key(config) for config in configs
    }
    if workers > 1 and lifetime == "transient":
        assert attempts == [(0, False), (1, True)]
        assert model.prepared == [configs[1:]]
        assert model.serial_misses == configs[:1]
        assert model.consumed == [configs[i] for i in (1, 3, 4, 5)]
    else:
        assert attempts == (
            [(i, False) for i in range(len(configs))] if workers > 1 else []
        )
        assert not model.prepared and not model.consumed
        assert model.serial_misses == list(dict.fromkeys(configs))
    assert foreign.is_alive() == (lifetime == "persistent")


@pytest.mark.parametrize("expiry", ["initial", "after_serial", "during_prepare"])
def test_retry_keeps_budget_tail(
    expiry: str, monkeypatch: pytest.MonkeyPatch, foreign_thread
) -> None:
    foreign, quiesce = foreign_thread

    def after_serial() -> None:
        quiesce()
        if expiry == "after_serial":
            model.expired = True

    model = _CompileModel(32, after_serial)
    model.expired = expiry == "initial"
    model.expire_during_prepare = expiry == "during_prepare"
    attempts = _observe_admission(monkeypatch, model)
    configs = [Config(block_sizes=[n]) for n in (8, 4, 2, 1, 16, 32)]
    compiled, results = _compile_prefix()(model.provider, configs)
    count = {"initial": 0, "after_serial": 1, "during_prepare": 2}[expiry]
    assert list(compiled) == list(range(count))
    assert model.calls == configs[:count]
    assert list(model.bound._compile_cache) == configs[:count]
    assert list(model.bound._cache_path_map) == configs[:count]
    assert [row.config for row in results] == configs
    assert all(row.perf == float("inf") and row.status == "error" for row in results)
    assert attempts == ([(0, False), (1, True)] if count == 2 else [(0, False)])
    assert model.prepared == ([configs[1:]] if count == 2 else [])
    assert model.consumed == (configs[1:2] if count == 2 else [])
    assert foreign.is_alive() == (expiry == "initial")


def test_retry_preserves_prepared_source_revalidation(
    monkeypatch: pytest.MonkeyPatch, foreign_thread
) -> None:
    foreign, quiesce = foreign_thread
    model = _CompileModel(32, quiesce)
    model.corrupt_receipts = True
    attempts = _observe_admission(monkeypatch, model)
    configs = [Config(block_sizes=[n]) for n in (8, 4, 2)]
    compiled, results = _compile_prefix()(model.provider, configs)
    assert attempts == [(0, False), (1, True)]
    assert model.prepared == [configs[1:]]
    assert not model.consumed
    assert model.serial_misses == configs
    assert [fn() for fn in compiled.values()] == [8, 4, 2]
    assert [row.config for row in results] == configs
    assert not foreign.is_alive()
