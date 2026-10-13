from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
import weakref

import pytest

from test.test_cute_source_batch import _bound
from test.test_cute_source_batch import _compile_prefix

from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="fork source workers")


@pytest.mark.parametrize("lifetime", ["transient", "persistent", "none"])
@pytest.mark.parametrize("workers", [2, 4])
def test_readmission_after_serial_work(
    lifetime: str, workers: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Actual provider prefix, admission, fork/drain and module/cache tail."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sb, "_guard_targets", list)
    bound, tensor = _bound()
    bound.settings.autotune_cute_source_workers = workers
    owner = weakref.ref(tensor)
    original_bytes = tensor.clone()
    started = threading.Event()
    finished = threading.Event()

    def foreign_work() -> None:
        started.set()
        finished.wait()

    foreign = threading.Thread(target=foreign_work, name="foreign-test-worker")
    if lifetime != "none":
        foreign.start()
        assert started.wait(2)

    parent_pid = os.getpid()
    serial = []
    admissions = []
    reaps = []
    forks = []
    original_create = sb.SourceBatch.create
    original_workers = sb._run_workers
    original_fork = os.fork
    original_source = bound.to_triton_code

    def create(kernel: object, args: object) -> object:
        batch = original_create(kernel, args)
        admissions.append((len(serial), batch is not None))
        return batch

    def fork() -> int:
        # Assert at the real process boundary, not only at early admission.
        assert threading.enumerate() == [threading.main_thread()]
        assert owner() is tensor
        forks.append(len(serial))
        return original_fork()

    def run_workers(*args: object, **kwargs: object) -> object:
        result = original_workers(*args, **kwargs)
        reaps.extend(result)
        assert all(row["exitcode"] == 0 for row in result)
        for row in result:
            with pytest.raises(ChildProcessError):
                os.waitpid(row["pid"], os.WNOHANG)
        return result

    def source(config: Config, **kwargs: object) -> str:
        if os.getpid() == parent_pid:
            serial.append(config)
            if lifetime == "transient" and len(serial) == 1:
                # The unrelated operation completes during serial work. The
                # production path neither signals nor joins foreign threads.
                finished.set()
                foreign.join(2)
                assert not foreign.is_alive()
        return original_source(config, **kwargs)

    monkeypatch.setattr(sb.SourceBatch, "create", create)
    monkeypatch.setattr(sb, "_run_workers", run_workers)
    monkeypatch.setattr(os, "fork", fork)
    bound.to_triton_code = source
    provider = SimpleNamespace(
        kernel=bound,
        args=[tensor],
        settings=bound.settings,
        config_spec=bound.config_spec,
        _validate_timing_policy=lambda: None,
        _effective_source_dedup_enabled=lambda: False,
        _effective_source_hashes=set(),
        _budget_exceeded_synced=lambda: False,
        _autotune_metrics=SimpleNamespace(num_successful_candidate_measurements=1),
    )
    configs = [Config(block_sizes=[n]) for n in [8, 4, 2, 1, 16, 32, 2, 64]]
    try:
        compiled, results = _compile_prefix()(provider, configs)
        assert list(compiled) == list(range(len(configs)))
        assert [fn() for fn in compiled.values()] == [8, 4, 2, 1, 16, 32, 2, 64]
        assert compiled[2] is compiled[6]
        assert [row.config for row in results] == configs
        assert owner() is tensor
        assert tensor.equal(original_bytes)
        if lifetime == "persistent":
            assert len(serial) == 7  # One original cache hit.
            assert not forks and not reaps
            assert len(admissions) == len(configs)
            assert not any(active for _, active in admissions)
            assert foreign.is_alive()
        else:
            expected_serial = 1 if lifetime == "transient" else 0
            assert len(serial) == expected_serial
            assert admissions == (
                [(0, False), (1, True)] if lifetime == "transient" else [(0, True)]
            )
            assert len(reaps) == 7 - expected_serial
            assert len(forks) == len(reaps)
    finally:
        finished.set()
        if lifetime != "none":
            foreign.join(2)
            assert not foreign.is_alive()


def test_budget_stops_before_readmission(monkeypatch: pytest.MonkeyPatch) -> None:
    bound, tensor = _bound()
    admissions = []

    def create(*args: object) -> None:
        admissions.append(args)
        return None

    monkeypatch.setattr(sb.SourceBatch, "create", create)
    provider = SimpleNamespace(
        kernel=bound,
        args=[tensor],
        settings=bound.settings,
        config_spec=bound.config_spec,
        _validate_timing_policy=lambda: None,
        _effective_source_dedup_enabled=lambda: False,
        _effective_source_hashes=set(),
        _budget_exceeded_synced=lambda: True,
        _autotune_metrics=SimpleNamespace(num_successful_candidate_measurements=1),
    )
    configs = [Config(block_sizes=[n]) for n in (8, 4, 2, 1)]
    compiled, results = _compile_prefix()(provider, configs)
    assert compiled == {}
    assert len(admissions) == 1
    assert [row.config for row in results] == configs
    assert all(row.perf == float("inf") for row in results)
