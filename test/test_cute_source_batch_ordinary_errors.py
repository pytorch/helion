from __future__ import annotations

from contextlib import suppress
import json
import os
import sys
import time

import pytest

from test.test_cute_chained_source_batch import chained
from test.test_cute_source_batch_advanced_region import _advanced

from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux source workers")


def _batch(monkeypatch, failure):
    bound, tensor, default = chained()
    configs = [
        Config(**{**_advanced(default).config, "block_sizes": [size]})
        for size in (16, 32, 64)
    ]
    parent = os.getpid()
    original_source = bound.to_triton_code
    original_run = sb._run_workers
    reaps = []
    error = ValueError("original candidate32 codegen error")

    def source(config, **kwargs):
        if config.config["block_sizes"] == [32]:
            if os.getpid() == parent:
                raise error
            if failure == "guard":
                # A swallowed forbidden operation is still fatal.
                with suppress(RuntimeError):
                    BoundKernel.compile_config(bound, config)
            elif failure == "authority":
                bound.host_function.device_ir.phases[0].roots.append(99)
            elif failure == "timeout":
                time.sleep(5)
            raise ValueError("ordinary speculative source error")
        return original_source(config, **kwargs)

    def run(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        finally:
            reaps.extend(json.loads((kwargs["output"] / "reaps.json").read_text()))

    bound.to_triton_code = source
    monkeypatch.setattr(sb, "_run_workers", run)
    monkeypatch.setattr(sb, "_guard_targets", lambda: [(BoundKernel, "compile_config")])
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    if failure == "timeout":
        batch.timeout = 0.03
    return batch, bound, configs, reaps, error


def _fully_reaped(reaps):
    assert reaps
    for row in reaps:
        with pytest.raises(ChildProcessError):
            os.waitpid(row["pid"], os.WNOHANG)


def test_ordinary_source_error_retains_only_valid_siblings_after_full_drain(
    monkeypatch,
):
    batch, bound, configs, reaps, error = _batch(monkeypatch, "ordinary")
    prepared = batch.prepare(configs, lambda: False)
    assert set(prepared) == {0, 2}
    assert {row["index"] for row in reaps} == {0, 1, 2}
    assert all(row["exitcode"] == 0 for row in reaps)
    _fully_reaped(reaps)
    assert [prepared[i].consume(bound, configs[i]) for i in (0, 2)] == [
        bound.to_triton_code(configs[i]) for i in (0, 2)
    ]
    with pytest.raises(ValueError) as caught:
        bound._compile_config(configs[1], allow_print=False)
    assert caught.value is error
    assert not bound._compile_cache and not bound._cache_path_map


@pytest.mark.parametrize("failure", ("guard", "authority", "timeout"))
def test_fatal_worker_or_supervisor_failure_still_discards_all_siblings(
    monkeypatch, failure
):
    batch, bound, configs, reaps, error = _batch(monkeypatch, failure)
    assert batch.prepare(configs, lambda: False) == {}
    _fully_reaped(reaps)
    assert any(row["exitcode"] != 0 for row in reaps)
    with pytest.raises(ValueError) as caught:
        bound._compile_config(configs[1], allow_print=False)
    assert caught.value is error
    assert not bound._compile_cache and not bound._cache_path_map


def test_parent_change_after_clean_worker_errors_discards_valid_siblings(monkeypatch):
    batch, bound, configs, reaps, _ = _batch(monkeypatch, "ordinary")
    run = sb._run_workers

    def changed(*args, **kwargs):
        result = run(*args, **kwargs)
        bound.host_function.device_ir.phases[0].roots.append(99)
        return result

    monkeypatch.setattr(sb, "_run_workers", changed)
    assert batch.prepare(configs, lambda: False) == {}
    assert all(row["exitcode"] == 0 for row in reaps)
    _fully_reaped(reaps)
    assert not bound._compile_cache and not bound._cache_path_map
