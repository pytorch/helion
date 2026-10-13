from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

import pytest

from test.test_cute_chained_source_batch import chained

from helion._compiler.device_ir import KernelPhase
from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux source workers")


def _advanced(default):
    return Config(
        **{
            **default.config,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_pointwise_vectorize": True,
            "cute_chained_scratch_layout": "xor",
            "cute_chained_group_contractions": True,
            "pid_type": "persistent",
        }
    )


def test_normalized_advanced_configs_do_not_request_or_compare_a_default(monkeypatch):
    bound, tensor, default = chained()
    config = _advanced(default)
    bound.config_spec.default_config = lambda: pytest.fail(
        "unrelated default normalization"
    )
    monkeypatch.setattr(sb, "_guard_targets", list)
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    prepared = batch.prepare([config, default], lambda: False)
    assert set(prepared) == {0, 1}
    for i, current in enumerate((config, default)):
        assert prepared[i].consume(bound, current) == bound.to_triton_code(current)
        assert prepared[i].config_key == sb._config_key(current)
    assert not bound._compile_cache and not bound._cache_path_map


@pytest.mark.parametrize("change", ("roots", "phases", "rng", "wrapper", "fission"))
def test_region_eligibility_is_rechecked_before_dispatch(monkeypatch, change):
    bound, tensor, default = chained()
    monkeypatch.setattr(sb, "_guard_targets", list)
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    if change == "roots":
        bound.host_function.device_ir.root_ids.append(1)
    elif change == "phases":
        bound.host_function.device_ir.phases.append(
            KernelPhase(roots=[1], root_nodes=[])
        )
    elif change == "rng":
        bound.host_function.rng_seed_slot_count = 1
    elif change == "wrapper":
        bound.env.cute_resolved_wrapper_plans.append({"new": "wrapper"})
    else:
        bound.env.cute_fission_plan = {"new": "fission"}
    monkeypatch.setattr(
        sb, "_run_workers", lambda *a, **kw: pytest.fail("ineligible fork")
    )
    assert batch.prepare([_advanced(default)], lambda: False) == {}


def test_invalid_normalization_preserves_original_serial_error(monkeypatch):
    bound, tensor, default = chained()
    config = _advanced(default)
    monkeypatch.setattr(sb, "_guard_targets", list)
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    error = ValueError("original unsupported advanced configuration")

    def reject(config):
        raise error

    bound.config_spec.normalized_config = reject
    monkeypatch.setattr(
        sb, "_run_workers", lambda *a, **kw: pytest.fail("invalid config fork")
    )
    assert batch.prepare([config], lambda: False) == {}
    with pytest.raises(ValueError) as caught:
        bound._normalized_config_copy(config)
    assert caught.value is error


@pytest.mark.parametrize("change", ("phase", "owner", "settings", "target", "host"))
def test_advanced_receipt_still_rejects_authority_change(change):
    bound, tensor, default = chained()
    config = _advanced(default)
    source = bound.to_triton_code(config)
    prepared = sb.PreparedSource(
        bound,
        sb._config_key(config),
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        sb._snapshot(bound, (tensor,)),
        (tensor,),
        (tensor.untyped_storage(),),
        "",
    )
    assert prepared.consume(bound, config) == source
    if change == "phase":
        bound.host_function.device_ir.phases[0].root_nodes[0].iter = ast.parse(
            "range(7)", mode="eval"
        ).body
    elif change == "owner":
        tensor.add_(1)
    elif change == "settings":
        bound.settings.static_shapes = False
    elif change == "target":
        bound.config_spec.num_sm += 1
    else:
        bound.host_function.body.append(ast.parse("changed = 1").body[0])
    assert prepared.consume(bound, config) is None


@pytest.mark.parametrize("failure", ("guard", "authority"))
def test_advanced_worker_failure_drains_then_keeps_original_serial_error(
    monkeypatch, tmp_path, failure
):
    bound, tensor, default = chained()
    config = _advanced(default)
    parent = os.getpid()
    original_run = sb._run_workers
    recorded = []

    def run(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        finally:
            recorded.extend(json.loads((kwargs["output"] / "reaps.json").read_text()))

    monkeypatch.setattr(sb, "_run_workers", run)
    monkeypatch.setattr(sb, "_guard_targets", lambda: [(BoundKernel, "compile_config")])
    error = ValueError("original serial source error")

    def source(*args, **kwargs):
        if os.getpid() == parent:
            raise error
        if failure == "guard":
            BoundKernel.compile_config(bound, config)
        else:
            bound.host_function.device_ir.phases[0].roots.append(9)
        return "def source_batch_fixture():\n    pass\n"

    bound.to_triton_code = source
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    assert batch.prepare([config], lambda: False) == {}
    assert len(recorded) == 1 and recorded[0]["exitcode"] != 0
    with pytest.raises(ChildProcessError):
        os.waitpid(recorded[0]["pid"], os.WNOHANG)
    with pytest.raises(ValueError) as caught:
        bound._compile_config(config, allow_print=False)
    assert caught.value is error
    assert not bound._compile_cache and not bound._cache_path_map
