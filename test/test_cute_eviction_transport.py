from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_source_batch import _bound

from helion._compiler.cute import eviction_relevance as er
from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb
from helion.runtime.kernel import BoundKernel

if TYPE_CHECKING:
    from pathlib import Path


def fixture() -> tuple[BoundKernel, torch.Tensor, Config]:
    bound, tensor = _bound()
    bound._cute_eviction_policy_receipts = {}
    bound.host_function.device_ir.root_ids = [0]
    bound.settings.static_shapes = True
    bound.env.backend.annotate_compiled_module = lambda module, source, name: setattr(
        getattr(module, name),
        "_helion_cute_source_hash",
        hashlib.sha256(source.encode()).hexdigest(),
    )
    bound.env.backend.generated_source_hash = lambda compiled: (
        compiled._helion_cute_source_hash
    )
    config = Config(block_sizes=[8], load_eviction_policies=["", ""])
    return bound, tensor, config


def receipt(bound: BoundKernel, config: Config, source: str) -> er.PolicyReceipt:
    uses = er.PolicyUses(config)
    owner = object()
    uses.record(owner, 0, ("",) * 7)
    uses.record(owner, 1, ("", "first", "last", "streaming", "", "", ""))
    value = uses.finish(er.authority(bound), source)
    assert value is not None
    return value


def prepared(
    bound: BoundKernel, tensor: torch.Tensor, config: Config, source: str | None = None
) -> sb.PreparedSource:
    source = bound.to_triton_code(config) if source is None else source
    return sb.PreparedSource(
        bound,
        sb._config_key(config),
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        sb._snapshot(bound, (tensor,)),
        (tensor,),
        (tensor.untyped_storage(),),
        "",
        policy_receipt=receipt(bound, config, source),
    )


def payload(bound: BoundKernel, config: Config, source: str) -> dict[str, object]:
    value = receipt(bound, config, source)
    bound._cute_eviction_policy_receipts[value.config_key] = value
    encoded = er.export_receipt(bound, config, source)
    return json.loads(json.dumps(encoded))


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("authority", "stale"),
        ("source", "wrong"),
        ("config", "{}"),
        ("classes", []),
        ("classes", [["0"] * 6, None]),
        ("classes", [["7"] * 7, None]),
        ("classes", [["1"] * 7, None]),
        ("classes", [["0"] * 7, [None] * 7]),
    ],
)
def test_bad_metadata_does_not_change_source_admission(
    field: str, value: object
) -> None:
    bound, tensor, config = fixture()
    item = prepared(bound, tensor, config)
    encoded = payload(bound, config, item.source)
    encoded[field] = value
    assert er.import_receipt(bound, config, item.source, encoded) is None
    assert (
        dataclasses.replace(item, policy_receipt=None).consume(bound, config)
        == item.source
    )


def test_round_trip_and_version_reset() -> None:
    bound, tensor, config = fixture()
    source = bound.to_triton_code(config)
    encoded = payload(bound, config, source)
    before = er.import_receipt(bound, config, source, encoded)
    assert before is not None
    tensor.add_(1)
    assert er.import_receipt(bound, config, source, encoded) == before
    bound.settings.fast_math = not bound.settings.fast_math
    assert er.import_receipt(bound, config, source, encoded) is None


def test_policy_free_receipt_round_trip_keeps_candidates() -> None:
    bound, tensor, _ = fixture()
    config = Config(block_sizes=[8])
    source = "original source"
    value = er.PolicyUses(config).finish(er.authority(bound), source)
    assert value is not None and value.classes == ()
    bound._cute_eviction_policy_receipts[value.config_key] = value
    encoded = json.loads(json.dumps(er.export_receipt(bound, config, source)))
    restored = er.import_receipt(bound, config, source, encoded)
    assert restored == value
    assert not restored.excludes(Config(block_sizes=[16]))
    assert not restored.excludes(Config(block_sizes=[8], load_eviction_policies=[""]))


def test_inference_owner_declines() -> None:
    bound, tensor, config = fixture()
    source = bound.to_triton_code(config)
    encoded = payload(bound, config, source)
    with torch.inference_mode():
        inference = torch.zeros(8)
    bound._runtime_tensor_refs_by_name = {"x": lambda: inference}
    assert er.import_receipt(bound, config, source, encoded) is None


def test_publication_only_after_real_materialization_and_disk_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    bound, tensor, config = fixture()
    item = prepared(bound, tensor, config)
    assert item.consume(bound, config) == item.source
    assert not bound._cute_eviction_policy_receipts
    fn = bound._compile_config(config, allow_print=False, prepared_source=item)
    assert er.receipt_for(bound, config) == item.policy_receipt
    bound._cute_eviction_policy_receipts.clear()
    assert bound._compile_config(config, allow_print=False, prepared_source=item) is fn
    assert not bound._cute_eviction_policy_receipts  # early compile cache hit
    bound._compile_cache.clear()
    assert bound._compile_config(config, allow_print=False, prepared_source=item) is fn
    assert (
        er.receipt_for(bound, config) == item.policy_receipt
    )  # same PyCodeCache object
    bound._compile_cache.clear()
    bound._cute_eviction_policy_receipts.clear()
    assert bound._compile_config(config, allow_print=False) is fn
    assert not bound._cute_eviction_policy_receipts  # disk-only input has no evidence


@pytest.mark.parametrize(
    "failure", ["import", "annotation", "host_lookup", "source_hash"]
)
def test_failure_or_unmatched_module_does_not_publish(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    bound, tensor, config = fixture()
    source = bound.to_triton_code(config)
    if failure == "import":
        source = "raise ValueError('import failed')\n" + source
    if failure == "host_lookup":
        source = "other_name = 1\n"
        bound.env.backend.annotate_compiled_module = lambda *args: None
    if failure == "annotation":

        def fail(*args: object) -> None:
            raise ValueError("annotation failed")

        bound.env.backend.annotate_compiled_module = fail
    if failure == "source_hash":
        bound.env.backend.generated_source_hash = lambda compiled: "wrong"
    item = prepared(bound, tensor, config, source)
    if failure == "source_hash":
        bound._compile_config(config, allow_print=False, prepared_source=item)
    else:
        with pytest.raises((ValueError, AttributeError)):
            bound._compile_config(config, allow_print=False, prepared_source=item)
        assert not bound._compile_cache
    assert not bound._cute_eviction_policy_receipts


def test_stale_prepared_falls_back_without_adoption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    bound, tensor, config = fixture()
    item = prepared(bound, tensor, config)
    tensor.add_(1)
    bound._compile_config(config, allow_print=False, prepared_source=item)
    assert not bound._cute_eviction_policy_receipts


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
@pytest.mark.parametrize("fresh", [False, True])
def test_fork_transports_only_fresh_receipts_without_early_publication(
    fresh: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    bound, tensor, config = fixture()
    original_source = bound.to_triton_code
    # Child must clear this inherited receipt even when generation produces none.
    inherited = receipt(bound, config, original_source(config))
    bound._cute_eviction_policy_receipts[inherited.config_key] = inherited

    def generate(config: Config, **kwargs: object) -> str:
        source = original_source(config)
        if fresh:
            value = receipt(bound, config, source)
            bound._cute_eviction_policy_receipts[value.config_key] = value
        return source

    bound.to_triton_code = generate
    monkeypatch.setattr(
        sb, "_guard_targets", lambda: [(BoundKernel, "_compile_config")]
    )
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    pending = batch.prepare([config], lambda: False)
    assert set(pending) == {0}
    assert (pending[0].policy_receipt is not None) == fresh
    assert bound._cute_eviction_policy_receipts == {inherited.config_key: inherited}
    bound._cute_eviction_policy_receipts.clear()
    assert not bound._compile_cache
    bound._compile_config(config, allow_print=False, prepared_source=pending[0])
    assert bool(bound._cute_eviction_policy_receipts) == fresh
