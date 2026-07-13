"""Manifest validation and hardware family resolution."""

from __future__ import annotations

import json
from pathlib import Path

import helion_rag.hardware as hardware
import helion_rag.manifest as manifest_mod
import helion_rag.setup_helpers as setup_helpers
import pytest


def test_manifest_validation() -> None:
    obj = {
        "version": 1,
        "families": {
            "h100": {"artifact_path": "h100", "aliases": ["h100"]},
            "b200": {"artifact_path": "b200", "aliases": ["b200"]},
            "mi350x": {"artifact_path": "mi350x", "aliases": ["mi350x"]},
            "tpu": {"artifact_path": "tpu", "aliases": ["tpu"]},
        },
    }
    manifest_mod.validate_manifest(obj)

    assert obj["version"] == 1
    assert sorted(obj["families"]) == ["b200", "h100", "mi350x", "tpu"]


@pytest.mark.parametrize(
    "bad_obj,match",
    [
        ({"families": {"h100": {}}}, "version"),
        ({"version": 1, "families": {"h100": {}}}, "artifact_path"),
        ({"version": 1, "families": {"h100": {"artifact_path": "h100"}}}, "aliases"),
    ],
)
def test_manifest_validation_errors(bad_obj: dict, match: str) -> None:
    with pytest.raises(manifest_mod.ManifestError, match=match):
        manifest_mod.validate_manifest(bad_obj)


def test_resolve_family_tokens_aliases_override_and_unknown() -> None:
    manifest = {
        "version": 1,
        "families": {
            "h100": {
                "artifact_path": "h100",
                "aliases": ["NVIDIA H100 PCIe", "hopper"],
                "compute_capabilities": ["9.0"],
            },
            "b200": {
                "artifact_path": "b200",
                "aliases": ["blackwell"],
                "compute_capabilities": ["10.0"],
            },
        },
    }

    def no_gpu() -> None:
        return None

    assert (
        hardware.resolve_family(
            device="NVIDIA H100 80GB", manifest=manifest, _device_fn=no_gpu
        )
        == "h100"
    )
    assert (
        hardware.resolve_family(
            device="hopper accelerator", manifest=manifest, _device_fn=no_gpu
        )
        == "h100"
    )
    assert (
        hardware.resolve_family(
            device="unknown",
            manifest=manifest,
            compute_capability="10.0",
            _device_fn=no_gpu,
        )
        == "b200"
    )
    assert (
        hardware.resolve_family(
            device="unknown", manifest=manifest, env_family="tpu", _device_fn=no_gpu
        )
        == "tpu"
    )
    assert (
        hardware.resolve_family(
            device="NVIDIA H100 80GB",
            manifest=manifest,
            env_family="b200",
            override="mi350x",
            _device_fn=no_gpu,
        )
        == "mi350x"
    )
    assert (
        hardware.resolve_family(device="unknown", manifest=manifest, _device_fn=no_gpu)
        is None
    )


class _FakeBucket:
    """In-memory stand-in for the Manifold transport used by publish_manifest."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})
        self.puts: list[str] = []

    def get(self, src: str, dest: str) -> bool:
        if src not in self.files:
            return False
        Path(dest).write_text(self.files[src], encoding="utf-8")
        return True

    def put(self, src: str, dest: str) -> None:
        self.files[dest] = Path(src).read_text(encoding="utf-8")
        self.puts.append(dest)


def test_publish_manifest_creates_when_bucket_has_none() -> None:
    bucket = _FakeBucket()
    base = "helion_ci_artifacts/tree/run"

    summary = setup_helpers.publish_manifest(
        base, "h100", manifold_get=bucket.get, manifold_put=bucket.put
    )

    assert summary == {"published": True, "family": "h100", "families": ["h100"]}
    assert bucket.puts == [f"{base}/manifest.json"]
    published = json.loads(bucket.files[f"{base}/manifest.json"])
    manifest_mod.validate_manifest(published)
    assert published["families"]["h100"]["artifact_path"] == "h100"
    assert published["families"]["h100"]["aliases"] == ["h100"]


def test_publish_manifest_merges_missing_family_and_is_noop_when_present() -> None:
    base = "helion_ci_artifacts/tree/run"
    existing = {
        "version": 1,
        "families": {"h100": {"artifact_path": "h100", "aliases": ["h1"]}},
    }
    bucket = _FakeBucket({f"{base}/manifest.json": json.dumps(existing)})

    merged = setup_helpers.publish_manifest(
        base,
        "b200",
        manifold_get=bucket.get,
        manifold_put=bucket.put,
        artifact_path="blackwell/b200",
        aliases=["blackwell"],
    )
    assert merged["published"] is True
    published = json.loads(bucket.files[f"{base}/manifest.json"])
    assert sorted(published["families"]) == ["b200", "h100"]
    assert published["families"]["b200"] == {
        "artifact_path": "blackwell/b200",
        "aliases": ["b200", "blackwell"],
    }

    bucket.puts.clear()
    noop = setup_helpers.publish_manifest(
        base, "h100", manifold_get=bucket.get, manifold_put=bucket.put
    )
    assert noop == {"published": False, "reason": "already-present", "family": "h100"}
    assert bucket.puts == []  # nothing written back on no-op
