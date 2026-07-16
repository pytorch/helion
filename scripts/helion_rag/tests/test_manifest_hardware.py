"""Manifest validation and hardware family resolution."""

from __future__ import annotations

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


_MANIFEST = {
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


def test_family_from_device_tokens() -> None:
    """Device-string token matching is pure and host-independent."""
    assert hardware._family_from_device("NVIDIA H100 80GB") == "h100"
    assert hardware._family_from_device("gfx950") == "mi350x"
    assert hardware._family_from_device("mystery accelerator") is None
    assert hardware._family_from_device(None) is None


def test_family_from_manifest_alias_and_compute_capability() -> None:
    assert (
        hardware._family_from_manifest_alias("hopper accelerator", _MANIFEST) == "h100"
    )
    assert hardware._family_from_manifest_alias("mystery", _MANIFEST) is None
    assert hardware._family_from_compute_capability("10.0", _MANIFEST) == "b200"
    assert hardware._family_from_compute_capability("7.5", _MANIFEST) is None


def test_resolve_family_precedence_before_device_fallback() -> None:
    """Precedence resolvable without the torch device probe (override > env > token).

    The torch fallback and the manifest alias/CC branches sit *after* the device
    token, so they're covered by the pure-helper tests above rather than driven
    through resolve_family (which would depend on the host GPU)."""
    assert (
        hardware.resolve_family(device="NVIDIA H100 80GB", manifest=_MANIFEST) == "h100"
    )
    assert (
        hardware.resolve_family(device="unknown", manifest=_MANIFEST, env_family="tpu")
        == "tpu"
    )
    assert (
        hardware.resolve_family(
            device="NVIDIA H100 80GB",
            manifest=_MANIFEST,
            env_family="b200",
            override="mi350x",
        )
        == "mi350x"
    )


def test_register_family_adds_missing_family() -> None:
    manifest = {
        "version": 1,
        "families": {"h100": {"artifact_path": "h100", "aliases": ["h100"]}},
    }

    assert setup_helpers._register_family(
        manifest,
        "b200",
        artifact_path="blackwell/b200",
        aliases=["blackwell"],
    )
    assert manifest["families"]["b200"] == {
        "artifact_path": "blackwell/b200",
        "aliases": ["b200", "blackwell"],
    }


def test_register_family_is_noop_when_present() -> None:
    manifest = {
        "version": 1,
        "families": {"h100": {"artifact_path": "h100", "aliases": ["h100"]}},
    }

    assert not setup_helpers._register_family(manifest, "h100")
    assert sorted(manifest["families"]) == ["h100"]
