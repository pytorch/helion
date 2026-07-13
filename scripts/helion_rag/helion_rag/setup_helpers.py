"""Helpers for setup-helion-rag.sh to replace Python heredocs."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import sys
import tempfile

from helion_rag.hardware import resolve_family
from helion_rag.manifest import load_manifest
from helion_rag.manifest import validate_manifest


def check_helion_import(repo_path: str | Path) -> int:
    """Exit code 0 if repo_path's helion importable and matches, 3 if mismatch."""
    repo = Path(repo_path).resolve()
    sys.path.insert(0, str(repo))
    import helion

    helion_file = Path(helion.__file__).resolve()
    return 0 if str(helion_file).startswith(str(repo)) else 3


def synthesize_manifest(out_path: str | Path, families: list[str]) -> None:
    obj = {
        "version": 1,
        "families": {f: {"artifact_path": f, "aliases": [f]} for f in families if f},
    }
    validate_manifest(obj)
    Path(out_path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def validate_manifest_cli(path: str | Path) -> int:
    load_manifest(path)
    return 0


def resolve_family_cli(manifest_path: str | Path, device_env: str | None = None) -> str:
    m = load_manifest(manifest_path)
    dev = device_env or os.environ.get("HELION_RAG_FAKE_DEVICE")
    fam = resolve_family(device=dev, manifest=m)
    return fam or ""


def is_represented(manifest_path: str | Path, family: str) -> bool:
    m = load_manifest(manifest_path)
    return family in (m.get("families") or {})


def artifact_path(manifest_path: str | Path, family: str) -> str:
    m = load_manifest(manifest_path)
    return m["families"][family]["artifact_path"]


def publish_manifest(
    manifold_base: str,
    family: str,
    *,
    manifold_get: Callable[[str, str], bool],
    manifold_put: Callable[[str, str], None],
    artifact_path: str | None = None,
    aliases: list[str] | None = None,
) -> dict:
    """Register a new hardware family in the shared manifest.json.
    Downloads the existing manifest, adds the new family if it's missing, and
    uploads it back. Does nothing if the family is already registered.
    """
    dest = f"{manifold_base}/manifest.json"
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "manifest.json"
        if manifold_get(dest, str(local)):
            obj = json.loads(local.read_text(encoding="utf-8"))
        else:
            obj = {"version": 1, "families": {}}
        families = obj.setdefault("families", {})
        if family in families:
            return {"published": False, "reason": "already-present", "family": family}
        families[family] = {
            "artifact_path": artifact_path or family,
            "aliases": [family, *(aliases or [])],
        }
        validate_manifest(obj)
        local.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        manifold_put(str(local), dest)
    return {"published": True, "family": family, "families": sorted(obj["families"])}
