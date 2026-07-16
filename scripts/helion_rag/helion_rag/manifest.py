"""Load and validate hardware manifest JSON."""

from __future__ import annotations

import json
from pathlib import Path


class ManifestError(Exception):
    """Raised when manifest JSON does not match expected schema."""


def _fail(msg: str) -> None:
    raise ManifestError(msg)


def validate_manifest(obj: dict) -> None:
    """Ensure manifest has version int and families dict with required fields."""
    if not isinstance(obj, dict):
        _fail("manifest must be a JSON object")
    if not isinstance(obj.get("version"), int):
        _fail("manifest 'version' must be an integer")
    families = obj.get("families")
    if not isinstance(families, dict) or not families:
        _fail("manifest 'families' must be a non-empty object")

    for fam, spec in families.items():
        if not isinstance(spec, dict):
            _fail(f"family {fam!r} must be an object")
        if not isinstance(spec.get("artifact_path"), str) or not spec.get(
            "artifact_path"
        ):
            _fail(f"family {fam!r} missing required 'artifact_path'")
        aliases = spec.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            _fail(f"family {fam!r} missing required non-empty 'aliases'")
        cc = spec.get("compute_capabilities")
        if cc is not None and not isinstance(cc, list):
            _fail(f"family {fam!r} 'compute_capabilities' must be a list")


def load_manifest(path: str | Path) -> dict:
    """Read manifest JSON from disk and validate it."""
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    validate_manifest(obj)
    return obj
