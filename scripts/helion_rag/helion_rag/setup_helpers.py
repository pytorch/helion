"""Helpers for setup-helion-rag.sh to replace Python heredocs."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import sys
import tempfile
from typing import cast

from helion_rag.hardware import resolve_family
from helion_rag.manifest import load_manifest
from helion_rag.manifest import validate_manifest
from helion_rag.signing import generate_keypair


def ensure_signing_keypair(private_path: str | Path, public_path: str | Path) -> bool:
    """Create one deployment keypair, or preserve the complete existing pair."""
    private = Path(private_path)
    public = Path(public_path)
    if private.exists() != public.exists():
        raise RuntimeError(
            f"partial signing keypair: expected both {private} and {public}"
        )
    if private.exists():
        private.chmod(0o600)
        return False
    private.parent.mkdir(parents=True, exist_ok=True)
    public.parent.mkdir(parents=True, exist_ok=True)
    private_pem, public_pem = generate_keypair()
    private.write_bytes(private_pem)
    private.chmod(0o600)
    public.write_bytes(public_pem)
    return True


def _shell_quote(value: str | Path) -> str:
    text = str(value).replace("'", "'\\''")
    return f"'{text}'"


def write_environment(
    path: str | Path,
    *,
    rag_root: str | Path,
    hardware_family: str,
    manifold_base: str,
    embed_model: str,
    private_key_path: str | Path,
    public_key_path: str | Path,
    llm_provider: str,
    llm_model: str,
    llm_ca_bundle: str,
    vertex_base_url: str | None = None,
    vertex_project_id: str | None = None,
    cloud_ml_region: str | None = None,
    hf_home: str | Path | None = None,
    collect_autotune: bool = False,
) -> None:
    """Write the managed RAG/Vertex environment without persisting credentials."""
    root = Path(rag_root)
    values: list[tuple[str, str | Path | None]] = [
        ("HELION_RAG_HARDWARE_FAMILY", hardware_family),
        ("HELION_RAG_MANIFOLD_BASE", manifold_base),
        ("HELION_RAG_MANIFEST", root / "manifest.json"),
        ("HELION_RAG_DATA_DIR", root / "ci_artifacts"),
        ("HELION_RAG_INDEX_DIR", root / "rag_index"),
        ("HELION_RAG_WRITEBACK_DIR", root / "rag_writeback"),
        ("HELION_RAG_AUTOTUNE_LOG_DIR", root / "autotune_logs"),
        ("HELION_RAG_UPLOADS_DIR", root / "uploads"),
        ("HELION_RAG_PRIVATE_KEY_PATH", private_key_path),
        ("HELION_RAG_PUBLIC_KEY_PATH", public_key_path),
        ("HELION_EMBED_MODEL", embed_model),
        ("HELION_LLM_PROVIDER", llm_provider),
        ("HELION_LLM_MODEL", llm_model),
        ("HELION_LLM_CA_BUNDLE", llm_ca_bundle),
        ("ANTHROPIC_VERTEX_BASE_URL", vertex_base_url),
        ("ANTHROPIC_VERTEX_PROJECT_ID", vertex_project_id),
        ("CLOUD_ML_REGION", cloud_ml_region),
        ("HF_HOME", hf_home),
    ]
    if collect_autotune:
        values.extend(
            [
                ("HELION_AUTOTUNE_LOG", root / "autotune_logs" / "helion-rag"),
                ("HELION_AUTOTUNE_LOG_DETAILS", "1"),
            ]
        )
    lines = ["# Managed by setup-helion-rag.sh"]
    lines.extend(
        f"export {name}={_shell_quote(value)}"
        for name, value in values
        if value is not None and str(value) != ""
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def resolve_family_cli(manifest_path: str | Path, device: str | None = None) -> str:
    m = load_manifest(manifest_path)
    fam = resolve_family(device=device, manifest=m)
    return fam or ""


def is_represented(manifest_path: str | Path, family: str) -> bool:
    m = load_manifest(manifest_path)
    return family in (m.get("families") or {})


def artifact_path(manifest_path: str | Path, family: str) -> str:
    m = load_manifest(manifest_path)
    return m["families"][family]["artifact_path"]


def _register_family(
    manifest: dict,
    family: str,
    *,
    artifact_path: str | None = None,
    aliases: list[str] | None = None,
) -> bool:
    """Add one family to a manifest; return False when already present."""
    families = manifest.setdefault("families", {})
    if family in families:
        return False
    families[family] = {
        "artifact_path": artifact_path or family,
        "aliases": [family, *(aliases or [])],
    }
    validate_manifest(manifest)
    return True


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
        if not _register_family(
            obj, family, artifact_path=artifact_path, aliases=aliases
        ):
            return {"published": False, "reason": "already-present", "family": family}
        local.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        manifold_put(str(local), dest)
    families = cast("dict[str, object]", obj["families"])
    return {"published": True, "family": family, "families": sorted(families)}
