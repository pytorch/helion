"""Artifact trust primitives: canonical hashing + Ed25519-signed manifests (§3.2).

A published generation carries a ``manifest.json`` listing every artifact's
SHA-256 and a detached ``manifest.sig`` — an Ed25519 signature over the manifest's
canonical bytes. Consumers verify the signature with a public key distributed
through trusted deployment configuration (never from the artifact bundle), then
verify every artifact hash, before any vector data is read. Any failure raises a
typed error the caller maps to a fail-closed fallback; the taxonomy mirrors the
distinct fallback events in §3.1 (signature / missing artifact / corruption /
version mismatch).

Uses the ``cryptography`` Ed25519 implementation. The private key stays offline
with the single designated publisher.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# cryptography is imported lazily inside the functions that need it, matching
# how index.py defers langchain: `pytest .` from the repo root collects this
# package's tests without installing it, so a module-level import here breaks
# collection for the whole repo.
if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
SIGNATURE_NAME = "manifest.sig"
_PUBLIC_KEY_ENV = "HELION_RAG_PUBLIC_KEY_PATH"
_PRIVATE_KEY_ENV = "HELION_RAG_PRIVATE_KEY_PATH"


class ArtifactVerificationError(Exception):
    """Base class for any artifact-trust failure (fails closed to baseline)."""


class SignatureError(ArtifactVerificationError):
    """The manifest signature did not verify against the trusted public key."""


class MissingArtifactError(ArtifactVerificationError):
    """A manifest-listed artifact (or the manifest/signature itself) is absent."""


class HashMismatchError(ArtifactVerificationError):
    """An artifact's content hash does not match the signed manifest."""


class VersionMismatchError(ArtifactVerificationError):
    """The manifest schema version is not understood by this consumer."""


class ArtifactFormatError(ArtifactVerificationError):
    """A trusted artifact cannot be decoded or violates its storage schema."""


# --------------------------------------------------------------------------- #
# Canonical bytes and hashing
# --------------------------------------------------------------------------- #
def canonical_json_bytes(obj: object) -> bytes:
    """Deterministic UTF-8 JSON bytes (sorted keys, no insignificant whitespace)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def generate_keypair() -> tuple[bytes, bytes]:
    """Return ``(private_pem, public_pem)`` for the offline publisher / tests."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def load_private_key(private_pem: bytes) -> Ed25519PrivateKey:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        key = serialization.load_pem_private_key(private_pem, password=None)
    except (TypeError, ValueError) as exc:
        raise ArtifactFormatError("malformed publisher private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ArtifactFormatError("publisher key is not an Ed25519 private key")
    return key


def load_public_key(public_pem: bytes) -> Ed25519PublicKey:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        key = serialization.load_pem_public_key(public_pem)
    except (TypeError, ValueError) as exc:
        raise ArtifactFormatError("malformed trusted public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ArtifactFormatError("trusted key is not an Ed25519 public key")
    return key


def load_trusted_public_key(env: str = _PUBLIC_KEY_ENV) -> Ed25519PublicKey:
    """Load the publisher public key from trusted deployment config, not the bundle."""
    path = os.environ.get(env)
    if not path:
        raise MissingArtifactError(
            f"trusted public key not configured; set {env} to its PEM path"
        )
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise MissingArtifactError(f"trusted public key is unreadable: {path}") from exc
    return load_public_key(payload)


def load_publisher_private_key(env: str = _PRIVATE_KEY_ENV) -> Ed25519PrivateKey:
    """Load the offline publisher signing key (only the designated publisher has it)."""
    path = os.environ.get(env)
    if not path:
        raise MissingArtifactError(
            f"publisher private key not configured; set {env} to its PEM path"
        )
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise MissingArtifactError(
            f"publisher private key is unreadable: {path}"
        ) from exc
    return load_private_key(payload)


# --------------------------------------------------------------------------- #
# Manifest build / sign / verify
# --------------------------------------------------------------------------- #
def build_manifest(
    generation_id: str, gen_dir: Path, artifact_names: Sequence[str]
) -> dict:
    """Build a manifest listing each artifact's SHA-256 (sorted, deterministic)."""
    return {
        "version": MANIFEST_VERSION,
        "generation_id": generation_id,
        "artifacts": {
            name: sha256_file(gen_dir / name) for name in sorted(artifact_names)
        },
    }


def sign_manifest(manifest: dict, private_key: Ed25519PrivateKey) -> bytes:
    return private_key.sign(canonical_json_bytes(manifest))


def verify_manifest_signature(
    manifest: dict, signature: bytes, public_key: Ed25519PublicKey
) -> None:
    """Raise :class:`SignatureError` if the signature does not match the manifest."""
    from cryptography.exceptions import InvalidSignature

    try:
        public_key.verify(signature, canonical_json_bytes(manifest))
    except (InvalidSignature, ValueError) as exc:
        raise SignatureError("manifest signature verification failed") from exc


def _validate_manifest(manifest: object) -> dict[str, object]:
    """Validate the signed generation manifest before artifact path access."""
    if not isinstance(manifest, dict):
        raise ArtifactFormatError("generation manifest must be a JSON object")
    expected_fields = {"version", "generation_id", "artifacts"}
    if set(manifest) != expected_fields:
        raise ArtifactFormatError(
            "generation manifest must contain only version, generation_id, and artifacts"
        )
    if manifest.get("version") != MANIFEST_VERSION:
        raise VersionMismatchError(
            f"unsupported manifest version {manifest.get('version')!r}"
        )
    generation_id = manifest.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ArtifactFormatError("generation manifest ID must be a non-empty string")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ArtifactFormatError("generation manifest artifacts must be an object")
    for name, digest in artifacts.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {MANIFEST_NAME, SIGNATURE_NAME}
        ):
            raise ArtifactFormatError(
                f"generation manifest has invalid artifact name {name!r}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ArtifactFormatError(
                f"generation manifest has invalid SHA-256 for artifact {name!r}"
            )
    return manifest


def verify_artifacts(manifest: dict, gen_dir: Path) -> None:
    """Verify the manifest version and every listed artifact's content hash."""
    trusted_manifest = _validate_manifest(manifest)
    artifacts = trusted_manifest["artifacts"]
    assert isinstance(artifacts, dict)
    for name, expected in artifacts.items():
        assert isinstance(name, str)
        assert isinstance(expected, str)
        path = gen_dir / name
        if not path.is_file():
            raise MissingArtifactError(f"artifact {name!r} missing from {gen_dir}")
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise MissingArtifactError(
                f"artifact {name!r} unreadable under {gen_dir}"
            ) from exc
        if actual != expected:
            raise HashMismatchError(f"artifact {name!r} hash mismatch")


def write_signed_manifest(
    generation_id: str,
    gen_dir: Path,
    artifact_names: Sequence[str],
    private_key: Ed25519PrivateKey,
) -> dict:
    """Build, sign, and write ``manifest.json`` + ``manifest.sig`` into ``gen_dir``."""
    manifest = build_manifest(generation_id, gen_dir, artifact_names)
    signature = sign_manifest(manifest, private_key)
    (gen_dir / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest))
    (gen_dir / SIGNATURE_NAME).write_bytes(signature)
    return manifest


def verify_generation(gen_dir: Path, public_key: Ed25519PublicKey) -> dict:
    """Fully verify a generation directory; return the trusted manifest.

    Verifies, in order: manifest + signature present, signature valid, schema
    version understood, and every listed artifact's hash. Raises a typed
    :class:`ArtifactVerificationError` subclass on the first failure so the caller
    fails closed to baseline tuning.
    """
    manifest_path = gen_dir / MANIFEST_NAME
    signature_path = gen_dir / SIGNATURE_NAME
    if not manifest_path.exists() or not signature_path.exists():
        raise MissingArtifactError(f"manifest or signature missing under {gen_dir}")
    try:
        payload = manifest_path.read_bytes()
        signature = signature_path.read_bytes()
    except OSError as exc:
        raise MissingArtifactError(
            f"manifest or signature unreadable under {gen_dir}"
        ) from exc
    try:
        raw_manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactFormatError("generation manifest is not valid JSON") from exc
    manifest = _validate_manifest(raw_manifest)
    verify_manifest_signature(manifest, signature, public_key)
    if payload != canonical_json_bytes(manifest):
        raise ArtifactFormatError("generation manifest JSON is not canonical")
    verify_artifacts(manifest, gen_dir)
    return manifest
