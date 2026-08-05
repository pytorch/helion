from __future__ import annotations

import json
from pathlib import Path

import pytest

from helion_rag import signing
from helion_rag import setup_helpers
from helion_rag.signing import HashMismatchError
from helion_rag.signing import MissingArtifactError
from helion_rag.signing import SignatureError
from helion_rag.signing import VersionMismatchError


def _write(gen_dir, name, content=b"data"):
    (gen_dir / name).write_bytes(content)


# --- canonical bytes + hashing --------------------------------------------
def test_canonical_json_is_key_order_independent():
    a = signing.canonical_json_bytes({"b": 1, "a": 2})
    b = signing.canonical_json_bytes({"a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1}'


def test_sha256_file_matches_hex(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"hello")
    assert signing.sha256_file(p) == signing.sha256_hex(b"hello")


# --- signing roundtrip -----------------------------------------------------
def test_sign_and_verify_roundtrip():
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    manifest = {"version": 1, "generation_id": "000001", "artifacts": {}}
    sig = signing.sign_manifest(manifest, priv)
    signing.verify_manifest_signature(manifest, sig, pub)  # no raise


def test_verify_fails_on_tampered_manifest():
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    manifest = {"version": 1, "generation_id": "000001", "artifacts": {}}
    sig = signing.sign_manifest(manifest, priv)
    tampered = {**manifest, "generation_id": "000002"}
    with pytest.raises(SignatureError):
        signing.verify_manifest_signature(tampered, sig, pub)


def test_verify_fails_with_wrong_key():
    priv_pem, _ = signing.generate_keypair()
    _, other_pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    other_pub = signing.load_public_key(other_pub_pem)
    manifest = {"version": 1, "generation_id": "1", "artifacts": {}}
    sig = signing.sign_manifest(manifest, priv)
    with pytest.raises(SignatureError):
        signing.verify_manifest_signature(manifest, sig, other_pub)


# --- artifact hash verification -------------------------------------------
def test_verify_artifacts_ok_and_tamper(tmp_path):
    _write(tmp_path, "index.faiss", b"vectors")
    _write(tmp_path, "metadata.json", b"{}")
    manifest = signing.build_manifest("1", tmp_path, ["index.faiss", "metadata.json"])
    signing.verify_artifacts(manifest, tmp_path)  # ok

    _write(tmp_path, "index.faiss", b"tampered")
    with pytest.raises(HashMismatchError):
        signing.verify_artifacts(manifest, tmp_path)


def test_verify_artifacts_missing(tmp_path):
    _write(tmp_path, "index.faiss", b"vectors")
    manifest = signing.build_manifest("1", tmp_path, ["index.faiss"])
    (tmp_path / "index.faiss").unlink()
    with pytest.raises(MissingArtifactError):
        signing.verify_artifacts(manifest, tmp_path)


def test_verify_artifacts_bad_version(tmp_path):
    _write(tmp_path, "a", b"x")
    manifest = signing.build_manifest("1", tmp_path, ["a"])
    manifest["version"] = 999
    with pytest.raises(VersionMismatchError):
        signing.verify_artifacts(manifest, tmp_path)


# --- full generation verify -----------------------------------------------
def test_write_and_verify_generation(tmp_path):
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    _write(tmp_path, "index.faiss", b"vectors")
    _write(tmp_path, "metadata.json", b"{}")
    signing.write_signed_manifest(
        "42", tmp_path, ["index.faiss", "metadata.json"], priv
    )
    manifest = signing.verify_generation(tmp_path, pub)
    assert manifest["generation_id"] == "42"


def test_verify_generation_detects_tampered_artifact(tmp_path):
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    _write(tmp_path, "index.faiss", b"vectors")
    signing.write_signed_manifest("1", tmp_path, ["index.faiss"], priv)
    _write(tmp_path, "index.faiss", b"evil")  # tamper after signing
    with pytest.raises(HashMismatchError):
        signing.verify_generation(tmp_path, pub)


def test_verify_generation_missing_manifest(tmp_path):
    _, pub_pem = signing.generate_keypair()
    pub = signing.load_public_key(pub_pem)
    with pytest.raises(MissingArtifactError):
        signing.verify_generation(tmp_path, pub)


@pytest.mark.parametrize("loader", [signing.load_private_key, signing.load_public_key])
def test_malformed_pem_is_a_typed_artifact_error(loader) -> None:
    with pytest.raises(signing.ArtifactVerificationError, match="key"):
        loader(b"not a PEM key")


def test_verify_generation_wraps_malformed_manifest_json(tmp_path) -> None:
    _, pub_pem = signing.generate_keypair()
    pub = signing.load_public_key(pub_pem)
    (tmp_path / signing.MANIFEST_NAME).write_bytes(b"{not-json")
    (tmp_path / signing.SIGNATURE_NAME).write_bytes(b"signature")

    with pytest.raises(signing.ArtifactVerificationError, match="manifest.*JSON"):
        signing.verify_generation(tmp_path, pub)


def test_verify_generation_rejects_noncanonical_signed_manifest(tmp_path) -> None:
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    manifest = {"version": 1, "generation_id": "000001", "artifacts": {}}
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    (tmp_path / signing.MANIFEST_NAME).write_bytes(payload)
    (tmp_path / signing.SIGNATURE_NAME).write_bytes(
        signing.sign_manifest(manifest, priv)
    )

    with pytest.raises(signing.ArtifactVerificationError, match="canonical"):
        signing.verify_generation(tmp_path, pub)


# --- trusted public key from deployment config ----------------------------
def test_load_trusted_public_key_from_env(tmp_path, monkeypatch):
    _, pub_pem = signing.generate_keypair()
    key_path = tmp_path / "pub.pem"
    key_path.write_bytes(pub_pem)
    monkeypatch.setenv("HELION_RAG_PUBLIC_KEY_PATH", str(key_path))
    assert signing.load_trusted_public_key() is not None


def test_load_trusted_public_key_unset(monkeypatch):
    monkeypatch.delenv("HELION_RAG_PUBLIC_KEY_PATH", raising=False)
    with pytest.raises(MissingArtifactError):
        signing.load_trusted_public_key()


def test_setup_generates_signing_keypair_once_with_private_permissions(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "keys" / "publisher-private.pem"
    public_path = tmp_path / "keys" / "publisher-public.pem"

    assert setup_helpers.ensure_signing_keypair(private_path, public_path) is True
    original = (private_path.read_bytes(), public_path.read_bytes())
    assert private_path.stat().st_mode & 0o777 == 0o600

    assert setup_helpers.ensure_signing_keypair(private_path, public_path) is False
    assert (private_path.read_bytes(), public_path.read_bytes()) == original


def test_setup_rejects_partial_signing_keypair(tmp_path: Path) -> None:
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_text("orphan", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial signing keypair"):
        setup_helpers.ensure_signing_keypair(private_path, public_path)
