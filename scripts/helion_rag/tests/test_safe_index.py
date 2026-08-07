from __future__ import annotations

import numpy as np
import pytest

from helion_rag import safe_index
from helion_rag import signing
from helion_rag.signing import HashMismatchError
from helion_rag.signing import MissingArtifactError
from helion_rag.signing import SignatureError

pytest.importorskip("cryptography")

_VECTORS = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    dtype="float32",
)
_METADATA = [{"id": 0}, {"id": 1}, {"id": 2}]


def _keys():
    priv_pem, pub_pem = signing.generate_keypair()
    return signing.load_private_key(priv_pem), signing.load_public_key(pub_pem)


def _save(gen_dir, priv, extra=None):
    idx = safe_index.build_safe_index(_VECTORS, _METADATA)
    return safe_index.save_generation(gen_dir, idx, priv, extra_json=extra)


def test_build_validates_shapes():
    with pytest.raises(ValueError):
        safe_index.build_safe_index(_VECTORS, [{"id": 0}])  # length mismatch


def test_roundtrip_and_search(tmp_path):
    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv, extra={"exact.json": {"k": "v"}})

    loaded, extra = safe_index.load_generation(gen, pub)
    assert extra["exact.json"] == {"k": "v"}

    hits = loaded.search(np.array([0.0, 0.9, 0.0, 0.0], dtype="float32"), k=2)
    assert hits[0][1]["id"] == 1  # nearest to the row-1 direction
    assert hits[0][0] == pytest.approx(1.0, abs=1e-5)


def test_load_detects_tampered_index(tmp_path):
    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)
    (gen / safe_index.INDEX_FILE).write_bytes(b"corrupted")
    with pytest.raises(HashMismatchError):
        safe_index.load_generation(gen, pub)


def test_load_detects_tampered_metadata(tmp_path):
    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)
    (gen / safe_index.METADATA_FILE).write_bytes(b'[{"id": 99}]')
    with pytest.raises(HashMismatchError):
        safe_index.load_generation(gen, pub)


def test_load_fails_closed_with_wrong_key(tmp_path):
    priv, _ = _keys()
    _, other_pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)
    with pytest.raises(SignatureError):
        safe_index.load_generation(gen, other_pub)


def test_load_missing_manifest(tmp_path):
    _, pub = _keys()
    gen = tmp_path / "000001"
    gen.mkdir()
    with pytest.raises(MissingArtifactError):
        safe_index.load_generation(gen, pub)


def test_extra_artifact_name_reserved(tmp_path):
    priv, _ = _keys()
    gen = tmp_path / "000001"
    idx = safe_index.build_safe_index(_VECTORS, _METADATA)
    with pytest.raises(ValueError):
        safe_index.save_generation(gen, idx, priv, extra_json={"metadata.json": {}})


def _replace_and_resign(gen, priv, name, payload) -> None:
    (gen / name).write_bytes(payload)
    manifest = signing.build_manifest(
        gen.name,
        gen,
        [safe_index.INDEX_FILE, safe_index.METADATA_FILE],
    )
    (gen / signing.MANIFEST_NAME).write_bytes(signing.canonical_json_bytes(manifest))
    (gen / signing.SIGNATURE_NAME).write_bytes(signing.sign_manifest(manifest, priv))


def test_load_wraps_signed_malformed_faiss_artifact(tmp_path):
    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)
    _replace_and_resign(gen, priv, safe_index.INDEX_FILE, b"not-faiss")

    with pytest.raises(signing.ArtifactVerificationError, match="FAISS"):
        safe_index.load_generation(gen, pub)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "metadata.*JSON"),
        (b'{"not":"a list"}', "metadata.*list"),
        (b'[ {"id":0} ]', "metadata.*canonical"),
    ],
)
def test_load_rejects_malformed_or_noncanonical_metadata(tmp_path, payload, message):
    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)
    _replace_and_resign(gen, priv, safe_index.METADATA_FILE, payload)

    with pytest.raises(signing.ArtifactVerificationError, match=message):
        safe_index.load_generation(gen, pub)


def test_load_validates_faiss_vector_and_metadata_row_count(tmp_path):
    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)
    _replace_and_resign(
        gen,
        priv,
        safe_index.METADATA_FILE,
        signing.canonical_json_bytes(_METADATA[:1]),
    )

    with pytest.raises(
        signing.ArtifactVerificationError,
        match="vector count 3.*metadata row count 1",
    ):
        safe_index.load_generation(gen, pub)


def test_unexpected_programming_error_from_faiss_propagates(tmp_path, monkeypatch):
    import faiss

    priv, pub = _keys()
    gen = tmp_path / "000001"
    _save(gen, priv)

    def fail_programming_error(path):
        raise AssertionError("programming bug")

    monkeypatch.setattr(faiss, "read_index", fail_programming_error)
    with pytest.raises(AssertionError, match="programming bug"):
        safe_index.load_generation(gen, pub)
