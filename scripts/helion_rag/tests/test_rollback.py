"""Roll-forward rollback: republish last-good content as a newer signed generation."""

from __future__ import annotations

import numpy as np
import pytest

from helion_rag import index as index_mod
from helion_rag import safe_index
from helion_rag import signing
from helion_rag.config import Config

pytest.importorskip("faiss")

_FAMILY = "h100"


def _cfg(tmp_path) -> Config:
    return Config(
        embed_model="unused",
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        writeback_dir=tmp_path / "writeback",
        hardware_family=_FAMILY,
    )


def _publish(cfg, gen_id, priv, *, marker):
    gen = cfg.index_dir / _FAMILY / "generations" / gen_id
    idx = safe_index.build_safe_index(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), [{"record_id": marker}]
    )
    safe_index.save_generation(
        gen, idx, priv, generation_id=gen_id, extra_json={"exact.json": {marker: 1}}
    )
    (cfg.index_dir / _FAMILY / "current").write_text(f"{gen_id}\n", encoding="utf-8")


def test_rollforward_republishes_last_good(tmp_path):
    cfg = _cfg(tmp_path)
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)

    _publish(cfg, "000000", priv, marker="good")
    _publish(cfg, "000001", priv, marker="bad")

    # Roll forward the last-good generation (000000) as a new signed generation.
    new_gen = index_mod.republish_generation(cfg, _FAMILY, "000000", priv, pub)
    assert new_gen.name == "000002"

    cur = (cfg.index_dir / _FAMILY / "current").read_text().strip()
    assert cur == "000002"

    _, extra = index_mod.load_family_generation(cfg, _FAMILY, pub)
    assert extra["exact.json"] == {"good": 1}  # content matches the last-good gen


def test_rollforward_refuses_unverified_source_generation(tmp_path):
    cfg = _cfg(tmp_path)
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    _publish(cfg, "000000", priv, marker="good")
    source = cfg.index_dir / _FAMILY / "generations" / "000000"
    (source / "metadata.json").write_bytes(b"[]")

    with pytest.raises(signing.HashMismatchError):
        index_mod.republish_generation(cfg, _FAMILY, "000000", priv, pub)

    assert not (cfg.index_dir / _FAMILY / "generations" / "000001").exists()


def test_rollforward_copies_only_manifest_covered_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    _publish(cfg, "000000", priv, marker="good")
    source = cfg.index_dir / _FAMILY / "generations" / "000000"
    (source / "untrusted.bin").write_bytes(b"not manifest covered")

    new_gen = index_mod.republish_generation(cfg, _FAMILY, "000000", priv, pub)

    assert not (new_gen / "untrusted.bin").exists()


def test_rollforward_rejects_source_with_mismatched_signed_generation_id(tmp_path):
    cfg = _cfg(tmp_path)
    priv_pem, pub_pem = signing.generate_keypair()
    priv = signing.load_private_key(priv_pem)
    pub = signing.load_public_key(pub_pem)
    source = cfg.index_dir / _FAMILY / "generations" / "000000"
    idx = safe_index.build_safe_index(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"),
        [{"record_id": "good"}],
    )
    safe_index.save_generation(
        source,
        idx,
        priv,
        generation_id="000999",
        extra_json={"exact.json": {"good": 1}},
    )

    with pytest.raises(index_mod.GenerationPinError, match="000999.*000000"):
        index_mod.republish_generation(cfg, _FAMILY, "000000", priv, pub)

    assert not (cfg.index_dir / _FAMILY / "generations" / "000001").exists()
