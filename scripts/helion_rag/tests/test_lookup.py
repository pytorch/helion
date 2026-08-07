"""Standalone Tier-0 exact lookup and Tier-2 miss tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helion_rag import safe_index
from helion_rag import signing
from helion_rag.config import Config
from helion_rag.config import _config
import helion_rag.corpus as corpus
import helion_rag.index as index_mod
import helion_rag.lookup as lookup_mod

from ._fixtures import DTYPES
from ._fixtures import FAMILY
from ._fixtures import SHAPES
from ._fixtures import SRC

pytest.importorskip("faiss")
pytest.importorskip("cryptography")


def _cfg(tmp_path: Path) -> Config:
    return Config(
        embed_model="unused",
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        writeback_dir=tmp_path / "writeback",
        hardware_family=FAMILY,
    )


def _write_signed_generation(
    cfg,
    family,
    *,
    exact,
    tmp_path,
    monkeypatch,
    generation_id="000000",
    keypair=None,
) -> tuple[bytes, bytes]:
    """Publish a real signed generation and point the public key at its keypair."""
    priv_pem, pub_pem = keypair or signing.generate_keypair()
    key_path = tmp_path / "pub.pem"
    key_path.write_bytes(pub_pem)
    monkeypatch.setenv("HELION_RAG_PUBLIC_KEY_PATH", str(key_path))

    gen = cfg.index_dir / family / "generations" / generation_id
    idx = safe_index.build_safe_index(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), [{"record_id": "x"}]
    )
    safe_index.save_generation(
        gen,
        idx,
        signing.load_private_key(priv_pem),
        generation_id=generation_id,
        extra_json={"exact.json": exact, "runids.json": {}},
    )
    (cfg.index_dir / family / "current").write_text(
        f"{generation_id}\n", encoding="utf-8"
    )
    return priv_pem, pub_pem


def test_lookup_tier0_exact_hit(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    hit = {
        "best_config": {"block_size": 16},
        "best_config_id": "cfg0",
        "run_id": "RUN1",
        "ref": {"family": FAMILY, "source_file": "add.meta.jsonl", "run_id": "RUN1"},
        "tier0_eligible": True,
    }
    _write_signed_generation(
        cfg, FAMILY, exact={key: hit}, tmp_path=tmp_path, monkeypatch=monkeypatch
    )

    res = lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)

    assert res["tier"] == 0
    assert res["best_config"] == {"block_size": 16}
    assert res["best_config_id"] == "cfg0"
    assert res["artifact_identity"]["index_id"] == "000000"
    assert len(res["artifact_identity"]["manifest_id"]) == 64
    assert len(res["artifact_identity"]["corpus_id"]) == 64


def test_live_lookup_requires_explicit_generation_pin_but_cli_lookup_does_not(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={key: {"tier0_eligible": True, "best_config": {}}},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)["tier"] == 0
    with pytest.raises(index_mod.GenerationPinError, match="explicit generation pin"):
        lookup_mod.lookup(
            SRC,
            SHAPES,
            DTYPES,
            "unknown",
            cfg=cfg,
            require_generation_pin=True,
            propagate_artifact_errors=True,
        )


def test_lookup_tier2_fails_closed_without_public_key(tmp_path, monkeypatch) -> None:
    """A published generation with no trusted public key configured fails closed."""
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={key: {"tier0_eligible": True, "best_config": {}}},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    monkeypatch.delenv("HELION_RAG_PUBLIC_KEY_PATH", raising=False)
    assert lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)["tier"] == 2


def test_live_lookup_propagates_artifact_verification_failure(
    tmp_path, monkeypatch
) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={key: {"tier0_eligible": False, "best_config": {}}},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    monkeypatch.delenv("HELION_RAG_PUBLIC_KEY_PATH", raising=False)

    with pytest.raises(signing.MissingArtifactError):
        lookup_mod.lookup(
            SRC,
            SHAPES,
            DTYPES,
            "unknown",
            cfg=cfg,
            propagate_artifact_errors=True,
        )


def test_lookup_tier2_when_no_index_for_family(tmp_path: Path) -> None:
    # The explicit test family avoids a host-device probe, but no index exists.
    cfg = _cfg(tmp_path)
    assert lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg) == {
        "tier": 2,
        "family": FAMILY,
    }


def test_config_reads_explicit_generation_pin(monkeypatch) -> None:
    monkeypatch.setenv("HELION_RAG_GENERATION_ID", "000123")

    assert _config().generation_id == "000123"


def test_config_reads_frozen_embedding_revision_and_policy(monkeypatch) -> None:
    monkeypatch.setenv("HELION_RAG_MODEL_REVISION", "revision-123")
    monkeypatch.setenv("HELION_RAG_TOKENIZER_REVISION", "revision-123")
    monkeypatch.setenv("HELION_RAG_EMBED_DEVICE", "cuda")

    cfg = _config()

    assert cfg.model_revision == "revision-123"
    assert cfg.tokenizer_revision == "revision-123"
    assert cfg.embedding_policy == {
        "device": "cuda",
        "normalize_embeddings": True,
        "pooling": "model_default",
        "precision": "model_default",
        "sequence_length": "model_default",
        "truncation": "model_default",
    }


def test_embedding_loader_applies_the_frozen_revision(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_embeddings(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFaceEmbeddings=fake_embeddings),
    )
    monkeypatch.setenv("HELION_RAG_EMBED_DEVICE", "cuda")
    cfg = SimpleNamespace(
        embed_model="qwen-4b",
        model_revision="revision-123",
        tokenizer_revision="revision-123",
    )

    index_mod._embeddings(cfg)

    assert captured["model_kwargs"] == {
        "device": "cuda",
        "revision": "revision-123",
    }


def test_lookup_uses_pinned_generation_instead_of_current(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    keypair = signing.generate_keypair()
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={
            key: {
                "tier0_eligible": True,
                "best_config": {"block_size": 16},
            }
        },
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        generation_id="000000",
        keypair=keypair,
    )
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={
            key: {
                "tier0_eligible": True,
                "best_config": {"block_size": 32},
            }
        },
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        generation_id="000001",
        keypair=keypair,
    )

    result = lookup_mod.lookup(
        SRC,
        SHAPES,
        DTYPES,
        "unknown",
        cfg=replace(cfg, generation_id="000000"),
    )

    assert result["best_config"] == {"block_size": 16}
    assert result["artifact_identity"]["index_id"] == "000000"


def test_missing_generation_pin_fails_closed(tmp_path: Path, monkeypatch) -> None:
    cfg = replace(_cfg(tmp_path), generation_id="999999")
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)["tier"] == 2
    with pytest.raises(signing.MissingArtifactError, match="999999"):
        lookup_mod.lookup(
            SRC,
            SHAPES,
            DTYPES,
            "unknown",
            cfg=cfg,
            propagate_artifact_errors=True,
        )


def test_signed_generation_identity_must_match_pin(tmp_path: Path, monkeypatch) -> None:
    cfg = replace(_cfg(tmp_path), generation_id="000000")
    priv_pem, pub_pem = signing.generate_keypair()
    key_path = tmp_path / "pub.pem"
    key_path.write_bytes(pub_pem)
    monkeypatch.setenv("HELION_RAG_PUBLIC_KEY_PATH", str(key_path))
    gen = cfg.index_dir / FAMILY / "generations" / "000000"
    idx = safe_index.build_safe_index(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), [{"record_id": "x"}]
    )
    safe_index.save_generation(
        gen,
        idx,
        signing.load_private_key(priv_pem),
        generation_id="000001",
        extra_json={"exact.json": {}, "runids.json": {}},
    )

    with pytest.raises(index_mod.GenerationPinError, match="000001.*000000"):
        lookup_mod.lookup(
            SRC,
            SHAPES,
            DTYPES,
            "unknown",
            cfg=cfg,
            propagate_artifact_errors=True,
        )


def test_artifact_identity_uses_verified_manifest(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    _write_signed_generation(
        cfg,
        FAMILY,
        exact={key: {"tier0_eligible": True, "best_config": {}}},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    generation = cfg.index_dir / FAMILY / "generations" / "000000"
    manifest_path = generation / signing.MANIFEST_NAME
    trusted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_corpus_id = trusted_manifest["artifacts"]["metadata.json"]
    expected_manifest_id = signing.sha256_hex(
        signing.canonical_json_bytes(trusted_manifest)
    )
    load_generation = safe_index.load_generation

    def load_then_tamper(gen_dir, public_key):
        loaded = load_generation(gen_dir, public_key)
        forged = json.loads(manifest_path.read_text(encoding="utf-8"))
        forged["artifacts"]["metadata.json"] = "f" * 64
        manifest_path.write_bytes(signing.canonical_json_bytes(forged))
        return loaded

    monkeypatch.setattr(safe_index, "load_generation", load_then_tamper)

    result = lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)

    assert result["artifact_identity"]["corpus_id"] == expected_corpus_id
    assert result["artifact_identity"]["manifest_id"] == expected_manifest_id
