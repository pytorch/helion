"""Standalone Tier-1 lookup over a real FAISS index."""

from __future__ import annotations

import json
import os
from pathlib import Path

from helion_rag.config import Config
import helion_rag.corpus as corpus
import helion_rag.index as index_mod
import helion_rag.lookup as lookup_mod
import pytest

from ._fixtures import DTYPES
from ._fixtures import FAMILY
from ._fixtures import SHAPES
from ._fixtures import SRC

# Small model by default for CI; override with a locally-cached model (e.g.
# Qwen/Qwen3-Embedding-4B) to validate against the audit-diagnostic models.
_EMBED_MODEL = os.environ.get(
    "HELION_RAG_TEST_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)


@pytest.fixture
def _index_deps(tmp_path, monkeypatch):
    pytest.importorskip("langchain_huggingface")
    pytest.importorskip("faiss")
    from helion_rag import signing

    priv_pem, pub_pem = signing.generate_keypair()
    priv_path = tmp_path / "priv.pem"
    pub_path = tmp_path / "pub.pem"
    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)
    monkeypatch.setenv("HELION_RAG_PRIVATE_KEY_PATH", str(priv_path))
    monkeypatch.setenv("HELION_RAG_PUBLIC_KEY_PATH", str(pub_path))


def _cfg(tmp_path: Path) -> Config:
    return Config(
        embed_model=_EMBED_MODEL,
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        writeback_dir=tmp_path / "writeback",
        hardware_family=FAMILY,
    )


def _record(shapes: str, dtypes: str, *, run_id: str, config: dict) -> dict:
    return {
        "kernel_name": "add",
        "run_id": run_id,
        "input_shapes": shapes,
        "dtypes": dtypes,
        "kernel_source": SRC,
        "settings": {},
        "configs": {
            "cfg0": {"config": config, "perf_stats": {"median": 0.5, "n_samples": 5}}
        },
    }


def _build_index(cfg: Config, family: str, records: list[dict]) -> None:
    fam_dir = cfg.corpus_dir / family
    fam_dir.mkdir(parents=True)
    (fam_dir / "add.meta.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    index_mod.build_family_index(cfg, family, corpus.load_corpus(cfg.corpus_dir))


def test_lookup_tier1_over_real_index(_index_deps, tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _build_index(
        cfg,
        FAMILY,
        [_record(SHAPES, DTYPES, run_id="R", config={"block_sizes": [16]})],
    )
    # Query a different shape so Tier-0 misses and Tier-1 similarity runs.
    res = lookup_mod.lookup(
        SRC, "[(1024, 1024), (1024, 1024)]", DTYPES, "unknown", cfg=cfg
    )

    assert res["tier"] == 1
    assert res["neighbors"]
    assert res["neighbors"][0]["kernel_name"] == "add"
