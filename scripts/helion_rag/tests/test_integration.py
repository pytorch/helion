"""End-to-end tests over real components — no mocks, no seams.

These replace the old mock-based unit tests that faked the vector store, the
BoundKernel, and family resolution:

* ``_extract`` / ``apply`` run against a real ``BoundKernel`` built from
  ``examples.add`` on the live CUDA device (``requires_cuda``).
* Tier-1 runs over a real FAISS index built by the bundle's own
  ``build_family_index`` (needs langchain + faiss + an embedding model, so it
  ``importorskip``s; run in ``.rag-venv`` with a small model cached and
  ``HF_HUB_DISABLE_XET=1``).

``apply``'s ``original`` argument is the real wrapped method that ``install()``
passes through — not a test seam — so tests pass a lightweight ``original`` to
observe fallback without running a full autotune. Anything that can only be
exercised by mocking hardware/deps is intentionally not tested here.
"""

from __future__ import annotations

import json
from pathlib import Path

from helion_rag.config import Config
import helion_rag.corpus as corpus
import helion_rag.index as index_mod
import helion_rag.lookup as lookup_mod
import helion_rag.patch as patch_mod
import pytest
import torch

from ._fixtures import DTYPES
from ._fixtures import SHAPES
from ._fixtures import SRC

_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real CUDA device"
)


@pytest.fixture
def _index_deps():
    pytest.importorskip("langchain_community")
    pytest.importorskip("faiss")


def _bound_add():
    """A real BoundKernel for examples.add on 16x16 fp16 CUDA tensors."""
    from examples.add import add

    x = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    y = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    return add.bind((x, y)), (x, y)


def _cfg(tmp_path: Path) -> Config:
    return Config(
        embed_model=_EMBED_MODEL,
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        writeback_dir=tmp_path / "writeback",
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


def _write_corrupt_generation(cfg: Config, family: str) -> None:
    """Present index bundle whose exact.json is unparseable JSON."""
    gen = cfg.index_dir / family / "generations" / "000000"
    gen.mkdir(parents=True)
    (gen / "exact.json").write_text("{ this is not json", encoding="utf-8")
    (gen / "index.faiss").write_text("placeholder", encoding="utf-8")
    (cfg.index_dir / family / "current").write_text("000000\n", encoding="utf-8")


@requires_cuda
def test_apply_falls_back_on_corrupt_index(tmp_path, monkeypatch) -> None:
    """A corrupt index must log and fall back to the original autotune, not crash."""
    bound, args = _bound_add()
    cfg = _cfg(tmp_path)
    _write_corrupt_generation(cfg, "h100")
    monkeypatch.setenv("HELION_RAG_ENABLED", "1")
    monkeypatch.setenv("HELION_RAG_INDEX_DIR", str(cfg.index_dir))

    called = []

    def original(kernel, a, *rest, **kwargs):
        called.append(a)
        return "fell-back"

    result = patch_mod.apply(bound, original, args, (), {})

    assert result == "fell-back"
    assert called == [args]


@requires_cuda
def test_lookup_tier0_over_real_index(_index_deps, tmp_path) -> None:
    """Tier-0 hits when the query identity matches what _extract produces for a
    real BoundKernel (real source/shapes/dtypes/settings → same workload key)."""
    bound, args = _bound_add()
    info = patch_mod._extract(bound, args)
    assert info is not None
    best = {"block_sizes": [16]}
    cfg = _cfg(tmp_path)
    rec = _record(info["shapes"], info["dtypes"], run_id="R", config=best)
    rec["kernel_source"] = info["kernel_source"]  # match _extract's real source
    rec["settings"] = info["settings"]
    _build_index(cfg, "h100", [rec])

    res = lookup_mod.lookup(
        info["kernel_source"],
        info["shapes"],
        info["dtypes"],
        info["hardware"],
        settings=info["settings"],
        cfg=cfg,
    )

    assert res["tier"] == 0
    assert res["best_config"] == best


def test_lookup_tier1_over_real_index(_index_deps, tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    _build_index(
        cfg,
        "h100",
        [_record(SHAPES, DTYPES, run_id="R", config={"block_sizes": [16]})],
    )
    # Query a different shape so Tier-0 misses and Tier-1 similarity runs.
    monkeypatch.setenv("HELION_RAG_SIM_THRESHOLD", "0.1")
    res = lookup_mod.lookup(
        SRC, "[(1024, 1024), (1024, 1024)]", DTYPES, "h100", cfg=cfg
    )

    assert res["tier"] == 1
    assert res["neighbors"]
    assert res["neighbors"][0]["kernel_name"] == "add"
