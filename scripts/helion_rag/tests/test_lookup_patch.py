"""Tier-0 exact lookup, Tier-2 miss, and seed-config merge — real data, no mocks.

Tier-1 (FAISS) and the ``apply`` runtime hook need a real vector store / BoundKernel,
so they live in ``test_integration.py`` behind ``importorskip``/``requires_cuda``.
"""

from __future__ import annotations

import json
from pathlib import Path

from helion_rag.config import Config
import helion_rag.corpus as corpus
import helion_rag.lookup as lookup_mod

from ._fixtures import DTYPES
from ._fixtures import SHAPES
from ._fixtures import SRC


def _cfg(tmp_path: Path) -> Config:
    return Config(
        embed_model="unused",
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        writeback_dir=tmp_path / "writeback",
    )


def _write_generation(cfg: Config, family: str, *, exact: dict) -> None:
    """Write a real on-disk generation (Tier-0 reads exact.json; no FAISS needed)."""
    gen = cfg.index_dir / family / "generations" / "000000"
    gen.mkdir(parents=True)
    (gen / "exact.json").write_text(json.dumps(exact), encoding="utf-8")
    (gen / "index.faiss").write_text("placeholder", encoding="utf-8")
    (cfg.index_dir / family / "current").write_text("000000\n", encoding="utf-8")


def test_lookup_tier0_exact_hit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, "h100")
    hit = {
        "best_config": {"block_size": 16},
        "best_config_id": "cfg0",
        "run_id": "RUN1",
        "ref": {"family": "h100", "source_file": "add.meta.jsonl", "run_id": "RUN1"},
        "tier0_eligible": True,
    }
    _write_generation(cfg, "h100", exact={key: hit})

    res = lookup_mod.lookup(SRC, SHAPES, DTYPES, "h100", cfg=cfg)

    assert res["tier"] == 0
    assert res["best_config"] == {"block_size": 16}
    assert res["best_config_id"] == "cfg0"


def test_lookup_tier2_when_no_index_for_family(tmp_path: Path) -> None:
    # "h100" resolves via the device-token list (no torch probe needed), but there's
    # no index bundle on disk -> Tier-2 miss.
    cfg = _cfg(tmp_path)
    assert lookup_mod.lookup(SRC, SHAPES, DTYPES, "h100", cfg=cfg) == {
        "tier": 2,
        "family": "h100",
    }


def test_merge_seed_configs_keeps_user_first_and_dedups() -> None:
    user = [{"block_size": 16}, {"block_size": 32}]
    rag = [{"block_size": 16}, {"block_size": 64}]

    assert lookup_mod.merge_seed_configs(user, rag) == [
        {"block_size": 16},
        {"block_size": 32},
        {"block_size": 64},
    ]
