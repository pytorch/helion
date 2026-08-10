"""Standalone Tier-0 exact lookup and Tier-2 miss tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from helion_rag.config import Config
import helion_rag.corpus as corpus
import helion_rag.lookup as lookup_mod
import pytest

from ._fixtures import DTYPES
from ._fixtures import FAMILY
from ._fixtures import SHAPES
from ._fixtures import SRC


def _cfg(tmp_path: Path) -> Config:
    return Config(
        embed_model="unused",
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        writeback_dir=tmp_path / "writeback",
        hardware_family=FAMILY,
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
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    perf_stats = [
        {
            "min": 0.4,
            "median": 0.5,
            "mean": 0.55,
            "p90": 0.7,
            "std": 0.1,
            "n_samples": 50,
        }
    ]
    hit = {
        "best_config": {"block_size": 16},
        "best_config_id": "cfg0",
        "source_hash": "source-hash",
        "perf_stats": perf_stats,
        "run_id": "RUN1",
        "ref": {"family": FAMILY, "source_file": "add.meta.jsonl", "run_id": "RUN1"},
        "tier0_eligible": True,
    }
    _write_generation(cfg, FAMILY, exact={key: hit})

    res = lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)

    assert res["tier"] == 0
    assert res["best_config"] == {"block_size": 16}
    assert res["best_config_id"] == "cfg0"
    assert res["source_hash"] == "source-hash"
    assert res["perf_stats"] == perf_stats


def test_lookup_tier2_when_no_index_for_family(tmp_path: Path) -> None:
    # The explicit test family avoids a host-device probe, but no index exists.
    cfg = _cfg(tmp_path)
    assert lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg) == {
        "tier": 2,
        "family": FAMILY,
    }


@pytest.mark.parametrize("shapes", ["not-a-shape", "[1, 2]"])
def test_lookup_tier2_for_invalid_shapes(tmp_path: Path, shapes: str) -> None:
    cfg = _cfg(tmp_path)

    assert lookup_mod.lookup(SRC, shapes, DTYPES, "unknown", cfg=cfg) == {
        "tier": 2,
        "family": FAMILY,
    }


def test_lookup_tier2_for_invalid_similarity_threshold(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_generation(cfg, FAMILY, exact={})
    previous = os.environ.get("HELION_RAG_SIM_THRESHOLD")
    os.environ["HELION_RAG_SIM_THRESHOLD"] = "not-a-threshold"
    try:
        result = lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)
    finally:
        if previous is None:
            os.environ.pop("HELION_RAG_SIM_THRESHOLD")
        else:
            os.environ["HELION_RAG_SIM_THRESHOLD"] = previous

    assert result == {"tier": 2, "family": FAMILY}


def test_lookup_invalid_similarity_threshold_precedes_exact_hit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    _write_generation(
        cfg,
        FAMILY,
        exact={key: {"tier0_eligible": True}},
    )
    previous = os.environ.get("HELION_RAG_SIM_THRESHOLD")
    os.environ["HELION_RAG_SIM_THRESHOLD"] = "not-a-threshold"
    try:
        result = lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)
    finally:
        if previous is None:
            os.environ.pop("HELION_RAG_SIM_THRESHOLD")
        else:
            os.environ["HELION_RAG_SIM_THRESHOLD"] = previous

    assert result == {"tier": 2, "family": FAMILY}


@pytest.mark.parametrize("settings", [[], ["backend"], "triton"])
def test_lookup_tier2_for_non_mapping_settings(
    tmp_path: Path, settings: object
) -> None:
    cfg = _cfg(tmp_path)

    result = lookup_mod.lookup(
        SRC,
        SHAPES,
        DTYPES,
        "unknown",
        settings=cast("dict | None", settings),
        cfg=cfg,
    )

    assert result == {"tier": 2, "family": FAMILY}


def test_lookup_raises_for_invalid_exact_entry(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, FAMILY)
    _write_generation(
        cfg,
        FAMILY,
        exact={
            key: {
                "best_config": {"block_size": 16},
                "best_config_id": "cfg0",
                "run_id": "RUN1",
                "ref": {},
                "tier0_eligible": True,
            }
        },
    )

    with pytest.raises(KeyError, match="source_hash"):
        lookup_mod.lookup(SRC, SHAPES, DTYPES, "unknown", cfg=cfg)
