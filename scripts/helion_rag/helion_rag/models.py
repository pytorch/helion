"""Dataclasses for our saved RAG data.

These define the shape of what we save to disk (like `*.meta.jsonl` and
`exact.json`) and store in FAISS metadata. This makes sure field names don't
silently get out of sync. Use `dataclasses.asdict()` before writing anything
out."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerfStats:
    """Aggregated timing summary for one config across its perf samples."""

    median: float | None
    mean: float | None
    min: float | None
    p90: float | None
    std: float | None
    n_samples: int


@dataclass
class Ref:
    """Provenance for a corpus record: where a config came from."""

    family: str
    source_file: str
    run_id: str | None


@dataclass
class ExactEntry:
    """Tier-0 payload keyed by workload_key: the measured-best config + origin."""

    best_config: dict
    best_config_id: str
    run_id: str | None
    ref: Ref
    tier0_eligible: bool
