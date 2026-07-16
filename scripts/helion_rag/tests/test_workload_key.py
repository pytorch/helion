"""Workload key identity, parity, and Tier-0 eligibility tests."""

from __future__ import annotations

import helion_rag.corpus as C
import helion_rag.patch as P

from ._fixtures import INGEST_DTYPES
from ._fixtures import INGEST_SHAPES
from ._fixtures import RUNTIME_DTYPES
from ._fixtures import RUNTIME_SHAPES
from ._fixtures import SETTINGS
from ._fixtures import SRC_EPILOGUE
from ._fixtures import SRC_PLAIN


def test_workload_key_ingest_runtime_parity() -> None:
    """Ingest and runtime forms produce same key."""
    ingest = C._workload_key(SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, "h100")
    runtime = C._workload_key(
        SRC_PLAIN, RUNTIME_SHAPES, RUNTIME_DTYPES, P._settings_dict(SETTINGS), "h100"
    )
    assert ingest == runtime


def test_non_codegen_setting_does_not_change_key() -> None:
    trimmed = {k: v for k, v in SETTINGS.items() if k != "autotune_random_seed"}
    assert C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, "h100"
    ) == C._workload_key(SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, trimmed, "h100")


def test_family_and_settings_change_key() -> None:
    base = C._workload_key(SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, "h100")
    assert base != C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, "b200"
    )
    other = {**SETTINGS, "dot_precision": "ieee"}
    assert base != C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, other, "h100"
    )


def test_runid_map_keeps_all_run_ids_but_dedup_keeps_fastest() -> None:
    """Run_id map keeps duplicates; dedup keeps the lowest-median record per key."""
    recs = [
        {
            "workload_key": "K",
            "run_id": "R1",
            "family": "h100",
            "best": {"median": 2.0},
        },
        {
            "workload_key": "K",
            "run_id": "R2",
            "family": "h100",
            "best": {"median": 1.0},
        },
    ]
    assert C._runid_map(recs) == {"R1": "K", "R2": "K"}
    deduped = C._dedup_by_key(recs)
    assert len(deduped) == 1
    assert deduped[0]["run_id"] == "R2"  # lower median wins regardless of load order


def test_tier0_eligible() -> None:
    assert C._tier0_eligible(SRC_PLAIN) is True
    assert C._tier0_eligible(SRC_EPILOGUE) is False


def _record(shapes: str, dtypes: str) -> dict:
    return {
        "run_id": "R",
        "kernel_source": SRC_PLAIN,
        "input_shapes": shapes,
        "dtypes": dtypes,
        "settings": SETTINGS,
        "configs": {
            "c0": {
                "config": {"block_sizes": [64]},
                "perf_stats": {"median": 1.0, "n_samples": 5},
            }
        },
    }


def test_parse_record_skips_unparsable_shapes() -> None:
    """Unparsable/dynamic-SymInt shapes are skipped, not raised, so one bad
    record can't abort the whole corpus load."""
    assert (
        C._parse_record(_record("not-a-literal", INGEST_DTYPES), "h100", "f.jsonl")
        is None
    )
    assert C._parse_record(_record("", INGEST_DTYPES), "h100", "f.jsonl") is None
    ok = C._parse_record(_record(INGEST_SHAPES, INGEST_DTYPES), "h100", "f.jsonl")
    assert ok is not None and ok["workload_key"] and ok["run_id"] == "R"
