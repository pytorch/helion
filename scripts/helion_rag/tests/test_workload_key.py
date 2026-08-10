"""Workload key identity, parity, and Tier-0 eligibility tests."""

from __future__ import annotations

import helion_rag.corpus as C
import pytest

from ._fixtures import INGEST_DTYPES
from ._fixtures import INGEST_SHAPES
from ._fixtures import FAMILY
from ._fixtures import OTHER_FAMILY
from ._fixtures import RUNTIME_DTYPES
from ._fixtures import RUNTIME_SHAPES
from ._fixtures import SETTINGS
from ._fixtures import SRC_EPILOGUE
from ._fixtures import SRC_PLAIN


def _perf_stats(median: float, *, std: float = 0.1, n_samples: int = 10) -> dict:
    return {
        "min": median - 0.1,
        "median": median,
        "mean": median,
        "p90": median + 0.1,
        "std": std,
        "n_samples": n_samples,
    }


def test_workload_key_canonicalizes_container_types() -> None:
    list_forms = C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, FAMILY
    )
    tuple_forms = C._workload_key(
        SRC_PLAIN, RUNTIME_SHAPES, RUNTIME_DTYPES, SETTINGS, FAMILY
    )
    assert list_forms == tuple_forms


def test_non_codegen_setting_does_not_change_key() -> None:
    trimmed = {k: v for k, v in SETTINGS.items() if k != "autotune_random_seed"}
    assert C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, FAMILY
    ) == C._workload_key(SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, trimmed, FAMILY)


def test_family_and_settings_change_key() -> None:
    base = C._workload_key(SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, FAMILY)
    assert base != C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, SETTINGS, OTHER_FAMILY
    )
    other = {**SETTINGS, "dot_precision": "ieee"}
    assert base != C._workload_key(
        SRC_PLAIN, INGEST_SHAPES, INGEST_DTYPES, other, FAMILY
    )


def test_runid_map_keeps_all_run_ids_but_dedup_keeps_fastest() -> None:
    """Run_id map keeps duplicates; dedup keeps the lowest-median record per key."""
    recs = [
        {
            "workload_key": "K",
            "run_id": "R1",
            "family": FAMILY,
            "best": {"median": 2.0},
        },
        {
            "workload_key": "K",
            "run_id": "R2",
            "family": FAMILY,
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
        "kernel_name": "add",
        "kernel_source": SRC_PLAIN,
        "input_shapes": shapes,
        "dtypes": dtypes,
        "settings": SETTINGS,
        "configs": {
            "c0": {
                "config": {"block_sizes": [64]},
                "source_hash": "hash-c0",
                "perf_stats": [_perf_stats(1.0)],
            }
        },
    }


def test_parse_record_ranks_by_median_and_keeps_per_shape_stats() -> None:
    multi_shapes = f"[{INGEST_SHAPES}, {INGEST_SHAPES}]"
    multi_dtypes = f"[{INGEST_DTYPES}, {INGEST_DTYPES}]"
    record = _record(multi_shapes, multi_dtypes)
    slower_stats = [_perf_stats(1.0), _perf_stats(9.0)]
    faster_stats = [_perf_stats(3.0, std=0.5), _perf_stats(3.0, std=0.2)]
    record["configs"] = {
        "slower": {
            "config": {"block_sizes": [32]},
            "source_hash": "hash-slower",
            "perf_stats": slower_stats,
        },
        "faster": {
            "config": {"block_sizes": [64]},
            "source_hash": "hash-faster",
            "perf_stats": faster_stats,
        },
    }

    parsed = C._parse_record(record, FAMILY, "f.jsonl")

    assert parsed is not None
    assert parsed["best"]["config_id"] == "faster"
    assert parsed["best"]["median"] == pytest.approx(3.0)
    assert parsed["best"]["perf_stats"] == faster_stats
    exact = C._exact_map([parsed])
    assert exact[parsed["workload_key"]]["perf_stats"] == faster_stats
    assert exact[parsed["workload_key"]]["source_hash"] == "hash-faster"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min", None),
        ("median", float("nan")),
        ("mean", float("inf")),
        ("p90", "fast"),
        ("std", None),
        ("n_samples", 0),
        ("n_samples", True),
    ],
)
def test_parse_record_rejects_invalid_perf_stats(field: str, value: object) -> None:
    record = _record(INGEST_SHAPES, INGEST_DTYPES)
    record["configs"]["c0"]["perf_stats"][0][field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        C._parse_record(record, FAMILY, "f.jsonl")


def test_parse_record_requires_every_perf_stats_field() -> None:
    record = _record(INGEST_SHAPES, INGEST_DTYPES)
    del record["configs"]["c0"]["perf_stats"][0]["std"]

    with pytest.raises(KeyError, match="std"):
        C._parse_record(record, FAMILY, "f.jsonl")


@pytest.mark.parametrize(
    ("shapes", "dtypes", "stats"),
    [
        (INGEST_SHAPES, INGEST_DTYPES, [_perf_stats(1.0), _perf_stats(2.0)]),
        (
            f"[{INGEST_SHAPES}, {INGEST_SHAPES}]",
            f"[{INGEST_DTYPES}, {INGEST_DTYPES}]",
            [_perf_stats(1.0)],
        ),
    ],
)
def test_parse_record_requires_one_stats_record_per_shape_case(
    shapes: str, dtypes: str, stats: list[dict]
) -> None:
    record = _record(shapes, dtypes)
    record["configs"]["c0"]["perf_stats"] = stats

    with pytest.raises(ValueError, match="performance-statistics records"):
        C._parse_record(record, FAMILY, "f.jsonl")


@pytest.mark.parametrize("shapes", ["not-a-literal", ""])
def test_parse_record_rejects_unparsable_shapes(shapes: str) -> None:
    with pytest.raises((SyntaxError, ValueError)):
        C._parse_record(_record(shapes, INGEST_DTYPES), FAMILY, "f.jsonl")


def test_parse_record_raises_for_missing_schema_field() -> None:
    record = _record(INGEST_SHAPES, INGEST_DTYPES)
    del record["configs"]

    with pytest.raises(KeyError, match="configs"):
        C._parse_record(record, FAMILY, "f.jsonl")
