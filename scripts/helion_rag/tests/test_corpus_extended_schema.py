"""Tests for §6.2 extended-schema (S4/S5) recording in the corpus layer."""

from __future__ import annotations

from helion_rag import corpus


def _base_record(run_id: str, dim: int) -> dict:
    return {
        "run_id": run_id,
        "kernel_name": "rope_fwd",
        "kernel_source": "def rope(x):\n    return x\n",
        "input_shapes": f"[(2, {dim})]",
        "dtypes": "['torch.float32']",
        "settings": {"backend": "triton"},
        "configs": {
            "0": {
                "config": {"block_sizes": [dim]},
                "perf_stats": {"median": 1.0, "n_samples": 5},
            }
        },
    }


def _extended_record(run_id: str, dim: int) -> dict:
    record = _base_record(run_id, dim)
    record.update(
        {
            "canonical_source_hash": "csh-" + run_id,
            "normalized_specialization_identity": "spec",
            "config_spec_fingerprint": "fp",
            "record_id": run_id,
            "toolchain_identity": "helion=0.1;torch=2.9;triton=3.5",
            "tolerance_policy_version": "v1",
        }
    )
    return record


def test_historical_record_is_not_tier0_eligible() -> None:
    parsed = corpus._parse_record(
        _base_record("h1", 64), "h100", "autotune-rope.meta.jsonl"
    )
    assert parsed is not None
    assert parsed["tier0_eligible"] is False
    # No extended identity fields are carried through.
    for field in (*corpus.EXTENDED_S4_FIELDS, *corpus.EXTENDED_S5_FIELDS):
        assert field not in parsed


def test_extended_record_is_tier0_eligible_and_carries_identity() -> None:
    parsed = corpus._parse_record(
        _extended_record("e1", 64), "h100", "autotune-rope.meta.jsonl"
    )
    assert parsed is not None
    assert parsed["tier0_eligible"] is True
    assert parsed["canonical_source_hash"] == "csh-e1"
    assert parsed["config_spec_fingerprint"] == "fp"
    assert parsed["tolerance_policy_version"] == "v1"


def test_tier0_eligible_requires_full_s4_identity() -> None:
    partial = _extended_record("p1", 64)
    del partial["config_spec_fingerprint"]  # missing one S4 field
    assert corpus._tier0_eligible(partial) is False


def test_extended_schema_report_partitions_populations() -> None:
    records = [
        corpus._parse_record(_extended_record("e1", 64), "h100", "s"),
        corpus._parse_record(_extended_record("e2", 128), "h100", "s"),
        corpus._parse_record(_base_record("h1", 64), "h100", "s"),
    ]
    report = corpus.extended_schema_report([r for r in records if r])
    assert report.total == 3
    assert report.extended == 2
    assert report.historical == 1


def test_regen_extended_records_carry_identity() -> None:
    runs = [_extended_record("e1", 64), _extended_record("e2", 128)]
    parsed = corpus.regen_extended(runs, "h100")
    assert len(parsed) == 2
    assert all(record["tier0_eligible"] for record in parsed)
    assert corpus.extended_schema_report(parsed).extended == 2
