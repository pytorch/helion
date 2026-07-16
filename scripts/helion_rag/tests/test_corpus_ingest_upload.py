"""Corpus extract, ingest idempotency, upload markers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import helion_rag.corpus as corpus
import helion_rag.ingest as ingest
import helion_rag.upload as upload

from ._fixtures import DTYPES
from ._fixtures import SHAPES
from ._fixtures import SRC


def _meta_record(run_id: str = "RUN1", median: float = 1.0) -> dict:
    return {
        "run_id": run_id,
        "kernel_name": "add",
        "kernel_source": SRC,
        "input_shapes": SHAPES,
        "dtypes": DTYPES,
        "settings": {"backend": "triton"},
        "configs": {
            "cfg0": {
                "generated_code": "large generated code",
                "config": {"block_size": 16},
                "perf_stats": {"median": median, "n_samples": 3},
            }
        },
    }


def _write_zip(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("logs/add.meta.jsonl", json.dumps(record) + "\n")


def test_extract_corpus_strips_generated_code_and_dedups(tmp_path: Path) -> None:
    zips = tmp_path / "zips"
    out = tmp_path / "corpus"
    _write_zip(zips / "h100" / "a.zip", _meta_record("RUN1", median=1.0))
    _write_zip(zips / "h100" / "b.zip", _meta_record("RUN1", median=1.0))
    _write_zip(zips / "h100" / "c.zip", _meta_record("RUN2", median=2.0))

    assert corpus.extract_corpus(zips, out) == 2

    written = sorted((out / "h100").glob("*.meta.jsonl"))
    assert [p.name for p in written] == ["add.meta.jsonl", "c__add.meta.jsonl"]
    for path in written:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert "generated_code" not in record["configs"]["cfg0"]

    loaded = corpus.load_corpus(out)
    assert {r["run_id"] for r in loaded} == {"RUN1", "RUN2"}


def test_cli_extract_subcommand(tmp_path: Path) -> None:
    """Drives `python -m helion_rag extract` end-to-end (guards the cli import path)."""
    data_dir = tmp_path / "data"
    _write_zip(data_dir / "h100" / "a.zip", _meta_record("RUN1", median=1.0))

    subprocess.run(
        [sys.executable, "-m", "helion_rag", "extract"],
        check=True,
        env={**os.environ, "HELION_RAG_DATA_DIR": str(data_dir)},
    )
    assert (data_dir / "corpus" / "h100" / "add.meta.jsonl").is_file()


def test_ingest_joins_aggregates_and_is_idempotent(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.meta.jsonl").write_text(json.dumps(_meta_record("RUN1")) + "\n")
    (logs / "perf.csv").write_text(
        "run_id,config_id,status,perf_ms\nRUN1,cfg0,ok,4.0\nRUN1,cfg0,pass,2.0\nRUN1,cfg0,fail,1.0\nRUN1,missing,ok,0.5"
        "\n",
        encoding="utf-8",
    )

    writeback = tmp_path / "writeback"
    ledger = tmp_path / "ledger.json"

    first = ingest.ingest(
        autotune_log_dir=logs,
        writeback_dir=writeback,
        family="h100",
        ledger_path=ledger,
        reindex=False,
    )
    second = ingest.ingest(
        autotune_log_dir=logs,
        writeback_dir=writeback,
        family="h100",
        ledger_path=ledger,
        reindex=False,
    )

    assert first == {"family": "h100", "ingested_run_ids": ["RUN1"], "skipped": 0}
    assert second == {"family": "h100", "ingested_run_ids": [], "skipped": 1}
    lines = (
        (writeback / "h100" / "local-autotune.meta.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1
    record = json.loads(lines[0])
    stats = record["configs"]["cfg0"]["perf_stats"]
    assert stats["median"] == 3.0
    assert stats["mean"] == 3.0
    assert stats["n_samples"] == 2
    assert json.loads(ledger.read_text(encoding="utf-8")) == {"run_ids": ["RUN1"]}


def _write_two_runs(logs: Path) -> None:
    logs.mkdir()
    (logs / "runs.meta.jsonl").write_text(
        json.dumps(_meta_record("RUN1"))
        + "\n"
        + json.dumps(_meta_record("RUN2"))
        + "\n",
        encoding="utf-8",
    )


def test_upload_without_transport_builds_archive_without_markers(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    _write_two_runs(logs)
    uploads = tmp_path / "uploads"

    result = upload.upload(
        autotune_log_dir=logs,
        uploads_dir=uploads,
        family="h100",
        contributor="tester",
    )
    assert result["uploaded"] is False
    assert result["run_ids"] == []
    assert Path(result["archive_path"]).is_file()
    with zipfile.ZipFile(result["archive_path"]) as zf:
        manifest = json.loads(zf.read("batch-manifest.json"))
    assert manifest == {
        "family": "h100",
        "contributor": "tester",
        "run_ids": ["RUN1", "RUN2"],
    }
    assert not (uploads / "uploaded-runs").exists()


def test_record_upload_writes_run_and_archive_markers(tmp_path: Path) -> None:
    runs_dir = tmp_path / "uploaded-runs"
    archives_dir = tmp_path / "uploaded-archives"

    upload.record_upload(["RUN1", "RUN2"], "abc123", runs_dir, archives_dir)

    assert (runs_dir / "RUN1.json").is_file()
    assert (runs_dir / "RUN2.json").is_file()
    assert json.loads((archives_dir / "abc123.json").read_text(encoding="utf-8")) == {
        "archive_sha256": "abc123",
        "run_ids": ["RUN1", "RUN2"],
    }
