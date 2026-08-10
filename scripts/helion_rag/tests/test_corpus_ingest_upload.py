"""Corpus extract, ingest idempotency, upload markers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import helion_rag.cli as cli
from helion_rag.config import Config
import helion_rag.corpus as corpus
import helion_rag.ingest as ingest
import helion_rag.upload as upload
import pytest

from ._fixtures import DTYPES
from ._fixtures import FAMILY
from ._fixtures import SHAPES
from ._fixtures import SRC


def _perf_stats(median: float) -> dict:
    return {
        "min": median - 0.1,
        "median": median,
        "mean": median,
        "p90": median + 0.1,
        "std": 0.1,
        "n_samples": 50,
    }


def _meta_record(run_id: str = "RUN1", median: float = 1.0) -> dict:
    return {
        "run_id": run_id,
        "kernel_name": "add",
        "kernel_source": SRC,
        "input_shapes": SHAPES,
        "dtypes": DTYPES,
        "hardware_info": {"device_kind": "cuda", "device_name": "test-device"},
        "settings": {"backend": "triton"},
        "configs": {
            "cfg0": {
                "generated_code": "large generated code",
                "source_hash": "source-hash",
                "config": {"block_size": 16},
                "perf_stats": [_perf_stats(median)],
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
    _write_zip(zips / FAMILY / "a.zip", _meta_record("RUN1", median=1.0))
    _write_zip(zips / FAMILY / "b.zip", _meta_record("RUN1", median=1.0))
    _write_zip(zips / FAMILY / "c.zip", _meta_record("RUN2", median=2.0))

    assert corpus.extract_corpus(zips, out) == 2

    written = sorted((out / FAMILY).glob("*.meta.jsonl"))
    assert len(written) == 2
    assert "add.meta.jsonl" in {p.name for p in written}
    for path in written:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert "generated_code" not in record["configs"]["cfg0"]

    loaded = corpus.load_corpus(out)
    assert {r["run_id"] for r in loaded} == {"RUN1", "RUN2"}


def test_extract_corpus_never_overwrites_colliding_names(tmp_path: Path) -> None:
    zips = tmp_path / "zips"
    archive = zips / FAMILY / "same.zip"
    archive.parent.mkdir(parents=True)
    records = [_meta_record(f"RUN{i}", median=float(i)) for i in range(3)]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, record in enumerate(records):
            zf.writestr(f"logs-{index}/add.meta.jsonl", json.dumps(record) + "\n")

    assert corpus.extract_corpus(zips, tmp_path / "corpus") == 3
    written = list((tmp_path / "corpus" / FAMILY).glob("*.meta.jsonl"))
    assert len(written) == 3
    assert {
        json.loads(path.read_text(encoding="utf-8"))["run_id"] for path in written
    } == {"RUN0", "RUN1", "RUN2"}
    assert corpus.extract_corpus(zips, tmp_path / "corpus") == 0


def test_cli_extract_subcommand(tmp_path: Path) -> None:
    """Drives `python -m helion_rag extract` end-to-end (guards the cli import path)."""
    data_dir = tmp_path / "data"
    _write_zip(data_dir / FAMILY / "a.zip", _meta_record("RUN1", median=1.0))

    # Fresh interpreter: pytest's rootdir sys.path insertion doesn't propagate
    # to subprocesses and helion_rag isn't pip-installed in CI, so put the
    # package dir (scripts/helion_rag/) on PYTHONPATH for `-m helion_rag`.
    pkg_root = Path(corpus.__file__).resolve().parents[1]
    env = {
        **os.environ,
        "HELION_RAG_DATA_DIR": str(data_dir),
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(pkg_root), os.environ.get("PYTHONPATH", "")) if p
        ),
    }
    subprocess.run([sys.executable, "-m", "helion_rag", "extract"], check=True, env=env)
    assert (data_dir / "corpus" / FAMILY / "add.meta.jsonl").is_file()


def test_ingest_preserves_jsonl_metadata_and_is_idempotent(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    source_record = _meta_record("RUN1")
    (logs / "run.meta.jsonl").write_text(
        json.dumps(source_record) + "\n", encoding="utf-8"
    )

    writeback = tmp_path / "writeback"
    ledger = tmp_path / "ledger.json"

    first = ingest.ingest(
        autotune_log_dir=logs,
        writeback_dir=writeback,
        family=FAMILY,
        ledger_path=ledger,
        reindex=False,
    )
    second = ingest.ingest(
        autotune_log_dir=logs,
        writeback_dir=writeback,
        family=FAMILY,
        ledger_path=ledger,
        reindex=False,
    )

    assert first == {"family": FAMILY, "ingested_run_ids": ["RUN1"], "skipped": 0}
    assert second == {"family": FAMILY, "ingested_run_ids": [], "skipped": 1}
    lines = (
        (writeback / FAMILY / "local-autotune.meta.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == source_record
    assert json.loads(ledger.read_text(encoding="utf-8")) == {"run_ids": ["RUN1"]}


def test_ingest_deduplicates_batch_and_existing_writeback(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runs.meta.jsonl").write_text(
        "\n".join(json.dumps(_meta_record("RUN1")) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "writeback" / FAMILY / "local-autotune.meta.jsonl"
    out_file.parent.mkdir(parents=True)
    out_file.write_text(
        "\n".join(json.dumps(_meta_record("RUN0")) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"

    result = ingest.ingest(
        autotune_log_dir=logs,
        writeback_dir=out_file.parents[1],
        family=FAMILY,
        ledger_path=ledger,
        reindex=False,
    )

    records = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert [record["run_id"] for record in records] == ["RUN0", "RUN1"]
    assert result["ingested_run_ids"] == ["RUN1"]
    assert json.loads(ledger.read_text()) == {"run_ids": ["RUN0", "RUN1"]}


def test_ingest_validates_the_batch_before_writing(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    good = _meta_record("RUN1")
    bad = _meta_record("RUN2")
    del bad["configs"]["cfg0"]["perf_stats"][0]["std"]
    (logs / "runs.meta.jsonl").write_text(
        f"{json.dumps(good)}\n{json.dumps(bad)}\n", encoding="utf-8"
    )
    writeback = tmp_path / "writeback"
    ledger = tmp_path / "ledger.json"

    with pytest.raises(KeyError, match="std"):
        ingest.ingest(
            autotune_log_dir=logs,
            writeback_dir=writeback,
            family=FAMILY,
            ledger_path=ledger,
            reindex=False,
        )

    assert not writeback.exists()
    assert not ledger.exists()


def test_ingest_restores_writeback_when_reindex_fails(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.meta.jsonl").write_text(
        json.dumps(_meta_record("RUN1")) + "\n", encoding="utf-8"
    )
    out_file = tmp_path / "writeback" / FAMILY / "local-autotune.meta.jsonl"
    out_file.parent.mkdir(parents=True)
    original = json.dumps(_meta_record("RUN0")) + "\n"
    out_file.write_text(original, encoding="utf-8")
    index_path = tmp_path / "index-is-a-file"
    index_path.write_text("not a directory", encoding="utf-8")
    cfg = Config(
        embed_model="unused",
        data_dir=tmp_path / "data",
        index_dir=index_path,
        writeback_dir=out_file.parents[1],
    )
    ledger = tmp_path / "ledger.json"

    with pytest.raises((FileExistsError, ModuleNotFoundError)):
        ingest.ingest(
            autotune_log_dir=logs,
            writeback_dir=out_file.parents[1],
            family=FAMILY,
            ledger_path=ledger,
            cfg=cfg,
        )

    assert out_file.read_text(encoding="utf-8") == original
    assert not ledger.exists()


def test_cli_upload_requires_autotune_log_dir(capsys) -> None:
    keys = ("HELION_RAG_HARDWARE_FAMILY", "HELION_RAG_AUTOTUNE_LOG_DIR")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HELION_RAG_HARDWARE_FAMILY"] = FAMILY
    os.environ.pop("HELION_RAG_AUTOTUNE_LOG_DIR", None)
    try:
        result = cli.main(["upload", "--dry-run"])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert result == 2
    assert "HELION_RAG_AUTOTUNE_LOG_DIR must be set" in capsys.readouterr().err


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
    (logs / "perf.csv").write_text("not used by RAG\n", encoding="utf-8")
    (logs / "run.log").write_text("not used by RAG\n", encoding="utf-8")
    uploads = tmp_path / "uploads"

    result = upload.upload(
        autotune_log_dir=logs,
        uploads_dir=uploads,
        family=FAMILY,
        contributor="tester",
    )
    assert result["uploaded"] is False
    assert result["run_ids"] == []
    assert Path(result["archive_path"]).is_file()
    with zipfile.ZipFile(result["archive_path"]) as zf:
        manifest = json.loads(zf.read("batch-manifest.json"))
        names = zf.namelist()
    assert manifest == {
        "family": FAMILY,
        "contributor": "tester",
        "run_ids": ["RUN1", "RUN2"],
    }
    assert not any(name.endswith((".csv", ".log")) for name in names)
    assert not (uploads / "uploaded-runs").exists()


def test_upload_archive_name_is_content_addressed(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    archive_paths = []
    for directory, run_ids in (("first", ("A1", "A2")), ("second", ("B1", "B2"))):
        logs = tmp_path / directory
        logs.mkdir()
        (logs / "runs.meta.jsonl").write_text(
            "\n".join(json.dumps(_meta_record(run_id)) for run_id in run_ids) + "\n",
            encoding="utf-8",
        )
        result = upload.upload(
            autotune_log_dir=logs,
            uploads_dir=uploads,
            family=FAMILY,
            contributor="tester",
        )
        archive_paths.append(Path(result["archive_path"]))

    assert archive_paths[0] != archive_paths[1]
    assert all(path.is_file() for path in archive_paths)


def test_upload_archive_contains_only_manifest_runs(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write_two_runs(logs)
    uploads = tmp_path / "uploads"
    upload.record_upload(
        ["RUN1"],
        "prior",
        uploads / "uploaded-runs",
        uploads / "uploaded-archives",
    )

    result = upload.upload(
        autotune_log_dir=logs,
        uploads_dir=uploads,
        family=FAMILY,
        contributor="tester",
    )

    with zipfile.ZipFile(result["archive_path"]) as zf:
        manifest = json.loads(zf.read("batch-manifest.json"))
        records = [
            json.loads(line)
            for name in zf.namelist()
            if name.endswith(".meta.jsonl")
            for line in zf.read(name).decode().splitlines()
        ]
    assert manifest["run_ids"] == ["RUN2"]
    assert [record["run_id"] for record in records] == ["RUN2"]


def test_upload_rejects_malformed_jsonl_record(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "bad.meta.jsonl").write_text('{"configs": {}}\n', encoding="utf-8")

    with pytest.raises(KeyError, match="run_id"):
        upload.upload(
            autotune_log_dir=logs,
            uploads_dir=tmp_path / "uploads",
            family=FAMILY,
            contributor="tester",
            dry_run=True,
        )


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
