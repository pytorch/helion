"""Turn autotune logs into writeback corpus and rebuild index."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from helion_rag.corpus import _parse_record

_WRITEBACK_FILE = "local-autotune.meta.jsonl"


def _read_meta_records(log_dir: Path) -> list[dict]:
    """Read all *.meta.jsonl under log dir."""
    out = []
    for f in sorted(Path(log_dir).rglob("*.meta.jsonl")):
        text = f.read_text(encoding="utf-8")
        out.extend(json.loads(ln) for ln in text.splitlines() if ln.strip())
    return out


def _load_ledger(ledger_path: Path) -> set[str]:
    """Load set of already processed run ids, empty if missing."""
    p = Path(ledger_path)
    if not p.is_file():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get("run_ids", []))


def _save_ledger(ledger_path: Path, run_ids: set[str]) -> None:
    """Write ledger atomically."""
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"run_ids": sorted(run_ids)}), encoding="utf-8")
    tmp.replace(p)


def _replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(data)
    try:
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _validate_record(record: dict, family: str, source_file: str) -> str:
    run_id = record["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{source_file}: run_id must be a non-empty string")
    configs = record["configs"]
    if not isinstance(configs, dict):
        raise TypeError(f"{source_file}: configs must be an object")
    if configs:
        _parse_record(record, family, source_file)
    return run_id


def ingest(
    *,
    autotune_log_dir,
    writeback_dir,
    family: str,
    ledger_path,
    reindex: bool = True,
    cfg=None,
) -> dict:
    """Add new runs to writeback tree, skip already seen via ledger, optionally rebuild index."""
    autotune_log_dir, writeback_dir, ledger_path = map(
        Path, (autotune_log_dir, writeback_dir, ledger_path)
    )

    meta_records = _read_meta_records(autotune_log_dir)
    processed = _load_ledger(ledger_path)
    out_dir = writeback_dir / family
    out_file = out_dir / _WRITEBACK_FILE

    previous = out_file.read_bytes() if out_file.is_file() else None
    existing = (
        [json.loads(line) for line in previous.decode().splitlines() if line.strip()]
        if previous is not None
        else []
    )

    records_by_run_id: dict[str, dict] = {}
    for record in existing:
        run_id = _validate_record(record, family, out_file.name)
        if record["configs"]:
            records_by_run_id.setdefault(run_id, record)
    processed.update(records_by_run_id)

    newly: list[str] = []
    skipped = 0
    for record in meta_records:
        run_id = _validate_record(record, family, "autotune log")
        if not record["configs"]:
            skipped += 1
            continue
        if run_id in processed or run_id in records_by_run_id:
            skipped += 1
            continue
        records_by_run_id[run_id] = record
        newly.append(run_id)

    output = "".join(
        f"{json.dumps(record)}\n" for record in records_by_run_id.values()
    ).encode()
    if output != previous:
        _replace_bytes(out_file, output)

    if newly and reindex:
        try:
            _reindex_family(family, writeback_dir, cfg)
        except BaseException:
            if previous is None:
                out_file.unlink(missing_ok=True)
            else:
                _replace_bytes(out_file, previous)
            raise

    processed.update(newly)
    if records_by_run_id or processed:
        _save_ledger(ledger_path, processed)

    return {
        "family": family,
        "ingested_run_ids": newly,
        "skipped": skipped,
    }


def _reindex_family(family: str, writeback_dir, cfg) -> None:
    """Rebuild index for one family from writeback corpus."""
    import dataclasses

    from helion_rag.config import _config
    from helion_rag.corpus import _group_by_family
    from helion_rag.corpus import load_corpus
    from helion_rag.index import build_family_index

    writeback_dir = Path(writeback_dir)
    if cfg is None:
        cfg = dataclasses.replace(
            _config(),
            writeback_dir=writeback_dir,
            index_dir=writeback_dir.parent / "rag_index",
        )
    records = load_corpus(writeback_dir)
    fam_records = _group_by_family(records).get(family, [])
    if fam_records:
        build_family_index(cfg, family, fam_records)
