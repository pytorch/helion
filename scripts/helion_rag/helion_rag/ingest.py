"""Turn autotune logs into writeback corpus and rebuild index."""

from __future__ import annotations

from collections import defaultdict
import csv as _csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from helion_rag.models import PerfStats

_OK_STATUSES = {"ok", "success", "successful", "pass", "passed"}
_WRITEBACK_FILE = "local-autotune.meta.jsonl"


def is_ok_row(row: dict) -> bool:
    """True if row status is ok and perf_ms is a finite number."""
    if str(row.get("status", "")).strip().lower() not in _OK_STATUSES:
        return False
    return math.isfinite(float(str(row.get("perf_ms", "")).strip() or "nan"))


def join_records(meta_records: list[dict], csv_rows: list[dict]) -> dict:
    """Join csv perf rows to meta configs by run_id and config_id, keep only ok rows."""
    valid = {
        (rec.get("run_id"), cid)
        for rec in meta_records
        for cid in (rec.get("configs") or {})
    }
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in csv_rows:
        pair = (r.get("run_id"), r.get("config_id"))
        if pair in valid and None not in pair and is_ok_row(r):
            out[pair].append(float(r["perf_ms"]))
    return dict(out)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear interpolation percentile like numpy."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return (
        sorted_vals[lo]
        if lo == hi
        else sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)
    )


def aggregate_perf_stats(perf_ms: list[float]) -> PerfStats:
    """Summarize list of times into median, mean, min, p90, std and count."""
    vals = sorted(float(v) for v in perf_ms)
    n = len(vals)
    if n == 0:
        return PerfStats(
            median=None, mean=None, min=None, p90=None, std=None, n_samples=0
        )
    return PerfStats(
        median=statistics.median(vals),
        mean=statistics.fmean(vals),
        min=vals[0],
        p90=_percentile(vals, 0.90),
        std=statistics.pstdev(vals) if n > 1 else 0.0,
        n_samples=n,
    )


def _read_meta_records(log_dir: Path) -> list[dict]:
    """Read all *.meta.jsonl under log dir."""
    out = []
    for f in sorted(Path(log_dir).rglob("*.meta.jsonl")):
        text = f.read_text(encoding="utf-8")
        out.extend(json.loads(ln) for ln in text.splitlines() if ln.strip())
    return out


def _read_csv_rows(log_dir: Path) -> list[dict]:
    """Read all csv files under log dir."""
    rows: list[dict] = []
    for f in sorted(Path(log_dir).rglob("*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            rows.extend(_csv.DictReader(fh))
    return rows


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
    joined = join_records(meta_records, _read_csv_rows(autotune_log_dir))

    processed = _load_ledger(ledger_path)
    out_dir = writeback_dir / family
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / _WRITEBACK_FILE

    newly: list[str] = []
    with out_file.open("a", encoding="utf-8") as fh:
        for rec in meta_records:
            run_id = rec.get("run_id")
            if not run_id or run_id in processed:
                continue
            configs_out = {
                cid: {
                    **(
                        {"generated_code": e["generated_code"]}
                        if e.get("generated_code")
                        else {}
                    ),
                    "config": e.get("config", {}),
                    "perf_stats": asdict(aggregate_perf_stats(joined[(run_id, cid)])),
                }
                for cid, e in (rec.get("configs") or {}).items()
                if (run_id, cid) in joined
            }
            if not configs_out:
                continue
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "kernel_name": rec.get("kernel_name", ""),
                        "kernel_source": rec.get("kernel_source", ""),
                        "input_shapes": rec.get("input_shapes", ""),
                        "dtypes": rec.get("dtypes", ""),
                        "hardware": rec.get("hardware", ""),
                        "settings": rec.get("settings", {}),
                        "configs": configs_out,
                    }
                )
                + "\n"
            )
            newly.append(run_id)

    if newly:
        processed.update(newly)
        _save_ledger(ledger_path, processed)
        if reindex:
            _reindex_family(family, writeback_dir, cfg)

    return {
        "family": family,
        "ingested_run_ids": newly,
        "skipped": len(processed) - len(newly),
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
