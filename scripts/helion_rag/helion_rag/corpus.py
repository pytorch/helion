"""Turn CI benchmark artifacts into a searchable corpus.

The workload key must match Helion's own run_id logic, so the codegen-setting
list and signature are imported straight from `helion.autotuner.metrics` rather
than vendored here - single source of truth. Corpus and index generation
therefore require `helion` to be importable."""

from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
import math
import operator
from pathlib import Path
import statistics
import zipfile

from helion.autotuner.metrics import _CODEGEN_SETTINGS
from helion.autotuner.metrics import _codegen_signature
from helion_rag._util import DEFAULT_TOP_N
from helion_rag._util import _die
from helion_rag._util import _log
from helion_rag.models import ExactEntry
from helion_rag.models import Ref

__all__ = ["_CODEGEN_SETTINGS"]


def _to_canonical_nested(v):
    """Turn nested lists into tuples so repr is stable across runs."""
    return (
        tuple(_to_canonical_nested(x) for x in v) if isinstance(v, (list, tuple)) else v
    )


def _literal_sequence(raw: str, name: str) -> list | tuple:
    value = ast.literal_eval(raw)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _canon_shapes(s: str) -> str:
    shapes = _literal_sequence(s, "input_shapes")
    _perf_stats_count(shapes)
    return repr(_to_canonical_nested(shapes))


def _canon_dtypes(s: str) -> str:
    return repr(_to_canonical_nested(_literal_sequence(s, "dtypes")))


def _normalize_kernel_source(src: str) -> str:
    """Normalize source via AST dump."""
    return ast.dump(ast.parse(src))


def _tier0_eligible(kernel_source: str) -> bool:
    """Reject kernels with epilogue or Callable args."""
    tree = ast.parse(kernel_source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        for p in (*a.posonlyargs, *a.args, *a.kwonlyargs):
            if p.arg == "epilogue":
                return False
            if p.annotation is not None and "Callable" in ast.unparse(p.annotation):
                return False
        for d in (*a.defaults, *(d for d in a.kw_defaults if d is not None)):
            if isinstance(d, ast.Lambda):
                return False
    return True


def _workload_key(
    kernel_source: str, shapes: str, dtypes: str, settings: dict, family: str
) -> str:
    """Stable hash of normalized source + shapes + dtypes + settings + family.
    Same at ingest time and query time, unlike Helion run_id which is device-specific."""
    if not isinstance(kernel_source, str) or not kernel_source.strip():
        raise ValueError("kernel_source must be non-empty")
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dictionary")
    payload = "\x00".join(
        (
            _normalize_kernel_source(kernel_source),
            _codegen_signature(settings or {}),
            _canon_shapes(shapes),
            _canon_dtypes(dtypes),
            family,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _perf_stats_count(shapes: list | tuple) -> int:
    if any(not isinstance(item, (list, tuple)) for item in shapes):
        raise ValueError("input_shapes entries must be sequences")
    nested = [
        any(isinstance(item, (list, tuple)) for item in shape_or_case)
        for shape_or_case in shapes
    ]
    if any(nested):
        if not all(nested):
            raise ValueError("input_shapes has inconsistent nesting")
        return len(shapes)
    return 1


def _validate_perf_stats(config_id: str, perf_stats: object, expected: int) -> list:
    if not isinstance(perf_stats, list):
        raise TypeError(f"config {config_id} perf_stats must be a list")
    if len(perf_stats) != expected:
        raise ValueError(
            f"config {config_id} must have {expected} performance-statistics records"
        )
    for stats in perf_stats:
        if not isinstance(stats, dict):
            raise TypeError(
                f"config {config_id} performance statistics must be objects"
            )
        for field in ("min", "median", "mean", "p90", "std"):
            value = stats[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"config {config_id} {field} must be finite")
        n_samples = stats["n_samples"]
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise TypeError(f"config {config_id} n_samples must be an integer")
        if n_samples <= 0:
            raise ValueError(f"config {config_id} n_samples must be positive")
    return perf_stats


def _ok_configs(configs: dict, expected_stats: int) -> list:
    """Keep configs with perf samples, sorted fastest first."""
    oks = []
    for config_id, entry in configs.items():
        perf_stats = _validate_perf_stats(
            config_id, entry["perf_stats"], expected_stats
        )
        oks.append(
            {
                "config_id": config_id,
                "config": entry["config"],
                "source_hash": entry["source_hash"],
                "median": statistics.median(stats["median"] for stats in perf_stats),
                "perf_stats": perf_stats,
            }
        )
    oks.sort(key=operator.itemgetter("median"))
    return oks


def _parse_record(
    record: dict, family: str, source_file: str, top_n: int = DEFAULT_TOP_N
) -> dict | None:
    """Turn one meta.jsonl line into index record, or skip if unusable."""
    raw_shapes = _literal_sequence(record["input_shapes"], "input_shapes")
    shapes = repr(_to_canonical_nested(raw_shapes))
    dtypes = _canon_dtypes(record["dtypes"])
    oks = _ok_configs(record["configs"], _perf_stats_count(raw_shapes))
    run_id = record["run_id"]
    if not oks:
        _log(f"{source_file}: run {run_id} has no ok configs; skipping")
        return None
    ksrc = record["kernel_source"]
    key = _workload_key(ksrc, shapes, dtypes, record["settings"], family)
    ref = Ref(family=family, source_file=source_file, run_id=run_id)
    return {
        "family": family,
        "kernel_name": record["kernel_name"],
        "run_id": run_id,
        "source_file": source_file,
        "input_shapes": shapes,
        "dtypes": dtypes,
        "workload_key": key,
        "tier0_eligible": _tier0_eligible(ksrc),
        "embed_text": ksrc.strip(),
        "best": oks[0],
        "top_n": oks[:top_n],
        "ref": ref,
    }


def load_corpus(corpus_dir, required: bool = True) -> list:
    """Load all *.meta.jsonl under corpus_dir/<family>/ into parsed records."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        if not required:
            return []
        _die(f"corpus dir not found: {corpus_dir} — run extraction step.")
    out = []
    nfiles = 0
    for f in sorted(corpus_dir.rglob("*.meta.jsonl")):
        family = f.relative_to(corpus_dir).parts[0]
        nfiles += 1
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                parsed = _parse_record(rec, family, f.name)
                if parsed:
                    out.append(parsed)
    _log(f"loaded {len(out)} records from {nfiles} files under {corpus_dir}")
    if not out and required:
        _die(f"no usable *.meta.jsonl records under {corpus_dir}")
    return out


def _strip_generated_code(data: bytes) -> bytes:
    """Remove per-config generated_code from meta jsonl to save space."""
    out = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for entry in (rec.get("configs") or {}).values():
            entry.pop("generated_code", None)
        out.append(json.dumps(rec, default=str))
    return ("\n".join(out) + "\n").encode("utf-8") if out else b""


def extract_corpus(zips_dir, out_dir) -> int:
    """Unzip benchmark archives, keep only meta.jsonl stripped of generated code.
    Dedupes by content hash per family. Returns number of files written."""
    zips_dir, out_dir = Path(zips_dir).resolve(), Path(out_dir).resolve()
    written = 0

    def _emit(family: str, base: str, data: bytes) -> None:
        nonlocal written
        fam_out = out_dir / family
        fam_out.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        candidate = base
        while True:
            path = fam_out / candidate
            try:
                with path.open("xb") as fh:
                    fh.write(data)
            except FileExistsError:
                if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                    return
                if candidate != base:
                    raise RuntimeError(f"content-address collision for {path}")
                candidate = f"{digest}__{base}"
                _log(
                    f"WARN: {family}/{base} recurs with different content; "
                    f"writing as {candidate}"
                )
                continue
            written += 1
            return

    for zf_path in sorted(zips_dir.rglob("*.zip")):
        family = zf_path.relative_to(zips_dir).parts[0]
        with zipfile.ZipFile(zf_path) as zf:
            for member in zf.namelist():
                if member.endswith(".meta.jsonl"):
                    _emit(
                        family,
                        Path(member).name,
                        _strip_generated_code(zf.read(member)),
                    )
    for stray in sorted(zips_dir.rglob("*.meta.jsonl")):
        if out_dir == stray.parent or out_dir in stray.parents:
            continue
        family = stray.relative_to(zips_dir).parts[0]
        _emit(family, stray.name, _strip_generated_code(stray.read_bytes()))
    return written


def _group_by_family(records: list) -> dict:
    """Group by family, no dedup yet — keeps all run_ids for runid map."""
    by = {}
    for r in records:
        by.setdefault(r["family"], []).append(r)
    return by


def _dedup_by_key(records: list) -> list:
    """Keep the fastest config for each workload. If a workload was measured in
    multiple runs, the one with the lowest median time wins."""
    best_by_key: dict[str, dict] = {}
    for r in records:
        cur = best_by_key.get(r["workload_key"])
        if cur is None or r["best"]["median"] < cur["best"]["median"]:
            best_by_key[r["workload_key"]] = r
    return list(best_by_key.values())


def _exact_map(records: list) -> dict:
    """workload_key -> best config and provenance for Tier-0 exact match."""
    return {
        r["workload_key"]: asdict(
            ExactEntry(
                best_config=r["best"]["config"],
                best_config_id=r["best"]["config_id"],
                source_hash=r["best"]["source_hash"],
                perf_stats=r["best"]["perf_stats"],
                run_id=r["run_id"],
                ref=r["ref"],
                tier0_eligible=r["tier0_eligible"],
            )
        )
        for r in records
    }


def _runid_map(records: list) -> dict:
    """run_id -> workload_key for provenance lookups."""
    return {r["run_id"]: r["workload_key"] for r in records if r.get("run_id")}
