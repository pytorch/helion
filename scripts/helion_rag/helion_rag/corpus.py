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
import operator
from pathlib import Path
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


def _canon_shapes(s: str) -> str:
    return repr(_to_canonical_nested(ast.literal_eval(s)))


def _canon_dtypes(s: str) -> str:
    return repr(_to_canonical_nested(ast.literal_eval(s)))


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


def _ok_configs(configs: dict) -> list:
    """Keep configs with perf samples, sorted fastest first."""
    oks = [
        {
            "config_id": cid,
            "config": e.get("config", {}),
            "median": e["perf_stats"]["median"],
        }
        for cid, e in configs.items()
        if (ps := e.get("perf_stats") or {}).get("n_samples", 0) > 0
        and ps.get("median") is not None
    ]
    oks.sort(key=operator.itemgetter("median"))
    return oks


def _parse_record(
    record: dict, family: str, source_file: str, top_n: int = DEFAULT_TOP_N
) -> dict | None:
    """Turn one meta.jsonl line into index record, or skip if unusable."""
    oks = _ok_configs(record.get("configs", {}))
    if not oks:
        _log(f"{source_file}: run {record.get('run_id')} has no ok configs; skipping")
        return None
    try:
        shapes = _canon_shapes(record.get("input_shapes", ""))
        dtypes = _canon_dtypes(record.get("dtypes", ""))
    except (SyntaxError, ValueError):
        # Dynamic-SymInt shapes aren't parseable; skip so one bad record
        # can't abort the whole corpus load / index build.
        _log(
            f"{source_file}: run {record.get('run_id')} has unparsable shapes/dtypes (e.g. dynamic SymInt); skipping"
        )
        return None
    ksrc = record.get("kernel_source", "")
    run_id = record.get("run_id")
    key = _workload_key(ksrc, shapes, dtypes, record.get("settings") or {}, family)
    ref = Ref(family=family, source_file=source_file, run_id=run_id)
    return {
        "family": family,
        "kernel_name": record.get("kernel_name", ""),
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
    seen: dict[tuple[str, str], str] = {}
    written = 0

    def _emit(family: str, base: str, data: bytes, origin: str) -> None:
        nonlocal written
        fam_out = out_dir / family
        fam_out.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        key = (family, base)
        if key in seen:
            if seen[key] == digest:
                return
            base = f"{origin}__{base}"
            _log(
                f"WARN: {family}/{key[1]} recurs with different content; writing as {base}"
            )
            key = (family, base)
        (fam_out / base).write_bytes(data)
        seen[key] = digest
        written += 1

    for zf_path in sorted(zips_dir.rglob("*.zip")):
        family = zf_path.relative_to(zips_dir).parts[0]
        with zipfile.ZipFile(zf_path) as zf:
            for member in zf.namelist():
                if member.endswith(".meta.jsonl"):
                    _emit(
                        family,
                        Path(member).name,
                        _strip_generated_code(zf.read(member)),
                        zf_path.stem,
                    )
    for stray in sorted(zips_dir.rglob("*.meta.jsonl")):
        if out_dir == stray.parent or out_dir in stray.parents:
            continue
        family = stray.relative_to(zips_dir).parts[0]
        _emit(family, stray.name, _strip_generated_code(stray.read_bytes()), stray.stem)
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
