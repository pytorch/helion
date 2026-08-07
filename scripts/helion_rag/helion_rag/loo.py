"""Pure helpers for leave-one-workload-out folds."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable


def _semantic_sig(record: dict) -> tuple:
    """Identity of a workload for holdout purposes: kernel + shapes + dtypes."""
    return (
        record.get("kernel_name"),
        record.get("input_shapes"),
        record.get("dtypes"),
    )


def records_for_workload_fold(
    records: Iterable[dict], target_workload_key: str
) -> list[dict]:
    """Return a fold corpus with the held-out workload removed **semantically**.

    The same workload (identical kernel + shapes + dtypes) can appear under
    several ``workload_key``s across CI snapshots when codegen settings or source
    differ slightly. Removing only one key would leave a sibling that the live
    query can still Tier-0/Tier-1 match — leaking the held-out workload's own
    tuned config. So we drop *every* record sharing the target's
    (kernel, shapes, dtypes), while leaving the kernel's other shapes intact so
    retrieval may still draw on same-kernel neighbours.
    """
    records = list(records)
    target = next(
        (r for r in records if r.get("workload_key") == target_workload_key), None
    )
    if target is None:
        return records
    sig = _semantic_sig(target)
    return [record for record in records if _semantic_sig(record) != sig]


def _workload_size(record: dict) -> int:
    shapes = ast.literal_eval(record["input_shapes"])
    return sum(math.prod(shape) for shape in shapes)


def select_heldout_workloads(
    eligible_workloads: Iterable[dict], target_kernel: str, *, count: int = 3
) -> list[dict]:
    """Choose up to ``count`` distinct workloads of one kernel spanning sizes.

    Operates on an already-frozen *eligible* set; eligibility is never
    recomputed here. Picks evenly-spaced size quantiles (min .. max) so held-out
    shapes cover small/median/large regimes.
    """
    by_key = {
        record["workload_key"]: record
        for record in eligible_workloads
        if record.get("kernel_name") == target_kernel
    }
    ordered = sorted(
        by_key.values(),
        key=lambda record: (_workload_size(record), record["workload_key"]),
    )
    if count < 1:
        raise ValueError("count must be at least 1")
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    last = len(ordered) - 1
    indices = sorted(
        dict.fromkeys(round(last * step / (count - 1)) for step in range(count))
    )
    return [ordered[index] for index in indices]


def corpus_fingerprint(records: Iterable[dict]) -> str:
    """Hash the retrieval-relevant content of a fold corpus."""
    rows = [
        {
            "kernel_name": record.get("kernel_name"),
            "workload_key": record.get("workload_key"),
            "input_shapes": record.get("input_shapes"),
            "dtypes": record.get("dtypes"),
            "top_n": record.get("top_n"),
        }
        for record in records
    ]
    rows.sort(key=lambda row: (str(row["kernel_name"]), str(row["workload_key"])))
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fold_slug(kernel_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", kernel_name).strip("-")
    if not slug:
        raise ValueError("target kernel must contain a usable name")
    return slug


def _workload_fold_slug(kernel_name: str, workload_key: str) -> str:
    """Directory slug for a single held-out (kernel, workload) fold."""
    return f"{_fold_slug(kernel_name)}__{workload_key[:16]}"


def prepare_workload_fold(
    cfg,
    family: str,
    target_workload_key: str,
    records: Iterable[dict],
    *,
    retrieval: dict | None = None,
):
    """Build an isolated index with a single (kernel, shape) workload excluded.

    The held-out ``workload_key`` is physically absent from the fold's FAISS
    index and its ``exact.json`` (both are rebuilt from ``fold_records``), so
    Tier-0 for the excluded workload cannot fire and Tier-1 cannot return it.
    Other shapes of the same kernel remain, so the kernel stays in
    ``included_kernels``.
    """
    from helion_rag.index import build_family_index

    all_records = list(records)
    target = next(
        (
            record
            for record in all_records
            if record.get("workload_key") == target_workload_key
        ),
        None,
    )
    if target is None:
        raise ValueError(
            f"held-out workload {target_workload_key!r} is not in the corpus"
        )
    kernel_name = target.get("kernel_name", "")
    fold_records = [
        record
        for record in records_for_workload_fold(all_records, target_workload_key)
        if record.get("family") == family
    ]
    if not fold_records:
        raise ValueError(
            f"fold for workload {target_workload_key!r} has no {family!r} records"
        )
    fold_root = (
        Path(cfg.index_dir)
        / "loo_folds_shape"
        / family
        / _workload_fold_slug(kernel_name, target_workload_key)
    )
    fold_cfg = replace(cfg, index_dir=fold_root)
    build_family_index(fold_cfg, family, fold_records)
    fold_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "family": family,
        "regime": "loo_workload",
        "excluded_workload_key": target_workload_key,
        "kernel_name": kernel_name,
        "included_records": len(fold_records),
        "included_kernels": sorted(
            {record.get("kernel_name", "") for record in fold_records}
        ),
        "corpus_fingerprint": corpus_fingerprint(fold_records),
        "embed_model": cfg.embed_model,
        "embed_text": cfg.embed_text,
        "retrieval": retrieval or {},
    }
    (fold_root / "fold.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return fold_cfg
