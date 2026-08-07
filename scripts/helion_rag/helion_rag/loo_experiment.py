"""Resumable leave-one-workload-out experiment: RAG-seeded LFBO vs plain LFBO.

The two arms run the identical ``LFBOTreeSearch`` autotuner and differ only by
whether retrieval seeds the search, so the measured difference is attributable
to the seeds alone. Each held-out workload gets its own fold whose index and
exact map physically exclude that workload, forcing retrieval to generalize
from sibling shapes rather than replay the stored answer.

Three phases, each resumable: ``preflight`` establishes which candidate
workloads the baseline can tune at all, ``select`` freezes the held-out set and
builds its folds, ``eval`` runs the matched pairs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Iterable
from typing import TypedDict


class RetrievalSpec(TypedDict):
    """The retrieval controls a cell was produced under."""

    embed_model: str
    embed_text: str
    retrieval: dict[str, object]


class _CellIdentity(TypedDict):
    """The fields every cell carries; together they define ``resume_key``."""

    kernel: str
    workload_key: str
    shapes: str
    dtypes: str
    arm: str
    rep: int
    corpus_fingerprint: str
    code_revision: str


class Cell(_CellIdentity, total=False):
    """One unit of work, and the record written for it.

    The dict *is* the record: it is hashed into ``resume_key``, written to the
    matrix JSONL, passed to the worker, and copied into the result row. Adding a
    field therefore changes every resume key and invalidates an in-progress
    campaign -- deliberately, because a cell produced under a different spec must
    never be silently reused. The keys below are set by particular producers
    rather than by every cell.
    """

    # Assigned immediately after construction; it hashes the other fields, so it
    # cannot be part of the literal it summarizes.
    resume_key: str
    phase: str  # eval and preflight cells; oracle cells have no phase
    run_index: int  # execution position, added by counterbalanced_order
    oracle_configs: list[object]  # oracle cells only
    embed_model: str
    embed_text: str
    retrieval: dict[str, object]


WORKLOAD_ARMS = ("lfbo", "rag_lfbo")
# Candidate -> matched baseline.
WORKLOAD_PAIRS = {"rag_lfbo": "lfbo"}
_AUTOTUNERS = {"lfbo": "LFBOTreeSearch", "rag_lfbo": "LFBOTreeSearch"}
_RAG_ARMS = frozenset({"rag_lfbo"})


def arm_environment(arm: str, fold_index_dir: str | Path) -> dict[str, str]:
    """Return the environment overrides for one experiment arm.

    ``HELION_RAG_LOO_SEEDING`` (not ``HELION_RAG_ENABLED``) drives this study's
    seeding path; see ``helion_rag.patch``.
    """
    if arm not in _AUTOTUNERS:
        raise ValueError(f"unknown experiment arm: {arm}")
    return {
        "HELION_AUTOTUNER": _AUTOTUNERS[arm],
        "HELION_RAG_LOO_SEEDING": "1" if arm in _RAG_ARMS else "0",
        "HELION_RAG_INDEX_DIR": str(fold_index_dir),
        "HELION_SKIP_CACHE": "1",
    }


def completed_keys(path: str | Path) -> set[str]:
    """Read successful resume keys, tolerating failed or truncated rows."""
    path = Path(path)
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ok") and row.get("resume_key"):
            completed.add(row["resume_key"])
    return completed


def pending_matrix(matrix: Iterable[Cell], results_path: str | Path) -> list[Cell]:
    """Return cells that do not yet have a successful result."""
    done = completed_keys(results_path)
    return [item for item in matrix if item["resume_key"] not in done]


def _resume_key(item: Cell) -> str:
    payload = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_matrix(
    workloads: Iterable[dict],
    *,
    repetitions: int,
    corpus_fingerprint: str,
    phase: str,
    code_revision: str = "working-tree",
    experiment_spec: RetrievalSpec | None = None,
    arms: tuple[str, ...] = WORKLOAD_ARMS,
) -> list[Cell]:
    """Create a deterministic experiment matrix with stable resume keys.

    ``phase`` is part of the resume key, so preflight and eval cells for the
    same (workload, arm, rep) never collide.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    ordered = sorted(
        workloads,
        key=lambda record: (
            str(record.get("kernel_name")),
            str(record.get("workload_key")),
        ),
    )
    matrix: list[Cell] = []
    for rep in range(repetitions):
        for workload in ordered:
            for arm in arms:
                item: Cell = {
                    "kernel": workload["kernel_name"],
                    "workload_key": workload["workload_key"],
                    "shapes": workload["input_shapes"],
                    "dtypes": workload["dtypes"],
                    "arm": arm,
                    "rep": rep,
                    "phase": phase,
                    "corpus_fingerprint": corpus_fingerprint,
                    "code_revision": code_revision,
                }
                item.update(experiment_spec or {})
                item["resume_key"] = _resume_key(item)
                matrix.append(item)
    return matrix


def counterbalanced_order(matrix: Iterable[Cell], *, seed: int = 0) -> list[Cell]:
    """Order evaluation cells so each matched pair runs adjacently, AB/BA.

    For every (kernel, workload, rep) the two cells are emitted next to each
    other; a seeded coin flip alternates which runs first, de-confounding
    thermal/service drift from the paired time comparison. Each returned cell
    gains a ``run_index`` recording its global execution position.
    """
    rows = list(matrix)
    by_cell: dict[tuple[str, str, int, str], Cell] = {
        (row["kernel"], row["workload_key"], row["rep"], row["arm"]): row
        for row in rows
    }
    rng = random.Random(seed)
    ordered: list[Cell] = []
    for candidate, baseline in sorted(WORKLOAD_PAIRS.items()):
        triples = sorted(
            {
                (row["kernel"], row["workload_key"], row["rep"])
                for row in rows
                if row["arm"] in (candidate, baseline)
            }
        )
        for kernel, workload, rep in triples:
            base = by_cell.get((kernel, workload, rep, baseline))
            cand = by_cell.get((kernel, workload, rep, candidate))
            if base is None or cand is None:
                continue
            block = [base, cand] if rng.random() < 0.5 else [cand, base]
            ordered.extend(block)
    return [{**cell, "run_index": index} for index, cell in enumerate(ordered)]


def _workload_fold_dir(cfg, family: str, kernel: str, workload_key: str) -> Path:
    from helion_rag.loo import _workload_fold_slug

    return (
        Path(cfg.index_dir)
        / "loo_folds_shape"
        / family
        / _workload_fold_slug(kernel, workload_key)
    )


def pair_eligible_workload_keys(preflight_rows: Iterable[dict], arm: str) -> set[str]:
    """Workload keys whose preflight baseline ``arm`` completed successfully."""
    return {
        row["workload_key"]
        for row in preflight_rows
        if row.get("arm") == arm and row.get("ok")
    }


def select_workload_folds(
    records: Iterable[dict],
    eligible_by_arm: dict[str, set[str]],
    *,
    count: int = 3,
    min_per_kernel: int = 2,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Union-select held-out workloads and derive per-pair kernel inclusion.

    ``eligible_by_arm`` maps each *baseline* arm to its preflight-eligible
    ``workload_key`` set. Selection draws up to ``count`` size-spanning shapes
    per kernel from the **union** of eligibility (one fold built per workload);
    a kernel enters a given pair's analysis only if it has ``min_per_kernel``
    eligible selected workloads for that pair.
    """
    from helion_rag.loo import select_heldout_workloads

    record_list = list(records)
    kernels = sorted({record["kernel_name"] for record in record_list})
    union_eligible: set[str] = (
        set().union(*eligible_by_arm.values()) if eligible_by_arm else set()
    )
    selected: list[dict] = []
    selected_keys: set[str] = set()
    for kernel in kernels:
        eligible = [
            record
            for record in record_list
            if record["kernel_name"] == kernel
            and record["workload_key"] in union_eligible
        ]
        for workload in select_heldout_workloads(eligible, kernel, count=count):
            if workload["workload_key"] not in selected_keys:
                selected_keys.add(workload["workload_key"])
                selected.append(workload)
    pair_kernels: dict[str, list[str]] = {}
    for candidate, baseline in WORKLOAD_PAIRS.items():
        base_eligible = eligible_by_arm.get(baseline, set())
        counts: dict[str, int] = {}
        for workload in selected:
            if workload["workload_key"] in base_eligible:
                counts[workload["kernel_name"]] = (
                    counts.get(workload["kernel_name"], 0) + 1
                )
        pair_kernels[candidate] = sorted(
            kernel for kernel, n in counts.items() if n >= min_per_kernel
        )
    return selected, pair_kernels


def prepare_workload_folds(
    cfg,
    family: str,
    records: list[dict],
    heldout_workloads: Iterable[dict],
    *,
    retrieval: dict,
) -> None:
    from helion_rag.loo import prepare_workload_fold

    for workload in heldout_workloads:
        key = workload["workload_key"]
        print(f"[loo] preparing workload fold excluding {key}", file=sys.stderr)
        prepare_workload_fold(cfg, family, key, records, retrieval=retrieval)


def build_oracle_matrix(
    workloads: Iterable[dict],
    *,
    corpus_fingerprint: str,
    code_revision: str,
) -> list[Cell]:
    """Create one drift-resistant top-five rebenchmark task per workload."""
    matrix: list[Cell] = []
    for workload in sorted(workloads, key=lambda record: record["workload_key"]):
        item: Cell = {
            "kernel": workload["kernel_name"],
            "workload_key": workload["workload_key"],
            "shapes": workload["input_shapes"],
            "dtypes": workload["dtypes"],
            "arm": "oracle",
            "rep": 0,
            "corpus_fingerprint": corpus_fingerprint,
            "code_revision": code_revision,
            "oracle_configs": [
                entry.get("config")
                for entry in (workload.get("top_n") or [])[:5]
                if entry.get("config")
            ],
        }
        item["resume_key"] = _resume_key(item)
        matrix.append(item)
    return matrix


def _git_revision(repo: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status:
        return revision
    digest = hashlib.sha256()
    digest.update(status.encode())
    digest.update(
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    )
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        path = repo / line[3:]
        if path.is_file():
            digest.update(line[3:].encode())
            digest.update(path.read_bytes())
    return f"{revision}-dirty-{digest.hexdigest()[:12]}"


def _retrieval_env(manifest: dict) -> dict[str, str]:
    """RAG embed/retrieval env derived from a fold manifest (empty if none)."""
    if not manifest or not manifest.get("retrieval"):
        return {}
    retrieval = manifest["retrieval"]
    return {
        "HELION_RAG_EMBED_TEXT": manifest["embed_text"],
        "HELION_EMBED_MODEL": manifest["embed_model"],
        "HELION_RAG_SIM_THRESHOLD": str(retrieval["sim_threshold"]),
        "HELION_RAG_K": str(retrieval["k"]),
        "HELION_RAG_TOP_N": str(retrieval["top_n"]),
        "HELION_RAG_DISTINCT_KERNELS": "1" if retrieval["distinct_kernels"] else "0",
        "HELION_RAG_SHAPE_RERANK": "1" if retrieval["shape_rerank"] else "0",
    }


def run_matrix(
    matrix: list[Cell],
    *,
    results_path: Path,
    timeout_s: int,
    fold_dir_fn,
    manifest_reader,
) -> int:
    """Run pending cells as fresh subprocesses.

    A ``manifest_reader`` returning ``{}`` yields no retrieval env, which is
    correct for baseline and oracle preflight cells.
    """
    pending = pending_matrix(matrix, results_path)
    print(
        f"[loo] {len(pending)}/{len(matrix)} cells pending -> {results_path}",
        file=sys.stderr,
    )
    failures = 0
    for index, item in enumerate(pending, start=1):
        fold_dir = fold_dir_fn(item)
        manifest = manifest_reader(fold_dir, item)
        command = [
            sys.executable,
            "-m",
            "helion_rag.loo_run",
            "--item-json",
            json.dumps(item, separators=(",", ":"), default=str),
            "--fold-index-dir",
            str(fold_dir),
            "--out",
            str(results_path),
        ]
        print(
            f"[loo] {index}/{len(pending)} {item['kernel']} "
            f"{item['arm']} rep={item['rep']}",
            file=sys.stderr,
        )
        try:
            child_env = {**os.environ, **_retrieval_env(manifest)}
            result = subprocess.run(command, timeout=timeout_s, env=child_env)
        except subprocess.TimeoutExpired:
            failures += 1
            # Record an explicit timeout row so the result manifest has no
            # silent gap for a cell that started but never finished.
            _append_row(results_path, {**item, "ok": False, "status": "timeout"})
            print(f"[loo] timeout after {timeout_s}s", file=sys.stderr)
            continue
        if result.returncode != 0:
            failures += 1
    return failures


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _write_matrix(path: Path, matrix: list[Cell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, default=str) + "\n" for item in matrix
        ),
        encoding="utf-8",
    )


_ARM_PAIR = {candidate: candidate for candidate in WORKLOAD_PAIRS}
_ARM_PAIR.update(
    {baseline: candidate for candidate, baseline in WORKLOAD_PAIRS.items()}
)


def _load_result_rows(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _workload_run_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "preflight_matrix": output_dir / "workload_preflight_matrix.jsonl",
        "preflight": output_dir / "workload_preflight.jsonl",
        "oracle_matrix": output_dir / "workload_oracle_matrix.jsonl",
        "oracles": output_dir / "workload_oracles.jsonl",
        "selection": output_dir / "workload_selection.json",
        "eval_matrix": output_dir / "workload_eval_matrix.jsonl",
        "results": output_dir / "workload_results.jsonl",
    }


def _read_workload_fold_manifest(fold_dir: Path, workload_key: str) -> dict:
    manifest_path = Path(fold_dir) / "fold.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing workload fold for {workload_key}: run --phase select"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("excluded_workload_key") != workload_key:
        raise ValueError(f"fold {fold_dir} excludes the wrong workload")
    return manifest


def _run_workload_regime(args, cfg, repo: Path) -> int:
    """Drive the study: preflight -> select -> eval."""
    from helion_rag.corpus import load_corpus
    from helion_rag.loo import corpus_fingerprint
    from helion_rag.loo import select_heldout_workloads
    from helion_rag.loo_inputs import SUPPORTED_KERNELS

    records = load_corpus(cfg.corpus_dir)
    records += load_corpus(cfg.writeback_dir, required=False)
    records = [record for record in records if record.get("family") == args.family]
    if not records:
        raise ValueError(f"no {args.family!r} corpus records found")
    # Retrieval draws from the FULL corpus; only *evaluation targets* are
    # restricted to runnable kernels (loo_inputs) and the --kernels subset.
    corpus_kernels = {record.get("kernel_name") for record in records}
    target_kernels = set(SUPPORTED_KERNELS) & corpus_kernels
    if args.kernels:
        unknown = set(args.kernels) - set(SUPPORTED_KERNELS)
        if unknown:
            raise ValueError(f"unknown/unrunnable kernels: {sorted(unknown)}")
        target_kernels &= set(args.kernels)
    target_kernels = sorted(target_kernels)
    if not target_kernels:
        raise ValueError("no runnable target kernels selected")
    # A pinned --code-revision keeps resume keys stable when the shared working
    # tree is churned by unrelated concurrent activity (the default git dirty
    # fingerprint would otherwise change and force a full redo on resume).
    code_revision = args.code_revision or _git_revision(repo)
    fingerprint = corpus_fingerprint(records)
    retrieval = {
        "sim_threshold": args.similarity_threshold,
        "k": args.neighbors,
        "top_n": args.configs_per_neighbor,
        "shape_rerank": args.shape_rerank,
        "distinct_kernels": False,
    }
    spec: RetrievalSpec = {
        "embed_model": args.embed_model,
        "embed_text": args.embed_text,
        "retrieval": retrieval,
    }
    repetitions = args.repetitions or 3
    baseline_arms = tuple(sorted(set(WORKLOAD_PAIRS.values())))
    output_dir = args.output_dir or (Path(cfg.index_dir).parent / "loo_evaluation")
    paths = _workload_run_paths(output_dir)

    def _fold_free(item):
        return cfg.index_dir

    def _no_manifest(fold_dir, item):
        return {}

    if args.phase == "preflight":
        pool: list[dict] = []
        for kernel in target_kernels:
            pool.extend(
                select_heldout_workloads(records, kernel, count=args.candidate_pool)
            )
        matrix = build_matrix(
            pool,
            repetitions=repetitions,
            corpus_fingerprint=fingerprint,
            code_revision=code_revision,
            experiment_spec=spec,
            arms=baseline_arms,
            phase="preflight",
        )
        oracles = build_oracle_matrix(
            pool, corpus_fingerprint=fingerprint, code_revision=code_revision
        )
        _write_matrix(paths["preflight_matrix"], matrix)
        _write_matrix(paths["oracle_matrix"], oracles)
        print(
            f"[loo] workload preflight: {len(pool)} candidate workloads, "
            f"{len(matrix)} baseline cells, {len(oracles)} oracle cells",
            file=sys.stderr,
        )
        if args.dry_run:
            return 0
        failures = run_matrix(
            oracles,
            results_path=paths["oracles"],
            timeout_s=args.timeout_s,
            fold_dir_fn=_fold_free,
            manifest_reader=_no_manifest,
        )
        failures += run_matrix(
            matrix,
            results_path=paths["preflight"],
            timeout_s=args.timeout_s,
            fold_dir_fn=_fold_free,
            manifest_reader=_no_manifest,
        )
        return 1 if failures else 0

    if args.phase == "select":
        preflight_rows = _load_result_rows(paths["preflight"])
        eligible = {
            arm: pair_eligible_workload_keys(preflight_rows, arm)
            for arm in baseline_arms
        }
        selected, pair_kernels = select_workload_folds(
            records, eligible, count=args.count
        )
        prepare_workload_folds(cfg, args.family, records, selected, retrieval=retrieval)
        selection = {
            "family": args.family,
            "code_revision": code_revision,
            "corpus_fingerprint": fingerprint,
            "retrieval": retrieval,
            "spec": spec,
            "repetitions": repetitions,
            "ab_seed": args.ab_seed,
            "eligible": {arm: sorted(keys) for arm, keys in eligible.items()},
            "pair_kernels": pair_kernels,
            "selected": [
                {
                    "kernel_name": workload["kernel_name"],
                    "workload_key": workload["workload_key"],
                    "input_shapes": workload["input_shapes"],
                    "dtypes": workload["dtypes"],
                }
                for workload in selected
            ],
        }
        paths["selection"].parent.mkdir(parents=True, exist_ok=True)
        paths["selection"].write_text(
            json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[loo] selected {len(selected)} held-out workloads; "
            f"pair kernels: {pair_kernels}",
            file=sys.stderr,
        )
        return 0

    if args.phase == "eval":
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
        pair_kernels = selection["pair_kernels"]
        matrix = build_matrix(
            selection["selected"],
            repetitions=selection["repetitions"],
            corpus_fingerprint=fingerprint,
            code_revision=code_revision,
            experiment_spec=spec,
            phase="eval",
        )
        matrix = [
            item
            for item in matrix
            if item["kernel"] in pair_kernels.get(_ARM_PAIR[item["arm"]], [])
        ]
        ordered = counterbalanced_order(matrix, seed=selection["ab_seed"])
        _write_matrix(paths["eval_matrix"], ordered)
        print(
            f"[loo] workload eval: {len(ordered)} cells "
            f"({len(ordered) // 2} matched pairs)",
            file=sys.stderr,
        )
        if args.dry_run:
            return 0

        def _fold_dir_fn(item):
            return _workload_fold_dir(
                cfg, args.family, item["kernel"], item["workload_key"]
            )

        def _manifest_reader(fold_dir, item):
            return _read_workload_fold_manifest(fold_dir, item["workload_key"])

        failures = run_matrix(
            ordered,
            results_path=paths["results"],
            timeout_s=args.timeout_s,
            fold_dir_fn=_fold_dir_fn,
            manifest_reader=_manifest_reader,
        )
        return 1 if failures else 0

    raise ValueError(f"unknown workload phase: {args.phase!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leave-one-workload-out RAG-seeded LFBO evaluation"
    )
    parser.add_argument(
        "--phase", choices=("preflight", "select", "eval"), required=True
    )
    parser.add_argument("--env-file")
    parser.add_argument("--family", default="h100")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--embed-text",
        choices=("cleaned", "comprehensive", "minimalist"),
        default="minimalist",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--configs-per-neighbor", type=int, default=3)
    parser.add_argument(
        "--shape-rerank", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-pool", type=int, default=4)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--ab-seed", type=int, default=0)
    parser.add_argument(
        "--code-revision",
        help="pin the resume-key code revision (stable across unrelated tree churn)",
    )
    parser.add_argument(
        "--kernels",
        help="comma-separated kernel subset to restrict the study to",
    )
    args = parser.parse_args(argv)
    args.kernels = (
        tuple(k for k in args.kernels.split(",") if k) if args.kernels else None
    )

    from helion_rag.cli import _load_env_file
    from helion_rag.config import _config

    if args.env_file:
        _load_env_file(args.env_file)
    cfg = replace(_config(), embed_model=args.embed_model, embed_text=args.embed_text)
    repo = Path(__file__).resolve().parents[3]
    return _run_workload_regime(args, cfg, repo)


if __name__ == "__main__":
    raise SystemExit(main())
