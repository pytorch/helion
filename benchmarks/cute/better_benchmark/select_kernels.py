"""Pick a time-weighted, family-stratified sample of better-benchmark kernels.

Inputs: a better-benchmark ``bench_parallel.py --all-shapes`` sweep (Inductor
time per repro shape point) and the B200 model accounting
(``benchmarks/model_accounting/b200``: each model's fusible kernels with their
shape points and occurrence counts).

Each (repro, shape) point gets a *model share*: for every model that runs it,
``occurrences * inductor_us / model_fusible_us``, summed over models. A share
of 1.0 is a whole model's fusible time, so the shares of all points sum to the
number of models. Points are ranked by share; the selection takes the top
points overall and then fills in the best points of families the top did not
cover, skipping points faster than ``--min-us`` (launch-bound, where every
backend reads alike). ``--extra`` points follow the selection, marked
``extra``: repros ported for other reasons that the study measures too.

The committed ``selection.json`` came from::

    python benchmarks/cute/better_benchmark/select_kernels.py \\
        --sweep artifacts/bb-sweep-2026-11-07/inductor_all_shapes.json \\
        --extra var_mean_42fad1ece813:652894ae var_mean_004f9a9ea198:536b6e86 \\
        sum_sum_643db2887a01:2cd54028 \\
        --output benchmarks/cute/better_benchmark/selection.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common


def load_points(sweep_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    sweep = json.loads(sweep_path.read_text())
    points = {}
    for key, rows in sweep.items():
        if key.startswith("_"):
            continue
        repro = Path(key).parent.name
        for label, row in rows.items():
            if not isinstance(row, dict) or "compiled_us" not in row:
                continue
            times = [row.get("compiled_us"), row.get("coord_descent_us")]
            times = [t for t in times if isinstance(t, (int, float)) and t > 0]
            if not times:
                continue
            shape = label.rsplit("_", 1)[-1]
            points[(repro, shape)] = {
                "repro": repro,
                "shape": shape,
                "family": common.family(repro),
                "inductor_us": min(times),
                "memcopy_sol_us": row.get("memcopy_sol_us"),
                "total_bytes": row.get("total_bytes"),
                "n_kernels": row.get("n_kernels"),
            }
    return points


def select_points(
    ranked: list[dict[str, Any]],
    count: int,
    top_fraction: float,
    max_per_pattern: int,
) -> list[dict[str, Any]]:
    """Up to ``count`` of the ``ranked`` points (best first), at most
    ``max_per_pattern`` per repro: the top ``round(count * top_fraction)``,
    then round-robin over the families (by total share), each round giving
    every family whose pick count is lowest one more of its best remaining
    points."""
    chosen: list[dict[str, Any]] = []
    chosen_keys: set[tuple[str, str]] = set()
    per_pattern: collections.Counter[str] = collections.Counter()

    def available(point: dict[str, Any]) -> bool:
        return (point["repro"], point["shape"]) not in chosen_keys and per_pattern[
            point["repro"]
        ] < max_per_pattern

    def take(point: dict[str, Any], reason: str) -> None:
        per_pattern[point["repro"]] += 1
        chosen_keys.add((point["repro"], point["shape"]))
        chosen.append(dict(point, reason=reason))

    top = round(count * top_fraction)
    for point in ranked:
        if len(chosen) >= top:
            break
        if available(point):
            take(point, "top model share")
    family_share: dict[str, float] = collections.defaultdict(float)
    for point in ranked:
        family_share[point["family"]] += point["model_share"]
    families = sorted(family_share, key=lambda f: -family_share[f])
    while len(chosen) < count:
        # Drop the families with nothing left to take, so the lowest pick
        # count belongs to a family that can still add a point.
        families = [
            f
            for f in families
            if any(p["family"] == f and available(p) for p in ranked)
        ]
        if not families:
            break
        covered = collections.Counter(p["family"] for p in chosen)
        level = min(covered[f] for f in families)
        for fam in families:
            if len(chosen) >= count:
                break
            if covered[fam] == level:
                point = next(p for p in ranked if p["family"] == fam and available(p))
                take(point, f"family {fam}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--top-fraction", type=float, default=0.6)
    parser.add_argument("--min-us", type=float, default=12.0)
    parser.add_argument("--max-per-pattern", type=int, default=1)
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=6,
        help="skip points Inductor compiles to more kernels than this",
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="REPRO:SHAPE",
        help="points to measure besides the selection, appended to it",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The ports read the repros' inputs by position, so the selection pins
    # the checkout's commit; uncommitted repros could not be pinned.
    revision = common.git_revision(common.BB_ROOT)
    if revision is not None and revision.endswith("-dirty"):
        parser.error(
            f"better-benchmark checkout {common.BB_ROOT} has uncommitted "
            "changes; the selection pins its commit"
        )

    points = load_points(args.sweep)
    by_hash: dict[str, str] = {}
    for repro, _ in points:
        if (pattern := common.pattern_hash(repro)) is not None:
            by_hash[pattern] = repro
    accounting = common.bb_path() / "benchmarks" / "model_accounting" / "b200"
    share: dict[tuple[str, str], float] = collections.defaultdict(float)
    models: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(dict)
    unpriced = 0
    for path in sorted(accounting.rglob("*.json")):
        model = json.loads(path.read_text())
        identity = model["identity"]
        uses: list[tuple[tuple[str, str], int, float]] = []
        for pattern, shapes in model["model"]["fusible"].items():
            repro = by_hash.get(pattern)
            for shape, occurrences in shapes.items():
                point = points.get((repro, shape)) if repro is not None else None
                if repro is None or point is None:
                    unpriced += occurrences
                    continue
                uses.append(((repro, shape), occurrences, point["inductor_us"]))
        total = sum(occ * us for _, occ, us in uses)
        if total <= 0:
            continue
        for key, occ, us in uses:
            share[key] += occ * us / total
            models[key][identity] = occ

    import oracle_harness  # pyrefly: ignore [missing-import]

    def stochastic(repro: str) -> bool:
        # Repros better-benchmark marks stochastic (aten dropout, bernoulli,
        # rand, ...). Inductor's random prims (inductor_random) are not in that
        # list, so a selected repro can still draw random numbers; the checks
        # compare the outputs that depend on them with Inductor's
        # (common.AnchoredGate).
        return oracle_harness.has_stochastic_ops(common.repro_dir(repro) / "repro.py")

    ranked: list[dict[str, Any]] = sorted(
        (
            {**points[key], "model_share": s, "models": models[key]}
            for key, s in share.items()
            if points[key]["inductor_us"] >= args.min_us
            and (points[key]["n_kernels"] or 0) <= args.max_kernels
            and points[key]["repro"] not in args.exclude
            and not stochastic(points[key]["repro"])
        ),
        key=lambda p: -p["model_share"],
    )
    chosen = select_points(ranked, args.count, args.top_fraction, args.max_per_pattern)
    extras: list[dict[str, Any]] = []
    for spec in args.extra:
        key = tuple(spec.split(":"))
        if key not in points:
            parser.error(f"--extra {spec}: the sweep has no such repro:shape point")
        point = {**points[key], "model_share": share.get(key, 0.0)}
        extras.append({**point, "models": models.get(key, {}), "reason": "extra"})
    total_share = sum(share.values())
    result = {
        "description": __doc__.split("\n\n")[0],
        "sweep": str(args.sweep),
        # The ports read the repros' inputs by position; common.bb_path() warns
        # when the checkout is at another commit.
        "better_benchmark_commit": revision,
        "params": {k: v for k, v in vars(args).items() if k not in ("output", "sweep")},
        "total_model_share": total_share,
        "selected_model_share": sum(p["model_share"] for p in chosen),
        "extra_model_share": sum(p["model_share"] for p in extras),
        "unpriced_occurrences": unpriced,
        "kernels": chosen + extras,
    }
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(
        f"{len(chosen)} kernels, {result['selected_model_share']:.2f} of "
        f"{total_share:.1f} model-shares, and {len(extras)} extras"
    )
    for p in result["kernels"]:
        print(
            f"{p['model_share']:7.3f} {p['inductor_us']:9.1f}us "
            f"{p['repro']:32s} {p['shape']} n_kernels={p['n_kernels']} "
            f"models={len(p['models'])} ({p['reason']})"
        )


if __name__ == "__main__":
    main()
