"""Summarize a better-benchmark campaign: aggregates, families and kernels.

Reads ``<run>/bench/*.json`` (and the autotune records under
``<run>/configs``) and writes ``<run>/report.md`` and ``<run>/report.json``.
Speedups are ``baseline_us / helion_cute_us`` (>1 means Helion+CuTe is
faster); ``inductor`` is the better of Inductor's default and
coordinate-descent compiles, as better-benchmark reports it. An
implementation is left out of every comparison and aggregate when it failed
its run or the anchored numerics gate (``FAIL`` in the numerics column),
or, for a Helion port, when its pinned kernels no longer generate the code
the autotune measured (``stale``: ``code_matches_autotune`` is false, so the
timing is not of the tuned kernels).

The aggregates count only records whose foreign GPU activity was sampled
and clean (``common.activity_issue``): a bench record during which a
process outside the job computed on its GPU, or one whose activity samples
do not cover its job, leaves its kernel out of them, and such an autotune
record leaves out its port (``--include-unsampled`` counts the records made
before the sampling existed, which have no samples at all; never one whose
samples miss its job). The report lists these records. A port's timing counts
only while its point's autotune record holds the kernels ``bench.py``
pinned (``autotune_code_sha256``; records made before name their
decorators only): a port timed on a provisional config (an earlier run's,
standing in until the port's own autotune has run: ``plan_run.py
--provisional-triton``), or on a record that an autotune has replaced since
or removes while it runs, shows its numbers, even when that config now
generates other code, but stays out of the aggregates too. For each timer
it gives

- aggregates over every counted kernel and without the ``--without``
  kernels (by default T5's relative-position-bias backward, the selection's
  longest kernel under Inductor, which dominates any time-weighted total):
  the geomean speedup of Helion+CuTe over Inductor; the time-weighted speedup,
  total Inductor time over total Helion+CuTe time; the model-weighted one,
  each kernel weighted by its ``model_share`` in ``--selection`` (the
  speedup of the selected kernels' time with every model weighing the
  same); the geomean fraction of the speed of light (memcopy time over
  kernel time, ``%SOL``) of Inductor and Helion+CuTe on those kernels;
  Helion-Triton's geomean speedup over Inductor and Helion+CuTe's over
  Helion-Triton and over the oracle, each on the kernels where both count;
  and, with ``--baseline``, the earlier run's geomean on the same kernels
  (those its own report would count, records without activity samples
  included: early runs have none);
- the same per family;
- a row per kernel, next to the ``--baseline`` run's Helion+CuTe speedup.

It also lists the records whose numerics gate fell back from the float64
reference to the eager one (``common.AnchoredGate``'s ``ref_fallback``).

``--rebalance`` recomputes every ``cold_us`` and ``bb_us``, the baseline
run's included, from the recorded rounds by ``bench.py``'s current rule, the
mean of the forward-order and reverse-order values (``rebalance``). Records
made before that rule took the median of 5 cold rounds, a forward-order
reading, and the minimum over all bb rounds.

Example::

    python benchmarks/cute/better_benchmark/report.py artifacts/bb-new \\
        --baseline artifacts/bb-old
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common

IMPLS = ("inductor", "oracle", "helion_triton", "helion_cute", "memcopy")
PORTS = {"helion_cute": "cute", "helion_triton": "triton"}
# T5's relative-position-bias backward, the selection's longest kernel under
# Inductor by far: it dominates any time-weighted total.
OUTLIERS = ["sum_sum_sum_d2c97d17f3dc"]


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fmt(value: float | None, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def usable(info: dict[str, Any], stale_ok: bool = False) -> bool:
    """Whether an implementation's timing enters the comparisons (with
    ``stale_ok``, whether it shows: a port timed on a provisional config
    shows whatever code that config generates now)."""
    return (
        info.get("status") == "ok"
        and info.get("numerics_pass") is not False
        and (stale_ok or info.get("code_matches_autotune") is not False)
    )


def verdict(info: dict[str, Any]) -> str:
    """The numerics column's entry for one implementation."""
    if info.get("numerics_pass") is False:
        return "FAIL"
    if info.get("code_matches_autotune") is False:
        return "stale"
    return {True: "ok", None: "?"}[info.get("numerics_pass")]


def impl_time(
    record: dict[str, Any], impl: str, metric: str, stale_ok: bool = False
) -> float | None:
    """An implementation's usable timing: finite and positive (a median of
    flush-subtracted samples none of which stayed positive is infinite)."""
    info = record["impls"].get(impl) or {}
    if not usable(info, stale_ok):
        return None
    value = info.get(metric)
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return value
    return None


def rebalance(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` with each implementation's ``cold_us`` and ``bb_us``
    recomputed from its recorded rounds (``cold_runs_us``, ``bb_runs_us``)
    the way ``bench.py`` computes them now: the median, for bb the minimum,
    over each order's rounds, averaged (``common.order_balanced``; with 5
    rounds the orders are forward, reverse, forward, reverse, forward). The
    per-order values go to ``cold_order_medians_us`` and
    ``bb_order_mins_us``, as ``bench.py`` records them."""
    for info in record["impls"].values():
        for metric, rounds, statistic, per_order in (
            ("cold_us", "cold_runs_us", statistics.median, "cold_order_medians_us"),
            ("bb_us", "bb_runs_us", min, "bb_order_mins_us"),
        ):
            if info.get(rounds):
                info[metric], info[per_order] = common.order_balanced(
                    info[rounds], statistic
                )
    return record


def timed_record_issue(
    info: dict[str, Any], config: dict[str, Any] | None
) -> str | None:
    """Why a port's timing (``info``, its entry in a bench record) is not of
    its point's current autotune record (``config``; None when the point has
    no ok one), or None when it is. ``bench.py`` records the code digests of
    the record it pinned (records made before name its decorators only); an
    autotune that runs after the bench (phase 2 of a run on provisional
    configs) moves that record aside while it runs and replaces it if it
    succeeds."""
    kernels = (config or {}).get("kernels", {})
    if "autotune_code_sha256" in info:
        same = info["autotune_code_sha256"] == {
            name: k["code_sha256"] for name, k in kernels.items()
        }
    else:
        same = info.get("decorators") == {
            name: k["decorator"] for name, k in kernels.items()
        }
    if config is not None and same and not config.get("provisional"):
        return None
    if info.get("autotune_provisional"):
        return f"provisional: {info.get('autotune_reused_from')}"
    if config is None:
        return "missing: no ok autotune record"
    if config.get("provisional"):
        return f"provisional: {config.get('reused_from')}"
    return "replaced: autotuned again after the bench"


def summarize(
    record: dict[str, Any],
    metric: str,
    autotunes: dict[str, dict[str, Any]] | None = None,
    include_unsampled: bool = False,
) -> dict[str, Any]:
    """One kernel's row: its timings, speedups and %SOL, and which timings
    the aggregates count given the activity of its bench record and of its
    ports' autotune records (``autotunes``, by backend), none of a port timed
    on another record than its point's current one (``timed_record_issue``)."""
    tuned = autotunes or {}
    issues: dict[str, str] = {}
    if issue := common.activity_issue(record, include_unsampled):
        issues["bench"] = issue
    # A port timed on another record shows whatever code that record's
    # configs generate now.
    elsewhere = set()
    for impl, backend in PORTS.items():
        info = record["impls"].get(impl) or {}
        if info.get("status") != "ok":
            continue
        config = tuned.get(backend)
        if issue := timed_record_issue(info, config):
            issues[f"{backend} autotune"] = issue
            elsewhere.add(impl)
        elif config is not None and (
            issue := common.activity_issue(config, include_unsampled)
        ):
            issues[f"{backend} autotune"] = issue
    ind = [impl_time(record, i, metric) for i in ("inductor", "inductor_cd")]
    ind = [t for t in ind if t is not None]
    row: dict[str, Any] = {
        "repro": record["repro"],
        "shape": record["shape"],
        "family": common.family(record["repro"]),
        "total_bytes": record.get("total_bytes"),
        "inductor": min(ind) if ind else None,
        "oracle": impl_time(record, "oracle", metric),
        **{impl: impl_time(record, impl, metric, impl in elsewhere) for impl in PORTS},
        "memcopy": impl_time(record, "memcopy", metric),
    }
    cute = row["helion_cute"]
    for base in ("inductor", "oracle", "helion_triton"):
        b = row[base]
        row[f"cute_vs_{base}"] = b / cute if b and cute else None
    triton = row["helion_triton"]
    row["triton_vs_inductor"] = (
        row["inductor"] / triton if row["inductor"] and triton else None
    )
    if cute and row["total_bytes"]:
        row["cute_gbps"] = row["total_bytes"] / cute / 1e3
    if row["inductor"] and row["total_bytes"]:
        row["inductor_gbps"] = row["total_bytes"] / row["inductor"] / 1e3
    if cute and row["memcopy"]:
        row["cute_of_sol"] = row["memcopy"] / cute
    if row["inductor"] and row["memcopy"]:
        row["inductor_of_sol"] = row["memcopy"] / row["inductor"]
    row["numerics"] = {
        impl: verdict(record["impls"].get(impl) or {})
        for impl in (
            "inductor",
            "inductor_cd",
            "oracle",
            "helion_triton",
            "helion_cute",
        )
    }
    kernels = {}
    for impl in ("inductor", "inductor_cd", "helion_cute", "helion_triton"):
        info = record["impls"].get(impl) or {}
        kernels[impl] = [k["name"] for k in info.get("kernels", [])]
    row["kernels"] = kernels
    row["foreign_gpu_processes"] = sorted(
        set(record.get("foreign_gpu_processes_before") or [])
        | set(record.get("foreign_gpu_processes_after") or [])
    )
    row["foreign_activity"] = common.foreign_activity(record)
    row["issues"] = issues

    def counted(impl: str) -> bool:
        if row[impl] is None or "bench" in issues:
            return False
        return impl not in PORTS or f"{PORTS[impl]} autotune" not in issues

    row["counted"] = [impl for impl in IMPLS if counted(impl)]
    return row


def ratio(row: dict[str, Any], impl: str, base: str) -> float | None:
    """``base`` time over ``impl`` time, when the aggregates count both."""
    if impl in row["counted"] and base in row["counted"]:
        return row[base] / row[impl]
    return None


def aggregate(
    rows: list[dict[str, Any]],
    weights: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
) -> dict[str, Any]:
    """The aggregates over ``rows`` (see the module docstring); ``weights``
    and ``baseline`` map (repro, shape) to the model share and to the
    baseline run's Helion+CuTe speedup over Inductor."""
    timed = [
        (row, value)
        for row in rows
        if (value := ratio(row, "helion_cute", "inductor")) is not None
    ]
    speedups = [value for _, value in timed]
    out: dict[str, Any] = {
        "kernels": len(timed),
        "at_or_above_parity": sum(1 for value in speedups if value >= 1.0),
        "cute_vs_inductor": geomean(speedups),
        "time_weighted": (
            sum(row["inductor"] for row, _ in timed)
            / sum(row["helion_cute"] for row, _ in timed)
            if timed
            else None
        ),
    }
    shares = [
        (weights[row["repro"], row["shape"]], value)
        for row, value in timed
        if (row["repro"], row["shape"]) in weights
    ]
    out["model_weighted"] = (
        sum(share for share, _ in shares)
        / sum(share / value for share, value in shares)
        if shares
        else None
    )
    out["model_weighted_kernels"] = len(shares)
    sol = [row for row, _ in timed if "memcopy" in row["counted"]]
    out["inductor_of_sol"] = geomean([r["memcopy"] / r["inductor"] for r in sol])
    out["cute_of_sol"] = geomean([r["memcopy"] / r["helion_cute"] for r in sol])
    out["sol_kernels"] = len(sol)
    for key, impl, base in (
        ("triton_vs_inductor", "helion_triton", "inductor"),
        ("cute_vs_triton", "helion_cute", "helion_triton"),
        ("cute_vs_oracle", "helion_cute", "oracle"),
    ):
        values = [v for row in rows if (v := ratio(row, impl, base)) is not None]
        out[key] = geomean(values)
        out[f"{key}_kernels"] = len(values)
    same = [
        (baseline[row["repro"], row["shape"]], value)
        for row, value in timed
        if (row["repro"], row["shape"]) in baseline
    ]
    out["baseline"] = {
        "kernels": len(same),
        "baseline": geomean([before for before, _ in same]),
        "this_run": geomean([after for _, after in same]),
    }
    return out


def ok_records(run_dir: Path, kind: str) -> dict[Path, dict[str, Any]]:
    """The ok records of a run's ``bench`` or ``configs`` directory."""
    found = {}
    for path in sorted((run_dir / kind).glob("*__*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "ok":
            found[path] = record
    return found


def by_point(autotunes: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Autotune records by (repro, shape), then by backend."""
    tuned: dict[tuple[str, str], dict[str, Any]] = collections.defaultdict(dict)
    for rec in autotunes:
        tuned[rec["repro"], rec["shape"]][rec["backend"]] = rec
    return tuned


def baseline_speedups(
    run_dir: Path, metric: str, rebalanced: bool = False
) -> dict[tuple[str, str], float]:
    """An earlier run's Helion+CuTe speedups over Inductor by (repro, shape),
    for the kernels its own aggregates would count (``summarize`` with its
    autotune records), except that records without activity samples count:
    early runs have none. ``rebalanced`` recomputes its timings
    (``rebalance``)."""
    tuned = by_point(list(ok_records(run_dir, "configs").values()))
    out = {}
    for record in ok_records(run_dir, "bench").values():
        if rebalanced:
            record = rebalance(record)
        point = tuned[record["repro"], record["shape"]]
        row = summarize(record, metric, point, include_unsampled=True)
        if (speedup := ratio(row, "helion_cute", "inductor")) is not None:
            out[row["repro"], row["shape"]] = speedup
    return out


def aggregate_lines(
    columns: dict[str, dict[str, Any]], baseline_name: str | None
) -> list[str]:
    """The aggregates table, a column per kernel set."""

    def times(value: float | None, n: int | None = None) -> str:
        text = "-" if value is None else f"{value:.3f}x"
        return text if n is None else f"{text} ({n})"

    def percent(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.0f}%"

    entries: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("kernels (Helion+CuTe and Inductor counted)", lambda a: str(a["kernels"])),
        ("Helion+CuTe vs Inductor, geomean", lambda a: times(a["cute_vs_inductor"])),
        ("at or above parity", lambda a: str(a["at_or_above_parity"])),
        (
            "Helion+CuTe vs Inductor, time-weighted (total Inductor / total CuTe)",
            lambda a: times(a["time_weighted"]),
        ),
        (
            "Helion+CuTe vs Inductor, model-weighted (kernels with a model share)",
            lambda a: times(a["model_weighted"], a["model_weighted_kernels"]),
        ),
        (
            "Inductor %SOL, geomean (kernels with a memcopy)",
            lambda a: f"{percent(a['inductor_of_sol'])} ({a['sol_kernels']})",
        ),
        ("Helion+CuTe %SOL, geomean", lambda a: percent(a["cute_of_sol"])),
        (
            "Helion-Triton vs Inductor, geomean (kernels)",
            lambda a: times(a["triton_vs_inductor"], a["triton_vs_inductor_kernels"]),
        ),
        (
            "Helion+CuTe vs Helion-Triton, geomean (kernels)",
            lambda a: times(a["cute_vs_triton"], a["cute_vs_triton_kernels"]),
        ),
        (
            "Helion+CuTe vs oracle, geomean (kernels)",
            lambda a: times(a["cute_vs_oracle"], a["cute_vs_oracle_kernels"]),
        ),
    ]
    if baseline_name is not None:
        entries.append(
            (
                f"Helion+CuTe vs Inductor, {baseline_name} -> this run, same kernels",
                lambda a: (
                    f"{times(a['baseline']['baseline'])} -> "
                    f"{times(a['baseline']['this_run'], a['baseline']['kernels'])}"
                ),
            )
        )
    lines = [
        f"| | {' | '.join(columns)} |",
        f"|---|{'---|' * len(columns)}",
    ]
    for label, cell in entries:
        lines.append(f"| {label} | {' | '.join(cell(a) for a in columns.values())} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metric", default="cold_us", choices=("cold_us", "bb_us"))
    parser.add_argument(
        "--goal",
        type=Path,
        default=None,
        help="write a cute-hillclimb goal file (ratio = inductor / helion_cute "
        "in --metric, one variant per bench record; no ratio where the "
        "aggregates do not count both)",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=common.SELECTION_PATH,
        help="the selection whose model shares weigh the model-weighted speedup",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="an earlier run directory to compare each kernel's speedup with",
    )
    parser.add_argument(
        "--without",
        nargs="*",
        default=OUTLIERS,
        help="repro ids the second aggregates column leaves out",
    )
    parser.add_argument(
        "--include-unsampled",
        action="store_true",
        help="count records without activity samples in the aggregates",
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="recompute cold_us and bb_us from the recorded rounds by bench.py's "
        "order-balanced rule, for runs recorded before it",
    )
    args = parser.parse_args()
    records = []
    paths = {}
    for path, record in ok_records(args.run_dir, "bench").items():
        records.append(rebalance(record) if args.rebalance else record)
        paths[(record["repro"], record["shape"])] = path
    autotunes = list(ok_records(args.run_dir, "configs").values())
    tuned = by_point(autotunes)
    autotune_s: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for rec in autotunes:
        if not rec.get("provisional"):
            point = f"{rec['repro']}__{rec['shape']}"
            autotune_s[point][rec["backend"]] = rec["autotune_s"]
    rows = {
        metric: [
            summarize(r, metric, tuned[r["repro"], r["shape"]], args.include_unsampled)
            for r in records
        ]
        for metric in ("cold_us", "bb_us")
    }
    weights = {}
    if args.selection.exists():
        weights = {
            (k["repro"], k["shape"]): k["model_share"]
            for k in json.loads(args.selection.read_text())["kernels"]
            if k.get("model_share")
        }
    contaminated: dict[str, dict[str, list[str]]] = {"bench": {}, "autotune": {}}
    unsampled: dict[str, list[str]] = {"bench": [], "autotune": []}
    uncovered: dict[str, dict[str, str]] = {"bench": {}, "autotune": {}}
    reused: list[str] = []
    provisional: list[str] = []
    for kind, recs in (("bench", records), ("autotune", autotunes)):
        for rec in recs:
            found = common.foreign_activity(rec)
            name = f"{rec['repro']}__{rec['shape']}"
            if kind == "autotune":
                name += f"__{rec['backend']}"
            if rec.get("provisional"):
                provisional.append(name)
                continue
            if rec.get("reused_from"):
                reused.append(name)
            if rec.get("contaminated"):
                attempts = len(rec.get("contaminated_attempts", []))
                found = [*(found or []), f"the cleanest of {attempts} attempts"]
            if found:
                contaminated[kind][name] = found
            elif found is None and rec.get("foreign_gpu_activity") is None:
                unsampled[kind].append(name)
            elif found is None:
                uncovered[kind][name] = common.activity_coverage_issue(rec) or ""
    # The gate's reference is the eager module when its float64 run failed
    # (common.AnchoredGate): ops without float64 kernels, or anything else.
    eager_reference = {
        f"{rec['repro']}__{rec['shape']}": anchored.get(
            "ref_fallback", "no reason recorded"
        )
        for rec in records
        for info in rec["impls"].values()
        if (anchored := info.get("anchored") or {}).get("ref_precision") == "eager"
    }

    lines = [f"# better-benchmark campaign {args.run_dir.name}", ""]
    if records:
        gpu = records[0].get("gpu", {})
        ver = records[0].get("versions", {})
        lines += [
            (
                f"GPU: {gpu.get('name')} ({gpu.get('power_limit_w')} W cap); "
                f"torch {ver.get('torch')}, triton {ver.get('triton')}, "
                f"cutlass-dsl {ver.get('nvidia-cutlass-dsl')}, "
                f"helion {ver.get('helion_git')}"
            ),
            "",
        ]
    if reused:
        lines += [
            f"Autotune records reused from earlier runs (plan_run.py): {len(reused)}.",
            "",
        ]
    if provisional:
        lines += [
            (
                f"Provisional autotune records: {len(provisional)}. Their ports were "
                "timed on an earlier run's configs ahead of their own autotunes "
                "(campaign.py --backends cute); their numbers show, flagged, and "
                "stay out of the aggregates."
            ),
            "",
        ]
    baseline_name = None if args.baseline is None else args.baseline.name
    aggregates: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in ("cold_us", "bb_us"):
        table = rows[metric]
        baseline = (
            {}
            if args.baseline is None
            else baseline_speedups(args.baseline, metric, args.rebalance)
        )
        without = [r for r in table if r["repro"] not in args.without]
        columns = {"all kernels": aggregate(table, weights, baseline)}
        if len(without) < len(table):
            columns[f"without {', '.join(args.without)}"] = aggregate(
                without, weights, baseline
            )
        aggregates[metric] = columns
        members: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in table:
            members[r["family"]].append(r)
        families[metric] = {
            name: aggregate(group, weights, baseline) for name, group in members.items()
        }
        lines += [f"## {metric}", "", *aggregate_lines(columns, baseline_name), ""]
        lines += [
            (
                "| family | kernels | CuTe/ind geomean | CuTe/ind time-weighted"
                " | ind %SOL | CuTe %SOL | Triton/ind geomean (kernels) |"
            ),
            "|---|---|---|---|---|---|---|",
        ]
        for name, agg in sorted(
            families[metric].items(), key=lambda kv: (-kv[1]["kernels"], kv[0])
        ):
            lines.append(
                f"| {name} | {agg['kernels']} | {fmt(agg['cute_vs_inductor'], '.3f')}"
                f" | {fmt(agg['time_weighted'], '.3f')}"
                f" | {fmt(100 * agg['inductor_of_sol'] if agg['inductor_of_sol'] else None, '.0f')}"
                f" | {fmt(100 * agg['cute_of_sol'] if agg['cute_of_sol'] else None, '.0f')}"
                f" | {fmt(agg['triton_vs_inductor'], '.3f')} ({agg['triton_vs_inductor_kernels']}) |"
            )
        baseline_column = f" | {baseline_name} cute/ind" if baseline_name else ""
        lines += [
            "",
            (
                "| repro | shape | MB | inductor | helion-cute | cute/ind"
                f"{baseline_column} | helion-triton | triton/ind | oracle | memcopy"
                " | ind %SOL | cute %SOL | numerics | left out of the aggregates |"
            ),
            f"|{'---|' * (15 if baseline_name else 14)}",
        ]
        for r in sorted(table, key=lambda r: r["cute_vs_inductor"] or 0):
            num = ",".join(f"{k.split('_')[-1]}={v}" for k, v in r["numerics"].items())
            before = (
                f" | {fmt(baseline.get((r['repro'], r['shape'])), '.3f')}"
                if baseline_name
                else ""
            )
            left_out = "; ".join(
                f"{kind} {issue.split(':')[0]}" for kind, issue in r["issues"].items()
            )
            lines.append(
                f"| {r['repro']} | {r['shape']} | {fmt((r['total_bytes'] or 0) / 1e6, '.1f')}"
                f" | {fmt(r['inductor'])} | {fmt(r['helion_cute'])}"
                f" | {fmt(r['cute_vs_inductor'], '.3f')}{before}"
                f" | {fmt(r['helion_triton'])} | {fmt(r['triton_vs_inductor'], '.3f')}"
                f" | {fmt(r['oracle'])} | {fmt(r['memcopy'])}"
                f" | {fmt(100 * r['inductor_of_sol'] if r.get('inductor_of_sol') else None, '.0f')}"
                f" | {fmt(100 * r['cute_of_sol'] if r.get('cute_of_sol') else None, '.0f')}"
                f" | {num} | {left_out or '-'} |"
            )
        lines.append("")
    counted = "counted" if args.include_unsampled else "left out of the aggregates"
    lines += [
        "## Foreign GPU activity",
        "",
        (
            f"Records during which a process outside the job reached "
            f"{common.CONTAMINATION_SM_PCT}% SM utilization on its GPU, an "
            "autotune record's for at least "
            f"{common.AUTOTUNE_CONTAMINATION_SHARE:.0%} of the job and "
            f"{common.AUTOTUNE_CONTAMINATION_MIN_S:.0f} s (contaminated: left "
            "out of the aggregates; run them again):"
        ),
        "",
    ]
    for kind, found in contaminated.items():
        lines += [
            f"- {kind} {name}: {', '.join(processes)}"
            for name, processes in sorted(found.items())
        ]
    if not any(contaminated.values()):
        lines.append("- none")
    lines += ["", f"Records without activity samples ({counted}):", ""]
    lines += [f"- {kind} {name}" for kind, names in unsampled.items() for name in names]
    if not any(unsampled.values()):
        lines.append("- none")
    lines += [
        "",
        (
            "Records whose activity samples do not cover their jobs (left out "
            "of the aggregates; run them again):"
        ),
        "",
    ]
    lines += [
        f"- {kind} {name}: {why}"
        for kind, found in uncovered.items()
        for name, why in sorted(found.items())
    ]
    if not any(uncovered.values()):
        lines.append("- none")
    lines += [
        "",
        "## Numerics references",
        "",
        (
            "Bench records whose numerics gate compared with the eager module on "
            "the original inputs, its float64 reference having failed:"
        ),
        "",
    ]
    lines += [f"- {name}: {why}" for name, why in sorted(eager_reference.items())]
    if not eager_reference:
        lines.append("- none")
    lines.append("")
    if args.goal is not None:
        shapes = []
        for r in rows[args.metric]:
            artifact = paths[(r["repro"], r["shape"])].resolve()
            with contextlib.suppress(ValueError):
                artifact = artifact.relative_to(Path.cwd().resolve())
            shapes.append(
                {
                    "name": f"{r['repro']}[{r['shape']}]",
                    "ratio": ratio(r, "helion_cute", "inductor"),
                    "artifact": str(artifact),
                }
            )
        goal = {
            "task": "better-benchmark: Helion+CuTe vs Inductor Triton "
            f"({args.metric}, {args.run_dir.name})",
            "status": "running",
            "criteria": {"geomean_min": 1.0},
            "shapes": shapes,
        }
        args.goal.write_text(json.dumps(goal, indent=1) + "\n")
    report = {
        "rows": rows,
        "aggregates": aggregates,
        "families": families,
        "autotune_s": autotune_s,
        "reused": reused,
        "provisional": provisional,
        "contaminated": contaminated,
        "unsampled": unsampled,
        "uncovered": uncovered,
        "eager_reference": eager_reference,
    }
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
