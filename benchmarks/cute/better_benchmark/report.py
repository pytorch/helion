"""Summarize a better-benchmark campaign: aggregates, families and kernels.

Reads ``<run>/bench/*.json`` (and the autotune records under
``<run>/configs``) and writes ``<run>/report.md`` and ``<run>/report.json``.
Speedups are ``baseline_us / helion_cute_us`` (>1 means Helion+CuTe is
faster); ``inductor`` is the fastest of the Inductor compiles a record
holds (``common.INDUCTOR_IMPLS``: default, coordinate descent and the
options of ``mode="max-autotune"``). A record of
several bench processes (``campaign.py --bench-repeats``: the aggregate
``<point>.json``; its per-process records ``<point>.r<k>.json`` are no rows
of their own) gives each timing as the median over the processes and each
speedup as the median of the per-process ratios, Inductor being the fastest
of its compiles in each process (``summarize``; the README's Baseline
section has the rule and why the ratio of the medians is not it). An
implementation is left out of every comparison and aggregate when it failed
its run or the anchored numerics gate (``FAIL`` in the numerics column),
or, for a Helion port, when its pinned kernels no longer generate the code
the autotune measured (``stale``: ``code_matches_autotune`` is false, so the
timing is not of the tuned kernels). One exception, in a record of several
processes: an Inductor compile that failed in some of them (``FAIL`` in the
column as well) still stands in the baseline of the processes in which it
ran and passed (``impl_times``), and the report lists it.

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

With ``--baseline`` it also says how many of the run's own autotunes
searched from the random seed of the baseline's autotune of the same port
(``campaign.py --seeds-from``; ``seed_pairing``) and names the others: a
cold search's pick follows its seed, so a port's before and after compare
the two trees alone only where both searched from one seed (and, since the
seed draws from the tree's search space, only where the two trees build the
same one for the port, which the seeds do not tell).

It also lists the records whose numerics gate fell back from the float64
reference to the eager one (``common.AnchoredGate``'s ``ref_fallback``), the
config each Triton kernel of Inductor's compiles ended on where the
bench records name them (``inductor_configs``; of a repeated bench, each
distinct set of picks with the number of processes that ended on it), the
Inductor compiles that are not usable in some process of a repeated bench
(that process's baseline is of the other compiles alone), the benches whose
listing timed an Inductor compile right behind a Helion port's batch
(``bench.py``'s ``carryover_risk``: that compile's reading, and the ratio
over it, can be high), and the implementations whose kernels' profiler
spans overlap (dependent launches: their ``kernel_us`` is of the kernels'
steps).

When bench records carry ``rot_us`` (``bench.py --rotate-inputs``: the cold
timer on private input sets, one per call, which hinted loads cannot keep in
L2), the ``cold_us`` section gains two aggregate rows, the geomean speedup
of Helion+CuTe over Inductor by ``rot_us`` over the kernels that have it
counted for both (with their number) and the ``cold_us`` geomean of those
same kernels, and a ``cute/ind rot`` column per kernel; a last section says
which records were timed that way, why the others were not, and that the two
speedups bracket the hint-free one only where the CuTe config has no such
hints (``ROTATION_CAVEAT``). Every other number is computed as without them.

Inductor's time is a minimum of up to three compiles' readings, which reads
low where they are close. Where bench records hold the figure or the rounds
and kernel configs to take it from (``common.fastest_inductor``: the
compiles that ended on the same code pooled, the fastest distinct one
chosen on other rounds than it is timed on), the per-kernel table gains
Helion+CuTe's speedup over that de-biased baseline (``cute/ind de-biased``,
and ``cute/ind rot de-biased`` beside the ``rot_us`` speedup), and under
the ``cold_us`` aggregates a line gives the two geomeans over the kernels
that have both and how many of them lie more than 0.5% apart, for
``cold_us`` and for ``rot_us``. The speedups above, the aggregates and the
``--goal`` file stay over the fastest compile.

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
import hashlib
import json
import math
import operator
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

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
# How to read the rot_us speedup beside the cold_us one, said in the report
# of a run that has it (the measurements are in the README's Timers section).
ROTATION_CAVEAT = (
    "Loads with an `L2::evict_last` hint, which Inductor emits by policy and a "
    "CuTe search can pick (`l2_last`), bias both timers: `cold_us` reads a "
    "hinted kernel partly warm, `rot_us` reads it slower than the same kernel "
    "without its hints. The two speedups bracket the hint-free one only where "
    "the CuTe config has no such hints; where it has, both can lean the same "
    "way and their agreement settles nothing (README, Timers)."
)


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
    return common.ran_and_passed(info) and (
        stale_ok or info.get("code_matches_autotune") is not False
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
    if common.is_timing(value):
        return value
    return None


def tuning_regimes(autotunes: list[dict[str, Any]]) -> dict[str, Any]:
    """How the run's autotune records timed their kernels' candidates
    (``common.kernels_in_sequence``), per backend (the backends differ in
    their default regime, ``common.default_regime``): ``records`` counts a
    backend's records by the regime they name (``common.record_regime``:
    ``alone`` for one made before ``autotune.py`` named any), ``kernels``
    its kernels by how their search was timed (``unrecorded``: the kernels
    of records that say nothing per kernel). ``later_alone`` names
    (``<repro>__<shape>__<backend>.<kernel>``) the kernels of records in
    the port regime that a search timed alone although earlier kernels of
    their port run before them, with the record's reason: the exceptions
    within that regime, not the kernels of a record tuned alone as a
    whole."""
    records: dict[str, dict[str, int]] = {}
    kernels: dict[str, dict[str, int]] = {}
    later_alone: dict[str, str] = {}
    for record in autotunes:
        if record.get("provisional"):
            continue
        backend = record["backend"]
        regime = common.record_regime(record)
        by_regime = records.setdefault(backend, {})
        by_regime[regime] = by_regime.get(regime, 0) + 1
        counts = kernels.setdefault(backend, {"port": 0, "alone": 0, "unrecorded": 0})
        entries = record.get("regimes")
        if entries is None:
            counts["unrecorded"] += len(record.get("kernels", ()))
            continue
        for name, entry in entries.items():
            counts[entry["regime"]] += 1
            if regime == "port" and entry["regime"] == "alone" and entry.get("behind"):
                point = f"{record['repro']}__{record['shape']}__{backend}"
                later_alone[f"{point}.{name}"] = entry.get("reason", "")
    return {"records": records, "kernels": kernels, "later_alone": later_alone}


def tuning_regime_lines(regimes: dict[str, Any]) -> list[str]:
    """The report's section on ``tuning_regimes``; nothing for a run without
    autotune records."""
    if not regimes["records"]:
        return []
    lines = [
        "## Tuning regime",
        "",
        (
            "How the autotunes timed their candidates (`autotune.py --regime`), "
            "per backend:"
        ),
        "",
    ]
    for backend, by_regime in sorted(regimes["records"].items()):
        kernels = regimes["kernels"][backend]
        lines.append(
            f"- {backend}: "
            + ", ".join(
                f"{count} record(s) `{regime}`"
                for regime, count in sorted(by_regime.items())
            )
            + f". Kernels: {kernels['port']} tuned behind the earlier kernels of "
            f"their port (run between the L2 flush and every timed call), "
            f"{kernels['alone']} tuned alone after a flush"
            + (
                f", {kernels['unrecorded']} in records that name no regime "
                "(tuned alone)"
                if kernels["unrecorded"]
                else ""
            )
            + "."
        )
    lines.append("")
    if regimes["later_alone"]:
        lines += [
            (
                "Kernels of records in the port regime that were tuned alone "
                "although earlier kernels of their port run before them:"
            ),
            "",
        ]
        lines += [
            f"- {name}: {reason}"
            for name, reason in sorted(regimes["later_alone"].items())
        ]
        lines.append("")
    return lines


def processes(record: dict[str, Any]) -> int:
    """How many bench processes a record is of: one, or those an aggregate
    record counts (``common.aggregate_bench_records``)."""
    return record["aggregate"]["processes"] if "aggregate" in record else 1


def impl_times(
    record: dict[str, Any], impl: str, metric: str, stale_ok: bool = False
) -> list[float | None]:
    """An implementation's usable timing in each process of a record
    (``impl_time``'s rule; an aggregate record's are its ``per_process``
    values).

    A port, an oracle or the memory copy of an aggregate record is usable
    in every process or in none: what one process does not reproduce is not
    the point's result. An Inductor compile is the baseline, and is taken
    process by process, as that process's own record gives it: usable where
    it ran and passed in that process. One that failed in a single process
    is then missing from that process's baseline alone, and the others
    still take the fastest of them all (``summarize`` names such processes:
    ``inductor_missing``)."""
    if "aggregate" not in record:
        return [impl_time(record, impl, metric, stale_ok)]
    count = processes(record)
    info = record["impls"].get(impl) or {}
    per_process = info.get("per_process") or {}
    values = per_process.get(metric)
    if values is None:
        return [None] * count
    if impl in common.INDUCTOR_IMPLS:
        counts = [
            usable({"status": status, "numerics_pass": passed})
            for status, passed in zip(
                per_process["status"], per_process["numerics_pass"], strict=True
            )
        ]
    else:
        counts = [usable(info, stale_ok)] * count
    return [
        value if counted and common.is_timing(value) else None
        for value, counted in zip(values, counts, strict=True)
    ]


def debiased_baseline(record: dict[str, Any], metric: str) -> list[float | None]:
    """The de-biased fastest Inductor compile's time in each process of a
    record (``common.fastest_inductor``: the baseline without what choosing
    the fastest of several noisy readings takes off), None where a process
    has none. A record of one process made before ``bench.py`` stored the
    figure gets it from its rounds here; an aggregate holds each
    process's (``fastest_inductor_per_process``), and one made before has
    them once ``main`` has read its processes' records
    (``with_fastest_inductor``)."""
    if "aggregate" in record:
        per_process = record.get("fastest_inductor_per_process") or [None] * processes(
            record
        )
    else:
        held = record.get("fastest_inductor")
        per_process = [held or common.fastest_inductor(record["impls"])]
    return [((entry or {}).get(metric) or {}).get("debiased") for entry in per_process]


def with_fastest_inductor(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """``record`` (read from ``path``) with each process's baseline two ways
    (``fastest_inductor_per_process``): an aggregate made before the figure
    takes it from the records of the processes it counted, which lie beside
    it, while every one of them is the record the aggregate was made of
    (the digests it holds, ``aggregate.records``). Otherwise the aggregate
    stays without the figure: a baseline of other readings than the
    headline's is not shown beside it."""
    if "aggregate" in record and "fastest_inductor_per_process" not in record:
        digests = record["aggregate"]["records"]
        per_process = []
        for name in record["aggregate"]["counted"]:
            process = path.with_name(name)
            if not process.exists():
                return record
            held = process.read_bytes()
            if hashlib.sha256(held).hexdigest() != digests.get(name):
                return record
            per_process.append(common.fastest_inductor(json.loads(held)["impls"]))
        record["fastest_inductor_per_process"] = per_process
    return record


def middle(values: list[float | None]) -> float | None:
    """The median over a record's processes; None unless every process has
    a value."""
    if any(value is None for value in values):
        return None
    return statistics.median(cast("list[float]", values))


def rebalance(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` with each implementation's ``cold_us`` and ``bb_us``
    recomputed from its recorded rounds (``cold_runs_us``, ``bb_runs_us``)
    the way ``bench.py`` computes them now: the median, for bb the minimum,
    over each order's rounds, averaged (``common.order_balanced``; with 5
    rounds the orders are forward, reverse, forward, reverse, forward). The
    per-order values go to ``cold_order_medians_us`` and
    ``bb_order_mins_us``, as ``bench.py`` records them. An aggregate record
    holds no rounds (its processes' records do) and stays as it is."""
    if "aggregate" in record:
        return record
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
    same = common.timed_kernels_held(info, config)
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
    # Each implementation's timing in each process of the record (one, or
    # those of an aggregate); Inductor's is the fastest of its compiles
    # (default, coordinate descent, max-autotune) in that process, of those
    # that are usable there.
    compiles = {
        impl: impl_times(record, impl, metric) for impl in common.INDUCTOR_IMPLS
    }
    times: dict[str, list[float | None]] = {
        "inductor": [
            min(usable_times) if usable_times else None
            for usable_times in (
                [t for t in process if t is not None]
                for process in zip(*compiles.values(), strict=True)
            )
        ],
        # The same baseline without the bias of choosing the fastest of
        # noisy readings; beside the headline, which stays the minimum.
        "inductor_debiased": debiased_baseline(record, metric),
        "oracle": impl_times(record, "oracle", metric),
        **{impl: impl_times(record, impl, metric, impl in elsewhere) for impl in PORTS},
        "memcopy": impl_times(record, "memcopy", metric),
    }
    row: dict[str, Any] = {
        "repro": record["repro"],
        "shape": record["shape"],
        "family": common.family(record["repro"]),
        "total_bytes": record.get("total_bytes"),
        **{impl: middle(values) for impl, values in times.items()},
        "processes": processes(record),
    }

    def speedup(impl: str, base: str) -> tuple[float | None, list[float]]:
        """``base`` time over ``impl`` time: the ratio in each process, and
        the median of those (the gate's rule; one process's ratio for a
        plain record). None unless both are timed in every process."""
        if row[impl] is None or row[base] is None:
            return None, []
        ratios = [
            cast("float", b) / cast("float", t)
            for t, b in zip(times[impl], times[base], strict=True)
        ]
        return statistics.median(ratios), ratios

    # ratio() reads these: (implementation, base) -> speedup.
    row["speedups"] = {}
    for key, impl, base in (
        ("cute_vs_inductor", "helion_cute", "inductor"),
        ("cute_vs_inductor_debiased", "helion_cute", "inductor_debiased"),
        ("cute_vs_oracle", "helion_cute", "oracle"),
        ("cute_vs_helion_triton", "helion_cute", "helion_triton"),
        ("triton_vs_inductor", "helion_triton", "inductor"),
    ):
        row[key], per_process = speedup(impl, base)
        row["speedups"].setdefault(impl, {})[base] = row[key]
        if key == "cute_vs_inductor" and len(per_process) > 1:
            row["cute_vs_inductor_per_process"] = per_process
    cute = row["helion_cute"]
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
        for impl in (*common.INDUCTOR_IMPLS, "oracle", "helion_triton", "helion_cute")
    }
    kernels = {}
    for impl in (*common.INDUCTOR_IMPLS, "helion_cute", "helion_triton"):
        info = record["impls"].get(impl) or {}
        kernels[impl] = [k["name"] for k in info.get("kernels", [])]
    row["kernels"] = kernels
    # The configs Inductor's kernels ended on (bench.py's inductor_configs):
    # a list per process.
    row["inductor_configs"] = {
        impl: picks
        for impl in common.INDUCTOR_IMPLS
        if (picks := inductor_picks(record["impls"].get(impl) or {}))
    }
    # The processes of a repeated bench in which an Inductor compile is not
    # usable, by their records: their baseline is of the other compiles alone.
    row["inductor_missing"] = {}
    if "aggregate" in record:
        names = record["aggregate"]["counted"]
        row["inductor_missing"] = {
            impl: missing
            for impl, values in compiles.items()
            if impl in record["impls"]
            and (
                missing := [
                    name
                    for name, value in zip(names, values, strict=True)
                    if value is None
                ]
            )
        }
    # Implementations whose kernels' profiler spans overlap (dependent
    # launches): their kernel_us is of the steps, not of the spans. An
    # aggregate has the two totals only of an implementation that every
    # process profiled.
    row["overlapping_spans"] = {
        impl: {"kernel_us": info["kernel_us"], "kernel_span_us": info["kernel_span_us"]}
        for impl, info in record["impls"].items()
        if info.get("kernel_spans_overlap")
        and "kernel_us" in info
        and "kernel_span_us" in info
    }
    row["foreign_gpu_processes"] = sorted(
        set(record.get("foreign_gpu_processes_before") or [])
        | set(record.get("foreign_gpu_processes_after") or [])
    )
    row["foreign_activity"] = common.foreign_activity(record)
    # The Inductor compiles that the bench timed right behind a Helion port
    # (bench.py's carryover_risk; an aggregate's are of all its processes).
    row["carryover_risk"] = list(record.get("carryover_risk") or [])
    row["issues"] = issues

    def counted(impl: str) -> bool:
        if row[impl] is None or "bench" in issues:
            return False
        return impl not in PORTS or f"{PORTS[impl]} autotune" not in issues

    row["counted"] = [impl for impl in IMPLS if counted(impl)]
    return row


def ratio(row: dict[str, Any], impl: str, base: str) -> float | None:
    """``base`` time over ``impl`` time, when the aggregates count both: of
    a record of several processes, the median of the ratios in each
    (``summarize``)."""
    if impl in row["counted"] and base in row["counted"]:
        return row["speedups"][impl][base]
    return None


def inductor_picks(info: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """The configs that the kernels of an Inductor compile ended on, a list
    per bench process (``info``: the compile's entry in a bench record or in
    an aggregate; none for a record made before ``bench.py`` named them)."""
    if "per_process" in info:
        return [picks or [] for picks in info["per_process"]["inductor_configs"]]
    return [info["inductor_configs"]] if "inductor_configs" in info else []


def pick_text(kernels: list[dict[str, Any]]) -> str:
    """One compile's picks on a line: each kernel's block sizes, warps and
    stages (a mix-order reduction's split size is among the block sizes).
    A process that names no kernel says so: its compile failed before the
    bench read its picks, or launches no Triton kernel."""
    if not kernels:
        return "no kernel recorded"
    parts = []
    for kernel in kernels:
        launchers = kernel.get("launchers", [kernel])
        configs = " | ".join(
            " ".join(
                [
                    *(
                        f"{name}={value}"
                        for name, value in config["block_sizes"].items()
                    ),
                    f"warps={config['num_warps']}",
                    f"stages={config['num_stages']}",
                ]
            )
            for config in launchers
        )
        parts.append(f"{kernel['name']}: {configs or 'no launcher'}")
    return "; ".join(parts)


def inductor_config_lines(rows: list[dict[str, Any]]) -> list[str]:
    """The report's section on the configs Inductor's kernels ended on: per
    kernel row and compile, each distinct set of picks with the number of
    bench processes that ended on it."""
    lines = []
    for row in sorted(rows, key=operator.itemgetter("repro", "shape")):
        for impl, picks in row["inductor_configs"].items():
            counts = collections.Counter(pick_text(kernels) for kernels in picks)
            for text, count in counts.most_common():
                share = (
                    f" ({count} of {len(picks)} processes)" if len(picks) > 1 else ""
                )
                lines.append(f"- {row['repro']}[{row['shape']}] {impl}{share}: {text}")
    return lines


def aggregate(
    rows: list[dict[str, Any]],
    weights: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
    rotating: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """The aggregates over ``rows`` (see the module docstring); ``weights``
    and ``baseline`` map (repro, shape) to the model share and to the
    baseline run's Helion+CuTe speedup over Inductor, ``rotating`` (None
    when no record has one) to this run's by ``rot_us``."""
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
    if rotating is not None:
        both = [
            (value, rotating[row["repro"], row["shape"]])
            for row, value in timed
            if (row["repro"], row["shape"]) in rotating
        ]
        out["rot_us"] = {
            "kernels": len(both),
            "cute_vs_inductor": geomean([rotated for _, rotated in both]),
            "same_kernels_by_this_timer": geomean([value for value, _ in both]),
        }
    return out


def ok_records(run_dir: Path, kind: str) -> dict[Path, dict[str, Any]]:
    """The ok records of a run's ``bench`` or ``configs`` directory."""
    found = {}
    for path in sorted((run_dir / kind).glob("*__*.json")):
        if common.is_repeat_record(path):
            # One process of a repeated bench: its point's aggregate record
            # stands for it (campaign.py --bench-repeats).
            continue
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


def seed_pairing(run_dir: Path, baseline: Path) -> dict[str, Any]:
    """Which of the autotunes the run ``run_dir`` made searched from the
    random seed of the ``baseline`` run's autotune of the same port, as
    ``campaign.py --seeds-from`` has them do: ``paired``, by record name,
    and for the others (``unpaired``) where their seed came from (the
    record's ``seed_from``, the run ``--seeds-from`` took it from). The
    run's ok records that are no search of its own stay out: those
    ``plan_run.py`` took from earlier runs, reused or provisional
    (``reused_from``), and the copies ``campaign.py --configs-from`` made
    (``common.holds_copy``, by the run's ``common.COPIED_CONFIGS``). It goes
    by the seeds the records of the two runs name (``common.record_seed``,
    ``common.standing_seed``), so two runs that took their seeds from a
    third are paired as well, and a baseline whose port was autotuned again
    since is not."""
    manifest = run_dir / common.COPIED_CONFIGS
    copied = json.loads(manifest.read_text()) if manifest.exists() else {}
    paired: list[str] = []
    unpaired: dict[str, str] = {}
    for path, rec in ok_records(run_dir, "configs").items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if rec.get("reused_from") or common.holds_copy(copied.get(path.name), digest):
            continue
        name = path.stem
        seed = common.record_seed(rec)
        before, why = common.standing_seed(
            baseline, rec["repro"], rec["shape"], rec["backend"]
        )
        source = rec.get("seed_from")
        if seed is not None and seed == before:
            paired.append(name)
        elif source is None:
            unpaired[name] = "a seed of its own"
        elif Path(source).resolve() != baseline.resolve():
            unpaired[name] = f"a seed taken from {Path(source).name}"
        else:
            now = "another now" if before is not None else f"none now ({why})"
            unpaired[name] = f"a seed taken from {baseline.name}, which names {now}"
    return {"baseline": str(baseline), "paired": paired, "unpaired": unpaired}


def seed_pairing_lines(pairing: dict[str, Any], baseline_name: str) -> list[str]:
    """The report's paragraph on ``seed_pairing``: how many of the run's
    autotunes searched from the baseline's seeds, and where the seeds of the
    others came from (with their names where some are paired: the
    exceptions of a paired run)."""
    paired, unpaired = pairing["paired"], pairing["unpaired"]
    total = len(paired) + len(unpaired)
    if not total:
        return [
            (
                "Autotune seeds: this run made no autotune of its own (its "
                "autotune records are other runs')."
            ),
            "",
        ]
    made = f"the {total} autotune{'' if total == 1 else 's'} this run made"
    how = (
        f"searched from the seed of {baseline_name}'s autotune of the same port "
        "(`campaign.py --seeds-from`)"
    )
    caveat = (
        "against that run also compares two random seeds, which a cold "
        "search's pick follows"
    )
    if paired:
        backends = sorted({name.rsplit("__", 1)[1] for name in (*paired, *unpaired)})
        counts = ", ".join(
            f"{backend} {sum(name.endswith(f'__{backend}') for name in paired)} of "
            f"{sum(name.endswith(f'__{backend}') for name in (*paired, *unpaired))}"
            for backend in backends
        )
        summary = (
            f"Autotune seeds: {len(paired)} of {made} {how}: {counts}. A "
            f"kernel's speedup {caveat}, where its port searched from another:"
        )
    else:
        summary = (
            f"Autotune seeds: none of {made} {how}, so each kernel's speedup "
            f"{caveat}. Their seeds:"
        )
    others: dict[str, list[str]] = collections.defaultdict(list)
    for name, why in sorted(unpaired.items()):
        others[why].append(name)
    lines = [summary, ""]
    lines += [
        f"- {why}: {len(names)}" + (f" ({', '.join(names)})" if paired else "")
        for why, names in sorted(others.items())
    ]
    if not unpaired:
        lines.append("- none")
    if paired:
        lines += [
            "",
            (
                "The same seed draws the same initial population only where both "
                "trees build the same search space for a port; generation 0 of "
                "the two runs' `logs/search/*/search.csv` shows whether they did."
            ),
        ]
    return [*lines, ""]


# From this relative difference on, a kernel's two speedups (over the
# fastest Inductor compile and over the de-biased baseline) count as apart.
DEBIASED_APART = 0.005


def debiased_lines(table: list[dict[str, Any]], rot: bool = False) -> list[str]:
    """What the report says of the de-biased baseline under a timer's
    aggregates: over the kernels that the aggregates count and that have
    both speedups, the two geomeans and how many kernels' speedups lie more
    than ``DEBIASED_APART`` apart (``rot``: of the ``rot_us`` rows, under
    the ``cold_us`` ones). Nothing for a run whose records have no rounds
    to take the figure from, nor for ``bb_us``, which has none."""
    pairs = [
        (headline, r["cute_vs_inductor_debiased"])
        for r in table
        if (headline := ratio(r, "helion_cute", "inductor")) is not None
        and r["cute_vs_inductor_debiased"] is not None
    ]
    if not pairs:
        return []
    apart = sum(abs(other / headline - 1) > DEBIASED_APART for headline, other in pairs)
    what = (
        "The same by `rot_us`"
        if rot
        else (
            "De-biased baseline (`common.fastest_inductor`: the compiles that "
            "ended on the same code pooled, the fastest distinct one chosen on "
            "other rounds than it is timed on; the headline above divides by "
            "the fastest compile's own reading, a minimum of noisy readings)"
        )
    )
    return [
        (
            f"{what}: CuTe/ind geomean "
            f"{fmt(geomean([other for _, other in pairs]), '.3f')} "
            f"over {len(pairs)} kernels, {fmt(geomean([h for h, _ in pairs]), '.3f')} "
            f"by the headline over the same; {apart} of them differ by more "
            f"than {DEBIASED_APART:.1%}."
        ),
        "",
    ]


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
    if any("rot_us" in a for a in columns.values()):
        entries += [
            (
                "Helion+CuTe vs Inductor by rot_us, private inputs, geomean (kernels)",
                lambda a: times(
                    a["rot_us"]["cute_vs_inductor"], a["rot_us"]["kernels"]
                ),
            ),
            (
                "the same kernels by cold_us, geomean",
                lambda a: times(a["rot_us"]["same_kernels_by_this_timer"]),
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
        record = with_fastest_inductor(record, path)
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
    # rot_us, where bench.py --rotate-inputs timed it: its rows, and each
    # kernel's Helion+CuTe speedup over Inductor where both count.
    rotated = [
        r for r in records if any("rot_us" in info for info in r["impls"].values())
    ]
    rotating: dict[tuple[str, str], float] | None = None
    if rotated:
        rows["rot_us"] = [
            summarize(
                r, "rot_us", tuned[r["repro"], r["shape"]], args.include_unsampled
            )
            for r in rotated
        ]
        rotating = {
            (row["repro"], row["shape"]): value
            for row in rows["rot_us"]
            if (value := ratio(row, "helion_cute", "inductor")) is not None
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
            if rec.get("contaminated") and "aggregate" in rec:
                # No process of the repeated bench was clean.
                found = [*(found or []), f"all {processes(rec)} bench processes"]
            elif rec.get("contaminated"):
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
        # The Triton build that compiled the Inductor kernels, where the
        # records name it (bench.py's triton_backend): a baseline run's is
        # fbtriton or the Tile IR fork with its switch on.
        build = records[0].get("triton_backend")
        triton = "" if build is None else f" ({common.triton_label(build)})"
        lines += [
            (
                f"GPU: {gpu.get('name')} ({gpu.get('power_limit_w')} W cap); "
                f"torch {ver.get('torch')}, triton {ver.get('triton')}{triton}, "
                f"cutlass-dsl {ver.get('nvidia-cutlass-dsl')}, "
                f"helion {ver.get('helion_git')}"
            ),
            "",
        ]
    counts = sorted({processes(record) for record in records} - {1})
    if counts:
        lines += [
            (
                f"{sum(processes(record) > 1 for record in records)} of the "
                f"{len(records)} bench records are aggregates over "
                f"{', '.join(map(str, counts))} bench processes (`campaign.py "
                "--bench-repeats`): a timing is the median over the processes, "
                "and a speedup the median of the ratios taken in each process, "
                "Inductor being the fastest of its compiles in that process."
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
    baseline_name = None
    pairing = None
    if args.baseline is not None:
        baseline_name = args.baseline.name
        pairing = seed_pairing(args.run_dir, args.baseline)
        lines += seed_pairing_lines(pairing, baseline_name)
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
        # rot_us cross-checks the cold timer.
        check = rotating if metric == "cold_us" else None
        columns = {"all kernels": aggregate(table, weights, baseline, check)}
        if len(without) < len(table):
            columns[f"without {', '.join(args.without)}"] = aggregate(
                without, weights, baseline, check
            )
        aggregates[metric] = columns
        members: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in table:
            members[r["family"]].append(r)
        families[metric] = {
            name: aggregate(group, weights, baseline, check)
            for name, group in members.items()
        }
        lines += [f"## {metric}", "", *aggregate_lines(columns, baseline_name), ""]
        lines += debiased_lines(table)
        if check is not None:
            lines += debiased_lines(rows["rot_us"], rot=True)
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
        # Each kernel's speedup by rot_us, as timed (the aggregates count it
        # under the conditions of the cold_us one beside it).
        by_rot = (
            {(r["repro"], r["shape"]): r["cute_vs_inductor"] for r in rows["rot_us"]}
            if check is not None
            else None
        )
        rot_column = "" if by_rot is None else " | cute/ind rot"
        # ... and over rot_us's de-biased baseline, where records have it.
        by_rot_debiased = None
        if check is not None and any(
            r["cute_vs_inductor_debiased"] is not None for r in rows["rot_us"]
        ):
            by_rot_debiased = {
                (r["repro"], r["shape"]): r["cute_vs_inductor_debiased"]
                for r in rows["rot_us"]
            }
            rot_column += " | cute/ind rot de-biased"
        # The speedup's range over the processes of a repeated bench.
        repeated = any(r["processes"] > 1 for r in table)
        spread_column = " | cute/ind per process (n)" if repeated else ""
        # The speedup over the de-biased baseline, where records have it.
        debiased = any(r["cute_vs_inductor_debiased"] is not None for r in table)
        debiased_column = " | cute/ind de-biased" if debiased else ""
        lines += [
            "",
            (
                "| repro | shape | MB | inductor | helion-cute | cute/ind"
                f"{debiased_column}{spread_column}{rot_column}{baseline_column}"
                " | helion-triton"
                " | triton/ind | oracle | memcopy | ind %SOL | cute %SOL | numerics"
                " | left out of the aggregates |"
            ),
            f"|{'---|' * (14 + debiased + repeated + rot_column.count('|') + bool(baseline_column))}",
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
            rot = (
                ""
                if by_rot is None
                else f" | {fmt(by_rot.get((r['repro'], r['shape'])), '.3f')}"
            )
            if by_rot_debiased is not None:
                rot += f" | {fmt(by_rot_debiased.get((r['repro'], r['shape'])), '.3f')}"
            spread = ""
            if repeated:
                ratios = r.get("cute_vs_inductor_per_process")
                spread = " | " + (
                    f"{min(ratios):.3f}..{max(ratios):.3f} ({len(ratios)})"
                    if ratios
                    else "-"
                )
            other = (
                f" | {fmt(r['cute_vs_inductor_debiased'], '.3f')}" if debiased else ""
            )
            lines.append(
                f"| {r['repro']} | {r['shape']} | {fmt((r['total_bytes'] or 0) / 1e6, '.1f')}"
                f" | {fmt(r['inductor'])} | {fmt(r['helion_cute'])}"
                f" | {fmt(r['cute_vs_inductor'], '.3f')}{other}{spread}{rot}{before}"
                f" | {fmt(r['helion_triton'])} | {fmt(r['triton_vs_inductor'], '.3f')}"
                f" | {fmt(r['oracle'])} | {fmt(r['memcopy'])}"
                f" | {fmt(100 * r['inductor_of_sol'] if r.get('inductor_of_sol') else None, '.0f')}"
                f" | {fmt(100 * r['cute_of_sol'] if r.get('cute_of_sol') else None, '.0f')}"
                f" | {num} | {left_out or '-'} |"
            )
        lines.append("")
    regimes = tuning_regimes(autotunes)
    lines += tuning_regime_lines(regimes)
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
    # What bench.py --rotate-inputs decided for each record it ran on.
    rotation = {
        f"{rec['repro']}__{rec['shape']}": rec["rotate_inputs"]
        for rec in records
        if "rotate_inputs" in rec
    }
    if rotation:
        timed = [d for d in rotation.values() if d.get("status") == "ok"]
        sets = sorted({d["sets"] for d in timed})
        lines += [
            "## Rotating inputs (rot_us)",
            "",
            (
                f"`bench.py --rotate-inputs` timed {len(timed)} of {len(rotation)} "
                f"records on private input sets ({', '.join(map(str, sets)) or '-'} "
                "per implementation), one per call of a flushed batch; not timed:"
            ),
            "",
        ]
        # A skipped record gives its reason, a failed pass its exception.
        left = {
            name: d.get("skipped")
            or f"failed: {(d.get('error') or '?').strip().splitlines()[-1]}"
            for name, d in rotation.items()
            if d.get("status") != "ok"
        }
        lines += [f"- {name}: {why}" for name, why in sorted(left.items())]
        if not left:
            lines.append("- none")
        lines += ["", ROTATION_CAVEAT, ""]
    picks = inductor_config_lines(rows["cold_us"])
    if picks:
        lines += [
            "## Inductor configs",
            "",
            (
                "The config that each Triton kernel of Inductor's compiles "
                "ended on in the bench (block sizes, a mix-order reduction's "
                "`RSPLIT_SIZE` among them, warps, stages). Coordinate descent "
                "tunes `inductor_cd` and `inductor_max` again in every bench "
                "process, and its pick moves a reading (README, Baseline)."
            ),
            "",
            *picks,
            "",
        ]
    missing = [
        f"- {r['repro']}[{r['shape']}] {impl}: not usable in {len(names)} of "
        f"{r['processes']} bench processes ({', '.join(names)})"
        for r in sorted(rows["cold_us"], key=operator.itemgetter("repro", "shape"))
        for impl, names in r["inductor_missing"].items()
    ]
    if missing:
        lines += [
            "## Inductor compiles missing from a process",
            "",
            (
                "A repeated bench takes Inductor process by process, as the "
                "fastest of the compiles that ran and passed in that process. "
                "Where one of them failed, that process's baseline is of the "
                "other compiles alone (as in a single bench), and its ratio "
                "can be higher than the other processes' for that reason."
            ),
            "",
            *missing,
            "",
        ]
    behind = [
        f"- {r['repro']}[{r['shape']}]: {'; '.join(r['carryover_risk'])}"
        for r in sorted(rows["cold_us"], key=operator.itemgetter("repro", "shape"))
        if r["carryover_risk"]
    ]
    if behind:
        lines += [
            "## Inductor compiles behind a Helion port",
            "",
            (
                "Benches whose listing timed an Inductor compile right behind "
                "a Helion port's batch, or first in an order that the port "
                "ends (`bench.py`'s `carryover_risk`). On some ports that "
                "compile then reads up to a microsecond a call higher, and "
                "the ratio over the fastest Inductor compile with it (README, "
                "Timers): list the oracle between the Inductor compiles and "
                "the Helion ports, and the memcopy last."
            ),
            "",
            *behind,
            "",
        ]
    overlapping = {
        f"{r['repro']}__{r['shape']} {impl}": spans
        for r in rows["cold_us"]
        for impl, spans in r["overlapping_spans"].items()
    }
    if overlapping:
        lines += [
            "## Dependent launches",
            "",
            (
                "Implementations whose kernels' profiler spans overlap (a kernel "
                "launched as a dependent launch starts its span while its "
                "predecessor runs): `kernel_us` and each kernel's `us` are of "
                "the kernels' steps, end minus the later of start and the "
                "latest earlier end; the spans add up to more."
            ),
            "",
        ]
        lines += [
            f"- {name}: steps {spans['kernel_us']:.1f} us, spans "
            f"{spans['kernel_span_us']:.1f} us"
            for name, spans in sorted(overlapping.items())
        ]
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
        "tuning_regimes": regimes,
        "contaminated": contaminated,
        "unsampled": unsampled,
        "uncovered": uncovered,
        "eager_reference": eager_reference,
    }
    if rotation:
        report["rotate_inputs"] = rotation
    if pairing is not None:
        report["seed_pairing"] = pairing
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
