"""CPU-only reports for the example backend comparison's saved records."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import operator
from statistics import geometric_mean
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute.compare_example_backends import PROVIDERS

if TYPE_CHECKING:
    from pathlib import Path


def latency_field(candidates: list[dict[str, Any]]) -> str:
    """The sustained per-call time of a graph grown past the event timer's
    quantum (graph_long_ms) when every candidate has it, else the 16-call
    graph's (graph_calls_ms), else the autotuner-regime cold-L2 time
    (bench_ms), else the single-call replay time (whose launch-latency quantum
    hides sub-20 us differences). One field per (module, case, shape) row
    keeps each comparison like for like."""
    for field in ("graph_long_ms", "graph_calls_ms", "bench_ms"):
        if all(
            isinstance(row.get(field), (int, float))
            and math.isfinite(row[field])
            and row[field] > 0
            for row in candidates
        ):
            return field
    return "graph_ms"


def aggregate(run: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if run["schema"] != "example-backend-comparison-v1" or run["status"] != "complete":
        raise ValueError("Expected a completed example comparison")
    # The providers the run compared, in the run's order (a results.json from
    # before the field compared exactly ``PROVIDERS``).  A shape row is complete
    # when each of them has a record; the speedups are CuTe against Triton
    # when both ran and CuTe against the best of the others that ran.
    providers: list[str] = run.get("providers", list(PROVIDERS))
    compared = [provider for provider in providers if provider != "cute"]
    ratios = [
        name
        for name, available in (
            ("cute_vs_triton", "cute" in providers and "triton" in providers),
            ("cute_vs_best", "cute" in providers and bool(compared)),
        )
        if available
    ]
    groups: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    variants: set[tuple[str, str]] = set()
    inventories: dict[tuple[str, int], dict[str, Any]] = {}
    failures = []
    uncapturable = []
    # The CUDA graph dispatch state each successful record's readings were
    # taken in (``graph_state_warmup``; records from before the field existed
    # count as "unrecorded"), keyed by note.  More than one key means the
    # run's readings are not all comparable with each other.
    graph_states: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, int, str, str]] = set()
    for row in records:
        module, shape, provider = row["module"], row["shape"], row["provider"]
        if module not in run["modules"] or shape not in run["shapes"]:
            raise ValueError("Record outside the requested module/shape set")
        if provider == "inventory":
            if (module, shape) in inventories:
                raise ValueError(f"Duplicate inventory: {module}/shape{shape}")
            inventories[(module, shape)] = row
            if row["status"] == "ok":
                variants.update((module, case["name"]) for case in row["cases"])
            else:
                failures.append(row)
            continue
        if row["status"] != "ok":
            failures.append(row)
            continue
        if provider not in providers:
            raise ValueError(
                f"Record from a provider the run did not request: {provider}"
            )
        key = (module, row["case"], shape)
        identity = (*key, provider, row["impl"])
        if identity in seen:
            raise ValueError(f"Duplicate candidate: {identity}")
        seen.add(identity)
        eager_only = row.get("graph_capturable") is False
        if eager_only and provider != "baselines":
            raise ValueError(f"Helion kernel not captured in a CUDA graph: {identity}")
        if any(
            row.get(field) != "pass"
            for field in (
                "eager_correctness",
                *(() if eager_only else ("graph_correctness",)),
                "post_measurement_correctness",
            )
        ):
            raise ValueError(f"Successful result missing correctness: {identity}")
        if eager_only:
            # A baseline that synchronizes with the host has eager timings
            # only; they are not comparable with the graph timings ranked
            # here, so the record is listed instead.
            uncapturable.append(row)
            variants.add((module, row["case"]))
            continue
        latency = row["graph_ms"]
        if (
            not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError(f"Invalid graph latency: {identity}")
        groups[key][provider].append(row)
        variants.add((module, row["case"]))
        graph_states[row.get("graph_state_warmup") or "unrecorded"] += 1

    coverage = [
        {
            "module": module,
            "shape": shape,
            "status": inventories.get((module, shape), {}).get("status", "missing"),
            "case_count": len(inventories.get((module, shape), {}).get("cases", [])),
        }
        for module in run["modules"]
        for shape in run["shapes"]
    ]
    for module in run["modules"]:
        if not any(known == module for known, _ in variants):
            variants.add((module, "inventory unavailable"))

    shape_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    for module, case in sorted(variants):
        selected: list[dict[str, Any]] = []
        for shape in run["shapes"]:
            candidates = groups[(module, case, shape)]
            row: dict[str, Any] = {
                "module": module,
                "case": case,
                "shape": shape,
                "status": "incomplete",
            }
            field = latency_field([c for rows in candidates.values() for c in rows])
            row["latency_field"] = field
            for provider in providers:
                if candidates[provider]:
                    best = min(candidates[provider], key=operator.itemgetter(field))
                    row[f"{provider}_ms"] = best[field]
                    row[f"{provider}_impl"] = best["impl"]
            if inventories.get((module, shape), {}).get("status") == "ok" and all(
                candidates[provider] for provider in providers
            ):
                row["status"] = "ok"
                if "cute_vs_triton" in ratios:
                    row["cute_vs_triton"] = row["triton_ms"] / row["cute_ms"]
                if "cute_vs_best" in ratios:
                    best_ms = min(row[f"{provider}_ms"] for provider in compared)
                    row["cute_vs_best"] = best_ms / row["cute_ms"]
                selected.append(row)
            shape_rows.append(row)
        # Do not improve a type's score by dropping its failed/missing shapes.
        type_row: dict[str, Any] = {
            "variant": f"{module}/{case}",
            "valid_shapes": len(selected),
            "requested_shapes": len(run["shapes"]),
            "status": "incomplete",
        }
        if len(selected) == len(run["shapes"]):
            type_row["status"] = "ok"
            for ratio in ratios:
                type_row[ratio] = geometric_mean(row[ratio] for row in selected)
        type_rows.append(type_row)
    return {
        "providers": providers,
        "ratios": ratios,
        "shapes": shape_rows,
        "variants": type_rows,
        "failures": failures,
        "uncapturable": uncapturable,
        "inventory": coverage,
        "graph_states": dict(
            sorted(graph_states.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


RATIO_LABELS = {
    "cute_vs_triton": "vs Triton",
    "cute_vs_best": "vs best compared",
}


def speedup_line(providers: list[str], ratios: list[str]) -> str:
    """The report line defining the speedups a run with these providers
    reports (``aggregate``'s ``ratios``), or saying why there are none."""
    if not ratios:
        return (
            f"The run compared {', '.join(providers)} only, so no CuTe speedups "
            "are reported; shapes.csv lists the latencies."
        )
    compared = [provider for provider in providers if provider != "cute"]
    definitions = []
    if "cute_vs_triton" in ratios:
        definitions.append("vs Triton is Triton/CuTe")
    if "cute_vs_best" in ratios:
        definitions.append(f"vs best compared is min({', '.join(compared)})/CuTe")
    return (
        f"Speedup uses CUDA-graph latency ({'; '.join(definitions)}); higher is "
        "better. Each implementation ran in a separate process on the same GPU. "
        "These are exploratory comparisons; no interleaved fixed-config "
        "qualification is implied."
    )


def graph_state_lines(graph_states: dict[str, int]) -> list[str]:
    """Report lines naming the graph dispatch state of the run's readings, a
    warning when the records disagree (see ``compare_example_backends.
    graph_state_regime`` for why readings of different states do not
    compare)."""
    if len(graph_states) == 1:
        ((state, count),) = graph_states.items()
        return [f"Graph dispatch state of all {count} measured records: {state}."]
    listed = "; ".join(f"{state} ({count})" for state, count in graph_states.items())
    return [
        (
            "WARNING: the measured records disagree on the graph dispatch state "
            f"their readings were taken in, so they are not all comparable: {listed}."
        )
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_results(input_path: Path, output: Path) -> None:
    import matplotlib  # pyrefly: ignore [missing-import]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]

    run = json.loads(input_path.read_text())
    records = []
    for job in run["jobs"]:
        path = (input_path.parent / job["path"]).resolve()
        if not path.is_relative_to(input_path.parent.resolve()):
            raise ValueError("Result path escapes the run directory")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != job["sha256"]:
            raise ValueError(f"Result hash changed: {path}")
        records.append(json.loads(content))
    data = aggregate(run, records)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "shapes.csv", data["shapes"])
    write_csv(output / "variants.csv", data["variants"])
    write_csv(output / "inventory.csv", data["inventory"])
    (output / "provenance.json").write_text(
        json.dumps(
            {key: value for key, value in run.items() if key != "jobs"}, indent=2
        )
        + "\n"
    )
    (output / "failures.json").write_text(json.dumps(data["failures"], indent=2) + "\n")
    (output / "uncapturable.json").write_text(
        json.dumps(data["uncapturable"], indent=2) + "\n"
    )
    complete = [row for row in data["variants"] if row["status"] == "ok"]
    ratios = data["ratios"]
    if ratios:
        figure, axis = plt.subplots(figsize=(12, max(3, 0.45 * len(complete))))
        positions = list(range(len(complete)))
        spacing = 0.76 / len(ratios)
        for index, ratio in enumerate(ratios):
            offset = (index - (len(ratios) - 1) / 2) * spacing
            axis.barh(
                [p + offset for p in positions],
                [r[ratio] for r in complete],
                height=spacing - 0.02,
                label=f"CuTe {RATIO_LABELS[ratio]}",
            )
        axis.set_yticks(positions, [r["variant"] for r in complete])
        axis.invert_yaxis()
        axis.axvline(1, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Geometric mean speedup across shapes (higher is better)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "speedups.png", dpi=160)
        figure.savefig(output / "speedups.pdf")
        plt.close(figure)
    successful_inventories = sum(row["status"] == "ok" for row in data["inventory"])
    gpu = run.get("gpu_properties", {})
    versions = ", ".join(
        f"{name}={version}" for name, version in run.get("software", {}).items()
    )
    lines = [
        "# Example backend comparison",
        "",
        f"GPU: {gpu.get('name', 'unknown')} ({gpu.get('uuid', 'unknown')}), power limit {gpu.get('power_limit_w', 'unknown')} W, driver {gpu.get('driver_version', 'unknown')}. Software: {versions}. Run seed: {run.get('autotune_seed', 'unknown')}.",
        "",
        f"All {len(run['modules'])} requested modules are represented. Inventory succeeded for {successful_inventories}/{len(data['inventory'])} requested module/shape pairs; unavailable inventories remain N/A in the table and inventory.csv.",
        "",
        *graph_state_lines(data["graph_states"]),
        "",
        speedup_line(data["providers"], ratios),
        "",
        f"{len(complete)}/{len(data['variants'])} variants have all requested shapes; {len(data['failures'])} failed or unavailable candidate jobs remain in failures.json. Incomplete variants are N/A and are not plotted.",
        "",
        f"{len(data['uncapturable'])} baseline jobs synchronize with the host and could only be timed eagerly (no CUDA graph capture); they are correct, listed in uncapturable.json, and not ranked.",
        "",
        *(["![Speedups](speedups.png)", ""] if ratios else []),
        "| Variant | Shapes |" + "".join(f" {RATIO_LABELS[r]} |" for r in ratios),
        "|---|---:|" + "---:|" * len(ratios),
    ]
    for row in data["variants"]:
        values = "".join(
            f" {row[ratio]:.3f}× |" if row["status"] == "ok" else " N/A |"
            for ratio in ratios
        )
        lines.append(
            f"| {row['variant']} | {row['valid_shapes']}/{row['requested_shapes']} |{values}"
        )
    lines.extend(
        [
            "",
            "[Per-shape latencies and selected implementations](shapes.csv) · [Variant geomeans](variants.csv) · [Inventory coverage](inventory.csv) · [Provenance](provenance.json) · [Failed and unavailable candidates](failures.json) · [Eager-only baselines](uncapturable.json).",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
