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

if TYPE_CHECKING:
    from pathlib import Path


def aggregate(run: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if run["schema"] != "example-backend-comparison-v1" or run["status"] != "complete":
        raise ValueError("Expected a completed example comparison")
    groups: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    variants: set[tuple[str, str]] = set()
    inventories: dict[tuple[str, int], dict[str, Any]] = {}
    failures = []
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
        if provider not in {"cute", "triton", "baselines"}:
            raise ValueError(f"Unknown provider: {provider}")
        key = (module, row["case"], shape)
        identity = (*key, provider, row["impl"])
        if identity in seen:
            raise ValueError(f"Duplicate candidate: {identity}")
        seen.add(identity)
        if any(
            row.get(field) != "pass"
            for field in (
                "eager_correctness",
                "graph_correctness",
                "post_measurement_correctness",
            )
        ):
            raise ValueError(f"Successful result missing correctness: {identity}")
        latency = row["graph_ms"]
        if (
            not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError(f"Invalid graph latency: {identity}")
        groups[key][provider].append(row)
        variants.add((module, row["case"]))

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
            providers = groups[(module, case, shape)]
            row: dict[str, Any] = {
                "module": module,
                "case": case,
                "shape": shape,
                "status": "incomplete",
            }
            for provider in ("cute", "triton", "baselines"):
                candidates = providers[provider]
                if candidates:
                    best = min(candidates, key=operator.itemgetter("graph_ms"))
                    row[f"{provider}_ms"] = best["graph_ms"]
                    row[f"{provider}_impl"] = best["impl"]
            if inventories.get((module, shape), {}).get("status") == "ok" and all(
                providers[provider] for provider in ("cute", "triton", "baselines")
            ):
                a, t, b = (
                    row[f"{provider}_ms"]
                    for provider in ("cute", "triton", "baselines")
                )
                row.update(
                    status="ok", cute_vs_triton=t / a, cute_vs_best=min(b, t) / a
                )
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
            type_row.update(
                status="ok",
                cute_vs_triton=geometric_mean(
                    row["cute_vs_triton"] for row in selected
                ),
                cute_vs_best=geometric_mean(row["cute_vs_best"] for row in selected),
            )
        type_rows.append(type_row)
    return {
        "shapes": shape_rows,
        "variants": type_rows,
        "failures": failures,
        "inventory": coverage,
    }


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
    complete = [row for row in data["variants"] if row["status"] == "ok"]
    figure, axis = plt.subplots(figsize=(12, max(3, 0.45 * len(complete))))
    positions = list(range(len(complete)))
    axis.barh(
        [p - 0.19 for p in positions],
        [r["cute_vs_triton"] for r in complete],
        height=0.36,
        label="CuTe vs Triton",
    )
    axis.barh(
        [p + 0.19 for p in positions],
        [r["cute_vs_best"] for r in complete],
        height=0.36,
        label="CuTe vs best compared",
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
        "Speedup is Triton/CuTe or min(best baseline, Triton)/CuTe using CUDA-graph latency; higher is better. Each implementation ran in a separate process on the same GPU. These are exploratory comparisons; no interleaved fixed-config qualification is implied.",
        "",
        f"{len(complete)}/{len(data['variants'])} variants have all requested shapes; {len(data['failures'])} failed or unavailable candidate jobs remain in failures.json. Incomplete variants are N/A and are not plotted.",
        "",
        "![Speedups](speedups.png)",
        "",
        "| Variant | Shapes | vs Triton | vs best compared |",
        "|---|---:|---:|---:|",
    ]
    for row in data["variants"]:
        values = [
            f"{row[key]:.3f}×" if row["status"] == "ok" else "N/A"
            for key in ("cute_vs_triton", "cute_vs_best")
        ]
        lines.append(
            f"| {row['variant']} | {row['valid_shapes']}/{row['requested_shapes']} | {values[0]} | {values[1]} |"
        )
    lines.extend(
        [
            "",
            "[Per-shape latencies and selected implementations](shapes.csv) · [Variant geomeans](variants.csv) · [Inventory coverage](inventory.csv) · [Provenance](provenance.json) · [Failed and unavailable candidates](failures.json).",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
