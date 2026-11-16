"""Result schema, checkpointing, speedups, the summary, and the export.

``results.json`` is the source of truth and is rewritten atomically after
every step.  Raw latency samples are appended to ``samples.jsonl`` so the
checkpoint stays small; ``results.json`` keeps per-block statistics.
"""

from __future__ import annotations

import json
import math
import operator
import os
from typing import TYPE_CHECKING
from typing import Any

from . import dashboard
from .manifest import BASELINES
from .manifest import PROVIDERS
from .timing import median

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = 1
# A batch whose host enqueue time is at least this fraction of its measured
# time kept the GPU waiting on the host.
DISPATCH_LIMITED = 0.9


class Results:
    def __init__(self, out_dir: Path, data: dict[str, Any]) -> None:
        self.out_dir = out_dir
        self.data = data

    @classmethod
    def create(cls, out_dir: Path, header: dict[str, Any]) -> Results:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "samples.jsonl").unlink(missing_ok=True)
        results = cls(out_dir, {"schema": SCHEMA, **header, "cells": {}})
        results.save()
        return results

    @classmethod
    def load(cls, out_dir: Path) -> Results:
        data = json.loads((out_dir / "results.json").read_text())
        assert data["schema"] == SCHEMA, data["schema"]
        return cls(out_dir, data)

    @property
    def cells(self) -> dict[str, dict[str, Any]]:
        return self.data["cells"]

    def save(self) -> None:
        path = self.out_dir / "results.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, path)

    def append_samples(self, record: dict[str, Any]) -> None:
        with open(self.out_dir / "samples.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def drop_samples(self, key: str) -> None:
        """Forget a cell's samples before it is timed again."""
        path = self.out_dir / "samples.jsonl"
        if not path.exists():
            return
        kept = [
            line
            for line in path.read_text().splitlines(keepends=True)
            if json.loads(line)["cell"] != key
        ]
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(kept))
        os.replace(tmp, path)


def done_cells(results: Results) -> list[dict[str, Any]]:
    """Finished cells; an interrupted cell is never reported."""
    return [c for c in results.cells.values() if c["state"] == "done"]


def cell_key(kernel: str, index: int) -> str:
    return f"{kernel}/{index}"


def block_stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median": median(ordered),
        "p10": ordered[int(0.1 * (len(ordered) - 1))],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "n": len(ordered),
    }


def block_medians(provider: dict[str, Any]) -> list[float]:
    return [block["median"] for block in provider["latency"]]


def center(provider: dict[str, Any]) -> float:
    return median(block_medians(provider))


def paired_speedup(baseline: dict[str, Any], helion: dict[str, Any]) -> float:
    """Median over blocks of baseline / Helion, so > 1 means Helion is faster.

    Pairing within a block cancels drift that affects both providers.
    """
    ratios = [
        b / h
        for b, h in zip(block_medians(baseline), block_medians(helion), strict=True)
    ]
    return median(ratios)


def timed(cell: dict[str, Any], provider: str) -> dict[str, Any] | None:
    record = cell["providers"].get(provider)
    if record is None or record["status"] != "ok" or not record.get("latency"):
        return None
    return record


def best_baseline(cell: dict[str, Any]) -> str | None:
    candidates = [p for p in BASELINES if timed(cell, p) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda p: center(cell["providers"][p]))


def _dispatch_limited(record: dict[str, Any]) -> bool:
    return median(record["enqueue_ms"]) >= DISPATCH_LIMITED * median(
        record["kernel_ms"]
    )


def loss_tag(cell: dict[str, Any], best: str) -> str:
    """Why Helion loses to ``best``: dispatch-bound, kernel-bound or unknown.

    Dispatch-bound: the host could not keep the GPU busy with Helion's calls,
    or Helion's device time already wins.  Device times are only comparable
    as MPS event measurements, and a dispatch-limited baseline's is only an
    upper bound.
    """
    helion = cell["providers"]["helion"]
    baseline = cell["providers"][best]
    if helion["kernel_ms_source"] != "mps_events":
        return "unknown"
    if _dispatch_limited(helion):
        return "dispatch-bound"
    if baseline["kernel_ms_source"] != "mps_events":
        return "unknown"
    if median(helion["kernel_ms"]) > median(baseline["kernel_ms"]):
        return "kernel-bound"
    return "unknown" if _dispatch_limited(baseline) else "dispatch-bound"


def _fmt_ms(value: float) -> str:
    return f"{value:.4f}" if value < 1 else f"{value:.3f}"


def _fmt_x(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}x"


def _status(cell: dict[str, Any], provider: str) -> str:
    record = cell["providers"].get(provider)
    if record is None:
        # The cell itself failed (its inputs, reference or sync floor).
        return "cell failed" if "error" in cell else "—"
    return record["status"]


def summarize(results: Results) -> str:
    data = results.data
    cells = done_cells(results)
    run = data["run"]
    env = data["environment"]
    lines = [
        "# Helion Metal benchmark snapshot",
        "",
        f"- manifest `{data['manifest']['version']}` (`{data['manifest']['hash'][:12]}`)",
        (
            f"- {env['chip']}, macOS {env['macos']}, torch {env['torch']}, "
            f"MLX {env['mlx']}, Helion `{env['helion_commit']}`"
        ),
        (
            f"- autotune effort `{run['autotune_effort']}`, seed "
            f"{run['autotune_seed']}, {run['blocks']} blocks, order seed "
            f"{run['order_seed']}"
        ),
        (
            f"- preflight: {'forced' if run['force'] else 'clean'}; "
            f"stability: {run.get('stability', 'unmeasured')}"
        ),
        *(
            f"- resumed {r['at']} after {r['after'] or 'an interruption'}"
            + (
                f"; preflight then: {'; '.join(r['preflight_problems'])}"
                if r["preflight_problems"]
                else ""
            )
            for r in run.get("resumes", [])
        ),
        "",
        (
            "Speedup = baseline latency / Helion latency (> 1 means Helion is "
            "faster), the median of per-block paired ratios.  Latency is a warm "
            "single call plus synchronize.  MLX runs uncompiled (no mx.compile), "
            "so its composed formulations are unfused."
        ),
        "",
        "## Aggregates",
        "",
        "| provider | correct / total |",
        "|---|---|",
    ]
    total = len(cells)
    for provider in PROVIDERS:
        ok = sum(
            1 for c in cells if c["providers"].get(provider, {}).get("status") == "ok"
        )
        lines.append(f"| {provider} | {ok} / {total} |")
    speedups = []
    losses = []
    for c in cells:
        best = best_baseline(c)
        helion = timed(c, "helion")
        if best is None or helion is None:
            continue
        s = paired_speedup(c["providers"][best], helion)
        speedups.append(s)
        if s < 1:
            losses.append((s, c, best))
    lines.append("")
    if speedups:
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        lines.append(
            f"Helion vs best baseline: geomean {geomean:.2f}x over the "
            f"{len(speedups)} cells where both are correct and timed; "
            f"Helion loses {len(losses)} of them."
        )
    else:
        lines.append("No cell has both Helion and a baseline correct and timed.")

    lines += ["", "## Correctness", ""]
    lines.append(
        "| kernel | case | shape | " + " | ".join(PROVIDERS) + " | dashboard 1e-2 |"
    )
    lines.append("|---|---|---|" + "---|" * len(PROVIDERS) + "---|")
    for c in cells:
        helion = c["providers"].get("helion", {})
        dashboard = helion.get("dashboard")
        verdict = (
            "—" if dashboard is None else ("pass" if dashboard["passed"] else "fail")
        )
        lines.append(
            f"| {c['kernel']} | {c['index']} | `{c['label']}` | "
            + " | ".join(_status(c, p) for p in PROVIDERS)
            + f" | {verdict} |"
        )

    lines += ["", "## Helion failures", ""]
    failures = [
        (c, c["providers"]["helion"])
        for c in cells
        if c["providers"].get("helion", {}).get("status") not in (None, "ok")
    ]
    if not failures:
        lines.append("None.")
    for c, record in failures:
        lines.append(
            f"- `{c['kernel']}` case {c['index']} `{c['label']}`: "
            f"**{record['status']}** — {record['error']}"
        )

    lines += ["", "## Cell failures", ""]
    failed = [c for c in cells if "error" in c]
    if not failed:
        lines.append("None.")
    for c in failed:
        lines.append(
            f"- `{c['kernel']}` case {c['index']} `{c['label']}`: {c['error']}"
        )

    lines += ["", "## Baseline failures", ""]
    lines.append(
        "A torch provider failing where MLX is correct points at an upstream bug."
    )
    lines.append("")
    baseline_failures = [
        (c, p, c["providers"][p])
        for c in cells
        for p in BASELINES
        if c["providers"].get(p, {}).get("status") not in (None, "ok", "unavailable")
    ]
    if not baseline_failures:
        lines.append("None.")
    for c, p, record in baseline_failures:
        lines.append(
            f"- `{c['kernel']}` case {c['index']} `{c['label']}` {p}: "
            f"**{record['status']}** — {record['error']}"
        )

    lines += ["", "## Performance", ""]
    header = ["kernel", "case", "shape", "helion ms"]
    header += [f"vs {p}" for p in BASELINES] + ["vs best", "helion kernel ms"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for c in cells:
        helion = timed(c, "helion")
        if helion is None:
            continue
        best = best_baseline(c)
        row = [c["kernel"], str(c["index"]), f"`{c['label']}`", _fmt_ms(center(helion))]
        for p in BASELINES:
            base = timed(c, p)
            row.append(_fmt_x(None if base is None else paired_speedup(base, helion)))
        if best is None:
            row.append(_fmt_x(None))
        else:
            row.append(
                f"{_fmt_x(paired_speedup(c['providers'][best], helion))} ({best})"
            )
        row.append(_fmt_ms(median(helion["kernel_ms"])))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Helion losses", ""]
    if not losses:
        lines.append("None.")
    losses.sort(key=operator.itemgetter(0))
    for s, c, best in losses:
        lines.append(
            f"- `{c['kernel']}` case {c['index']} `{c['label']}`: {s:.2f}x vs "
            f"{best}, {loss_tag(c, best)} (sync floor "
            f"{_fmt_ms(c['sync_floor_ms']['torch'])} ms)"
        )
    lines.append("")
    return "\n".join(lines)


def helionbench_records(results: Results) -> list[dict[str, Any]]:
    """Rows in the format ``benchmarks/run.py`` and ``run_tpu.py`` emit."""
    device = results.data["environment"]["chip"]
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for c in done_cells(results):
        by_kernel.setdefault(c["kernel"], []).append(c)
    records = []

    def add(kernel: str, metric: str, pairs: list[tuple[str, float]]) -> None:
        if pairs:
            records.append(
                {
                    "benchmark": {
                        "name": "Helion Metal Benchmark",
                        "extra_info": {"device": device, "backend": "metal"},
                    },
                    "model": {"name": kernel},
                    "metric": {
                        "name": metric,
                        "benchmark_values": [value for _, value in pairs],
                    },
                    "shape": [label for label, _ in pairs],
                }
            )

    for kernel, cells in by_kernel.items():
        accuracy = []
        latency = []
        speedup = []
        compile_speedup = []
        for c in cells:
            helion = c["providers"].get("helion")
            if helion is None:
                continue
            # The dashboard's accuracy is its 1e-2 gate against eager; without
            # an eager output, the strict verdict stands in.
            dashboard = helion.get("dashboard")
            passed = (
                dashboard["passed"]
                if dashboard is not None
                else helion["status"] == "ok"
            )
            accuracy.append((c["label"], 1.0 if passed else 0.0))
            eager = timed(c, "torch_eager")
            if timed(c, "helion") is None or eager is None:
                continue
            latency.append((c["label"], center(helion)))
            speedup.append((c["label"], paired_speedup(eager, helion)))
            compiled = timed(c, "torch_compile")
            if compiled is not None:
                compile_speedup.append((c["label"], paired_speedup(eager, compiled)))
        add(kernel, "helion_accuracy", accuracy)
        add(kernel, "helion_latency_ms", latency)
        add(kernel, "helion_speedup", speedup)
        add(kernel, "torch_compile_speedup", compile_speedup)
    return records


def write_reports(results: Results) -> None:
    (results.out_dir / "summary.md").write_text(summarize(results))
    (results.out_dir / "helionbench.json").write_text(
        json.dumps(helionbench_records(results), indent=2)
    )
    (results.out_dir / "dashboard.html").write_text(
        dashboard.render(results.data, done_cells(results))
    )
