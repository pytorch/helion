"""Create the attention-gym comparison table and heatmap from saved JSONL.

No GPU or Helion import is needed to aggregate results. Plotting additionally
requires matplotlib. For example::

    python benchmarks/cute/attngym_report.py --input results.jsonl \
        --manifest campaign_manifest.json \
        --plot-output comparison.png --csv-output comparison.csv \
        --summary-output comparison.json

Each throughput cell is the arithmetic mean of ``median_tflops`` across its
correct shapes. Speedup first selects the fastest correct, displayed baseline
for each shape, then takes the geometric mean of baseline/Helion median latency.
FAIL, SKIP, ERROR, and missing records never contribute performance numbers.
Partial cells and comparisons retain their coverage in all outputs.

A suite or campaign manifest defines the exact expected cases and dimensions.
Without one, the default figure expects three cases for each of its 29 rows;
``--expected-shapes`` selects generic count-based coverage for other totals.
Campaign completion requires all six terminal backend results for every case.
Full comparison coverage additionally requires correct Helion and at least one
correct baseline per case. SKIP, FAIL, and ERROR are terminal results.

Result cells must be unique across the input files. Select one campaign before
reporting; repeated cells and mixed GPU models or commits are rejected. All
implementations of a shape must use the same physical GPU; different shapes may
be assigned to different GPUs of that model. Legacy records may
omit ``shape_id`` if their complete ``shape`` is present. A legacy SKIP without
either field cannot identify a multi-shape result and is rejected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import hashlib
import json
import math
import operator
from pathlib import Path
from statistics import fmean
from statistics import geometric_mean
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


FIGURE_GROUPS = (
    ("Dense", ("noop", "causal", "gqa-causal", "mla-causal")),
    (
        "Static mask",
        (
            "sliding-window",
            "prefix-lm",
            "dilated-sw",
            "block-diffusion",
            "natten2d",
            "sta2d",
            "sta3d",
            "batchify",
            "svg-spatial",
            "svg-temporal",
        ),
    ),
    (
        "Gather mask",
        ("document", "shared-prefix", "global-sw", "flamingo", "vsa"),
    ),
    ("Speculative", ("jetspec-tree", "jetspec-training")),
    (
        "Score mod",
        (
            "alibi",
            "rel-bias",
            "softcap",
            "sandwich",
            "sigmoid-act",
            "graphormer-spatial",
            "graphormer-edge",
        ),
    ),
    ("Combined", ("gemma2",)),
)
FIGURE_IMPLS = (
    "sdpa",
    "fa4",
    "flex-triton",
    "flex-triton-ma",
    "flex-cute",
    "helion-cute",
)
IMPL_LABELS = (
    "SDPA\n(cuDNN)",
    "FA4",
    "FlexAttention\n(Triton)",
    "FlexAttention\n(Triton, max-autotune)",
    "FlexAttention\n(CuTe)",
    "Helion\n(CuTe)",
)
_STATUS_LABELS = {
    "FAIL": "wrong output",
    "SKIP": "no kernel",
    "ERROR": "error",
    "MISSING": "not run",
    "NOT_REQUESTED": "not requested",
}
_SHAPE_FIELDS = ("b", "hq", "hkv", "m", "n", "d", "dv", "dtype")
_CASE_SETTINGS = (
    "input_seed",
    "benchmark_timer",
    "benchmark_statistic",
    "num_runs",
    "warmup_ms",
    "rep_ms",
    "thermal_warmup_ms",
    "cooldown_temp",
    "skip_correctness",
    "accuracy_atol",
    "accuracy_rtol",
    "parameters",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_shape(shape: object, *, complete: bool = False) -> dict[str, Any]:
    if not isinstance(shape, dict):
        raise ValueError("Expected a shape object")
    result = dict(shape)
    if "d" in result and "dv" not in result:
        result["dv"] = result["d"]
    if isinstance(result.get("dtype"), str):
        result["dtype"] = result["dtype"].removeprefix("torch.")
    if complete and any(field not in result for field in _SHAPE_FIELDS):
        raise ValueError("Manifest comparison requires complete shape dimensions")
    if complete:
        if any(
            isinstance(result[field], bool)
            or not isinstance(result[field], int)
            or result[field] <= 0
            for field in _SHAPE_FIELDS[:-1]
        ):
            raise ValueError("Manifest shape dimensions must be positive integers")
        if not isinstance(result["dtype"], str) or not result["dtype"]:
            raise ValueError("Manifest shape must contain a dtype")
    _canonical_json(result)
    return result


def _shape_id(record: Mapping[str, Any], shape: Mapping[str, Any]) -> str:
    shape_id = record.get("shape_id")
    if shape_id is None:
        if not shape:
            raise ValueError(
                f"Cannot identify shape for {record['variant']}: "
                "provide shape_id or shape, including for SKIP/ERROR records"
            )
        shape_id = _canonical_json(shape)
    if not isinstance(shape_id, str) or not shape_id:
        raise ValueError(f"Invalid shape_id for {record['variant']}")
    return shape_id


def _manifest_cases(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty cases list")
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    aliases: dict[str, tuple[str, str]] = {}
    explicit_ids: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("variant"), str)
            or not entry["variant"]
        ):
            raise ValueError("Manifest case must contain a variant")
        shape = _normalize_shape(entry.get("shape"), complete=True)
        shape_id = _shape_id(entry, shape)
        key = (entry["variant"], shape_id)
        if key in cases or (entry.get("shape_id") and shape_id in explicit_ids):
            raise ValueError(f"Duplicate manifest case ID: {shape_id}")
        if entry.get("shape_id"):
            explicit_ids.add(shape_id)
        alias = entry.get("shape_id") or entry.get("case") or entry["variant"]
        if alias in aliases:
            raise ValueError(f"Duplicate manifest case ID: {alias}")
        aliases[alias] = key
        cases[key] = {**entry, "shape_id": shape_id, "shape": shape}
    campaign = manifest.get("campaign", {})
    if not isinstance(campaign, dict):
        raise ValueError("Manifest campaign must be an object")
    selected = campaign.get("selected_cases")
    if selected is not None:
        if not isinstance(selected, list) or not selected:
            raise ValueError("Campaign selected_cases must be a nonempty list")
        if not all(isinstance(case_id, str) for case_id in selected):
            raise ValueError("Campaign selected case IDs must be strings")
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate campaign selected case ID")
        if set(selected) - set(aliases):
            raise ValueError("Campaign selects a case absent from the suite manifest")
        cases = {aliases[alias]: cases[aliases[alias]] for alias in selected}
    return cases


def read_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                records.append({**record, "_source": f"{path}:{line_number}"})
    return records


def _number(record: Mapping[str, Any], field: str) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (field == "median_ms" and value == 0)
    ):
        raise ValueError(f"Invalid {field} for {record['variant']}/{record['impl']}")
    return float(value)


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("variant", "impl", "accuracy"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"Result must contain a nonempty {field}")
    if record["accuracy"] not in {"PASS", "FAIL", "SKIP", "ERROR"}:
        raise ValueError(f"Unknown accuracy status: {record['accuracy']}")
    shape = _normalize_shape(record.get("shape", {}))
    shape_id = _shape_id(record, shape)
    latency = _number(record, "median_ms")
    throughput = _number(record, "median_tflops")
    if record["accuracy"] == "PASS" and latency is None:
        raise ValueError(f"PASS result lacks median_ms: {record['variant']}/{shape_id}")
    return {
        **record,
        "shape": shape,
        "shape_id": shape_id,
        "median_ms": latency,
        "median_tflops": throughput,
    }


def _one_metadata_value(
    records: Sequence[Mapping[str, Any]],
    field: str,
    *,
    consistent_presence: bool = False,
) -> object:
    present = [row[field] for row in records if row.get(field) is not None]
    values = {_canonical_json(value): value for value in present}
    if len(values) > 1:
        raise ValueError(f"Mixed {field} measurements: {', '.join(sorted(values))}")
    if consistent_presence and present and len(present) != len(records):
        raise ValueError(f"Missing {field} metadata on some compared results")
    return next(iter(values.values()), None)


def _aggregate_cell(
    records: Sequence[dict[str, Any]], shape_count: int
) -> dict[str, Any]:
    counts = Counter(row["accuracy"] for row in records)
    counts["MISSING"] = max(0, shape_count - len(records))
    passing = [row for row in records if row["accuracy"] == "PASS"]
    throughput = [
        row["median_tflops"] for row in passing if row["median_tflops"] is not None
    ]
    mean_tflops = fmean(throughput) if throughput else None
    if shape_count == 0 and not records:
        status = "NOT_REQUESTED"
    elif not records:
        status = "MISSING"
    elif counts["PASS"] == shape_count:
        status = "PASS"
    elif counts["PASS"]:
        status = "PARTIAL"
    else:
        statuses = [name for name, count in counts.items() if count]
        status = statuses[0] if len(statuses) == 1 else "MIXED"
    if mean_tflops is not None:
        display = f"{mean_tflops:.1f}"
        if len(throughput) != shape_count:
            display += f"\n{len(throughput)}/{shape_count} shapes"
    elif passing:
        display = "timing only"
        if len(passing) != shape_count:
            display += f"\n{len(passing)}/{shape_count} shapes"
    elif status in _STATUS_LABELS:
        display = _STATUS_LABELS[status]
    else:
        display = "\n".join(
            f"{_STATUS_LABELS[name]} {count}" for name, count in counts.items() if count
        )
    return {
        "status": status,
        "expected_shapes": shape_count,
        "recorded_shapes": len(records),
        "extra_shapes": max(0, len(records) - shape_count),
        "mean_tflops": mean_tflops,
        "throughput_shapes": len(throughput),
        "pass_shapes": counts["PASS"],
        "fail_shapes": counts["FAIL"],
        "skip_shapes": counts["SKIP"],
        "error_shapes": counts["ERROR"],
        "missing_shapes": counts["MISSING"],
        "display": display,
    }


def aggregate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_shapes: int | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate one campaign, preserving shape identity and correctness coverage."""
    if expected_shapes is not None and expected_shapes < 1:
        raise ValueError("expected_shapes must be positive")
    expected_cases = _manifest_cases(manifest) if manifest is not None else None
    if expected_cases is not None:
        if expected_shapes is not None and expected_shapes != len(expected_cases):
            raise ValueError(
                "expected_shapes disagrees with the selected manifest cases"
            )
        expected_shapes = len(expected_cases)
    elif expected_shapes is None:
        expected_shapes = 87
    default_figure = expected_cases is None and expected_shapes == 87
    normalized = [_normalize_record(row) for row in records]
    if not normalized:
        raise ValueError("No benchmark records found")
    sources: dict[tuple[str, str, str], str] = {}
    signatures: dict[tuple[str, str], str] = {}
    explicit_ids: dict[str, str] = {}
    for index, row in enumerate(normalized):
        shape_key = (row["variant"], row["shape_id"])
        if records[index].get("shape_id") is not None:
            previous = explicit_ids.setdefault(row["shape_id"], row["variant"])
            if previous != row["variant"]:
                raise ValueError(
                    f"Duplicate shape_id across variants: {row['shape_id']}"
                )
        key = (*shape_key, row["impl"])
        source = row.get("_source", f"record {index + 1}")
        if key in sources:
            raise ValueError(
                f"Duplicate result cell {key}: {sources[key]} and {source}. "
                "Select a single run; repeated measurements are not merged."
            )
        sources[key] = source
        if expected_cases is not None:
            if shape_key not in expected_cases:
                raise ValueError(f"Unexpected case outside the manifest: {shape_key}")
            actual = _normalize_shape(row["shape"], complete=True)
            if actual != expected_cases[shape_key]["shape"]:
                raise ValueError(
                    f"Shape dimensions disagree with manifest: {shape_key}"
                )
            expected_parameters = expected_cases[shape_key].get("parameters")
            if (
                expected_parameters is not None
                and row.get("parameters") != expected_parameters
            ):
                raise ValueError(f"Parameters disagree with manifest: {shape_key}")
        if row["shape"]:
            signature = json.dumps(row["shape"], sort_keys=True, allow_nan=False)
            if shape_key in signatures and signatures[shape_key] != signature:
                raise ValueError(f"Inconsistent shape metadata for {shape_key}")
            signatures[shape_key] = signature
    selected = [row for row in normalized if row["impl"] in FIGURE_IMPLS]
    if not selected:
        raise ValueError("No records for the displayed implementations")
    measured = [row for row in selected if row["median_ms"] is not None]
    metadata: dict[str, Any] = {
        field: _one_metadata_value(normalized, field)
        for field in ("gpu", "helion_commit", "source_commit", "suite")
    }
    for field in ("source_fingerprint", "manifest_sha256"):
        metadata[field] = _one_metadata_value(
            normalized, field, consistent_presence=True
        )
    if manifest is not None:
        provenance = manifest.get("source_provenance", {})
        base_manifest = {
            key: value
            for key, value in manifest.items()
            if key not in {"campaign", "source_provenance"}
        }
        manifest_hash = hashlib.sha256(
            _canonical_json(base_manifest).encode()
        ).hexdigest()
        _one_metadata_value(
            [provenance, {"manifest_sha256": manifest_hash}, metadata],
            "manifest_sha256",
        )
        _one_metadata_value([provenance, metadata], "source_fingerprint")
        _one_metadata_value([provenance, metadata], "source_commit")
        _one_metadata_value([manifest, metadata], "suite")
    for field in ("physical_gpu", "gpu_uuid"):
        values = sorted({str(row[field]) for row in selected if row.get(field)})
        metadata[field] = values[0] if len(values) == 1 else None
        metadata[f"{field}s"] = values
    warnings = []
    for field in (
        "gpu",
        "physical_gpu",
        "helion_commit",
        "source_fingerprint",
        "manifest_sha256",
    ):
        if (
            field in ("source_fingerprint", "manifest_sha256")
            and metadata[field] is None
        ):
            continue
        missing = sum(not row.get(field) for row in measured)
        if missing:
            warnings.append(f"{missing} measured records lack {field} metadata")
    excluded_impls = sorted({row["impl"] for row in normalized} - set(FIGURE_IMPLS))
    if excluded_impls:
        warnings.append(
            f"Excluded implementations from figure and best-other baseline: {', '.join(excluded_impls)}"
        )
    shapes: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        shapes[(row["variant"], row["shape_id"])][row["impl"]] = row
    comparisons: list[dict[str, Any]] = []
    shape_keys = expected_cases if expected_cases is not None else shapes
    for variant, shape_id in sorted(shape_keys):
        cells = shapes.get((variant, shape_id), {})
        case_records = list(cells.values())
        case_metadata = {
            field: _one_metadata_value(case_records, field)
            for field in ("physical_gpu", "gpu_uuid")
        }
        campaign = manifest.get("campaign", {}) if manifest is not None else {}
        for field in _CASE_SETTINGS:
            value = _one_metadata_value(case_records, field, consistent_presence=True)
            if case_records and campaign.get(field) is not None and value is None:
                raise ValueError(
                    f"Missing {field} metadata required by campaign manifest"
                )
            if value is not None and campaign.get(field) is not None:
                _one_metadata_value([campaign, {field: value}], field)
            case_metadata[field] = value
        expected_gpu = campaign.get("case_gpus", {}).get(shape_id)
        if expected_gpu is not None and case_metadata["physical_gpu"] is not None:
            if str(expected_gpu) != str(case_metadata["physical_gpu"]):
                raise ValueError(f"GPU assignment disagrees with manifest: {shape_id}")
        helion = cells.get("helion-cute")
        baselines = [
            row
            for impl, row in cells.items()
            if impl != "helion-cute" and row["accuracy"] == "PASS"
        ]
        best = (
            min(baselines, key=operator.itemgetter("median_ms")) if baselines else None
        )
        speedup = None
        reason = None
        if helion is None:
            reason = "Helion result missing"
        elif helion["accuracy"] != "PASS":
            reason = f"Helion {helion['accuracy']}"
        elif best is None:
            reason = "No correct baseline"
        else:
            speedup = best["median_ms"] / helion["median_ms"]
        comparisons.append(
            {
                "variant": variant,
                "shape_id": shape_id,
                "observed": bool(cells),
                "missing_impls": [impl for impl in FIGURE_IMPLS if impl not in cells],
                **case_metadata,
                "shape": expected_cases[(variant, shape_id)]["shape"]
                if expected_cases is not None
                else next((row["shape"] for row in cells.values() if row["shape"]), {}),
                "helion_ms": helion["median_ms"]
                if helion and helion["accuracy"] == "PASS"
                else None,
                "best_other_ms": best["median_ms"] if best else None,
                "best_other_impl": best["impl"] if best else None,
                "speedup_vs_best_other": speedup,
                "reason": reason,
                "results": {
                    impl: {key: value for key, value in row.items() if key != "_source"}
                    for impl, row in cells.items()
                },
            }
        )
    variant_groups = [(name, group) for group, names in FIGURE_GROUPS for name in names]
    known_variants = {name for name, group in variant_groups}
    observed_variants = {row["variant"] for row in selected}
    if expected_cases is not None:
        expected_counts = Counter(variant for variant, shape_id in expected_cases)
    elif default_figure:
        expected_counts = Counter(dict.fromkeys(known_variants, 3))
    else:
        expected_counts = Counter(variant for variant, shape_id in shapes)
    missing_variants = sorted(set(expected_counts) - observed_variants)
    variant_groups.extend(
        (name, "Other")
        for name in sorted((observed_variants | set(expected_counts)) - known_variants)
    )
    variants: list[dict[str, Any]] = []
    for variant, group in variant_groups:
        variant_shapes = [row for row in comparisons if row["variant"] == variant]
        speedups = [
            row["speedup_vs_best_other"]
            for row in variant_shapes
            if row["speedup_vs_best_other"] is not None
        ]
        observed = sum(row["observed"] for row in variant_shapes)
        expected = expected_counts.get(variant, observed)
        impl_cells = {
            impl: _aggregate_cell(
                [
                    row
                    for row in selected
                    if row["variant"] == variant and row["impl"] == impl
                ],
                expected,
            )
            for impl in FIGURE_IMPLS
        }
        campaign_complete = observed == expected and all(
            not row["missing_impls"] for row in variant_shapes
        )
        variants.append(
            {
                "variant": variant,
                "group": group,
                "expected_shapes": expected,
                "observed_shapes": observed,
                "missing_shapes": max(0, expected - observed),
                "compared_shapes": len(speedups),
                "campaign_complete": campaign_complete,
                "comparison_complete": observed == expected
                and len(speedups) == expected,
                "speedup_vs_best_other": geometric_mean(speedups) if speedups else None,
                "impls": impl_cells,
            }
        )
    speedups = [
        row["speedup_vs_best_other"]
        for row in comparisons
        if row["speedup_vs_best_other"] is not None
    ]
    observed_count = len(shapes)
    compared_count = len(speedups)
    shape_coverage_complete = observed_count == expected_shapes
    if default_figure:
        shape_coverage_complete = shape_coverage_complete and all(
            sum(name == variant for name, shape_id in shapes) == count
            for variant, count in expected_counts.items()
        )
    expected_backend_cells = expected_shapes * len(FIGURE_IMPLS)
    covered_backend_cells = sum(
        min(cell["recorded_shapes"], row["expected_shapes"])
        for row in variants
        if row["variant"] in expected_counts
        for cell in row["impls"].values()
    )
    missing_backend_cells = max(0, expected_backend_cells - covered_backend_cells)
    campaign_complete = shape_coverage_complete and missing_backend_cells == 0
    comparison_complete = shape_coverage_complete and compared_count == expected_shapes
    complete = campaign_complete and comparison_complete
    missing_shapes = sum(row["missing_shapes"] for row in variants)
    if expected_cases is None and not default_figure:
        missing_shapes = max(0, expected_shapes - observed_count)
    coverage = f"{compared_count}/{expected_shapes} expected shapes compared ({observed_count} observed)"
    if default_figure and missing_shapes and observed_count >= expected_shapes:
        coverage = f"{compared_count} observed shapes compared; {missing_shapes} expected shape slots missing"
    coverage += f"; {covered_backend_cells}/{expected_backend_cells} expected backend results recorded"
    if not complete:
        coverage = f"INCOMPLETE: {coverage}"
    return {
        "schema_version": 2,
        **metadata,
        "expected_shapes": expected_shapes,
        "observed_shapes": observed_count,
        "compared_shapes": compared_count,
        "uncompared_observed_shapes": observed_count - compared_count,
        "unobserved_expected_shapes": missing_shapes,
        "missing_case_ids": [
            row["shape_id"] for row in comparisons if not row["observed"]
        ]
        if expected_cases is not None
        else None,
        "manifest_validated": expected_cases is not None,
        "expected_backend_cells": expected_backend_cells,
        "recorded_backend_cells": len(selected),
        "covered_backend_cells": covered_backend_cells,
        "missing_backend_cells": missing_backend_cells,
        "campaign_complete": campaign_complete,
        "comparison_complete": comparison_complete,
        "complete": complete,
        "coverage": coverage,
        "geomean_speedup_vs_best_other": geometric_mean(speedups) if speedups else None,
        "throughput_aggregation": "Arithmetic mean of per-shape median_tflops with accuracy PASS; partial cells show contributing shape count.",
        "speedup_aggregation": "Geometric mean of per-shape min(PASS baseline median_ms) / PASS Helion median_ms; baseline limited to the five other displayed implementations.",
        "excluded_impls": excluded_impls,
        "missing_variants": missing_variants,
        "warnings": warnings,
        "variants": variants,
        "shapes": comparisons,
    }


def write_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows = []
    for variant in report["variants"]:
        row = {
            key: variant[key]
            for key in (
                "group",
                "variant",
                "expected_shapes",
                "observed_shapes",
                "missing_shapes",
                "compared_shapes",
                "campaign_complete",
                "comparison_complete",
                "speedup_vs_best_other",
            )
        }
        for impl, cell in variant["impls"].items():
            row[impl] = (
                cell["mean_tflops"]
                if cell["mean_tflops"] is not None
                else cell["display"].replace("\n", "; ")
            )
            row[f"{impl}_status"] = cell["status"]
            row[f"{impl}_pass_shapes"] = cell["pass_shapes"]
            row[f"{impl}_throughput_shapes"] = cell["throughput_shapes"]
            row[f"{impl}_missing_shapes"] = cell["missing_shapes"]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_report(
    report: Mapping[str, Any], path: Path, *, title: str | None = None
) -> None:
    import matplotlib  # pyrefly: ignore [missing-import]

    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle  # pyrefly: ignore [missing-import]
    import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]

    rows = report["variants"]
    values = [
        [row["impls"][impl]["mean_tflops"] for impl in FIGURE_IMPLS] for row in rows
    ]
    finite = [value for row in values for value in row if value is not None]
    matrix = [[math.nan if value is None else value for value in row] for row in values]
    figure, axis = plt.subplots(figsize=(15, max(7, len(rows) * 0.37)))
    image = axis.imshow(
        matrix, cmap="viridis", vmin=0, vmax=max(finite, default=1) or 1, aspect="auto"
    )
    for row_index, row in enumerate(rows):
        for column, impl in enumerate(FIGURE_IMPLS):
            cell = row["impls"][impl]
            if cell["mean_tflops"] is None:
                axis.add_patch(
                    Rectangle((column - 0.5, row_index - 0.5), 1, 1, color="#eeeeee")
                )
            color = (
                "white"
                if cell["mean_tflops"] is not None
                and cell["mean_tflops"] < 0.62 * (max(finite, default=1) or 1)
                else "black"
            )
            axis.text(
                column,
                row_index,
                cell["display"],
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
        speedup = row["speedup_vs_best_other"]
        text = "not compared" if speedup is None else f"{speedup:.2f}×"
        if speedup is not None and row["compared_shapes"] != row["expected_shapes"]:
            text += f"\n{row['compared_shapes']}/{row['expected_shapes']} shapes"
        axis.text(
            6,
            row_index,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="#245b27" if speedup is not None and speedup >= 1 else "#9c2929",
        )
    axis.set_xlim(-0.5, 6.5)
    axis.set_xticks(range(7), [*IMPL_LABELS, "Helion speedup\nvs best other"])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=10, labelsize=9)
    axis.set_yticks(
        range(len(rows)),
        [
            f"{row['variant']}  ({row['observed_shapes']}/{row['expected_shapes']})"
            for row in rows
        ],
    )
    axis.tick_params(axis="y", length=0, labelsize=9)
    axis.set_xticks([value - 0.5 for value in range(8)], minor=True)
    axis.set_yticks([value - 0.5 for value in range(len(rows) + 1)], minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="minor", length=0)
    for group in dict.fromkeys(row["group"] for row in rows):
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        axis.axhline(indices[0] - 0.5, color="#666666", linewidth=1.2)
        axis.text(
            -0.31,
            fmean(indices),
            group.replace(" ", "\n"),
            transform=axis.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=10,
        )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
    colorbar.set_label("Mean throughput (TFLOPS)")
    heading = (
        title
        or f"Attention-gym backend comparison — {report['gpu'] or 'GPU not recorded'}"
    )
    overall = report["geomean_speedup_vs_best_other"]
    score = (
        "No valid shape comparisons"
        if overall is None
        else f"Geomean speedup {overall:.2f}×"
    )
    coverage = report["coverage"].replace("; ", "\n", 1)
    figure.suptitle(f"{heading}\n{score} · {coverage}", fontsize=12, y=0.985)
    footer = "Throughput: arithmetic mean of correct shapes. Speedup: geomean of per-shape fastest correct baseline / Helion median latency."
    footer += "\nPartial cells show contributing / expected shape counts. Only PASS measurements contribute."
    if report["missing_variants"]:
        footer += (
            f" {len(report['missing_variants'])} figure variants have no measurements."
        )
    figure.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(left=0.27, right=0.94, bottom=0.085, top=0.86)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument(
        "--manifest", type=Path, help="Exported suite or campaign manifest JSON"
    )
    parser.add_argument(
        "--expected-shapes", type=int, help="Default: manifest case count, otherwise 87"
    )
    parser.add_argument("--title")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    if manifest is not None and not isinstance(manifest, dict):
        parser.error("--manifest must contain a JSON object")
    report = aggregate(
        read_records(args.input),
        expected_shapes=args.expected_shapes,
        manifest=manifest,
    )
    if args.csv_output:
        write_csv(args.csv_output, report)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if args.plot_output:
        plot_report(report, args.plot_output, title=args.title)
    print(report["coverage"])
    speedup = report["geomean_speedup_vs_best_other"]
    if speedup is not None:
        print(
            f"Geomean speedup on {report['compared_shapes']} compared shapes: {speedup:.3f}x"
        )
    for warning in report["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
