"""Build concise native-source optimization prompts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


_BACKEND_LABELS = {"cute": "CuTe DSL (CUTLASS)", "triton": "Triton"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Keep measured outcomes; full samples and provenance stay in the records."""
    return {
        "ok": evaluation["ok"],
        "perf": evaluation["perf"],
        "unit": evaluation["unit"],
        "cases": [
            {
                key: case[key]
                for key in ("index", "status", "perf", "noise", "error")
                if case[key] is not None
            }
            for case in evaluation["cases"]
        ],
    }


def build_handoff_prompt(manifest: Mapping[str, Any]) -> str:
    """Describe only the native task and contracts."""
    objective = manifest["objective"]
    aggregation = {"geomean": "geometric mean", "max": "maximum"}[
        objective["aggregation"]
    ]
    if objective["unit"] == "ms":
        goal = (
            "Minimize kernel latency (ms)."
            if len(manifest["cases"]) == 1
            else f"Minimize the {aggregation} of per-case latencies (ms)."
        )
    else:
        goal = (
            f"Minimize the {aggregation} of latency/reference-latency ratios. "
            "Reference latencies (ms, in case order): "
            + _json(objective["reference_latencies"])
            + "."
        )
    cases = [
        {
            **{
                key: case[key]
                for key in (
                    "source",
                    "entrypoint",
                    "hardware",
                    "input_metadata",
                    "dependencies",
                    "static_shapes",
                    "timing",
                    "atol",
                    "rtol",
                    "scale_atol",
                )
            },
            "backend": _BACKEND_LABELS.get(case["backend"], case["backend"]),
            **(
                {"alias_signature": case["alias_signature"]}
                if case.get("alias_signature") is not None
                else {}
            ),
        }
        for case in manifest["cases"]
    ]
    contract = (
        "Preserve entrypoints, output structure, input mutations, aliases, and "
        "numerical tolerances."
    )
    if any("alias_signature" in case for case in cases):
        contract += (
            " alias_signature lists storage groups of flattened output tensors "
            "followed by input tensors: equal IDs share storage; different IDs "
            "do not. IDs are local to each signature."
        )
    return "\n\n".join(
        (
            "Optimize the native kernel source and launch code in the listed files.",
            goal,
            "Cases:\n" + _json(cases),
            contract,
            (
                "The caller checks correctness and benchmarks changes. "
                "Change only the listed source files."
            ),
        )
    )
