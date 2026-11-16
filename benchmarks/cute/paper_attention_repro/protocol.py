"""CPU-only workload selection and measurement contracts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from pathlib import Path

IMPLEMENTATIONS = (
    "helion-cute",
    "sdpa",
    "fa4",
    "flex-triton",
    "flex-triton-ma",
    "flex-cute",
)
HELION_IMPLEMENTATIONS = {"helion-cute"}
SDPA_DENSE = {"noop", "causal", "gqa-causal", "mla-causal"}
MEASUREMENT = {
    "num_runs": 5,
    "warmup_ms": 500,
    "rep_ms": 500,
    "thermal_warmup_ms": 5000,
    "sdpa_thermal_warmup_ms": 2000,
    "cache_warmup_calls": 3,
    "statistic": "median of five CUDA-event run medians",
    "autotune_effort": "full",
    "autotune_budget_seconds": 1800,
    "input_seed": 0,
    "correctness": True,
    "idle_between_processes_seconds": 20,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id(case: dict[str, Any]) -> str:
    return f"{case['variant']}.{case['label']}"


def select_cases(
    profile: dict[str, Any], variants: str, shapes: str
) -> list[dict[str, Any]]:
    cases = profile["cases"]
    available = {case["variant"] for case in cases}
    selected_variants = available if variants == "all" else set(variants.split(","))
    unknown = selected_variants - available
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")
    selected_shapes = {int(value) for value in shapes.split(",")}
    if not selected_shapes or selected_shapes - {0, 1, 2}:
        raise ValueError("--shapes must contain indices from 0,1,2")
    result = [
        case
        for case in cases
        if case["variant"] in selected_variants
        and case["shape_index"] in selected_shapes
    ]
    if len({case_id(case) for case in result}) != len(result):
        raise ValueError("Reference profile contains duplicate cases")
    return result


def parse_gpus(value: str) -> list[int]:
    gpus = [int(gpu) for gpu in value.split(",")]
    if not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct nonnegative GPU indices")
    return gpus


def parse_impls(value: str) -> list[str]:
    impls = list(IMPLEMENTATIONS) if value == "all" else value.split(",")
    unknown = set(impls) - set(IMPLEMENTATIONS)
    if unknown or len(set(impls)) != len(impls):
        raise ValueError(f"Invalid or repeated implementations: {value}")
    return impls


def shard_cases(
    cases: list[dict[str, Any]], gpus: list[int]
) -> dict[int, list[dict[str, Any]]]:
    """Keep all shapes and implementations of a variant on one GPU."""
    assignments: dict[str, int] = {}
    shards: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    for case in cases:
        variant = case["variant"]
        if variant not in assignments:
            assignments[variant] = gpus[len(assignments) % len(gpus)]
        shards[assignments[variant]].append(case)
    return shards


def contract(case: dict[str, Any], mode: str, impl: str) -> dict[str, Any]:
    helion = impl in HELION_IMPLEMENTATIONS
    sparse_helion = helion and case["batch"] == "sparse"
    masked_sdpa = impl == "sdpa" and case["variant"] not in SDPA_DENSE
    sdpa_in_timing = masked_sdpa and (mode == "backward" or case["batch"] == "sparse")
    if mode == "backward":
        output = "dq, dk, dv"
        region = (
            "delta and exp2-LSE preprocess, fused gradients, dq dtype conversion"
            if helion
            else "autograd.grad, including the library's backward preprocess/postprocess"
        )
        forward_state = (
            "FP32 reference output and natural-log LSE prepared outside timing"
            if helion
            else "native forward and saved state prepared outside timing"
        )
    else:
        output = (
            "attention output and natural-log LSE"
            if sparse_helion
            else "attention output"
        )
        region = "forward attention with prepared layouts and mask metadata"
        forward_state = None
    if sdpa_in_timing:
        region += (
            "; extra mask/bias conversion cost, then backward uses the forward-saved mask"
            if mode == "backward"
            else "; compiled dense mask/bias materialization inside timing"
        )
    if (
        impl in {"flex-triton", "flex-triton-ma"}
        or sparse_helion
        or (mode == "backward" and helion)
    ):
        block_size = [128, 128]
    elif impl == "flex-cute":
        block_size = [256, 128]
    else:
        block_size = None
    return {
        "output": output,
        "timed_region": region,
        "forward_state": forward_state,
        "input_layout": "contiguous B,H,S,D",
        "fa4_layout": "contiguous B,S,H,D prepared outside timing"
        if impl == "fa4"
        else None,
        "block_mask_size": block_size,
        "block_mask_construction": "outside timing" if block_size else None,
        "sdpa_mask_or_bias": ("inside timing" if sdpa_in_timing else "outside timing")
        if masked_sdpa
        else None,
        "sdpa_backend": "fastest correct cuDNN/flash/efficient"
        if impl == "sdpa"
        else None,
        "sdpa_backward_mask": (
            "conversion result discarded; autograd reads the forward-saved mask"
            if masked_sdpa and mode == "backward" and sdpa_in_timing
            else "forward-saved mask"
            if masked_sdpa and mode == "backward"
            else None
        ),
        "flex_compile_mode": "max-autotune-no-cudagraphs"
        if impl == "flex-triton-ma"
        else "default",
        "fa4_return_lse": False if impl == "fa4" and mode == "forward" else None,
        "internal_flex_lse": "library implementation"
        if impl.startswith("flex-")
        else None,
        "require_fused_flash": mode == "forward" and impl == "helion-cute",
        "latest_backward_mla_reference": (
            "pass7 source, configs and full metadata unavailable; canonical earlier workload"
            if mode == "backward" and case["variant"] == "mla-causal"
            else None
        ),
    }


def runtime_target(case: dict[str, Any], mode: str) -> str:
    if mode == "forward":
        return f"forward_{'fused' if case['batch'] == 'fused' else 'sparse'}"
    return "backward"


def config_file(case: dict[str, Any], impl: str, directory: Path) -> Path:
    return directory / f"{case_id(case)}.{impl}.json"


def protected_output(out: Path, protected: list[Path]) -> None:
    resolved = out.resolve()
    for path in protected:
        if resolved == path.resolve() or path.resolve() in resolved.parents:
            raise ValueError(
                f"Output directory is inside protected input/source: {path}"
            )
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {out}")
