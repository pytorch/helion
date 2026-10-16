"""Torch-free access to the shared ROCm validation and tuning policy.

The JSON is the source of numerical limits. Read it on each access so a policy
edit cannot silently reuse cached acceptance limits or validation receipts.
"""

from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

POLICY_PATH = Path(__file__).with_suffix(".json")


@lru_cache(maxsize=8)
def _parse(contents: bytes) -> dict:
    policy = json.loads(contents)
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported numerical policy schema")
    fixed_contract = {
        "oracle_dtype": "float64",
        "oracle_final_rounding": False,
        "finite_outputs": True,
        "zero_reference_norm": "elementwise_and_structural_zero",
        "conventional_matmul_tf32": False,
    }
    if any(policy["shared"].get(key) != value for key, value in fixed_contract.items()):
        raise ValueError("Unsupported oracle contract in numerical policy")
    for kernel in policy["kernels"].values():
        reference = kernel.get("reference")
        if kernel["id"] in ("K19", "K21", "K22") and reference is None:
            raise ValueError("Kernel requires its selected reference contract")
        if reference is not None:
            expected_reference = {
                "K19": ("torch_bf16_moe", "bfloat16"),
                "K21": ("flashkda_prefill", "native"),
                "K22": ("flashinfer_eager_decode", "float32"),
            }.get(kernel["id"])
            if (
                expected_reference is None
                or (reference.get("kind"), reference.get("oracle_dtype"))
                != expected_reference
                or reference.get("oracle_final_rounding") is not True
                or reference.get("comparison_dtype") != "float64"
            ):
                raise ValueError("Unsupported kernel reference contract")
        # Reduction axes are part of each operation's validator, not tunable
        # limits. Reject a changed description instead of silently ignoring it.
        group_scopes = {
            "K1": "per_expert",
            "K7": "per_query_head",
            "K8": "per_query_head",
            "K9": "per_row",
            "K10": "per_row",
            "K11": "per_sequence_head",
            "K12": "per_sequence_head",
            "K13": "per_sequence_head",
            "K14": "per_sequence_head",
            "K19": "per_token",
            "K21": "per_sequence_head",
            "K22": "per_sequence_head",
        }
        scope = ["global"]
        if kernel["id"] in group_scopes:
            scope.append(group_scopes[kernel["id"]])
        rules = kernel.get("rules", {})
        for name in ("clip_calibration_factor",):
            if name in rules and (
                isinstance(rules[name], bool)
                or not isinstance(rules[name], (int, float))
                or not math.isfinite(rules[name])
                or rules[name] < 0
            ):
                raise ValueError(f"Invalid numerical policy {name}")
        for name in ("familywise_alpha", "advancement_alpha", "miss_probability"):
            if name in rules and (
                isinstance(rules[name], bool)
                or not isinstance(rules[name], (int, float))
                or not 0 < rules[name] < 1
            ):
                raise ValueError(f"Invalid numerical policy {name}")
        for name in (
            "benchmark_draws_per_row",
            "qualification_draws_per_row",
            "qualification_rows",
            "audit_draws",
        ):
            if name in rules and (type(rules[name]) is not int or rules[name] < 2):
                raise ValueError(f"Invalid numerical policy {name}")
        gates = [
            gate
            for dtypes in kernel.get("outputs", {}).values()
            for gate in dtypes.values()
        ]
        gates.extend(kernel.get("reference_checks", {}).values())
        for gate in gates:
            if "l2_limit" in gate and gate.get("l2_scope") != scope:
                raise ValueError("Unsupported L2 reduction scope in numerical policy")
            if "group_l2_limit" in gate and (
                kernel["id"] not in {"K7", "K8"}
                or gate.get("l2_limit") is None
                or len(scope) != 2
            ):
                raise ValueError(
                    "Separate group L2 limits require attention global/group gates"
                )
            if gate.get("comparison") == "exact" and (
                gate.get("atol") != 0 or gate.get("rtol") != 0
            ):
                raise ValueError("Exact output contracts require zero tolerances")
            for name in ("atol", "rtol", "l2_limit", "group_l2_limit"):
                value = gate.get(name)
                if name == "group_l2_limit" and name in gate and value is None:
                    raise ValueError("Invalid numerical policy group_l2_limit: None")
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid numerical policy {name}: {value!r}")
    return policy


def _read() -> dict:
    return _parse(POLICY_PATH.read_bytes())


def shared_policy() -> dict:
    return copy.deepcopy(_read()["shared"])


def kernel_policy(kernel: str) -> dict:
    kernels = _read()["kernels"]
    if kernel not in kernels:
        kernel = next(
            (
                name
                for name, value in kernels.items()
                if value["id"] == kernel or kernel in value.get("aliases", [])
            ),
            kernel,
        )
    return copy.deepcopy(kernels[kernel])


def dtype_policies(kernel: str, output: str = "output") -> dict:
    return kernel_policy(kernel)["outputs"][output]


def tolerances(
    kernel: str,
    dtype: object | None = None,
    *,
    output: str = "output",
    atol: float | None = None,
    rtol: float | None = None,
) -> dict:
    """Return the selected gate, optionally tightened by explicit CLI limits."""
    dtypes = dtype_policies(kernel, output)
    if dtype is None:
        if len(dtypes) != 1:
            raise ValueError(f"Select an output dtype for {kernel}/{output}")
        dtype = next(iter(dtypes))
    dtype = str(dtype).removeprefix("torch.")
    gate = dtypes[dtype]
    for name, requested in (("atol", atol), ("rtol", rtol)):
        if requested is not None:
            if not math.isfinite(requested) or requested < 0:
                raise ValueError("Use finite nonnegative tolerances")
            gate[name] = min(requested, gate[name])
    return gate


def reference_tolerances(kernel: str, check: str = "default") -> dict:
    return kernel_policy(kernel)["reference_checks"][check]


def reference_policy(kernel: str | None = None) -> dict:
    """Identify oracle arithmetic separately from FP64 error measurement."""
    selected = kernel_policy(kernel).get("reference") if kernel is not None else None
    if selected is not None:
        return selected
    return {
        "kind": "fp64",
        "oracle_dtype": "float64",
        "oracle_final_rounding": False,
        "comparison_dtype": "float64",
    }


def require_exact(kernel: str, recipe: str | None = None) -> dict:
    rules = kernel_policy(kernel)["rules"]
    if rules["comparison"] != "byte_exact" or rules["atol"] != 0 or rules["rtol"] != 0:
        raise ValueError(f"{kernel} requires the exact byte contract")
    if recipe is not None and recipe not in rules["recipes"]:
        raise ValueError(f"Unsupported {kernel} recipe: {recipe}")
    return rules


def policy_identity() -> dict:
    contents = POLICY_PATH.read_bytes()
    policy = _parse(contents)
    return {
        "version": policy["policy_version"],
        "policy_sha256": hashlib.sha256(contents).hexdigest(),
        "policy_loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
