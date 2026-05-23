# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Matmul-family autotune seed heuristics.

This intentionally starts with a narrow surface area: matmul-family kernels on
B200, seeded from measured configs stored in
``heuristics/matmul_b200.json``. The seeds are used only as
additional initial-population candidates; regular autotuning still validates
and benchmarks them before selecting a config.

Rules match dtype plus M/N/K interval and exact-value predicates. Dimension
buckets in the JSON are interval predicates, not cumulative predicates. When
multiple rules match, rules with more matching bucket fields are used first and
JSON order breaks ties.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .._compat import extract_device
from .._compat import get_device_name
from ..runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base_search import _AutotunableKernel
    from .config_spec import ConfigSpec
    from .config_spec import MatmulFact


_RUNTIME_HEURISTICS_PATH = (
    Path(__file__).resolve().parent / "heuristics" / "matmul_b200.json"
)
_SUPPORTED_DEVICE_NAME = "B200"

_SHAPE_BUCKET_KEYS = frozenset(
    {
        "dtype",
        "k_bucket",
        "m_bucket",
        "n_bucket",
        "m_value",
        "n_value",
        "k_value",
    }
)


def matmul_heuristics_enabled() -> bool:
    value = os.environ.get("HELION_AUTOTUNE_MATMUL_HEURISTICS")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes", "on"}


def matmul_heuristics_supported_on_args(args: Sequence[object]) -> bool:
    device = extract_device(args)
    device_name = get_device_name(device) if device is not None else None
    return device_name is not None and _SUPPORTED_DEVICE_NAME in device_name


def _tensor_shapes(args: Sequence[object]) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for arg in args:
        shape = getattr(arg, "shape", None)
        if shape is None:
            continue
        try:
            shapes.append(tuple(int(dim) for dim in shape))
        except (TypeError, ValueError):
            continue
    return shapes


def _tensor_dtypes(args: Sequence[object]) -> list[str]:
    dtypes: list[str] = []
    for arg in args:
        dtype = getattr(arg, "dtype", None)
        if dtype is not None:
            dtypes.append(str(dtype))
    return dtypes


def _dtype_family(args: Sequence[object]) -> str:
    dtypes = _tensor_dtypes(args)
    return _dtype_family_from_dtype(dtypes[0] if dtypes else "unknown")


def _dtype_family_from_dtype(dtype: object) -> str:
    dtype = str(dtype)
    if "float16" in dtype or "bfloat16" in dtype:
        return "fp16_bf16"
    if "float32" in dtype:
        return "fp32"
    return "other"


def _infer_int4_matmul_class(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
) -> str | None:
    dtypes = _tensor_dtypes(args)
    if len(dtypes) < 2 or "int8" not in dtypes[1]:
        return None

    shapes = _tensor_shapes(args)
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        return None
    # Packed-int8 signature: packed weight K dimension is original K // 2.
    if shapes[0][-1] != shapes[1][-2] * 2:
        return None
    if kernel is None:
        return None
    try:
        source = inspect.getsource(cast("Any", kernel).kernel.fn).lower()
    except (AttributeError, OSError, TypeError):
        return None
    return "matmul_int4" if "int4" in source else None


def _single_2d_matmul_fact(config_spec: ConfigSpec) -> MatmulFact | None:
    facts = config_spec.matmul_facts
    if len(facts) != 1:
        return None
    fact = facts[0]
    if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
        return None
    if fact.static_m is None or fact.static_n is None or fact.static_k is None:
        return None
    return fact


def _infer_matmul_family_class(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
) -> str | None:
    default = dict(config_spec.default_config())
    block_sizes = default.get("block_sizes")
    if not isinstance(block_sizes, list) or len(block_sizes) != 3:
        return None

    int4_kernel_class = _infer_int4_matmul_class(kernel, args)
    if int4_kernel_class is not None:
        return int4_kernel_class

    if _single_2d_matmul_fact(config_spec) is not None:
        return "matmul"
    return None


def _shape_bucket_from_args(args: Sequence[object]) -> dict[str, object] | None:
    shapes = _tensor_shapes(args)
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        return None
    return {
        "dtype": _dtype_family(args),
        "m_value": shapes[0][-2],
        "n_value": shapes[1][-1],
        "k_value": shapes[0][-1],
    }


def _shape_bucket_from_fact(fact: MatmulFact) -> dict[str, object]:
    assert fact.static_m is not None
    assert fact.static_n is not None
    assert fact.static_k is not None
    return {
        "dtype": _dtype_family_from_dtype(fact.lhs_dtype),
        "m_value": fact.static_m,
        "n_value": fact.static_n,
        "k_value": fact.static_k,
    }


@functools.cache
def _heuristic_rules() -> tuple[dict[str, object], ...]:
    with _RUNTIME_HEURISTICS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    raw_rules = data.get("rules", []) if isinstance(data, dict) else []
    if not isinstance(raw_rules, list):
        return ()
    rules: list[dict[str, object]] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        kernel_class = raw_rule.get("kernel_class")
        shape_bucket = raw_rule.get("shape_bucket")
        if not isinstance(kernel_class, str) or not isinstance(shape_bucket, dict):
            continue
        if not shape_bucket or not set(shape_bucket).issubset(_SHAPE_BUCKET_KEYS):
            continue
        rules.append(raw_rule)
    return tuple(rules)


def _shape_bucket_matches(
    rule_bucket: dict[str, object],
    query_bucket: dict[str, object],
) -> bool:
    for key, value in rule_bucket.items():
        if key in {"k_bucket", "m_bucket", "n_bucket"}:
            dim_value = query_bucket.get(f"{key[0]}_value")
            intervals = value if isinstance(value, list) else [value]
            if not any(
                isinstance(interval, str)
                and isinstance(dim_value, int)
                and _interval_contains(interval, dim_value)
                for interval in intervals
            ):
                return False
            continue
        query_value = query_bucket.get(key)
        if isinstance(value, list) and query_value not in value:
            return False
        if not isinstance(value, list) and query_value != value:
            return False
    return True


def _interval_contains(interval: str, value: int) -> bool:
    if len(interval) < 5 or interval[0] not in "([" or interval[-1] not in ")]":
        return False
    try:
        lower_text, upper_text = interval[1:-1].split(",", maxsplit=1)
        lower = float(lower_text)
        upper = float("inf") if upper_text == "inf" else float(upper_text)
    except ValueError:
        return False

    lower_ok = value >= lower if interval[0] == "[" else value > lower
    upper_ok = value <= upper if interval[-1] == "]" else value < upper
    return lower_ok and upper_ok


def _rules_for_bucket(
    kernel_class: str,
    shape_bucket: dict[str, object],
) -> list[dict[str, object]]:
    matches = [
        rule
        for rule in _heuristic_rules()
        if rule["kernel_class"] == kernel_class
        and _shape_bucket_matches(
            cast("dict[str, object]", rule["shape_bucket"]),
            shape_bucket,
        )
    ]
    matches.sort(
        key=lambda rule: len(cast("dict[str, object]", rule["shape_bucket"])),
        reverse=True,
    )
    return matches


def _materialize_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config:
    flat_fields = config_spec._flat_fields()
    supported = {key: value for key, value in raw.items() if key in flat_fields}
    allowed_pid_types = config_spec.allowed_pid_types
    if (
        "pid_type" in supported
        and allowed_pid_types
        and supported["pid_type"] not in allowed_pid_types
    ):
        supported.pop("pid_type")
    merged = dict(config_spec.default_config())
    merged.update(supported)
    config_spec.normalize(merged, _fix_invalid=True)
    return Config(**cast("dict[str, Any]", merged))


def matmul_heuristic_seed_configs(
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
    max_configs: int,
    kernel_class: str = "matmul",
) -> list[Config]:
    shape_bucket = _shape_bucket_from_args(args)
    if shape_bucket is None:
        return []
    return _seed_configs_for_bucket(
        shape_bucket,
        config_spec=config_spec,
        max_configs=max_configs,
        kernel_class=kernel_class,
    )


def _seed_configs_for_bucket(
    shape_bucket: dict[str, object],
    *,
    config_spec: ConfigSpec,
    max_configs: int,
    kernel_class: str,
) -> list[Config]:
    if max_configs <= 0 or not matmul_heuristics_enabled():
        return []
    rules = _rules_for_bucket(kernel_class, shape_bucket)
    if not rules:
        return []

    seeds: list[Config] = []
    seen: set[str] = set()
    for rule in rules:
        templates = rule.get("templates")
        if not isinstance(templates, list):
            continue
        for template in templates:
            if not isinstance(template, dict):
                continue
            config = _materialize_config(template, config_spec=config_spec)
            key = repr(config)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(config)
            if len(seeds) >= max_configs:
                return seeds
    return seeds


def matmul_heuristic_seed_configs_for_kernel(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
    max_configs: int,
) -> list[Config]:
    if not matmul_heuristics_supported_on_args(args):
        return []
    kernel_class = _infer_matmul_family_class(
        kernel,
        args,
        config_spec=config_spec,
    )
    if kernel_class is None:
        return []
    shape_bucket = None
    if kernel_class == "matmul":
        fact = _single_2d_matmul_fact(config_spec)
        if fact is not None:
            shape_bucket = _shape_bucket_from_fact(fact)
    if shape_bucket is None:
        shape_bucket = _shape_bucket_from_args(args)
    if shape_bucket is None:
        return []
    return _seed_configs_for_bucket(
        shape_bucket,
        config_spec=config_spec,
        max_configs=max_configs,
        kernel_class=kernel_class,
    )


def matmul_heuristic_default_config_for_kernel(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
) -> Config | None:
    configs = matmul_heuristic_seed_configs_for_kernel(
        kernel,
        args,
        config_spec=config_spec,
        max_configs=1,
    )
    return configs[0] if configs else None
