# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Matmul-family autotune seed heuristics.

This intentionally starts with a narrow surface area: matmul-family kernels on
B200, seeded from measured configs stored in
``heuristics/matmul_b200.json``. The seeds are used only as
additional initial-population candidates; regular autotuning still validates
and benchmarks them before selecting a config.

Shape buckets may be partial. When multiple rules match, rules with more
matching bucket fields are used first and JSON order breaks ties.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import textwrap
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


MATMUL_HEURISTICS_ENV = "HELION_AUTOTUNE_MATMUL_HEURISTICS"
MATMUL_HEURISTICS_PATH_ENV = "HELION_AUTOTUNE_MATMUL_HEURISTICS_PATH"
_RUNTIME_HEURISTICS_PATH = (
    Path(__file__).resolve().parent / "heuristics" / "matmul_b200.json"
)

_MATMUL_KERNEL_CLASSES = frozenset(
    {"matmul", "matmul_int4", "matmul_int16", "matmul_fp4"}
)
_QUANTIZED_KERNEL_FINGERPRINTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("matmul_fp4", frozenset({"e2m1", "fp4", "nvfp4"})),
    ("matmul_int4", frozenset({"int4", "pack_int4", "unpack_int4"})),
)
_MATMUL_BINS = {
    "m": [4, 8, 16, 64, 128, 256, 512, 1024, 4096],
    "n": [64, 128, 256, 512, 1024, 4096],
    "k": [64, 128, 256, 512, 1024, 4096, 32768],
}
_SHAPE_BUCKET_KEYS = frozenset({"aspect", "dtype", "k_bin", "m_bin", "n_bin"})


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def matmul_heuristics_enabled() -> bool:
    return _env_flag_enabled(MATMUL_HEURISTICS_ENV, default=True)


def matmul_heuristics_supported_on_args(args: Sequence[object]) -> bool:
    device = extract_device(args)
    device_name = get_device_name(device) if device is not None else None
    return device_name is not None and "B200" in device_name


@functools.cache
def _runtime_heuristics() -> dict[str, object]:
    path = Path(os.environ.get(MATMUL_HEURISTICS_PATH_ENV) or _RUNTIME_HEURISTICS_PATH)
    if not path.exists():
        return {"rules": []}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"rules": []}


def _kernel_source_text(kernel: _AutotunableKernel) -> str:
    try:
        raw_source = inspect.getsource(cast("Any", kernel).kernel.fn)
    except (AttributeError, OSError, TypeError):
        return "# Source unavailable"

    source_lines = textwrap.dedent(raw_source).splitlines()
    start_idx = 0
    while start_idx < len(source_lines) and not source_lines[
        start_idx
    ].lstrip().startswith("def "):
        start_idx += 1
    return "\n".join(source_lines[start_idx:])


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


def _bin_le(value: int | None, bins: Sequence[int]) -> str:
    if value is None:
        return "unknown"
    for bound in bins:
        if value <= bound:
            return f"<={bound}"
    return f">{bins[-1]}"


def _matmul_shape(shapes: Sequence[tuple[int, ...]]) -> tuple[int, int, int] | None:
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        return None
    return (shapes[0][-2], shapes[1][-1], shapes[0][-1])


def _infer_quantized_matmul_class(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
) -> str | None:
    dtypes = _tensor_dtypes(args)
    if len(dtypes) < 2:
        return None
    if "int16" in dtypes[1]:
        return "matmul_int16"
    if "int8" not in dtypes[1]:
        return None

    shapes = _tensor_shapes(args)
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        return None
    # Packed-int8 signature: packed weight K dimension is original K // 2.
    if shapes[0][-1] != shapes[1][-2] * 2:
        return None
    if kernel is None:
        return None

    source = _kernel_source_text(kernel).lower()
    for class_name, markers in _QUANTIZED_KERNEL_FINGERPRINTS:
        if any(marker in source for marker in markers):
            return class_name
    return None


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

    quantized = _infer_quantized_matmul_class(kernel, args)
    if quantized is not None:
        return quantized

    if _single_2d_matmul_fact(config_spec) is not None:
        return "matmul"
    return None


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


def _aspect_bucket(m: int, n: int, k: int) -> str:
    min_dim = min(m, n, k)
    max_dim = max(m, n, k)
    if max_dim / min_dim < 4:
        return "balanced"
    if m == min_dim:
        return "skinny_m"
    if n == min_dim:
        return "skinny_n"
    return "skinny_k"


def _matmul_shape_bucket(args: Sequence[object]) -> dict[str, object] | None:
    shape = _matmul_shape(_tensor_shapes(args))
    if shape is None:
        return None
    m, n, k = shape
    return _matmul_shape_bucket_from_values(
        m,
        n,
        k,
        dtype_family=_dtype_family(args),
    )


def _matmul_shape_bucket_from_values(
    m: int,
    n: int,
    k: int,
    *,
    dtype_family: str,
) -> dict[str, object]:
    return {
        "aspect": _aspect_bucket(m, n, k),
        "dtype": dtype_family,
        "k_bin": _bin_le(k, _MATMUL_BINS["k"]),
        "m_bin": _bin_le(m, _MATMUL_BINS["m"]),
        "n_bin": _bin_le(n, _MATMUL_BINS["n"]),
    }


def _matmul_shape_bucket_from_fact(fact: MatmulFact) -> dict[str, object]:
    assert fact.static_m is not None
    assert fact.static_n is not None
    assert fact.static_k is not None
    return _matmul_shape_bucket_from_values(
        fact.static_m,
        fact.static_n,
        fact.static_k,
        dtype_family=_dtype_family_from_dtype(fact.lhs_dtype),
    )


@functools.cache
def _heuristic_rules() -> tuple[dict[str, object], ...]:
    raw_rules = _runtime_heuristics().get("rules", [])
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
    return all(query_bucket.get(key) == value for key, value in rule_bucket.items())


def _rules_for_bucket(
    kernel_class: str,
    shape_bucket: dict[str, object],
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for rule in _heuristic_rules():
        if rule.get("kernel_class") != kernel_class:
            continue
        rule_bucket = rule.get("shape_bucket")
        if not isinstance(rule_bucket, dict):
            continue
        if _shape_bucket_matches(rule_bucket, shape_bucket):
            matches.append(rule)
    matches.sort(
        key=lambda rule: len(cast("dict[str, object]", rule["shape_bucket"])),
        reverse=True,
    )
    return matches


def _supported_sparse_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> dict[str, object]:
    flat_fields = dict(config_spec._flat_fields())
    supported = {key: value for key, value in raw.items() if key in flat_fields}
    allowed_pid_types = config_spec.allowed_pid_types
    if (
        "pid_type" in supported
        and allowed_pid_types
        and supported["pid_type"] not in allowed_pid_types
    ):
        supported.pop("pid_type")
    return supported


def _materialize_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config:
    supported = _supported_sparse_config(raw, config_spec=config_spec)
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
    shape_bucket: dict[str, object] | None = None,
) -> list[Config]:
    if max_configs <= 0 or not matmul_heuristics_enabled():
        return []
    if kernel_class not in _MATMUL_KERNEL_CLASSES:
        return []
    if shape_bucket is None:
        shape_bucket = _matmul_shape_bucket(args)
    if shape_bucket is None:
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
    if not matmul_heuristics_enabled():
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
            shape_bucket = _matmul_shape_bucket_from_fact(fact)
    return matmul_heuristic_seed_configs(
        args,
        config_spec=config_spec,
        max_configs=max_configs,
        kernel_class=kernel_class,
        shape_bucket=shape_bucket,
    )


def matmul_heuristic_default_config_for_kernel(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
) -> Config | None:
    if not matmul_heuristics_supported_on_args(args):
        return None
    configs = matmul_heuristic_seed_configs_for_kernel(
        kernel,
        args,
        config_spec=config_spec,
        max_configs=1,
    )
    return configs[0] if configs else None
