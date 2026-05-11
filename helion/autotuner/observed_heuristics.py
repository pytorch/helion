# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Data-derived autotune heuristics.

The seed templates in this module are not hand-authored configs.  They are read
from ``helion/autotuner/data/observed_heuristics_b200.json``, which is generated
from measured AOT CSV data.  Runtime code only classifies the active kernel into
a workload class and compact shape bucket, then looks up validated structural
templates for that exact bucket.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .config_validation import validate_sparse_config_shape
from .workload import detect_workload_traits
from .workload import kernel_source_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel
    from .config_spec import ConfigSpec


OBSERVED_HEURISTICS_ENV = "HELION_AUTOTUNE_OBSERVED_HEURISTICS"
OBSERVED_HEURISTIC_SEEDS_ENV = "HELION_AUTOTUNE_OBSERVED_HEURISTIC_SEEDS"
OBSERVED_HEURISTICS_PATH_ENV = "HELION_AUTOTUNE_OBSERVED_HEURISTICS_PATH"
OBSERVED_HEURISTIC_MAX_TEMPLATES_ENV = (
    "HELION_AUTOTUNE_OBSERVED_HEURISTIC_MAX_TEMPLATES"
)
OBSERVED_HEURISTIC_DISABLED_CLASSES_ENV = (
    "HELION_AUTOTUNE_OBSERVED_HEURISTIC_DISABLED_CLASSES"
)
_RUNTIME_HEURISTICS_PATH = (
    Path(__file__).resolve().parent / "data" / "observed_heuristics_b200.json"
)


def _env_value(name: str) -> str | None:
    return os.environ.get(name)


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def observed_heuristics_enabled() -> bool:
    """Return whether data-derived autotune heuristics should be used."""
    return _env_flag_enabled(
        OBSERVED_HEURISTICS_ENV,
        default=True,
    )


def observed_heuristic_seeds_enabled() -> bool:
    """Return whether data-derived first-round seed configs should be used."""
    return observed_heuristics_enabled() and _env_flag_enabled(
        OBSERVED_HEURISTIC_SEEDS_ENV,
        default=True,
    )


@functools.cache
def _runtime_heuristics() -> dict[str, object]:
    """Load the observed-heuristic JSON once per process."""
    path = Path(_env_value(OBSERVED_HEURISTICS_PATH_ENV) or _RUNTIME_HEURISTICS_PATH)
    if not path.exists():
        return {"rules": [], "class_summary": {}}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"rules": [], "class_summary": {}}
    return data


def _disabled_kernel_classes() -> set[str]:
    value = (
        _env_value(OBSERVED_HEURISTIC_DISABLED_CLASSES_ENV) or ""
    )
    return {item.strip() for item in value.split(",") if item.strip()}


def _max_observed_templates() -> int | None:
    value = _env_value(OBSERVED_HEURISTIC_MAX_TEMPLATES_ENV)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _float_field(raw: dict[str, object], name: str, default: float) -> float:
    try:
        return float(raw.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_field(raw: dict[str, object], name: str, default: int) -> int:
    try:
        return int(raw.get(name, default))
    except (TypeError, ValueError):
        return default


def _template_rank_key(
    raw_template: dict[str, object],
) -> tuple[int, int, float, float]:
    return (
        -_int_field(raw_template, "win_count", 0),
        -_int_field(raw_template, "shape_coverage", 0),
        _float_field(raw_template, "geomean_slowdown", float("inf")),
        _float_field(raw_template, "p90_slowdown", float("inf")),
    )


def _has_template_rank_metadata(raw_template: dict[str, object]) -> bool:
    return any(
        name in raw_template
        for name in (
            "win_count",
            "shape_coverage",
            "geomean_slowdown",
            "p90_slowdown",
        )
    )


def _selected_templates(rule: dict[str, object]) -> list[dict[str, object]]:
    templates = rule.get("templates", [])
    if not isinstance(templates, list):
        return []
    valid_templates = [
        raw_template for raw_template in templates if isinstance(raw_template, dict)
    ]
    if any(_has_template_rank_metadata(raw_template) for raw_template in valid_templates):
        ranked = [
            raw_template
            for _index, raw_template in sorted(
                enumerate(valid_templates),
                key=lambda item: (_template_rank_key(item[1]), item[0]),
            )
        ]
    else:
        ranked = valid_templates
    limit = _max_observed_templates()
    if limit is not None:
        ranked = ranked[:limit]
    return ranked


def _default_block_rank(config_spec: ConfigSpec) -> int:
    default = dict(config_spec.default_config())
    block_sizes = default.get("block_sizes")
    if isinstance(block_sizes, list):
        return len(block_sizes)
    return 0


def _has_reduction(config_spec: ConfigSpec) -> bool:
    return bool(config_spec.reduction_loops)


def _flat_fields(config_spec: ConfigSpec) -> dict[str, object]:
    return dict(config_spec._flat_fields())


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
    dtype = dtypes[0] if dtypes else "unknown"
    if "float8" in dtype:
        return "fp8"
    if "float16" in dtype or "bfloat16" in dtype:
        return "fp16_bf16"
    if "float32" in dtype:
        return "fp32"
    if "int" in dtype:
        return "int"
    return "other"


def _bin_le(value: int | None, bins: Sequence[int]) -> str:
    if value is None:
        return "unknown"
    for bound in bins:
        if value <= bound:
            return f"<={bound}"
    return f">{bins[-1]}"


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _is_fp8_matmul(args: Sequence[object]) -> bool:
    return any("float8" in dtype for dtype in _tensor_dtypes(args)[:2])


_QUANTIZED_KERNEL_FINGERPRINTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("matmul_fp4", frozenset({"e2m1", "fp4", "nvfp4"})),
    ("matmul_int4", frozenset({"int4", "pack_int4", "unpack_int4"})),
)


def _infer_quantized_matmul_class(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
) -> str | None:
    """Return a quantized-matmul sub-class if the kernel is one of int4/int16/fp4.

    int16 is unambiguous from arg1 dtype. int4 vs fp4 share the packed-int8
    signature, so we fall back to scanning the kernel source for the
    distinguishing term. Order matters: check fp4 before int4 (the fp4 source
    can mention "int4" only in comments that refer to nibbles).
    """
    dtypes = _tensor_dtypes(args)
    if len(dtypes) < 2:
        return None
    if "int16" in dtypes[1]:
        return "matmul_int16"
    if "int8" not in dtypes[1]:
        return None
    # Packed-int8 shape signature: arg1 dim0 is K//2 of arg0 dim1.
    shapes = _tensor_shapes(args)
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        return None
    if shapes[0][-1] != shapes[1][-2] * 2:
        return None
    if kernel is None:
        return None
    try:
        source = kernel_source_text(kernel).lower()
    except Exception:  # noqa: BLE001
        return None
    for class_name, markers in _QUANTIZED_KERNEL_FINGERPRINTS:
        if any(marker in source for marker in markers):
            return class_name
    return None


def _matmul_shape(shapes: Sequence[tuple[int, ...]]) -> tuple[int, int, int] | None:
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        return None
    return (shapes[0][-2], shapes[1][-1], shapes[0][-1])


def _row_shape(shapes: Sequence[tuple[int, ...]]) -> tuple[int | None, int | None]:
    if not shapes or len(shapes[0]) < 2:
        return None, None
    rows = 1
    for dim in shapes[0][:-1]:
        rows *= dim
    return rows, shapes[0][-1]


def _attention_shape(
    shapes: Sequence[tuple[int, ...]],
) -> tuple[int | None, int | None, int | None, int | None] | None:
    if not shapes or len(shapes[0]) < 4:
        return None
    batch_heads = shapes[0][0] * shapes[0][1]
    q_seq = shapes[0][-2]
    kv_seq = shapes[1][-2] if len(shapes) > 1 and len(shapes[1]) >= 4 else q_seq
    return batch_heads, q_seq, kv_seq, shapes[0][-1]


def _aspect_bucket(m: int | None, n: int | None, k: int | None) -> str:
    values = [value for value in (m, n, k) if value is not None and value > 0]
    if len(values) != 3:
        return "unknown"
    min_dim = min(values)
    max_dim = max(values)
    if max_dim / min_dim < 4:
        return "balanced"
    if m == min_dim:
        return "skinny_m"
    if n == min_dim:
        return "skinny_n"
    return "skinny_k"


def _rank1_reduction_class(
    shapes: Sequence[tuple[int, ...]],
    dtypes: Sequence[str],
    workload_traits: frozenset[str],
) -> str | None:
    if "cross_entropy" in workload_traits:
        return "row_cross_entropy"
    if (
        len(shapes) >= 2
        and len(dtypes) >= 2
        and dtypes[1]
        in {
            "torch.int64",
            "torch.int32",
        }
    ):
        return "row_cross_entropy"

    cols = shapes[0][-1] if shapes and shapes[0] else None

    def rank1_matches_cols(shape: tuple[int, ...]) -> bool:
        return cols is not None and len(shape) == 1 and shape[0] == cols

    if (
        len(shapes) >= 3
        and rank1_matches_cols(shapes[1])
        and rank1_matches_cols(shapes[2])
    ):
        return "row_norm_layer"
    if len(shapes) >= 2 and rank1_matches_cols(shapes[1]):
        return "row_norm_rms"
    if "softmax" in workload_traits or (
        "exp" in workload_traits and "sum_reduction" in workload_traits
    ):
        return "row_softmax"
    return None


def _looks_dense_elementwise(shapes: Sequence[tuple[int, ...]]) -> bool:
    if not shapes:
        return False
    output_shape = shapes[0]
    if len(shapes) == 1:
        return True
    return all(shape == output_shape for shape in shapes[1:])


def classify_runtime_kernel(
    args: Sequence[object],
    *,
    workload_traits: frozenset[str],
    config_spec: ConfigSpec,
    kernel: _AutotunableKernel | None = None,
) -> str | None:
    """Classify the runtime kernel into the CSV-derived workload taxonomy.

    ``kernel`` is optional and used only for fingerprinting specific quantized
    matmul variants (int4 vs fp4, which are otherwise indistinguishable from
    arg shapes/dtypes alone). Classification works without it for every class
    except ``matmul_int4``/``matmul_fp4``, which will fall back to ``matmul``.
    """
    block_rank = _default_block_rank(config_spec)
    shapes = _tensor_shapes(args)
    if "attention_reduction" in workload_traits and block_rank == 3:
        return "attention"
    if "matmul" in workload_traits and "split_k" in _flat_fields(config_spec):
        return "split_k_matmul"
    if "matmul" in workload_traits and block_rank == 4:
        if len(shapes) >= 2 and len(shapes[0]) == 3 and len(shapes[1]) == 3:
            return "batched_matmul"
        return "grouped_matmul"
    if "matmul" in workload_traits and block_rank == 3:
        if _is_fp8_matmul(args):
            return "matmul_fp8"
        quantized = _infer_quantized_matmul_class(kernel, args)
        if quantized is not None:
            return quantized
        return "matmul"
    # Quantized matmuls (int4/fp4) don't call hl.dot directly — they do a
    # manual outer-product + sum-reduction over unpacked weights — so they
    # emit {"reduction", "sum_reduction"} instead of {"matmul"}. Classify by
    # shape signature + source fingerprint.
    if (
        block_rank == 3
        and "sum_reduction" in workload_traits
        and "reduction" in workload_traits
    ):
        quantized = _infer_quantized_matmul_class(kernel, args)
        if quantized is not None:
            return quantized
    if _has_reduction(config_spec) and block_rank == 1:
        return _rank1_reduction_class(shapes, _tensor_dtypes(args), workload_traits)
    if (
        not _has_reduction(config_spec)
        and block_rank == 1
        and _looks_dense_elementwise(shapes)
    ):
        return "elementwise"
    return None


def _shape_bucket_for_class(
    kernel_class: str, args: Sequence[object]
) -> dict[str, object]:
    shapes = _tensor_shapes(args)
    dtype_family = _dtype_family(args)
    if kernel_class == "attention":
        attention_shape = _attention_shape(shapes)
        batch_heads, q_seq, kv_seq, head_dim = (
            attention_shape
            if attention_shape is not None
            else (None, None, None, None)
        )
        bucket = {
            "batch_heads_bin": _bin_le(batch_heads, [32, 64, 128, 256]),
            "dtype": dtype_family,
            "head_dim_bin": _bin_le(head_dim, [64, 128, 256]),
            "seq_bin": _bin_le(q_seq, [1, 16, 1024, 2048, 4096, 8192, 16384]),
        }
        if q_seq != kv_seq:
            bucket["kv_seq_bin"] = _bin_le(kv_seq, [1024, 2048, 4096, 8192, 16384])
        return bucket
    if kernel_class in {
        "matmul",
        "matmul_fp8",
        "grouped_matmul",
        "matmul_int4",
        "matmul_int16",
        "matmul_fp4",
    }:
        matmul_shape = _matmul_shape(shapes)
        m, n, k = matmul_shape if matmul_shape is not None else (None, None, None)
        return {
            "aspect": _aspect_bucket(m, n, k),
            "dtype": dtype_family,
            "k_bin": _bin_le(k, [64, 128, 256, 512, 1024, 4096, 32768]),
            "m_bin": _bin_le(m, [4, 8, 16, 64, 128, 256, 512, 1024, 4096]),
            "n_bin": _bin_le(n, [64, 128, 256, 512, 1024, 4096]),
        }
    if kernel_class.startswith("row_"):
        rows, cols = _row_shape(shapes)
        return {
            "cols_bin": _bin_le(cols, [512, 1024, 2048, 4096, 8192, 16384, 32768]),
            "dtype": dtype_family,
            "rows_bin": _bin_le(rows, [512, 2048, 4096, 16384, 65536, 262144]),
        }
    if kernel_class == "elementwise":
        numel = _numel(shapes[0]) if shapes else None
        return {
            "dtype": dtype_family,
            "numel_bin": _bin_le(numel, [4096, 65536, 1048576, 16777216, 134217728]),
        }
    return {"dtype": dtype_family}


def _fallback_group_for_class(
    kernel_class: str, args: Sequence[object]
) -> str | None:
    """Coarse shape-group label for fallback lookup.

    Used only when exact-bucket rule lookup misses. The grouping is
    deliberately much coarser than ``_shape_bucket_for_class`` —
    enough partitions to capture the dominant config-shape
    correlation (skinny axes, balanced-vs-rect), not fine enough
    to need many archive shapes per group.

    Returns None for kernel classes without a defined grouping. The
    lookup will simply skip fallbacks in that case (safe default).
    """
    shapes = _tensor_shapes(args)
    if kernel_class in {
        "matmul",
        "matmul_fp8",
        "grouped_matmul",
        "matmul_int4",
        "matmul_int16",
        "matmul_fp4",
    }:
        matmul_shape = _matmul_shape(shapes)
        if matmul_shape is None:
            return None
        m, n, k = matmul_shape
        if m is None or n is None or k is None:
            return None
        if m <= 256:
            return "small_m"
        if n <= 256:
            return "small_n"
        if k <= 256:
            return "small_k"
        dims = [m, n, k]
        if max(dims) / max(1, min(dims)) < 2:
            return "balanced"
        return "rect"
    if kernel_class.startswith("row_"):
        rows, cols = _row_shape(shapes)
        if rows is None or cols is None:
            return None
        if rows <= 512:
            return "short"
        if cols <= 1024:
            return "narrow"
        if cols >= 8192:
            return "wide"
        return "square"
    if kernel_class == "elementwise":
        numel = _numel(shapes[0]) if shapes else None
        if numel is None:
            return None
        if numel <= 65536:
            return "tiny"
        if numel <= 1048576:
            return "mid"
        return "huge"
    if kernel_class == "attention":
        attention_shape = _attention_shape(shapes)
        if attention_shape is None:
            return None
        batch_heads, q_seq, _kv_seq, head_dim = attention_shape
        if q_seq is None or head_dim is None:
            return None
        if q_seq <= 1024:
            return "short_seq"
        if q_seq >= 8192:
            return "long_seq"
        if head_dim is not None and head_dim <= 64:
            return "small_head"
        return "mid_seq"
    return None


@functools.cache
def _rules_by_key() -> dict[str, dict[str, object]]:
    data = _runtime_heuristics()
    raw_rules = data.get("rules", [])
    if not isinstance(raw_rules, list):
        return {}
    rules: dict[str, dict[str, object]] = {}
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        kernel_class = raw_rule.get("kernel_class")
        shape_bucket = raw_rule.get("shape_bucket")
        if not isinstance(kernel_class, str) or not isinstance(shape_bucket, dict):
            continue
        rules[f"{kernel_class}:{_stable_json(shape_bucket)}"] = raw_rule
    return rules


def _find_rule(
    kernel_class: str,
    shape_bucket: dict[str, object],
) -> dict[str, object] | None:
    if kernel_class in _disabled_kernel_classes():
        return None
    return _rules_by_key().get(f"{kernel_class}:{_stable_json(shape_bucket)}")


@functools.cache
def _fallbacks_by_class() -> dict[str, dict[str, dict[str, object]]]:
    """Return the ``fallbacks`` map from the loaded JSON, or empty."""
    data = _runtime_heuristics()
    raw = data.get("fallbacks", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, dict[str, object]]] = {}
    for kernel_class, group_map in raw.items():
        if not isinstance(kernel_class, str) or not isinstance(group_map, dict):
            continue
        clean_groups: dict[str, dict[str, object]] = {}
        for group, entry in group_map.items():
            if isinstance(group, str) and isinstance(entry, dict):
                clean_groups[group] = entry
        if clean_groups:
            out[kernel_class] = clean_groups
    return out


def _find_fallback(
    kernel_class: str,
    group: str | None,
) -> dict[str, object] | None:
    """Look up the fallback entry for a (kernel_class, group) pair.

    Returns the raw entry (same shape as a rule's ``templates[i]``)
    or None if no fallback is defined.
    """
    if group is None:
        return None
    if kernel_class in _disabled_kernel_classes():
        return None
    return _fallbacks_by_class().get(kernel_class, {}).get(group)


def _matched_rule(
    args: Sequence[object],
    *,
    workload_traits: frozenset[str],
    config_spec: ConfigSpec,
    kernel: _AutotunableKernel | None = None,
) -> tuple[str | None, dict[str, object], dict[str, object] | None]:
    kernel_class = classify_runtime_kernel(
        args,
        workload_traits=workload_traits,
        config_spec=config_spec,
        kernel=kernel,
    )
    if kernel_class is None:
        return None, {}, None
    shape_bucket = _shape_bucket_for_class(kernel_class, args)
    return kernel_class, shape_bucket, _find_rule(kernel_class, shape_bucket)


def _supported_sparse_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> dict[str, object]:
    flat_fields = _flat_fields(config_spec)
    supported = {key: value for key, value in raw.items() if key in flat_fields}
    if (
        "pid_type" in supported
        and supported["pid_type"] not in config_spec.allowed_pid_types
    ):
        supported.pop("pid_type")
    validate_sparse_config_shape(supported, config_spec=config_spec)
    return supported


def _materialize_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config:
    import helion

    supported = _supported_sparse_config(raw, config_spec=config_spec)
    merged = dict(config_spec.default_config())
    merged.update(supported)
    config_spec.normalize(merged, _fix_invalid=True)
    return helion.Config(**merged)


def observed_heuristic_seed_configs(
    args: Sequence[object],
    *,
    workload_traits: frozenset[str],
    config_spec: ConfigSpec,
    max_configs: int,
    kernel: _AutotunableKernel | None = None,
) -> list[Config]:
    """Return valid CSV-derived seed configs for this config space."""
    if max_configs <= 0 or not observed_heuristic_seeds_enabled():
        return []

    kernel_class, _shape_bucket, rule = _matched_rule(
        args,
        workload_traits=workload_traits,
        config_spec=config_spec,
        kernel=kernel,
    )
    templates: list[dict[str, object]]
    if rule is not None:
        templates = _selected_templates(rule)
    elif kernel_class is not None:
        # Exact-bucket lookup missed — try the per-kernel-class fallback.
        group = _fallback_group_for_class(kernel_class, args)
        fallback_entry = _find_fallback(kernel_class, group)
        if fallback_entry is None:
            return []
        templates = [fallback_entry]
    else:
        return []

    seeds: list[Config] = []
    seen: set[str] = set()
    for raw_template in templates:
        if not isinstance(raw_template, dict):
            continue
        raw = raw_template.get("template")
        if not isinstance(raw, dict):
            continue
        config = _materialize_config(raw, config_spec=config_spec)
        key = repr(config)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(config)
        if len(seeds) >= max_configs:
            break
    return seeds


def observed_heuristic_seed_configs_for_kernel(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
    max_configs: int,
) -> list[Config]:
    """Return observed seed configs after inferring workload traits from a kernel."""
    if max_configs <= 0 or not observed_heuristic_seeds_enabled():
        return []

    return observed_heuristic_seed_configs(
        args,
        workload_traits=detect_workload_traits(kernel, config_spec=config_spec),
        config_spec=config_spec,
        max_configs=max_configs,
        kernel=kernel,
    )


def observed_heuristic_default_config(
    kernel: _AutotunableKernel | None,
    args: Sequence[object],
    *,
    config_spec: ConfigSpec,
) -> Config | None:
    """Return the first observed seed config for no-autotune/default execution."""
    configs = observed_heuristic_seed_configs_for_kernel(
        kernel,
        args,
        config_spec=config_spec,
        max_configs=1,
    )
    return configs[0] if configs else None
