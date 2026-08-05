"""
Search space analysis and logging for the Helion autotuner.

This module provides tools to analyze and log the valid search space
for autotuning, including:
- Which config keys are enabled/disabled and why
- The size of each search dimension
- Total search space size (when computable)
- Coverage metrics (configs tested vs. total space)
- Per-feature exploration tracking
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..autotuner.config_spec import ConfigSpec
    from ..autotuner.config_spec import SearchDimensionInfo
    from ..runtime.config import Config

log = logging.getLogger(__name__)


def canonical_config_id(config: Config) -> str:
    """Stable 16-hex id for a config: sha256 of its canonical (sorted) JSON.

    The same config always maps to the same id, so it is safe as a set key for
    counting distinct configs. Shared with the autotuner dataset logger.
    """
    canonical = json.dumps(config.config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass
class DimensionStats:
    """One search dimension: its possible values and what was observed.

    Merges the static description of a dimension (cardinality / enumerable
    values / why it is constrained) with the dynamic exploration record
    (distinct observed values), so coverage is computed from a single object.
    """

    name: str
    dim_type: str  # "discrete" | "categorical"
    cardinality: int  # number of possible values (0 == inapplicable)
    possible_values: list[object] | None = None  # explicit values if enumerable
    constrained_by: str | None = None
    observed_values: set[object] = dataclasses.field(default_factory=set)

    def observe(self, value: object) -> None:
        try:
            self.observed_values.add(value)
        except TypeError:
            # unhashable (e.g. nested list) -- coerce to repr for tracking
            self.observed_values.add(repr(value))

    @property
    def observed_count(self) -> int:
        """Distinct observed values, clamped to ``cardinality``.

        The observed set can exceed a projected-size estimate, so clamp to keep
        the displayed fraction sane (never ``7/4``).
        """
        if not self.cardinality:
            return len(self.observed_values)
        return min(len(self.observed_values), self.cardinality)

    @property
    def coverage_percent(self) -> float:
        if not self.cardinality:
            return 0.0
        return (self.observed_count / self.cardinality) * 100

    @property
    def applicable(self) -> bool:
        return self.cardinality > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.dim_type,
            "cardinality": self.cardinality,
            "possible_values": (
                self.possible_values if self.cardinality <= 100 else None
            ),
            "constrained_by": self.constrained_by,
            "observed_count": self.observed_count,
            "observed_values": sorted(self.observed_values, key=repr),
            "coverage_percent": round(self.coverage_percent, 2),
        }

    def to_summary_line(self) -> str:
        if not self.cardinality:
            return f"{self.name}: no autotunable choices for this kernel"
        return (
            f"{self.name}: {self.observed_count}/{self.cardinality} "
            f"options tested ({self.coverage_percent:.1f}%)"
        )


@dataclasses.dataclass
class SearchSpaceReport:
    """Unified search-space + exploration report for one kernel.

    Combines the search-space description (identity, dimensions, restrictions,
    total size) with the exploration outcome (configs attempted/valid/invalid,
    timing, algorithm) and owns serialization + logging + saving.
    """

    # Identity
    kernel_name: str
    specialization_key: str | None
    backend: str
    hardware: str | None

    # Search space structure
    dimensions: list[DimensionStats]
    total_search_space_size: int | None  # exact product, or None if unknown
    disabled_features: list[str]  # "feature: reason" strings
    shape_constraints: list[str]

    # Exploration outcome (populated at finish()). ``configs_tested`` counts
    # distinct configs (by canonical id) and is the denominator for coverage.
    # ``explored_valid``/``explored_invalid`` are raw attempt counts (a config
    # re-attempted across generations counts each time), kept on the same scale
    # so ``explored_valid_percent`` is meaningful.
    search_algorithm: str = ""
    elapsed_seconds: float = 0.0
    configs_tested: int = 0
    explored_valid: int = 0
    explored_invalid: int = 0

    @property
    def enabled_features(self) -> list[str]:
        return [d.name for d in self.dimensions]

    @property
    def explored_total(self) -> int:
        return self.explored_valid + self.explored_invalid

    @property
    def explored_valid_percent(self) -> float:
        total = self.explored_total
        return (self.explored_valid / total) * 100 if total else 0.0

    @property
    def applicable_dimensions(self) -> list[DimensionStats]:
        return [d for d in self.dimensions if d.applicable]

    @property
    def avg_feature_coverage(self) -> float:
        applicable = self.applicable_dimensions
        if not applicable:
            return 0.0
        return sum(d.coverage_percent for d in applicable) / len(applicable)

    @property
    def min_feature_coverage(self) -> float:
        applicable = self.applicable_dimensions
        return min((d.coverage_percent for d in applicable), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_name": self.kernel_name,
            "specialization_key": self.specialization_key,
            "backend": self.backend,
            "hardware": self.hardware,
            "total_search_space_size": (
                str(self.total_search_space_size)
                if self.total_search_space_size is not None
                else "unknown"
            ),
            "search_algorithm": self.search_algorithm,
            "elapsed_seconds": self.elapsed_seconds,
            "configs_tested": self.configs_tested,
            "explored_valid": self.explored_valid,
            "explored_invalid": self.explored_invalid,
            "explored_total": self.explored_total,
            "explored_valid_percent": round(self.explored_valid_percent, 2),
            "coverage_percent": (
                round((self.configs_tested / self.total_search_space_size) * 100, 6)
                if self.total_search_space_size
                else None
            ),
            "avg_feature_coverage": round(self.avg_feature_coverage, 2),
            "min_feature_coverage": round(self.min_feature_coverage, 2),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "disabled_features": self.disabled_features,
            "shape_constraints": self.shape_constraints,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """Log a human-readable search-space + exploration summary."""
        size_str = (
            f"{self.total_search_space_size:,}"
            if self.total_search_space_size is not None
            else "unknown"
        )
        logger.log(level, f"Search space for {self.kernel_name}:")
        logger.log(
            level, f"  Backend: {self.backend}, Hardware: {self.hardware or 'unknown'}"
        )
        logger.log(level, f"  Total search space size: {size_str}")
        logger.log(level, f"  Search dimensions: {len(self.dimensions)}")

        if self.disabled_features:
            logger.log(level, f"  Disabled features ({len(self.disabled_features)}):")
            # Collapse features disabled solely because the selected backend
            # doesn't support them into one line; list specific reasons.
            generic_suffix = f": Not supported by {self.backend} backend"
            backend_specific = [
                feat[: -len(generic_suffix)]
                for feat in self.disabled_features
                if feat.endswith(generic_suffix)
            ]
            other = [
                feat
                for feat in self.disabled_features
                if not feat.endswith(generic_suffix)
            ]
            for feat in other[:10]:
                logger.log(level, f"    - {feat}")
            if len(other) > 10:
                logger.log(level, f"    ... and {len(other) - 10} more")
            if backend_specific:
                logger.log(
                    level,
                    f"    - {len(backend_specific)} feature(s) not supported by "
                    f"{self.backend} backend (e.g. {', '.join(backend_specific[:3])})",
                )

        if self.shape_constraints:
            logger.log(level, f"  Shape constraints ({len(self.shape_constraints)}):")
            for constraint in self.shape_constraints[:5]:
                logger.log(level, f"    - {constraint}")
            if len(self.shape_constraints) > 5:
                logger.log(level, f"    ... and {len(self.shape_constraints) - 5} more")

        if not self.search_algorithm:
            return

        # Exploration outcome.
        logger.log(level, f"  Search algorithm: {self.search_algorithm}")
        logger.log(
            level,
            f"  Time: {self.elapsed_seconds:.1f}s, "
            f"Configs tested: {self.configs_tested:,}",
        )
        if self.explored_total > 0:
            logger.log(
                level,
                f"  Configs attempted: {self.explored_total:,} "
                f"({self.explored_valid:,} valid, "
                f"{self.explored_invalid:,} invalid, "
                f"{self.explored_valid_percent:.1f}% valid)",
            )
        if self.total_search_space_size:
            coverage = (self.configs_tested / self.total_search_space_size) * 100
            logger.log(level, f"  Overall search space coverage: {coverage:.6f}%")
        logger.log(level, "  Per-feature exploration:")
        logger.log(
            level, f"    Average feature coverage: {self.avg_feature_coverage:.1f}%"
        )
        logger.log(
            level, f"    Minimum feature coverage: {self.min_feature_coverage:.1f}%"
        )
        for dim in sorted(self.dimensions, key=lambda d: d.coverage_percent):
            logger.log(level, f"    - {dim.to_summary_line()}")
        poor = [d for d in self.applicable_dimensions if d.coverage_percent < 50.0]
        if poor:
            logger.log(level, "\n  Features with <50% exploration:")
            for dim in poor:
                logger.log(
                    level,
                    f"    - {dim.name}: only {dim.observed_count} "
                    f"of {dim.cardinality} values tested",
                )

    def save(self, output_path: str, cache_hash: str | None = None) -> str:
        """Best-effort write of the report as one JSON document.

        Returns the written path, or an empty string on failure. Never raises:
        search-space logging is diagnostic and must not crash the autotuner.
        """
        try:
            path = resolve_report_path(
                output_path,
                kernel_name=self.kernel_name,
                cache_hash=cache_hash,
            )
            path.write_text(self.to_json())
            return str(path)
        except Exception:
            log.debug(
                "Failed to save search space report to %r", output_path, exc_info=True
            )
            return ""


class SearchSpaceTracker:
    """Record which config values are tested during autotuning.

    Owns the exploration counters and feeds observed values into the report's
    :class:`DimensionStats`. ``finish()`` returns the completed report.
    """

    def __init__(self, report: SearchSpaceReport) -> None:
        self.report = report
        self._dimensions = {d.name: d for d in report.dimensions}
        self._seen_keys: set[str] = set()
        # Raw count of valid configs recorded (duplicates counted), kept on the
        # same scale as invalid_config_count for the validity breakdown.
        self.valid_config_count: int = 0
        self.invalid_config_count: int = 0

    def record_config(self, config: Config) -> None:
        """Record a tested (valid) configuration and its observed values."""
        self.valid_config_count += 1
        self._seen_keys.add(canonical_config_id(config))
        for name, dim in self._dimensions.items():
            value = _extract_feature_value(config, name)
            if value is not None:
                dim.observe(value)

    def record_invalid(self, count: int = 1) -> None:
        """Record ``count`` candidate configs rejected as InvalidConfig."""
        if count > 0:
            self.invalid_config_count += count

    def finish(
        self, search_algorithm: str, elapsed_seconds: float
    ) -> SearchSpaceReport:
        """Populate the report's exploration outcome and return it."""
        self.report.search_algorithm = search_algorithm
        self.report.elapsed_seconds = elapsed_seconds
        # configs_tested = distinct configs (coverage denominator);
        # explored_valid/invalid = raw attempt counts (validity breakdown).
        self.report.configs_tested = len(self._seen_keys)
        self.report.explored_valid = self.valid_config_count
        self.report.explored_invalid = self.invalid_config_count
        return self.report


def _extract_feature_value(config: Config, feature_name: str) -> object:
    """Extract a feature value from a Config object (None if not applicable)."""
    # List-valued config attributes are coerced to a hashable tuple.
    if feature_name in ("block_sizes", "loop_orders", "l2_groupings", "flatten_loops"):
        return tuple(getattr(config, feature_name))
    # ``pallas_loop_type`` is stored in the config dict, not as an attribute.
    if feature_name == "pallas_loop_type":
        return config.get("pallas_loop_type")
    # Generic scalar tunables (pid_type, num_warps, num_stages, maxnreg, ...).
    return getattr(config, feature_name, None)


def analyze_search_space(
    config_spec: ConfigSpec,
    kernel_name: str = "",
    specialization_key: str | None = None,
    hardware: str | None = None,
) -> SearchSpaceReport:
    """Analyze the valid search space for a kernel's config spec.

    This examines which features are enabled/disabled based on:
    - Backend capabilities (via supports_config_key)
    - Hardware constraints (e.g., maxnreg only on CUDA)
    - Kernel properties (e.g., epilogue_subtile only for matmul-like)
    - Shape-dependent constraints (e.g., block_size limits)

    Args:
        config_spec: The configuration specification to analyze
        kernel_name: Optional kernel name for logging
        specialization_key: Optional specialization key
        hardware: Optional hardware identifier

    Returns:
        A SearchSpaceReport describing the valid search space
    """
    from ..autotuner.config_spec import VALID_KEYS
    from ..autotuner.config_spec import VALID_PID_TYPES

    dimensions: list[DimensionStats] = []
    disabled_features: list[str] = []
    shape_constraints: list[str] = []

    flat_fields = config_spec._flat_fields()
    for info in config_spec.iter_search_dimensions():
        dim = _dimension_from_info(info, config_spec)
        if dim is not None:
            dimensions.append(dim)

    for key in sorted(VALID_KEYS):
        if key in flat_fields:
            continue
        if config_spec.supports_config_key(key):
            # Supported but not materialized as a tunable field for this kernel.
            continue
        reason = _get_disable_reason(config_spec, key)
        disabled_features.append(f"{key}: {reason}")

    # Check shape-dependent constraints
    if config_spec.block_sizes:
        for i, spec in enumerate(config_spec.block_sizes):
            if spec.autotuner_min != spec.min_size:
                shape_constraints.append(
                    f"block_size[{i}] autotuner range constrained to "
                    f"[{spec.autotuner_min}, {spec.max_size}] "
                    f"(natural min_size={spec.min_size})"
                )

    # Surface pid_types that were disabled for this kernel. pid_type stays an
    # enabled feature (some values remain), so this won't appear in
    # disabled_features; report it as a constraint instead.
    if config_spec.supports_config_key("pid_type"):
        disabled_pid_types = [
            pt for pt in VALID_PID_TYPES if pt not in config_spec.allowed_pid_types
        ]
        if disabled_pid_types:
            # Only annotate pid_types that are *currently* disabled. Iterating
            # disabled_pid_types (derived from allowed_pid_types) means a stale
            # reason left for a later re-allowed pid_type is ignored, so the two
            # structures can't drift into wrong output.
            reasons = config_spec.disallowed_pid_type_reasons
            disabled_desc = ", ".join(
                f"{pt} ({reasons[pt]})" if pt in reasons else pt
                for pt in disabled_pid_types
            )
            shape_constraints.append(
                f"pid_type restricted to {list(config_spec.allowed_pid_types)} "
                f"(disabled: {disabled_desc})"
            )

    # Surface non-pid_type search-space restrictions (e.g. tcgen05 cluster_m /
    # ab_stages / narrowing) recorded as (feature, reason) pairs at compile time.
    for feature, reason in getattr(config_spec, "restriction_reasons", []):
        shape_constraints.append(f"{feature} ({reason})")

    if config_spec.cute_flash_search_enabled:
        shape_constraints.append(
            "CuTe flash attention search enabled (restricted surface)"
        )

    if config_spec.epilogue_subtile_autotune_choices is not None:
        shape_constraints.append(
            f"epilogue_subtile enabled for k_hint={config_spec.epilogue_subtile_k_hint}"
        )

    # Total search space size: exact product of per-dimension cardinalities.
    # A dimension whose cardinality is unknown (0 sentinel from a fragment that
    # can't report one) makes the total unknown; otherwise the product is exact
    # (Python big ints, so large attention-style spaces report a real number
    # rather than being truncated).
    product = 1
    total_size: int | None = None
    if all(dim.cardinality != 0 for dim in dimensions):
        for dim in dimensions:
            product *= dim.cardinality
        total_size = product

    return SearchSpaceReport(
        kernel_name=kernel_name,
        specialization_key=specialization_key,
        backend=config_spec.backend_name,
        hardware=hardware,
        dimensions=dimensions,
        total_search_space_size=total_size,
        disabled_features=disabled_features,
        shape_constraints=shape_constraints,
    )


def _dimension_from_info(
    info: SearchDimensionInfo,
    config_spec: ConfigSpec,
) -> DimensionStats | None:
    """Build a :class:`DimensionStats` from a spec-provided dimension.

    Cardinality and values come from the config spec (fragment-derived); this
    only attaches the human-readable ``constrained_by`` annotation, which is
    spec-state specific and not encoded in the fragment itself.
    """
    from ..autotuner.config_spec import VALID_PID_TYPES

    cardinality = info.cardinality if info.cardinality is not None else 0
    values = info.values if info.values is not None and cardinality <= 100 else None

    constrained_by: str | None = None
    if info.name == "block_sizes" and config_spec.tensor_numel_constraints:
        constrained_by = "tensor numel constraints"
    elif info.name == "pid_type":
        disabled = [
            pt for pt in VALID_PID_TYPES if pt not in config_spec.allowed_pid_types
        ]
        if disabled:
            constrained_by = (
                f"{len(disabled)} pid_type(s) disabled by kernel "
                f"({', '.join(disabled)})"
            )
    elif info.is_sequence and info.num_items:
        constrained_by = f"{info.num_items} loop(s)"

    return DimensionStats(
        name=info.name,
        dim_type="discrete" if info.is_sequence else "categorical",
        cardinality=cardinality,
        possible_values=values,
        constrained_by=constrained_by,
    )


def _get_disable_reason(config_spec: ConfigSpec, key: str) -> str:
    """Get human-readable reason why a config key is disabled."""
    backend_name = config_spec.backend_name

    if backend_name == "pallas":
        if key in ("num_warps", "num_stages"):
            return "Pallas backend (handled by XLA)"

    elif backend_name == "triton":
        if key == "pallas_loop_type":
            return "Triton backend (no Pallas loops)"
        if key == "pallas_pre_broadcast":
            return "Triton backend"
        if key == "num_threads":
            return "Triton backend (uses num_warps)"

    elif backend_name == "cute":
        if key == "num_threads" and not config_spec.target_device_capability:
            return "CuTe requires CUDA target"

    if key == "epilogue_subtile":
        if not config_spec.epilogue_subtile_candidate_enabled:
            return "Not a matmul-like kernel"
        if config_spec.epilogue_subtile_k_hint < 1024:
            return f"k_hint={config_spec.epilogue_subtile_k_hint} too small (<1024)"

    if key == "pallas_loop_type":
        if not config_spec.has_pallas_inner_loops:
            return "No Pallas inner loops in kernel"

    # Check if it's a backend-specific key
    from ..autotuner.config_spec import BACKEND_SPECIFIC_KEYS

    if key in BACKEND_SPECIFIC_KEYS:
        return f"Not supported by {backend_name} backend"

    return f"Not supported by {backend_name} backend"


def log_search_space_comparison(
    logger: logging.Logger,
    report: SearchSpaceReport,
) -> None:
    """Log a search-space vs. searched comparison banner.

    All figures come from ``report`` so the banner, the report summary, and the
    saved JSON never disagree.
    """
    logger.info("=" * 60)
    logger.info("Autotune Search Space Analysis")
    logger.info("=" * 60)

    report.log_summary(logger, logging.INFO)

    total = report.total_search_space_size
    configs_tested = report.configs_tested
    logger.info("\nSearch Coverage:")
    logger.info(f"  Configs tested: {configs_tested:,}")
    if total is not None and total > 0:
        coverage = (configs_tested / total) * 100
        logger.info(f"  Total space: {total:,}")
        logger.info(f"  Coverage: {coverage:.6f}%")
        logger.info(f"  Search algorithm: {report.search_algorithm}")
        logger.info(f"  Time elapsed: {report.elapsed_seconds:.1f}s")
    else:
        logger.info(
            "  Total space: unknown (a dimension's cardinality could not be determined)"
        )
        logger.info(f"  Search algorithm: {report.search_algorithm}")
        logger.info(f"  Time elapsed: {report.elapsed_seconds:.1f}s")

    logger.info("=" * 60)


def resolve_report_path(
    output_path: str,
    kernel_name: str,
    cache_hash: str | None,
    default_filename: str = "autotune_search_space.json",
) -> Path:
    """Resolve ``output_path`` to a per-kernel report file path.

    Directory paths (existing or trailing-separator) get ``default_filename``.
    The kernel name and the autotuner's stable cache hash are injected into the
    filename stem so each kernel/shape writes a distinct file (matching the
    ``.best_config`` cache key). Re-tuning the same kernel/shape reuses the hash
    and intentionally rewrites its file; without a hash, a numeric suffix guards
    against clobbering an unrelated file.

    Example: ``analysis.json`` -> ``analysis.my_kernel.3f9a1c2e.json``.
    """

    def token(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")[:64]

    path = Path(output_path)
    if path.is_dir() or output_path.endswith(("/", "\\")):
        path = path / default_filename
    path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        path.stem,
        *(t for t in (token(kernel_name), token(cache_hash or "")) if t),
    ]
    candidate = path.with_name(f"{'.'.join(parts)}{path.suffix}")
    if token(cache_hash or ""):
        return candidate
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{'.'.join(parts)}.{counter}{path.suffix}")
        counter += 1
    return candidate
