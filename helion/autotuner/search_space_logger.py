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
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from ..runtime.config import Config
    from ..autotuner.config_spec import ConfigSpec
    from ..autotuner.block_id_sequence import BlockIdSequence

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SearchSpaceDimension:
    """Description of one dimension of the search space."""

    name: str
    dim_type: str  # "discrete", "continuous", "boolean", "categorical"
    size: int  # Number of possible values
    values: list[object] | None = None  # Explicit values if small enough
    constrained_by: str | None = None  # Why this dimension is restricted


@dataclasses.dataclass
class SearchSpaceSummary:
    """Summary of the valid search space for one kernel."""

    # Identity
    kernel_name: str
    specialization_key: str | None

    # Backend/hardware constraints
    backend: str
    hardware: str | None

    # Search space structure
    dimensions: list[SearchSpaceDimension]
    total_search_space_size: int | None  # None if too large/infinite

    # What's enabled/disabled
    enabled_features: list[str]
    disabled_features: list[str]  # (feature, reason) pairs

    # Shape-dependent constraints
    shape_constraints: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_name": self.kernel_name,
            "specialization_key": self.specialization_key,
            "backend": self.backend,
            "hardware": self.hardware,
            "dimensions": [
                {
                    "name": d.name,
                    "type": d.dim_type,
                    "size": d.size,
                    "values": d.values if d.size <= 100 else None,
                    "constrained_by": d.constrained_by,
                }
                for d in self.dimensions
            ],
            "total_search_space_size": (
                str(self.total_search_space_size)
                if self.total_search_space_size is not None
                else "infinite"
            ),
            "enabled_features": self.enabled_features,
            "disabled_features": self.disabled_features,
            "shape_constraints": self.shape_constraints,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """Log a human-readable summary."""
        size_str = (
            f"{self.total_search_space_size:,}"
            if self.total_search_space_size is not None
            else "infinite"
        )
        logger.log(level, f"Search space for {self.kernel_name}:")
        logger.log(level, f"  Backend: {self.backend}, Hardware: {self.hardware or 'unknown'}")
        logger.log(level, f"  Total search space size: {size_str}")
        logger.log(level, f"  Search dimensions: {len(self.dimensions)}")
        if self.disabled_features:
            logger.log(level, f"  Disabled features ({len(self.disabled_features)}):")
            for feat in self.disabled_features[:10]:
                logger.log(level, f"    - {feat}")
            if len(self.disabled_features) > 10:
                logger.log(level, f"    ... and {len(self.disabled_features) - 10} more")
        if self.shape_constraints:
            logger.log(level, f"  Shape constraints ({len(self.shape_constraints)}):")
            for constraint in self.shape_constraints[:5]:
                logger.log(level, f"    - {constraint}")
            if len(self.shape_constraints) > 5:
                logger.log(level, f"    ... and {len(self.shape_constraints) - 5} more")


@dataclasses.dataclass
class FeatureExplorationStats:
    """Statistics about which values of a feature were explored."""

    feature_name: str
    all_possible_values: list[object]  # All values in the search space
    tested_values: list[object]  # Values that were actually tested
    coverage_percent: float  # (tested / all) * 100

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "all_possible_values": self.all_possible_values,
            "tested_values": self.tested_values,
            "total_options": len(self.all_possible_values),
            "tested_options": len(self.tested_values),
            "coverage_percent": round(self.coverage_percent, 2),
        }

    def to_summary_line(self) -> str:
        """Single-line summary: 'pid_type: 2/8 options tested (25.0%)'."""
        return (
            f"{self.feature_name}: {len(self.tested_values)}/{len(self.all_possible_values)} "
            f"options tested ({self.coverage_percent:.1f}%)"
        )


@dataclasses.dataclass
class ExplorationReport:
    """Complete report on feature exploration during autotuning."""

    kernel_name: str
    backend: str
    search_algorithm: str
    elapsed_seconds: float
    configs_tested: int
    total_search_space_size: int | None

    # Per-feature exploration stats
    feature_stats: list[FeatureExplorationStats]

    # Overall exploration quality
    avg_feature_coverage: float  # Average coverage across all features
    min_feature_coverage: float  # Minimum coverage (bottleneck feature)

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_name": self.kernel_name,
            "backend": self.backend,
            "search_algorithm": self.search_algorithm,
            "elapsed_seconds": self.elapsed_seconds,
            "configs_tested": self.configs_tested,
            "total_search_space_size": (
                str(self.total_search_space_size)
                if self.total_search_space_size is not None
                else "infinite"
            ),
            "feature_stats": [stats.to_dict() for stats in self.feature_stats],
            "avg_feature_coverage": round(self.avg_feature_coverage, 2),
            "min_feature_coverage": round(self.min_feature_coverage, 2),
        }

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """Log a human-readable exploration summary."""
        logger.log(level, f"\nFeature Exploration Report for {self.kernel_name}:")
        logger.log(level, f"  Backend: {self.backend}")
        logger.log(level, f"  Search algorithm: {self.search_algorithm}")
        logger.log(level, f"  Time: {self.elapsed_seconds:.1f}s, Configs tested: {self.configs_tested:,}")

        if self.total_search_space_size is not None:
            overall_coverage = (self.configs_tested / self.total_search_space_size) * 100
            logger.log(level, f"  Overall search space coverage: {overall_coverage:.6f}%")

        logger.log(level, f"  Per-feature exploration:")
        logger.log(level, f"    Average feature coverage: {self.avg_feature_coverage:.1f}%")
        logger.log(level, f"    Minimum feature coverage: {self.min_feature_coverage:.1f}%")

        # List all features with their exploration stats
        for stats in sorted(self.feature_stats, key=lambda s: s.coverage_percent):
            logger.log(level, f"    - {stats.to_summary_line()}")

        # Highlight poorly explored features
        poor_coverage = [s for s in self.feature_stats if s.coverage_percent < 50.0]
        if poor_coverage:
            logger.log(level, f"\n  Features with <50% exploration:")
            for stats in poor_coverage:
                logger.log(level, f"    - {stats.feature_name}: only {len(stats.tested_values)} "
                          f"of {len(stats.all_possible_values)} values tested")


class FeatureExplorationTracker:
    """Track which feature values are tested during autotuning.

    This tracker records every config tested and aggregates statistics
    about which feature values were explored vs. which were available.
    """

    def __init__(self, search_summary: SearchSpaceSummary) -> None:
        self.search_summary = search_summary
        self.tested_configs: list[Config] = []
        self._feature_value_sets: dict[str, set[object]] = {}

        # Initialize empty sets for each feature
        for dim in search_summary.dimensions:
            self._feature_value_sets[dim.name] = set()

    def record_config(self, config: Config) -> None:
        """Record a tested configuration."""
        self.tested_configs.append(config)

        # Extract feature values from config
        for feature_name in self._feature_value_sets.keys():
            value = _extract_feature_value(config, feature_name)
            if value is not None:
                self._feature_value_sets[feature_name].add(value)

    def generate_report(
        self,
        search_algorithm: str,
        elapsed_seconds: float,
    ) -> ExplorationReport:
        """Generate an exploration report from recorded configs."""
        feature_stats: list[FeatureExplorationStats] = []

        # Build mapping from search_summary dimensions
        for dim in self.search_summary.dimensions:
            all_values = dim.values if dim.values is not None else []
            tested_values = sorted(self._feature_value_sets.get(dim.name, set()))

            if len(all_values) > 0:
                coverage = (len(tested_values) / len(all_values)) * 100
            else:
                coverage = 0.0

            feature_stats.append(FeatureExplorationStats(
                feature_name=dim.name,
                all_possible_values=all_values,
                tested_values=tested_values,
                coverage_percent=coverage,
            ))

        # Calculate aggregate metrics
        if feature_stats:
            avg_coverage = sum(s.coverage_percent for s in feature_stats) / len(feature_stats)
            min_coverage = min(s.coverage_percent for s in feature_stats)
        else:
            avg_coverage = 0.0
            min_coverage = 0.0

        return ExplorationReport(
            kernel_name=self.search_summary.kernel_name,
            backend=self.search_summary.backend,
            search_algorithm=search_algorithm,
            elapsed_seconds=elapsed_seconds,
            configs_tested=len(self.tested_configs),
            total_search_space_size=self.search_summary.total_search_space_size,
            feature_stats=feature_stats,
            avg_feature_coverage=avg_coverage,
            min_feature_coverage=min_coverage,
        )


def _extract_feature_value(config: Config, feature_name: str) -> object:
    """Extract a feature value from a Config object.

    Args:
        config: The tested configuration
        feature_name: Name of the feature to extract

    Returns:
        The feature value, or None if not applicable
    """
    # Handle special cases
    if feature_name == "block_sizes":
        return tuple(config.block_sizes)

    if feature_name == "pid_type":
        return config.pid_type

    if feature_name == "num_warps":
        return config.num_warps

    if feature_name == "num_stages":
        return config.num_stages

    if feature_name == "maxnreg":
        return config.maxnreg

    if feature_name == "loop_orders":
        return tuple(config.loop_orders)

    if feature_name == "pallas_loop_type":
        return config.pallas_loop_type

    if feature_name == "epilogue_subtile":
        return config.epilogue_subtile

    if feature_name == "l2_groupings":
        return tuple(config.l2_groupings)

    if feature_name == "flatten_loops":
        return tuple(config.flatten_loops)

    # Generic attribute access
    return getattr(config, feature_name, None)


def analyze_search_space(
    config_spec: ConfigSpec,
    kernel_name: str = "",
    specialization_key: str | None = None,
    hardware: str | None = None,
) -> SearchSpaceSummary:
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
        A SearchSpaceSummary describing the valid search space
    """
    from ..autotuner.config_spec import VALID_KEYS

    dimensions: list[SearchSpaceDimension] = []
    enabled_features: list[str] = []
    disabled_features: list[str] = []
    shape_constraints: list[str] = []

    # Analyze each config key
    for key in sorted(VALID_KEYS):
        supported = config_spec.supports_config_key(key)

        if supported:
            enabled_features.append(key)

            # Calculate dimension size for this key
            dim = _analyze_dimension_size(config_spec, key)
            if dim is not None:
                dimensions.append(dim)
        else:
            reason = _get_disable_reason(config_spec, key)
            disabled_features.append(f"{key}: {reason}")

    # Check shape-dependent constraints
    if config_spec.block_sizes:
        for i, spec in enumerate(config_spec.block_sizes):
            if spec.autotuner_min != spec.min_size:
                shape_constraints.append(
                    f"block_size[{i}] constrained to [{spec.autotuner_min}, {spec.max_size}] "
                    f"(flash attention target)"
                )

    if config_spec.cute_flash_search_enabled:
        shape_constraints.append("CuTe flash attention search enabled (restricted surface)")

    if config_spec.epilogue_subtile_autotune_choices is not None:
        shape_constraints.append(
            f"epilogue_subtile enabled for k_hint={config_spec.epilogue_subtile_k_hint}"
        )

    # Calculate total search space size (product of dimensions)
    total_size: int | None = 1
    for dim in dimensions:
        if dim.size > 1:
            if total_size is None:
                break
            total_size *= dim.size
            if total_size > 10**12:  # Cap at 1 trillion
                total_size = None
                break

    return SearchSpaceSummary(
        kernel_name=kernel_name,
        specialization_key=specialization_key,
        backend=config_spec.backend_name,
        hardware=hardware,
        dimensions=dimensions,
        total_search_space_size=total_size,
        enabled_features=enabled_features,
        disabled_features=disabled_features,
        shape_constraints=shape_constraints,
    )


def _analyze_dimension_size(
    config_spec: ConfigSpec,
    key: str,
) -> SearchSpaceDimension | None:
    """Analyze the size of one config dimension."""
    from ..autotuner.config_spec import (
        VALID_PID_TYPES,
        VALID_MAXNREG,
        AUTOTUNED_PALLAS_LOOP_TYPES,
        VALID_PALLAS_LOOP_TYPES,
    )
    from ..autotuner.config_fragment import (
        IntegerFragment,
        PowerOfTwoFragment,
        EnumFragment,
        BooleanFragment,
        ListOf,
    )

    # Handle special cases
    if key == "block_sizes":
        total = 1
        for spec in config_spec.block_sizes:
            # Count power-of-two values in range
            range_size = _count_power_of_two_values(spec.min_size, spec.max_size)
            total *= range_size
        return SearchSpaceDimension(
            name="block_sizes",
            dim_type="discrete",
            size=total,
            constrained_by="tensor numel constraints" if config_spec.tensor_numel_constraints else None,
        )

    elif key == "pid_type":
        return SearchSpaceDimension(
            name="pid_type",
            dim_type="categorical",
            size=len(config_spec.allowed_pid_types),
            values=list(config_spec.allowed_pid_types),
        )

    elif key == "num_warps":
        # num_warps is typically [1, 32] power-of-two
        return SearchSpaceDimension(
            name="num_warps",
            dim_type="discrete",
            size=6,  # 1, 2, 4, 8, 16, 32
            values=[1, 2, 4, 8, 16, 32],
        )

    elif key == "num_stages":
        frag = config_spec._num_stages_fragment()
        if isinstance(frag, IntegerFragment):
            size = frag.max_value - frag.min_value + 1
            return SearchSpaceDimension(
                name="num_stages",
                dim_type="discrete",
                size=size,
                values=list(range(frag.min_value, frag.max_value + 1)) if size <= 20 else None,
            )

    elif key == "maxnreg":
        return SearchSpaceDimension(
            name="maxnreg",
            dim_type="categorical",
            size=len(VALID_MAXNREG),
            values=[str(v) for v in VALID_MAXNREG],
        )

    elif key == "loop_orders":
        # Loop orders are permutations - factorial growth
        num_loops = len(config_spec.loop_orders)
        if num_loops <= 1:
            size = 1
        elif num_loops <= 6:
            size = math.factorial(num_loops)
        else:
            size = 720  # Cap at 6!
        return SearchSpaceDimension(
            name="loop_orders",
            dim_type="discrete",
            size=size,
            constrained_by=f"{num_loops} loops" if num_loops > 1 else None,
        )

    elif key == "pallas_loop_type":
        if config_spec.has_pallas_inner_loops:
            choices = VALID_PALLAS_LOOP_TYPES if config_spec.has_symbolic_or_data_dependent_bounds else AUTOTUNED_PALLAS_LOOP_TYPES
            return SearchSpaceDimension(
                name="pallas_loop_type",
                dim_type="categorical",
                size=len(choices),
                values=list(choices),
            )
        return SearchSpaceDimension(
            name="pallas_loop_type",
            dim_type="categorical",
            size=0,
            constrained_by="no pallas inner loops",
        )

    elif key == "epilogue_subtile":
        if config_spec.epilogue_subtile_autotune_choices:
            return SearchSpaceDimension(
                name="epilogue_subtile",
                dim_type="categorical",
                size=len(config_spec.epilogue_subtile_autotune_choices),
                values=[str(v) for v in config_spec.epilogue_subtile_autotune_choices],
            )
        return SearchSpaceDimension(
            name="epilogue_subtile",
            dim_type="categorical",
            size=0,
            constrained_by="not a matmul-like kernel or k_hint too small",
        )

    elif key == "l2_groupings":
        # L2 groupings is a ListOf with typically small integer values
        num_specs = len(config_spec.l2_groupings)
        if num_specs == 0:
            return None
        # Each slot can be 1, 2, 4, 8 typically
        per_slot_size = 4
        return SearchSpaceDimension(
            name="l2_groupings",
            dim_type="discrete",
            size=per_slot_size ** num_specs,
            constrained_by=f"{num_specs} loop(s)",
        )

    elif key == "flatten_loops":
        # Boolean per loop
        num_specs = len(config_spec.flatten_loops)
        if num_specs == 0:
            return None
        return SearchSpaceDimension(
            name="flatten_loops",
            dim_type="discrete",
            size=2 ** num_specs,
            constrained_by=f"{num_specs} loop(s)",
        )

    # Generic fragment analysis for other keys
    return None


def _count_power_of_two_values(min_val: int, max_val: int) -> int:
    """Count power-of-two values in [min_val, max_val]."""
    if min_val > max_val:
        return 0
    count = 0
    val = 1
    while val <= max_val:
        if val >= min_val:
            count += 1
        val *= 2
    return max(count, 1)  # At least one value


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
    summary: SearchSpaceSummary,
    configs_tested: int,
    search_algorithm: str,
    elapsed_seconds: float,
) -> None:
    """Log comparison of search space vs. what was actually searched.

    Args:
        logger: Logger to write to
        summary: The search space summary
        configs_tested: Number of configs actually tested
        search_algorithm: Name of the search algorithm used
        elapsed_seconds: Total autotune time
    """
    logger.info("=" * 60)
    logger.info("Autotune Search Space Analysis")
    logger.info("=" * 60)

    summary.log_summary(logger, logging.INFO)

    if summary.total_search_space_size is not None and summary.total_search_space_size > 0:
        coverage = (configs_tested / summary.total_search_space_size) * 100
        logger.info(f"\nSearch Coverage:")
        logger.info(f"  Configs tested: {configs_tested:,}")
        logger.info(f"  Total space: {summary.total_search_space_size:,}")
        logger.info(f"  Coverage: {coverage:.6f}%")
        logger.info(f"  Search algorithm: {search_algorithm}")
        logger.info(f"  Time elapsed: {elapsed_seconds:.1f}s")
        if coverage < 0.01:
            logger.info(f"  Note: Very low coverage - consider increasing autotune_budget_seconds")
    else:
        logger.info(f"\nSearch Coverage:")
        logger.info(f"  Configs tested: {configs_tested:,}")
        logger.info(f"  Total space: too large to enumerate")
        logger.info(f"  Search algorithm: {search_algorithm}")
        logger.info(f"  Time elapsed: {elapsed_seconds:.1f}s")

    logger.info("=" * 60)


def save_search_space_summary(
    summary: SearchSpaceSummary,
    configs_tested: int,
    search_algorithm: str,
    elapsed_seconds: float,
    output_path: str,
) -> str:
    """Save search space analysis to a JSON file.

    Args:
        summary: The search space summary
        configs_tested: Number of configs actually tested
        search_algorithm: Name of the search algorithm used
        elapsed_seconds: Total autotune time
        output_path: Path to save the JSON file

    Returns:
        The path to the saved file
    """
    output = {
        **summary.to_dict(),
        "configs_tested": configs_tested,
        "search_algorithm": search_algorithm,
        "elapsed_seconds": elapsed_seconds,
        "coverage_percent": (
            (configs_tested / summary.total_search_space_size) * 100
            if summary.total_search_space_size and summary.total_search_space_size > 0
            else None
        ),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str))
    return str(path)
