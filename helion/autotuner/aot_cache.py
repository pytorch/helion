"""
AOT (Ahead-of-Time) Autotuning Cache Implementation
====================================================

This module provides a cache implementation for AOT autotuning workflows that:
1. Collects tuned configs for each shape during benchmark runs
2. Measures all configs across all shapes
3. Generates heuristics using decision trees to select optimal configs
4. Supports multiple hardware architectures

The workflow is:
1. collect_tuned_configs: Tune each shape, record (kernel, shape, config) triples
2. measure_configs: Measure each shape with all observed configs
3. Generate heuristics to select configs based on performance goals
4. evaluate: Validate performance goals are achieved
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import logging
import marshal
import operator
import os
from pathlib import Path
import sys
import threading
import traceback
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import cast
import weakref

import torch

from .. import exc
from .._hardware import get_hardware_info
from .._utils import indexing_uses_tensor_descriptor
from ..runtime.config import Config
from ..runtime.cute.structural_config import CuteStructuralConfig
from ..runtime.cute.structural_config import config_artifact_json
from ..runtime.cute.structural_config import decode_structural_policy
from ..runtime.kernel import _BoolValueKey
from ..runtime.kernel import _FloatValueKey
from ..runtime.kernel import _FragmentSetValueKey
from ..runtime.kernel import _FragmentValueKey
from ..runtime.kernel import _resolve_index_widths
from ..runtime.structural_policy import StructuralConfig
from ..runtime.structural_policy import StructuralPolicyError
from ..runtime.structural_policy import bound_structural_policy
from ..runtime.structural_policy import require_same_structural_policy
from .aot_kernel import _flatten_key_value
from .aot_kernel import extract_key_features
from .aot_kernel import extract_shape_features
from .aot_structural_policy import load_policy_module
from .aot_structural_policy import measurement_config_hash
from .aot_structural_policy import measurement_metadata
from .aot_structural_policy import model_configs
from .aot_structural_policy import policy_path
from .base_cache import AutotuneCacheBase
from .base_cache import BoundKernelInMemoryCacheKey
from .base_cache import LooseAutotuneCacheKey
from .base_cache import _requires_uncached_tuning
from .benchmark_provider import _MultiShapeAutotuneArgs
from .benchmark_provider import _searching_candidates
from .benchmarking import l2_flush_sweep

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Callable

    from ..runtime.kernel import BoundKernel
    from ..runtime.structural_policy import StructuralPolicy
    from .aot_structural_export import GuardedCallKey
    from .base_search import BaseSearch
    from .config_fragment import ConfigSpecFragment

log: logging.Logger = logging.getLogger(__name__)

# Environment variable to control AOT mode
AOT_MODE_ENV = "HELION_AOT_MODE"
AOT_DATA_DIR_ENV = "HELION_AOT_DATA_DIR"
# Environment variable to override heuristic search path (for comparing heuristics)
HEURISTIC_DIR_ENV = "HELION_HEURISTIC_DIR"
# Environment variable to enable verbose output in quiet AOT modes.
AOT_VERBOSE_ENV = "HELION_AOT_VERBOSE"

AOTMode = Literal["collect", "measure", "evaluate", "compile", "disabled"]


def _config_uses_tensor_descriptor(config: Config) -> bool:
    # A prefetch's box form reads a tensor map, which needs an aligned base too.
    return any(
        map(
            indexing_uses_tensor_descriptor,
            (config.indexing, config.atomic_indexing, config.prefetch_indexing),
        )
    )


def get_aot_mode() -> AOTMode:
    """Get the current AOT mode from environment."""
    mode = os.environ.get(AOT_MODE_ENV, "evaluate").lower()
    if mode in ("collect", "measure", "evaluate", "compile", "disabled"):
        return mode  # type: ignore[return-value]
    raise ValueError(
        f"Invalid {AOT_MODE_ENV} value: {mode}. "
        "Must be one of: collect, measure, evaluate, compile, disabled"
    )


def is_aot_verbose() -> bool:
    """Check if verbose output is enabled for AOT mode.

    In evaluate and compile mode, output is quiet by default.
    Set HELION_AOT_VERBOSE=1 to enable verbose output.
    """
    return os.environ.get(AOT_VERBOSE_ENV, "").lower() in ("1", "true", "yes")


def get_aot_data_dir() -> Path:
    """Get the AOT data directory from environment or default."""
    if (path := os.environ.get(AOT_DATA_DIR_ENV)) is not None:
        return Path(path)
    return Path.cwd() / ".helion_aot"


# Cache for heuristic file lookups
_heuristic_file_cache: dict[tuple[str, ...], Path | None] = {}
# The kernels, each a source file and a name, that a lookup without a
# structural policy and without a data directory was made for: the kernel's
# first answer stands for all its lookups
_heuristic_file_shared: set[tuple[str, str]] = set()


def find_heuristic_file(
    kernel_source_file: str | Path,
    kernel_name: str | None = None,
    data_dir: Path | None = None,
    *,
    policy: StructuralPolicy | None = None,
) -> Path | None:
    """
    Find the heuristic file for a kernel.

    This is the single source of truth for heuristic file discovery, used by both
    AOTKeyFunction and AOTAutotuneCache.

    Search order:
    1. HELION_HEURISTIC_DIR env var (if set) - for comparing different heuristics
    2. Next to kernel source file: _<filename>_<device>_<compute>.py
    3. Fallback to older compute capabilities within the same device family
    4. AOT data directory: heuristic_<kernel_name>.py (fallback)

    The answer is kept until ``clear_heuristic_cache()``. Without a
    structural policy it is kept by the kernel, which is its source file
    and its name, and by the data directory: the kernels of a source file,
    and kernels with tuning data in two directories, each find their own.
    A lookup without a data directory (the key function of an
    ``aot_kernel``) takes the first answer its kernel got, and that answer
    then stands for every lookup of the kernel, whatever data directory it
    names: the key function and the cache of a kernel select from one file.

    The data directory is resolved, for the search and for the key, and the
    answer is the name of the file found in its resolved directory: a
    relative ``HELION_AOT_DATA_DIR`` is another directory once the working
    directory changes, and what callers keep by the answer (a module, its
    selections) is kept by the file. The name itself is not resolved: a
    heuristic that is a link is read through the link, as a file rewritten
    in place is read anew. Under a structural policy, whose model is read
    at each selection, the key holds the other two directories a candidate
    is searched in resolved as well, the source file's and
    ``HELION_HEURISTIC_DIR``: one that is relative, or a link, is looked up
    again once it names another directory. Without a policy the key holds
    the source file as the kernel's code names it, and neither that
    variable nor the hardware. Like the files on disk, these are as the
    first lookup of a key found them: the answer stays the file in the
    directory they named then.

    Args:
        kernel_source_file: Path to the kernel's source file
        kernel_name: Optional kernel name for fallback lookup
        data_dir: Optional AOT data directory for fallback lookup

    Returns:
        Path to heuristic file, in its resolved directory, if found, None
        otherwise
    """
    if data_dir is not None:
        data_dir = data_dir.resolve()
    heuristic_dir = os.environ.get(HEURISTIC_DIR_ENV)
    kernel = (str(kernel_source_file), kernel_name or "")
    cache_key: tuple[str, ...] = kernel
    if policy is not None:
        # With the three directories of the candidates, each resolved as
        # the answer's is.
        cache_key = (
            str(kernel_source_file),
            kernel_name or "",
            str(Path(kernel_source_file).parent.resolve()),
            str(data_dir),
            "" if heuristic_dir is None else str(Path(heuristic_dir).resolve()),
            get_hardware_info().hardware_id,
            policy.identity(),
        )
    elif data_dir is None:
        _heuristic_file_shared.add(kernel)
    elif kernel not in _heuristic_file_shared:
        cache_key = (*kernel, str(data_dir))
    if cache_key in _heuristic_file_cache:
        return _heuristic_file_cache[cache_key]

    source_path = Path(kernel_source_file)
    base_name = source_path.stem
    hw = get_hardware_info()
    compatible_computes = hw.get_compatible_compute_ids()

    candidates: list[Path] = []

    # 1. Check HELION_HEURISTIC_DIR override
    if heuristic_dir is not None:
        heuristic_dir_path = Path(heuristic_dir)
        for compat_compute in compatible_computes:
            candidates.append(
                heuristic_dir_path
                / f"_helion_aot_{base_name}_{hw.device_kind}_{compat_compute}.py"
            )
        if kernel_name:
            candidates.append(heuristic_dir_path / f"heuristic_{kernel_name}.py")

    # 2. Check next to kernel source file with compute capability fallback
    for compat_compute in compatible_computes:
        heuristic_name = f"_helion_aot_{base_name}_{hw.device_kind}_{compat_compute}.py"
        candidates.append(source_path.parent / heuristic_name)

    # 3. Check AOT data directory (fallback)
    if data_dir is not None and kernel_name is not None:
        candidates.append(data_dir / f"heuristic_{kernel_name}.py")

    # Find first existing file
    result: Path | None = None
    for candidate in candidates:
        candidate = policy_path(candidate, policy)
        if candidate.exists():
            log.debug(f"Found heuristic file: {candidate}")
            result = candidate.parent.resolve() / candidate.name
            break

    _heuristic_file_cache[cache_key] = result
    if policy is None:
        # The first answer of the kernel.
        _heuristic_file_cache.setdefault(kernel, result)
    return result


def clear_heuristic_cache() -> None:
    """Clear the heuristic file cache (useful for testing)."""
    _heuristic_file_cache.clear()
    _heuristic_file_shared.clear()


def load_kernel_source_files(
    data_dir: Path,
    hardware_id: str,
    *,
    policy: StructuralPolicy | None = None,
) -> dict[str, str]:
    """
    Load kernel source file mappings from tuned configs JSON.

    This is a standalone function for use by aot_runner.py during heuristic generation.

    Args:
        data_dir: Directory containing the tuned configs file
        hardware_id: Hardware ID used in the filename

    Returns:
        Dict mapping kernel_name -> source_file_path
    """
    configs_file = data_dir / f"tuned_configs_{hardware_id}.json"
    if not configs_file.exists():
        return {}

    try:
        data = json.loads(configs_file.read_text())
        result: dict[str, str] = {}
        for kernel_name, configs in data.items():
            for cfg in configs:
                if policy is not None and (
                    "cute_structural_policy" not in cfg
                    or decode_structural_policy(cfg["cute_structural_policy"]) != policy
                ):
                    continue
                if cfg.get("kernel_source_file"):
                    result[kernel_name] = cfg["kernel_source_file"]
                    break
        return result
    except StructuralPolicyError:
        raise
    except Exception as e:
        log.warning(f"Failed to load kernel source files: {e}")
        return {}


@dataclass
class ShapeKey:
    """Represents a unique shape/dtype combination for a kernel."""

    kernel_name: str
    specialization_key: tuple[Any, ...]
    hardware_id: str
    search_policy_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        result = {
            "kernel_name": self.kernel_name,
            "specialization_key": _serialize_tuple(self.specialization_key),
            "hardware_id": self.hardware_id,
        }
        if self.search_policy_hash:
            result["search_policy_hash"] = self.search_policy_hash
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShapeKey:
        """Create from a dict."""
        return cls(
            kernel_name=data["kernel_name"],
            specialization_key=_deserialize_tuple(data["specialization_key"]),
            hardware_id=data["hardware_id"],
            search_policy_hash=data.get("search_policy_hash", ""),
        )

    def stable_hash(self) -> str:
        """Get a stable hash for this shape key."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute SHA256 hash (first 8 chars) of tensor bytes."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    # Convert dtypes not supported by numpy (e.g., bfloat16)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    # fp8 (and other sub-byte-named) dtypes have no numpy equivalent; hash the
    # raw bytes by viewing as uint8 (fp8_e4m3fn / fp8_e5m2 are 1 byte each).
    elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.view(torch.uint8)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:8]


@dataclass
class TunedConfig:
    """A tuned configuration with its benchmark results."""

    config: Config
    shape_key: ShapeKey
    timing_ms: float | None = None
    kernel_source_file: str | None = None
    shape_features: dict[str, Any] | None = None
    # SHA256 hashes (first 8 chars) for correctness verification:
    # [0] = input tensor hashes before kernel runs
    # [1] = input tensor hashes after kernel runs (to detect in-place modifications)
    # [2] = output tensor hashes
    tensor_hashes: list[list[str]] | None = None
    cute_structural_policy: StructuralPolicy | None = None


def _configs_text(configs: Sequence[Mapping[str, object]]) -> str:
    """``configs`` as one string, equal for equal options."""
    return json.dumps(
        [dict(config) for config in configs], sort_keys=True, default=repr
    )


class AOTAutotuneCache(AutotuneCacheBase):
    """
    Cache implementation for AOT autotuning workflows.

    Behavior depends on the HELION_AOT_MODE environment variable:
    - collect: Tune each shape individually, record results
    - measure: Measure each shape with all observed configs
    - evaluate: Use heuristics to select configs, validate performance
    - disabled: Fall through to underlying autotuner (default)

    When collect_fn/measure_fn are set on the kernel:
    - collect_fn: In collect mode, only these inputs are autotuned
    - measure_fn: In measure mode, only these inputs are measured
    - One-shot: If both set in collect mode, runs both phases in one invocation
    """

    # Preserve explicit evaluate/compile dispatch for fresh bindings. This does
    # not make their collect-mode search results reusable: that read is guarded
    # separately by _search_policy_allows_cache_io.
    _supports_fresh_bindings = True

    # Tracks which AOT modes have been announced to avoid repeated stderr messages.
    # Class-level so announcements happen only once per mode across all instances.
    _mode_announced: ClassVar[set[str]] = set()

    # Class-level caches for heuristic lookup (shared across instances)
    # Maps heuristic file path -> loaded module
    _heuristic_modules: ClassVar[dict[Path | tuple[Path, str], Any]] = {}
    # Held while the module of a plain heuristic is executed for its path
    _heuristic_import_lock: ClassVar[Any] = threading.RLock()
    # Maps (kernel_source_file, kernel_name, shape_features_hash,
    # heuristic file) -> Config
    # Using source file ensures kernels with same name in different modules don't collide
    _heuristic_results: ClassVar[
        dict[tuple[str, ...], Config | CuteStructuralConfig]
    ] = {}
    # Tracks which kernels have shown the "no heuristic" warning (to avoid spam)
    _no_heuristic_warned: ClassVar[set[str]] = set()
    # Tracks which kernels have already been compiled in compile mode
    _compiled_kernels: ClassVar[set[str | tuple[str, ...]]] = set()
    # Static-shape compile mode accumulates one source variant per normalized
    # call signature and rewrites the dispatcher whenever a shape is observed.
    _compiled_kernel_variants: ClassVar[
        dict[
            tuple[str | bool, ...],
            dict[tuple[object, ...], tuple[str, str]],
        ]
    ] = {}
    _compiled_kernel_variants_lock: ClassVar[Any] = threading.RLock()
    # Keep the last ordinary dynamic table, even while a static file is
    # published at its path. Do not keep the actual bound kernel alive.
    _ordinary_dynamic_children: ClassVar[
        dict[
            Path,
            tuple[
                weakref.ReferenceType[BoundKernel], tuple[object, ...], dict[int, str]
            ],
        ]
    ] = {}
    _policy_dynamic_variants: ClassVar[
        dict[tuple[str | bool, ...], dict[tuple[object, ...], tuple[str, ...]]]
    ] = {}
    # Under held overrides (``BoundKernel._unsearched_overrides_identity``):
    # what the ordinary dynamic table last published at a path was generated
    # from besides its binding, the stored configs, and those configs as the
    # backend's hook left them for that binding
    # (``_check_ordinary_dynamic_answer``).
    _ordinary_dynamic_answers: ClassVar[
        dict[Path, tuple[tuple[object, ...], str, str]]
    ] = {}
    _standalone_export_owners: ClassVar[dict[Path, tuple[str | bool, ...]]] = {}
    _dynamic_export_contracts: ClassVar[
        dict[Path, tuple[int | None, tuple[str, str] | None]]
    ] = {}

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all class-level caches (heuristic modules and results)."""
        with cls._heuristic_import_lock:
            cls._heuristic_modules.clear()
        cls._heuristic_results.clear()
        cls._no_heuristic_warned.clear()
        cls._compiled_kernels.clear()
        with cls._compiled_kernel_variants_lock:
            cls._compiled_kernel_variants.clear()
            cls._ordinary_dynamic_children.clear()
            cls._ordinary_dynamic_answers.clear()
            cls._policy_dynamic_variants.clear()
            cls._standalone_export_owners.clear()
            cls._dynamic_export_contracts.clear()
        clear_heuristic_cache()  # Clear module-level cache
        cls._mode_announced.clear()
        log.debug("Cleared AOTAutotuneCache caches")

    def __init__(
        self,
        autotuner: BaseSearch,
        *,
        autotuner_factory: Callable[[], BaseSearch] | None = None,
    ) -> None:
        super().__init__(autotuner, autotuner_factory=autotuner_factory)
        if not isinstance(self.args, _MultiShapeAutotuneArgs):
            self.args = self.kernel.kernel.normalize_args(*self.args)
        self.mode = get_aot_mode()
        self.hardware_id = get_hardware_info().hardware_id
        self.data_dir = get_aot_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._tuned_configs: dict[str, list[TunedConfig]] = self._load_tuned_configs()
        self.shape_key = self._create_shape_key()
        self._verbose = is_aot_verbose()

        # Look up optional collect_fn/measure_fn from the Kernel object
        # These are set by @aot_kernel() decorator
        self._collect_fn = getattr(self.kernel.kernel, "_aot_collect_fn", None)
        self._measure_fn = getattr(self.kernel.kernel, "_aot_measure_fn", None)

        # Announce mode once per mode type (quiet in evaluate/compile unless verbose)
        should_announce = (
            self.mode != "disabled"
            and self.mode not in AOTAutotuneCache._mode_announced
            and (self.mode not in ("evaluate", "compile") or self._verbose)
        )
        if should_announce:
            print(
                f"[AOT] Mode: {self.mode}, Data dir: {self.data_dir}, "
                f"Hardware: {self.hardware_id}",
                file=sys.stderr,
            )
            num_configs = sum(len(v) for v in self._tuned_configs.values())
            if num_configs > 0:
                print(f"[AOT] Loaded {num_configs} existing configs", file=sys.stderr)
            AOTAutotuneCache._mode_announced.add(self.mode)

    def _should_report_cache_hit(self) -> bool:
        return self.mode not in ("evaluate", "compile") or self._verbose

    def _search_policy_allows_cache_io(self) -> bool:
        # Evaluate/compile/disabled select an AOT heuristic or default config;
        # they are not search-result cache reads and must never start tuning.
        if self.mode in ("disabled", "evaluate", "compile"):
            return True
        if self.mode == "collect":
            # Explicit collection owns its artifacts even for custom providers.
            # A fresh binding still has no reusable automatic selection key.
            return not _requires_uncached_tuning(self.kernel, self.args)
        return super()._search_policy_allows_cache_io()

    def _search_policy_allows_cache_write(self) -> bool:
        # Collect must record a newly tuned config even when its custom search
        # policy is intentionally ineligible for later cache reuse.
        if self.mode == "collect":
            return True
        return super()._search_policy_allows_cache_write()

    @property
    def _configs_file(self) -> Path:
        """Path to the tuned configs JSON file."""
        return self.data_dir / f"tuned_configs_{self.hardware_id}.json"

    @property
    def _measurements_file(self) -> Path:
        """Path to the measurements CSV file."""
        return policy_path(
            self.data_dir / f"measurements_{self.hardware_id}.csv",
            bound_structural_policy(self.kernel),
        )

    def _load_tuned_configs(self) -> dict[str, list[TunedConfig]]:
        """Load tuned configs from disk."""
        if not self._configs_file.exists():
            return {}
        try:
            data = json.loads(self._configs_file.read_text())
            result: dict[str, list[TunedConfig]] = {}
            for kernel_name, configs in data.items():
                result[kernel_name] = []
                for cfg in configs:
                    policy = (
                        decode_structural_policy(cfg["cute_structural_policy"])
                        if "cute_structural_policy" in cfg
                        else None
                    )
                    result[kernel_name].append(
                        TunedConfig(
                            config=(
                                Config.from_dict(cfg["config"])
                                if policy is not None
                                else Config(**cfg["config"])
                            ),
                            shape_key=ShapeKey.from_dict(cfg["shape_key"]),
                            timing_ms=cfg.get("timing_ms"),
                            kernel_source_file=cfg.get("kernel_source_file"),
                            shape_features=cfg.get("shape_features"),
                            tensor_hashes=cfg.get("tensor_hashes"),
                            cute_structural_policy=policy,
                        )
                    )
            return result
        except StructuralPolicyError:
            raise
        except Exception as e:
            log.warning(f"Failed to load tuned configs: {e}")
            return {}

    def _save_tuned_configs(self) -> None:
        """Save tuned configs to disk."""
        data: dict[str, list[dict[str, Any]]] = {}
        for kernel_name, config_list in self._tuned_configs.items():
            data[kernel_name] = [
                {
                    "config": dict(cfg.config),
                    "shape_key": cfg.shape_key.to_dict(),
                    "timing_ms": cfg.timing_ms,
                    "kernel_source_file": cfg.kernel_source_file,
                    "shape_features": cfg.shape_features,
                    "tensor_hashes": cfg.tensor_hashes,
                    **(
                        {"cute_structural_policy": cfg.cute_structural_policy.to_dict()}
                        if cfg.cute_structural_policy is not None
                        else {}
                    ),
                }
                for cfg in config_list
            ]
        self._configs_file.write_text(json.dumps(data, indent=2))

    def _add_tuned_config(
        self,
        kernel_name: str,
        config: Config,
        shape_key: ShapeKey,
        timing_ms: float | None = None,
        kernel_source_file: str | None = None,
        shape_features: dict[str, Any] | None = None,
        tensor_hashes: list[list[str]] | None = None,
    ) -> None:
        """Add a tuned config for a kernel/shape combination."""
        if kernel_name not in self._tuned_configs:
            self._tuned_configs[kernel_name] = []

        shape_hash = shape_key.stable_hash()
        config_dict = dict(config)
        policy = bound_structural_policy(self.kernel)

        # Check if this exact config already exists for this shape
        for existing in self._tuned_configs[kernel_name]:
            if (
                existing.shape_key.stable_hash() == shape_hash
                and dict(existing.config) == config_dict
                and existing.cute_structural_policy == policy
            ):
                # Update if we have better timing
                if timing_ms is not None:
                    if existing.timing_ms is None or timing_ms < existing.timing_ms:
                        existing.timing_ms = timing_ms
                if kernel_source_file is not None:
                    existing.kernel_source_file = kernel_source_file
                if shape_features is not None:
                    existing.shape_features = shape_features
                if tensor_hashes is not None:
                    existing.tensor_hashes = tensor_hashes
                return

        self._tuned_configs[kernel_name].append(
            TunedConfig(
                config=copy.deepcopy(config) if policy is not None else config,
                shape_key=shape_key,
                timing_ms=timing_ms,
                kernel_source_file=kernel_source_file,
                shape_features=shape_features,
                tensor_hashes=tensor_hashes,
                cute_structural_policy=policy,
            )
        )

    def _get_all_configs_for_kernel(self, kernel_name: str) -> list[Config]:
        """Get all unique configs observed for a kernel."""
        if kernel_name not in self._tuned_configs:
            return []
        seen: set[str] = set()
        result: list[Config] = []
        policy = bound_structural_policy(self.kernel)
        for tc in self._tuned_configs[kernel_name]:
            if tc.cute_structural_policy != policy:
                continue
            config_hash = hashlib.sha256(
                config_artifact_json(tc.config, tc.cute_structural_policy).encode()
            ).hexdigest()
            if config_hash not in seen:
                seen.add(config_hash)
                result.append(
                    copy.deepcopy(tc.config) if policy is not None else tc.config
                )
        return result

    def _save_measurement(
        self,
        kernel_name: str,
        shape_key: ShapeKey,
        config: Config,
        timing_ms: float,
        shape_features: dict[str, Any],
    ) -> None:
        """Save a measurement to CSV."""
        policy = bound_structural_policy(self.kernel)
        config_hash = measurement_config_hash(config, policy)
        row = {
            "kernel_name": kernel_name,
            "shape_hash": shape_key.stable_hash(),
            "config_hash": config_hash,
            "config": json.dumps(dict(config)),
            "shape_features": json.dumps(shape_features),
            "timing_ms": timing_ms,
            **measurement_metadata(policy),
        }
        file_exists = self._measurements_file.exists()
        with open(self._measurements_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _create_shape_key(self) -> ShapeKey:
        """Create a shape key for the current kernel invocation."""
        from .local_cache import _search_policy_hash

        search_policy_hash = _search_policy_hash(
            self.autotuner,
            cute_flash_search_enabled=self.kernel.config_spec.cute_flash_search_enabled,
            explicit_aot=True,
        )
        return ShapeKey(
            kernel_name=self.kernel.kernel.name,
            specialization_key=self.kernel.kernel.specialization_key(self.args),
            hardware_id=self.hardware_id,
            search_policy_hash=search_policy_hash,
        )

    def _extract_shape_features(
        self, args: Sequence[object] | None = None
    ) -> dict[str, Any]:
        """Extract numeric features from the shape for ML model.

        If a user key function is provided, extracts features from the
        flattened key output instead of raw args.
        """
        if args is None:
            args = self.args

        # Check if user provided a key function
        user_key = getattr(self.kernel.kernel, "_aot_user_key", None)
        if user_key is not None:
            # Extract features from flattened key output
            key_value = user_key(*args)
            return extract_key_features(key_value)

        # Use single source of truth from aot_kernel module
        return extract_shape_features(args)

    def get(self) -> Config | None:
        """Get a cached config based on current mode."""
        if self.mode == "collect":
            # In collect mode, check if we already have a config for this exact shape
            kernel_name = self.kernel.kernel.name
            configs = self._tuned_configs.get(kernel_name, [])
            for tc in configs:
                if (
                    tc.shape_key.stable_hash() == self.shape_key.stable_hash()
                    and tc.cute_structural_policy
                    == bound_structural_policy(self.kernel)
                ):
                    log.info(f"AOT collect: Using existing config for {kernel_name}")
                    return (
                        copy.deepcopy(tc.config)
                        if tc.cute_structural_policy is not None
                        else tc.config
                    )
            return None  # Need to tune

        if self.mode == "measure":
            # In measure mode, we don't use cache - we measure all configs
            return None

        if self.mode == "compile":
            # In compile mode: use heuristic + generate standalone Triton code
            self._maybe_run_compile()

        # For disabled/evaluate/compile modes: try heuristic, fall back to default config
        # (never trigger autotuning for aot_kernel)
        config = self._get_heuristic_config(self.args)
        if config is not None:
            return config

        # No heuristic available - use default config with warning (once per kernel)
        kernel_name = self.kernel.kernel.name
        from .. import exc

        if kernel_name not in AOTAutotuneCache._no_heuristic_warned:
            AOTAutotuneCache._no_heuristic_warned.add(kernel_name)
            if exc.NoAOTHeuristicWarning not in self.autotuner.settings.ignore_warnings:
                print(
                    f"[AOT] Warning: No heuristic found for '{kernel_name}'. "
                    f"Using default config. "
                    f"Use `python -m helion.autotuner.aot_runner` to generate tuned configs.",
                    file=sys.stderr,
                )
        return self.autotuner.config_spec.default_config()

    def _compute_tensor_hashes(
        self, tensors: Sequence[object] | None = None
    ) -> list[str]:
        """Compute hashes for tensors. Non-tensors get "n/a"."""
        if tensors is None:
            tensors = self.args
        return [
            compute_tensor_hash(arg) if isinstance(arg, torch.Tensor) else "n/a"
            for arg in tensors
        ]

    def put(self, config: Config, timing_ms: float | None = None) -> None:
        """Store a tuned config based on current mode."""
        if self.mode == "disabled":
            return

        if self.mode == "collect":
            kernel_name = self.kernel.kernel.name
            kernel_source_file = self.kernel.kernel.__code__.co_filename
            shape_features = self._extract_shape_features()

            # Hash inputs, run kernel, hash inputs again and outputs
            input_hashes = self._compute_tensor_hashes()
            fn = self.kernel.compile_config(config)
            outputs = fn(*self.args)
            input_after_hashes = self._compute_tensor_hashes()
            if outputs is None:
                outputs = ()
            elif not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            output_hashes = self._compute_tensor_hashes(outputs)

            tensor_hashes = [input_hashes, input_after_hashes, output_hashes]

            self._add_tuned_config(
                kernel_name=kernel_name,
                config=config,
                shape_key=self.shape_key,
                timing_ms=timing_ms,
                kernel_source_file=kernel_source_file,
                shape_features=shape_features,
                tensor_hashes=tensor_hashes,
            )
            self._save_tuned_configs()

            print(
                f"[AOT collect] Saved config for kernel={kernel_name} "
                f"shape_hash={self.shape_key.stable_hash()[:8]} "
                f"hashes={tensor_hashes} "
                f"to {self._configs_file}",
                file=sys.stderr,
            )
            log.info(
                f"AOT collect: Saved config for {kernel_name} "
                f"shape={self.shape_key.stable_hash()}"
            )

    def measure_all_configs(self) -> list[tuple[Config, float]]:
        """
        Measure all known configs for the current shape.
        Returns list of (config, timing_ms) pairs.
        """
        # (with the search's L2 flush sweep, ``autotune_l2_flush_sweep``)
        with l2_flush_sweep(self.autotuner.settings.autotune_l2_flush_sweep):
            return self._measure_all_configs()

    def _measure_all_configs(self) -> list[tuple[Config, float]]:
        self.autotuner._prepare()
        kernel_name = self.kernel.kernel.name
        all_configs = self._get_all_configs_for_kernel(kernel_name)

        if not all_configs:
            log.warning(f"No configs found for kernel {kernel_name}")
            return []

        print(
            f"[AOT measure] Testing {len(all_configs)} configs for {kernel_name} "
            f"shape_hash={self.shape_key.stable_hash()[:8]}",
            file=sys.stderr,
        )

        results: list[tuple[Config, float]] = []
        shape_features = self._extract_shape_features()

        # Temporarily disable subprocess precompile for direct benchmark calls
        old_precompile = self.autotuner.settings.autotune_precompile
        self.autotuner.settings.autotune_precompile = None

        # Set up provider resources if needed (normally done inside autotune())
        benchmark_provider = self.autotuner.benchmark_provider
        try:
            benchmark_provider.setup()
            for i, config in enumerate(all_configs):
                try:
                    # Collected tuning data: no search under the kernel's
                    # overrides need have written it, so the config is timed
                    # as the kernel compiles it. The measurement stays under
                    # the stored config, which a heuristic selects by.
                    result = self.autotuner.benchmark(
                        self.kernel._unsearched_config(config)
                    )
                    timing = result.perf
                    if timing < float("inf"):
                        results.append((config, timing))

                        # Save measurement
                        self._save_measurement(
                            kernel_name=kernel_name,
                            shape_key=self.shape_key,
                            config=config,
                            timing_ms=timing,
                            shape_features=shape_features,
                        )
                        print(
                            f"[AOT measure] Config {i + 1}/{len(all_configs)}: {timing:.4f}ms",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[AOT measure] Config {i + 1}/{len(all_configs)}: invalid (inf timing)",
                            file=sys.stderr,
                        )
                except Exception as e:
                    error_msg = str(e) or type(e).__name__
                    tb = traceback.format_exc()
                    print(
                        f"[AOT measure] Config {i + 1}/{len(all_configs)}: failed - {error_msg}",
                        file=sys.stderr,
                    )
                    # Print last few lines of traceback for debugging
                    tb_lines = tb.strip().split("\n")
                    if len(tb_lines) > 4:
                        print(f"  Traceback: ...{tb_lines[-3]}", file=sys.stderr)
                        print(f"             {tb_lines[-2]}", file=sys.stderr)
                    log.debug(f"Failed to benchmark config {config}: {e}\n{tb}")
        finally:
            # Restore settings
            self.autotuner.settings.autotune_precompile = old_precompile
            benchmark_provider.cleanup()

        print(
            f"[AOT measure] Completed: {len(results)}/{len(all_configs)} configs succeeded",
            file=sys.stderr,
        )
        return results

    def _find_heuristic_file(self) -> Path | None:
        """Find the heuristic file for this kernel using shared lookup."""
        kernel_name = self.kernel.kernel.name
        kernel_source_file = self.kernel.kernel.__code__.co_filename
        return find_heuristic_file(
            kernel_source_file,
            kernel_name=kernel_name,
            data_dir=self.data_dir,
            policy=bound_structural_policy(self.kernel),
        )

    def _get_heuristic_config(
        self,
        args: Sequence[object] | None = None,
        *,
        snapshot: tuple[Path, bytes] | None = None,
    ) -> Config | None:
        """
        Use the heuristic to select a config.

        Looks for autotune_<kernel>(*args) function in the heuristic file.

        Args:
            args: Optional arguments to use. If None, uses self.args.
            snapshot: The heuristic file and its bytes as the caller read
                them: the selection is made from them and kept by their
                digest. Without one, a kernel under a structural policy
                reads the file here, and a kernel without a policy selects
                from the module kept for the path.

        For CUDA/ROCm, if heuristics for the current compute capability aren't found,
        we try older compatible architectures (e.g., sm80 heuristics on sm90 hardware).
        """
        heuristic_file = (
            self._find_heuristic_file() if snapshot is None else snapshot[0]
        )
        if heuristic_file is None:
            return None

        if args is None:
            args = self.args

        kernel_name = self.kernel.kernel.name
        kernel_source_file = self.kernel.kernel.__code__.co_filename

        # Compute cache key based on shape features
        shape_features = self._extract_shape_features(args)
        shape_hash = hashlib.sha256(
            json.dumps(shape_features, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Check if we already have a cached result for this kernel+shape
        policy = bound_structural_policy(self.kernel)
        cache_key = (kernel_source_file, kernel_name, shape_hash)
        source = snapshot[1] if snapshot is not None else None
        if policy is not None or source is not None:
            if source is None:
                source = heuristic_file.read_bytes()
            cache_key = (
                *cache_key,
                policy.identity() if policy is not None else "",
                str(heuristic_file.resolve()),
                hashlib.sha256(source).hexdigest(),
            )
        else:
            # The results of the module kept for this heuristic file
            # (``_load_heuristic_module``), by the path the lookup answers
            # with, which names a resolved directory: two caches over one
            # kernel may have their tuning data in different files.
            cache_key = (*cache_key, str(heuristic_file))
        if cache_key in AOTAutotuneCache._heuristic_results:
            log.debug(
                f"Using cached heuristic result for {kernel_name} shape={shape_hash}"
            )
            result = AOTAutotuneCache._heuristic_results[cache_key]
            if isinstance(result, StructuralConfig):
                require_same_structural_policy(
                    result.policy, policy, context="Cached AOT heuristic"
                )
                return result.config
            require_same_structural_policy(None, policy, context="Cached AOT heuristic")
            return result

        try:
            module = self._load_heuristic_module(heuristic_file, policy, source)
            model_configs(module, kernel_name, policy, required=False)

            # Call autotune_<kernel>(*args) to get the config
            # If there's a user key, we need to pass flattened key values, not raw args
            config: Config | None = None
            autotune_fn = getattr(module, f"autotune_{kernel_name}", None)
            if autotune_fn is not None:
                user_key = getattr(self.kernel.kernel, "_aot_user_key", None)
                if user_key is not None:
                    # User key: pass flattened key values to heuristic
                    key_value = user_key(*args)
                    flat_key = _flatten_key_value(key_value)
                    config_dict = autotune_fn(*flat_key)
                else:
                    # No user key: pass raw args to heuristic
                    config_dict = autotune_fn(*args)
                if isinstance(config_dict, StructuralConfig):
                    require_same_structural_policy(
                        config_dict.policy, policy, context="AOT heuristic"
                    )
                    config = config_dict.config
                else:
                    require_same_structural_policy(
                        None, policy, context="Unversioned AOT heuristic"
                    )
                    config = Config(**config_dict)

            # Cache the result
            if config is not None:
                AOTAutotuneCache._heuristic_results[cache_key] = (
                    CuteStructuralConfig(config, policy)
                    if policy is not None
                    else config
                )
                log.debug(
                    f"Cached heuristic result for {kernel_name} shape={shape_hash}"
                )

            return config
        except StructuralPolicyError:
            raise
        except Exception as e:
            log.warning(f"Failed to load heuristic from {heuristic_file}: {e}")

        return None

    @classmethod
    def _load_heuristic_module(
        cls, path: Path, policy: StructuralPolicy | None, source: bytes | None = None
    ) -> types.ModuleType:
        """The module of the heuristic file ``path``, executed from bytes
        read from the file (``load_policy_module``), never from the bytecode
        Python keeps for the path.

        A caller that passes ``source``, the bytes it read from the file,
        gets the module executed from those bytes and kept by their digest,
        and so does a policy-bearing heuristic, whose file is read here. A
        plain heuristic without ``source`` is the module kept for its path:
        the last one executed for it, or one executed here, once, from the
        bytes the file then holds. ``_get_heuristic_config`` keeps its
        selections by the path as well (``clear_caches`` drops both)."""
        key: Path | tuple[Path, str] = path
        if source is not None or policy is not None:
            if source is None:
                source = path.read_bytes()
            key = (path, hashlib.sha256(source).hexdigest())
            if key not in cls._heuristic_modules:
                module = load_policy_module(path, source)
                cls._heuristic_modules[key] = module
                cls._heuristic_modules[path] = module
            return cls._heuristic_modules[key]
        # One module per path, also where two threads get here at once.
        with cls._heuristic_import_lock:
            if key not in cls._heuristic_modules:
                module = load_policy_module(path, path.read_bytes())
                cls._heuristic_modules[key] = module
                # Retain the public diagnostic view of the last module at a path.
                cls._heuristic_modules[path] = module
            return cls._heuristic_modules[key]

    def _maybe_run_compile(self) -> None:
        """
        In compile mode, generate code for all heuristic-selected configs and
        write a standalone ``.py`` file. Backends with a dependency-free
        launcher need no Helion runtime; CuTe currently retains its launcher
        import.

        Dynamic-shape kernels compile every heuristic config once. Static-shape
        kernels compile the selected config for each observed BoundKernel and
        dispatch by a normalized call signature.
        """
        kernel_name = self.kernel.kernel.name
        heuristic_file = self._find_heuristic_file()
        if heuristic_file is None:
            log.warning(
                "No heuristic for '%s', skipping standalone compile", kernel_name
            )
            return

        policy = bound_structural_policy(self.kernel)
        # Read once: the chosen children, the ownership key, and the emitted
        # selector must all describe the same model, even if it is replaced.
        model_source = heuristic_file.read_bytes()
        module = self._load_heuristic_module(heuristic_file, policy, model_source)
        records = model_configs(module, kernel_name, policy)

        # -- extract selected configs ---------------------------------------
        # nearest_neighbor backend: module-level CONFIGS
        # decision_tree backend: _C = [...] inside autotune_<kernel>
        configs_list: list[dict[str, object]] | None = (
            [dict(record.config) for record in records]
            if records is not None
            else getattr(module, "CONFIGS", None)
        )
        if configs_list is None and policy is None:
            configs_list = self._parse_configs_from_autotune(
                module, kernel_name, heuristic_file, model_source
            )

        # What the backend holds this kernel's unsearched configs to: the
        # same call exports other code once it changes, so it is part of
        # every key an export is known by. No call counts as exported on
        # what other held overrides generated.
        held = self.kernel._unsearched_overrides_identity()

        if self.kernel.settings.static_shapes:
            # Static dispatch already keys exact shape and stride. Enable the two
            # remaining descriptor predicates consistently for the whole file if
            # any heuristic-selected config can use descriptor indexing.
            # Structural-policy (CuTe) files have no Triton descriptor predicates.
            # Under held overrides the backend's hook may change any config,
            # and it answers by the binding, while the predicates are one
            # flag for the file and part of its entry. They are enabled for
            # every call then: predicates read off what the hook made of the
            # configs for one binding would set the calls of a kernel apart,
            # each in an entry that writes the file without the other's
            # variants. Without held overrides the hook is not asked, and
            # the stored configs are what the kernel compiles.
            tensor_descriptor_guards = policy is None and (
                configs_list is None
                or held is not None
                or any(
                    _config_uses_tensor_descriptor(
                        # pyrefly: ignore [bad-argument-type]
                        self.kernel._normalized_config_copy(Config(**config_dict))
                    )
                    for config_dict in configs_list
                )
            )
            self._compile_current_static_shape(
                heuristic_file,
                kernel_name,
                tensor_descriptor_guards=tensor_descriptor_guards,
                model_source=model_source,
            )
            return

        if configs_list is None:
            log.warning("Cannot extract configs from heuristic for '%s'", kernel_name)
            return

        # The stored configs as the kernel selects them for this call's
        # binding, and as its export compiles them.
        selected = [
            self._selected_config(config_dict, policy) for config_dict in configs_list
        ]
        configs = [self._exported_config(config, policy) for config in selected]
        if policy is None and any(map(_config_uses_tensor_descriptor, configs)):
            raise exc.InvalidAPIUsage(
                "dynamic standalone AOT does not support tensor-descriptor "
                "indexing; use pointer or block-pointer indexing"
            )

        args = tuple(self.args)
        env = self.kernel.env
        dtype = self.kernel.settings.index_dtype
        index_limit = None if dtype is None else torch.iinfo(dtype).max
        expected_width = (
            int(env.index_dtype is torch.int64)
            if dtype is None
            else int(env.wide_empty_grid)
        )
        if _resolve_index_widths(self.kernel.settings, args) != (
            env.index_dtype,
            env.wide_empty_grid,
        ):
            raise exc.InvalidAPIUsage(
                "Dynamic standalone inputs no longer match the index width of the bound kernel"
            )
        source = Path(self.kernel.kernel.__code__.co_filename)
        if policy is not None:
            source = source.resolve()
        output_path = policy_path(
            source.with_name(f"{source.stem}_{kernel_name}_standalone.py")
            if source.is_file() or policy is None
            else self.data_dir / f"{kernel_name}_standalone.py",
            policy,
        )
        # Atomic publication replaces this directory entry, not its symlink target.
        output_path = output_path.parent.resolve() / output_path.name
        compile_key: str | tuple[str, ...] = (str(output_path), kernel_name)
        key_source = None
        user_key = getattr(self.kernel.kernel, "_aot_user_key", None)
        if user_key is not None and (policy is not None or len(configs) > 1):
            from .aot_structural_export import _user_key_source

            key_source = _user_key_source(user_key)
        call_keys = None
        source_digest = None
        if policy is not None:
            from .aot_structural_export import _export_call_key
            from .aot_structural_export import export_runtime_guards

            source_digest = hashlib.sha256(
                source.read_bytes()
                if source.is_file()
                else marshal.dumps(self.kernel.kernel.__code__)
            ).hexdigest()
            guards, guard_key = export_runtime_guards(self.kernel, args)
            observed_widths: list[int] = []
            call_keys = (
                _export_call_key(args),
                _export_call_key(
                    args, dynamic=True, index_limit=index_limit, widths=observed_widths
                ),
            )
            if max(observed_widths, default=0) != expected_width:
                raise exc.InvalidAPIUsage(
                    "Standalone signature index width disagrees with the bound kernel"
                )
            compile_key = (
                self.kernel.kernel.__code__.co_filename,
                kernel_name,
                str(output_path),
                self.hardware_id,
                policy.identity(),
                held or "",
                source_digest,
                hashlib.sha256(model_source).hexdigest(),
                self.kernel.config_spec.cache_fingerprint_hash(),
                str(self.kernel.settings.index_dtype),
                repr(call_keys[0]),
                repr((guards, guard_key)),
            )
        ordinary_identity: tuple[object, ...] = ()
        answer: tuple[tuple[object, ...], str, str] | None = None
        saved_codes: dict[int, str] = {}
        initial_facts = ()
        with self._compiled_kernel_variants_lock:
            self._check_dynamic_export_contract(output_path, index_limit, key_source)
            if policy is not None:
                if compile_key in self._compiled_kernels:
                    return
            else:
                from .aot_structural_export import export_specialized_input_facts

                initial_facts = export_specialized_input_facts(self.kernel, args)
                ordinary_identity = (
                    hashlib.sha256(model_source).digest(),
                    hashlib.sha256(source.read_bytes()).digest()
                    if source.is_file()
                    else None,
                    self.hardware_id,
                    index_limit,
                    expected_width,
                    key_source,
                    held,
                    initial_facts,
                )
                if held is not None:
                    # The table is generated from this call's binding and
                    # serves every call, and the backend's hook answers for
                    # a binding: the stored configs, and what the hook made
                    # of them for this one (``_selected_config``).
                    answer = (
                        (*ordinary_identity[:3], held),
                        _configs_text(configs_list),
                        _configs_text(selected),
                    )
                    self._check_ordinary_dynamic_answer(
                        output_path, kernel_name, answer
                    )
                old = self._ordinary_dynamic_children.get(output_path)
                if (
                    old is not None
                    and old[0]() is self.kernel
                    and old[1] == ordinary_identity
                ):
                    saved_codes = old[2]
                    if (
                        len(saved_codes) == len(configs)
                        and self._standalone_export_owners.get(output_path)
                        == compile_key
                    ):
                        return

        # -- generate Triton code for each config --------------------------
        if not configs:
            raise exc.InvalidAPIUsage(
                "Dynamic standalone model has no configs to compile"
            )
        triton_codes: list[str] = []
        compiled_any = False
        failed: set[int] = set()
        for i, config in enumerate(configs):
            if i in saved_codes:
                triton_codes.append(saved_codes[i])
                compiled_any = True
                continue
            try:
                triton_codes.append(self.kernel.to_triton_code(config))
                compiled_any = True
            except Exception:
                failed.add(i)
                if policy is not None or (not compiled_any and i == len(configs) - 1):
                    raise
                log.warning(
                    "Config %d failed to compile for '%s'",
                    i,
                    kernel_name,
                    exc_info=True,
                )
                triton_codes.append(
                    f"def {kernel_name}(*args, **kwargs):\n"
                    f"    raise RuntimeError('Config {i} failed to compile')\n"
                )

        if self.kernel.settings.index_dtype != dtype:
            raise exc.InvalidAPIUsage(
                "Standalone index dtype changed during code generation"
            )
        # -- emit standalone file -------------------------------------------
        from .aot_compile import generate_standalone_file

        if policy is not None:
            assert call_keys is not None and source_digest is not None
            self._compile_policy_dynamic(
                model_source,
                source_digest,
                triton_codes,
                policy,
                call_keys,
                output_path,
                index_limit,
                key_source,
                held,
            )
            AOTAutotuneCache._compiled_kernels.add(compile_key)
            return
        from .aot_structural_export import export_specialized_input_facts

        specialized_input_facts = export_specialized_input_facts(self.kernel, args)
        if saved_codes and specialized_input_facts != initial_facts:
            raise exc.InvalidAPIUsage(
                "Standalone specialization changed during compilation"
            )
        ordinary_identity = (*ordinary_identity[:-1], specialized_input_facts)
        with self._compiled_kernel_variants_lock:
            self._check_dynamic_export_contract(output_path, index_limit, key_source)
            previous = self._ordinary_dynamic_children.get(output_path)
            active = self._standalone_export_owners.get(output_path) == compile_key
            if previous is not None:
                binding_ref, old_identity, old_codes = previous
                if binding_ref() is self.kernel and old_identity == ordinary_identity:
                    if active and len(old_codes) == len(configs):
                        return
                    for index in failed.intersection(old_codes):
                        triton_codes[index] = old_codes[index]
                        failed.remove(index)
                elif active and failed:
                    raise exc.InvalidAPIUsage(
                        "Cannot replace an active standalone file with an incompatible partial binding or model"
                    )
            if answer is not None:
                # Another binding may have published while this one
                # generated its code.
                self._check_ordinary_dynamic_answer(output_path, kernel_name, answer)
            out_path = generate_standalone_file(
                kernel_name=kernel_name,
                index_limit=index_limit,
                expected_width=expected_width,
                specialized_input_facts=specialized_input_facts,
                triton_codes=triton_codes,
                _key_source=key_source,
                heuristic_code=model_source.decode(),
                output_dir=self.data_dir,
                kernel_source_file=self.kernel.kernel.__code__.co_filename,
            )
            self._standalone_export_owners[output_path] = (
                str(output_path),
                kernel_name,
            )
            self._dynamic_export_contracts[output_path] = (index_limit, key_source)
            self._ordinary_dynamic_children[output_path] = (
                weakref.ref(self.kernel),
                ordinary_identity,
                {i: code for i, code in enumerate(triton_codes) if i not in failed},
            )
            if answer is None:
                self._ordinary_dynamic_answers.pop(output_path, None)
            else:
                self._ordinary_dynamic_answers[output_path] = answer
            if not failed:
                AOTAutotuneCache._compiled_kernels.add(compile_key)
        print(f"[AOT] Standalone: {out_path}", file=sys.stderr)

    def _selected_config(
        self, config_dict: dict[str, object], policy: StructuralPolicy | None
    ) -> Config:
        """A config of the heuristic's tuning data as this kernel selects
        it: no search under the kernel's overrides produced it, so the
        backend's hook answers for the kernel's binding
        (``BoundKernel._unsearched_config``)."""
        return self.kernel._unsearched_config(
            Config.from_dict(config_dict)
            if policy is not None
            else Config(**config_dict)  # pyrefly: ignore [bad-argument-type]
        )

    def _exported_config(
        self, config: Config, policy: StructuralPolicy | None
    ) -> Config:
        """``config``, of the tuning data as this kernel selects it
        (``_selected_config``), as the kernel's dynamic export compiles it: a
        plain export normalizes it, so the descriptor check of
        ``_maybe_run_compile`` reads the config that is compiled, not the
        stored one."""
        if policy is not None:
            return config
        return self.kernel._normalized_config_copy(config)

    @classmethod
    def _check_ordinary_dynamic_answer(
        cls,
        path: Path,
        kernel_name: str,
        answer: tuple[tuple[object, ...], str, str],
    ) -> None:
        """Reject an ordinary dynamic export under held overrides whose
        binding the backend's hook answers otherwise for than for the
        binding of the table ``path`` holds.

        Such a table holds the code of one binding for every call, while
        the hook answers for a binding. ``answer`` is what the table is
        generated from besides its binding (the heuristic's and the source's
        bytes, the hardware and the held overrides), the stored configs and
        those configs as the hook left them for the calling binding
        (``_maybe_run_compile``). Where the table last published at
        ``path`` was generated from the same and the hook made something
        else of the same configs, this call would replace that binding's
        code for its calls too, without a variant per binding to tell them
        apart. Without held overrides the hook is not asked and no call is
        rejected. Called with ``_compiled_kernel_variants_lock`` held."""
        exported = cls._ordinary_dynamic_answers.get(path)
        if (
            exported is not None
            and exported[:2] == answer[:2]
            and exported[2] != answer[2]
        ):
            raise exc.InvalidAPIUsage(
                "dynamic standalone AOT holds the code of one binding "
                f"of '{kernel_name}' for every call, and the kernel's "
                f"autotune_config_overrides ({answer[0][-1]}) change the tuning "
                "data's configs for this call's binding otherwise than "
                "for the binding its export is generated from; export "
                "the kernel with static_shapes=True, which keeps a "
                "variant per call"
            )

    @classmethod
    def _check_dynamic_export_contract(
        cls, path: Path, index_limit: int | None, key_source: tuple[str, str] | None
    ) -> None:
        previous = cls._dynamic_export_contracts.get(path)
        if previous is None:
            return
        if previous[0] != index_limit:
            raise StructuralPolicyError(
                "Dynamic standalone variants must use the same index dtype"
            )
        if previous[1] != key_source:
            raise StructuralPolicyError(
                "Dynamic standalone variants must use the same custom key"
            )

    @classmethod
    def _may_publish_structural_variant(
        cls,
        path: Path,
        key: tuple[str | bool, ...],
        unchanged: Callable[[tuple[str | bool, ...]], bool],
    ) -> bool:
        owner = cls._standalone_export_owners.get(path)
        if owner is None or owner == key:
            return True
        # A source-only revision can recompile and verify an existing
        # signature. It cannot install a new table over another binding.
        if owner[:-1] == key[:-1] and unchanged(owner):
            return False
        raise StructuralPolicyError(
            "A different heuristic model, kernel source, hardware, export mode, or "
            "set of held autotune_config_overrides already owns this standalone file"
        )

    def _compile_policy_dynamic(
        self,
        model_source: bytes,
        source_digest: str,
        codes: list[str],
        policy: StructuralPolicy,
        call_keys: tuple[tuple[object, ...], tuple[object, ...]],
        output_path: Path,
        index_limit: int | None,
        key_source: tuple[str, str] | None,
        held: str | None,
    ) -> None:
        from ..language.constexpr import ConstExpr
        from .aot_compile import generate_standalone_file
        from .aot_structural_export import _export_call_key
        from .aot_structural_export import export_runtime_guards

        code_object = self.kernel.kernel.__code__
        source_path = Path(code_object.co_filename).resolve()
        source_file = str(source_path) if source_path.is_file() else None
        if (
            source_digest
            != hashlib.sha256(
                source_path.read_bytes() if source_file else marshal.dumps(code_object)
            ).hexdigest()
        ):
            raise StructuralPolicyError(
                "Kernel source changed during standalone compilation"
            )
        guards, guard_key = export_runtime_guards(self.kernel, tuple(self.args))
        if _export_call_key(tuple(self.args)) != call_keys[0]:
            raise exc.InvalidAPIUsage(
                "Dynamic standalone inputs changed during code generation"
            )
        model_digest = hashlib.sha256(model_source).hexdigest()
        key = (
            "dynamic",
            str(source_path),
            self.kernel.kernel.name,
            str(output_path),
            self.hardware_id,
            policy.identity(),
            # The held overrides ``codes`` were generated under, as the
            # caller read them (``_maybe_run_compile``).
            held or "",
            model_digest,
            source_digest,
        )
        # Structural rewrites can specialize dimensions even with static_shapes
        # disabled (e.g. operand materialization's hl.specialize(shape)). These
        # facts normally live in BoundKernel, not the generated host wrapper.
        env = self.kernel.env
        requires_bound = bool(
            env.specialized_vars
            or env.specialized_strides
            or env.tensor_descriptor_layout_guards
            or ConstExpr in self.kernel.kernel._annotations
            or any(
                fact.fact == "input_tensor_metadata"
                for fact in self.kernel._compiler_seed_specialization_extractors
            )
        )
        group = (
            ("exact", (guards, call_keys[0], guard_key))
            if requires_bound
            else (
                "dynamic",
                (
                    guards,
                    call_keys[1],
                    guard_key,
                ),
            )
        )
        with self._compiled_kernel_variants_lock:
            self._check_dynamic_export_contract(output_path, index_limit, key_source)
            if not self._may_publish_structural_variant(
                output_path,
                key,
                lambda owner: (
                    self._policy_dynamic_variants[owner].get(group) == tuple(codes)
                ),
            ):
                return
            variants = self._policy_dynamic_variants.get(key, {})
            if group in variants:
                if variants[group] != tuple(codes):
                    raise StructuralPolicyError(
                        "Dynamic standalone binding collision: specialize the "
                        "metadata that changes generated code before exporting"
                    )
                return
            updated = variants | {group: tuple(codes)}
            flattened: list[str] = []
            groups: list[tuple[tuple[object, ...], list[int]]] = []
            for signature, group_codes in sorted(
                updated.items(), key=lambda item: repr(item[0])
            ):
                start = len(flattened)
                flattened.extend(group_codes)
                groups.append((signature, list(range(start, len(flattened)))))
            out_path = generate_standalone_file(
                kernel_name=self.kernel.kernel.name,
                triton_codes=flattened,
                heuristic_code=model_source.decode(),
                output_dir=self.data_dir,
                kernel_source_file=source_file,
                structural_policy=policy,
                dynamic_groups=groups,
                index_limit=index_limit,
                runtime_guard_plans=tuple(
                    cast("GuardedCallKey", signature[1])[0] for signature in updated
                ),
                _key_source=key_source,
            )
            self._policy_dynamic_variants[key] = updated
            self._standalone_export_owners[output_path] = key
            self._dynamic_export_contracts[output_path] = (index_limit, key_source)
        print(f"[AOT] Standalone: {out_path}", file=sys.stderr)

    def _compile_current_static_shape(
        self,
        heuristic_file: Path,
        kernel_name: str,
        *,
        tensor_descriptor_guards: bool = False,
        model_source: bytes | None = None,
    ) -> None:
        """Compile one observed static call and update its exact dispatcher."""
        normalized_args = self.kernel.kernel.normalize_args(*self.args)
        if model_source is None:
            config = self._get_heuristic_config(normalized_args)
            model_source = heuristic_file.read_bytes()
        else:
            config = self._get_heuristic_config(
                normalized_args, snapshot=(heuristic_file, model_source)
            )
        if config is None:
            log.warning(
                "No heuristic config for static kernel '%s', skipping standalone compile",
                kernel_name,
            )
            return
        # No search under the kernel's overrides produced the selection: it
        # is exported as the kernel compiles it, in an entry of the overrides
        # the backend holds it to (``_maybe_run_compile``).
        config = self.kernel._unsearched_config(config)
        held = self.kernel._unsearched_overrides_identity() or ""

        from .aot_compile import _standalone_call_key
        from .aot_compile import generate_standalone_file

        policy = bound_structural_policy(self.kernel)
        code_object = self.kernel.kernel.__code__
        source_path = Path(code_object.co_filename).resolve()
        if source_path.is_file():
            source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            kernel_source_file: str | None = str(source_path)
            output_identity = str(
                policy_path(
                    source_path.parent
                    / f"{source_path.stem}_{kernel_name}_standalone.py",
                    policy,
                )
            )
        else:
            source_digest = hashlib.sha256(marshal.dumps(code_object)).hexdigest()
            kernel_source_file = None
            output_identity = str(
                policy_path(
                    self.data_dir.resolve() / f"{kernel_name}_standalone.py", policy
                )
            )
        variants_key = (
            output_identity,
            kernel_name,
            self.hardware_id,
            hashlib.sha256(model_source).hexdigest(),
            source_digest,
            tensor_descriptor_guards,
            held,
        )
        if policy is not None:
            variants_key = (
                "static",
                output_identity,
                kernel_name,
                self.hardware_id,
                policy.identity(),
                held,
                hashlib.sha256(model_source).hexdigest(),
                source_digest,
            )
        config_fingerprint = (
            config_artifact_json(config, policy)
            if policy is not None
            else json.dumps(dict(config), sort_keys=True, default=repr)
        )
        try:
            code = self.kernel.to_triton_code(config)
        except Exception as error:
            raise RuntimeError(
                f"static standalone variant failed to compile for {kernel_name}"
            ) from error
        if code is None:
            raise RuntimeError(
                f"static standalone variant emitted no code for {kernel_name}"
            )
        # Use independent tensor metadata for both the binding check and the
        # dispatcher key. A resize during a classifier must not change the key
        # after validation. A shallow copy shares storage; memoization preserves
        # repeated arguments and their descriptor alignment.
        views: dict[int, torch.Tensor] = {}

        def snapshot(tensor: torch.Tensor) -> torch.Tensor:
            if id(tensor) not in views:
                views[id(tensor)] = copy.copy(tensor)
            return views[id(tensor)]

        normalized_args = torch.utils._pytree.tree_map_only(
            torch.Tensor, snapshot, normalized_args
        )
        # Reuse the recorded custom key without re-running user code.
        saved = self.kernel._base_spec_key
        if (
            self.kernel.kernel._base_specialization_key(
                normalized_args, custom_key=saved[-1]
            )
            != saved
        ):
            raise exc.InvalidAPIUsage(
                "Standalone inputs no longer match the static bound input metadata"
            )
        export_options: dict[str, Any] = {}
        if policy is not None:
            from .aot_structural_export import _export_call_key
            from .aot_structural_export import export_runtime_guards

            exact_key = _export_call_key(normalized_args)
            guards, guard_key = export_runtime_guards(self.kernel, normalized_args)
            call_key = (guards, exact_key, guard_key)
            export_options = {"structural_policy": policy}
        else:
            from .aot_structural_export import export_specialized_input_facts

            export_specialized_input_facts(self.kernel, normalized_args)
            call_key = _standalone_call_key(
                normalized_args, tensor_descriptor_guards=tensor_descriptor_guards
            )

        # The compile workflow runs in one process. Serialize concurrent calls
        # in that process so two newly observed signatures cannot each rewrite
        # an incomplete dispatcher. ``generate_standalone_file`` replaces the
        # output atomically, so readers never observe a partial module.
        with AOTAutotuneCache._compiled_kernel_variants_lock:
            if policy is not None and not self._may_publish_structural_variant(
                Path(output_identity),
                variants_key,
                lambda owner: (
                    self._compiled_kernel_variants[owner].get(call_key)
                    == (config_fingerprint, code)
                ),
            ):
                return
            variants = AOTAutotuneCache._compiled_kernel_variants.setdefault(
                variants_key, {}
            )
            existing = variants.get(call_key)
            if existing is not None:
                if existing != (config_fingerprint, code):
                    raise RuntimeError(
                        f"static standalone call-key collision for {kernel_name}: "
                        "the normalized arguments do not distinguish all generated "
                        "specializations; expose value-derived compile-time metadata "
                        "as scalar or container kernel arguments"
                    )
                if (
                    self._standalone_export_owners.get(Path(output_identity))
                    == variants_key
                ):
                    return

            updated = {**variants, call_key: (config_fingerprint, code)}
            ordered = sorted(updated.items(), key=lambda item: repr(item[0]))
            if policy is not None:
                export_options["runtime_guard_plans"] = tuple(
                    cast("GuardedCallKey", key)[0] for key in updated
                )
            out_path = generate_standalone_file(
                kernel_name=kernel_name,
                triton_codes=[variant[1][1] for variant in ordered],
                heuristic_code=model_source.decode(),
                output_dir=self.data_dir,
                kernel_source_file=kernel_source_file,
                dispatch_keys=[variant[0] for variant in ordered],
                tensor_descriptor_guards=tensor_descriptor_guards,
                **export_options,
            )
            variants[call_key] = (config_fingerprint, code)
            self._standalone_export_owners[Path(output_identity)] = variants_key
            if policy is None:
                # The static table just replaced this ordinary dynamic file.
                self._compiled_kernels.discard((output_identity, kernel_name))
        print(f"[AOT] Standalone: {out_path}", file=sys.stderr)

    @staticmethod
    def _parse_configs_from_autotune(
        module: object, kernel_name: str, path: Path, source: bytes
    ) -> list[dict[str, object]] | None:
        """Extract the ``_C`` config list from ``autotune_<kernel>``. Its
        text is read from ``source``, the bytes ``module`` was executed from
        as the file ``path``, where they define the function, and as
        ``inspect.getsource`` reads it otherwise: from the file of a
        function the heuristic imports. ``None`` where there is no such text
        or no list in it."""
        autotune_fn = getattr(module, f"autotune_{kernel_name}", None)
        if autotune_fn is None:
            return None
        # The function ``inspect.getsource`` reads: the one under decorators.
        defined = inspect.unwrap(autotune_fn)
        if inspect.isfunction(defined) and defined.__code__.co_filename == str(path):
            # The module and the selector being published came from these bytes.
            # inspect.getsource() would reopen the model and could see a replacement.
            lines = source.decode().splitlines(keepends=True)
            src = "".join(
                inspect.getblock(lines[defined.__code__.co_firstlineno - 1 :])
            )
        else:
            try:
                src = inspect.getsource(autotune_fn)
            except (OSError, TypeError):
                # A function of no file, or a callable that is no function.
                return None
        start = src.find("_C = [")
        if start < 0:
            return None
        start += len("_C = ")
        depth = 0
        end = start
        for i in range(start, len(src)):
            if src[i] == "[":
                depth += 1
            elif src[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            return eval(src[start:end])  # list of config dicts
        except Exception:
            return None

    def _get_cache_key(self) -> BoundKernelInMemoryCacheKey:
        """Return a cache key for compatibility."""
        return self.kernel.kernel._create_bound_kernel_cache_key(
            self.kernel,
            tuple(self.args),
            self.kernel.kernel._base_specialization_key(self.args),
        )

    def _list_cache_entries(self) -> Sequence[tuple[str, LooseAutotuneCacheKey]]:
        """List cache entries for compatibility.

        Returns empty list because AOTAutotuneCache uses heuristics rather than
        a traditional cache. The tuned configs are stored in JSON files per
        hardware ID, not in a queryable cache structure.
        """
        return []

    def _run_collect_fn_workflow(self) -> None:
        """Run autotuning on all inputs from collect_fn."""
        assert self._collect_fn is not None
        kernel_name = self.kernel.kernel.name

        print(
            f"[AOT collect_fn] Starting autotuning for {kernel_name}",
            file=sys.stderr,
        )

        count = 0
        for i, input_args in enumerate(self._collect_fn()):
            print(
                f"[AOT collect_fn] Tuning shape {i + 1}...",
                file=sys.stderr,
            )
            self.kernel.kernel(*input_args)
            count = i + 1

        # Reload configs from disk since collect saved new configs
        self._tuned_configs = self._load_tuned_configs()
        print(
            f"[AOT collect_fn] Completed {count} shapes for {kernel_name}",
            file=sys.stderr,
        )

    def _run_measure_fn_workflow(self) -> None:
        """Run measurement on all inputs from measure_fn."""
        assert self._measure_fn is not None
        kernel_name = self.kernel.kernel.name
        all_configs = self._get_all_configs_for_kernel(kernel_name)

        if not all_configs:
            print(
                f"[AOT measure_fn] Warning: No configs found for {kernel_name}",
                file=sys.stderr,
            )
            return

        print(
            f"[AOT measure_fn] Starting measurement of {len(all_configs)} configs "
            f"for {kernel_name}",
            file=sys.stderr,
        )

        count = 0
        for i, input_args in enumerate(self._measure_fn()):
            print(
                f"[AOT measure_fn] Measuring shape {i + 1}...",
                file=sys.stderr,
            )
            count = i + 1
            spec_key = self.kernel.kernel.specialization_key(input_args)
            shape_key = ShapeKey(
                kernel_name=kernel_name,
                specialization_key=spec_key,
                hardware_id=self.hardware_id,
            )
            shape_features = self._extract_shape_features(input_args)

            for config in all_configs:
                try:
                    bound = self.kernel.kernel.bind(input_args)
                    # Timed as this binding compiles it, and saved under the
                    # stored config (``measure_all_configs``).
                    compiled = bound._unsearched_config(config)
                    with _searching_candidates():
                        fn = bound.compile_config(compiled)

                    from triton.testing import do_bench

                    timing = do_bench(lambda fn=fn, args=input_args: fn(*args))
                    assert isinstance(timing, float)

                    self._save_measurement(
                        kernel_name=kernel_name,
                        shape_key=shape_key,
                        config=config,
                        timing_ms=timing,
                        shape_features=shape_features,
                    )
                except Exception as e:
                    log.debug(f"Failed to measure config {config}: {e}")

        print(
            f"[AOT measure_fn] Completed {count} shapes! "
            f"Results saved to {self._measurements_file}",
            file=sys.stderr,
        )

    def _maybe_run_input_fn_workflows(self) -> None:
        """Run collect_fn/measure_fn workflows if applicable."""
        # Check if input_fn workflow should run (only once per kernel)
        if getattr(self.kernel.kernel, "_aot_workflow_done", False):
            return
        # Mark done FIRST to prevent recursive calls when we invoke the kernel
        self.kernel.kernel._aot_workflow_done = True  # type: ignore[attr-defined]

        if self.mode == "collect" and self._collect_fn is not None:
            self._run_collect_fn_workflow()
            if self._measure_fn is not None:
                # One-shot: run measure immediately after collect
                self._run_measure_fn_workflow()

        elif self.mode == "measure" and self._measure_fn is not None:
            # Only run if measurements don't already exist (avoids duplicate work
            # when runner calls measure phase after one-shot collect already ran it)
            if not self._measurements_file.exists():
                self._run_measure_fn_workflow()

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Perform autotuning based on current mode."""
        self._maybe_run_input_fn_workflows()

        if self.mode == "collect":
            # Collect mode: autotune this shape and save + return the config
            return super().autotune(skip_cache=skip_cache)

        if self.mode == "measure":
            # Measure mode: benchmark all known configs for this shape and return the best config
            results = self.measure_all_configs()
            if results:
                best_config, best_timing = min(results, key=operator.itemgetter(1))
                log.info(
                    f"AOT measure: Best config for {self.kernel.kernel.name} "
                    f"shape={self.shape_key.stable_hash()} timing={best_timing:.4f}ms"
                )
                # Collected tuning data: no search under the kernel's
                # overrides need have written it.
                return self.kernel._unsearched_config(best_config, report=True)
            # Fall through to regular autotuning if no configs available
            log.warning("No configs to measure, falling through to autotuner")

        # Use parent implementation for other modes
        # Note: super().autotune() internally calls self.put() before returning
        return super().autotune(skip_cache=skip_cache)


def _code_identity(
    code: types.CodeType,
) -> tuple[bytes, tuple[Any, ...], tuple[str, ...]]:
    """Semantic identity of a code object: bytecode + constants + referenced names."""
    return (code.co_code, code.co_consts, code.co_names)


def _serialize_value(val: object) -> object:
    """Serialize a single value to JSON-compatible format.

    Supports: None, bool, int, float, str, type, tuple, frozenset, set,
    torch.dtype, torch.device, list, dict, code, bytes, ellipsis, complex,
    and the runtime's typed boolean, float and fragment specialization keys.
    """
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, _BoolValueKey):
        return {"__bool_value_key__": val.value}
    if isinstance(val, _FloatValueKey):
        return {"__float_value_key__": val.bits.hex()}
    if isinstance(val, _FragmentValueKey):
        return {
            "__fragment_value_key__": [
                _serialize_value(val.fragment_type),
                _serialize_value(val.fields),
            ]
        }
    if isinstance(val, _FragmentSetValueKey):
        return {"__fragment_set_value_key__": _serialize_value(val.entries)}
    if isinstance(val, type):
        return {"__type__": f"{val.__module__}.{val.__qualname__}"}
    if isinstance(val, tuple):
        return {"__tuple__": [_serialize_value(v) for v in val]}
    if isinstance(val, frozenset):
        return {"__frozenset__": [_serialize_value(v) for v in sorted(val, key=str)]}
    if isinstance(val, set):
        return {"__set__": [_serialize_value(v) for v in sorted(val, key=str)]}
    if isinstance(val, torch.dtype):
        return {"__dtype__": str(val)}
    if isinstance(val, torch.device):
        return {"__device__": str(val)}
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, types.CodeType):
        return _serialize_value(_code_identity(val))
    if isinstance(val, bytes):
        return {"__bytes__": val.hex()}
    if isinstance(val, types.EllipsisType):
        return {"__ellipsis__": True}
    if isinstance(val, complex):
        return {"__complex__": [val.real, val.imag]}
    raise TypeError(f"Cannot serialize type: {type(val).__name__}")


def _deserialize_value(val: object) -> object:
    """Deserialize a JSON value back to Python object.

    Handles tagged dicts: __tuple__, __frozenset__, __set__, __dtype__, __device__,
    __type__, __bytes__, __ellipsis__, __complex__, and typed runtime keys.

    Note: __code__ is not turned back into a code object. This is fine because the AOT
    cache only compares specialization keys, it never calls the original function.
    """
    if isinstance(val, dict):
        if "__bool_value_key__" in val:
            return _BoolValueKey(val["__bool_value_key__"])
        if "__float_value_key__" in val:
            return _FloatValueKey(bytes.fromhex(val["__float_value_key__"]))
        if "__fragment_value_key__" in val:
            fragment_type, fields = val["__fragment_value_key__"]
            return _FragmentValueKey(
                cast("type[ConfigSpecFragment]", _deserialize_value(fragment_type)),
                cast("Hashable", _deserialize_value(fields)),
            )
        if "__fragment_set_value_key__" in val:
            return _FragmentSetValueKey(
                cast(
                    "frozenset[tuple[Hashable, int]]",
                    _deserialize_value(val["__fragment_set_value_key__"]),
                )
            )
        if "__tuple__" in val:
            return tuple(_deserialize_value(v) for v in val["__tuple__"])
        if "__frozenset__" in val:
            return frozenset(_deserialize_value(v) for v in val["__frozenset__"])
        if "__set__" in val:
            return {_deserialize_value(v) for v in val["__set__"]}
        if "__dtype__" in val:
            dtype_name = val["__dtype__"].replace("torch.", "")
            return getattr(torch, dtype_name)
        if "__device__" in val:
            return torch.device(val["__device__"])
        if "__type__" in val:
            return _import_type(val["__type__"])
        if "__bytes__" in val:
            return bytes.fromhex(val["__bytes__"])
        if "__ellipsis__" in val:
            return ...
        if "__complex__" in val:
            real, imag = _deserialize_tuple(val["__complex__"])
            return complex(real, imag)
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_deserialize_value(v) for v in val]
    return val


def _import_type(type_name: str) -> type:
    """Import a type from its fully qualified name."""
    if type_name == "builtins.NoneType":
        # NoneType's qualified name has no matching builtins attribute. Do
        # not fall back to the unrelated torch.NoneType IR class.
        return type(None)
    parts = type_name.rsplit(".", 1)
    if len(parts) == 2:
        module_name, class_name = parts
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass

    # Fallback: try common modules
    for module_name in ["builtins", "torch", "helion"]:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, type_name.split(".")[-1]):
                return getattr(module, type_name.split(".")[-1])
        except (ImportError, AttributeError):
            pass

    raise ValueError(f"Cannot import type: {type_name}")


def _serialize_tuple(t: tuple[Any, ...]) -> list[Any]:
    """Serialize a tuple to JSON-compatible list."""
    return [_serialize_value(item) for item in t]


def _deserialize_tuple(data: list[Any]) -> tuple[Any, ...]:
    """Deserialize a list back to tuple."""
    return tuple(_deserialize_value(item) for item in data)
