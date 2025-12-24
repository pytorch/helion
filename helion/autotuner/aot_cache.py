"""
AOT (Ahead-of-Time) Autotuning Cache Implementation
====================================================

This module provides a cache implementation for AOT autotuning workflows that:
1. Collects tuned configs for each shape during benchmark runs
2. Measures all configs across all shapes
3. Generates heuristics using LightGBM to select optimal configs
4. Supports multiple hardware architectures

The workflow is:
1. collect_tuned_configs: Tune each shape, record (kernel, shape, config) triples
2. measure_configs: Measure each shape with all observed configs
3. Generate heuristics to select configs based on performance goals
4. evaluate: Validate performance goals are achieved
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import logging
import operator
import os
from pathlib import Path
import platform
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import torch

from ..runtime.config import Config
from .base_cache import AutotuneCacheBase
from .base_cache import LooseAutotuneCacheKey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)

# Environment variable to control AOT mode
AOT_MODE_ENV = "HELION_AOT_MODE"
AOT_DATA_DIR_ENV = "HELION_AOT_DATA_DIR"
# Environment variable to override heuristic search path (for comparing heuristics)
HEURISTIC_DIR_ENV = "HELION_HEURISTIC_DIR"
# Environment variable to enable verbose output in evaluate mode (default: quiet)
AOT_VERBOSE_ENV = "HELION_AOT_VERBOSE"

AOTMode = Literal["collect", "measure", "evaluate", "disabled"]


def get_aot_mode() -> AOTMode:
    """Get the current AOT mode from environment."""
    mode = os.environ.get(AOT_MODE_ENV, "evaluate").lower()
    if mode in ("collect", "measure", "evaluate", "disabled"):
        return mode  # type: ignore[return-value]
    raise ValueError(
        f"Invalid {AOT_MODE_ENV} value: {mode}. "
        "Must be one of: collect, measure, evaluate, disabled"
    )


def is_aot_verbose() -> bool:
    """Check if verbose output is enabled for AOT mode.

    In evaluate mode, output is quiet by default (just using heuristics).
    Set HELION_AOT_VERBOSE=1 to enable verbose output.
    """
    return os.environ.get(AOT_VERBOSE_ENV, "").lower() in ("1", "true", "yes")


def get_aot_data_dir() -> Path:
    """Get the AOT data directory from environment or default."""
    if (path := os.environ.get(AOT_DATA_DIR_ENV)) is not None:
        return Path(path)
    return Path.cwd() / ".helion_aot"


def get_hardware_id() -> str:
    """Get a unique identifier for the current hardware."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if torch.version.cuda is not None:
            return f"cuda_{props.name.replace(' ', '_')}_{torch.version.cuda}"
        if torch.version.hip is not None:
            return f"rocm_{props.gcnArchName}_{torch.version.hip}"
    return f"cpu_{platform.machine()}"


def get_device_compute_id() -> tuple[str, str]:
    """
    Get the device kind and compute capability/architecture.

    Returns:
        Tuple of (device_kind, compute_kind) where:
        - device_kind: 'cuda', 'rocm', or 'cpu'
        - compute_kind: 'sm90' for CUDA, 'gfx90a' for ROCm, or architecture for CPU
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if torch.version.cuda is not None:
            # CUDA: use compute capability like sm90, sm89, sm80
            return ("cuda", f"sm{props.major}{props.minor}")
        if torch.version.hip is not None:
            # ROCm: use gcnArchName like gfx90a, gfx942
            return ("rocm", props.gcnArchName)
    return ("cpu", platform.machine())


# Known compute capabilities in descending order (newest first)
# This allows fallback to older architectures when heuristics aren't available
_CUDA_COMPUTE_CAPS: list[str] = [
    "sm100",
    "sm90",
    "sm89",
    "sm87",
    "sm86",
    "sm80",
    "sm75",
    "sm72",
    "sm70",
]

_ROCM_ARCHS: list[str] = [
    "gfx950",
    "gfx942",
    "gfx941",
    "gfx940",
    "gfx90a",
    "gfx908",
    "gfx906",
    "gfx900",
]


def get_compatible_compute_ids(device_kind: str, compute_kind: str) -> list[str]:
    """
    Get a list of compatible compute IDs for fallback, ordered from current to oldest.

    For CUDA/ROCm, returns the current compute capability followed by all older
    compatible architectures. This allows using heuristics tuned on older hardware
    when newer hardware-specific heuristics aren't available.

    Args:
        device_kind: 'cuda', 'rocm', or 'cpu'
        compute_kind: The current compute capability (e.g., 'sm90', 'gfx942')

    Returns:
        List of compute IDs to try, starting with the exact match
    """
    if device_kind == "cuda":
        arch_list = _CUDA_COMPUTE_CAPS
    elif device_kind == "rocm":
        arch_list = _ROCM_ARCHS
    else:
        # CPU or unknown - no fallback
        return [compute_kind]

    # Find current architecture in the list
    try:
        current_idx = arch_list.index(compute_kind)
        # Return current and all older architectures
        return arch_list[current_idx:]
    except ValueError:
        # Unknown architecture - try it alone, then try all known ones
        return [compute_kind, *arch_list]


def get_heuristic_path_for_kernel(kernel_source_file: str | Path) -> Path:
    """
    Get the path where heuristics should be stored for a kernel.

    The heuristic file is placed next to the kernel source file with naming:
    _<original_filename>_<device_kind>_<compute_kind>.py

    For example:
    - my_kernels.py on CUDA sm90 -> _my_kernels_cuda_sm90.py
    - ops.py on ROCm gfx90a -> _ops_rocm_gfx90a.py
    """
    source_path = Path(kernel_source_file)
    device_kind, compute_kind = get_device_compute_id()

    # Get the base filename without extension
    base_name = source_path.stem

    # Create the heuristic filename
    heuristic_name = f"_{base_name}_{device_kind}_{compute_kind}.py"

    return source_path.parent / heuristic_name


@dataclass
class ShapeKey:
    """Represents a unique shape/dtype combination for a kernel."""

    kernel_name: str
    specialization_key: tuple[Any, ...]
    hardware_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "kernel_name": self.kernel_name,
            "specialization_key": _serialize_tuple(self.specialization_key),
            "hardware_id": self.hardware_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShapeKey:
        """Create from a dict."""
        return cls(
            kernel_name=data["kernel_name"],
            specialization_key=_deserialize_tuple(data["specialization_key"]),
            hardware_id=data["hardware_id"],
        )

    def stable_hash(self) -> str:
        """Get a stable hash for this shape key."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass
class TensorHashes:
    """SHA256 hashes (first 8 chars) for input/output tensors."""

    input_hashes: list[str]  # Hash of each input tensor's bytes
    output_hashes: list[str] | None = (
        None  # Hash of each output tensor's bytes (after run)
    )
    input_after_run_hashes: list[str] | None = (
        None  # Hash of inputs after run (detect modification)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "input_hashes": self.input_hashes,
            "output_hashes": self.output_hashes,
            "input_after_run_hashes": self.input_after_run_hashes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorHashes:
        """Create from dict."""
        return cls(
            input_hashes=data.get("input_hashes", []),
            output_hashes=data.get("output_hashes"),
            input_after_run_hashes=data.get("input_after_run_hashes"),
        )


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute SHA256 hash (first 8 chars) of tensor bytes."""
    # Ensure tensor is contiguous for consistent hashing
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    # Move to CPU if needed for hashing
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    # Get raw bytes and hash
    data = tensor.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:8]


@dataclass
class TunedConfig:
    """A tuned configuration with its benchmark results."""

    config: Config
    shape_key: ShapeKey
    timing_ms: float | None = None
    is_optimal_for_shape: bool = False
    kernel_source_file: str | None = None  # Path to the kernel source file
    shape_features: dict[str, Any] | None = None  # Shape features for this config
    tensor_hashes: TensorHashes | None = None  # Hashes of input/output tensors


@dataclass
class AOTDataStore:
    """Manages persisted AOT tuning data."""

    data_dir: Path
    hardware_id: str

    # In-memory caches
    _tuned_configs: dict[str, list[TunedConfig]] = field(default_factory=dict)
    _measurements: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def configs_file(self) -> Path:
        """Path to the tuned configs JSON file."""
        return self.data_dir / f"tuned_configs_{self.hardware_id}.json"

    @property
    def measurements_file(self) -> Path:
        """Path to the measurements CSV file."""
        return self.data_dir / f"measurements_{self.hardware_id}.csv"

    @property
    def heuristic_file(self) -> Path:
        """Path to the generated heuristic Python file."""
        return self.data_dir / f"heuristic_{self.hardware_id}.py"

    @property
    def model_file(self) -> Path:
        """Path to the LightGBM model file."""
        return self.data_dir / f"model_{self.hardware_id}.txt"

    def load_tuned_configs(self) -> dict[str, list[TunedConfig]]:
        """Load tuned configs from disk."""
        if not self.configs_file.exists():
            return {}

        try:
            data = json.loads(self.configs_file.read_text())
            result: dict[str, list[TunedConfig]] = {}
            for kernel_name, configs in data.items():
                result[kernel_name] = [
                    TunedConfig(
                        config=Config(**_deserialize_config(cfg["config"])),
                        shape_key=ShapeKey.from_dict(cfg["shape_key"]),
                        timing_ms=cfg.get("timing_ms"),
                        is_optimal_for_shape=cfg.get("is_optimal_for_shape", False),
                        kernel_source_file=cfg.get("kernel_source_file"),
                        shape_features=cfg.get("shape_features"),
                        tensor_hashes=TensorHashes.from_dict(cfg["tensor_hashes"])
                        if cfg.get("tensor_hashes")
                        else None,
                    )
                    for cfg in configs
                ]
            return result
        except Exception as e:
            log.warning(f"Failed to load tuned configs: {e}")
            return {}

    def save_tuned_configs(self, configs: dict[str, list[TunedConfig]]) -> None:
        """Save tuned configs to disk."""
        data: dict[str, list[dict[str, Any]]] = {}
        for kernel_name, config_list in configs.items():
            data[kernel_name] = [
                {
                    "config": _serialize_config(cfg.config),
                    "shape_key": cfg.shape_key.to_dict(),
                    "timing_ms": cfg.timing_ms,
                    "is_optimal_for_shape": cfg.is_optimal_for_shape,
                    "kernel_source_file": cfg.kernel_source_file,
                    "shape_features": cfg.shape_features,
                    "tensor_hashes": cfg.tensor_hashes.to_dict()
                    if cfg.tensor_hashes
                    else None,
                }
                for cfg in config_list
            ]

        self.configs_file.write_text(json.dumps(data, indent=2))

    def add_tuned_config(
        self,
        kernel_name: str,
        config: Config,
        shape_key: ShapeKey,
        timing_ms: float | None = None,
        is_optimal: bool = True,
        kernel_source_file: str | None = None,
        shape_features: dict[str, Any] | None = None,
        tensor_hashes: TensorHashes | None = None,
    ) -> None:
        """
        Add a newly tuned config.

        If timing_ms is provided and a config already exists for this shape,
        only updates if the new config is better (lower timing).
        """
        if kernel_name not in self._tuned_configs:
            self._tuned_configs[kernel_name] = []

        shape_hash = shape_key.stable_hash()
        config_dict = dict(config)

        # Check if this exact config already exists for this shape
        for existing in self._tuned_configs[kernel_name]:
            if (
                existing.shape_key.stable_hash() == shape_hash
                and dict(existing.config) == config_dict
            ):
                # Same config for same shape - update if we have better timing
                if timing_ms is not None:
                    if existing.timing_ms is None or timing_ms < existing.timing_ms:
                        existing.timing_ms = timing_ms
                # Update other fields if provided
                if kernel_source_file is not None:
                    existing.kernel_source_file = kernel_source_file
                if shape_features is not None:
                    existing.shape_features = shape_features
                if tensor_hashes is not None:
                    existing.tensor_hashes = tensor_hashes
                return

        # Check if we have a different config for this shape - keep best
        # Find existing configs for this shape
        existing_for_shape = [
            tc
            for tc in self._tuned_configs[kernel_name]
            if tc.shape_key.stable_hash() == shape_hash and tc.is_optimal_for_shape
        ]

        if existing_for_shape and timing_ms is not None:
            # We have existing configs for this shape - check if new one is better
            for existing in existing_for_shape:
                if existing.timing_ms is not None and timing_ms >= existing.timing_ms:
                    # New config is not better, still add but mark as not optimal
                    is_optimal = False
                    break
            else:
                # New config is better - mark old ones as not optimal
                for existing in existing_for_shape:
                    existing.is_optimal_for_shape = False
                is_optimal = True

        self._tuned_configs[kernel_name].append(
            TunedConfig(
                config=config,
                shape_key=shape_key,
                timing_ms=timing_ms,
                is_optimal_for_shape=is_optimal,
                kernel_source_file=kernel_source_file,
                shape_features=shape_features,
                tensor_hashes=tensor_hashes,
            )
        )

    def get_all_configs_for_kernel(self, kernel_name: str) -> list[Config]:
        """Get all unique configs observed for a kernel."""
        if kernel_name not in self._tuned_configs:
            return []

        seen: set[str] = set()
        result: list[Config] = []
        for tc in self._tuned_configs[kernel_name]:
            config_hash = hashlib.sha256(
                json.dumps(dict(tc.config), sort_keys=True).encode()
            ).hexdigest()
            if config_hash not in seen:
                seen.add(config_hash)
                result.append(tc.config)
        return result

    def get_kernel_source_files(self) -> dict[str, str]:
        """
        Get mapping of kernel names to their source files.

        Returns dict mapping kernel_name -> source_file_path.
        Only includes kernels where source file was recorded.
        """
        result: dict[str, str] = {}
        for kernel_name, configs in self._tuned_configs.items():
            for tc in configs:
                if tc.kernel_source_file:
                    result[kernel_name] = tc.kernel_source_file
                    break  # Only need one source file per kernel
        return result

    def load_measurements(self) -> list[dict[str, Any]]:
        """Load measurements from CSV."""
        if not self.measurements_file.exists():
            return []

        try:
            measurements: list[dict[str, Any]] = []
            with open(self.measurements_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    measurements.append(
                        {
                            "kernel_name": row["kernel_name"],
                            "shape_hash": row["shape_hash"],
                            "config_hash": row["config_hash"],
                            "config": json.loads(row["config"]),
                            "shape_features": json.loads(row["shape_features"]),
                            "timing_ms": float(row["timing_ms"]),
                        }
                    )
            return measurements
        except Exception as e:
            log.warning(f"Failed to load measurements: {e}")
            return []

    def save_measurement(
        self,
        kernel_name: str,
        shape_key: ShapeKey,
        config: Config,
        timing_ms: float,
        shape_features: dict[str, Any],
    ) -> None:
        """Save a measurement to CSV."""
        config_hash = hashlib.sha256(
            json.dumps(dict(config), sort_keys=True).encode()
        ).hexdigest()[:16]

        row = {
            "kernel_name": kernel_name,
            "shape_hash": shape_key.stable_hash(),
            "config_hash": config_hash,
            "config": json.dumps(dict(config)),
            "shape_features": json.dumps(shape_features),
            "timing_ms": timing_ms,
        }

        file_exists = self.measurements_file.exists()
        with open(self.measurements_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self._measurements.append(row)

    def flush(self) -> None:
        """Flush all in-memory data to disk."""
        if self._tuned_configs:
            # _tuned_configs already contains loaded data + new additions
            # Just save the complete set
            self.save_tuned_configs(self._tuned_configs)
            log.debug(
                f"Flushed {sum(len(v) for v in self._tuned_configs.values())} configs to {self.configs_file}"
            )


class AOTAutotuneCache(AutotuneCacheBase):
    """
    Cache implementation for AOT autotuning workflows.

    Behavior depends on the HELION_AOT_MODE environment variable:
    - collect: Tune each shape individually, record results
    - measure: Measure each shape with all observed configs
    - evaluate: Use heuristics to select configs, validate performance
    - disabled: Fall through to underlying autotuner (default)
    """

    _mode_announced: ClassVar[set[str]] = (
        set()
    )  # Class-level to avoid repeated messages

    # Class-level caches for heuristic lookup (shared across instances)
    # Maps heuristic file path -> loaded module
    _heuristic_modules: ClassVar[dict[Path, Any]] = {}
    # Maps (kernel_source_file, kernel_name, shape_features_hash) -> Config
    # Using source file ensures kernels with same name in different modules don't collide
    _heuristic_results: ClassVar[dict[tuple[str, str, str], Config]] = {}
    # Maps kernel_source_file -> heuristic file Path (or None if not found)
    # This avoids repeated filesystem lookups for the same kernel
    _heuristic_file_cache: ClassVar[dict[str, Path | None]] = {}

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all class-level caches (heuristic modules and results)."""
        cls._heuristic_modules.clear()
        cls._heuristic_results.clear()
        cls._heuristic_file_cache.clear()
        cls._mode_announced.clear()
        log.debug("Cleared AOTAutotuneCache caches")

    def __init__(self, autotuner: BaseSearch) -> None:
        super().__init__(autotuner)
        self.mode = get_aot_mode()
        self.hardware_id = get_hardware_id()
        self.data_store = AOTDataStore(get_aot_data_dir(), self.hardware_id)
        self.shape_key = self._create_shape_key()
        self._verbose = is_aot_verbose()

        # Load existing data
        self.data_store._tuned_configs = self.data_store.load_tuned_configs()

        # Announce mode once per mode type (quiet in evaluate mode unless verbose)
        should_announce = (
            self.mode != "disabled"
            and self.mode not in AOTAutotuneCache._mode_announced
            and (self.mode != "evaluate" or self._verbose)
        )
        if should_announce:
            import sys

            print(
                f"[AOT] Mode: {self.mode}, Data dir: {self.data_store.data_dir}, "
                f"Hardware: {self.hardware_id}",
                file=sys.stderr,
            )
            num_configs = sum(len(v) for v in self.data_store._tuned_configs.values())
            if num_configs > 0:
                print(f"[AOT] Loaded {num_configs} existing configs", file=sys.stderr)
            AOTAutotuneCache._mode_announced.add(self.mode)

    def _create_shape_key(self) -> ShapeKey:
        """Create a shape key for the current kernel invocation."""
        return ShapeKey(
            kernel_name=self.kernel.kernel.name,
            specialization_key=self.kernel.kernel.specialization_key(self.args),
            hardware_id=self.hardware_id,
        )

    def _extract_shape_features(
        self, args: Sequence[object] | None = None
    ) -> dict[str, Any]:
        """Extract numeric features from the shape for ML model."""
        if args is None:
            args = self.args

        features: dict[str, Any] = {}

        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                features[f"arg{i}_ndim"] = arg.ndim
                for j, size in enumerate(arg.shape):
                    features[f"arg{i}_dim{j}"] = int(size)
                features[f"arg{i}_numel"] = int(arg.numel())
                # Keep string dtype for readability
                features[f"arg{i}_dtype"] = str(arg.dtype)
                # Add numeric dtype size (bytes) for ML model - larger types = larger numbers
                features[f"arg{i}_dtype_size"] = arg.element_size()
                # Add dtype category: 0=int, 1=float, 2=complex, 3=other
                features[f"arg{i}_dtype_cat"] = _get_dtype_category(arg.dtype)
            elif isinstance(arg, (int, float)):
                features[f"arg{i}_scalar"] = arg

        return features

    def get(self) -> Config | None:
        """Get a cached config based on current mode."""
        if self.mode == "disabled":
            return None

        if self.mode == "collect":
            # In collect mode, check if we already have a config for this exact shape
            kernel_name = self.kernel.kernel.name
            configs = self.data_store._tuned_configs.get(kernel_name, [])
            for tc in configs:
                if tc.shape_key.stable_hash() == self.shape_key.stable_hash():
                    log.info(f"AOT collect: Using existing config for {kernel_name}")
                    return tc.config
            return None  # Need to tune

        if self.mode == "measure":
            # In measure mode, we don't use cache - we measure all configs
            return None

        if self.mode == "evaluate":
            # In evaluate mode, use the heuristic to select config
            return self._get_heuristic_config()

        return None

    def _compute_tensor_hashes(self) -> TensorHashes:
        """Compute SHA256 hashes (first 8 chars) for input tensors."""
        input_hashes = []
        for arg in self.args:
            if isinstance(arg, torch.Tensor):
                try:
                    input_hashes.append(compute_tensor_hash(arg))
                except Exception as e:
                    log.debug(f"Failed to hash tensor: {e}")
                    input_hashes.append("error")
            else:
                # For non-tensor args, use a placeholder
                input_hashes.append("n/a")
        return TensorHashes(input_hashes=input_hashes)

    def put(self, config: Config, timing_ms: float | None = None) -> None:
        """Store a tuned config based on current mode."""
        if self.mode == "disabled":
            return

        if self.mode == "collect":
            # Store the tuned config with kernel source file location
            kernel_name = self.kernel.kernel.name
            kernel_source_file = self.kernel.kernel.__code__.co_filename

            # Extract shape features and tensor hashes
            shape_features = self._extract_shape_features()
            tensor_hashes = self._compute_tensor_hashes()

            self.data_store.add_tuned_config(
                kernel_name=kernel_name,
                config=config,
                shape_key=self.shape_key,
                timing_ms=timing_ms,
                is_optimal=True,
                kernel_source_file=kernel_source_file,
                shape_features=shape_features,
                tensor_hashes=tensor_hashes,
            )
            self.data_store.flush()

            # Print to stderr so it's visible even without logging configured
            import sys

            # Build hash info string
            hash_info = ""
            if tensor_hashes.input_hashes:
                hash_info = f" input_hashes=[{','.join(tensor_hashes.input_hashes)}]"

            print(
                f"[AOT collect] Saved config for kernel={kernel_name} "
                f"shape_hash={self.shape_key.stable_hash()[:8]}"
                f"{hash_info} "
                f"to {self.data_store.configs_file}",
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
        import sys
        import tempfile
        import traceback

        kernel_name = self.kernel.kernel.name
        all_configs = self.data_store.get_all_configs_for_kernel(kernel_name)

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

        # Set up tmpdir if needed (normally done inside autotune())
        tmpdir_created = False
        if self.autotuner._precompile_tmpdir is None:
            self.autotuner._precompile_tmpdir = tempfile.TemporaryDirectory()
            tmpdir_created = True

        try:
            for i, config in enumerate(all_configs):
                try:
                    # Benchmark this config
                    fn, timing = self.autotuner.benchmark(config)
                    if timing < float("inf"):
                        results.append((config, timing))

                        # Save measurement
                        self.data_store.save_measurement(
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
            if tmpdir_created and self.autotuner._precompile_tmpdir is not None:
                self.autotuner._precompile_tmpdir.cleanup()
                self.autotuner._precompile_tmpdir = None

        print(
            f"[AOT measure] Completed: {len(results)}/{len(all_configs)} configs succeeded",
            file=sys.stderr,
        )
        return results

    def _find_heuristic_file(self) -> Path | None:
        """
        Find the heuristic file for this kernel.

        Search order:
        1. HELION_HEURISTIC_DIR env var (if set) - for comparing different heuristics
        2. Next to kernel source file: _<filename>_<device>_<compute>.py
        3. Fallback to older compute capabilities within the same device family
        4. AOT data directory: heuristic_<kernel_name>.py (fallback)

        Results are cached to avoid repeated filesystem lookups.
        """
        kernel_name = self.kernel.kernel.name

        # Get the kernel source file path
        kernel_source_file = self.kernel.kernel.__code__.co_filename

        # Check cache first (use source file + kernel name as key since
        # different kernels in the same file may have different heuristics)
        cache_key = kernel_source_file
        if cache_key in AOTAutotuneCache._heuristic_file_cache:
            return AOTAutotuneCache._heuristic_file_cache[cache_key]

        source_path = Path(kernel_source_file)
        base_name = source_path.stem

        # Get device info and compatible compute capabilities
        device_kind, compute_kind = get_device_compute_id()
        compatible_computes = get_compatible_compute_ids(device_kind, compute_kind)

        # Build list of candidate heuristic files in priority order
        candidates: list[Path] = []

        # 1. Check HELION_HEURISTIC_DIR override (for comparing heuristics)
        if (heuristic_dir := os.environ.get(HEURISTIC_DIR_ENV)) is not None:
            heuristic_dir_path = Path(heuristic_dir)
            # Try each compatible compute capability in order
            for compat_compute in compatible_computes:
                candidates.append(
                    heuristic_dir_path
                    / f"_{base_name}_{device_kind}_{compat_compute}.py"
                )
            # Also check kernel-specific file in override dir
            candidates.append(heuristic_dir_path / f"heuristic_{kernel_name}.py")

        # 2. Check next to kernel source file with compute capability fallback
        for compat_compute in compatible_computes:
            heuristic_name = f"_{base_name}_{device_kind}_{compat_compute}.py"
            candidates.append(source_path.parent / heuristic_name)

        # 3. Check AOT data directory (fallback for backward compatibility)
        candidates.extend(
            [
                self.data_store.data_dir / f"heuristic_{kernel_name}.py",
                self.data_store.heuristic_file,
            ]
        )

        # Find first existing heuristic file
        result: Path | None = None
        for candidate in candidates:
            if candidate.exists():
                log.debug(f"Found heuristic file: {candidate}")
                result = candidate
                break

        if result is None:
            log.debug(
                f"Heuristic file not found for {kernel_name}. Searched: "
                f"{[str(c) for c in candidates[:3]]}..."
            )

        # Cache the result (even if None, to avoid repeated searches)
        AOTAutotuneCache._heuristic_file_cache[cache_key] = result
        return result

    def _get_heuristic_config(
        self, args: Sequence[object] | None = None
    ) -> Config | None:
        """
        Use the heuristic to select a config.

        Args:
            args: Optional arguments to use for shape feature extraction.
                  If None, uses self.args.

        For CUDA/ROCm, if heuristics for the current compute capability aren't found,
        we try older compatible architectures (e.g., sm80 heuristics on sm90 hardware).
        """
        heuristic_file = self._find_heuristic_file()
        if heuristic_file is None:
            return None

        kernel_name = self.kernel.kernel.name
        kernel_source_file = self.kernel.kernel.__code__.co_filename

        # Extract shape features and compute hash for caching
        shape_features = self._extract_shape_features(args)
        shape_hash = hashlib.sha256(
            json.dumps(shape_features, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Check if we already have a cached result for this kernel+shape
        # Include source file in key to avoid collisions between kernels with same name
        cache_key = (kernel_source_file, kernel_name, shape_hash)
        if cache_key in AOTAutotuneCache._heuristic_results:
            log.debug(
                f"Using cached heuristic result for {kernel_name} shape={shape_hash}"
            )
            return AOTAutotuneCache._heuristic_results[cache_key]

        try:
            # Load heuristic module from cache or import fresh
            if heuristic_file in AOTAutotuneCache._heuristic_modules:
                module = AOTAutotuneCache._heuristic_modules[heuristic_file]
            else:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "heuristic", heuristic_file
                )
                if spec is None or spec.loader is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                AOTAutotuneCache._heuristic_modules[heuristic_file] = module
                log.debug(f"Loaded heuristic module: {heuristic_file}")

            # Call the heuristic function
            config: Config | None = None
            if hasattr(module, f"select_config_{kernel_name}"):
                select_fn = getattr(module, f"select_config_{kernel_name}")
                config_dict = select_fn(shape_features)
                config = Config(**config_dict)
            elif hasattr(module, "select_config"):
                select_fn = module.select_config
                config_dict = select_fn(kernel_name, shape_features)
                config = Config(**config_dict)

            # Cache the result
            if config is not None:
                AOTAutotuneCache._heuristic_results[cache_key] = config
                log.debug(
                    f"Cached heuristic result for {kernel_name} shape={shape_hash}"
                )

            return config
        except Exception as e:
            log.warning(f"Failed to load heuristic from {heuristic_file}: {e}")

        return None

    def supports_per_shape_config(self) -> bool:
        """
        Return True if heuristics are available for per-shape config selection.

        When True, the kernel can use get_config_for_args() on each invocation
        to get shape-specific configs, even when static_shapes=False.
        """
        # Only support per-shape config in evaluate mode with available heuristics
        if self.mode != "evaluate":
            return False
        return self._find_heuristic_file() is not None

    def get_config_for_args(self, args: Sequence[object]) -> Config | None:
        """
        Get a config for the given arguments using heuristics.

        This enables per-shape config selection independent of static_shapes setting.

        Args:
            args: The kernel arguments for this invocation

        Returns:
            Config if heuristics provide a config for this shape, None otherwise
        """
        return self._get_heuristic_config(args)

    def _get_cache_key(self) -> LooseAutotuneCacheKey:
        """Return a cache key for compatibility."""
        return self.autotuner.kernel.kernel._create_bound_kernel_cache_key(
            self.kernel,
            tuple(self.args),
            self.kernel.kernel.specialization_key(self.args),
        )

    def _list_cache_entries(self) -> Sequence[tuple[str, LooseAutotuneCacheKey]]:
        """List cache entries for compatibility."""
        return []

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Perform autotuning based on current mode."""
        if self.mode == "measure":
            # In measure mode, benchmark all configs and return the best
            results = self.measure_all_configs()
            if results:
                best_config, best_timing = min(results, key=operator.itemgetter(1))
                log.info(
                    f"AOT measure: Best config for {self.kernel.kernel.name} "
                    f"shape={self.shape_key.stable_hash()} timing={best_timing:.4f}ms"
                )
                return best_config
            # Fall through to regular autotuning if no configs available
            log.warning("No configs to measure, falling through to autotuner")

        # Use parent implementation for other modes
        return super().autotune(skip_cache=skip_cache)


def _get_dtype_category(dtype: torch.dtype) -> int:
    """
    Get numeric category for dtype.

    Categories ordered by "complexity":
    - 0: boolean
    - 1: integer types
    - 2: floating point types
    - 3: complex types
    - 4: other/unknown
    """
    if dtype == torch.bool:
        return 0
    if dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        return 1
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return 2
    if dtype in (torch.complex64, torch.complex128):
        return 3
    return 4


def _serialize_value(val: Any) -> Any:
    """Serialize a single value to JSON-compatible format."""
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, type):
        # Handle class/type objects
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
    if hasattr(val, "__dict__"):
        # Handle arbitrary objects by storing their class and dict
        return {
            "__object__": f"{val.__class__.__module__}.{val.__class__.__qualname__}",
            "__data__": _serialize_value(vars(val)),
        }
    # Last resort: convert to string
    return {"__str__": str(val)}


def _deserialize_value(val: Any) -> Any:
    """Deserialize a JSON value back to Python object."""
    if isinstance(val, dict):
        if "__tuple__" in val:
            return tuple(_deserialize_value(v) for v in val["__tuple__"])
        if "__frozenset__" in val:
            return frozenset(_deserialize_value(v) for v in val["__frozenset__"])
        if "__set__" in val:
            return {_deserialize_value(v) for v in val["__set__"]}
        if "__dtype__" in val:
            dtype_str = val["__dtype__"]
            # Parse dtype string like "torch.float32"
            dtype_name = dtype_str.replace("torch.", "")
            return getattr(torch, dtype_name)
        if "__device__" in val:
            return torch.device(val["__device__"])
        if "__type__" in val:
            # Reconstruct type from fully qualified name
            type_name = val["__type__"]
            return _import_type(type_name)
        if "__object__" in val:
            # Reconstruct object from class name and data
            cls = _import_type(val["__object__"])
            data = _deserialize_value(val["__data__"])
            obj = object.__new__(cls)
            obj.__dict__.update(data)
            return obj
        if "__str__" in val:
            # String representation - return as-is (can't reconstruct)
            return val["__str__"]
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_deserialize_value(v) for v in val]
    return val


def _import_type(type_name: str) -> type:
    """Import a type from its fully qualified name."""
    import importlib

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


def _serialize_config(config: Config) -> dict[str, Any]:
    """Serialize a Config to JSON-compatible dict."""
    return {k: _serialize_value(v) for k, v in dict(config).items()}


def _deserialize_config(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a config dict, converting special types back."""
    return {k: _deserialize_value(v) for k, v in data.items()}
