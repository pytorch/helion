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

import csv
import dataclasses
from dataclasses import dataclass
import functools
import hashlib
import json
import logging
import operator
import os
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import torch

from ..experimental.aot_kernel import _flatten_key_value
from ..experimental.aot_kernel import extract_key_features
from ..experimental.aot_kernel import extract_shape_features
from ..runtime.config import Config
from .base_cache import AutotuneCacheBase
from .base_cache import BoundKernelInMemoryCacheKey
from .base_cache import LooseAutotuneCacheKey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)

# Compute capability lists for fallback (newest to oldest)
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


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """
    Hardware information for cache keys and heuristic file discovery.

    Attributes:
        device_kind: Device type ('cuda', 'rocm', 'xpu', 'cpu')
        hardware_name: Device name (e.g., 'NVIDIA H100', 'gfx90a', 'cpu')
        runtime_version: Runtime version (e.g., '12.4', 'gfx90a', 'x86_64')
        compute_capability: Compute capability for heuristics (e.g., 'sm90', 'gfx90a')
    """

    device_kind: str
    hardware_name: str
    runtime_version: str
    compute_capability: str

    @property
    def hardware_id(self) -> str:
        """Get a unique identifier string for this hardware."""
        safe_name = self.hardware_name.replace(" ", "_")
        return f"{self.device_kind}_{safe_name}_{self.runtime_version}"

    def get_compatible_compute_ids(self) -> list[str]:
        """
        Get a list of compatible compute IDs for fallback, ordered from current to oldest.

        For CUDA/ROCm, returns the current compute capability followed by all older
        compatible architectures. This allows using heuristics tuned on older hardware
        when newer hardware-specific heuristics aren't available.
        """
        if self.device_kind == "cuda":
            arch_list = _CUDA_COMPUTE_CAPS
        elif self.device_kind == "rocm":
            arch_list = _ROCM_ARCHS
        else:
            return [self.compute_capability]

        try:
            current_idx = arch_list.index(self.compute_capability)
            return arch_list[current_idx:]
        except ValueError:
            return [self.compute_capability, *arch_list]


@functools.cache
def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
    """
    Get hardware information for the current or specified device.

    This is the single source of truth for hardware detection, used by both
    local cache and AOT cache.

    Args:
        device: Optional device to get info for. If None, uses first available GPU or CPU.

    Returns:
        HardwareInfo with device details for caching and heuristic lookup.
    """
    # CPU fallback
    if device is not None and device.type == "cpu":
        return HardwareInfo(
            device_kind="cpu",
            hardware_name="cpu",
            runtime_version=platform.machine().lower(),
            compute_capability=platform.machine().lower(),
        )

    # XPU (Intel) path
    if (
        device is not None
        and device.type == "xpu"
        and getattr(torch, "xpu", None) is not None
        and torch.xpu.is_available()
    ):
        props = torch.xpu.get_device_properties(device)
        return HardwareInfo(
            device_kind="xpu",
            hardware_name=props.name,
            runtime_version=props.driver_version,
            compute_capability=props.name,  # XPU doesn't have compute capability
        )

    # CUDA/ROCm path
    if torch.cuda.is_available():
        dev = (
            device
            if device is not None and device.type == "cuda"
            else torch.device("cuda:0")
        )
        props = torch.cuda.get_device_properties(dev)

        if torch.version.cuda is not None:
            return HardwareInfo(
                device_kind="cuda",
                hardware_name=props.name,
                runtime_version=str(torch.version.cuda),
                compute_capability=f"sm{props.major}{props.minor}",
            )
        if torch.version.hip is not None:
            return HardwareInfo(
                device_kind="rocm",
                hardware_name=props.gcnArchName,
                runtime_version=torch.version.hip,
                compute_capability=props.gcnArchName,
            )

    # CPU fallback
    return HardwareInfo(
        device_kind="cpu",
        hardware_name="cpu",
        runtime_version=platform.machine().lower(),
        compute_capability=platform.machine().lower(),
    )


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


# Cache for heuristic file lookups
_heuristic_file_cache: dict[str, Path | None] = {}


def find_heuristic_file(
    kernel_source_file: str | Path,
    kernel_name: str | None = None,
    data_dir: Path | None = None,
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

    Args:
        kernel_source_file: Path to the kernel's source file
        kernel_name: Optional kernel name for fallback lookup
        data_dir: Optional AOT data directory for fallback lookup

    Returns:
        Path to heuristic file if found, None otherwise
    """
    cache_key = str(kernel_source_file)
    if cache_key in _heuristic_file_cache:
        return _heuristic_file_cache[cache_key]

    source_path = Path(kernel_source_file)
    base_name = source_path.stem
    hw = get_hardware_info()
    compatible_computes = hw.get_compatible_compute_ids()

    candidates: list[Path] = []

    # 1. Check HELION_HEURISTIC_DIR override
    if (heuristic_dir := os.environ.get(HEURISTIC_DIR_ENV)) is not None:
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
        if candidate.exists():
            log.debug(f"Found heuristic file: {candidate}")
            result = candidate
            break

    _heuristic_file_cache[cache_key] = result
    return result


def clear_heuristic_cache() -> None:
    """Clear the heuristic file cache (useful for testing)."""
    _heuristic_file_cache.clear()


def load_kernel_source_files(data_dir: Path, hardware_id: str) -> dict[str, str]:
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
                if cfg.get("kernel_source_file"):
                    result[kernel_name] = cfg["kernel_source_file"]
                    break
        return result
    except Exception as e:
        log.warning(f"Failed to load kernel source files: {e}")
        return {}


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

    # Tracks which AOT modes have been announced to avoid repeated stderr messages.
    # Class-level so announcements happen only once per mode across all instances.
    _mode_announced: ClassVar[set[str]] = set()

    # Class-level caches for heuristic lookup (shared across instances)
    # Maps heuristic file path -> loaded module
    _heuristic_modules: ClassVar[dict[Path, Any]] = {}
    # Maps (kernel_source_file, kernel_name, shape_features_hash) -> Config
    # Using source file ensures kernels with same name in different modules don't collide
    _heuristic_results: ClassVar[dict[tuple[str, str, str], Config]] = {}
    # Tracks which kernels have shown the "no heuristic" warning (to avoid spam)
    _no_heuristic_warned: ClassVar[set[str]] = set()

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all class-level caches (heuristic modules and results)."""
        cls._heuristic_modules.clear()
        cls._heuristic_results.clear()
        cls._no_heuristic_warned.clear()
        clear_heuristic_cache()  # Clear module-level cache
        cls._mode_announced.clear()
        log.debug("Cleared AOTAutotuneCache caches")

    def __init__(self, autotuner: BaseSearch) -> None:
        super().__init__(autotuner)
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

        # Announce mode once per mode type (quiet in evaluate mode unless verbose)
        should_announce = (
            self.mode != "disabled"
            and self.mode not in AOTAutotuneCache._mode_announced
            and (self.mode != "evaluate" or self._verbose)
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

    @property
    def _configs_file(self) -> Path:
        """Path to the tuned configs JSON file."""
        return self.data_dir / f"tuned_configs_{self.hardware_id}.json"

    @property
    def _measurements_file(self) -> Path:
        """Path to the measurements CSV file."""
        return self.data_dir / f"measurements_{self.hardware_id}.csv"

    def _load_tuned_configs(self) -> dict[str, list[TunedConfig]]:
        """Load tuned configs from disk."""
        if not self._configs_file.exists():
            return {}
        try:
            data = json.loads(self._configs_file.read_text())
            result: dict[str, list[TunedConfig]] = {}
            for kernel_name, configs in data.items():
                result[kernel_name] = [
                    TunedConfig(
                        config=Config(**cfg["config"]),
                        shape_key=ShapeKey.from_dict(cfg["shape_key"]),
                        timing_ms=cfg.get("timing_ms"),
                        kernel_source_file=cfg.get("kernel_source_file"),
                        shape_features=cfg.get("shape_features"),
                        tensor_hashes=cfg.get("tensor_hashes"),
                    )
                    for cfg in configs
                ]
            return result
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

        # Check if this exact config already exists for this shape
        for existing in self._tuned_configs[kernel_name]:
            if (
                existing.shape_key.stable_hash() == shape_hash
                and dict(existing.config) == config_dict
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
                config=config,
                shape_key=shape_key,
                timing_ms=timing_ms,
                kernel_source_file=kernel_source_file,
                shape_features=shape_features,
                tensor_hashes=tensor_hashes,
            )
        )

    def _get_all_configs_for_kernel(self, kernel_name: str) -> list[Config]:
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

    def _save_measurement(
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
        file_exists = self._measurements_file.exists()
        with open(self._measurements_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

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
                if tc.shape_key.stable_hash() == self.shape_key.stable_hash():
                    log.info(f"AOT collect: Using existing config for {kernel_name}")
                    return tc.config
            return None  # Need to tune

        if self.mode == "measure":
            # In measure mode, we don't use cache - we measure all configs
            return None

        # For disabled/evaluate modes: try heuristic, fall back to default config
        # (never trigger autotuning for aot_kernel)
        config = self._get_heuristic_config()
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
                    f"Use `python -m helion.experimental.aot_runner` to generate tuned configs.",
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
        import tempfile
        import traceback

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
            if tmpdir_created and self.autotuner._precompile_tmpdir is not None:
                self.autotuner._precompile_tmpdir.cleanup()
                self.autotuner._precompile_tmpdir = None

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
        )

    def _get_heuristic_config(
        self, args: Sequence[object] | None = None
    ) -> Config | None:
        """
        Use the heuristic to select a config.

        Looks for autotune_<kernel>(*args) function in the heuristic file.

        Args:
            args: Optional arguments to use. If None, uses self.args.

        For CUDA/ROCm, if heuristics for the current compute capability aren't found,
        we try older compatible architectures (e.g., sm80 heuristics on sm90 hardware).
        """
        heuristic_file = self._find_heuristic_file()
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

    def _get_cache_key(self) -> BoundKernelInMemoryCacheKey:
        """Return a cache key for compatibility."""
        return self.autotuner.kernel.kernel._create_bound_kernel_cache_key(
            self.kernel,
            tuple(self.args),
            self.kernel.kernel.specialization_key(self.args),
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
        collect_inputs = list(self._collect_fn())

        print(
            f"[AOT collect_fn] Autotuning {len(collect_inputs)} shapes for {kernel_name}",
            file=sys.stderr,
        )

        for i, input_args in enumerate(collect_inputs):
            print(
                f"[AOT collect_fn] Tuning shape {i + 1}/{len(collect_inputs)}",
                file=sys.stderr,
            )
            self.kernel.kernel(*input_args)

        print(f"[AOT collect_fn] Completed for {kernel_name}", file=sys.stderr)

    def _run_measure_fn_workflow(self) -> None:
        """Run measurement on all inputs from measure_fn."""
        assert self._measure_fn is not None
        kernel_name = self.kernel.kernel.name
        measure_inputs = list(self._measure_fn())
        all_configs = self._get_all_configs_for_kernel(kernel_name)

        if not all_configs:
            print(
                f"[AOT measure_fn] Warning: No configs found for {kernel_name}",
                file=sys.stderr,
            )
            return

        print(
            f"[AOT measure_fn] Measuring {len(all_configs)} configs "
            f"across {len(measure_inputs)} shapes for {kernel_name}",
            file=sys.stderr,
        )

        for input_args in measure_inputs:
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
                    fn = bound.compile_config(config)

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
            f"[AOT measure_fn] Completed! Results saved to {self._measurements_file}",
            file=sys.stderr,
        )

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Perform autotuning based on current mode."""
        # Check if input_fn workflow should run (only once per kernel)
        workflow_done = getattr(self.kernel.kernel, "_aot_workflow_done", False)

        if not workflow_done:
            # Mark done FIRST to prevent recursive calls when we invoke the kernel
            self.kernel.kernel._aot_workflow_done = True  # type: ignore[attr-defined]

            if self.mode == "collect" and self._collect_fn is not None:
                self._run_collect_fn_workflow()
                if self._measure_fn is not None:
                    self._run_measure_fn_workflow()
                return super().autotune(skip_cache=skip_cache)

            if self.mode == "measure" and self._measure_fn is not None:
                self._run_measure_fn_workflow()

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
        # Note: super().autotune() internally calls self.put() before returning
        return super().autotune(skip_cache=skip_cache)


def _serialize_value(val: object) -> object:
    """Serialize a single value to JSON-compatible format.

    Supports: None, bool, int, float, str, type, tuple, frozenset, set,
    torch.dtype, torch.device, list, dict.
    """
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
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
    raise TypeError(f"Cannot serialize type: {type(val).__name__}")


def _deserialize_value(val: object) -> object:
    """Deserialize a JSON value back to Python object.

    Handles tagged dicts: __tuple__, __frozenset__, __set__, __dtype__, __device__, __type__.
    """
    if isinstance(val, dict):
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
