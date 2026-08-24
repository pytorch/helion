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

import ast
from collections.abc import Mapping
import contextlib
import csv
from dataclasses import dataclass
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import marshal
import operator
import os
from pathlib import Path
import sys
import tempfile
import threading
import traceback
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

from filelock import FileLock
import torch

from .._hardware import HardwareInfo
from .._hardware import get_hardware_info
from ..exc import AOTHardwareManifestError
from ..runtime.config import Config
from .aot_compile import _standalone_call_key
from .aot_compile import generate_standalone_file
from .aot_compile import kernel_source_identity
from .aot_compile import standalone_output_path
from .aot_kernel import _flatten_key_value
from .aot_kernel import extract_key_features
from .aot_kernel import extract_shape_features
from .base_cache import AutotuneCacheBase
from .base_cache import BoundKernelInMemoryCacheKey
from .base_cache import LooseAutotuneCacheKey
from .benchmark_provider import _MultiShapeAutotuneArgs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType
    from typing import Callable

    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)

# Environment variable to control AOT mode
AOT_MODE_ENV = "HELION_AOT_MODE"
AOT_DATA_DIR_ENV = "HELION_AOT_DATA_DIR"
AOT_HARDWARE_MANIFEST = "hardware.json"
# Internal runner-to-child contract: validate the run manifest against the
# hardware resolved from the benchmark's actual kernel arguments.
AOT_RUNNER_HARDWARE_VALIDATION_ENV = "HELION_AOT_RUNNER_VALIDATE_HARDWARE"
_HARDWARE_MANIFEST_FIELDS = (
    "hardware_id",
    "device_kind",
    "hardware_name",
    "runtime_version",
    "compute_capability",
)
# Environment variable to override heuristic search path (for comparing heuristics)
HEURISTIC_DIR_ENV = "HELION_HEURISTIC_DIR"
# Optional module-level artifact contract. When present, discovery and loading
# require an exact hardware-name match instead of relying only on compatible
# compute-capability filenames.
SUPPORTED_HARDWARE_NAMES = "SUPPORTED_HARDWARE_NAMES"
# Environment variable to enable verbose output in quiet AOT modes.
AOT_VERBOSE_ENV = "HELION_AOT_VERBOSE"

AOTMode = Literal["collect", "measure", "evaluate", "compile", "disabled"]


class HeuristicArtifactMetadataError(RuntimeError):
    """Raised when a heuristic artifact declares invalid support metadata."""


class _AOTHeuristicAbstention:
    """Internal marker for an intentional per-key heuristic fallback."""


_AOT_HEURISTIC_ABSTENTION = _AOTHeuristicAbstention()


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


def _missing_manifest_message(data_dir: Path) -> str:
    return (
        f"AOT directory {data_dir} is missing required {AOT_HARDWARE_MANIFEST}. "
        "Helion cannot safely identify artifacts from a pre-manifest AOT run. "
        "Recollect into a fresh AOT data directory: set HELION_AOT_DATA_DIR to "
        "an empty path or use a new aot_runner --run-id. A directory containing "
        "legacy AOT artifacts such as heuristic_*.py, tuned_configs_*.json, or "
        "measurements_*.csv is intentionally not adopted in place."
    )


def _hardware_manifest_data(hardware: HardwareInfo) -> dict[str, str]:
    return {field: getattr(hardware, field) for field in _HARDWARE_MANIFEST_FIELDS}


def load_hardware_manifest(data_dir: Path) -> HardwareInfo:
    """Load and validate the hardware identity recorded by an AOT benchmark."""
    manifest_file = data_dir / AOT_HARDWARE_MANIFEST
    if not manifest_file.is_file():
        raise AOTHardwareManifestError(_missing_manifest_message(data_dir))

    try:
        data = json.loads(manifest_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AOTHardwareManifestError(
            f"Invalid AOT hardware manifest: {manifest_file}"
        ) from exc

    required_fields = set(_HARDWARE_MANIFEST_FIELDS)
    if (
        not isinstance(data, dict)
        or not required_fields.issubset(data)
        or any(
            not isinstance(data[field], str) or not data[field]
            for field in _HARDWARE_MANIFEST_FIELDS
        )
    ):
        raise AOTHardwareManifestError(
            f"Invalid AOT hardware manifest: {manifest_file}"
        )

    hardware = HardwareInfo(
        device_kind=data["device_kind"],
        hardware_name=data["hardware_name"],
        runtime_version=data["runtime_version"],
        compute_capability=data["compute_capability"],
    )
    # Keep the derived ID in the manifest as a corruption/inconsistency check.
    if data["hardware_id"] != hardware.hardware_id:
        raise AOTHardwareManifestError(
            f"Inconsistent AOT hardware identity in {manifest_file}: "
            f"{data['hardware_id']!r} != {hardware.hardware_id!r}"
        )
    return hardware


def validate_hardware_manifest(data_dir: Path, hardware: HardwareInfo) -> None:
    """Reject an AOT data directory recorded for different hardware."""
    recorded = load_hardware_manifest(data_dir)
    if recorded != hardware:
        raise AOTHardwareManifestError(
            f"AOT hardware identity mismatch in {data_dir / AOT_HARDWARE_MANIFEST}: "
            f"recorded {recorded!r}, observed {hardware!r}; use a run directory "
            "collected on the current hardware"
        )


def validate_runner_hardware(data_dir: Path, hardware: HardwareInfo) -> None:
    """Validate the runner's manifest using hardware resolved inside its child."""
    if os.environ.get(AOT_RUNNER_HARDWARE_VALIDATION_ENV) == "1":
        validate_hardware_manifest(data_dir, hardware)


_LEGACY_AOT_ARTIFACT_PREFIXES = (
    "_helion_aot_",
    "evaluation_",
    "heuristic_",
    "measurements_",
    "tuned_configs_",
)


def _is_legacy_aot_artifact_name(name: str) -> bool:
    return name.startswith(_LEGACY_AOT_ARTIFACT_PREFIXES) or "_standalone." in name


def _legacy_aot_artifacts(data_dir: Path) -> list[Path]:
    """Find artifacts that must not be assigned a new hardware identity."""
    artifacts = [
        entry
        for entry in data_dir.iterdir()
        if _is_legacy_aot_artifact_name(entry.name)
    ]
    pycache = data_dir / "__pycache__"
    if pycache.is_dir():
        artifacts.extend(
            entry
            for entry in pycache.iterdir()
            if _is_legacy_aot_artifact_name(entry.name)
        )
    return sorted(artifacts)


def write_hardware_manifest(data_dir: Path, hardware: HardwareInfo) -> None:
    """Record one unambiguous hardware identity for an AOT benchmark run."""
    manifest_file = data_dir / AOT_HARDWARE_MANIFEST
    lock_file = data_dir / f".{AOT_HARDWARE_MANIFEST}.lock"
    temporary_file: Path | None = None
    try:
        # FileLock uses an OS-held lock on supported platforms, which is
        # released if the writer exits or crashes. A persistent lock path is
        # therefore harmless, unlike a create-once sentinel that a dead writer
        # could strand indefinitely.
        with FileLock(str(lock_file)):
            if manifest_file.is_file():
                existing = load_hardware_manifest(data_dir)
                if existing != hardware:
                    raise AOTHardwareManifestError(
                        f"Ambiguous AOT hardware identity in {manifest_file}: "
                        f"already recorded {existing.hardware_id!r}, "
                        f"but benchmark observed {hardware.hardware_id!r}"
                    )
                return

            if artifacts := _legacy_aot_artifacts(data_dir):
                artifact_names = ", ".join(
                    str(path.relative_to(data_dir)) for path in artifacts[:5]
                )
                raise AOTHardwareManifestError(
                    f"{_missing_manifest_message(data_dir)} Conflicting legacy "
                    f"artifacts: {artifact_names}"
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=data_dir,
                prefix=f".{AOT_HARDWARE_MANIFEST}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_file = Path(output.name)
                output.write(
                    json.dumps(_hardware_manifest_data(hardware), indent=2) + "\n"
                )
                output.flush()
                os.fsync(output.fileno())

            # The fully written, fsynced file appears at the public path in one
            # atomic rename. Readers can observe either no manifest or the
            # complete JSON document, never the writer's partial output.
            os.replace(temporary_file, manifest_file)
            temporary_file = None
    except AOTHardwareManifestError:
        raise
    except OSError as exc:
        raise AOTHardwareManifestError(
            "Failed to atomically create AOT hardware manifest "
            f"{manifest_file}. Set {AOT_DATA_DIR_ENV} to a fresh, writable "
            "directory, then recollect the AOT data."
        ) from exc
    finally:
        if temporary_file is not None:
            with contextlib.suppress(OSError):
                temporary_file.unlink(missing_ok=True)


def _resolved_path(path: str | Path) -> Path:
    """Return the canonical identity used by all AOT file caches."""
    return Path(path).expanduser().resolve()


def _normalize_supported_hardware_names(
    value: object,
    artifact: Path,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list, set, frozenset)) or not value:
        raise HeuristicArtifactMetadataError(
            f"{artifact}: {SUPPORTED_HARDWARE_NAMES} must be a nonempty "
            "sequence or set of hardware names"
        )
    if any(type(name) is not str or not name for name in value):
        raise HeuristicArtifactMetadataError(
            f"{artifact}: {SUPPORTED_HARDWARE_NAMES} entries must be nonempty strings"
        )
    return tuple(sorted(set(value)))


@functools.cache
def _declared_supported_hardware_names(artifact: Path) -> tuple[str, ...] | None:
    """Read the optional exact-hardware contract without executing the artifact."""
    try:
        tree = ast.parse(artifact.read_text(), filename=str(artifact))
    except (OSError, SyntaxError) as exc:
        raise HeuristicArtifactMetadataError(
            f"Unable to inspect heuristic artifact metadata in {artifact}"
        ) from exc

    declaration: ast.Assign | ast.AnnAssign | None = None
    for node in tree.body:
        candidate_declaration: ast.Assign | ast.AnnAssign | None = None
        if isinstance(node, ast.Assign):
            matching_targets = [
                item
                for item in node.targets
                if isinstance(item, ast.Name) and item.id == SUPPORTED_HARDWARE_NAMES
            ]
            if matching_targets:
                candidate_declaration = node
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == SUPPORTED_HARDWARE_NAMES
            ):
                candidate_declaration = node
        elif isinstance(node, ast.AugAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == SUPPORTED_HARDWARE_NAMES
            ):
                raise HeuristicArtifactMetadataError(
                    f"{artifact}: {SUPPORTED_HARDWARE_NAMES} must be one "
                    "module-level literal assignment"
                )
        if candidate_declaration is None:
            continue
        if declaration is not None:
            raise HeuristicArtifactMetadataError(
                f"{artifact}: {SUPPORTED_HARDWARE_NAMES} must be one "
                "module-level literal assignment"
            )
        declaration = candidate_declaration

    if declaration is None:
        return None
    value_node = declaration.value
    if value_node is None:
        raise HeuristicArtifactMetadataError(
            f"{artifact}: {SUPPORTED_HARDWARE_NAMES} must have a literal value"
        )
    try:
        value = ast.literal_eval(value_node)
    except (ValueError, TypeError) as exc:
        raise HeuristicArtifactMetadataError(
            f"{artifact}: {SUPPORTED_HARDWARE_NAMES} must have a literal value"
        ) from exc
    return _normalize_supported_hardware_names(value, artifact)


def get_heuristic_hardware(device: torch.device | None) -> HardwareInfo:
    """Return the hardware identity used for heuristic discovery and caching."""
    if device is not None:
        from .._argument_device import _canonicalize_argument_device

        # Keep this boundary canonical even though get_hardware_info also
        # canonicalizes: callers and tests may replace that downstream seam.
        device = _canonicalize_argument_device(device)
    return get_hardware_info(_explicit_hardware_device(device))


def heuristic_artifact_identity(
    artifact: Path,
    hardware: HardwareInfo,
) -> str | None:
    """Return the artifact identity, or ``None`` for unsupported hardware."""
    names = _declared_supported_hardware_names(artifact)
    if names is None:
        return str(artifact)
    if hardware.hardware_name not in names:
        return None
    return (
        f"{artifact}::{SUPPORTED_HARDWARE_NAMES}={','.join(names)}"
        f"::hardware_name={hardware.hardware_name}"
    )


def heuristic_module_supports_hardware(
    module: ModuleType,
    artifact: Path,
    hardware: HardwareInfo,
) -> bool:
    """Validate a loaded module against its declared hardware support."""
    declared_names = _declared_supported_hardware_names(artifact)
    runtime_value = getattr(module, SUPPORTED_HARDWARE_NAMES, None)
    if declared_names is None:
        if runtime_value is not None:
            raise HeuristicArtifactMetadataError(
                f"{artifact}: {SUPPORTED_HARDWARE_NAMES} must be a "
                "module-level literal assignment"
            )
        return True
    runtime_names = _normalize_supported_hardware_names(runtime_value, artifact)
    if runtime_names != declared_names:
        raise HeuristicArtifactMetadataError(
            f"{artifact}: loaded {SUPPORTED_HARDWARE_NAMES} does not match "
            "its static declaration"
        )
    return hardware.hardware_name in declared_names


# Cache for heuristic file lookups. Every value that affects the candidate list
# belongs in the key so one process can safely switch devices or override dirs.
_heuristic_file_cache: dict[
    tuple[str | None, str | None, str | None, str | None, HardwareInfo], Path | None
] = {}


class _UseEnvironmentHeuristicDir:
    """Sentinel selecting the process override at discovery time."""


_USE_ENVIRONMENT_HEURISTIC_DIR = _UseEnvironmentHeuristicDir()


def _explicit_hardware_device(device: torch.device | None) -> torch.device | None:
    """Return devices that hardware discovery can resolve explicitly.

    Pallas uses CPU tensors as a bridge for TPU execution.  Keeping that CPU
    device explicit lets ``get_hardware_info`` prefer TPU on mixed-device hosts
    while retaining its ordinary accelerator fallback. Unknown device kinds leave
    accelerator discovery automatic.
    """
    if device is not None and device.type in ("cpu", "cuda", "xpu"):
        return device
    return None


def find_heuristic_file(
    kernel_source_file: str | Path | None,
    kernel_name: str | None = None,
    data_dir: Path | None = None,
    device: torch.device | None = None,
    resolved_heuristic_dir: Path | _UseEnvironmentHeuristicDir | None = (
        _USE_ENVIRONMENT_HEURISTIC_DIR
    ),
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
        kernel_source_file: Canonical kernel source path, or ``None`` when the
            source is not safely file-backed. Source-adjacent lookup is skipped
            for ``None`` while explicit and AOT data directories remain usable.
        kernel_name: Optional kernel name for fallback lookup
        data_dir: Optional AOT data directory for fallback lookup
        device: Optional device whose hardware-specific heuristic should be used
        resolved_heuristic_dir: Pre-resolved override directory or explicit
            ``None``. When omitted, ``HELION_HEURISTIC_DIR`` is resolved from
            the environment.

    Returns:
        Path to heuristic file if found, None otherwise
    """
    source_path = (
        None if kernel_source_file is None else _resolved_path(kernel_source_file)
    )
    hw = get_heuristic_hardware(device)
    compatible_computes = tuple(hw.get_compatible_compute_ids())
    if isinstance(resolved_heuristic_dir, _UseEnvironmentHeuristicDir):
        heuristic_dir_value = os.environ.get(HEURISTIC_DIR_ENV)
        heuristic_dir = (
            None if heuristic_dir_value is None else _resolved_path(heuristic_dir_value)
        )
    else:
        heuristic_dir = resolved_heuristic_dir
    resolved_data_dir = None if data_dir is None else _resolved_path(data_dir)
    cache_key = (
        None if source_path is None else str(source_path),
        kernel_name,
        None if resolved_data_dir is None else str(resolved_data_dir),
        None if heuristic_dir is None else str(heuristic_dir),
        hw,
    )
    if cache_key in _heuristic_file_cache:
        return _heuristic_file_cache[cache_key]

    candidates: list[tuple[Path, bool]] = []

    # 1. Check HELION_HEURISTIC_DIR override
    if heuristic_dir is not None:
        if source_path is not None:
            for compat_compute in compatible_computes:
                candidates.append(
                    (
                        heuristic_dir
                        / f"_helion_aot_{source_path.stem}_{hw.device_kind}_{compat_compute}.py",
                        False,
                    )
                )
        if kernel_name:
            candidates.append((heuristic_dir / f"heuristic_{kernel_name}.py", False))

    # 2. Check next to kernel source file with compute capability fallback
    if source_path is not None:
        for compat_compute in compatible_computes:
            heuristic_name = (
                f"_helion_aot_{source_path.stem}_{hw.device_kind}_{compat_compute}.py"
            )
            candidates.append((source_path.parent / heuristic_name, False))

    # 3. Check AOT data directory (fallback)
    if resolved_data_dir is not None and kernel_name is not None:
        candidates.append((resolved_data_dir / f"heuristic_{kernel_name}.py", True))

    # Find first existing file
    result: Path | None = None
    for candidate, requires_manifest in candidates:
        if candidate.is_file():
            candidate = candidate.resolve()
            try:
                supported_names = _declared_supported_hardware_names(candidate)
            except HeuristicArtifactMetadataError:
                log.warning(
                    "Skipping heuristic with invalid artifact metadata: %s",
                    candidate,
                    exc_info=True,
                )
                continue
            if supported_names is not None and hw.hardware_name not in supported_names:
                log.debug(
                    "Skipping heuristic %s: %s is not in %s",
                    candidate,
                    hw.hardware_name,
                    supported_names,
                )
                continue
            if requires_manifest:
                assert resolved_data_dir is not None
                validate_hardware_manifest(resolved_data_dir, hw)
            log.debug(f"Found heuristic file: {candidate}")
            result = candidate
            break

    _heuristic_file_cache[cache_key] = result
    return result


def clear_heuristic_cache() -> None:
    """Clear the heuristic file cache (useful for testing)."""
    _heuristic_file_cache.clear()
    _declared_supported_hardware_names.cache_clear()


def load_kernel_source_files(data_dir: Path, hardware_id: str) -> dict[str, str]:
    """
    Load kernel source file mappings from tuned configs JSON.

    This is a best-effort legacy helper for offline heuristic generation and
    intentionally keeps the first nonempty source recorded for each kernel.
    Standalone compile freshness uses a stricter parser because conflicting
    eligible source paths make the expected output location ambiguous.

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
    standalone: bool = True
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
    - disabled: Ignore AOT artifacts and return the compiler default config

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
    _heuristic_modules: ClassVar[dict[Path, ModuleType]] = {}
    _supported_heuristics: ClassVar[
        dict[
            tuple[Path, HardwareInfo],
            tuple[ModuleType, str] | None,
        ]
    ] = {}
    # Maps (kernel source, kernel name, hardware, heuristic file, shape hash) to
    # the selected Config or an explicit per-key abstention. Missing or invalid
    # artifacts are not cached here.
    # Source and hardware prevent same-name kernels and cross-device results from aliasing;
    # the heuristic path keeps directory overrides isolated.
    _heuristic_results: ClassVar[
        dict[
            tuple[str, str, HardwareInfo, str, str],
            Config | _AOTHeuristicAbstention,
        ]
    ] = {}
    # Tracks which exact kernels have shown the "no heuristic" warning.
    _no_heuristic_warned: ClassVar[set[tuple[str, str, HardwareInfo]]] = set()
    # Dynamic compile suppression is scoped to the exact source, hardware, and
    # heuristic. Same-name kernels and runtime override changes must not alias.
    _compiled_kernels: ClassVar[set[tuple[str, str, HardwareInfo, str]]] = set()
    # Static-shape compile mode accumulates one source variant per normalized
    # call signature and rewrites the dispatcher whenever a shape is observed.
    _compiled_kernel_variants: ClassVar[
        dict[
            tuple[str, str, str, str, str],
            dict[tuple[object, ...], tuple[str, str]],
        ]
    ] = {}
    _compiled_kernel_variants_lock: ClassVar[Any] = threading.RLock()

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all class-level caches (heuristic modules and results)."""
        cls._heuristic_modules.clear()
        cls._supported_heuristics.clear()
        cls._heuristic_results.clear()
        cls._no_heuristic_warned.clear()
        cls._compiled_kernels.clear()
        with cls._compiled_kernel_variants_lock:
            cls._compiled_kernel_variants.clear()
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
        hardware_device = _explicit_hardware_device(self.kernel.env.device)
        self._hardware_device = hardware_device
        self.hardware = get_hardware_info(hardware_device)
        self.hardware_id = self.hardware.hardware_id
        code_object = self.kernel.kernel.__code__
        self._kernel_source_file, self._kernel_source = kernel_source_identity(
            code_object.co_filename,
            code_object,
        )
        self.data_dir = get_aot_data_dir()
        if self.mode == "disabled":
            self._tuned_configs: dict[str, list[TunedConfig]] = {}
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.mode == "collect":
                write_hardware_manifest(self.data_dir, self.hardware)
            elif self.mode == "measure":
                # Measure consumes tuned configs produced by collect, so its
                # data directory is always a recorded run rather than an
                # optional deployment lookup location.
                validate_hardware_manifest(self.data_dir, self.hardware)
            else:
                # The runner extends the same exact check to evaluate and
                # compile inside the child, where the benchmark's actual
                # argument device is known.
                validate_runner_hardware(self.data_dir, self.hardware)
            self._tuned_configs = self._load_tuned_configs()
        self.shape_key = self._create_shape_key()
        self._verbose = is_aot_verbose()

        self._collect_fn = self.kernel.kernel._aot_collect_fn
        self._measure_fn = self.kernel.kernel._aot_measure_fn

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
                        standalone=cfg.get("standalone", True),
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
                    "standalone": cfg.standalone,
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
        standalone: bool = True,
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
                existing.standalone = standalone
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
                standalone=standalone,
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
        user_key = self.kernel.kernel._aot_user_key
        if user_key is not None:
            # Extract features from flattened key output
            key_value = user_key(*args)
            return extract_key_features(key_value)

        # Use single source of truth from aot_kernel module
        return extract_shape_features(args)

    def get(self) -> Config | None:
        """Get a cached config based on current mode."""
        if self.mode == "disabled":
            # This is an explicit AOT bypass: do not load a checked-in heuristic
            # and do not delegate to a live autotuning search.
            return self.autotuner.config_spec.default_config()

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

        if self.mode == "compile":
            # In compile mode: use heuristic + generate standalone Triton code
            self._maybe_run_compile()

        # For evaluate/compile modes: try heuristic, fall back to default config
        # (never trigger autotuning for aot_kernel)
        config = self._get_heuristic_config(self.args)
        if isinstance(config, _AOTHeuristicAbstention):
            return self.autotuner.config_spec.default_config()
        if config is not None:
            return config

        # No heuristic available - use default config with warning (once per kernel)
        kernel_name = self.kernel.kernel.name
        from .. import exc

        warning_key = (self._kernel_source, kernel_name, self.hardware)
        if warning_key not in AOTAutotuneCache._no_heuristic_warned:
            AOTAutotuneCache._no_heuristic_warned.add(warning_key)
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
            kernel_source_file = (
                None
                if self._kernel_source_file is None
                else str(self._kernel_source_file)
            )
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
                standalone=self.kernel.kernel._aot_standalone,
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
        benchmark_provider.setup()

        try:
            for i, config in enumerate(all_configs):
                try:
                    # Benchmark this config
                    result = self.autotuner.benchmark(config)
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
        return find_heuristic_file(
            self._kernel_source_file,
            kernel_name=self.kernel.kernel.name,
            data_dir=self.data_dir,
            device=self._hardware_device,
        )

    def _load_supported_heuristic(
        self, heuristic_file: Path
    ) -> tuple[ModuleType, tuple[str, str, HardwareInfo, str]] | None:
        """Load and validate one heuristic for this cache's exact hardware."""
        validation_key = (heuristic_file, self.hardware)
        if validation_key in AOTAutotuneCache._supported_heuristics:
            cached = AOTAutotuneCache._supported_heuristics[validation_key]
            if cached is None:
                return None
            module, artifact_identity = cached
            return (
                module,
                (
                    self._kernel_source,
                    self.kernel.kernel.name,
                    self.hardware,
                    artifact_identity,
                ),
            )

        if heuristic_file in AOTAutotuneCache._heuristic_modules:
            module = AOTAutotuneCache._heuristic_modules[heuristic_file]
        else:
            spec = importlib.util.spec_from_file_location("heuristic", heuristic_file)
            if spec is None or spec.loader is None:
                AOTAutotuneCache._supported_heuristics[validation_key] = None
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AOTAutotuneCache._heuristic_modules[heuristic_file] = module
            log.debug(f"Loaded heuristic module: {heuristic_file}")

        try:
            supports_hardware = heuristic_module_supports_hardware(
                module,
                heuristic_file,
                self.hardware,
            )
        except HeuristicArtifactMetadataError:
            log.warning(
                "Skipping heuristic with invalid artifact metadata: %s",
                heuristic_file,
                exc_info=True,
            )
            AOTAutotuneCache._supported_heuristics[validation_key] = None
            return None
        if not supports_hardware:
            AOTAutotuneCache._supported_heuristics[validation_key] = None
            return None
        artifact_identity = heuristic_artifact_identity(heuristic_file, self.hardware)
        if artifact_identity is None:
            AOTAutotuneCache._supported_heuristics[validation_key] = None
            return None
        AOTAutotuneCache._supported_heuristics[validation_key] = (
            module,
            artifact_identity,
        )
        return (
            module,
            (
                self._kernel_source,
                self.kernel.kernel.name,
                self.hardware,
                artifact_identity,
            ),
        )

    def _get_heuristic_config(
        self, args: Sequence[object] | None = None
    ) -> Config | _AOTHeuristicAbstention | None:
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
        # Compute cache key based on shape features
        shape_features = self._extract_shape_features(args)
        shape_hash = hashlib.sha256(
            json.dumps(shape_features, sort_keys=True).encode()
        ).hexdigest()[:16]

        try:
            loaded = self._load_supported_heuristic(heuristic_file)
            if loaded is None:
                return None
            module, result_key_prefix = loaded
            cache_key = (*result_key_prefix, shape_hash)
            if cache_key in AOTAutotuneCache._heuristic_results:
                log.debug(
                    f"Using cached heuristic result for {kernel_name} shape={shape_hash}"
                )
                return AOTAutotuneCache._heuristic_results[cache_key]

            # Call autotune_<kernel>(*args) to get the config.
            # If there's a user key, we need to pass flattened key values, not raw args
            autotune_fn = getattr(module, f"autotune_{kernel_name}", None)
            if autotune_fn is None:
                return None
            user_key = self.kernel.kernel._aot_user_key
            if user_key is not None:
                # User key: pass flattened key values to heuristic
                key_value = user_key(*args)
                flat_key = _flatten_key_value(key_value)
                config_dict = autotune_fn(*flat_key)
            else:
                # No user key: pass raw args to heuristic
                config_dict = autotune_fn(*args)
            if config_dict is None:
                config: Config | _AOTHeuristicAbstention = _AOT_HEURISTIC_ABSTENTION
            else:
                if not isinstance(config_dict, Mapping):
                    raise TypeError(
                        f"autotune_{kernel_name} must return a mapping or None, got "
                        f"{type(config_dict).__name__}"
                    )
                config = Config(**config_dict)

            # Cache explicit per-key abstention as well as selected configs.
            # Heuristic artifacts are immutable for their artifact identity, so
            # repeating an intentional fallback cannot produce a new result.
            AOTAutotuneCache._heuristic_results[cache_key] = config
            log.debug(f"Cached heuristic result for {kernel_name} shape={shape_hash}")

            return config
        except Exception:
            log.warning(
                "Failed to load heuristic from %s",
                heuristic_file,
                exc_info=True,
            )

        return None

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
        if not self.kernel.kernel._aot_standalone:
            log.info(
                "Skipping standalone compile for AOT kernel '%s' (standalone=False)",
                kernel_name,
            )
            return
        heuristic_file = self._find_heuristic_file()
        if heuristic_file is None:
            log.warning(
                "No heuristic for '%s', skipping standalone compile", kernel_name
            )
            return

        loaded = self._load_supported_heuristic(heuristic_file)
        if loaded is None:
            return
        module, compiled_kernel_key = loaded

        if self.kernel.settings.static_shapes:
            self._compile_current_static_shape(heuristic_file, kernel_name)
            return

        selected = self._get_heuristic_config(self.args)
        if isinstance(selected, _AOTHeuristicAbstention):
            raise RuntimeError(
                "dynamic standalone compilation cannot encode a per-workload "
                "heuristic abstention; declare the AOT kernel with standalone=False"
            )

        # -- extract selected configs ---------------------------------------
        # nearest_neighbor backend: module-level CONFIGS
        # decision_tree backend: _C = [...] inside autotune_<kernel>
        configs_list: list[dict[str, object]] | None = getattr(module, "CONFIGS", None)
        if configs_list is None:
            configs_list = self._parse_configs_from_autotune(module, kernel_name)
        if configs_list is None:
            log.warning("Cannot extract configs from heuristic for '%s'", kernel_name)
            return

        if compiled_kernel_key in AOTAutotuneCache._compiled_kernels:
            return
        AOTAutotuneCache._compiled_kernels.add(compiled_kernel_key)

        # -- generate Triton code for each config --------------------------
        triton_codes: list[str] = []
        for i, config_dict in enumerate(configs_list):
            config = Config(**config_dict)  # pyrefly: ignore [bad-argument-type]
            try:
                triton_codes.append(self.kernel.to_triton_code(config))
            except Exception:
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

        # -- emit standalone file -------------------------------------------
        out_path = generate_standalone_file(
            kernel_name=kernel_name,
            triton_codes=triton_codes,
            heuristic_code=heuristic_file.read_text(),
            output_dir=self.data_dir,
            kernel_source_file=self._kernel_source_file,
        )
        print(f"[AOT] Standalone: {out_path}", file=sys.stderr)

    def _compile_current_static_shape(
        self,
        heuristic_file: Path,
        kernel_name: str,
    ) -> None:
        """Compile one observed static call and update its exact dispatcher."""
        normalized_args = self.kernel.kernel.normalize_args(*self.args)
        config = self._get_heuristic_config(normalized_args)
        if isinstance(config, _AOTHeuristicAbstention):
            log.debug(
                "Heuristic abstained for static kernel '%s', skipping standalone "
                "compile",
                kernel_name,
            )
            return
        if config is None:
            log.warning(
                "No heuristic config for static kernel '%s', skipping standalone compile",
                kernel_name,
            )
            return

        call_key = _standalone_call_key(normalized_args)
        code_object = self.kernel.kernel.__code__
        source_path = self._kernel_source_file
        output_path = standalone_output_path(
            kernel_name=kernel_name,
            output_dir=self.data_dir,
            kernel_source_file=source_path,
        )
        if source_path is not None:
            source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        else:
            source_digest = hashlib.sha256(marshal.dumps(code_object)).hexdigest()
        output_identity = str(output_path.resolve())
        variants_key = (
            output_identity,
            kernel_name,
            self.hardware_id,
            hashlib.sha256(heuristic_file.read_bytes()).hexdigest(),
            source_digest,
        )
        config_fingerprint = json.dumps(dict(config), sort_keys=True, default=repr)
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

        # The compile workflow runs in one process. Serialize concurrent calls
        # in that process so two newly observed signatures cannot each rewrite
        # an incomplete dispatcher. ``generate_standalone_file`` replaces the
        # output atomically, so readers never observe a partial module.
        with AOTAutotuneCache._compiled_kernel_variants_lock:
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
                return

            updated = {**variants, call_key: (config_fingerprint, code)}
            ordered = sorted(updated.items(), key=lambda item: repr(item[0]))
            out_path = generate_standalone_file(
                kernel_name=kernel_name,
                triton_codes=[variant[1][1] for variant in ordered],
                heuristic_code=heuristic_file.read_text(),
                output_dir=self.data_dir,
                kernel_source_file=source_path,
                dispatch_keys=[variant[0] for variant in ordered],
            )
            variants[call_key] = (config_fingerprint, code)
        print(f"[AOT] Standalone: {out_path}", file=sys.stderr)

    @staticmethod
    def _parse_configs_from_autotune(
        module: object, kernel_name: str
    ) -> list[dict[str, object]] | None:
        """Extract the ``_C`` config list from ``autotune_<kernel>``."""
        autotune_fn = getattr(module, f"autotune_{kernel_name}", None)
        if autotune_fn is None:
            return None
        try:
            src = inspect.getsource(autotune_fn)
        except OSError:
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
            f"[AOT measure_fn] Completed {count} shapes! "
            f"Results saved to {self._measurements_file}",
            file=sys.stderr,
        )

    def _maybe_run_input_fn_workflows(self) -> None:
        """Run collect_fn/measure_fn workflows if applicable."""
        # Check if input_fn workflow should run (only once per kernel)
        if self.kernel.kernel._aot_workflow_done:
            return
        # Mark done FIRST to prevent recursive calls when we invoke the kernel
        self.kernel.kernel._aot_workflow_done = True

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
                return best_config
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
    torch.dtype, torch.device, list, dict, code, bytes, ellipsis, complex.
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
    __type__, __bytes__, __ellipsis__, __complex__.

    Note: __code__ is not turned back into a code object. This is fine because the AOT
    cache only compares specialization keys, it never calls the original function.
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
