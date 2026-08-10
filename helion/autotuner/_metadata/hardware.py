"""Hardware and runtime metadata for autotuning datasets.

Identity and version probes are strict. Numeric device properties are best-effort.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
from typing import TypedDict

import torch

from ..._hardware import get_hardware_info

logger = logging.getLogger(__name__)


class _HardwareInfoBase(TypedDict):
    device_kind: str
    device_name: str
    compute_capability: str | None
    cpu_num_threads: int
    versions: dict[str, str]


class HardwareInfoRecord(_HardwareInfoBase, total=False):
    """``total=False`` keeps ``device_props`` optional; a single-class ``NotRequired``
    does not, being inert under ``from __future__ import annotations``."""

    # GPU-only; keys are the raw per-backend get_device_properties attr names.
    device_props: dict[str, int | None]


def _cpu_name() -> str:
    return platform.machine() or "cpu"


def _hardware_identity(
    device: torch.device,
) -> tuple[str, str, str | None]:
    """Device identity from the canonical hardware record."""
    if device.type == "cpu":
        return "cpu", _cpu_name(), None
    hw = get_hardware_info(device)
    return hw.device_kind, hw.hardware_name, hw.compute_capability


# CUDA and ROCm both use the torch.cuda properties object.
_CUDA_PROPS_ATTRS: tuple[str, ...] = (
    "multi_processor_count",
    "max_threads_per_multi_processor",
    "max_threads_per_block",
    "warp_size",
    "shared_memory_per_block",
    "shared_memory_per_block_optin",
    "regs_per_multiprocessor",
    "total_memory",
    "L2_cache_size",
)
_DEVICE_PROPS_ATTRS: dict[str, tuple[str, ...]] = {
    "cuda": _CUDA_PROPS_ATTRS,
    "rocm": _CUDA_PROPS_ATTRS,
    "xpu": (
        "max_compute_units",
        "max_work_group_size",
        "local_mem_size",
        "total_memory",
    ),
}

_GPU_BACKENDS: frozenset[str] = frozenset(_DEVICE_PROPS_ATTRS)

_BACKEND_PACKAGES: dict[str, tuple[str, ...]] = {
    "cuda": ("triton",),
    "rocm": ("triton",),
    "xpu": ("triton",),  # XPU codegen goes through triton
    "tpu": ("jax", "jaxlib", "libtpu"),
}


def _package_version(name: str) -> str:
    """Read distribution metadata, falling back for ROCm's Triton package."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return importlib.import_module(name).__version__


def _hardware_versions(device_kind: str) -> dict[str, str]:
    versions: dict[str, str] = {
        "torch": torch.__version__,
        "helion": importlib.metadata.version("helion"),
    }
    if device_kind in _GPU_BACKENDS:
        toolkit_key = "hip" if device_kind == "rocm" else device_kind
        toolkit = getattr(torch.version, toolkit_key)
        if toolkit is None:
            raise RuntimeError(f"torch.version.{toolkit_key} is unavailable")
        versions[toolkit_key] = str(toolkit)
    for name in _BACKEND_PACKAGES.get(device_kind, ()):
        versions[name] = _package_version(name)
    return versions


def _device_props(
    device: torch.device, device_kind: str
) -> dict[str, int | None] | None:
    """Return backend-native numeric properties, using ``None`` when unavailable."""
    attrs = _DEVICE_PROPS_ATTRS.get(device_kind)
    if attrs is None:
        return None
    props: object | None = None
    try:
        if device_kind in ("cuda", "rocm") and torch.cuda.is_available():
            dev = device if device.type == "cuda" else torch.device("cuda:0")
            props = torch.cuda.get_device_properties(dev)
        elif (
            device_kind == "xpu"
            and getattr(torch, "xpu", None) is not None
            and torch.xpu.is_available()
        ):
            dev = device if device.type == "xpu" else torch.device("xpu:0")
            props = torch.xpu.get_device_properties(dev)
    except Exception:
        logger.debug("device property probe failed", exc_info=True)
    return {name: getattr(props, name, None) for name in attrs}


def collect_hardware_info(device: torch.device) -> HardwareInfoRecord:
    """Collect a hardware snapshot excluded from ``run_id``."""
    if device is None:
        raise ValueError("device is required to collect hardware info")
    device_kind, device_name, compute_capability = _hardware_identity(device)
    record: HardwareInfoRecord = {
        "device_kind": device_kind,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "cpu_num_threads": torch.get_num_threads(),
        "versions": _hardware_versions(device_kind),
    }
    props = _device_props(device, device_kind)
    if props is not None:
        record["device_props"] = props
    return record
