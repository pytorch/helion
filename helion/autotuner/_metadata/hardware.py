"""Hardware and runtime metadata for autotuning datasets."""

from __future__ import annotations

import importlib.metadata
import platform
from typing import TypedDict

import torch

from ..._hardware import get_hardware_info


class _HardwareInfoBase(TypedDict):
    device_kind: str
    device_name: str
    compute_capability: str | None
    cpu_num_threads: int
    versions: dict[str, str]


class HardwareInfoRecord(_HardwareInfoBase, total=False):
    """``total=False`` keeps ``device_props`` optional; a single-class ``NotRequired``
    does not, being inert under ``from __future__ import annotations``."""

    # Accelerator-only; keys are raw per-backend get_device_properties attr names.
    device_props: dict[str, int]


def _cpu_name() -> str:
    name = platform.machine()
    if not name:
        raise RuntimeError("platform.machine() returned an empty CPU name")
    return name


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
    "cpu": (),
    "cuda": ("triton",),
    "rocm": ("triton",),
    "xpu": ("triton",),  # XPU codegen goes through triton
    "tpu": ("jax", "jaxlib", "libtpu"),
}


def _package_version(name: str) -> str:
    """Read required package distribution metadata."""
    return importlib.metadata.version(name)


def _toolkit_version(name: str, version: str | None) -> str:
    if version is None:
        raise RuntimeError(f"torch.version.{name} is unavailable")
    return str(version)


def _hardware_versions(device_kind: str) -> dict[str, str]:
    versions: dict[str, str] = {
        "torch": torch.__version__,
        "helion": importlib.metadata.version("helion"),
    }
    if device_kind == "cuda":
        versions["cuda"] = _toolkit_version("cuda", torch.version.cuda)
    elif device_kind == "rocm":
        versions["hip"] = _toolkit_version("hip", torch.version.hip)
    elif device_kind == "xpu":
        versions["xpu"] = _toolkit_version("xpu", torch.version.xpu)
    for name in _BACKEND_PACKAGES[device_kind]:
        if (device_kind, name) == ("rocm", "triton"):
            try:
                versions[name] = _package_version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = importlib.import_module(name).__version__
        else:
            versions[name] = _package_version(name)
    return versions


def _device_props(device: torch.device, device_kind: str) -> dict[str, int] | None:
    """Return required backend-native numeric properties for accelerators."""
    if device_kind in ("cpu", "tpu"):
        return None
    attrs = _DEVICE_PROPS_ATTRS[device_kind]
    if device_kind in ("cuda", "rocm"):
        props = torch.cuda.get_device_properties(device)
    else:
        props = torch.xpu.get_device_properties(device)
    return {name: getattr(props, name) for name in attrs}


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
