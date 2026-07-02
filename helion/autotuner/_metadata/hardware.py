"""Hardware/runtime snapshot for the autotuner cost-model dataset.

``collect_hardware_info`` runs lazily from ``KernelMetadata.to_dict`` (dataset-only
write path), never on the normal autotune path. The GPU-only ``device_props`` block
carries raw per-backend ``get_device_properties`` values (per-field ``None`` on miss);
device identity and backend-required tool versions raise on real failure, surfacing
through ``end_run``.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import platform
from typing import TypedDict

import torch

from ..._compat import get_device_name
from ..._hardware import get_hardware_info

logger = logging.getLogger(__name__)


class _HardwareInfoBase(TypedDict):
    device_kind: str | None
    device_name: str | None
    compute_capability: str | None
    cpu_num_threads: int | None
    # Keys vary by backend; see _hardware_versions.
    versions: dict[str, str | None]


class HardwareInfoRecord(_HardwareInfoBase, total=False):
    """``total=False`` keeps ``device_props`` optional; a single-class ``NotRequired``
    does not, being inert under ``from __future__ import annotations``."""

    # GPU-only; keys are the raw per-backend get_device_properties attr names.
    device_props: dict[str, int | None]


def _cpu_name() -> str:
    return platform.machine() or "cpu"


def _hardware_identity(
    device: torch.device | None,
) -> tuple[str | None, str | None, str | None]:
    """Device identity via the canonical ``get_hardware_info``, preferring the
    marketing name. Raises for an unsupported device, which cannot happen on
    the success path."""
    if device is not None and device.type == "cpu":
        return "cpu", _cpu_name(), None
    hw = get_hardware_info(device)
    name = get_device_name(device) or hw.hardware_name
    return hw.device_kind, name, hw.compute_capability


# Maps GPU device properties using PyTorch attribute names, not native HIP/SYCL names.
# CUDA & ROCm hardware share the exact same attribute names because they both rely on
# the torch.cuda properties object.
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

# GPU backends are exactly those with a device-props table.
_GPU_BACKENDS: frozenset[str] = frozenset(_DEVICE_PROPS_ATTRS)


# Packages whose versions we record per backend.
_BACKEND_PACKAGES: dict[str, tuple[str, ...]] = {
    "cuda": ("triton",),
    "rocm": ("triton",),
    "xpu": ("triton",),  # XPU codegen goes through triton
    "tpu": ("jax", "jaxlib", "libtpu"),
}


def _hardware_versions(device_kind: str | None) -> dict[str, str | None]:
    """PyTorch and Helion versions are always included. GPU backends also record the
    toolkit version (CUDA, HIP, or XPU) under its own key. Each backend's required
    packages come from _BACKEND_PACKAGES.
    """
    versions: dict[str, str | None] = {
        "torch": torch.__version__,
        "helion": importlib.metadata.version("helion"),
    }
    if device_kind is not None and device_kind in _GPU_BACKENDS:
        toolkit_key = "hip" if device_kind == "rocm" else device_kind
        toolkit = getattr(torch.version, toolkit_key, None)
        versions[toolkit_key] = None if toolkit is None else str(toolkit)
    for name in _BACKEND_PACKAGES.get(device_kind or "", ()):
        versions[name] = importlib.metadata.version(name)
    return versions


def _device_props(
    device: torch.device | None, device_kind: str | None
) -> dict[str, int | None] | None:
    """Raw ``get_device_properties`` values for the active GPU backend, keyed by the torch
    attribute name (per-backend; normalized downstream). ``None`` for non-GPU kinds; a
    probe miss yields an all-``None`` block."""
    attrs = _DEVICE_PROPS_ATTRS.get(device_kind or "")
    if attrs is None:
        return None
    props: object | None = None
    try:
        if device_kind in ("cuda", "rocm") and torch.cuda.is_available():
            dev = (
                device
                if device is not None and device.type == "cuda"
                else torch.device("cuda:0")
            )
            props = torch.cuda.get_device_properties(dev)
        elif (
            device_kind == "xpu"
            and getattr(torch, "xpu", None) is not None
            and torch.xpu.is_available()
        ):
            dev = (
                device
                if device is not None and device.type == "xpu"
                else torch.device("xpu:0")
            )
            props = torch.xpu.get_device_properties(dev)
    except Exception:
        logger.debug("device property probe failed", exc_info=True)
    return {name: getattr(props, name, None) for name in attrs}


def collect_hardware_info(device: torch.device | None = None) -> HardwareInfoRecord:
    """Hardware snapshot excluded from ``run_id``. Identity and required-version probes
    may raise; end_run calls this only on a successful tune, so it surfaces."""
    device_kind, device_name, compute_capability = _hardware_identity(device)
    cpu_num_threads = None
    with contextlib.suppress(Exception):
        cpu_num_threads = torch.get_num_threads()
    record: HardwareInfoRecord = {
        "device_kind": device_kind,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "cpu_num_threads": cpu_num_threads,
        "versions": _hardware_versions(device_kind),
    }
    props = _device_props(device, device_kind)
    if props is not None:
        record["device_props"] = props
    return record
