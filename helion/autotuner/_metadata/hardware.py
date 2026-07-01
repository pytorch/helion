"""Hardware/runtime snapshot for the autotuner cost-model dataset.

``collect_hardware_info`` runs lazily from ``KernelMetadata.to_dict`` (dataset-only
write path), never on the normal autotune path. Numeric props are best-effort
(per-field ``None`` on miss); device identity and backend-required tool versions
raise on real failure, which the dataset sink (``logger.end_run``) guards.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import platform
import re
import sys
from typing import TypedDict

import torch

from ..._compat import get_device_name
from ..._hardware import get_hardware_info

logger = logging.getLogger(__name__)


class DevicePropsRecord(TypedDict):
    """Nested under ``hardware_info`` only on GPU backends; absent on TPU/CPU."""

    sm_count: int | None
    max_threads_per_sm: int | None
    max_threads_per_block: int | None
    warp_size: int | None
    shared_mem_per_block: int | None
    regs_per_multiprocessor: int | None
    total_mem: int | None
    l2_cache_size: int | None


class _HardwareInfoBase(TypedDict):
    device_id: str | None
    device_kind: str | None
    device_name: str | None
    compute_capability: str | None
    cpu_num_threads: int | None
    # Keys vary by backend; see _hardware_versions.
    versions: dict[str, str | None]


class HardwareInfoRecord(_HardwareInfoBase, total=False):
    """``total=False`` keeps ``device_props`` optional; a single-class ``NotRequired``
    does not, being inert under ``from __future__ import annotations``."""

    device_props: DevicePropsRecord


def _device_id(
    device_kind: str | None, device_name: str | None, compute_capability: str | None
) -> str | None:
    """Stable id with no driver/runtime version, so it survives driver bumps."""
    if not device_name:
        return None
    # Collapse anything outside the id alphabet (notably ':') so the kind:name:cap
    # split stays unambiguous for downstream parsing.
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", device_name)
    return f"{device_kind or 'unknown'}:{safe_name}:{compute_capability or 'none'}"


def _cpu_name() -> str:
    return platform.machine() or "cpu"


def _hardware_identity(
    device: torch.device | None,
) -> tuple[str | None, str | None, str | None]:
    """Device identity via the canonical ``get_hardware_info``, preferring the
    marketing name (it reports ROCm's gfx ISA). Raises for an unsupported device
    (you cannot be autotuning on one); the sink guard keeps that non-fatal."""
    if device is not None and device.type == "cpu":
        return "cpu", _cpu_name(), None
    hw = get_hardware_info(device)
    name = get_device_name(device) or hw.hardware_name
    return hw.device_kind, name, hw.compute_capability


def _package_version(name: str, module: str | None = None) -> str | None:
    """Dist metadata, then the live module's ``__version__`` -- the two differ when
    CI ships triton as the ``pytorch-triton`` dist (metadata misses, ``import triton``
    works). ``None`` only when genuinely absent."""
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(name)
    mod = sys.modules.get(module or name)
    return getattr(mod, "__version__", None) if mod else None


# Version-block key per GPU backend, named for the toolkit (ROCm's is HIP), not the
# device kind.
_TOOLKIT_KEY: dict[str, str] = {"cuda": "cuda", "rocm": "hip", "xpu": "xpu"}


# Derived from _TOOLKIT_KEY so the toolkit-version and device_props gates can't drift.
_GPU_BACKENDS: frozenset[str] = frozenset(_TOOLKIT_KEY)


def _toolkit_version(device_kind: str | None) -> str | None:
    """Build/runtime toolkit version (CUDA/HIP/XPU), not the driver (kept out of
    identity; see ``_device_id``). Emitted under the ``_TOOLKIT_KEY`` name."""
    if device_kind == "cuda":
        return None if torch.version.cuda is None else str(torch.version.cuda)
    if device_kind == "rocm":
        return None if torch.version.hip is None else str(torch.version.hip)
    if device_kind == "xpu":
        # torch.version.xpu (sibling of cuda/hip); torch.xpu has no __version__.
        xpu = getattr(torch.version, "xpu", None)
        return None if xpu is None else str(xpu)
    return None


# Per backend, as (dist_name, import_name, required). A required package that is
# absent raises (you could not have autotuned without it); libtpu is best-effort (C
# plugin, often no dist/__version__).
_BACKEND_PACKAGES: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "cuda": (("triton", "triton", True),),
    "rocm": (("triton", "triton", True),),
    "xpu": (("triton", "triton", True),),  # XPU codegen goes through triton
    "tpu": (
        ("jax", "jax", True),
        ("jaxlib", "jaxlib", True),
        ("libtpu", "libtpu", False),
    ),
}


def _hardware_versions(device_kind: str | None) -> dict[str, str | None]:
    """Backend-aware version block. Always torch + helion + the toolkit version
    (under its ``_TOOLKIT_KEY`` name, e.g. ``hip`` for ROCm). Backend packages come
    from ``_BACKEND_PACKAGES``; a missing required one raises."""
    versions: dict[str, str | None] = {
        "torch": torch.__version__,
        "helion": _package_version("helion"),
    }
    # GPU backends always carry the toolkit key (``None`` if unavailable).
    toolkit_key = _TOOLKIT_KEY.get(device_kind or "")
    if toolkit_key is not None:
        versions[toolkit_key] = _toolkit_version(device_kind)
    for dist, module, required in _BACKEND_PACKAGES.get(device_kind or "", ()):
        version = _package_version(dist, module)
        if version is None and required:
            raise RuntimeError(
                f"{dist} required for {device_kind} backend but not found"
            )
        versions[dist] = version
    return versions


def _first_attr(props: object, *names: str) -> int | None:
    for name in names:
        value = getattr(props, name, None)
        if value is not None:
            return value
    return None


def _probe_device_props(
    device: torch.device | None, device_kind: str | None
) -> object | None:
    """Best-effort ``get_device_properties`` for the active GPU backend; ``None`` on
    any miss so the props block degrades to all-``None`` instead of raising."""
    try:
        if device_kind in ("cuda", "rocm") and torch.cuda.is_available():
            dev = (
                device
                if device is not None and device.type == "cuda"
                else torch.device("cuda:0")
            )
            return torch.cuda.get_device_properties(dev)
        if (
            device_kind == "xpu"
            and getattr(torch, "xpu", None) is not None
            and torch.xpu.is_available()
        ):
            dev = (
                device
                if device is not None and device.type == "xpu"
                else torch.device("xpu:0")
            )
            return torch.xpu.get_device_properties(dev)
    except Exception:
        logger.debug("device property probe failed", exc_info=True)
    return None


def _collect_device_props(props: object | None) -> DevicePropsRecord:
    """``None`` props yields an all-``None`` block. XPU maps onto the cuda-named
    fields where the analog is clean; no-analog fields stay ``None``."""
    return {
        "sm_count": _first_attr(props, "multi_processor_count", "max_compute_units"),
        "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", None),
        "max_threads_per_block": _first_attr(
            props, "max_threads_per_block", "max_work_group_size"
        ),
        "warp_size": getattr(props, "warp_size", None),
        "shared_mem_per_block": _first_attr(
            props, "shared_memory_per_block", "local_mem_size"
        ),
        "regs_per_multiprocessor": getattr(props, "regs_per_multiprocessor", None),
        "total_mem": getattr(props, "total_memory", None),
        "l2_cache_size": getattr(props, "L2_cache_size", None),
    }


def collect_hardware_info(device: torch.device | None = None) -> HardwareInfoRecord:
    """Hardware snapshot excluded from ``run_id``. Identity and required-version
    probes may raise; the dataset sink guard keeps that non-fatal."""
    device_kind, device_name, compute_capability = _hardware_identity(device)
    cpu_num_threads = None
    with contextlib.suppress(Exception):
        cpu_num_threads = torch.get_num_threads()
    record: HardwareInfoRecord = {
        "device_id": _device_id(device_kind, device_name, compute_capability),
        "device_kind": device_kind,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "cpu_num_threads": cpu_num_threads,
        "versions": _hardware_versions(device_kind),
    }
    # Present on every GPU backend, even when the probe failed (all-None block then).
    if device_kind in _GPU_BACKENDS:
        record["device_props"] = _collect_device_props(
            _probe_device_props(device, device_kind)
        )
    return record
