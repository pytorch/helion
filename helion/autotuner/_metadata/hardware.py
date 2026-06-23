"""Hardware/runtime snapshot for the autotuner cost-model dataset.

``collect_hardware_info`` is called lazily from ``KernelMetadata.to_dict`` (the
dataset-only write path), so the autotuner never probes the device on the normal
path. Best-effort: every field is ``None`` on miss and it never raises. The record
shape is the ``HardwareInfoRecord`` TypedDict below.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import platform
import re
from typing import TypedDict

import torch

from ..._compat import get_device_name
from ..._hardware import get_hardware_info

logger = logging.getLogger(__name__)


class HardwareInfoRecord(TypedDict):
    """Fixed-shape ``hardware_info`` record (one per autotune run)."""

    device_id: str | None
    device_kind: str | None
    device_name: str | None
    compute_capability: str | None
    sm_count: int | None
    max_threads_per_sm: int | None
    max_threads_per_block: int | None
    warp_size: int | None
    shared_mem_per_block: int | None
    regs_per_multiprocessor: int | None
    total_mem: int | None
    l2_cache_size: int | None
    cpu_num_threads: int | None
    versions: dict[str, str | None]


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
    if device is not None and device.type == "cpu":
        return "cpu", _cpu_name(), None
    try:
        hw = get_hardware_info(device)
        # get_hardware_info returns ROCm's gfx ISA; prefer the marketing name.
        name = hw.hardware_name
        with contextlib.suppress(Exception):
            name = get_device_name(device) or hw.hardware_name
        return hw.device_kind, name, hw.compute_capability
    except Exception:
        logger.debug(
            "get_hardware_info failed; falling back to device type", exc_info=True
        )
    kind = device.type if device is not None else "cpu"
    name = _cpu_name() if kind == "cpu" else None
    return kind, name, None


def _package_version(name: str) -> str | None:
    """Version from dist metadata, not an import (triton is optional)."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tpu_runtime_version() -> str | None:
    """TPU runtime version from dist metadata -- avoids importing torch_xla/jax,
    which can trigger XLA runtime init side effects."""
    return _package_version("torch_xla") or _package_version("jax")


def _hardware_versions(device_kind: str | None) -> dict[str, str | None]:
    """torch/triton/helion always, plus exactly one backend key named by
    ``device_kind`` (value may be ``None``); ``cpu``/unknown add none. The key is
    added unconditionally so the count is the same across backends."""
    versions: dict[str, str | None] = {
        "torch": torch.__version__,
        "triton": _package_version("triton"),
        "helion": _package_version("helion"),
    }
    if device_kind == "cuda":
        cuda = torch.version.cuda
        versions["cuda"] = None if cuda is None else str(cuda)
    elif device_kind == "rocm":
        hip = torch.version.hip
        versions["hip"] = None if hip is None else str(hip)
    elif device_kind == "xpu":
        # torch.version.xpu (sibling of cuda/hip); torch.xpu has no __version__.
        xpu = getattr(torch.version, "xpu", None)
        versions["xpu"] = None if xpu is None else str(xpu)
    elif device_kind == "tpu":
        versions["xla"] = _tpu_runtime_version()
    return versions


def _first_attr(props: object, *names: str) -> int | None:
    for name in names:
        value = getattr(props, name, None)
        if value is not None:
            return value
    return None


def _device_props(
    device: torch.device | None, device_kind: str | None
) -> dict[str, int | None]:
    """Numeric device props, ``None`` per field on miss, never raises. XPU maps to
    the cuda/rocm-named fields where the analog is clean."""
    fields: dict[str, int | None] = dict.fromkeys(
        (
            "sm_count",
            "max_threads_per_sm",
            "max_threads_per_block",
            "warp_size",
            "shared_mem_per_block",
            "regs_per_multiprocessor",
            "total_mem",
            "l2_cache_size",
        ),
        None,
    )
    props = None
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
        # A driver/runtime API change (e.g. get_device_properties) surfaces here as
        # all-null props; log at debug so it is not a silent degradation.
        logger.debug("device property probe failed", exc_info=True)
    if props is None:
        return fields
    fields["sm_count"] = _first_attr(
        props, "multi_processor_count", "max_compute_units"
    )
    fields["max_threads_per_sm"] = getattr(
        props, "max_threads_per_multi_processor", None
    )
    fields["max_threads_per_block"] = _first_attr(
        props, "max_threads_per_block", "max_work_group_size"
    )
    fields["warp_size"] = getattr(props, "warp_size", None)
    fields["shared_mem_per_block"] = _first_attr(
        props, "shared_memory_per_block", "local_mem_size"
    )
    fields["regs_per_multiprocessor"] = getattr(props, "regs_per_multiprocessor", None)
    fields["total_mem"] = getattr(props, "total_memory", None)
    fields["l2_cache_size"] = getattr(props, "L2_cache_size", None)
    return fields


def collect_hardware_info(device: torch.device | None = None) -> HardwareInfoRecord:
    """Best-effort hardware snapshot; never raises, excluded from ``run_id``."""
    device_kind, device_name, compute_capability = _hardware_identity(device)
    cpu_num_threads = None
    with contextlib.suppress(Exception):
        cpu_num_threads = torch.get_num_threads()
    props = _device_props(device, device_kind)
    return {
        "device_id": _device_id(device_kind, device_name, compute_capability),
        "device_kind": device_kind,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "sm_count": props["sm_count"],
        "max_threads_per_sm": props["max_threads_per_sm"],
        "max_threads_per_block": props["max_threads_per_block"],
        "warp_size": props["warp_size"],
        "shared_mem_per_block": props["shared_mem_per_block"],
        "regs_per_multiprocessor": props["regs_per_multiprocessor"],
        "total_mem": props["total_mem"],
        "l2_cache_size": props["l2_cache_size"],
        "cpu_num_threads": cpu_num_threads,
        "versions": _hardware_versions(device_kind),
    }
