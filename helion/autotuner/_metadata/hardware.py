"""Hardware/runtime snapshot for the autotuner cost-model dataset.

Derives the flat ``hardware`` device name and the structured, JSON-safe
``hardware_info`` record recorded on the per-run ``.meta.jsonl`` line. Best-effort
and read-only: every field is ``None`` on miss and collection never raises. Builds
on the pre-existing device identity in :mod:`helion._hardware`.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import platform
from typing import TypedDict

import torch

from ..._compat import get_device_name
from ..._hardware import get_hardware_info


class HardwareInfoRecord(TypedDict):
    """Per-run hardware/runtime snapshot for ``.meta.jsonl`` (JSON-safe; None on miss)."""

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
    """Stable physical-device id ``kind:name:compute_capability`` (no driver version)."""
    if not device_name:
        return None
    safe_name = device_name.replace(" ", "_")
    return f"{device_kind or 'unknown'}:{safe_name}:{compute_capability or 'none'}"


def _cpu_name() -> str:
    """Host CPU architecture (``platform.machine()``), or ``"cpu"`` if unknown."""
    return platform.machine() or "cpu"


def _hardware_identity(
    device: torch.device | None,
) -> tuple[str | None, str | None, str | None]:
    """Best-effort ``(device_kind, device_name, compute_capability)``; cpu first."""
    if device is not None and device.type == "cpu":
        return "cpu", _cpu_name(), None
    try:
        hw = get_hardware_info(device)
        # Product name (get_hardware_info gives ROCm's gfx ISA, not the marketing name).
        name = hw.hardware_name
        with contextlib.suppress(Exception):
            name = get_device_name(device) or hw.hardware_name
        return hw.device_kind, name, hw.compute_capability
    except Exception:
        pass
    kind = device.type if device is not None else "cpu"
    name = _cpu_name() if kind == "cpu" else None
    return kind, name, None


def _package_version(name: str) -> str | None:
    """Installed distribution version via dist metadata (no import), else ``None``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tpu_runtime_version() -> str | None:
    """torch_xla version if present, else jax version, else ``None`` (lazy import)."""
    for mod_name in ("torch_xla", "jax"):
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            return str(version)
    return None


def _hardware_versions(device_kind: str | None) -> dict[str, str | None]:
    """torch/triton/helion versions plus only the active backend's runtime key."""
    versions: dict[str, str | None] = {
        "torch": torch.__version__,
        "triton": _package_version("triton"),
        "helion": _package_version("helion"),
    }
    if device_kind == "cuda" and torch.version.cuda is not None:
        versions["cuda"] = str(torch.version.cuda)
    elif device_kind == "rocm" and torch.version.hip is not None:
        versions["hip"] = str(torch.version.hip)
    elif device_kind == "xpu":
        # torch.version.xpu (sibling of cuda/hip); torch.xpu has no __version__.
        versions["xpu"] = getattr(torch.version, "xpu", None)
    elif device_kind == "tpu":
        versions["xla"] = _tpu_runtime_version()
    return versions


def _first_attr(props: object, *names: str) -> int | None:
    """First present (non-``None``) attribute among ``names``, else ``None``."""
    for name in names:
        value = getattr(props, name, None)
        if value is not None:
            return value
    return None


def _device_props(
    device: torch.device | None, device_kind: str | None
) -> dict[str, int | None]:
    """Fixed-key numeric device props (``None`` per field on miss, never raises);
    cuda/rocm-named, with XPU fallbacks where the analog is clean."""
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
    with contextlib.suppress(Exception):
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
    """Best-effort JSON-safe hardware snapshot for ``.meta.jsonl`` (None on miss,
    never raises, excluded from ``run_id``)."""
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


def describe_device(
    device: torch.device | None, *, collect_info: bool
) -> tuple[str, HardwareInfoRecord | None]:
    """Derive the flat ``hardware`` name and (optionally) the structured snapshot
    from one ``device``.

    ``hardware`` is always the ``get_device_name`` string (a ``run_id`` input, so it
    must not depend on whether the snapshot was collected); ``hardware_info`` is the
    descriptive record, collected only when ``collect_info`` (the dataset path).
    """
    hardware = get_device_name(device) or ""
    info = collect_hardware_info(device) if collect_info else None
    return hardware, info
