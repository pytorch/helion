from __future__ import annotations

from typing import cast
from typing import overload

import torch

from . import exc


def _current_device_index(device_type: str) -> int:
    device_module = getattr(torch, device_type, None)
    is_available = getattr(device_module, "is_available", None)
    available = None if not callable(is_available) else is_available()
    current_device = getattr(device_module, "current_device", None)
    if available is not False and callable(current_device):
        return cast("int", current_device())

    accelerator_device = torch.accelerator.current_accelerator(check_available=True)
    if accelerator_device is not None and accelerator_device.type == device_type:
        return torch.accelerator.current_device_index()
    raise exc.InvalidAPIUsage(
        f"no current indexed accelerator is available for {device_type!r}"
    )


@overload
def _canonicalize_argument_device(device: None) -> None: ...


@overload
def _canonicalize_argument_device(device: torch.device) -> torch.device: ...


def _canonicalize_argument_device(
    device: torch.device | None,
) -> torch.device | None:
    """Resolve an indexless accelerator while preserving discovery fallback."""
    if device is None:
        return None
    if device.index is not None or device.type in ("cpu", "meta", "mps"):
        return device
    try:
        index = _current_device_index(device.type)
    except exc.InvalidAPIUsage:
        # Preserve device kinds without PyTorch current-device support. Their
        # existing backend-specific discovery remains authoritative.
        return device
    return torch.device(device.type, index)
