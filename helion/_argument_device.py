from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Hashable
from typing import Literal
from typing import NamedTuple
from typing import cast
from typing import overload

import torch

from . import exc
from .language.constexpr import ConstExpr

if TYPE_CHECKING:
    from collections.abc import Sequence


class _DevicePathStep(NamedTuple):
    """One validated hop through an argument container."""

    kind: Literal["sequence", "mapping"]
    key: int | Hashable
    container_type: type[object]
    position: int


_DevicePath = tuple[_DevicePathStep, ...]


def _has_argument_device(value: object) -> bool:
    """Return whether normal kernel traversal would find a device in ``value``."""
    if isinstance(value, (torch.device, torch.Tensor)):
        return True
    if isinstance(value, ConstExpr):
        return False
    if isinstance(value, (tuple, list)):
        return any(_has_argument_device(item) for item in value)
    if isinstance(value, dict):
        return any(_has_argument_device(item) for item in value.values())
    return False


def _leaf_device(value: object) -> torch.device | None:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, torch.Tensor):
        return value.device
    return None


def _find_argument_device_with_path(
    values: Sequence[object],
) -> tuple[torch.device, _DevicePath] | None:
    """Find the first device-bearing value and its container path."""

    def visit(
        value: object,
        path: _DevicePath,
    ) -> tuple[torch.device, _DevicePath] | None:
        if (device := _leaf_device(value)) is not None:
            return device, path
        if isinstance(value, ConstExpr):
            return None
        if isinstance(value, (tuple, list)):
            container_type = type(value)
            for index, item in enumerate(value):
                step = _DevicePathStep(
                    "sequence",
                    index,
                    container_type,
                    index,
                )
                if result := visit(item, (*path, step)):
                    return result
        elif isinstance(value, dict):
            container_type = type(value)
            for position, (key, item) in enumerate(value.items()):
                step = _DevicePathStep(
                    "mapping",
                    cast("Hashable", key),
                    container_type,
                    position,
                )
                if result := visit(item, (*path, step)):
                    return result
        return None

    return visit(values, ())


def _find_argument_device(values: Sequence[object]) -> torch.device:
    """Return the first device found by normal kernel argument traversal."""
    result = _find_argument_device_with_path(values)
    if result is None:
        raise exc.NoTensorArgs
    return result[0]


def _mapping_key_matches(actual: object, expected: object) -> bool:
    if actual is expected:
        return True
    try:
        return bool(actual == expected)
    except (RuntimeError, TypeError, ValueError):
        # Tensor-valued or otherwise non-scalar mapping keys may not define a
        # usable boolean equality. Treat that as a cache-path miss; the caller
        # falls back to a complete argument traversal.
        return False


def _device_at_path(values: Sequence[object], path: _DevicePath) -> torch.device | None:
    """Read a cached device path after validating structure and precedence."""
    value: object = values
    for step in path:
        if type(value) is not step.container_type:
            return None
        if step.kind == "sequence":
            if not isinstance(value, (tuple, list)) or not isinstance(step.key, int):
                return None
            if step.key < 0 or step.key >= len(value):
                return None
            if any(_has_argument_device(item) for item in value[: step.key]):
                return None
            value = value[step.key]
            continue

        if not isinstance(value, dict):
            return None
        for position, (key, item) in enumerate(value.items()):
            if position == step.position:
                if not _mapping_key_matches(key, step.key):
                    return None
                value = item
                break
            if _has_argument_device(item):
                return None
        else:
            return None

    return _leaf_device(value)


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


@dataclasses.dataclass
class _ArgumentDeviceResolver:
    """Cache the first argument-device path while preserving traversal semantics."""

    path: _DevicePath | None = None

    @classmethod
    def from_values(cls, values: Sequence[object]) -> _ArgumentDeviceResolver:
        result = _find_argument_device_with_path(values)
        if result is None:
            raise exc.NoTensorArgs
        return cls(result[1])

    def resolve(self, values: Sequence[object]) -> torch.device | None:
        device = None if self.path is None else _device_at_path(values, self.path)
        if device is None:
            result = _find_argument_device_with_path(values)
            if result is None:
                self.path = None
                return None
            device, self.path = result
        return _canonicalize_argument_device(device)

    def __call__(self, values: Sequence[object]) -> torch.device:
        device = self.resolve(values)
        if device is None:
            raise exc.NoTensorArgs
        return device
