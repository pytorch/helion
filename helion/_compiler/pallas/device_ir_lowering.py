"""Pallas-specific device IR lowering pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..device_ir_lowering import DeviceIRLowering

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from ..host_function import HostFunction


class PallasDeviceIRLowering(DeviceIRLowering):
    """Pallas-specific overrides for the device IR lowering pipeline."""

    def trace(self, device_ir: DeviceIR, func: HostFunction) -> None:
        super().trace(device_ir, func)

    def transform(self, device_ir: DeviceIR, func: HostFunction) -> None:
        super().transform(device_ir, func)

    def lower(self, device_ir: DeviceIR) -> None:
        super().lower(device_ir)

    def optimize(self, device_ir: DeviceIR) -> None:
        super().optimize(device_ir)

    def register(self, device_ir: DeviceIR) -> None:
        super().register(device_ir)
