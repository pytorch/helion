from __future__ import annotations

import dataclasses
import functools
import logging

import torch

log: logging.Logger = logging.getLogger(__name__)

# Compute capability lists for fallback (newest to oldest)
_CUDA_COMPUTE_CAPS: list[str] = [
    "sm100",
    "sm90",
    "sm89",
    "sm87",
    "sm86",
    "sm80",
    "sm75",
    "sm72",
    "sm70",
]

_ROCM_ARCHS: list[str] = [
    "gfx950",
    "gfx942",
    "gfx941",
    "gfx940",
    "gfx90a",
    "gfx908",
    "gfx906",
    "gfx900",
]


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """
    Hardware information for cache keys and heuristic selection.

    Attributes:
        device_kind: Device type ('cuda', 'rocm', 'xpu', or 'tpu')
        hardware_name: Device name (e.g., 'NVIDIA H100', 'gfx90a')
        runtime_version: Runtime version (e.g., '12.4', 'gfx90a')
        compute_capability: Compute capability for heuristics (e.g., 'sm90', 'gfx90a')
    """

    device_kind: str
    hardware_name: str
    runtime_version: str
    compute_capability: str

    @property
    def hardware_id(self) -> str:
        """Get a unique identifier string for this hardware."""
        safe_name = self.hardware_name.replace(" ", "_")
        return f"{self.device_kind}_{safe_name}_{self.runtime_version}"

    def get_compatible_compute_ids(self) -> list[str]:
        """
        Get a list of compatible compute IDs for fallback, ordered from current to oldest.

        For CUDA/ROCm, returns the current compute capability followed by all older
        compatible architectures. This allows using heuristics tuned on older hardware
        when newer hardware-specific heuristics aren't available.
        """
        if self.device_kind == "cuda":
            arch_list = _CUDA_COMPUTE_CAPS
        elif self.device_kind == "rocm":
            arch_list = _ROCM_ARCHS
        else:
            return [self.compute_capability]

        try:
            current_idx = arch_list.index(self.compute_capability)
            return arch_list[current_idx:]
        except ValueError:
            return [self.compute_capability, *arch_list]


def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
    """
    Get hardware information for the current or specified device.

    Args:
        device: Optional device to get info for. If None, discovers an accelerator.
            An explicit CPU device prefers a TPU because Pallas uses CPU tensors as
            its bridge, then retains the normal CUDA/ROCm fallback.

    Returns:
        HardwareInfo with device details for caching and heuristic lookup.
    """
    if device is not None:
        from ._argument_device import _canonicalize_argument_device

        device = _canonicalize_argument_device(device)
    return _get_hardware_info(device)


@functools.cache
def _get_hardware_info(device: torch.device | None) -> HardwareInfo:
    # Pallas represents TPU inputs as CPU tensors.  Prefer TPU for that explicit
    # bridge device, but preserve the historical accelerator fallback when no TPU
    # is present.
    prefer_tpu = device is not None and device.type == "cpu"
    if prefer_tpu and (hardware := _get_tpu_hardware_info()) is not None:
        return hardware

    # XPU (Intel) path
    if (
        device is not None
        and device.type == "xpu"
        and getattr(torch, "xpu", None) is not None
        and torch.xpu.is_available()
    ):
        props = torch.xpu.get_device_properties(device)
        return HardwareInfo(
            device_kind="xpu",
            hardware_name=props.name,
            runtime_version=props.driver_version,
            compute_capability=props.name,  # XPU doesn't have compute capability
        )

    # CUDA/ROCm path.  Unsupported or unavailable explicit device kinds retain
    # the historical fallback to the first visible CUDA/ROCm device.
    if torch.cuda.is_available():
        dev = (
            device
            if device is not None and device.type == "cuda"
            else torch.device("cuda:0")
        )
        props = torch.cuda.get_device_properties(dev)

        if torch.version.cuda is not None:
            return HardwareInfo(
                device_kind="cuda",
                hardware_name=props.name,
                runtime_version=str(torch.version.cuda),
                compute_capability=f"sm{props.major}{props.minor}",
            )
        if torch.version.hip is not None:
            return HardwareInfo(
                device_kind="rocm",
                hardware_name=props.gcnArchName,
                runtime_version=torch.version.hip,
                compute_capability=props.gcnArchName,
            )

    if not prefer_tpu and (hardware := _get_tpu_hardware_info()) is not None:
        return hardware

    raise RuntimeError(
        "No supported GPU or TPU device found. Helion requires CUDA, ROCm, XPU, or TPU."
    )


def clear_hardware_info_cache() -> None:
    """Clear cached explicit-device hardware identities."""

    _get_hardware_info.cache_clear()


def _get_tpu_hardware_info() -> HardwareInfo | None:
    """Return the first JAX TPU, when the optional TPU runtime is available."""
    try:
        import jax
    except ImportError:
        return None

    try:
        devices = jax.devices("tpu")
    except (ImportError, OSError, RuntimeError):
        log.debug(
            "JAX TPU discovery failed; continuing accelerator discovery",
            exc_info=True,
        )
        return None

    if devices:
        first_tpu = devices[0]
        return HardwareInfo(
            device_kind="tpu",
            hardware_name=first_tpu.device_kind,
            runtime_version=jax.__version__,
            compute_capability=first_tpu.device_kind,
        )
    return None
