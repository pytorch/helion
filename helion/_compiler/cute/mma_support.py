from __future__ import annotations

import dataclasses
import os

import torch

_VALID_IMPLS = ("universal", "warp", "warpgroup", "tcgen05")


@dataclasses.dataclass(frozen=True)
class CuteMmaSupport:
    device_name: str | None
    capability: tuple[int, int] | None
    cutlass_arch: str | None
    universal: bool
    warp_f16bf16: bool
    warpgroup_f16bf16: bool
    tcgen05_f16bf16: bool
    warp_error: str | None = None
    warpgroup_error: str | None = None
    tcgen05_error: str | None = None

    @property
    def supported_impls(self) -> tuple[str, ...]:
        impls: list[str] = []
        if self.universal:
            impls.append("universal")
        if self.warp_f16bf16:
            impls.append("warp")
        if self.warpgroup_f16bf16:
            impls.append("warpgroup")
        if self.tcgen05_f16bf16:
            impls.append("tcgen05")
        return tuple(impls)

    @property
    def default_impl(self) -> str | None:
        # Helion's current MMA lowering is built around the universal FMA atom.
        if self.universal:
            return "universal"
        return None


def _current_cuda_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _current_cutlass_arch_name() -> str | None:
    try:
        from cutlass.cutlass_dsl import BaseDSL

        return BaseDSL._get_dsl().get_arch_enum().name
    except Exception:
        return None


def _probe_warp_f16bf16() -> tuple[bool, str | None]:
    try:
        import cutlass
        from cutlass.cute.nvgpu import warp

        warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _probe_warpgroup_f16bf16() -> tuple[bool, str | None]:
    try:
        import cutlass
        from cutlass.cute.nvgpu import warpgroup

        warpgroup.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (64, 8, 16),
            warpgroup.OperandSource.SMEM,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _probe_tcgen05_f16bf16() -> tuple[bool, str | None]:
    try:
        import cutlass
        from cutlass.cute.nvgpu import tcgen05

        tcgen05.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (128, 8, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def get_cute_mma_support() -> CuteMmaSupport:
    device = _current_cuda_device()
    if device is None:
        return CuteMmaSupport(
            device_name=None,
            capability=None,
            cutlass_arch=None,
            universal=False,
            warp_f16bf16=False,
            warpgroup_f16bf16=False,
            tcgen05_f16bf16=False,
            warp_error="CUDA unavailable",
            warpgroup_error="CUDA unavailable",
            tcgen05_error="CUDA unavailable",
        )

    device_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    cutlass_arch = _current_cutlass_arch_name()

    # The universal atom is the only lowering Helion currently wires up end-to-end.
    universal = cutlass_arch is not None
    warp_ok, warp_error = _probe_warp_f16bf16()
    warpgroup_ok, warpgroup_error = _probe_warpgroup_f16bf16()
    tcgen05_ok, tcgen05_error = _probe_tcgen05_f16bf16()

    return CuteMmaSupport(
        device_name=device_name,
        capability=capability,
        cutlass_arch=cutlass_arch,
        universal=universal,
        warp_f16bf16=warp_ok,
        warpgroup_f16bf16=warpgroup_ok,
        tcgen05_f16bf16=tcgen05_ok,
        warp_error=warp_error,
        warpgroup_error=warpgroup_error,
        tcgen05_error=tcgen05_error,
    )


def select_cute_mma_impl() -> str:
    support = get_cute_mma_support()
    override = os.environ.get("HELION_CUTE_MMA_IMPL", "auto").strip().lower()
    if override == "auto":
        default_impl = support.default_impl
        if default_impl is None:
            raise RuntimeError("CuTe MMA probing found no usable implementation")
        return default_impl
    if override not in _VALID_IMPLS:
        raise ValueError(
            f"Invalid HELION_CUTE_MMA_IMPL={override!r}; expected one of {_VALID_IMPLS}"
        )
    if override not in support.supported_impls:
        raise RuntimeError(
            "Requested HELION_CUTE_MMA_IMPL is not supported on this machine: "
            f"{override}. Supported: {support.supported_impls}"
        )
    return override
