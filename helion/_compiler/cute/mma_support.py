from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class CuteMmaSupport:
    device_name: str | None
    capability: tuple[int, int] | None
    cutlass_arch: str | None
    universal: bool
    warp_f16bf16: bool
    warpgroup_f16bf16: bool
    tcgen05_f16bf16: bool
    tcgen05_f8: bool = False
    warp_error: str | None = None
    warpgroup_error: str | None = None
    tcgen05_error: str | None = None
    tcgen05_f8_error: str | None = None

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
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.cute.nvgpu import warpgroup

        warpgroup.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (64, 8, 16),
            warpgroup.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _probe_tcgen05_f16bf16() -> tuple[bool, str | None]:
    try:
        import cutlass
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.cute.nvgpu import tcgen05

        tcgen05.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (128, 8, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _probe_tcgen05_f8() -> tuple[bool, str | None]:
    try:
        import cutlass
        from cutlass.cute.nvgpu import tcgen05

        # fp8 (e4m3) MMA on tcgen05 uses the F8F6F4 op with MMA-K=32 (vs 16
        # for BF16/FP16) and a separate a_dtype/b_dtype.
        tcgen05.MmaF8F6F4Op(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            (128, 8, 32),
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
            tcgen05_f8=False,
            warp_error="CUDA unavailable",
            warpgroup_error="CUDA unavailable",
            tcgen05_error="CUDA unavailable",
            tcgen05_f8_error="CUDA unavailable",
        )

    device_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    cutlass_arch = _current_cutlass_arch_name()

    # The universal atom is the only lowering Helion currently wires up end-to-end.
    universal = cutlass_arch is not None
    warp_ok, warp_error = _probe_warp_f16bf16()
    warpgroup_ok, warpgroup_error = _probe_warpgroup_f16bf16()
    tcgen05_ok, tcgen05_error = _probe_tcgen05_f16bf16()
    tcgen05_f8_ok, tcgen05_f8_error = _probe_tcgen05_f8()

    return CuteMmaSupport(
        device_name=device_name,
        capability=capability,
        cutlass_arch=cutlass_arch,
        universal=universal,
        warp_f16bf16=warp_ok,
        warpgroup_f16bf16=warpgroup_ok,
        tcgen05_f16bf16=tcgen05_ok,
        tcgen05_f8=tcgen05_f8_ok,
        warp_error=warp_error,
        warpgroup_error=warpgroup_error,
        tcgen05_error=tcgen05_error,
        tcgen05_f8_error=tcgen05_f8_error,
    )
