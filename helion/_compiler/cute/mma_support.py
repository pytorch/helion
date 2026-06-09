from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class CuteMmaSupport:
    universal: bool
    warp_f16bf16: bool
    warpgroup_f16bf16: bool
    tcgen05_f16bf16: bool

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


def _probe_warp_f16bf16() -> bool:
    try:
        import cutlass
        from cutlass.cute.nvgpu import warp

        warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
        return True
    except Exception:
        return False


def _probe_warpgroup_f16bf16() -> bool:
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
        return True
    except Exception:
        return False


def _probe_tcgen05_f16bf16() -> bool:
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
        return True
    except Exception:
        return False


def get_cute_mma_support() -> CuteMmaSupport:
    device = _current_cuda_device()
    if device is None:
        return CuteMmaSupport(
            universal=False,
            warp_f16bf16=False,
            warpgroup_f16bf16=False,
            tcgen05_f16bf16=False,
        )

    cutlass_arch = _current_cutlass_arch_name()

    # The universal atom is the only lowering Helion currently wires up end-to-end.
    universal = cutlass_arch is not None
    return CuteMmaSupport(
        universal=universal,
        warp_f16bf16=_probe_warp_f16bf16(),
        warpgroup_f16bf16=_probe_warpgroup_f16bf16(),
        tcgen05_f16bf16=_probe_tcgen05_f16bf16(),
    )
