from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class CuteMmaSupport:
    universal: bool
    warp_f16bf16: bool
    warpgroup_f16bf16: bool
    tcgen05_f16bf16: bool
    tcgen05_f8: bool = False
    tcgen05_tf32: bool = False

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


def _probe_tcgen05_tf32() -> bool:
    try:
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.cute.nvgpu import tcgen05

        # fp32 operands run on tcgen05 as tf32 with MMA-K=8 (256 bits of K per
        # instruction / 32-bit operands). The op takes no a/acc dtype args:
        # tf32 in, f32 accumulate is the only shape.
        tcgen05.MmaTF32Op(
            (128, 8, 8),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        return True
    except Exception:
        return False


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
            universal=False,
            warp_f16bf16=False,
            warpgroup_f16bf16=False,
            tcgen05_f16bf16=False,
            tcgen05_f8=False,
            tcgen05_tf32=False,
        )

    cutlass_arch = _current_cutlass_arch_name()

    # The universal atom is the only lowering Helion currently wires up end-to-end.
    universal = cutlass_arch is not None
    tcgen05_f8_ok, _ = _probe_tcgen05_f8()
    return CuteMmaSupport(
        universal=universal,
        warp_f16bf16=_probe_warp_f16bf16(),
        warpgroup_f16bf16=_probe_warpgroup_f16bf16(),
        tcgen05_f16bf16=_probe_tcgen05_f16bf16(),
        tcgen05_f8=tcgen05_f8_ok,
        tcgen05_tf32=_probe_tcgen05_tf32(),
    )


def tcgen05_supports_input_dtype(
    support: CuteMmaSupport, input_dtype: torch.dtype
) -> bool:
    """Per-dtype tcgen05 capability lookup.

    A free function (not a CuteMmaSupport method) so callers keep working with
    the duck-typed SimpleNamespace doubles tests patch in for
    ``get_cute_mma_support``; those predate the tf32 field, hence the getattr
    default.
    """
    if input_dtype == torch.float8_e4m3fn:
        return support.tcgen05_f8
    if input_dtype == torch.float32:
        return getattr(support, "tcgen05_tf32", False)
    return support.tcgen05_f16bf16


def cute_fp32_dot_uses_tf32() -> bool:
    """Whether fp32 matmul operands may be computed as tf32 on tensor cores.

    Helion's default ``settings.dot_precision`` maps to ``"tf32"`` (matching
    triton's ``tl.dot`` default), which permits the tcgen05 tf32 MMA path for
    fp32 inputs. ``"ieee"``/``"tf32x3"`` keep fp32 matmuls on the exact SIMT /
    universal-FMA lowerings.
    """
    from ..compile_environment import CompileEnvironment

    if not CompileEnvironment.has_current():
        return False
    env = CompileEnvironment.current()
    return env.backend.map_dot_precision(env.settings.dot_precision) == "tf32"
