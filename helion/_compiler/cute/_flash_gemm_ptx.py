# PTX-path tcgen05 MMA for the fused flash-attention codegen (Stage 2b).
#
# Vendors the FA4 ``gemm_ptx_partial`` (flash_attn/cute/blackwell_helpers.py) +
# the ``mma_sm100_desc`` descriptor helpers it needs, self-contained (no external
# flash-attention path load). Used by the fa4 MMA warp to issue QK/PV via a SINGLE
# inline-asm region with literal-immediate descriptors: this fits the MMA warp at
# 48 regs (the cute.gemm path spills ~116 STL / 133 LDL) AND folds the pfor2
# mbarrier wait INSIDE the PV tcgen05.mma issue stream (vs a Python-level
# mbar_spin_wait that breaks the tensor-core stream between K-chunks).
#
# NOTE: deliberately NO ``from __future__ import annotations`` -- the generated
# Helion module carries it, which would stringify the @cute.jit annotations.
from enum import IntEnum
import re

import cutlass
from cutlass import Boolean
from cutlass import Int32
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute


# ===========================================================================
# Descriptor helpers, vendored from flash_attn/cute/mma_sm100_desc.py (a port of
# CUTLASS mma_sm100_desc.hpp / mma_traits_sm100.hpp). Values MUST match the HW
# encodings.
# ===========================================================================
class Major(IntEnum):
    K = 0
    MN = 1


class ScaleIn(IntEnum):
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


def to_UMMA_format(cutlass_type) -> int:
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    if cutlass_type is cutlass.FloatE4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.FloatE5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(
        f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}"
    )


def make_instr_desc(
    a_type,
    b_type,
    c_type,
    M: int,
    N: int,
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))
    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8...256")
    m_dim = M >> 4
    n_dim = N >> 3
    desc = 0
    desc |= (0 & 0x3) << 0
    desc |= (int(is_sparse) & 0x1) << 2
    desc |= (int(c_sat) & 0x1) << 3
    desc |= (c_fmt & 0x3) << 4
    desc |= (a_fmt & 0x7) << 7
    desc |= (b_fmt & 0x7) << 10
    desc |= (int(a_neg) & 0x1) << 13
    desc |= (int(b_neg) & 0x1) << 14
    desc |= (int(a_major) & 0x1) << 15
    desc |= (int(b_major) & 0x1) << 16
    desc |= (n_dim & 0x3F) << 17
    desc |= (m_dim & 0x1F) << 24
    desc |= (int(max_shift) & 0x3) << 30
    return desc & 0xFFFF_FFFF


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp) -> int:
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K
        if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else Major.MN,
        Major.K
        if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else Major.MN,
    )


def _mma_cta_group_qualifier(cta_group: cute.nvgpu.tcgen05.CtaGroup | int) -> str:
    value = (
        cta_group.value
        if isinstance(cta_group, cute.nvgpu.tcgen05.CtaGroup)
        else cta_group
    )
    if value == 1:
        return "cta_group::1"
    if value == 2:
        return "cta_group::2"
    raise ValueError(f"Unsupported tcgen05 MMA CTA group: {cta_group!r}")


class LayoutType(IntEnum):
    SWIZZLE_NONE = 0
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    swz_str = str(swizzle)
    inside = swz_str[swz_str.index("<") + 1 : swz_str.index(">")]
    b, m, s = (int(x) for x in inside.split(","))
    if m == 4:
        if s != 3:
            raise ValueError("Unexpected swizzle shift - want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[b]
    if m == 5:
        if (b, s) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B
    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(
    layout: cute.Layout, swizzle: cute.Swizzle, major: Major
) -> int:
    layout_type = _layout_type(swizzle)
    VERSION = 1
    LBO_MODE = 0
    BASE_OFFSET = 0
    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]
    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(
            layout, (swizzle_atom_mn_size, swizzle_atom_k_size)
        )
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError(
                "Not a canonical UMMA_MN Layout: Expected profile failure."
            )
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = (
            canonical_layout.stride[0][1],
            canonical_layout.stride[1][1],
        )
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError(
                "Not a canonical UMMA_K Layout: Expected MN-size multiple of 8."
            )
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10
    desc = 0
    desc |= (leading_byte_offset & 0x3FFF) << 16
    desc |= (stride_byte_offset & 0x3FFF) << 32
    desc |= (VERSION & 0x3) << 46
    desc |= (BASE_OFFSET & 0x7) << 49
    desc |= (LBO_MODE & 0x1) << 52
    desc |= (int(layout_type) & 0x7) << 61
    return desc & 0xFFFF_FFFF_FFFF_FFFF


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    return (start_addr.toint() & 0x3FFFF) >> 4


# ===========================================================================
# parse_swizzle_from_pointer + i64_to_i32x2, vendored from flash_attn/cute/utils.py
# and blackwell_helpers.py.
# ===========================================================================
def parse_swizzle_from_pointer(ptr: cute.Pointer) -> cute.Swizzle:
    swizzle_str = str(ptr.type.swizzle_type)
    match = re.search(r"S<(\d+),(\d+),(\d+)>", swizzle_str)
    if match:
        b, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return cute.make_swizzle(b, m, s)
    raise ValueError(f"Could not parse swizzle_type: {swizzle_str}")


def smem_desc_base_from_tensor(sA: cute.Tensor, major: Major) -> int:
    return make_smem_desc_base(
        cute.recast_layout(128, sA.element_type.width, sA.layout[0]),
        parse_swizzle_from_pointer(sA.iterator),
        major,
    )


def i64_to_i32x2(i: int) -> tuple[int, int]:
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


def _not_zero_init(zero_init):
    """Negate zero_init for the ``Int32(not zero_init)`` MMA-enable input. Python
    ``not`` raises on a dynamic cutlass.Boolean; a Boolean must be negated with ~
    inside a kernel region. This is the ONE behavioral fix vs the verbatim FA4
    source, needed so a LOOP-CARRIED Boolean zero_init works."""
    if isinstance(zero_init, Boolean):
        return ~zero_init
    return not zero_init


# ===========================================================================
# gemm_ptx_partial, vendored VERBATIM from flash_attn/cute/blackwell_helpers.py
# (the only change vs source: ``sm100_desc`` calls are now the module-local
# vendored helpers above, and ``not zero_init`` -> ``_not_zero_init(zero_init)``).
# ===========================================================================
@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start_a: Int32,
    smem_desc_base_a: int | None,
    tCrA_layout: cute.Layout,
    var_name_prefix: str = "smem_desc",
) -> None:
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    smem_desc_base_a_lo, smem_desc_a_hi = None, None
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
        llvm.inline_asm(
            None,
            [Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value()],
            f".reg .b32 {var_name_prefix}_lo;\n\t"
            f".reg .b64 {var_name_prefix}_<{num_k_tile}>;\n\t"
            f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_a_hi)}}};\n\t"
            + "".join(
                (
                    f"add.s32 {var_name_prefix}_lo, $0, {hex(offset_a[k])};\n\t"
                    f"mov.b64 {var_name_prefix}_{k}, "
                    f"{{{var_name_prefix}_lo, {hex(smem_desc_a_hi)}}};\n\t"
                )
                for k in range(1, num_k_tile)
            ),
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_smem_desc_from_tensor(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    sA: cute.Tensor,
    tCrA: cute.Tensor,
    stage: int,
    var_name_prefix: str,
) -> None:
    sA_swizzle = parse_swizzle_from_pointer(sA.iterator)
    smem_desc_base_a: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.a_dtype.width, sA.layout[0]),
            sA_swizzle,
            Major.K
            if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    declare_ptx_smem_desc(
        make_smem_desc_start_addr(sA[None, None, None, stage].iterator),
        smem_desc_base_a,
        tCrA.layout,
        var_name_prefix,
    )


@cute.jit
def declare_ptx_idesc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    var_name: str = "idesc",
) -> None:
    idesc: int = const_expr(mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\tmov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_qk(
    acc_tmem_addr: Int32,
    smem_desc_start_b: Int32,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    smem_offset: int,
    zero_init: bool | Boolean = False,
    cta_group: cute.nvgpu.tcgen05.CtaGroup | int = 1,
) -> None:
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    num_k_tile = cute.size(tCrB_layout.shape[2])
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    zero_init_dynamic = const_expr(isinstance(zero_init, Boolean))
    pred_str = "p" if zero_init_dynamic else "0" if zero_init else "1"
    cta_group_qualifier = const_expr(_mma_cta_group_qualifier(cta_group))
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(_not_zero_init(zero_init)).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 tmem_acc;\n\t"
        ".reg .b32 smem_desc_b_lo_start;\n\t"
        ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
        ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
        f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b32 tmem_acc, $2;\n\t"
        "mov.b32 smem_desc_b_lo_start, $0;\n\t"
        f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
        f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_0;\n\t"
        f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
        f"mov.b64 {smem_var_name_prefix}_0, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
        f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
        + "".join(
            (
                f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, "
                f"{smem_var_name_prefix}_{k};\n\t"
                f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                f"mov.b64 {smem_var_name_prefix}_{k}, "
                "{smem_desc_a_lo, smem_desc_a_hi};\n\t"
                f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            )
            for k in range(1, num_k_tile)
        )
        + "setp.ne.b32 p, $1, 0;\n\t"
        f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
        f"[tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, "
        f"{idesc_var_name}, {pred_str};\n\t"
        + "".join(
            (
                f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
                f"[tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, "
                f"{idesc_var_name}, 1;\n\t"
            )
            for k in range(1, num_k_tile)
        )
        + "}\n",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_qk_static(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrB: cute.Tensor,
    sB: cute.Tensor,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    zero_init: bool | Boolean = False,
    cta_group: cute.nvgpu.tcgen05.CtaGroup | int | None = None,
) -> None:
    """QK MMA with per-Q-stage descriptors declared up front.

    ``gemm_ptx_precomputed_qk`` keeps one descriptor set and mutates its low
    bits by a stage stride before every MMA issue. For the FA4 two-Q-stage flash
    body the Q stage is compile-time known at each call site, so this variant
    consumes the already-correct descriptor registers directly.
    """
    sB_swizzle = parse_swizzle_from_pointer(sB.iterator)
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB.layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    num_k_tile = cute.size(tCrB.shape[2])
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(num_k_tile)]
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    zero_init_dynamic = const_expr(isinstance(zero_init, Boolean))
    pred_str = "p" if zero_init_dynamic else "0" if zero_init else "1"
    mma_cta_group = op.cta_group if const_expr(cta_group is None) else cta_group
    cta_group_qualifier = const_expr(_mma_cta_group_qualifier(mma_cta_group))
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(_not_zero_init(zero_init)).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 tmem_acc;\n\t"
        ".reg .b32 smem_desc_b_lo_start;\n\t"
        ".reg .b32 smem_desc_b_lo;\n\t"
        f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b32 tmem_acc, $2;\n\t"
        "mov.b32 smem_desc_b_lo_start, $0;\n\t"
        f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, {hex(smem_desc_b_hi)}}};\n\t"
        + "".join(
            (
                f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, {hex(smem_desc_b_hi)}}};\n\t"
            )
            for k in range(1, num_k_tile)
        )
        + "setp.ne.b32 p, $1, 0;\n\t"
        f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
        f"[tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, "
        f"{idesc_var_name}, {pred_str};\n\t"
        + "".join(
            (
                f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
                f"[tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, "
                f"{idesc_var_name}, 1;\n\t"
            )
            for k in range(1, num_k_tile)
        )
        + "}\n",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_pv_ts(
    acc_tmem_addr: Int32,
    tmem_a_addr: Int32,
    smem_desc_start_b: Int32,
    smem_desc_base_b: int,
    tCrA_layout: cute.Layout,
    tCrB_layout: cute.Layout,
    idesc_var_name: str,
    mbar_ptr: cutlass.Pointer | None = None,
    mbar_phase: Int32 | None = None,
    zero_init: bool | Boolean = False,
    cta_group: cute.nvgpu.tcgen05.CtaGroup | int = 1,
) -> None:
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    tCrA_layout = cute.recast_layout(32, 16, tCrA_layout)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, num_k_tile)]
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    zero_init_dynamic = const_expr(isinstance(zero_init, Boolean))
    pred_str = "p" if zero_init_dynamic else "0" if zero_init else "1"
    cta_group_qualifier = const_expr(_mma_cta_group_qualifier(cta_group))

    input_args = [
        Int32(cute.arch.make_warp_uniform(tmem_a_addr)).ir_value(),
        Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
        Int32(_not_zero_init(zero_init)).ir_value(),
        Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
    ]
    if const_expr(mbar_ptr is not None):
        assert mbar_phase is not None, (
            "mbar_phase must be provided when mbar_ptr is not None"
        )
        input_args.append(mbar_ptr.toint().ir_value())
        input_args.append(Int32(mbar_phase).ir_value())
        mbar_wait_str = (
            ".reg .pred P1; \n\t"
            "LAB_WAIT: \n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
            "@P1 bra DONE; \n\t"
            "bra     LAB_WAIT; \n\t"
            "DONE: \n\t"
        )
    else:
        mbar_wait_str = ""
    llvm.inline_asm(
        None,
        input_args,
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 tmem_acc;\n\t"
        ".reg .b32 tmem_a;\n\t"
        ".reg .b32 smem_desc_b_lo_start;\n\t"
        ".reg .b32 smem_desc_b_lo;\n\t"
        ".reg .b32 smem_desc_b_hi;\n\t"
        ".reg .b64 smem_desc_b;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b32 tmem_acc, $3;\n\t"
        "mov.b32 tmem_a, $0;\n\t"
        "mov.b32 smem_desc_b_lo_start, $1;\n\t"
        f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
        "mov.b64 smem_desc_b, {smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
        "setp.ne.b32 p, $2, 0;\n\t"
        f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
        f"[tmem_acc], [tmem_a], smem_desc_b, {idesc_var_name}, {pred_str};\n\t"
        + "".join(
            (
                f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                "mov.b64 smem_desc_b, {smem_desc_b_lo, smem_desc_b_hi};\n\t"
                f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
                f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], "
                f"smem_desc_b, {idesc_var_name}, 1;\n\t"
            )
            for k in range(
                1,
                num_k_tile if const_expr(mbar_ptr is None) else num_k_tile // 4 * 3,
            )
        )
        + mbar_wait_str
        + (
            "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, "
                    f"{hex(offset_b_diff[k - 1])};\n\t"
                    "mov.b64 smem_desc_b, {smem_desc_b_lo, smem_desc_b_hi};\n\t"
                    f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 "
                    f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], "
                    f"smem_desc_b, {idesc_var_name}, 1;\n\t"
                )
                for k in range(num_k_tile // 4 * 3, num_k_tile)
            )
            if const_expr(mbar_ptr is not None)
            else ""
        )
        + "}\n",
        "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor | None,
    sB: cute.Tensor,
    mbar_ptr: cutlass.Pointer | None = None,
    mbar_phase: Int32 | None = None,
    zero_init: bool | Boolean = False,
    tA_addr: Int32 | None = None,
    cta_group: cute.nvgpu.tcgen05.CtaGroup | int | None = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(mma_op_to_idesc(op))
    mma_cta_group = op.cta_group if const_expr(cta_group is None) else cta_group
    cta_group_qualifier = const_expr(_mma_cta_group_qualifier(mma_cta_group))
    if const_expr(not is_ts):
        sA_swizzle = parse_swizzle_from_pointer(sA.iterator)
        smem_desc_base_a: int = const_expr(
            make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                Major.K
                if const_expr(
                    op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = parse_swizzle_from_pointer(sB.iterator)
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [
        cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(_not_zero_init(zero_init)).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "mov.b64 smem_desc_a, {smem_desc_a_lo_start, smem_desc_a_hi};\n\t"
            "mov.b64 smem_desc_b, {smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    "mov.b64 smem_desc_a, {smem_desc_a_lo, smem_desc_a_hi};\n\t"
                    "mov.b64 smem_desc_b, {smem_desc_b_lo, smem_desc_b_hi};\n\t"
                    f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, tCrA.iterator.toint() returns 0 no matter what, so the
        # caller passes tA_addr explicitly for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(_not_zero_init(zero_init)).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, (
                "mbar_phase must be provided when mbar_ptr is not None"
            )
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 tmem_a, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "mov.b64 smem_desc_b, {smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    "mov.b64 smem_desc_b, {smem_desc_b_lo, smem_desc_b_hi};\n\t"
                    f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2])
                    if const_expr(mbar_ptr is None)
                    else cute.size(tCrA.shape[2]) // 4 * 3,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        "mov.b64 smem_desc_b, {smem_desc_b_lo, smem_desc_b_hi};\n\t"
                        f"@leader_thread tcgen05.mma.{cta_group_qualifier}.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(
                        cute.size(tCrA.shape[2]) // 4 * 3, cute.size(tCrA.shape[2])
                    )
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
