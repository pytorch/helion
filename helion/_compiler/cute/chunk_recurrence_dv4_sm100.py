# ruff: noqa: ANN001, ANN202, C408, RUF005
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""SM100 warp-MMA DV4 schedule for the v2 workspace layout."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.utils
import torch

from .kda_device_primitives import ldmatrix_x2
from .kda_device_primitives import ldmatrix_x2_trans
from .kda_device_primitives import ldmatrix_x4_trans
from .kda_device_primitives import mma_m16n8k16_bf16
from .kda_device_primitives import movmatrix_b16
from .kda_device_primitives import pack_bf16x2
from .kda_device_primitives import stmatrix_x2
from .kda_device_primitives import stmatrix_x2_trans
from .kda_device_primitives import store_vec8_bf16
from .kda_device_primitives import tma_load_3d
from .kda_device_primitives import tma_store_3d
from .kda_device_primitives import tma_store_commit_group
from .kda_device_primitives import tma_store_wait_read
from .kda_device_primitives import vec8_bf16
from .kda_device_primitives import warp_arrive


def _upload_tensor_map_bytes(
    payload: bytes | bytearray, device: torch.device
) -> torch.Tensor:
    """Upload descriptor bytes before graph capture; the launcher owns reuse."""

    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().to(device)


BT = 16
DK = 128
DV = 128
RECURRENCE_DV_PARTITIONS = 4
DV_HALF = DV // RECURRENCE_DV_PARTITIONS
BF16_SEGMENT_ELEMS = 64
BF16_GROUP_ELEMS = 8
BF16_SEGMENTS = DK // BF16_SEGMENT_ELEMS
BF16_SEGMENT_STRIDE = BT * BF16_SEGMENT_ELEMS  # 1024 elements
BF16_ROW_XOR_MASK = BF16_SEGMENT_ELEMS // BF16_GROUP_ELEMS - 1  # 7
KR_AK_TOKEN_XOR = BT // 2  # 8, token-row permutation for the Ak.T image
FACTOR_KEY_XOR = 8
PAIRWISE_COL_XOR = 8  # column permutation of the 16x16 pairwise image
PAIRWISE_ROW_STRIDE = 16


def raw_bf16_s128(token, dim):
    """Physical BF16 element index of logical ``(token, dim)``.

    Used for raw Q, raw K, ``Ki``, ``Kd`` and ``Qd``.  ``Kd``/``Qd`` carry no
    feature permutation, so they share this image with the
    raw stages and with ``Ki``.
    """
    segment = dim // BF16_SEGMENT_ELEMS
    local = dim - segment * BF16_SEGMENT_ELEMS
    group = local // BF16_GROUP_ELEMS
    inner = local - group * BF16_GROUP_ELEMS
    return (
        segment * BF16_SEGMENT_STRIDE
        + token * BF16_SEGMENT_ELEMS
        + (group ^ (token & BF16_ROW_XOR_MASK)) * BF16_GROUP_ELEMS
        + inner
    )


def pairwise_sw32(row, col):
    """Physical BF16 element index of a 16x16 pairwise tile element."""
    storage_col = col ^ PAIRWISE_COL_XOR
    byte_offset = 2 * (row * PAIRWISE_ROW_STRIDE + storage_col)
    return (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2


factor_idx = raw_bf16_s128
VO_ROW_ELEMS = DV_HALF
VO_STAGE_ELEMS = BT * VO_ROW_ELEMS
VO_SWIZZLE = "128B" if DV_HALF == 64 else "NONE"
VO_VECTORS_PER_ROW = DV_HALF // 8
OUTPUT_TAIL_TASKS = BT * VO_VECTORS_PER_ROW
OUTPUT_TAIL_ROUNDS = (OUTPUT_TAIL_TASKS + 31) // 32
STATE_BF16_ROWS_PER_VALUE = DK // BF16_SEGMENT_ELEMS  # 2


def _s128_index(row, column, segment_elems, group_elems):
    """S128 element index of ``column`` within 128-byte row ``row``.

    ``row`` here is the *segment* row -- the unit the swizzle XOR keys on -- not
    a logical matrix row.  The three recurrence images differ only in what they
    call a segment row and how they split a logical coordinate into one.
    """
    group = column // group_elems
    inner = column - group * group_elems
    return (
        row * segment_elems
        + (group ^ (row & (segment_elems // group_elems - 1))) * group_elems
        + inner
    )


def vo_idx(row, v_local):
    """Physical BF16 index of ``(token row, value)`` in a V/output half stage.

    ``v_local`` is in ``[0, 64)`` and is the CTA-local value, so the two DV
    halves address byte-identical stages at different global coordinates.
    """
    if VO_ROW_ELEMS == 32:
        return row * VO_ROW_ELEMS + v_local
    return _s128_index(row, v_local, VO_ROW_ELEMS, BF16_GROUP_ELEMS)


def state_bf16_idx(v_local, k):
    """Physical BF16 index of logical ``H[k][v]`` in the persistent state.

    The unswizzled address function is ``state_idx(k, v) = v * 128 + k`` (plan
    Section 12.2 Q1): physical ``[V, K]`` row-major *and* logical ``[K, V]``
    column-major at once, which is what lets an external ``[V, K]`` state half
    land by TMA with no transpose and still be read as ``H[K, V]`` by the MMA.
    """
    segment = k // BF16_SEGMENT_ELEMS
    local = k - segment * BF16_SEGMENT_ELEMS
    line = STATE_BF16_ROWS_PER_VALUE * v_local + segment
    return _s128_index(line, local, BF16_SEGMENT_ELEMS, BF16_GROUP_ELEMS)


def vo_global_index(token, head, heads, dv_half, v_local):
    """Flat element index of ``out[0, token, head, 64 * dv_half + v_local]``.

    The partial-output tail stores through this map with 16-byte vectors (plan
    Section 8.2); the full path never needs it, because TMA addresses the same
    element through the descriptor instead.
    """
    return (token * heads + head) * DV + dv_half * DV_HALF + v_local


WARP_VALUES = 8
COMPUTE_WARPS = DV_HALF // WARP_VALUES


def state_x2_ptr(lane, kb, v_base):
    """SMEM index lane ``lane`` addresses for a state 16x8 ``ldmatrix.x2``.

    The same 16 addresses serve both state reads .2: without
    ``.trans`` they produce the MMA **B** operand of ``Kd @ H`` and ``Qd @ H``,
    and with ``.trans`` they produce the **C** view the state update needs.  The
    two passes therefore differ only in one instruction modifier, which is why
    the second pass can reload from SMEM instead of keeping eight fragments
    live.  Lanes 16-31 are ignored by an x2 copy.
    """
    matrix = (lane // 8) & 1
    row = lane - (lane // 8) * 8
    return state_bf16_idx(v_base + row, kb * BT + 8 * matrix)


def vo_x2_ptr(lane, v_base):
    """SMEM index lane ``lane`` addresses for a V/output 16x8 ``ldmatrix.x2``.

    Non-transposed in both directions: the stage is token-major and the C tile's
    rows are tokens, so the loaded V and the stored output share this map.
    """
    matrix = (lane // 8) & 1
    row = (lane - (lane // 8) * 8) + 8 * matrix
    return vo_idx(row, v_base)


def factor_a_fragment_ptr(lane, kb):
    """SMEM index lane ``lane`` addresses for a ``Kd``/``Qd`` A-operand x4 load.

    Both stages are token-major, which is the A operand's orientation already,
    so this is the plain (non-transposed) x4 map at key block ``kb``.
    """
    matrix_id = lane // 8
    row = (lane - matrix_id * 8) + 8 * (matrix_id - (matrix_id // 2) * 2)
    # The unscaled workspace layout publishes Kd/Qd at storage key
    # ``logical_key ^ 8``. Applying that permutation to the ldmatrix base
    # recovers the logical A fragment without moving the staged tile.
    col = (kb * BT + 8 * (matrix_id // 2)) ^ FACTOR_KEY_XOR
    return factor_idx(row, col)


def pairwise_a_fragment_ptr(lane):
    """SMEM index lane ``lane`` addresses for the ``Aq`` A-operand x4 load."""
    matrix_id = lane // 8
    row = (lane - matrix_id * 8) + 8 * (matrix_id - (matrix_id // 2) * 2)
    col = 8 * (matrix_id // 2)
    return pairwise_sw32(row, col)


def ak_a_fragment_ptr(lane, kb):
    """SMEM index lane ``lane`` addresses for the ``Ak`` A-operand x4 trans.

    ``Ak`` is published by prepare as ``Ak.T`` with a ``token ^ 8`` row
    permutation, so the stage holds ``[token][key]`` while the MMA wants
    ``[key][token]``.  ``ldmatrix.x4.trans`` supplies the transpose, and the
    four returned registers are ``(a0, a1, a2, a3)`` directly -- plan
    Section 7.1 forbids permuting them afterwards.
    """
    matrix_id = lane // 8
    row8 = lane - matrix_id * 8
    logical_j = (matrix_id // 2) * 8 + row8
    key = kb * BT + (matrix_id - (matrix_id // 2) * 2) * 8
    return factor_idx(logical_j ^ KR_AK_TOKEN_XOR, key)


_LLVM_STRUCT_TYPE = cast("Any", llvm).StructType


@dsl_user_op
def ldmatrix_x4(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16`` -> four b32 registers."""
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        _LLVM_STRUCT_TYPE.get_literal([_T.IntegerType.get_signless(32)] * 4),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            llvm.extractvalue(
                _T.IntegerType.get_signless(32), struct, [i], loc=loc, ip=ip
            )
        )
        for i in range(4)
    )


@dsl_user_op
def sub_bf16x2(a, b, *, loc=None, ip=None):
    """Packed BF16 subtract of two b32 registers, rounding each difference.

    the design builds the residual with this rather than with
    FP32 arithmetic: both operands are already BF16, so the exact difference is
    representable and a single ``sub.rn`` reproduces the contract's
    ``BF16(V - X)`` boundary for two values at once.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "sub.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unpack_bf16x2(value, *, loc=None, ip=None):
    """Widen a packed BF16 pair to two FP32, low half first.

    The state's decay term is ``FP32(GTotal) * FP32(H_bf16)``,
    so the reloaded BF16 state has to reach the FP32 accumulator; this is the
    widening, and it recovers nothing the entry rounding already discarded.
    """
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        _LLVM_STRUCT_TYPE.get_literal([_T.F32Type.get()] * 2),
        [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi;"
        "  mov.b32 {lo, hi}, $2;"
        "  cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(
            llvm.extractvalue(_T.F32Type.get(), struct, [i], loc=loc, ip=ip)
        )
        for i in range(2)
    )


@dsl_user_op
def fence_tensormap_acquire(desc_addr, *, loc=None, ip=None):
    """Publish a host-written tensor map to the tensormap proxy.

    The descriptor is written by the host and read by the TMA unit through a
    different proxy, so the kernel has to acquire it before first use.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip)],
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


DESCRIPTOR_BYTES = 128
SWIZZLES = ("128B", "64B", "NONE")


def encode_tensor_map(
    dtype: torch.dtype,
    base_ptr: int,
    global_dim,
    strides_bytes,
    box_dim,
    *,
    swizzle: str = "128B",
) -> bytes:
    """Encode one ``cuTensorMapEncodeTiled`` descriptor as 128 raw bytes.

    Shared by prepare and the recurrence.  Every map fixes ``interleave=NONE``,
    unit element strides, 128-byte L2 promotion and ``OOB_FILL_NONE``; only the
    dtype, geometry and swizzle vary.
    """
    drv = importlib.import_module("cuda.bindings.driver")

    if dtype is torch.bfloat16:
        tma_dtype = drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    elif dtype is torch.float32:
        tma_dtype = drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
    else:
        raise ValueError(f"unsupported TMA element type {dtype}")
    if swizzle not in SWIZZLES:
        raise ValueError(f"swizzle must be one of {SWIZZLES}, got {swizzle!r}")

    if swizzle == "128B":
        swizzle_enum = drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    elif swizzle == "64B":
        swizzle_enum = drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B
    else:
        swizzle_enum = drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE

    rank = len(global_dim)
    err, tmap = drv.cuTensorMapEncodeTiled(
        tma_dtype,
        rank,
        base_ptr,
        [drv.cuuint64_t(d) for d in global_dim],
        [drv.cuuint64_t(s) for s in strides_bytes],
        [drv.cuuint32_t(b) for b in box_dim],
        [drv.cuuint32_t(1)] * rank,
        drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_enum,
        drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    # cuda-python wraps the descriptor, so take its address via getPtr().
    return bytes(ctypes.string_at(tmap.getPtr(), DESCRIPTOR_BYTES))


@dataclass(frozen=True)
class TensorMapSpec:
    """Everything ``cuTensorMapEncodeTiled`` is given, and nothing else.

    Two roles that produce equal specs are the same descriptor.  ``role`` is
    deliberately absent from the comparison; it is carried alongside.
    """

    dtype: torch.dtype
    base_ptr: int
    global_dim: tuple[int, ...]
    global_stride_bytes: tuple[int, ...]
    box_dim: tuple[int, ...]
    swizzle: str

    def validate(self) -> None:
        """Check the alignment rules a wrong descriptor would otherwise hide."""
        element_bytes = 2 if self.dtype is torch.bfloat16 else 4
        if self.base_ptr % 16:
            raise ValueError(
                f"TMA global base must be 16-byte aligned, got {self.base_ptr}"
            )
        inner_bytes = self.box_dim[0] * element_bytes
        limit = {"128B": 128, "64B": 64, "NONE": 512}[self.swizzle]
        if inner_bytes % 16 or inner_bytes > limit:
            raise ValueError(
                f"TMA inner box is {inner_bytes} B; must be a multiple of 16 "
                f"and at most {limit} for swizzle {self.swizzle}"
            )
        for extent in self.box_dim:
            if not 1 <= extent <= 256:
                raise ValueError(f"TMA box extent {extent} outside [1, 256]")
        for stride in self.global_stride_bytes:
            if stride % 16:
                raise ValueError(f"TMA global stride {stride} is not 16-byte aligned")
        for dim, box in zip(self.global_dim, self.box_dim, strict=True):
            if dim <= 0:
                raise ValueError(f"TMA global dim {dim} must be positive")
            if box > dim and box != self.box_dim[1]:
                # A box may exceed a degenerate outer dim only through the row
                # mode, which the tail path relies on; the inner and plane modes
                # must fit.
                raise ValueError(f"TMA box {box} exceeds global dim {dim}")

    def encode(self) -> bytes:
        return encode_tensor_map(
            self.dtype,
            self.base_ptr,
            self.global_dim,
            self.global_stride_bytes,
            self.box_dim,
            swizzle=self.swizzle,
        )


def _spec_fields(geometry: dict) -> dict:
    """Rename :func:`factor_slab_geometry`'s keys for :class:`TensorMapSpec`.

    The encoder takes ``strides_bytes``; the spec dataclass calls the same
    field ``global_stride_bytes``.  One adapter beats two copies of the tuple.
    """
    return dict(
        global_dim=geometry["global_dim"],
        global_stride_bytes=geometry["strides_bytes"],
        box_dim=geometry["box_dim"],
    )


def factor_slab_geometry(rows: int, heads: int, element_bytes: int = 2) -> dict:
    """The one description of the fused Kd/Qd/Ak slab, as ``(DK, rows, 3H)``.

    The three regions are the same geometry, the same dtype and exactly
    adjacent -- ``prepare_region_offsets`` aligns each to 256 bytes and every
    region size is a multiple of it, so no padding is inserted -- which makes
    them 3H planes of one tensor rather than three tensors.  Plane ``r*H + h``
    is region ``r`` (0=Kd, 1=Qd, 2=Ak) of head ``h``.

    prepare writes through this and the recurrence reads through it, and the
    fused entry passes prepare's encoded descriptor to both kernels, so the two
    sides cannot be allowed to disagree about it.  Defining it once is what
    makes that structural rather than a comment asking for care.
    """
    return dict(
        global_dim=(DK, rows, 3 * heads),
        strides_bytes=(DK * element_bytes, DK * rows * element_bytes),
        box_dim=(BF16_SEGMENT_ELEMS, BT, 1),
    )


ROLES = ("factor", "aq", "gt", "v", "out", "state_in", "state_out")


def recurrence_tensor_map_specs(
    *,
    kd_ptr: int,
    aq_ptr: int,
    gt_ptr: int,
    v_ptr: int,
    out_ptr: int,
    heads: int,
    total_tokens: int,
    total_chunks: int,
    sequences: int,
    state_ptr: int,
) -> dict[str, TensorMapSpec]:
    """Build the exact v2-workspace TensorMap specifications as plain data."""
    rows = BT * total_chunks
    specs: dict[str, TensorMapSpec] = {
        # Kd/Qd/Ak fused: 3H planes of R rows of 128 BF16 keys.  The geometry
        # is prepare's -- literally the same function -- because the fused
        # entry gives prepare's encoded descriptor to this kernel as well.
        "factor": TensorMapSpec(
            dtype=torch.bfloat16,
            base_ptr=kd_ptr,
            swizzle="128B",
            **_spec_fields(factor_slab_geometry(rows, heads)),
        ),
        # The 16x16 pairwise record prepare already wrote in its SW32 image:
        # moved verbatim, so no second swizzle is applied here.
        "aq": TensorMapSpec(
            dtype=torch.bfloat16,
            base_ptr=aq_ptr,
            global_dim=(BT * BT, total_chunks, heads),
            global_stride_bytes=(BT * BT * 2, total_chunks * BT * BT * 2),
            box_dim=(BT * BT, 1, 1),
            swizzle="NONE",
        ),
        "gt": TensorMapSpec(
            dtype=torch.float32,
            base_ptr=gt_ptr,
            global_dim=(DK, total_chunks, heads),
            global_stride_bytes=(DK * 4, total_chunks * DK * 4),
            box_dim=(DK, 1, 1),
            swizzle="NONE",
        ),
    }
    # V and out are the same geometry over [1, T, H, 128]; when they are also
    # the same storage they collapse to one descriptor by equality.
    activation = dict(
        dtype=torch.bfloat16,
        global_dim=(DV, total_tokens, heads),
        global_stride_bytes=(heads * DV * 2, DV * 2),
        box_dim=(DV_HALF, BT, 1),
        swizzle=VO_SWIZZLE,
    )
    specs["v"] = TensorMapSpec(base_ptr=v_ptr, **activation)
    specs["out"] = TensorMapSpec(base_ptr=out_ptr, **activation)

    state = _state_spec_fields(sequences * heads)
    specs["state_in"] = TensorMapSpec(base_ptr=state_ptr, **state)
    specs["state_out"] = TensorMapSpec(base_ptr=state_ptr, **state)

    for spec in specs.values():
        spec.validate()
    return specs


def _state_spec_fields(planes: int) -> dict:
    """Describe the exact in-place BF16 state consumed by this lowering."""

    rows = STATE_BF16_ROWS_PER_VALUE * DV
    return dict(
        dtype=torch.bfloat16,
        global_dim=(BF16_SEGMENT_ELEMS, rows, planes),
        global_stride_bytes=(
            BF16_SEGMENT_ELEMS * 2,
            rows * BF16_SEGMENT_ELEMS * 2,
        ),
        box_dim=(
            BF16_SEGMENT_ELEMS,
            rows // RECURRENCE_DV_PARTITIONS,
            1,
        ),
        swizzle="128B",
    )


def unique_descriptors(specs: dict[str, TensorMapSpec]) -> list[TensorMapSpec]:
    """Distinct descriptors, in first-use order; equality *is* exact alias."""
    seen: list[TensorMapSpec] = []
    for role in ROLES:
        spec = specs.get(role)
        if spec is not None and spec not in seen:
            seen.append(spec)
    return seen


@dataclass(frozen=True)
class RecurrenceTensorMaps:
    """Device addresses of each role's descriptor, plus the backing storage."""

    storage: torch.Tensor
    addresses: dict[str, int]

    def address(self, role: str) -> int:
        """Descriptor address, or 0 for a role this launch does not use."""
        return self.addresses.get(role, 0)


def build_recurrence_tensor_maps(
    specs: dict[str, TensorMapSpec], device: torch.device
) -> RecurrenceTensorMaps:
    """Encode the distinct descriptors and upload them as one device buffer."""
    distinct = unique_descriptors(specs)
    packed = bytearray()
    for spec in distinct:
        packed += spec.encode()
    storage = _upload_tensor_map_bytes(packed, device)
    if storage.data_ptr() % 64:
        raise RuntimeError("tensor map storage must be 64-byte aligned")

    base = storage.data_ptr()
    slot = {spec: base + i * DESCRIPTOR_BYTES for i, spec in enumerate(distinct)}
    return RecurrenceTensorMaps(
        storage=storage,
        addresses={role: slot[spec] for role, spec in specs.items()},
    )


LOAD_WARP = COMPUTE_WARPS  # 4
STORE_WARP = COMPUTE_WARPS + 1  # 5
REC_WARPS = COMPUTE_WARPS + 2  # 6
REC_THREADS = REC_WARPS * 32
INPUT_STAGES = 8 if RECURRENCE_DV_PARTITIONS == 2 else 6
OUTPUT_STAGES = 3
STAGE_KD = 0
STAGE_QD = 4096
STAGE_AK = 8192
STAGE_AQ = 12288
STAGE_GT = 12800
STAGE_V = 13312
INPUT_STAGE_BYTES = STAGE_V + VO_STAGE_ELEMS * 2
INPUT_STAGE_TX_BYTES = INPUT_STAGE_BYTES
OUTPUT_STAGE_BYTES = BT * DV_HALF * 2  # 2048
SMEM_STATE = 0
STATE_BYTES = DV_HALF * DK * 2  # 16384
SMEM_INPUT = SMEM_STATE + STATE_BYTES  # 16384
SMEM_OUTPUT = SMEM_INPUT + INPUT_STAGES * INPUT_STAGE_BYTES  # 93184
REC_SMEM_BARRIERS = SMEM_OUTPUT + OUTPUT_STAGES * OUTPUT_STAGE_BYTES  # 97280
MBAR_INPUT_READY = 0
MBAR_INPUT_CONSUMED = MBAR_INPUT_READY + INPUT_STAGES * 8  # 40
MBAR_OUTPUT_READY = MBAR_INPUT_CONSUMED + INPUT_STAGES * 8  # 80
MBAR_OUTPUT_CONSUMED = MBAR_OUTPUT_READY + OUTPUT_STAGES * 8  # 96
MBAR_STATE_READY = MBAR_OUTPUT_CONSUMED + OUTPUT_STAGES * 8  # 112
BARRIER_BYTES = MBAR_STATE_READY + 8  # 120
SMEM_RAW_END = REC_SMEM_BARRIERS + BARRIER_BYTES  # 97400
SMEM_ALIGNMENT = 256
SMEM_DYNAMIC_BYTES = (
    (SMEM_RAW_END + SMEM_ALIGNMENT - 1) // SMEM_ALIGNMENT * SMEM_ALIGNMENT
)  # 97536
MIN_BLOCKS_PER_MP = 1
INPUT_READY_ARRIVALS = 1  # completed by transaction bytes
INPUT_CONSUMED_ARRIVALS = COMPUTE_WARPS  # one per compute warp, not per thread
OUTPUT_READY_ARRIVALS = COMPUTE_WARPS
OUTPUT_CONSUMED_ARRIVALS = 1
STATE_READY_ARRIVALS = 1


def input_stage(chunk):
    return chunk - (chunk // INPUT_STAGES) * INPUT_STAGES


def input_generation(chunk):
    return chunk // INPUT_STAGES


def input_ready_parity(chunk):
    """Compute warps: pass once the slot's TMA has landed ``ig + 1`` times."""
    return input_generation(chunk) & 1


def input_consumed_parity(chunk):
    """Load warp: pass once all 4 compute warps have released the slot ``ig``
    times.  Generation 0 passes against the initial phase with no seeding."""
    return 1 ^ (input_generation(chunk) & 1)


def output_stage(chunk):
    return chunk - (chunk // OUTPUT_STAGES) * OUTPUT_STAGES


def output_generation(chunk):
    return chunk // OUTPUT_STAGES


def output_ready_parity(chunk):
    """Store warp: pass once all 4 compute warps have filled the slot."""
    return output_generation(chunk) & 1


def output_consumed_parity(chunk):
    """Compute warps: pass once the store warp has finished reading the slot."""
    return 1 ^ (output_generation(chunk) & 1)


KEY_BLOCKS = DK // BT  # 8
BAR_IN_READY = MBAR_INPUT_READY // 8  # 0
BAR_IN_CONSUMED = MBAR_INPUT_CONSUMED // 8  # 5
BAR_OUT_READY = MBAR_OUTPUT_READY // 8  # 10
BAR_OUT_CONSUMED = MBAR_OUTPUT_CONSUMED // 8  # 12
BAR_STATE = MBAR_STATE_READY // 8  # 14
FACTOR_SEGMENT_ELEMS = BF16_SEGMENT_STRIDE  # 1024


@cute.jit
def zero_acc4():
    z = cutlass.Float32(0.0)
    return (z, z, z, z)


@cute.jit
def mma_n8(a, b, c):
    """One native ``m16n8k16``: A is four registers, B two, C four."""
    return mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])


@cute.kernel
def emit_bt16_recurrence(
    gout: cute.Tensor,
    gcu_seqlens: cute.Tensor,
    gcu_chunks: cute.Tensor,
    desc_factor: cutlass.Int64,
    desc_aq: cutlass.Int64,
    desc_gt: cutlass.Int64,
    desc_v: cutlass.Int64,
    desc_out: cutlass.Int64,
    desc_state_in: cutlass.Int64,
    desc_state_out: cutlass.Int64,
    heads: cutlass.Int32,
    SCALE: cutlass.Float32,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    # x = P * seq + dv_half, y = head: the DV partition is the
    # fastest-varying coordinate, so the partitions of a head are adjacent
    # blocks and share a wave, which is what lets L2 serve the second one's
    # factor loads.  Everything coarser keeps the order (N, 2H, 1) had.
    seq = bidx // cutlass.Int32(RECURRENCE_DV_PARTITIONS)
    dv_half = bidx - seq * cutlass.Int32(RECURRENCE_DV_PARTITIONS)
    head = bidy

    # --- fixed arena ------------------------------------
    # One allocation of exactly SMEM_DYNAMIC_BYTES, so the launch parameter is
    # the plan's number and every address is a compile-time constant offset.
    alloc = cutlass.utils.SmemAllocator()
    p_base = alloc.allocate(SMEM_DYNAMIC_BYTES, 1024)
    p16 = cute.recast_ptr(p_base, dtype=cutlass.BFloat16)
    p32 = cute.recast_ptr(p_base, dtype=cutlass.Float32)
    p64 = cute.recast_ptr(p_base, dtype=cutlass.Int64)

    p_state = p16 + (SMEM_STATE // 2)
    p_bar = p64 + (REC_SMEM_BARRIERS // 8)

    if warp_id == 0:
        if lane == 0:
            for s in cutlass.range_constexpr(INPUT_STAGES):
                cute.arch.mbarrier_init(p_bar + BAR_IN_READY + s, INPUT_READY_ARRIVALS)
                cute.arch.mbarrier_init(
                    p_bar + BAR_IN_CONSUMED + s, INPUT_CONSUMED_ARRIVALS
                )
            for s in cutlass.range_constexpr(OUTPUT_STAGES):
                cute.arch.mbarrier_init(
                    p_bar + BAR_OUT_READY + s, OUTPUT_READY_ARRIVALS
                )
                cute.arch.mbarrier_init(
                    p_bar + BAR_OUT_CONSUMED + s, OUTPUT_CONSUMED_ARRIVALS
                )
            cute.arch.mbarrier_init(p_bar + BAR_STATE, STATE_READY_ARRIVALS)
            cute.arch.mbarrier_init_fence()
            fence_tensormap_acquire(desc_factor)
            fence_tensormap_acquire(desc_aq)
            fence_tensormap_acquire(desc_gt)
            fence_tensormap_acquire(desc_v)
            fence_tensormap_acquire(desc_out)
            fence_tensormap_acquire(desc_state_in)
            fence_tensormap_acquire(desc_state_out)
    cute.arch.barrier()

    # --- sequence coordinates ---------------------------
    token_start = cutlass.Int32(gcu_seqlens[seq])
    token_end = cutlass.Int32(gcu_seqlens[seq + 1])
    chunk_base = cutlass.Int32(gcu_chunks[seq])
    num_chunks = cute.ceil_div(token_end - token_start, BT)
    state_plane = seq * heads + head

    # --- initial state ----------------------------------
    if warp_id == 0:
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(p_bar + BAR_STATE, DV_HALF * DK * 2)
            tma_load_3d(
                p_state,
                desc_state_in,
                p_bar + BAR_STATE,
                0,
                dv_half * (STATE_BF16_ROWS_PER_VALUE * DV_HALF),
                state_plane,
            )
    cute.arch.mbarrier_wait(p_bar + BAR_STATE, 0)
    cute.arch.barrier()

    # --- roles ---------------------------------
    if warp_id < COMPUTE_WARPS:
        v_base = warp_id * WARP_VALUES
        # The state lives in registers for the whole kernel, read from shared
        # memory once here and written back once after the chunk loop.  Sixteen
        # packed BF16 registers per lane hold this warp's [128 key, 8 value]
        # slice, which is warp-private -- ``v_base`` is ``warp_id *
        # WARP_VALUES`` and no warp ever addresses another's columns -- so
        # nothing here needs a cross-warp exchange.
        #
        # It replaces 24 shared-memory instructions per chunk per warp (eight
        # ``ldmatrix_x2`` in pass 1, eight ``ldmatrix_x2_trans`` and eight
        # ``stmatrix_x2_trans`` in pass 2) with 16 ``movmatrix``.  The
        # ``.trans`` read is the C view, which is the layout pass 2 accumulates
        # in; pass 1 wants the B operand and gets it with one ``movmatrix`` per
        # register.  ``tests/test_recurrence_layouts.py`` enumerates that
        # equality, 64 of 64.
        h_state: tuple = ()
        for kb in cutlass.range_constexpr(KEY_BLOCKS):
            lo, hi = ldmatrix_x2_trans(p_state + state_x2_ptr(lane, kb, v_base))
            h_state = h_state + (lo, hi)

        for c in range(num_chunks):
            in_stage = input_stage(c)
            out_stage = output_stage(c)
            stage16 = (SMEM_INPUT // 2) + in_stage * (INPUT_STAGE_BYTES // 2)
            p_kd = p16 + (stage16 + STAGE_KD // 2)
            p_qd = p16 + (stage16 + STAGE_QD // 2)
            p_ak = p16 + (stage16 + STAGE_AK // 2)
            p_aq = p16 + (stage16 + STAGE_AQ // 2)
            p_v = p16 + (stage16 + STAGE_V // 2)
            smem_gt = cute.make_tensor(
                p32
                + ((SMEM_INPUT + STAGE_GT) // 4 + in_stage * (INPUT_STAGE_BYTES // 4)),
                cute.make_layout(DK),
            )
            p_out = p16 + ((SMEM_OUTPUT // 2) + out_stage * (OUTPUT_STAGE_BYTES // 2))

            token_base = token_start + c * BT
            valid_rows = token_end - token_base
            if valid_rows > BT:
                valid_rows = cutlass.Int32(BT)

            # The nine IKET ranges below -- five here, two on the producer, two
            # on the store warp -- are emitted only under the research build; see
            # the research tracing hooks.  They split a chunk into stall and
            # work for each role, which is what NCU cannot show.
            cute.arch.mbarrier_wait(
                p_bar + BAR_IN_READY + in_stage, input_ready_parity(c)
            )

            # ---- pass 1: X = Kd @ H and O = Qd @ H --------------------------
            # One state B fragment per key block feeds both MMAs, and nothing
            # writes the state until pass 2, so the fragment can be dropped
            # immediately after use.
            acc_x = zero_acc4()
            acc_o = zero_acc4()
            for kb in cutlass.range_constexpr(KEY_BLOCKS):
                # C -> B is a within-tile 8x8 transpose, one instruction per
                # packed register, and the fragment is fresh rather than
                # aliased onto ``h_state``: aliasing an MMA operand onto
                # persistent state registers is what made engine's equivalent
                # probe spill.
                b = (
                    movmatrix_b16(h_state[2 * kb]),
                    movmatrix_b16(h_state[2 * kb + 1]),
                )
                acc_x = mma_n8(
                    ldmatrix_x4(p_kd + factor_a_fragment_ptr(lane, kb)),
                    b,
                    acc_x,
                )
                acc_o = mma_n8(
                    ldmatrix_x4(p_qd + factor_a_fragment_ptr(lane, kb)),
                    b,
                    acc_o,
                )

            # ---- residual -------------------------
            # Both halves of a packed C register are the same token row, so the
            # tail mask is one predicate per register.  An invalid row selects an
            # exact packed BF16 zero instead of subtracting a V that belongs to
            # the next sequence.

            v_lo, v_hi = ldmatrix_x2(p_v + vo_x2_ptr(lane, v_base))
            row_lo = lane // 4
            res_lo = cutlass.Int32(0)
            res_hi = cutlass.Int32(0)
            if row_lo < valid_rows:
                res_lo = sub_bf16x2(v_lo, pack_bf16x2(acc_x[0], acc_x[1]))
            if row_lo + 8 < valid_rows:
                res_hi = sub_bf16x2(v_hi, pack_bf16x2(acc_x[2], acc_x[3]))
            # C layout -> B layout: transpose each packed 8x8 quadrant in place.
            res_b = (movmatrix_b16(res_lo), movmatrix_b16(res_hi))

            # ---- O += Aq @ R, into the same FP32 accumulator ----------------
            acc_o = mma_n8(
                ldmatrix_x4(p_aq + pairwise_a_fragment_ptr(lane)), res_b, acc_o
            )

            # ---- publish the output before the state update -----------------
            cute.arch.mbarrier_wait(
                p_bar + BAR_OUT_CONSUMED + out_stage, output_consumed_parity(c)
            )
            stmatrix_x2(
                p_out + vo_x2_ptr(lane, v_base),
                pack_bf16x2(acc_o[0] * SCALE, acc_o[1] * SCALE),
                pack_bf16x2(acc_o[2] * SCALE, acc_o[3] * SCALE),
            )
            warp_arrive(p_bar + BAR_OUT_READY + out_stage, lane)

            # ---- pass 2: H_next = Diag(GTotal) H + Ak @ R -------------------
            # The same addresses as pass 1, read with ``.trans`` so the state
            # arrives already in the accumulator's layout.  Each warp writes
            # only its own value columns, and pass 1 is complete, so the update
            # is safe in place.
            next_state: tuple = ()
            for kb in cutlass.range_constexpr(KEY_BLOCKS):
                h0, h1 = unpack_bf16x2(h_state[2 * kb])
                h2, h3 = unpack_bf16x2(h_state[2 * kb + 1])
                decay_lo = cutlass.Float32(smem_gt[kb * BT + row_lo])
                decay_hi = cutlass.Float32(smem_gt[kb * BT + row_lo + 8])
                acc_s = (
                    decay_lo * h0,
                    decay_lo * h1,
                    decay_hi * h2,
                    decay_hi * h3,
                )
                acc_s = mma_n8(
                    ldmatrix_x4_trans(p_ak + ak_a_fragment_ptr(lane, kb)),
                    res_b,
                    acc_s,
                )
                next_state = next_state + (
                    pack_bf16x2(acc_s[0], acc_s[1]),
                    pack_bf16x2(acc_s[2], acc_s[3]),
                )
            h_state = next_state

            warp_arrive(p_bar + BAR_IN_CONSUMED + in_stage, lane)

        # The final-state path below reads the state from shared memory with all
        # 192 threads, so the compute warps publish their registers first.  The
        # converging barrier after this branch is what orders it.
        for kb in cutlass.range_constexpr(KEY_BLOCKS):
            stmatrix_x2_trans(
                p_state + state_x2_ptr(lane, kb, v_base),
                h_state[2 * kb],
                h_state[2 * kb + 1],
            )

    elif warp_id == LOAD_WARP:
        for c in range(num_chunks):
            in_stage = input_stage(c)
            cute.arch.mbarrier_wait(
                p_bar + BAR_IN_CONSUMED + in_stage, input_consumed_parity(c)
            )

            if lane == 0:
                gchunk = chunk_base + c
                token_base = token_start + c * BT
                factor_row = gchunk * BT
                stage16 = SMEM_INPUT // 2 + in_stage * (INPUT_STAGE_BYTES // 2)
                mbar = p_bar + BAR_IN_READY + in_stage
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, INPUT_STAGE_TX_BYTES)
                # Nine instructions, one completion barrier, 15,360 bytes.
                for factor in cutlass.range_constexpr(3):
                    plane = factor * heads + head
                    dst = stage16 + (STAGE_KD // 2) + factor * (4096 // 2)
                    for segment in cutlass.range_constexpr(BF16_SEGMENTS):
                        tma_load_3d(
                            p16 + (dst + segment * FACTOR_SEGMENT_ELEMS),
                            desc_factor,
                            mbar,
                            segment * BF16_SEGMENT_ELEMS,
                            factor_row,
                            plane,
                        )
                tma_load_3d(
                    p16 + (stage16 + STAGE_AQ // 2),
                    desc_aq,
                    mbar,
                    0,
                    gchunk,
                    head,
                )
                tma_load_3d(
                    p32
                    + (
                        (SMEM_INPUT + STAGE_GT) // 4
                        + in_stage * (INPUT_STAGE_BYTES // 4)
                    ),
                    desc_gt,
                    mbar,
                    0,
                    gchunk,
                    head,
                )
                tma_load_3d(
                    p16 + (stage16 + STAGE_V // 2),
                    desc_v,
                    mbar,
                    dv_half * DV_HALF,
                    token_base,
                    head,
                )

    else:
        for c in range(num_chunks):
            out_stage = output_stage(c)
            p_out = p16 + ((SMEM_OUTPUT // 2) + out_stage * (OUTPUT_STAGE_BYTES // 2))
            token_base = token_start + c * BT
            valid_rows = token_end - token_base
            if valid_rows > BT:
                valid_rows = cutlass.Int32(BT)

            cute.arch.mbarrier_wait(
                p_bar + BAR_OUT_READY + out_stage, output_ready_parity(c)
            )

            if valid_rows == BT:
                # The full path: the stage is released only after
                # the store has read it, not when it is merely committed.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(desc_out, p_out, dv_half * DV_HALF, token_base, head)
                    tma_store_commit_group()
                    tma_store_wait_read(0)
                    cute.arch.mbarrier_arrive(p_bar + BAR_OUT_CONSUMED + out_stage)
            else:
                # Tail path: the whole warp stores 16-byte vectors, and no TMA,
                # fence, commit or wait-group is involved.
                for rep in cutlass.range_constexpr(OUTPUT_TAIL_ROUNDS):
                    task = lane + rep * 32
                    if task < valid_rows * VO_VECTORS_PER_ROW:
                        row = task // VO_VECTORS_PER_ROW
                        vec = task - row * VO_VECTORS_PER_ROW
                        frag = vec8_bf16(p_out, vo_idx(row, vec * 8))
                        store_vec8_bf16(
                            gout.iterator,
                            vo_global_index(
                                token_base + row, head, heads, dv_half, vec * 8
                            ),
                            frag,
                        )
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(p_bar + BAR_OUT_CONSUMED + out_stage)

    # --- final state handoff ----------------------------
    # All three roles converge here: the loads are issued, the last state update
    # is written, and every output store has completed its ``wait_group.read``.
    cute.arch.barrier()
    if warp_id == STORE_WARP:
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        if lane == 0:
            tma_store_3d(
                desc_state_out,
                p_state,
                0,
                dv_half * (STATE_BF16_ROWS_PER_VALUE * DV_HALF),
                state_plane,
            )
            tma_store_commit_group()
            tma_store_wait_read(0)


@cute.jit
def _recurrence_entry(
    gout: cute.Tensor,
    gcu_seqlens: cute.Tensor,
    gcu_chunks: cute.Tensor,
    desc_factor: cutlass.Int64,
    desc_aq: cutlass.Int64,
    desc_gt: cutlass.Int64,
    desc_v: cutlass.Int64,
    desc_out: cutlass.Int64,
    desc_state_in: cutlass.Int64,
    desc_state_out: cutlass.Int64,
    heads: cutlass.Int32,
    scale: cutlass.Float32,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    stream,
):
    emit_bt16_recurrence(
        gout,
        gcu_seqlens,
        gcu_chunks,
        desc_factor,
        desc_aq,
        desc_gt,
        desc_v,
        desc_out,
        desc_state_in,
        desc_state_out,
        heads,
        scale,
    ).launch(
        grid=(grid_x, grid_y, 1),
        block=(REC_THREADS, 1, 1),
        smem=SMEM_DYNAMIC_BYTES,
        min_blocks_per_mp=MIN_BLOCKS_PER_MP,
        stream=stream,
    )


__all__ = [
    "COMPUTE_WARPS",
    "DV_HALF",
    "INPUT_STAGES",
    "OUTPUT_TAIL_ROUNDS",
    "OUTPUT_TAIL_TASKS",
    "RECURRENCE_DV_PARTITIONS",
    "REC_THREADS",
    "ROLES",
    "SMEM_DYNAMIC_BYTES",
    "VO_VECTORS_PER_ROW",
    "_recurrence_entry",
    "build_recurrence_tensor_maps",
    "factor_a_fragment_ptr",
    "factor_idx",
    "recurrence_tensor_map_specs",
    "vo_idx",
]
