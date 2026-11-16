"""``cuTensorMapEncodeTiled`` descriptors the launcher encodes on the host.

The KDA prepare and recurrence kernels take raw 128-byte TMA descriptors as
scalar wrapper arguments (the device code reads them through
``fence_tensormap_acquire``).  :class:`TensorMapSpec` is everything the
encoder is given and nothing else -- two roles with equal specs are one
descriptor -- and :func:`build_tensor_maps` encodes the distinct specs of a
launch into one device buffer.  :func:`factor_slab_geometry` is the one
description of the fused Kd/Qd/Ak factor slab both the prepare and the
recurrence descriptors are built from.  Leaf module: torch, ctypes and the
CUDA driver bindings (imported lazily); never cutlass.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# The KDA packed-workspace geometry: tokens per chunk, the key/value head dim
# and the BF16 elements of one 128-byte swizzle segment.
BT = 16
DK = 128
BF16_SEGMENT_ELEMS = 64


def upload_tensor_map_bytes(
    payload: bytes | bytearray, device: torch.device
) -> torch.Tensor:
    """Upload descriptor bytes before graph capture; the launcher owns reuse."""

    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().to(device)


DESCRIPTOR_BYTES = 128
SWIZZLES = ("128B", "64B", "NONE")


def encode_tensor_map(
    dtype: torch.dtype,
    base_ptr: int,
    global_dim: Sequence[int],
    strides_bytes: Sequence[int],
    box_dim: Sequence[int],
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


def spec_fields(geometry: dict) -> dict:
    """Rename :func:`factor_slab_geometry`'s keys for :class:`TensorMapSpec`.

    The encoder takes ``strides_bytes``; the spec dataclass calls the same
    field ``global_stride_bytes``.  One adapter beats two copies of the tuple.
    """
    return {
        "global_dim": geometry["global_dim"],
        "global_stride_bytes": geometry["strides_bytes"],
        "box_dim": geometry["box_dim"],
    }


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
    return {
        "global_dim": (DK, rows, 3 * heads),
        "strides_bytes": (DK * element_bytes, DK * rows * element_bytes),
        "box_dim": (BF16_SEGMENT_ELEMS, BT, 1),
    }


def unique_descriptors(specs: dict[str, TensorMapSpec]) -> list[TensorMapSpec]:
    """Distinct descriptors, in first-use (insertion) order; equality *is* exact
    alias."""
    seen: list[TensorMapSpec] = []
    for spec in specs.values():
        if spec not in seen:
            seen.append(spec)
    return seen


@dataclass(frozen=True)
class TensorMaps:
    """Device addresses of each role's descriptor, plus the backing storage."""

    storage: torch.Tensor
    addresses: dict[str, int]

    def address(self, role: str) -> int:
        """Descriptor address, or 0 for a role this launch does not use."""
        return self.addresses.get(role, 0)


def build_tensor_maps(
    specs: dict[str, TensorMapSpec], device: torch.device
) -> TensorMaps:
    """Encode the distinct descriptors and upload them as one device buffer."""
    distinct = unique_descriptors(specs)
    packed = bytearray()
    for spec in distinct:
        packed += spec.encode()
    storage = upload_tensor_map_bytes(packed, device)
    if storage.data_ptr() % 64:
        raise RuntimeError("tensor map storage must be 64-byte aligned")

    base = storage.data_ptr()
    slot = {spec: base + i * DESCRIPTOR_BYTES for i, spec in enumerate(distinct)}
    return TensorMaps(
        storage=storage,
        addresses={role: slot[spec] for role, spec in specs.items()},
    )
