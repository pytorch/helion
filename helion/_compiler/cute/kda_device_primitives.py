# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility exports for the former KDA-specific primitive module."""

from __future__ import annotations

from .affine_recurrence_primitives import _ldmatrix as _ldmatrix
from .affine_recurrence_primitives import _mma_m16n8k16 as _mma_m16n8k16
from .affine_recurrence_primitives import ldmatrix_x2 as ldmatrix_x2
from .affine_recurrence_primitives import ldmatrix_x2_trans as ldmatrix_x2_trans
from .affine_recurrence_primitives import ldmatrix_x4_trans as ldmatrix_x4_trans
from .affine_recurrence_primitives import mma_m16n8k16_bf16 as mma_m16n8k16_bf16
from .affine_recurrence_primitives import movmatrix_b16 as movmatrix_b16
from .affine_recurrence_primitives import pack_bf16x2 as pack_bf16x2
from .affine_recurrence_primitives import stmatrix_x2 as stmatrix_x2
from .affine_recurrence_primitives import stmatrix_x2_trans as stmatrix_x2_trans
from .affine_recurrence_primitives import store_u32x4_if_valid as store_u32x4_if_valid
from .affine_recurrence_primitives import store_vec8_bf16 as store_vec8_bf16
from .affine_recurrence_primitives import tma_load_3d as tma_load_3d
from .affine_recurrence_primitives import tma_store_3d as tma_store_3d
from .affine_recurrence_primitives import (
    tma_store_commit_group as tma_store_commit_group,
)
from .affine_recurrence_primitives import tma_store_wait_read as tma_store_wait_read
from .affine_recurrence_primitives import vec4_f32 as vec4_f32
from .affine_recurrence_primitives import vec8_bf16 as vec8_bf16
from .affine_recurrence_primitives import vec_at as vec_at
from .affine_recurrence_primitives import warp_arrive as warp_arrive

__all__ = [
    "_mma_m16n8k16",
    "ldmatrix_x2",
    "ldmatrix_x2_trans",
    "ldmatrix_x4_trans",
    "mma_m16n8k16_bf16",
    "movmatrix_b16",
    "pack_bf16x2",
    "stmatrix_x2",
    "stmatrix_x2_trans",
    "store_u32x4_if_valid",
    "store_vec8_bf16",
    "tma_load_3d",
    "tma_store_3d",
    "tma_store_commit_group",
    "tma_store_wait_read",
    "vec4_f32",
    "vec8_bf16",
    "vec_at",
    "warp_arrive",
]
