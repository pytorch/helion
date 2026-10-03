# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Packed global stores without experimental CUTLASS primitive dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from ._mlir_compat import ir


@dsl_user_op
def store_u32x4_if_valid(
    ptr: object,
    value0: object,
    value1: object,
    value2: object,
    value3: object,
    state_index: object,
    state_size: object,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store four packed words only when their slot is in range."""

    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    values = [
        cutlass.Uint32(value).ir_value(loc=loc, ip=ip)
        for value in (value0, value1, value2, value3)
    ]
    index = cutlass.Int64(state_index).ir_value(loc=loc, ip=ip)
    size = cutlass.Int64(state_size).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address, *values, index, size],
        (
            "{ .reg .pred valid; "
            "setp.lt.u64 valid, $5, $6; "
            "@valid st.global.L1::no_allocate.v4.u32 [$0], {$1, $2, $3, $4}; }"
        ),
        "l,r,r,r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
