"""L2 prefetch."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda.utils import num_threads


@triton.jit
def prefetch_l2(base, offset: tl.constexpr, nbytes: tl.constexpr):  # noqa: ANN001, ANN201
    """One thread bulk-prefetches ``nbytes`` at byte ``offset`` past ``base`` into L2."""
    lane = tl.arange(0, num_threads())
    tl.inline_asm_elementwise(
        "{ .reg .pred %p; setp.eq.s32 %p, $2, 0; "
        f"@%p cp.async.bulk.prefetch.L2.global [$1], {nbytes}; mov.u32 $0, 0; }}",
        "=r,l,r",
        [base.to(tl.int64) + offset, lane],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
