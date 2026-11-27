"""What helion.inductor tells a backend about a matmul it lowers, and the plan
the backend answers with (``Backend.inductor_matmul_plan``)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass(frozen=True)
class MatmulFacts:
    """A matmul helion.inductor lowers, ``a @ b`` (rank 2) or ``torch.bmm(a, b)``
    (rank 3), or a kernel's dot of that form. Sizes, strides and storage offsets
    are in elements; a field is None where helion.inductor does not know it:
    an operand's layout before ``use_native_matmul()`` fixes it, a dot's
    strides."""

    dtype: torch.dtype
    out_dtype: torch.dtype
    # fp32 under torch.backends.cuda.matmul.fp32_precision == "tf32"
    tf32: bool
    # the device's compute capability major (None off GPUs) and SM count
    major: int | None
    sm_count: int
    rank: int = 2
    batch: int | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None
    a_strides: tuple[int, ...] | None = None
    b_strides: tuple[int, ...] | None = None
    a_offset: int | None = None
    b_offset: int | None = None


@dataclasses.dataclass(frozen=True)
class MatmulPlan:
    """A backend's plan for a matmul; the defaults lower it as written."""

    # Answered without layouts: whether helion.inductor reads the operands'
    # layouts and asks again for `transpose` of a @ b, or the sizes and layouts
    # for the other fields (a bmm's `transpose` included).
    may_transpose: bool = False
    may_rewrite: bool = False
    # Lower as the transposed product: (b.T @ a.T).T, (b.mT @ a.mT).mT for bmm.
    transpose: bool = False
    # Copy b to a K-major (column-major) buffer.
    copy_b_k_major: bool = False
    # Row strides of aligned copies of a and b; None reads an operand as it is.
    stage: tuple[int | None, int | None] = (None, None)
    # Whether pad_mm's zero pad of a's M (b's N) may move to the output.
    pad_output: tuple[bool, bool] = (False, False)
    # A kernel's dot keeps its K loop however short K is, or reads K whole.
    keep_k_loop: bool = False
    whole_k: bool = False
