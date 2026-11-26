"""What helion.inductor tells a backend about a matmul it lowers, and the plan
the backend answers with (``Backend.inductor_matmul_plan``)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass(frozen=True)
class MatmulFacts:
    """A matmul helion.inductor lowers, ``a @ b``, or a kernel's dot (rank 2,
    or 3 for a batch). Sizes, strides and storage offsets are in elements; a
    field is None where helion.inductor does not know it: an operand's layout
    before ``use_native_matmul()`` fixes it, a dot's strides."""

    dtype: torch.dtype
    # the device's compute capability major (None off GPUs) and SM count
    major: int | None
    sm_count: int
    rank: int = 2
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
    # layouts and asks again for `transpose`, or the sizes for the dot fields.
    may_transpose: bool = False
    may_rewrite: bool = False
    # Lower as the transposed product (b.T @ a.T).T.
    transpose: bool = False
    # A kernel's dot keeps its K loop however short K is.
    keep_k_loop: bool = False
