"""CuteBackend's plan for the matmuls helion.inductor lowers
(``Backend.inductor_matmul_plan``): the tcgen05 rules of
``tcgen05/search_plan.py`` and the CTA-pair tile model of
``dense_aligned_n.py``, for 16-bit products (and TF32 transposes) on
SM100-family GPUs."""

from __future__ import annotations

import functools
import hashlib
import pathlib
import sys
from typing import TYPE_CHECKING
from typing import Any

from torch._inductor.runtime.hints import DeviceProperties

from ..inductor_matmul import MatmulPlan
from . import cp_async_operand
from . import cute_mma_dag
from . import cute_warp_mma_gemm
from . import dense_aligned_n
from . import stage_aligned_operands
from . import tcgen05_constants
from .cute_mma_dag import mma_dag_takes_row
from .dense_aligned_n import dense_aligned_n_sixteen_bit
from .dense_aligned_n import transposed_pair_tile_orientation
from .stage_aligned_operands import STAGED_ALIGNMENT_BYTES
from .stage_aligned_operands import STAGING_MIN_ELEMENTS
from .tcgen05 import search_plan
from .tcgen05.search_plan import tcgen05_copies_n_major_b
from .tcgen05.search_plan import tcgen05_min_n
from .tcgen05.search_plan import tcgen05_shape_rejected
from .tcgen05.search_plan import tcgen05_small_n_zero_fills
from .tcgen05.search_plan import tcgen05_transposes_epilogue_bound_product
from .tcgen05.search_plan import tcgen05_transposes_misaligned_a_rows
from .tcgen05.search_plan import tcgen05_transposes_misaligned_b_rows
from .tcgen05.search_plan import tcgen05_transposes_misaligned_batched_output
from .tcgen05.search_plan import tcgen05_transposes_n_edge_b_rows
from .tcgen05.search_plan import tcgen05_transposes_skinny_batched_product
from .tcgen05.search_plan import tcgen05_transposes_skinny_product
from .tcgen05.search_plan import tcgen05_transposes_wide_tf32_product

if TYPE_CHECKING:
    import torch

    from ..inductor_matmul import MatmulFacts

# The longest K a batch of small matrices reads whole (MatmulPlan.whole_k).
SMALL_MATRIX_WHOLE_K_MAX = 128

# The modules whose code decides the plan, which rules_key() hashes.
RULE_MODULES = (
    cp_async_operand,
    cute_mma_dag,
    cute_warp_mma_gemm,
    dense_aligned_n,
    stage_aligned_operands,
    search_plan,
    tcgen05_constants,
    sys.modules[__name__],
)


@functools.cache
def rules_key() -> str:
    """A hash of RULE_MODULES' sources."""
    sources = (
        pathlib.Path(str(module.__file__)).read_bytes() for module in RULE_MODULES
    )
    return hashlib.sha256(b"".join(sources)).hexdigest()


def transposed_product_wins(
    m: int, n: int, k: int, device: torch.device, *, b_n_major: bool = False
) -> bool:
    """Whether the CTA-pair tile model runs a 16-bit m x n x k product of a
    row-major a better transposed on `device`, b K-major or, with b_n_major,
    N-major."""
    props = DeviceProperties.create(device)
    return _pair_transposes(
        m, n, k, props.major, props.multi_processor_count, b_n_major=b_n_major
    )


def _pair_transposes(
    m: int, n: int, k: int, major: int | None, sm_count: int, *, b_n_major: bool
) -> bool:
    return (
        major == 10
        and transposed_pair_tile_orientation(
            m, n, k, sm_pairs=sm_count // 2, b_n_major=b_n_major
        ).transposes
    )


def matmul_plan(facts: MatmulFacts) -> MatmulPlan:
    sixteen_bit = dense_aligned_n_sixteen_bit(facts.dtype)
    native = sixteen_bit and facts.major == 10
    plan: dict[str, Any] = {
        "may_transpose": sixteen_bit or facts.tf32,
        "may_rewrite": native,
    }
    m, n, k = facts.m, facts.n, facts.k
    if m is None or n is None or k is None:
        return MatmulPlan(**plan)
    if native:
        small_n = facts.rank == 2 and tcgen05_small_n_zero_fills(m, n, k, facts.dtype)
        plan["keep_k_loop"] = not tcgen05_shape_rejected(
            m, n, k, facts.dtype, small_n_supported=small_n
        )
        # Asked without the batch: however many small matrices a batch holds,
        # the stored product runs them.
        plan["whole_k"] = (
            facts.rank == 3
            and facts.out_dtype == facts.dtype
            and tcgen05_shape_rejected(m, n, k, facts.dtype)
            and n >= tcgen05_min_n(facts.dtype)
            and k <= SMALL_MATRIX_WHOLE_K_MAX
            and mma_dag_takes_row(k)
        )
    a, b = facts.a_strides, facts.b_strides
    if facts.rank == 3:
        if native and a is not None and b is not None and facts.batch is not None:
            a0, a1, a2 = a
            b0, b1, b2 = b
            plan["transpose"] = tcgen05_transposes_skinny_batched_product(
                facts.batch,
                m,
                n,
                k,
                facts.dtype,
                a_strides=(a0, a1, a2),
                b_strides=(b0, b1, b2),
                num_sms=facts.sm_count,
            ) or tcgen05_transposes_misaligned_batched_output(
                facts.batch,
                m,
                n,
                k,
                facts.dtype,
                a_strides=(a1, a2),
                b_strides=(b1, b2),
                num_sms=facts.sm_count,
            )
        return MatmulPlan(**plan)
    if a is not None and b is not None:
        plan["transpose"] = _transposes(facts, m, n, k, (a[0], a[1]), (b[0], b[1]))
        if native:
            plan["copy_b_k_major"] = tcgen05_copies_n_major_b(
                m,
                n,
                k,
                facts.dtype,
                a_strides=(a[0], a[1]),
                b_strides=(b[0], b[1]),
                num_sms=facts.sm_count,
            )
    plan["stage"] = (
        _staged_row_stride(facts, k, 0, m * k, a, facts.a_offset),
        _staged_row_stride(facts, k, 1, k * n, b, facts.b_offset),
    )
    plan["pad_output"] = (
        _loads_k_major_rows(facts, 0, a, facts.a_offset),
        _loads_k_major_rows(facts, 1, b, facts.b_offset),
    )
    return MatmulPlan(**plan)


def _transposes(
    facts: MatmulFacts,
    m: int,
    n: int,
    k: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    if facts.major != 10:
        return False
    rule_args: dict[str, Any] = {
        "a_strides": a,
        "b_strides": b,
        "num_sms": facts.sm_count,
    }
    if facts.tf32:
        return tcgen05_transposes_wide_tf32_product(m, n, k, facts.dtype, **rule_args)
    if not dense_aligned_n_sixteen_bit(facts.dtype):
        return False
    if (
        tcgen05_transposes_skinny_product(
            m,
            n,
            k,
            facts.dtype,
            **rule_args,
            a_offset=facts.a_offset or 0,
            b_offset=facts.b_offset or 0,
        )
        or tcgen05_transposes_misaligned_b_rows(m, n, k, facts.dtype, **rule_args)
        or tcgen05_transposes_n_edge_b_rows(m, n, k, facts.dtype, **rule_args)
        or tcgen05_transposes_misaligned_a_rows(m, n, k, facts.dtype, **rule_args)
        or tcgen05_transposes_epilogue_bound_product(m, n, k, facts.dtype, **rule_args)
    ):
        return True
    b_k_major = b[-2] == 1
    return (
        a[-1] == 1
        and (b_k_major or b[-1] == 1)
        and _pair_transposes(
            m, n, k, facts.major, facts.sm_count, b_n_major=not b_k_major
        )
    )


def _aligned(facts: MatmulFacts, row_stride: int, offset: int | None) -> bool:
    itemsize = facts.dtype.itemsize
    return (row_stride * itemsize) % 16 == 0 and ((offset or 0) * itemsize) % 16 == 0


def _staged_row_stride(
    facts: MatmulFacts,
    k: int,
    operand: int,
    numel: int,
    strides: tuple[int, ...] | None,
    offset: int | None,
) -> int | None:
    """The row stride of an aligned copy of K-major operand `operand` (0: a,
    1: b), whose rows the TMA producer cannot load as they are. An operand
    expanded along M or N (a 0 stride) is read as it is: a copy would write
    every repeat."""
    k_dim = 1 - operand
    if (
        strides is None
        or strides[k_dim] != 1
        or strides[1 - k_dim] == 0
        or numel < STAGING_MIN_ELEMENTS
        or _aligned(facts, strides[1 - k_dim], offset)
    ):
        return None
    align = STAGED_ALIGNMENT_BYTES // facts.dtype.itemsize
    return -(-k // align) * align


def _loads_k_major_rows(
    facts: MatmulFacts,
    operand: int,
    strides: tuple[int, ...] | None,
    offset: int | None,
) -> bool:
    """Whether the TMA producer loads operand `operand`'s K-major rows as they
    are."""
    k_dim = 1 - operand
    return (
        strides is not None
        and strides[k_dim] == 1
        and _aligned(facts, strides[1 - k_dim], offset)
    )
