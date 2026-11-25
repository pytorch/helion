"""CuteBackend's plan for the matmuls helion.inductor lowers
(``Backend.inductor_matmul_plan``): the tcgen05 rules of
``tcgen05/search_plan.py`` and the CTA-pair tile model of
``dense_aligned_n.py``, for 16-bit products on SM100-family GPUs."""

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
from . import cute_warp_mma_gemm
from . import dense_aligned_n
from . import tcgen05_constants
from .dense_aligned_n import dense_aligned_n_sixteen_bit
from .dense_aligned_n import transposed_pair_tile_orientation
from .tcgen05 import search_plan
from .tcgen05.search_plan import tcgen05_shape_rejected
from .tcgen05.search_plan import tcgen05_small_n_zero_fills
from .tcgen05.search_plan import tcgen05_transposes_epilogue_bound_product
from .tcgen05.search_plan import tcgen05_transposes_misaligned_b_rows
from .tcgen05.search_plan import tcgen05_transposes_skinny_product

if TYPE_CHECKING:
    import torch

    from ..inductor_matmul import MatmulFacts

# The modules whose code decides the plan, which rules_key() hashes.
RULE_MODULES = (
    cp_async_operand,
    cute_warp_mma_gemm,
    dense_aligned_n,
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
    plan: dict[str, Any] = {"may_transpose": sixteen_bit, "may_rewrite": native}
    m, n, k = facts.m, facts.n, facts.k
    if m is None or n is None or k is None:
        return MatmulPlan(**plan)
    if native:
        small_n = facts.rank == 2 and tcgen05_small_n_zero_fills(m, n, k, facts.dtype)
        plan["keep_k_loop"] = not tcgen05_shape_rejected(
            m, n, k, facts.dtype, small_n_supported=small_n
        )
    a, b = facts.a_strides, facts.b_strides
    if a is not None and b is not None:
        plan["transpose"] = _transposes(facts, m, n, k, (a[0], a[1]), (b[0], b[1]))
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
