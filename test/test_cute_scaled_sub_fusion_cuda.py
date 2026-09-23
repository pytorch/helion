from __future__ import annotations

import pytest
import torch

from .test_cute_bounded_cache_cuda import CacheCase
from .test_cute_bounded_cache_cuda import run_guarded_cache_case
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["dense", "unaligned"])
def test_scaled_sub_fusion_exact_masked_and_vector_replays(fast_math, dtype, layout):
    case = CacheCase(
        f"scaled-fusion-{dtype}-{layout}-fast{int(fast_math)}",
        (17, 513),
        dtype,
        False,
        4096,
        128,
        4 if dtype == torch.float32 else 8,
        layout=layout,
    )
    run_guarded_cache_case(case, fast_math=fast_math)
