"""A grouped candidate the strict normalization rejects must not abort the search.

CPU-only (fake CUDA target): the grouped jagged kernel at the catalogue's s2
shape (10240 x 1024 x 1024 over 16 groups). Its default config normalizes
strictly; a search candidate carrying ``tcgen05_warp_register_reallocation=False``
on a row-union schedule (which has no static register profile) is rejected by
the strict normalization, repaired by the search's normalization, and answers
``has_compiled_config() == False`` instead of raising into the render-ahead
batch (the after4 grouped_gemm s2 failure).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _target

import helion
from helion import exc
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])

ROW_UNION_KEY = "tcgen05_grouped_dense_row_union"
SCHEDULE_KEY = "tcgen05_grouped_row_union_schedule"
REALLOCATION_KEY = "tcgen05_warp_register_reallocation"


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _target(),
    ):
        yield
    torch.set_num_threads(old_threads)


def _bound_s2() -> BoundKernel:
    m, n, k, groups = 10240, 1024, 1024, 16
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=torch.int32) * (m // groups),
    )
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
        cute_segmented_matmul_tiling=True,
    )
    return kernel.bind(args)


# The after-sweep #3 s2 winner (row-union schedule nm_m256_n192_k64_clc), the
# config the catalogue's s2 search had landed on before the failure.
S2_WINNER = {
    "block_sizes": [128, 64, 128],
    "loop_orders": [[0, 1, 2]],
    "l2_groupings": [1],
    "cute_proven_bounds": False,
    "cute_pdl": False,
    "indexing": ["pointer", "pointer", "pointer", "tensor_descriptor", "pointer"],
    "pid_type": "persistent_interleaved",
    "tcgen05_cta_group": "auto",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 2,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_l2_swizzle_size": 1,
    "tcgen05_batch_raster": "batch_fastest",
    "tcgen05_grouped_dense_row_union": True,
    "tcgen05_grouped_row_union_schedule": "nm_m256_n192_k64_clc",
    "tcgen05_ab_startup_prefill": True,
    "tcgen05_c_store_mode": "normal",
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_layout_strategy": "default",
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_warp_spec_register_decrease": 120,
    "tcgen05_warp_spec_register_increase": 256,
    "tcgen05_consumer_regs": 256,
    "tcgen05_persistence_model": "static_persistent",
    "cute_min_blocks_per_mp": 1,
    "cute_collective_mma": False,
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_seed": False,
    "cute_collective_tmem_a": False,
    "cute_collective_copy": "scalar",
    "cute_collective_compute": "warp",
    "cute_vector_widths": [1, 1, 1, 1],
    "cute_lane_layouts": ["blocked", "blocked", "blocked", "blocked"],
}


def test_default_config_normalizes_and_a_rejected_candidate_does_not_raise() -> None:
    bound = _bound_s2()
    with bound.env:
        spec = bound.config_spec
        default = spec.default_config()
        # the default config carries no register profile knob and normalizes
        normalized = spec.normalized_config(default)
        assert REALLOCATION_KEY not in normalized.config
        assert bound.has_compiled_config(default) is False
        # the s2 winner itself normalizes
        winner = spec.normalized_config(helion.Config(**S2_WINNER))
        assert winner.config[SCHEDULE_KEY] == "nm_m256_n192_k64_clc"
        # a candidate the search can draw from it: the register profile knob's
        # False value on a row-union schedule, which has no static profile
        candidate = helion.Config(**{**S2_WINNER, REALLOCATION_KEY: False})
        with pytest.raises(exc.InvalidConfig, match="warp_register_reallocation"):
            spec.normalized_config(candidate)
        # the search's normalization repairs it by dropping the knob
        repaired = dict(candidate.config)
        spec.normalize(repaired, _fix_invalid=True)
        assert REALLOCATION_KEY not in repaired
        # and asking whether it is compiled is a plain "no", not an abort
        assert bound.has_compiled_config(candidate) is False
