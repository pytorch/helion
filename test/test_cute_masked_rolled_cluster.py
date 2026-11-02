from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_rolled_cluster import _rms_ref
from test.test_cute_rolled_cluster import rms_norm_rolled_kernel

import helion
from helion._compiler.autotuner_heuristics.cute import CuteMaskedRolledClusterHeuristic
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessCuteAvailable


def _kernel() -> helion.Kernel:
    return helion.kernel(
        rms_norm_rolled_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


@pytest.mark.parametrize("columns", [3584, 4097, 6144])
def test_static_cluster_tail_keeps_the_original_bound(columns: int) -> None:
    args = (
        torch.empty((3, columns), dtype=torch.bfloat16),
        torch.empty((columns,), dtype=torch.bfloat16),
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_kernel(), args)
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
            cute_cluster_n=8,
        )
        code = bound.to_code(config)
    assert "_helion_cute_cluster_shape = (1, 8, 1)" in code
    assert f"< {columns}" in code
    assert "cute.arch.block_idx()[1]) * 1024" in code
    if columns % 8 == 0:
        assert "_fuse_cache_0" in code


@pytest.mark.parametrize("columns", [32000, 65544, 128256, 151936])
def test_masked_cluster_seed_has_a_vectorized_register_slice(columns: int) -> None:
    args = (
        torch.empty((3, columns), dtype=torch.bfloat16),
        torch.empty((columns,), dtype=torch.bfloat16),
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_kernel(), args)
        host = bound.host_function
        assert host is not None
        seeds = CuteMaskedRolledClusterHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert seeds
        for seed in seeds:
            normalized = bound.config_spec.normalized_config(seed)
            code = bound.to_code(normalized)
            assert "_helion_cute_cluster_shape = (1," in code
            assert "cute.arch.load(" in code
            assert f"< {columns}" in code
            assert "_fuse_cache_0" in code


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
@skipIfNotCUDA()
@pytest.mark.cuda
@pytest.mark.parametrize("columns", [3584, 4097, 6144])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_static_cluster_tail_matches_rmsnorm(columns: int, dtype: torch.dtype) -> None:
    x = torch.randn((3, columns), device=DEVICE, dtype=dtype)
    weight = torch.randn((columns,), device=DEVICE, dtype=dtype)
    code, (out, inv_rms) = code_and_output(
        _kernel(),
        (x, weight),
        block_sizes=[1],
        num_threads=[1, 128],
        reduction_loops=[1024],
        cute_vector_widths=[8 if dtype == torch.bfloat16 else 4, 1],
        cute_reduction_reloads=["register"],
        cute_cluster_n=8,
    )
    reference, reference_inv = _rms_ref(x, weight)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(inv_rms, reference_inv, rtol=1e-4, atol=1e-4)
    assert "_helion_cute_cluster_shape = (1, 8, 1)" in code
