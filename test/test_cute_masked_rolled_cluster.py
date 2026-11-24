from __future__ import annotations

import pytest
import torch

from test.cute_population_contracts import _target
from test.test_cute_rolled_cluster import _rms_ref
from test.test_cute_rolled_cluster import rms_norm_rolled_kernel

import helion
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteMaskedRolledClusterHeuristic,
)
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable

pytestmark = pytest.mark.cpu_only


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
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_kernel(), args)
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
def test_masked_cluster_seed_has_a_vectorized_cluster_slice(columns: int) -> None:
    """The seeds lower to a vectorized cluster slice; a register seed caches it
    for the later sweeps, a gmem seed reads it again (none at 65544 columns,
    which 16-element vectors do not divide).  One seed of each kind renders."""
    args = (
        torch.empty((3, columns), dtype=torch.bfloat16),
        torch.empty((columns,), dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_kernel(), args)
        host = bound.host_function
        assert host is not None
        seeds = CuteMaskedRolledClusterHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        first: dict[str, helion.Config] = {}
        for seed in seeds:
            first.setdefault(seed.config["cute_reduction_reloads"][0], seed)
        assert set(first) == ({"register"} if columns % 16 else {"register", "gmem"})
        for reload, seed in first.items():
            code = bound.to_code(bound.config_spec.normalized_config(seed))
            assert "_helion_cute_cluster_shape = (1," in code
            assert "cute.arch.load(" in code
            assert f"< {columns}" in code
            assert ("_fuse_cache_0" in code) == (reload == "register")


def test_wide_row_gmem_seed_halves_the_cluster() -> None:
    """At 262144 bf16 columns the 256-thread register seed takes a cluster of
    16 (64 elements of the row per thread); its gmem seed sweeps 128 per
    thread in a cluster of 8, in chunks of two 16-element packets per thread,
    at four blocks per SM."""
    columns = 262144
    args = (
        torch.empty((3, columns), dtype=torch.bfloat16),
        torch.empty((columns,), dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_kernel(), args)
        host = bound.host_function
        assert host is not None
        seeds = CuteMaskedRolledClusterHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        # The seeds themselves, without their independent-reduction twins.
        plain = [
            seed for seed in seeds if not seed.config.get("cute_independent_reduction")
        ]
        shapes: dict[tuple[str, int], tuple[object, ...]] = {}
        for seed in plain:
            config = seed.config
            shapes.setdefault(
                (config["cute_reduction_reloads"][0], config["num_threads"][1]),
                (
                    config["cute_cluster_n"],
                    config["reduction_loops"][0],
                    max(config["cute_vector_widths"]),
                    config["cute_min_blocks_per_mp"],
                ),
            )
        assert shapes[("register", 256)][0] == 16
        assert {
            threads: shape
            for (reload, threads), shape in shapes.items()
            if reload == "gmem"
        } == {
            256: (8, 8192, 16, 4),
            128: (16, 4096, 16, 8),
            512: (4, 16384, 16, 2),
        }
        (gmem,) = [
            seed
            for seed in plain
            if seed.config["cute_reduction_reloads"] == ["gmem"]
            and seed.config["num_threads"][1] == 256
        ]
        code = bound.to_code(bound.config_spec.normalized_config(gmem))
    assert "_helion_cute_cluster_shape = (1, 8, 1)" in code
    assert "cute.arch.load(" in code
    assert "_fuse_cache_0" not in code


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
