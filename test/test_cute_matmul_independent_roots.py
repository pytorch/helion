from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise_first(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    pointwise = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        pointwise[i] = x[i] + 1
    for m, n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([m, n], dtype=torch.float32)
        for k in hl.tile(a.size(1)):
            acc = hl.dot(a[m, k], b[k, n], acc)
        out[m, n] = acc.to(out.dtype)
    return out, pointwise


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise_last(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    pointwise = torch.empty_like(x)
    for m, n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([m, n], dtype=torch.float32)
        for k in hl.tile(a.size(1)):
            acc = hl.dot(a[m, k], b[k, n], acc)
        out[m, n] = acc.to(out.dtype)
    for i in hl.tile(x.numel()):
        pointwise[i] = x[i] + 1
    return out, pointwise


@pytest.mark.parametrize("first", [False, True])
@skipIfRefEager("requires compiler seeds")
def test_unrelated_root_retains_its_default_tile(first: bool) -> None:
    args = (
        torch.empty(64, 128, dtype=torch.float16),
        torch.empty(128, 64, dtype=torch.float16),
        torch.empty(257),
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(_pointwise_first if first else _pointwise_last, args)
        assert bound.host_function is not None
        spec = bound.config_spec
        fact = spec.matmul_facts[0]
        mma_axes = {fact.m_block_id, fact.n_block_id, fact.k_block_id}
        (other,) = set(spec.block_sizes.valid_block_ids()) - mma_axes
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds
        for seed in seeds:
            fragment = spec.block_sizes.block_id_lookup(other)._fragment(spec)
            assert (
                spec.block_sizes.config_get(seed.block_sizes, other)
                == fragment.default()
            )
            assert spec.num_threads.config_get(seed.num_threads, other) == 0
        assert "cute.gemm(" in bound.to_code(seeds[0])


@pytest.mark.parametrize("nested", [False, True])
@skipIfRefEager("requires compiler seeds")
def test_non_matmul_axis_in_matmul_tree_is_not_admitted(nested: bool) -> None:
    if nested:

        def kernel(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (x.size(0), a.size(0), b.size(1)), device=a.device, dtype=a.dtype
            )
            for m, n in hl.tile([a.size(0), b.size(1)]):
                for extra in hl.tile(x.size(0)):
                    acc = hl.zeros([m, n], dtype=torch.float32)
                    for k in hl.tile(a.size(1)):
                        acc = hl.dot(a[m, k], b[k, n], acc)
                    out[extra, m, n] = (
                        acc[None, :, :] + hl.zeros([extra, 1, 1], dtype=torch.float32)
                    ).to(out.dtype)
            return out
    else:

        def kernel(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                (x.size(0), a.size(0), b.size(1)), device=a.device, dtype=a.dtype
            )
            for extra, m, n in hl.tile([x.size(0), a.size(0), b.size(1)]):
                acc = hl.zeros([m, n], dtype=torch.float32)
                for k in hl.tile(a.size(1)):
                    acc = hl.dot(a[m, k], b[k, n], acc)
                out[extra, m, n] = (
                    acc[None, :, :] + hl.zeros([extra, 1, 1], dtype=torch.float32)
                ).to(out.dtype)
            return out

    variant = helion.kernel(
        kernel, backend="cute", static_shapes=True, autotune_effort="none"
    )

    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            variant,
            (
                torch.empty(64, 128, dtype=torch.float16),
                torch.empty(128, 64, dtype=torch.float16),
                torch.empty(8),
            ),
        )
        assert bound.host_function is not None
        assert not CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("explicit_tile", [False, True])
@skipIfNotCUDA()
@skipIfRefEager("requires compiler seeds")
def test_mixed_root_seed_executes(first: bool, explicit_tile: bool) -> None:
    a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    x = torch.randn(257, device="cuda")
    args = (a, b, x)
    bound = (_pointwise_first if first else _pointwise_last).bind(args)
    assert bound.host_function is not None
    seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    seed = next(s for s in seeds if s.get("cute_collective_compute", "warp") == "warp")
    if explicit_tile:
        spec = bound.config_spec
        fact = spec.matmul_facts[0]
        (other,) = set(spec.block_sizes.valid_block_ids()) - {
            fact.m_block_id,
            fact.n_block_id,
            fact.k_block_id,
        }
        seed = helion.Config.from_dict(
            seed.config
            | {
                "block_sizes": [
                    4096 if block_id == other else size
                    for block_id, size in zip(
                        spec.block_sizes.valid_block_ids(),
                        seed.block_sizes,
                        strict=True,
                    )
                ]
            }
        )
    actual, pointwise = bound.compile_config(seed)(*args)
    torch.testing.assert_close(actual, a @ b)
    torch.testing.assert_close(pointwise, x + 1)
