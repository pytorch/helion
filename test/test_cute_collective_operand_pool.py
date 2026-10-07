from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes
from test.test_cute_collective_seeds import _independent_matmuls
from test.test_cute_collective_tf32 import _round_tf32

import helion
from helion._compiler.cute.collective_tcgen05 import CollectiveOperandSmem
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

CUDA_DEVICE = "cuda"


def test_operand_pool_sizes_are_bytes_and_cover_different_largest_operands() -> None:
    pool = CollectiveOperandSmem("pool")
    pool.pointers("first", "cutlass.BFloat16", 8192, 2048)
    pool.pointers("second", "cutlass.TFloat32", 1024, 8192)
    pool.pointers("third", "cutlass.Float16", 2048, 4096)
    assert (pool.a_bytes, pool.b_bytes) == (16384, 32768)


def _without_operand_storage(source: str) -> str:
    class RemoveOperandStorage(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith(("_ap", "_bp"))
            ):
                return None
            return node

    return ast.dump(RemoveOperandStorage().visit(ast.parse(source)))


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("recipe", ["scalar", "vector_unrolled"])
@skipUnlessBackends(["cute"])
def test_pool_preserves_mamba_arithmetic_and_every_synchronization(
    compute: str, recipe: str
) -> None:
    arguments = tuple(
        torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2)
    )
    bound = _cpu_bind(_mamba_kernel(static_shapes=False), arguments)
    config = helion.Config(
        block_sizes=[64, 64, 64],
        num_threads=[2, 64, 1, 1],
        cute_vector_widths=[1] * 7,
        cute_lane_layouts=["blocked"] * 7,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_native_seeded=True,
        cute_collective_recipe=recipe,
    )
    pooled = bound.to_code(config)
    with patch(
        "helion._compiler.cute.collective_matmul.CollectiveOperandSmem",
        return_value=None,
    ):
        separate = bound.to_code(config)
    assert "cute.arch.alloc_smem(cutlass.Uint8" in pooled
    assert "cute.arch.alloc_smem(cutlass.Uint8" not in separate
    assert _without_operand_storage(pooled) == _without_operand_storage(separate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
@skipUnlessBackends(["cute"])
def test_mixed_dtype_sequential_pool_tails_and_mutating_graphs(
    compute: str, dtypes: tuple[torch.dtype, torch.dtype]
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(1917)
    arguments = (
        torch.randn((65, 78), device=CUDA_DEVICE, dtype=dtypes[0]),
        torch.randn((78, 71), device=CUDA_DEVICE, dtype=dtypes[0]),
        torch.randn((131, 37), device=CUDA_DEVICE, dtype=dtypes[1]),
        torch.randn((37, 39), device=CUDA_DEVICE, dtype=dtypes[1]),
    )
    kernel = helion.kernel(
        _independent_matmuls.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        dot_precision="tf32",
    )
    config = helion.Config(
        block_sizes=[64, 32, 32, 64, 32, 64],
        num_threads=[4, 32, 1, 4, 32, 1],
        cute_vector_widths=[1] * 6,
        cute_lane_layouts=["blocked"] * 6,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_recipe="vector_unrolled",
    )
    control = kernel._bind_isolated(arguments)
    # Compile the unpooled control once, before the pooled candidate. Isolated
    # bindings retain different generated callables despite identical configs.
    with patch(
        "helion._compiler.cute.collective_matmul.CollectiveOperandSmem",
        return_value=None,
    ):
        assert "cute.arch.alloc_smem(cutlass.Uint8" not in control.to_code(config)
        control.set_config(config)
        control(*arguments)
    bound = kernel._bind_isolated(arguments)
    bound.set_config(config)
    assert "cute.arch.alloc_smem(cutlass.Uint8" in bound.to_code(config)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return bound(*arguments)

    replay = _make_cudagraph_replay(run)
    for factor in (1.0, 0.0, -0.5):
        for argument in arguments:
            argument.mul_(factor)
            argument.add_(0.125)
        reference = control(*arguments)
        for index in range(2):
            a, b = arguments[2 * index : 2 * index + 2]
            left = (a * 0.5).float()
            right = b.float()
            if a.dtype is torch.float32:
                left, right = _round_tf32(left), _round_tf32(right)
            expected = (left @ right).to(a.dtype)
            torch.testing.assert_close(reference[index], expected, rtol=0.02, atol=0.02)
        for actual in (run(), replay()):
            for value, expected in zip(actual, reference, strict=True):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
