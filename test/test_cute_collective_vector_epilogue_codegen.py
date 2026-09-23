from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

CUDA_DEVICE = "cuda"


def _mamba_config(strategy: str, compute: str = "tcgen05") -> helion.Config:
    return helion.Config(
        block_sizes=[64, 64, 64],
        num_threads=[2, 64, 1, 1],
        cute_vector_widths=[1] * 7,
        cute_lane_layouts=["blocked"] * 7,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_native_seeded=True,
        cute_collective_recipe="vector_unrolled",
        cute_collective_epilogue=strategy,
    )


@pytest.mark.parametrize("strategy", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_mamba_epilogue_removes_dead_seed_replay(
    strategy: str, compute: str, static_shapes: bool
) -> None:
    arguments = tuple(
        torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2)
    )
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), arguments)
    source = bound.to_code(_mamba_config(strategy, compute))
    loops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("epilogue_vector_slot")
    ]
    assert len(loops) == 1
    epilogue = ast.unparse(loops[0])
    assert "epilogue_vector_values" in epilogue
    assert "collective_1_c[" in epilogue
    assert "collective_c[" not in epilogue
    assert "dA_cumsum.iterator" not in epilogue
    assert "cute.make_ptr(x.element_type" in epilogue
    assert "cute.make_ptr(out.element_type" in epilogue
    assert "out.iterator.toint() % 16 == 0" in source
    assert ("cutlass.range_constexpr" in ast.unparse(loops[0].iter)) == (
        strategy == "vector_unrolled"
    )


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _residual_matmul(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[row, reduction], b[reduction, column])
        value = torch.relu(acc) + residual[row, column].to(torch.float32) * 0.5
        out[row, column] = value.to(out.dtype)
    return out


def _residual_config(strategy: str, compute: str, n_first: bool) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_lane_layouts=["blocked"] * 3,
        loop_orders=[[1, 0] if n_first else [0, 1]],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_recipe="vector_unrolled",
        cute_collective_epilogue=strategy,
    )


def _residual_inputs(
    dtype: torch.dtype, out_dtype: torch.dtype, unaligned: bool, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((130, 80), dtype=dtype, device=device)[:, :78],
        torch.empty((78, 80), dtype=dtype, device=device)[:, :70],
        torch.empty((130, 80), dtype=out_dtype, device=device)[:, :70],
        torch.empty((130 * 80 + int(unaligned),), dtype=out_dtype, device=device)[
            int(unaligned) :
        ].view(130, 80)[:, :70],
    )


_CASES = [
    (torch.float16, torch.float16, "tcgen05", False, False),
    (torch.bfloat16, torch.bfloat16, "tcgen05", False, True),
    (torch.float16, torch.float32, "warp", False, True),
    (torch.bfloat16, torch.float32, "warp", True, False),
]


@pytest.mark.parametrize("strategy", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("dtype,out_dtype,compute,unaligned,n_first", _CASES)
def test_residual_epilogue_codegen(
    strategy: str,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
    compute: str,
    unaligned: bool,
    n_first: bool,
) -> None:
    arguments = _residual_inputs(dtype, out_dtype, unaligned)
    bound = _cpu_bind(_residual_matmul, arguments)
    source = bound.to_code(_residual_config(strategy, compute, n_first))
    assert "epilogue_vector_values" in source
    assert "epilogue_store_lane" in source
    assert ("tcgen05.CtaGroup.ONE" in source) == (compute == "tcgen05")
    if unaligned:
        assert "out.iterator.toint() % 16 == 0" in source
        assert "out.layout.stride[0] == 80" in source


@pytest.mark.parametrize("strategy", [None, True, 8, "automatic"])
def test_invalid_epilogue_strategy_rejected(strategy: object) -> None:
    arguments = _residual_inputs(torch.float16, torch.float16, False)
    bound = _cpu_bind(_residual_matmul, arguments)
    config = _residual_config("scalar", "warp", False).config | {
        "cute_collective_epilogue": strategy
    }
    with pytest.raises(helion.exc.InvalidConfig, match="cute_collective_epilogue"):
        bound.to_code(helion.Config.from_dict(config))


def test_epilogue_search_and_roundtrip() -> None:
    arguments = _residual_inputs(torch.float16, torch.float16, False)
    bound = _cpu_bind(_residual_matmul, arguments)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert {"vector", "vector_unrolled"} <= {
            seed.get("cute_collective_epilogue") for seed in seeds
        }
        generation = ConfigGeneration(bound.config_spec)
        for strategy in ("scalar", "vector", "vector_unrolled"):
            flat, recovered = generation.canonicalize_flat(
                generation.flatten(_residual_config(strategy, "warp", False))
            )
            assert recovered["cute_collective_epilogue"] == strategy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("strategy", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("dtype,out_dtype,compute,unaligned,n_first", _CASES)
def test_residual_epilogue_cuda_tails_and_mutations(
    strategy: str,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
    compute: str,
    unaligned: bool,
    n_first: bool,
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(20260913)
    arguments = _residual_inputs(dtype, out_dtype, unaligned, "cuda")
    a, b, residual, out = arguments
    for tensor in (a, b, residual):
        tensor.copy_(torch.randint(-4, 5, tensor.shape, device=CUDA_DEVICE) * 0.25)
    control = _residual_matmul._bind_isolated(arguments)
    control.set_config(_residual_config("scalar", compute, n_first))
    bound = _residual_matmul._bind_isolated(arguments)
    config = _residual_config(strategy, compute, n_first)
    assert "epilogue_vector_values" in bound.to_code(config)
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(*arguments)

    replay = _make_cudagraph_replay(run)
    for scale in (1.0, -1.0, 0.5):
        b.mul_(scale)
        residual.mul_(-1)
        expected = (torch.relu(a.float() @ b.float()) + residual.float() * 0.5).to(
            out_dtype
        )
        torch.testing.assert_close(control(*arguments), expected, rtol=0, atol=0)
        out.fill_(float("nan"))
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)
        out.fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = _residual_inputs(dtype, out_dtype, unaligned, "cuda")
    for old, new in zip(arguments[:3], replacements[:3], strict=True):
        new.copy_(old)
    replacements[-1].fill_(float("nan"))
    torch.testing.assert_close(bound(*replacements), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("strategy", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_mamba_epilogue_cuda_mutations(
    strategy: str, dtype: torch.dtype, static_shapes: bool
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(20260913)
    arguments = tuple(
        torch.randn(size, device=CUDA_DEVICE, dtype=torch.bfloat16).to(dtype)
        for size in _mamba_sizes(2)
    )
    arguments[2].uniform_()
    arguments[3].copy_(-torch.rand_like(arguments[2]).cumsum(-1))
    kernel = _mamba_kernel(static_shapes=static_shapes)
    reference = kernel._bind_isolated(arguments)
    reference.set_config(reference.config_spec.autotune_reference_config())
    control = kernel._bind_isolated(arguments)
    control.set_config(_mamba_config("scalar"))
    bound = kernel._bind_isolated(arguments)
    config = _mamba_config(strategy)
    assert "epilogue_vector_values" in bound.to_code(config)
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(*arguments)

    replay = _make_cudagraph_replay(run)
    for factor in (1.0, 0.875, 1.125):
        arguments[1].mul_(factor)
        arguments[5].mul_(factor)
        expected = control(*arguments)
        _chunked_assert_close(
            expected,
            reference(*arguments),
            atol=0.01,
            rtol=0.01,
            scale_atol_by_expected_rms=True,
        )
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)
        output = replay()
        output.fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = tuple(value.clone() for value in arguments)
    torch.testing.assert_close(
        bound(*replacements), control(*replacements), rtol=0, atol=0
    )
