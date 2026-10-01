from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes
from test.test_cute_collective_tmem_seed import _dependent_config
from test.test_cute_collective_tmem_seed import _dependent_contractions
from test.test_cute_collective_tmem_seed import _dependent_inputs
from test.test_cute_collective_vector_epilogue_codegen import _mamba_config

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


def _config(m: int = 64, n: int = 64) -> helion.Config:
    return helion.Config.from_dict(
        _mamba_config("vector_unrolled").config
        | {
            "block_sizes": [m, n, 64],
            "num_threads": [128 // n, n, 1, 1],
            "cute_collective_tmem_seed": True,
            "cute_proven_bounds": True,
        }
    )


@pytest.mark.parametrize("static_shapes", [True, False])
@pytest.mark.parametrize(("m", "n"), [(64, 32), (64, 64), (128, 64)])
def test_original_dependent_contraction_codegen(
    static_shapes: bool, m: int, n: int
) -> None:
    args = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2))
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), args)
    enabled = bound.to_code(_config(m, n))
    control = bound.to_code(
        helion.Config.from_dict(_config(m, n).config | {"cute_proven_bounds": False})
    )
    assert "_cute_proven_loop_bounds_" in enabled
    assert "_cute_proven_loop_bounds_" not in control
    assert "collective_tmem_seed_values" in enabled
    assert sum(
        isinstance(node, ast.Compare) for node in ast.walk(ast.parse(enabled))
    ) < sum(isinstance(node, ast.Compare) for node in ast.walk(ast.parse(control)))
    if static_shapes:
        assert "recipe_tail_lane" not in enabled


def test_search_exposes_proved_bounds_with_complete_collective_configuration() -> None:
    args = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2))
    bound = _cpu_bind(_mamba_kernel(static_shapes=True), args)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        candidates = [
            seed
            for seed in seeds
            if seed.get("cute_proven_bounds") and seed.get("cute_collective_tmem_seed")
        ]
        assert candidates
        generator = ConfigGeneration(bound.env.config_spec)
        candidate = candidates[0]
        flattened = generator.flatten(candidate)
        restored = generator.unflatten(flattened)
        assert generator.flatten(restored) == flattened
        assert all(
            restored.get(key) == value for key, value in candidate.config.items()
        )
    assert "_cute_proven_loop_bounds_" in bound.to_code(candidate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("static_shapes", [True, False])
@pytest.mark.parametrize("shape", [0, 1, 2])
def test_mamba_cuda_proved_bounds_preserve_exact_outputs(
    shape: int, static_shapes: bool
) -> None:
    torch.manual_seed(20260913)
    sizes = (
        (
            (1, 4, 1, 64, 64),
            (1, 256, 4, 32),
            (1, 4, 4, 64),
            (1, 4, 4, 64),
            (1, 256, 1, 32),
            (1, 4, 4, 32, 32),
            (4,),
        )
        if shape == 0
        else _mamba_sizes(shape)
    )
    args = tuple(
        torch.randn(size, device=CUDA_DEVICE, dtype=torch.bfloat16) for size in sizes
    )
    args[2].uniform_()
    args[3].copy_(-torch.rand_like(args[2]).cumsum(-1))
    kernel = _mamba_kernel(static_shapes=static_shapes)
    config = _config(n=32 if shape == 0 else 64)
    candidate = kernel._bind_isolated(args)
    control = kernel._bind_isolated(args)
    candidate.set_config(config)
    control.set_config(
        helion.Config.from_dict(config.config | {"cute_proven_bounds": False})
    )
    replay = _make_cudagraph_replay(lambda: candidate(*args))
    for factor in (1.0, 0.875, 1.125):
        args[1].mul_(factor)
        args[5].mul_(factor)
        expected = control(*args)
        torch.testing.assert_close(candidate(*args), expected, rtol=0, atol=0)
        replay().fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = tuple(value.clone() for value in args)
    torch.testing.assert_close(
        candidate(*replacements), control(*replacements), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("m", "n"), [(64, 32), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_memory_derived_k_tails_and_empty_ranges_stay_guarded(
    m: int, n: int, dtype: torch.dtype
) -> None:
    args = _dependent_inputs(dtype, "cuda")
    a, b, scale, lengths = args
    torch.manual_seed(916)
    for value in (a, b, scale):
        value.copy_(torch.randint(-4, 5, value.shape, device=CUDA_DEVICE) * 0.25)
    lengths.fill_(78)
    config = helion.Config.from_dict(
        _dependent_config(m, n, True).config | {"cute_proven_bounds": True}
    )
    candidate = _dependent_contractions._bind_isolated(args)
    candidate.set_config(config)
    replay = _make_cudagraph_replay(lambda: candidate(*args))
    for first, second in ((78, 78), (0, 78), (78, 0), (0, 0), (17, 37), (78, 78)):
        lengths.copy_(
            torch.tensor([first, second], device=CUDA_DEVICE, dtype=torch.int32)
        )
        a.mul_(-1)
        scale.mul_(-1)
        expected = (a[:, :first].float() @ b[:, :first].float().T) * scale[:, None]
        expected += (a[:, :second].float() * 0.5) @ b[:, :second].float().T
        torch.testing.assert_close(candidate(*args), expected, rtol=0, atol=0)
        replay().fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = _dependent_inputs(dtype, "cuda")
    for old, new in zip(args, replacements, strict=True):
        new.copy_(old)
    torch.testing.assert_close(candidate(*replacements), expected, rtol=0, atol=0)
