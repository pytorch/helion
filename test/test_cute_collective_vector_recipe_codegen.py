from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_native_seeded import _config
from test.test_cute_collective_native_seeded import _inputs
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes
from test.test_cute_collective_native_seeded import _seed_written_in_place
from test.test_cute_collective_native_seeded import _seeded_contraction

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


def _recipe_config(recipe: str, *, compute: str = "tcgen05") -> helion.Config:
    return helion.Config.from_dict(
        _config(64, 32, 32, compute=compute).config | {"cute_collective_recipe": recipe}
    )


@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_original_mamba_vector_recipe_codegen(
    recipe: str, compute: str, static_shapes: bool
) -> None:
    inputs = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(2))
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), inputs)
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
    source = bound.to_code(config)
    assert "recipe_registers" in source
    assert "recipe_lane" in source
    assert source.count("mul.rn.f32") == 5
    assert source.count("tcgen05.CtaGroup.ONE") == (2 if compute == "tcgen05" else 0)
    if recipe == "vector_unrolled":
        assert "for collective_1_ai in cutlass.range_constexpr(4)" in source
    else:
        assert "for collective_1_ai in cutlass.range(4, unroll=1)" in source


def test_vector_recipe_seeds_and_config_roundtrip() -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.bfloat16))
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert {"vector", "vector_unrolled"} <= {
            seed.get("cute_collective_recipe") for seed in seeds
        }
        generation = ConfigGeneration(bound.config_spec)
        for recipe in ("scalar", "vector", "vector_unrolled"):
            _flat, recovered = generation.canonicalize_flat(
                generation.flatten(_recipe_config(recipe))
            )
            assert recovered["cute_collective_recipe"] == recipe


@pytest.mark.parametrize("recipe", [None, True, 8, "automatic"])
def test_vector_recipe_config_rejects_unknown_strategies(recipe: object) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.float16))
    config = helion.Config.from_dict(
        _config().config | {"cute_collective_recipe": recipe}
    )
    with pytest.raises(helion.exc.InvalidConfig, match="cute_collective_recipe"):
        bound.to_code(config)


@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
def test_vector_recipe_keeps_existing_alias_rejection(recipe: str) -> None:
    a, b, seed, limit = _inputs(torch.bfloat16)
    bound = _cpu_bind(_seed_written_in_place, (a, b, seed))
    with pytest.raises(
        helion.exc.BackendUnsupported, match="operand loads may alias row-loop writes"
    ):
        bound.to_code(_recipe_config(recipe))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
@pytest.mark.parametrize(
    "dtype,compute,unaligned",
    [
        (torch.float16, "warp", False),
        (torch.bfloat16, "warp", True),
        (torch.float16, "tcgen05", False),
        (torch.bfloat16, "tcgen05", True),
    ],
)
@pytest.mark.parametrize("limit", [0, 33, 78])
def test_vector_recipe_seeded_tails_and_mutating_graphs(
    recipe: str,
    dtype: torch.dtype,
    compute: str,
    unaligned: bool,
    limit: int,
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(857)
    offset = int(unaligned)
    a = torch.randn((65, 88), device=CUDA_DEVICE, dtype=dtype)[:, offset : offset + 78]
    b = torch.randn((71, 88), device=CUDA_DEVICE, dtype=dtype)[:, offset : offset + 78]
    seed = torch.randn((65, 71), device=CUDA_DEVICE, dtype=torch.float32)
    inputs = (a, b, seed, limit)
    reference = _seeded_contraction._bind_isolated(inputs)
    reference.set_config(_recipe_config("scalar", compute=compute))
    bound = _seeded_contraction._bind_isolated(inputs)
    bound.set_config(_recipe_config(recipe, compute=compute))

    def run() -> torch.Tensor:
        return bound(*inputs)

    replay = _make_cudagraph_replay(run)
    for factor in (1.0, 0.5, -1.0):
        a.mul_(factor)
        b.mul_(0.75)
        seed.mul_(0.5)
        expected = seed + (a[:, :limit] * 0.5).float() @ b[:, :limit].float().T
        control = reference(*inputs)
        torch.testing.assert_close(control, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(run(), control, rtol=0, atol=0)
        torch.testing.assert_close(replay(), control, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_original_mamba_vector_recipe_mutating_graphs(
    recipe: str, dtype: torch.dtype, static_shapes: bool
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    sizes = _mamba_sizes(2)
    torch.manual_seed(20260913)
    arguments = tuple(
        torch.randn(size, device=CUDA_DEVICE, dtype=torch.bfloat16).to(dtype)
        for size in sizes
    )
    arguments[2].uniform_()
    arguments[3].copy_(-torch.rand_like(arguments[2]).cumsum(-1))
    kernel = _mamba_kernel(static_shapes=static_shapes)
    config = helion.Config(
        block_sizes=[64, 64, 64],
        num_threads=[2, 64, 1, 1],
        cute_vector_widths=[1] * 7,
        cute_lane_layouts=["blocked"] * 7,
        cute_collective_mma=True,
        cute_collective_compute="tcgen05",
        cute_collective_copy="async_cached",
        cute_collective_native_seeded=True,
        cute_collective_recipe="scalar",
    )
    reference = kernel._bind_isolated(arguments)
    reference.set_config(reference.config_spec.autotune_reference_config())
    control = kernel._bind_isolated(arguments)
    control.set_config(config)
    bound = kernel._bind_isolated(arguments)
    bound.set_config(
        helion.Config.from_dict(config.config | {"cute_collective_recipe": recipe})
    )

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
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = tuple(value.clone() for value in arguments)
    torch.testing.assert_close(
        bound(*replacements), control(*replacements), rtol=0, atol=0
    )
