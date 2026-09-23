from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.mamba2_chunk_scan import helion_mamba2_chunk_scan_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_matmul import _is_scalar_carry_alias
from helion._compiler.cute.collective_matmul import _remove_dead_staged_operands
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _seeded_dot(a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = seed[tile_m, tile_n].float() * 0.75
        for tile_k in hl.tile(k):
            left = a[tile_m, tile_k] * 0.5
            acc = hl.dot(left, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _invariant_seed_dot(
    a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        fixed = seed[tile_m, tile_n].float()
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            left = a[tile_m, tile_k] * 0.5
            acc = hl.dot(left, b[tile_k, tile_n], acc=fixed)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _full_slice_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        left = a[tile_m, :] * 0.5
        acc = hl.dot(left, b[:, tile_n], acc=acc)
        out[tile_m, tile_n] = torch.relu(acc).to(out.dtype)
    return out


def _config(*, full_slice: bool = False) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 32] if full_slice else [64, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
    )


def _seed_inputs(dtype: torch.dtype, device: str = "cpu") -> tuple[torch.Tensor, ...]:
    return (
        torch.randn((130, 78), dtype=dtype, device=device),
        torch.randn((78, 70), dtype=dtype, device=device),
        torch.randn((130, 70), dtype=dtype, device=device),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_explicit_dot_stages_original_fp32_seed(dtype: torch.dtype) -> None:
    inputs = _seed_inputs(dtype)
    source = _cpu_bind(_seeded_dot, inputs).to_code(_config())
    ast.parse(source)
    assert source.count("cute.gemm(") == 1
    assert "_acc.fill(0.0)" not in source
    assert "cute.autovec_copy(collective_seed, collective_acc)" in source
    assert "seed.iterator" in source
    assert source.index("collective_seed =") < source.index("cute.gemm(")
    assert "dot_acc_base" not in source
    assert "_helion_pending_collective_mma" not in source


@skipUnlessBackends(["cute"])
def test_invariant_seed_is_not_a_loop_carry() -> None:
    inputs = _seed_inputs(torch.float16)
    source = _cpu_bind(_invariant_seed_dot, inputs).to_code(_config())
    assert "cute.gemm(" not in source
    assert "_helion_pending_collective_mma" not in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [5, 21, 32, 128])
@skipUnlessBackends(["cute"])
def test_full_slice_dot_owns_entire_synthetic_k(dtype: torch.dtype, k: int) -> None:
    inputs = (
        torch.empty((130, k), dtype=dtype),
        torch.empty((k, 70), dtype=dtype),
    )
    source = _cpu_bind(_full_slice_dot, inputs).to_code(_config(full_slice=True))
    assert source.count("cute.gemm(") == 1
    assert "synthetic_lane_" not in source
    assert "_helion_pending_collective_mma" not in source
    if k == 21:
        assert "collective_k_offset + collective_k < 21" in source
        assert "collective_a[collective_m, collective_k] =" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("collective_requested", [False, True])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_partial_synthetic_k_cannot_fall_back_to_a_thread_subset(
    dtype: torch.dtype, collective_requested: bool, static_shapes: bool
) -> None:
    inputs = (
        torch.empty((130, 64), dtype=dtype),
        torch.empty((64, 70), dtype=dtype),
    )
    config = dict(_config(full_slice=True))
    config["cute_collective_mma"] = collective_requested
    if collective_requested:
        # Vectorized operands do not satisfy the collective staging proof.
        # A requested-but-ineligible collective must reject the same unsafe
        # scalar fallback as a configuration that disables it explicitly.
        config["cute_vector_widths"] = [2, 1, 1]
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        kernel = helion.kernel(
            _full_slice_dot.fn,
            backend="cute",
            static_shapes=static_shapes,
            autotune_effort="none",
        )
        bound = _cpu_bind(kernel, inputs)
        with pytest.raises(exc.BackendUnsupported, match="K axis split across"):
            bound.to_code(helion.Config.from_dict(config))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_full_thread_k_keeps_the_complete_scalar_dot(
    dtype: torch.dtype, static_shapes: bool
) -> None:
    inputs = (
        torch.empty((130, 64), dtype=dtype),
        torch.empty((64, 70), dtype=dtype),
    )
    config = helion.Config(
        block_sizes=[16, 16],
        num_threads=[1, 1, 64],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=False,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        kernel = helion.kernel(
            _full_slice_dot.fn,
            backend="cute",
            static_shapes=static_shapes,
            autotune_effort="none",
        )
        source = _cpu_bind(kernel, inputs).to_code(config)
    assert "synthetic_lane_" not in source
    assert "cute.gemm(" not in source
    assert "group_span=64" in source


def test_scalar_carry_proof_allows_only_direct_copies() -> None:
    value = ast.Name(id="copy2", ctx=ast.Load())
    assert _is_scalar_carry_alias(
        value, "acc", ast.parse("copy1 = acc\ncopy2 = copy1").body
    )
    assert not _is_scalar_carry_alias(
        value, "acc", ast.parse("copy1 = acc + 1\ncopy2 = copy1").body
    )
    assert not _is_scalar_carry_alias(
        value, "acc", ast.parse("copy1 = other\ncopy2 = copy1").body
    )


def test_staged_dce_preserves_effectful_rebinding_with_the_same_name() -> None:
    body = ast.parse(
        "value = (a.iterator + index).load()\nvalue = opaque_effect()"
    ).body
    df = SimpleNamespace(
        dce_vars=[],
        cute_state=SimpleNamespace(
            collective_mma_pure_stmt_ids={id(body[0])}, collective_mma_sites=[]
        ),
    )
    _remove_dead_staged_operands(body, cast("DeviceFunction", df))
    assert (
        ast.unparse(ast.Module(body=body, type_ignores=[])) == "value = opaque_effect()"
    )


@pytest.mark.parametrize("shape", [0, 1, 2])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_mamba_contractions_receive_complete_ordinary_seeds(
    shape: int, static_shapes: bool
) -> None:
    batch, heads, groups, sequence, chunk, head, state = (
        (1, 4, 1, 256, 64, 32, 32),
        (2, 8, 2, 1024, 64, 64, 64),
        (2, 16, 4, 2048, 128, 64, 128),
    )[shape]
    chunks = sequence // chunk
    inputs = tuple(
        torch.empty(size, dtype=torch.bfloat16)
        for size in (
            (batch, chunks, groups, chunk, chunk),
            (batch, sequence, heads, head),
            (batch, heads, chunks, chunk),
            (batch, heads, chunks, chunk),
            (batch, sequence, groups, state),
            (batch, chunks, heads, head, state),
            (heads,),
        )
    )
    kernel = helion.kernel(
        helion_mamba2_chunk_scan_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = _cpu_bind(kernel, inputs)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    for seed in seeds:
        assert len(seed.block_sizes) == len(bound.config_spec.block_sizes)
        assert len(seed.num_threads) == len(bound.config_spec.num_threads)
        widths = seed["cute_vector_widths"]
        assert isinstance(widths, list)
        assert len(widths) == len(bound.config_spec.cute_vector_widths)
    source = bound.to_code(seeds[0])
    device_function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert all(argument.arg != "_" for argument in device_function.args.args)
    assert not any(
        isinstance(node, ast.Name) and node.id == "_"
        for node in ast.walk(device_function)
    )
    assert source.count("cute.gemm(") == 2
    assert "synthetic_lane_" not in source
    assert "cute.autovec_copy(collective_1_seed, collective_1_acc)" in source
    assert "collective_1_c[collective_1_m, collective_1_n] = cutlass.Float32(" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_seeded_dot_correctness(dtype: torch.dtype) -> None:
    torch.manual_seed(173)
    inputs = _seed_inputs(dtype, "cuda")
    bound = _seeded_dot._bind_isolated(inputs)
    bound.set_config(_config())
    a, b, seed = inputs
    expected = (torch.mm((a * 0.5).float(), b.float()) + seed.float() * 0.75).to(dtype)
    torch.testing.assert_close(bound(*inputs), expected, atol=0.02, rtol=0.02)
