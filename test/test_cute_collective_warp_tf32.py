"""Warp TF32 staging, precision policy, and shared-ring regressions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_collective_tf32 import _config
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel
from test.test_cute_collective_tf32 import _round_tf32

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@helion.kernel(
    backend="cute",
    static_shapes=False,
    autotune_effort="none",
    dot_precision="tf32",
)
def _seeded_fp32(a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = seed[row, column]
        for inner in hl.tile(k):
            acc = hl.dot(a[row, inner], b[inner, column], acc=acc)
        output[row, column] = acc
    return output


@pytest.mark.parametrize("tile", [(32, 32, 16), (64, 64, 64), (128, 64, 128)])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_computed_tail_uses_tf32_warp_fragments(
    cpu_only: None, tile: tuple[int, int, int], static_shapes: bool
) -> None:
    bound = _cpu_bind(
        _kernel(batched=True, static_shapes=static_shapes),
        (torch.empty((3, 130, 78)), torch.empty((3, 78, 70))),
    )
    source = bound.to_code(_config(*tile, compute="warp", copy="async_cached"))
    assert "warp.MmaTF32Op((16, 8, 8))" in source
    assert "CopyUniversalOp(), cutlass.TFloat32, num_bits_per_copy=32" in source
    assert "cvt_f32_tf32" in source
    assert "LdMatrix8x8x16bOp" not in source
    assert "MmaF16BF16Op" not in source and "TmemAllocator" not in source
    assert "cutlass.Float16" not in source and "cutlass.BFloat16" not in source
    assert "mask_1" in source and "mask_2" in source and "mask_3" in source


@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
def test_warp_precision_policy_keeps_fp32_fallback(
    cpu_only: None, precision: str
) -> None:
    bound = _cpu_bind(
        _kernel(precision=precision),
        _jagged_inputs(batch=3, rows=137, k=64, n=64),
    )
    source = bound.to_code(_config(32, 32, 32, compute="warp"))
    assert "MmaTF32Op" not in source and "cvt_f32_tf32" not in source
    assert "cute.gemm(" not in source
    assert "cutlass.Float16" not in source and "cutlass.BFloat16" not in source


@pytest.mark.parametrize(
    "capability,enabled", [((7, 5), False), ((8, 0), True), ((9, 0), True)]
)
def test_warp_tf32_requires_sm80_or_newer(
    cpu_only: None, capability: tuple[int, int], enabled: bool
) -> None:
    bound = _cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=137, k=64, n=64))
    bound.config_spec.target_device_capability = capability
    source = bound.to_code(_config(32, 32, 32, compute="warp"))
    assert ("MmaTF32Op" in source) is enabled
    with bound.env:
        assert bound.host_function is not None
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert bool(seeds) is enabled
    assert all(seed.get("cute_collective_compute", "warp") == "warp" for seed in seeds)


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_async_tf32_rounds_only_the_waited_shared_stage(
    cpu_only: None, stages: int
) -> None:
    arguments = _jagged_inputs(batch=3, rows=137, k=128, n=64)
    bound = _cpu_bind(_kernel(), arguments)
    config = _config(32, 32, 32, compute="warp", copy="async_cached")
    config.config["cute_collective_stages"] = stages
    source = bound.to_code(config)
    tree = ast.parse(source)
    conversions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.cvt_f32_tf32"
    ]
    assert len(conversions) == 2
    for conversion in conversions:
        assert isinstance(conversion.args[0], ast.Subscript)
        index = conversion.args[0].slice
        assert isinstance(index, ast.Tuple) and len(index.elts) == 3
        assert isinstance(index.elts[-1], ast.Name)
        assert index.elts[-1].id.endswith("_stage_read")
    assert f"cute.arch.cp_async_wait_group({stages - 1})" in source
    assert "cute.arch.cp_async_wait_group(0)" in source
    assert (
        "cute.recast_ptr(collective_ap, collective_as, dtype=cutlass.Float32)" in source
    )
    assert (
        "cute.recast_ptr(collective_bp, collective_bs, dtype=cutlass.Float32)" in source
    )
    for operand in ("a", "b"):
        assert (
            f"collective_{operand}s = cute.recast_layout(8, cutlass.TFloat32.width, collective_{operand}l).inner"
            in source
        )
        assert (
            f"cute.recast_ptr(collective_{operand}p, collective_{operand}s, dtype=cutlass.TFloat32)"
            in source
        )
    async_atoms = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.make_copy_atom"
        and isinstance(node.args[0], ast.Call)
        and ast.unparse(node.args[0].func) == "cute.nvgpu.cpasync.CopyG2SOp"
    ]
    assert async_atoms
    assert all(
        any(
            keyword.arg == "num_bits_per_copy"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 128
            for keyword in node.keywords
        )
        for node in async_atoms
    )
    assert "cp_async_shared_global" not in source
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        normalized = bound._normalized_config_copy(config)
        roundtrip = generation.unflatten(generation.flatten(normalized))
        assert all(
            roundtrip.config[key] == value for key, value in normalized.config.items()
        )
    assert "warp.MmaTF32Op((16, 8, 8))" in bound.to_code(roundtrip)


def test_fp32_seed_uses_exact_register_accumulator(cpu_only: None) -> None:
    arguments = (torch.empty((65, 48)), torch.empty((48, 70)), torch.empty((65, 70)))
    bound = _cpu_bind(_seeded_fp32, arguments)
    config = helion.Config(
        block_sizes=[32, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute="warp",
    )
    source = bound.to_code(config)
    assert "MmaTF32Op" in source
    assert "collective_seed = collective_thr.partition_C(collective_c)" in source
    assert "cute.autovec_copy(collective_seed, collective_acc)" in source
    assert source.count("cvt_f32_tf32") == 2
    assert "TmemAllocator" not in source


def _require_sm80() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80 or newer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tile,copy,layout",
    [
        ((32, 32, 16), "scalar", "unaligned"),
        ((32, 32, 32), "async_cached", "n_major"),
        ((64, 64, 64), "async_cached", "k_major"),
        ((128, 64, 128), "scalar", "n_major"),
        ((128, 64, 128), "async_cached", "k_major"),
        ((64, 32, 64), "async_cached", "unaligned"),
    ],
)
@pytest.mark.parametrize("static_shapes", [False, True])
def test_warp_tf32_computed_tail_and_mutating_graph(
    tile: tuple[int, int, int],
    copy: str,
    layout: str,
    static_shapes: bool,
) -> None:
    _require_sm80()
    torch.manual_seed(1719)
    a = torch.randn((3, 130, 78), device=CUDA_DEVICE)
    if layout == "k_major":
        b = torch.randn((3, 70, 80), device=CUDA_DEVICE)[:, :, :78].transpose(1, 2)
    elif layout == "n_major":
        b = torch.randn((3, 78, 72), device=CUDA_DEVICE)[:, :, :70]
    else:
        b = torch.randn(3 * 78 * 70 + 1, device=CUDA_DEVICE)[1:].view(3, 78, 70)
    kernel = _kernel(batched=True, static_shapes=static_shapes)
    bound = kernel.bind((a, b))
    config = _config(*tile, compute="warp", copy=copy)
    source = bound.to_code(config)
    assert "MmaTF32Op" in source and "TmemAllocator" not in source
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(a, b)

    def expected() -> torch.Tensor:
        return (
            torch.bmm(_round_tf32(a * 0.75).double(), _round_tf32(b).double()).float()
            + 0.25
        )

    torch.testing.assert_close(run(), expected(), rtol=2e-4, atol=2e-4)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(replay(), expected(), rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tile,stages",
    [
        ((32, 32, 32), 1),
        ((32, 32, 32), 2),
        ((32, 32, 32), 3),
        ((32, 32, 32), 4),
        ((64, 64, 64), 2),
        ((128, 64, 128), 2),
    ],
)
def test_warp_tf32_jagged_pipeline_reuse_and_mutated_offsets(
    tile: tuple[int, int, int], stages: int
) -> None:
    _require_sm80()
    torch.manual_seed(20260912)
    arguments = _jagged_inputs(batch=3, rows=137, k=256, n=64, device=CUDA_DEVICE)
    offsets, a, b, bias = arguments
    lengths = torch.tensor([0, 67, 70], dtype=torch.int64)
    kernel = _kernel()
    bound = kernel.bind(arguments)
    config = _config(*tile, compute="warp", copy="async_cached")
    config.config["cute_collective_stages"] = stages
    source = bound.to_code(config)
    assert "MmaTF32Op" in source and "TmemAllocator" not in source
    bound.set_config(config)

    def mutate() -> None:
        nonlocal lengths
        lengths = lengths.roll(1)
        offsets[0] = 0
        offsets[1:].copy_(lengths.cumsum(0))
        a.normal_()
        b.normal_()
        bias.normal_()

    def expected() -> torch.Tensor:
        left, right = _round_tf32(a).double(), _round_tf32(b).double()
        pieces = []
        begin = 0
        for group, count in enumerate(lengths.tolist()):
            end = begin + count
            pieces.append((left[begin:end] @ right[group]).float() + bias[group])
            begin = end
        return torch.cat(pieces)

    def run() -> torch.Tensor:
        return bound(*arguments)

    mutate()
    torch.testing.assert_close(run(), expected(), rtol=2e-4, atol=2e-4)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        mutate()
        torch.testing.assert_close(replay(), expected(), rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("k", [0, 48])
def test_warp_tf32_keeps_fp32_seed_without_half_conversion(k: int) -> None:
    _require_sm80()
    torch.manual_seed(872)
    a = torch.randn((65, k), device=CUDA_DEVICE) * 1e5
    b = torch.randn((k, 70), device=CUDA_DEVICE) * 1e-5
    seed = torch.full((65, 70), 0.12345679, device=CUDA_DEVICE)
    bound = _seeded_fp32.bind((a, b, seed))
    config = helion.Config(
        block_sizes=[32, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute="warp",
    )
    source = bound.to_code(config)
    assert "MmaTF32Op" in source
    bound.set_config(config)

    def expected() -> torch.Tensor:
        return (
            _round_tf32(a).double() @ _round_tf32(b).double() + seed.double()
        ).float()

    def run() -> torch.Tensor:
        return bound(a, b, seed)

    torch.testing.assert_close(
        run(), expected(), rtol=2e-4 if k else 0, atol=2e-4 if k else 0
    )
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.normal_().mul_(1e5)
        b.normal_().mul_(1e-5)
        seed.normal_()
        torch.testing.assert_close(
            replay(), expected(), rtol=2e-4 if k else 0, atol=2e-4 if k else 0
        )
