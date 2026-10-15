from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.matmul import matmul
import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics import compiler_promotion_specialization_key
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.matmul_utils import cute_matmul_root_placements
from helion._hardware import HardwareInfo
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import KernelGridFact
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import RootGridFact
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _independent_matmuls(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    second = torch.empty((c.size(0), d.size(1)), device=c.device, dtype=c.dtype)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for contraction in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[row, contraction] * 0.5, b[contraction, column])
        first[row, column] = acc.to(first.dtype)
    for row2, column2 in hl.tile([c.size(0), d.size(1)]):
        acc2 = hl.zeros([row2, column2], dtype=torch.float32)
        for contraction2 in hl.tile(c.size(1)):
            acc2 = torch.addmm(
                acc2, c[row2, contraction2] * 0.5, d[contraction2, column2]
            )
        second[row2, column2] = acc2.to(second.dtype)
    return first, second


@skipUnlessBackends(["cute"])
def test_independent_contractions_receive_complete_collective_seeds() -> None:
    inputs = tuple(
        torch.empty(shape, dtype=torch.float16)
        for shape in ((128, 256), (256, 64), (96, 64), (64, 256))
    )
    with ExitStack() as stack:
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
        ):
            stack.enter_context(patch(name, return_value=value))
        bound = _independent_matmuls._bind_isolated(inputs)
        assert bound.host_function is not None
        with bound.env:
            seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        assert seeds
        assert bound.to_code(seeds[0]).count("cute.gemm(") == 2
        assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
        spec = bound.config_spec
        assert len(spec.matmul_facts) == 2
        for seed in seeds:
            assert len(seed.block_sizes) == len(spec.block_sizes) == 6
            assert len(seed.num_threads) == len(spec.num_threads)
            widths = seed["cute_vector_widths"]
            assert isinstance(widths, list)
            assert len(widths) == len(spec.cute_vector_widths)
            assert seed["cute_collective_mma"]
            for fact in spec.matmul_facts:
                n = fact.n_block_id
                m = fact.m_block_id
                k = fact.k_block_id
                assert m is not None and n is not None and k is not None
                bn = spec.block_sizes.config_get(seed.block_sizes, n)
                assert bn is not None
                assert spec.num_threads.config_get(seed.num_threads, m) == 128 // bn
                assert spec.num_threads.config_get(seed.num_threads, n) == bn
                assert spec.num_threads.config_get(seed.num_threads, k) == 1


def test_incomplete_attribution_does_not_seed_independent_roots() -> None:
    # Attribution is a correctness fact; a same-length list of unpaired dots
    # must never be guessed to correspond to root grids in source order.
    env = SimpleNamespace(
        config_spec=SimpleNamespace(
            matmul_facts=[object(), object()],
            kernel_matmul_fact=SimpleNamespace(attribution_complete=False),
            kernel_grid_fact=object(),
        )
    )
    ir = SimpleNamespace(grid_block_ids=[[0, 1], [3, 4]])
    assert (
        CuteCollectiveMatmulHeuristic._axes(
            cast("CompileEnvironment", env), cast("DeviceIR", ir)
        )
        == ()
    )


def _placement_env() -> CompileEnvironment:
    first = MatmulFact(2, 2, 0, 1, 2, 128, 64, 256, torch.float16, torch.float16)
    second = MatmulFact(2, 2, 3, 4, 5, 96, 256, 64, torch.float16, torch.float16)
    # Reverse the resolved list: placement must follow graph attribution.
    return cast(
        "CompileEnvironment",
        SimpleNamespace(
            config_spec=SimpleNamespace(
                matmul_facts=[first, second],
                kernel_matmul_fact=SimpleNamespace(
                    attribution_complete=True,
                    matmuls=(
                        SimpleNamespace(fact=second, site=SimpleNamespace(graph_id=20)),
                        SimpleNamespace(fact=first, site=SimpleNamespace(graph_id=10)),
                    ),
                ),
                kernel_grid_fact=KernelGridFact(
                    (RootGridFact(1, (0, 1)), RootGridFact(2, (3, 4))),
                    ((10, 1), (20, 2)),
                ),
            )
        ),
    )


@pytest.mark.parametrize("active", [[[0, 1], [3, 4]], [[0, 1]], [[3, 4]]])
def test_matmul_placement_uses_graph_attribution_and_active_roots(
    active: list[list[int]],
) -> None:
    env = _placement_env()
    ir = cast("DeviceIR", SimpleNamespace(grid_block_ids=active))
    placed = cute_matmul_root_placements(env, ir)
    assert len(placed) == len(active)
    for fact, root in placed:
        assert list(root) in active
        assert root == (fact.m_block_id, fact.n_block_id)


@pytest.mark.parametrize("unknown", ["graph", "active_root"])
def test_matmul_placement_rejects_missing_attribution(unknown: str) -> None:
    env = _placement_env()
    if unknown == "graph":
        env.config_spec.kernel_grid_fact = KernelGridFact(
            (RootGridFact(1, (0, 1)), RootGridFact(2, (3, 4))), ((10, 1),)
        )
        active = [[0, 1], [3, 4]]
    else:
        active = [[0, 1], [6, 7]]
    ir = cast("DeviceIR", SimpleNamespace(grid_block_ids=active))
    assert cute_matmul_root_placements(env, ir) == ()


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("hardware_name", "capability", "dtype", "m", "preferred"),
    [
        ("NVIDIA H100 PCIe", (9, 0), torch.float16, 128, True),
        ("NVIDIA H100 PCIe", (9, 0), torch.float16, 129, True),
        ("NVIDIA H100 PCIe", (9, 0), torch.float16, 32, False),
        ("NVIDIA H100 PCIe", (9, 0), torch.bfloat16, 128, False),
        ("NVIDIA H200", (9, 0), torch.float16, 128, False),
        ("NVIDIA B200", (10, 0), torch.float16, 128, False),
    ],
)
def test_collective_async_priority_preserves_candidates_and_default(
    hardware_name: str,
    capability: tuple[int, int],
    dtype: torch.dtype,
    m: int,
    preferred: bool,
) -> None:
    hardware = HardwareInfo(
        "cuda", hardware_name, "13.0", f"sm{capability[0]}{capability[1]}"
    )
    kernel = helion.kernel(
        matmul.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (torch.empty((m, 256), dtype=dtype), torch.empty((256, 64), dtype=dtype))
    heuristic = CuteCollectiveMatmulHeuristic
    with ExitStack() as stack:
        stack.enter_context(_mock_cuda_unavailable())
        stack.enter_context(_forbid_native_compile())
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden"))
        )
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", capability),
            (
                "helion._compiler.compile_environment.target_device_capability",
                capability,
            ),
            (
                "helion.runtime.get_num_sm",
                {"NVIDIA H100 PCIe": 114, "NVIDIA H200": 132, "NVIDIA B200": 148}[
                    hardware_name
                ],
            ),
            ("helion._hardware.get_hardware_info", hardware),
        ):
            stack.enter_context(patch(name, return_value=value))
        bound = kernel._bind_isolated(args)
        assert bound.host_function is not None
        with bound.env:
            ir = bound.host_function.device_ir
            actual = heuristic.get_seed_configs(bound.env, ir)
            default = bound.config_spec.default_config()
            assert bound.config_spec.compiler_default_config is None
            with patch.object(heuristic, "CACHE_NAMED_TARGETS", frozenset()):
                original = heuristic.get_seed_configs(bound.env, ir)
                assert bound.config_spec.default_config() == default
            assert original[0]["cute_collective_copy"] == "scalar"
            if preferred:
                async_seed = helion.Config.from_dict(
                    original[0].config | {"cute_collective_copy": "async"}
                )
                assert actual == [async_seed, *[s for s in original if s != async_seed]]
            else:
                assert actual == original
            assert not heuristic.promote_seed_to_default
            with patch.object(bound.config_spec, "compiler_seed_configs", actual):
                normalized = ConfigGeneration(
                    bound.config_spec
                ).seed_flat_config_pairs()
            assert normalized[0][1]["cute_collective_copy"] == (
                "async" if preferred else "scalar"
            )
        key = dict(compiler_promotion_specialization_key("cute", torch.device("cuda")))
        assert key[heuristic.name] == (
            hardware_name if hardware_name == "NVIDIA H100 PCIe" else None
        )
