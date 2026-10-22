from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.matmul import matmul
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_matmul import _is_direct_zero_seed
from helion._compiler.cute.collective_matmul import has_collective_native_seed_candidate
from helion._compiler.cute.matmul_utils import cute_matmul_root_placements
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import KernelGridFact
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import RootGridFact
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel


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


def _admitted_inputs() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=torch.float16)
        for shape in ((128, 256), (256, 64), (96, 64), (64, 256))
    )


@skipUnlessBackends(["cute"])
def test_independent_contractions_receive_complete_collective_seeds() -> None:
    bound = _cpu_bind(_independent_matmuls, _admitted_inputs())
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
    ("capability", "num_sm", "dtype", "m", "preferred"),
    [
        ((8, 0), 108, torch.float16, 128, True),
        ((8, 9), 128, torch.float16, 128, True),
        ((9, 0), 114, torch.float16, 128, True),
        ((9, 0), 114, torch.float16, 129, True),
        ((9, 0), 114, torch.float16, 32, False),
        ((9, 0), 114, torch.bfloat16, 128, False),
        ((10, 0), 148, torch.float16, 128, True),
        ((10, 0), 148, torch.bfloat16, 128, False),
    ],
)
def test_collective_async_priority_preserves_candidates_and_default(
    capability: tuple[int, int],
    num_sm: int,
    dtype: torch.dtype,
    m: int,
    preferred: bool,
) -> None:
    kernel = helion.kernel(
        matmul.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (torch.empty((m, 256), dtype=dtype), torch.empty((256, 64), dtype=dtype))
    heuristic = CuteCollectiveMatmulHeuristic
    original: list[helion.Config] = []
    interleave = heuristic._interleave_bounded_k_seeds

    def capture_order(
        spec: ConfigSpec, seeds: list[helion.Config], parameters: tuple[str, ...]
    ) -> list[helion.Config]:
        result = interleave(spec, seeds, parameters)
        original[:] = result
        return result

    with (
        _mock_cuda_unavailable(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(kernel, args, capability=capability, num_sm=num_sm)
        assert bound.host_function is not None
        with bound.env:
            ir = bound.host_function.device_ir
            default = bound.config_spec.default_config()
            assert bound.config_spec.compiler_default_config is None
            with patch.object(
                heuristic, "_interleave_bounded_k_seeds", side_effect=capture_order
            ):
                actual = heuristic.get_seed_configs(bound.env, ir)
            assert bound.config_spec.default_config() == default
            assert original[0]["cute_collective_copy"] == "scalar"
            if preferred:
                async_seed = helion.Config.from_dict(
                    original[0].config | {"cute_collective_copy": "async"}
                )
                assert actual == [async_seed, *[s for s in original if s != async_seed]]
                if capability[0] == 10:
                    assert (
                        helion.Config.from_dict(
                            original[0].config
                            | {
                                "cute_collective_copy": "async_cached",
                                "cute_collective_compute": "tcgen05",
                            }
                        )
                        in actual
                    )
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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reverse_registered_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    k = hl.register_block_size(a.size(1))
    n = hl.register_block_size(b.size(1))
    m = hl.register_block_size(a.size(0))
    out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    for rows, columns in hl.tile([a.size(0), b.size(1)], block_size=[m, n]):
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(a.size(1), block_size=k):
            acc = hl.dot(a[rows, inner], b[inner, columns], acc)
        out[rows, columns] = acc.to(out.dtype)
    return out


@pytest.mark.parametrize("missing_async", [False, True])
def test_async_priority_keeps_excluded_candidates_in_order(missing_async: bool) -> None:
    kernel = helion.kernel(
        matmul.fn if missing_async else _reverse_registered_matmul.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    args = (
        torch.empty(128, 256, dtype=torch.float16),
        torch.empty(256, 64, dtype=torch.float16),
    )
    heuristic = CuteCollectiveMatmulHeuristic
    original: list[helion.Config] = []
    interleave = heuristic._interleave_bounded_k_seeds

    def capture(
        spec: ConfigSpec, seeds: list[helion.Config], parameters: tuple[str, ...]
    ) -> list[helion.Config]:
        result = interleave(spec, seeds, parameters)
        if missing_async:
            sibling = helion.Config.from_dict(
                result[0].config | {"cute_collective_copy": "async"}
            )
            result = [seed for seed in result if seed != sibling]
        original[:] = result
        return result

    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        assert bound.host_function is not None
        with (
            bound.env,
            patch.object(heuristic, "_interleave_bounded_k_seeds", side_effect=capture),
        ):
            actual = heuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        assert actual == original
        assert actual
        if not missing_async:
            assert actual[0].num_threads != [4, 32, 1]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _serial_state_recurrence(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    bh, n, _c, d = k.shape
    dv = v.size(3)
    h_all = torch.empty((bh, n, d, dv), dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([bh, d, dv], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i in hl.grid(n):
            h_all[idx, i, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            h_acc = hl.dot(
                k[idx, i, :, tile_d].transpose(-2, -1), v[idx, i, :, tile_dv], acc=h_acc
            )
    return h_all


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _serial_fresh_zero(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    bh, n, _c, d = k.shape
    dv = v.size(3)
    h_all = torch.empty((bh, n, d, dv), dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([bh, d, dv], block_size=[1, None, None]):
        idx = tile_bh.id
        for i in hl.grid(n):
            h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
            h_acc = hl.dot(
                k[idx, i, :, tile_d].transpose(-2, -1), v[idx, i, :, tile_dv], acc=h_acc
            )
            h_all[idx, i, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return h_all


def _state_inputs() -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty((4, 8, 64, 64), dtype=torch.bfloat16) for _ in range(2))


def _assert_geometry_without_seeds(bound: BoundKernel) -> None:
    # Only the seeds are withheld. The geometry still exposes the
    # proven-bounds knob and registers the stride facts, so an explicit
    # (or previously autotuned) config keeps normalizing and compiling.
    spec = bound.config_spec
    assert not any(
        seed.get("cute_collective_mma", False) for seed in spec.compiler_seed_configs
    )
    assert spec.cute_proven_bounds_enabled
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    default = spec.default_config()
    assert default["cute_proven_bounds"] is False
    explicit = helion.Config.from_dict(
        default.config | {"cute_proven_bounds": False, "cute_collective_mma": True}
    )
    assert "cute.gemm(" not in bound.to_code(explicit)


@skipUnlessBackends(["cute"])
def test_serial_loop_carried_accumulator_receives_no_collective_seeds() -> None:
    # The full-slice K contraction accumulates the chunk loop's carry, so only
    # the first chunk starts from the captured zero. The lowering refuses such
    # a site under every config, so the heuristic must not seed the kernel.
    bound = _cpu_bind(_serial_state_recurrence, _state_inputs())
    assert bound.host_function is not None
    ir = bound.host_function.device_ir
    (fact,) = bound.config_spec.matmul_facts
    assert fact.k_block_id is not None
    assert bound.env.block_sizes[fact.k_block_id].reduction
    with bound.env:
        (node,) = [
            node
            for graph in ir.graphs
            for node in graph.graph.nodes
            if node.target is hl.dot
        ]
        acc = node.args[2]
        assert isinstance(acc, torch.fx.Node)
        assert not _is_direct_zero_seed(ir.graphs, node, acc, fact.k_block_id)
        assert has_collective_native_seed_candidate(bound.env, ir)
        assert CuteCollectiveMatmulHeuristic._axes(bound.env, ir)
        assert CuteCollectiveMatmulHeuristic.get_seed_configs(bound.env, ir) == []
    _assert_geometry_without_seeds(bound)


@skipUnlessBackends(["cute"])
def test_serial_loop_indexed_operands_receive_no_collective_seeds() -> None:
    # A zero created in every chunk passes the seed proof, but the operands
    # index the chunk loop's coordinate, which no scalar recipe can replay.
    bound = _cpu_bind(_serial_fresh_zero, _state_inputs())
    assert bound.host_function is not None
    ir = bound.host_function.device_ir
    with bound.env:
        assert not has_collective_native_seed_candidate(bound.env, ir)
        assert CuteCollectiveMatmulHeuristic._axes(bound.env, ir)
        assert CuteCollectiveMatmulHeuristic.get_seed_configs(bound.env, ir) == []
    _assert_geometry_without_seeds(bound)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _branch_around_reduction(
    a: torch.Tensor, b: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    out = torch.zeros((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    for row, col in hl.tile([a.size(0), b.size(1)]):
        if flag[0] > 0:
            acc = hl.zeros([row, col], dtype=torch.float32)
            for kk in hl.tile(a.size(1)):
                acc = torch.addmm(acc, a[row, kk] * 0.5, b[kk, col])
            out[row, col] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _block_size_scaled_operand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((4, a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    for row, col in hl.tile([a.size(0), b.size(1)]):
        for rep in hl.tile(4, block_size=1):
            acc = hl.zeros([row, col], dtype=torch.float32)
            for kk in hl.tile(a.size(1)):
                acc = torch.addmm(acc, a[row, kk] * rep.block_size, b[kk, col])
            out[rep.begin, row, col] = acc.to(out.dtype)
    return out


@skipUnlessBackends(["cute"])
def test_branch_between_grid_rows_and_reduction_receives_no_collective_seeds() -> None:
    # The predicate is scalar, but the row lane of a grid M wraps the whole
    # root body, so the branch sits inside the row/reduction region the
    # lowering refuses under every config.
    inputs = (
        torch.empty((256, 256), dtype=torch.float16),
        torch.empty((256, 256), dtype=torch.float16),
        torch.empty((1,), dtype=torch.float32),
    )
    bound = _cpu_bind(_branch_around_reduction, inputs)
    assert bound.host_function is not None
    ir = bound.host_function.device_ir
    with bound.env:
        assert CuteCollectiveMatmulHeuristic._axes(bound.env, ir)
        assert CuteCollectiveMatmulHeuristic.get_seed_configs(bound.env, ir) == []
    _assert_geometry_without_seeds(bound)


@skipUnlessBackends(["cute"])
def test_enclosing_loop_block_size_keeps_collective_seeds() -> None:
    # The enclosing loop's block size is replayed as a constant; only its
    # position (grid index, tile begin, or the tile as a load coordinate)
    # would deny the operands a scalar recipe.
    inputs = tuple(torch.empty((256, 256), dtype=torch.float16) for _ in range(2))
    bound = _cpu_bind(_block_size_scaled_operand, inputs)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    assert bound.to_code(seeds[0]).count("cute.gemm(") == 1


class TestCollectiveSeedCodegen(TestCase):
    @skipUnlessBackends(["cute"])
    def test_admitted_contractions_keep_their_seeds_and_code(self) -> None:
        # The structural seed filter must leave an admitted kernel untouched:
        # pin the seed count and the first seed's generated code.
        bound = _cpu_bind(_independent_matmuls, _admitted_inputs())
        assert bound.host_function is not None
        with bound.env:
            seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        self.assertEqual(len(seeds), 112)
        code = bound.to_code(seeds[0], output_origin_lines=False)
        self.assertEqual(code.count("cute.gemm("), 2)
        self.assertExpectedJournal(code)
