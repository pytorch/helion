from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute import _seq_config_list
from helion._compiler.cute.local_atomic import _allocation_origin_resolver
from helion._compiler.cute.local_atomic import _reachable_graphs
from helion._compiler.device_ir import GraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import control_flow_child_graph_ids
from helion._compiler.device_ir import control_flow_parent_entries
from helion._compiler.device_ir import control_flow_parent_ids
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


def _pointwise_block_id(bound: BoundKernel) -> int:
    spec = bound.config_spec
    fact = spec.matmul_facts[0]
    (block_id,) = set(spec.block_sizes.valid_block_ids()) - {
        fact.m_block_id,
        fact.n_block_id,
        fact.k_block_id,
    }
    return block_id


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
            acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
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
            acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
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
        other = _pointwise_block_id(bound)
        fragment = spec.block_sizes.block_id_lookup(other)._fragment(spec)
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds
        for seed in seeds:
            assert (
                spec.block_sizes.config_get(seed.block_sizes, other)
                == fragment.default()
            )
            assert spec.num_threads.config_get(seed.num_threads, other) == 0
        assert "cute.gemm(" in bound.to_code(seeds[0])
        invalid = helion.Config.from_dict(
            seeds[0].config
            | {
                "num_threads": _seq_config_list(spec.num_threads, {other: 64}),
                "block_sizes": [
                    4096 if bid == other else size
                    for bid, size in zip(
                        spec.block_sizes.valid_block_ids(),
                        seeds[0].block_sizes,
                        strict=True,
                    )
                ],
            }
        )
        with pytest.raises(
            helion.exc.BackendUnsupported, match=f"sibling block {other}"
        ):
            bound.to_code(invalid)


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("block_size", [96, 95])
@skipIfNotCUDA()
@skipIfRefEager("requires compiler seeds")
def test_fixed_sibling_atomic_has_one_owner(compute: str, block_size: int) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs SM100")

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def mixed(
        a: torch.Tensor, b: torch.Tensor, counts: torch.Tensor, width: hl.constexpr
    ) -> torch.Tensor:
        out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
        for m, n in hl.tile([a.size(0), b.size(1)]):
            acc = hl.zeros([m, n], dtype=torch.float32)
            for k in hl.tile(a.size(1)):
                acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
            out[m, n] = acc.to(out.dtype)
        for i in hl.tile(counts.numel(), block_size=width):
            hl.atomic_add(counts, [i], 1.0)
        return out

    a = torch.randn(64, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    counts = torch.zeros(4 * block_size, device=DEVICE)
    args = (a, b, counts, block_size)
    bound = mixed.bind(args)
    assert bound.host_function is not None
    spec = bound.config_spec
    n_block_id = spec.matmul_facts[0].n_block_id
    assert n_block_id is not None
    seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    seed = next(
        seed
        for seed in seeds
        if seed.get("cute_collective_compute", "warp") == compute
        and spec.num_threads.config_get(seed.num_threads, n_block_id) == 64
    )
    result = bound.compile_config(seed)(*args)
    torch.testing.assert_close(result, (a * 0.5) @ b)
    torch.testing.assert_close(counts, torch.ones_like(counts))


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
                        acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
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
                    acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
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
@pytest.mark.parametrize("explicit_tile", [None, 4096])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("reverse_order", [False, True])
@skipIfNotCUDA()
@skipIfRefEager("requires compiler seeds")
def test_mixed_root_seed_executes(
    first: bool, explicit_tile: int | None, compute: str, reverse_order: bool
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs SM100")
    a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    x = torch.randn(1029, device="cuda")
    args = (a, b, x)
    bound = (_pointwise_first if first else _pointwise_last).bind(args)
    assert bound.host_function is not None
    seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    seed = next(s for s in seeds if s.get("cute_collective_compute", "warp") == compute)
    spec = bound.config_spec
    fact = spec.matmul_facts[0]
    if explicit_tile:
        other = _pointwise_block_id(bound)
        seed = helion.Config.from_dict(
            seed.config
            | {
                "block_sizes": [
                    explicit_tile if block_id == other else size
                    for block_id, size in zip(
                        spec.block_sizes.valid_block_ids(),
                        seed.block_sizes,
                        strict=True,
                    )
                ]
            }
        )
    if reverse_order:
        root = next(
            root
            for root in bound.host_function.device_ir.grid_block_ids
            if fact.m_block_id in root
        )
        seed = helion.Config.from_dict(
            seed.config
            | {"loop_orders": _seq_config_list(spec.loop_orders, {root[0]: [1, 0]})}
        )
    actual, pointwise = bound.compile_config(seed)(*args)
    torch.testing.assert_close(actual, (a * 0.5) @ b)
    torch.testing.assert_close(pointwise, x + 1)


@skipIfRefEager("requires compiler seeds")
def test_sibling_reduction_does_not_receive_unlaunchable_seeds() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def mixed(
        a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
        reduced = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
        for m, n in hl.tile([a.size(0), b.size(1)]):
            acc = hl.zeros([m, n], dtype=torch.float32)
            for k in hl.tile(a.size(1)):
                acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
            out[m, n] = acc.to(out.dtype)
        for row in hl.tile(x.size(0)):
            reduced[row] = x[row, :].sum(-1)
        return out, reduced

    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            mixed,
            (
                torch.empty(64, 128, dtype=torch.float16),
                torch.empty(128, 64, dtype=torch.float16),
                torch.empty(5, 128),
            ),
        )
        assert bound.host_function is not None
        assert not CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _inactive_sibling_axis(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    pointwise = torch.empty_like(x)
    for m, n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([m, n], dtype=torch.float32)
        for k in hl.tile(a.size(1)):
            acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
        out[m, n] = acc.to(out.dtype)
    for _unused, i in hl.tile([9, x.numel()]):
        pointwise[i] = x[i] + 1
    return out, pointwise


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipIfNotCUDA()
@skipIfRefEager("checks collective launch dimensions")
def test_inactive_leading_sibling_axis_keeps_the_collective_launch(
    compute: str,
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs SM100")
    a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    x = torch.randn(257, device="cuda")
    bound = _inactive_sibling_axis.bind((a, b, x))
    assert bound.host_function is not None
    seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    seed = next(s for s in seeds if s.get("cute_collective_compute", "warp") == compute)
    actual, pointwise = bound.compile_config(seed)(a, b, x)
    torch.testing.assert_close(actual, (a * 0.5) @ b)
    torch.testing.assert_close(pointwise, x + 1)


@skipIfRefEager("requires compiler seeds")
def test_epilogue_reduction_keeps_its_existing_seed_admission() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def mixed(
        a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, flag: int
    ) -> torch.Tensor:
        out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
        for m, n in hl.tile([a.size(0), b.size(1)]):
            acc = hl.zeros([m, n], dtype=torch.float32)
            acc = torch.addmm(acc, a[m, :] * 0.5, b[:, n])
            if flag > 0:
                out[m, n] = (acc + bias[m, :].sum(-1)[:, None]).to(out.dtype)
            else:
                out[m, n] = acc.to(out.dtype)
        return out

    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            mixed,
            (
                torch.empty(64, 128, dtype=torch.float16),
                torch.empty(128, 64, dtype=torch.float16),
                torch.empty(64, 129),
                1,
            ),
        )
        assert bound.host_function is not None
        assert CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def test_control_flow_parent_maps_include_while_else_without_replaying_state() -> None:
    graph = torch.fx.Graph()
    initial = graph.placeholder("initial")
    call = graph.call_function(_tracing_ops._while_loop, (1, 2, [initial], 3))
    body = torch.fx.Graph()
    state = body.placeholder("state")
    graphs = [
        cast("GraphInfo", SimpleNamespace(graph=graph, graph_id=0)),
        cast("GraphInfo", SimpleNamespace(graph=body, graph_id=2)),
    ]
    assert control_flow_child_graph_ids(call) == (1, 2, 3)
    assert control_flow_parent_ids(graphs) == {1: 0, 2: 0, 3: 0}
    assert control_flow_parent_entries(graphs) == {}
    assert _allocation_origin_resolver(graphs)(state) is state


def test_parent_identity_prefers_original_graph_over_rolling_alias() -> None:
    graph = torch.fx.Graph()
    graph.call_function(_tracing_ops._for_loop, (1, [0], [4], []))
    graphs = [
        GraphInfo(0, graph),
        GraphInfo(1, torch.fx.Graph()),
        GraphInfo(2, graph),
    ]
    assert control_flow_parent_ids(graphs) == {1: 0}


def test_local_allocation_reachability_includes_while_else() -> None:
    graph = torch.fx.Graph()
    graph.call_function(_tracing_ops._while_loop, (1, 2, [], 3))
    graphs = [RootGraphInfo(0, graph)] + [
        GraphInfo(graph_id, torch.fx.Graph()) for graph_id in (1, 2, 3)
    ]
    assert {info.graph_id for info in _reachable_graphs(graphs)} == {0, 1, 2, 3}


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipIfNotCUDA()
@skipIfRefEager("requires compiler seeds")
def test_two_sibling_roots_reuse_the_collective_launch(compute: str) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs SM100")

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def mixed(
        a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
        px, py = torch.empty_like(x), torch.empty_like(y)
        for i in hl.tile(x.numel()):
            px[i] = x[i] + 1
        for m, n in hl.tile([a.size(0), b.size(1)]):
            acc = hl.zeros([m, n], dtype=torch.float32)
            for k in hl.tile(a.size(1)):
                acc = hl.dot(a[m, k] * 0.5, b[k, n], acc)
            out[m, n] = acc.to(out.dtype)
        for outer in hl.tile(y.size(0)):
            for middle in hl.tile(y.size(1), block_size=1):
                for inner in hl.tile(y.size(2)):
                    py[outer, middle, inner] = y[outer, middle, inner] * 2
        return out, px, py

    args = (
        torch.randn(64, 128, device="cuda", dtype=torch.float16),
        torch.randn(128, 64, device="cuda", dtype=torch.float16),
        torch.randn(257, device="cuda"),
        torch.randn(5, 2, 257, device="cuda"),
    )
    bound = mixed.bind(args)
    assert bound.host_function is not None
    seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    assert seeds
    seed = next(s for s in seeds if s.get("cute_collective_compute", "warp") == compute)
    actual = bound.compile_config(seed)(*args)
    a, b, x, y = args
    for output, expected in zip(actual, ((a * 0.5) @ b, x + 1, y * 2), strict=True):
        torch.testing.assert_close(output, expected)


def test_control_flow_graph_queries_skip_incomplete_calls() -> None:
    graph = torch.fx.Graph()
    for target, args in (
        (_tracing_ops._for_loop, ()),
        (_tracing_ops._if, (True, 1)),
        (_tracing_ops._while_loop, (1,)),
        (_tracing_ops._if, (True, "not_graph_id", 2, [], [])),
    ):
        assert control_flow_child_graph_ids(graph.call_function(target, args)) == ()
