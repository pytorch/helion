from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.graph_cse import cse_pure_nodes
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import NodeArgsGraphInfo
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

BLOCK_REDUCE = "_cute_grouped_reduce_shared_two_stage("


def _jsd(
    beta: float, temperature: float, student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """examples/fused_linear_jsd.py::fused_linear_jsd_kernel's row body."""
    batch, _vocab = student.size()
    loss = student.new_empty(batch, dtype=torch.float32)
    for tile in hl.tile(batch):
        s = student[tile, :] / temperature
        t = teacher[tile, :] / temperature
        sp = torch.softmax(s.to(torch.float32), dim=-1)
        tp = torch.softmax(t.to(torch.float32), dim=-1)
        slp = torch.log_softmax(s.to(torch.float32), dim=-1)
        tlp = torch.log_softmax(t.to(torch.float32), dim=-1)
        m = (1 - beta) * sp + beta * tp
        log_m = torch.log(m)
        skl = (sp * (slp - log_m)).sum(dim=-1)
        tkl = (tp * (tlp - log_m)).sum(dim=-1)
        loss[tile] = (1 - beta) * skl + beta * tkl
    return loss


def _reload_between_stores(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile in hl.tile(n):
        first = x[tile, :]
        y[tile, :] = first * 2
        second = x[tile, :]
        out[tile] = (first + second).sum(-1)
    return out


def _scale_feeds_inner_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :]
        scale = row.amax(-1)
        scale_again = row.amax(-1)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (
                x[tile_m, tile_n] - scale[:, None] - scale_again[:, None]
            )
    return out


def _softmax_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax and log_softmax of one operand: two amax, two exp, two sum
    nodes as traced; one of each once merged."""
    n, _m = x.size()
    p = torch.empty_like(x)
    lp = torch.empty_like(x)
    for tile in hl.tile(n):
        row = x[tile, :]
        p[tile, :] = torch.softmax(row, dim=-1)
        lp[tile, :] = torch.log_softmax(row, dim=-1)
    return p, lp


jsd_exact = helion.kernel(backend="cute", static_shapes=True)(_jsd)
jsd_traced = helion.kernel(backend="cute", static_shapes=True, cute_graph_cse=False)(
    _jsd
)
softmax_pair = helion.kernel(backend="cute", static_shapes=True)(_softmax_pair)
softmax_pair_traced = helion.kernel(
    backend="cute", static_shapes=True, cute_graph_cse=False
)(_softmax_pair)
reload_between_stores = helion.kernel(backend="cute", static_shapes=True)(
    _reload_between_stores
)
scale_feeds_inner_loop = helion.kernel(backend="cute", static_shapes=True)(
    _scale_feeds_inner_loop
)

ROLLED: dict[str, Any] = {
    "block_sizes": [1],
    "num_threads": [1, 256],
    "reduction_loops": [4096],
    "cute_vector_widths": [4, 1],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_reduction_reloads": ["register"],
    # The sweep counts below are the merged graph's; the exact division
    # hoist's redo form would add a second copy of every sweep
    # (test_cute_exact_division.py covers that).
    "cute_exact_div_form": "plain",
}


def _bind(kernel: helion.Kernel, args: tuple[object, ...]) -> BoundKernel:
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        return _cpu_bind(kernel, args)


def _device_ir(bound: BoundKernel) -> DeviceIR:
    host_function = bound.host_function
    assert host_function is not None
    return host_function.device_ir


def _count_targets(graph: torch.fx.Graph, target: object) -> int:
    return sum(1 for node in graph.nodes if node.target is target)


def _jsd_args() -> tuple[object, ...]:
    student = torch.empty((128, 4096), dtype=torch.float32)
    teacher = torch.empty((128, 4096), dtype=torch.float32)
    return (0.5, 1.0, student, teacher)


def test_softmax_and_log_softmax_share_their_reductions() -> None:
    """One operand's softmax and log_softmax trace two amax, two exp and two
    sum nodes each; the merged graph rolls into five sweeps with six block
    reductions (two maxima, two denominators, two divergence sums)."""
    bound = _bind(jsd_exact, _jsd_args())
    device_ir = _device_ir(bound)
    root = device_ir.graphs[device_ir.root_ids[0]].graph
    assert _count_targets(root, torch.ops.aten.amax.default) == 2
    assert _count_targets(root, torch.ops.aten.exp.default) == 2
    assert _count_targets(root, torch.ops.aten.sum.dim_IntList) == 4
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        code = bound.to_code(helion.Config(**ROLLED))
    assert code.count(BLOCK_REDUCE) == 6
    assert code.count("for roffset_") == 5


def test_the_setting_keeps_the_traced_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cute_graph_cse=False`` (or HELION_CUTE_GRAPH_CSE=0) leaves every
    duplicate as traced: four amax, four exp, six sums, nine sweeps."""
    bound = _bind(jsd_traced, _jsd_args())
    root = _device_ir(bound).graphs[_device_ir(bound).root_ids[0]].graph
    assert _count_targets(root, torch.ops.aten.amax.default) == 4
    assert _count_targets(root, torch.ops.aten.exp.default) == 4
    assert _count_targets(root, torch.ops.aten.sum.dim_IntList) == 6
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        code = bound.to_code(helion.Config(**ROLLED))
    assert code.count(BLOCK_REDUCE) == 10
    assert code.count("for roffset_") == 9
    assert helion.Settings().cute_graph_cse is True
    monkeypatch.setenv("HELION_CUTE_GRAPH_CSE", "0")
    assert helion.Settings().cute_graph_cse is False
    assert helion.Settings(cute_graph_cse=True).cute_graph_cse is True
    monkeypatch.setenv("HELION_CUTE_GRAPH_CSE", "maybe")
    with pytest.raises(ValueError, match="HELION_CUTE_GRAPH_CSE"):
        helion.Settings()


def test_loads_keep_their_order_around_a_store() -> None:
    """Memory operations never merge: a reload after a store into another
    tensor stays a second load."""
    x = torch.empty((64, 1024), dtype=torch.float32)
    y = torch.empty((64, 1024), dtype=torch.float32)
    bound = _bind(reload_between_stores, (x, y))
    device_ir = _device_ir(bound)
    root = device_ir.graphs[device_ir.root_ids[0]].graph
    assert _count_targets(root, hl.load) == 2
    assert _count_targets(root, hl.store) == 2


def test_subgraph_arguments_follow_the_surviving_node() -> None:
    """An inner tile loop that consumed the duplicate reads the survivor."""
    x = torch.empty((64, 2048), dtype=torch.float32)
    bound = _bind(scale_feeds_inner_loop, (x,))
    device_ir = _device_ir(bound)
    root = device_ir.graphs[device_ir.root_ids[0]].graph
    assert _count_targets(root, torch.ops.aten.amax.default) == 1
    live = {node for info in device_ir.graphs for node in info.graph.nodes}
    for info in device_ir.graphs:
        if isinstance(info, NodeArgsGraphInfo):
            assert all(node in live for node in info.node_args)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        code = bound.to_code(
            helion.Config(
                block_sizes=[1, 256],
                num_threads=[1, 256, 256],
                reduction_loops=[None],
                cute_vector_widths=[4, 1, 1],
                cute_lane_layouts=["strided", "blocked", "blocked"],
                cute_reduction_reloads=["register"],
            )
        )
    assert "_helion__scale_feeds_inner_loop" in code


def _graph_with(
    make: object,
) -> tuple[torch.fx.Graph, list[torch.fx.Node]]:
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    for node in (a, b):
        node.meta["val"] = torch.empty(4)
    nodes = make(graph, a, b)  # pyrefly: ignore [not-callable]
    for node in nodes:
        node.meta.setdefault("val", torch.empty(4))
    graph.output(tuple(nodes))
    return graph, nodes


def test_cse_merges_duplicate_roots_and_their_consequences() -> None:
    """A duplicate root merges; a duplicate fed by a merged node merges; an
    unrelated duplicate, a swapped operand order, a different kwarg and a
    different float payload all stay."""

    def make(graph: torch.fx.Graph, a: torch.fx.Node, b: torch.fx.Node) -> list:
        add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        add_again = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        doubled = graph.call_function(torch.ops.aten.mul.Scalar, (add, 2.0))
        doubled_again = graph.call_function(torch.ops.aten.mul.Scalar, (add_again, 2.0))
        swapped = graph.call_function(torch.ops.aten.add.Tensor, (b, a))
        scaled = graph.call_function(torch.ops.aten.add.Tensor, (a, b), {"alpha": 2})
        neg_zero = graph.call_function(torch.ops.aten.add.Scalar, (a, -0.0))
        pos_zero = graph.call_function(torch.ops.aten.add.Scalar, (a, 0.0))
        neg = graph.call_function(torch.ops.aten.neg.default, (a,))
        neg_again = graph.call_function(torch.ops.aten.neg.default, (a,))
        return [
            add,
            add_again,
            doubled,
            doubled_again,
            swapped,
            scaled,
            neg_zero,
            pos_zero,
            neg,
            neg_again,
        ]

    graph, nodes = _graph_with(make)
    replaced = cse_pure_nodes(
        graph, is_root=lambda node: node.target is torch.ops.aten.add.Tensor
    )
    assert replaced == {nodes[1]: nodes[0], nodes[3]: nodes[2]}
    outputs = next(iter(graph.find_nodes(op="output"))).args[0]
    assert outputs[1] is nodes[0]
    assert outputs[3] is nodes[2]
    assert _count_targets(graph, torch.ops.aten.add.Tensor) == 3
    assert _count_targets(graph, torch.ops.aten.mul.Scalar) == 1
    assert _count_targets(graph, torch.ops.aten.add.Scalar) == 2
    assert _count_targets(graph, torch.ops.aten.neg.default) == 2


def test_cse_admits_the_survivors_consumer_whatever_the_trace_order() -> None:
    """Duplicate roots whose consumers are traced in the other order: the
    erased root's consumer is seen first and survives its key; the surviving
    root's consumer, traced second, merges into it all the same (its input is
    a survivor of this pass)."""

    def make(graph: torch.fx.Graph, a: torch.fx.Node, b: torch.fx.Node) -> list:
        r1 = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        r2 = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        c_dup = graph.call_function(torch.ops.aten.mul.Scalar, (r2, 2.0))
        c_surv = graph.call_function(torch.ops.aten.mul.Scalar, (r1, 2.0))
        # The chain continues through the merged consumers in either order.
        d_surv = graph.call_function(torch.ops.aten.neg.default, (c_surv,))
        d_dup = graph.call_function(torch.ops.aten.neg.default, (c_dup,))
        # A consumer of the survivor root with no twin stays.
        lone = graph.call_function(torch.ops.aten.mul.Scalar, (r1, 3.0))
        return [r1, r2, c_dup, c_surv, d_surv, d_dup, lone]

    graph, nodes = _graph_with(make)
    r1, r2, c_dup, c_surv, d_surv, d_dup, lone = nodes
    replaced = cse_pure_nodes(
        graph, is_root=lambda node: node.target is torch.ops.aten.add.Tensor
    )
    assert replaced == {r2: r1, c_surv: c_dup, d_dup: d_surv}
    outputs = next(iter(graph.find_nodes(op="output"))).args[0]
    assert outputs == (r1, r1, c_dup, c_dup, d_surv, d_surv, lone)
    assert c_dup.args == (r1, 2.0)
    assert d_surv.args == (c_dup,)
    assert _count_targets(graph, torch.ops.aten.add.Tensor) == 1
    assert _count_targets(graph, torch.ops.aten.mul.Scalar) == 2
    assert _count_targets(graph, torch.ops.aten.neg.default) == 1


def test_cse_leaves_uninitialized_allocations_alone() -> None:
    """``aten.empty_like`` and friends are pure by schema, but two of them
    hold whatever memory they were given: never merged, even on the chain
    from a merged root."""

    def make(graph: torch.fx.Graph, a: torch.fx.Node, b: torch.fx.Node) -> list:
        add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        add_again = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        fresh = graph.call_function(torch.ops.aten.empty_like.default, (add,))
        fresh_again = graph.call_function(
            torch.ops.aten.empty_like.default, (add_again,)
        )
        sized = graph.call_function(
            torch.ops.aten.new_empty.default, (add, [4]), {"dtype": torch.float32}
        )
        sized_again = graph.call_function(
            torch.ops.aten.new_empty.default, (add_again, [4]), {"dtype": torch.float32}
        )
        blank = graph.call_function(
            torch.ops.aten.empty.memory_format, ([4],), {"dtype": torch.float32}
        )
        blank_again = graph.call_function(
            torch.ops.aten.empty.memory_format, ([4],), {"dtype": torch.float32}
        )
        return [
            add,
            add_again,
            fresh,
            fresh_again,
            sized,
            sized_again,
            blank,
            blank_again,
        ]

    graph, nodes = _graph_with(make)
    replaced = cse_pure_nodes(graph, is_root=lambda node: True)
    assert replaced == {nodes[1]: nodes[0]}
    assert _count_targets(graph, torch.ops.aten.empty_like.default) == 2
    assert _count_targets(graph, torch.ops.aten.new_empty.default) == 2
    assert _count_targets(graph, torch.ops.aten.empty.memory_format) == 2


def test_cse_leaves_random_and_python_level_nodes_alone() -> None:
    def make(graph: torch.fx.Graph, a: torch.fx.Node, b: torch.fx.Node) -> list:
        rand = graph.call_function(torch.ops.aten.rand.default, ([4],))
        rand_again = graph.call_function(torch.ops.aten.rand.default, ([4],))
        item = graph.call_function(operator.getitem, (a, 0))
        item_again = graph.call_function(operator.getitem, (a, 0))
        for node in (item, item_again):
            node.meta["val"] = 1.0
        return [rand, rand_again, item, item_again]

    graph, _nodes = _graph_with(make)
    assert cse_pure_nodes(graph, is_root=lambda node: True) == {}
    assert _count_targets(graph, torch.ops.aten.rand.default) == 2
    assert _count_targets(graph, operator.getitem) == 2


@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestGraphCseCuda(TestCase):
    def test_jsd_rows_match_torch(self) -> None:
        torch.manual_seed(0)
        student = 4 * torch.randn(6, 1000, device=DEVICE, dtype=torch.float32)
        teacher = 4 * torch.randn(6, 1000, device=DEVICE, dtype=torch.float32)
        beta, temperature = 0.3, 2.0
        code, out = code_and_output(
            jsd_exact,
            (beta, temperature, student, teacher),
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[1024],
            cute_vector_widths=[4, 1],
            cute_lane_layouts=["strided", "blocked"],
            cute_reduction_reloads=["register"],
        )
        assert code.count(BLOCK_REDUCE) == 6
        s = student / temperature
        t = teacher / temperature
        sp, tp = torch.softmax(s, -1), torch.softmax(t, -1)
        slp, tlp = torch.log_softmax(s, -1), torch.log_softmax(t, -1)
        log_m = torch.log((1 - beta) * sp + beta * tp)
        expected = (1 - beta) * (sp * (slp - log_m)).sum(-1) + beta * (
            tp * (tlp - log_m)
        ).sum(-1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_pair_is_bitwise_the_traced_render(self) -> None:
        """The merged render and the traced render (``cute_graph_cse=False``)
        of softmax + log_softmax agree bit for bit on every input class --
        overflow-scale magnitudes, a large common offset, -inf columns and
        rows, +inf, NaN and a signed zero -- under the rolled sweep config
        and under the default config."""
        n, m = 64, 1000
        generator = torch.Generator().manual_seed(0)

        def randn() -> torch.Tensor:
            return torch.randn(n, m, generator=generator)

        cases = {"randn": randn(), "scale_1e30": randn() * 1e30}
        cases["offset_1e8"] = randn() + 1e8
        x = randn()
        x[:, ::3] = float("-inf")
        cases["neginf_columns"] = x
        x = randn()
        x[0, :] = float("-inf")
        cases["neginf_row"] = x
        x = randn()
        x[1, 5] = float("inf")
        cases["posinf"] = x
        x = randn()
        x[2, 7] = float("nan")
        cases["nan"] = x
        x = randn()
        x[3, :] = 0.0
        x[3, 0] = -0.0
        cases["signed_zero"] = x
        probe = cases["randn"].to(DEVICE)
        configs: list[dict[str, Any]] = [
            {
                "block_sizes": [1],
                "num_threads": [1, 256],
                "reduction_loops": [1024],
                "cute_vector_widths": [4, 1],
                "cute_lane_layouts": ["strided", "blocked"],
                "cute_reduction_reloads": ["register"],
            },
            dict(softmax_pair.bind((probe,)).config_spec.default_config().config),
        ]
        for config in configs:
            merged_code, _ = code_and_output(softmax_pair, (probe,), **config)
            traced_code, _ = code_and_output(softmax_pair_traced, (probe,), **config)
            assert merged_code.count(BLOCK_REDUCE) < traced_code.count(BLOCK_REDUCE)
            merged = softmax_pair.bind((probe,)).compile_config(helion.Config(**config))
            traced = softmax_pair_traced.bind((probe,)).compile_config(
                helion.Config(**config)
            )
            for name, rows in cases.items():
                x = rows.to(DEVICE)
                for lhs, rhs in zip(merged(x), traced(x), strict=True):
                    equal = (lhs == rhs) | (lhs.isnan() & rhs.isnan())
                    assert bool(equal.all()), (name, config)
            p, lp = merged(probe)
            torch.testing.assert_close(p, torch.softmax(probe, -1))
            torch.testing.assert_close(lp, torch.log_softmax(probe, -1))


if __name__ == "__main__":
    pytest.main([__file__])
