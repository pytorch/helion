"""Tests for the fast-math distribution of sum reductions over affine updates."""

from __future__ import annotations

import torch

import helion
from helion._compiler.pallas import distribute_reductions
from helion._compiler.pallas.distribute_reductions import distribute_affine_reductions
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl
from helion.language import view_ops

_MUL = torch.ops.aten.mul.Tensor
_ADD = torch.ops.aten.add.Tensor
_SUM = torch.ops.aten.sum.dim_IntList
_FULL = slice(None)

H, V, K = 2, 4, 8


def _node(
    graph: torch.fx.Graph, target: object, args: tuple[object, ...], *shape: int
) -> torch.fx.Node:
    # pyrefly: ignore [bad-argument-type]
    node = graph.call_function(target, args)
    node.meta["val"] = torch.empty(*shape)
    return node


def _placeholder(graph: torch.fx.Graph, name: str, *shape: int) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(*shape)
    return node


def _lane_view(graph: torch.fx.Graph, vector: torch.fx.Node) -> torch.fx.Node:
    """``vector[:, None, :]``: invariant along V."""
    return _node(graph, view_ops.subscript, (vector, [_FULL, None, _FULL]), H, 1, K)


def _reduce(
    graph: torch.fx.Graph, state: torch.fx.Node, weight: torch.fx.Node
) -> torch.fx.Node:
    product = _node(graph, _MUL, (state, weight), H, V, K)
    return _node(graph, _SUM, (product, [-1]), H, V)


def _step_graph(
    coef_varies_along_k: bool = False,
) -> tuple[torch.fx.Graph, torch.fx.Node, torch.fx.Node]:
    """One delta-rule step on ``s [H, V, K]``::

    s1 = s * a[:, :, None];  p = sum(s1 * k[:, None, :], -1)
    s2 = s1 + c[:, :, None] * k[:, None, :];  o = sum(s2 * q[:, None, :], -1)
    """
    graph = torch.fx.Graph()
    s = _placeholder(graph, "s", H, V, K)
    a = _placeholder(graph, "a", H, V)
    k = _placeholder(graph, "k", H, K)
    q = _placeholder(graph, "q", H, K)
    if coef_varies_along_k:
        c_b = _lane_view(graph, _placeholder(graph, "c", H, K))
    else:
        c = _placeholder(graph, "c", H, V)
        c_b = _node(graph, view_ops.subscript, (c, [_FULL, _FULL, None]), H, V, 1)
    a_b = _node(graph, view_ops.subscript, (a, [_FULL, _FULL, None]), H, V, 1)
    k_b = _lane_view(graph, k)
    s1 = _node(graph, _MUL, (s, a_b), H, V, K)
    p = _reduce(graph, s1, k_b)
    s2 = _node(graph, _ADD, (s1, _node(graph, _MUL, (c_b, k_b), H, V, K)), H, V, K)
    o = _reduce(graph, s2, _lane_view(graph, q))
    graph.output((p, o, s2))
    return graph, s, s2


def _full_reduction_inputs(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """The full-size operands of the full-size products that are summed."""
    inputs = []
    for node in graph.find_nodes(op="call_function", target=_SUM):
        product = node.args[0]
        assert isinstance(product, torch.fx.Node)
        if tuple(product.meta["val"].shape) != (H, V, K):
            continue
        inputs.extend(
            arg
            for arg in product.args
            if isinstance(arg, torch.fx.Node)
            and tuple(arg.meta["val"].shape) == (H, V, K)
        )
    return inputs


def _run(graph: torch.fx.Graph, args: list[torch.Tensor]) -> list[torch.Tensor]:
    """Evaluate ``graph`` eagerly (``subscript`` as plain indexing)."""
    env: dict[torch.fx.Node, object] = {}
    inputs = iter(args)
    for node in graph.nodes:
        if node.op == "placeholder":
            env[node] = next(inputs)
        elif node.op == "output":
            return list(torch.fx.map_arg(node.args[0], env.__getitem__))
        else:
            values = torch.fx.map_arg(node.args, env.__getitem__)
            if node.target is view_ops.subscript:
                env[node] = values[0][tuple(values[1])]
            else:
                env[node] = node.target(*values)
    raise AssertionError("graph has no output")


def test_moves_reductions_to_the_step_input() -> None:
    graph, s, s2 = _step_graph()

    assert distribute_affine_reductions(graph, fast_math=True) == 3

    # p and o both reduce s; neither waits for the update s2.
    assert _full_reduction_inputs(graph) == [s, s]
    assert s2 in graph.nodes
    # k . q is reduced at the broadcast shape [H, 1, K].
    small = [
        node
        for node in graph.find_nodes(op="call_function", target=_SUM)
        if tuple(node.args[0].meta["val"].shape) == (H, 1, K)
    ]
    assert len(small) == 1
    assert tuple(small[0].meta["val"].shape) == (H, 1)


def test_values_match() -> None:
    graph, _, _ = _step_graph()
    g = torch.Generator().manual_seed(0)
    args = [
        torch.randn(shape, generator=g)
        for shape in ((H, V, K), (H, V), (H, K), (H, K), (H, V))
    ]
    expected = _run(graph, args)

    distribute_affine_reductions(graph, fast_math=True)

    got = _run(graph, args)
    for actual, want in zip(got, expected, strict=True):
        torch.testing.assert_close(actual, want)


def test_requires_fast_math() -> None:
    graph, s, s2 = _step_graph()

    assert distribute_affine_reductions(graph, fast_math=False) == 0
    assert _full_reduction_inputs(graph) == [s2.args[0], s2]


def test_requires_coefficient_invariant_along_reduced_dims() -> None:
    # Only the decay of p is factored; o still reduces the update.
    graph, s, s2 = _step_graph(coef_varies_along_k=True)

    assert distribute_affine_reductions(graph, fast_math=True) == 1
    assert _full_reduction_inputs(graph) == [s, s2]


def test_depth_limits_how_far_back_a_reduction_moves() -> None:
    # The next step's reduction of the decayed output, sum(s2 * a2 * q2), stays
    # on s2 at depth 1 (s2 is that step's input); depth 2 moves it to s.
    for depth, count, source in ((1, 4, "s2"), (2, 6, "s")):
        graph, s, s2 = _step_graph()
        output = next(n for n in graph.nodes if n.op == "output")
        with graph.inserting_before(output):
            a2 = _placeholder(graph, "a2", H, V)
            a2_b = _node(graph, view_ops.subscript, (a2, [_FULL, _FULL, None]), H, V, 1)
            s3 = _node(graph, _MUL, (s2, a2_b), H, V, K)
            q2 = _lane_view(graph, _placeholder(graph, "q2", H, K))
            o2 = _reduce(graph, s3, q2)
        output.args = ((*output.args[0], o2),)
        original = distribute_reductions._MAX_DEPTH
        distribute_reductions._MAX_DEPTH = depth
        try:
            assert distribute_affine_reductions(graph, fast_math=True) == count
        finally:
            distribute_reductions._MAX_DEPTH = original
        assert _full_reduction_inputs(graph) == [s, s, {"s": s, "s2": s2}[source]]


@onlyBackends(["pallas"])
class TestDistributeReductionsKernel(TestCase):
    def test_delta_rule_recurrence(self) -> None:
        def recurrence(
            state_in: torch.Tensor,
            k: torch.Tensor,
            q: torch.Tensor,
            v: torch.Tensor,
            beta: torch.Tensor,
            gain: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, hv, _ = k.shape
            dv = v.size(2)
            out = torch.empty([m, hv, dv], dtype=torch.float32, device=k.device)
            state_out = torch.empty_like(state_in)
            for _ in hl.grid(1):
                state = state_in[:, :, :]
                for b in hl.static_range(m):
                    state = state * gain[b, :][:, None, None]
                    k_b = k[b, :, :][:, None, :]
                    prediction = torch.sum(k_b * state, -1)
                    delta = (v[b, :, :] - prediction) * beta[b, :][:, None]
                    state = state + delta[:, :, None] * k_b
                    out[b, :, :] = torch.sum(q[b, :, :][:, None, :] * state, -1)
                state_out[:, :, :] = state
            return out, state_out

        m, hv, dv, dk = 3, 2, 128, 128
        state, k, q, v, beta, gain = (
            0.1 * torch.randn(hv, dv, dk, device=DEVICE),
            torch.nn.functional.normalize(
                torch.randn(m, hv, dk, device=DEVICE), dim=-1
            ),
            torch.randn(m, hv, dk, device=DEVICE) / 16,
            torch.randn(m, hv, dv, device=DEVICE),
            torch.sigmoid(torch.randn(m, hv, device=DEVICE)),
            torch.rand(m, hv, device=DEVICE),
        )
        args = (state, k, q, v, beta, gain)
        outs = []
        for b in range(m):
            state = state * gain[b][:, None, None]
            prediction = (k[b][:, None, :] * state).sum(-1)
            delta = (v[b] - prediction) * beta[b][:, None]
            state = state + delta[:, :, None] * k[b][:, None, :]
            outs.append((q[b][:, None, :] * state).sum(-1))
        sums = {}
        for fast_math in (False, True):
            kernel = helion.kernel(static_shapes=True, fast_math=fast_math)(recurrence)
            code, (out, state_out) = code_and_output(kernel, args)
            torch.testing.assert_close(out, torch.stack(outs), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(state_out, state, atol=1e-5, rtol=1e-5)
            sums[fast_math] = code.count("jnp.sum(")
        # Two full-size reductions per row either way, plus k . q per row.
        self.assertEqual(sums, {False: 2 * m, True: 3 * m})
