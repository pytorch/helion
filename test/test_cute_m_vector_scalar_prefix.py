"""Typed pure arithmetic before the explicit per-M singleton view."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_epilogue_dag import _evaluate

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._compiler.cute.epilogue_fanout import _chain_key
from helion._compiler.cute.m_vector_scalar_prefix import prove_m_vector_scalar_prefix
from helion._compiler.cute.shared_epilogue_dag import dag_keys
from helion._compiler.cute.shared_epilogue_dag import raw_aux_key
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl

_DEFAULT_PREFIX_MODE = hl.constexpr("typed")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def prefix_kernel(a, b, u, v, first, second, mode: hl.constexpr = _DEFAULT_PREFIX_MODE):
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        base = u[row]
        if mode == "typed":
            denom = base + hl.full([], 1e-6, dtype=torch.float32)
        elif mode == "scalar":
            denom = base + 1e-6
        elif mode == "scalar_reverse":
            denom = 2.0 - base
        elif mode == "multiply":
            denom = (base * hl.full([], 2.0, dtype=torch.float32)) - hl.full(
                [], 0.5, dtype=torch.float32
            )
        elif mode == "divide":
            denom = hl.full([], 2.0, dtype=torch.float32) / base
        elif mode == "reverse":
            denom = hl.full([], 2.0, dtype=torch.float32) - base
        elif mode == "cast":
            denom = (
                (base + hl.full([], 1e-6, dtype=torch.float32))
                .to(torch.float16)
                .to(torch.float32)
            )
        elif mode == "bfcast":
            denom = (
                (base + hl.full([], 1e-6, dtype=torch.float32))
                .to(torch.bfloat16)
                .to(torch.float32)
            )
        elif mode == "unary":
            denom = torch.abs(-base) + hl.full([], 1e-6, dtype=torch.float32)
        elif mode == "two_vectors":
            denom = base + v[row]
        elif mode == "half_math":
            denom = base.to(torch.float16) + hl.full([], 1e-6, dtype=torch.float16)
        elif mode == "mixed_typed_scalar":
            denom = base + hl.full([], 1e-6, dtype=torch.bfloat16)
        elif mode == "runtime_scalar":
            denom = base + hl.load(v, [0])
        elif mode == "unknown":
            denom = torch.sin(base) + hl.full([], 1e-6, dtype=torch.float32)
        else:
            denom = base + hl.full([], float("inf"), dtype=torch.float32)
        shaped = denom.reshape([row.index.size(0), 1])
        value = acc / shaped / shaped
        first[row, col] = value.to(first.dtype)
        second[row, col] = (-acc / shaped).to(second.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def different_prefixes(a, b, u, first, second):
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        base = u[row]
        one = base + hl.full([], 1e-6, dtype=torch.float32)
        two = base + hl.full([], 2e-6, dtype=torch.float32)
        first[row, col] = (acc / one.reshape([row, 1]) / one.reshape([row, 1])).to(
            first.dtype
        )
        second[row, col] = (acc / two.reshape([row, 1]) / two.reshape([row, 1])).to(
            second.dtype
        )


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def args(dtype=torch.bfloat16, m=256, n=128, stride=1, offset=0):
    raw = torch.empty(m * stride + offset, dtype=torch.float32)
    return (
        torch.empty((m, 128), dtype=dtype),
        torch.empty((128, n), dtype=dtype),
        raw[offset : offset + m * stride : stride],
        torch.empty(m),
        torch.empty((m, n)),
        torch.empty((m, n)),
    )


def chains(bound):
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        anchor = next(
            n
            for g in graphs
            for n in g.graph.nodes
            if n.target is torch.ops.aten.addmm.default
        )
        return analyze_tcgen05_matmul_store_chains(graphs, anchor)


@pytest.mark.parametrize(
    "mode",
    (
        "typed",
        "scalar",
        "scalar_reverse",
        "multiply",
        "divide",
        "reverse",
        "cast",
        "bfcast",
        "unary",
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("stride,offset", ((1, 0), (2, 7)))
def test_full_native_prefix_codegen_preserves_exact_operations(
    mode, dtype, stride, offset
):
    actual = args(dtype, stride=stride, offset=offset)
    with patch.object(prefix_kernel, "_bind_isolated", side_effect=prefix_kernel.bind):
        bound = cpu_bind(prefix_kernel, (*actual, mode))
    assert bound.config_spec.cute_tcgen05_search_enabled
    complete = chains(bound)
    assert complete is not None and len(complete) == 2
    prefixes = [
        leaf
        for _, c in complete
        for leaf in c.auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorScalarPrefixLoadExpr)
    ]
    assert prefixes
    assert all(p.raw_expr.load_node is p.load_node for p in prefixes)
    assert all(p.prefix.provenance for p in prefixes)
    for config in (
        bound.config_spec.default_config(),
        helion.Config(block_sizes=[128, 32, 64], pid_type="flat"),
    ):
        source = bound.to_code(config)
        assert "tcgen05_ab_tma" in source and ".layout.stride[0], 0)" in source
        assert (
            "cute.make_rmem_tensor(cute.make_layout(tcgen05_tRS_rAcc.shape)" in source
        )
        assert ".producer_commit()" in source
    original_graph = complete[0][0].graph
    divisions = [
        n for n in original_graph.nodes if n.target is torch.ops.aten.div.Tensor
    ]
    assert len(divisions) == (4 if mode == "divide" else 3)


@pytest.mark.parametrize(
    "mode",
    (
        "two_vectors",
        "half_math",
        "mixed_typed_scalar",
        "runtime_scalar",
        "unknown",
        "nonfinite",
    ),
)
def test_unproved_prefix_grammar_or_promotion_does_not_enter_native_surface(mode):
    bound = cpu_bind(prefix_kernel, (*args(), mode))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert chains(bound) is None


@pytest.mark.parametrize(
    "mode",
    (
        "typed",
        "scalar",
        "scalar_reverse",
        "multiply",
        "divide",
        "reverse",
        "cast",
        "bfcast",
        "unary",
    ),
)
def test_IEEE_prefix_and_two_original_divisions_are_bitwise_exact(mode):
    actual = args()
    bound = cpu_bind(prefix_kernel, (*actual, mode))
    complete = chains(bound)
    assert complete is not None
    chain = complete[0][1]
    leaves = chain.auxiliary_tensor_loads
    raw = (
        torch.tensor(
            [
                0.0,
                -0.0,
                1e-6,
                -1e-6,
                1.0,
                -1.0,
                float("inf"),
                -float("inf"),
                float("nan"),
                2**-126,
                -(2**-126),
                1e10,
            ],
            dtype=torch.float32,
        )
        .reshape(12, 1)
        .expand(12, 5)
    )
    acc = torch.linspace(-2, 2, 60).reshape(12, 5)
    counter = itertools.count()
    source, expr = chain.render_prelude_and_expr(
        "original_acc",
        lambda p: f"{p}{next(counter)}",
        "",
        dict.fromkeys(leaves, "raw_u"),
    )
    got = _evaluate(
        source,
        expr,
        acc,
        raw_u=raw,
        cute=SimpleNamespace(math=SimpleNamespace(absf=torch.abs)),
    )
    assert isinstance(got, torch.Tensor)
    c = torch.tensor(1e-6, dtype=torch.float32)
    if mode in ("typed", "scalar"):
        denom = raw + c
    elif mode in ("reverse", "scalar_reverse"):
        denom = torch.tensor(2.0, dtype=torch.float32) - raw
    elif mode == "multiply":
        denom = (raw * torch.tensor(2.0, dtype=torch.float32)) - torch.tensor(
            0.5, dtype=torch.float32
        )
    elif mode == "divide":
        denom = torch.tensor(2.0, dtype=torch.float32) / raw
    elif mode == "cast":
        denom = (raw + c).half().float()
    elif mode == "bfcast":
        denom = (raw + c).bfloat16().float()
    else:
        denom = raw.abs() + c
    expected = (acc / denom) / denom
    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        got[finite].view(torch.int32), expected[finite].view(torch.int32)
    )
    nonnan = ~torch.isnan(expected)
    assert torch.equal(torch.signbit(got[nonnan]), torch.signbit(expected[nonnan]))


def test_different_prefixes_of_same_raw_load_have_distinct_typed_shared_keys():
    a, b, u, _v, first, second = args()
    bound = cpu_bind(different_prefixes, (a, b, u, first, second))
    complete = chains(bound)
    assert complete is not None
    c1, c2 = complete[0][1], complete[1][1]
    leaf1, leaf2 = c1.auxiliary_tensor_loads[0], c2.auxiliary_tensor_loads[0]
    assert isinstance(leaf1, epi._MVectorScalarPrefixLoadExpr) and isinstance(
        leaf2, epi._MVectorScalarPrefixLoadExpr
    )
    assert leaf1.load_node is leaf2.load_node
    assert _chain_key(c1) != _chain_key(c2)
    assert raw_aux_key(leaf1) != raw_aux_key(leaf2)
    assert dag_keys(leaf1.prefix) != dag_keys(leaf2.prefix)
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            pid_type="persistent_interleaved",
            tcgen05_ab_stages=2,
            tcgen05_epilogue_fanout="shared",
        )
    )
    assert "Float32(1e-06)" in source and "Float32(2e-06)" in source
    assert source.count("PipelineTmaStore.create(") == 2


@pytest.mark.parametrize("fault", ("fake", "alias", "missing_facts", "no_classifier"))
def test_prefix_does_not_replace_real_readonly_alias_proof(fault):
    actual = list(args())
    if fault == "alias":
        actual[2] = actual[4].view(-1)[:256]
    elif fault == "fake":
        with FakeTensorMode() as mode:
            actual = [mode.from_tensor(x) for x in actual]
    bound = cpu_bind(prefix_kernel, tuple(actual))
    if fault in ("fake", "alias"):
        assert not bound.config_spec.cute_tcgen05_search_enabled
        return
    key = "tensor_storage_disjoint_matrix_v1"
    table = (
        bound.env.bound_runtime_input_specialization_results
        if fault == "missing_facts"
        else bound.env.runtime_input_specializations
    )
    saved = table.pop(key)
    try:
        assert chains(bound) is None
    finally:
        table[key] = saved


def prefix_graph(count=1):
    graph = torch.fx.Graph()
    source = graph.placeholder("u")
    source.meta["val"] = torch.empty(64)
    dim = graph.placeholder("m")
    load = graph.call_function(hl.load, (source, [dim], None, None))
    load.meta["val"] = torch.empty(64)
    previous = load
    nodes = []
    for _ in range(count):
        c = graph.call_function(hl.full, ([], 1e-6, torch.float32, None))
        c.meta["val"] = torch.empty(())
        current = graph.call_function(torch.ops.aten.add.Tensor, (previous, c))
        current.meta["val"] = torch.empty(64)
        nodes.extend((c, current))
        previous = current
    return graph, load, nodes, previous


@pytest.mark.parametrize(
    "fault",
    (
        "cycle",
        "cross_graph",
        "wrong_dtype",
        "wrong_shape",
        "nonfinite",
        "unknown",
        "kwargs",
        "two_loads",
    ),
)
def test_bounded_prefix_prover_rejects_untyped_or_impure_graphs(fault):
    g, load, nodes, root = prefix_graph()
    if fault == "cycle":
        root.args = (root, nodes[0])
    elif fault == "cross_graph":
        other = torch.fx.Graph()
        c = other.placeholder("c")
        c.meta["val"] = torch.empty(())
        root.args = (load, c)
    elif fault == "wrong_dtype":
        load.meta["val"] = torch.empty(64, dtype=torch.bfloat16)
    elif fault == "wrong_shape":
        root.meta["val"] = torch.empty((64, 1))
    elif fault == "nonfinite":
        nodes[0].args = ([], float("nan"), torch.float32, None)
    elif fault == "unknown":
        root.target = torch.ops.aten.sin.default
    elif fault == "kwargs":
        root.kwargs = {"alpha": 2}
    elif fault == "two_loads":
        second = g.call_function(hl.load, load.args)
        second.meta["val"] = load.meta["val"]
        root.args = (load, second)
    assert prove_m_vector_scalar_prefix(root, (64,), torch.device("cpu")) is None


@pytest.mark.parametrize(
    "fault", ("wrong_M_index", "wrong_view_axis", "inferred_view", "mask", "effect")
)
def test_explicit_prefix_view_retains_M_provenance_mask_and_effect_guards(fault):
    bound = cpu_bind(prefix_kernel, args(m=128, n=128))
    complete = chains(bound)
    assert complete is not None
    leaf = complete[0][1].auxiliary_tensor_loads[0]
    assert isinstance(leaf, epi._MVectorScalarPrefixLoadExpr)
    graph = leaf.view_node.graph
    if fault == "wrong_M_index":
        store_value = complete[0][0].args[2]
        assert isinstance(store_value, torch.fx.Node)
        indices = epi._carrier_tile_index_nodes(store_value)
        assert indices is not None
        leaf.load_node.args = (leaf.load_node.args[0], [indices[1]], None, None)
    elif fault == "wrong_view_axis":
        shape = leaf.view_node.args[1]
        assert isinstance(shape, (list, tuple))
        leaf.view_node.args = (leaf.view_node.args[0], [1, shape[0]])
    elif fault == "inferred_view":
        leaf.view_node.args = (leaf.view_node.args[0], [-1, 1])
    elif fault == "mask":
        leaf.load_node.args = (*leaf.load_node.args[:2], True, None)
    else:
        with graph.inserting_before(complete[0][0]):
            graph.call_function(
                hl.store,
                (leaf.load_node.args[0], leaf.load_node.args[1], leaf.load_node, None),
            )
    assert chains(bound) is None


@pytest.mark.parametrize("literal", (1e100, 2**80, True, float("nan")))
def test_weak_literal_nonfinite_overflow_and_boolean_are_declined(literal):
    _g, load, _nodes, root = prefix_graph()
    root.args = (load, literal)
    assert prove_m_vector_scalar_prefix(root, (64,), torch.device("cpu")) is None


def test_prefix_owned_DCE_preserves_other_live_prefix_uses_and_stores():
    import ast

    from helion._compiler.ast_read_writes import dead_assignment_elimination
    from helion._compiler.generate_ast import GenerateAST

    _g, _load, _nodes, root = prefix_graph()
    proof = prove_m_vector_scalar_prefix(root, (64,), torch.device("cpu"))
    assert proof is not None
    body: list[ast.AST] = list(
        ast.parse(
            "raw = load_u()\nconstant = full_scalar()\nprefix = raw + constant\nunused = raw + constant\nstore_out(prefix)\n"
        ).body
    )
    owners: dict[int, list[tuple[list[ast.AST], ast.AST]]] = {
        id(proof.load_node): [(body, body[0])],
        id(root.args[1]): [(body, body[1])],
        id(root): [(body, body[2]), (body, body[3])],
    }
    from helion._compiler.device_function import DeviceFunction

    codegen = GenerateAST.__new__(GenerateAST)
    device = DeviceFunction.__new__(DeviceFunction)
    device.dce_vars = []
    codegen.device_function = device
    codegen._statements_by_owner_node_id = owners
    codegen.allow_dead_assignments_owned_by_nodes(proof.owned_nodes)
    assert len(body) == 5  # Registration never deletes source statements.
    dead_assignment_elimination(body, device.dce_vars)
    live = "\n".join(ast.unparse(statement) for statement in body)
    assert "store_out(prefix)" in live and "prefix = raw + constant" in live
    assert "raw = load_u()" in live and "constant = full_scalar()" in live
    assert "unused =" not in live


@pytest.mark.parametrize("count,accepted", ((28, True), (32, False), (40, False)))
def test_prefix_depth_and_whole_node_caps_are_bounded(count, accepted):
    _g, _load, _nodes, root = prefix_graph(count)
    proof = prove_m_vector_scalar_prefix(root, (64,), torch.device("cpu"))
    assert (proof is not None) == accepted
