from __future__ import annotations

import ast
from contextlib import ExitStack
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import cute_mma
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._compiler.cute.epilogue_fanout import _chain_key
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl
from helion.language import creation_ops

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_codegen_target(), forbid_cuda_work():
        support = cute_mma.get_cute_mma_support()
        support.tcgen05_tf32 = True
        yield


def _call(graph, target, args, dtype=torch.float32, shape=(4, 16)):
    node = graph.call_function(target, args)
    node.meta["val"] = torch.empty(shape, dtype=dtype, device="cpu")
    return node


def _input(graph):
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
    return anchor


def _constant(graph, value, dtype=torch.float32):
    return _call(graph, creation_ops.full, ([], value, dtype, None), dtype, ())


def _gelu(graph, x):
    half = _call(graph, torch.ops.aten.mul.Tensor, (x, _constant(graph, 0.5)))
    square = _call(graph, torch.ops.aten.mul.Tensor, (x, x))
    cube = _call(graph, torch.ops.aten.mul.Tensor, (square, x))
    term = _call(graph, torch.ops.aten.mul.Tensor, (cube, _constant(graph, 0.044715)))
    term = _call(graph, torch.ops.aten.add.Tensor, (x, term))
    term = _call(
        graph, torch.ops.aten.mul.Tensor, (term, _constant(graph, 0.7978845608028654))
    )
    term = _call(graph, torch.ops.aten.tanh.default, (term,))
    term = _call(graph, torch.ops.aten.add.Tensor, (term, _constant(graph, 1.0)))
    return _call(graph, torch.ops.aten.mul.Tensor, (half, term))


def _analyze(value, anchor, *, graphs=None):
    if graphs is None:
        graphs = [SimpleNamespace(graph=value.graph, graph_id=0)]
    with ExitStack() as stack:
        if not CompileEnvironment.has_current():
            stack.enter_context(
                patch.object(
                    CompileEnvironment,
                    "current",
                    return_value=SimpleNamespace(
                        settings=SimpleNamespace(fast_math=False)
                    ),
                )
            )
        return epi.analyze_tcgen05_unary_epilogue_chain(
            None,
            value,
            target_fx_nodes={anchor},
            inner_outputs_by_graph_id={},
            graphs=graphs,
        )


def _render(analysis, *, aux=None):
    assert analysis is not None
    names = itertools.count()
    source, expression = analysis[0].render_prelude_and_expr(
        "original_acc",
        lambda prefix: f"{prefix}_{next(names)}",
        "",
        aux,
    )
    ast.parse(source + f"result = {expression}")
    return source, expression


def _evaluate(source, expression, values, **extra):
    class Constructors(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cutlass"
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="cast", ctx=ast.Load()),
                        args=[*node.args, node.func],
                        keywords=[],
                    ),
                    node,
                )
            return node

    def cast(value, dtype):
        return (
            value.to(dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=dtype, device="cpu")
        )

    tree = ast.fix_missing_locations(
        Constructors().visit(ast.parse(source + f"result = {expression}"))
    )
    namespace = {
        "original_acc": values,
        "cast": cast,
        "cutlass": SimpleNamespace(
            Float32=torch.float32, Float16=torch.float16, BFloat16=torch.bfloat16
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(tanh=torch.tanh, expm1=torch.expm1), where=torch.where
        ),
        **extra,
    }
    exec(compile(tree, "<cpu_epilogue_dag>", "exec"), namespace)
    return namespace["result"]


def test_gelu_same_anchor_dag_preserves_association_and_common_prefix():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    result = _gelu(graph, anchor)
    analysis = _analyze(result, anchor)
    assert isinstance(analysis[0].steps[0].expr, epi._PointwiseDagExpr)
    source, expression = _render(analysis)
    assert source.count("original_acc * original_acc") == 1
    values = torch.linspace(-3.5, 3.5, 64, device="cpu").reshape(4, 16)
    expected = (values * 0.5) * (
        torch.tanh(
            (values + ((values * values) * values) * 0.044715) * 0.7978845608028654
        )
        + 1.0
    )
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dag_reconvergence_keeps_intermediate_rounding_and_original_accumulator(dtype):
    graph = torch.fx.Graph()
    anchor = _input(graph)
    rounded = _call(
        graph, torch.ops.prims.convert_element_type.default, (anchor, dtype), dtype
    )
    bias = _call(graph, torch.ops.aten.add.Tensor, (rounded, 0.03125), dtype)
    square = _call(graph, torch.ops.aten.mul.Tensor, (bias, bias), dtype)
    promoted = _call(
        graph, torch.ops.prims.convert_element_type.default, (square, torch.float32)
    )
    result = _call(graph, torch.ops.aten.sub.Tensor, (promoted, anchor))
    source, expression = _render(_analyze(result, anchor))
    assert source.count(" + (0.03125)") <= 1
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    expected = (
        (values.to(dtype) + 0.03125) * (values.to(dtype) + 0.03125)
    ).float() - values
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, rtol=0, atol=0
    )


def test_typed_constants_signed_zero_and_dag_fanout_keys():
    def chain(literal, dtype):
        graph = torch.fx.Graph()
        anchor = _input(graph)
        scaled = _call(
            graph, torch.ops.aten.mul.Tensor, (anchor, _constant(graph, literal, dtype))
        )
        result = _call(graph, torch.ops.aten.mul.Tensor, (scaled, scaled))
        return _analyze(result, anchor)[0]

    first = chain(0.3, torch.bfloat16)
    assert _chain_key(first) == _chain_key(chain(0.3, torch.bfloat16))
    assert _chain_key(first) != _chain_key(chain(0.3, torch.float32))
    assert _chain_key(chain(0.0, torch.float32)) != _chain_key(
        chain(-0.0, torch.float32)
    )
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    source, expression = _render((first, None))
    scale = torch.tensor(0.3, dtype=torch.bfloat16, device="cpu").float()
    expected = (values * scale) * (values * scale)
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "second_anchor",
        "unknown",
        "shape",
        "stride",
        "mutable",
        "cross_graph",
        "depth",
        "node_limit",
    ],
)
def test_unproved_dag_structure_stays_rejected(mutation):
    graph = torch.fx.Graph()
    anchor = _input(graph)
    result = _call(graph, torch.ops.aten.mul.Tensor, (anchor, anchor))
    if mutation == "second_anchor":
        rhs = _call(graph, torch.ops.aten.mm.default, (anchor, anchor))
        result = _call(graph, torch.ops.aten.add.Tensor, (result, rhs))
    elif mutation == "unknown":
        result = _call(graph, torch.ops.aten.sin.default, (result,))
    elif mutation == "shape":
        result.meta["val"] = torch.empty((2, 32), device="cpu")
    elif mutation == "stride":
        result.meta["val"] = torch.empty((16, 4), device="cpu").T
    elif mutation == "mutable":
        graph.call_function(torch.ops.aten.add_.Tensor, (anchor, 1.0))
    elif mutation == "cross_graph":
        other = torch.fx.Graph()
        term = _call(other, torch.ops.aten.mul.Tensor, (anchor, anchor))
        result = _call(graph, torch.ops.aten.add.Tensor, (result, term))
    elif mutation == "depth":
        for _ in range(35):
            result = _call(graph, torch.ops.aten.mul.Tensor, (result, anchor))
    elif mutation == "node_limit":
        terms = [
            _call(
                graph, torch.ops.aten.mul.Tensor, (anchor, _constant(graph, float(i)))
            )
            for i in range(40)
        ]
        while len(terms) > 1:
            terms = [
                _call(graph, torch.ops.aten.add.Tensor, tuple(terms[i : i + 2]))
                if i + 1 < len(terms)
                else terms[i]
                for i in range(0, len(terms), 2)
            ]
        result = terms[0]
    assert _analyze(result, anchor) is None


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", dot_precision="tf32"
)
def _biased_gelu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    activated: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        x = acc + bias[col]
        out[row, col] = x
        half = x * hl.full([], 0.5, dtype=torch.float32)
        square = x * x
        cube = square * x
        term = x + cube * hl.full([], 0.044715, dtype=torch.float32)
        term = term * hl.full([], 0.7978845608028654, dtype=torch.float32)
        term = torch.tanh(term) + hl.full([], 1.0, dtype=torch.float32)
        activated[row, col] = half * term


def _args(*, alias=False, offset=False):
    a, b = (
        torch.empty((128, 128), dtype=torch.float32, device="cpu") for _ in range(2)
    )
    out = torch.empty((128, 128), dtype=torch.float32, device="cpu")
    bias = (
        out.reshape(-1)[1:129]
        if offset
        else out[0]
        if alias
        else torch.empty((128,), dtype=torch.float32, device="cpu")
    )
    return a, b, bias, out, torch.empty_like(out)


def _native_config():
    return helion.Config(block_sizes=[128, 64, 64], pid_type="flat")


def test_real_disjoint_external_output_bias_admits_native_dag_and_two_original_stores():
    bound = cpu_bind(_biased_gelu, _args())
    assert bound.config_spec.cute_tcgen05_search_enabled
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        anchor = next(
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        chains = analyze_tcgen05_matmul_store_chains(graphs, anchor)
        assert chains is not None and len(chains) == 2
        assert isinstance(chains[1][1].steps[0].expr, epi._PointwiseDagExpr)
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    bias = torch.linspace(-0.1, 0.1, 16, device="cpu")
    x = values + bias
    expected = [
        x,
        (x * 0.5)
        * (torch.tanh((x + (x * x) * x * 0.044715) * 0.7978845608028654) + 1.0),
    ]
    for (_store, chain), wanted in zip(chains, expected, strict=True):
        aux = dict.fromkeys(chain.auxiliary_tensor_loads, "bias_values")
        prelude, expression = _render((chain, anchor), aux=aux)
        assert prelude.count("original_acc + bias_values") == 1
        torch.testing.assert_close(
            _evaluate(prelude, expression, values, bias_values=bias),
            wanted,
            atol=0,
            rtol=0,
        )
    source = bound.to_code(_native_config())
    ast.parse(source)
    assert "tcgen05_ab_tma" in source
    assert "tcgen05_dag_step" in source
    assert "tcgen05_d_tma" in source


@pytest.mark.parametrize("offset", [False, True])
def test_overlapping_bias_and_output_views_do_not_admit_native_dag(offset):
    args = _args(alias=True, offset=offset)
    bound = cpu_bind(_biased_gelu, args)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert "cute_collective_mma" in bound.config_spec._flat_fields()


def test_fake_only_or_absent_bound_alias_facts_fail_closed():
    from torch._subclasses.fake_tensor import FakeTensorMode

    args = _args()
    with FakeTensorMode() as mode:
        fake = tuple(mode.from_tensor(value) for value in args)
    bound = cpu_bind(_biased_gelu, fake)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    real_bound = cpu_bind(_biased_gelu, args)
    with real_bound.env, real_bound.host_function:
        saved = real_bound.env.bound_runtime_input_specialization_results
        real_bound.env.bound_runtime_input_specialization_results = {}
        try:
            graphs = real_bound.host_function.device_ir.graphs
            anchor = next(
                node
                for graph in graphs
                for node in graph.graph.nodes
                if node.target is torch.ops.aten.addmm.default
            )
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
        finally:
            real_bound.env.bound_runtime_input_specialization_results = saved


def test_alias_pattern_rebinding_cannot_reuse_a_disjoint_preflight():
    disjoint = cpu_bind(_biased_gelu, _args())
    overlap = cpu_bind(_biased_gelu, _args(alias=True))
    key = "tensor_storage_disjoint_matrix_v1"
    assert (
        disjoint.env.bound_runtime_input_specialization_results[key]
        != overlap.env.bound_runtime_input_specialization_results[key]
    )
    real_disjoint = _args()
    real_overlap = _args(alias=True)
    real_offset = _args(alias=True, offset=True)
    with disjoint.env, disjoint.host_function:
        specialization = disjoint.env.runtime_input_specializations[key]
        values = disjoint.env.bound_runtime_input_specialization_results[key]
        extractors = tuple(
            (lambda args, i=i: args[i]) for i in range(len(specialization.sources))
        )
        guard = specialization.prepared_guard_factory(extractors, values)
        assert guard(real_disjoint)
        assert not guard(real_overlap)
        assert not guard(real_offset)


@pytest.mark.parametrize(
    "integer",
    [
        0,
        768,
        (1 << 24) - 1,
        1 << 24,
        (1 << 24) + 1,
        (1 << 24) + 3,
        -(1 << 24) - 1,
        (1 << 31) + 1,
        (1 << 53) + 1,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_integer_scalar_leaves_render_validated_float_constructors(integer, dtype):
    graph = torch.fx.Graph()
    constant = _constant(graph, integer, dtype)
    leaf = epi._scalar_constant(constant)
    if abs(integer) > (1 << 24):
        assert leaf is None
        return
    assert leaf is not None and leaf.literal == repr(float(integer))
    source, expression = epi._render_auxiliary_tensor_expr(
        leaf, "unused", {}, lambda prefix: prefix, ""
    )
    tree = ast.parse(expression, mode="eval")
    literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert literals and all(isinstance(value, float) for value in literals)
    actual = _evaluate(source, expression, torch.empty((), device="cpu"))
    expected = torch.full((), integer, dtype=dtype, device="cpu").float()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_huge_integer_leaf_rejects_without_conversion_overflow_and_floats_unchanged():
    graph = torch.fx.Graph()
    assert epi._scalar_constant(_constant(graph, 10**1000)) is None
    midpoint = (1 << 62) + (1 << 38) + 1
    assert torch.full((), midpoint, dtype=torch.float32, device="cpu") != torch.full(
        (), float(midpoint), dtype=torch.float32, device="cpu"
    )
    assert epi._scalar_constant(_constant(graph, midpoint)) is None
    for value in [0.044715, -0.0, 16777217.0]:
        assert epi._scalar_constant(_constant(graph, value)).literal == repr(value)


def test_duplicate_operand_metadata_survives_codegen_copy_without_node_references():
    bound = cpu_bind(_biased_gelu, _args())
    square = next(
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.name == "square"
    )
    assert square.args[1] is None
    assert square.meta["cute_duplicate_arg_positions"] == ((1, 0),)
    source = bound.to_code(_native_config())
    assert "tcgen05_dag_step" in source
    assert "None *" not in source


def test_simple_linear_chains_do_not_enter_dag_rendering():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    result = _call(graph, torch.ops.aten.mul.Tensor, (anchor, 0.5))
    analysis = _analyze(result, anchor)
    assert not isinstance(analysis[0].steps[0].expr, epi._PointwiseDagExpr)
    source, _expression = _render(analysis)
    assert "tcgen05_chain_step" in source and "tcgen05_dag_step" not in source


def test_cached_binding_alias_changes_choose_another_source_family():
    disjoint_args = _args()
    overlap_args = _args(alias=True)
    disjoint = _biased_gelu.bind(disjoint_args)
    overlap = _biased_gelu.bind(overlap_args)
    rebound = _biased_gelu.bind(_args())
    assert disjoint.config_spec.cute_tcgen05_search_enabled
    assert not overlap.config_spec.cute_tcgen05_search_enabled
    assert disjoint is not overlap
    assert rebound.config_spec.cute_tcgen05_search_enabled
    disjoint_keys = [
        key for key, bound in _biased_gelu._bound_kernels.items() if bound is disjoint
    ]
    overlap_keys = [
        key for key, bound in _biased_gelu._bound_kernels.items() if bound is overlap
    ]
    assert disjoint_keys and overlap_keys
    assert disjoint_keys[0].extra_results != overlap_keys[0].extra_results


def test_late_registration_cannot_refresh_a_cleared_alias_snapshot():
    bound = cpu_bind(_biased_gelu, _args())
    key = "tensor_storage_disjoint_matrix_v1"
    with bound.env, bound.host_function:
        spec = bound.env.runtime_input_specializations[key]
        saved = bound.env.bound_runtime_input_specialization_results.pop(key)
        try:
            bound.env.register_runtime_input_specialization(key, spec)
            assert key not in bound.env.bound_runtime_input_specialization_results
        finally:
            bound.env.bound_runtime_input_specialization_results[key] = saved


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _biased_selu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    activated: torch.Tensor,
    two_outputs: hl.constexpr,
    recast_inputs: hl.constexpr = False,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            if recast_inputs:
                left = a[row, inner].to(torch.float32).to(torch.bfloat16)
                right = b[inner, col].to(torch.float32).to(torch.bfloat16)
                acc = torch.addmm(acc, left, right)
            else:
                acc = torch.addmm(acc, a[row, inner], b[inner, col])
        bias_f32 = bias[col].to(torch.float32)
        one = hl.full([], 1.0, dtype=torch.float32)
        x = one * bias_f32 + one * acc
        if two_outputs:
            out[row, col] = x.to(torch.bfloat16)
        x = x.to(torch.float32)
        condition = x > hl.full([], 0.0, dtype=torch.float32)
        positive = x * hl.full([], 1.0507009873554805, dtype=torch.float32)
        negative = torch.expm1(x) * hl.full([], 1.7580993408473766, dtype=torch.float32)
        result = torch.where(condition, positive, negative)
        activated[row, col] = result.to(torch.bfloat16)


@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("two_outputs", [False, True])
@pytest.mark.parametrize("recast_inputs", [False, True])
def test_direct_or_exact_recast_bf16_operands_and_biased_selu_admit_native_dag(
    bias_dtype, two_outputs, recast_inputs
):
    args = (
        torch.empty((128, 128), dtype=torch.bfloat16, device="cpu"),
        torch.empty((128, 128), dtype=torch.bfloat16, device="cpu"),
        torch.empty((128,), dtype=bias_dtype, device="cpu"),
        torch.empty((128, 128), dtype=torch.bfloat16, device="cpu"),
        torch.empty((128, 128), dtype=torch.bfloat16, device="cpu"),
        two_outputs,
        recast_inputs,
    )
    bound = cpu_bind(_biased_selu, args)
    assert bound.config_spec.cute_tcgen05_search_enabled
    source = bound.to_code(_native_config())
    ast.parse(source)
    assert "tcgen05_ab_tma" in source
    assert "cute.math.expm1(" in source and "cute.where(" in source
    assert "dot_product" not in source


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_selu_predicate_expm1_and_selection_preserve_ieee_and_near_zero(dtype):
    graph = torch.fx.Graph()
    anchor = _input(graph)
    condition = _call(
        graph, torch.ops.aten.gt.Tensor, (anchor, _constant(graph, 0.0)), torch.bool
    )
    positive = _call(
        graph, torch.ops.aten.mul.Tensor, (anchor, _constant(graph, 1.0507009873554805))
    )
    expm1 = _call(graph, torch.ops.aten.expm1.default, (anchor,))
    negative = _call(
        graph, torch.ops.aten.mul.Tensor, (expm1, _constant(graph, 1.7580993408473766))
    )
    result = _call(graph, torch.ops.aten.where.self, (condition, positive, negative))
    result = _call(
        graph, torch.ops.prims.convert_element_type.default, (result, dtype), dtype
    )
    analysis = _analyze(result, anchor)
    source, expression = _render(analysis)
    assert source.count("cute.math.expm1(") == 1
    _chain_key(analysis[0])
    values = torch.tensor(
        [-float("inf"), -3.0, -1e-8, -0.0, 0.0, 1e-8, 3.0, float("inf"), float("nan")],
        dtype=torch.float32,
        device="cpu",
    )
    expected = (
        torch.where(
            values > 0.0,
            values * 1.0507009873554805,
            torch.expm1(values) * 1.7580993408473766,
        )
        .to(dtype)
        .float()
    )
    actual = _evaluate(source, expression, values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual[3:5]), torch.signbit(expected[3:5]))


def test_boolean_or_mismatched_selection_without_proof_is_rejected():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    predicate = _call(
        graph, torch.ops.aten.gt.Tensor, (anchor, _constant(graph, 0.0)), torch.bool, ()
    )
    squared = _call(graph, torch.ops.aten.mul.Tensor, (anchor, anchor))
    result = _call(graph, torch.ops.aten.where.self, (predicate, squared, anchor))
    assert _analyze(result, anchor) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_exact_operand_roundtrip_records_casts_and_preserves_supported_values(dtype):
    from helion.language import memory_ops

    graph = torch.fx.Graph()
    source = graph.placeholder("host_tensor")
    source.meta["val"] = torch.empty((4, 16), dtype=dtype, device="cpu")
    load = _call(
        graph, memory_ops.load, (source, [slice(None), slice(None)], None, None), dtype
    )
    promoted = _call(
        graph, torch.ops.prims.convert_element_type.default, (load, torch.float32)
    )
    recast = _call(
        graph, torch.ops.prims.convert_element_type.default, (promoted, dtype), dtype
    )
    before = tuple((node, node.args, node.kwargs) for node in graph.nodes)
    proof = cute_mma._exact_half_mma_operand_round_trip(recast)
    assert proof == (load, (promoted, recast))
    native = cute_mma._trace_to_mma_operand(recast, role="lhs")
    assert (
        native is not None
        and native.source_fake.dtype is dtype
        and native.mma_dtype is dtype
    )
    assert native.exact_round_trip_casts == (promoted, recast)
    assert (
        native.terminal is recast and native.load is load and native.cast_dtype is None
    )
    assert before == tuple((node, node.args, node.kwargs) for node in graph.nodes)
    bits = torch.arange(65536, dtype=torch.int32, device="cpu").to(torch.int16)
    values = bits.view(dtype)
    roundtrip = values.float().to(dtype)
    torch.testing.assert_close(roundtrip, values, atol=0, rtol=0, equal_nan=True)
    finite = ~torch.isnan(values)
    assert torch.equal(roundtrip.view(torch.int16)[finite], bits[finite])


@pytest.mark.parametrize(
    "mutation",
    [
        "narrowing",
        "other_half",
        "arithmetic",
        "masked",
        "kwargs",
        "shape",
        "stride",
        "source_width",
        "extra_cast",
    ],
)
def test_operand_roundtrip_rejects_any_loss_or_intervening_work(mutation):
    from helion.language import memory_ops

    graph = torch.fx.Graph()
    dtype = torch.float32 if mutation == "narrowing" else torch.bfloat16
    source = graph.placeholder("host_tensor")
    source.meta["val"] = torch.empty((4, 16), dtype=dtype, device="cpu")
    load = _call(
        graph, memory_ops.load, (source, [slice(None), slice(None)], None, None), dtype
    )
    promoted = _call(
        graph, torch.ops.prims.convert_element_type.default, (load, torch.float32)
    )
    recast = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (promoted, torch.bfloat16),
        torch.bfloat16,
    )
    if mutation == "other_half":
        recast.args = (promoted, torch.float16)
        recast.meta["val"] = torch.empty((4, 16), dtype=torch.float16, device="cpu")
    elif mutation == "arithmetic":
        middle = _call(graph, torch.ops.aten.add.Tensor, (promoted, 0.25))
        recast.args = (middle, torch.bfloat16)
    elif mutation == "masked":
        mask = _call(graph, torch.ops.aten.gt.Tensor, (load, 0.0), torch.bool)
        load.args = (source, [slice(None), slice(None)], mask, None)
    elif mutation == "kwargs":
        promoted.kwargs = {"device": None}
    elif mutation == "shape":
        promoted.meta["val"] = torch.empty((2, 32), device="cpu")
    elif mutation == "stride":
        promoted.meta["val"] = torch.empty((16, 4), device="cpu").T
    elif mutation == "source_width":
        source.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
    elif mutation == "extra_cast":
        wide = _call(
            graph,
            torch.ops.prims.convert_element_type.default,
            (load, torch.float64),
            torch.float64,
        )
        promoted.args = (wide, torch.float32)
    assert cute_mma._exact_half_mma_operand_round_trip(recast) is None


def test_bound_operand_planning_shares_exact_source_format_and_complete_store_lifetime():
    args = (
        *(
            torch.empty(shape, dtype=torch.bfloat16, device="cpu")
            for shape in [(128, 128), (128, 128), (128,), (128, 128), (128, 128)]
        ),
        True,
        True,
    )
    bound = cpu_bind(_biased_selu, args)
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        mma = next(
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        for operand, role in zip(mma.args[1:], ("lhs", "rhs"), strict=True):
            admitted = cute_mma._analyze_mma_operand(operand, bound.env)
            lowered = cute_mma._trace_to_mma_operand(operand, role=role)
            assert admitted is not None and lowered is not None
            assert (
                admitted.load is lowered.load and admitted.terminal is lowered.terminal
            )
            assert admitted.exact_round_trip_casts == lowered.exact_round_trip_casts
            assert admitted.source_fake is lowered.source_fake
            assert admitted.mma_dtype is lowered.mma_dtype is torch.bfloat16
            assert cute_mma._operand_load_path_exclusive(admitted, mma)
        stores = cute_mma._trace_mma_to_stores(mma, graphs)
        assert stores is not None and len(stores) == 2


@pytest.mark.parametrize("alias", ["input", "same_output", "overlapping_outputs"])
def test_dag_cannot_reorder_an_input_write_or_aliased_output_pair(alias):
    args = list(_args())
    if alias == "input":
        args[3] = args[0]
    elif alias == "same_output":
        args[4] = args[3]
    else:
        backing = torch.empty((129, 128), dtype=torch.float32, device="cpu")
        args[3], args[4] = backing[:128], backing[1:]
    bound = cpu_bind(_biased_gelu, tuple(args))
    assert not bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cast_factor", [False, True])
def test_rank0_constant_with_half_operand_requires_explicit_normalization(
    dtype, cast_factor
):
    graph = torch.fx.Graph()
    anchor = _input(graph)
    x = _call(
        graph, torch.ops.prims.convert_element_type.default, (anchor, dtype), dtype
    )
    factor = _constant(graph, 0.3, torch.float32)
    if cast_factor:
        factor = _call(
            graph,
            torch.ops.prims.convert_element_type.default,
            (factor, dtype),
            dtype,
            (),
        )
    y = _call(graph, torch.ops.aten.mul.Tensor, (x, factor), dtype)
    result = _call(graph, torch.ops.aten.mul.Tensor, (y, y), dtype)
    analysis = _analyze(result, anchor)
    if not cast_factor:
        assert analysis is None
        return
    source, expression = _render(analysis)
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    # Explicit tensor normalization is the oracle; a CPU rank-zero scalar
    # fast path may keep wider opmath and cannot prove all-device semantics.
    factor_value = torch.full((), 0.3, dtype=torch.float32, device="cpu").to(dtype)
    y_value = values.to(dtype) * factor_value
    expected = (y_value * y_value).float()
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cast_threshold", [False, True])
def test_half_threshold_predicate_uses_operand_precision_not_bool_result(
    dtype, cast_threshold
):
    graph = torch.fx.Graph()
    anchor = _input(graph)
    x = _call(
        graph, torch.ops.prims.convert_element_type.default, (anchor, dtype), dtype
    )
    threshold = _constant(graph, 0.3, torch.float32)
    if cast_threshold:
        threshold = _call(
            graph,
            torch.ops.prims.convert_element_type.default,
            (threshold, dtype),
            dtype,
            (),
        )
    condition = _call(graph, torch.ops.aten.gt.Tensor, (x, threshold), torch.bool)
    square = _call(graph, torch.ops.aten.mul.Tensor, (x, x), dtype)
    result = _call(graph, torch.ops.aten.where.self, (condition, x, square), dtype)
    analysis = _analyze(result, anchor)
    if not cast_threshold:
        assert analysis is None
        return
    source, expression = _render(analysis)
    threshold_value = torch.full((), 0.3, dtype=torch.float32, device="cpu")
    values = torch.full(
        (4, 16), threshold_value.to(dtype).item(), dtype=torch.float32, device="cpu"
    )
    x_value = values.to(dtype)
    expected_condition = x_value > threshold_value
    assert not expected_condition.any()
    assert (x_value.float() > threshold_value).all()
    expected = torch.where(expected_condition, x_value, x_value * x_value).float()
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, atol=0, rtol=0
    )


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", dot_precision="tf32"
)
def _rank0_normalization_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    kind: hl.constexpr,
    explicit: hl.constexpr,
    comparison: hl.constexpr,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        x = acc.to(torch.bfloat16)
        if kind == "load":
            raw_factor = scale[()]
        else:
            raw_factor = hl.full([], 0.3, dtype=torch.float32)
        if explicit:
            factor = raw_factor.to(torch.bfloat16)
        else:
            factor = raw_factor
        if comparison:
            condition = x > factor
            value = torch.where(condition, x, x * x)
        else:
            y = x * factor
            value = y * y
        out[row, col] = value.to(torch.float32) + acc
    return out


@pytest.mark.parametrize("kind", ["constant", "load"])
@pytest.mark.parametrize("comparison", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_rank0_device_load_and_constant_admission_fail_closed_on_mixed_precision(
    kind, comparison, explicit
):
    args = (
        torch.empty((128, 128), dtype=torch.float32, device="cpu"),
        torch.empty((128, 128), dtype=torch.float32, device="cpu"),
        torch.full((), 0.3, dtype=torch.float32, device="cpu"),
        kind,
        explicit,
        comparison,
    )
    bound = cpu_bind(_rank0_normalization_matmul, args)
    assert bound.config_spec.cute_tcgen05_search_enabled == explicit
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        mma = next(
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        chains = analyze_tcgen05_matmul_store_chains(graphs, mma)
    if not explicit:
        assert chains is None
        return
    assert chains is not None
    chain = chains[0][1]
    bindings = dict.fromkeys(chain.runtime_scalars, "scalar_value")
    source, expression = _render((chain, mma), aux=bindings)
    values = torch.full(
        (4, 16),
        torch.tensor(0.3, dtype=torch.bfloat16, device="cpu").item(),
        device="cpu",
    )
    x = values.to(torch.bfloat16)
    scalar = args[2]
    normalized = scalar.to(torch.bfloat16)
    if comparison:
        expected = torch.where(x > normalized, x, x * x).float() + values
    else:
        y = x * normalized
        expected = (y * y).float() + values
    torch.testing.assert_close(
        _evaluate(source, expression, values, scalar_value=scalar),
        expected,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_boolean_arithmetic_operand_is_explicitly_promoted_in_dag(dtype):
    graph = torch.fx.Graph()
    anchor = _input(graph)
    x = _call(
        graph, torch.ops.prims.convert_element_type.default, (anchor, dtype), dtype
    )
    condition = _call(
        graph, torch.ops.aten.gt.Tensor, (x, _constant(graph, 0.0, dtype)), torch.bool
    )
    masked = _call(graph, torch.ops.aten.mul.Tensor, (x, condition), dtype)
    promoted = _call(
        graph, torch.ops.prims.convert_element_type.default, (masked, torch.float32)
    )
    result = _call(graph, torch.ops.aten.add.Tensor, (promoted, anchor))
    source, expression = _render(_analyze(result, anchor))
    values = (
        torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan"), 1e-8],
            device="cpu",
        )
        .repeat(8)
        .reshape(4, 16)
    )
    expected = (values.to(dtype) * (values.to(dtype) > 0)).float() + values
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, atol=0, rtol=0, equal_nan=True
    )
    assert ".to(cutlass.Float32)" in source


def test_where_branches_allow_only_proved_lossless_widening():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    rounded = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (anchor, torch.bfloat16),
        torch.bfloat16,
    )
    condition = _call(
        graph, torch.ops.aten.gt.Tensor, (anchor, _constant(graph, 0.0)), torch.bool
    )
    selected = _call(graph, torch.ops.aten.where.self, (condition, rounded, anchor))
    source, expression = _render(_analyze(selected, anchor))
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    expected = torch.where(values > 0, values.to(torch.bfloat16), values)
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, atol=0, rtol=0
    )
    selected.meta["val"] = torch.empty((4, 16), dtype=torch.bfloat16, device="cpu")
    assert _analyze(selected, anchor) is None


def test_explicit_cast_cannot_hide_unproved_half_binary_child():
    graph = torch.fx.Graph()
    anchor = _input(graph)
    x = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (anchor, torch.bfloat16),
        torch.bfloat16,
    )
    mixed = _call(
        graph, torch.ops.aten.mul.Tensor, (x, _constant(graph, 0.3)), torch.bfloat16
    )
    promoted = _call(
        graph, torch.ops.prims.convert_element_type.default, (mixed, torch.float32)
    )
    result = _call(graph, torch.ops.aten.add.Tensor, (promoted, anchor))
    assert _analyze(result, anchor) is None
