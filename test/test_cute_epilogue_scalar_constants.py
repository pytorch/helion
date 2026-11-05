from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

import helion
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute.epilogue_fanout import _chain_key
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl
from helion.language import creation_ops


def _full(
    graph: torch.fx.Graph,
    value: object = 0.3,
    dtype: torch.dtype = torch.float32,
) -> torch.fx.Node:
    node = graph.call_function(creation_ops.full, ([], value, dtype, None))
    node.meta["val"] = torch.empty((), dtype=dtype, device="cpu")
    return node


def _call(
    graph: torch.fx.Graph,
    target: Callable[..., object],
    args: tuple[object, ...],
    dtype: torch.dtype = torch.float32,
    shape: tuple[int, ...] = (4, 16),
) -> torch.fx.Node:
    node = graph.call_function(target, args)
    node.meta["val"] = torch.empty(shape, dtype=dtype, device="cpu")
    return node


def _chain(
    value: torch.fx.Node, anchor: torch.fx.Node
) -> cute_epilogue.Tcgen05UnaryEpilogueChain:
    analysis = cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
        None, value, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
    )
    assert analysis is not None and analysis[1] is anchor
    return analysis[0]


def _render(
    chain: cute_epilogue.Tcgen05UnaryEpilogueChain,
) -> tuple[str, str]:
    names = itertools.count()
    source, expression = chain.render_prelude_and_expr(
        "values", lambda prefix: f"{prefix}_{next(names)}", ""
    )
    # Parse the actual CuTe source before interpreting its constructor calls
    # as CPU tensor casts. No SDK compilation or device launch is involved.
    ast.parse(source + f"result = {expression}\n")
    return source, expression


def _evaluate(source: str, expression: str, values: torch.Tensor) -> torch.Tensor:
    class Constructors(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
                constructor = node.func.value
                if (
                    not isinstance(constructor, ast.Call)
                    or not isinstance(constructor.func, ast.Attribute)
                    or not isinstance(constructor.func.value, ast.Name)
                    or constructor.func.value.id != "cutlass"
                    or constructor.func.attr not in ("Int16", "Int32")
                    or len(constructor.args) != 1
                    or constructor.keywords
                    or len(node.args) != 1
                    or node.keywords
                    or not isinstance(node.args[0], ast.Attribute)
                    or not isinstance(node.args[0].value, ast.Name)
                    or node.args[0].value.id != "cutlass"
                ):
                    raise ValueError("unknown integer bitcast pattern")
                value = ast.literal_eval(constructor.args[0])
                integer_name = constructor.func.attr
                floating_name = node.args[0].attr
                expected = -32768 if integer_name == "Int16" else -2147483648
                if (
                    type(value) is not int
                    or value != expected
                    or (
                        integer_name == "Int16"
                        and floating_name not in ("Float16", "BFloat16")
                        or integer_name == "Int32"
                        and floating_name != "Float32"
                    )
                ):
                    raise ValueError("mismatched integer bitcast width or value")
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="bitcast_integer", ctx=ast.Load()),
                        args=[ast.Constant(value), constructor.func, node.args[0]],
                        keywords=[],
                    ),
                    node,
                )
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cutlass"
            ):
                if node.func.attr in ("Int16", "Int32"):
                    raise ValueError("integer constructor outside exact bitcast")
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="cast", ctx=ast.Load()),
                        args=[*node.args, node.func],
                        keywords=[],
                    ),
                    node,
                )
            return node

    def cast(value: object, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype)
        return torch.tensor(value, dtype=dtype, device="cpu")

    def bitcast_integer(
        value: int, integer_dtype: torch.dtype, floating_dtype: torch.dtype
    ) -> torch.Tensor:
        if (integer_dtype, floating_dtype) not in (
            (torch.int16, torch.float16),
            (torch.int16, torch.bfloat16),
            (torch.int32, torch.float32),
        ):
            raise ValueError("mismatched integer and floating storage widths")
        return torch.tensor(value, dtype=integer_dtype, device="cpu").view(
            floating_dtype
        )

    tree = ast.fix_missing_locations(
        Constructors().visit(ast.parse(source + f"result = {expression}\n"))
    )
    namespace = {
        "values": values,
        "cast": cast,
        "bitcast_integer": bitcast_integer,
        "cutlass": SimpleNamespace(
            Float16=torch.float16,
            BFloat16=torch.bfloat16,
            Float32=torch.float32,
            Int16=torch.int16,
            Int32=torch.int32,
        ),
    }
    exec(compile(tree, "<cpu_epilogue_constants>", "exec"), namespace)
    return namespace["result"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("literal", [0.3, 0.044715, 768, -0.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_typed_constant_leaf_preserves_value_dtype_and_operand_order(
    dtype: torch.dtype, literal: float, reverse: bool
) -> None:
    graph = torch.fx.Graph()
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
    constant = _full(graph, literal, dtype)
    leaf = cute_epilogue._scalar_constant(constant)
    expected_literal = (
        repr(float(literal)) if isinstance(literal, int) else repr(literal)
    )
    assert leaf == cute_epilogue._ScalarConstantExpr(expected_literal, dtype)
    assert cute_epilogue._is_auxiliary_tensor_expr_node(constant)
    assert not cute_epilogue._is_host_scalar_expr(leaf)
    args = (constant, anchor) if reverse else (anchor, constant)
    result = _call(graph, torch.ops.aten.sub.Tensor, args)
    chain = _chain(result, anchor)
    assert chain.auxiliary_tensor_loads == ()
    assert chain.runtime_scalars == ()
    source, expression = _render(chain)
    if expected_literal == "-0.0":
        integer = (
            "cutlass.Int32(-2147483648)"
            if dtype is torch.float32
            else "cutlass.Int16(-32768)"
        )
        assert f"{integer}.bitcast({cute_epilogue._FLOAT_CAST_TYPES[dtype]})" in source
    else:
        assert f"{cute_epilogue._FLOAT_CAST_TYPES[dtype]}({expected_literal})" in source
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    rounded = torch.tensor(literal, dtype=dtype, device="cpu").float()
    expected = rounded - values if reverse else values - rounded
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_constant_upcast_and_auxiliary_arithmetic_preserve_rounding(
    dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    constant = _full(graph, 0.3, dtype)
    product = _call(graph, torch.ops.aten.mul.Tensor, (constant, 1.0625), dtype, ())
    promoted = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (product, torch.float32),
        shape=(),
    )
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
    result = _call(graph, torch.ops.aten.mul.Tensor, (anchor, promoted))
    chain = _chain(result, anchor)
    source, expression = _render(chain)
    assert f".to({cute_epilogue._FLOAT_CAST_TYPES[dtype]})" in source
    assert ".to(cutlass.Float32)" in source
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    expected = values * (torch.tensor(0.3, dtype=dtype, device="cpu") * 1.0625).float()
    actual = _evaluate(source, expression, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(expected, values * (0.3 * 1.0625))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_constant_downcast_and_carrier_rounding_remain_explicit(
    dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    constant = _full(graph, 0.3)
    rounded_constant = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (constant, dtype),
        dtype,
        (),
    )
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
    rounded_carrier = _call(
        graph, torch.ops.prims.convert_element_type.default, (anchor, dtype), dtype
    )
    result = _call(
        graph, torch.ops.aten.add.Tensor, (rounded_carrier, rounded_constant), dtype
    )
    source, expression = _render(_chain(result, anchor))
    values = torch.linspace(-2, 2, 64, device="cpu").reshape(4, 16)
    expected = (
        values.to(dtype)
        + torch.tensor(0.3, dtype=torch.float32, device="cpu").to(dtype)
    ).float()
    torch.testing.assert_close(
        _evaluate(source, expression, values), expected, rtol=0, atol=0
    )


def test_constant_fanout_identity_preserves_dtype_literal_and_signed_zero() -> None:
    def chain(literal: float, dtype: torch.dtype):
        graph = torch.fx.Graph()
        anchor = graph.placeholder("mma")
        anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
        result = _call(
            graph, torch.ops.aten.mul.Tensor, (anchor, _full(graph, literal, dtype))
        )
        return _chain(result, anchor)

    first = chain(0.3, torch.float16)
    assert first == chain(0.3, torch.float16)
    assert _chain_key(first) == _chain_key(chain(0.3, torch.float16))
    assert _chain_key(first) != _chain_key(chain(0.3, torch.bfloat16))
    assert _chain_key(first) != _chain_key(chain(0.4, torch.float16))
    positive = chain(0.0, torch.float32)
    negative = chain(-0.0, torch.float32)
    assert positive != negative
    assert _chain_key(positive) != _chain_key(negative)


@pytest.mark.parametrize(
    "invalid",
    [
        "shape",
        "result_shape",
        "result_dtype",
        "missing_metadata",
        "dynamic",
        "bool_literal",
        "bool_dtype",
        "integer_dtype",
        "float64_dtype",
        "nan",
        "inf",
        "negative_inf",
        "overflow_fp16",
        "overflow_fp32",
        "huge_integer",
        "kwargs",
        "device",
        "wrong_target",
        "missing_dtype",
        "extra_argument",
    ],
)
def test_nonliteral_or_unsupported_full_stays_rejected(invalid: str) -> None:
    graph = torch.fx.Graph()
    node = _full(graph)
    args = list(node.args)
    if invalid == "shape":
        args[0] = [1]
    elif invalid == "result_shape":
        node.meta["val"] = torch.empty((1,), device="cpu")
    elif invalid == "result_dtype":
        node.meta["val"] = torch.empty((), dtype=torch.float16, device="cpu")
    elif invalid == "missing_metadata":
        node.meta.clear()
    elif invalid == "dynamic":
        args[1] = graph.placeholder("dynamic")
    elif invalid == "bool_literal":
        args[1] = True
    elif invalid in ("bool_dtype", "integer_dtype", "float64_dtype"):
        dtype = {
            "bool_dtype": torch.bool,
            "integer_dtype": torch.int32,
            "float64_dtype": torch.float64,
        }[invalid]
        args[2] = dtype
        node.meta["val"] = torch.empty((), dtype=dtype, device="cpu")
    elif invalid in ("nan", "inf", "negative_inf"):
        args[1] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative_inf": float("-inf"),
        }[invalid]
    elif invalid == "overflow_fp16":
        args[1:3] = [1e5, torch.float16]
        node.meta["val"] = torch.empty((), dtype=torch.float16, device="cpu")
    elif invalid == "overflow_fp32":
        args[1] = 1e50
    elif invalid == "huge_integer":
        args[1] = 10**1000
    elif invalid == "kwargs":
        node.kwargs = {"unexpected": 1}
    elif invalid == "device":
        args[3] = torch.device("cpu")
    elif invalid == "wrong_target":
        node.target = torch.full
    elif invalid == "missing_dtype":
        args = args[:2]
    elif invalid == "extra_argument":
        args.append(None)
    node.args = tuple(args)
    assert cute_epilogue._scalar_constant(node) is None
    assert not cute_epilogue._is_auxiliary_tensor_expr_node(node)
    assert (
        cute_epilogue._classify_auxiliary_tensor_expr(
            node,
            carrier_tile_shape=(4, 16),
            carrier_tile_index_nodes=None,
            carrier_global_shape=(4, 16),
        )
        is None
    )


def test_same_anchor_binary_dag_remains_outside_the_chain_contract() -> None:
    graph = torch.fx.Graph()
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32, device="cpu")
    result = _call(graph, torch.ops.aten.mul.Tensor, (anchor, anchor))
    assert (
        cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
            None, result, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
        )
        is None
    )


@helion.kernel(backend="cute", static_shapes=True)
def _typed_constant_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = y.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, col])
        scale = hl.full([], 0.3, dtype=torch.bfloat16).to(torch.float32)
        out[row, col] = acc * scale
    return out


def test_cpu_trace_normalized_full_and_store_chain() -> None:
    from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains

    # This is CPU-only tracing, even in a worker whose CUDA context was
    # initialized by an earlier GPU test. Hardware metadata is mocked, not
    # queried; the native/stream/allocation guards remain in force.
    with forbid_cuda_work(), mock_cuda_unavailable():
        x = torch.empty((128, 128), dtype=torch.float32, device="cpu")
        bound = cpu_bind(_typed_constant_matmul, (x, x.clone()))
        assert bound.host_function is not None
        graphs = bound.host_function.device_ir.graphs
        nodes = [node for graph in graphs for node in graph.graph.nodes]
        constants = [node for node in nodes if cute_epilogue._scalar_constant(node)]
        assert len(constants) == 1
        assert constants[0].args == ([], 0.3, torch.bfloat16, None)
        matmul = next(
            node for node in nodes if node.target is torch.ops.aten.addmm.default
        )
        stores = analyze_tcgen05_matmul_store_chains(graphs, matmul)
        assert stores is not None and len(stores) == 1
        source, expression = _render(stores[0][1])
        assert "cutlass.BFloat16(0.3)" in source
        assert ".to(cutlass.Float32)" in source
