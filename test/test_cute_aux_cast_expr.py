from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_batched_aux_tma import _config
from test.test_cute_batched_aux_tma import _cpu_codegen

import helion
from helion import exc
from helion._compiler.cute import cute_epilogue as epilogue
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import memory_ops

pytestmark = skipUnlessBackends(["cute"])


class _AuxGraph:
    def __init__(self) -> None:
        self.graph = Graph()
        self.indices = (
            self.graph.placeholder("mi"),
            self.graph.placeholder("ni"),
        )

    def load(self, dtype: torch.dtype, *, broadcast: bool = False) -> Node:
        tensor = self.graph.placeholder(f"aux_{len(self.graph.nodes)}")
        tensor.meta["val"] = torch.empty((8,) if broadcast else (8, 8), dtype=dtype)
        indices = [self.indices[1]] if broadcast else list(self.indices)
        node = self.graph.call_function(memory_ops.load, args=(tensor, indices))
        node.meta["val"] = torch.empty((4,) if broadcast else (4, 4), dtype=dtype)
        return node

    def cast(self, node: Node, dtype: torch.dtype) -> Node:
        cast = self.graph.call_function(
            torch.ops.prims.convert_element_type.default, args=(node, dtype)
        )
        cast.meta["val"] = torch.empty(node.meta["val"].shape, dtype=dtype)
        return cast

    def product(self, lhs: Node, rhs: Node) -> Node:
        node = self.graph.call_function(torch.ops.aten.mul.Tensor, args=(lhs, rhs))
        node.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
        return node

    def classify(self, node: Node) -> epilogue._TensorExpr | None:
        return epilogue._classify_auxiliary_tensor_expr(
            node,
            carrier_tile_shape=(4, 4),
            carrier_tile_index_nodes=self.indices,
            carrier_global_shape=(8, 8),
        )


def _names():
    counter = itertools.count()
    return lambda prefix: f"{prefix}_{next(counter)}"


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("reverse", (False, True))
def test_aux_cast_roundtrip_ir_render_and_hoist(
    dtype: torch.dtype, reverse: bool
) -> None:
    graph = _AuxGraph()
    product = graph.product(graph.load(torch.float32), graph.load(torch.float32))
    cast = graph.cast(graph.cast(product, dtype), torch.float32)
    expr = graph.classify(cast)
    assert isinstance(expr, epilogue._CastTensorExpr)
    assert expr.source_dtype == dtype and expr.target_dtype == torch.float32
    assert isinstance(expr.operand, epilogue._CastTensorExpr)
    assert expr.operand.source_dtype == torch.float32
    assert expr.operand.target_dtype == dtype
    assert epilogue._is_auxiliary_tensor_expr_node(cast)
    assert not epilogue._tensor_expr_contains_current(expr)
    current = epilogue._CurrentTensorExpr()
    step = epilogue._AuxiliaryTensorExprStep(
        epilogue._BinaryTensorExpr(
            "sub",
            "{lhs} - {rhs}",
            expr if reverse else current,
            current if reverse else expr,
        )
    )
    assert epilogue._tensor_expr_contains_current(step.expr)
    assert step.hoistable_aux_expr is expr
    assert len(step.operands) == 2
    bindings = dict(zip(reversed(step.operands), ("b", "a"), strict=True))
    prelude, result = step.render_prelude_and_expr("acc", bindings, _names(), "")
    hoisted, aux = step.render_hoistable_aux_prelude_and_expr(bindings, _names(), "")
    narrow = "BFloat16" if dtype == torch.bfloat16 else "Float16"
    assert hoisted == (
        "tcgen05_aux_product_0 = a * b\n"
        f"tcgen05_aux_expr_1 = (tcgen05_aux_product_0).to(cutlass.{narrow})\n"
        "tcgen05_aux_expr_2 = (tcgen05_aux_expr_1).to(cutlass.Float32)\n"
    )
    assert prelude == hoisted + (
        f"tcgen05_chain_step_3 = {step.render_with_hoisted_aux('acc', aux)}\n"
    )
    assert result == "tcgen05_chain_step_3"
    # Execute only the rendered ordinary Python tensor expressions on CPU.
    # The dtype names stand in for CuTe types; no CuTe import or compilation.
    values = torch.tensor(
        [
            0.0,
            -0.0,
            1.0001,
            -1.0001,
            1.5,
            65520.0,
            2.0**-24,
            2.0**-133,
            float("inf"),
            -float("inf"),
            float("nan"),
            3.0,
            -3.0,
            0.5,
            -0.5,
            2.0,
        ]
    ).reshape(4, 4)
    namespace = {
        "a": values,
        "b": torch.ones_like(values),
        "acc": torch.ones_like(values),
        "cutlass": SimpleNamespace(
            Float32=torch.float32, Float16=torch.float16, BFloat16=torch.bfloat16
        ),
    }
    exec(prelude, namespace)
    rounded = values.to(dtype).to(torch.float32)
    expected = rounded - 1 if reverse else 1 - rounded
    actual = namespace[result]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual[finite].view(torch.int32), expected[finite].view(torch.int32)
    )
    assert rounded[0, 2] != values[0, 2]  # Both boundaries are observable.


@pytest.mark.parametrize("source", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("target", (torch.float16, torch.bfloat16, torch.float32))
def test_aux_cast_supported_dtypes_and_exact_metadata(
    source: torch.dtype, target: torch.dtype
) -> None:
    graph = _AuxGraph()
    load = graph.load(source)
    node = graph.cast(load, target)
    assert epilogue._auxiliary_cast_operand(node) == (load, source, target)
    assert graph.classify(node) is not None
    assert epilogue._is_auxiliary_tensor_expr_node(node)


@pytest.mark.parametrize(
    "kind",
    [
        "source_int",
        "source_double",
        "target_int",
        "target_bool",
        "target_double",
        "target_float8",
        "missing_source",
        "missing_result",
        "wrong_dtype",
        "wrong_shape",
        "kwargs",
        "extra_arg",
        "literal_operand",
        "literal_dtype",
        "to_copy",
    ],
)
def test_aux_cast_unsupported_forms_fail_closed(kind: str) -> None:
    graph = _AuxGraph()
    load = graph.load(torch.float32)
    node = graph.cast(load, torch.bfloat16)
    if kind.startswith("source_"):
        load.meta["val"] = torch.empty(
            (4, 4), dtype=(torch.int32 if kind == "source_int" else torch.float64)
        )
        node = graph.cast(load, torch.float32)
    elif kind.startswith("target_"):
        dtype = {
            "target_int": torch.int32,
            "target_bool": torch.bool,
            "target_double": torch.float64,
            "target_float8": torch.float8_e4m3fn,
        }[kind]
        node = graph.cast(load, dtype)
    elif kind == "missing_source":
        load.meta.clear()
    elif kind == "missing_result":
        node.meta.clear()
    elif kind == "wrong_dtype":
        node.meta["val"] = torch.empty((4, 4), dtype=torch.float16)
    elif kind == "wrong_shape":
        node.meta["val"] = torch.empty((4, 1), dtype=torch.bfloat16)
    elif kind == "kwargs":
        node.kwargs = {"copy": True}
    elif kind == "extra_arg":
        node.args = (*node.args, True)
    elif kind == "literal_operand":
        node.args = (1.0, torch.bfloat16)
    elif kind == "literal_dtype":
        node.args = (load, "bfloat16")
    else:
        node.target = torch.ops.aten._to_copy.default
    assert epilogue._auxiliary_cast_operand(node) is None
    assert graph.classify(node) is None
    assert not epilogue._is_auxiliary_tensor_expr_node(node)


def test_aux_cast_preserves_broadcast_leaf_and_depth_guard() -> None:
    graph = _AuxGraph()
    cast = graph.cast(
        graph.cast(graph.load(torch.float32, broadcast=True), torch.bfloat16),
        torch.float32,
    )
    expr = graph.classify(cast)
    assert isinstance(expr, epilogue._CastTensorExpr)
    (leaf,) = epilogue._auxiliary_tensor_expr_operands(expr)
    assert leaf.broadcast_axis == 1
    for _ in range(32):
        cast = graph.cast(cast, torch.float32)
    assert graph.classify(cast) is None
    assert not epilogue._is_auxiliary_tensor_expr_node(cast)


def test_aux_cast_does_not_turn_carrier_or_unknown_node_into_auxiliary() -> None:
    graph = _AuxGraph()
    carrier = graph.graph.placeholder("acc")
    carrier.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    node = graph.cast(carrier, torch.bfloat16)
    assert graph.classify(node) is None
    assert not epilogue._is_auxiliary_tensor_expr_node(node)
    # Traversal is nevertheless exhaustive over the new typed variant.
    expr = epilogue._CastTensorExpr(
        epilogue._CurrentTensorExpr(), torch.float32, torch.bfloat16
    )
    assert epilogue._tensor_expr_contains_current(expr)
    assert epilogue._auxiliary_tensor_expr_operands(expr) == ()


@pytest.mark.parametrize("kind", ("raw", "fp32", "scaled_fp32"))
def test_aux_cast_legacy_leaf_source_and_hoisting_unchanged(kind: str) -> None:
    graph = _AuxGraph()
    load = graph.load(torch.bfloat16)
    node, template = load, "{aux}"
    if kind != "raw":
        node = graph.cast(load, torch.float32)
        template = "({aux}).to(cutlass.Float32)"
    if kind == "scaled_fp32":
        node = graph.graph.call_function(torch.ops.aten.mul.Tensor, args=(node, 0.5))
        node.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
        template = f"(({template}) * 0.5)"
    expr = graph.classify(node)
    assert isinstance(expr, epilogue._AuxiliaryTensorLoadExpr)
    assert expr.load_node is load and expr.template == template
    step = epilogue._AuxiliaryTensorExprStep(
        epilogue._BinaryTensorExpr(
            "add", "{lhs} + {rhs}", epilogue._CurrentTensorExpr(), expr
        )
    )
    assert step.hoistable_aux_expr is None
    prefix, result = step.render_prelude_and_expr("acc", {expr: "aux"}, _names(), "")
    if kind == "raw":
        assert prefix == "tcgen05_chain_step_0 = acc + aux\n"
        assert result == "tcgen05_chain_step_0"
    else:
        assert prefix == (
            f"tcgen05_aux_expr_0 = {template.format(aux='aux')}\n"
            "tcgen05_chain_step_1 = acc + tcgen05_aux_expr_0\n"
        )
        assert result == "tcgen05_chain_step_1"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _computed_cast_aux(
    lhs: torch.Tensor, rhs: torch.Tensor, first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    m, k = lhs.shape
    n = rhs.shape[1]
    out = torch.empty((m, n), dtype=lhs.dtype, device=lhs.device)
    for mi, ni in hl.tile([m, n]):
        acc = hl.zeros([mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.addmm(acc, lhs[mi, ki], rhs[ki, ni])
        aux = (
            (first[mi, ni].to(torch.float32) * second[mi, ni].to(torch.float32))
            .to(lhs.dtype)
            .to(torch.float32)
        )
        out[mi, ni] = (acc * aux).to(out.dtype)
    return out


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("simt", "tma"))
def test_computed_aux_cast_codegen_keeps_both_rounding_boundaries(
    dtype: torch.dtype, mode: str
) -> None:
    args = (
        torch.empty((256, 128), dtype=dtype),
        torch.empty((128, 512), dtype=dtype),
        torch.empty((256, 512), dtype=dtype),
        torch.empty((256, 512), dtype=dtype),
    )
    with _cpu_codegen():
        code = _computed_cast_aux._bind_isolated(args).to_code(
            _config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode=mode)
        )
    tree = ast.parse(code)
    cast_targets = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("tcgen05_aux_expr")
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "to"
        ):
            cast_targets.append(ast.unparse(node.value.args[0]))
    narrow = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    assert narrow in cast_targets and "cutlass.Float32" in cast_targets
    assert "tcgen05_aux_product" in code
    assert "cute.arch.fence_acq_rel_cta()" in code


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_aux_cast_does_not_widen_tma_transport_dtype_admission(
    dtype: torch.dtype,
) -> None:
    args = (
        torch.empty((256, 128), dtype=dtype),
        torch.empty((128, 512), dtype=dtype),
        torch.empty((256, 512), dtype=torch.float32),
        torch.empty((256, 512), dtype=torch.float32),
    )
    with (
        _cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="auxiliary tensor dtype to match"),
    ):
        _computed_cast_aux._bind_isolated(args).to_code(
            _config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode="tma")
        )
