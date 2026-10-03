"""Native sigmoid uses pointwise math without moving dtype boundaries."""

from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from torch.fx.node import Argument


@pytest.fixture
def cpu_b200() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _matmul_sigmoid(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = y.size(1)
    gate_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    scaled_out = torch.empty_like(gate_out)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, column])
        gate = torch.sigmoid(acc).to(x.dtype)
        gate_out[row, column] = gate
        scaled_out[row, column] = gate.float() * 3.03125
    return gate_out, scaled_out


def _pointwise_sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for index in hl.tile(x.numel()):
        out[index] = torch.sigmoid(x[index])
    return out


def _native_config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 128, 32],
        num_warps=4,
        pid_type="flat",
        tcgen05_strategy="role_local_monolithic",
        tcgen05_persistence_model="non_persistent",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        cute_collective_mma=False,
    )


def _sigmoid_signatures(source: str) -> list[tuple[object, ...]]:
    signatures: list[tuple[object, ...]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or ast.unparse(node.func) != "cute.math.rcp":
            continue
        assert len(node.args) == 1
        denominator = node.args[0]
        assert isinstance(denominator, ast.BinOp)
        assert isinstance(denominator.op, ast.Add)
        assert ast.literal_eval(denominator.left) == 1.0
        exponent = denominator.right
        assert isinstance(exponent, ast.Call)
        assert ast.unparse(exponent.func) == "cute.math.exp2"
        product = exponent.args[0]
        assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
        assert isinstance(product.right, ast.UnaryOp)
        assert isinstance(product.right.op, ast.USub)
        signatures.append(
            (
                ast.literal_eval(product.right),
                tuple(
                    (item.arg, ast.literal_eval(item.value)) for item in node.keywords
                ),
                tuple(
                    (item.arg, ast.literal_eval(item.value))
                    for item in exponent.keywords
                ),
            )
        )
    return signatures


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("pointwise_static_shapes", (False, True))
@pytest.mark.parametrize("fast_math", (False, True))
@skipUnlessBackends(["cute"])
def test_native_and_pointwise_sigmoid_share_the_same_math(
    cpu_b200: None, dtype: torch.dtype, pointwise_static_shapes: bool, fast_math: bool
) -> None:
    # These tails also select masked stores in the native epilogue. Keep the
    # arguments alive so pointer-alignment facts are available to codegen.
    args = (
        torch.empty((65, 64), dtype=dtype),
        torch.empty((64, 72), dtype=dtype)[:, :70],
    )
    native = helion.kernel(
        _matmul_sigmoid,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
        fast_math=fast_math,
    )._bind_isolated(args)
    native.env.config_spec.cute_tcgen05_search_enabled = True
    native_source = native.to_code(_native_config())
    pointwise_args = (torch.empty(129, dtype=dtype),)
    pointwise = helion.kernel(
        _pointwise_sigmoid,
        backend="cute",
        autotune_effort="none",
        static_shapes=pointwise_static_shapes,
        fast_math=fast_math,
    )._bind_isolated(pointwise_args)
    pointwise_source = pointwise.to_code(helion.Config(block_sizes=[128]))
    expected = (
        -1.4426950408889634,
        (("approx", True), ("ftz", True)),
        (("fastmath", True),),
    )
    native_signatures = _sigmoid_signatures(native_source)
    pointwise_signatures = _sigmoid_signatures(pointwise_source)
    assert native_signatures and pointwise_signatures
    assert all(item == expected for item in native_signatures + pointwise_signatures)
    assert "tcgen05_acc_loaded" in native_source
    assert "tcgen05_chain_step" in native_source
    assert "1.0 /" not in native_source
    cast = "cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16"
    assert f".to({cast}).to(cutlass.Float32)" in native_source


def _call(
    graph: torch.fx.Graph,
    target: Callable[..., object],
    args: tuple[Argument, ...],
    dtype: torch.dtype,
) -> torch.fx.Node:
    node = graph.call_function(target, args=args)
    node.meta["val"] = torch.empty((4, 16), dtype=dtype)
    return node


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("round_input", (False, True))
def test_sigmoid_keeps_rounding_before_the_next_operation(
    dtype: torch.dtype, round_input: bool
) -> None:
    graph = torch.fx.Graph()
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32)
    carrier = anchor
    if round_input:
        carrier = _call(
            graph, torch.ops.prims.convert_element_type.default, (anchor, dtype), dtype
        )
    sigmoid_dtype = dtype if round_input else torch.float32
    sigmoid = _call(graph, torch.ops.aten.sigmoid.default, (carrier,), sigmoid_dtype)
    rounded = _call(
        graph, torch.ops.prims.convert_element_type.default, (sigmoid, dtype), dtype
    )
    scaled = _call(graph, torch.ops.aten.mul.Tensor, (rounded, 1.0625), dtype)
    result = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (scaled, torch.float32),
        torch.float32,
    )
    analysis = cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
        None, result, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
    )
    assert analysis is not None and analysis[1] is anchor
    names = itertools.count()
    source, expression = analysis[0].render_prelude_and_expr(
        "values", lambda prefix: f"{prefix}_{next(names)}", ""
    )

    # CPU evaluation checks the placement of dtype conversions, independently
    # of the GPU intrinsics' approximation error.
    def exp2(value: torch.Tensor, *, fastmath: bool) -> torch.Tensor:
        assert fastmath
        return torch.exp2(value)

    def reciprocal(value: torch.Tensor, *, approx: bool, ftz: bool) -> torch.Tensor:
        assert approx and ftz
        return value.reciprocal()

    values = torch.linspace(-4.125, 4.125, 64).reshape(4, 16)
    namespace = {
        "values": values,
        "cute": SimpleNamespace(math=SimpleNamespace(exp2=exp2, rcp=reciprocal)),
        "cutlass": SimpleNamespace(
            Float16=torch.float16, BFloat16=torch.bfloat16, Float32=torch.float32
        ),
    }
    exec(source + f"result = {expression}\n", namespace)
    input_values = values.to(dtype).float() if round_input else values
    unrounded = (1.0 + torch.exp2(-input_values * 1.4426950408889634)).reciprocal()
    expected = (unrounded.to(dtype) * 1.0625).float()
    torch.testing.assert_close(namespace["result"], expected, rtol=0, atol=0)
    assert not torch.equal(expected, unrounded * 1.0625)


def test_explicit_division_keeps_its_arithmetic_form() -> None:
    graph = torch.fx.Graph()
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32)
    inverse = _call(graph, torch.ops.aten.div.Tensor, (1.0, anchor), torch.float32)
    analysis = cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
        None, inverse, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
    )
    assert analysis is not None
    source, expression = analysis[0].render_prelude_and_expr(
        "values", lambda prefix: prefix, ""
    )
    assert "cute.math.rcp" not in source + expression
    assert any(isinstance(node, ast.Div) for node in ast.walk(ast.parse(source)))
    assert "cute.math.rcp" not in cute_epilogue._SILU_TEMPLATE
