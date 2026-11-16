"""Native sigmoid and silu use pointwise math without moving dtype boundaries."""

from __future__ import annotations

import ast
import itertools
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute import target_facts
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from torch.fx.node import Argument


@pytest.fixture
def cpu_b200() -> Iterator[None]:
    with (
        cpu_target(),
        patch.object(target_facts, "per_cta_smem_capacity_bytes", return_value=232448),
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


def _matmul_scaled(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, column])
        out[row, column] = acc.to(x.dtype) * scale[()]
    return out


def _matmul_integer_scaled(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = y.size(1)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, column])
        out[row, column] = acc.to(x.dtype) * scale[row, column]
    return out


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
    native.env.config_spec.tcgen05.search_enabled = True
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


def test_a_rank0_scale_takes_the_carriers_dtype_first(cpu_b200: None) -> None:
    # ``acc.to(bf16) * scale[()]`` is a bf16 product: eager and the pointwise
    # lowering round the 0-d fp32 scale to bf16 before it, as an explicit
    # ``scale[()].to(bf16)`` would.
    args = (
        torch.empty((256, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        torch.empty((), dtype=torch.float32),
    )
    native = helion.kernel(
        _matmul_scaled, backend="cute", autotune_effort="none", static_shapes=True
    )._bind_isolated(args)
    native.env.config_spec.tcgen05.search_enabled = True
    source = native.to_code(_native_config())
    assert "tcgen05_chain_step" in source
    rounding = " = tcgen05_scalar.to(cutlass.BFloat16).to(cutlass.Float32)"
    rounded = [
        line.split(" = ")[0].strip()
        for line in source.splitlines()
        if line.endswith(rounding)
    ]
    assert len(rounded) == 1
    assert f"* {rounded[0]}).to(cutlass.BFloat16)" in source
    assert "* tcgen05_scalar)" not in source


def test_an_integer_aux_takes_the_carriers_dtype_first(cpu_b200: None) -> None:
    # ``acc.to(bf16) * scale[row, column]`` with an int32 aux tensor is a
    # bf16 product: eager rounds the integers to bf16 before it.
    args = (
        torch.empty((256, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        torch.empty((256, 128), dtype=torch.int32),
    )
    native = helion.kernel(
        _matmul_integer_scaled,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
    )._bind_isolated(args)
    native.env.config_spec.tcgen05.search_enabled = True
    source = native.to_code(_native_config())
    suffix = ".to(cutlass.BFloat16).to(cutlass.Float32)"
    rounded = [
        line.strip().split(" = ")
        for line in source.splitlines()
        if " = tcgen05_aux_loaded_" in line and line.endswith(suffix)
    ]
    assert len(rounded) == 1
    name, value = rounded[0]
    assert f"* {name}).to(cutlass.BFloat16)" in source
    assert f"* {value.removesuffix(suffix)})" not in source


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
    # So does silu's division form without fast_math (the pointwise quotient).
    assert "cute.math.rcp" not in cute_epilogue._SILU_DIVISION_TEMPLATE


def _matmul_silu(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = y.size(1)
    division = torch.empty((m, n), dtype=torch.float32, device=x.device)
    product = torch.empty_like(division)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[row, inner], y[inner, column])
        # Inductor's decomposition ``x / (1 + exp(-x))`` and the core one.
        division[row, column] = torch.nn.functional.silu(acc)
        product[row, column] = acc * torch.sigmoid(acc)
    return division, product


def _pointwise_silu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    division = torch.empty_like(x)
    product = torch.empty_like(x)
    for index in hl.tile(x.numel()):
        value = x[index]
        division[index] = torch.nn.functional.silu(value)
        product[index] = value * torch.sigmoid(value)
    return division, product


def _flush(value: np.ndarray) -> np.ndarray:
    """Flush float32 denormals to signed zero, as the ``ftz`` forms do."""
    tiny = np.finfo(np.float32).tiny
    return np.where(np.abs(value) < tiny, np.copysign(0, value), value).astype(
        np.float32
    )


def _approximate(value: np.ndarray) -> np.ndarray:
    """``value`` (float64) truncated to float32 and flushed: the stand-in for
    an approximate ``ftz`` intrinsic, which a correctly rounded form cannot
    stand in for."""
    rounded = value.astype(np.float32)
    away = np.abs(rounded.astype(np.float64)) > np.abs(value)
    return _flush(np.where(away, np.nextafter(rounded, np.float32(0)), rounded))


def _cpu_math() -> dict[str, object]:
    """CPU stand-ins for the DSL math a silu renders: correctly rounded exp,
    exp2 and IEEE arithmetic; truncated, flushed ``ex2.approx.ftz`` and
    ``rcp.approx.ftz``; ``div.approx.ftz`` a flushed product with the
    latter."""

    def exp(value: np.ndarray, fastmath: bool = False) -> np.ndarray:
        assert not fastmath
        return np.exp(value.astype(np.float64)).astype(np.float32)

    def exp2(value: np.ndarray, fastmath: bool = False) -> np.ndarray:
        result = np.exp2(value.astype(np.float64))
        return _approximate(result) if fastmath else result.astype(np.float32)

    def rcp(value: np.ndarray, approx: bool = False, ftz: bool = False) -> np.ndarray:
        assert approx and ftz
        return _approximate(1.0 / _flush(value).astype(np.float64))

    def div(
        left: np.ndarray, right: np.ndarray, approx: bool = False, ftz: bool = False
    ) -> np.ndarray:
        assert approx and ftz
        return _flush(_flush(left) * rcp(right, approx=True, ftz=True))

    return {
        "cute": SimpleNamespace(
            math=SimpleNamespace(exp=exp, exp2=exp2, rcp=rcp, div=div)
        ),
        "cutlass": SimpleNamespace(Float32=lambda value: np.float32(value)),
    }


def _named_assigns(tree: ast.AST, pattern: str) -> list[ast.Assign]:
    return sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and re.fullmatch(pattern, node.targets[0].id)
        ),
        key=lambda node: node.lineno,
    )


def _evaluate(expression: ast.expr, namespace: dict[str, object]) -> np.ndarray:
    return eval(compile(ast.Expression(expression), "<rendered>", "eval"), namespace)


@pytest.mark.parametrize("fast_math", (False, True))
def test_native_and_pointwise_silu_compute_the_same_math(
    cpu_b200: None, fast_math: bool
) -> None:
    """Both silu decompositions fuse into one chain step each (no ``aten``
    op survives the splice), and every step computes what the pointwise
    lowering computes for its form: ``x * sigmoid(x)`` the sigmoid contract,
    ``x / (1 + exp(-x))`` the IEEE quotient of the exact ``cute.math.exp``,
    under ``fast_math`` the approximate quotient of ``ex2.approx``.  Both
    sources run on CPU stand-ins for the DSL math, so a form that rounds
    differently (a reciprocal then a product, ``exp`` for ``exp2(x * log2
    e)``, a denormal sigmoid kept at -88) fails here; the GPU check is
    ``test_tcgen05_fused_silu_matches_the_pointwise_lowering``."""
    x = torch.empty((128, 64), dtype=torch.bfloat16)
    y = torch.empty((64, 128), dtype=torch.bfloat16)
    native = helion.kernel(
        _matmul_silu,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
        fast_math=fast_math,
    )._bind_isolated((x, y))
    native.env.config_spec.tcgen05.search_enabled = True
    native_source = native.to_code(_native_config())
    pointwise = helion.kernel(
        _pointwise_silu,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
        fast_math=fast_math,
    )._bind_isolated((torch.empty(129, dtype=torch.float32),))
    pointwise_source = pointwise.to_code(helion.Config(block_sizes=[128]))

    steps = _named_assigns(ast.parse(native_source), r"tcgen05_chain_step_\d+")
    assert len(steps) == 2
    kernel = next(
        node
        for node in ast.parse(pointwise_source).body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(item) == "cute.kernel" for item in node.decorator_list)
    )
    stores = sorted(
        (
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
        ),
        key=lambda node: node.lineno,
    )
    assert len(stores) == 2
    # No denormal inputs: ``div.approx.ftz`` also flushes a denormal product.
    values = np.concatenate(
        [
            np.array([-88.0, -87.5, -90.0, -104.0, -0.0, 0.0, 1.0, 88.0, 100.0]),
            np.linspace(-110.0, 40.0, 4093),
        ]
    ).astype(np.float32)
    with np.errstate(over="ignore"):
        fused = []
        for step in steps:
            (carrier,) = {
                node.id
                for node in ast.walk(step.value)
                if isinstance(node, ast.Name)
                and node.id.startswith("tcgen05_acc_loaded")
            }
            fused.append(_evaluate(step.value, {**_cpu_math(), carrier: values}))
        namespace = {**_cpu_math(), "value": values}
        for statement in _named_assigns(kernel, r"v_\d+"):
            exec(
                compile(ast.Module([statement], type_ignores=[]), "<rendered>", "exec"),
                namespace,
            )
        unfused = [_evaluate(store.args[0], namespace) for store in stores]
    for fused_values, unfused_values in zip(fused, unfused, strict=True):
        np.testing.assert_array_equal(
            fused_values.view(np.int32), unfused_values.view(np.int32)
        )
