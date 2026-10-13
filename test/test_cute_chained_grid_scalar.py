from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import sympy
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute.chained_grid_scalar import grid_scalar_bounds
from helion._compiler.cute.chained_grid_scalar import grid_scalar_expression
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.variable_origin import BlockSizeOrigin
from helion._compiler.variable_origin import TileEndOrigin
import helion.language as hl
from helion.language import _tracing_ops


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _captured_scalar_loop(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    division: hl.constexpr,
    divisor: int,
    overflow: hl.constexpr,
) -> torch.Tensor:
    divisor = hl.specialize(divisor)
    for bt in hl.tile(initial.size(0), block_size=1):
        if overflow:
            selected = (bt.id * 1073741824 - 1073741824) % 3 + 4
        elif division:
            selected = (bt.id - 3) // divisor + 4
        else:
            selected = (bt.id - 3) % divisor + 4
        state = initial[bt.id, :, :]
        for step in hl.grid(2):
            state = hl.dot(a[selected, :, :], b[step, :, :], acc=state * 0.5)
            output[bt.id, step, :, :] = state
    return output


def _args(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((8, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((4, 64, 64), dtype=torch.float32, device=device),
        torch.full((4, 3, 64, 64), 19.0, device=device),
    )


def _config(schedule: str) -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule=schedule)


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize(
    "division,divisor", [(True, 2), (True, -2), (False, 3), (False, -3)]
)
def test_captured_composite_grid_scalars_codegen(
    schedule: str, division: bool, divisor: int
) -> None:
    with _cpu_codegen():
        bound = _captured_scalar_loop._bind_isolated(
            (*_args(), division, divisor, False)
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            composites = [
                node
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if grid_scalar_expression(node) is not None
            ]
            assert composites
        source = bound.to_code(_config(schedule))
    assert "chain_loop_index" in source and "cute.gemm(" in source


@pytest.mark.parametrize("replacement", ["unknown", "extent", "end"])
def test_composite_proof_rejects_unknown_or_nonscalar_origins(replacement: str) -> None:
    with _cpu_codegen():
        bound = _captured_scalar_loop._bind_isolated((*_args(), True, 2, False))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            node = next(
                node
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if grid_scalar_expression(node) is not None
            )
            expression = grid_scalar_expression(node)
            assert expression is not None
            symbol = next(iter(expression.free_symbols))
            origins = bound.host_function.expr_to_origin
            mapping = dict(origins)
            if replacement == "unknown":
                mapping.pop(symbol)
            else:
                origin = (
                    BlockSizeOrigin(0) if replacement == "extent" else TileEndOrigin(0)
                )
                mapping[symbol] = SimpleNamespace(origin=origin)
            with patch.dict(origins, mapping, clear=True):
                assert grid_scalar_expression(node) is None
            # Real tile extents retain the existing vector-index classification.
            extents = [
                item
                for graph in bound.host_function.device_ir.graphs
                for item in graph.graph.nodes
                if item.target is _tracing_ops._get_symnode
                and bound.env.resolve_block_id(item.meta.get("val")) is not None
                and _Expression._grid_origin(item) is None
            ]
            assert all(grid_scalar_expression(item) is None for item in extents)


def test_emitted_composite_signed_arithmetic_matches_python() -> None:
    # Execute the generated scalar statements with C-style signed remainder,
    # independently of Python's already-correct // and % operators.
    class CArithmetic(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.op, (ast.Mod, ast.FloorDiv)):
                name = "rem" if isinstance(node.op, ast.Mod) else "truncdiv"
                return ast.Call(ast.Name(name, ast.Load()), [node.left, node.right], [])
            return node

    with _cpu_codegen():
        bound = _captured_scalar_loop._bind_isolated((*_args(), True, 2, False))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            composites = [
                expression
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if (expression := grid_scalar_expression(node)) is not None
            ]
            symbol = cast("sympy.Symbol", next(iter(composites[0].free_symbols)))
            from torch.utils._sympy.functions import FloorDiv
            from torch.utils._sympy.functions import PythonMod

            for divisor in (-7, -2, 2, 7):
                for operation in (FloorDiv, PythonMod):
                    counter = itertools.count()
                    cg = SimpleNamespace(
                        device_function=SimpleNamespace(
                            new_var=lambda prefix, counter=counter: (
                                f"{prefix}_{next(counter)}"
                            ),
                            resolved_block_size=lambda axis: 1,
                        )
                    )
                    plan = SimpleNamespace(
                        axes=((0, 41, 1),), loop=None, prepared_widenings=None
                    )
                    expression = _Expression(cast("Any", cg), cast("Any", plan), {})
                    expression.origins[0] = "index"
                    value = expression._grid_scalar(
                        cast("Any", operation(sympy.Add(symbol, -3), divisor))
                    )
                    program = ast.fix_missing_locations(
                        CArithmetic().visit(
                            ast.parse(
                                "\n".join([*expression.lines, f"result = {value}"])
                            )
                        )
                    )
                    code = compile(program, "<grid scalar>", "exec")
                    for index in range(-20, 21):

                        def truncdiv(a: int, b: int) -> int:
                            q = abs(a) // abs(b)
                            return -q if (a < 0) != (b < 0) else q

                        namespace: dict[str, Any] = {
                            "index": index,
                            "truncdiv": truncdiv,
                            "rem": lambda a, b: a - truncdiv(a, b) * b,
                            "cutlass": SimpleNamespace(Int32=int, Int64=int),
                        }
                        exec(code, namespace)
                        expected = (
                            (index - 3) // divisor
                            if operation is FloorDiv
                            else (index - 3) % divisor
                        )
                        assert namespace["result"] == expected


@pytest.mark.parametrize("bits", [32, 64])
def test_composite_intermediate_overflow_rejects_despite_safe_final_range(
    bits: int,
) -> None:
    symbol = sympy.Symbol("index", integer=True)
    scale = 2 ** (bits - 2)
    # At index2 the product overflows before subtraction returns to range.
    product = sympy.Mul(symbol, scale, evaluate=False)
    expression = sympy.Add(product, -scale, evaluate=False)
    limits = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    assert all(limits[0] <= value * scale - scale <= limits[1] for value in range(3))
    assert grid_scalar_bounds(expression, {symbol: (0, 2)}, limits) is None
    assert grid_scalar_bounds(expression, {symbol: (0, 1)}, limits) is not None
    assert grid_scalar_bounds(expression, {}, limits) is None


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_actual_composite_emission_uses_requested_integer_width(
    index_dtype: torch.dtype,
) -> None:
    with _cpu_codegen():
        kernel = helion.kernel(
            _captured_scalar_loop.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            index_dtype=index_dtype,
        )
        bound = kernel._bind_isolated((*_args(), False, 3, True))
        assert bound.config_spec.cute_chained_matmul_search_enabled
        if index_dtype == torch.int32:
            with pytest.raises(exc.BackendUnsupported):
                bound.to_code(_config("coalesced"))
        else:
            source = bound.to_code(_config("coalesced"))
            assert "cutlass.Int64(chain_origin_0)" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize(
    "division,divisor", [(True, 2), (True, -2), (False, 3), (False, -3)]
)
def test_captured_grid_arithmetic_and_loop_exports_gpu(
    schedule: str, division: bool, divisor: int
) -> None:
    if schedule == "tcgen05_tmem" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    a, b, initial, output = _args("cuda")
    torch.manual_seed(1127)
    for tensor in (a, b, initial):
        tensor.copy_(torch.randint(-4, 5, tensor.shape, device="cuda") * 0.0625)
    expected = output.clone()
    for index in range(4):
        selected = (index - 3) // divisor + 4 if division else (index - 3) % divisor + 4
        state = initial[index].double()
        for step in range(2):
            state = a[selected].double() @ b[step].double() + state * 0.5
            expected[index, step] = state.float()
    args = (a, b, initial, output, division, divisor, False)
    actual = _captured_scalar_loop._bind_isolated(args).compile_config(
        _config(schedule)
    )(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _clamped_chunk_endpoints(
    a: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    for batch in hl.tile(initial.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for chunk in hl.tile(gate.size(1), block_size=64):
            last = min(chunk.begin + 64, gate.size(1)) - 1
            previous = max(chunk.begin - 1, 0)
            factor = gate[batch.id, last] + gate[batch.id, previous]
            state = hl.dot(a[batch.id, chunk.id, :, :], b[:, :], acc=state * factor)
            output[batch.id, chunk.id, :, :] = state
    return output


def _clamped_args(length: int, device: str = "cpu") -> tuple[torch.Tensor, ...]:
    chunks = (length + 63) // 64
    return (
        torch.empty((2, chunks, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, length), device=device),
        torch.empty((2, 64, 64), device=device),
        torch.full((2, chunks + 1, 64, 64), 19.0, device=device),
    )


@pytest.mark.parametrize("length", (73, 128))
@pytest.mark.parametrize("schedule", ("coalesced", "tcgen05_tmem"))
def test_captured_min_max_chunk_endpoints_codegen(length: int, schedule: str) -> None:
    with _cpu_codegen():
        bound = _clamped_chunk_endpoints._bind_isolated(_clamped_args(length))
        source = bound.to_code(_config(schedule))
    assert "min(" in source and "max(" in source
    assert "chain_loop_index" in source and "cute.gemm(" in source


@pytest.mark.parametrize("length", (73, 512))
def test_real_gdn_example_clamped_last_gate_codegen(length: int) -> None:
    from examples.gdn_fwd_h import helion_gdn_fwd_h

    args = (
        torch.empty((2, length, 4, 16), dtype=torch.bfloat16),
        torch.empty((2, length, 4, 16), dtype=torch.bfloat16),
        torch.empty((2, length, 4, 32), dtype=torch.bfloat16),
        torch.empty((2, length, 4)),
        64,
    )
    with _cpu_codegen():
        kernel = helion.kernel(helion_gdn_fwd_h.fn, backend="cute", static_shapes=True)
        bound = kernel._bind_isolated(args)
        source = bound.to_code(bound.config_spec.default_config())
    assert f"min({length}," in source
    assert "chain_loop_index" in source and source.count("cute.gemm(") == 2


@pytest.mark.parametrize("bits", (32, 64))
def test_min_max_bounds_check_all_branches_before_clamping(bits: int) -> None:
    index = sympy.Symbol("index", integer=True)
    limits = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    shifted = sympy.Add(index, -3)
    for expression, expected in (
        (sympy.Min(shifted, 7), (-3, 7)),
        (sympy.Max(shifted, 0), (0, 7)),
        (sympy.Max(-2, sympy.Min(shifted, 4)), (-2, 4)),
    ):
        assert grid_scalar_bounds(expression, {index: (0, 10)}, limits) == expected
        assert grid_scalar_bounds(expression, {}, limits) is None
    overflowing = sympy.Mul(index, 2 ** (bits - 2), evaluate=False)
    for operation in (sympy.Min, sympy.Max):
        expression = operation(overflowing, 0, evaluate=False)
        assert grid_scalar_bounds(expression, {index: (0, 2)}, limits) is None
        assert grid_scalar_bounds(expression, {index: (0, 1)}, limits) is not None


def test_unsupported_grid_composite_never_lifts_inactive_legacy_loop() -> None:
    from helion._compiler.cute.chained_matmul import _UnsupportedChain

    with _cpu_codegen():
        bound = _captured_scalar_loop._bind_isolated((*_args(), True, 2, False))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            node = next(
                node
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if grid_scalar_expression(node) is not None
            )
            original = node.meta["val"]
            symbol = next(iter(original._sympy_().free_symbols))
            node.meta["val"] = bound.env.shape_env.create_symintnode(
                cast("sympy.Expr", sympy.Abs(sympy.Add(symbol, -3))), hint=None
            )
            assert grid_scalar_expression(node) is None
            cg = SimpleNamespace(
                device_function=SimpleNamespace(
                    literal_expr=lambda value: pytest.fail("legacy offset lookup"),
                )
            )
            plan = SimpleNamespace(
                axes=((0, 4, 1),), loop=None, prepared_widenings=None
            )
            expression = _Expression(cast("Any", cg), cast("Any", plan), {})
            with pytest.raises(
                _UnsupportedChain, match="unsupported composite grid scalar"
            ):
                expression.scalar(node)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("length", (73, 128))
@pytest.mark.parametrize("schedule", ("coalesced", "tcgen05_tmem"))
def test_clamped_grid_endpoints_and_tail_loop_exports_gpu(
    length: int, schedule: str
) -> None:
    if schedule == "tcgen05_tmem" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = _clamped_args(length, "cuda")
    a, b, gate, initial, output = args
    torch.manual_seed(2601)
    for tensor in (a, b, initial):
        tensor.copy_(torch.randint(-2, 3, tensor.shape, device="cuda") * 0.0625)
    gate.copy_(torch.randint(1, 4, gate.shape, device="cuda") * 0.125)
    expected = output.clone()
    for batch in range(2):
        state = initial[batch].double()
        for chunk in range((length + 63) // 64):
            last, previous = min((chunk + 1) * 64, length) - 1, max(chunk * 64 - 1, 0)
            factor = gate[batch, last].double() + gate[batch, previous].double()
            state = a[batch, chunk].double() @ b.double() + state * factor
            expected[batch, chunk] = state.float()
    actual = _clamped_chunk_endpoints._bind_isolated(args).compile_config(
        _config(schedule)
    )(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
