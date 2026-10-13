from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_row_registers import _args
from .test_cute_chained_row_registers import _config
from .test_cute_chained_row_registers import _fixture
from .test_cute_chained_row_registers import _sequence
import helion
from helion._compiler.cute.chained_coordinates import CoordinateContext
from helion._compiler.cute.chained_coordinates import CoordinateDomain
from helion._compiler.cute.chained_coordinates import IndexExpr
from helion._compiler.cute.chained_coordinates import binary
from helion._compiler.cute.chained_coordinates import constant
from helion._compiler.cute.chained_coordinates import original_coordinate
from helion._compiler.cute.chained_coordinates import prove
from helion._compiler.cute.chained_coordinates import render
from helion._compiler.cute.chained_coordinates import row_scope
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_row_registers import plan_row_registers
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@pytest.mark.parametrize("rows,columns", [(8, 8), (13, 17), (16, 16), (32, 32)])
@pytest.mark.parametrize("selected", [None, 3])
def test_mixed_radix_reshape_compositions_at_every_owned_coordinate(
    rows: int, columns: int, selected: int | None
) -> None:
    domain = CoordinateDomain(
        "owned_row",
        "static_column",
        (0, rows - 1) if selected is None else (selected, selected),
        (0, columns - 1),
    )
    context = CoordinateContext(domain)
    # Original 2x16x2x16 -> 32x32 -> 64x64 typed composition with split-member
    # coordinates. Evaluate the exact renderings at all points, not a second
    # copy of the simplifier's affine inference.
    first = context.reshape(
        ("0", domain.row, "0", domain.column), (32, 32), (2, 16, 2, 16)
    )
    assert first is not None
    second = context.reshape(("0", first[0], "0", first[1]), (64, 64), (2, 32, 2, 32))
    assert second is not None
    points = domain.points()
    assert points is not None
    for row, col in points:
        ns = {
            domain.row: np.int32(row),
            domain.column: col,
            "cutlass": SimpleNamespace(Int32=np.int32),
        }
        actual = tuple(int(eval(text, ns)) for text in second)
        original_flat1 = row * 32 + col
        original_flat2 = ((original_flat1 // 32) % 32) * 64 + original_flat1 % 32
        expected = ((original_flat2 // 64) % 64, original_flat2 % 64)
        assert actual == expected
    assert all(len(text) < 100 for text in second)


def test_runtime_type_survives_selected_owner_and_constant_result() -> None:
    domain = CoordinateDomain("long_runtime_row", "column", (7, 7), (0, 15))
    expr = binary("mod", binary("mul", IndexExpr("row"), constant(16)), constant(16))
    proof = prove(expr, domain)
    assert proof is not None and proof.replacement.runtime
    assert render(proof.replacement, domain) == "cutlass.Int32(0)"
    ns = {"cutlass": SimpleNamespace(Int32=np.int32)}
    assert type(eval(render(proof.replacement, domain), ns)) is np.int32
    constexpr = prove(binary("mul", IndexExpr("column"), constant(0)), domain)
    assert constexpr is not None and not constexpr.replacement.runtime
    assert render(constexpr.replacement, domain) == "0"


@pytest.mark.parametrize(
    "kind",
    [
        "overflow",
        "cancelled_overflow",
        "negative",
        "zero_divisor",
        "negative_divisor",
        "unknown",
        "non_affine",
        "budget",
    ],
)
def test_unproved_intermediates_keep_original_spelling(kind: str) -> None:
    row = IndexExpr("row")
    expression = {
        "overflow": binary("mul", row, constant(2**30)),
        "cancelled_overflow": binary(
            "div", binary("mul", row, constant(2**30)), constant(2**30)
        ),
        "negative": binary("sub", row, constant(1)),
        "zero_divisor": binary("div", row, constant(0)),
        "negative_divisor": binary("mod", row, constant(-3)),
        "unknown": IndexExpr("loaded_index"),
        "non_affine": binary("mod", row, constant(3)),
        "budget": row,
    }[kind]
    if kind == "budget":
        for _ in range(130):
            expression = binary("add", expression, constant(0))
    domain = CoordinateDomain("row", "column", (0, 31), (0, 31))
    assert prove(expression, domain) is None
    assert (
        CoordinateContext(domain).normalize(expression, "original_expression")
        == "original_expression"
    )


def test_unbound_strings_casts_and_invalid_domains_are_never_parsed() -> None:
    context = CoordinateContext(CoordinateDomain("row", "column", (0, 15), (0, 15)))
    for text in ("unknown", "row + 1", "cutlass.Int64(row)", "data[row]", "-1"):
        assert context.lookup(text) is None
        assert context.coordinate(text) == text
    assert context.reshape(("row + 1", "column"), (16, 16), (16, 16)) is None
    assert (
        prove(IndexExpr("row"), CoordinateDomain("row", "column", (0, 32), (0, 15)))
        is None
    )


def test_shorter_coordinate_retains_original_domain_identity() -> None:
    context = CoordinateContext(CoordinateDomain("row", "column", (0, 15), (0, 15)))
    result = context.reshape(("row", "column"), (16, 16), (16, 16))
    assert result is not None and result == ("row", "cutlass.Int32(column)")
    assert original_coordinate(result[0]) != "row"
    assert original_coordinate(result[1]) != "column"
    assert original_coordinate(context.coordinate(result[0])) == original_coordinate(
        result[0]
    )


@pytest.mark.parametrize("warp", [None, 0, 3])
def test_original_lexical_branch_and_event_own_the_bound(warp: int | None) -> None:
    plan, _, _ = _fixture(16, 16)
    proposal = plan_row_registers(plan)
    plan = replace(plan, row_registers=proposal)
    island = proposal.islands[0]
    for event in island.events:
        scope = row_scope(
            plan,
            island,
            event,
            f"{island.name}_row",
            f"{island.name}_column",
            ChainedExecution(plan.threads),
            warp,
        )
        scope.check(plan)
        proof = event.reduction
        assert scope.domain.rows == (
            (proof.position, proof.position)
            if proof is not None and proof.axis == 0
            else (0, 15)
        )
        assert scope.domain.columns == (
            (proof.position, proof.position)
            if proof is not None and proof.axis == 1
            else (0, 15)
        )
        for changed in (
            replace(scope, domain=replace(scope.domain, rows=(0, 31))),
            replace(scope, domain=replace(scope.domain, row="other_event_row")),
            replace(
                scope, execution=ChainedExecution(plan.threads, thread="other_thread")
            ),
            replace(scope, warp=99),
        ):
            with pytest.raises(_UnsupportedChain, match="branch or owner"):
                changed.check(plan)
        with pytest.raises(_UnsupportedChain, match="foreign"):
            scope.check(replace(plan, row_registers=None))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_row_value_domain_calls_and_shared_default_remain_owned(dtype) -> None:
    with _cpu_codegen():
        bound = _sequence._bind_isolated((*_args(dtype), True))
        shared = bound.to_code(_config("shared"))
        with patch(
            "helion._compiler.cute.chained_row_register_emission.row_scope",
            side_effect=AssertionError("row-only coordinate scope"),
        ):
            assert bound.to_code(_config("shared")) == shared
        with patch.object(
            CoordinateContext,
            "normalize",
            side_effect=lambda expression, original: original,
        ):
            original = bound.to_code(_config("warp_rows"))
        simplified = bound.to_code(_config("warp_rows"))
    assert len(simplified) < len(original)

    def calls(text):
        return [
            ast.unparse(n.func)
            for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.Call) and ast.unparse(n.func) != "cutlass.Int32"
        ]

    assert calls(original) == calls(simplified)
    assert original.count("sync_threads()") == simplified.count("sync_threads()")
    assert original.count("cutlass.Float32(0) + ") == simplified.count(
        "cutlass.Float32(0) + "
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _split_view_sequence(a, b, rhs):
    output = torch.empty((a.shape[0], 16, 32), device=a.device, dtype=a.dtype)
    for batch in hl.tile(a.shape[0], block_size=1):
        product = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :])
        split_columns, unused_columns = hl.split(
            product.reshape(2, 16, 2, 16).permute(1, 3, 0, 2)
        )
        base, unused_rows = hl.split(split_columns)
        state = base
        index = hl.arange(16)
        for position in hl.static_range(1, 4):
            row = torch.where(index[:, None] == position, state, 0.0).sum(0)
            column = torch.where(index[None, :] == position, base, 0.0).sum(1)
            state = torch.tanh(state * 0.5 + column[:, None] * row[None, :] + 0.0625)
            state = torch.where(index[:, None] >= index[None, :], state, 0.0)
        output[batch.begin, :, :] = hl.dot(
            state.to(a.dtype), rhs[batch.begin, :, :]
        ).to(a.dtype)
    return output


def _view_args(dtype, device="cpu"):
    return tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape in ((3, 32, 32), (3, 32, 32), (3, 16, 32))
    )


def test_actual_split_view_composition_reduces_domains_without_fp_changes() -> None:
    changed = []
    normalize = CoordinateContext.normalize

    def observe(context, expression, original):
        result = normalize(context, expression, original)
        if result != original:
            changed.append(expression)
        return result

    with _cpu_codegen():
        bound = _split_view_sequence._bind_isolated(_view_args(torch.bfloat16))
        with patch.object(
            CoordinateContext,
            "normalize",
            side_effect=lambda expression, original: original,
        ):
            old = bound.to_code(_config("warp_rows"))
        with patch.object(CoordinateContext, "normalize", observe):
            new = bound.to_code(_config("warp_rows"))
    assert len(new) < len(old)
    pending = list(changed)
    kinds = set()
    while pending:
        value = pending.pop()
        kinds.add(value.kind)
        pending.extend(value.args)
    assert {"div", "mod"} <= kinds
    # The BF16 K32 stage uses the existing prepared warp helper; the second
    # contraction is inline. Both original native stages must remain.
    for spelling in ("cute.gemm(", "execute_prepared_warp_k("):
        assert old.count(spelling) == new.count(spelling) == 1
    assert old.count("shuffle_sync(") == new.count("shuffle_sync(") == 3
    assert old.count("sync_threads()") == new.count("sync_threads()")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_gpu_split_view_owned_coordinates_match_original_and_double(dtype) -> None:
    torch.manual_seed(927)
    args = _view_args(dtype, DEVICE)
    for arg in args:
        arg.normal_(0, 0.125)
    with patch.object(
        CoordinateContext,
        "normalize",
        side_effect=lambda expression, original: original,
    ):
        original = _split_view_sequence._bind_isolated(args).compile_config(
            _config("warp_rows")
        )
    simplified = _split_view_sequence._bind_isolated(args).compile_config(
        _config("warp_rows")
    )
    for iteration in range(2):
        if iteration:
            args[0].mul_(-1)
        saved = tuple(x.clone() for x in args)
        expected = []
        for a, b, rhs in zip(*args, strict=True):
            base = (a.double() @ b.double())[:16, :16]
            state = base
            for position in range(1, 4):
                state = torch.tril(
                    torch.tanh(
                        state * 0.5
                        + base[:, position, None] * state[position, None, :]
                        + 0.0625
                    )
                )
            expected.append(state.to(dtype).double() @ rhs.double())
        result = simplified(*args)
        torch.testing.assert_close(result, original(*args), atol=0, rtol=0)
        torch.testing.assert_close(
            result, torch.stack(expected).to(dtype), atol=0.002, rtol=0.002
        )
        torch.testing.assert_close(args, saved, atol=0, rtol=0)
