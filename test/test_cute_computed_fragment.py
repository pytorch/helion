from __future__ import annotations

import ast
from collections import Counter
from contextlib import ExitStack
import itertools
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._sympy.functions import FloorDiv

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion import exc
from helion._compiler.cute.computed_fragment import Fragment
from helion._compiler.cute.computed_fragment import FragmentCompiler
from helion._compiler.host_function import HostFunction
from helion._compiler.variable_origin import BlockSizeOrigin
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl
from helion.runtime.ref_mode import RefMode

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_recurrence(x: torch.Tensor, steps: hl.constexpr):
    batches = x.size(0)
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    mask_out = torch.empty_like(x, dtype=torch.bool)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(size)
        mask = index[:, None] > index[None, :]
        value = torch.where(mask, x[batch, :, :], 0.0)
        mask_out[batch, :, :] = mask[None, :, :]
        for _ in range(steps):
            value = hl.dot(value, value.transpose(-2, -1)) + value
        out[batch, :, :] = value
    return out, mask_out


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_broadcast_dot(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor):
    batches = x.size(0)
    rows = hl.specialize(x.size(1))
    columns = y.size(2)
    contraction = hl.specialize(x.size(2))
    assert contraction == y.size(1)
    out = torch.empty((batches, rows, columns), device=x.device, dtype=x.dtype)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(contraction)
        left = x[batch, :, index]
        right = y[batch, index, :] * scale[batch, index, None]
        out[batch, :, :] = hl.dot(left, right)
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("size,steps", [(5, 0), (5, 1), (8, 3)])
def test_static_fragment_axes_transpose_and_loop_carry(size, steps):
    torch.manual_seed(123)
    x = torch.randn((3, size, size), device=DEVICE) * 0.1
    expected_mask = (
        torch.arange(size, device=DEVICE)[:, None]
        > torch.arange(size, device=DEVICE)[None, :]
    )
    expected = torch.where(expected_mask, x, 0.0)
    for _ in range(steps):
        expected = expected @ expected.transpose(-2, -1) + expected
    actual, mask = _fragment_recurrence(x, steps)
    torch.testing.assert_close(mask, expected_mask.expand_as(mask))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_parallel_carries(x: torch.Tensor, y: torch.Tensor, steps: hl.constexpr):
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    other = torch.empty_like(y)
    for batch in hl.tile(x.size(0), block_size=1):
        index = hl.arange(size)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        left = x[batch, :, :] + eye[None, :, :]
        right = y[batch, :, :]
        for _ in range(steps):
            left, right = right, left + right
        out[batch, :, :] = left
        other[batch, :, :] = right
    return out, other


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("steps", [1, 3])
def test_fragment_parallel_carry_aliases_and_lazy_updates(steps):
    x = torch.randn((3, 5, 5), device=DEVICE)
    y = torch.randn_like(x)
    left, right = x + torch.eye(5, device=DEVICE), y
    for _ in range(steps):
        left, right = right, left + right
    actual = _fragment_parallel_carries(x, y, steps)
    torch.testing.assert_close(actual, (left, right))


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_half_carry_dot(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    flags: torch.Tensor,
    steps: hl.constexpr,
):
    out = torch.empty_like(x, dtype=torch.float32)
    for batch in hl.tile(x.size(0), block_size=1):
        state = x[batch, :, :].float()
        for _ in range(steps):
            if flags[batch.begin] > 0:
                product = hl.dot(z[batch, :, :], state.to(z.dtype))
                state = hl.dot(y[batch, :, :], product.to(y.dtype), acc=state)
            else:
                state = state * 0.5
        out[batch, :, :] = state
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
def test_half_dot_operand_reads_resident_loop_carry(dtype, steps):
    x = torch.randn((3, 3, 7), device=DEVICE, dtype=dtype) * 0.1
    y = torch.randn((3, 3, 5), device=DEVICE, dtype=dtype) * 0.1
    z = torch.randn((3, 5, 3), device=DEVICE, dtype=dtype) * 0.1
    flags = torch.tensor([0, 1, 1], device=DEVICE)
    expected = x.float()
    for _ in range(steps):
        product = z.float() @ expected.to(dtype).float()
        updated = y.float() @ product.to(dtype).float() + expected
        expected = torch.where(flags[:, None, None] > 0, updated, expected * 0.5)
    torch.testing.assert_close(
        _fragment_half_carry_dot(x, y, z, flags, steps),
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_scaled_half_dot(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty((x.size(0), x.size(1), y.size(2)), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        left = x[batch, :, :] * 0.5
        for columns in hl.tile(y.size(2), block_size=4):
            out[batch, :, columns] = hl.dot(left, y[batch, :, columns])
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_half_dot_keeps_repeated_operand_axes_independent(dtype):
    x = torch.randn((3, 5, 5), device=DEVICE, dtype=dtype).transpose(-2, -1)
    y = torch.randn((3, 7, 5), device=DEVICE, dtype=dtype).transpose(-2, -1)
    torch.testing.assert_close(
        _fragment_scaled_half_dot(x, y),
        (x * 0.5).float() @ y.float(),
        atol=2e-5,
        rtol=1e-5,
    )


@skipUnlessBackends(["cute"])
def test_computed_batched_dot_noncontiguous_broadcast():
    torch.manual_seed(124)
    x = torch.randn((3, 7, 5), device=DEVICE).transpose(-2, -1)
    y = torch.randn((3, 9, 7), device=DEVICE).transpose(-2, -1)
    scale = torch.randn((3, 7), device=DEVICE)
    actual = _fragment_broadcast_dot(x, y, scale)
    torch.testing.assert_close(
        actual, x @ (y * scale[:, :, None]), atol=2e-5, rtol=1e-5
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_batched_load_dot(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty((x.size(0), x.size(1), y.size(2)), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = hl.dot(x[batch, :, :], y[batch, :, :])
    return out


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_batched_computed_dot(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty((x.size(0), x.size(1), y.size(2)), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = hl.dot(x[batch, :, :] + 1, y[batch, :, :] + 1)
    return out


@skipUnlessBackends(["cute"])
def test_fragment_batched_dot_of_direct_loads():
    x = torch.randn((3, 5, 7), device=DEVICE)
    y = torch.randn((3, 7, 9), device=DEVICE)
    torch.testing.assert_close(_fragment_batched_load_dot(x, y), x @ y)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_select(x: torch.Tensor, flags: torch.Tensor, scale: float):
    batches = x.size(0)
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(size)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        value = x[batch, :, :] + eye[None, :, :]
        if flags[batch.begin] > 0:
            value = hl.dot(value, value.transpose(-2, -1))
        else:
            value = value * scale
        out[batch, :, :] = value
    return out


@skipUnlessBackends(["cute"])
def test_fragment_uniform_branch_and_offset_view():
    torch.manual_seed(125)
    x = torch.randn((4, 11, 11), device=DEVICE)[:, 1:11:2, 2:7]
    flags = torch.tensor([0, 1, 1, 0], device=DEVICE)
    actual = _fragment_select(x, flags, 0.125)
    value = x + torch.eye(5, device=DEVICE)
    expected = torch.where(
        flags[:, None, None] > 0, value @ value.transpose(-2, -1), value * 0.125
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_subrange(x: torch.Tensor, out: torch.Tensor):
    batches = x.size(0)
    rows = hl.specialize(x.size(1))
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(rows)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        for columns in hl.tile(2, x.size(2) - 3, block_size=4):
            out[batch, :, columns] = hl.dot(eye[None, :, :], x[batch, :, columns]) + 1.0
    return out


@skipUnlessBackends(["cute"])
def test_fragment_tiled_subrange_keeps_sentinels():
    x = torch.randn((3, 5, 14), device=DEVICE)
    out = torch.full_like(x, -123.0)
    actual = _fragment_subrange(x, out)
    expected = torch.full_like(x, -123.0)
    expected[:, :, 2:-3] = x[:, :, 2:-3] + 1.0
    torch.testing.assert_close(actual, expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_sum(x: torch.Tensor):
    batches = x.size(0)
    size = hl.specialize(x.size(1))
    rows = torch.empty((batches, size, 1), device=x.device)
    columns = torch.empty((batches, 1, size), device=x.device)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(size)
        value = torch.where(index[:, None] >= index[None, :], x[batch, :, :], 0.0)
        product = hl.dot(value, value.transpose(-2, -1))
        rows[batch, :, :] = product.sum(-1, keepdim=True)
        columns[batch, :, :] = product.sum(-2, keepdim=True)
    return rows, columns


@skipUnlessBackends(["cute"])
def test_fragment_sum_on_both_matrix_axes():
    x = torch.randn((3, 7, 7), device=DEVICE)
    rows, columns = _fragment_sum(x)
    value = torch.tril(x)
    product = value @ value.transpose(-2, -1)
    torch.testing.assert_close(
        rows, product.sum(-1, keepdim=True), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        columns, product.sum(-2, keepdim=True), atol=1e-5, rtol=1e-5
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_scan(x: torch.Tensor, reverse: hl.constexpr):
    batches = x.size(0)
    out = torch.empty_like(x)
    for batch in hl.tile(batches, block_size=1):
        value = x[batch, :, :] * 2.0 + 1.0
        out[batch, :, :] = hl.cumsum(value, dim=1, reverse=bool(reverse))
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fragment_computed_scan_nonlast_axis(reverse):
    x = torch.randn((3, 7, 5), device=DEVICE)
    value = x * 2 + 1
    expected = value.flip(1).cumsum(1).flip(1) if reverse else value.cumsum(1)
    torch.testing.assert_close(_fragment_scan(x, reverse), expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_scalar_fill(x: torch.Tensor, scales: torch.Tensor):
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    for batch in hl.tile(x.size(0), block_size=1):
        first = scales[batch.begin, 0]
        filled = hl.full([size, size], first, dtype=x.dtype)
        second = scales[batch.begin, 1]
        value = x[batch, :, :] * second
        out[batch, :, :] = hl.dot(filled[None, :, :], value)
    return out


@skipUnlessBackends(["cute"])
def test_fragment_dynamic_fill_keeps_scalar_storage():
    x = torch.randn((3, 5, 5), device=DEVICE)
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], device=DEVICE)
    filled = scales[:, 0, None, None].expand_as(x)
    expected = filled @ (x * scales[:, 1, None, None])
    torch.testing.assert_close(_fragment_scalar_fill(x, scales), expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_masked_sum(x: torch.Tensor):
    rows = hl.specialize(x.size(1))
    out = torch.empty((x.size(0), rows), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        index = hl.arange(rows)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        total = (
            hl.zeros([batch, rows], dtype=torch.float32) + eye.sum(-1).float()[None, :]
        )
        for columns in hl.tile(x.size(2), block_size=4):
            total = total + (x[batch, :, columns] + 1.0).sum(-1).float()
        out[batch, :] = total
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fragment_lowprecision_sum_masks_computed_tail(dtype):
    x = torch.randn((3, 8, 7), device=DEVICE, dtype=dtype)
    expected = torch.ones((3, 8), device=DEVICE)
    for start in range(0, 7, 4):
        expected += (x[:, :, start : start + 4] + 1).sum(-1).float()
    torch.testing.assert_close(_fragment_masked_sum(x), expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_masks(
    x: torch.Tensor, flags: torch.Tensor, out: torch.Tensor, unchanged: torch.Tensor
):
    size = hl.specialize(x.size(1))
    for batch in hl.tile(x.size(0), block_size=1):
        index = hl.arange(size)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        disabled = hl.full([], False, dtype=torch.bool)
        value = hl.load(x, [batch, slice(None), slice(None)], extra_mask=disabled) + 3.0
        result = hl.dot(eye[None, :, :], value)
        hl.store(
            out,
            [batch, slice(None), slice(None)],
            result,
            extra_mask=flags[batch.begin] > 0,
        )
        hl.store(
            unchanged, [batch, slice(None), slice(None)], result, extra_mask=disabled
        )
    return out, unchanged


@skipUnlessBackends(["cute"])
def test_fragment_scalar_memory_masks():
    x = torch.randn((3, 5, 5), device=DEVICE)
    flags = torch.tensor([0, 1, 0], device=DEVICE)
    out, unchanged = _fragment_masks(
        x, flags, torch.full_like(x, -7), torch.full_like(x, -9)
    )
    expected = torch.full_like(x, -7)
    expected[1] = 3
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(unchanged, torch.full_like(x, -9))


@skipUnlessBackends(["cute"])
def test_fragment_rejects_oversized_shared_storage_before_launch():
    with FakeTensorMode():
        x = torch.empty((1, 512, 512), device=DEVICE)
        bound = _fragment_recurrence.bind((x, 1))
    with pytest.raises(exc.InvalidConfig, match="shared bytes, exceeding"):
        bound.to_code(bound.config_spec.default_config())


def _fragment_extent_fixture(rows_per_block, columns_per_block):
    rows, columns, parts = sympy.symbols(
        "rows columns parts", integer=True, positive=True
    )
    symbols = (rows, columns, rows, columns, parts)
    numels = (17, 65, rows, columns, FloorDiv(columns + 64, columns))
    configured = (rows_per_block, columns_per_block, 64, 128, 8192)
    blocks = [
        SimpleNamespace(
            block_id=i,
            var=SimpleNamespace(_sympy_=lambda symbol=symbol: symbol),
            numel=sympy.sympify(numel),
            reduction=i >= 2,
        )
        for i, (symbol, numel) in enumerate(zip(symbols, numels, strict=True))
    ]
    compiler = object.__new__(FragmentCompiler)
    compiler.df = SimpleNamespace(
        resolved_block_size=lambda block_id: configured[block_id], literal_expr=str
    )
    compiler.env = SimpleNamespace(
        block_sizes=blocks,
        specialize_expr=lambda expr: expr,
        backend=SimpleNamespace(sympy_printer_expr=str),
    )
    compiler.sym_indices = {}
    compiler.offsets = {}
    host = SimpleNamespace(
        expr_to_origin={
            symbol: SimpleNamespace(origin=BlockSizeOrigin(block_id))
            for symbol, block_id in ((rows, 0), (columns, 1), (parts, 4))
        }
    )
    return compiler, host, rows, columns, parts


@pytest.mark.parametrize(
    "rows_per_block,columns_per_block", [(8, 16), (16, 32), (32, 64), (16, 128)]
)
def test_fragment_configured_extents_preserve_coordinate_geometry(
    rows_per_block, columns_per_block
):
    compiler, host, rows, columns, parts = _fragment_extent_fixture(
        rows_per_block, columns_per_block
    )
    expected_parts = (65 + columns_per_block - 1) // columns_per_block
    with patch.object(HostFunction, "current", return_value=host):
        assert compiler.shape((rows, parts)) == (rows_per_block, expected_parts)
        assert compiler.shape((rows, columns)) == (rows_per_block, columns_per_block)
        # The same symbolic dimension determines both allocation and coordinate
        # stride; alias reductions must not change either one.
        row, column = sympy.symbols("row column", integer=True)
        address = sympy.sympify(
            compiler.sym(row * parts + column), locals={"row": row, "column": column}
        )
        for r in range(rows_per_block):
            for c in range(expected_parts):
                assert int(address.subs({row: r, column: c})) == r * expected_parts + c
        assert compiler.extent(FloorDiv(columns + 64, columns)) == expected_parts


def test_fragment_configured_extents_preserve_dynamic_symbols_and_reject_cycles():
    compiler, host, rows, columns, parts = _fragment_extent_fixture(16, 32)
    dynamic = sympy.Symbol("dynamic", integer=True, positive=True)
    with patch.object(HostFunction, "current", return_value=host):
        expression = FloorDiv(dynamic + columns - 1, columns)
        assert compiler.configured_expr(expression) == FloorDiv(dynamic + 31, 32)
        with pytest.raises(exc.InvalidConfig, match="static local extents"):
            compiler.extent(expression)
        compiler.env.block_sizes[4].numel = parts
        with pytest.raises(exc.InvalidConfig, match="cyclic"):
            compiler.extent(parts)
        with pytest.raises(exc.InvalidConfig, match="cyclic"):
            compiler.sym(parts)


def test_fragment_coordinate_locals_preserve_arithmetic_and_read_scope():
    compiler = object.__new__(FragmentCompiler)
    compiler.expression = None
    compiler.snapshot_owner = None
    compiler.iteration_local_returns = frozenset()
    compiler.iteration_owner = None
    compiler.iteration_defined = set()
    statements = []

    def lift(expression, *, prefix):
        name = f"{prefix}_{len(statements)}"
        statements.append(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expression)
        )
        return ast.Name(id=name, ctx=ast.Load())

    compiler.cg = SimpleNamespace(lift=lift)
    assert compiler.coordinate_locals(("index", "0", "17")) == ("index", "0", "17")
    assert not statements
    expression = "((index // 3) % 7) + index * 5"
    first = compiler.coordinate_locals((expression,))[0]
    second = compiler.coordinate_locals((expression,))[0]
    assert first != second
    # The same coordinate must be read again after a value changes. No
    # expression or load cache may reuse the preceding lexical evaluation.
    tree = ast.Module(
        body=[
            statements[0],
            ast.Assign(
                targets=[ast.Name(id="index", ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id="index", ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=11),
                ),
            ),
            statements[1],
        ],
        type_ignores=[],
    )
    code = compile(ast.fix_missing_locations(tree), "<coordinate-locals>", "exec")
    for index in (-(2**60), -37, 0, 19, 2**60):
        values = {"index": index}
        exec(code, values)
        assert values[first] == (index // 3) % 7 + index * 5
        assert values[second] == ((index + 11) // 3) % 7 + (index + 11) * 5


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_coordinate_views(x: torch.Tensor):
    rows = x.size(1)
    columns = x.size(2)
    out = torch.empty((x.size(0), columns, rows), device=x.device, dtype=x.dtype)
    for batch in hl.tile(x.size(0), block_size=1):
        values = (x[batch, :, :] + 1).transpose(-1, -2)
        scanned = hl.cumsum(values, dim=-1)
        out[batch, :, :] = scanned + values
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_fragment_coordinate_views_scan_and_source_reuse(dtype, strided):
    x = torch.arange(3 * 5 * 14, device=DEVICE, dtype=dtype).reshape(3, 5, 14)
    x = x[:, :, 1::2] if strided else x[:, :, :7].contiguous()
    values = (x + 1).transpose(-1, -2)
    expected = values.cumsum(-1).to(dtype) + values
    torch.testing.assert_close(_fragment_coordinate_views(x), expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_byte_storage(
    x: torch.Tensor,
    mask: torch.Tensor,
    destination: torch.Tensor,
    reverse: hl.constexpr,
):
    copied = torch.empty_like(x)
    prefix = torch.empty_like(x)
    mask_copy = torch.empty_like(mask)
    for row in hl.tile(x.size(0), block_size=1):
        enabled = mask[row, :]
        values = hl.load(x, [row, slice(None)], extra_mask=enabled)
        copied[row, :] = values
        mask_copy[row, :] = enabled
        prefix[row, :] = hl.cumsum(values + 17, dim=-1, reverse=reverse)
        hl.store(
            destination,
            [row, slice(None)],
            values.float() + 0.75,
            extra_mask=values >= 128,
        )
    return copied, prefix, mask_copy


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fragment_unsigned_byte_masked_roundtrip_and_overflow(reverse):
    # Include high unsigned bytes, row/column tails, strided storage, and bool
    # bytes whose true representation is neither one nor a positive signed byte.
    backing = torch.arange(3 * 130, device=DEVICE).reshape(3, 130).to(torch.uint8)
    x = backing[:, ::2]
    x.copy_((torch.arange(195, device=DEVICE).reshape(3, 65) + 79).to(torch.uint8))
    mask_bytes = (
        torch.tensor([0, 1, 2, 128, 255], dtype=torch.uint8, device=DEVICE)
        .repeat(39)
        .reshape(3, 65)
    )
    mask = mask_bytes.view(torch.bool)
    output_backing = torch.full((5, 132), 77, dtype=torch.uint8, device=DEVICE)
    destination = output_backing[1:4, 1:131:2]
    expected = torch.where(mask, x, 0)
    values = expected + 17
    prefix = values.flip([-1]) if reverse else values
    prefix = prefix.cumsum(-1, dtype=torch.uint8)
    if reverse:
        prefix = prefix.flip([-1])
    expected_backing = output_backing.clone()
    expected_backing[1:4, 1:131:2] = torch.where(expected >= 128, expected, 77)
    inputs = [value.clone() for value in (backing, mask_bytes)]
    actual = _fragment_byte_storage(x, mask, destination, reverse)
    torch.testing.assert_close(actual, (expected, prefix, mask), rtol=0, atol=0)
    torch.testing.assert_close(output_backing, expected_backing, rtol=0, atol=0)
    torch.testing.assert_close((backing, mask_bytes), tuple(inputs), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_fp8_storage(x: torch.Tensor):
    copied = torch.empty_like(x)
    widened = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    prefix = torch.empty_like(widened)
    for row in hl.tile(x.size(0), block_size=1):
        raw = x[row, :]
        values = raw.float()
        narrowed = (values + 0.5).to(x.dtype)
        copied[row, :] = raw
        widened[row, :] = narrowed.float()
        prefix[row, :] = hl.cumsum(values + 1, dim=-1)
    return copied, widened, prefix


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fragment_fp8_typed_storage_and_conversions(dtype):
    values = torch.tensor([-4.0, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0, 4.0], device=DEVICE)
    x = values.repeat(49)[:390].reshape(3, 130).to(dtype)[:, ::2]
    copied, widened, prefix = _fragment_fp8_storage(x)
    torch.testing.assert_close(
        copied.view(torch.uint8), x.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        widened, (x.float() + 0.5).to(dtype).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(prefix, (x.float() + 1).cumsum(-1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2])
def test_fragment_logical_storage_cpu_codegen(dtype):
    import ast

    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        if dtype is torch.uint8:
            inputs = (
                torch.ones(3, 65, dtype=dtype),
                torch.ones(3, 65, dtype=torch.bool),
                torch.empty(3, 65, dtype=dtype),
                False,
            )
            kernel = _fragment_byte_storage
            types = {"Uint8", "Boolean"}
        else:
            inputs = (torch.ones(3, 65).to(dtype),)
            kernel = _fragment_fp8_storage
            types = {"Float8E4M3FN" if dtype is torch.float8_e4m3fn else "Float8E5M2"}
        bound = _cpu_bind(kernel, inputs)
        tree = ast.parse(bound.to_code(bound.config_spec.default_config()))
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("fragment_load")
    ]
    assert loads
    assert {
        node.value.func.attr
        for node in loads
        if isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "cutlass"
    } == types
    assert all(
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in types
        for node in loads
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_fp8e4m3fn_to_float32"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("kind", ["scan", "dot"])
def test_fragment_dynamic_capacity_codegen_and_rebinding(kind):
    kernel = helion.kernel(
        (_fragment_scan if kind == "scan" else _fragment_batched_computed_dot).fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    shapes = [(3, 7, 5), (3, 17, 9), (3, 3, 2)]
    inputs = [
        (torch.randn(shape), False)
        if kind == "scan"
        else (torch.randn(shape), torch.randn(shape[0], shape[2], shape[1]))
        for shape in shapes
    ]
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
    ):
        bounds = [kernel.bind(args) for args in inputs]
        assert len({id(bound) for bound in bounds}) == 3
        assert kernel.bind((inputs[0][0].clone(), inputs[0][1])) is bounds[0]
        allocate = FragmentCompiler.allocate
        for bound, capacity in zip(
            bounds, [(1, 8, 8), (1, 32, 16), (1, 4, 2)], strict=True
        ):
            allocations = []

            def record_allocation(compiler, value, allocations=allocations):
                allocations.append(value.shape)
                return allocate(compiler, value)

            assert (
                "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
            )
            with patch.object(FragmentCompiler, "allocate", record_allocation):
                code = bound.to_code(bound.config_spec.default_config())
            assert capacity in allocations
            # Runtime logical dimensions still control bounds and masking;
            # exact metadata guards make the static capacity safe to replay.
            assert "x.size(1)" in code
            assert "x.size(2)" in code
            if kind == "dot":
                contractions = [
                    node
                    for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id.startswith("fragment_k")
                ]
                assert contractions
                # Transformed padding is nonzero. The contraction must retain
                # a logical runtime bound rather than just storage capacity.
                for loop in contractions:
                    assert isinstance(loop.iter, ast.Call)
                    assert any(
                        isinstance(node, ast.Name)
                        for arg in loop.iter.args
                        for node in ast.walk(arg)
                    )
        # Direct calls to an old binding must also select the binding whose
        # capacity covers the current metadata, including a smaller replay.
        with ExitStack() as stack:
            for index, bound in enumerate(bounds):
                stack.enter_context(patch.object(bound, "_run", return_value=index))
                stack.enter_context(patch.object(bound, "_prepare_direct_call"))
            for index, args in enumerate(inputs):
                assert bounds[0](*args) == index


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("kind", ["scan", "dot"])
def test_fragment_dynamic_capacity_replay(kind):
    kernel = helion.kernel(
        (_fragment_scan if kind == "scan" else _fragment_batched_computed_dot).fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    for shape in [(3, 7, 5), (3, 17, 9), (3, 3, 2), (3, 7, 5)]:
        x = torch.arange(math.prod(shape), device=DEVICE, dtype=torch.float32).reshape(
            shape
        )
        if kind == "scan":
            output, expected = kernel(x, False), (x * 2 + 1).cumsum(1)
        else:
            y = (
                torch.arange(
                    shape[0] * shape[2] * shape[1], device=DEVICE, dtype=torch.float32
                ).reshape(shape[0], shape[2], shape[1])
                % 7
            )
            output, expected = kernel(x, y), (x + 1) @ (y + 1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _generated_output_address_coverage(
    source: str, output_name: str, blocks: int, threads: int
):
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(
            ast.Module(
                body=[n for n in tree.body if isinstance(n, ast.Assign)]
                + function.body,
                type_ignores=[],
            )
        )
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    stores = []

    def visit(node, loops=(), guards=()):
        if isinstance(node, ast.For):
            for child in node.body:
                visit(child, (*loops, node), guards)
            return
        if isinstance(node, ast.If):
            for child in node.body:
                visit(child, loops, (*guards, node.test))
            for child in node.orelse:
                visit(
                    child,
                    loops,
                    (*guards, ast.UnaryOp(op=ast.Not(), operand=node.test)),
                )
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            and any(
                isinstance(n, ast.Name) and n.id == output_name
                for n in ast.walk(node.func.value)
            )
        ):
            stores.append((node.func.value, loops, guards))
        for child in ast.iter_child_nodes(node):
            visit(child, loops, guards)

    visit(function)
    assert len(stores) == 1
    pointer, loops, guards = stores[0]
    assert len(loops) == 1
    loop = loops[0]
    assert isinstance(loop.target, ast.Name)
    result = Counter()
    for block in range(blocks):
        for thread in range(threads):
            env = {
                "cutlass": SimpleNamespace(
                    Int32=int, Int64=int, Uint32=int, Uint64=int
                ),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(
                        thread_idx=lambda thread=thread: (thread, 0, 0),
                        block_idx=lambda block=block: (block, 0, 0),
                    )
                ),
                "range": range,
                output_name: SimpleNamespace(
                    iterator=0, layout=SimpleNamespace(stride=(1,))
                ),
            }

            def value(expression, env=env):
                for node in ast.walk(expression):
                    if (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id not in env
                    ):
                        assert node.id in assignments, node.id
                        env[node.id] = value(assignments[node.id])
                return eval(
                    compile(
                        ast.fix_missing_locations(ast.Expression(expression)),
                        "<generated-address>",
                        "eval",
                    ),
                    {"__builtins__": {}},
                    env,
                )

            iterations = value(loop.iter)
            for position in iterations:
                # Local scalar assignments must be re-evaluated at every loop
                # iteration; retain only the external primitive environment.
                for name in assignments:
                    env.pop(name, None)
                env[loop.target.id] = position
                if all(value(guard) for guard in guards):
                    result[int(value(pointer))] += 1
    return dict(sorted(result.items()))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_grid_origin_scan(x: torch.Tensor, begin: hl.constexpr):
    out = torch.full((x.size(0),), -1, dtype=torch.int64, device=x.device)
    for row in hl.tile(begin, x.size(0)):
        values = hl.cumsum(x[row, :] + 1, dim=-1)
        out[row] = values.amax(-1).to(torch.int64) + row.index.to(torch.int64)
    return out


@pytest.mark.parametrize("block_rows", [1, 16, 64])
@pytest.mark.parametrize("begin", [0, 3])
@pytest.mark.parametrize("mode", ["serial", "cooperative"])
def test_fragment_grid_origins_cover_each_output_once(block_rows, begin, mode):
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(_fragment_grid_origin_scan, (torch.ones(37, 5), begin))
        config = bound.config_spec.default_config()
        config.config["block_sizes"][0] = block_rows
        config.config["cute_fragment_scan"] = mode
        source = bound.to_code(config)
    counts = _generated_output_address_coverage(
        source, "out", (37 - begin + block_rows - 1) // block_rows, 128
    )
    assert counts == dict.fromkeys(range(begin, 37), 1)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("block_rows", [1, 16])
@pytest.mark.parametrize("mode", ["serial", "cooperative"])
def test_fragment_grid_origins_native_scan_rows_and_tail(block_rows, mode):
    kernel = helion.kernel(
        _fragment_grid_origin_scan.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    x = torch.arange(37 * 5, device=DEVICE, dtype=torch.float32).reshape(37, 5)
    bound = kernel.bind((x, 3))
    config = bound.config_spec.default_config()
    config.config["block_sizes"][0] = block_rows
    config.config["cute_fragment_scan"] = mode
    expected = torch.full((37,), -1, device=DEVICE, dtype=torch.int64)
    expected[3:] = (x[3:] + 1).sum(-1).long() + torch.arange(3, 37, device=DEVICE)
    torch.testing.assert_close(
        bound.compile_config(config)(x, 3), expected, rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_independent_statistic(
    stats: torch.Tensor, values: torch.Tensor, offset: torch.Tensor | float
):
    rows, columns = values.shape
    block_columns = hl.register_block_size(columns)
    out = torch.empty(
        (rows, (columns + block_columns - 1) // block_columns),
        dtype=values.dtype,
        device=values.device,
    )
    for row, column in hl.tile([rows, columns], block_size=[None, block_columns]):
        maximum = stats[row, :].amax(-1)
        if isinstance(offset, torch.Tensor):
            shift = offset[row]
        else:
            shift = hl.full([row], offset, dtype=torch.float32)
        filtered = torch.where(
            values[row, column] >= maximum[:, None] + shift[:, None],
            values[row, column],
            0.0,
        )
        out[row, column.id] = filtered.sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_existing_tile_reduction(values: torch.Tensor):
    out = torch.empty_like(values)
    for row, column in hl.tile(values.shape):
        tile = values[row, column]
        out[row, column] = tile - tile.amax(-1)[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_existing_row_reduction(values: torch.Tensor):
    out = torch.empty((values.size(0),), dtype=values.dtype, device=values.device)
    for row in hl.tile(values.size(0)):
        out[row] = values[row, :].amax(-1)
    return out


def _simulate_independent_fragment(source, inputs, outputs, blocks):
    """Execute generated fragment phases serially, preserving every coordinate.

    Each fragment element loop visits all coordinates instead of one thread's
    residue class. The barriers therefore delimit complete CPU phases. This
    validates generated indexing/math, not native CUDA synchronization.
    """
    import operator

    import numpy as np

    packets = {"vector": 0, "scalar": 0}

    class Pointer:
        def __init__(self, tensor, offset=0):
            self.tensor = tensor
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.tensor, self.offset + int(offset))

        def storage(self):
            return self.tensor.as_strided(
                (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
                (1,),
                storage_offset=0,
            )

        def align(self, alignment):
            assert (
                self.tensor.data_ptr() + self.offset * self.tensor.element_size()
            ) % alignment == 0
            return self

        def load(self):
            packets["scalar"] += 1
            storage = self.storage()
            offset = self.tensor.storage_offset() + self.offset
            assert 0 <= offset < storage.numel()
            return storage[offset].item()

        def store(self, value):
            storage = self.storage()
            offset = self.tensor.storage_offset() + self.offset
            assert 0 <= offset < storage.numel()
            storage[offset] = value.item() if isinstance(value, np.generic) else value

    class Shared:
        def __init__(self, dtype, shape):
            self.values = np.empty(math.prod(shape), dtype=dtype)
            self.written = np.zeros(math.prod(shape), dtype=bool)

        def __getitem__(self, index):
            assert self.written[int(index)], f"read before shared write: {index}"
            return self.values[int(index)]

        def __setitem__(self, index, value):
            self.values[int(index)] = value
            self.written[int(index)] = True

    class Allocator:
        def allocate_tensor(self, dtype, layout, byte_alignment):
            return Shared(dtype, layout)

    def copy_packet(atom, source, destination):
        assert atom == 128
        assert destination.values.size == 4
        assert (
            source.iterator.tensor.data_ptr() + source.iterator.offset * 4
        ) % 16 == 0
        packets["vector"] += 1
        for lane in range(4):
            destination[lane] = (source.iterator + lane).load()

    class SerialElements(ast.NodeTransformer):
        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id.startswith(
                ("fragment_index", "fragment_packet_index")
            ):
                assert isinstance(node.iter, ast.Call) and len(node.iter.args) == 3
                node.iter.args[0] = ast.Constant(0)
                node.iter.args[2] = ast.Constant(1)
            return node

    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    function.decorator_list = []
    function = SerialElements().visit(function)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    for block in range(blocks):
        environment = {
            "operator": operator,
            "_cute_python_mod": operator.mod,
            "cutlass": SimpleNamespace(
                Float32=np.float32,
                Float64=np.float64,
                Int32=int,
                Int64=int,
                Uint32=int,
                Uint64=int,
                Boolean=bool,
                utils=SimpleNamespace(SmemAllocator=Allocator),
            ),
            "cute": SimpleNamespace(
                math=SimpleNamespace(min=np.minimum, max=np.maximum, tanh=np.tanh),
                make_layout=lambda shape: shape,
                make_rmem_tensor=lambda layout, dtype: Shared(dtype, layout),
                make_tensor=lambda iterator, layout: SimpleNamespace(
                    iterator=iterator, layout=layout
                ),
                make_copy_atom=lambda op, dtype, num_bits_per_copy: num_bits_per_copy,
                nvgpu=SimpleNamespace(CopyUniversalOp=lambda: None),
                copy=copy_packet,
                arch=SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0),
                    block_idx=lambda block=block: (block, 0, 0),
                    sync_threads=lambda: None,
                ),
            ),
        }
        for name, tensor in (inputs | outputs).items():
            if isinstance(tensor, torch.Tensor):
                environment[name] = SimpleNamespace(
                    iterator=Pointer(tensor),
                    layout=SimpleNamespace(stride=tensor.stride()),
                )
            else:
                environment[name] = tensor
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[*constants, function], type_ignores=[])
                ),
                "<generated-fragment-cpu>",
                "exec",
            ),
            environment,
        )
        environment[function.name](
            *(environment[arg.arg] for arg in function.args.args)
        )
    return packets


@pytest.mark.parametrize("widths", [(17, 65), (31, 19), (17, 17)])
@pytest.mark.parametrize("row_offset", [False, True])
@pytest.mark.parametrize("block_rows", [1, 4])
def test_independent_full_reduction_fragment_codegen_and_values(
    widths, row_offset, block_rows
):
    stats_width, value_width = widths
    rows, block_columns = 5, 8
    stats = (torch.arange(rows * stats_width).reshape(rows, stats_width) % 7).float()
    values = (torch.arange(rows * value_width).reshape(rows, value_width) % 19).float()
    offset = torch.arange(rows).float() if row_offset else 0.5
    inputs = (stats, values, offset)
    kernel = helion.kernel(
        _fragment_independent_statistic.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, inputs)
        config = bound.config_spec.default_config()
        config.config["block_sizes"][:] = [block_columns, block_rows]
        config.config["reduction_loops"] = [None]
        source = bound.to_code(config)
    assert "fragment_max" in source and "fragment_sum" in source
    parts = (value_width + block_columns - 1) // block_columns
    actual = torch.full((rows, parts), float("nan"))
    threshold = stats.amax(-1) + offset
    filtered = torch.where(values >= threshold[:, None], values, 0.0)
    expected = torch.stack(
        [
            filtered[:, begin : begin + block_columns].sum(-1)
            for begin in range(0, value_width, block_columns)
        ],
        -1,
    )
    _simulate_independent_fragment(
        source,
        {"stats": stats, "values": values, "offset": offset},
        {"out": actual},
        ((rows + block_rows - 1) // block_rows) * parts,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tiled", [False, True])
def test_existing_tile_or_row_reduction_keeps_native_path(tiled):
    kernel = helion.kernel(
        (
            _fragment_existing_tile_reduction
            if tiled
            else _fragment_existing_row_reduction
        ).fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, (torch.ones(5, 17),))
        source = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" not in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _native_same_axis_statistic(
    key: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(values)
    for row in hl.tile(values.size(0)):
        column = hl.arange(values.size(1))
        normalized = key[column] - key[column].amax()
        out[row, column] = values[row, column] * normalized[None, :]
    return out


def test_same_axis_statistic_broadcast_preserves_native_owner():
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(
            _native_same_axis_statistic, (torch.ones(17), torch.ones(5, 17))
        )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [None]
        source = bound.to_code(config)
    assert "fragment_smem" not in source
    assert "warp_reduction_max" in source


def test_independent_full_reduction_records_dynamic_metadata_before_projection():
    from helion._compiler.cute.computed_fragment import (
        computed_fragment_specialization_facts,
    )

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        kernel = helion.kernel(
            _fragment_independent_statistic.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
        )
        bound = _cpu_bind(kernel, (torch.ones(5, 17), torch.ones(5, 65), 0.5))
        facts = computed_fragment_specialization_facts(
            bound._env, bound.host_function.device_ir
        )
    assert "input_tensor_metadata" in facts


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("row_offset", [False, True])
def test_independent_full_reduction_fragment_native_values(row_offset):
    stats = (torch.arange(5 * 17, device=DEVICE).reshape(5, 17) % 7).float()
    values = (torch.arange(5 * 65, device=DEVICE).reshape(5, 65) % 19).float()
    offset = torch.arange(5, device=DEVICE).float() if row_offset else 0.5
    kernel = helion.kernel(
        _fragment_independent_statistic.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel.bind((stats, values, offset))
    config = bound.config_spec.default_config()
    config.config["block_sizes"][:] = [8, 4]
    config.config["reduction_loops"] = [None]
    threshold = stats.amax(-1) + offset
    filtered = torch.where(values >= threshold[:, None], values, 0.0)
    expected = torch.stack(
        [filtered[:, begin : begin + 8].sum(-1) for begin in range(0, 65, 8)], -1
    )
    torch.testing.assert_close(
        bound.compile_config(config)(stats, values, offset), expected, rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_factored_tiled_reduction(x: torch.Tensor, kind: hl.constexpr):
    rows, columns = x.shape
    dtype = torch.int64 if kind in ("argmin", "argmax") else x.dtype
    out = torch.empty((rows,), device=x.device, dtype=dtype)
    for row in hl.tile(rows):
        loaded = x[row, :]
        factored = loaded.reshape(loaded.size(0), 4, columns // 4)
        group_max = torch.amax(factored, dim=-1, keepdim=True)
        values = (factored + group_max).reshape(loaded.size(0), columns)
        if kind == "sum":
            result = torch.sum(values, dim=-1)
        elif kind == "prod":
            result = torch.prod(values, dim=-1)
        elif kind == "min":
            result = torch.amin(values, dim=-1)
        elif kind == "max":
            result = torch.amax(values, dim=-1)
        elif kind == "argmin":
            result = torch.argmin(values, dim=-1)
        else:
            result = torch.argmax(values, dim=-1)
        out[row] = result
    return out


def _factored_tiled_reference(x, kind):
    factored = x.reshape(x.size(0), 4, x.size(1) // 4)
    values = (factored + factored.amax(-1, keepdim=True)).reshape_as(x)
    return {
        "sum": torch.sum,
        "prod": torch.prod,
        "min": torch.amin,
        "max": torch.amax,
        "argmin": torch.argmin,
        "argmax": torch.argmax,
    }[kind](values, dim=-1)


def _factored_tiled_config(bound, tile):
    config = bound.config_spec.default_config()
    config.config["block_sizes"][:] = [2] * len(config.block_sizes)
    assert config.reduction_loops, "the regression must exercise a rollable reduction"
    config.config["reduction_loops"] = [tile] * len(config.reduction_loops)
    return config


@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max", "argmin", "argmax"])
@pytest.mark.parametrize("columns,tile", [(128, 16), (128, 32), (128, None)])
def test_fragment_owned_reduction_tiles_preserve_full_producer_and_carry(
    kind, columns, tile
):
    generator = torch.Generator().manual_seed(139)
    x = 0.49 + 0.02 * torch.rand((3, columns), generator=generator)
    # Equal extrema spanning tile boundaries check global first-index ties.
    x[0, 1] = x[0, columns - 2] = 0.75
    x[1, 3] = x[1, columns - 3] = 0.25
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(_fragment_factored_tiled_reduction, (x, kind))
        config = _factored_tiled_config(bound, tile)
        with bound.env:
            effective_tiles = bound._normalized_config_copy(config).reduction_loops
        source = bound.to_code(config)
    expected = _factored_tiled_reference(x, kind)
    actual = torch.full_like(expected, -9)
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    tile_loops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("fragment_reduce_tile")
    ]
    if all(value is None for value in effective_tiles):
        assert not tile_loops
    else:
        assert tile_loops
        for loop in tile_loops:
            assert isinstance(loop.iter, ast.Call)
            assert isinstance(loop.iter.args[-1], ast.Constant)
            assert loop.iter.args[-1].value in effective_tiles


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max", "argmin", "argmax"])
def test_fragment_owned_reduction_tiles_native(dtype, kind):
    generator = torch.Generator(device=DEVICE).manual_seed(139)
    x = (0.49 + 0.02 * torch.rand((3, 128), device=DEVICE, generator=generator)).to(
        dtype
    )
    x[0, 1] = x[0, 126] = 0.75
    x[1, 3] = x[1, 125] = 0.25
    bound = _fragment_factored_tiled_reduction.bind((x, kind))
    config = _factored_tiled_config(bound, 32)
    actual = bound.compile_config(config)(x, kind)
    torch.testing.assert_close(actual, _factored_tiled_reference(x, kind))


@pytest.mark.parametrize(
    "target,arguments,shape",
    [
        (torch.ops.aten.unsqueeze.default, (-1,), (2, 3)),
        (torch.ops.aten.unsqueeze.default, (0,), (2, 3)),
        (torch.ops.aten.unsqueeze.default, (-3,), (2, 3)),
        (torch.ops.aten.unsqueeze.default, (0,), ()),
        (torch.ops.aten.squeeze.default, (), (1, 2, 1, 3, 1)),
        (torch.ops.aten.squeeze.default, (), (1, 1)),
        (torch.ops.aten.squeeze.dim, (-1,), (3, 2, 1)),
        (torch.ops.aten.squeeze.dim, (1,), (3, 2, 1)),
        (torch.ops.aten.squeeze.dims, ([0, -1],), (1, 2, 3, 1)),
        (torch.ops.aten.squeeze.dims, ([0, 1],), (1, 2, 3, 1)),
    ],
)
@pytest.mark.parametrize("resident", [False, True])
def test_fragment_singleton_view_logical_coordinates(
    target, arguments, shape, resident
):
    # Build a genuinely noncontiguous source whenever its rank permits it.
    physical = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape[::-1])
    value = physical.permute(tuple(reversed(range(len(shape))))) if shape else physical
    expected = target(value, *arguments)
    compiler = object.__new__(FragmentCompiler)
    compiler.pending_local_atomics = set()
    compiler.snapshot_owner = None
    compiler.snapshot_shapes = set()
    compiler.local_register_nodes = set()
    compiler.local_register_slot = None
    compiler.shape = lambda sizes: tuple(sizes)
    compiler.coordinate_locals = lambda coordinates: coordinates
    graph = torch.fx.Graph()
    input_node = graph.placeholder("source")
    input_node.meta["val"] = value
    node = graph.call_function(target, (input_node, *arguments))
    node.meta.update(val=expected, lowering=None)
    source = Fragment(
        tuple(value.shape),
        value.dtype,
        lambda coords: f"source[{', '.join(coords) if coords else '()'}]",
        resident,
        storage="resident_storage" if resident else None,
    )
    result = compiler.node(node, {input_node: source})
    assert isinstance(result, Fragment)
    assert result.shape == tuple(expected.shape)
    assert result.resident == resident
    assert result.dependencies == (source,)
    assert FragmentCompiler.referenced_buffers([result]) == (
        {"resident_storage"} if resident else set()
    )
    for indices in itertools.product(*(range(size) for size in expected.shape)):
        expression = result.read(tuple(map(str, indices)))
        actual = eval(expression, {"__builtins__": {}}, {"source": value})
        torch.testing.assert_close(actual, expected[indices], rtol=0, atol=0)
    # A view must read its current source, including an updated resident carry.
    value.add_(7)
    indices = tuple(0 for _ in expected.shape)
    actual = eval(
        result.read(tuple(map(str, indices))),
        {"__builtins__": {}},
        {"source": value},
    )
    torch.testing.assert_close(
        actual, target(value, *arguments)[indices], rtol=0, atol=0
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_singleton_scan(x: torch.Tensor, steps: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        value = (x[row, :, :] + 1).transpose(-2, -1)
        value = value.unsqueeze(-1).squeeze(-1)
        value = value.unsqueeze(1).unsqueeze(-1).squeeze((1, -1))
        # Restore the logical shape after testing singleton and non-singleton
        # last axes. Default squeeze also removes the temporary leading one.
        shape = value.shape
        value = value.squeeze(-1).reshape(shape)
        value = value.unsqueeze(0).squeeze().reshape(shape)
        value = value.unsqueeze(0).squeeze(0)
        carry = hl.cumsum(value, dim=-1)
        for _ in range(steps):
            carry = hl.cumsum(carry.unsqueeze(-2).squeeze(-2) + 1, dim=-1)
        out[row, :, :] = carry.transpose(-2, -1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_singleton_native_path(x: torch.Tensor):
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape):
        value = x[row, column].unsqueeze(-1).squeeze(-1)
        out[row, column] = value + 1
    return out


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("steps", [0, 2])
def test_fragment_singleton_scan_codegen_and_rebind(static_shapes, steps):
    kernel = helion.kernel(
        _fragment_singleton_scan.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    shapes = [(3, 5, 7), (3, 9, 3), (3, 1, 4)]
    inputs = [
        torch.randn(shape).transpose(-2, -1).contiguous().transpose(-2, -1)
        for shape in shapes
    ]
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bounds = [kernel.bind((value, steps)) for value in inputs]
        assert len({id(bound) for bound in bounds}) == len(inputs)
        for value, bound in zip(inputs, bounds, strict=True):
            assert (
                kernel.bind((value.clone(memory_format=torch.preserve_format), steps))
                is bound
            )
            if not static_shapes:
                assert (
                    "input_tensor_metadata"
                    in bound.env.compiler_fact_specialization_facts
                )
            code = bound.to_code(bound.config_spec.default_config())
            assert "fragment_index" in code and "SmemAllocator" in code
        with ExitStack() as stack:
            for index, bound in enumerate(bounds):
                stack.enter_context(patch.object(bound, "_run", return_value=index))
                stack.enter_context(patch.object(bound, "_prepare_direct_call"))
            for index, value in enumerate(inputs):
                assert bounds[0](value, steps) == index


def test_fragment_singleton_views_alone_preserve_native_codegen():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_singleton_native_path, (torch.ones(3, 7),))
        code = bound.to_code(bound.config_spec.default_config())
    assert "fragment_index" not in code and "SmemAllocator" not in code


@pytest.mark.parametrize("steps", [0, 2])
@pytest.mark.parametrize("width", [1, 5])
def test_fragment_singleton_scan_eager_contract(steps, width):
    x = (
        torch.arange(3 * width * 7, dtype=torch.float32)
        .reshape(3, 7, width)
        .transpose(-2, -1)
    )
    expected = (x + 1).transpose(-2, -1).cumsum(-1)
    for _ in range(steps):
        expected = (expected + 1).cumsum(-1)
    kernel = helion.kernel(
        _fragment_singleton_scan.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ref_mode=RefMode.EAGER,
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        actual = kernel(x, steps)
    torch.testing.assert_close(actual, expected.transpose(-2, -1), rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("steps", [0, 2])
def test_fragment_singleton_views_native_scan_and_carry(steps):
    x = (
        torch.arange(3 * 5 * 7, device=DEVICE, dtype=torch.float32)
        .reshape(3, 7, 5)
        .transpose(-2, -1)
    )
    expected = (x + 1).transpose(-2, -1).cumsum(-1)
    for _ in range(steps):
        expected = (expected + 1).cumsum(-1)
    actual = _fragment_singleton_scan(x, steps)
    torch.testing.assert_close(actual, expected.transpose(-2, -1), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_reduction_mixed(x: torch.Tensor, kind: hl.constexpr):
    rows = x.size(0)
    dtype = (
        torch.int64 if x.dtype == torch.int32 and kind in ("sum", "prod") else x.dtype
    )
    out = torch.empty((rows,), device=x.device, dtype=dtype)
    out_index = torch.empty((rows,), device=x.device, dtype=torch.int64)
    for row in hl.tile(rows):
        values = hl.cumsum(x[row, :] + 1, dim=-1)
        if kind == "sum":
            result = values.sum(-1)
        elif kind == "prod":
            result = values.prod(-1)
        elif kind == "min":
            result = values.amin(-1)
        else:
            result = values.amax(-1)
        out[row] = result
        out_index[row] = values.argmax(-1)
    return out, out_index


def _fragment_warp_reduction_inputs(rows, columns, dtype, kind, device="cpu"):
    x = (torch.arange(rows * columns, device=device).reshape(rows, columns) % 5).to(
        dtype
    )
    if kind == "prod":
        x.fill_(-1)
        x[:, 0] = 0
    elif kind == "min":
        x += 1
    elif kind == "max":
        x = -x - 2
    if dtype == torch.int64 and kind != "prod":
        large = 1 << (48 if kind == "sum" else 54)
        x += -large if kind == "max" else large
    if dtype == torch.float64 and kind != "prod":
        x += -(2**40 + 0.25) if kind == "max" else 2**40 + 0.25
    return x


def _fragment_warp_reduction_reference(x, kind):
    values = (x + 1).cumsum(-1, dtype=x.dtype)
    reduction = {
        "sum": torch.sum,
        "prod": torch.prod,
        "min": torch.amin,
        "max": torch.amax,
    }[kind]
    return reduction(values, dim=-1), values.argmax(-1)


def _fragment_warp_reduction_config(bound, mode, tile):
    config = bound.config_spec.default_config()
    config.config["block_sizes"][:] = [2] * len(config.block_sizes)
    config.config["reduction_loops"] = [tile] * len(config.reduction_loops)
    config.config["cute_fragment_reduction"] = mode
    return config


def _simulate_fragment_warp_reduction(
    source, inputs, outputs, blocks, threads=128, *, allow_lane_stores=False
):
    """Execute emitted scalar programs with real 32-lane shuffle exchanges.

    Ordinary fragment element loops run each element once per phase. Cooperative
    element loops run 32 Python generators in lockstep; a yielded shuffle reads
    the peer's value from the same instruction before any lane advances. This
    checks generated addressing, identities and full participation, not CUDA's
    implementation of shuffle/barrier intrinsics.
    """
    import operator
    from typing import Any

    import numpy as np

    state = SimpleNamespace(lane=None, exchanges=0, leader_stores=0)

    class Pointer:
        def __init__(self, tensor, offset=0):
            self.tensor, self.offset = tensor, offset

        def __add__(self, offset):
            return Pointer(self.tensor, self.offset + int(offset))

        def load(self):
            assert 0 <= self.offset < self.tensor.numel()
            return self.tensor.reshape(-1)[self.offset].item()

        def store(self, value):
            assert 0 <= self.offset < self.tensor.numel()
            # Preserve Int64 low bits rather than converting through float.
            self.tensor.reshape(-1)[self.offset] = (
                value.item() if isinstance(value, np.generic) else value
            )

    class Shared:
        def __init__(self, dtype, layout):
            self.values = [None] * math.prod(layout)
            self.dtype = dtype

        def __getitem__(self, index):
            assert 0 <= int(index) < len(self.values)
            assert self.values[int(index)] is not None, f"read before write: {index}"
            return self.values[int(index)]

        def __setitem__(self, index, value):
            assert 0 <= int(index) < len(self.values)
            if state.lane is not None:
                assert allow_lane_stores or state.lane == 0, (
                    "a nonleader wrote a cooperative result"
                )
                state.leader_stores += 1
            self.values[int(index)] = self.dtype(value)

    class Allocator:
        def allocate_tensor(self, dtype, layout, byte_alignment):
            return Shared(dtype, layout)

    def run_warp(function):
        lanes = [function(lane) for lane in range(32)]

        def advance(values=None):
            events = []
            for lane, program in enumerate(lanes):
                state.lane = lane
                try:
                    events.append(
                        next(program) if values is None else program.send(values[lane])
                    )
                except StopIteration:
                    events.append(None)
            state.lane = None
            assert all(event is None for event in events) or all(
                event is not None for event in events
            ), "partial-warp collective participation"
            return events

        events = advance()
        while events[0] is not None:
            offset, kind = events[0][1:]
            assert all(event[1:] == (offset, kind) for event in events)
            if kind == "index":
                assert 0 <= offset < 32
                peers = [events[offset][0] for lane in range(32)]
            else:
                assert offset in (16, 8, 4, 2, 1)
                peers = [
                    events[
                        lane ^ offset
                        if kind == "bfly"
                        else max(lane - offset, lane if lane < offset else 0)
                    ][0]
                    for lane in range(32)
                ]
            state.exchanges += 1
            events = advance(peers)

    class Shuffle(ast.NodeTransformer):
        def visit_Call(self, node):
            kind = {
                "cute.arch.shuffle_sync_bfly": "bfly",
                "cute.arch.shuffle_sync_up": "up",
                "cute.arch.shuffle_sync": "index",
            }.get(ast.unparse(node.func))
            if kind is None:
                return self.generic_visit(node)
            kwargs = {keyword.arg: keyword.value for keyword in node.keywords}
            assert ast.literal_eval(kwargs["mask"]) == 0xFFFFFFFF
            assert ast.literal_eval(kwargs["mask_and_clamp"]) == (
                0 if kind == "up" else 31
            )
            assert len(node.args) == (2 if kind == "index" else 1)
            offset = node.args[1] if kind == "index" else kwargs["offset"]
            return ast.Yield(
                ast.Tuple([node.args[0], offset, ast.Constant(kind)], ast.Load())
            )

    class SerialPhases(ast.NodeTransformer):
        def visit_For(self, node):
            if not isinstance(node.target, ast.Name) or not node.target.id.startswith(
                "fragment_index"
            ):
                return self.generic_visit(node)
            assert isinstance(node.iter, ast.Call) and len(node.iter.args) == 3
            cooperative = isinstance(node.iter.args[0], ast.BinOp) and isinstance(
                node.iter.args[0].op, ast.FloorDiv
            )
            if cooperative:
                start = node.iter.args[0]
                assert isinstance(start, ast.BinOp)
                assert isinstance(start.left, ast.Name)
                assert ast.literal_eval(start.right) == 32
                assert ast.literal_eval(node.iter.args[2]) == threads // 32
                count = ast.literal_eval(node.iter.args[1])
                ownership = [
                    list(range(thread // 32, count, threads // 32))
                    for thread in range(threads)
                ]
                for warp in range(threads // 32):
                    assert all(
                        indices == ownership[warp * 32]
                        for indices in ownership[warp * 32 : (warp + 1) * 32]
                    )
                assert Counter(
                    index
                    for thread in range(0, threads, 32)
                    for index in ownership[thread]
                ) == Counter(range(count))
                function = ast.parse("def _warp_element(_lane):\n    pass").body[0]
                assert isinstance(function, ast.FunctionDef)
                function.body = [
                    ast.Assign(
                        [ast.Name(start.left.id, ast.Store())],
                        ast.Name("_lane", ast.Load()),
                    ),
                    *[Shuffle().visit(statement) for statement in node.body],
                ]
                assert any(isinstance(item, ast.Yield) for item in ast.walk(function))
                node.body = [
                    function,
                    ast.Expr(
                        ast.Call(
                            ast.Name("_run_warp", ast.Load()),
                            [ast.Name("_warp_element", ast.Load())],
                            [],
                        )
                    ),
                ]
            node.iter.args[0] = ast.Constant(0)
            node.iter.args[2] = ast.Constant(1)
            return node

    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    function.decorator_list = []
    for parent in ast.walk(function):
        for _, body in ast.iter_fields(parent):
            if not isinstance(body, list):
                continue
            phases = [
                index
                for index, statement in enumerate(body)
                if (
                    isinstance(statement, ast.For)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id.startswith("fragment_index")
                )
            ]
            # Code cleanup removes the redundant first/last barrier and merges
            # adjacent barriers. Every dependent fragment phase still needs a
            # CTA barrier before the following phase can read/reuse its storage.
            for before, after in itertools.pairwise(phases):
                assert any(
                    ast.unparse(statement) == "cute.arch.sync_threads()"
                    for statement in body[before + 1 : after]
                )
    function = SerialPhases().visit(function)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    for block in range(blocks):
        environment: dict[str, Any] = {
            "operator": operator,
            "_cute_python_mod": operator.mod,
            "_run_warp": run_warp,
            "cutlass": SimpleNamespace(
                Float16=np.float16,
                BFloat16=lambda value: torch.tensor(float(value)).bfloat16().item(),
                Float32=np.float32,
                Float64=np.float64,
                Int32=np.int32,
                Int64=np.int64,
                Uint32=np.uint32,
                Uint64=np.uint64,
                Boolean=bool,
                utils=SimpleNamespace(SmemAllocator=Allocator),
            ),
            "cute": SimpleNamespace(
                math=SimpleNamespace(min=np.minimum, max=np.maximum),
                make_layout=lambda shape: shape,
                arch=SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0),
                    block_idx=lambda block=block: (block, 0, 0),
                    sync_threads=lambda: None,
                ),
            ),
        }
        for name, tensor in (inputs | outputs).items():
            environment[name] = SimpleNamespace(
                iterator=Pointer(tensor), layout=SimpleNamespace(stride=tensor.stride())
            )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[*constants, function], type_ignores=[])
                ),
                "<generated-fragment-warp-cpu>",
                "exec",
            ),
            environment,
        )
        environment[function.name](
            *(environment[arg.arg] for arg in function.args.args)
        )
    return state


@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    ],
)
def test_fragment_warp_reduction_cpu_mixed_argmax_and_logical_tail(kind, dtype):
    x = _fragment_warp_reduction_inputs(3, 65, dtype, kind)
    expected = _fragment_warp_reduction_reference(x, kind)
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(_fragment_warp_reduction_mixed, (x, kind))
        for mode in ("serial", "warp"):
            config = _fragment_warp_reduction_config(bound, mode, 16)
            source = bound.to_code(config)
            actual = tuple(torch.full_like(value, -9) for value in expected)
            model = _simulate_fragment_warp_reduction(
                source, {"x": x}, {"out": actual[0], "out_index": actual[1]}, 2
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert (model.exchanges > 0) == (mode == "warp")
            assert (model.leader_stores > 0) == (mode == "warp")
            assert "fragment_selected_index" in source


@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max", "argmin", "argmax"])
def test_fragment_warp_reduction_cpu_nan_across_tiles(kind):
    x = torch.full((3, 128), 0.5)
    x[0, 0] = float("nan")
    x[1, 33] = float("nan")
    x[2, 127] = float("nan")
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(_fragment_factored_tiled_reduction, (x, kind))
        config = _fragment_warp_reduction_config(bound, "warp", 16)
        source = bound.to_code(config)
    assert "fragment_reduce_tile" in source
    expected = _factored_tiled_reference(x, kind)
    actual = torch.full_like(expected, -9)
    model = _simulate_fragment_warp_reduction(source, {"x": x}, {"out": actual}, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert model.exchanges > 0


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize("columns", [5, 65])
def test_fragment_warp_reduction_native_mixed_argmax_and_tails(kind, dtype, columns):
    x = _fragment_warp_reduction_inputs(3, columns, dtype, kind, "cuda")
    bound = _fragment_warp_reduction_mixed.bind((x, kind))
    config = _fragment_warp_reduction_config(bound, "warp", 16)
    actual = bound.compile_config(config)(x, kind)
    torch.testing.assert_close(actual, _fragment_warp_reduction_reference(x, kind))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max", "argmin", "argmax"])
def test_fragment_warp_reduction_native_nan_across_tiles(kind):
    x = torch.full((3, 128), 0.5, device=DEVICE)
    x[0, 0] = float("nan")
    x[1, 33] = float("nan")
    x[2, 127] = float("nan")
    bound = _fragment_factored_tiled_reduction.bind((x, kind))
    config = _fragment_warp_reduction_config(bound, "warp", 16)
    actual = bound.compile_config(config)(x, kind)
    torch.testing.assert_close(
        actual, _factored_tiled_reference(x, kind), equal_nan=True
    )


class _FragmentExpressionCodegen:
    def __init__(self):
        self.statements_stack = [[]]
        self.counter = itertools.count()

    def lift(self, expression, *, prefix="v"):
        name = f"{prefix}_{next(self.counter)}"
        self.statements_stack[-1].append(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expression)
        )
        return ast.Name(id=name, ctx=ast.Load())

    def execute(self, namespace):
        code = ast.fix_missing_locations(
            ast.Module(body=self.statements_stack[0], type_ignores=[])
        )
        exec(compile(code, "<fragment-expression>", "exec"), namespace)
        return namespace


def _fragment_expression_compiler():
    compiler = object.__new__(FragmentCompiler)
    codegen = _FragmentExpressionCodegen()
    compiler.cg = cast("GenerateAST", codegen)
    compiler.expression = None
    compiler.snapshot_owner = None
    compiler.iteration_local_returns = frozenset()
    compiler.iteration_owner = None
    compiler.iteration_defined = set()
    return compiler, codegen


@pytest.mark.parametrize("depth", [2, 8, 32])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32]
)
def test_fragment_expression_shared_dag_is_linear_and_typed(depth, dtype):
    compiler, codegen = _fragment_expression_compiler()
    calls = Counter()
    fragments = [Fragment((5,), dtype, lambda coords: f"source[{coords[0]}]")]
    for level in range(depth):
        parent = fragments[-1]
        producer = object()

        def read(coords, parent=parent, producer=producer, level=level):
            def evaluate():
                calls[level] += 1
                # Two syntactically different paths to the same logical input.
                lhs = parent.read(compiler.coordinate_locals(coords))
                rhs = parent.read(
                    compiler.coordinate_locals(
                        tuple(f"(({x}) // 1) + 0" for x in coords)
                    )
                )
                return f"cast((({lhs}) + ({rhs})) * 0.25 + 0.125)"

            return compiler.pointwise_read(producer, coords, evaluate)

        fragments.append(Fragment((5,), dtype, read, dependencies=(parent,)))
    result = fragments[-1].read(("position",))
    assert calls == Counter(dict.fromkeys(range(depth), 1))
    assert compiler.expression is None
    assert len(codegen.statements_stack[0]) <= 2 * depth + 2
    source = torch.tensor([-2.5, -0.125, 0.0, 1.5, 4.25], dtype=dtype)

    def typed(value):
        return torch.as_tensor(value).to(dtype).item()

    for position in range(5):
        expected = source[position].item()
        for _ in range(depth):
            expected = typed((expected + expected) * 0.25 + 0.125)
        namespace = codegen.execute(
            {"source": source, "position": position, "cast": typed}
        )
        assert namespace[result] == expected


def test_fragment_expression_resident_reads_refresh_after_write_and_tile_change():
    compiler, codegen = _fragment_expression_compiler()
    storage = Fragment(
        (3,),
        torch.int32,
        lambda coords: f"storage[{coords[0]}]",
        True,
        storage="storage",
    )
    producer = object()
    value = Fragment(
        (3,),
        torch.int32,
        lambda coords: compiler.pointwise_read(
            producer, coords, lambda: f"int({storage.read(coords)}) + 1"
        ),
        dependencies=(storage,),
    )
    first = value.read(("position",))
    codegen.statements_stack[-1].extend(ast.parse("storage[position] = 11").body)
    second = value.read(("position",))
    codegen.statements_stack[-1].extend(ast.parse("position = 2").body)
    third = value.read(("position",))
    namespace = codegen.execute({"storage": [3, 5, 7], "position": 0})
    assert [namespace[name] for name in (first, second, third)] == [4, 12, 8]


@pytest.mark.parametrize("flag", [False, True])
def test_fragment_expression_scopes_do_not_reuse_branch_or_loop_locals(flag):
    compiler, codegen = _fragment_expression_compiler()
    producer = object()

    def read():
        return compiler.pointwise_read(
            producer, ("position",), lambda: "storage[position] + 1"
        )

    branches = []
    for side in (1, -1):
        body = ast.parse(f"storage[position] += {side}").body
        codegen.statements_stack.append(body)
        value = read()
        body.extend(ast.parse(f"outputs.append({value})").body)
        codegen.statements_stack.pop()
        branches.append(body)
    codegen.statements_stack[0].append(
        ast.If(
            test=ast.Name(id="flag", ctx=ast.Load()),
            body=branches[0],
            orelse=branches[1],
        )
    )
    loop = ast.parse("for position in range(3):\n    storage[position] += 2").body[0]
    assert isinstance(loop, ast.For)
    codegen.statements_stack.append(loop.body)
    value = read()
    loop.body.extend(ast.parse(f"outputs.append({value})").body)
    codegen.statements_stack.pop()
    codegen.statements_stack[0].append(loop)
    namespace = codegen.execute(
        {"flag": flag, "storage": [3, 5, 7], "position": 0, "outputs": []}
    )
    expected_first = 5 if flag else 3
    assert namespace["outputs"] == [expected_first, expected_first + 2, 8, 10]


def test_fragment_expression_coordinate_keys_preserve_permutations_and_offsets():
    compiler, codegen = _fragment_expression_compiler()
    producer = object()
    calls = Counter()

    def read(coords):
        def evaluate():
            calls[coords] += 1
            return f"10 * ({coords[0]}) + ({coords[1]})"

        return compiler.pointwise_read(producer, coords, evaluate)

    def evaluate():
        a = read(("row", "column"))
        b = read(("column", "row"))
        c = read(("row + 1", "column"))
        d = read(("row", "column"))
        return f"({a}, {b}, {c}, {d})"

    result = compiler.pointwise_read(object(), (), evaluate)
    namespace = codegen.execute({"row": 2, "column": 5})
    assert namespace[result] == (25, 52, 35, 25)
    assert len(calls) == 3 and sum(calls.values()) == 3


@pytest.mark.parametrize("coordinate", [-65, -1, 0, 1, 65])
def test_fragment_expression_coordinate_integer_boundaries(coordinate):
    compiler, codegen = _fragment_expression_compiler()
    producer = object()

    def evaluate():
        indices = compiler.coordinate_locals(
            ("index // 1", "index % 1", "index // 4", "index % 4", "index + 1")
        )
        return f"({', '.join(indices)})"

    result = compiler.pointwise_read(producer, ("index",), evaluate)
    namespace = codegen.execute({"index": coordinate})
    assert namespace[result] == (
        coordinate,
        0,
        coordinate // 4,
        coordinate % 4,
        coordinate + 1,
    )


def test_fragment_expression_failed_evaluation_restores_scope():
    compiler, codegen = _fragment_expression_compiler()

    def evaluate():
        compiler.coordinate_locals(("position + 1",))
        raise ValueError("failed producer")

    with pytest.raises(ValueError, match="failed producer"):
        compiler.pointwise_read(object(), ("position",), evaluate)
    assert compiler.expression is None


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_expression_diamond_scan(x: torch.Tensor, depth: int):
    depth = hl.specialize(depth)
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        value = x[row, :] + 0.125
        for _ in hl.static_range(depth):
            first = value + 0.25
            second = value - 0.25
            value = first * 0.25 + second * 0.25
        out[row, :] = hl.cumsum(value.float(), dim=-1)
    return out


@pytest.mark.parametrize("row_tile", [1, 4])
def test_fragment_expression_real_graph_codegen_growth(row_tile):
    counts = []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        for depth in (2, 4, 8):
            bound = _cpu_bind(
                _fragment_expression_diamond_scan, (torch.ones(3, 31), depth)
            )
            config = bound.config_spec.default_config()
            config.config["block_sizes"] = [row_tile]
            source = bound.to_code(config)
            counts.append(sum(1 for _ in ast.walk(ast.parse(source))))
            assert "fragment_value" in source
    # Producer count doubles; the scalar AST must grow proportionally, not
    # duplicate every path through the unrolled diamond graph.
    assert counts[1] < counts[0] * 2
    assert counts[2] < counts[1] * 2


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("row_tile", [1, 4])
def test_fragment_expression_native_diamond_scan(dtype, row_tile):
    x = torch.linspace(-1, 1, 3 * 31, device=DEVICE, dtype=dtype).reshape(3, 31)
    expected = x + 0.125
    for _ in range(8):
        first, second = expected + 0.25, expected - 0.25
        expected = first * 0.25 + second * 0.25
    expected = expected.float().cumsum(-1)
    bound = _fragment_expression_diamond_scan.bind((x, 8))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [row_tile]
    torch.testing.assert_close(bound.compile_config(config)(x, 8), expected)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_captured_reduction(x: torch.Tensor, steps: int):
    out = torch.empty((x.size(0),), dtype=torch.int32, device=x.device)
    for row in hl.tile(x.size(0)):
        weights = x[row, :]
        threshold = hl.full([row], 0, dtype=torch.int32)
        for _step in range(steps):
            count = (weights > threshold[:, None]).sum(-1)
            threshold = torch.where(count >= 3, threshold + 1, threshold)
        out[row] = threshold
    return out


def _captured_reduction_reference(x, steps):
    threshold = torch.zeros(x.size(0), dtype=torch.int32, device=x.device)
    for _step in range(steps):
        count = (x > threshold[:, None]).sum(-1)
        threshold = torch.where(count >= 3, threshold + 1, threshold)
    return threshold


def _captured_reduction_config(bound, row_tile, reduction_tile):
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [row_tile]
    config.config["reduction_loops"] = [reduction_tile] * len(config.reduction_loops)
    # Four lanes intentionally require a synthetic lane loop in the old scalar
    # path even for width 17. A complete fragment owns its own CTA geometry.
    config.config["num_threads"] = [1, 4]
    return config


@pytest.mark.parametrize(
    "width,row_tile,reduction_tile", [(17, 1, None), (65, 2, 16), (1025, 2, 32)]
)
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_captured_full_reduction_loop_generated_values(
    width, row_tile, reduction_tile, steps
):
    x = (torch.arange(3 * width).reshape(3, width) % 9).float() - 2
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_captured_reduction, (x, steps))
        source = bound.to_code(
            _captured_reduction_config(bound, row_tile, reduction_tile)
        )
    actual = torch.full((3,), -1, dtype=torch.int32)
    _simulate_independent_fragment(
        source,
        {"x": x, "steps": steps},
        {"out": actual},
        (3 + row_tile - 1) // row_tile,
    )
    torch.testing.assert_close(
        actual, _captured_reduction_reference(x, steps), rtol=0, atol=0
    )


def test_captured_full_reduction_loop_default_values():
    x = torch.ones(3, 1025)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_captured_reduction, (x, 3))
        source = bound.to_code(bound.config_spec.default_config())
    actual = torch.full((3,), -1, dtype=torch.int32)
    _simulate_independent_fragment(source, {"x": x, "steps": 3}, {"out": actual}, 3)
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["capacity", "admission"])
def test_captured_full_reduction_loop_rejects_unsafe_fallback(failure):
    x = torch.ones(3, 131072 if failure == "capacity" else 17)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_captured_reduction, (x, 3))
        config = _captured_reduction_config(bound, 1, None)
        with ExitStack() as stack:
            if failure == "admission":
                stack.enter_context(
                    patch(
                        "helion._compiler.cute.computed_fragment.computed_fragment_supported",
                        return_value=False,
                    )
                )
            with pytest.raises(
                exc.InvalidConfig, match="shared bytes|captured full reductions"
            ):
                bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_tiled_capture_control(x: torch.Tensor, steps: int):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        for col in hl.tile(x.size(1)):
            weights = x[row, col]
            threshold = hl.full([row], 0, dtype=torch.int32)
            for _step in range(steps):
                count = (weights > threshold[:, None]).sum(-1)
                threshold = torch.where(count >= 3, threshold + 1, threshold)
            out[row, col] = threshold[:, None].to(x.dtype)
    return out


def test_captured_explicit_tile_reduction_keeps_native_owner():
    from helion._compiler.cute.captured_reduction import captured_reduction_coordinates

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_tiled_capture_control, (torch.ones(3, 17), 3))
        with bound.env, bound.host_function:
            assert not captured_reduction_coordinates(
                bound.env, bound.host_function.device_ir.graphs
            )
        source = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" not in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "width,row_tile,reduction_tile", [(17, 1, None), (65, 2, 16), (1025, 2, 32)]
)
def test_captured_full_reduction_loop_native(width, row_tile, reduction_tile):
    x = (torch.arange(3 * width, device=DEVICE).reshape(3, width) % 9).float() - 2
    bound = _fragment_captured_reduction.bind((x, 3))
    actual = bound.compile_config(
        _captured_reduction_config(bound, row_tile, reduction_tile)
    )(x, 3)
    torch.testing.assert_close(
        actual, _captured_reduction_reference(x, 3), rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_captured_conditional(
    x: torch.Tensor, steps: int, unchanged: hl.constexpr
):
    out = torch.empty((x.size(0),), dtype=torch.int32, device=x.device)
    for row in hl.tile(x.size(0)):
        weights = x[row, :]
        value = hl.full([row], 0, dtype=torch.int32)
        for _step in range(steps):
            if steps > 1:
                count = (weights > value[:, None]).sum(-1)
                value = torch.where(count >= 3, value + 1, value)
            elif not unchanged:
                value = value + 2
        out[row] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_captured_conditional_missing(x: torch.Tensor, steps: int):
    out = torch.empty((x.size(0),), dtype=torch.int32, device=x.device)
    for row in hl.tile(x.size(0)):
        weights = x[row, :]
        value = hl.full([row], 0, dtype=torch.int32)
        for _step in range(steps):
            if steps > 1:
                count = (weights > value[:, None]).sum(-1)
                value = torch.where(count >= 3, value + 1, value)
        out[row] = value
    return out


@pytest.mark.parametrize("steps", [3, 4])
def test_captured_conditional_other_branch_supplies_unchanged_output(steps):
    x = torch.ones(3, 17)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_captured_conditional_missing, (x, steps))
        source = bound.to_code(_captured_reduction_config(bound, 1, 16))
    actual = torch.full((3,), -1, dtype=torch.int32)
    _simulate_independent_fragment(source, {"x": x, "steps": steps}, {"out": actual}, 3)
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 3])
def test_captured_conditional_explicit_outputs_keep_generated_values(steps):
    x = torch.ones(3, 17)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_captured_conditional, (x, steps, False))
        source = bound.to_code(_captured_reduction_config(bound, 1, 16))
    actual = torch.full((3,), -1, dtype=torch.int32)
    _simulate_independent_fragment(source, {"x": x, "steps": steps}, {"out": actual}, 3)
    expected = torch.full_like(actual, 2 if steps == 1 else 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_fragment_threads_mixed_warp_reduction_cpu(threads, dtype):
    x = _fragment_warp_reduction_inputs(3, 65, dtype, "max")
    expected = _fragment_warp_reduction_reference(x, "max")
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_warp_reduction_mixed, (x, "max"))
        config = _fragment_warp_reduction_config(bound, "warp", 16)
        config.config["cute_fragment_threads"] = threads
        source = bound.to_code(config)
    assert f"block=({threads}, 1, 1)" in source
    actual = tuple(torch.full_like(value, -9) for value in expected)
    _simulate_fragment_warp_reduction(
        source, {"x": x}, {"out": actual[0], "out_index": actual[1]}, 2, threads
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512, 1024])
def test_fragment_threads_native_mixed_reduction(threads):
    x = _fragment_warp_reduction_inputs(3, 65, torch.int64, "max", "cuda")
    bound = _fragment_warp_reduction_mixed.bind((x, "max"))
    config = _fragment_warp_reduction_config(bound, "warp", 16)
    config.config["cute_fragment_threads"] = threads
    torch.testing.assert_close(
        bound.compile_config(config)(x, "max"),
        _fragment_warp_reduction_reference(x, "max"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("kind", ["min", "max"])
def test_fragment_threads_nan_order_cpu(threads, kind):
    x = torch.full((3, 128), 0.5)
    x[0, 0] = float("nan")
    x[1, 33] = float("nan")
    x[2, 127] = float("nan")
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_factored_tiled_reduction, (x, kind))
        config = _fragment_warp_reduction_config(bound, "warp", 16)
        config.config["cute_fragment_threads"] = threads
        source = bound.to_code(config)
    expected = _factored_tiled_reference(x, kind)
    actual = torch.full_like(expected, -9)
    _simulate_fragment_warp_reduction(source, {"x": x}, {"out": actual}, 2, threads)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("mode", ["serial", "cooperative"])
def test_fragment_threads_native_scan(threads, mode):
    from test.test_cute_fragment_scan_config import _computed_scan

    x = torch.arange(3 * 5 * 65, device=DEVICE).reshape(3, 5, 65).to(torch.int64)
    bound = _computed_scan.bind((x, 2, False))
    config = bound.config_spec.default_config()
    config.config.update(cute_fragment_threads=threads, cute_fragment_scan=mode)
    actual, reused = bound.compile_config(config)(x, 2, False)
    torch.testing.assert_close(actual, (x + 1).cumsum(2), rtol=0, atol=0)
    torch.testing.assert_close(reused, (x + 1) * 2, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _physical_vector_capture(x: torch.Tensor, rows: int, steps: int):
    out = torch.empty((rows,), dtype=torch.float32, device=x.device)
    for row in hl.tile(rows):
        vector = x[:]
        total = hl.zeros([row], dtype=torch.float32)
        for step in range(steps):
            total += (vector + step).sum()
        out[row] = total
    return out


@pytest.mark.parametrize(
    "width,threads,native",
    [(16, 16, True), (32, 32, True), (32, 16, False), (64, 64, False)],
)
def test_captured_vector_uses_complete_physical_lanes(width, threads, native):
    x = torch.arange(width, dtype=torch.float32)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_physical_vector_capture, (x, 3, 3))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [1]
        config.config["num_threads"] = [1, threads]
        source = bound.to_code(config)
    assert ("fragment_smem" not in source) == native
    if native:
        assert f"threads_in_group={width}" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("width,threads", [(16, 16), (32, 32), (32, 16), (64, 64)])
def test_captured_vector_physical_or_fragment_native(width, threads):
    x = torch.arange(width, dtype=torch.float32, device=DEVICE)
    expected = torch.full(
        (3,), sum((x + step).sum().item() for step in range(3)), device=DEVICE
    )
    original = x.clone()
    bound = _physical_vector_capture.bind((x, 3, 3))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [1]
    config.config["num_threads"] = [1, threads]
    actual = bound.compile_config(config)(x, 3, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options", [{"cute_fragment_reduction": "warp"}, {"cute_fragment_threads": 64}]
)
def test_physical_capture_preserves_explicit_fragment_modes(options):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_physical_vector_capture, (torch.ones(16), 3, 3))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[1], num_threads=[1, 16], **options)
        source = bound.to_code(config)
    assert "fragment_smem" in source


def test_physical_capture_vector_packets_keep_fragment_owner():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_physical_vector_capture, (torch.ones(16), 3, 3))
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[1],
            num_threads=[1, 16],
            cute_vector_widths=[2] * len(bound.config_spec.cute_vector_widths),
        )
        source = bound.to_code(config)
    assert "fragment_smem" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_tensor_index_scan(x: torch.Tensor, mode: hl.constexpr):
    out = torch.empty_like(x)
    for _ in hl.grid(1):
        row = hl.arange(x.size(0))
        column = hl.arange(x.size(1))
        if mode == "cartesian":
            loaded = hl.load(x, [row, column])
        elif mode == "masked":
            loaded = hl.load(
                x,
                [row[:, None], column[None, :]],
                extra_mask=(row[:, None] + column[None, :]) % 3 != 0,
            )
        else:
            loaded = hl.load(x, [row[:, None], column[None, :]])
        scanned = torch.cumsum(loaded + 1, dim=-1)
        if mode == "reverse_store":
            hl.store(out, [row[:, None], (x.size(1) - 1 - column)[None, :]], scanned)
        else:
            out[:, :] = scanned
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_nonconsecutive_tensor_index_scan(x: torch.Tensor):
    out = torch.empty_like(x)
    for _ in hl.grid(1):
        row = hl.arange(x.size(0))[:, None]
        column = hl.arange(x.size(2))[None, :]
        loaded = hl.load(x, [row, slice(None), column])
        out[:, :, :] = torch.cumsum(loaded + 1, dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_scalar_tensor_index_scan(x: torch.Tensor):
    out = torch.empty((x.size(1), x.size(2)), device=x.device, dtype=x.dtype)
    for _ in hl.grid(1):
        row = hl.arange(x.size(1))[:, None]
        column = hl.arange(x.size(2))[None, :]
        loaded = hl.load(x, [1, row, column])
        out[:, :] = torch.cumsum(loaded + 1, dim=-1)
    return out


def _tensor_index_source(kernel, args):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        return bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize("shape", [(3, 5), (17, 33), (1, 17), (17, 1)])
@pytest.mark.parametrize("mode", ["broadcast", "cartesian", "masked", "reverse_store"])
def test_fragment_tensor_index_generated_values(shape, mode):
    x = (torch.arange(math.prod(shape)).reshape(shape) % 13).float()
    before = x.clone()
    source = _tensor_index_source(_fragment_tensor_index_scan, (x, mode))
    actual = torch.full_like(x, -999)
    values = x
    if mode == "masked":
        mask = (torch.arange(shape[0])[:, None] + torch.arange(shape[1])) % 3 != 0
        values = torch.where(mask, values, 0)
    expected = torch.cumsum(values + 1, -1)
    if mode == "reverse_store":
        expected = expected.flip(-1)
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("scalar", [False, True])
def test_fragment_nonconsecutive_and_scalar_tensor_indices(scalar):
    x = (torch.arange(3 * 2 * 17).reshape(3, 2, 17) % 13).float()
    kernel = (
        _fragment_scalar_tensor_index_scan
        if scalar
        else _fragment_nonconsecutive_tensor_index_scan
    )
    expected = torch.cumsum((x[1] if scalar else x) + 1, -1)
    actual = torch.full_like(expected, -999)
    source = _tensor_index_source(kernel, (x,))
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("mode", ["broadcast", "cartesian", "masked", "reverse_store"])
def test_fragment_tensor_index_native(mode):
    x = (torch.arange(17 * 33, device=DEVICE).reshape(17, 33) % 13).float()
    values = x
    if mode == "masked":
        mask = (
            torch.arange(17, device=DEVICE)[:, None] + torch.arange(33, device=DEVICE)
        ) % 3 != 0
        values = torch.where(mask, x, 0)
    expected = torch.cumsum(values + 1, -1)
    if mode == "reverse_store":
        expected = expected.flip(-1)
    torch.testing.assert_close(
        _fragment_tensor_index_scan(x, mode), expected, rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_loaded_tensor_index_scan(x, rows, columns):
    out = torch.empty((rows.numel(), columns.numel()), device=x.device, dtype=x.dtype)
    for _ in hl.grid(1):
        row = rows[:][:, None]
        column = columns[:][None, :]
        loaded = hl.load(x, [row, column])
        out[:, :] = torch.cumsum(loaded + 1, dim=-1)
    return out


@pytest.mark.parametrize("transpose", [False, True])
def test_fragment_loaded_tensor_index_bounds_and_strides(transpose):
    storage = (torch.arange(7 * 11).reshape(7, 11) % 13).float()
    x = storage.t() if transpose else storage
    rows = torch.tensor([x.size(0) - 1, 1, 0, 1, x.size(0)], dtype=torch.int64)
    columns = torch.tensor([2, 0, x.size(1) - 1], dtype=torch.int64)
    source = _tensor_index_source(
        _fragment_loaded_tensor_index_scan, (x, rows, columns)
    )
    expected = torch.cat((x[rows[:-1]][:, columns], torch.zeros(1, columns.numel())))
    expected = torch.cumsum(expected + 1, -1)
    actual = torch.full_like(expected, -999)
    # The generated pointer arithmetic sees underlying storage, with the actual
    # strided input's compile-time strides retained in the generated code.
    _simulate_independent_fragment(
        source, {"x": storage, "rows": rows, "columns": columns}, {"out": actual}, 1
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scalar", [False, True])
@skipUnlessBackends(["cute"])
def test_fragment_tensor_index_native_rank3(scalar):
    x = (torch.arange(3 * 2 * 17, device=DEVICE).reshape(3, 2, 17) % 13).float()
    kernel = (
        _fragment_scalar_tensor_index_scan
        if scalar
        else _fragment_nonconsecutive_tensor_index_scan
    )
    expected = torch.cumsum((x[1] if scalar else x) + 1, -1)
    torch.testing.assert_close(kernel(x), expected, rtol=0, atol=0)


def test_fragment_tensor_index_mapping_rejects_unowned_axes():
    from helion._compiler.cute.fragment_indexing import memory_index_coordinates

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_tensor_index_scan, (torch.ones(3, 5), "broadcast"))
        with bound.env:
            index = torch.ones(2, 3, dtype=torch.int64)
            for shape in ((4,), (1, 2)):
                with pytest.raises(
                    exc.BackendUnsupported, match="fragment tensor index"
                ):
                    memory_index_coordinates(bound.env, [index], {0: (2, 3)}, shape)
            singleton = torch.ones(1, dtype=torch.int64)
            with pytest.raises(exc.BackendUnsupported, match="fragment tensor index"):
                memory_index_coordinates(bound.env, [singleton], {0: (1,)}, ())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_conditional_cross_branch_capture(x: torch.Tensor, flag: int):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        values = torch.cumsum(x[row, :] + 1, dim=-1)
        if flag > 0:
            branch_temporary = values * 2
            values = values + branch_temporary
        out[row, :] = values
    return out


@pytest.mark.parametrize("flag", [-1, 1])
def test_fragment_conditional_cross_branch_capture_preserves_branch_effects(flag):
    x = (torch.arange(3 * 17).reshape(3, 17) % 7).float()
    source = _tensor_index_source(_fragment_conditional_cross_branch_capture, (x, flag))
    actual = torch.full_like(x, -999)
    _simulate_independent_fragment(source, {"x": x, "flag": flag}, {"out": actual}, 3)
    expected = torch.cumsum(x + 1, -1) * (3 if flag > 0 else 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_conditional_capture_effects(x: torch.Tensor, flag: int):
    out = torch.empty_like(x)
    effect = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        left = torch.cumsum(x[row, :] + 1, dim=-1)
        right = left + 1
        if flag > 0:
            left = left + right
            effect[row, :] = left * 2
        else:
            right = right - left
            effect[row, :] = right * 3
        out[row, :] = left + right
    return out, effect


@pytest.mark.parametrize("flag", [-1, 1])
def test_fragment_conditional_capture_preserves_all_effects(flag):
    x = (torch.arange(3 * 17).reshape(3, 17) % 7).float()
    source = _tensor_index_source(_fragment_conditional_capture_effects, (x, flag))
    outputs = {name: torch.full_like(x, -999) for name in ("out", "effect")}
    _simulate_independent_fragment(source, {"x": x, "flag": flag}, outputs, 3)
    left = torch.cumsum(x + 1, -1)
    right = left + 1
    if flag > 0:
        left = left + right
        effect = left * 2
    else:
        right = right - left
        effect = right * 3
    torch.testing.assert_close(outputs["out"], left + right, rtol=0, atol=0)
    torch.testing.assert_close(outputs["effect"], effect, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["missing", "conflicting"])
def test_fragment_conditional_outer_capture_proof_rejects_invalid_ir(kind):
    original = FragmentCompiler.conditional

    def corrupt_capture(compiler, node, values):
        info = compiler.graphs[node.args[1]]
        saved_outputs, saved_names, saved_args = (
            info.branches_outputs,
            info.else_arg_names,
            node.args,
        )
        if kind == "missing":
            info.branches_outputs = [(0, "unbound_outer_value")]
        else:
            info.else_arg_names = list(info.if_arg_names)
            # A different outer node under the same lexical name is ambiguous.
            conflicting = next(n for n in node.graph.nodes if n not in node.args[3])
            node.args = (*node.args[:4], [conflicting])
        try:
            return original(compiler, node, values)
        finally:
            info.branches_outputs, info.else_arg_names, node.args = (
                saved_outputs,
                saved_names,
                saved_args,
            )

    with (
        patch.object(FragmentCompiler, "conditional", corrupt_capture),
        pytest.raises(
            exc.InvalidConfig,
            match="captured unchanged output|conflicting outer captures",
        ),
    ):
        _tensor_index_source(
            _fragment_conditional_cross_branch_capture, (torch.ones(3, 17), 1)
        )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("flag", [-1, 1])
def test_fragment_conditional_cross_branch_capture_native(flag):
    x = (torch.arange(3 * 17, device=DEVICE).reshape(3, 17) % 7).float()
    expected = torch.cumsum(x + 1, -1) * (3 if flag > 0 else 1)
    torch.testing.assert_close(
        _fragment_conditional_cross_branch_capture(x, flag), expected, rtol=0, atol=0
    )


_FRAGMENT_FIXED_CHUNK = 8


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_host_fixed_chunk_scan(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        for column in hl.tile(x.size(1), block_size=_FRAGMENT_FIXED_CHUNK):
            out[row, column] = torch.cumsum(x[row, column], dim=-1)
    return out


def test_fragment_unresolved_fixed_extent_preserves_ordinary_owner():
    from helion._compiler import generate_ast as generate_module

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_host_fixed_chunk_scan, (torch.ones(3, 17),))
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
        with patch.object(
            generate_module.GenerateAST,
            "_try_codegen_computed_fragment_root",
            return_value=False,
        ):
            ordinary = bound.to_code(config)
    assert "fragment_smem" not in source
    assert ast.dump(ast.parse(source), include_attributes=False) == ast.dump(
        ast.parse(ordinary), include_attributes=False
    )


@pytest.mark.parametrize("option", ["cute_fragment_scan", "cute_fragment_threads"])
def test_fragment_unresolved_fixed_extent_rejects_explicit_owner(option):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_host_fixed_computed_scan, (torch.ones(3, 17),))
        config = bound.config_spec.default_config()
        config.config[option] = "cooperative" if option == "cute_fragment_scan" else 32
        with pytest.raises(exc.InvalidConfig, match="computed fragment root"):
            bound.to_code(config)


@skipUnlessBackends(["cute"])
def test_fragment_host_fixed_chunk_native():
    x = (torch.arange(3 * 17, device=DEVICE).reshape(3, 17) % 7).float()
    expected = torch.cat(
        [
            torch.cumsum(x[:, start : start + _FRAGMENT_FIXED_CHUNK], -1)
            for start in range(0, 17, _FRAGMENT_FIXED_CHUNK)
        ],
        -1,
    )
    torch.testing.assert_close(
        _fragment_host_fixed_chunk_scan(x), expected, rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_host_fixed_computed_scan(x: torch.Tensor):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        for column in hl.tile(x.size(1), block_size=_FRAGMENT_FIXED_CHUNK):
            out[row, column] = torch.cumsum(x[row, column] + 1, dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _free_iota_vector_reduction(x, chunk: hl.constexpr, kind: hl.constexpr):
    parts = (x.size(1) + chunk - 1) // chunk
    out = torch.empty((x.size(0), parts), dtype=torch.int32, device=x.device)
    for row, part in hl.grid((x.size(0), parts)):
        columns = part * chunk + hl.arange(chunk)
        valid = (columns < x.size(1)) & (columns % 5 != row % 5)
        value = hl.load(x, [row, columns], extra_mask=valid)
        if kind == "count":
            reduced = ((value > 3) & valid).to(torch.int32).sum(dtype=torch.int32)
        elif kind == "sum":
            reduced = torch.where(valid, value + 2, 0).sum(dtype=torch.int32)
        else:
            reduced = torch.where(valid, value + 2, -1000).amax()
        out[row, part] = reduced
    return out


def _free_iota_reference(x, chunk, kind):
    result = torch.empty(
        (x.size(0), (x.size(1) + chunk - 1) // chunk),
        dtype=torch.int32,
        device=x.device,
    )
    for row in range(x.size(0)):
        for part, start in enumerate(range(0, x.size(1), chunk)):
            columns = torch.arange(
                start, min(start + chunk, x.size(1)), device=x.device
            )
            values = x[row, columns][columns % 5 != row % 5]
            result[row, part] = (
                (values > 3).sum()
                if kind == "count"
                else (values + 2).sum()
                if kind == "sum"
                else (values + 2).amax()
                if values.numel()
                else -1000
            )
    return result


def _free_iota_config(bound, reduction):
    config = bound.config_spec.default_config()
    config.config["reduction_loops"] = [reduction]
    return config


@pytest.mark.parametrize("chunk,reduction", [(16, 8), (64, 16), (128, 32)])
@pytest.mark.parametrize("kind", ["count", "sum", "max"])
def test_free_iota_reduction_complete_producer_values(chunk, reduction, kind):
    # Strided source, independent row/partition coordinates, masked tails and a
    # nonzero pointwise offset distinguish full-vector semantics from repeating
    # one masked scalar. The original looped lowering did precisely the latter.
    x = (torch.arange(3 * (4 * chunk + 6), dtype=torch.int32) % 13).reshape(3, -1)[
        :, ::2
    ]
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_free_iota_vector_reduction, (x, chunk, kind))
        source = bound.to_code(_free_iota_config(bound, reduction))
    assert "arange_lane" not in source
    assert "fragment_reduce_tile" in source
    expected = _free_iota_reference(x, chunk, kind)
    actual = torch.full_like(expected, -999)
    before = x.clone()
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, actual.numel())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    coverage = _generated_output_address_coverage(source, "out", actual.numel(), 128)
    assert coverage == dict.fromkeys(range(actual.numel()), 1)


def test_free_iota_large_default_reduction_values():
    x = (torch.arange(2 * 8195).reshape(2, 8195) % 7).int()
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(_free_iota_vector_reduction, (x, 8192, "count"))
        source = bound.to_code(bound.config_spec.default_config())
    expected = _free_iota_reference(x, 8192, "count")
    actual = torch.full_like(expected, -999)
    _simulate_independent_fragment(source, {"x": x}, {"out": actual}, actual.numel())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "arange_lane" not in source


def test_free_iota_existing_complete_axis_keeps_native_program():
    # The persistent 64-element axis already owns the ordinary iota coordinate.
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _free_iota_vector_reduction, (torch.ones(3, 129).int(), 64, "sum")
        )
        config = _free_iota_config(bound, None)
        actual = bound.to_code(config)
        with patch(
            "helion._compiler.cute.computed_fragment.free_iota_reductions",
            return_value={},
        ):
            original = bound.to_code(config)
    assert "fragment_smem" not in actual
    assert actual == original


def test_free_iota_declines_incompatible_collective_config():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _free_iota_vector_reduction, (torch.ones(3, 129).int(), 64, "count")
        )
        config = _free_iota_config(bound, 16)
        config.config["cute_collective_mma"] = True
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(config)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "chunk,reduction,kind",
    [(16, 8, "count"), (64, 16, "sum"), (128, 32, "max"), (8192, 4096, "count")],
)
def test_free_iota_complete_vector_native(chunk, reduction, kind):
    x = (torch.arange(3 * (2 * chunk + 3), device=DEVICE).reshape(3, -1) % 13).int()
    before = x.clone()
    bound = _free_iota_vector_reduction.bind((x, chunk, kind))
    actual = bound.compile_config(_free_iota_config(bound, reduction))(x, chunk, kind)
    torch.testing.assert_close(
        actual, _free_iota_reference(x, chunk, kind), rtol=0, atol=0
    )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_int32_negated_extrema(x: torch.Tensor):
    low = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    high = torch.empty_like(low)
    for row in hl.tile(x.size(0), block_size=1):
        # The scan establishes a complete-fragment producer. Integer overflow is
        # intentional: reductions must preserve the logical Int32 value bits.
        values = -hl.cumsum(x[row, :] + 1, dim=-1)
        low[row] = values.amin(-1)
        high[row] = values.amax(-1)
    return low, high


def _fragment_int32_extrema_inputs(columns, device="cpu"):
    bounds = torch.iinfo(torch.int32)
    prefix = torch.tensor(
        [bounds.min, bounds.max, -1, 0, 1, bounds.min + 1, bounds.max - 1],
        dtype=torch.int32,
        device=device,
    )
    prefix = prefix.repeat((columns + 6) // 7)[:columns]
    positive = (torch.arange(columns, device=device) % 13 + 1).int()
    prefix = torch.stack((prefix, positive, -positive))
    x = prefix.clone()
    x[:, 1:] = prefix[:, 1:] - prefix[:, :-1]
    return x - 1


def _fragment_int32_extrema_reference(x):
    values = -(x + 1).cumsum(-1, dtype=torch.int32)
    return values.amin(-1), values.amax(-1)


@pytest.mark.parametrize("mode", ["serial", "warp"])
@pytest.mark.parametrize("columns", [17, 65])
def test_fragment_int32_negated_extrema_generated_boundaries(mode, columns):
    x = _fragment_int32_extrema_inputs(columns)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_int32_negated_extrema, (x,))
        config = bound.config_spec.default_config()
        config.config["cute_fragment_reduction"] = mode
        code = bound.to_code(config)
    expected = _fragment_int32_extrema_reference(x)
    actual = tuple(torch.full_like(value, -99) for value in expected)
    _simulate_fragment_warp_reduction(
        code, {"x": x}, {"low": actual[0], "high": actual[1]}, 3
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["min", "max"])
def test_int32_extrema_signed_order_boundary_equivalence(kind):
    import numpy as np

    from helion._compiler.cute.backend import CuteBackend

    limits = torch.iinfo(torch.int32)
    values = [
        limits.min,
        limits.min + 1,
        -65537,
        -1,
        0,
        1,
        65537,
        limits.max - 1,
        limits.max,
    ]
    # Include negation overflow, where -INT_MIN is still INT_MIN.
    values.extend(int(value) for value in -torch.tensor(values, dtype=torch.int32))
    expression = CuteBackend().reduction_combine_expr(kind, "a", "b", torch.int32)
    scope = {
        "cutlass": SimpleNamespace(Int32=np.int32, Uint32=np.uint32),
        "cute": SimpleNamespace(math=SimpleNamespace(min=np.minimum, max=np.maximum)),
    }
    reference = min if kind == "min" else max
    for left, right in itertools.product(values, repeat=2):
        scope.update(a=np.int32(left), b=np.int32(right))
        assert int(eval(expression, scope)) == reference(left, right)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("mode", ["serial", "warp"])
@pytest.mark.parametrize("columns", [17, 65])
def test_fragment_int32_negated_extrema_native(mode, columns):
    x = _fragment_int32_extrema_inputs(columns, "cuda")
    before = x.clone()
    bound = _fragment_int32_negated_extrema.bind((x,))
    config = bound.config_spec.default_config()
    config.config["cute_fragment_reduction"] = mode
    actual = bound.compile_config(config)(x)
    torch.testing.assert_close(
        actual, _fragment_int32_extrema_reference(x), rtol=0, atol=0
    )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _fragment_dynamic_host_strides(x, out):
    hl.specialize(x.size(1))
    for row in hl.tile(x.size(0), block_size=1):
        out[row, :] = torch.cumsum(x[row, :] + 1, dim=-1)
    return out


def test_fragment_host_stride_arguments_preserve_views_and_rebinds():
    inputs = torch.arange(3 * 18).reshape(3, 18).float()[:, 1::2]
    outputs = torch.full((3, 27), -999.0)[:, 2::3]
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_dynamic_host_strides, (inputs, outputs))
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
    tree = ast.parse(source)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    parameters = {arg.arg for arg in device.args.args}
    assert {"x_stride_0", "x_stride_1", "out_stride_0", "out_stride_1"} <= parameters
    for input_step, output_step in ((2, 3), (3, 5)):
        storage = torch.arange(3 * (9 * input_step + 1)).reshape(3, -1).float()
        x = storage[:, 1::input_step]
        target = torch.full((3, 9 * output_step + 2), -999.0)
        out = target[:, 2::output_step]
        kwargs = {"x": x, "out": out}
        for name, tensor in (("x", x), ("out", out)):
            for dim in range(2):
                kwargs[f"{name}_stride_{dim}"] = tensor.stride(dim)
                kwargs[f"{name}_size_{dim}"] = tensor.size(dim)
        _simulate_independent_fragment(source, kwargs, {}, 3)
        torch.testing.assert_close(out, (x + 1).cumsum(-1), rtol=0, atol=0)
        untouched = target.clone()
        untouched[:, 2::output_step] = -999
        assert torch.all(untouched == -999)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _free_iota_two_destinations(
    x, left, right, chunk: hl.constexpr, same_target: hl.constexpr
):
    for row, part in hl.grid([x.size(0), left.size(1)]):
        index = part * chunk + hl.arange(chunk)
        loaded = hl.load(x, [row, index], extra_mask=index < x.size(1))
        total = (loaded + 1).sum()
        left[row, part] = total
        if same_target:
            left[row, part] = total + 7
        else:
            right[row, part] = total + 7
    return left, right


@pytest.mark.parametrize("mode", ["disjoint", "same_target", "overlapping_views"])
def test_free_iota_store_ownership_keeps_alias_guards(mode):
    x = torch.arange(2 * 129).reshape(2, 129).float()
    storage = torch.full((2, 4), -999.0)
    left = storage[:, :3]
    right = (
        torch.full((2, 3), -999.0)
        if mode == "disjoint"
        else left
        if mode == "same_target"
        else storage[:, 1:]
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _free_iota_two_destinations,
            (x, left, right, 64, mode == "same_target"),
        )
        config = _free_iota_config(bound, 16)
        if mode == "overlapping_views":
            with pytest.raises(exc.BackendUnsupported, match="disjoint storage"):
                bound.to_code(config)
            return
        source = bound.to_code(config)
    assert "fragment_reduce_tile" in source
    _simulate_independent_fragment(
        source, {"x": x, "left": left, "right": right}, {}, 6
    )
    expected = torch.stack(
        [
            torch.stack(
                [x[row, start : start + 64].sum() + 64 for start in (0, 64, 128)]
            )
            for row in range(2)
        ]
    )
    torch.testing.assert_close(right, expected + 7, rtol=0, atol=0)
    torch.testing.assert_close(
        left, expected + (7 if mode == "same_target" else 0), rtol=0, atol=0
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("width", [9, 17])
def test_fragment_host_stride_native_rebind(width):
    compiled = None
    for input_step, output_step in ((2, 3), (3, 5)):
        storage = torch.arange(
            3 * (width * input_step + 1), device=DEVICE, dtype=torch.float32
        ).reshape(3, -1)
        x = storage[:, 1::input_step]
        target = torch.full((3, width * output_step + 2), -999.0, device=DEVICE)
        out = target[:, 2::output_step]
        before = x.clone()
        if compiled is None:
            bound = _fragment_dynamic_host_strides.bind((x, out))
            compiled = bound.compile_config(bound.config_spec.default_config())
        actual = compiled(x, out)
        assert actual is out
        torch.testing.assert_close(actual, (x + 1).cumsum(-1), rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        untouched = target.clone()
        untouched[:, 2::output_step] = -999
        assert torch.all(untouched == -999)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_cached_transcendental(x: torch.Tensor):
    out = torch.empty((x.size(0), 2), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        value = torch.tanh(x[row, :])
        out[row, 0] = value.sum(-1)
        out[row, 1] = hl.cumsum(value, dim=-1).amax(-1)
    return out


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_uncached_cheap(x: torch.Tensor):
    out = torch.empty((x.size(0), 2), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        value = x[row, :] + 1
        out[row, 0] = value.sum(-1)
        out[row, 1] = hl.cumsum(value, dim=-1).amax(-1)
    return out


def test_fragment_producer_cache_cost_and_unsupported_configs():
    x = torch.arange(3 * 65, dtype=torch.float32).reshape(3, 65) * 0.01
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_uncached_cheap, (x,))
        assert not bound.config_spec.cute_fragment_producer_cache_root_ids
        config = bound.config_spec.default_config()
        original = bound.to_code(config)
        config.config["cute_fragment_producer_cache"] = False
        assert bound.to_code(config) == original
        for value in (True, 1, "shared", None):
            config.config["cute_fragment_producer_cache"] = value
            with pytest.raises(exc.InvalidConfig):
                bound.to_code(config)


@pytest.mark.parametrize("columns", [17, 65, 128])
def test_fragment_producer_cache_generated_values_and_roundtrip(columns):
    x = torch.linspace(-3, 3, 3 * columns).reshape(3, columns)
    before = x.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_cached_transcendental, (x,))
        assert bound.config_spec.cute_fragment_producer_cache_root_ids
        config = bound.config_spec.default_config()
        config.config["block_sizes"][:] = [2] * len(config.block_sizes)
        original = bound.to_code(config)
        config.config["cute_fragment_producer_cache"] = False
        assert bound.to_code(config) == original
        config.config["cute_fragment_producer_cache"] = True
        generation = bound.config_spec.create_config_generation()
        flat, effective = generation.strict_config_pair(config)
        assert generation.strict_config_pair(effective)[0] == flat
        cached = bound.to_code(effective)
        assert cached.count("cute.math.tanh(") < original.count("cute.math.tanh(")
        for conflicting in (
            "cute_fragment_register_loads",
            "cute_fragment_warp_results",
        ):
            conflict = helion.Config.from_dict(effective.config | {conflicting: True})
            for repair in (False, True):
                with pytest.raises(exc.InvalidConfig, match="lane-private"):
                    bound.config_spec.normalize(
                        helion.Config.from_dict(conflict.config.copy()),
                        _fix_invalid=repair,
                    )
    actual = []
    for source in (original, cached):
        output = torch.full((3, 2), float("nan"))
        _simulate_independent_fragment(source, {"x": x}, {"out": output}, 2)
        actual.append(output)
    torch.testing.assert_close(actual[0], actual[1], rtol=0, atol=0)
    reference = torch.stack((x.tanh().sum(-1), x.tanh().cumsum(-1).amax(-1)), -1)
    torch.testing.assert_close(actual[1], reference)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_fragment_producer_cache_effect_and_control_boundaries():
    from helion._compiler.cute.producer_cache import producer_cache_candidates
    from helion.language import _tracing_ops
    from helion.language import atomic_ops

    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(4, 16)
    value = graph.call_function(torch.ops.aten.tanh.default, (source,))
    value.meta["val"] = torch.empty(4, 16)
    for operation in (torch.ops.aten.sum.dim_IntList, torch.ops.aten.amax.default):
        user = graph.call_function(operation, (value, [-1]))
        user.meta["val"] = torch.empty(4)
    graph.output(value)
    assert producer_cache_candidates(graph) == frozenset((value,))
    for operation in (
        _tracing_ops._if,
        _tracing_ops._while_loop,
        atomic_ops.atomic_add,
    ):
        effect = graph.call_function(operation, ())
        assert not producer_cache_candidates(graph)
        graph.erase_node(effect)
    random = graph.call_function(torch.ops.aten.rand_like.default, (source,))
    random.meta["val"] = torch.empty(4, 16)
    assert random not in producer_cache_candidates(graph)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("columns", [17, 128])
def test_fragment_producer_cache_native(columns):
    x = torch.randn((3, columns), device=DEVICE)
    bound = _fragment_cached_transcendental.bind((x,))
    config = bound.config_spec.default_config()
    config.config["cute_fragment_producer_cache"] = True
    actual = bound.compile_config(config)(x)
    torch.testing.assert_close(
        actual, torch.stack((x.tanh().sum(-1), x.tanh().cumsum(-1).amax(-1)), -1)
    )


@pytest.mark.parametrize("strategy_name", ["FROM_RANDOM", "FROM_BEST_AVAILABLE"])
def test_fragment_producer_cache_preserves_prefix_and_rng(strategy_name):
    import random

    from test.test_compiler_coverage import make_search

    from helion.autotuner.pattern_search import InitialPopulationStrategy

    x = torch.ones((3, 65))
    # Later independent coverage groups are tested in their own appended order.
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch(
            "helion._compiler.autotuner_heuristics.register_fragment_warp_scan_coverage"
        ),
        patch(
            "helion._compiler.autotuner_heuristics.register_fragment_published_scalars_coverage"
        ),
        patch(
            "helion._compiler.autotuner_heuristics.register_fragment_packet_loads_coverage"
        ),
    ):
        with patch(
            "helion._compiler.autotuner_heuristics.register_fragment_producer_cache_coverage"
        ):
            previous = _cpu_bind(
                helion.kernel(
                    _fragment_cached_transcendental.fn,
                    backend="cute",
                    static_shapes=True,
                ),
                (x,),
            )
        current = _cpu_bind(
            helion.kernel(
                _fragment_cached_transcendental.fn, backend="cute", static_shapes=True
            ),
            (x,),
        )
    strategy = InitialPopulationStrategy[strategy_name]
    old = make_search(previous.config_spec, count=20, strategy=strategy)
    new = make_search(current.config_spec, count=20, strategy=strategy)
    for seed in (73, 741, 2031):
        random.seed(seed)
        prior = old._generate_initial_population_flat()
        state = random.getstate()
        random.seed(seed)
        current_rows = new._generate_initial_population_flat()
        assert random.getstate() == state
        expected = [old.config_gen.unflatten(row) for row in prior]
        actual = [new.config_gen.unflatten(row) for row in current_rows]
        assert actual[: len(expected)] == expected
        assert len(actual) == len(expected) + 1
        assert actual[-1]["cute_fragment_producer_cache"] is True
    assert previous.config_spec.default_config() == current.config_spec.default_config()
    assert (
        previous.config_spec.compiler_seed_configs
        == current.config_spec.compiler_seed_configs
    )
    for overrides, disabled in (
        ({"cute_fragment_producer_cache": False}, False),
        ({}, True),
    ):
        search = make_search(
            current.config_spec, count=20, overrides=overrides, disabled=disabled
        )
        assert all(
            not search.config_gen.unflatten(row).get(
                "cute_fragment_producer_cache", False
            )
            for row in search._generate_initial_population_flat()
        )


def test_fragment_producer_cache_capacity_is_not_bypassed():
    x = torch.ones((3, 65))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_cached_transcendental, (x,))
        config = bound.config_spec.default_config()
        config.config["cute_fragment_producer_cache"] = True
        for capacity in (0, 16):
            with (
                patch(
                    "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
                    return_value=capacity,
                ),
                pytest.raises(exc.InvalidConfig, match="shared"),
            ):
                bound.to_code(config)


def test_fragment_producer_cache_preserves_special_value_bits():
    x = torch.zeros((3, 17))
    x[0, 0] = float("nan")
    x[1, :2] = torch.tensor([float("inf"), -float("inf")])
    x[2, ::2] = -0.0
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_cached_transcendental, (x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"][:] = [2] * len(config.block_sizes)
        outputs = []
        for enabled in (False, True):
            config.config["cute_fragment_producer_cache"] = enabled
            source = bound.to_code(config)
            actual = torch.empty((3, 2))
            _simulate_independent_fragment(source, {"x": x}, {"out": actual}, 2)
            outputs.append(actual)
    torch.testing.assert_close(
        outputs[0].view(torch.int32), outputs[1].view(torch.int32)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
@pytest.mark.parametrize(
    "axis,reverse,columns,threads",
    [(2, False, 65, 32), (2, True, 250, 128), (1, True, 33, 512), (2, True, 513, 1024)],
)
def test_fragment_warp_scan_cpu_typed_axes_tail_and_input_reuse(
    dtype, axis, reverse, columns, threads
):
    from test.test_cute_fragment_scan_config import _computed_scan

    shape = (3, columns, 5) if axis == 1 else (3, 5, columns)
    x = (torch.arange(math.prod(shape)).reshape(shape) % 7 - 3).to(dtype)
    if dtype == torch.int64:
        x += 2**40
    before = x.clone()
    expected = (
        torch.flip(torch.cumsum(torch.flip(x + 1, (axis,)), axis, dtype=dtype), (axis,))
        if reverse
        else torch.cumsum(x + 1, axis, dtype=dtype)
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_computed_scan, (x, axis, reverse))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [2],
                "cute_fragment_threads": threads,
                "cute_fragment_warp_scan": True,
            }
        )
        source = bound.to_code(config)
    out, reused = torch.full_like(x, -9), torch.full_like(x, -9)
    state = _simulate_fragment_warp_reduction(
        source,
        {"x": x},
        {"out": out, "reused": reused},
        2,
        threads,
        allow_lane_stores=True,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(reused, (x + 1) * 2, rtol=0, atol=0)
    assert torch.equal(x, before)
    assert state.exchanges > 0


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_scan_special(x: torch.Tensor, reverse: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = hl.cumsum(x[row, :] * 1, dim=-1, reverse=reverse)
    return out


def _warp_scan_tree_reference(x, reverse):
    """Synchronous tensor tree, independent of emitted scalar addresses."""
    import numpy as np

    array = x.numpy().copy()
    if reverse:
        array = array[:, ::-1].copy()
    width = array.shape[-1]
    chunks = (width + 31) // 32
    data = np.zeros((len(array), chunks * 32), dtype=array.dtype)
    data[:, :width] = array
    data = data.reshape(len(array), chunks, 32)

    def scan(values):
        for distance in (1, 2, 4, 8, 16):
            if distance < values.shape[-1]:
                old = values.copy()
                values[..., distance:] = np.add(
                    old[..., :-distance], old[..., distance:], dtype=values.dtype
                )
        return values

    with np.errstate(over="ignore", invalid="ignore"):
        data = scan(data)
        totals = data[..., 31].copy()
        totals[:, -1] = data[:, -1, (width - 1) % 32]
        totals = scan(totals)
        for chunk in range(1, chunks):
            data[:, chunk, :] = np.add(
                totals[:, chunk - 1, None], data[:, chunk, :], dtype=data.dtype
            )
    result = data.reshape(len(array), -1)[:, :width]
    return torch.from_numpy((result[:, ::-1] if reverse else result).copy())


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
@pytest.mark.parametrize(
    "width,reverse", [(1, True), (33, False), (65, True), (513, False), (1024, True)]
)
def test_fragment_warp_scan_cpu_exact_tree_boundaries(dtype, width, reverse):
    x = (torch.arange(3 * width).reshape(3, width) % 7 - 3).to(dtype)
    if dtype.is_floating_point:
        x[0] = -0.0
        x[1, width // 2] = float("nan")
        if width > 32:
            x[2, 0] = 16777216
            x[2, 31] = -16777216
    else:
        x[:, width // 2] = torch.iinfo(dtype).max
    expected = _warp_scan_tree_reference(x, reverse)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_warp_scan_special, (x, reverse))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [2],
                "cute_fragment_threads": 128,
                "cute_fragment_warp_scan": True,
            }
        )
        source = bound.to_code(config)
    out = torch.full_like(x, -9)
    _simulate_fragment_warp_reduction(
        source, {"x": x}, {"out": out}, 2, allow_lane_stores=True
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    if dtype.is_floating_point:
        assert torch.signbit(out[0]).all()


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_fragment_warp_scan_sdk_shuffles_preserve_32_and_64_bit_types():
    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    for dtype in (cutlass.Float32, cutlass.Float64, cutlass.Int32, cutlass.Int64):
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("scan_shuffle", ([dtype.mlir_type], []))
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    value = dtype(block.arguments[0])
                    up = cute.arch.shuffle_sync_up(
                        value, offset=16, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    indexed = cute.arch.shuffle_sync(
                        up, 31, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    assert up.dtype == indexed.dtype == dtype
                    func.ReturnOp([])
            text = str(module)
            assert "nvvm.shfl.sync" in text
            assert "up" in text and "idx" in text
            assert module.operation.verify()


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
@pytest.mark.parametrize(
    "reverse,threads", [(False, 32), (True, 128), (True, 512), (True, 1024)]
)
def test_fragment_warp_scan_native_tails_typed_reuse(dtype, reverse, threads):
    from test.test_cute_fragment_scan_config import _computed_scan

    x = (torch.arange(3 * 5 * 65, device=DEVICE).reshape(3, 5, 65) % 7 - 3).to(dtype)
    if dtype == torch.int64:
        x += 2**40
    before = x.clone()
    bound = _computed_scan.bind((x, 2, reverse))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "block_sizes": [2],
            "cute_fragment_threads": threads,
            "cute_fragment_warp_scan": True,
        }
    )
    actual, reused = bound.compile_config(config)(x, 2, reverse)
    values = x + 1
    expected = (
        values.flip((2,)).cumsum(2, dtype=dtype).flip((2,))
        if reverse
        else values.cumsum(2, dtype=dtype)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(reused, values * 2, rtol=0, atol=0)
    assert torch.equal(before, x)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_scan_exclusive(x: torch.Tensor, reverse: hl.constexpr):
    out = torch.empty_like(x)
    width = hl.specialize(x.size(1))
    for row in hl.tile(x.size(0)):
        index = hl.arange(width)
        if reverse:
            source = index + 1
        else:
            source = index - 1
        values = hl.load(
            x,
            [row.index[:, None], source[None, :]],
            extra_mask=((source >= 0) & (source < width))[None, :],
        )
        out[row, :] = hl.cumsum(values * 1, dim=-1, reverse=reverse)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_scan_tiled(x: torch.Tensor, reverse: hl.constexpr):
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape, block_size=[2, 32]):
        out[row, column] = hl.cumsum(x[row, column] + 1, dim=-1, reverse=reverse)
    return out


@pytest.mark.parametrize(
    "kernel", [_fragment_warp_scan_exclusive, _fragment_warp_scan_tiled]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_fragment_warp_scan_cpu_exclusive_producer_and_tiled_extent(kernel, reverse):
    x = (torch.arange(3 * 65).reshape(3, 65) % 7 - 3).float()
    if kernel is _fragment_warp_scan_exclusive:
        values = torch.zeros_like(x)
        if reverse:
            values[:, :-1] = x[:, 1:]
        else:
            values[:, 1:] = x[:, :-1]
        expected = (
            values.flip((1,)).cumsum(1).flip((1,)) if reverse else values.cumsum(1)
        )
        blocks = 2
    else:
        expected = torch.empty_like(x)
        for start in range(0, 65, 32):
            values = x[:, start : start + 32] + 1
            expected[:, start : start + 32] = (
                values.flip((1,)).cumsum(1).flip((1,)) if reverse else values.cumsum(1)
            )
        blocks = 6
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, (x, reverse))
        config = bound.config_spec.default_config()
        if kernel is _fragment_warp_scan_exclusive:
            config.config["block_sizes"] = [2]
        config.config["cute_fragment_warp_scan"] = True
        code = bound.to_code(config)
    out = torch.full_like(x, -9)
    _simulate_fragment_warp_reduction(
        code, {"x": x}, {"out": out}, blocks, allow_lane_stores=True
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_fragment_warp_scan_native_boundary_and_signed_zero(dtype):
    x = (torch.arange(3 * 1024).reshape(3, 1024) % 7 - 3).to(dtype)
    if dtype.is_floating_point:
        x[0] = -0.0
        x[1, 31] = float("nan")
    else:
        x[:, 31] = torch.iinfo(dtype).max
    expected = _warp_scan_tree_reference(x, True).cuda()
    x = x.cuda()
    before = x.clone()
    bound = _fragment_warp_scan_special.bind((x, True))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "block_sizes": [2],
            "cute_fragment_threads": 128,
            "cute_fragment_warp_scan": True,
        }
    )
    out = bound.compile_config(config)(x, True)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)
    if dtype.is_floating_point:
        assert torch.signbit(out[0]).all()


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "kernel", [_fragment_warp_scan_exclusive, _fragment_warp_scan_tiled]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_fragment_warp_scan_native_exclusive_and_tiled(kernel, reverse):
    x = (torch.arange(3 * 65, device=DEVICE).reshape(3, 65) % 7 - 3).float()
    before = x.clone()
    bound = kernel.bind((x, reverse))
    config = bound.config_spec.default_config()
    if kernel is _fragment_warp_scan_exclusive:
        config.config["block_sizes"] = [2]
        values = torch.zeros_like(x)
        if reverse:
            values[:, :-1] = x[:, 1:]
        else:
            values[:, 1:] = x[:, :-1]
        expected = (
            values.flip((1,)).cumsum(1).flip((1,)) if reverse else values.cumsum(1)
        )
    else:
        expected = torch.empty_like(x)
        for start in range(0, 65, 32):
            values = x[:, start : start + 32] + 1
            expected[:, start : start + 32] = (
                values.flip((1,)).cumsum(1).flip((1,)) if reverse else values.cumsum(1)
            )
    config.config["cute_fragment_warp_scan"] = True
    actual = bound.compile_config(config)(x, reverse)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(before, x)


@pytest.mark.parametrize("reverse", [False, True])
def test_fragment_warp_scan_cpu_empty_logical_extent(reverse):
    # Empty frontend tensors have zero physical capacity and are rejected.
    # Exercise the emitter's distinct zero-logical-extent boundary with a
    # nonzero physical tile: every slot must be initialized and reverse source
    # positions must never be read. Only the bound expression is substituted.
    x = torch.full((3, 65), float("nan"))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_warp_scan_special, (x, reverse))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"block_sizes": [2], "cute_fragment_warp_scan": True}
        )
        with patch.object(FragmentCompiler, "logical_axis_extent", return_value="0"):
            code = bound.to_code(config)
    out = torch.full_like(x, float("nan"))
    _simulate_fragment_warp_reduction(
        code, {"x": x}, {"out": out}, 2, allow_lane_stores=True
    )
    assert torch.equal(out, torch.zeros_like(out))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
@pytest.mark.parametrize("threads", [32, 1024])
def test_fragment_warp_scan_sdk_staged_values_and_guarded_load(
    dtype, threads, tmp_path
):
    import importlib.util

    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    from test.test_cute_fragment_scan_config import _computed_scan

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_computed_scan, (torch.ones(3, 5, 65, dtype=dtype), 2, False))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [2],
                "cute_fragment_threads": threads,
                "cute_fragment_warp_scan": True,
            }
        )
        tree = ast.parse(bound.to_code(config))
    phase = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "fragment_scan_lane"
            for stmt in node.body
        )
    )
    # Keep the actual generated control flow and shuffle network. Substitute
    # only its shared source reads with a typed runtime argument, so the SDK
    # stages the real branch-local definitions without requiring a GPU.

    class SourceValue(ast.NodeTransformer):
        def visit_Subscript(self, node):
            assert isinstance(node.ctx, ast.Load)
            assert isinstance(node.value, ast.Name)
            assert node.value.id.startswith("fragment_buffer")
            return ast.copy_location(ast.Name("input_value", ast.Load()), node)

    body = []
    for stmt in phase.body:
        if isinstance(stmt, ast.If) and any(
            isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
            for node in ast.walk(stmt)
        ):
            break
        body.append(SourceValue().visit(stmt))
    assert isinstance(phase.target, ast.Name)
    code = (
        "import cutlass\nimport cutlass.cute as cute\n@cute.jit\n"
        f"def partial(fragment_thread, {phase.target.id}, input_value):\n"
        + "\n".join(
            "    " + line for stmt in body for line in ast.unparse(stmt).splitlines()
        )
        + "\n    return fragment_scan_value\n"
    )
    carry = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "fragment_scan_chunks"
    )
    assert isinstance(carry.value, ast.IfExp)
    assert isinstance(carry.value.body, ast.Subscript)
    assert isinstance(carry.value.body.value, ast.Name)
    assert isinstance(carry.value.test, ast.Compare)
    assert isinstance(carry.value.test.left, ast.Name)
    lane = carry.value.test.left.id
    output_coordinates = sorted(
        {
            node.id
            for node in ast.walk(carry.value.body.slice)
            if isinstance(node, ast.Name)
        }
        - {lane}
    )
    carry.value.body.value.id = "totals"
    code += (
        f"@cute.jit\ndef carry(totals, {lane}):\n"
        + "".join(f"    {name} = 0\n" for name in output_coordinates)
        + f"    {ast.unparse(carry)}\n    return fragment_scan_chunks\n"
    )
    path = tmp_path / "generated_scan.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("generated_scan_sdk", path)
    assert spec is not None and spec.loader is not None
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    scalar = {
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }[dtype]
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "partial",
                (
                    [cutlass.Int32.mlir_type] * 2 + [scalar.mlir_type],
                    [scalar.mlir_type],
                ),
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                result = generated.partial(
                    cutlass.Int32(block.arguments[0]),
                    cutlass.Int32(block.arguments[1]),
                    scalar(block.arguments[2]),
                )
                assert result.dtype == scalar
                func.ReturnOp([result.ir_value()])
            function = func.FuncOp(
                "carry",
                (
                    [cutlass.Int64.mlir_type, cutlass.Int32.mlir_type],
                    [scalar.mlir_type],
                ),
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                pointer = cute.make_ptr(
                    scalar,
                    cutlass.Int64(block.arguments[0]),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                totals = cute.make_tensor(pointer, cute.make_layout((3,)))
                result = generated.carry(totals, cutlass.Int32(block.arguments[1]))
                assert result.dtype == scalar
                func.ReturnOp([result.ir_value()])
        assert module.operation.verify()
        assert "nvvm.shfl.sync" in str(module)
        branches = [op for op in block.operations if op.operation.name == "scf.if"]
        assert len(branches) == 1
        then_ops = branches[0].regions[0].blocks[0].operations
        else_ops = branches[0].regions[1].blocks[0].operations
        assert sum(op.operation.name == "cute.memref.load" for op in then_ops) == 1
        assert not any(op.operation.name == "cute.memref.load" for op in else_ops)
        assert not any(
            op.operation.name == "cute.memref.load" for op in block.operations
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_mixed_warp_scans(x: torch.Tensor, y: torch.Tensor, reverse: hl.constexpr):
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    reused = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        out_x[row, :] = hl.cumsum(x[row, :] + 1, dim=-1, reverse=reverse)
        values = y[row, :] * 2
        out_y[row, :] = hl.cumsum(values, dim=-1, reverse=not reverse)
        reused[row, :] = values
    return out_x, out_y, reused


def _mixed_warp_scan_inputs(dtype, fallback_dtype, width):
    x = (torch.arange(3 * width).reshape(3, width) % 7 - 3).to(fallback_dtype)
    y = (torch.arange(3 * 65).reshape(3, 65) % 5 - 2).to(dtype)
    if dtype == torch.int64:
        y += 2**40
    return x, y


def _mixed_warp_scan_reference(x, y, reverse):
    def scan(value, backwards):
        if backwards:
            return value.flip((1,)).cumsum(1, dtype=value.dtype).flip((1,))
        return value.cumsum(1, dtype=value.dtype)

    return scan(x + 1, reverse), scan(y * 2, not reverse), y * 2


@pytest.mark.parametrize("mode", ["serial", "cooperative"])
@pytest.mark.parametrize(
    "dtype,fallback_dtype,width,reverse",
    [
        (torch.int64, torch.int32, 1057, True),
        (torch.float64, torch.float32, 1057, False),
        (torch.float32, torch.float16, 33, True),
        (torch.int32, torch.bfloat16, 33, False),
    ],
)
def test_fragment_mixed_warp_scans_cpu(mode, dtype, fallback_dtype, width, reverse):
    x, y = _mixed_warp_scan_inputs(dtype, fallback_dtype, width)
    before = (x.clone(), y.clone())
    expected = _mixed_warp_scan_reference(x, y, reverse)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_mixed_warp_scans, (x, y, reverse))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [2],
                "cute_fragment_scan": mode,
                "cute_fragment_warp_scan": True,
            }
        )
        code = bound.to_code(config)
    assert "shuffle_sync_up" in code
    # Wide integer phases now use hierarchical prefixes; floating and narrow
    # unsupported dtypes retain the explicitly selected fallback schedule.
    fallback = fallback_dtype not in (torch.int32, torch.int64)
    assert ("fragment_scan_initialized" in code) == (mode == "serial" and fallback)
    actual = [torch.full_like(value, -9) for value in expected]
    state = _simulate_fragment_warp_reduction(
        code,
        {"x": x, "y": y},
        dict(zip(("out_x", "out_y", "reused"), actual, strict=True)),
        2,
        allow_lane_stores=True,
    )
    assert state.exchanges > 0
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert torch.equal(x, before[0]) and torch.equal(y, before[1])


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("mode", ["serial", "cooperative"])
@pytest.mark.parametrize(
    "dtype,fallback_dtype,width,reverse",
    [
        (torch.int64, torch.int32, 1057, True),
        (torch.float64, torch.float32, 1057, False),
        (torch.float32, torch.float16, 33, True),
        (torch.int32, torch.bfloat16, 33, False),
    ],
)
def test_fragment_mixed_warp_scans_native(mode, dtype, fallback_dtype, width, reverse):
    x, y = (
        value.cuda() for value in _mixed_warp_scan_inputs(dtype, fallback_dtype, width)
    )
    before = (x.clone(), y.clone())
    expected = _mixed_warp_scan_reference(x, y, reverse)
    bound = _fragment_mixed_warp_scans.bind((x, y, reverse))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "block_sizes": [2],
            "cute_fragment_scan": mode,
            "cute_fragment_warp_scan": True,
        }
    )
    bound.set_config(config)
    actual = bound(x, y, reverse)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert torch.equal(x, before[0]) and torch.equal(y, before[1])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_readonly_snapshots(
    x, kind: hl.constexpr, integer_dtype: hl.constexpr = torch.int64
):
    out = torch.empty_like(x)
    reduced = torch.empty((x.size(0),), dtype=integer_dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1))
        values = hl.load(x, [row, index])
        integers = values.to(integer_dtype) + 1
        if kind == "sum":
            reduced[row] = integers.sum(dtype=integer_dtype)
        elif kind == "min":
            reduced[row] = integers.min()
        else:
            reduced[row] = integers.max()
        hl.store(out, [row, index], values + 2)
    return out, reduced


def _snapshot_codegen(kernel, args, threads=128, enabled=True):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_threads=threads, cute_fragment_register_snapshots=enabled
        )
        return bound.to_code(config)


@pytest.mark.parametrize("kind", ["sum", "min", "max"])
@pytest.mark.parametrize(
    "threads,width", [(32, 65), (128, 129), (512, 513), (1024, 1025)]
)
def test_readonly_snapshot_integer_generated(kind, threads, width):
    from test.test_atomic_ops import _simulate_register_load_program

    x = torch.arange(2 * width).reshape(2, width).float() - width
    code = _snapshot_codegen(_fragment_readonly_snapshots, (x, kind), threads)
    assert "fragment_snapshot_reduce" in code
    out = torch.full_like(x, -99)
    reduced = torch.full((2,), -99, dtype=torch.int64)
    _simulate_register_load_program(
        code, x, threads, host_tensors={"out": out, "reduced": reduced}
    )
    expected = x.to(torch.int64) + 1
    expected = (
        expected.sum(-1)
        if kind == "sum"
        else expected.amin(-1)
        if kind == "min"
        else expected.amax(-1)
    )
    torch.testing.assert_close(out, x + 2, rtol=0, atol=0)
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "kind,threads", [("sum", 32), ("min", 32), ("max", 32), ("sum", 1024)]
)
def test_readonly_snapshot_actual_sdk(tmp_path, kind, threads):
    import importlib.util

    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    source = _snapshot_codegen(
        _fragment_readonly_snapshots, (torch.zeros(2, 65), kind), threads
    )
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    fn.name = "staged"
    fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
    ] + [fn]
    path = tmp_path / "snapshot.py"
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    spec = importlib.util.spec_from_file_location("snapshot_sdk", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with ir.Context(), ir.Location.unknown():
        emitted = ir.Module.create()
        with ir.InsertionPoint(emitted.body):
            entry = func.FuncOp("entry", ([], []))
            with ir.InsertionPoint(entry.add_entry_block()):
                args = [
                    cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Int64 if a.arg == "reduced" else cutlass.Float32,
                            0,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((4096,)),
                    )
                    for a in fn.args.args
                ]
                module.staged(*args)
                func.ReturnOp([])
        assert emitted.operation.verify()
        text = str(emitted)
        assert "nvvm.shfl.sync" in text and "nvvm.barrier" in text


def test_snapshot_requested_coordinate_proof():
    from helion._compiler.cute.register_snapshots import SnapshotOwner

    owner = SnapshotOwner("index", "slot", 65)
    owner.aliases["coordinate"] = "index // 1 % 128"
    owner.prove("coordinate // 1")
    owner.prove("index * 4 // 1 - index * 3")
    for expression in (
        "64 - index",
        "index + 1",
        "index % 64",
        "foreign",
        "index // 2",
    ):
        with pytest.raises(exc.InvalidConfig, match="active owner"):
            owner.prove(expression)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_snapshot_unsupported(x, kind: hl.constexpr):
    out = torch.empty_like(x)
    reduced = torch.empty((x.size(0),), device=x.device)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1))
        value = hl.load(x, [row, index])
        if kind == "float":
            reduced[row] = value.sum()
        elif kind == "arg":
            reduced[row] = value.argmax().to(torch.float32)
        elif kind == "flip":
            hl.store(out, [row, index], torch.flip(value, [0]))
        elif kind == "gather":
            hl.store(out, [row, index], torch.gather(value, 0, x.size(1) - 1 - index))
        elif kind == "carry":
            for _ in range(2):
                value = value + 1
            hl.store(out, [row, index], value)
        elif kind == "phi":
            if row % 2 == 0:
                changed = value + 1
            else:
                changed = value - 1
            hl.store(out, [row, index], changed)
    return out, reduced


@pytest.mark.parametrize("kind", ["float", "arg", "flip", "gather", "carry", "phi"])
def test_snapshot_unsupported_consumers_decline(kind):
    with pytest.raises(exc.InvalidConfig):
        _snapshot_codegen(_fragment_snapshot_unsupported, (torch.ones(2, 65), kind))


@pytest.mark.parametrize("width", [0, 1])
def test_snapshot_scalarized_or_empty_domain_declines(width):
    with pytest.raises(exc.InvalidConfig):
        _snapshot_codegen(_fragment_readonly_snapshots, (torch.ones(2, width), "sum"))


def test_snapshot_budget_and_boolean_config():
    x = torch.ones(2, 1025)
    with pytest.raises(exc.InvalidConfig):
        _snapshot_codegen(_fragment_readonly_snapshots, (x, "sum"), 32)
    x = torch.ones(2, 65)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_readonly_snapshots, (x, "sum"))
        config = bound.config_spec.default_config()
        original = bound.to_code(config)
        config.config["cute_fragment_register_snapshots"] = False
        assert bound.to_code(config) == original
        for value in (1, "true", None):
            config.config["cute_fragment_register_snapshots"] = value
            with pytest.raises(exc.InvalidConfig):
                bound.to_code(config)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("kind", ["sum", "min", "max"])
def test_snapshot_integer_boundary_payloads(dtype, kind):
    from test.test_atomic_ops import _simulate_register_load_program

    limit = torch.iinfo(dtype)
    x = torch.tensor([limit.min, limit.max, -1, 0, 1] * 13, dtype=dtype).reshape(1, 65)
    code = _snapshot_codegen(_fragment_readonly_snapshots, (x, kind, dtype), 32)
    out = torch.empty_like(x)
    reduced = torch.empty((1,), dtype=dtype)
    _simulate_register_load_program(
        code, x, 32, host_tensors={"out": out, "reduced": reduced}
    )
    values = x + 1
    expected = (
        values.sum(-1, dtype=dtype)
        if kind == "sum"
        else values.amin(-1)
        if kind == "min"
        else values.amax(-1)
    )
    torch.testing.assert_close(out, x + 2, rtol=0, atol=0)
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)


def test_snapshot_strided_offset_storage_and_input_integrity():
    from test.test_atomic_ops import _simulate_register_load_program

    backing = torch.arange(2 * 131).reshape(2, 131).float()
    x = backing[:, 1::2]
    before = backing.clone()
    code = _snapshot_codegen(_fragment_readonly_snapshots, (x, "sum"), 32)
    out = torch.full_like(x, -99)
    reduced = torch.empty((2,), dtype=torch.int64)
    _simulate_register_load_program(
        code, x, 32, host_tensors={"out": out, "reduced": reduced}
    )
    torch.testing.assert_close(out, x + 2, rtol=0, atol=0)
    torch.testing.assert_close(reduced, (x.to(torch.int64) + 1).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_snapshot_alias(x, out):
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1))
        values = hl.load(x, [row, index])
        hl.store(out, [row, x.size(1) - 1 - index], values + 1)
    return out


def test_snapshot_alias_conflict_fails_closed():
    backing = torch.ones(2, 128)
    for x, out in ((backing, backing), (backing[:, :65], backing[:, 63:])):
        with pytest.raises(exc.InvalidConfig):
            _snapshot_codegen(_fragment_snapshot_alias, (x, out), 32)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("threads", [32, 1024])
@pytest.mark.parametrize("kind", ["sum", "min", "max"])
def test_snapshot_native_integer_boundaries(threads, kind, dtype):
    limits = torch.iinfo(dtype)
    x = torch.tensor(
        [limits.min, limits.max, -1, 0, 1] * 13, dtype=dtype, device=DEVICE
    ).reshape(1, 65)
    before = x.clone()
    bound = _fragment_readonly_snapshots.bind((x, kind, dtype))
    config = bound.config_spec.default_config()
    config.config.update(
        cute_fragment_register_snapshots=True, cute_fragment_threads=threads
    )
    out, reduced = bound.compile_config(config)(x, kind, dtype)
    values = x + 1
    expected = (
        values.sum(-1, dtype=dtype)
        if kind == "sum"
        else values.amin(-1)
        if kind == "min"
        else values.amax(-1)
    )
    torch.testing.assert_close(out, x + 2, rtol=0, atol=0)
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_snapshot_native_typed_loads(dtype):
    x = torch.arange(130, device=DEVICE).reshape(2, 65).to(dtype) - 65
    before = x.clone()
    bound = _fragment_readonly_snapshots.bind((x, "sum"))
    config = bound.config_spec.default_config()
    config.config.update(
        cute_fragment_register_snapshots=True, cute_fragment_threads=128
    )
    out, reduced = bound.compile_config(config)(x, "sum")
    torch.testing.assert_close(out, x + 2, rtol=0, atol=0)
    torch.testing.assert_close(reduced, (x.to(torch.int64) + 1).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_packet_masked_scan(x: torch.Tensor, modulus: hl.constexpr):
    raw = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        index = hl.arange(x.size(1))
        value = hl.load(x, [row, index], extra_mask=index % modulus != 0)
        raw[row, :] = value
        out[row, :] = hl.cumsum(value + 1, dim=-1) + value.sum(-1)[:, None]
    return raw, out


@pytest.mark.parametrize("width", [4, 17, 65, 128])
@pytest.mark.parametrize("layout", ["dense", "stride", "offset", "transpose"])
def test_fragment_packet_loads_generated_masks_layouts(width, layout):
    base = torch.arange(3 * width * 2, dtype=torch.float32).reshape(3, width * 2)
    if layout == "dense":
        x = base[:, :width].contiguous()
    elif layout == "stride":
        x = base[:, ::2]
    elif layout == "offset":
        x = base[:, 1 : width + 1]
    else:
        x = torch.arange(3 * width, dtype=torch.float32).reshape(width, 3).T
    before = x.clone()
    codes = []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_packet_masked_scan, (x, 13))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [2]
        for enabled in (False, True):
            config.config["cute_fragment_packet_loads"] = enabled
            codes.append(bound.to_code(config))
    assert codes[0].count("sync_threads") == codes[1].count("sync_threads")
    if layout == "offset":
        assert "num_bits_per_copy=128" not in codes[1]
    else:
        assert "num_bits_per_copy=128" in codes[1]
    outputs = []
    for code in codes:
        raw, out = torch.empty_like(x), torch.empty_like(x)
        stats = _simulate_independent_fragment(
            code, {"x": x}, {"raw": raw, "out": out}, 2
        )
        outputs.append((raw, out, stats))
    expected = torch.where(torch.arange(width) % 13 != 0, x, 0)
    for raw, out, _stats in outputs:
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            out, (expected + 1).cumsum(-1) + expected.sum(-1)[:, None], rtol=0, atol=0
        )
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    if layout == "dense" and width >= 17:
        assert outputs[1][2]["vector"] > 0
    if layout in ("stride", "transpose", "offset"):
        assert outputs[1][2]["vector"] == 0


@pytest.mark.parametrize("strategy_name", ["FROM_RANDOM", "FROM_BEST_AVAILABLE"])
def test_fragment_packet_loads_preserves_old_population_rng(strategy_name):
    import random

    from test.test_compiler_coverage import make_search

    from helion.autotuner.pattern_search import InitialPopulationStrategy

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        with patch(
            "helion._compiler.autotuner_heuristics.register_fragment_packet_loads_coverage"
        ):
            old = _cpu_bind(
                helion.kernel(
                    _fragment_uncached_cheap.fn, backend="cute", static_shapes=True
                ),
                (torch.ones(3, 65),),
            )
        new = _cpu_bind(
            helion.kernel(
                _fragment_uncached_cheap.fn, backend="cute", static_shapes=True
            ),
            (torch.ones(3, 65),),
        )
        assert old.config_spec.default_config() == new.config_spec.default_config()
        assert (
            old.config_spec.compiler_seed_configs
            == new.config_spec.compiler_seed_configs
        )
        old_code = old.to_code(old.config_spec.default_config())
        assert new.to_code(new.config_spec.default_config()) == old_code
    strategy = InitialPopulationStrategy[strategy_name]
    for seed in (31, 987, 2026):
        prior = make_search(old.config_spec, count=100, strategy=strategy)
        current = make_search(new.config_spec, count=100, strategy=strategy)
        random.seed(seed)
        expected = [
            prior.config_gen.unflatten(row)
            for row in prior._generate_initial_population_flat()
        ]
        state = random.getstate()
        random.seed(seed)
        actual = [
            current.config_gen.unflatten(row)
            for row in current._generate_initial_population_flat()
        ]
        assert random.getstate() == state
        assert actual[: len(expected)] == expected
        assert len(actual) == len(expected) + 1
        assert actual[-1]["cute_fragment_packet_loads"] is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_fragment_packet_loads_unsupported_dtype_default_parity(dtype):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_uncached_cheap, (torch.ones(3, 17, dtype=dtype),))
        assert not bound.config_spec.cute_fragment_packet_load_root_ids
        config = bound.config_spec.default_config()
        before = bound.to_code(config)
        config.config["cute_fragment_packet_loads"] = False
        assert bound.to_code(config) == before
        config.config["cute_fragment_packet_loads"] = True
        with pytest.raises(exc.InvalidConfig, match="readonly Float32"):
            bound.to_code(config)


@pytest.mark.parametrize(
    "bad",
    [
        "value = (x.iterator + shared[index]).load()",
        "value = (indices.iterator + offset).load()",
        "offset = unknown(index)\nvalue = (x.iterator + offset).load()",
        "if flag:\n    x[index] = value\nvalue = (x.iterator + index).load()",
        "value = (x.iterator + index).load()\ncute.arch.sync_threads()",
    ],
)
def test_fragment_packet_loads_rejects_effectful_coordinate_recipes(bad):
    from helion._compiler.cute.packet_loads import _load_assignment

    assert _load_assignment(ast.parse(bad).body, "x") is None


@pytest.mark.parametrize(
    "layout,width", [("dense", 17), ("dense", 128), ("stride", 65), ("offset", 17)]
)
@skipUnlessBackends(["cute"])
def test_fragment_packet_loads_native(layout, width):
    base = torch.arange(3 * width * 2, device=DEVICE, dtype=torch.float32).reshape(
        3, width * 2
    )
    x = (
        base[:, ::2]
        if layout == "stride"
        else base[:, 1 : width + 1]
        if layout == "offset"
        else base[:, :width].contiguous()
    )
    before = x.clone()
    bound = _fragment_packet_masked_scan.bind((x, 13))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[2], cute_fragment_packet_loads=True)
    actual = bound.compile_config(config)(x, 13)
    expected = torch.where(torch.arange(width, device=DEVICE) % 13 != 0, x, 0)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[1], (expected + 1).cumsum(-1) + expected.sum(-1)[:, None], rtol=0, atol=0
    )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_fragment_packet_loads_preserves_special_bits():
    import numpy as np

    bits = torch.tensor(
        [
            0,
            -2147483648,
            1,
            -2147483647,
            1065353216,
            -1082130432,
            2139095040,
            -8388608,
            2143289635,
            8388607,
            8388608,
            -2139095040,
            0,
            -2147483648,
            1065353216,
            1,
        ],
        dtype=torch.int32,
    )
    x = bits.view(torch.float32).repeat(3, 1)
    codes = []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_packet_masked_scan, (x, 1000))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [2]
        for enabled in (False, True):
            config.config["cute_fragment_packet_loads"] = enabled
            codes.append(bound.to_code(config))
    snapshots = []
    with np.errstate(invalid="ignore"):
        for code in codes:
            raw, out = torch.empty_like(x), torch.empty_like(x)
            _simulate_independent_fragment(code, {"x": x}, {"raw": raw, "out": out}, 2)
            snapshots.append((raw, out))
    expected = x.clone()
    expected[:, 0] = 0
    assert torch.equal(snapshots[0][0].view(torch.int32), expected.view(torch.int32))
    assert torch.equal(snapshots[1][0].view(torch.int32), expected.view(torch.int32))
    torch.testing.assert_close(
        snapshots[0][1], snapshots[1][1], rtol=0, atol=0, equal_nan=True
    )


def test_fragment_packet_loads_declines_aliasing_host_mutation():
    x = torch.arange(3 * 17, dtype=torch.float32).reshape(3, 17)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_dynamic_host_strides, (x, x))
        assert not bound.config_spec.cute_fragment_packet_load_root_ids
        config = bound.config_spec.default_config()
        config.config["cute_fragment_packet_loads"] = True
        with pytest.raises(exc.InvalidConfig, match="readonly Float32"):
            bound.to_code(config)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fragment_packet_loads_zero_extent_does_not_issue_packets(dtype):
    x = torch.empty((3, 0), dtype=dtype)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_uncached_cheap, (x,))
        config = bound.config_spec.default_config()
        before = bound.to_code(config)
        assert "num_bits_per_copy=128" not in before
        config.config["cute_fragment_packet_loads"] = True
        if dtype == torch.float32:
            assert bound.to_code(config) == before
        else:
            with pytest.raises(exc.InvalidConfig, match="readonly Float32"):
                bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_packet_shared_address(
    x, offsets, width: hl.constexpr, modulus: hl.constexpr
):
    out = torch.empty((2, width), device=x.device, dtype=x.dtype)
    raw = torch.empty_like(out)
    for row in hl.tile(2):
        index = hl.arange(width)
        samples = hl.load(offsets, [row, hl.arange(offsets.size(1))])
        offset = samples.sum(-1).to(torch.int32)
        selected = index[None, :] + offset[:, None]
        flat = row.index[:, None] * width + selected
        value = hl.load(
            x,
            [flat],
            extra_mask=(selected >= 0)
            & (selected < width)
            & (index[None, :] % modulus != 0),
        )
        raw[row, :] = value
        out[row, :] = hl.cumsum(value + 1, dim=-1) + value.sum(-1)[:, None]
    return raw, out


@pytest.mark.parametrize("width", [4, 17, 65])
@pytest.mark.parametrize("layout", ["dense", "stride", "offset"])
@pytest.mark.parametrize("offset", [-1, 0, 3])
def test_fragment_packet_shared_address_generated(width, layout, offset):
    base = torch.arange(4 * width, dtype=torch.float32)
    x = (
        base[: 2 * width]
        if layout == "dense"
        else base[::2]
        if layout == "stride"
        else base[1 : 2 * width + 1]
    )
    before = x.clone()
    offsets = torch.zeros((2, 4), dtype=torch.float32)
    offsets[:, 0] = offset
    codes = []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_packet_shared_address, (x, offsets, width, 13))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [1]
        for enabled in (False, True):
            config.config["cute_fragment_packet_loads"] = enabled
            codes.append(bound.to_code(config))
    assert codes[0].count("sync_threads") == codes[1].count("sync_threads")
    index = torch.arange(width)
    wanted = index + offset
    matrix = x.reshape(2, width)
    expected = torch.where(
        (wanted >= 0) & (wanted < width) & (index % 13 != 0),
        matrix[:, wanted.clamp(0, width - 1)],
        0,
    )
    for code in codes:
        out = torch.full((2, width), -99, dtype=x.dtype)
        raw = torch.full_like(out, -99)
        stats = _simulate_independent_fragment(
            code, {"x": x, "offsets": offsets}, {"raw": raw, "out": out}, 2
        )
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            out, (expected + 1).cumsum(-1) + expected.sum(-1)[:, None], rtol=0, atol=0
        )
        if code == codes[1] and layout == "dense" and width == 65 and offset == 0:
            assert stats["vector"] > 0
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def _packet_shared_proof_fixture():
    from helion._compiler.cute.packet_loads import _published_dependencies

    slot = Fragment((), torch.float32, lambda _: "slot[0]", True, storage="slot")
    view = Fragment((), torch.float32, lambda _: "slot[0]", True, (slot,))
    # A lazy view is not register-backed: it has transitive shared storage.
    lazy = Fragment((), torch.float32, lambda _: "slot[0]", dependencies=(slot,))
    compiler = object.__new__(FragmentCompiler)
    compiler.thread = "thread"
    compiler.threads = 128
    compiler.buffers = [("slot", torch.float32, 1), ("unused", torch.float32, 1)]
    compiler.scopes = [{"lazy": lazy}]
    compiler.held = []
    compiler.pending_local_atomics = set()
    compiler.cg = SimpleNamespace(
        statements_stack=[
            ast.parse(
                "for owner in range(thread, 1, 128):\n    slot[0] = cutlass.Float32(3)\ncute.arch.sync_threads()\n"
            ).body
        ]
    )
    return compiler, slot, view, lazy, _published_dependencies


def test_fragment_packet_shared_publication_and_transitive_lifetime():
    compiler, slot, view, lazy, proof = _packet_shared_proof_fixture()
    assert proof(compiler, (lazy,)) == frozenset({"slot"})
    assert proof(compiler, (view,)) == frozenset({"slot"})
    result = Fragment((4,), torch.float32, lambda _: "0")
    compiler.held.append((result, lazy))
    compiler.scopes.clear()
    compiler.smem_bytes = 0
    selected = compiler.allocate(result)
    assert selected.storage == "unused" and selected.storage != slot.storage
    assert compiler.live_buffers() == {"slot"}


@pytest.mark.parametrize(
    "bad",
    [
        "pending",
        "alias",
        "later_write",
        "multiple",
        "missing_publication",
        "unknown_storage",
        "register",
    ],
)
def test_fragment_packet_shared_publication_declines(bad):
    compiler, slot, _view, lazy, proof = _packet_shared_proof_fixture()
    dependencies = (lazy,)
    body = compiler.cg.statements_stack[-1]
    if bad == "pending":
        compiler.pending_local_atomics.add("slot")
    elif bad == "alias":
        body.extend(ast.parse("alias = slot").body)
    elif bad == "later_write":
        body.extend(
            ast.parse(
                "for owner in range(thread, 1, 128):\n    slot[0] = cutlass.Float32(9)"
            ).body
        )
    elif bad == "multiple":
        body.extend(
            ast.parse("slot[0] = cutlass.Float32(9)\ncute.arch.sync_threads()").body
        )
    elif bad == "missing_publication":
        body.pop()
    elif bad == "unknown_storage":
        compiler.buffers.clear()
    else:
        register = Fragment((), torch.float32, lambda _: "private", True)
        dependencies = (lazy, register)
    assert not proof(compiler, dependencies)


@pytest.mark.parametrize("index", ["0", "0 * 1", "2 - 2"])
def test_fragment_packet_shared_slot_keeps_conditional_scope(index):
    from helion._compiler.cute.packet_loads import _load_assignment

    body = ast.parse(
        f"if valid:\n    offset = cutlass.Int64(slot[{index}])\n    result = cutlass.Float32((x.iterator + offset).load())"
    ).body
    before = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert _load_assignment(body, "x") is None
    assert _load_assignment(body, "x", frozenset({"slot"})) is not None
    assert ast.dump(ast.Module(body=body, type_ignores=[])) == before


@pytest.mark.parametrize(
    "expression",
    [
        "slot[lane]",
        "slot[1]",
        "slot[False]",
        "slot[0.0]",
        "private[0]",
        "unknown[0]",
        "slot[0 * call()]",
    ],
)
def test_fragment_packet_shared_slot_unknown_or_register_declines(expression):
    from helion._compiler.cute.packet_loads import _load_assignment

    body = ast.parse(
        f"offset = {expression}\nresult = (x.iterator + offset).load()"
    ).body
    assert _load_assignment(body, "x", frozenset({"slot"})) is None


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "width,layout,offset", [(65, "dense", 0), (65, "stride", 3), (17, "offset", -1)]
)
def test_fragment_packet_shared_address_native(width, layout, offset):
    base = torch.arange(4 * width, dtype=torch.float32, device=DEVICE)
    x = (
        base[: 2 * width]
        if layout == "dense"
        else base[::2]
        if layout == "stride"
        else base[1 : 2 * width + 1]
    )
    offsets = torch.zeros((2, 4), dtype=torch.float32, device=DEVICE)
    offsets[:, 0] = offset
    before = x.clone(), offsets.clone()
    bound = _fragment_packet_shared_address._bind_isolated((x, offsets, width, 13))
    config = bound.config_spec.default_config()
    config.config.update(block_sizes=[1], cute_fragment_packet_loads=True)
    raw, out = bound.compile_config(config)(x, offsets, width, 13)
    index = torch.arange(width, device=DEVICE)
    selected = index + offset
    expected = torch.where(
        (selected >= 0) & (selected < width) & (index % 13 != 0),
        x.reshape(2, width)[:, selected.clamp(0, width - 1)],
        0,
    )
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        out, (expected + 1).cumsum(-1) + expected.sum(-1)[:, None], rtol=0, atol=0
    )
    torch.testing.assert_close(x, before[0], rtol=0, atol=0)
    torch.testing.assert_close(offsets, before[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "width,reverse,threads", [(1025, False, 32), (2048, True, 128), (4097, True, 512)]
)
def test_fragment_hierarchical_integer_scan_cpu(dtype, width, reverse, threads):
    x = (torch.arange(2 * width).reshape(2, width) % 19 - 9).to(dtype)
    limits = torch.iinfo(dtype)
    x[0, ::37] = limits.max
    x[1, ::41] = limits.min
    before = x.clone()
    ordered = x.flip((-1,)) if reverse else x
    expected = ordered.cumsum(-1, dtype=dtype)
    if reverse:
        expected = expected.flip((-1,))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_warp_scan_special, (x, reverse))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [1],
                "cute_fragment_threads": threads,
                "cute_fragment_warp_scan": True,
            }
        )
        code = bound.to_code(config)
    actual = torch.full_like(x, -123)
    _simulate_fragment_warp_reduction(
        code, {"x": x}, {"out": actual}, 2, threads, allow_lane_stores=True
    )
    assert torch.equal(actual, expected)
    assert torch.equal(x, before)
    assert "shuffle_sync_up" in code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width,reverse", [(1025, False), (2048, True), (4097, True)])
def test_fragment_hierarchical_integer_scan_native(dtype, width, reverse):
    x = (torch.arange(3 * width, device=DEVICE).reshape(3, width) % 19 - 9).to(dtype)
    limits = torch.iinfo(dtype)
    x[0, ::37] = limits.max
    x[1, ::41] = limits.min
    before = x.clone()
    ordered = x.flip((-1,)) if reverse else x
    expected = ordered.cumsum(-1, dtype=dtype)
    if reverse:
        expected = expected.flip((-1,))
    bound = _fragment_warp_scan_special.bind((x, reverse))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "block_sizes": [1],
            "cute_fragment_threads": 128,
            "cute_fragment_warp_scan": True,
        }
    )
    actual = bound.compile_config(config)(x, reverse)
    assert torch.equal(actual, expected)
    assert torch.equal(x, before)


@pytest.mark.parametrize("axis,reverse", [(1, False), (1, True), (2, True)])
def test_fragment_hierarchical_integer_scan_cpu_strides_and_input_reuse(axis, reverse):
    from test.test_cute_fragment_scan_config import _computed_scan

    width = 1057
    shape = (3, width, 2) if axis == 1 else (3, 2, width)
    storage = torch.arange(2 * math.prod(shape)).reshape(*shape, 2)
    x = (storage[..., 0] % 13 - 6).to(torch.int32)
    # Preserve a noncontiguous input after the dtype conversion/pointwise work.
    backing = torch.stack((x, x + 7), -1)
    x = backing[..., 0]
    assert not x.is_contiguous()
    before = x.clone()
    backing_before = backing.clone()
    values = x + 1
    ordered = values.flip((axis,)) if reverse else values
    expected = ordered.cumsum(axis, dtype=x.dtype)
    if reverse:
        expected = expected.flip((axis,))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_computed_scan, (x, axis, reverse))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [2],
                "cute_fragment_threads": 128,
                "cute_fragment_warp_scan": True,
            }
        )
        code = bound.to_code(config)
    out, reused = torch.full_like(x, -123), torch.full_like(x, -123)
    # The simulator's pointer reads physical storage offsets. Codegen was
    # bound to the strided view; its pointer starts at this backing allocation.
    _simulate_fragment_warp_reduction(
        code,
        {"x": backing},
        {"out": out, "reused": reused},
        2,
        128,
        allow_lane_stores=True,
    )
    assert torch.equal(out, expected)
    assert torch.equal(reused, values * 2)
    assert torch.equal(x, before)
    assert torch.equal(backing, backing_before)


def test_fragment_hierarchical_integer_scan_resource_limit():
    x = torch.ones((1, 16385), dtype=torch.int64)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_warp_scan_special, (x, False))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "block_sizes": [1],
                "cute_fragment_threads": 1024,
                "cute_fragment_warp_scan": True,
            }
        )
        with pytest.raises(exc.InvalidConfig, match="shared bytes, exceeding"):
            bound.to_code(config)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "dtype,axis,reverse",
    [(torch.int32, 1, False), (torch.int64, 1, True), (torch.int64, 2, True)],
)
def test_fragment_hierarchical_integer_scan_native_strides(dtype, axis, reverse):
    from test.test_cute_fragment_scan_config import _computed_scan

    shape = (3, 1057, 2) if axis == 1 else (3, 2, 1057)
    backing = (torch.arange(2 * math.prod(shape), device=DEVICE) % 13 - 6).to(dtype)
    backing = backing.reshape(*shape, 2)
    x = backing[..., 0]
    before = backing.clone()
    values = x + 1
    ordered = values.flip((axis,)) if reverse else values
    expected = ordered.cumsum(axis, dtype=dtype)
    if reverse:
        expected = expected.flip((axis,))
    bound = _computed_scan.bind((x, axis, reverse))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "block_sizes": [2],
            "cute_fragment_threads": 128,
            "cute_fragment_warp_scan": True,
        }
    )
    out, reused = bound.compile_config(config)(x, axis, reverse)
    assert torch.equal(out, expected)
    assert torch.equal(reused, values * 2)
    assert torch.equal(backing, before)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_while_plan_recipe(x, limits, swap: hl.constexpr, steps: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))
        left = x[row, lane]
        right = left + 1
        iteration = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            if swap:
                left, right = right, left
            else:
                left, right = left + right, left - right
            for _packet in range(steps):
                left = torch.gather(left + 1, 0, ((lane * 3 + 1) % x.size(1)).long())
            iteration = iteration + 1
        out[row, lane] = left + right
    return out


def _resident_while_bound(swap=False, steps=2):
    from helion.language import _tracing_ops

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _resident_while_plan_recipe,
            (
                torch.arange(48).reshape(3, 16).float(),
                torch.tensor([0, 1, 3], dtype=torch.int32),
                swap,
                steps,
            ),
        )
    graphs = bound.host_function.device_ir.graphs
    call = next(
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.target is _tracing_ops._while_loop
    )
    return bound, call, graphs


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_resident_while_plan_actual_current_edges(steps):
    from helion._compiler.cute.resident_while import resident_while_plan

    bound, call, graphs = _resident_while_bound(steps=steps)
    before = tuple(str(graph.graph) for graph in graphs)
    plan = resident_while_plan(call, graphs)
    assert plan.body.captures == tuple(call.args[2])
    assert plan.condition.captures == plan.body.captures
    assert plan.body.carry_map == ((0, 0), (1, 2), (2, 3))
    assert plan.invariant_slots == (1, 4)
    assert len(plan.nested_fors) == 1
    child = plan.nested_fors[0]
    assert child.parent_call is call
    assert child.captures == tuple(child.call.args[3])
    assert child.carry_map == ((0, 0),)
    assert "whole_cta_entry_and_shared_predicate_publication" in plan.requirements
    assert "current_call_logical_domains_and_lowering_support" in plan.requirements
    assert tuple(str(graph.graph) for graph in graphs) == before
    # Stage 2 requires the explicit gather capability; the structural plan is
    # still insufficient without logical domains and the shared emitter.
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        config = bound.config_spec.default_config()
        config.config["cute_fragment_bounded_gather"] = True
        code = bound.to_code(config)
        assert "while fragment_while_take" in code


def test_resident_while_plan_current_captures_not_node_args():
    from helion._compiler.cute.resident_while import resident_while_plan

    _bound, call, graphs = _resident_while_bound()
    plan = resident_while_plan(call, graphs)
    for context in (plan.condition, plan.body, *plan.nested_fors):
        context.graph.node_args = list(reversed(context.captures))
    current = list(call.args[2])
    current[0], current[1] = current[1], current[0]
    call.args = (*call.args[:2], current, call.args[3])
    changed = resident_while_plan(call, graphs)
    assert changed.body.captures == tuple(current)
    assert changed.condition.captures == tuple(current)
    assert changed.body.carry_map == ((0, 1), (1, 2), (2, 3))
    assert changed.invariant_slots == (0, 4)


def test_resident_while_plan_rejects_actual_ambiguous_swap():
    from helion._compiler.cute.resident_while import resident_while_plan

    _bound, call, graphs = _resident_while_bound(swap=True)
    with pytest.raises(exc.InvalidConfig, match="explicit initialized phi"):
        resident_while_plan(call, graphs)


@pytest.mark.parametrize(
    "kind",
    [
        "dtype",
        "shape",
        "condition",
        "identity",
        "missing_phi",
        "duplicate_destination",
        "duplicate_capture",
    ],
)
def test_resident_while_plan_rejects_mutated_actual_edges(kind):
    from helion._compiler.cute.resident_while import resident_while_plan
    from helion.language import _tracing_ops

    _bound, call, graphs = _resident_while_bound()
    plan = resident_while_plan(call, graphs)
    expected = ""
    if kind in ("dtype", "shape"):
        plan.body.outputs[2].meta["val"] = torch.empty(
            (16 if kind == "dtype" else 17,),
            dtype=torch.int32 if kind == "dtype" else torch.float32,
        )
        expected = "carry physical shape or dtype"
    elif kind == "condition":
        plan.condition.outputs[0].meta["val"] = torch.empty((1,), dtype=torch.bool)
        expected = "Boolean scalar condition"
    elif kind == "identity":
        plan.body.graph.cond_graph_id = plan.body.graph.graph_id
        expected = "condition/body graph identity"
    elif kind in ("missing_phi", "duplicate_destination"):
        projection = next(node for node in call.users if node.args[1] == 2)
        phi = next(
            node for node in projection.users if node.target is _tracing_ops._phi
        )
        if kind == "missing_phi":
            phi.target = torch.ops.aten.clone.default
            expected = "explicit initialized phi"
        else:
            phi.args = (call.args[2][2], projection)
            expected = "duplicate carry destination"
    else:
        captures = list(call.args[2])
        captures[1] = captures[0]
        call.args = (*call.args[:2], captures, None)
        expected = "duplicate capture identity"
    with pytest.raises(exc.InvalidConfig, match=expected):
        resident_while_plan(call, graphs)


@pytest.mark.parametrize(
    "kind", ["dynamic_bound", "while", "if", "mutation", "random", "memory"]
)
def test_resident_while_plan_rejects_nested_effects(kind):
    from helion._compiler.cute.resident_while import resident_while_plan
    from helion.language import _tracing_ops
    from helion.language import memory_ops

    _bound, call, graphs = _resident_while_bound()
    plan = resident_while_plan(call, graphs)
    child = plan.nested_fors[0]
    if kind == "dynamic_bound":
        args = list(child.call.args)
        args[2] = [plan.body.placeholders[0]]
        child.call.args = tuple(args)
        expected = "bounds must be static integers"
    elif kind in ("while", "if"):
        child.call.target = (
            _tracing_ops._while_loop if kind == "while" else _tracing_ops._if
        )
        expected = "nested while/if"
    else:
        node = next(
            node
            for node in child.graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        )
        node.target = {
            "mutation": torch.ops.aten.add_.Tensor,
            "random": torch.ops.aten.rand.default,
            "memory": memory_ops.store,
        }[kind]
        expected = "mutable/nondeterministic" if kind != "memory" else "memory-effect"
    with pytest.raises(exc.InvalidConfig, match=expected):
        resident_while_plan(call, graphs)


@pytest.mark.parametrize(
    "kind", ["foreign_capture", "late_capture", "missing_graph", "duplicate_graph"]
)
def test_resident_while_plan_rejects_call_context_drift(kind):
    from helion._compiler.cute.resident_while import resident_while_plan

    _bound, call, graphs = _resident_while_bound()
    plan = resident_while_plan(call, graphs)
    if kind == "foreign_capture":
        captures = list(call.args[2])
        captures[0] = plan.body.placeholders[0]
        call.args = (*call.args[:2], captures, None)
        expected = "current caller"
    elif kind == "late_capture":
        captures = list(call.args[2])
        captures[0] = next(iter(call.users))
        call.args = (*call.args[:2], captures, None)
        expected = "precede current call"
    elif kind == "missing_graph":
        call.args = (len(graphs) + 1, *call.args[1:])
        expected = "missing graph ID"
    else:
        graphs = [*graphs, plan.condition.graph]
        expected = "duplicate graph ID"
    with pytest.raises(exc.InvalidConfig, match=expected):
        resident_while_plan(call, graphs)


@pytest.mark.parametrize(
    "kind", ["reversed_list", "graph_id", "same_graph", "second_while", "second_for"]
)
def test_resident_while_plan_rejects_cache_context_aliases(kind):
    from helion._compiler.cute.resident_while import resident_while_plan

    _bound, call, graphs = _resident_while_bound()
    plan = resident_while_plan(call, graphs)
    if kind == "reversed_list":
        graphs = list(reversed(graphs))
        expected = "graph list/index identity"
    elif kind == "graph_id":
        plan.condition.graph.graph_id = len(graphs) + 2
        expected = "graph list/index identity"
    elif kind == "same_graph":
        plan.condition.graph.graph = plan.body.graph.graph
        expected = "duplicate underlying graph"
    else:
        original = call if kind == "second_while" else plan.nested_fors[0].call
        with original.graph.inserting_after(original):
            duplicate = original.graph.call_function(
                original.target, original.args, original.kwargs
            )
        assert duplicate is not original
        expected = "callee must have one actual call"
    with pytest.raises(exc.InvalidConfig, match=expected):
        resident_while_plan(call, graphs)


@pytest.mark.parametrize(
    "kind",
    [
        "projection_op",
        "phi_op",
        "projection_owner",
        "phi_owner",
        "projection_order",
        "phi_kwargs",
    ],
)
def test_resident_while_plan_rejects_projection_phi_identity(kind):
    from helion._compiler.cute.resident_while import resident_while_plan

    _bound, call, graphs = _resident_while_bound()
    plan = resident_while_plan(call, graphs)
    item = next(iter(call.users))
    phi = next(iter(item.users))
    if kind == "projection_op":
        item.op = "call_method"
        expected = "invalid loop output projection"
    elif kind == "phi_op":
        phi.op = "call_method"
        expected = "explicit initialized phi"
    elif kind.endswith("owner"):
        node = item if kind == "projection_owner" else phi
        node.graph = plan.body.graph.graph
        expected = "node/graph ownership"
    elif kind == "projection_order":
        call.prepend(item)
        expected = "invalid loop output projection"
    else:
        phi.kwargs = {"unrecognized": True}
        expected = "explicit initialized phi"
    with pytest.raises(exc.InvalidConfig, match=expected):
        resident_while_plan(call, graphs)


@pytest.mark.parametrize("value", ["projection", "phi"])
@pytest.mark.parametrize("change", ["dtype", "shape"])
def test_resident_while_plan_rejects_projection_phi_signature(value, change):
    from helion._compiler.cute.resident_while import resident_while_plan

    _bound, call, graphs = _resident_while_bound()
    resident_while_plan(call, graphs)
    item = next(node for node in call.users if node.args[1] == 0)
    target = item if value == "projection" else next(iter(item.users))
    assert target.meta["val"].dtype == torch.int32
    assert target.meta["val"].shape == ()
    target.meta["val"] = torch.empty(
        () if change == "dtype" else (11,),
        dtype=torch.int16 if change == "dtype" else torch.int32,
    )
    with pytest.raises(exc.InvalidConfig, match=f"{value} physical shape or dtype"):
        resident_while_plan(call, graphs)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_while_recurrence(x, limits, steps: hl.constexpr, capacity: hl.constexpr):
    out = torch.empty_like(x)
    counts = torch.empty((x.size(0),), device=x.device, dtype=torch.int32)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(capacity)
        left = hl.load(x, [row, lane], extra_mask=lane < x.size(1))
        right = left + 1
        iteration = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            left, right = left + right, left - right
            for _packet in range(steps):
                left = torch.gather(left + 1, 0, ((lane * 3 + 1) % x.size(1)).long())
            iteration = iteration + 1
        hl.store(out, [row, lane], left + right, extra_mask=lane < x.size(1))
        counts[row] = iteration
    return out, counts


def _resident_while_codegen(x, limits, steps, threads):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _resident_while_recurrence,
            (x, limits, steps, 1 << (x.size(1) - 1).bit_length()),
        )
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_bounded_gather=True, cute_fragment_threads=threads
        )
        return bound, config, bound.to_code(config)


@pytest.mark.parametrize(
    "width,steps,threads", [(1, 0, 128), (17, 2, 128), (32, 1, 128), (65, 2, 128)]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_resident_while_emitted_shared_epochs(width, steps, threads, dtype):
    from test.test_atomic_ops import _simulate_register_load_program

    x = (torch.arange(3 * width).reshape(3, width) % 7).to(dtype)
    limits = torch.tensor([0, 1, 3], dtype=torch.int32)
    before = x.clone(), limits.clone()
    _bound, _config, code = _resident_while_codegen(x, limits, steps, threads)
    expected = []
    for row in range(3):
        left, right = x[row].clone(), x[row] + 1
        for _iteration in range(limits[row].item()):
            left, right = left + right, left - right
            for _packet in range(steps):
                left = (left + 1)[(torch.arange(width) * 3 + 1) % width]
        expected.append(left + right)
    for order in (list(range(threads)), list(reversed(range(threads)))):
        out = torch.full_like(x, -99)
        counts = torch.full_like(limits, -99)
        _unused, barriers = _simulate_register_load_program(
            code,
            x,
            threads,
            host_tensors={"limits": limits, "out": out, "counts": counts},
            lane_order=order,
        )
        assert barriers > 0
        torch.testing.assert_close(out, torch.stack(expected), rtol=0, atol=0)
        torch.testing.assert_close(counts, limits, rtol=0, atol=0)
        assert torch.equal(x, before[0]) and torch.equal(limits, before[1])


def test_resident_while_predicate_publication_and_reader_barriers():
    from test.test_atomic_ops import _simulate_register_load_program

    x = torch.ones((3, 16))
    limits = torch.tensor([0, 1, 3], dtype=torch.int32)
    _bound, _config, code = _resident_while_codegen(x, limits, 2, 128)
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    sites = []
    for node in ast.walk(fn):
        for _name, body in ast.iter_fields(node):
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name)
                    and target.id.startswith("fragment_while_take")
                    for target in statement.targets
                ):
                    assert ast.unparse(body[index - 1]) == "cute.arch.sync_threads()"
                    assert ast.unparse(body[index + 1]) == "cute.arch.sync_threads()"
                    sites.append((body, index))
    assert len(sites) == 2
    body, index = sites[0]
    body.pop(index - 1)
    mutant = ast.unparse(ast.fix_missing_locations(tree))
    with pytest.raises(AssertionError, match="shared read before initialization"):
        _simulate_register_load_program(
            mutant,
            x,
            128,
            host_tensors={
                "limits": limits,
                "out": torch.empty_like(x),
                "counts": torch.empty_like(limits),
            },
            lane_order=list(reversed(range(128))),
        )


def test_resident_while_existing_thread_and_cache_config_rejections():
    x = torch.ones((3, 16))
    limits = torch.tensor([0, 1, 3], dtype=torch.int32)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, _code = _resident_while_codegen(x, limits, 2, 128)
        for key, value, message in (
            ("cute_fragment_threads", 32, "supported computed fragment root"),
            ("cute_fragment_register_loads", True, "register_loads|shared uncached"),
            (
                "cute_fragment_register_snapshots",
                True,
                "register_snapshots|shared uncached",
            ),
        ):
            changed = helion.Config.from_dict(dict(config) | {key: value})
            with pytest.raises(exc.InvalidConfig, match=message):
                bound.to_code(changed)


def test_resident_while_rejects_captured_host_alias_write():
    from helion._compiler.cute.gather_domains import loop_domain_facts
    from helion.language import memory_ops

    bound, call, graphs = _resident_while_bound()
    with bound.env, bound.host_function:
        assert call in loop_domain_facts(bound._env, graphs).resident_whiles
        load = next(node for node in call.graph.nodes if node.target is memory_ops.load)
        store = next(
            node for node in call.graph.nodes if node.target is memory_ops.store
        )
        store.args = (load.args[0], *store.args[1:])
        assert call not in loop_domain_facts(bound._env, graphs).resident_whiles


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_while_index_drift(x, limits, bounded: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1)).long()
        values = x[row, index]
        iteration = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            if bounded:
                selected = index % x.size(1)
            else:
                selected = index
            values = torch.gather(values + 1, 0, selected)
            index = index + 1
            iteration = iteration + 1
        out[row, :] = values
    return out


@pytest.mark.parametrize("bounded", [False, True])
def test_resident_while_mutable_index_requires_current_bounds(bounded):
    from helion._compiler.cute.gather_domains import loop_domain_facts

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _resident_while_index_drift,
            (torch.ones((3, 16)), torch.tensor([0, 1, 3], dtype=torch.int32), bounded),
        )
        config = bound.config_spec.default_config()
        config.config["cute_fragment_bounded_gather"] = True
        with bound.env, bound.host_function:
            facts = loop_domain_facts(bound._env, bound.host_function.device_ir.graphs)
            for info in bound.host_function.device_ir.graphs:
                for node in info.graph.nodes:
                    if node.op == "placeholder":
                        assert node not in facts.readonly_ranges
        if bounded:
            assert "while fragment_while_take" in bound.to_code(config)
        else:
            with pytest.raises(
                exc.InvalidConfig, match="proved bounded last-axis gather"
            ):
                bound.to_code(config)


def test_resident_while_preserves_shared_resource_limit():
    with pytest.raises(exc.InvalidConfig, match="shared bytes, exceeding"):
        _resident_while_codegen(
            torch.ones((3, 8193)), torch.tensor([0, 1, 3], dtype=torch.int32), 0, 128
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_while_invariant_index(x, limits, nested: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1)).long()
        values = x[row, index]
        iteration = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            if nested:
                for _outer in range(2):
                    for _inner in range(2):
                        values = torch.gather(values + 1, 0, index ^ 1)
            else:
                values = torch.gather(values + 1, 0, index ^ 1)
            iteration = iteration + 1
        out[row, :] = values
    return out


@pytest.mark.parametrize("width", [16, 64, 256])
@pytest.mark.parametrize("nested", [False, True])
def test_resident_while_invariant_index_bounds(width, nested):
    from test.test_atomic_ops import _simulate_register_load_program

    from helion._compiler.cute.gather_domains import loop_domain_facts

    x = torch.arange(3 * width).reshape(3, width).float()
    limits = torch.tensor([0, 1, 3], dtype=torch.int32)
    expected = x.clone()
    for row in range(3):
        for _iteration in range(limits[row].item() * (4 if nested else 1)):
            expected[row] = (expected[row] + 1)[torch.arange(width) ^ 1]
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_resident_while_invariant_index, (x, limits, nested))
        with bound.env, bound.host_function:
            facts = loop_domain_facts(bound.env, bound.host_function.device_ir.graphs)
            assert (0, width - 1) in facts.readonly_ranges.values()
        config = bound.config_spec.default_config()
        config.config["cute_fragment_bounded_gather"] = True
        code = bound.to_code(config)
    for order in (list(range(128)), list(reversed(range(128)))):
        out = torch.full_like(x, -1)
        _simulate_register_load_program(
            code,
            x,
            128,
            host_tensors={"limits": limits, "out": out},
            lane_order=order,
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_while_readonly_program(
    x, limits, out, steps: hl.constexpr, capacity: hl.constexpr
):
    checksums = torch.empty((x.size(0),), device=x.device, dtype=torch.int32)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(capacity)
        value = hl.full([capacity], 0, dtype=torch.int32)
        iteration = hl.full([], 0, dtype=torch.int32)
        checksum = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            for packet in range(steps):
                index = (lane + iteration + packet) % x.size(1)
                fresh = hl.load(x, [row, index])
                adjusted = hl.inline_asm_elementwise(
                    "add.s32 $0, $1, $2;",
                    "=r,r,r",
                    [value, fresh],
                    dtype=torch.int32,
                    is_pure=True,
                    pack=1,
                )
                value = torch.gather(adjusted, 0, ((lane * 3 + 1) % x.size(1)).long())
            checksum = torch.where(lane < x.size(1), value, 0).sum().to(torch.int32)
            iteration = iteration + 1
        out[row, :] = value
        checksums[row] = checksum
    return out, checksums


def _resident_while_readonly_bound(x, limits, out, steps):
    bound = _cpu_bind(
        _resident_while_readonly_program,
        (x, limits, out, steps, 1 << (x.size(1) - 1).bit_length()),
    )
    config = bound.config_spec.default_config()
    config.config["cute_fragment_bounded_gather"] = True
    return bound, config


@pytest.mark.parametrize("width,steps", [(1, 2), (17, 0), (65, 2), (257, 1)])
def test_resident_while_readonly_load_assembly_and_sum(width, steps):
    from test.test_atomic_ops import _simulate_register_load_program
    from test.test_indexing import _asm_gather_model_source

    x = (torch.arange(3 * width).reshape(3, width) % 7).int()
    limits = torch.tensor([0, 1, 4], dtype=torch.int32)
    expected = torch.zeros_like(x)
    lane = torch.arange(width)
    for row in range(3):
        for iteration in range(limits[row].item()):
            for packet in range(steps):
                fresh = x[row, (lane + iteration + packet) % width]
                expected[row] = (expected[row] + fresh)[(lane * 3 + 1) % width]
    before = x.clone(), limits.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config = _resident_while_readonly_bound(
            x, limits, torch.empty_like(x), steps
        )
        code = bound.to_code(config)
    for order in (list(range(128)), list(reversed(range(128)))):
        out = torch.full_like(x, -1)
        checksums = torch.full_like(limits, -1)
        _simulate_register_load_program(
            _asm_gather_model_source(code),
            x,
            128,
            host_tensors={"limits": limits, "out": out, "checksums": checksums},
            lane_order=order,
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(checksums, expected.sum(-1).int(), rtol=0, atol=0)
    assert torch.equal(x, before[0]) and torch.equal(limits, before[1])


@pytest.mark.parametrize("alias", ["same", "view", "dlpack"])
def test_resident_while_readonly_rejects_aliased_output(alias):
    x = torch.ones((3, 17), dtype=torch.int32)
    out = x
    if alias == "view":
        out = x.view_as(x)
    elif alias == "dlpack":
        out = torch.utils.dlpack.from_dlpack(x)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config = _resident_while_readonly_bound(
            x, torch.tensor([0, 1, 4], dtype=torch.int32), out, 2
        )
        with pytest.raises(exc.InvalidConfig, match="proved bounded last-axis gather"):
            bound.to_code(config)


def test_resident_while_readonly_discovery_live_facts_before_snapshot():
    from helion._compiler.autotuner_heuristics import cute_fragment_bounded_gather

    original = cute_fragment_bounded_gather.bounded_gather_roots
    observed = []

    def discover(env, ir, *, allow_unbound=False):
        roots = original(env, ir, allow_unbound=allow_unbound)
        observed.append(
            (allow_unbound, bool(env.bound_runtime_input_specialization_results), roots)
        )
        return roots

    x = torch.ones((3, 1), dtype=torch.int32)
    limits = torch.tensor([0, 1, 4], dtype=torch.int32)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        with patch.object(
            cute_fragment_bounded_gather, "bounded_gather_roots", discover
        ):
            bound, config = _resident_while_readonly_bound(
                x, limits, torch.empty_like(x), 2
            )
        roots = bound.config_spec.cute_fragment_bounded_gather_root_ids
        assert roots and (True, False, roots) in observed
        assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
        assert "while fragment_while_take" in bound.to_code(config)


def test_resident_while_readonly_discovery_does_not_weaken_codegen_facts():
    from helion._compiler.autotuner_heuristics.cute_fragment_bounded_gather import (
        bounded_gather_roots,
    )
    from helion._compiler.cute.bounded_gather import prove_gather
    from helion._compiler.cute.computed_fragment import computed_fragment_supported

    x = torch.ones((3, 1), dtype=torch.int32)
    limits = torch.tensor([0, 1, 4], dtype=torch.int32)
    out = torch.empty_like(x)
    live = {"x": x, "limits": limits, "out": out, "steps": 2, "capacity": 1}
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config = _resident_while_readonly_bound(x, limits, out, 2)
        env = bound.env
        graphs = bound.host_function.device_ir.graphs
        gather = next(
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.gather.default
        )
        roots = bound.config_spec.cute_fragment_bounded_gather_root_ids
        assert roots
        with env, bound.host_function, env.use_runtime_arg_values(live):
            with patch.dict(
                env.bound_runtime_input_specialization_results, {}, clear=True
            ):
                assert not bounded_gather_roots(env, bound.host_function.device_ir)
                assert prove_gather(env, gather, graphs=graphs) is None
                assert not computed_fragment_supported(
                    env, graphs, bounded_gather_owned=True
                )
                assert (
                    bounded_gather_roots(
                        env, bound.host_function.device_ir, allow_unbound=True
                    )
                    == roots
                )
                assert (
                    prove_gather(env, gather, graphs=graphs, allow_unbound=True)
                    is not None
                )
            with patch.dict(env.runtime_input_specializations, {}, clear=True):
                assert not bounded_gather_roots(
                    env, bound.host_function.device_ir, allow_unbound=True
                )
        with (
            patch.dict(env.bound_runtime_input_specialization_results, {}, clear=True),
            pytest.raises(exc.InvalidConfig),
        ):
            bound.to_code(config)
        assert bound.config_spec.cute_fragment_bounded_gather_root_ids == roots


def test_resident_while_readonly_discovery_rejects_contradictory_live_alias():
    from helion._compiler.autotuner_heuristics.cute_fragment_bounded_gather import (
        bounded_gather_roots,
    )

    x = torch.ones((3, 1), dtype=torch.int32)
    limits = torch.tensor([0, 1, 4], dtype=torch.int32)
    out = torch.empty_like(x)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, _config = _resident_while_readonly_bound(x, limits, out, 2)
        env = bound.env
        roots = bound.config_spec.cute_fragment_bounded_gather_root_ids
        assert roots
        with (
            env,
            bound.host_function,
            env.use_runtime_arg_values(
                {"x": x, "limits": limits, "out": x, "steps": 2, "capacity": 1}
            ),
        ):
            assert not bounded_gather_roots(env, bound.host_function.device_ir)
            assert not bounded_gather_roots(
                env, bound.host_function.device_ir, allow_unbound=True
            )
        assert bound.config_spec.cute_fragment_bounded_gather_root_ids == roots


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_pure_region_diamond(x, depth: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :]
        for _iteration in hl.static_range(depth):
            word = hl.inline_asm_elementwise(
                "add.s32 $0, $1, 3;",
                "=r,r",
                [value],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            value = (word + 2) + (word - 2)
        out[row, :] = hl.cumsum(value, dim=0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_pure_region_owner(x, broadcast: hl.constexpr):
    if broadcast:
        out = torch.empty_like(x)
    else:
        out = torch.empty(
            (x.size(0), x.size(2), x.size(1)), device=x.device, dtype=x.dtype
        )
    for row in hl.grid(x.size(0)):
        if broadcast:
            seed = hl.full([1], 0, dtype=torch.int32)
        else:
            seed = x[row, :, :]
        owner = hl.inline_asm_elementwise(
            "mov.u32 $0, %tid.x;",
            "=r,r",
            [seed],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        value = owner + 1
        if broadcast:
            out[row, :] = hl.cumsum(x[row, :] + value, dim=0)
        else:
            out[row, :, :] = hl.cumsum(value.transpose(0, 1), dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_pure_region_alias(x):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :]
        word = hl.inline_asm_elementwise(
            "add.s32 $0, $1, 3;",
            "=r,r",
            [value],
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        adjusted = word + 1
        x[row, :] = value * 0
        out[row, :] = hl.cumsum(adjusted, dim=0)
    return out


def _pure_region_model_source(code):
    """Interpret only the two declared test ASM instructions, with their casts."""

    class Model(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) != "_cute_inline_asm_elementwise":
                return node
            keywords = {item.arg: item.value for item in node.keywords}
            assert ast.literal_eval(keywords["is_pure"]) is True
            assert ast.unparse(keywords["dtype"]) == "cutlass.Int32"
            assert ast.literal_eval(keywords["constraints"]) == "=r,r"
            instruction = ast.literal_eval(keywords["asm"])
            if instruction == "mov.u32 $0, %tid.x;":
                expression = ast.parse("cute.arch.thread_idx()[0]", mode="eval").body
            else:
                assert instruction == "add.s32 $0, $1, 3;"
                expression = ast.BinOp(node.args[0].elts[0], ast.Add(), ast.Constant(3))
            return ast.Call(keywords["dtype"], [expression], [])

    return ast.unparse(ast.fix_missing_locations(Model().visit(ast.parse(code))))


def _pure_region_codes(kernel, args, cache):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        assert bound.config_spec.cute_fragment_pure_producer_regions_root_ids
        config = bound.config_spec.default_config()
        config.config["cute_fragment_producer_cache"] = cache
        old = bound.to_code(config)
        config.config["cute_fragment_pure_producer_regions"] = False
        assert bound.to_code(config) == old
        config.config["cute_fragment_pure_producer_regions"] = True
        generation = bound.config_spec.create_config_generation()
        flat, canonical = generation.strict_config_pair(config)
        assert generation.strict_config_pair(canonical)[0] == flat
        new = bound.to_code(canonical)
    assert new.count("_cute_inline_asm_elementwise(") == old.count(
        "_cute_inline_asm_elementwise("
    )
    assert new.count("cute.arch.sync_threads()") <= old.count(
        "cute.arch.sync_threads()"
    )
    return old, new


@pytest.mark.parametrize("columns,depth", [(17, 2), (65, 4), (129, 8)])
def test_fragment_pure_producer_regions_diamond_and_wrap(columns, depth):
    from test.test_atomic_ops import _simulate_register_load_program

    x = (
        torch.arange(2 * columns, dtype=torch.int64).reshape(2, columns) + 2147483600
    ).int()
    codes = _pure_region_codes(_fragment_pure_region_diamond, (x, depth), False)
    assert codes[1].count("cute.arch.sync_threads()") < codes[0].count(
        "cute.arch.sync_threads()"
    )
    expected = x.clone()
    for _iteration in range(depth):
        expected = (expected + 3) * 2
    expected = expected.cumsum(-1).int()
    for code, reverse in itertools.product(codes, (False, True)):
        out = torch.full_like(x, -999)
        _simulate_register_load_program(
            _pure_region_model_source(code),
            x.clone(),
            128,
            host_tensors={"out": out},
            lane_order=list(reversed(range(128))) if reverse else list(range(128)),
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,broadcast", [((2, 65), True), ((2, 4, 8), False), ((2, 8, 8), False)]
)
def test_fragment_pure_producer_regions_preserve_opaque_owner(shape, broadcast):
    from test.test_atomic_ops import _simulate_register_load_program

    x = torch.ones(shape, dtype=torch.int32)
    codes = _pure_region_codes(_fragment_pure_region_owner, (x, broadcast), False)
    expected = (
        (x * 2).cumsum(-1).int()
        if broadcast
        else (
            (torch.arange(math.prod(shape[1:])).reshape(shape[1:]).T + 1)
            .cumsum(-1)
            .expand(shape[0], shape[2], shape[1])
            .int()
        )
    )
    for code, reverse in itertools.product(codes, (False, True)):
        out = torch.full_like(expected, -999)
        _simulate_register_load_program(
            _pure_region_model_source(code),
            x,
            128,
            host_tensors={"out": out},
            lane_order=list(reversed(range(128))) if reverse else list(range(128)),
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fragment_pure_producer_regions_alias_snapshot():
    from test.test_atomic_ops import _simulate_register_load_program

    x = torch.arange(2 * 65).reshape(2, 65).int()
    codes = _pure_region_codes(_fragment_pure_region_alias, (x,), False)
    expected = (x + 4).cumsum(-1).int()
    for code, reverse in itertools.product(codes, (False, True)):
        current = x.clone()
        out = torch.full_like(x, -999)
        _simulate_register_load_program(
            _pure_region_model_source(code),
            current,
            128,
            host_tensors={"out": out},
            lane_order=list(reversed(range(128))) if reverse else list(range(128)),
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert torch.count_nonzero(current) == 0


def test_fragment_pure_producer_regions_plan_boundaries():
    import operator

    from helion._compiler.cute.pure_producer_regions import pure_producer_plan
    from helion.language import _tracing_ops
    from helion.language import atomic_ops
    from helion.language import inline_asm_ops
    from helion.language import memory_ops

    env = SimpleNamespace(known_equal=operator.eq)
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(65, dtype=torch.int32)
    asm = graph.call_function(
        inline_asm_ops.inline_asm_elementwise,
        ("mov.u32 $0, %tid.x;", "=r,r", [source], torch.int32, True, 1),
    )
    asm.meta["val"] = source.meta["val"]
    value = graph.call_function(torch.ops.aten.add.Tensor, (asm, 1))
    value.meta["val"] = source.meta["val"]
    reduced = graph.call_function(torch.ops.aten.sum.default, (value,))
    reduced.meta["val"] = torch.empty((), dtype=torch.int32)
    later = graph.call_function(torch.ops.aten.add.Tensor, (value, reduced))
    later.meta["val"] = source.meta["val"]
    graph.output(later)
    plan = pure_producer_plan(graph, env)
    assert plan.lazy == frozenset((asm,))
    assert plan.publications == frozenset((value,))
    # Same padding is not a proof of equal logical owners.
    value.meta["val"] = torch.empty(128, dtype=torch.int32)
    assert not pure_producer_plan(graph, env, shape=lambda sizes: (128,)).lazy
    value.meta["val"] = source.meta["val"]
    assert not pure_producer_plan(graph, env, shape=lambda sizes: (0,)).lazy
    assert not pure_producer_plan(graph, env, cache_nodes=frozenset((asm,))).lazy
    for operation in (
        _tracing_ops._for_loop,
        _tracing_ops._while_loop,
        _tracing_ops._if,
        memory_ops.load,
        memory_ops.store,
        atomic_ops.atomic_add,
    ):
        with graph.inserting_before(value):
            boundary = graph.call_function(operation, ())
        assert not pure_producer_plan(graph, env).lazy
        graph.erase_node(boundary)
    original = asm.args
    for pure, pack in ((False, 1), (True, 2)):
        asm.args = (*original[:4], pure, pack)
        assert not pure_producer_plan(graph, env).lazy
    asm.args = original


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_uniform_region(
    x, seed, mutate: hl.constexpr, opaque: hl.constexpr = False
):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        key = seed[0]
        if opaque:
            key = hl.inline_asm_elementwise(
                "mov.u32 $0, %tid.x;",
                "=r,r",
                [key],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
        value = x[row, :]
        if mutate:
            seed[0] = key + 1
        for _iteration in hl.static_range(12):
            key = (key + 17) ^ (key >> 3)
            word = hl.inline_asm_elementwise(
                "add.s32 $0, $1, 3;",
                "=r,r",
                [value + key],
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            value = word ^ key
        out[row, :] = hl.cumsum(value, dim=0)
    return out


@pytest.mark.parametrize("columns", [17, 65, 129])
def test_fragment_uniform_producer_regions_typed_snapshot(columns):
    from test.test_atomic_ops import _simulate_register_load_program

    x = torch.arange(columns, dtype=torch.int32).reshape(1, columns)
    seed = torch.tensor([2147483600], dtype=torch.int32)
    codes = _pure_region_codes(_fragment_uniform_region, (x, seed, False), True)
    assert codes[1].count("cute.arch.sync_threads()") < codes[0].count(
        "cute.arch.sync_threads()"
    )
    expected = x.clone()
    key = seed[0]
    for _iteration in range(12):
        key = (key + 17) ^ (key >> 3)
        expected = ((expected + key) + 3) ^ key
    expected = expected.cumsum(-1).int()
    for code, reverse in itertools.product(codes, (False, True)):
        out = torch.full_like(x, -999)
        _simulate_register_load_program(
            _pure_region_model_source(code),
            x,
            128,
            host_tensors={"seed": seed.clone(), "out": out},
            lane_order=list(reversed(range(128))) if reverse else list(range(128)),
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fragment_uniform_producer_regions_mutable_seed_declines():
    from helion._compiler.cute import computed_fragment

    x = torch.arange(65, dtype=torch.int32).reshape(1, 65)
    seed = torch.tensor([19], dtype=torch.int32)
    original = computed_fragment.pure_producer_plan
    plans = []

    def record(*args, **kwargs):
        plan = original(*args, **kwargs)
        if kwargs.get("cache_nodes"):
            plans.append(plan)
        return plan

    with patch.object(computed_fragment, "pure_producer_plan", record):
        _pure_region_codes(_fragment_uniform_region, (x, seed, True), True)
    assert plans and all(not plan.replicated for plan in plans)


def test_fragment_uniform_producer_regions_scalar_asm_is_not_uniform():
    from test.test_atomic_ops import _simulate_register_load_program

    from helion._compiler.cute import computed_fragment

    x = torch.arange(65, dtype=torch.int32).reshape(1, 65)
    seed = torch.tensor([19], dtype=torch.int32)
    original = computed_fragment.pure_producer_plan
    plans = []

    def record(*args, **kwargs):
        plan = original(*args, **kwargs)
        if kwargs.get("cache_nodes"):
            plans.append(plan)
        return plan

    with patch.object(computed_fragment, "pure_producer_plan", record):
        codes = _pure_region_codes(
            _fragment_uniform_region, (x, seed, False, True), True
        )
    assert plans and all(not plan.replicated for plan in plans)
    key = torch.tensor(0, dtype=torch.int32)
    expected = x.clone()
    for _iteration in range(12):
        key = (key + 17) ^ (key >> 3)
        expected = ((expected + key) + 3) ^ key
    expected = expected.cumsum(-1).int()
    for code, reverse in itertools.product(codes, (False, True)):
        out = torch.full_like(x, -999)
        _simulate_register_load_program(
            _pure_region_model_source(code),
            x,
            128,
            host_tensors={"seed": seed.clone(), "out": out},
            lane_order=list(reversed(range(128))) if reverse else list(range(128)),
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fragment_uniform_producer_regions_cache_and_owner_boundaries():
    import operator

    from helion._compiler.cute.pure_producer_regions import pure_producer_plan
    from helion.language import inline_asm_ops
    from helion.language import memory_ops

    env = SimpleNamespace(known_equal=operator.eq)
    graph = torch.fx.Graph()
    vector = graph.placeholder("vector")
    vector.meta["val"] = torch.empty(65, dtype=torch.int32)
    scalar = graph.call_function(torch.ops.aten.scalar_tensor.default, (17,))
    scalar.meta["val"] = torch.empty((), dtype=torch.int32)
    key = graph.call_function(torch.ops.aten.add.Tensor, (scalar, 1))
    key.meta["val"] = scalar.meta["val"]
    mixed = graph.call_function(torch.ops.aten.add.Tensor, (vector, key))
    mixed.meta["val"] = vector.meta["val"]
    word = graph.call_function(
        inline_asm_ops.inline_asm_elementwise,
        ("mov.u32 $0, %tid.x;", "=r,r", [mixed], torch.int32, True, 1),
    )
    word.meta["val"] = vector.meta["val"]
    output = graph.call_function(torch.ops.aten.add.Tensor, (word, key))
    output.meta["val"] = vector.meta["val"]
    reduced = graph.call_function(torch.ops.aten.sum.default, (output,))
    reduced.meta["val"] = scalar.meta["val"]
    later = graph.call_function(torch.ops.aten.add.Tensor, (reduced, 1))
    later.meta["val"] = scalar.meta["val"]
    graph.output(later)
    caches = frozenset((key, later))

    def plan():
        return pure_producer_plan(graph, env, cache_nodes=caches, shape=tuple)

    assert plan().replicated == frozenset((key,))
    assert plan().lazy == frozenset((word,))
    assert plan().publications == frozenset((output,))
    # Pure opaque scalar code is not an immutable uniform recipe.
    scalar.target = inline_asm_ops.inline_asm_elementwise
    original = scalar.args
    scalar.args = ("mov.u32 $0, %tid.x;", "=r", [], torch.int32, True, 1)
    assert not plan().replicated
    scalar.target = torch.ops.aten.scalar_tensor.default
    scalar.args = original
    # Remapping a row owner and an escaping second publication both retain it.
    old_target, old_args = output.target, output.args
    output.target, output.args = torch.ops.aten.permute.default, (word, [0])
    assert not plan().replicated
    output.target, output.args = old_target, old_args
    with graph.inserting_before(reduced):
        escaped = graph.call_function(torch.ops.aten.add.Tensor, (key, 99))
        escaped.meta["val"] = scalar.meta["val"]
    assert not plan().replicated
    graph.erase_node(escaped)
    for target in (memory_ops.store, memory_ops.load):
        with graph.inserting_before(word):
            effect = graph.call_function(target, ())
        assert not plan().replicated
        graph.erase_node(effect)
    # Same padding is not proof of equal logical or physical owners.
    mixed.meta["val"] = torch.empty(64, dtype=torch.int32)
    assert not plan().replicated


@pytest.mark.parametrize("strategy_name", ["FROM_RANDOM", "FROM_BEST_AVAILABLE"])
def test_fragment_pure_producer_regions_prefix_and_rng(strategy_name):
    import random

    from test.test_compiler_coverage import make_search

    from helion.autotuner.pattern_search import InitialPopulationStrategy

    def bind():
        return _cpu_bind(
            helion.kernel(
                _fragment_pure_region_diamond.fn, backend="cute", static_shapes=True
            ),
            (torch.ones((2, 65), dtype=torch.int32), 4),
        )

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        with patch(
            "helion._compiler.autotuner_heuristics.register_fragment_pure_producer_regions_coverage"
        ):
            previous = bind()
        current = bind()
    strategy = InitialPopulationStrategy[strategy_name]
    old = make_search(previous.config_spec, count=20, strategy=strategy)
    new = make_search(current.config_spec, count=20, strategy=strategy)
    for seed in (73, 741, 2031):
        random.seed(seed)
        prior = old._generate_initial_population_flat()
        state = random.getstate()
        random.seed(seed)
        rows = new._generate_initial_population_flat()
        assert random.getstate() == state
        expected = [old.config_gen.unflatten(row) for row in prior]
        actual = [new.config_gen.unflatten(row) for row in rows]
        assert actual[: len(expected)] == expected
        assert len(actual) == len(expected) + 1
        assert actual[-1]["cute_fragment_pure_producer_regions"] is True
    assert previous.config_spec.default_config() == current.config_spec.default_config()
    assert (
        previous.config_spec.compiler_seed_configs
        == current.config_spec.compiler_seed_configs
    )


def test_fragment_pure_producer_regions_strict_config():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_uncached_cheap, (torch.ones((3, 65)),))
        original = bound.to_code(bound.config_spec.default_config())
        for value in (True, 1, None, "region"):
            config = bound.config_spec.default_config()
            config.config["cute_fragment_pure_producer_regions"] = value
            before = dict(config.config)
            with pytest.raises(exc.InvalidConfig, match="same-owner"):
                bound.to_code(config)
            assert config.config == before
        config = bound.config_spec.default_config()
        config.config["cute_fragment_pure_producer_regions"] = False
        assert bound.to_code(config) == original


def test_fragment_pure_producer_regions_nested_root_declines():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        x = torch.ones((3, 16), dtype=torch.int32)
        bound, config = _resident_while_readonly_bound(
            x, torch.tensor([0, 1, 3], dtype=torch.int32), torch.empty_like(x), 2
        )
        assert not bound.config_spec.cute_fragment_pure_producer_regions_root_ids
        config.config["cute_fragment_pure_producer_regions"] = True
        with pytest.raises(exc.InvalidConfig, match="same-owner"):
            bound.to_code(config)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("owner", [False, True])
def test_fragment_pure_producer_regions_native(owner):
    x = torch.arange(2 * 65, dtype=torch.int32, device=DEVICE).reshape(2, 65)
    if owner:
        bound = _fragment_pure_region_owner.bind((x, True))
        args = (x, True)
        expected = (x + 1).cumsum(-1).int()
    else:
        bound = _fragment_pure_region_diamond.bind((x, 4))
        args = (x, 4)
        expected = x.clone()
        for _iteration in range(4):
            expected = (expected + 3) * 2
        expected = expected.cumsum(-1).int()
    config = bound.config_spec.default_config()
    config.config["cute_fragment_pure_producer_regions"] = True
    actual = bound.compile_config(config)(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("opaque", [False, True])
def test_fragment_uniform_producer_regions_native(opaque):
    x = torch.arange(129, dtype=torch.int32, device=DEVICE).reshape(1, 129)
    seed = torch.tensor([2147483600], dtype=torch.int32, device=DEVICE)
    args = (x, seed, False, opaque)
    bound = _fragment_uniform_region.bind(args)
    config = bound.config_spec.default_config()
    config.config["cute_fragment_pure_producer_regions"] = True
    config.config["cute_fragment_producer_cache"] = True
    actual = bound.compile_config(config)(*args)
    key = torch.zeros((), dtype=torch.int32, device=DEVICE) if opaque else seed[0]
    expected = x.clone()
    for _iteration in range(12):
        key = (key + 17) ^ (key >> 3)
        expected = ((expected + key) + 3) ^ key
    torch.testing.assert_close(actual, expected.cumsum(-1).int(), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_where_logical_tail(
    x: torch.Tensor, idx: torch.Tensor, form: hl.constexpr, kind: hl.constexpr
):
    out = torch.empty((idx.size(0),), device=x.device, dtype=x.dtype)
    flat = idx.flatten()
    width = hl.specialize(idx.size(1))
    for row in hl.tile(idx.size(0), block_size=1):
        j = hl.arange(width)
        indices = flat[row.index[:, None] * width + j[None, :]]
        if form == "masked":
            values = hl.load(x, [indices], extra_mask=j[None, :] % 3 != 0)
        else:
            values = x[indices]
        if form == "condition":
            selected = torch.where(values > 0, 3.0, 7.0)
        elif form == "branch":
            selected = torch.where(
                torch.full((), True, device=x.device), values + 2, 1.0
            )
        elif form == "nested":
            transposed = values.transpose(0, 1)
            selected = torch.where(
                j[:, None] % 2 == 0, transposed + 2, transposed + 1
            ).transpose(0, 1)
            selected = torch.where(selected > 0, selected * 2, selected - 3) + 1
        else:
            selected = torch.where(j[None, :] % 2 == 0, values + 2, values + 1)
        if kind == "min":
            reduced = selected.amin(-1)
        elif kind == "max":
            reduced = selected.amax(-1)
        else:
            reduced = selected.sum(-1)
        out[row] = hl.cumsum(reduced, dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_where_explicit_padding(x: torch.Tensor, extent: hl.constexpr):
    width = hl.specialize(x.size(1))
    out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0), block_size=1):
        # Here padding is explicitly part of the user's tensor, including the
        # false load-mask positions. They must contribute the where constants.
        j = hl.arange(extent)
        values = hl.load(x, [row, j], extra_mask=j[None, :] < width)
        selected = torch.where(j[None, :] % 2 == 0, values + 2, values + 1)
        out[row] = hl.cumsum(selected.sum(-1), dim=-1)
    return out


def _where_tail_inputs(width, device="cpu"):
    x = (torch.arange(43, device=device, dtype=torch.float32) % 7) - 3
    idx = (torch.arange(3 * width, device=device, dtype=torch.int32) * 5 % 43).reshape(
        3, width
    )
    return x, idx


def _where_tail_reference(x, idx, form, kind):
    j = torch.arange(idx.size(1), device=x.device)[None, :]
    values = x[idx.long()]
    if form == "masked":
        values = torch.where(j % 3 != 0, values, 0)
    if form == "condition":
        selected = torch.where(values > 0, 3.0, 7.0)
    elif form == "branch":
        selected = values + 2
    else:
        selected = torch.where(j % 2 == 0, values + 2, values + 1)
        if form == "nested":
            selected = torch.where(selected > 0, selected * 2, selected - 3) + 1
    if kind == "min":
        return selected.amin(-1)
    if kind == "max":
        return selected.amax(-1)
    return selected.sum(-1)


def _where_tail_config(bound, mode):
    config = bound.config_spec.default_config()
    config.config["cute_fragment_reduction"] = mode
    # Keep the independent reduction model on its supported scalar owner path.
    config.config["cute_fragment_producer_cache"] = False
    _, config = bound.config_spec.create_config_generation().strict_config_pair(config)
    return config


@pytest.mark.parametrize("width", [17, 33, 65])
@pytest.mark.parametrize(
    "form", ["indirect", "condition", "branch", "nested", "masked"]
)
def test_fragment_where_implicit_tail_cpu(width, form):
    x, idx = _where_tail_inputs(width)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        for kind in ("sum", "min", "max"):
            args = (x, idx, form, kind)
            expected = _where_tail_reference(*args)
            eager = helion.kernel(
                _fragment_where_logical_tail.fn,
                ref_mode="eager",
                backend="cute",
                static_shapes=True,
            )
            torch.testing.assert_close(
                _cpu_bind(eager, args).run_ref(*args), expected, rtol=0, atol=0
            )
            bound = _cpu_bind(_fragment_where_logical_tail, args)
            for mode in ("serial", "warp"):
                code = bound.to_code(_where_tail_config(bound, mode))
                actual = torch.full_like(expected, -999)
                _simulate_fragment_warp_reduction(
                    code, {"x": x, "flat": idx.flatten()}, {"out": actual}, 3
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [17, 33, 65])
def test_fragment_where_explicit_padding_cpu(width):
    x = torch.zeros((3, width))
    extent = 1 << (width - 1).bit_length()
    expected = torch.full((3,), 1.5 * extent)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        eager = helion.kernel(
            _fragment_where_explicit_padding.fn,
            ref_mode="eager",
            backend="cute",
            static_shapes=True,
        )
        torch.testing.assert_close(
            _cpu_bind(eager, (x, extent)).run_ref(x, extent), expected, rtol=0, atol=0
        )
        bound = _cpu_bind(_fragment_where_explicit_padding, (x, extent))
        for mode in ("serial", "warp"):
            code = bound.to_code(_where_tail_config(bound, mode))
            actual = torch.full_like(expected, -999)
            _simulate_fragment_warp_reduction(code, {"x": x}, {"out": actual}, 3)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "form", ["indirect", "condition", "branch", "nested", "masked"]
)
def test_fragment_where_implicit_tail_native(form):
    x, idx = _where_tail_inputs(17, "cuda")
    for mode, kind in itertools.product(("serial", "warp"), ("sum", "min", "max")):
        args = (x, idx, form, kind)
        bound = _fragment_where_logical_tail.bind(args)
        actual = bound.compile_config(_where_tail_config(bound, mode))(*args)
        torch.testing.assert_close(actual, _where_tail_reference(*args), rtol=0, atol=0)


@skipUnlessBackends(["cute"])
def test_fragment_where_explicit_padding_native():
    x = torch.zeros((3, 17), device=DEVICE)
    bound = _fragment_where_explicit_padding.bind((x, 32))
    for mode in ("serial", "warp"):
        actual = bound.compile_config(_where_tail_config(bound, mode))(x, 32)
        torch.testing.assert_close(
            actual, torch.full((3,), 48.0, device=DEVICE), rtol=0, atol=0
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_where_adversarial_tail(
    x: torch.Tensor, idx: torch.Tensor, form: hl.constexpr, kind: hl.constexpr
):
    out = torch.empty((idx.size(0),), device=x.device, dtype=x.dtype)
    flat = idx.flatten()
    width = hl.specialize(idx.size(1))
    for row in hl.tile(idx.size(0), block_size=1):
        j = hl.arange(width)
        values = x[flat[row.index[:, None] * width + j[None, :]]]
        if form == "rhs":
            selected = torch.where(
                torch.full((), False, device=x.device), 1.0, values + 2
            )
        elif form == "condition":
            selected = torch.where(values == 0, 3.0, 7.0)
        else:
            selected = torch.where(j[None, :] % 2 == 0, values + 2, values + 1)
        if kind == "min":
            reduced = selected.amin(-1)
        else:
            reduced = selected.amax(-1)
        out[row] = hl.cumsum(reduced, dim=-1)
    return out


@pytest.mark.parametrize(
    "form,kind,value,expected_value",
    [
        ("parity", "max", -10.0, -8.0),
        ("parity", "min", 10.0, 11.0),
        ("rhs", "max", -10.0, -8.0),
        ("rhs", "min", 10.0, 12.0),
        ("condition", "min", 10.0, 7.0),
    ],
)
def test_fragment_where_adversarial_tail_cpu(form, kind, value, expected_value):
    # Zero-valued padding wins max over negative inputs and min over positive
    # inputs unless every where operand preserves the logical reduction tail.
    x, idx = _where_tail_inputs(17)
    x.fill_(value)
    args = (x, idx, form, kind)
    expected = torch.full((3,), expected_value)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        eager = helion.kernel(
            _fragment_where_adversarial_tail.fn,
            ref_mode="eager",
            backend="cute",
            static_shapes=True,
        )
        torch.testing.assert_close(
            _cpu_bind(eager, args).run_ref(*args), expected, rtol=0, atol=0
        )
        bound = _cpu_bind(_fragment_where_adversarial_tail, args)
        for mode in ("serial", "warp"):
            code = bound.to_code(_where_tail_config(bound, mode))
            actual = torch.full_like(expected, -999)
            _simulate_fragment_warp_reduction(
                code, {"x": x, "flat": idx.flatten()}, {"out": actual}, 3
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_where_explicit_padding_min(x: torch.Tensor, extent: hl.constexpr):
    width = hl.specialize(x.size(1))
    out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0), block_size=1):
        j = hl.arange(extent)
        values = hl.load(x, [row, j], extra_mask=j[None, :] < width)
        selected = torch.where(j[None, :] % 2 == 0, values + 2, values + 1)
        out[row] = hl.cumsum(selected.amin(-1), dim=-1)
    return out


def test_fragment_where_explicit_padding_min_cpu():
    # Explicitly requested positions beyond x still contribute 1 or 2. The
    # memory mask must not turn them into a logical reduction tail.
    x = torch.full((3, 17), 10.0)
    args = (x, 32)
    expected = torch.ones(3)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        eager = helion.kernel(
            _fragment_where_explicit_padding_min.fn,
            ref_mode="eager",
            backend="cute",
            static_shapes=True,
        )
        torch.testing.assert_close(
            _cpu_bind(eager, args).run_ref(*args), expected, rtol=0, atol=0
        )
        bound = _cpu_bind(_fragment_where_explicit_padding_min, args)
        for mode in ("serial", "warp"):
            code = bound.to_code(_where_tail_config(bound, mode))
            actual = torch.full_like(expected, -999)
            _simulate_fragment_warp_reduction(code, {"x": x}, {"out": actual}, 3)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
