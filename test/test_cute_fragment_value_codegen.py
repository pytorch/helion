"""CPU code generation and replay checks for fragment values and memory bounds."""

from __future__ import annotations

import ast
from ctypes import c_int32

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
from helion._testing.cute_fragment_simulator import (
    simulate_independent_fragment as _simulate_independent_fragment,
)
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _adjacent_window_scan(x, base_ptr, out, width: hl.constexpr, start: hl.constexpr):
    for row in hl.grid(1):
        i = hl.arange(start, start + width, dtype=torch.int64)
        if start:
            base = base_ptr[row]
            left = x[base + i]
            right = x[base + (i + width)]
        else:
            left = x[i]
            right = x[i + width]
        out[i - start] = torch.cumsum(left + right, dim=0)


@pytest.mark.parametrize("width,start", [(3, 0), (4, 2**31 - 2)])
def test_adjacent_fragment_windows_keep_their_lanes(width, start):
    x = torch.arange(1, 2 * width + 1, dtype=torch.int64)
    base = torch.tensor([-start])
    backing = torch.full((width + 2,), -73, dtype=torch.int64)
    out = backing[1:-1]
    expected = (x[:width] + x[width:]).cumsum(0)
    with mock_cuda_unavailable():
        bound = cpu_bind(_adjacent_window_scan, (x, base, out, width, start))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        sources = [
            bound.to_code(
                helion.Config(block_sizes=[], cute_fragment_slab_loads=stage),
                output_origin_lines=False,
            )
            for stage in (False, True)
        ]
    for source in sources:
        if start:
            # Python's unbounded loop indices cannot model a device Int32
            # lane. Check the addition itself sees a wide operand on both paths.
            crossing = [
                expr
                for expr in ast.walk(ast.parse(source))
                if isinstance(expr, ast.BinOp)
                and isinstance(expr.op, ast.Add)
                and isinstance(expr.left, ast.Constant)
                and expr.left.value == start
            ]
            assert all("cutlass.Int64(" in ast.unparse(expr.right) for expr in crossing)
            assert crossing or f"cutlass.Int64({start}) +" in source
        out.fill_(-73)
        _simulate_independent_fragment(
            source, {"x": x, "base_ptr": base}, {"out": out}, 1
        )
        torch.testing.assert_close(out, expected)
        assert backing[[0, -1]].tolist() == [-73, -73]


@pytest.mark.parametrize("width", [4097, 8193])
def test_padded_slab_windows_materialize_only_the_sum(width):
    x = torch.zeros(2 * width, dtype=torch.int64)
    out = torch.zeros(width, dtype=torch.int64)
    with _target():
        bound = cpu_bind(_adjacent_window_scan, (x, torch.tensor([0]), out, width, 0))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        cfg = helion.Config(
            block_sizes=[], cute_fragment_slab_loads=True, cute_fragment_threads=64
        )
        if width == 8193:
            with pytest.raises(
                exc.InvalidConfig, match="shared bytes, exceeding 232448"
            ):
                bound.to_code(cfg)
        else:
            # The two windows share the slab; adding two separate padded
            # copies would push this scan above the same target's capacity.
            source = bound.to_code(cfg, output_origin_lines=False)
            assert "cute.make_layout((8194,))" in source


@helion.kernel(backend="cute", static_shapes=True)
def _padded_slab_consumer(x, out, scatter: hl.constexpr):
    for _row in hl.grid(1):
        i = hl.arange(3, dtype=torch.int64)
        left = x[i]
        right = x[i + 3]
        prefix = torch.cumsum(left + right, 0)
        if scatter:
            out[left] = prefix
        else:
            out[i] = torch.cumsum(left, 0) + prefix


@pytest.mark.parametrize("scatter", [False, True])
def test_padded_slab_members_keep_their_bounds_in_consumers(scatter):
    x = torch.arange(1, 7, dtype=torch.int64)
    backing = torch.full((6,), -73, dtype=torch.int64)
    out = backing[1:-1]
    expected = [-73, -73, 5, 12, 21, -73] if scatter else [-73, 6, 15, 27, -73, -73]
    with mock_cuda_unavailable():
        bound = cpu_bind(_padded_slab_consumer, (x, out, scatter))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        for stage in (False, True):
            source = bound.to_code(
                helion.Config(block_sizes=[], cute_fragment_slab_loads=stage),
                output_origin_lines=False,
            )
            # Both windows of x must share one staged load; the canary must
            # also pass when the two ordinary loads are selected.
            assert source.count(".load()") == (1 if stage else 2)
            backing.fill_(-73)
            _simulate_independent_fragment(source, {"x": x}, {"out": out}, 1)
            assert backing.tolist() == expected


@helion.kernel(backend="cute", static_shapes=True)
def _empty_window_and_scan(x, out):
    for _row in hl.grid(1):
        empty = hl.arange(0, dtype=torch.int64)
        i = hl.arange(4, dtype=torch.int64)
        out[i] = torch.cumsum(x[i] + 1, dim=0)
        # Keep the empty load live past its own index's final FX use.
        out[hl.arange(0, dtype=torch.int32)] = x[empty]


def test_empty_tensor_index_keeps_its_axis_next_to_staged_windows():
    x = torch.arange(1, 5, dtype=torch.int64)
    backing = torch.full((6,), -73, dtype=torch.int64)
    out = backing[1:-1]
    with mock_cuda_unavailable():
        bound = cpu_bind(_empty_window_and_scan, (x, out))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        for staged in (False, True):
            source = bound.to_code(
                helion.Config(block_sizes=[], cute_fragment_slab_loads=staged),
                output_origin_lines=False,
            )
            assert source.count(".load()") == (1 if staged else 2)
            backing.fill_(-73)
            _simulate_independent_fragment(source, {"x": x}, {"out": out}, 1)
            assert backing.tolist() == [-73, 2, 5, 9, 14, -73]


@helion.kernel(backend="cute", static_shapes=True)
def _symbolic_iota_scan(out):
    for row in hl.grid(3):
        start = (2**31 - 2) + row
        i = hl.arange(start, start + 4, dtype=torch.int64)
        out[row, :] = torch.cumsum(i, 0)


def test_fragment_iota_widens_the_symbolic_start_before_adding():
    backing = torch.full((3, 6), -73, dtype=torch.int64)
    out = backing[:, 1:-1]
    with mock_cuda_unavailable():
        source = cpu_bind(_symbolic_iota_scan, (out,)).to_code(
            helion.Config(block_sizes=[]), output_origin_lines=False
        )
    # A wide final cast cannot recover the third row's Int32 addition.
    (start,) = (
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and isinstance(node.left, ast.Constant)
        and node.left.value == 2**31 - 2
    )
    assert (
        isinstance(start.right, ast.Call)
        and ast.unparse(start.right.func) == "cutlass.Int64"
    )
    _simulate_independent_fragment(
        source, {}, {"out": out}, 3, int32=np.int32, int64=np.int64
    )
    expected = ((2**31 - 2) + torch.arange(3)[:, None] + torch.arange(4)).cumsum(1)
    torch.testing.assert_close(out, expected)
    assert backing[:, [0, -1]].tolist() == [[-73, -73]] * 3


def test_hoisted_slab_observes_source_shape_before_branch():
    @helion.kernel(backend="cute", static_shapes=True)
    def prefix(x, out):
        for row in hl.tile(out.size(0), block_size=1):
            if row.begin < x.size(0):
                out[row, :] = torch.cumsum(x[row, 0, :] + x[row, 1, :], dim=-1)
        return out

    x = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 4)
    out = torch.empty(3, 4)
    expected = torch.full_like(out, -73)
    expected[:2] = (x[:, 0, :] + x[:, 1, :]).cumsum(-1)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(prefix, (x, out))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        for staged, packets in ((False, False), (True, False), (True, True)):
            source = bound.to_code(
                helion.Config(
                    block_sizes=[],
                    cute_fragment_slab_loads=staged,
                    cute_fragment_vector_io=packets,
                )
            )
            if packets:
                # The 128-bit copy must have the physical row predicate too;
                # the serial phase simulator models the scalar slab walk.
                (guard,) = (
                    n.test
                    for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.If)
                    and any(
                        isinstance(stmt, ast.Expr)
                        and "cute.copy(" in ast.unparse(stmt)
                        and "x.iterator" in ast.unparse(stmt)
                        for stmt in n.body
                    )
                )
                assert any(
                    isinstance(n, ast.Compare)
                    and isinstance(n.ops[0], ast.Lt)
                    and ast.literal_eval(n.comparators[0]) == x.size(0)
                    for n in ast.walk(guard)
                    if isinstance(n, ast.Compare)
                    and isinstance(n.comparators[0], ast.Constant)
                )
            else:
                out.fill_(-73)
                _simulate_independent_fragment(source, {"x": x}, {"out": out}, 3)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("columns", [(2, 3), (-2, -1)])
def test_slab_loads_with_inserted_axis_keep_their_shape(columns):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan_pair(x, out, lo: hl.constexpr, hi: hl.constexpr):
        for i in hl.grid(x.size(0)):
            left = x[i, None, lo]
            right = x[i, None, hi]
            # The adjacent plain columns still qualify for a slab.
            out[i, :] = torch.cumsum(left + right, dim=0) + x[i, 0] + x[i, 1]
        return out

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = torch.full((3, 1), -73.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(scan_pair, (x, out, *columns))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        source = bound.to_code(
            helion.Config(block_sizes=[], cute_fragment_slab_loads=True)
        )
    _simulate_independent_fragment(source, {"x": x}, {"out": out}, 3)
    torch.testing.assert_close(out, x.sum(dim=1, keepdim=True), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("pos", "index_dtype", "mask_dtype"),
    [
        (-1, torch.int32, torch.int32),
        (1, torch.int64, torch.int64),
        (-1, torch.int32, torch.int64),
    ],
)
def test_slab_offset_and_limit_follow_vector_promotion(pos, index_dtype, mask_dtype):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan_windows(
        x,
        base_ptr,
        limit_ptr,
        out,
        pos: hl.constexpr,
        dtype: hl.constexpr,
        mask_type: hl.constexpr,
    ):
        for row in hl.grid(x.size(0)):
            base = base_ptr[pos]
            limit = limit_ptr[pos]
            i0 = hl.arange(4, dtype=dtype)
            i1 = 4 + i0
            m0 = hl.arange(4, dtype=mask_type)
            low = hl.load(x, [row, base + i0], extra_mask=m0 < limit)
            high = hl.load(x, [row, base + i1], extra_mask=4 + m0 < limit)
            out[row, :] = torch.cumsum(low + high, dim=0)
        return out

    x = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    base, limit = torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)
    out = torch.empty(2, 4)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(
            scan_windows, (x, base, limit, out, pos, index_dtype, mask_dtype)
        )
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        sources = [
            bound.to_code(
                helion.Config(block_sizes=[], cute_fragment_slab_loads=enabled)
            )
            for enabled in (False, True)
        ]
    assert (
        "fragment_slab_scalar" not in sources[0]
        and "fragment_slab_scalar" in sources[1]
    )
    i = torch.arange(8, dtype=index_dtype)
    m = torch.arange(8, dtype=mask_dtype)
    for start, count in (
        (0, 2**32),
        (2**32, 8),
        (0, 3),
        (0, -(2**32) + 4),
        (-1, 8),
        (2**31 - 2, 8),
    ):
        base[pos], limit[pos] = start, count
        # Keep the 0-D ATen operands: extracting Python ints or widening the
        # vector would change wrapped addresses and comparison results.
        address = base[pos] + i
        mask = (m < limit[pos]) & (address >= 0) & (address < x.size(1))
        vals = torch.where(mask, x[:, address.clamp(0, x.size(1) - 1).long()], 0)
        expected = (vals[:, :4] + vals[:, 4:]).cumsum(1)
        for source in sources:
            out.fill_(-73)
            _simulate_independent_fragment(
                source,
                {"x": x, "base_ptr": base, "limit_ptr": limit},
                {"out": out},
                2,
                int32=lambda v: c_int32(int(v)).value,
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_masked_singleton_tile_applies_each_lane_mask():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, live, out):
        for t in hl.tile(out.size(0), block_size=4):
            values = hl.load(x, [t], extra_mask=live[t])
            out[t] = torch.cumsum(values + 1, dim=0)

    x = torch.tensor([7.0])
    live = torch.ones(4, dtype=torch.bool)
    out = torch.full((4,), -73.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(scan, (x, live, out))
        src = bound.to_code(helion.Config(block_sizes=[]))
    for keep in ([True] * 4, [True, False, True, False], [False] * 4):
        live.copy_(torch.tensor(keep))
        out.fill_(-73.0)
        _simulate_independent_fragment(src, {"x": x, "live": live}, {"out": out}, 1)
        expected = (torch.where(live, x[0], 0.0) + 1).cumsum(0)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_masked_singleton_tile_in_a_runtime_slab():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, table, out):
        for t in hl.tile(out.size(0), block_size=4):
            i = hl.arange(4)
            j = i + 4
            base = table[0]
            limit = table[1]
            lo = hl.load(x, [t, base + i], extra_mask=(i < limit)[None, :])
            hi = hl.load(x, [t, base + j], extra_mask=(j < limit)[None, :])
            out[t, :] = torch.cumsum(lo + hi + 1, dim=1)

    x = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8)
    table = torch.tensor([0, 8], dtype=torch.int32)
    out = torch.full((4, 4), -73.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(scan, (x, table, out))
        sources = [
            bound.to_code(helion.Config(block_sizes=[], cute_fragment_slab_loads=stage))
            for stage in (False, True)
        ]
    assert sources[0].count("x.iterator +") == 2
    assert sources[1].count("x.iterator +") == 1
    for limit in (8, 3):
        table[1] = limit
        active = torch.arange(8) < limit
        masked = torch.where(active, x, 0.0)
        expected = (masked[:, :4] + masked[:, 4:] + 1).cumsum(1).expand_as(out)
        for source in sources:
            out.fill_(-73)
            _simulate_independent_fragment(
                source, {"x": x, "table": table}, {"out": out}, 1
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("width", "start"), [(1, 0), (1, 1), (4, 0)])
def test_singleton_tile_broadcast_feeds_fragment_sum_and_scan(width, start):
    @helion.kernel(backend="cute", static_shapes=True)
    def compute(x, totals, out, begin: hl.constexpr, step: hl.constexpr):
        for row in hl.grid(1):
            for t in hl.tile(begin, out.size(1), block_size=step):
                values = x[t]
                totals[t.begin] = values.sum()
                out[row, t] = torch.cumsum(values + 1, dim=0)

    # A one-row source broadcasts into every live position of the tile.
    # A partial final tile contributes only its two live lanes, not four.
    x = torch.tensor([3.0])
    totals = torch.full((6,), -73.0)
    out = torch.full((1, 6), -73.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(compute, (x, totals, out, start, width))
        src = bound.to_code(helion.Config(block_sizes=[]))
    _simulate_independent_fragment(src, {"x": x}, {"totals": totals, "out": out}, 1)
    expected_totals = torch.full_like(totals, -73.0)
    expected_out = torch.full_like(out, -73.0)
    for begin in range(start, 6, width):
        live = min(width, 6 - begin)
        expected_totals[begin] = 3 * live
        expected_out[0, begin : begin + live] = torch.full((live,), 4.0).cumsum(0)
    torch.testing.assert_close(totals, expected_totals, rtol=0, atol=0)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)


def test_negative_tile_start_broadcasts_all_logically_live_lanes():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, totals, out):
        for row in hl.grid(1):
            for t in hl.tile(-2, 2, block_size=4):
                values = hl.load(x, [t], extra_mask=t.index >= -2)
                totals[row] = values.sum()
                out[row, t.index + 2] = torch.cumsum(values + 1, dim=0)

    x = torch.ones(1)
    totals = torch.full((1,), -73.0)
    out = torch.full((1, 4), -73.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        source = cpu_bind(scan, (x, totals, out)).to_code(helion.Config(block_sizes=[]))
    _simulate_independent_fragment(source, {"x": x}, {"totals": totals, "out": out}, 1)
    torch.testing.assert_close(totals, torch.tensor([4.0]), rtol=0, atol=0)
    torch.testing.assert_close(
        out[0], torch.arange(2, 9, 2, dtype=x.dtype), rtol=0, atol=0
    )
