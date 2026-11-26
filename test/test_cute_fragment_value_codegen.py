"""CPU code generation and replay checks for fragment values and memory bounds."""

from __future__ import annotations

import ast
from ctypes import c_int32
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
from helion._testing.cute_fragment_kernels import singleton_tail_scan
from helion._testing.cute_fragment_simulator import (
    simulate_independent_fragment as _simulate_independent_fragment,
)
import helion.language as hl
from helion.runtime.ref_mode import RefMode

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


def test_fragment_mean_clips_a_wide_interval_before_subtracting():
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int64)
    def scan(x, out):
        for t in hl.tile(-(2**63) + 1, 1, block_size=4):
            values = x[t]
            out[t.begin // 4 + 2**61, :] = torch.cumsum(values * 0 + values.mean(), 0)

    # Simulate the first CTA. The mathematical remaining length exceeds
    # INT64_MAX, but the four-lane mean must still divide by exactly four.
    x = torch.tensor([7.0])
    out = torch.full((1, 4), -73.0)
    with mock_cuda_unavailable():
        source = cpu_bind(scan, (x, out)).to_code(
            helion.Config(block_sizes=[]), output_origin_lines=False
        )
    _simulate_independent_fragment(
        source, {"x": x}, {"out": out}, 1, int32=np.int32, int64=np.int64
    )
    assert out.tolist() == [[7, 14, 21, 28]]


@pytest.mark.parametrize("checked", [False, True], ids=["ordinary", "checked"])
def test_nested_fragment_mean_uses_the_live_tail(checked):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, checked: hl.constexpr):
        for _ in hl.grid(1):
            for t in hl.tile(3, 16, block_size=4):
                mean = x[t].mean()
                if checked:
                    torch._assert_async(mean > 0.5, "tail mean")
                out[t] = torch.cumsum(x[t] * 0 + mean, dim=0)

    x = torch.ones(16)
    expected = torch.full_like(x, -1)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected, checked)
    assert expected.tolist() == [-1] * 3 + [1, 2, 3, 4] * 3 + [1]
    with mock_cuda_unavailable():
        code = cpu_bind(scan, (x, expected, checked)).to_code(
            helion.Config(block_sizes=[])
        )
    seen = []

    def observe_assert(operands, *, asm, **options):
        assert (
            "call __assertfail" in asm and len(operands) == 1 and not options["is_pure"]
        )
        seen.append(bool(operands[0]))

    actual = torch.full_like(x, -1)
    _simulate_independent_fragment(
        code, {"x": x}, {"out": actual}, 1, inline_asm=observe_assert
    )
    torch.testing.assert_close(actual, expected)
    assert seen == ([True] * 4 if checked else [])


@pytest.mark.parametrize(
    "dynamic,index_dtype",
    [(False, torch.int64), (True, torch.int64), (True, None)],
    ids=["literals", "device-bounds", "device-bounds-default-index"],
)
def test_ordinary_fragment_int64_range_preserves_both_endpoints(dynamic, index_dtype):
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=index_dtype)
    def scan(x, out, bounds, dynamic: hl.constexpr):
        for row in hl.grid(1):
            if dynamic:
                start, stop = bounds[row, 0], bounds[row, 1]
            else:
                start, stop = 2**32 - 1, 2**32 + 1
            for i in hl.grid(start, stop):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + i, dim=0)

    x = torch.zeros((1, 4), dtype=torch.int64)
    expected = torch.zeros_like(x)
    center = 2**31 if index_dtype is None else 2**32
    bounds = torch.tensor([[center - 1, center + 1]])
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected, bounds, dynamic)
    assert expected.tolist() == [[(2 * center - 1) * n for n in range(1, 5)]]
    with mock_cuda_unavailable():
        code = cpu_bind(scan, (x, expected, bounds, dynamic)).to_code(
            helion.Config(block_sizes=[])
        )
    actual = torch.zeros_like(x)
    _simulate_independent_fragment(
        code,
        {"x": x, "bounds": bounds},
        {"out": actual},
        1,
        int32=lambda v: (int(v) + 2**31) % 2**32 - 2**31,
    )
    torch.testing.assert_close(actual, expected)


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


def test_singleton_broadcast_excludes_padded_lanes_at_int32_max():
    x = torch.ones(1)
    totals = torch.full((2,), -73.0)
    scans = torch.full((2, 4), -73.0)
    with mock_cuda_unavailable():
        source = cpu_bind(
            singleton_tail_scan, (x, totals, scans, 2**31 - 6, 2**31 - 1)
        ).to_code(helion.Config(block_sizes=[]), output_origin_lines=False)
    counts = _simulate_independent_fragment(
        source,
        {"x": x},
        {"totals": totals, "scans": scans},
        2,
        int32=np.int32,
        int64=np.int64,
    )
    assert totals.tolist() == [4, 1]
    assert counts["scalar"] == 5


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


def test_wide_interval_keeps_first_tile_gather_and_scan_live():
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int32)
    def compute(x, table, totals, out):
        for t in hl.tile(-(2**31), 0, block_size=4):
            # Only the first CTA has an output row. Its true remaining distance
            # is 2**31, which must not become a negative scan extent or mask.
            row = t.begin // 4 + 2**29
            values = table[x[t]]
            totals[row] = values.sum()
            out[row, :] = torch.cumsum(values + 1, dim=0)

    x = torch.zeros(1, dtype=torch.int32)
    table = torch.tensor([7, 9], dtype=torch.int32)
    totals = torch.full((1,), -73, dtype=torch.int32)
    out = torch.full((1, 4), -73, dtype=torch.int32)
    with mock_cuda_unavailable():
        src = cpu_bind(compute, (x, table, totals, out)).to_code(
            helion.Config(block_sizes=[]), output_origin_lines=False
        )
    _simulate_independent_fragment(
        src,
        {"x": x, "table": table},
        {"totals": totals, "out": out},
        1,
        int32=np.int32,
        int64=np.int64,
    )
    assert totals.tolist() == [28]
    assert out.tolist() == [[8, 16, 24, 32]]


def test_indirect_store_at_int64_max_preserves_its_tail():
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int64)
    def compute(x, totals, out):
        for t in hl.tile(2**63 - 3, 2**63 - 1, block_size=4):
            values = x[t]
            totals[0] = values.sum()
            totals[1] = t.end
            out[t.index - (2**63 - 3)] = torch.cumsum(values + 1, dim=0)

    x = torch.tensor([3], dtype=torch.int64)
    totals = torch.full((2,), -73, dtype=torch.int64)
    out = torch.full((4,), -73, dtype=torch.int64)
    with mock_cuda_unavailable():
        src = cpu_bind(compute, (x, totals, out)).to_code(
            helion.Config(block_sizes=[]), output_origin_lines=False
        )
    with np.errstate(over="ignore"):
        _simulate_independent_fragment(
            src,
            {"x": x},
            {"totals": totals, "out": out},
            1,
            int32=np.int32,
            int64=np.int64,
        )
    assert totals.tolist() == [6, 2**63 - 1]
    assert out.tolist() == [4, 8, -73, -73]


def test_captured_index_keeps_its_defining_tile_lanes():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        width = hl.register_block_size(4)
        for outer in hl.tile(4, block_size=width):
            ids = outer.index
            for inner in hl.tile(2, block_size=width):
                out[ids] = torch.cumsum(x[:] + inner.begin, dim=0)

    x = torch.arange(1, 5, dtype=torch.int64)
    out = torch.full_like(x, -73)
    with mock_cuda_unavailable():
        src = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[4]))
    _simulate_independent_fragment(src, {"x": x}, {"out": out}, 1)
    torch.testing.assert_close(out, torch.cumsum(x, dim=0))


@pytest.mark.parametrize(
    "kind", ["host", "uint32", "nested", "expression", "signed-unit", "nonunit"]
)
def test_fragment_ranges_preserve_unproved_operands(kind):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, start: int, stop: int, bounds, kind: hl.constexpr):
        for row in hl.grid(1):
            if kind == "nested" or kind == "expression":
                if kind == "nested":
                    begin, end = bounds[row, 0], bounds[row, 1]
                else:
                    begin, end = 2, 4
                for i in hl.grid(begin, end):
                    if kind == "expression":
                        low = i * 2**30
                    else:
                        low = i
                    for j in hl.grid(low, low + 1):
                        out[row, :] = out[row, :] + torch.cumsum(x[row, :] + j, dim=0)
            else:
                if kind == "host":
                    low, high = start, stop
                else:
                    low, high = bounds[row, 0], bounds[row, 1]
                step = 2 if kind == "nonunit" else 1
                for j in hl.grid(low, high, step):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + j, dim=0)

    x = torch.zeros((1, 4), dtype=torch.int64)
    start, stop = 2**31 - 1, 2**31 + 1
    if kind in ("nonunit", "signed-unit"):
        # With a unit stride stop is representable; with stride 2, the final
        # increment would cross 2**31, even though both endpoints fit Int32.
        bounds = torch.tensor([[2**31 - 2, 2**31 - 1]], dtype=torch.int32)
    else:
        bounds = torch.tensor(
            [[start, stop]], dtype=torch.uint32 if kind == "uint32" else torch.int64
        )
    expected = torch.zeros_like(x)
    args = x, expected, start, stop, bounds, kind
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(*args)
    total = 5 * 2**30 if kind == "expression" else 2**32 - 1
    if kind in ("nonunit", "signed-unit"):
        total = 2**31 - 2
    assert expected.tolist() == [[total * n for n in range(1, 5)]]
    with mock_cuda_unavailable():
        code = cpu_bind(scan, args).to_code(helion.Config(block_sizes=[]))
    actual = torch.zeros_like(x)
    _simulate_independent_fragment(
        code,
        {"x": x, "bounds": bounds, "start": start, "stop": stop},
        {"out": actual},
        1,
        int32=np.int32,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if kind in ("nonunit", "signed-unit"):
        (loop,) = (
            line for line in code.splitlines() if "for fragment_tile in range(" in line
        )
        assert (
            "range(cutlass.Int32(" if kind == "signed-unit" else "range(cutlass.Int64("
        ) in loop
    if kind == "expression":
        # Python's range simulator yields plain integers. Check the native
        # counter is widened before constructing the nested endpoint.
        assert "1073741824 * cutlass.Int64(fragment_tile)" in code


def test_fragment_iota_uses_the_entire_symbolic_step():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(out):
        for row in hl.grid(out.size(0)):
            values = torch.ops.prims.iota.default(
                3,
                start=0,
                step=row + 1,
                dtype=torch.int64,
                device=out.device,
                requires_grad=False,
            )
            out[row, :] = torch.cumsum(values, dim=0)

    expected = torch.empty((2, 3), dtype=torch.int64)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(expected)
    assert expected.tolist() == [[0, 1, 3], [0, 2, 6]]
    with mock_cuda_unavailable():
        code = cpu_bind(scan, (expected,)).to_code(helion.Config(block_sizes=[]))
    actual = torch.full_like(expected, -73)
    _simulate_independent_fragment(code, {}, {"out": actual}, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
