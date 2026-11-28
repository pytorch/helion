"""Signed fragment loads, stores and staged windows."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute.fragment_slab_loads import runtime_window
from helion._compiler.cute.fragment_slab_loads import static_window
from helion._testing import DEVICE
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
from helion._testing.cute_fragment_index_domains import _masked_wide_window
from helion._testing.cute_fragment_index_domains import signed_fragment_address
from helion._testing.cute_fragment_simulator import (
    simulate_independent_fragment as _simulate_independent_fragment,
)
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("allocation", "access"),
    [
        ("cute.make_rmem_tensor((4,), cutlass.Int32)", "scratch[-1] = 7"),
        (
            "cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Int32, (4,), 16)",
            "value = scratch[-1]",
        ),
    ],
)
def test_fragment_model_rejects_negative_scratch_index(
    allocation: str, access: str
) -> None:
    source = f"""
def _helion_bad_access():
    scratch = {allocation}
    scratch[3] = 42
    {access}
"""
    with pytest.raises(AssertionError, match="fragment scratch index -1 out of bounds"):
        _simulate_independent_fragment(source, {}, {}, 1)


@pytest.mark.parametrize("read", [True, False])
@pytest.mark.parametrize(
    ("masked", "dtype"),
    [(False, torch.int32), (True, torch.int32), (False, torch.int16)],
)
def test_signed_fragment_address_respects_explicit_mask(
    read: bool, masked: bool, dtype: torch.dtype
) -> None:
    # Row zero uses -1; row one uses an in-bounds address with a false mask.
    # Independent inputs and canaries distinguish wrapping, zero and no write.
    x = (torch.arange(128, dtype=torch.float32).view(2, 8, 8) % 13) - 6
    storage = torch.full((2, 10, 8), -9000.0)
    out = storage[:, 1:-1]
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(signed_fragment_address, (x, out, read, masked, dtype))
        source = bound.to_code(bound.config_spec.default_config())
    expected = storage.clone()
    if read:
        lhs = x.roll(1, dims=1)
        lhs[:, 0] = 0
        if masked:
            lhs[:, :2] = 0
        expected[:, 1:-1] = (2 * lhs) @ x.transpose(-1, -2)
    else:
        gram = (2 * x) @ x.transpose(-1, -2)
        if masked:
            expected[:, 2:-2] = gram[:, 2:]
        else:
            expected[:, 1:-2] = gram[:, 1:]
    _simulate_independent_fragment(source, {"x": x}, {"out": out}, 2)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _signed_slab_gram(x, offsets, out, index_dtype: hl.constexpr):
    for batch in hl.tile(x.size(0), block_size=1):
        base = offsets[batch.begin]
        rows = hl.arange(8, dtype=index_dtype)
        cols = hl.arange(8)
        # The scalar selects a non-varying slab dimension. The two windows
        # can straddle row zero in one batch and stay positive in the other.
        lhs = x[batch, -1, base + rows, cols]
        rhs = x[batch, -1, base + (8 + rows), cols]
        out[batch, :, :] = hl.dot(2 * lhs, rhs.transpose(-1, -2))
    return out


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_signed_slab_matches_scalar_windows(index_dtype):
    x = (torch.arange(2 * 3 * 32 * 8).reshape(2, 3, 32, 8) % 13).float() - 6
    offsets = torch.tensor([-4, 3], dtype=torch.int64)
    rows = offsets[:, None] + torch.arange(8)
    batch = torch.arange(2)[:, None]
    # The scalar -1 picks the last plane. The tensor row lanes below
    # zero are guarded in both the ordinary and staged loads.
    left = x[batch, -1, rows.clamp_min(0)] * (rows >= 0)[..., None]
    expected = (2 * left) @ x[batch, -1, rows + 8].transpose(-1, -2)
    out = torch.empty_like(expected)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_signed_slab_gram, (x, offsets, out, index_dtype))
        default = bound.config_spec.default_config()
        for slab in (False, True):
            config = helion.Config.from_dict(
                dict(default) | {"cute_fragment_slab_loads": slab}
            )
            source = bound.to_code(config)
            assert ("fragment_slab_scalar" in source) == slab
            if slab:
                bits = torch.iinfo(index_dtype).bits
                assert f"fragment_slab_scalar = cutlass.Int{bits}(0)" in source
            out.fill_(-9000)
            _simulate_independent_fragment(
                source, {"x": x, "offsets": offsets}, {"out": out}, 2
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_slab_of_adjacent_negative_literal_rows_widens_before_adding() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def adjacent(x, out):
        for i in hl.grid(1):
            col = hl.arange(8)
            out[i, :] = torch.cumsum(x[-2, col] + x[-1, col], dim=0)

    start = 2**31 - 1
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        # Two real, disjoint CPU storages satisfy the read-only proof. A
        # broadcast view fits the large logical axis in eight elements.
        x = torch.empty(8).expand(start + 2, 8)
        out = torch.empty(1, 8)
        bound = cpu_bind(adjacent, (x, out))
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_slab_loads": True}
        )
        source = bound.to_code(cfg)
    # This is an address calculation from the genuine slab load, not the
    # simulator's unbounded Python integer addition.
    assert any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and ast.unparse(node.left) == f"cutlass.Int64({start})"
        and isinstance(node.right, ast.Call)
        and ast.unparse(node.right.func) == "cutlass.Int64"
        for node in ast.walk(ast.parse(source))
    )


def test_masked_window_crossing_int32_does_not_wrap_before_comparison() -> None:
    start = 2**31 - 4
    x = torch.ones(1, 16, 8)
    bases = torch.tensor([-start], dtype=torch.int64)
    lengths = torch.tensor([2**31], dtype=torch.int64)
    out = torch.empty(1, 8, 8)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(
            _masked_wide_window, (x, bases, lengths, out, torch.int64, start)
        )
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_slab_loads": True}
        )
        source = bound.to_code(cfg)
    assert "fragment_slab_scalar" in source
    # Simulated Python loop indices are unbounded; independently require
    # the native Int64 cast on the addition's coordinate, before the mask.
    wide_adds = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and isinstance(node.left, ast.Call)
        and ast.unparse(node.left.func) == "cutlass.Int64"
        and isinstance(node.left.args[0], ast.Constant)
        and node.left.args[0].value == start
        and isinstance(node.right, ast.Call)
        and ast.unparse(node.right.func) == "cutlass.Int64"
    ]
    assert wide_adds
    # No column in the upper operand is below the limit; its dot is zero.
    _simulate_independent_fragment(
        source,
        {"x": x, "bases": bases, "lengths": lengths},
        {"out": out},
        1,
        int32=lambda v: (int(v) + 2**31) % 2**32 - 2**31,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_masked_slab_uses_the_window_and_comparison_dtypes(index_dtype) -> None:
    # Truncating the first offset makes an invalid row live. Truncating the
    # second limit masks every valid row. The third target has a real tail.
    x = (torch.arange(3 * 16 * 8).reshape(3, 16, 8) % 13).float() - 6
    bases = torch.tensor([2**32, 0, 0], dtype=torch.int64)
    lengths = torch.tensor([16, 2**32, 11], dtype=torch.int64)
    out = torch.full((3, 8, 8), -9000.0)
    expected = torch.zeros_like(out)
    if index_dtype == torch.int64:
        expected[1] = (2 * x[1, :8]) @ x[1, 8:].T
    else:
        # The int32 vector narrows a 0-d int64 argument in both the index
        # sum and comparison. A blind promotion to the table dtype is wrong.
        expected[0] = (2 * x[0, :8]) @ x[0, 8:].T
    expected[2, :, :3] = (2 * x[2, :8]) @ x[2, 8:11].T
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_masked_wide_window, (x, bases, lengths, out, index_dtype))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_slab_loads": True}
        )
        code = bound.to_code(cfg)
    # Include Int32 narrowing in the generated-code replay: its default
    # Python integers would hide a truncating load.
    _simulate_independent_fragment(
        code,
        {"x": x, "bases": bases, "lengths": lengths},
        {"out": out},
        3,
        int32=lambda v: (int(v) + 2**31) % 2**32 - 2**31,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_mask_widths(x, bases, lengths, out):
    for batch in hl.grid(x.size(0)):
        base, limit = bases[batch], lengths[batch]
        lo = hl.arange(8, dtype=torch.int64)
        hi = lo + 8
        m = hl.arange(8, dtype=torch.int32)
        cols = hl.arange(8)
        a = hl.load(x, [batch, base + lo, cols], extra_mask=(lo < limit)[:, None])
        b = hl.load(x, [batch, base + hi, cols], extra_mask=(hi < limit)[:, None])
        c = hl.load(x, [batch, base + lo, cols], extra_mask=(m < limit)[:, None])
        d = hl.load(x, [batch, base + hi, cols], extra_mask=(m + 8 < limit)[:, None])
        out[batch, :, :] = hl.dot(a * 3, b.T) + hl.dot(c * 2, d.T)
    return out


def test_equal_offsets_do_not_share_different_mask_widths() -> None:
    x = (torch.arange(2 * 16 * 8).reshape(2, 16, 8) % 13).float() - 6
    bases = torch.zeros(2, dtype=torch.int64)
    lengths = torch.tensor([2**32, 16], dtype=torch.int64)
    out = torch.empty(2, 8, 8)
    expected = (x[:, :8] @ x[:, 8:].transpose(-1, -2)) * torch.tensor([3, 5])[
        :, None, None
    ]
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_two_mask_widths, (x, bases, lengths, out))
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_slab_loads": True}
        )
        code = bound.to_code(cfg)
    staged = [
        ln
        for ln in code.splitlines()
        if "fragment_slab_scalar" in ln and "lengths.iterator +" in ln
    ]
    assert len(staged) == 2
    assert any(" = cutlass.Int32(" in ln for ln in staged)
    assert any(" = cutlass.Int64(" in ln for ln in staged)
    _simulate_independent_fragment(
        code,
        {"x": x, "bases": bases, "lengths": lengths},
        {"out": out},
        2,
        int32=lambda v: (int(v) + 2**31) % 2**32 - 2**31,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _symbolic_slab_coordinates(x, table, out):
    for batch in hl.tile(x.size(0), block_size=1):
        rows, cols = hl.arange(8), hl.arange(8)
        offset = table[batch.begin - 1]
        lhs = x[batch.begin, batch.begin - 1, offset + rows, cols]
        rhs = x[batch.begin, batch.begin - 1, offset + (8 + rows), cols]
        out[batch.begin - 1, :, :] = hl.dot(2 * lhs, rhs.transpose(-1, -2))
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _literal_slab_coordinate(x, out, row: hl.constexpr, starts: hl.constexpr):
    batches = hl.specialize(x.size(0))
    height = hl.specialize(x.size(2))
    width = hl.specialize(out.size(2))
    for batch in hl.tile(batches, block_size=1):
        rows = torch.add(hl.arange(height // 2), starts[0])
        cols = torch.add(hl.arange(width), starts[1])
        a = x[batch.begin, row, rows, cols]
        b = x[batch.begin, row, rows + height // 2, cols]
        out[batch.begin, :, :] = hl.dot(a * 2, b.transpose(-2, -1))
    return out


def test_symbolic_slab_dimension_checks_literal_at_runtime() -> None:
    x = (torch.arange(2 * 3 * 16 * 11).reshape(2, 3, 16, 11) % 17).float() - 8
    out = torch.full((2, 8, 8), -9000.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_literal_slab_coordinate, (x, out, -3, (0, 0)))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_slab_loads": True}
        )
        code = bound.to_code(cfg)
    assert code.count(".load()") == 1
    # The same installed slab checks both the scalar plane and its shared
    # eight-column window at run time. A partial column must never read the
    # next row, and a fully out-of-range plane must read zero.
    for planes, columns in ((3, 11), (3, 5), (2, 8)):
        data = x[:, :planes, :, :columns].contiguous()
        out.fill_(-9000)
        arguments = {"x": data}
        for name, tensor in (("x", data), ("out", out)):
            for dim in range(tensor.ndim):
                arguments[f"{name}_size_{dim}"] = tensor.size(dim)
                arguments[f"{name}_stride_{dim}"] = tensor.stride(dim)
        _simulate_independent_fragment(code, arguments, {"out": out}, 2)
        expected = (
            (data[:, 0, :8, :8] * 2) @ data[:, 0, 8:, :8].transpose(-1, -2)
            if planes == 3
            else torch.zeros_like(out)
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if (planes, columns) == (3, 5):
            # The very same five-column tensor must keep the slab when its
            # size is known during compilation rather than passed at runtime.
            static = helion.kernel(
                _literal_slab_coordinate.fn, backend="cute", static_shapes=True
            )
            with mock_cuda_unavailable(), _target(), forbid_native_compile():
                static_bound = cpu_bind(static, (data, out, -3, (0, 0)))
                assert static_bound.config_spec.cute_fragment_slab_loads_root_ids == {0}
                static_code = static_bound.to_code(cfg)
            assert static_code.count(".load()") == 1
            out.fill_(-9000)
            _simulate_independent_fragment(static_code, {"x": data}, {"out": out}, 2)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("start", [-1, -4])
def test_slab_zero_fills_negative_common_columns(start: int) -> None:
    # A complete FP32 packet before column zero must be fully suppressed;
    # a partially negative packet must use scalar per-lane bounds. The CUDA
    # counterpart below executes both packet settings against this reference.
    storage = (torch.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16) % 17).float() - 8
    storage[..., :4] = 313
    x = storage[..., 4:12]
    destination = torch.full((2 * 8 * 8 + 8,), -9000.0)
    out = destination[4:-4].view(2, 8, 8)
    live = x[:, 0, :, : 8 + start]
    expected = (2 * live[:, :8]) @ live[:, 8:].transpose(-1, -2)
    for known in (False, True):
        kernel = helion.kernel(
            _literal_slab_coordinate.fn, backend="cute", static_shapes=known
        )
        with mock_cuda_unavailable(), _target(), forbid_native_compile():
            bound = cpu_bind(kernel, (x, out, 0, (0, start)))
            assert bound.config_spec.cute_fragment_slab_loads_root_ids == {0}
            cfg = helion.Config.from_dict(
                dict(bound.config_spec.default_config())
                | {"cute_fragment_slab_loads": True}
            )
            code = bound.to_code(cfg)
            packets = bound.to_code(
                helion.Config.from_dict(dict(cfg) | {"cute_fragment_vector_io": True})
            )
            has_input_packet = any(
                "cute.copy(" in line and "x.iterator" in line
                for line in packets.splitlines()
            )
            assert has_input_packet == (known and start == -4)
            if start == -1:
                # Moving the same start to the varying row axis excludes its
                # first window from the static union, leaving only one member.
                varying = cpu_bind(kernel, (x, out, 0, (start, 0)))
                assert not varying.config_spec.cute_fragment_slab_loads_root_ids
        assert code.count(".load()") == 1
        assert "cute.copy(" not in code
        arguments = {"x": x}
        for name, tensor in (("x", x), ("out", out)):
            for dim in range(tensor.ndim):
                arguments[f"{name}_size_{dim}"] = tensor.size(dim)
                arguments[f"{name}_stride_{dim}"] = tensor.stride(dim)
        out.fill_(-9000)
        _simulate_independent_fragment(code, arguments, {"out": out}, 2)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert (destination[:4] == -9000).all() and (destination[-4:] == -9000).all()


def test_signed_scalar_expressions_match_slab_table_and_plain_store() -> None:
    x = (torch.arange(2 * 3 * 32 * 8).reshape(2, 3, 32, 8) % 17).float() - 8
    table = torch.tensor([-4, 3], dtype=torch.int32)
    backing = torch.full((2, 10, 8), -9000.0)
    out = backing[:, 1:-1]
    expected = backing.clone()
    for batch in range(2):
        i = batch - 1
        rows = table[i] + torch.arange(8)
        left = x[batch, i, rows.clamp_min(0)] * (rows >= 0)[:, None]
        expected[i, 1:-1] = (2 * left) @ x[batch, i, rows + 8].T
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_symbolic_slab_coordinates, (x, table, out))
        default = bound.config_spec.default_config()
        for enabled in (False, True):
            config = helion.Config.from_dict(
                dict(default) | {"cute_fragment_slab_loads": enabled}
            )
            code = bound.to_code(config)
            assert ("fragment_slab_scalar" in code) == enabled
            backing.fill_(-9000)
            _simulate_independent_fragment(
                code, {"x": x, "table": table}, {"out": out}, 2
            )
            torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_indexed_gram(x, out, mode: hl.constexpr):
    for batch in hl.tile(x.size(0), block_size=1):
        rows = hl.arange(8)
        cols = hl.arange(8)
        acc = hl.full([8, 8], 0, torch.float32)
        start = -1 if mode == "shifted" else 0
        for k in hl.grid(start, x.size(1) + start):
            tile = x[batch.begin, k, rows, cols]
            acc = acc + hl.dot(tile * 2, tile.transpose(-2, -1))
        out[batch.begin, :, :] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shared_grid_origins(x, out):
    width = hl.register_block_size(x.size(0))
    for batch in hl.tile(x.size(0), block_size=width):
        acc = hl.full([8, 8], 0, torch.float32)
        for inner in hl.tile(-1, x.size(0) - 1, block_size=width):
            a = x[inner.begin, :, :]
            acc = acc + hl.dot(a * 2, a.transpose(-2, -1))
        out[batch.begin, :, :] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_slab_rows(x, out):
    width = hl.register_block_size(x.size(0))
    for row in hl.tile(x.size(0), block_size=width):
        acc = hl.full([4, 4], 0.0, torch.float32)
        for inner in hl.tile(-1, x.size(0) - 1, block_size=width):
            indices, cols = hl.arange(4), hl.arange(4)
            lo = x[inner.begin, indices, cols]
            hi = x[inner.begin, indices + 4, cols]
            acc = acc + hl.dot(2 * lo, hi.transpose(-2, -1))
        out[row.begin, :, :] = acc
    return out


def test_slab_cannot_hoist_a_nested_origin_with_the_root_block_id() -> None:
    x = (torch.arange(3 * 8 * 4).reshape(3, 8, 4) % 17).float() - 8
    out = torch.zeros(3, 4, 4)
    expected = ((2 * x[:, :4]) @ x[:, 4:].transpose(-2, -1)).sum(0).expand_as(out)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_nested_slab_rows, (x, out))
        assert bound.host_function.device_ir.noncanonical_task_origin_block_ids == {0}
        for enabled in (False, True):
            cfg = helion.Config.from_dict(
                dict(bound.config_spec.default_config())
                | {"block_sizes": [1], "cute_fragment_slab_loads": enabled}
            )
            source = bound.to_code(cfg)
            assert "fragment_slab_" not in source
            out.fill_(-9000)
            _simulate_independent_fragment(source, {"x": x}, {"out": out}, 3)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _slab_shifted_block(x, out):
    for block in hl.tile(-1, x.size(0) - 1, block_size=1):
        rows, cols = hl.arange(4), hl.arange(4)
        a = x[block, rows, cols]
        b = x[block, rows + 4, cols]
        out[block.index + 1, :, :] = hl.dot(2 * a, b.transpose(-2, -1))
    return out


def test_slab_applies_the_input_bounds_to_shifted_block_coordinates() -> None:
    storage = (torch.arange(4 * 8 * 4).reshape(4, 8, 4) % 13).float() - 6
    storage[0].fill_(23)
    x = storage[1:]
    out = torch.zeros(3, 4, 4)
    expected = torch.zeros_like(out)
    expected[1:] = (2 * x[:2, :4]) @ x[:2, 4:].transpose(-2, -1)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_slab_shifted_block, (x, out))
        for enabled in (False, True):
            cfg = helion.Config.from_dict(
                dict(bound.config_spec.default_config())
                | {"cute_fragment_slab_loads": enabled}
            )
            source = bound.to_code(cfg)
            assert ("cute.make_layout((32,))" in source) == enabled
            out.fill_(-9000)
            _simulate_independent_fragment(source, {"x": x}, {"out": out}, 3)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_shared_block_id_preserves_shifted_loop() -> None:
    x = (torch.arange(2 * 8 * 5).reshape(2, 8, 5) % 13).float() - 6
    out = torch.full((2, 8, 8), -9000.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_shared_grid_origins, (x, out))
        ir = bound.host_function.device_ir
        assert ir.noncanonical_task_origin_block_ids == {0}
        assert ir.task_families[0].axes[0].canonical_origin
        assert ir.task_families[0].axes[0].block_id == 0
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config()) | {"block_sizes": [1]}
        )
        code = bound.to_code(cfg)
    assert "fragment_wrapped_index" in code
    _simulate_independent_fragment(code, {"x": x}, {"out": out}, 2)
    expected = ((2 * x) @ x.transpose(-1, -2)).sum(0)
    torch.testing.assert_close(out, expected.expand_as(out), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _divided_grid_and_tile(x, out):
    for batch in hl.grid(out.size(0)):
        acc = hl.full([8, 8], 0, torch.float32)
        for k in hl.tile(x.size(1), block_size=2):
            v = x[batch // 2, k.begin, :, :]
            acc = acc + hl.dot(v * 2, v.transpose(-2, -1))
        out[batch, :, :] = acc
    return out


def test_divided_canonical_grid_and_inner_tile_are_nonnegative() -> None:
    x = (torch.arange(2 * 4 * 8 * 8).reshape(2, 4, 8, 8) % 13).float() - 6
    out = torch.full((4, 8, 8), -9000.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_divided_grid_and_tile, (x, out))
        code = bound.to_code(bound.config_spec.default_config())
    assert "fragment_wrapped_index" not in code
    _simulate_independent_fragment(code, {"x": x}, {"out": out}, 4)
    v = x.repeat_interleave(2, dim=0)[:, ::2]
    torch.testing.assert_close(
        out, ((2 * v) @ v.transpose(-1, -2)).sum(1), rtol=0, atol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_index_gram(x, out):
    for batch in hl.tile(x.size(0), block_size=1):
        rows, cols = hl.arange(8), hl.arange(8)
        acc = hl.full([8, 8], 0, torch.float32)
        for k in hl.grid(x.size(1)):
            tile = x[batch.begin, k, rows, cols]
            acc = acc + hl.dot(tile * 2, tile.transpose(-2, -1))
            rows = rows - 1
        out[batch.begin, :, :] = acc
    return out


@pytest.mark.parametrize("mode", ["canonical", "shifted", "carry"])
def test_loop_indices_wrap_only_possible_negatives(mode: str) -> None:
    x = (torch.arange(2 * 3 * 8 * 8).reshape(2, 3, 8, 8) % 13).float() - 6
    out = torch.full((2, 8, 8), -9000.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = (
            cpu_bind(_carried_index_gram, (x, out))
            if mode == "carry"
            else cpu_bind(_loop_indexed_gram, (x, out, mode))
        )
        code = bound.to_code(bound.config_spec.default_config())
    assert ("fragment_wrapped_index" in code) == (mode == "shifted")
    _simulate_independent_fragment(code, {"x": x}, {"out": out}, 2)
    expected = torch.zeros_like(out)
    for k in range(x.size(1)):
        tile = x[:, k]
        if mode == "carry":
            tile = tile.roll(k, dims=1)
            tile[:, :k] = 0
        expected += (2 * tile) @ tile.transpose(-1, -2)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_grid_fragment(x, out):
    for batch in hl.grid(-1, x.size(0) - 1):
        rows, cols = hl.arange(8), hl.arange(8)
        tile = x[batch, rows, cols]
        out[batch, rows, cols] = hl.dot(tile * 2, tile.transpose(-2, -1))
    return out


def test_shifted_root_grid_keeps_signed_scalar_coordinates() -> None:
    # -1 and 0 both execute. A predicate that assumed all root grids start
    # at zero would leave the final output matrix at its original sentinel.
    x = (torch.arange(128).reshape(2, 8, 8) % 13).float() - 6
    out = torch.full_like(x, -9000)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_shifted_grid_fragment, (x, out))
        code = bound.to_code(bound.config_spec.default_config())
    assert "fragment_wrapped_index" in code
    _simulate_independent_fragment(code, {"x": x}, {"out": out}, 2)
    torch.testing.assert_close(out, (2 * x) @ x.transpose(-1, -2), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _slab_with_newaxis(x, offsets, out):
    for batch in hl.grid(x.size(0)):
        base = offsets[batch]
        rows, cols = hl.arange(8), hl.arange(8)
        lhs = x[None, batch, base + rows, cols].squeeze(0)
        rhs = x[None, batch, base + (8 + rows), cols].squeeze(0)
        out[batch, :, :] = hl.dot(2 * lhs, rhs.transpose(-1, -2))
    return out


def test_slab_preserves_newaxis_in_the_plain_load() -> None:
    x = (torch.arange(2 * 32 * 8).reshape(2, 32, 8) % 13).float() - 6
    offsets = torch.tensor([3, -4], dtype=torch.int64)
    out = torch.full((2, 8, 8), -9000.0)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_slab_with_newaxis, (x, offsets, out))
        assert not bound.config_spec.cute_fragment_slab_loads_root_ids
        code = bound.to_code(bound.config_spec.default_config())
    rows = offsets[:, None] + torch.arange(8)
    batch = torch.arange(2)[:, None]
    left = x[batch, rows.clamp_min(0)] * (rows >= 0)[..., None]
    expected = (2 * left) @ x[batch, rows + 8].transpose(-1, -2)
    _simulate_independent_fragment(code, {"x": x, "offsets": offsets}, {"out": out}, 2)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_slab_zero_fills_partial_nonvarying_dimension() -> None:
    # Both row windows index eight columns of a six-column input. A slab
    # varied along rows must zero-fill its last two columns, not read
    # the next row.
    storage = (torch.arange(2 * 3 * 32 * 8).reshape(2, 3, 32, 8) % 13).float() - 6
    storage[..., 6:] = 313
    x = storage[..., :6]
    offsets = torch.tensor([-4, 3], dtype=torch.int64)
    destination = torch.full((2, 8, 9), -9000.0)
    out = destination[..., :8]
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_signed_slab_gram, (x, offsets, out, torch.int64))
        assert bound.config_spec.cute_fragment_slab_loads_root_ids == {0}
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_slab_loads": True, "cute_fragment_vector_io": True}
        )
        code = bound.to_code(cfg)
        assert "fragment_slab_" in code
        assert "cute.copy(" not in code
    rows = offsets[:, None] + torch.arange(8)
    batch = torch.arange(2)[:, None]
    # The scalar -1 picks the last plane. The tensor row lanes below
    # zero are guarded in both the ordinary and staged loads.
    left = x[batch, -1, rows.clamp_min(0)] * (rows >= 0)[..., None]
    expected = (2 * left) @ x[batch, -1, rows + 8].transpose(-1, -2)
    _simulate_independent_fragment(code, {"x": x, "offsets": offsets}, {"out": out}, 2)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert (destination[..., 8] == -9000).all()


def test_slab_window_requires_unit_coefficient() -> None:
    x = torch.empty(2, 3, 32, 8)
    offsets = torch.empty(2, dtype=torch.int64)
    out = torch.empty(2, 8, 8)
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_signed_slab_gram, (x, offsets, out, torch.int64))
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            additions = [
                n
                for g in graphs
                for n in g.graph.nodes
                if n.target is torch.ops.aten.add.Tensor
            ]
            static = next(n for n in additions if static_window(n) is not None)
            dynamic = next(
                n
                for n in additions
                if runtime_window(bound.env, n, {}, graphs) is not None
            )
            for node in (static, dynamic):
                # alpha=2 changes either the shift or the stride; neither
                # remains the original unscaled unit-stride window.
                old = node.kwargs
                try:
                    node.kwargs = dict(old, alpha=2)
                    if node is static:
                        assert static_window(node) is None
                    else:
                        assert runtime_window(bound.env, node, {}, graphs) is None
                finally:
                    node.kwargs = old


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _literal_row_gram(x, out, masked: hl.constexpr):
    for batch in hl.tile(x.size(0), block_size=1):
        cols = hl.arange(8)
        a = x[batch, :, cols]
        gram = hl.dot(a * 2, a.transpose(-1, -2))
        if masked:
            mask = (cols != 1)[None, :]
            row = hl.load(x, [batch, -1, cols], extra_mask=mask)
            hl.store(out, [batch, -1, cols], gram.sum(1) + row, extra_mask=mask)
        else:
            out[batch, -1, cols] = gram.sum(1) + x[batch, -1, cols]
    return out


@pytest.mark.parametrize("masked", [False, True])
def test_negative_scalar_wraps_with_or_without_extra_mask(masked):
    x = (torch.arange(128).view(2, 8, 8) % 13).float() - 6
    out = torch.full_like(x, -9000)
    expected = out.clone()
    expected[:, -1] = ((2 * x) @ x.transpose(-1, -2)).sum(1) + x[:, -1]
    if masked:
        expected[:, -1, 1] = -9000
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_literal_row_gram, (x, out, masked))
        source = bound.to_code(bound.config_spec.default_config())
    _simulate_independent_fragment(source, {"x": x}, {"out": out}, 2)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_negative_common_columns_execute_real_packet_and_scalar_guards() -> None:
    storage = (
        torch.arange(2 * 3 * 16 * 16, device=DEVICE).view(2, 3, 16, 16) % 17
    ).float() - 8
    storage[..., :4] = 313
    storage[..., 12:] = -271
    x = storage[..., 4:12]
    destination = torch.full((2 * 8 * 8 + 8,), -9000.0, device=DEVICE)
    out = destination[4:-4].view(2, 8, 8)
    for known in (False, True):
        kernel = helion.kernel(
            _literal_slab_coordinate.fn, backend="cute", static_shapes=known
        )
        for start in (-1, -4):
            args = (x, out, 0, (0, start))
            bound = kernel.bind(args)
            assert bound.config_spec.cute_fragment_slab_loads_root_ids == {0}
            for packet in (False, True):
                cfg = helion.Config.from_dict(
                    dict(bound.config_spec.default_config())
                    | {
                        "cute_fragment_slab_loads": True,
                        "cute_fragment_vector_io": packet,
                    }
                )
                code = bound.to_code(cfg)
                assert any(
                    "cute.copy(" in line and "x.iterator" in line
                    for line in code.splitlines()
                ) == (known and start == -4 and packet)
                run = bound.compile_config(cfg)
                for _ in range(2):
                    live = x[:, 0, :, : 8 + start]
                    expected = (2 * live[:, :8]) @ live[:, 8:].transpose(-1, -2)
                    destination.fill_(-9000)
                    run(*args)
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    assert (destination[:4] == -9000).all() and (
                        destination[-4:] == -9000
                    ).all()
                    x[:, 0, 8:, 0].neg_()
