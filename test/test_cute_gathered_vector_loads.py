"""CPU codegen coverage for CuTe vector loads with a gathered row coordinate."""

from __future__ import annotations

import ast
import math

import numpy as np
import pytest
import torch

import helion
from helion._testing import import_example
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_thread_simulator import execute_pointwise_thread_program
import helion.language as hl

pytestmark = pytest.mark.cpu_only

embedding = import_example("embedding").embedding


def _generate(kernel: object, args: tuple[object, ...], **config: object) -> str:
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
        return bound.to_code(helion.Config.from_dict(config))


def _device_function(module: ast.Module) -> ast.FunctionDef:
    """The innermost generated function that owns the vectorized lane loop."""
    found = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(child, ast.For)
            and isinstance(child.target, ast.Name)
            and child.target.id.startswith("vec_lane_")
            for child in ast.walk(node)
        ):
            found = node
    assert found is not None, ast.unparse(module)
    return found


def _vector_loop(function: ast.FunctionDef) -> ast.For:
    """The constexpr V-loop of the vectorized lane."""
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("vec_lane_")
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) == "cutlass.range_constexpr"
    ]
    assert len(loops) == 1, ast.unparse(function)
    return loops[0]


def _statements_above(function: ast.FunctionDef, vloop: ast.For) -> list[ast.stmt]:
    """Statements that run before the V-loop, from the kernel body inwards.

    The gathered index is bound wherever the lane-loop placement leaves it:
    before the outer lane loop when it does not depend on the lane, or inside
    that loop above the V-loop.
    """

    def walk(body: list[ast.stmt]) -> list[ast.stmt] | None:
        for index, stmt in enumerate(body):
            if stmt is vloop:
                return list(body[:index])
            if isinstance(stmt, ast.For):
                inner = walk(stmt.body)
                if inner is not None:
                    return [*body[:index], *inner]
        return None

    above = walk(function.body)
    assert above is not None, "the V-loop has no enclosing lane loop"
    return above


def _load_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Attribute) and call.func.attr == "load")
            or ast.unparse(call.func) == "cute.arch.load"
        )
    ]


def _packet_loads(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in _load_calls(node)
        if ast.unparse(call.func) == "cute.arch.load"
        and len(call.args) > 1
        and "ir.VectorType.get(" in ast.unparse(call.args[1])
    ]


@pytest.mark.parametrize(
    ("dtype", "vec_width", "carrier", "zero"),
    [
        (torch.float32, 4, "cutlass.Uint32", "cutlass.Float32(0)"),
        (torch.bfloat16, 8, "cutlass.Uint16", "cutlass.BFloat16(0)"),
    ],
)
@skipUnlessBackends(["cute"])
def test_gathered_row_load_is_one_packet_per_row(
    dtype: torch.dtype, vec_width: int, carrier: str, zero: str
) -> None:
    kernel = helion.kernel(
        embedding.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    # 35 lookups in 4-row tiles leave a partial row tile (an outer row mask)
    # on top of the gathered index bound; 256 columns over 256 // V threads
    # give every thread exactly one V-wide packet.
    code = _generate(
        kernel,
        (torch.zeros((7, 5), dtype=torch.int32), torch.empty((48, 256), dtype=dtype)),
        block_sizes=[4, 256],
        num_threads=[0, 256 // vec_width],
        cute_vector_widths=[1, vec_width],
    )
    function = _device_function(ast.parse(code))
    vloop = _vector_loop(function)
    assert ast.unparse(vloop.iter) == f"cutlass.range_constexpr({vec_width})"
    above = _statements_above(function, vloop)
    # Exactly one packet load of the gathered row sits above the V-loop ...
    packets = [call for stmt in above for call in _packet_loads(stmt)]
    assert len(packets) == 1, code
    (packet,) = packets
    assert (
        ast.unparse(packet.args[1])
        == f"ir.VectorType.get([{vec_width}], {carrier}.mlir_type)"
    )
    pointer = packet.args[0]
    assert "weight.iterator" in ast.unparse(pointer)
    # ... fed by the gathered index, which is loaded once per row (also above
    # the V-loop) rather than once per element.
    index_loads = [
        stmt
        for stmt in above
        if isinstance(stmt, ast.Assign)
        and any(call is not packet for call in _load_calls(stmt))
    ]
    assert len(index_loads) == 1, code
    (index_stmt,) = index_loads
    assert isinstance(index_stmt.targets[0], ast.Name)
    index_name = index_stmt.targets[0].id
    # The packet pointer is guarded: the row bound and the row-tile mask select
    # an in-bounds anchor for masked threads instead of reading out of range,
    # and the gathered row is part of the loaded address.
    assert isinstance(pointer, ast.IfExp), ast.unparse(pointer)
    assert index_name in ast.unparse(pointer.body)
    assert index_name in ast.unparse(pointer.test)
    assert "mask_0" in ast.unparse(pointer.test)
    assert "weight.iterator" in ast.unparse(pointer.orelse)
    # The V-loop only extracts lanes: no scalar weight or index loads remain,
    # and the per-element mask gate still zeroes masked values.
    assert not _load_calls(vloop), ast.unparse(vloop)
    assert f"else {zero}" in ast.unparse(vloop)
    assert "weight.iterator" not in ast.unparse(vloop)
    assert ast.unparse(function).count("weight.iterator") == 2


@skipUnlessBackends(["cute"])
def test_lane_varying_gather_keeps_scalar_loads() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def gather_along_lanes(perm: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        rows, cols = weight.size()
        out = torch.empty_like(weight)
        for tile_r, tile_c in hl.tile([rows, cols]):
            out[tile_r, tile_c] = weight[tile_r, perm[tile_c]]
        return out

    code = _generate(
        gather_along_lanes,
        (torch.zeros((256,), dtype=torch.int32), torch.empty((32, 256))),
        block_sizes=[4, 256],
        num_threads=[0, 64],
        cute_vector_widths=[1, 4],
    )
    function = _device_function(ast.parse(code))
    vloop = _vector_loop(function)
    # A gather along the contiguous lane dimension is not one contiguous
    # packet: the weight load stays a per-element scalar load inside the
    # V-loop while the contiguous store keeps its packet.
    assert not any(
        "weight.iterator" in ast.unparse(call) for call in _packet_loads(function)
    )
    weight_loads = [
        call for call in _load_calls(vloop) if "weight.iterator" in ast.unparse(call)
    ]
    assert len(weight_loads) == 1, code
    assert "_cute_store_u32_vec(" in ast.unparse(function)


@skipUnlessBackends(["cute"])
def test_row_mask_predicates_the_whole_packet() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def scale_rows(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.size()
        out = torch.empty_like(x)
        for tile_r, tile_c in hl.tile([rows, cols]):
            out[tile_r, tile_c] = x[tile_r, tile_c] * 2.0
        return out

    # 35 rows in 4-row tiles: the row mask is uniform across the V lanes, so
    # it guards the packet pointer rather than forcing scalar loads.
    code = _generate(
        scale_rows,
        (torch.empty((35, 256)),),
        block_sizes=[4, 256],
        num_threads=[0, 64],
        cute_vector_widths=[1, 4],
    )
    function = _device_function(ast.parse(code))
    vloop = _vector_loop(function)
    packets = [
        call for call in _packet_loads(function) if "x.iterator" in ast.unparse(call)
    ]
    assert len(packets) == 1, code
    pointer = packets[0].args[0]
    assert isinstance(pointer, ast.IfExp), ast.unparse(pointer)
    assert "mask_0" in ast.unparse(pointer.test)
    assert not _load_calls(vloop), ast.unparse(vloop)
    # The product only reaches the store, guarded by the same row mask: a
    # masked-off lane is never written, so it needs no zero
    # (``store_only_loads``).
    assert "else cutlass.Float32(0)" not in ast.unparse(vloop)


@skipUnlessBackends(["cute"])
def test_row_mask_keeps_the_packet_zero_for_an_atomic() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def scale_and_add_rows(
        x: torch.Tensor, acc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = x.size()
        out = torch.empty_like(x)
        for tile_r, tile_c in hl.tile([rows, cols]):
            scaled = x[tile_r, tile_c] * 2.0
            out[tile_r, tile_c] = scaled
            hl.atomic_add(acc, [tile_r, tile_c], scaled)
        return out, acc

    # ``scale_rows`` with the product also added atomically, a consumer
    # ``store_only_loads`` does not model: the packet's masked-off lanes keep
    # their zero.
    code = _generate(
        scale_and_add_rows,
        (torch.empty((35, 256)), torch.zeros((35, 256))),
        block_sizes=[4, 256],
        num_threads=[0, 64],
        cute_vector_widths=[1, 4],
    )
    function = _device_function(ast.parse(code))
    vloop = _vector_loop(function)
    packets = [
        call for call in _packet_loads(function) if "x.iterator" in ast.unparse(call)
    ]
    assert len(packets) == 1, code
    pointer = packets[0].args[0]
    assert isinstance(pointer, ast.IfExp), ast.unparse(pointer)
    assert "mask_0" in ast.unparse(pointer.test)
    assert "else cutlass.Float32(0)" in ast.unparse(vloop)


@pytest.mark.parametrize(
    ("leading", "shape"),
    [
        pytest.param(True, (4, 4, 8), id="leading"),
        pytest.param(False, (4, 8, 2), id="trailing"),
        pytest.param(False, (4, 8, 1), id="out-of-bounds-singleton"),
    ],
)
@skipIfRefEager("checks the generated CuTe gather")
@skipUnlessBackends(["cute"])
def test_gather_rejects_direct_scalar_source_index(
    leading: bool, shape: tuple[int, int, int]
) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather_plane(
        x: torch.Tensor, ids: torch.Tensor, plane: torch.Tensor, leading: hl.constexpr
    ) -> torch.Tensor:
        out = torch.empty_like(ids, dtype=x.dtype)
        for tm, tn in hl.tile(ids.shape):
            if leading:
                source = x[plane, tm, tn]
            else:
                source = x[tm, tn, plane]
            out[tm, tn] = torch.gather(source, 0, ids[tm, tn])
        return out

    ids = torch.zeros((4, 8), dtype=torch.long)
    args = torch.ones(shape), ids, torch.tensor(1), leading
    with pytest.raises(
        helion.exc.BackendUnsupported, match="unmasked rectangular load"
    ):
        _generate(gather_plane, args, block_sizes=[4, 8])


@pytest.mark.parametrize(("dim", "plane"), [(0, False), (-1, True)])
@skipIfRefEager("executes the generated CUDA thread program on CPU")
@skipUnlessBackends(["cute"])
def test_gather_addresses_the_selected_lane_of_each_rectangular_tile(
    dim: int, plane: bool
) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids, dim: hl.constexpr, plane: hl.constexpr):
        out = torch.empty_like(ids, dtype=x.dtype)
        for m, n in hl.tile(ids.shape):
            if plane:
                tile = x[1, m, n]
            else:
                tile = x[m, n]
            out[m, n] = torch.gather(tile, dim, ids[m, n])
        return out

    # Four tiles with distinct origins. The last row and column tiles have
    # masked lanes; a valid index within the loaded tile can select one.
    source = 1 + torch.arange(77, dtype=torch.float32).reshape(7, 11)
    x = torch.stack((source + 1000, source)) if plane else source
    ids = torch.empty_like(source, dtype=torch.long)
    expected = torch.empty_like(source)
    bm, bn = 4, 8
    for m in range(0, 7, bm):
        for n in range(0, 11, bn):
            part = source[m : m + bm, n : n + bn]
            h, w = part.shape
            padded = torch.zeros(bm, bn)
            padded[:h, :w] = part
            selected = (torch.arange(h * w).view(h, w) * 3 + 1) % padded.shape[dim]
            ids[m : m + bm, n : n + bn] = selected
            expected[m : m + bm, n : n + bn] = torch.gather(padded, dim, selected)
    args = (x, ids, dim, plane)
    code = _generate(gather, args, block_sizes=[bm, bn])
    got, _ = execute_pointwise_thread_program(code, args)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize(
    "form",
    ["scalar_alias", "loaded_singleton", "explicit_mask", "correlated", "shifted"],
)
@skipIfRefEager("checks the CuTe source replay contract")
@skipUnlessBackends(["cute"])
def test_gather_rejects_sources_without_rectangular_coordinates(form: str) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids, plane, form: hl.constexpr):
        out = torch.empty_like(ids, dtype=x.dtype)
        for m, n in hl.tile(ids.shape):
            if form == "scalar_alias":
                p = plane
                tile = x[m, n, p]
            elif form == "loaded_singleton":
                tile = x[m, n, plane[()]]
            elif form == "explicit_mask":
                tile = hl.load(x, [m, n], extra_mask=m.index[:, None] >= 2)
            elif form == "correlated":
                idx = ids[m, n] + 1
                tile = x[idx, idx]
            else:
                tile = x[m.index + 1, n]
            out[m, n] = torch.gather(tile, 0, ids[m, n])
        return out

    shape = (4, 8, 1) if form in ("scalar_alias", "loaded_singleton") else (8, 8)
    args = (
        torch.ones(shape),
        torch.zeros((4, 8), dtype=torch.int64),
        torch.tensor(1),
        form,
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="unmasked rectangular load"
    ):
        _generate(gather, args, block_sizes=[4, 8])


@skipIfRefEager("checks CuTe gathers from full slices")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("form", ["grid_plane", "full_other_axis", "two_full_slices"])
def test_gather_preserves_grid_planes_and_other_full_axes(form: str) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def grid_plane(x, ids):
        out = torch.empty_like(x)
        for i in hl.grid(x.size(0)):
            out[i, :] = torch.gather(x[i, :], -1, ids[i, :])
        return out

    @helion.kernel(backend="cute", static_shapes=True)
    def full_other_axis(x, ids):
        out = torch.empty_like(x)
        for m in hl.tile(ids.size(0), block_size=2):
            out[m, :] = torch.gather(x[m, :], 0, ids[m, :])
        return out

    @helion.kernel(backend="cute", static_shapes=True)
    def two_full_slices(x, ids):
        out = torch.empty_like(x)
        for m in hl.tile(ids.size(0)):
            out[m, :, :] = torch.gather(x[m, :, :], 2, ids[m, :, :])
        return out

    kernel, shape, blocks = {
        "grid_plane": (grid_plane, (2, 4), []),
        "full_other_axis": (full_other_axis, (2, 4), []),
        "two_full_slices": (two_full_slices, (2, 4, 8), [2]),
    }[form]
    x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    ids = torch.zeros_like(x, dtype=torch.int64)
    code = _generate(kernel, (x, ids), block_sizes=blocks)
    got, _ = execute_pointwise_thread_program(code, (x, ids))
    torch.testing.assert_close(
        got, torch.gather(x, 0 if form == "full_other_axis" else -1, ids)
    )


@skipIfRefEager("executes the ordinary CuTe gather fallback")
@skipUnlessBackends(["cute"])
def test_gather_from_a_full_iota_keeps_its_grid_plane() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids):
        out = torch.empty_like(x)
        for row in hl.grid(x.size(0)):
            lane = hl.arange(x.size(1))
            out[row, lane] = torch.gather(x[row, lane], 0, ids[row, lane])
        return out

    x = torch.arange(130, dtype=torch.int32).reshape(2, 65)
    ids = (torch.arange(130).reshape(2, 65) * 3 + 5) % 65
    code = _generate(
        gather, (x, ids), block_sizes=[], cute_fragment_bounded_gather=False
    )
    got, _ = execute_pointwise_thread_program(code, (x, ids))
    torch.testing.assert_close(got, x.gather(1, ids))


@pytest.mark.parametrize("form", ["masked", "shifted"])
@skipIfRefEager("checks the CuTe source replay contract")
@skipUnlessBackends(["cute"])
def test_gather_requires_one_full_unmasked_iota(form: str) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids, form: hl.constexpr):
        out = torch.empty_like(ids, dtype=x.dtype)
        for row in hl.grid(out.size(0)):
            lane = hl.arange(out.size(1))
            if form == "masked":
                source = hl.load(x, [row, lane], extra_mask=lane != 1)
            else:
                source = x[row, lane + 1]
            out[row, lane] = torch.gather(source, 0, ids[row, lane])
        return out

    x = torch.ones(8, 8)
    ids = torch.zeros((2, 8), dtype=torch.int64)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="unmasked rectangular load"
    ):
        _generate(
            gather, (x, ids, form), block_sizes=[], cute_fragment_bounded_gather=False
        )


@pytest.mark.parametrize("form", ["literal", "configured", "dynamic"])
@skipIfRefEager("executes the ordinary CuTe gather fallback")
@skipUnlessBackends(["cute"])
def test_gather_bounds_a_prefix_iota(form: str) -> None:
    @helion.kernel(backend="cute", static_shapes=False)
    def configured(x, ids):
        out = torch.empty_like(ids, dtype=x.dtype)
        for tile in hl.tile(ids.size(0)):
            lane = hl.arange(tile.block_size)
            out[tile] = torch.gather(x[lane], 0, ids[tile])
        return out

    @helion.kernel(backend="cute", static_shapes=False)
    def gather(x, ids, form: hl.constexpr):
        out = torch.empty_like(ids, dtype=x.dtype)
        for _row in hl.grid(1):
            if form == "literal":
                lane = hl.arange(4)
            else:
                lane = hl.arange(ids[:].size(0))
            out[lane] = torch.gather(x[lane], 0, ids[lane])
        return out

    x = 10 + torch.arange(8, dtype=torch.int32)
    ids = (
        torch.tensor([3, 0, 4, -1, 2])
        if form == "dynamic"
        else torch.tensor([3, 0, 4, -1])
    )
    width = ids.numel()
    if form == "dynamic":
        ids[1] = width  # Below the physical storage size and the padded block size.
    args = (x, ids) if form == "configured" else (x, ids, form)
    code = _generate(
        configured if form == "configured" else gather,
        args,
        block_sizes=[4] if form == "configured" else [],
        cute_fragment_bounded_gather=False,
    )
    got, _ = execute_pointwise_thread_program(code, args)
    torch.testing.assert_close(got, torch.where((ids >= 0) & (ids < width), x[ids], 0))


@skipIfRefEager("checks the CuTe source replay contract")
@skipUnlessBackends(["cute"])
def test_gather_rejects_correlated_identity_iotas() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids):
        out = torch.empty_like(x)
        for _row in hl.grid(1):
            lane = hl.arange(8)
            out[:, :] = torch.gather(x[lane, lane], 0, ids[:, :])
        return out

    args = torch.ones(8, 8), torch.zeros(8, 8, dtype=torch.int64)
    with pytest.raises(helion.exc.BackendUnsupported):
        _generate(gather, args, block_sizes=[], cute_fragment_bounded_gather=False)


@skipIfRefEager("executes a CuTe gather on a nonzero tile")
@skipUnlessBackends(["cute"])
def test_gather_keeps_a_singleton_slice_separate_from_its_tile() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids):
        out = torch.empty((1,), device=x.device, dtype=x.dtype)
        for col in hl.tile(4, 5, block_size=4):
            value = torch.gather(x[:, col], 1, ids[:, col])
            out[col.index - 4] = value.squeeze(0)
        return out

    x = torch.arange(5, dtype=torch.float32).reshape(1, 5)
    args = (x, torch.zeros_like(x, dtype=torch.int64))
    code = _generate(gather, args, block_sizes=[])
    got, _ = execute_pointwise_thread_program(code, args)
    torch.testing.assert_close(got, x[0, 4:])


@skipIfRefEager("checks CuTe gather replay of per-row masks")
@skipUnlessBackends(["cute"])
def test_gather_rejects_jagged_source() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, lengths, ids):
        out = torch.empty_like(x)
        for m in hl.tile(lengths.size(0)):
            row_length = lengths[m]
            for n in hl.jagged_tile(row_length):
                out[m, n] = torch.gather(x[m, n], 1, ids[m, n])
        return out

    args = (
        torch.ones(2, 4),
        torch.tensor([1, 4]),
        torch.full((2, 4), 3, dtype=torch.int64),
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="unmasked rectangular load"
    ):
        _generate(gather, args, block_sizes=[2, 4])


@skipIfRefEager("executes the generated gather with literal tile bounds")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("lo", [2**63 - 4, -8, -2])
def test_gather_compares_the_literal_tile_stop_without_wrapping(lo: int) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids, out, lo: hl.constexpr, hi: hl.constexpr):
        for tile in hl.tile(lo, hi, block_size=4):
            value = torch.gather(x[tile], 0, ids[tile])
            out[tile.index - tile.begin] = value
        return out

    class SignedCarrier(np.int64):
        def ir_value(self) -> np.int64:
            return self

    args = (torch.tensor([7.0]), torch.tensor([3]), torch.full((4,), -1.0), lo, lo + 4)
    code = _generate(gather, args, block_sizes=[])
    # Python's integers are unbounded, but CuTe coerces a plain RHS of 2**63
    # to signed INT64_MIN. Do not let CPU replay hide that native comparison.
    assert not any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(rhs, ast.Constant) and rhs.value == 2**63
            for rhs in node.comparators
        )
        for node in ast.walk(ast.parse(code))
    )
    with np.errstate(over="ignore"):
        out, _ = execute_pointwise_thread_program(code, args, int64=SignedCarrier)
    torch.testing.assert_close(out, torch.full((4,), 7.0))


@pytest.mark.parametrize(
    ("size", "dtype", "constant"),
    [
        (1, torch.int32, False),
        (4, torch.int32, False),
        (4, torch.int32, True),
        (1, torch.uint32, False),
    ],
)
@skipIfRefEager("checks the generated gather's physical bounds")
@skipUnlessBackends(["cute"])
def test_gather_negative_origin_masks_the_selected_lane(size, dtype, constant):
    @helion.kernel(backend="cute", index_dtype=dtype)
    def gather(x, ids, lower, upper, out, literal: hl.constexpr):
        for row in hl.grid(1):
            lo = -1 if literal else lower[0]
            hi = 1 if literal else upper[0]
            for tile in hl.tile(lo, hi, block_size=4):
                lane = tile.index - tile.begin
                out[row, lane] = torch.gather(x[tile], 0, ids[lane])
        return out

    # Give the original x[tile] load backing at -1, distinct from the values
    # the gather can reach. ID 1 selects x[0]; ID 3 is outside this tile,
    # even when broadcasting x[0] or when its absolute index 2 fits in x.
    x = torch.arange(6, 7 + size, dtype=torch.int32)[1:]
    out = torch.full((1, 4), -73, dtype=torch.int32)
    # With unsigned index storage both saved bounds can carry negative bits.
    bounds = torch.tensor((-3, -1) if dtype == torch.uint32 else (-1, 1))
    bounds = bounds.to(torch.int32)
    args = (x, torch.tensor([1, 3, 0, 0]), bounds[:1], bounds[1:], out, constant)
    source = _generate(gather, args, block_sizes=[])
    with np.errstate(over="ignore"):
        result, _ = execute_pointwise_thread_program(
            source, args, int32=np.int32, int64=np.int64, expected_writes=2
        )
    assert result.tolist() == [[7, 0, -73, -73]]


@skipIfRefEager("executes the generated gather with signed 64-bit arithmetic")
@skipUnlessBackends(["cute"])
def test_gather_padded_lane_does_not_wrap_on_a_broadcast_source() -> None:
    @helion.kernel(backend="cute")
    def gather(x, ids, lo, hi):
        out = torch.empty_like(x)
        for tile in hl.tile(lo, hi, block_size=4):
            value = torch.gather(x[tile], 0, ids[tile])
            out[tile.index - tile.begin] = value
        return out

    args = (torch.tensor([7.0]), torch.tensor([0]), 2**63 - 2, 2**63 - 1)
    code = _generate(gather, args, block_sizes=[])
    # Lane 0 is live. Lane 2 is padding, and its absolute coordinate would
    # wrap to INT64_MIN before the broadcast rewrites the address to zero.
    for lane, expected in ((0, 7.0), (2, 0.0)):
        args[1].fill_(lane)
        with np.errstate(over="ignore"):
            out, _ = execute_pointwise_thread_program(code, args, int64=np.int64)
        assert out.item() == expected


@skipIfRefEager("checks the native nested device-loop counter")
@skipUnlessBackends(["cute"])
def test_nested_gather_requires_representable_absolute_bounds() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, ids, out, lo: hl.constexpr, hi: hl.constexpr):
        for row in hl.grid(2):
            for tile in hl.tile(lo, hi, block_size=4):
                out[row, tile.index - tile.begin] = torch.gather(
                    x[row, tile], 0, ids[row, tile]
                )
        return out

    tensors = (torch.tensor([[7.0], [9.0]]), torch.tensor([[0], [3]]))
    for lo in (2**63 - 4, 2**31 - 4, -(2**31) - 6):
        args = (*tensors, torch.full((2, 4), -1.0), lo, lo + 4)
        with pytest.raises(
            helion.exc.BackendUnsupported, match="must fit signed Int32"
        ):
            _generate(gather, args, block_sizes=[])

    for lo in (-8, 8):
        args = (*tensors, torch.full((2, 4), -1.0), lo, lo + 4)
        code = _generate(gather, args, block_sizes=[])
        assert "trap;" not in code
        result, _ = execute_pointwise_thread_program(code, args)
        torch.testing.assert_close(result, tensors[0].expand(2, 4))


@skipIfRefEager("checks a dynamic range before the native device loop")
@skipUnlessBackends(["cute"])
def test_nested_gather_checks_dynamic_bounds_before_the_loop() -> None:
    @helion.kernel(backend="cute")
    def gather(x, ids, out, lo, hi):
        for row in hl.grid(2):
            for tile in hl.tile(lo, hi, block_size=4):
                out[row, tile.index - tile.begin] = torch.gather(
                    x[tile, row], 0, ids[tile, row]
                )
        return out

    x = (10 + torch.arange(16, dtype=torch.float32)).reshape(8, 2)
    ids = torch.tensor([[3, 0], [1, 2], [2, 1], [0, 3]] * 2)
    out = torch.empty((2, 4))
    code = _generate(gather, (x, ids, out, 0, 4), block_sizes=[])
    for lo in (0, 4):
        got, _ = execute_pointwise_thread_program(code, (x, ids, out, lo, lo + 4))
        torch.testing.assert_close(
            got, torch.gather(x[lo : lo + 4], 0, ids[lo : lo + 4]).T
        )

    # A narrowed exclusive end can make range empty: the trap must run before it.
    device = next(
        node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)
    )
    guard = next(
        node
        for node in device.body
        if isinstance(node, ast.If) and "trap;" in ast.unparse(node)
    )
    loop = next(node for node in device.body if isinstance(node, ast.For))
    assert guard.lineno < loop.lineno
    predicate = compile(ast.Expression(guard.test), "<nested-range>", "eval")
    for lo, hi in ((2**31 - 4, 2**31), (-(2**31) - 4, -(2**31))):
        assert eval(predicate, {"lo": lo, "hi": hi})


@skipIfRefEager("checks the narrowed step of a nested grid")
@skipUnlessBackends(["cute"])
def test_nested_grid_rejects_an_unrepresentable_step() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def stepped(out, step: hl.constexpr):
        for row in hl.grid(2):
            for col in hl.grid(0, 4, step=step):
                out[row, col] = 1
        return out

    with pytest.raises(helion.exc.BackendUnsupported, match="must fit signed Int32"):
        _generate(stepped, (torch.empty(2, 4), 2**32 + 1), block_sizes=[])
