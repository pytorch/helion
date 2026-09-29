"""CPU codegen coverage for CuTe vector loads with a gathered row coordinate."""

from __future__ import annotations

import ast

from examples.embedding import embedding
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl


def _generate(kernel: object, args: tuple[torch.Tensor, ...], **config: object) -> str:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
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
    """Statements of the outer lane loop that precede the V-loop."""
    for node in ast.walk(function):
        if isinstance(node, ast.For) and any(stmt is vloop for stmt in node.body):
            return node.body[: node.body.index(vloop)]
    raise AssertionError("the V-loop has no enclosing lane loop")


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
    assert "else cutlass.Float32(0)" in ast.unparse(vloop)
