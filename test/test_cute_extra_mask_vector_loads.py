"""GPU-free codegen coverage for CuTe vector loads under ``hl.load(extra_mask=...)``.

A tile-vector load site used to fall back to V scalar masked loads as soon as
an ``extra_mask`` was present.  A mask term that reads the per-element lane
index is now proven uniform across the V-aligned packet when it compares the
index (plus a constant shift) against a bound that is a multiple of V, and the
packet is guarded by that term evaluated at the lane base.  A lane coordinate
``tile.index + k`` places the packet at ``lane_base + k`` when ``k`` is a
multiple of V.  Masks that can change inside a packet keep the scalar loads.
"""

from __future__ import annotations

import ast

from examples.concatenate import concat2d_dim1
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _generate(kernel: object, args: tuple[torch.Tensor, ...], **config: object) -> str:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        return bound.to_code(helion.Config.from_dict(config))


def _vector_loop(code: str) -> ast.For:
    """The single constexpr V-loop of the generated kernel."""
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("vec_lane_")
        and ast.unparse(node.iter).startswith("cutlass.range_constexpr(")
    ]
    assert len(loops) == 1, code
    return loops[0]


def _lane_body(code: str, vloop: ast.For) -> list[ast.stmt]:
    """Statements of the outer lane loop that own the V-loop."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.For) and any(
            ast.unparse(stmt) == ast.unparse(vloop) for stmt in node.body
        ):
            return node.body
    raise AssertionError(code)


def _packet_loads(statements: list[ast.stmt]) -> list[ast.Assign]:
    """Hoisted ``_tile_unroll_vec_* = <16-byte load>`` assignments."""
    return [
        stmt
        for stmt in statements
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id.startswith("_tile_unroll_vec_")
    ]


def _scalar_loads(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
    ]


def _concat_kernel() -> object:
    return helion.kernel(
        concat2d_dim1.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


def _concat_code(n1: int, n2: int, dtype: torch.dtype, vec_width: int) -> str:
    kernel = _concat_kernel()
    args = (torch.empty((8, n1), dtype=dtype), torch.empty((8, n2), dtype=dtype))
    return _generate(
        kernel,
        args,
        block_sizes=[1, 2048],
        num_threads=[0, 2048 // vec_width],
        cute_vector_widths=[1, vec_width],
    )


@pytest.mark.parametrize(
    ("dtype", "vec_width", "carrier"),
    [(torch.float32, 4, "cutlass.Uint32"), (torch.bfloat16, 8, "cutlass.Uint16")],
)
def test_concat_masked_loads_hoist_as_guarded_packets(
    dtype: torch.dtype, vec_width: int, carrier: str
) -> None:
    # 512 and 768 are multiples of V, so ``tile1.index < 512`` and
    # ``tile1.index >= 512`` cannot change inside an aligned packet, and the
    # ``tile1.index - 512`` coordinate of ``y`` keeps its packet aligned.
    code = _concat_code(512, 768, dtype, vec_width)
    vloop = _vector_loop(code)
    packets = _packet_loads(_lane_body(code, vloop))
    assert len(packets) == 2, code
    x_packet, y_packet = (ast.unparse(packet.value) for packet in packets)
    vector_type = f"ir.VectorType.get([{vec_width}], {carrier}.mlir_type)"
    assert vector_type in x_packet and vector_type in y_packet
    # The extra mask guards the packet pointer at the lane base, not per lane.
    assert "operator.lt(lane_base_1, 512)" in x_packet, x_packet
    assert "operator.ge(lane_base_1, 512)" in y_packet, y_packet
    assert "cutlass.Int32(lane_base_1 - 512) < 768" in y_packet, y_packet
    # The y packet is placed at the shifted lane base.
    assert "cutlass.Int32(lane_base_1 - 512)) * cutlass.Int32(y.layout.stride[1])" in (
        y_packet
    ), y_packet
    # No per-element scalar loads remain inside the V-loop.
    assert not _scalar_loads(vloop), ast.unparse(vloop)


def test_bound_not_a_multiple_of_vector_width_stays_scalar() -> None:
    # A packet starting at 500 spans 500..503: ``tile1.index < 502`` differs
    # inside it, and ``tile1.index - 502`` is a misaligned coordinate.
    code = _concat_code(502, 300, torch.float32, 4)
    vloop = _vector_loop(code)
    assert not _packet_loads(_lane_body(code, vloop)), code
    assert len(_scalar_loads(vloop)) == 2, ast.unparse(vloop)


def test_bound_multiple_of_vector_width_but_not_of_eight() -> None:
    # 500 is a multiple of 4 but not of 8: only the narrower packet is uniform.
    code = _concat_code(500, 512, torch.float32, 4)
    assert len(_packet_loads(_lane_body(code, _vector_loop(code)))) == 2, code
    code = _concat_code(500, 512, torch.bfloat16, 8)
    assert not _packet_loads(_lane_body(code, _vector_loop(code))), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _keep_even_columns(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(out.size()):
        out[tile0, tile1] = hl.load(
            x, [tile0, tile1], extra_mask=(tile1.index % 2 == 0)[None, :]
        )
    return out


def test_lane_varying_extra_mask_stays_scalar() -> None:
    args = (torch.empty((8, 1024)),)
    code = _generate(
        _keep_even_columns,
        args,
        block_sizes=[1, 1024],
        num_threads=[0, 256],
        cute_vector_widths=[1, 4],
    )
    vloop = _vector_loop(code)
    assert not _packet_loads(_lane_body(code, vloop)), code
    assert len(_scalar_loads(vloop)) == 1, ast.unparse(vloop)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prefix_columns(x: torch.Tensor, last: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(out.size()):
        out[tile0, tile1] = hl.load(
            x, [tile0, tile1], extra_mask=(tile1.index <= last)[None, :]
        )
    return out


@pytest.mark.parametrize(("last", "packets"), [(511, 1), (512, 0)])
def test_inclusive_bound_uniformity(last: int, packets: int) -> None:
    # ``index <= 511`` is ``index < 512``: uniform across aligned packets of 4.
    # ``index <= 512`` splits the packet 512..515.
    args = (torch.empty((8, 1024)), last)
    code = _generate(
        _prefix_columns,
        args,
        block_sizes=[1, 1024],
        num_threads=[0, 256],
        cute_vector_widths=[1, 4],
    )
    vloop = _vector_loop(code)
    assert len(_packet_loads(_lane_body(code, vloop))) == packets, code
    assert len(_scalar_loads(vloop)) == 1 - packets, ast.unparse(vloop)
