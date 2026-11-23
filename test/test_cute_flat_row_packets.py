"""Row packets of a flattened multi-dim tile on a permuted view (CPU codegen).

A flattened tile addresses a packet by its flat lane base only on a tensor it
walks contiguously.  On a view whose dims are the tile's dims in order and
whose innermost dim keeps unit stride (``x.view(b, s, h, d).permute(0, 2, 1,
3)``), every V-aligned chunk of the flat walk still lies inside one row, so
the packet is addressed at the element its lane base decodes to
(``memory_ops._cute_flat_row_packet``).  These tests pin that form and every
case the proof declines; ``test_cute_pointwise_vec.py`` checks the results on
a GPU.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute import memory_ops as cute_memory_ops
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import kernel_body
from helion._testing.cute_flat_row_kernels import cast_copy as _cast_copy
from helion._testing.cute_flat_row_kernels import cast_into as _cast_into
from helion._testing.cute_flat_row_kernels import (
    scale_between_views as _scale_between_views,
)
from helion._testing.cute_flat_row_kernels import scale_in_place as _scale_in_place
from helion._testing.cute_flat_row_kernels import scale_into as _scale_into
import helion.language as hl
from helion.language.memory_ops import load as language_load
from helion.language.memory_ops import store as language_store

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion._compiler.inductor_lowering import CodegenState


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_render_target():
        yield


def _bias_cast_copy(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    out = torch.empty([b, h, s, d], dtype=torch.float32, device=x.device)
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        out[tile_b, tile_h, tile_s, tile_d] = (
            x[tile_b, tile_h, tile_s, tile_d].to(torch.float32) + bias[tile_d]
        )
    return out


def _two_reads(
    x: torch.Tensor, out: torch.Tensor, out2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, s, d = x.shape
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        out[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d] * 2
        out2[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d] + 1
    return out, out2


# A flattened tile of 256 elements on 16 threads: four V4 packets per thread.
_TILE = {
    "flatten_loops": [True],
    "block_sizes": [2, 1, 8, 16],
    "num_threads": [1, 1, 1, 16],
    "cute_vector_widths": [1, 1, 1, 4],
}


def _render(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    *,
    static_shapes: bool = True,
    **overrides: object,
) -> str:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=static_shapes, autotune_effort="none"
    )
    bound = cpu_bind(kernel, args)
    config = bound.config_spec.default_config().config | _TILE | overrides
    return kernel_body(bound.to_code(helion.Config.from_dict(config)))


def _split_heads(
    batch: int = 4, seq: int = 8, heads: int = 2, head_dim: int = 16
) -> torch.Tensor:
    """``[batch, heads, seq, head_dim]`` view of a ``[batch, seq, heads,
    head_dim]`` activation: strides ``(seq * heads * head_dim, head_dim,
    heads * head_dim, 1)``."""
    return torch.empty(batch, seq, heads, head_dim, dtype=torch.bfloat16).permute(
        0, 2, 1, 3
    )


def _packet_loads(body: str, tensor: str = "x") -> list[str]:
    """The pointer expressions of the packet loads of ``tensor``."""
    return [
        line.strip()
        for line in body.splitlines()
        if "ir.VectorType.get(" in line and f"{tensor}.iterator" in line
    ]


def _row_address(tensor: str, extents: tuple[int, int, int, int]) -> str:
    """The pointer of the element a flat lane base decodes to."""
    base = "cutlass.Uint32(lane_base_0_1_2_3)"
    inner = [extents[1] * extents[2] * extents[3], extents[2] * extents[3], extents[3]]
    return (
        f"{tensor}.iterator"
        f" + cutlass.Int32({base} // cutlass.Uint32({inner[0]}))"
        f" * cutlass.Int32({tensor}.layout.stride[0])"
        f" + cutlass.Int32({base} // cutlass.Uint32({inner[1]})"
        f" % cutlass.Uint32({extents[1]}))"
        f" * cutlass.Int32({tensor}.layout.stride[1])"
        f" + cutlass.Int32({base} // cutlass.Uint32({inner[2]})"
        f" % cutlass.Uint32({extents[2]}))"
        f" * cutlass.Int32({tensor}.layout.stride[2])"
        f" + cutlass.Int32({base} % cutlass.Uint32({extents[3]}))"
        f" * cutlass.Int32({tensor}.layout.stride[3])"
    )


def _assert_scalar_load(body: str) -> None:
    assert _packet_loads(body) == []
    assert "cutlass.Uint32(lane_base" not in body
    assert re.search(r"x\.layout\.stride\[3\]\)\)\.load\(\)", body)


def test_permuted_view_loads_row_packets() -> None:
    """One packet per lane at the decoded row, guarded by the flat extent."""
    body = _render(_cast_copy, (_split_heads(),))
    (load,) = _packet_loads(body)
    address = _row_address("x", (4, 2, 8, 16))
    assert f"cute.arch.load({address} if lane_base_0_1_2_3 < 1024" in load
    assert "else x.iterator + cutlass.Int32(0), ir.VectorType.get([4]," in load
    assert ").load()" not in body
    # The contiguous output keeps the flat pointer.
    assert "(out.iterator + cutlass.Int32(lane_base_0_1_2_3), _tile_store" in body


def test_masked_flat_tail_guards_the_packet() -> None:
    """A partial last tile sends its masked-off threads to the view's first
    row: 768 elements in tiles of 512."""
    body = _render(
        _cast_copy,
        (_split_heads(seq=6),),
        block_sizes=[4, 1, 8, 16],
        num_threads=[1, 1, 2, 16],
    )
    (load,) = _packet_loads(body)
    address = _row_address("x", (4, 2, 6, 16))
    assert f"({address} if lane_base_0_1_2_3 < 768 else x.iterator" in load
    assert "mask_0_1_2_3 = offsets_0_1_2_3 < 768" in body


def test_proven_bounds_drop_the_guard() -> None:
    body = _render(_cast_copy, (_split_heads(),), cute_proven_bounds=True)
    (load,) = _packet_loads(body)
    assert f"cute.arch.load({_row_address('x', (4, 2, 8, 16))}, ir." in load


def test_expanded_outer_dim_keeps_the_packet() -> None:
    """A stride-0 dim adds nothing to the address and is trivially aligned."""
    x = torch.empty(4, 1, 8, 16, dtype=torch.bfloat16).expand(4, 2, 8, 16)
    assert x.stride() == (128, 0, 16, 1)
    (load,) = _packet_loads(_render(_cast_copy, (x,)))
    assert _row_address("x", (4, 2, 8, 16)) in load


def test_every_packet_dtype_loads_row_packets() -> None:
    """The row address does not depend on the dtype; the packet is the one
    the dtype has on any tile: a vector of 16- or 32-bit words, or, for the
    one-byte dtypes, four bytes in one 32-bit word.  A dtype without a tile
    packet (uint8, int64) keeps its per-element loads."""
    words16 = "ir.VectorType.get([4], cutlass.Uint16.mlir_type)))"
    words32 = "ir.VectorType.get([4], cutlass.Uint32.mlir_type)))"
    packed = "cutlass.Uint32))"
    address = _row_address("x", (4, 2, 8, 16))
    # One case per carrier and per decode of it: the three one-byte dtypes
    # unpack the word differently, and the 2- and 4-byte dtypes each have an
    # integer and a float form (bf16 is the form of the tests above).
    for dtype, carrier in (
        (torch.int8, packed),
        (torch.bool, packed),
        (torch.float8_e4m3fn, packed),
        (torch.int16, words16),
        (torch.float16, words16),
        (torch.int32, words32),
        (torch.float32, words32),
    ):
        x = torch.empty(4, 8, 2, 16, dtype=dtype).permute(0, 2, 1, 3)
        body = _render(_cast_copy, (x,))
        (load,) = (line for line in body.splitlines() if "x.iterator" in line)
        assert f"_lane_loads_0_0.append(cute.arch.load({address} if" in load, dtype
        assert load.endswith(carrier), dtype
    for dtype in (torch.uint8, torch.int64):
        x = torch.empty(4, 8, 2, 16, dtype=dtype).permute(0, 2, 1, 3)
        body = _render(_cast_copy, (x,))
        assert "_lane_loads_" not in body, dtype
        assert "cutlass.Uint32(lane_base" not in body, dtype


def test_store_into_permuted_view_is_a_row_packet() -> None:
    x = torch.empty(4, 2, 8, 16, dtype=torch.bfloat16)
    body = _render(_scale_into, (x, _split_heads()))
    # The contiguous input keeps the flat pointer; the output takes the row.
    assert "x.iterator + cutlass.Int32(lane_base_0_1_2_3)" in body
    assert (
        f"_cute_store_u16_vec({_row_address('out', (4, 2, 8, 16))}, _tile_store" in body
    )
    assert ".store(" not in body


def test_broadcast_operand_stays_scalar_beside_the_row_packet() -> None:
    body = _render(
        _bias_cast_copy, (_split_heads(), torch.empty(16, dtype=torch.float32))
    )
    assert len(_packet_loads(body)) == 1
    assert _packet_loads(body, "bias") == []
    assert "bias.layout.stride[0])).load()" in body


def test_load_hint_reaches_the_row_packet() -> None:
    """The eviction slot of the load was inert on its scalar form."""
    body = _render(_cast_copy, (_split_heads(),), load_eviction_policies=["l2_last"])
    assert f"_cute_load_l2_evict_last_8b({_row_address('x', (4, 2, 8, 16))}" in body


def test_streaming_store_and_packet_prefetch_keep_the_row_packet() -> None:
    """The store policy rewrites the flush only; ``cute_packet_prefetch``
    needs rank-1 tensors, so it leaves a multi-dim tile unchanged."""
    plain = _render(_cast_copy, (_split_heads(),), cute_proven_bounds=True)
    streaming = _render(
        _cast_copy,
        (_split_heads(),),
        cute_proven_bounds=True,
        cute_store_policy="streaming",
    )
    assert _packet_loads(streaming) == _packet_loads(plain)
    assert "_cute_store_u32x4_streaming(out.iterator" in streaming
    prefetch = _render(
        _cast_copy, (_split_heads(),), cute_proven_bounds=True, cute_packet_prefetch=2
    )
    assert prefetch == plain


# Declines: each case breaks one premise of the proof and keeps the
# per-element load.


def test_unaligned_storage_offset_stays_scalar() -> None:
    """One element into its storage, every row starts off the packet grid."""
    storage = torch.empty(4 * 8 * 2 * 16 + 4, dtype=torch.bfloat16)
    _assert_scalar_load(
        _render(_cast_copy, (storage[1:-3].view(4, 8, 2, 16).permute(0, 2, 1, 3),))
    )
    # Four elements in, V4 packets are aligned and V8 packets are not.
    aligned = storage[4:].view(4, 8, 2, 16).permute(0, 2, 1, 3)
    assert len(_packet_loads(_render(_cast_copy, (aligned,)))) == 1
    _assert_scalar_load(
        _render(_cast_copy, (aligned,), cute_vector_widths=[1, 1, 1, 8])
    )


def test_outer_stride_off_the_packet_grid_stays_scalar() -> None:
    """Rows 10 elements apart hold V2 packets, not V4."""
    x = torch.empty(4, 4, 10, dtype=torch.bfloat16)[:, :, :8]
    x = x.permute(1, 0, 2).unsqueeze(0)
    assert x.shape == (1, 4, 4, 8)
    assert x.stride() == (40, 10, 40, 1)
    tile = {"block_sizes": [1, 1, 4, 8], "num_threads": [1, 1, 1, 8]}
    _assert_scalar_load(_render(_cast_copy, (x,), **tile))
    body = _render(_cast_copy, (x,), **tile, cute_vector_widths=[1, 1, 1, 2])
    assert len(_packet_loads(body)) == 1


def test_reordered_loops_stay_scalar() -> None:
    """The flat walk no longer runs along the innermost dim."""
    _assert_scalar_load(
        _render(_cast_copy, (_split_heads(),), loop_orders=[[0, 1, 3, 2]])
    )


def test_unit_stride_dim_that_is_not_innermost_stays_scalar() -> None:
    """A transposed innermost dim: consecutive elements of the flat walk are
    eight elements apart in memory."""
    x = torch.empty(4, 2, 16, 8, dtype=torch.bfloat16).permute(0, 1, 3, 2)
    assert x.stride() == (256, 128, 1, 8)
    body = _render(_cast_copy, (x,))
    assert _packet_loads(body) == []
    assert "cutlass.Uint32(lane_base" not in body
    assert ").load()" in body


def test_inner_extent_not_a_multiple_of_the_packet_stays_scalar() -> None:
    """Six-element rows: a V4 chunk of the flat walk straddles two rows.  The
    contiguous output still takes its flat packet."""
    body = _render(
        _cast_copy,
        (_split_heads(head_dim=6),),
        block_sizes=[2, 1, 8, 4],
        num_threads=[1, 1, 4, 4],
    )
    _assert_scalar_load(body)
    assert "(out.iterator + cutlass.Int32(lane_base_0_1_2_3), _tile_store" in body


def test_rows_of_partial_packets_at_an_aligned_pitch_stay_scalar() -> None:
    """Rows of 12 elements, 16 apart: the base and every outer stride sit on
    the V8 grid, so only the row extent says that a V8 chunk of the flat walk
    straddles two rows (and would fault its 16-byte load).  V4 divides it."""
    x = torch.empty(4, 2, 6, 16, dtype=torch.bfloat16)[..., :12]
    assert x.stride() == (192, 96, 16, 1)
    tile = {"block_sizes": [1, 1, 8, 16], "num_threads": [1, 1, 1, 16]}
    body = _render(_cast_copy, (x,), **tile, cute_vector_widths=[1, 1, 1, 8])
    _assert_scalar_load(body)
    assert "(out.iterator + cutlass.Int32(lane_base_0_1_2_3), _tile_store" in body
    (load,) = _packet_loads(_render(_cast_copy, (x,), **tile))
    assert _row_address("x", (4, 2, 6, 12)) in load


def test_tensor_larger_than_the_tile_stays_scalar() -> None:
    """The tile walks the output's shape: a flat offset decoded by the
    tile's extents is another element of a wider or longer input."""
    tile = {"block_sizes": [1, 1, 8, 16], "num_threads": [1, 1, 2, 16]}
    wider = torch.empty(4, 6, 2, 16, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    _assert_scalar_load(_render(_cast_into, (wider, torch.empty(4, 2, 6, 12)), **tile))
    longer = torch.empty(4, 9, 2, 16, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    _assert_scalar_load(_render(_cast_into, (longer, torch.empty(4, 2, 6, 16)), **tile))
    # The same view at the tile's own shape takes its packets.
    exact = torch.empty(4, 6, 2, 16, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    (load,) = _packet_loads(
        _render(_cast_into, (exact, torch.empty(4, 2, 6, 16)), **tile)
    )
    assert _row_address("x", (4, 2, 6, 16)) in load


def test_in_place_update_keeps_per_element_accesses() -> None:
    """A tensor the tile both loads and stores: the barrier pass cannot
    confine a row packet to its lane and would refuse the kernel, so both
    accesses stay per element, as before row packets existed, with the
    barrier that orders them."""
    body = _render(_scale_in_place, (_split_heads(),))
    _assert_scalar_load(body)
    assert "_cute_store_" not in body
    assert re.search(r"x\.layout\.stride\[3\]\)\)\.store\(", body)
    assert body.count("cute.arch.sync_threads()") == 1


def test_views_of_one_storage_keep_per_element_accesses() -> None:
    """A load through one view of a storage and a store through another are
    one tensor to the barrier pass (it would refuse row packets of the
    pair), whatever the kernel calls them: both stay per element, with the
    barrier, as before row packets existed."""
    x = torch.empty(4, 8, 2, 16, dtype=torch.bfloat16)
    body = _render(_scale_between_views, (x,))
    assert _packet_loads(body, "src") == []
    assert "cutlass.Uint32(lane_base" not in body
    assert "_cute_store_" not in body
    assert re.search(r"src\.layout\.stride\[3\]\)\)\.load\(\)", body)
    assert re.search(r"dst\.layout\.stride\[3\]\)\)\.store\(", body)
    assert body.count("cute.arch.sync_threads()") == 1


def test_tensor_read_twice_keeps_its_row_packets() -> None:
    """Two loads of one tensor never pair; the stores name other tensors."""
    out = torch.empty(4, 2, 8, 16, dtype=torch.bfloat16)
    body = _render(_two_reads, (_split_heads(), out, out.clone()))
    assert all(_row_address("x", (4, 2, 8, 16)) in load for load in _packet_loads(body))
    assert _packet_loads(body)
    assert ").load()" not in body


def test_access_of_an_unnamed_tensor_counts_as_a_pair() -> None:
    """A store whose target is no host tensor (a stack tensor, a tensor made
    on the device) may be one of the loaded tensor."""
    kernel = helion.kernel(
        _cast_copy, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = cpu_bind(kernel, (_split_heads(),))
    graphs = bound.host_function.device_ir.graphs
    nodes = [node for info in graphs for node in info.graph.nodes]
    (load,) = (node for node in nodes if node.target is language_load)
    (store,) = (node for node in nodes if node.target is language_store)

    def unpaired(node: torch.fx.Node) -> bool:
        state = SimpleNamespace(
            fx_node=node, codegen=SimpleNamespace(codegen_graphs=graphs)
        )
        return cute_memory_ops._cute_flat_row_packet_meets_no_write(
            cast("CodegenState", state), node.args[0].meta["val"]
        )

    assert unpaired(load)
    assert unpaired(store)
    store.args = (load, *store.args[1:])
    assert not unpaired(load)


def test_dynamic_shapes_stay_scalar() -> None:
    _assert_scalar_load(_render(_cast_copy, (_split_heads(),), static_shapes=False))


def test_tile_past_the_int32_decode_stays_scalar() -> None:
    """2**31 elements: the lane base no longer fits the unsigned decode.
    Eight elements fewer, and the masked tail of the last tile still reaches
    2**31.  A view that ends a tile earlier takes its packets."""
    tile = {"block_sizes": [2, 8, 1, 8], "num_threads": [1, 8, 1, 2]}
    for shape in ((65536, 4096, 1, 8), (16383, 16385, 1, 8)):
        x = torch.empty(1, 1, 1, 8, dtype=torch.bfloat16).expand(*shape)
        body = _render(_cast_copy, (x,), **tile)
        assert _packet_loads(body) == []
        assert "cutlass.Uint32(lane_base" not in body
    x = torch.empty(1, 1, 1, 8, dtype=torch.bfloat16).expand(65535, 4096, 1, 8)
    (load,) = _packet_loads(_render(_cast_copy, (x,), **tile))
    assert "cutlass.Uint32(lane_base_0_1_2_3) // cutlass.Uint32(32768)" in load
    assert "if lane_base_0_1_2_3 < 2147450880 else x.iterator" in load
