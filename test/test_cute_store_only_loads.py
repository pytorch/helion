"""Packet loads whose masked-off lanes no store writes (``store_only_loads``).

A load whose value reaches only stores of the same tiles, through
``PointwiseLowering`` ops and ``None``/``:`` views, needs no zero for its
masked-off lanes: its packet lanes render without the per-lane select (a
bool byte keeps its compare against zero).  Any other consumer (``torch.where``
among them, a limitation of the analysis), a store of other tiles at the
value's position, or an ``extra_mask`` keeps it.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.store_only_loads import load_fill_is_dead
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl
from helion.language.memory_ops import load

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def scale_columns(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        scaled = x[tile_m, tile_n].float() * w[tile_n].float()[None, :]
        out[tile_m, tile_n] = scaled.to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def scale_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        row = x[tile_m, tile_n].float()
        out[tile_m, tile_n] = (row * 2.0).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def scale_and_sum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile(x.size()):
        row = x[tile_m, tile_n].float()
        out[tile_m, tile_n] = (row * 2.0).to(x.dtype)
        sums[tile_m] = row.sum(-1)
    return out, sums


@helion.kernel(backend="cute", static_shapes=True)
def scale_grid_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.grid(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[i, tile_n] = x[i, tile_n] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True)
def scale_tile_begin_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = tile_m.begin
        for tile_n in hl.tile(x.size(1)):
            out[row, tile_n] = x[row, tile_n] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True)
def rebound_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_n, tile_m] = x[tile_m, tile_n]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def copy_across_shared_block_sizes(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    block = hl.register_block_size(x.size(0))
    for tile_a in hl.tile(x.size(0), block_size=block):
        for tile_b in hl.tile(out.size(0), block_size=block):
            out[tile_b] = x[tile_a]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def transpose_shared_block_sizes(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    block = hl.register_block_size(x.size(0))
    for tile_m in hl.tile(x.size(0), block_size=block):
        for tile_n in hl.tile(x.size(1), block_size=block):
            out[tile_n, tile_m] = x[tile_m, tile_n]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def masked_copy(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = hl.load(
            x, [tile_m, tile_n], extra_mask=keep[tile_m, tile_n]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def select_values(m: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = torch.where(
            m[tile_m, tile_n], x[tile_m, tile_n], y[tile_m, tile_n]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def bool_to_float(m: torch.Tensor) -> torch.Tensor:
    out = torch.empty(m.size(), dtype=torch.float32, device=m.device)
    for tile_m, tile_n in hl.tile(m.size()):
        out[tile_m, tile_n] = m[tile_m, tile_n].float()
    return out


@helion.kernel(backend="cute", static_shapes=True)
def bool_rows_to_float(m: torch.Tensor) -> torch.Tensor:
    out = torch.empty(m.size(), dtype=torch.float32, device=m.device)
    for i in hl.grid(m.size(0)):
        for tile_n in hl.tile(m.size(1)):
            out[i, tile_n] = m[i, tile_n].float()
    return out


@helion.kernel(backend="cute", static_shapes=True)
def copy_and_accumulate(x: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        value = x[tile_m, tile_n]
        out[tile_m, tile_n] = value
        hl.atomic_add(acc, [tile_m, tile_n], value.float())
    return out


_BF16 = torch.bfloat16
# 1000 columns: the 1024-wide tile masks its last 24 lanes.
_PACKET_CONFIG = helion.Config(
    block_sizes=[1, 1024], num_threads=[0, 128], cute_vector_widths=[1, 8]
)


def _dead_fills(kernel: object, args: tuple[object, ...]) -> dict[str, bool]:
    bound = cpu_bind(kernel, args)  # pyrefly: ignore[bad-argument-type]
    verdicts: dict[str, bool] = {}
    with bound.env, bound.host_function:
        for graph in bound.host_function.device_ir.graphs:
            for node in graph.graph.find_nodes(op="call_function", target=load):
                tensor = node.args[0]
                assert isinstance(tensor, torch.fx.Node)
                verdicts[tensor.name] = verdicts.get(
                    tensor.name, True
                ) and load_fill_is_dead(node)
    return verdicts


def test_only_same_tile_stores_make_a_fill_dead() -> None:
    with cpu_target():
        x = torch.empty(8, 1000, dtype=_BF16)
        # Both operands reach only the store, the row vector through [None, :].
        assert _dead_fills(scale_columns, (x, torch.empty(1000, dtype=_BF16))) == {
            "x": True,
            "w": True,
        }
        # The same scaled row with and without a sum, which reads every lane
        # of the row.
        assert _dead_fills(scale_rows, (x,)) == {"x": True}
        assert _dead_fills(scale_and_sum, (x,)) == {"x": False}
        # An ``hl.grid`` index is a scalar without a mask, like an integer,
        # and so is ``tile.begin`` (``_cute_is_tile_scalar``).
        assert _dead_fills(scale_grid_rows, (x,)) == {"x": True}
        assert _dead_fills(scale_tile_begin_rows, (x,)) == {"x": True}
        # Lane (i, j) of the load is stored under the masks of (j, i).
        square = torch.empty(64, 64, dtype=_BF16)
        assert _dead_fills(rebound_store, (square,)) == {"x": False}
        # Tiles of one ``hl.register_block_size`` share one block id, which
        # no subscript tells apart (``_shared_blocks``): over 60 and 64
        # elements, lanes 60..63 of the load would be masked off and stored,
        # and a transpose stores lane (i, j) under the masks of (j, i) again.
        assert _dead_fills(
            copy_across_shared_block_sizes,
            (torch.empty(60, dtype=_BF16), torch.empty(64, dtype=_BF16)),
        ) == {"x": False}
        assert _dead_fills(transpose_shared_block_sizes, (square,)) == {"x": False}
        # An extra_mask load's zero is its documented value, and a load used
        # as a mask is no stored value.
        keep = torch.empty(8, 1000, dtype=torch.bool)
        assert _dead_fills(masked_copy, (x, keep)) == {"x": False, "keep": False}
        # A limitation: ``torch.where`` is an ``AtenLowering``, which the
        # analysis does not follow, though its inputs' zeros are as dead as
        # a pointwise op's; a masked fill keeps every select.
        assert _dead_fills(select_values, (keep, x, x)) == {
            "m": False,
            "x": False,
            "y": False,
        }


def _packet_lanes(kernel: object, args: tuple[object, ...]) -> list[str]:
    bound = cpu_bind(kernel, args)  # pyrefly: ignore[bad-argument-type]
    code = bound.to_code(bound.config_spec.normalized_config(_PACKET_CONFIG))
    return [
        line.strip()
        for line in code.splitlines()
        if "_tile_unroll_vec_" in line and "bitcast" in line
    ]


def test_store_only_packet_lanes_render_without_select() -> None:
    with cpu_target():
        x = torch.empty(8, 1000, dtype=_BF16)
        lanes = _packet_lanes(scale_columns, (x, torch.empty(1000, dtype=_BF16)))
        assert len(lanes) == 2
        assert not any(" if " in lane for lane in lanes), lanes
        # The analysis does not model atomics and fails closed on them: that
        # lane keeps its select.
        (lane,) = _packet_lanes(copy_and_accumulate, (x, torch.empty(8, 1000)))
        assert lane.endswith("if mask_1 else cutlass.BFloat16(0)"), lane


def test_store_only_bool_lanes_keep_their_compare() -> None:
    # The packet lane is the whole value of a dead fill: only the select
    # goes, and a bool byte is compared against zero as in every bool load.
    # The top-level tile defers its packet until the grid's body is wrapped;
    # the tile loop inside ``hl.grid`` emits it in place.
    with cpu_target():
        m = torch.empty(8, 1000, dtype=torch.bool)
        for kernel, config in (
            (bool_to_float, _PACKET_CONFIG),
            (
                bool_rows_to_float,
                helion.Config(
                    block_sizes=[1024], num_threads=[128], cute_vector_widths=[1, 8]
                ),
            ),
        ):
            bound = cpu_bind(kernel, (m,))  # pyrefly: ignore[bad-argument-type]
            code = bound.to_code(bound.config_spec.normalized_config(config))
            lanes = [
                line.strip()
                for line in code.splitlines()
                if "_tile_unroll_vec_" in line and ">> 8 *" in line
            ]
            assert lanes, code
            for lane in lanes:
                assert lane.endswith("& 255) != cutlass.Uint8(0)"), lane
