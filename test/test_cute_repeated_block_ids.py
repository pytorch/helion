"""CuTe refuses a store whose target tile binds one block id to two axes
(``cute/repeated_block_ids.py``): the SIMT lowering gives each block id one
lane coordinate, so the store would reach only the tile's diagonal."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _repeated_tile_store(x: torch.Tensor) -> torch.Tensor:
    rows, channels = x.shape
    out = torch.zeros([channels, rows, channels], dtype=x.dtype, device=x.device)
    for c in hl.tile(channels):
        for r in hl.tile(rows):
            out[c, r, c] = x[r, c][None, :, :]
    return out


@skipIfRefEager("generates code")
def test_a_store_target_repeating_a_block_id_is_refused() -> None:
    # ``out[c, r, c]`` broadcasts the ``[r, c]`` value along both of ``c``'s
    # axes, and eager writes the whole tile; one lane coordinate for both
    # axes stored only ``out[i, r, i]``.
    x = torch.empty(784, 128, dtype=torch.bfloat16)
    with cpu_target():
        bound = cpu_bind(_repeated_tile_store, (x,))
        config = bound.config_spec.normalized_config(
            helion.Config(block_sizes=[64, 32])
        )
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="indexes two axes of one tensor by block id 0",
        ):
            bound.to_code(config)


@skipIfRefEager("generates code")
def test_a_repeated_block_of_one_element_renders() -> None:
    # A one-element block's diagonal is the whole tile.
    x = torch.empty(64, 64)
    with cpu_target():
        bound = cpu_bind(_repeated_tile_store, (x,))
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(block_sizes=[1, 16]))
        )
    assert re.search(r"\(out\.iterator[^\n]*\)\.store\(", code), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fixed_one_element_store(x: torch.Tensor) -> torch.Tensor:
    batch, rows = x.shape
    out = torch.zeros([batch, rows, batch], dtype=x.dtype, device=x.device)
    for b, r in hl.tile([batch, rows], block_size=[1, None]):
        out[b, r, b] = x[b, r][:, :, None]
    return out


@skipIfRefEager("generates code")
def test_a_repeated_fixed_block_of_one_element_renders() -> None:
    # The block variable of a fixed one-element tile keeps a larger hint; the
    # config's block size decides.
    x = torch.empty(64, 64)
    with cpu_target():
        bound = cpu_bind(_fixed_one_element_store, (x,))
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(block_sizes=[16]))
        )
    assert re.search(r"\(out\.iterator[^\n]*\)\.store\(", code), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _expanded_tile_store(x: torch.Tensor) -> torch.Tensor:
    rows, channels = x.shape
    out = torch.zeros([channels, rows, channels], dtype=x.dtype, device=x.device)
    for c in hl.tile(channels):
        for r in hl.tile(rows):
            v = x[r, c]
            out[c, r, c] = v[None, :, :].expand(v.size(1), v.size(0), v.size(1))
    return out


@skipIfRefEager("generates code")
def test_an_expanded_value_on_two_tiles_of_a_block_is_refused() -> None:
    # The broadcast store's loop walks ``c``'s broadcast axis only when its
    # other axis is a tensor index; with two tiles both take the lane.
    x = torch.empty(64, 64)
    with cpu_target():
        bound = cpu_bind(_expanded_tile_store, (x,))
        config = bound.config_spec.normalized_config(
            helion.Config(block_sizes=[16, 16])
        )
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="indexes two axes of one tensor by block id 0",
        ):
            bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gathered_column_store(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0), x.size(0)], dtype=x.dtype, device=x.device)
    for c in hl.tile(x.size(0)):
        out[c, idx[c]] = 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_column_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0), x.size(0) + 1], dtype=x.dtype, device=x.device)
    for c in hl.tile(x.size(0)):
        out[c, c.index + 1] = 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _outer_index_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0), x.size(0)], dtype=x.dtype, device=x.device)
    for c in hl.tile(x.size(0)):
        out[c.index[:, None], c.index[None, :]] = 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_broadcast_store(x: torch.Tensor) -> torch.Tensor:
    rows, channels = x.shape
    out = torch.zeros([channels, rows, channels + 1], dtype=x.dtype, device=x.device)
    for c in hl.tile(channels):
        for r in hl.tile(rows):
            out[c, r, c.index + 1] = x[r, c][None, :, :]
    return out


@pytest.mark.parametrize(
    "kernel",
    [
        _gathered_column_store,
        _shifted_column_store,
        _outer_index_store,
        _shifted_broadcast_store,
    ],
    ids=lambda kernel: kernel.name,
)
@skipIfRefEager("generates code")
def test_a_tensor_index_repeating_a_block_id_is_refused(
    kernel: helion.Kernel,
) -> None:
    # A tensor index's dims take the block of the tile they follow, as in a
    # load: ``out[c, idx[c]]`` addresses ``[c, c]`` like ``x[c, idx[c]]``, and
    # one lane coordinate for both axes stored only its diagonal.
    x = torch.empty(64, 64)
    idx = torch.zeros(64, dtype=torch.int64)
    args = (x, idx) if kernel is _gathered_column_store else (x,)
    with cpu_target():
        bound = cpu_bind(kernel, args)
        config = bound.config_spec.normalized_config(
            helion.Config(block_sizes=[16] * len(bound.config_spec.block_sizes))
        )
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="indexes two axes of one tensor by block id 0",
        ):
            bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _paired_index_store(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0), x.size(0)], dtype=x.dtype, device=x.device)
    for c in hl.tile(x.size(0)):
        out[c.index[:, None], idx[c][:, None]] = 1.0
    return out


@skipIfRefEager("generates code")
def test_tensor_indices_broadcast_together_count_their_block_once() -> None:
    # Two ``[c, 1]`` indices pair up element by element, one store per lane.
    x = torch.empty(64, 64)
    idx = torch.zeros(64, dtype=torch.int64)
    with cpu_target():
        bound = cpu_bind(_paired_index_store, (x, idx))
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(block_sizes=[16]))
        )
    assert re.search(r"\(out\.iterator[^\n]*\)\.store\(", code), code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_row_store(data: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m = data.size(0)
    out = torch.zeros([m, 2 * m, 4], device=data.device, dtype=data.dtype)
    for tile_m in hl.tile(m):
        val = hl.load(data, [tile_m, hl.arange(4)])
        val_3d = val[:, None, :].expand(val.size(0), val.size(0), 4)
        hl.store(out, [rows[tile_m], tile_m.index, hl.arange(4)], val_3d)
    return out


@skipIfRefEager("generates code")
def test_a_broadcast_store_with_a_lane_loop_renders() -> None:
    # ``rows[tile_m]`` and ``tile_m.index`` index two axes by one block, and
    # the value is broadcast along the second: the store's lowering walks that
    # axis with a loop of its own, writing the whole tile.
    data = torch.empty(10, 4)
    rows = torch.zeros(10, dtype=torch.int32)
    with cpu_target():
        bound = cpu_bind(_broadcast_row_store, (data, rows))
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(block_sizes=[8]))
        )
    assert re.search(r"for bcast_lane\w* in range\(8\)", code), code


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _traced_prefix_store(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    half = n // 2
    for t in hl.tile(m):
        out[t, :half] = x[t, :half]
    return out


@skipIfRefEager("generates code")
def test_a_store_through_a_traced_slice_bound_renders() -> None:
    # A slice beside the tile is not counted, whatever its stop (here a node
    # of the graph, ``sym_min``).
    x = torch.empty(64, 128)
    with cpu_target():
        bound = cpu_bind(_traced_prefix_store, (x,))
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(block_sizes=[16]))
        )
    assert re.search(r"\(out\.iterator[^\n]*\)\.store\(", code), code
