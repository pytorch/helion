"""CPU tests: ``cute_staged_store`` writes a CTA's stores through a
shared-memory image flushed as packets, declining wherever the generated
kernel is not proved to write one aligned span of an output no load can
alias; the autotuner offers the knob, and the packet loads of the gather, on
the kernel's structure and seeds the register tile only with the staged
store."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune.heuristics import cute as heuristics
from helion._compiler.cute.staged_stores import stage_cta_stores
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.unfold_kernels import register_tile_config
from helion._testing.unfold_kernels import sliding_windows
from helion._testing.unfold_kernels import sliding_windows_and_copy
from helion._testing.unfold_kernels import sliding_windows_into
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

# 8 threads of one 8-lane bf16 packet each on 64 flat elements, the 3 window
# slots a per-thread lane loop over one 4-slot tile.
_TILE = register_tile_config(threads=8, window_block=4)
_SCALAR = {"block_sizes": [64, 4], "num_threads": [8, 1]}


def _bound(
    kernel: Kernel[Any], args: tuple[object, ...], **settings: object
) -> BoundKernel[Any]:
    return cpu_bind(
        helion.kernel(kernel.fn, backend="cute", **{"static_shapes": True, **settings}),
        args,
    )


def _code(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    config: dict[str, object],
    **settings: object,
) -> str:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = _bound(kernel, args, **settings)
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config.from_dict(config))
        )


def _windows(batch: int = 2, seq: int = 8, channels: int = 16) -> torch.Tensor:
    return torch.empty(batch, seq, channels, dtype=torch.bfloat16)


def _direct_stores(code: str) -> int:
    return code.count("(out.iterator + ") - code.count("cute.arch.store(out.iterator")


# ---------------------------------------------------------------------------
# The staged store
# ---------------------------------------------------------------------------


def test_a_tile_is_staged_in_shared_memory_and_flushed_in_packets() -> None:
    code = _code(sliding_windows, (_windows(), 3, 1), _TILE)
    # 64 elements x 3 slots of bf16: the image of a CTA's tile.
    assert "= cute.arch.alloc_smem(cutlass.BFloat16, 192, alignment=16)" in code
    # Its base is the tile's first element: 192 elements per tile of 64.
    assert "_cta_stage_base = cutlass.Int32(pid_0) * 192 + " in code
    # Every store goes to the image at the element's offset in the tile; the
    # slot of the padded tile that no window has stores nothing.
    assert _direct_stores(code) == 0
    assert code.count("- _cta_stage_base)).store(") == 3
    assert (
        "(_cta_stage + (cutlass.Int32(indices_0_u0) * cutlass.Int32(3) + "
        "cutlass.Int32(indices_1_u0) * cutlass.Int32(1) - _cta_stage_base))"
        ".store(cutlass.BFloat16(" in code
    )
    # One barrier, then 24 packets over 8 threads: thread t moves packets t,
    # t + 8 and t + 16, so a warp's stores are consecutive.
    assert code.count("cute.arch.sync_threads()") == 1
    flush = code[code.index("cute.arch.sync_threads()") :]
    assert flush.count("cute.arch.load(_cta_stage + ") == 3
    assert flush.count("cute.arch.store(out.iterator + (_cta_stage_base + ") == 3
    for first in (0, 8, 16):
        assert f"(_cta_stage_thread + {first}) * 8" in flush
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in flush
    assert "_cta_stage_full" not in code


def test_fp32_tiles_flush_four_element_packets() -> None:
    config = register_tile_config(threads=8, window_block=4, vec=4)
    code = _code(sliding_windows, (torch.empty(2, 8, 16), 3, 1), config)
    assert "= cute.arch.alloc_smem(cutlass.Float32, 96, alignment=16)" in code
    assert "ir.VectorType.get([4], cutlass.Uint32.mlir_type)" in code
    assert "(_cta_stage_thread + 16) * 4" in code
    assert _direct_stores(code) == 0


def test_scalar_layouts_stage_too() -> None:
    # One flat element per thread and its slots as lanes; then the slots on
    # threads as well (a 32 x 4 thread block, linear thread ids).
    for config in (
        {"block_sizes": [64, 4], "num_threads": [64, 1]},
        {"block_sizes": [32, 4], "num_threads": [32, 4]},
    ):
        code = _code(
            sliding_windows, (_windows(), 3, 1), {**config, "cute_staged_store": True}
        )
        assert _direct_stores(code) == 0
        assert code.count("cute.arch.sync_threads()") == 1
    assert (
        "_cta_stage_thread = cutlass.Int32(cute.arch.thread_idx()[0]) * 1 + "
        "cutlass.Int32(cute.arch.thread_idx()[1]) * 32" in code
    )
    # 96 elements are 12 packets: the first 12 of the 128 threads flush.
    assert "if _cta_stage_thread < 12:" in code


def test_a_partial_last_tile_keeps_its_direct_stores() -> None:
    # 144 elements in tiles of 64: the third CTA has 16.
    code = _code(sliding_windows, (_windows(1, 9, 16), 3, 1), _TILE)
    # A CTA stages when its whole tile lies inside the tensor (its last
    # packet starts at element 56 of the tile, below 144) ...
    assert "_cta_stage_full = (cutlass.Int32(pid_0) * 64 < 88) & " in code
    tree = ast.parse(code)
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_cta_stage_full"
    ]
    # ... each store keeps its direct, masked form in the other CTAs (the
    # padded tile's fourth slot, which no CTA stores, is left as it was) ...
    stores = [node for node in guards if node.orelse]
    assert len(stores) == 3
    assert _direct_stores(code) == 4
    for node in stores:
        assert "(_cta_stage + (" in ast.unparse(node.body[0])
        assert "(out.iterator + " in ast.unparse(node.orelse[0])
    # ... and the barrier and the flush run for the CTAs that staged.
    (flush,) = [node for node in guards if not node.orelse]
    assert "cute.arch.sync_threads()" in ast.unparse(flush.body[0])
    assert ast.unparse(flush).count("cute.arch.store(out.iterator") == 3


def _declines(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    config: dict[str, object],
    **settings: object,
) -> None:
    code = _code(kernel, args, config, **settings)
    assert "_cta_stage" not in code
    assert "cute.arch.sync_threads()" not in code
    # Where the proof fails the knob is inert.
    assert code == _code(
        kernel, args, {**config, "cute_staged_store": False}, **settings
    )


def test_several_tiles_on_the_last_dimension_decline() -> None:
    # Two-slot tiles of three slots: a CTA writes two of every three elements.
    _declines(
        sliding_windows,
        (_windows(), 3, 1),
        register_tile_config(threads=8, window_block=2),
    )


@pytest.mark.parametrize(
    ("args", "config"),
    [
        # An empty batch, and windows without a slot: no CTA is launched.
        ((_windows(0, 40, 16), 9, 4), _SCALAR | {"cute_staged_store": True}),
        ((_windows(), 0, 0), register_tile_config(threads=8, window_block=1)),
    ],
    ids=["empty_batch", "empty_window"],
)
def test_an_empty_tensor_declines(
    args: tuple[object, ...], config: dict[str, object]
) -> None:
    _declines(sliding_windows, args, config)


def test_an_image_over_the_shared_memory_budget_declines() -> None:
    # 16384 elements x 3 slots x 2 bytes: twice the 48 KiB planners' budget.
    _declines(
        sliding_windows,
        (_windows(8, 256, 16), 3, 1),
        register_tile_config(threads=128, window_block=4, block_sizes=[16384, 4]),
    )


def test_a_second_output_declines() -> None:
    _declines(sliding_windows_and_copy, (_windows(), 3, 1), _TILE)


def test_an_output_that_may_alias_the_input_declines() -> None:
    buffer = torch.empty(256 * 4, dtype=torch.bfloat16)
    flat, out = buffer[:256], buffer[256:].view(256, 3)
    _declines(sliding_windows_into, (flat, out, 8, 16, 1), _TILE)
    # Two arguments with storages of their own are proven apart (the bound
    # kernel is keyed on it).
    apart = torch.empty(256, 3, dtype=torch.bfloat16)
    code = _code(sliding_windows_into, (flat, apart, 8, 16, 1), _TILE)
    assert _direct_stores(code) == 0


def test_an_output_view_stages_only_at_an_aligned_offset() -> None:
    flat = torch.empty(256, dtype=torch.bfloat16)
    buffer = torch.empty(256 * 3 + 16, dtype=torch.bfloat16)
    # 6 bytes into the storage: no 16-byte store may start a tile there.
    _declines(sliding_windows_into, (flat, buffer[3:771].view(256, 3), 8, 16, 1), _TILE)
    code = _code(
        sliding_windows_into, (flat, buffer[8:776].view(256, 3), 8, 16, 1), _TILE
    )
    assert _direct_stores(code) == 0


# ---------------------------------------------------------------------------
# The proof on hand-written bodies
# ---------------------------------------------------------------------------


def _stage(source: str, **overrides: Any) -> tuple[bool, str]:
    """``stage_cta_stores`` over a kernel body of 4 threads on a 6-CTA grid
    writing the bf16 tensor ``out`` (strides ``(2, 1)``), beside a tensor
    ``x`` proven apart from it."""
    body = ast.parse(textwrap.dedent(source)).body
    names = iter(range(100))
    arguments: dict[str, Any] = {
        "tensors": {
            "out": torch.empty(24, 2, dtype=torch.bfloat16),
            "x": torch.empty(48, dtype=torch.bfloat16),
        },
        "base_aligned": lambda tensor: True,
        "proven_disjoint_tensor_pairs": {frozenset(("out", "x"))},
        "proven_tensor_stride_values": {("out", 0): 2, ("out", 1): 1},
        "constexpr_values": {"_BLOCK_SIZE_0": 4},
        "thread_block_dims": (4, 1, 1),
        "block_grid_dims": (6, 1, 1),
        "index_type": "cutlass.Int32",
        "dtype_str": lambda dtype: "cutlass.BFloat16",
        "smem_budget": 49152,
        "new_var": lambda name: f"{name}_{next(names)}",
    }
    staged = stage_cta_stores(body, **{**arguments, **overrides})
    return staged, ast.unparse(ast.Module(body=body, type_ignores=[]))


_ROWS = """
    tile_offset_0 = cutlass.Int32(cute.arch.block_idx()[0]) * _BLOCK_SIZE_0
    indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[0])
    for lane_1 in range(2):
        value = (x.iterator + cutlass.Int32(indices_0) * 2 + lane_1).load()
        {store}
"""
_STORE = (
    "(out.iterator + cutlass.Int32(indices_0) * cutlass.Int32(out.layout.stride[0])"
    " + cutlass.Int32(lane_1) * cutlass.Int32(out.layout.stride[1]))"
    ".store(cutlass.BFloat16(value))"
)


# Column 0 stored first, then column 1 under a mask over ``limit_0``, which
# is defined between the two stores: 22 rows in tiles of 4.
_LATE_MASK = """
    tile_offset_0 = cutlass.Int32(cute.arch.block_idx()[0]) * _BLOCK_SIZE_0
    indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[0])
    (out.iterator + cutlass.Int32(indices_0) * 2).store(cutlass.BFloat16(0))
    limit_0 = cutlass.Int32(cute.arch.block_idx()[0]) * _BLOCK_SIZE_0
    mask_0 = limit_0 + cutlass.Int32(cute.arch.thread_idx()[0]) < 22
    if mask_0:
        (out.iterator + cutlass.Int32(indices_0) * 2 + 1).store(cutlass.BFloat16(0))
"""


def test_the_proof_stages_a_row_tile() -> None:
    source = _ROWS.format(store=_STORE)
    staged, code = _stage(source)
    assert staged
    # 4 rows x 2 columns: one packet, which thread 0 flushes.
    assert "alloc_smem(cutlass.BFloat16, 8, alignment=16)" in code
    assert "_cta_stage_base_1 = cutlass.Int32(tile_offset_0) * 2" in code
    assert "if _cta_stage_thread_2 < 1:" in code
    assert "(out.iterator + cutlass.Int32(indices_0)" not in code


@pytest.mark.parametrize(
    ("source", "overrides"),
    [
        # Without the launch shape or a static grid there is nothing to run.
        (_ROWS.format(store=_STORE), {"thread_block_dims": None}),
        (_ROWS.format(store=_STORE), {"block_grid_dims": None}),
        # An unaligned base, a load that may alias the output, an unproven
        # stride.
        (_ROWS.format(store=_STORE), {"base_aligned": lambda tensor: False}),
        (_ROWS.format(store=_STORE), {"proven_disjoint_tensor_pairs": set()}),
        (_ROWS.format(store=_STORE), {"proven_tensor_stride_values": {}}),
        # Tiles of 3 rows on the 4 threads: the CTAs' spans start 12 bytes
        # apart.
        (_ROWS.format(store=_STORE), {"constexpr_values": {"_BLOCK_SIZE_0": 3}}),
        # A barrier, a load of the output, a second column written twice, a
        # store under a test that is no mask of it, an early return.
        (_ROWS.format(store=_STORE) + "    cute.arch.sync_threads()\n", {}),
        (
            _ROWS.format(store=_STORE)
            + "    again = (out.iterator + cutlass.Int32(indices_0)).load()\n",
            {},
        ),
        (_ROWS.format(store=_STORE.replace("cutlass.Int32(lane_1) *", "0 *")), {}),
        (
            _ROWS.format(
                store=f"if value > 0:\n            {_STORE}\n        else:\n"
                "            other = value"
            ),
            {},
        ),
        (_ROWS.format(store=_STORE) + "    return\n", {}),
        # A loop the proof cannot count around the store.
        (_ROWS.format(store=_STORE).replace("range(2)", "range(rows)"), {}),
        # A grid without a CTA, and an element type without a packet.
        (_ROWS.format(store=_STORE), {"block_grid_dims": (0, 1, 1)}),
        (
            _ROWS.format(store=_STORE),
            {
                "tensors": {
                    "out": torch.empty(24, 2, dtype=torch.float64),
                    "x": torch.empty(48, dtype=torch.bfloat16),
                }
            },
        ),
        # Three threads on the tiles of 4 rows: spans of 12 bytes, each
        # starting on a packet.
        (_ROWS.format(store=_STORE), {"thread_block_dims": (3, 1, 1)}),
        # The last thread's first element moved onto the one before it:
        # elements 0 to 7 with 5 written twice and 6 never.
        (
            _ROWS.format(
                store=_STORE.replace(
                    ").store(",
                    " + cutlass.Int32(cute.arch.thread_idx()[0]) // 3 * (lane_1 - 1)"
                    ").store(",
                )
            ),
            {},
        ),
        # A store that data skips, by a ``continue`` the proof never runs.
        (
            _ROWS.format(
                store=f"if value > 0:\n            continue\n        {_STORE}"
            ),
            {},
        ),
        # A tile origin assigned again ahead of the stores, and a mask over
        # a value defined after the first store: the staged code would read
        # either where it does not hold the value the proof used.
        (
            _ROWS.format(store=_STORE).replace(
                "    for lane_1",
                "    tile_offset_0 = tile_offset_0 * 3\n    for lane_1",
            ),
            {},
        ),
        (_LATE_MASK, {}),
        # A mask over a sum that leaves 32 bits from the second tile on: its
        # 32-bit cast wraps there, so stores the exact integers would stage
        # in every CTA run in the first one only.  Then a 32-bit sum over
        # the thread index that wraps for the last two threads, which never
        # store.
        (
            _ROWS.format(
                store="if cutlass.Int32(cutlass.Int64(indices_0) + 2147483644) >= 0:\n"
                f"            {_STORE}"
            ),
            {},
        ),
        (
            _ROWS.format(
                store="if cutlass.Int32(cute.arch.thread_idx()[0]) + 2147483646 >= 0:\n"
                f"            {_STORE}"
            ),
            {},
        ),
        # A mask whose two sides fit 32 bits at every row while the staging
        # test over the tile origin, ``origin * 160000000 < 520000000``,
        # would wrap from the fifth tile on and stage tiles that store
        # nothing.
        (
            _ROWS.format(
                store="if indices_0 * 80000000 < 1000000000 - indices_0 * 80000000:\n"
                f"            {_STORE}"
            ),
            {},
        ),
        # An element offset whose constant term puts the span's base, which
        # the staged code computes in the index type, past 32 bits from the
        # second tile on.
        (
            _ROWS.format(store=_STORE.replace(").store(", " + 2147483640).store(")),
            {},
        ),
        # An unsigned operand: ``Uint32(0) - 1`` is not below 3 in the
        # kernel, so the first thread never stores.
        (
            _ROWS.format(
                store="if cutlass.Uint32(cute.arch.thread_idx()[0]) - 1 < 3:\n"
                f"            {_STORE}"
            ),
            {},
        ),
    ],
    ids=[
        "no_launch_shape",
        "no_static_grid",
        "unaligned_base",
        "may_alias_a_load",
        "unproven_stride",
        "unaligned_spans",
        "barrier",
        "loads_the_output",
        "element_written_twice",
        "store_under_a_data_test",
        "early_return",
        "uncounted_loop",
        "empty_grid",
        "element_type_without_a_packet",
        "span_not_whole_packets",
        "element_written_twice_in_a_gapless_span",
        "store_skipped_by_continue",
        "rebound_tile_origin",
        "mask_defined_after_a_store",
        "mask_over_a_wrapping_cast",
        "thread_mask_over_a_wrapping_sum",
        "staging_test_that_would_wrap",
        "span_base_past_32_bits",
        "unsigned_mask_operand",
    ],
)
# One case per proof obligation of ``staged_stores``: each fails alone.
def test_the_proof_declines(source: str, overrides: dict[str, Any]) -> None:
    before = ast.unparse(ast.parse(textwrap.dedent(source)))
    staged, code = _stage(source, **overrides)
    assert not staged
    assert code == before


def test_the_proof_guards_tiles_past_the_end() -> None:
    # 22 rows in tiles of 4: the sixth CTA's last two rows are masked off.
    masked = f"mask_0 = indices_0 < 22\n        if mask_0:\n            {_STORE}"
    staged, code = _stage(_ROWS.format(store=masked))
    assert staged
    assert "_cta_stage_full_2 = cutlass.Int32(tile_offset_0) * 1 < 19" in code
    assert code.count("if _cta_stage_full_2:") == 2
    assert "(out.iterator + cutlass.Int32(indices_0)" in code


# ---------------------------------------------------------------------------
# The search surface and the seeds
# ---------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True)
def _transposed_copy(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([columns, rows], dtype=x.dtype, device=x.device)
    for tile_c, tile_r in hl.tile([columns, rows]):
        out[tile_c, tile_r] = x[tile_r, tile_c].T
    return out


def _spec(kernel: Kernel[Any], args: tuple[object, ...], **settings: object) -> Any:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        return _bound(kernel, args, **settings).config_spec


def test_the_knobs_are_offered_on_the_kernels_structure() -> None:
    specs = [
        _spec(sliding_windows, (_windows(*shape), window, pad))
        for shape, window, pad in (
            ((2, 8, 16), 3, 1),
            ((32, 512, 384), 9, 4),
            # Extents no register tile fits.
            ((3, 7, 5), 3, 1),
        )
    ]
    for spec in specs:
        assert spec.cute_lane_gather_store_sites
        fields = spec._flat_fields()
        assert "cute_staged_store" in fields
        assert fields["cute_vloop_sink"].search_choices == (False, True)
        assert fields["cute_lane_unroll"].search_choices == (1, 2, 4, 8, 16)
    # The search space does not follow the extents.
    assert len({repr(spec.structural_fingerprint()) for spec in specs}) == 1


@pytest.mark.parametrize(
    ("kernel", "args", "settings"),
    [
        # No flattened gather stored along another tile.
        (_transposed_copy, (torch.empty(64, 48, dtype=torch.bfloat16),), {}),
        # Dynamic shapes: no static launch grid, so the proof never holds.
        (sliding_windows, (_windows(), 3, 1), {"static_shapes": False}),
        # Element types without a 16-byte packet: nothing to flush or load.
        (sliding_windows, (torch.empty(2, 8, 16, dtype=torch.float64), 3, 1), {}),
        (sliding_windows, (torch.empty(2, 8, 16, dtype=torch.uint8), 3, 1), {}),
    ],
    ids=["no_gather", "dynamic_shapes", "fp64", "uint8"],
)
def test_the_knobs_are_not_offered_where_they_cannot_apply(
    kernel: Kernel[Any], args: tuple[object, ...], settings: dict[str, object]
) -> None:
    spec = _spec(kernel, args, **settings)
    assert not spec.cute_lane_gather_store_sites
    fields = spec._flat_fields()
    assert "cute_staged_store" not in fields
    # Nor are the packet loads, which are slower without the staged store.
    assert fields["cute_vloop_sink"].search_choices == (False,)
    assert fields["cute_lane_unroll"].search_choices == (1,)
    default = spec.default_config().config
    with pytest.raises(exc.InvalidConfig, match="cute_staged_store requires"):
        spec.normalize(default | {"cute_staged_store": True})
    # The knob-off spelling of a config tuned elsewhere is dropped.
    explicit = default | {"cute_staged_store": False}
    spec.normalize(explicit)
    assert "cute_staged_store" not in explicit


def test_the_register_tile_is_seeded_only_with_the_staged_store() -> None:
    spec = _spec(sliding_windows, (_windows(32, 512, 384), 9, 4))
    seeds = [seed.config for seed in spec.compiler_seed_configs]
    tiles = [seed for seed in seeds if seed.get("cute_vloop_sink")]
    assert [(seed["block_sizes"], seed["num_threads"]) for seed in tiles] == [
        ([512, 16], [64, 1]),
        ([1024, 16], [128, 1]),
    ]
    for seed in tiles:
        assert seed["cute_vector_widths"] == [8, 1]
        assert seed["cute_lane_unroll"] == 16
        assert seed["cute_proven_bounds"] is True
        # Packet loads with scattered scalar stores are the slowest layout
        # of this kernel: never planted without the staged store.
        assert seed["cute_staged_store"] is True
    # The seed renders what it promises: 9 packet loads kept, every store
    # staged, no test of a partial tile.
    code = _code(sliding_windows, (_windows(32, 512, 384), 9, 4), tiles[0])
    assert code.count("_cute_load_v16b_if(flat.iterator") == 16
    assert _direct_stores(code) == 0
    assert "_cta_stage_full" not in code


@helion.kernel(backend="cute", static_shapes=True)
def _windows_cast_into(
    flat: torch.Tensor, out: torch.Tensor, channels: int
) -> torch.Tensor:
    """Unmasked windows of ``flat`` converted to ``out``'s element type."""
    channels = hl.specialize(channels)
    n, window = out.shape
    for tile_n, tile_k in hl.tile([n, window]):
        offset = tile_n.index[:, None] + channels * tile_k.index[None, :]
        out[tile_n, tile_k] = flat[offset].to(out.dtype)
    return out


@pytest.mark.parametrize(
    ("gathered", "stored", "offered"),
    [
        (torch.int16, torch.bfloat16, True),
        # Packet loads have no one-byte form, the staged store no 8-byte one.
        (torch.int8, torch.bfloat16, False),
        (torch.bfloat16, torch.float64, False),
    ],
    ids=["int16_to_bf16", "int8_to_bf16", "bf16_to_fp64"],
)
def test_the_knobs_need_packets_of_both_element_types(
    gathered: torch.dtype, stored: torch.dtype, offered: bool
) -> None:
    args = (torch.empty(1024, dtype=gathered), torch.empty(256, 3, dtype=stored), 16)
    spec = _spec(_windows_cast_into, args)
    assert spec.cute_lane_gather_store_sites is offered
    assert ("cute_staged_store" in spec._flat_fields()) is offered


def _register_tiles(spec: Any) -> list[dict[str, object]]:
    return [
        dict(seed.config)
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink") or seed.config.get("cute_staged_store")
    ]


@pytest.mark.parametrize(
    ("dtype", "vec"),
    [(torch.float16, 8), (torch.float32, 4), (torch.int16, 8), (torch.int32, 4)],
)
def test_a_register_tile_seed_stages_and_loads_packets(
    dtype: torch.dtype, vec: int
) -> None:
    """Each element type with a 16-byte packet (bf16 is rendered above): the
    seeds are planted, and what they promise is what renders."""
    args = (torch.empty(4, 32, 32, dtype=dtype), 3, 1)
    tiles = _register_tiles(_spec(sliding_windows, args))
    assert [(seed["block_sizes"], seed["num_threads"]) for seed in tiles] == [
        ([vec * 64, 4], [64, 1]),
        ([vec * 128, 4], [128, 1]),
    ]
    assert all(seed["cute_staged_store"] is True for seed in tiles)
    code = _code(sliding_windows, args, tiles[0])
    assert code.count("_cute_load_v16b_if(flat.iterator") == 4
    assert _direct_stores(code) == 0
    assert code.count("- _cta_stage_base)).store(") == 3


@pytest.mark.parametrize(
    ("dtype", "vec"), [(torch.float32, 4), (torch.bfloat16, 8)], ids=["fp32", "bf16"]
)
def test_a_register_tile_seed_of_a_two_slot_window_stages_too(
    dtype: torch.dtype, vec: int
) -> None:
    """Two slots are also the two components of an output element, which a
    tile covering them stores as one packet per thread, with an element load
    per lane and slot (``test_cute_component_packets.py``).  The seed's
    ``cute_vloop_sink`` keeps the slots for the sink, so it renders what it
    promises here as well: a packet load per slot into the staged image."""
    args = (torch.empty(3, 40, 16, dtype=dtype), 2, 1)
    tiles = _register_tiles(_spec(sliding_windows, args))
    assert [(seed["block_sizes"], seed["num_threads"]) for seed in tiles] == [
        ([vec * 64, 2], [64, 1]),
        ([vec * 128, 2], [128, 1]),
    ]
    code = _code(sliding_windows, args, tiles[0])
    assert code.count("_cute_load_v16b_if(flat.iterator") == 2
    assert ".load()" not in code
    assert "_tile_store_vals" not in code
    assert code.count("- _cta_stage_base)).store(") == 2
    assert code.count("cute.arch.sync_threads()") == 1


@pytest.mark.parametrize(
    ("args", "settings"),
    [
        # Dynamic extents, too few elements for a 64-thread tile, and a
        # sequence width the packets cannot follow (12 channels).
        ((_windows(32, 512, 384), 9, 4), {"static_shapes": False}),
        ((_windows(2, 8, 16), 3, 1), {}),
        ((_windows(32, 512, 12), 9, 4), {}),
        # One-byte elements, which neither the packet loads nor the staged
        # store take, and a one-slot window, whose stores are contiguous
        # along the gathered tile already.
        ((torch.empty(32, 512, 384, dtype=torch.uint8), 9, 4), {}),
        ((torch.empty(32, 512, 384, dtype=torch.bool), 9, 4), {}),
        ((_windows(32, 512, 384), 1, 0), {}),
        # An output of 2**31 + 16 elements: the staged store's proof holds
        # every index to 32 bits.
        ((_windows(1, 14913081, 16), 9, 4), {}),
    ],
    ids=[
        "dynamic_shapes",
        "small_extent",
        "row_not_a_multiple",
        "uint8",
        "bool",
        "one_slot_window",
        "output_past_32_bits",
    ],
)
def test_no_register_tile_is_seeded_where_its_packets_are_unproved(
    args: tuple[object, ...], settings: dict[str, object]
) -> None:
    assert not _register_tiles(_spec(sliding_windows, args, **settings))


# Unmasked windows of 4096 flat bf16 elements, three slots 1024 elements
# apart, into ``[1024, 3]``: a kernel both register tiles fit.
_FLAT = 4096
_OUT = (1024, 3)
_CHANNELS = 1024


@helion.kernel(backend="cute", static_shapes=True)
def _windows_and_row_into(
    flat: torch.Tensor, out: torch.Tensor, row: torch.Tensor, channels: int
) -> torch.Tensor:
    """``_windows_cast_into`` plus a copy of ``flat``'s first elements: a
    second tensor written by a store along the gathered tile alone."""
    channels = hl.specialize(channels)
    n, window = out.shape
    for tile_n, tile_k in hl.tile([n, window]):
        offset = tile_n.index[:, None] + channels * tile_k.index[None, :]
        out[tile_n, tile_k] = flat[offset]
        row[tile_n] = flat[tile_n]
    return out


def test_no_register_tile_is_seeded_for_a_kernel_with_a_second_output() -> None:
    """The staged store declines a kernel that writes two tensors, and the
    register tile without it is the layout's slowest form: the seed asks
    for one output, although only one store is along the gather's lane
    tile."""
    flat = torch.empty(_FLAT, dtype=torch.bfloat16)
    out = torch.empty(_OUT, dtype=torch.bfloat16)
    row = torch.empty(_OUT[0], dtype=torch.bfloat16)
    spec = _spec(_windows_and_row_into, (flat, out, row, _CHANNELS))
    assert spec.cute_lane_gather_store_sites
    assert not _register_tiles(spec)
    assert len(_register_tiles(_spec(_windows_cast_into, (flat, out, _CHANNELS)))) == 2


@helion.kernel(backend="cute", static_shapes=True)
def _sum_of_two_windows_into(
    first: torch.Tensor, second: torch.Tensor, out: torch.Tensor, channels: int
) -> torch.Tensor:
    """The sum of the unmasked windows of two flat tensors: two gathers
    along the lane tile."""
    channels = hl.specialize(channels)
    n, window = out.shape
    for tile_n, tile_k in hl.tile([n, window]):
        offset = tile_n.index[:, None] + channels * tile_k.index[None, :]
        out[tile_n, tile_k] = first[offset] + second[offset]
    return out


def test_the_register_tile_needs_the_packets_of_every_gather() -> None:
    """The seed proves each gather's packets, not the first one's alone: a
    second gathered tensor whose base is 6 bytes into its storage, or whose
    elements are not the output's size, has no such packets, so the kernel
    gets no register tile."""
    first = torch.empty(_FLAT, dtype=torch.bfloat16)
    buffer = torch.empty(_FLAT + 16, dtype=torch.bfloat16)
    out = torch.empty(_OUT, dtype=torch.bfloat16)
    aligned = (first, buffer[8 : 8 + _FLAT], out, _CHANNELS)
    tiles = _register_tiles(_spec(_sum_of_two_windows_into, aligned))
    assert [seed["num_threads"] for seed in tiles] == [[64, 1], [128, 1]]
    code = _code(_sum_of_two_windows_into, aligned, tiles[0])
    assert "_cta_stage" in code
    assert ".load()" not in code
    unaligned = (first, buffer[3 : 3 + _FLAT], out, _CHANNELS)
    assert not _register_tiles(_spec(_sum_of_two_windows_into, unaligned))
    wider = (first, torch.empty(_FLAT, dtype=torch.int32), out, _CHANNELS)
    assert not _register_tiles(_spec(_sum_of_two_windows_into, wider))


def test_the_register_tile_needs_every_tensor_below_the_index_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof's index limit holds for each tensor: with it lowered to the
    gathered tensor's 4096 elements, above the output's 3072, the seed is
    gone (the 2**31-element output is a case above)."""
    flat = torch.empty(_FLAT, dtype=torch.bfloat16)
    args = (flat, torch.empty(_OUT, dtype=torch.bfloat16), _CHANNELS)
    assert len(_register_tiles(_spec(_windows_cast_into, args))) == 2
    monkeypatch.setattr(heuristics, "INDEX_LIMIT", _FLAT)
    assert not _register_tiles(_spec(_windows_cast_into, args))


def test_the_register_tile_is_seeded_within_the_shared_memory_budget() -> None:
    # 32 slots of bf16 are 64 bytes per element of the gathered tile: 512
    # elements (64 threads) fill 32 KiB of the 48 KiB budget, 1024 would not
    # fit.
    spec = _spec(sliding_windows, (_windows(32, 512, 384), 32, 16))
    assert [
        (seed["block_sizes"], seed["num_threads"]) for seed in _register_tiles(spec)
    ] == [([512, 32], [64, 1])]


@pytest.mark.parametrize(
    ("out_offset", "shared_storage", "seeded"),
    [
        # The caller's output 16 bytes into its buffer, 6 bytes in (no packet
        # may start a tile there), and inside the input's storage (not
        # proven apart from the loads, so the stores cannot wait).
        (8, False, True),
        (3, False, False),
        (8, True, False),
    ],
    ids=["aligned", "unaligned", "input_storage"],
)
def test_an_output_argument_is_seeded_where_it_stages(
    out_offset: int, shared_storage: bool, seeded: bool
) -> None:
    elements, window = 1024, 9
    buffer = torch.empty(elements * (window + 1) + 16, dtype=torch.bfloat16)
    flat = (
        buffer[elements * window + 16 :]
        if shared_storage
        else torch.empty(elements, dtype=torch.bfloat16)
    )
    out = buffer[out_offset : out_offset + elements * window].view(elements, window)
    args = (flat, out, 4, 16, 4)
    tiles = _register_tiles(_spec(sliding_windows_into, args))
    assert [(seed["block_sizes"], seed["num_threads"]) for seed in tiles] == (
        [([512, 16], [64, 1]), ([1024, 16], [128, 1])] if seeded else []
    )
    if seeded:
        code = _code(sliding_windows_into, args, tiles[0])
        assert code.count("_cute_load_v16b_if(flat.iterator") == 16
        assert _direct_stores(code) == 0
