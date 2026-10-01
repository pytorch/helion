from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from helion import exc
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_seed_tiles import SeedTiling
from helion._compiler.cute.chained_seed_tiles import plan_seed_panels
from helion._compiler.cute.chained_tmem_segments import emit_tmem_segment_load


@pytest.mark.parametrize("limit", [0, 32, 64])
def test_every_contained_member_is_covered_once_in_original_order(limit):
    for total in range(16, 257, 16):
        for offset in range(0, total, 16):
            for width in range(16, total - offset + 1, 16):
                panels = plan_seed_panels((128, total), offset, width, limit)
                assert tuple(
                    column
                    for panel in panels
                    for column in range(panel.offset, panel.offset + panel.width)
                ) == tuple(range(offset, offset + width))
                assert all(
                    panel.full_shape == (128, total)
                    and panel.member_offset == offset
                    and panel.member_width == width
                    and panel.width > 0
                    and panel.width % 16 == 0
                    for panel in panels
                )
                if limit:
                    assert all(panel.width <= limit for panel in panels)
                else:
                    assert len(panels) == 1
                    assert (panels[0].offset, panels[0].width) == (offset, width)


@pytest.mark.parametrize(
    "width,limit,expected",
    [(80, 64, (64, 16)), (112, 64, (64, 48)), (80, 32, (32, 32, 16))],
)
def test_unequal_remainder_and_immutable_member_coordinates(width, limit, expected):
    panels = plan_seed_panels((128, 160), 16, width, limit)
    assert tuple(panel.width for panel in panels) == expected
    with pytest.raises(FrozenInstanceError):
        panels[0].offset = 0  # pyrefly: ignore[read-only]


@pytest.mark.parametrize("limit", [True, False, None, -1, 1, 16, 128, 32.0, "32"])
def test_strict_maximum_rejects_in_constructor_and_planner(limit):
    with pytest.raises(ValueError, match="maximum columns must be 0, 32 or 64"):
        SeedTiling(limit)
    with pytest.raises(ValueError, match="maximum columns must be 0, 32 or 64"):
        plan_seed_panels((128, 128), 0, 128, limit)


@pytest.mark.parametrize(
    "shape,offset,width",
    [
        (None, 0, 16),
        ([128, 128], 0, 16),
        ((128,), 0, 16),
        ((64, 128), 0, 16),
        ((True, 128), 0, 16),
        ((128, 128.0), 0, 16),
        ((128, 0), 0, 16),
        ((128, 272), 0, 16),
        ((128, 136), 0, 16),
        ((128, 128), True, 16),
        ((128, 128), 0.0, 16),
        ((128, 128), -16, 16),
        ((128, 128), 8, 16),
        ((128, 128), 128, 16),
        ((128, 128), 112, 32),
        ((128, 128), 0, False),
        ((128, 128), 0, 16.0),
        ((128, 128), 0, 0),
        ((128, 128), 0, 8),
        ((128, 128), 0, -16),
    ],
)
def test_invalid_member_cannot_activate_tiling(shape, offset, width):
    tiling = SeedTiling(32)
    with pytest.raises(ValueError, match="full-M128"):
        tiling.panels(shape, offset, width)
    assert not tiling.activated


@pytest.mark.parametrize("limit", [32, 64])
def test_tracking_requires_a_real_split_and_is_per_codegen(limit):
    tiling = SeedTiling(limit)
    with pytest.raises(exc.BackendUnsupported, match="multi-panel accumulator seed"):
        tiling.validate()
    tiling.panels((128, 160), 16, limit)
    assert not tiling.activated
    with pytest.raises(exc.BackendUnsupported, match="multi-panel accumulator seed"):
        tiling.validate()
    assert len(tiling.panels((128, 160), 16, limit + 16)) == 2
    assert tiling.activated
    tiling.panels((128, 160), 128, 16)
    tiling.validate()
    assert not SeedTiling(limit).activated


def test_legacy_zero_requires_no_activation_and_never_splits():
    tiling = SeedTiling(0)
    tiling.validate()
    assert len(tiling.panels((128, 256), 16, 240)) == 1
    assert not tiling.activated
    tiling.validate()


@pytest.mark.parametrize(
    "width,offset,total", [(16, 0, 16), (48, 16, 160), (128, 32, 160), (240, 16, 256)]
)
def test_extracted_views_leave_entire_existing_load_source_byte_identical(
    width, offset, total
):
    panel = plan_seed_panels((128, total), offset, width)[0]
    origin, shape = ((0, offset), 0, 0), ((128, width), 1, 1)
    original = [
        f"member_segment = cute.composition(cute.domain_offset({origin!r}, group_acc), cute.make_layout({shape!r}))",
        f"member_identity = cute.composition(cute.domain_offset({origin!r}, group_slice.partition_C(cute.make_identity_tensor({(128, total)!r}))), cute.make_layout({shape!r}))",
        f"member_copy = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation((128, width))}, cutlass.Float32), member_segment)",
        "member_thread = member_copy.get_slice(chain_thread)",
        "member_source = member_thread.partition_S(member_segment)",
        "member_coords = member_thread.partition_D(member_identity)",
        "member_values = cute.make_rmem_tensor(member_coords.shape, cutlass.Float32)",
        "cute.copy(member_copy, member_source, member_values)",
        "cute.arch.fence_view_async_tmem_load()",
    ]
    assert panel.views("member", "group") == original[:2]
    assert (
        emit_tmem_segment_load("member", "group", (128, total), offset, width)
        == original
    )


@pytest.mark.parametrize("operand_source", ["SMEM", "TMEM"])
@pytest.mark.parametrize("limit", [0, 32, 64])
@pytest.mark.parametrize(
    "total,offset,width",
    [(16, 0, 16), (128, 16, 80), (160, 32, 128), (256, 144, 112), (256, 0, 256)],
)
def test_actual_fp32_cute_panel_load_store_and_identity_ownership_cpu(
    operand_source, limit, total, offset, width
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, total),
                tcgen05.OperandSource.SMEM
                if operand_source == "SMEM"
                else tcgen05.OperandSource.TMEM,
            )
            layout = mma.make_fragment_C(mma.partition_shape_C((128, total))).layout
            group = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 224, cute.AddressSpace.tmem), layout
            )
            coverage = set()
            for panel in plan_seed_panels((128, total), offset, width, limit):
                namespace: dict[str, Any] = {
                    "cute": cute,
                    "group_acc": group,
                    "group_slice": mma.get_slice(0),
                }
                exec("\n".join(panel.views("panel", "group")), namespace)
                segment, identity = (
                    namespace["panel_segment"],
                    namespace["panel_identity"],
                )
                load = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(
                        eval(load_operation((128, panel.width)), {"tcgen05": tcgen05}),
                        cutlass.Float32,
                    ),
                    segment,
                )
                store = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(
                        tcgen05.St32x32bOp(
                            tcgen05.Repetition(min(32, panel.width & -panel.width))
                        ),
                        cutlass.Float32,
                    ),
                    segment,
                )
                assert segment.element_type == cutlass.Float32
                for thread in range(128):
                    reader, writer = load.get_slice(thread), store.get_slice(thread)
                    loaded = reader.partition_D(identity)
                    stored = writer.partition_S(identity)
                    assert (
                        int(cute.size(loaded)) == int(cute.size(stored)) == panel.width
                    )
                    values = cute.make_rmem_tensor(loaded.shape, cutlass.Float32)
                    cute.copy(load, reader.partition_S(segment), values)
                    cute.copy(store, values, writer.partition_D(segment))
                    for index in range(panel.width):
                        point = tuple(map(int, loaded[index]))
                        assert point == tuple(map(int, stored[index]))
                        row, column = point
                        assert row == thread and offset <= column < offset + width
                        assert point not in coverage
                        coverage.add(point)
                        assert panel.offset + int(
                            segment.layout(((row, column - panel.offset), 0, 0))
                        ) == int(layout(((row, column), 0, 0)))
                        # Original expressions retain the MEMBER origin even
                        # in later panels; no panel-local reinterpretation.
                        assert 0 <= column - panel.member_offset < panel.member_width
            assert coverage == {
                (row, col)
                for row in range(128)
                for col in range(offset, offset + width)
            }
        assert module.operation.verify()
        source = str(module)
        assert "tmem_load<f32" in source and "tmem_store<f32" in source
    assert torch.cuda.is_initialized() == before
