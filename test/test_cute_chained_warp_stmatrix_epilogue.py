from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.fx import Graph

from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_fragment_epilogue import FragmentExpression
from helion._compiler.cute.chained_fragment_epilogue import WarpOperandEpilogue
from helion._compiler.cute.chained_root_warp_stage import WarpOperandLayout
from helion._compiler.cute.chained_warp_stmatrix_epilogue import render_stmatrix_operand


def _fixture(dtype=torch.bfloat16, shape=(64, 64), threads=128):
    graph = Graph()
    source, target = graph.placeholder("source"), graph.placeholder("target")
    coords = ("bridge_row", "bridge_col")
    fragment = "producer_acc[bridge_index]"
    scalar_type = {
        torch.bfloat16: "cutlass.BFloat16",
        torch.float16: "cutlass.Float16",
    }[dtype]
    expression = SimpleNamespace(
        cg=object(),
        context=object(),
        pointwise_policy=None,
        fragments={source: (coords, fragment)},
        pointwise_views={},
        memo={(source, coords): fragment},
        lines=[
            "weight = cute.math.exp2(cutlass.Float32(bridge_row - bridge_col) * log_gamma)",
            f"product = cutlass.Float32({fragment} * weight)",
            f"narrowed = {scalar_type}(product)",
        ],
    )
    value = FragmentExpression(
        cast("chain._Expression", expression),
        source,
        target,
        coords,
        coords,
        fragment,
        scalar_type,
        "narrowed",
        ("bridge_col <= bridge_row", "bridge_row < row_bound"),
        None,
    )
    object.__setattr__(value, "_selection", value.facts())
    epilogue = WarpOperandEpilogue(value, "bridge", "producer", shape, "consumer_a")
    layout = WarpOperandLayout(target, "a", shape, (shape[1] + 8, 1), 1, dtype)
    options = {
        "producer_threads": threads,
        "cta_threads": threads,
        "pointer_alignment": 128,
    }
    return epilogue, layout, options


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "shape,threads",
    (((16, 16), 32), ((32, 32), 64), ((64, 64), 128), ((128, 128), 256)),
)
def test_complete_original_panels_keep_expression_mask_and_cast_order(
    dtype, shape, threads
):
    epilogue, layout, options = _fixture(dtype, shape, threads)
    original = epilogue.render()
    result = render_stmatrix_operand(epilogue, layout, **options)
    assert result is not None
    assert result[0] == original[0] and result[2:-5] == original[2:-1]
    # Restore only the transport/allocation choice: complete original AST agrees.
    restored = (result[0], original[1], *result[2:-5], original[-1])
    assert ast.dump(ast.parse("\n".join(restored))) == ast.dump(
        ast.parse("\n".join(original))
    )
    text = "\n".join(result)
    assert "StMatrix8x8x16bOp(num_matrices=4, transpose=False)" in text
    assert "producer_mma" in text and "retile(bridge_values)" in text
    assert "partition_D(consumer_a)" in text
    assert "cute.arch.sync_threads" not in text and "alloc_smem" not in text
    # All per-warp 16-column panels and all row bases have 16-byte alignment.
    assert all(
        2 * (r * layout.stride[0] + c) % 16 == 0
        for r in range(shape[0])
        for c in range(0, shape[1], 16)
    )


@pytest.mark.parametrize(
    "change",
    (
        "role",
        "inner",
        "shape",
        "column_stride",
        "small_stride",
        "unaligned_stride",
        "dtype",
        "foreign_node",
        "transposed_coords",
        "partial_rows",
        "partial_columns",
        "narrow_panel",
        "partial_warps",
        "divergent_cta",
        "pointer_alignment",
    ),
)
def test_unsupported_geometry_declines_without_changing_original_epilogue(change):
    epilogue, layout, options = _fixture()
    before = epilogue.render()
    if change == "role":
        layout = replace(layout, role="b")
    elif change == "inner":
        layout = replace(layout, inner=0)
    elif change == "shape":
        layout = replace(layout, shape=(32, 64))
    elif change == "column_stride":
        layout = replace(layout, stride=(72, 2))
    elif change == "small_stride":
        layout = replace(layout, stride=(56, 1))
    elif change == "unaligned_stride":
        layout = replace(layout, stride=(68, 1))
    elif change == "dtype":
        layout = replace(layout, dtype=torch.float32)
    elif change == "foreign_node":
        layout = replace(layout, node=layout.node.graph.placeholder("foreign"))
    elif change == "transposed_coords":
        epilogue = replace(
            epilogue,
            value=replace(
                epilogue.value, target_coordinates=epilogue.value.coordinates[::-1]
            ),
        )
    elif change in ("partial_rows", "partial_columns", "narrow_panel"):
        shape = {
            "partial_rows": (63, 64),
            "partial_columns": (64, 63),
            "narrow_panel": (32, 32),
        }[change]
        epilogue = replace(epilogue, shape=shape)
        layout = replace(layout, shape=shape)
    elif change == "partial_warps":
        options.update(producer_threads=127, cta_threads=127)
    elif change == "divergent_cta":
        options["producer_threads"] = 64
    else:
        options["pointer_alignment"] = 8
    assert render_stmatrix_operand(epilogue, layout, **options) is None
    if change not in (
        "transposed_coords",
        "partial_rows",
        "partial_columns",
        "narrow_panel",
    ):
        assert epilogue.render() == before


@pytest.mark.parametrize("change", ("expression", "fragment", "mask"))
def test_valid_geometry_does_not_recertify_changed_original_expression(change):
    epilogue, layout, options = _fixture()
    if change == "expression":
        epilogue.value.expression.lines.append("unproved = 1")
    elif change == "fragment":
        epilogue.value.expression.fragments[epilogue.value.source] = (
            epilogue.value.coordinates,
            "other_acc[bridge_index]",
        )
    else:
        object.__setattr__(epilogue.value, "domain", ())
    with pytest.raises(chain._UnsupportedChain, match="fragment epilogue"):
        render_stmatrix_operand(epilogue, layout, **options)
