from __future__ import annotations

import ast
from dataclasses import replace

import pytest
import torch

from .test_cute_chained_vector_group import _calls
from .test_cute_chained_vector_group import _capture
from .test_cute_chained_vector_group import _expanded_outputs
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("height", [32, 65, 128])
def test_original_per_cell_arithmetic_masks_and_copy_partition(dtype, scalar, height):
    shape = (height, 128)
    ownership = plan_vector_ownership(shape, 128, tile_columns=32)
    assert ownership is not None
    before, old_single, _ = _capture(dtype=dtype, scalar=scalar, shape=shape)
    after, new_single, tracker = _capture(
        dtype=dtype, scalar=scalar, shape=shape, ownership=ownership
    )
    assert before is not None and after is not None and tracker.activated
    pairs = [(before, after, "joined")]
    pairs.extend(
        (old, new, f"ordinary_{index}")
        for index, (old, new) in enumerate(zip(old_single, new_single, strict=True))
    )
    for old, new, tag in pairs:
        assert old is not None and new is not None
        assert _expanded_outputs(old, f"{tag}_element") == _expanded_outputs(
            new, f"{tag}_element"
        )
        source = "\n".join(new)
        assert _calls("\n".join(old)) == _calls(source)
        assert (
            f"{tag}_row = {ownership.row_expression('chain_thread', f'{tag}_step')}"
            in source
        )
        assert (
            f"{tag}_base = {ownership.base_expression('chain_thread', f'{tag}_step')}"
            in source
        )
        assert f"if {tag}_row < {height}:" in source
        assert "sync_threads" not in source and "alloc_smem" not in source
        if not scalar:
            copies = [
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "cute.copy"
                and isinstance(node.args[-1], ast.Subscript)
                and "_target" in ast.unparse(node.args[-1].value)
            ]
            assert copies
            for copy in copies:
                assert isinstance(copy.args[-1], ast.Subscript)
                assert ast.dump(copy.args[-1].slice) == ast.dump(
                    ast.parse(ownership.copy_indices(f"{tag}_step"), mode="eval").body
                )


@pytest.mark.parametrize("columns", [0, 128])
def test_default_or_full_width_is_byte_identical(columns):
    ownership = plan_vector_ownership((128, 128), 128, tile_columns=columns)
    before, old_single, _ = _capture()
    after, new_single, _ = _capture(ownership=ownership)
    assert before == after and old_single == new_single


@pytest.mark.parametrize("shape,threads", [((32, 128), 128), ((128, 128), 256)])
def test_foreign_geometry_rejects_before_unroll(shape, threads):
    ownership = plan_vector_ownership(shape, threads, tile_columns=32)
    assert ownership is not None
    merged, single, tracker = _capture(ownership=ownership)
    assert merged is None and all(item is None for item in single)
    assert not tracker.activated


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, trips=1),
        lambda value: replace(value, row_tiles=1),
        lambda value: replace(value, column_tiles=1),
        lambda value: replace(value, thread_rows=1),
        lambda value: replace(value, thread_columns=1),
    ],
    ids=["trips", "row_tiles", "column_tiles", "thread_rows", "thread_columns"],
)
def test_inconsistent_derived_record_rejects_before_unroll(change):
    ownership = plan_vector_ownership((128, 128), 128, tile_columns=32)
    assert ownership is not None
    malformed = change(ownership)
    assert malformed != ownership
    merged, single, tracker = _capture(ownership=malformed)
    assert merged is None and all(item is None for item in single)
    assert not tracker.activated
