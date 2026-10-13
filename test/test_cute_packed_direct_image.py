from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_row_registers import _config
from .test_cute_chained_row_storage_search import CACHE
from .test_cute_chained_row_storage_search import PACKING
from .test_cute_chained_row_storage_search import _args
from .test_cute_chained_row_storage_search import _rows
from helion._compiler.cute.chained_direct_image import DirectImageEdit
from helion._compiler.cute.chained_direct_image import _apply
from helion._compiler.cute.chained_direct_image import _row_publication_names
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_row_parallel import RowParallelEmission


@pytest.fixture(scope="module")
def actual_rows():
    captures = []
    validate = RowParallelEmission.validate

    def observe(self, plan):
        validate(self, plan)
        captures.append((self, plan))

    with _cpu_codegen(), patch.object(RowParallelEmission, "validate", observe):
        bound = _rows._bind_isolated(_args())
        bound.to_code(_config("warp_rows_parallel", **{CACHE: True, PACKING: True}))
    assert captures
    return captures[0]


def test_each_packed_publication_retains_its_original_output(actual_rows):
    rows, plan = actual_rows
    for receipt in rows.batches:
        proof = receipt.packing
        assert proof is not None
        first = proof.programs[0]
        for item in receipt.batch.islands:
            island = item.island
            row, column, registers = _row_publication_names(plan, island, rows)
            assert (row, column, registers) == (
                first.row,
                first.column,
                first.registers,
            )
            value = f"{registers}[{island.events[-1].offset} + {column}]"
            prefix = (
                f"if {row} < {island.shape[0]}:",
                f"    for {column} in cutlass.range_constexpr({island.shape[1]}):",
            )
            target = f"{island.name}[{row}, {column}]"
            edit = DirectImageEdit(
                (*prefix, f"        {target} = {value}"),
                (
                    *prefix,
                    f"        {target} = cutlass.BFloat16(cutlass.Float32({value}))",
                ),
            )
            assert edit.locate(receipt.phase) is not None
            lowered = _apply(receipt.phase, (edit,), complete=True)
            assert lowered[-1] == receipt.execution.sync
            assert sum("cutlass.BFloat16(cutlass.Float32" in x for x in lowered) == 1
            assert (
                _apply(
                    lowered, (DirectImageEdit(edit.after, edit.before),), complete=True
                )
                == receipt.phase
            )


@pytest.mark.parametrize(
    "mutation", ["missing", "foreign_island", "foreign_owner", "phase", "duplicate"]
)
def test_packed_publication_requires_exact_completion(actual_rows, mutation):
    rows, plan = actual_rows
    island = rows.batches[0].batch.islands[-1].island
    if mutation == "missing":
        rows = None
    elif mutation == "foreign_island":
        island = replace(island, name="foreign")
    elif mutation == "foreign_owner":
        rows = replace(rows, owner=object())
    elif mutation == "phase":
        receipt = replace(rows.batches[0], phase=(*rows.batches[0].phase, "pass"))
        rows = replace(rows, batches=[receipt, *rows.batches[1:]])
    else:
        rows = replace(rows, batches=[*rows.batches, rows.batches[0]])
    with pytest.raises(_UnsupportedChain):
        _row_publication_names(plan, island, rows)


@pytest.mark.parametrize("mutation", ["names", "source", "island", "phase"])
def test_packed_name_pullback_replays_whole_receipt(actual_rows, mutation):
    rows, plan = actual_rows
    receipt = rows.batches[0]
    proof = receipt.packing
    assert proof is not None
    island = proof.islands[-1]
    phase = receipt.phase
    if mutation == "names":
        proof = replace(proof, names=(("foreign", "foreign"), *proof.names[1:]))
    elif mutation == "source":
        proof = replace(proof, source=proof.source + "pass\n")
        proof = replace(proof, selection=proof.facts())
    elif mutation == "island":
        island = replace(island, name="foreign")
    else:
        phase = (*phase[:-1], "cute.arch.sync_warp()")
    with pytest.raises(_UnsupportedChain):
        proof.publication_names(island, receipt.execution, phase)


def test_unpacked_publication_names_need_no_packing_receipt(actual_rows):
    rows, plan = actual_rows
    island = rows.batches[0].batch.islands[0].island
    plan = replace(plan, row_subwarp_packing=False)
    assert _row_publication_names(plan, island, None) == (
        f"{island.name}_row",
        f"{island.name}_column",
        f"{island.name}_values",
    )
