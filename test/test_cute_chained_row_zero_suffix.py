from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_row_registers import _args
from .test_cute_chained_row_registers import _config
from .test_cute_chained_row_registers import _fixture
from .test_cute_chained_row_registers import _poisoned_sequence
from .test_cute_chained_row_registers import _sequence
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_row_register_emission import _normalized_zero_adds
from helion._compiler.cute.chained_row_register_emission import _zero_adds
from helion._compiler.cute.chained_sparse_reduction import plan_sparse_reduction


def _patterns() -> np.ndarray:
    # Every BF16 encoding embedded in FP32 plus actual FP32 boundary/subnormal
    # and signaling/quiet NaN payloads. Execute emitted statements, not a second
    # implementation of the intended zero identity.
    bits = np.concatenate(
        (
            np.arange(65536, dtype=np.uint32) << 16,
            np.array(
                [
                    0,
                    0x80000000,
                    1,
                    0x80000001,
                    0x007FFFFF,
                    0x807FFFFF,
                    0x00800000,
                    0x80800000,
                    0x7F7FFFFF,
                    0xFF7FFFFF,
                    0x7F800000,
                    0xFF800000,
                    0x7F800001,
                    0xFF800001,
                    0x7FC00001,
                    0xFFC00001,
                    0x7FFFFFFF,
                    0xFFFFFFFF,
                ],
                dtype=np.uint32,
            ),
            np.random.default_rng(816).integers(0, 2**32, 8192, dtype=np.uint32),
        )
    )
    return bits.view(np.float32)


def _execute(lines: list[str], value: np.ndarray) -> np.ndarray:
    namespace = {"cutlass": SimpleNamespace(Float32=np.float32), "value": value}
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        exec("\n".join(lines), namespace)
    result = namespace["result"]
    assert isinstance(result, np.ndarray)
    return result


@pytest.mark.parametrize("position", range(32))
def test_emitted_normalized_suffix_is_fp32_bitwise_fixed_point(position: int) -> None:
    plan, reductions, _ = _fixture(32, 32)
    selection = reductions[0].args[0]
    assert isinstance(selection, Node)
    mask = selection.args[0]
    assert isinstance(mask, Node)
    condition = mask.args[0]
    assert isinstance(condition, Node)
    condition.args = (condition.args[0], position)
    proof = plan_sparse_reduction(plan, reductions[0])
    assert proof is not None and proof.position == position
    original = _zero_adds("value", "result", proof.zero_sides)
    folded = _normalized_zero_adds(plan, proof, "value", "result")
    assert len(original) == 6 and folded == original[:1]
    values = _patterns()
    np.testing.assert_array_equal(
        _execute(original, values).view(np.uint32),
        _execute(folded, values).view(np.uint32),
    )
    # The raw negative-zero input is not substituted for the normalized result.
    negative_zero = np.array([0x80000000], dtype=np.uint32).view(np.float32)
    assert _execute(folded, negative_zero).view(np.uint32).item() == 0
    assert negative_zero.view(np.uint32).item() == 0x80000000


@pytest.mark.parametrize(
    "mutation", ["negative_zero", "nonzero", "dtype", "position", "sides"]
)
def test_fold_requires_current_original_typed_sparse_proof(mutation: str) -> None:
    plan, reductions, _ = _fixture()
    proof = plan_sparse_reduction(plan, reductions[0])
    assert proof is not None
    if mutation in ("negative_zero", "nonzero"):
        zero = proof.source.args[2]
        assert isinstance(zero, Node)
        zero.args = (-0.0 if mutation == "negative_zero" else 1.0,)
    elif mutation == "dtype":
        proof.selected.meta["val"] = proof.selected.meta["val"].to(torch.float16)
    elif mutation == "position":
        proof = replace(proof, position=32)
    else:
        proof = replace(proof, zero_sides=())
    with pytest.raises(_UnsupportedChain, match="proof changed"):
        _normalized_zero_adds(plan, proof, "value", "result")


@pytest.mark.parametrize("poison", [False, True])
def test_real_row_emission_keeps_first_add_and_every_nonzero_recipe(
    poison: bool,
) -> None:
    kernel = _poisoned_sequence if poison else _sequence
    args = _args(torch.float16) if poison else (*_args(torch.float16), True)
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        source = bound.to_code(_config("warp_rows"))
    tree = ast.parse(source)
    selected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("chain_row_register_")
        and "_selected_" in node.targets[0].id
    ]
    assert len(selected) == 6
    for node in selected:
        assert isinstance(node.targets[0], ast.Name)
        assert isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
        assert ast.unparse(node.value.left) == "cutlass.Float32(0)"
        assert node.targets[0].id not in {
            n.id for n in ast.walk(node.value.right) if isinstance(n, ast.Name)
        }
    assert "shuffle_sync" in source and "cute.gemm" in source
    if poison:
        assert 'float("nan")' in source or "float('nan')" in source


def test_shared_schedule_does_not_apply_row_suffix_fold() -> None:
    with _cpu_codegen():
        bound = _sequence._bind_isolated((*_args(torch.float16), False))
        original = bound.to_code(_config("shared"))
        with patch(
            "helion._compiler.cute.chained_row_register_emission._normalized_zero_adds",
            side_effect=AssertionError("row fold reached shared schedule"),
        ):
            assert original == bound.to_code(_config("shared"))
