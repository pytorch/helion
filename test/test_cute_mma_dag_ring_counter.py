"""CPU source checks for bounded slot state with Int64 memory indices."""

from __future__ import annotations

import ast
import re

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_mma_dag_ring import ring_matmul

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("mode", "loads", "stages", "dtype", "wide", "pass_warps"),
    [
        ("warp", "lane", 3, torch.bfloat16, True, False),
        ("warp", "lane", 4, torch.float32, True, False),
        ("warp", "tma", 3, torch.bfloat16, True, False),
        ("tcgen05", "tma", 3, torch.bfloat16, True, False),
        ("warp", "lane", 3, torch.float32, True, False),
        ("warp", "tma", 3, torch.float32, True, False),
        ("tcgen05", "tma", 3, torch.float32, True, True),
        ("warp", "lane", 3, torch.bfloat16, False, False),
    ],
)
def test_wide_ring_state_keeps_a_bounded_int32_ordinal(
    mode: str,
    loads: str,
    stages: int,
    dtype: torch.dtype,
    wide: bool,
    pass_warps: bool,
) -> None:
    args = (
        torch.empty((19, 64, 64), dtype=dtype),
        torch.empty((19, 64, 64), dtype=torch.bfloat16),
        torch.empty(2, dtype=torch.int64 if wide else torch.int32),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(ring_matmul, backend="cute", static_shapes=True), args
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_loads=loads,
                    cute_mma_dag_stages=stages,
                    cute_mma_dag_warps=8 if pass_warps else 4,
                    cute_mma_dag_pass_warps="on" if pass_warps else "off",
                )
            )
        source = bound.to_code(config)
    assert ("mma_bf16" if mode == "warp" else "tcgen05") in source
    tree = ast.parse(source)
    ring_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id.startswith("md_ring")
    }
    if not wide:
        assert not ring_names
        return
    assert ".ldg_s64(" in source
    assert "cutlass.Int32(_helion_mma_dag.ldg_s64(" not in source
    if mode == "warp" and loads == "tma" and dtype == torch.bfloat16:
        # This path uses only the persistent TMA counter, with no raw ring.
        assert not ring_names
        assert "md_cnt0 = cutlass.Int32(0) if md_cnt0 == 6 else md_cnt0" in source
        return
    assert ring_names
    for name in ring_names:
        assert f"{name} = cutlass.Int32(0)" in source
        assert f"else {name}" in source
    # The ring ordinal replaces division of the wide induction variable, not
    # the induction variable itself or its relative memory address.
    assert ("md_ring =" if mode == "tcgen05" else "md_ring0 =") in source
    assert "for md_i" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.FloorDiv)):
            names = {n.id for n in ast.walk(node.left) if isinstance(n, ast.Name)}
            if isinstance(node.right, ast.Constant) and node.right.value == stages:
                assert not any(name.startswith("md_i") for name in names)
    if mode == "warp" and loads == "tma":
        assert "md_cnt0 = cutlass.Int32(0) if md_cnt0 == 6 else md_cnt0" in source
    if mode == "warp" and stages == 4:
        assert "md_ring0 == 12" in source
        assert re.search(r"md_rstage(?:_\d+)? = md_ring0 % 3", source)
    if pass_warps:
        assert source.count("md_ring = md_ring + 1") == 2
