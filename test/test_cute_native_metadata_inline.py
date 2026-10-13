from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_masked_async import _masked_loop
from .test_cute_chained_root_typed_operands import _args
from .test_cute_chained_root_typed_operands import _chain
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute import chained_root_warp_stage
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute import native_matmul_metadata as native

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from helion._compiler.cute.chained_coordinate_bounds import (
        CoordinateBoundsCanonicalization,
    )
    from helion._compiler.cute.chained_coordinate_sharing import CoordinateSharing
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.generate_ast import GenerateAST
    from helion.runtime.kernel import Kernel


def _case(route: str) -> tuple[Kernel, tuple[object, ...], helion.Config]:
    if route == "warp_loop":
        return (
            _masked_loop,
            (
                torch.empty((2, 3, 64, 64), dtype=torch.bfloat16),
                torch.empty((2, 64, 32)),
                64,
            ),
            helion.Config(num_warps=4, cute_chained_mma_schedule="cp_async"),
        )
    return (
        _chain,
        _args(),
        helion.Config(
            num_warps=4,
            cute_chained_mma_schedule="coalesced"
            if route == "warp_root"
            else "tcgen05_tmem",
            cute_chained_typed_operands=route == "tcgen_typed",
        ),
    )


def _code(route: str, enabled: bool = True) -> str:
    kernel, args, config = _case(route)
    config.config[native.KEY] = enabled
    with _cpu_codegen():
        return kernel._bind_isolated(args).to_code(config)


def _device(source: str) -> list[str]:
    return [
        ast.dump(node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion")
    ]


@pytest.mark.parametrize(
    "route", ["warp_root", "warp_loop", "tcgen_root", "tcgen_typed"]
)
def test_inline_stages_complete_and_device_source_unchanged(route: str) -> None:
    receipts: list[tuple[str, tuple[Node, ...], tuple[str, ...]]] = []
    commits: list[tuple[Node, ...]] = []
    record, commit = native.record_native_stage, native.commit_body

    def observe(
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        nodes: tuple[Node, ...],
        kind: str,
        lines: Sequence[str],
    ) -> None:
        receipts.append((kind, nodes, tuple(lines)))
        record(cg, plan, nodes, kind, lines)

    def committed(
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        lines: Sequence[str],
        *,
        coordinate_bounds: CoordinateBoundsCanonicalization | None = None,
        original_body: Sequence[ast.AST] | None = None,
        coordinate_sharing: CoordinateSharing | None = None,
        bounded_body: Sequence[ast.AST] | None = None,
    ) -> None:
        commit(
            cg,
            plan,
            lines,
            coordinate_bounds=coordinate_bounds,
            original_body=original_body,
            coordinate_sharing=coordinate_sharing,
            bounded_body=bounded_body,
        )
        commits.append(plan.dots)

    before = _code(route, False)
    with (
        patch.object(native, "record_native_stage", observe),
        patch.object(native, "commit_body", committed),
    ):
        after = _code(route)
    assert _device(before) == _device(after)
    assert len(commits) == 1
    assert Counter(node for _, nodes, _ in receipts for node in nodes) == Counter(
        commits[0]
    )
    assert len(receipts) == (2 if route == "warp_loop" else 3)
    assert {kind for kind, _, _ in receipts} == {
        "warp" if route.startswith("warp") else "tcgen05"
    }
    assert all("cute.gemm(" in "\n".join(lines) for _, _, lines in receipts)
    assert all("cute.arch.sync_threads()" in lines for _, _, lines in receipts)
    assert "_helion_cute_disable_bake_tensor_shapes = False" in after
    assert "_helion_cute_native_metadata_specialization = True" in after
    assert "_helion_cute_native_metadata_specialization" not in before


@pytest.mark.parametrize("route", ["warp_loop", "tcgen_typed"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "source", "unknown"])
def test_inline_receipts_cannot_bypass_existing_ledger(route: str, change: str) -> None:
    original = native.record_native_stage
    calls = 0

    def changed(
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        nodes: tuple[Node, ...],
        kind: str,
        lines: Sequence[str],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            if change == "missing":
                return
            if change == "unknown":
                native.record_unknown(cg)
            if change == "source":
                lines = [*lines, "uncommitted_statement()"]
            if change == "duplicate":
                original(cg, plan, nodes, kind, lines)
        original(cg, plan, nodes, kind, lines)

    with (
        patch.object(native, "record_native_stage", changed),
        pytest.raises(exc.BackendUnsupported, match="coverage"),
    ):
        _code(route)
    assert calls >= 2


@pytest.mark.parametrize("route", ["warp_loop", "tcgen_root"])
def test_failed_inline_attempt_cannot_publish_partial_receipts(route: str) -> None:
    captured: list[GenerateAST] = []
    original = native.record_native_stage

    def capture(
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        nodes: tuple[Node, ...],
        kind: str,
        lines: Sequence[str],
    ) -> None:
        captured.append(cg)
        original(cg, plan, nodes, kind, lines)

    module = chained_root_warp_stage if route == "warp_loop" else chained_tcgen05
    method = "emit_original_warp_operands" if route == "warp_loop" else "_stage"
    builder = getattr(module, method)

    def fail_later(*args: object, **kwargs: object) -> object:
        if captured:
            raise chained_matmul._UnsupportedChain("injected later-stage failure")
        return builder(*args, **kwargs)

    with (
        patch.object(native, "record_native_stage", capture),
        patch.object(module, method, fail_later),
        pytest.raises(exc.BackendUnsupported),
    ):
        _code(route)
    assert len(captured) == 1
    ledger = captured[0].cute_matmul_layouts
    assert ledger is not None and ledger.active is None and not ledger.committed
    assert "_helion_cute_native_metadata_specialization = True" in _code(route)
