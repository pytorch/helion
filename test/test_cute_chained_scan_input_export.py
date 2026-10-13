from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_chained_scan_export import _scan_export_config
from test.test_cute_chained_scan_export import _scan_export_cpu_codegen

import helion
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _export(a, b, raw, mode: str):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    side = torch.empty((batch, reduction), dtype=torch.float32, device=a.device)
    for bi, mi, ni in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        mb: Any = mi.begin
        nb: Any = ni.begin
        ki = hl.arange(reduction)
        value = raw[bi.begin, ki].float()
        if mode == "min":
            value = value.clamp(min=-0.25)
        elif mode == "max":
            value = value.clamp(max=0.75)
        elif mode == "both":
            value = value.clamp(min=-0.25, max=0.75)
        elif mode == "chain":
            value = value.clamp(max=0.75).clamp(min=-0.25)
        elif mode == "multiply":
            value = value * 2.0
        elif mode == "second_leaf":
            value = value + raw[bi.begin, 0].float()
        elif mode == "narrow":
            value = value.to(torch.bfloat16).float()
        elif mode == "reindex":
            value = raw[bi.begin, reduction - 1 - ki].float()
        elif mode == "unrelated":
            value = value.clamp(min=0)
        scan_input = value
        if mode == "unrelated":
            scan_input = raw[bi.begin, ki].float()
        prefix = hl.cumsum(scan_input, dim=0)
        left = (a[bi.begin, mi, ki].float() * prefix[None, :]).to(a.dtype)
        out[bi.begin, mi, ni] = hl.dot(left, b[bi.begin, ki, ni])
        hl.store(side, [bi.begin, ki], value, extra_mask=(mb == 0) & (nb == 0))
    return out, side


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mode", ["cast", "min", "max", "both", "chain"])
def test_cast_clamp_scan_input_source(dtype: torch.dtype, mode: str) -> None:
    with _scan_export_cpu_codegen():
        args = (
            torch.zeros(2, 256, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, dtype=dtype),
            mode,
        )
        source = _export._bind_isolated(args).to_code(_scan_export_config())
    assert "chain_export_0_index" in source
    assert ".store(cutlass.Float32(" in source


@pytest.mark.parametrize(
    "mode", ["multiply", "second_leaf", "narrow", "reindex", "unrelated"]
)
def test_nonlocal_or_computed_exports_reject(mode: str) -> None:
    with _scan_export_cpu_codegen():
        args = (
            torch.zeros(2, 256, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, dtype=torch.float32),
            mode,
        )
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            _export._bind_isolated(args).to_code(_scan_export_config())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cast_clamp_typed_edge_values(dtype: torch.dtype) -> None:
    # CPU semantics of the unchanged typed FX operation chain. GPU NaN lowering
    # is inherited; this does not claim a new device floating-point contract.
    from helion._compiler.cute.chained_scan_export import _scan_input_leaf
    from helion._compiler.device_ir import RootGraphInfo
    from helion.language import memory_ops

    edge = torch.tensor(
        [-float("inf"), -1, -0.0, 0.0, 1.0, float("inf"), float("nan")], dtype=dtype
    )
    with _scan_export_cpu_codegen():
        args = (
            torch.zeros(2, 256, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, dtype=dtype),
            "chain",
        )
        bound = _export._bind_isolated(args)
        assert bound.host_function is not None
        root = next(
            graph
            for graph in bound.host_function.device_ir.graphs
            if isinstance(graph, RootGraphInfo)
        )
        side = next(
            node
            for node in root.graph.nodes
            if node.target is memory_ops.store and node.args[0].args[0] == "side"
        )
        value = side.args[2]
        leaf = _scan_input_leaf(value)
        assert leaf is not None

        def evaluate(node: Any) -> torch.Tensor:
            if node is leaf:
                return edge
            if len(node.args) == 1:
                return evaluate(node.args[0])
            return node.target(evaluate(node.args[0]), *node.args[1:], **node.kwargs)

        actual = evaluate(value)
        expected = edge.float().clamp(max=0.75).clamp(min=-0.25)
        assert actual.dtype is torch.float32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        assert torch.equal(
            actual[:-1].view(torch.int32), expected[:-1].view(torch.int32)
        )
