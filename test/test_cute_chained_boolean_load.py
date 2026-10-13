from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _boolean_mask_dot(a, b, valid):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, columns in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        selected = hl.load(valid, [rows], extra_mask=(rows.index % 5) != 1)
        left = torch.where(selected[:, None], a[rows, kk], 0)
        out[rows, columns] = hl.dot(left, b[kk, columns], out_dtype=torch.float32)
    return out


def _config(schedule):
    return helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("coalesced", "k_major"))
def test_global_boolean_load_restores_predicate_before_mask(dtype, schedule):
    values = (
        torch.empty((129, 16), dtype=dtype),
        torch.empty((16, 32), dtype=dtype),
        torch.empty(129, dtype=torch.bool),
    )
    with _cpu_codegen():
        source = _boolean_mask_dot._bind_isolated(values).to_code(_config(schedule))
    conditionals = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.IfExp)
        and ast.unparse(node.orelse) == "cutlass.Boolean(0)"
    ]
    assert conditionals
    assert all(
        ast.unparse(node.body).startswith("cutlass.Boolean(") for node in conditionals
    )
    assert "chain_0_mma" in source


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("coalesced", "k_major"))
def test_global_boolean_bytes_mask_tail_and_replay_gpu(dtype, schedule):
    generator = torch.Generator(device=DEVICE).manual_seed(92813)
    a = torch.randn((129, 16), dtype=dtype, device=DEVICE, generator=generator) * 0.25
    b = torch.randn((16, 32), dtype=dtype, device=DEVICE, generator=generator) * 0.25
    # torch.bool is byte-addressed: every nonzero byte is true, not only bit0.
    payload = torch.tensor([0, 1, 2, 128, 255], dtype=torch.uint8, device=DEVICE)
    valid = payload[torch.arange(129, device=DEVICE) % 5].view(torch.bool)
    values = (a, b, valid)
    before = tuple(value.view(torch.uint8).clone() for value in values)
    compiled = _boolean_mask_dot._bind_isolated(values).compile_config(
        _config(schedule)
    )
    selected = valid & (torch.arange(129, device=DEVICE) % 5 != 1)
    expected = (torch.where(selected[:, None], a, 0).double() @ b.double()).float()
    actual = compiled(*values)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*values)
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        assert torch.equal(captured.view(torch.int32), actual.view(torch.int32))
    for value, saved in zip(values, before, strict=True):
        assert torch.equal(value.view(torch.uint8), saved)
