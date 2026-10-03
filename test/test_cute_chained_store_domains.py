from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_loop import _storage_sources
from helion._compiler.cute.chained_loop import discover_chained_loop
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_domain_loop(
    a,
    b,
    initial,
    history,
    axis: hl.constexpr,
    broadcast: hl.constexpr,
    extra: hl.constexpr,
):
    steps, m, k = a.shape
    n = b.shape[-1]
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[128, 64]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            product = hl.dot(
                a[step.id, rows, kk], b[step.id, kk, cols], out_dtype=torch.float32
            )
            state = state * 0.5 + product
            if axis == "raw":
                hl.store(history, [step.id, rows, cols], state)
            else:
                if axis == "rows" or axis == "both":
                    rr = (rows.index + 1) % m
                else:
                    rr = rows.index
                if axis == "columns" or axis == "both":
                    cc = (cols.index + 1) % n
                else:
                    cc = cols.index
                if broadcast:
                    if extra:
                        hl.store(
                            history,
                            [step.id, rr[:, None], cc[None, :]],
                            state,
                            extra_mask=(rows.index[:, None] + cols.index[None, :]) % 3
                            != 1,
                        )
                    else:
                        hl.store(history, [step.id, rr[:, None], cc[None, :]], state)
                else:
                    if extra:
                        hl.store(
                            history,
                            [step.id, rr, cc],
                            state,
                            extra_mask=(rows.index[:, None] + cols.index[None, :]) % 3
                            != 1,
                        )
                    else:
                        hl.store(history, [step.id, rr, cc], state)
        final[rows, cols] = state
    return history, final


def _args(device: str | torch.device, axis: str, broadcast: bool, extra: bool) -> tuple:
    generator = torch.Generator(device=device).manual_seed(4312)
    return (
        torch.randn(
            (2, 130, 16), device=device, dtype=torch.bfloat16, generator=generator
        )
        * 0.05,
        torch.randn(
            (2, 16, 66), device=device, dtype=torch.bfloat16, generator=generator
        )
        * 0.05,
        torch.randn((130, 66), device=device, dtype=torch.float32, generator=generator)
        * 0.05,
        torch.full((2, 258, 130), 13.0, device=device, dtype=torch.float32),
        axis,
        broadcast,
        extra,
    )


def _config(schedule: str) -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule=schedule)


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize(
    "axis,broadcast,extra",
    [
        ("raw", False, False),
        ("rows", False, False),
        ("columns", False, False),
        ("both", False, True),
        ("both", True, False),
        ("both", True, True),
    ],
)
def test_store_keeps_logical_index_domains(
    schedule: str, axis: str, broadcast: bool, extra: bool
) -> None:
    with _cpu_codegen():
        bound = _store_domain_loop._bind_isolated(_args("cpu", axis, broadcast, extra))
        source = bound.to_code(_config(schedule))
    assert "chain_loop_index" in source
    tree = ast.parse(source)
    store_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_store_step"
    )
    guard = next(
        node.test
        for node in ast.walk(store_loop)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Subscript) for target in statement.targets)
            for statement in node.body
        )
    )
    predicate = ast.unparse(guard)
    assert "chain_origin_0 + chain_store // 64 < 130" in predicate
    assert "chain_origin_1 + chain_store % 64 < 66" in predicate
    assert "chain_origin_0 + 0 < 130" not in predicate
    assert "chain_origin_1 + 0 < 66" not in predicate


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize(
    "axis,broadcast,extra",
    [
        ("raw", False, False),
        ("rows", False, False),
        ("columns", False, False),
        ("both", False, True),
        ("both", True, False),
        ("both", True, True),
    ],
)
def test_store_domains_gpu_preserve_padded_and_masked_destinations(
    schedule: str, axis: str, broadcast: bool, extra: bool
) -> None:
    args = _args(DEVICE, axis, broadcast, extra)
    a, b, initial, history = args[:4]
    expected = history.clone()
    state = initial.clone()
    rr, cc = torch.arange(130, device=DEVICE), torch.arange(66, device=DEVICE)
    if axis in ("rows", "both"):
        rr = (rr + 1) % 130
    if axis in ("columns", "both"):
        cc = (cc + 1) % 66
    mask = (
        torch.arange(130, device=DEVICE)[:, None]
        + torch.arange(66, device=DEVICE)[None, :]
    ) % 3 != 1
    for step in range(2):
        state = state * 0.5 + a[step].float() @ b[step].float()
        values = torch.where(mask, state, 13.0) if extra else state
        expected[step, rr[:, None], cc[None, :]] = values
    bound = _store_domain_loop._bind_isolated(args)
    config = _config(schedule)
    assert "chain_loop_index" in bound.to_code(config)
    actual_history, actual_final = bound.compile_config(config)(*args)
    torch.testing.assert_close(actual_history, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_final, state, atol=2e-3, rtol=2e-3)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _internal_read_loop(a, b, initial, history, positions, gather: hl.constexpr):
    steps, m, k = a.shape
    n = b.shape[-1]
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[16, 16]):
        state = hl.zeros([rows, cols], dtype=torch.float32)
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            if gather:
                gathered = hl.load(
                    a[step.id, rows, kk].float(), [slice(None), positions[kk]]
                )
            else:
                gathered = a[step.id, rows, kk].float()
            product = hl.dot(
                gathered.to(a.dtype), b[step.id, kk, cols], out_dtype=torch.float32
            )
            raw = initial[rows, cols]
            view = hl.load(raw, [slice(None), slice(None)])
            state = state * 0.5 + product + view
            history[step.id, rows, cols] = state
        final[rows, cols] = state
    return history, final


def _internal_args(aliased: bool, gather: bool = False) -> tuple:
    history = torch.empty((2, 16, 16), dtype=torch.float32)
    return (
        torch.empty((2, 16, 16), dtype=torch.bfloat16),
        torch.empty((2, 16, 16), dtype=torch.bfloat16),
        history[0] if aliased else torch.empty((16, 16), dtype=torch.float32),
        history,
        torch.arange(16, dtype=torch.int32),
        gather,
    )


@pytest.mark.parametrize("aliased", [False, True])
def test_internal_load_view_cannot_hide_external_read_write_alias(
    aliased: bool,
) -> None:
    with _cpu_codegen():
        bound = _internal_read_loop._bind_isolated(_internal_args(aliased))
        assert bound.host_function is not None
        assert any(
            node.target is memory_ops.load
            and node.args[0].target is not _tracing_ops._host_tensor
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
        )
        assert bound.config_spec.cute_chained_loop_search_enabled == (not aliased)
        if not aliased:
            assert "chain_loop_index" in bound.to_code(_config("tcgen05_tmem"))


def test_internal_store_remains_outside_external_storage_proof() -> None:
    with _cpu_codegen():
        bound = _internal_read_loop._bind_isolated(_internal_args(False))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            loop = discover_chained_loop(bound.host_function.device_ir.graphs)
            assert loop is not None
            internal = next(
                node
                for node in loop.body.graph.nodes
                if node.target is memory_ops.load
                and node.args[0].target is not _tracing_ops._host_tensor
            )
            store = loop.region.stores[0]
            store.args = (internal, *store.args[1:])
            loop.body.graph.lint()
            assert _storage_sources(loop) is None


def test_internal_loop_gather_keeps_data_dependent_domain_rejection() -> None:
    original = chained_matmul._operand_domain
    failures: list[str] = []

    def record_domain_failure(*args):
        try:
            return original(*args)
        except chained_matmul._UnsupportedChain as error:
            failures.append(str(error))
            raise

    with _cpu_codegen():
        bound = _internal_read_loop._bind_isolated(_internal_args(False, True))
        assert bound.config_spec.cute_chained_loop_search_enabled
        with (
            patch.object(chained_matmul, "_operand_domain", record_domain_failure),
            pytest.raises(exc.BackendUnsupported, match="late validation"),
        ):
            bound.to_code(_config("tcgen05_tmem"))
    assert failures == ["data-dependent contraction operand domain"]
