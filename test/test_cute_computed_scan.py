from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import computed_scan_root
from helion._compiler.cute import positional_root
from helion._compiler.cute.chained_collectives import warp_prefix_point
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.computed_scan_root import ComputedScanPlan
from helion._compiler.cute.positional_root import _Unsupported
from helion._testing import DEVICE
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prefix(
    x,
    bias,
    rate,
    extent: hl.constexpr,
    transpose: hl.constexpr,
):
    scan_extent = cast("int", extent)
    batch, tokens, channels = x.shape
    output = torch.empty(
        (batch, scan_extent, channels), dtype=torch.float32, device=x.device
    )
    for b, channel in hl.tile([batch, channels], block_size=[1, None]):
        token = hl.arange(scan_extent)
        raw = hl.load(
            x, [b.id, token, channel], extra_mask=(token < tokens)[:, None]
        ).float()
        shifted = raw + bias[channel][None, :]
        softplus = torch.clamp(shifted, min=0.0) + torch.log1p(
            torch.exp(-torch.abs(shifted))
        )
        steps = torch.where((token < tokens)[:, None], rate[b.id] * softplus, 0.0)
        if transpose:
            prefix = hl.cumsum(steps.transpose(0, 1), 1).transpose(0, 1)
        else:
            prefix = hl.cumsum(steps, 0)
        output[b.id, :, channel] = prefix
    return output


def _args(device="cpu", dtype=torch.bfloat16):
    return (
        torch.empty((3, 37, 35), dtype=dtype, device=device),
        torch.empty((35,), device=device),
        torch.empty((3,), device=device),
    )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("extent", [48, 64])
def test_computed_prefix_codegen(transpose, extent) -> None:
    plans = []
    original = computed_scan_root.plan_computed_scan_root

    def observe(graphs, df):
        plan = original(graphs, df)
        if plan is not None:
            plan.check()
        plans.append(plan)
        return plan

    with (
        _cpu_codegen(),
        patch.object(computed_scan_root, "plan_computed_scan_root", observe),
    ):
        bound = _prefix._bind_isolated((*_args(), extent, transpose))
        source = bound.to_code(helion.Config(block_sizes=[16], num_warps=4))
        assert len(plans) == 1 and isinstance(plans[0], ComputedScanPlan)
        plan = plans[0]
        assert plan.extent == extent
        assert "ptp_computed_scan_" in source and "shuffle_sync_up" in source
        assert source.count("alloc_smem(") == 1
        assert source.count("sync_threads()") == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _raw_scan(x):
    output = torch.empty_like(x)
    for row in hl.tile(x.shape[0], block_size=1):
        output[row, :] = hl.cumsum(x[row, :], 1)
    return output


def test_existing_raw_scan_source_is_unchanged() -> None:
    with _cpu_codegen():
        bound = _raw_scan._bind_isolated((torch.empty((3, 32)),))
        config = helion.Config(num_warps=4)
        actual = bound.to_code(config)
        with patch.object(
            computed_scan_root, "plan_computed_scan_root", return_value=None
        ):
            original = bound.to_code(config)
        assert actual == original
        assert "ptp_computed_scan_" not in actual


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rejected(x, kind: hl.constexpr):
    batch, extent, channels = x.shape
    extent = hl.specialize(extent)
    output = x if kind == "alias" else torch.empty_like(x)
    extra = torch.empty_like(x)
    for row, col in hl.tile([batch, channels], block_size=[1, None]):
        values = torch.exp(x[row.id, :, col].float())
        if kind == "output_read":
            values = values + output[row.id, :, col]
        if kind == "dtype":
            prefix = hl.cumsum(values.to(torch.float16), 0).float()
        elif kind == "reverse":
            prefix = hl.cumsum(values, 0, reverse=True)
        elif kind == "multiply":
            prefix = hl.cumprod(values, 0)
        else:
            prefix = hl.cumsum(values, 0)
        if kind == "omitted_axis":
            output[0, :, col] = prefix
        else:
            output[row.id, :, col] = prefix
        if kind == "extra_store":
            extra[row.id, :, col] = values
    return output


def _assert_ordinary_scan_fallback(bound, config) -> None:
    observed: list[tuple[str, object]] = []
    computed = computed_scan_root.plan_computed_scan_root
    positional = positional_root.plan_positional_root

    def observe_computed(graphs, df):
        plan = computed(graphs, df)
        observed.append(("computed", plan))
        return plan

    def observe_positional(graphs, df):
        plan = positional(graphs, df)
        observed.append(("positional", plan))
        return plan

    with (
        patch.object(computed_scan_root, "plan_computed_scan_root", observe_computed),
        patch.object(positional_root, "plan_positional_root", observe_positional),
    ):
        actual = bound.to_code(config)
    assert observed == [("positional", None), ("computed", None)]
    assert "ptp_computed_scan_" not in actual
    # Ordinary CuTe scans now support these cases. Declining a specialized
    # whole-root plan must preserve that existing fallback byte for byte.
    with (
        patch.object(computed_scan_root, "plan_computed_scan_root", return_value=None),
        patch.object(positional_root, "plan_positional_root", return_value=None),
    ):
        assert actual == bound.to_code(config)


@pytest.mark.parametrize(
    "kind",
    [
        "alias",
        "dtype",
        "reverse",
        "multiply",
        "omitted_axis",
        "extra_store",
        "output_read",
    ],
)
def test_unsafe_or_unsupported_roots_do_not_gain_admission(kind) -> None:
    with _cpu_codegen():
        bound = _rejected._bind_isolated((torch.empty((3, 48, 35)), kind))
        _assert_ordinary_scan_fallback(
            bound, helion.Config(block_sizes=[16], num_warps=4)
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _split_axis(x):
    output = torch.empty_like(x)
    for row, token, channel in hl.tile(x.shape, block_size=[1, 16, None]):
        values = torch.exp(x[row.id, token, channel])
        output[row.id, token, channel] = hl.cumsum(values, 0)
    return output


def test_scan_axis_split_between_ctas_declines() -> None:
    with _cpu_codegen():
        bound = _split_axis._bind_isolated((torch.empty((3, 37, 35)),))
        _assert_ordinary_scan_fallback(
            bound, helion.Config(block_sizes=[16], num_warps=4)
        )


@pytest.mark.parametrize(
    "mutation", ["dtype", "selector", "helper", "extent", "domain", "effect", "storage"]
)
def test_stale_complete_scan_proof_rejects(mutation) -> None:
    original = computed_scan_root.plan_computed_scan_root

    def observe(graphs, df):
        plan = original(graphs, df)
        assert isinstance(plan, ComputedScanPlan)
        if mutation == "extent":
            with pytest.raises(_Unsupported, match="changed"):
                replace(plan, extent=plan.extent - 1).check()
        elif mutation == "domain":
            with (
                patch.dict(plan.domains.extents, {plan.scan: (1, 1)}),
                pytest.raises(_Unsupported, match="changed"),
            ):
                plan.check()
        elif mutation == "dtype":
            with (
                patch.dict(plan.scan.meta, {"val": plan.scan.meta["val"].half()}),
                pytest.raises(_Unsupported, match="changed"),
            ):
                plan.check()
        elif mutation == "effect":
            with plan.scan.graph.inserting_before(plan.store):
                node = plan.scan.graph.call_function(torch.neg, (plan.scan,))
            try:
                with pytest.raises(_Unsupported, match="changed"):
                    plan.check()
            finally:
                plan.scan.graph.erase_node(node)
        elif mutation == "storage":
            env = CompileEnvironment.current()
            with (
                patch.object(env, "fresh_allocation_storages", set()),
                pytest.raises(_Unsupported, match="changed"),
            ):
                plan.check()
        else:
            node = plan.store if mutation == "selector" else plan.helper_nodes[-2]
            saved = node.args
            if mutation == "selector":
                selectors = saved[1]
                assert isinstance(selectors, (tuple, list))
                node.args = (saved[0], [0, slice(None), selectors[-1]], *saved[2:])
            else:
                node.args = tuple(reversed(saved))
            try:
                with pytest.raises(_Unsupported, match="changed"):
                    plan.check()
            finally:
                node.args = saved
        plan.check()
        return plan

    with (
        _cpu_codegen(),
        patch.object(computed_scan_root, "plan_computed_scan_root", observe),
    ):
        bound = _prefix._bind_isolated((*_args(), 64, False))
        assert "ptp_computed_scan_" in bound.to_code(
            helion.Config(block_sizes=[16], num_warps=4)
        )


def _execute_prefix_statements(statements, environments, active):
    """Execute emitted scalar statements in lockstep, including actual shuffle operands."""

    def evaluate(node, lane):
        if isinstance(node, ast.Call) and ast.unparse(node.func).startswith(
            "cute.arch.shuffle_sync"
        ):
            assert active == list(range(32)), "a shuffle must have all 32 lanes"
            value = node.args[0]
            assert isinstance(value, ast.Name)
            if ast.unparse(node.func).endswith("_up"):
                offset = int(ast.literal_eval(node.keywords[0].value))
                owner = lane - offset if lane >= offset else lane
            else:
                owner = int(ast.literal_eval(node.args[1]))
            return environments[owner][value.id]
        return eval(
            compile(ast.Expression(node), "<emitted prefix>", "eval"),
            {"cutlass": SimpleNamespace(Float32=np.float32)},
            environments[lane],
        )

    for statement in statements:
        if isinstance(statement, ast.Assign):
            assert len(statement.targets) == 1 and isinstance(
                statement.targets[0], ast.Name
            )
            values = [evaluate(statement.value, lane) for lane in active]
            for lane, value in zip(active, values, strict=True):
                environments[lane][statement.targets[0].id] = value
        else:
            assert isinstance(statement, ast.If) and not statement.orelse
            selected = [lane for lane in active if evaluate(statement.test, lane)]
            _execute_prefix_statements(statement.body, environments, selected)


@pytest.mark.parametrize("extent", [1, 17, 32, 48, 64, 95])
def test_actual_prefix_body_ownership_and_arithmetic(extent) -> None:
    source = np.asarray(
        [(-1.0 if i % 3 == 0 else 1.0) * (i + 1) / 64 for i in range(extent)],
        dtype=np.float32,
    )
    source[0] = np.float32(-0.0)
    text = "\n".join(
        warp_prefix_point(
            "test",
            extent,
            [],
            "source[test_position]",
            execution=ChainedExecution(128, thread="thread"),
            initial_add=False,
        )
    )
    statements = ast.parse(text).body
    envs = [
        {"thread": lane, "source": source, "test_carry": np.float32(0)}
        for lane in range(32)
    ]
    actual = []
    for part in range((extent + 31) // 32):
        for env in envs:
            env["test_part"] = part
        _execute_prefix_statements(statements, envs, list(range(32)))
        actual.extend(env["test_acc"] for env in envs)
    actual = np.asarray(actual[:extent], dtype=np.float32)
    expected = np.cumsum(source.astype(np.float64))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.view(np.uint32)[0] == source.view(np.uint32)[0]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _one_prefix(x):
    output = torch.empty_like(x, dtype=torch.float32)
    for _unused in hl.tile(1, block_size=1):
        output[:] = hl.cumsum(-x[:].float(), 0)
    return output


@pytest.mark.parametrize("extent", [1, 17, 95])
def test_rank_one_computed_prefix_codegen(extent) -> None:
    with _cpu_codegen():
        bound = _one_prefix._bind_isolated((torch.empty(extent),))
        source = bound.to_code(helion.Config(num_warps=4))
        assert "ptp_computed_scan_" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_gpu_rank_one_nonfinite_and_first_signed_zero() -> None:
    x = torch.arange(47, device=DEVICE, dtype=torch.float32) / 16
    x[9] = float("inf")
    x[21] = float("-inf")
    x[37] = float("nan")
    saved = x.clone()
    bound = _one_prefix._bind_isolated((x,))
    bound.set_config(helion.Config(num_warps=4))
    assert "ptp_computed_scan_" in bound.to_code()
    actual = bound(x)
    torch.testing.assert_close(
        actual, (-x).double().cumsum(0).float(), atol=1e-5, rtol=1e-5, equal_nan=True
    )
    assert actual.view(torch.int32)[0].item() == -(2**31)
    torch.testing.assert_close(x, saved, atol=0, rtol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
def test_gpu_computed_prefix_tail_and_transpose(dtype, transpose) -> None:
    torch.manual_seed(19)
    backing = torch.full((3, 41, 40), float("nan"), dtype=dtype, device=DEVICE)
    x = backing[:, :37, :35]
    x.copy_(torch.randn_like(x) * 2)
    bias = torch.linspace(-1, 1, 35, device=DEVICE)
    rate = torch.tensor([-0.01, -0.03, -0.07], device=DEVICE)
    saved = backing.clone()
    bias_saved, rate_saved = bias.clone(), rate.clone()
    shifted = x.float() + bias[None, None, :]
    softplus = torch.clamp(shifted, min=0) + torch.log1p(torch.exp(-torch.abs(shifted)))
    steps = rate[:, None, None] * softplus
    expected = torch.zeros((3, 64, 35), dtype=torch.float64, device=DEVICE)
    expected[:, :37, :] = steps.double()
    expected = expected.cumsum(1)
    bound = _prefix._bind_isolated((x, bias, rate, 64, transpose))
    bound.set_config(helion.Config(block_sizes=[16], num_warps=4))
    assert "ptp_computed_scan_" in bound.to_code()
    for _ in range(2):
        actual = bound(x, bias, rate, 64, transpose)
        torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(backing, saved, atol=0, rtol=0, equal_nan=True)
        assert torch.equal(bias, bias_saved) and torch.equal(rate, rate_saved)
