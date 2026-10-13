from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_collectives
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_scan_input import KEY
from helion._compiler.cute.chained_scan_input import bind_scan_input
from helion._compiler.cute.chained_scan_input import plan_scan_inputs
from helion._compiler.cute.producer_phase import CollectiveProducer
from helion._compiler.cute.producer_phase import PointStore
from helion._compiler.cute.producer_phase import PointValue
from helion._compiler.cute.producer_phase import emit_producer_phase
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import memory_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.generate_ast import GenerateAST
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence(raw, scale, left, right):
    output = torch.empty((raw.shape[0], 32, 32), device=raw.device, dtype=torch.float32)
    for batch in hl.tile(raw.shape[0], block_size=1):
        index = hl.arange(32)
        values = hl.load(raw, [batch.begin, index], index < raw.shape[1])
        nonlinear = torch.sigmoid(values.float())
        steps = torch.where(index < raw.shape[1], scale[batch.begin] * nonlinear, 0.0)
        prefix = hl.cumsum(steps, 0)
        lhs = left[batch.begin, :, :].float() + prefix[:, None] * 0.03125
        energy = (lhs * lhs).sum(1)
        result = hl.dot(lhs.to(left.dtype), right[batch.begin, :, :])
        output[batch.begin, :, :] = result + energy[:, None] * 0.125
    return output


def _args(dtype=torch.bfloat16, extent=19, device="cpu"):
    return (
        torch.empty((3, 32), dtype=dtype, device=device)[:, :extent],
        torch.empty((3,), device=device),
        torch.empty((3, 32, 32), dtype=dtype, device=device),
        torch.empty((3, 32, 32), dtype=dtype, device=device),
    )


def _config(enabled: object = True, *, schedule: str = "coalesced") -> helion.Config:
    return helion.Config.from_dict(
        {"num_warps": 4, "cute_chained_mma_schedule": schedule, KEY: enabled}
    )


@pytest.fixture(scope="module")
def _captured_source() -> tuple[BoundKernel, ChainedMatmulPlan, str]:
    plans = []
    original = chained_collectives.emit_collectives_before

    def observe(cg, plan, boundaries, stage, **kwargs):
        plans.append(plan)
        return original(cg, plan, boundaries, stage, **kwargs)

    with (
        _cpu_codegen(),
        patch.object(chained_collectives, "emit_collectives_before", observe),
    ):
        bound = _sequence._bind_isolated(_args())
        source = bound.to_code(_config())
        assert plans and len(plans[0].scan_inputs) == 1
        return bound, plans[0], source


@pytest.fixture
def captured(_captured_source) -> Iterator[tuple[BoundKernel, ChainedMatmulPlan, str]]:
    # Mock CUDA only for CPU tests that request this fixture. A module-scoped
    # mock would also cover the real GPU tests later in this file.
    with _cpu_codegen():
        yield _captured_source


def test_actual_scan_keeps_serial_combine_and_original_allocation(captured) -> None:
    bound, plan, source = captured
    selected = plan.scan_inputs[0]
    selected.check(plan)
    assert selected.extent == 32
    assert selected.frontier.target is torch.ops.aten.sigmoid.default
    assert "input_position" in source and "input_value" in source
    assert "cutlass.range(32, unroll=1)" in source
    original = bound.to_code(_config(False))
    assert source.count("alloc_smem(") == original.count("alloc_smem(")
    assert source.count("sync_threads()") == original.count("sync_threads()") + 1
    assert source.count("cute.gemm(") == original.count("cute.gemm(")
    # A disabled explicit coordinate is erased and must not change the device.
    implicit = _config(False)
    implicit.config.pop(KEY)
    assert bound.to_code(implicit) == original


@pytest.mark.parametrize("value", [None, 0, 1, "true", "serial"])
@pytest.mark.parametrize("repair", [False, True])
def test_strict_bool_before_repair(captured, value, repair) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        captured[0].config_spec.normalize(_config(value), _fix_invalid=repair)


def test_field_roundtrip_old_seed_prefix_and_explicit_override(captured) -> None:
    bound, _, _ = captured
    spec = bound.config_spec
    assert spec.cute_chained_scan_input_search_enabled
    assert KEY in spec._flat_fields()
    generation = spec.create_config_generation()
    for config in (_config(False), _config()):
        normalized = spec.normalized_config(config)
        assert generation.unflatten(generation.flatten(normalized)) == normalized
    assert KEY not in spec.normalized_config(_config(False))
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        with patch.object(
            CuteChainedMatmulHeuristic,
            "_with_scan_input_seed",
            side_effect=lambda env, seeds: seeds,
        ):
            old = CuteChainedMatmulHeuristic.get_seed_configs(bound.env, host.device_ir)
        new = CuteChainedMatmulHeuristic.get_seed_configs(bound.env, host.device_ir)
        assert old is not None and new is not None
        assert new[:-1] == old and new[-1].config[KEY] is True
        with patch.dict(bound.env.settings.autotune_config_overrides, {KEY: False}):
            assert (
                CuteChainedMatmulHeuristic._with_scan_input_seed(bound.env, old) == old
            )


@pytest.mark.parametrize("other", ["warp_scan", "loop", "cache", "pipeline"])
def test_alternate_ownership_is_rejected(captured, other) -> None:
    bound, plan, _ = captured
    if other == "warp_scan":
        config = _config()
        config.config["cute_chained_scan_schedule"] = "warp"
        with pytest.raises(exc.InvalidConfig, match="ordinary serial"):
            bound.config_spec.normalized_config(config)
    else:
        name = {
            "loop": "loop",
            "cache": "pointwise_cache",
            "pipeline": "preparation_pipeline",
        }[other]
        assert not plan_scan_inputs(replace(plan, **{name: object()}))


@pytest.mark.parametrize("mutation", ["dtype", "extent", "operand", "storage"])
def test_stale_typed_graph_and_allocation_are_rejected(captured, mutation) -> None:
    _, plan, _ = captured
    selected = plan.scan_inputs[0]
    if mutation == "storage":
        with pytest.raises(_UnsupportedChain, match="changed"):
            selected.check(replace(plan, warp_workspace=object()))
    elif mutation == "extent":
        with pytest.raises(_UnsupportedChain, match="changed"):
            replace(selected, extent=31).check(plan)
    elif mutation == "dtype":
        with (
            patch.dict(
                selected.frontier.meta, val=selected.frontier.meta["val"].half()
            ),
            pytest.raises(_UnsupportedChain, match="changed"),
        ):
            selected.check(plan)
    else:
        original_args = selected.frontier.args
        selected.frontier.args = (selected.scan,)
        try:
            with pytest.raises(_UnsupportedChain, match="changed"):
                selected.check(plan)
        finally:
            selected.frontier.args = original_args


@pytest.mark.parametrize("kind", ["store", "unknown", "cross_coordinate"])
def test_effects_and_cross_coordinate_frontier_cannot_be_staged(captured, kind) -> None:
    _, plan, _ = captured
    assert plan.region is not None
    selected = plan.scan_inputs[0]
    graph = plan.region.graph
    with graph.inserting_before(selected.scan):
        target = {
            "store": memory_ops.store,
            "unknown": torch.ops.aten.flip.default,
            "cross_coordinate": view_ops.subscript,
        }[kind]
        extra = graph.call_function(target, (selected.frontier, [slice(1, None)]))
        extra.meta["val"] = selected.frontier.meta["val"]
    try:
        changed = replace(plan, region=replace(plan.region, nodes=tuple(graph.nodes)))
        if kind == "cross_coordinate":
            product = next(n for n in selected.frontier.users if n is not extra)
            old = product.args
            product.args = tuple(
                extra if arg is selected.frontier else arg for arg in old
            )
            try:
                assert not plan_scan_inputs(changed)
            finally:
                product.args = old
        else:
            assert not plan_scan_inputs(changed)
    finally:
        graph.erase_node(extra)


@pytest.mark.parametrize("binding", ["output", "unpublished"])
def test_existing_read_of_output_buffer_is_not_a_completed_input(
    captured, binding
) -> None:
    _, plan, _ = captured
    selected = plan.scan_inputs[0]
    name = chained_collectives.collective_bindings(plan)[selected.scan]
    boundaries = {
        selected.scan: name,
        selected.frontier: name if binding == "output" else "unpublished",
    }
    with pytest.raises(_UnsupportedChain, match="independent original buffer"):
        bind_scan_input(
            cast("GenerateAST", None),
            plan,
            boundaries,
            selected,
            name,
            ChainedExecution(plan.threads),
        )


@pytest.mark.parametrize("extent", [13, 32, 33, 64, 129])
@pytest.mark.parametrize("threads", [32, 128])
def test_actual_typed_program_ownership_and_in_place_order(extent, threads) -> None:
    # Execute the emitted producer and original serial body, with a barrier
    # between teams, and trace each buffer read/write. This tests actual loop
    # indices/guards and publication order rather than a parallel ownership map.
    name = "prefix"
    point = PointValue(
        (),
        ast.parse("cutlass.Float32(inputs[prefix_input_position])", mode="eval").body,
        PointStore(ast.parse("prefix[prefix_input_position]", mode="eval").body),
    )
    program = CollectiveProducer(
        name,
        "serial_scan",
        extent,
        1,
        ("prefix_position",),
        ("prefix_position",),
        ("rhs = prefix[prefix_position]",),
        "scale * rhs if prefix_position < valid else cutlass.Float32(0)",
        ChainedExecution(threads),
        point,
    )
    body = emit_producer_phase(
        (program,), ast.parse("cute.arch.sync_threads()").body[0]
    )
    assert body is not None and len(body) == 4
    bits = np.resize(
        np.array(
            [
                0,
                0x80000000,
                1,
                0x80000001,
                0x3F000000,
                0x7F800000,
                0xFF800000,
                0x7F800001,
                0xFFC00001,
            ],
            dtype=np.uint32,
        ),
        extent,
    )
    inputs = bits.view(np.float32)
    values = np.empty(extent, dtype=np.float32)
    writes = np.zeros(extent, dtype=np.int32)
    reads = []

    class Buffer:
        def __getitem__(self, index):
            assert np.all(writes >= 1) and writes[index] == 1
            reads.append(index)
            return values[index]

        def __setitem__(self, index, value):
            if writes[index]:
                assert reads[-1] == index and writes[index] == 1
            values[index] = value
            writes[index] += 1

    namespace = {
        "prefix": Buffer(),
        "inputs": inputs,
        "scale": np.float32(-0.25),
        "valid": extent - 2,
        "cutlass": SimpleNamespace(
            range=lambda n, **kwargs: range(n),
            range_constexpr=range,
            Float32=np.float32,
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(sync_threads=lambda: None)),
    }
    with np.errstate(invalid="ignore", under="ignore"):
        for phase_number, phase in enumerate((body[:2], body[2:])):
            code = compile(
                ast.fix_missing_locations(
                    ast.Module(body=list(phase), type_ignores=[])
                ),
                "<typed-prefix>",
                "exec",
            )
            for thread in range(threads):
                namespace["chain_thread"] = thread
                exec(code, namespace)
            if phase_number == 0:
                np.testing.assert_array_equal(values.view(np.uint32), bits)
        expected = np.empty_like(inputs)
        accumulator = np.float32(0)
        for i, value in enumerate(inputs):
            accumulator += (
                np.float32(-0.25) * value if i < extent - 2 else np.float32(0)
            )
            expected[i] = accumulator
    assert reads == list(range(extent)) and np.all(writes == 2)
    np.testing.assert_array_equal(values.view(np.uint32), expected.view(np.uint32))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("extent", [19, 32])
@pytest.mark.parametrize("schedule", ["coalesced", "cp_async"])
def test_gpu_original_prefix_and_changing_input_frames(dtype, extent, schedule) -> None:
    torch.manual_seed(711)
    args = _args(dtype, extent, DEVICE)
    backing = args[0]._base
    assert backing is not None
    backing.fill_(811)
    for value in args:
        value.normal_(0, 0.125)
    saved = tuple(value.clone() for value in args)
    bound = _sequence._bind_isolated(args)
    original = bound.compile_config(_config(False))
    config = _config(schedule=schedule)
    if schedule == "cp_async":
        config.config["cute_native_matmul_metadata"] = True
        assert "CopyG2SOp(" in bound.to_code(config)
    staged = bound.compile_config(config)
    for iteration in range(2):
        raw, scale, left, right = args
        if iteration:
            raw.mul_(-1)
            scale.mul_(-1)
            saved = tuple(value.clone() for value in args)
        expected_original = original(*args)
        output = staged(*args)
        torch.testing.assert_close(output, expected_original, atol=0, rtol=0)
        steps = torch.zeros((3, 32), device=DEVICE, dtype=torch.float64)
        steps[:, :extent] = scale.double()[:, None] * torch.sigmoid(raw.double())
        lhs = left.double() + steps.cumsum(1)[:, :, None] * 0.03125
        reference = (
            lhs.to(dtype).double() @ right.double()
            + (lhs * lhs).sum(2)[:, :, None] * 0.125
        )
        torch.testing.assert_close(output, reference.float(), atol=0.002, rtol=0.002)
        assert torch.all(backing[:, extent:] == 811)
        torch.testing.assert_close(args, saved, atol=0, rtol=0)
