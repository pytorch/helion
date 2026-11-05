"""Private source certificates are recomputed, not retained by re-lowering."""

from __future__ import annotations

import ast

import pytest
import torch

from test.test_cute_bool_aux_native import _args
from test.test_cute_bool_aux_native import _external_bool_epilogue

from helion._compiler.device_ir import lower_to_device_ir
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.exc import NoDeviceLoopsInKernel


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _both_subjects():
    arguments = list(_args())
    arguments[7] = torch.empty((128, 128), dtype=torch.bfloat16)
    bound = cpu_bind(_external_bool_epilogue, tuple(arguments))
    state = bound.config_spec._cute_tcgen05_config
    assert state._readonly_bool_aux_seed_subject is not None
    assert state.mixed_output_store_plan is not None
    return bound


@pytest.mark.parametrize("change", ("unproved_bool", "removed_mma"))
def test_real_relowering_same_configspec_clears_both_previous_source_certificates(
    change,
):
    bound = _both_subjects()
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    old_subject = state._readonly_bool_aux_seed_subject
    assert old_subject is not None
    before = dict(bound.env.bound_runtime_input_specialization_results)
    rewritten = 0
    # Rewrite only this test's HostFunction AST, keeping its real typed shapes
    # and the exact same CompileEnvironment/ConfigSpec through a second lower.
    # Original kernel functions and production test expectations are untouched.
    for statement in bound.host_function.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if (
                change == "unproved_bool"
                and node.func.attr == "to"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mask"
            ):
                node.func.attr = "sin"
                node.args = []
                rewritten += 1
            elif change == "removed_mma" and node.func.attr == "addmm":
                node.func.attr = "add"
                node.args = node.args[:1] + node.args[:1]
                rewritten += 1
    assert rewritten
    with bound.env:
        new_ir = lower_to_device_ir(bound.host_function)
    assert bound.env.config_spec is spec
    assert spec._cute_tcgen05_config is state
    assert state._readonly_bool_aux_seed_subject is None
    assert state.mixed_output_store_plan is None
    assert state._multi_output_seed_configs() == []
    assert before == bound.env.bound_runtime_input_specialization_results
    matmuls = [
        node
        for graph in new_ir.graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    ]
    if change == "unproved_bool":
        assert matmuls and all(node is not old_subject[0] for node in matmuls)
    else:
        assert not matmuls


def test_declined_relowering_resets_private_certificates_before_no_loop_refusal():
    bound = _both_subjects()
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    original = bound.host_function.body[:]
    bound.host_function.body.clear()
    try:
        with bound.env, pytest.raises(NoDeviceLoopsInKernel):
            lower_to_device_ir(bound.host_function)
    finally:
        bound.host_function.body[:] = original
    assert bound.env.config_spec is spec
    assert state._readonly_bool_aux_seed_subject is None
    assert state.mixed_output_store_plan is None


def test_postconstructor_codegen_does_not_clear_current_source_certificates():
    import helion

    bound = _both_subjects()
    state = bound.config_spec._cute_tcgen05_config
    bool_subject = state._readonly_bool_aux_seed_subject
    mixed_subject = state.mixed_output_store_plan
    for _ in range(2):
        source = bound.to_code(
            helion.Config(block_sizes=[128, 64, 64], pid_type="flat")
        )
        assert "tcgen05_ab_tma" in source and "tcgen05_fp32_tmem_desc" in source
        assert "!= cutlass.Uint8(0)" in source
        assert state._readonly_bool_aux_seed_subject is bool_subject
        assert state.mixed_output_store_plan is mixed_subject
