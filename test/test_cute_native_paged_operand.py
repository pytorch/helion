from __future__ import annotations

from contextlib import nullcontext
import copy
import dataclasses
import functools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import helion
from helion import exc
from helion._compiler.cute.analysis.native_s2f6_operand import match_native_s2f6_operand
from helion._compiler.cute.analysis.paged_attention_match import _block_scaled_operand
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.fuse_native_s2f6 import fuse_native_s2f6
from helion._compiler.cute.paged_attention_transport import (
    paged_value_metadata_skip_supported,
)
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._compiler.device_ir import _get_custom_decomp_table
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_native_paged_operand import native_grouped_operand_fixture
from helion._testing.cute_native_paged_operand import native_paged_operand_fixture
from helion._testing.cute_native_s2f6 import integer_s2f6_decode
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture
from helion.autotuner.config_generation import ConfigGeneration
from helion.language.inline_asm_ops import inline_asm_elementwise

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(native=True, fused=True):
    kernel, args = (
        native_paged_operand_fixture()
        if native
        else optional_lse_fixture("per_key", True)
    )
    control = (
        nullcontext()
        if fused
        else patch(
            "helion._compiler.cute.fuse_native_s2f6.fuse_native_s2f6", return_value=0
        )
    )
    with cpu_render_target(fake_device_sm_count=148), control:
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    plan = match_paged_attention(program)
    assert plan is not None
    return bound, program, plan, args


@pytest.mark.parametrize("fused", [False, True])
def test_native_address_proof_survives_local_fusion(fused):
    bound, program, plan, args = _bound(fused=fused)
    for operand, raw, scale in zip(
        (plan.key, plan.value), args[1:3], args[3:5], strict=True
    ):
        assert operand.decode_kind == "native_s2f6_bf16"
        assert operand.post_scale_log2 == 0 and operand.positive_zero is False
        assert operand.group_size == 32 and operand.native_proof is not None
        assert operand.raw in program.accesses and operand.scale in program.accesses
        assert operand.raw.node is operand.native_proof.raw
        assert operand.scale.node is operand.native_proof.scale
        assert operand.raw.strides[-1] == operand.scale.strides[-1] == 1
        assert tuple(operand.raw.fake.stride()) == raw.stride()
        assert tuple(operand.scale.fake.stride()) == scale.stride()
        assert (
            any(n.target is inline_asm_elementwise for n in operand.native_proof.nodes)
            == fused
        )
    with bound.env:
        assert plan_paged_attention_transport(bound.env, plan) is None
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(exc.InvalidConfig, match="cute_mma_dag_pass_warps"),
    ):
        bound.to_code(optional_lse_config())
    assert not paged_value_metadata_skip_supported(SimpleNamespace(semantic=plan))


@pytest.mark.parametrize("fused", [False, True])
def test_native_generic_configs_refuse_and_default_keeps_ordinary_lowering(fused):
    bound, _, _, _ = _bound(fused=fused)
    with cpu_render_target(fake_device_sm_count=148):
        default = bound.config_spec.default_config()
        assert default.config.get("cute_mma_dag", "off") == "off"
        code = bound.to_code(default)
        assert "paged_attention_" not in code
        assert ("scaled::n2::ue8m0" in code) == fused
        for mode in ("warp", "tcgen05"):
            config = helion.Config(
                **{
                    **default.config,
                    "cute_mma_dag": mode,
                    "cute_mma_dag_pass_warps": "off",
                }
            )
            with pytest.raises(exc.InvalidConfig, match="native S2F6 operands"):
                bound.to_code(config)


@pytest.mark.parametrize("fused", [False, True])
def test_native_grouped_gemm_retains_successful_ordinary_default(fused):
    kernel, args = native_grouped_operand_fixture()
    control = (
        nullcontext()
        if fused
        else patch(
            "helion._compiler.cute.fuse_native_s2f6.fuse_native_s2f6", return_value=0
        )
    )
    with cpu_render_target(fake_device_sm_count=148), control:
        bound = cpu_bind(kernel, args)
        default = bound.config_spec.default_config()
        assert default.config.get("cute_mma_dag", "off") == "off"
        code = bound.to_code(default)
        assert ("scaled::n2::ue8m0" in code) == fused
        assert "_helion_mma_dag" not in code
        for mode in ("warp", "tcgen05"):
            config = helion.Config(
                **{
                    **default.config,
                    "cute_mma_dag": mode,
                    "cute_mma_dag_pass_warps": "off",
                }
            )
            with pytest.raises(exc.InvalidConfig, match="native S2F6 operands"):
                bound.to_code(config)


@pytest.mark.parametrize("native", [False, True])
def test_mma_mode_search_requires_an_admitted_native_transport(native):
    bound, _, _, _ = _bound(native=native, fused=False)
    with cpu_render_target(fake_device_sm_count=148):
        generation = ConfigGeneration(bound.config_spec)
        indices, sequence = generation._key_to_flat_indices["cute_mma_dag"]
        assert not sequence
        (index,) = indices
        choices = generation.flat_spec[index].search_values()
        assert choices is not None
        if native:
            assert choices == ["off"]
        else:
            assert any(mode != "off" for mode in choices)


def test_float_decode_keeps_existing_panel_and_default_kind():
    bound, _, plan, _ = _bound(native=False)
    assert plan.key.decode_kind == plan.value.decode_kind == "float_exp2"
    assert plan.key.native_proof is plan.value.native_proof is None
    with cpu_render_target(fake_device_sm_count=148):
        assert "paged_attention_partial_body(" in bound.to_code(optional_lse_config())


@pytest.mark.parametrize(
    "mutation", ["divisor", "group8", "axis", "offset", "dtype", "grouped_raw"]
)
def test_native_access_mutations_refuse(mutation):
    _, program, plan, _ = _bound(fused=True)
    accesses = {access.node: access for access in program.accesses}
    raw, scale = plan.key.raw, plan.key.scale
    if mutation == "dtype":
        accesses[scale.node] = dataclasses.replace(
            scale, fake=scale.fake.to(torch.int8)
        )
    elif mutation == "grouped_raw":
        accesses[raw.node] = dataclasses.replace(
            raw,
            subscripts=(
                *raw.subscripts[:-1],
                dataclasses.replace(raw.subscripts[-1], divisor=32),
            ),
        )
    else:
        subscripts = list(scale.subscripts)
        index = -1 if mutation in ("divisor", "group8") else 0
        changes = (
            {"offset": 1}
            if mutation == "offset"
            else {
                "divisor": None
                if mutation == "divisor"
                else 8
                if mutation == "group8"
                else 32
            }
        )
        subscripts[index] = dataclasses.replace(subscripts[index], **changes)
        accesses[scale.node] = dataclasses.replace(scale, subscripts=tuple(subscripts))
    assert (
        _block_scaled_operand(
            plan.key.value, accesses, torch.bfloat16, resolve=program.resolve
        )
        is None
    )


@pytest.mark.parametrize("mutation", ["nan_bits", "effect"])
def test_mma_admission_does_not_accept_other_assembly(mutation):
    bound, _, plan, _ = _bound(fused=True)
    asm = next(
        n for n in plan.key.native_proof.nodes if n.target is inline_asm_elementwise
    )
    arguments = list(asm.args)
    if mutation == "nan_bits":
        arguments[0] = arguments[0].replace("0x7fc0", "0x7fc1")
    else:
        arguments[4] = False
    original_args = asm.args
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        assert match_mma_dag(bound.host_function.device_ir.graphs) is not None
        asm.args = tuple(arguments)
        try:
            assert match_mma_dag(bound.host_function.device_ir.graphs) is None
        finally:
            asm.args = original_args
        assert match_mma_dag(bound.host_function.device_ir.graphs) is not None


@pytest.mark.parametrize(
    "target,constant",
    [(torch.ops.aten.mul.Tensor, 2.0**-6), (torch.ops.aten.add.Tensor, 0.0)],
)
def test_native_postscale_and_positive_zero_stay_unadmitted(target, constant):
    _, program, plan, _ = _bound(fused=True)
    value = plan.key.value
    with value.graph.inserting_after(value):
        altered = value.graph.call_function(target, (value, constant))
        altered.meta = value.meta.copy()
    try:
        assert (
            _block_scaled_operand(
                altered,
                {access.node: access for access in program.accesses},
                torch.bfloat16,
                allow_positive_zero=True,
                resolve=program.resolve,
            )
            is None
        )
    finally:
        value.graph.erase_node(altered)


@functools.cache
def _fused_graph():
    graph = make_fx(
        integer_s2f6_decode,
        decomposition_table=_get_custom_decomp_table(),
        tracing_mode="fake",
    )(torch.empty(8, 32, dtype=torch.int8), torch.empty(8, 1, dtype=torch.uint8)).graph
    assert fuse_native_s2f6(graph, (10, 0)) == 1
    return graph


@pytest.mark.parametrize(
    "mutation", [None, "nan_bits", "pure", "pack", "cast", "broadcast"]
)
def test_exact_fused_contract_and_broadcast(mutation):
    graph = copy.deepcopy(_fused_graph())
    value = next(n for n in graph.nodes if n.op == "output").args[0]
    asm = next(n for n in graph.nodes if n.target is inline_asm_elementwise)
    args = list(asm.args)
    if mutation == "nan_bits":
        args[0] = args[0].replace("0x7fc0", "0x7fc1")
    if mutation == "pure":
        args[4] = False
    if mutation == "pack":
        args[5] = 2
    if mutation == "cast":
        cast = args[2][0]
        cast.args = (cast.args[0], torch.uint32)
    if mutation == "broadcast":
        value.meta["val"] = torch.empty(8, 1, dtype=torch.bfloat16)
    asm.args = tuple(args)
    proof = match_native_s2f6_operand(value)
    assert (proof is not None) == (mutation is None)
    if proof is not None:
        assert {value, asm, proof.raw_int32, proof.scale_int32} <= proof.nodes
