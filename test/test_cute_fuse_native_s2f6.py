"""CPU-only selection and graph preservation tests for native S2F6 fusion."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

import helion
from helion._compiler.cute.fuse_native_s2f6 import fuse_native_s2f6
from helion._compiler.cute.pure_lane_packets import is_pure_scalar_assembly
from helion._compiler.device_ir import _get_custom_decomp_table
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_native_s2f6 import decode_domain
from helion._testing.cute_native_s2f6 import decode_kernel
from helion._testing.cute_native_s2f6 import grouped_decode_kernel
from helion._testing.cute_native_s2f6 import grouped_decode_sum
from helion._testing.cute_native_s2f6 import integer_s2f6_decode
from helion.language.inline_asm_ops import inline_asm_elementwise

pytestmark = pytest.mark.cpu_only


@forbid_cuda_work()
def test_grouped_decode_renders_inside_rolled_reduction():
    args = (
        torch.empty((3, 257), dtype=torch.int8),
        torch.empty((3, 1), dtype=torch.uint8),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(grouped_decode_sum, args)
        code = bound.to_code(
            helion.Config(block_sizes=[4], num_threads=[4, 32], reduction_loops=[64])
        )
    assert "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2" in code


@pytest.fixture(scope="module")
@forbid_cuda_work()
def decomposed():
    return make_fx(
        integer_s2f6_decode,
        decomposition_table=_get_custom_decomp_table(),
        tracing_mode="fake",
    )(torch.empty(64, dtype=torch.int8), torch.empty(64, dtype=torch.uint8))


@pytest.mark.parametrize(
    "capability", [(10, 0), (10, 3), (9, 0), (10, 1), (12, 0), None]
)
@forbid_cuda_work()
def test_target_selection_and_exact_operands(decomposed, capability):
    graph = copy.deepcopy(decomposed).graph
    original = str(graph)
    count = fuse_native_s2f6(graph, capability)
    if capability not in ((10, 0), (10, 3)):
        assert count == 0 and str(graph) == original
        return
    assert count == 1 and fuse_native_s2f6(graph, capability) == 0
    native = next(n for n in graph.nodes if n.target is inline_asm_elementwise)
    raw, scale = (n for n in graph.nodes if n.op == "placeholder")
    asm, constraints, operands, dtype, pure, pack = native.args
    assert constraints == "=h,r,r" and dtype is torch.int16 and pure and pack == 1
    assert [operand.args for operand in operands] == [
        (raw, torch.int32),
        (scale, torch.int32),
    ]
    assert "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2" in asm
    assert "setp.eq.u32 exceptional, $2, 255;" in asm
    assert "selp.b32 decoded, 0x7fc0, decoded, exceptional;" in asm
    assert is_pure_scalar_assembly(asm, constraints, pure)
    view = next(n for n in graph.nodes if n.target is torch.ops.aten.view.dtype)
    assert view.args == (native, torch.bfloat16)
    assert native.meta["val"].dtype is torch.int16


@forbid_cuda_work()
def test_live_integer_intermediate_is_preserved(decomposed):
    graph = copy.deepcopy(decomposed).graph
    magnitude = next(n for n in graph.nodes if n.target is torch.ops.aten.abs.default)
    output = next(n for n in graph.nodes if n.op == "output")
    output.args = ((output.args[0], magnitude),)
    original_args = magnitude.args
    assert fuse_native_s2f6(graph, (10, 0)) == 1
    assert magnitude in graph.nodes and magnitude.args == original_args
    assert output.args[0][1] is magnitude


@pytest.mark.parametrize(
    "mutation", ["changed_bias", "lifted_constant", "shared_storage"]
)
@forbid_cuda_work()
def test_unproved_recipe_or_observable_storage_is_unchanged(decomposed, mutation):
    graph = copy.deepcopy(decomposed).graph
    if mutation == "changed_bias":
        node = next(
            n
            for n in graph.nodes
            if n.target is torch.ops.aten.sub.Tensor and n.args[1] == 6
        )
        node.args = (node.args[0], 7)
    elif mutation == "lifted_constant":
        node = next(
            n
            for n in graph.nodes
            if n.target is torch.ops.aten.scalar_tensor.default and n.args[0] == 0x7FC0
        )
        with graph.inserting_before(node):
            lifted = graph.placeholder("unproved_constant")
        lifted.meta = node.meta.copy()
        node.replace_all_uses_with(lifted)
    else:
        node = next(n for n in graph.nodes if n.target is torch.ops.aten.view.dtype)
        output = next(n for n in graph.nodes if n.op == "output")
        output.args = ((output.args[0], node.args[0]),)
    original = str(graph)
    assert fuse_native_s2f6(graph, (10, 0)) == 0
    assert str(graph) == original


@forbid_cuda_work()
def test_normal_pipeline_emits_native_decode_without_cuda():
    args = (torch.empty(64, dtype=torch.int8), torch.empty(64, dtype=torch.uint8))
    with cpu_render_target():
        bound = cpu_bind(decode_kernel, args)
        assert bound.config_spec.pointwise_facts
        code = bound.to_code(helion.Config(block_sizes=[64]))
    assert "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2" in code
    assert "selp.b32 decoded, 0x7fc0, decoded, exceptional;" in code


@forbid_cuda_work()
def test_portable_recipe_matches_all_reference_bits():
    raw, scale, expected = decode_domain()
    actual = integer_s2f6_decode(raw, scale)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@forbid_cuda_work()
def test_zero_initializer_cannot_widen_asm_broadcast(decomposed):
    module = copy.deepcopy(decomposed)
    zero = next(
        n for n in module.graph.nodes if n.target is torch.ops.aten.full.default
    )
    zero.args = ([2, 64], zero.args[1])
    module.recompile()
    FakeTensorProp(module).propagate(
        torch.empty(64, dtype=torch.int8), torch.empty(64, dtype=torch.uint8)
    )
    result = next(
        n for n in module.graph.nodes if n.target is torch.ops.aten.view.dtype
    )
    assert result.meta["val"].shape == (2, 64)
    before = str(module.graph)
    assert fuse_native_s2f6(module.graph, (10, 0)) == 0
    assert str(module.graph) == before


@forbid_cuda_work()
def test_matched_casts_keep_input_aliases(decomposed):
    graph = copy.deepcopy(decomposed).graph
    raw = next(n for n in graph.nodes if n.op == "placeholder")
    cast = next(
        n for n in raw.users if n.target is torch.ops.prims.convert_element_type.default
    )
    with graph.inserting_before(cast):
        alias = graph.call_function(torch.ops.aten.alias.default, (raw,))
    alias.meta = raw.meta.copy()
    cast.args = (alias, torch.int32)
    assert fuse_native_s2f6(graph, (10, 0)) == 1
    native = next(n for n in graph.nodes if n.target is inline_asm_elementwise)
    assert native.args[2][0] is cast
    assert cast.args[0] is alias


@forbid_cuda_work()
def test_grouped_scale_broadcast_keeps_pointwise_codegen():
    args = (
        torch.empty((32, 128), dtype=torch.int8),
        torch.empty((32, 1), dtype=torch.uint8),
    )
    with cpu_render_target():
        bound = cpu_bind(grouped_decode_kernel, args)
        assert bound.config_spec.pointwise_facts
        code = bound.to_code(helion.Config(block_sizes=[4, 32], num_threads=[4, 32]))
    assert "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2" in code


@forbid_cuda_work()
def test_selection_consumer_keeps_portable_recipe():
    def selected(raw, scale):
        decoded = integer_s2f6_decode(raw, scale).reshape(4, 16)
        return torch.topk(decoded.float(), 3, dim=-1)

    module = make_fx(
        selected, decomposition_table=_get_custom_decomp_table(), tracing_mode="fake"
    )(torch.empty(64, dtype=torch.int8), torch.empty(64, dtype=torch.uint8))
    before = str(module.graph)
    assert fuse_native_s2f6(module.graph, (10, 0)) == 0
    assert str(module.graph) == before
