from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import pytest
import sympy
import torch

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_attention import paged_online_attention

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


def _program(
    dtype=torch.bfloat16, width=64, dim=128, window=15, length_dtype=torch.int32
):
    # Distinct K/V storage plus padded page strides and a nonzero scale offset
    # catch accidental dense-view or storage-origin assumptions.
    shape = (13, 2, width, 2, dim)
    page_stride = 2 * width * 2 * dim + width * dim // 2 + 256
    strides = (page_stride, width * 2 * dim, 2 * dim, dim, 1)
    raw_k = torch.empty(13 * page_stride, dtype=torch.int8)
    raw_v = torch.empty_like(raw_k)
    k = raw_k.as_strided(shape, strides)
    v = raw_v.as_strided(shape, strides)
    scale_shape = (13, 2, width, 2, dim // 8)
    scale_strides = (page_stride, width * dim // 4, dim // 4, dim // 8, 1)
    sk = raw_k.view(torch.uint8).as_strided(
        scale_shape, scale_strides, 2 * width * 2 * dim
    )
    sv = raw_v.view(torch.uint8).as_strided(
        scale_shape, scale_strides, 2 * width * 2 * dim
    )
    args = (
        torch.empty(6, 7, dim, dtype=dtype),
        k,
        v,
        sk,
        sv,
        torch.empty(3, 17, dtype=torch.int32),
        torch.empty(3, dtype=length_dtype),
        3,
        2,
        window,
    )
    with cpu_render_target():
        bound = cpu_bind(cast("Kernel[object]", paged_online_attention), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return program


@pytest.fixture(scope="module")
def program():
    return _program()


@pytest.mark.parametrize(
    "dtype,width,dim,window,length_dtype",
    [
        (torch.bfloat16, 64, 128, 15, torch.int32),
        (torch.bfloat16, 128, 128, 0, torch.int32),
        (torch.float16, 64, 32, -1, torch.int64),
        (torch.bfloat16, 64, 32, 31, torch.int64),
    ],
)
def test_paged_semantics_keep_source_layout(dtype, width, dim, window, length_dtype):
    plan = match_paged_attention(_program(dtype, width, dim, window, length_dtype))
    assert plan is not None
    assert plan.io_dtype == dtype
    assert plan.coordinates.width == width
    assert plan.coordinates.group == 2
    assert plan.coordinates.query_len == 3
    assert plan.coordinates.length_base_dtype == length_dtype
    # PyTorch's 0-D int64 length base is wrapped to the row vector's
    # int32 dtype at the add; preserve both widths, not just the input dtype.
    assert plan.coordinates.query_dtype == torch.int32
    assert plan.key.group_size == plan.value.group_size == 8
    assert plan.key.raw.name != plan.value.raw.name
    assert plan.key.raw.strides[0] == 2 * width * 2 * dim + width * dim // 2 + 256
    assert plan.key.scale.name == "k_scale"
    assert plan.value.scale.name == "v_scale"
    assert plan.page_table.strides == (17, 1)
    assert plan.page_dimension == 0
    assert plan.mask.window_left == (None if window < 0 else window)
    assert plan.program.resolve(plan.mask.query_position) is not plan.page_table.node
    assert plan.loop.begin is plan.program.begin
    assert plan.loop.end is plan.program.end


def _nodes(program):
    return {
        node.name: node
        for graph in (program.loop_node.graph, program.dots[0].node.graph)
        for node in graph.nodes
    }


# These are distinct semantic obligations, not a Cartesian product of knobs.
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_scale_bias",
        "wrong_decode_dtype",
        "wrong_max_fill",
        "wrong_max_axis",
        "wrong_alpha",
        "wrong_sum_fill",
        "wrong_sum_axis",
        "wrong_empty_max",
        "wrong_empty_denominator",
        "wrong_lse",
        "wrong_initial_maximum",
        "wrong_initial_denominator",
        "wrong_page_map",
        "extra_store",
        "wrong_iteration_stride",
        "wrong_iota_start",
        "zero_query_group",
        "wrong_query_axis",
        "alpha_decode",
        "alpha_denominator",
        "alpha_rescale",
        "alpha_key_position",
        "alpha_query_base",
        "alpha_lse",
        "rounded_score_constant",
        "bool_decode_bias",
        "malformed_window_predicate",
        "lse_inside_loop",
    ],
)
def test_rejects_changed_attention_semantics(program, mutation):
    candidate = copy_paged_attention_program(program)
    plan = match_paged_attention(candidate)
    assert plan is not None
    nodes = _nodes(candidate)
    if mutation.startswith("alpha_"):
        if mutation == "alpha_decode":
            product = cast("torch.fx.Node", plan.key.value.args[0])
            exponent = cast("torch.fx.Node", product.args[1])
            target = cast("torch.fx.Node", exponent.args[0])
        elif mutation == "alpha_denominator":
            target = plan.denominator.output
        elif mutation == "alpha_rescale":
            target = cast("torch.fx.Node", nodes["alpha"].args[0])
        elif mutation == "alpha_key_position":
            target = plan.mask.key_position
        elif mutation == "alpha_query_base":
            target = cast(
                "torch.fx.Node", candidate.resolve(plan.mask.query_position).args[0]
            )
        else:
            target = candidate.vec_stores[0].value
        target.kwargs = {**target.kwargs, "alpha": 2}
    elif mutation in ("rounded_score_constant", "bool_decode_bias"):
        if mutation == "rounded_score_constant":
            target = cast("torch.fx.Node", plan.score.args[1])
            literal, dtype = 0.1, torch.bfloat16
        else:
            product = cast("torch.fx.Node", plan.key.value.args[0])
            exponent = cast("torch.fx.Node", product.args[1])
            target = cast("torch.fx.Node", exponent.args[0])
            literal, dtype = 127, torch.bool
        with target.graph.inserting_before(target):
            constant = target.graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                (literal,),
                {"dtype": dtype, "device": "cpu"},
            )
        constant.meta["val"] = torch.scalar_tensor(literal, dtype=dtype)
        target.args = (target.args[0], constant)
    elif mutation == "wrong_scale_bias":
        product = cast("torch.fx.Node", plan.key.value.args[0])
        exponent = cast("torch.fx.Node", product.args[1])
        bias = cast("torch.fx.Node", exponent.args[0])
        bias.args = (bias.args[0], 126.0)
    elif mutation == "wrong_decode_dtype":
        plan.key.value.args = (plan.key.value.args[0], torch.float32)
    elif mutation == "wrong_max_fill":
        reduced = next(
            n for n in nodes.values() if n.target is torch.ops.aten.amax.default
        )
        reduced.args[0].args = (plan.score, 0)
    elif mutation == "wrong_max_axis":
        reduced = next(
            n for n in nodes.values() if n.target is torch.ops.aten.amax.default
        )
        reduced.args = (reduced.args[0], [0])
    elif mutation == "wrong_alpha":
        alpha = next(n for n in nodes.values() if n.name == "alpha")
        alpha.args[0].args = tuple(reversed(alpha.args[0].args))
    elif mutation == "wrong_sum_fill":
        reduced = next(
            n for n in nodes.values() if n.target is torch.ops.aten.sum.dim_IntList
        )
        reduced.args[0].args = (plan.probability, 1)
    elif mutation == "wrong_sum_axis":
        reduced = next(
            n for n in nodes.values() if n.target is torch.ops.aten.sum.dim_IntList
        )
        reduced.args = (reduced.args[0], [0])
    elif mutation == "wrong_empty_max":
        safe = nodes["safe_maximum"]
        safe.args = (safe.args[0], safe.args[2], safe.args[1])
    elif mutation == "wrong_empty_denominator":
        safe = nodes["safe_denominator"]
        safe.args = (safe.args[0], safe.args[2], safe.args[1])
    elif mutation == "wrong_lse":
        next(
            n for n in nodes.values() if n.target is torch.ops.aten.log.default
        ).target = torch.ops.aten.log2.default
    elif mutation in ("wrong_initial_maximum", "wrong_initial_denominator"):
        index = 0 if mutation == "wrong_initial_maximum" else 1
        carries = list(candidate.loops[0].vec_carries)
        carries[index] = dataclasses.replace(carries[index], init_fill=1.0)
        candidate = dataclasses.replace(
            candidate,
            loops=(
                dataclasses.replace(candidate.loops[0], vec_carries=tuple(carries)),
            ),
        )
    elif mutation == "wrong_page_map":
        # The public plan is backed by the typed source proof. Change that proof
        # to model V using a different physical page expression.
        value_dot = candidate.dots[1]
        access = value_dot.b.load
        assert access is not None
        subscripts = list(access.subscripts)
        subscripts[0] = dataclasses.replace(
            subscripts[0], expr=sympy.Add(plan.page_symbol, 1)
        )
        access = dataclasses.replace(access, subscripts=tuple(subscripts))
        changed = dataclasses.replace(
            value_dot, b=dataclasses.replace(value_dot.b, load=access)
        )
        candidate = dataclasses.replace(candidate, dots=(candidate.dots[0], changed))
    elif mutation in ("wrong_iteration_stride", "wrong_iota_start"):
        key_position = plan.mask.key_position
        iota = cast("torch.fx.Node", key_position.args[0])
        offset = cast("torch.fx.Node", key_position.args[1])
        if mutation == "wrong_iteration_stride":
            offset.args = (offset.args[0], 2 * plan.coordinates.width)
        else:
            iota.kwargs = {**iota.kwargs, "start": 1}
    elif mutation in ("zero_query_group", "wrong_query_axis"):
        position = candidate.resolve(plan.mask.query_position)
        divided = cast("torch.fx.Node", position.args[1])
        row = cast("torch.fx.Node", divided.args[0])
        if mutation == "zero_query_group":
            divided.args = (row, 0)
        else:
            indices = cast("list[torch.fx.Node]", plan.query.node.args[1])
            row.args = (indices[0],)
    elif mutation == "extra_store":
        candidate = dataclasses.replace(
            candidate, stores=(*candidate.stores, candidate.stores[0])
        )
    elif mutation == "malformed_window_predicate":
        plan.mask.predicate.args = plan.mask.predicate.args[:1]
    elif mutation == "lse_inside_loop":
        candidate = dataclasses.replace(
            candidate,
            vec_stores=(dataclasses.replace(candidate.vec_stores[0], loop=0),),
        )
    assert match_paged_attention(candidate) is None


def test_explicit_unit_alpha_is_preserved(program):
    candidate = copy_paged_attention_program(program)
    for node in _nodes(candidate).values():
        if node.target in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor):
            node.kwargs = {**node.kwargs, "alpha": 1}
    assert match_paged_attention(candidate) is not None
