from __future__ import annotations

import ast
import dataclasses
import functools

import pytest
import torch

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    paged_value_metadata_skip_supported,
)
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(post_scale=-22, *, padded=False):
    kernel, initial = positive_zero_attention_fixture(
        query_len=4, post_scale=post_scale
    )
    args = list(initial)
    scales = args[4]
    assert isinstance(scales, torch.Tensor)
    if padded:
        page_stride = 4096
        strides = (page_stride, 4, 768, 1)
        storage = torch.empty(scales.shape[0] * page_stride + 16, dtype=scales.dtype)
        scales = storage.as_strided(scales.shape, strides, storage_offset=16)
        args[4] = scales
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, tuple(args))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            transport = plan_paged_attention_transport(bound.env, semantic)
    assert transport is not None
    return bound, transport, scales


@pytest.mark.parametrize(
    "post_scale,available", [(-22, True), (-14, True), (-13, False), (0, False)]
)
def test_metadata_availability_preserves_bf16_underflow_boundary(post_scale, available):
    bound, transport, _ = _bound(post_scale)
    assert paged_value_metadata_skip_supported(transport) is available
    facts = bound.config_spec.cute_mma_dag_facts.attention
    assert facts.paired_items
    assert facts.value_metadata_skip_available is available
    ordinary = dataclasses.replace(
        transport,
        semantic=dataclasses.replace(
            transport.semantic,
            value=dataclasses.replace(transport.semantic.value, positive_zero=False),
        ),
    )
    assert not paged_value_metadata_skip_supported(ordinary)


@pytest.mark.parametrize("padded", [False, True])
def test_normalized_metadata_words_preserve_view_offsets(padded):
    _, transport, scales = _bound(padded=padded)
    assert paged_value_metadata_skip_supported(transport)
    operand = transport.value
    # The mapped source inserts its singleton subpage axis into this rank-four
    # argument; descriptor axes describe that view, not the original argument.
    assert operand.scale_shape == (128, scales.shape[2], 1, scales.shape[0])
    for page in range(scales.shape[0]):
        for head in range(scales.shape[2]):
            for key in range(128):
                for byte in range(4):
                    original = (page, key, head, byte)
                    normalized = (key, head, 0, page)
                    expected = scales.storage_offset() + sum(
                        i * stride
                        for i, stride in zip(original, scales.stride(), strict=True)
                    )
                    actual = (
                        scales.storage_offset()
                        + byte
                        + 4
                        * sum(
                            i * stride
                            for i, stride in zip(
                                normalized, operand.scale_strides, strict=True
                            )
                        )
                    )
                    assert actual == expected
    # CPU target normalization rebases fake storage. The emitter must use this
    # named view's runtime iterator; native tests exercise its nonzero offset.
    assert operand.scale.name == "v_scale_subpages"


def test_metadata_span_proof_is_stricter_than_individual_tma_limits():
    _, transport, _ = _bound()
    high = dataclasses.replace(
        transport.value,
        scale_shape=(128, 2, 1, 91577),
        scale_strides=(1, 128, 256, 65536),
    )
    assert 4 * 63550 * high.scale_strides[3] > 2**31
    assert paged_value_metadata_skip_supported(
        dataclasses.replace(transport, value=high)
    )
    overflow = dataclasses.replace(
        high, scale_shape=(128, 2, 1, 2**32 - 1), scale_strides=(1, 128, 256, 2**38 - 4)
    )
    assert all(extent < 2**32 for extent in overflow.scale_shape)
    assert all(4 * stride < 2**40 for stride in overflow.scale_strides)
    assert not paged_value_metadata_skip_supported(
        dataclasses.replace(transport, value=overflow)
    )


@pytest.mark.parametrize(
    "padded,enabled,overlap",
    [(False, False, False), (False, True, False), (True, True, True)],
)
def test_selected_metadata_uses_the_named_normalized_view(padded, enabled, overlap):
    bound, transport, _ = _bound(padded=padded)
    config = optional_lse_config()
    config.config.update(
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_value_metadata_skip=enabled,
        cute_mma_dag_value_metadata_read_overlap=overlap,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paged_attention_paired_body"
    ]
    assert len(calls) == 1
    keywords = {node.arg: node.value for node in calls[0].keywords}
    assert ast.literal_eval(keywords["value_positive_zero"]) is True
    assert ("value_metadata_read_overlap" in keywords) is overlap
    if overlap:
        assert ast.literal_eval(keywords["value_metadata_read_overlap"]) is True
    else:
        config.config.pop("cute_mma_dag_value_metadata_read_overlap")
        with cpu_render_target(fake_device_sm_count=148):
            assert bound.to_code(config) == source
    if not enabled:
        assert "value_metadata_scales" not in keywords
        return
    view = keywords["value_metadata_scales"]
    assert isinstance(view, ast.Call) and isinstance(view.func, ast.Attribute)
    assert view.func.attr == "make_tensor"
    pointer, layout = view.args
    assert isinstance(pointer, ast.Call) and isinstance(pointer.func, ast.Attribute)
    assert pointer.func.attr == "recast_ptr"
    assert ast.unparse(pointer.args[0]) == "v_scale_subpages.iterator"
    assert ast.unparse(pointer.keywords[0].value) == "cutlass.Int32"
    assert isinstance(layout, ast.Call) and isinstance(layout.func, ast.Attribute)
    assert layout.func.attr == "make_layout"
    assert ast.literal_eval(layout.args[0]) == transport.value.scale_shape
    assert ast.literal_eval(layout.keywords[0].value) == transport.value.scale_strides
