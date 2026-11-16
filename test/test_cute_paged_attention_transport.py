from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing import default_cute_mma_support
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention

if TYPE_CHECKING:
    from helion._compiler.cute.paged_attention_transport import PagedAttentionTransport
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


def _plan(
    width: int = 64,
    *,
    offset: int = 0,
    page_pitch: int = 33792,
    interleaved_scales: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    capability: tuple[int, int] = (10, 0),
) -> PagedAttentionTransport | None:
    subpages = 128 // width
    shape = (7, subpages, width, 2, 128)
    raw_stride = (page_pitch, width * 256, 256, 128, 1)
    raw = [torch.empty(7 * page_pitch + offset, dtype=torch.int8) for _ in range(2)]
    operands = [x.as_strided(shape, raw_stride, offset) for x in raw]
    scale_stride = (
        (page_pitch, width * 8, 8, 4, 1)
        if interleaved_scales
        else (page_pitch, width * 4, 4, 512, 1)
    )
    scales = [
        x.view(torch.uint8).as_strided((*shape[:-1], 4), scale_stride, 32768 + offset)
        for x in raw
    ]
    args = (
        torch.empty((6, 40, 128), dtype=dtype),
        *operands,
        *scales,
        torch.empty((3, 17), dtype=torch.int32),
        torch.empty((3,), dtype=torch.int32),
        5,
        8,
        31,
    )
    with cpu_render_target(
        fake_device_sm_count=148, mma_support=default_cute_mma_support()
    ):
        bound = cpu_bind(cast("Kernel[object]", paged_online_attention), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            bound.config_spec.target_device_capability = capability
            return plan_paged_attention_transport(bound.env, semantic)


@pytest.mark.parametrize("width,page_pitch", [(64, 33792), (128, 34048)])
def test_paged_descriptor_preserves_named_views(width: int, page_pitch: int) -> None:
    plan = _plan(width, page_pitch=page_pitch)
    assert plan is not None
    assert plan.key.raw.name == "k"
    assert plan.key.scale.name == "k_scale"
    assert plan.value.raw.name == "v"
    assert plan.value.scale.name == "v_scale"
    assert plan.key.raw_shape == (128, width, 2, 128 // width, 7)
    assert plan.key.raw_strides == (1, 256, 128, width * 256, page_pitch)
    assert plan.key.scale_shape == (width, 2, 128 // width, 7)
    assert plan.key.scale_strides == (1, 128, width, page_pitch // 4)
    assert plan.items == 6 and plan.query_rows == 40


@pytest.mark.parametrize(
    "offset,page_pitch,interleaved_scales,dtype",
    [
        (1, 33792, False, torch.bfloat16),
        (0, 33796, False, torch.bfloat16),
        (0, 33792, True, torch.bfloat16),
        (0, 33792, False, torch.float16),
    ],
)
def test_paged_transport_declines_unproved_layouts(
    offset: int, page_pitch: int, interleaved_scales: bool, dtype: torch.dtype
) -> None:
    assert (
        _plan(
            offset=offset,
            page_pitch=page_pitch,
            interleaved_scales=interleaved_scales,
            dtype=dtype,
        )
        is None
    )


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 1), (9, 0), (12, 0)])
def test_paged_transport_requires_a_native_decode_target(
    capability: tuple[int, int],
) -> None:
    assert (_plan(capability=capability) is not None) == (
        capability in ((10, 0), (10, 3))
    )
