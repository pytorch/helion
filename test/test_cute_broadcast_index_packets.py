"""Packet admission preserves broadcasts, integer coordinates and masks."""

from __future__ import annotations

import ast
import operator

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.memory_ops import _cute_lane_uniform_predicate
from helion._compiler.cute.memory_ops import _CuteLaneIndexFacts
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def gathered_sum(
    x: torch.Tensor, ids: torch.Tensor, limit: hl.constexpr, stride: hl.constexpr
) -> torch.Tensor:
    output = torch.empty((ids.numel(),), dtype=torch.float32, device=x.device)
    for rows in hl.tile(ids.numel()):
        source = ids[rows].to(torch.int64)
        total = hl.zeros([rows], dtype=torch.float32)
        for features in hl.tile(x.shape[1]):
            coordinate = features.index.to(torch.int64)
            if stride != 1:
                coordinate = coordinate // stride
            values = hl.load(
                x,
                [source[:, None], coordinate[None, :]],
                extra_mask=(source[:, None] >= 0)
                & (source[:, None] < x.shape[0])
                & (coordinate[None, :] < limit),
            )
            total = total + values.float().sum(dim=1)
        output[rows] = total
    return output


def packet_config(width: int) -> helion.Config:
    return helion.Config(
        block_sizes=[2, 128],
        num_threads=[2, 32],
        cute_vector_widths=[1, width],
        cute_lane_layouts="strided",
    )


def _generate(dtype: torch.dtype, width: int, limit: int, stride: int = 1) -> str:
    args = (
        torch.empty((12, 256), dtype=dtype),
        torch.empty((7,), dtype=torch.int32),
        limit,
        stride,
    )
    with _forbid_cuda_work(), _mock_cuda_unavailable():
        return _cpu_bind(gathered_sum, args).to_code(packet_config(width))


@pytest.mark.parametrize(("dtype", "width"), [(torch.float32, 4), (torch.bfloat16, 4)])
def test_broadcasted_gather_uses_guarded_packets(
    dtype: torch.dtype, width: int
) -> None:
    code = _generate(dtype, width, 128)
    packets = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_tile_unroll_vec_")
    ]
    assert len(packets) == 1, code
    pointer = ast.unparse(packets[0].value)
    assert "else x.iterator" in pointer, pointer
    assert "lane_base_" in pointer, pointer
    assert "operator.ge" in pointer and "operator.lt" in pointer, pointer
    assert "vec_lane_" not in pointer, pointer


@pytest.mark.parametrize(("limit", "stride"), [(127, 1), (128, 2)])
def test_partial_packet_mask_or_gathered_lane_stays_scalar(
    limit: int, stride: int
) -> None:
    code = _generate(torch.float32, 4, limit, stride)
    assert "_tile_unroll_vec_" not in code, code
    assert ").load() if " in code, code


@pytest.mark.parametrize(
    ("predicate", "uniform"),
    [
        ("index - offset < 128", True),
        ("128 > index - offset", True),
        ("index - offset >= 128", True),
        ("index - offset <= 128", False),
        ("index - offset > 128", False),
        ("index - unknown < 128", False),
        ("index - offset < 127", False),
        ("cutlass.Int32(index + unknown) < index", False),
    ],
)
def test_relative_bounds_require_proven_packet_alignment(
    predicate: str, uniform: bool
) -> None:
    lane = _CuteLaneIndexFacts("index", "base", "vi", 4, True, frozenset(("offset",)))
    assert (
        _cute_lane_uniform_predicate(ast.parse(predicate, mode="eval").body, lane)
        == uniform
    )
    if uniform:
        for offset in (0, 64, 128):
            for base in range(offset, offset + 256, 4):
                results = {
                    eval(
                        predicate,
                        {"operator": operator, "index": base + vi, "offset": offset},
                    )
                    for vi in range(4)
                }
                assert len(results) == 1
