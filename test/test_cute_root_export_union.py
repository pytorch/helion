from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.chained_root_exports import classify_root_exports
import helion.language as hl

if TYPE_CHECKING:
    from torch.fx import Node


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _independent_outputs(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    raw: torch.Tensor,
    first_dtype: hl.constexpr,
    independent_scan: hl.constexpr,
):
    first_type: Any = first_dtype
    batch = a.size(0)
    rows = 16
    width, columns = hl.specialize(a.size(2)), hl.specialize(b.size(2))
    first = torch.empty((batch, rows, columns), dtype=first_type, device=a.device)
    second = torch.empty((batch, rows, columns), dtype=c.dtype, device=a.device)
    side = torch.empty((batch, rows), dtype=torch.float32, device=a.device)
    for tile in hl.tile(batch, block_size=1):
        row = tile.id
        idx = hl.arange(rows)
        valid = idx < a.size(1)
        left = hl.load(a, [row, idx, hl.arange(width)], extra_mask=valid[:, None])
        right = hl.load(b, [row, hl.arange(width), hl.arange(columns)])
        other = hl.load(c, [row, hl.arange(width), hl.arange(columns)])
        values = hl.load(raw, [row, idx], extra_mask=valid).float()
        prefix = hl.cumsum(values, dim=0)
        first[row, :, :] = (hl.dot(left, right) * torch.exp2(prefix[:, None])).to(
            first_type
        )
        second[row, :, :] = hl.dot(left, other).to(c.dtype)
        side[row, :] = (
            hl.cumsum(values * 2, dim=0) if independent_scan else torch.exp2(prefix)
        )
    return first, second, side


def _args():
    return tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in (
            (3, 11, 32),
            (3, 32, 32),
            (3, 32, 32),
            (3, 11),
        )
    )


@pytest.mark.parametrize("first_dtype", [torch.float16, torch.bfloat16])
def test_independent_fresh_matrix_outputs_register_common_contraction(
    first_dtype,
) -> None:
    with _cpu_codegen():
        bound = _independent_outputs._bind_isolated((*_args(), first_dtype, False))
        assert bound.config_spec.cute_chained_matmul_search_enabled
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            proofs = [
                classify_root_exports(tuple(graph.graph.nodes))
                for graph in bound.host_function.device_ir.graphs
            ]
        admitted = [proof for proof in proofs if proof is not None]
        assert len(admitted) == 1 and len(admitted[0].stores) == 3
        assert admitted[0].primary is admitted[0].stores[0].store


def test_union_requires_every_contraction_and_retains_alias_guard() -> None:
    with _cpu_codegen():
        bound = _independent_outputs._bind_isolated((*_args(), torch.bfloat16, False))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            nodes = next(
                tuple(graph.graph.nodes)
                for graph in bound.host_function.device_ir.graphs
                if classify_root_exports(tuple(graph.graph.nodes)) is not None
            )
            proof = classify_root_exports(nodes)
            assert proof is not None
            second = proof.stores[1].store
            assert (
                classify_root_exports(
                    tuple(node for node in nodes if node is not second)
                )
                is None
            )
            first_target = cast("Node", proof.primary.args[0])
            second_target = cast("Node", second.args[0])
            with patch.dict(second_target.meta, {"val": first_target.meta["val"]}):
                assert classify_root_exports(nodes) is None
            # Complete fresh ownership is still required for every matrix output.
            indices = list(cast("list[object]", second.args[1]))
            indices[-1] = slice(0, 8)
            with patch.object(
                second, "_args", (second.args[0], indices, *second.args[2:])
            ):
                assert classify_root_exports(nodes) is None


def test_side_scan_must_belong_to_matrix_output_ancestry() -> None:
    with _cpu_codegen():
        bound = _independent_outputs._bind_isolated((*_args(), torch.bfloat16, True))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert all(
                classify_root_exports(tuple(graph.graph.nodes)) is None
                for graph in bound.host_function.device_ir.graphs
            )
