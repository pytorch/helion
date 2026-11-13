from __future__ import annotations

import pytest
import torch

from helion._compiler.cute.selection.selection_coarse import trace_coarse_rank_selection

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_selection_trace_geometry_sequence(recovery):
    dynamic_shapes = torch._dynamo.config.automatic_dynamic_shapes
    # Exercise changing register, subgroup, and selected-prefix widths in one
    # process. The fourth trace used to generalize the exact and payload branch
    # outputs to incompatible symbolic strides after the preceding geometries.
    for registers, k, bits, vector, lanes, payload in [
        (4, 4, 4, 1, 4, False),
        (32, 4, 5, 1, 1, True),
        (16, 8, 7, 4, 8, True),
        (16, 16, 7, 4, 8, True),
        (16, 32, 7, 4, 8, True),
    ]:
        graph = trace_coarse_rank_selection(
            32,
            registers,
            k,
            torch.int64,
            "batcher",
            "sequential",
            bits,
            vector,
            lanes,
            recovery,
            payload,
        )
        assert (
            sum(
                node.target is torch.ops.higher_order.cond for node in graph.graph.nodes
            )
            == 1
        )
        mask = (1 << bits) - 1
        for tied in (False, True):
            keys = torch.empty((32, registers), dtype=torch.int64)
            for group in range(32):
                for register in range(registers):
                    column = (
                        (register // vector) * lanes + group % lanes
                    ) * vector + register % vector
                    rank = 0x3F000001 if tied else 0x3F000000 + column * 8192
                    keys[group, register] = (rank << bits) | (mask - column)
            actual = graph(keys)
            # Distinct coarse buckets exercise refinement; tied ranks require
            # the full-key fallback. Compare both to ordinary integer sorting.
            for start in range(0, 32, lanes):
                expected = (
                    keys[start : start + lanes]
                    .flatten()
                    .sort(descending=True)
                    .values[:k]
                )
                selected = actual[start : start + lanes].transpose(0, 1).flatten()[:k]
                if payload:
                    expected = expected & mask
                    selected = selected & mask
                torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    assert torch._dynamo.config.automatic_dynamic_shapes == dynamic_shapes
