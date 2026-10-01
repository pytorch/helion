from __future__ import annotations

import ast
from itertools import pairwise
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_work_order_admission import test_actual_fused_admission as _fused
from .test_cute_work_order_public import _policies
from .test_cute_work_order_scalar import _ragged_sum
import helion
from helion._compiler.cute import work_order


def _signed(value, bits):
    value = int(value) % (1 << bits)
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


class _Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return _Pointer(self.values, self.offset + int(offset))

    def load(self):
        assert 0 <= self.offset < len(self.values)
        value = self.values[self.offset]
        assert value is not None
        return np.int64(value)


def _tensor(values, stride=1):
    storage = [None] * (len(values) * stride)
    storage[::stride] = values
    return SimpleNamespace(
        iterator=_Pointer(storage), layout=SimpleNamespace(stride=(stride,))
    )


def _replay(text, count, origin, tensors, expected):
    """Execute the actual emitted AST at its two warp-collective boundaries.

    This is CPU arithmetic/address coverage, not native shuffle qualification.
    Every lane runs the original projected loads and casts. The collective mocks
    deliver the values of the exact other lanes, including both Uint64 halves.
    """
    tree = ast.parse(text)
    cut = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "work_rank"
    )
    prefix = compile(
        ast.Module(body=tree.body[:cut], type_ignores=[]), "<actual-work-key>", "exec"
    )
    suffix = compile(
        ast.Module(body=tree.body[cut:], type_ignores=[]), "<actual-work-rank>", "exec"
    )
    cutlass = SimpleNamespace(
        Int32=lambda value: np.int32(_signed(value, 32)),
        Int64=lambda value: np.int64(_signed(value, 64)),
        Uint64=lambda value: np.uint64(int(value) % (1 << 64)),
        range_constexpr=range,
    )
    lanes = []
    with np.errstate(over="ignore"):
        for lane in range(32):
            state: dict[str, Any] = dict(
                tensors,
                cutlass=cutlass,
                cute=SimpleNamespace(
                    arch=SimpleNamespace(lane_idx=lambda lane=lane: lane)
                ),
            )
            exec(prefix, state)
            lanes.append(state)
        keys = [state["work_trips"] for state in lanes]
        assert [int(key) for key in keys[:count]] == expected
        assert all(int(key) == expected[lane % count] for lane, key in enumerate(keys))
        result = []
        for logical in range(count):
            candidates = []
            for state in lanes:
                state[origin] = np.int32(logical)
                state["cute"].arch.shuffle_sync = lambda value, peer: keys[int(peer)]
                state["cute"].arch.warp_reduction_max = lambda value: value
                exec(suffix, state)
                candidates.append(int(state["work_selected"]))
            result.append(max(candidates))
    assert result == sorted(range(count), key=lambda i: (-expected[i], i))
    assert sorted(result) == list(range(count))


@pytest.mark.parametrize("count", (2, 3, 5, 16, 31, 32))
def test_actual_ordinary_permutation_all_lanes(count, tmp_path):
    stride = 2 if count == 5 else 1
    args = (
        torch.empty((count, 64)),
        torch.arange(count * stride, dtype=torch.int64)[::stride],
        torch.arange(count * stride, dtype=torch.int64)[::stride] + 32,
        torch.empty(count),
    )
    captured = []
    original = work_order.emit_work_permutation

    def observe(cg, plan, coordinates):
        result = original(cg, plan, coordinates)
        captured.append(
            ast.unparse(ast.Module(body=list(result.statements), type_ignores=[]))
        )
        return result

    with _cpu_codegen(), patch.object(work_order, "emit_work_permutation", observe):
        bound = _ragged_sum._bind_isolated(args)
        config = helion.Config(
            block_sizes=[32],
            num_warps=4,
            pid_type="flat",
            cute_grid_work_order=_policies(bound, True),
        )
        with bound.env.use_runtime_arg_values(_runtime_values(_ragged_sum, args)):
            source = bound.to_code(config)
    assert len(captured) == 1
    # Nonzero starts, empty/reverse intervals, ceil/ties, and original Int32
    # narrowing. Large metadata are arithmetic probes, not body-valid inputs.
    patterns = (
        (0, 1),
        (-40, 25),
        (10, 10),
        (50, -10),
        (2**32 + 1, 2**32 + 34),
        (-(2**31), 2**31 - 1),
    )
    begin, end = zip(*(patterns[i % len(patterns)] for i in range(count)), strict=True)
    starts, ends = [value - 1 for value in begin], [value + 2 for value in end]
    expected = [
        max(0, (_signed(b, 32) - _signed(a, 32) + 31) // 32)
        for a, b in zip(begin, end, strict=True)
    ]
    _replay(
        captured[0],
        count,
        "indices_0",
        {"starts": _tensor(starts, stride), "ends": _tensor(ends)},
        expected,
    )
    _replay(
        captured[0],
        count,
        "indices_0",
        {"starts": _tensor([-1] * count, stride), "ends": _tensor([2] * count)},
        [0] * count,
    )
    (tmp_path / "source.py").write_text(source)
    (tmp_path / "permutation.py").write_text(captured[0])


def test_actual_fused_permutation_retains_high_words(tmp_path):
    _fused(tmp_path)
    text = (tmp_path / "permutation.py").read_text()
    for bounds in (
        (0, 33, 65),
        (0, 0, 0),
        (0, 2**40 + 1, 2**40 + 34),
        (-(2**62), 2**61, 2**61 + 1),
    ):
        expected = [max(0, (b - a + 31) // 32) for a, b in pairwise(bounds)]
        _replay(
            text, 2, "chain_origin_1", {"cu_seqlens": _tensor(list(bounds))}, expected
        )
