"""The attention matchers' shared graph-walk primitives
(``helion._compiler.cute.analysis.match_common``) and the per-family literal
acceptance they make explicit: the dense matchers read Python numbers and
``aten.scalar_tensor`` only, the flex matchers also a 0-d ``full``."""

from __future__ import annotations

import operator
from types import SimpleNamespace

import pytest
import torch

from helion._compiler.compile_environment import FixedBlockSizeSource
from helion._compiler.cute.analysis import match_common
from helion.language import creation_ops
from helion.language._tracing_ops import _host_tensor
from helion.language._tracing_ops import _new_var


def _graph() -> torch.fx.Graph:
    return torch.fx.Graph()


def test_scalar_literals_per_family() -> None:
    graph = _graph()
    scalar = graph.call_function(torch.ops.aten.scalar_tensor.default, (-float("inf"),))
    full = graph.call_function(torch.ops.aten.full.default, ([], 2.0))
    hl_full = graph.call_function(creation_ops.full, ([], 3))
    shaped = graph.call_function(torch.ops.aten.full.default, ([4], 2.0))
    # Both families: numbers (bools are ints) and scalar_tensor.
    assert match_common.scalar_value(2) == 2.0
    assert match_common.scalar_value(True) == 1.0
    assert match_common.scalar_value(scalar) == -float("inf")
    assert match_common.is_neg_inf(scalar)
    assert match_common.is_neg_inf(-float("inf"))
    assert not match_common.is_neg_inf(1.0)
    # The 0-d full forms: flex only.
    assert match_common.scalar_value(full) is None
    assert match_common.scalar_value(hl_full) is None
    assert match_common.scalar_value(full, allow_full=True) == 2.0
    assert match_common.scalar_value(hl_full, allow_full=True) == 3.0
    assert match_common.scalar_value(shaped, allow_full=True) is None
    assert match_common.scalar_value("2", allow_full=True) is None


def test_call_and_slice_predicates() -> None:
    graph = _graph()
    x = graph.placeholder("x")
    item = graph.call_function(operator.getitem, (x, 0))
    assert match_common.is_call(item, operator.getitem)
    assert match_common.is_call(item, operator.add, operator.getitem)
    assert not match_common.is_call(item, operator.add)
    assert not match_common.is_call(x, operator.getitem)
    assert not match_common.is_call(None, operator.getitem)
    assert match_common.is_full_slice(slice(None))
    assert not match_common.is_full_slice(slice(None, 4))
    assert not match_common.is_full_slice(None)


def test_new_var_and_host_tensor() -> None:
    graph = _graph()
    x = graph.placeholder("x")
    once = graph.call_function(_new_var, (x,))
    twice = graph.call_function(_new_var, (once,))
    assert match_common.new_var_source(once) is x
    assert match_common.new_var_source(twice) is once
    assert match_common.new_var_source(x) is None
    assert match_common.strip_new_var(twice) is x
    assert match_common.strip_new_var(x) is x
    assert match_common.strip_new_var(3) == 3
    host = graph.call_function(_host_tensor, ("q",))
    assert match_common.host_tensor_name(host) == "q"
    assert match_common.host_tensor_name(x) is None
    assert (
        match_common.host_tensor_name(graph.call_function(_host_tensor, (1,))) is None
    )


@pytest.mark.parametrize("value", [1, 2])
def test_unit_fixed_block(value: int) -> None:
    block = SimpleNamespace(block_size_source=FixedBlockSizeSource(value))
    env = SimpleNamespace(block_sizes=[block])
    assert match_common.unit_fixed_block(env, 0) is (value == 1)  # type: ignore[arg-type]
