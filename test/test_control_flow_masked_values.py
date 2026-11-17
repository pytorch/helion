from __future__ import annotations

import operator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ElseGraphInfo
from helion._compiler.device_ir import GraphInfo
from helion._compiler.device_ir import IfGraphInfo
from helion._compiler.node_masking import getitem_masked_value
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("initial_mask", (0, None))
def test_if_masks_follow_joined_slots_and_unchanged_outer_values(initial_mask):
    outer = torch.fx.Graph()
    initial = outer.placeholder("initial")
    initial.meta["masked_value"] = initial_mask
    then, otherwise = torch.fx.Graph(), torch.fx.Graph()
    then_only = then.placeholder("then_only")
    then_common = then.placeholder("then_common")
    else_common = otherwise.placeholder("else_common")
    else_only = otherwise.placeholder("else_only")
    for node, mask in (
        (then_only, 1),
        (then_common, 2),
        (else_common, 3),
        (else_only, 4),
    ):
        node.meta["masked_value"] = mask
    then.output((then_only, then_common))
    otherwise.output((else_common, else_only))
    info = IfGraphInfo(
        0,
        then,
        [],
        else_branch=1,
        if_arg_names=[],
        else_arg_names=["initial"],
        branches_outputs=[(1, 0), (0, "initial"), ("initial", 1)],
    )
    call = outer.call_function(_tracing_ops._if, (True, 0, 1, [], [initial]))
    graphs = [info, ElseGraphInfo(1, otherwise, [])]
    with patch.object(DeviceIR, "current", return_value=SimpleNamespace(graphs=graphs)):
        actual = [
            getitem_masked_value(outer.call_function(operator.getitem, (call, index)))
            for index in range(6)
        ]
    assert actual == [2, 1, initial_mask, 3, initial_mask, 4]


def test_loop_masks_keep_the_original_output_positions():
    body = torch.fx.Graph()
    values = [body.placeholder(name) for name in ("first", "second")]
    for value, mask in zip(values, (-float("inf"), 0), strict=True):
        value.meta["masked_value"] = mask
    body.output(tuple(values))
    outer = torch.fx.Graph()
    call = outer.call_function(_tracing_ops._for_loop, (0, [0], [2], []))
    with patch.object(
        DeviceIR, "current", return_value=SimpleNamespace(graphs=[GraphInfo(0, body)])
    ):
        actual = [
            getitem_masked_value(outer.call_function(operator.getitem, (call, index)))
            for index in range(2)
        ]
    assert actual == [-float("inf"), 0]
