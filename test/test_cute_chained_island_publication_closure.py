from __future__ import annotations

import math
import operator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.fx import Graph

from helion._compiler.cute.chained_island_publication import _closed_users
from helion._compiler.cute.chained_island_publication import _prepared_users
from helion.language import memory_ops
from helion.language import view_ops


def _graph():
    graph = Graph()
    export = graph.placeholder("original_fp32_export")
    entry = graph.placeholder("unrelated_input")
    point = graph.call_function(operator.add, (export, entry))
    contraction = graph.call_function(operator.matmul, (point, entry))
    bound: Any = SimpleNamespace(
        island=SimpleNamespace(values=(SimpleNamespace(node=export),)),
        exports=(SimpleNamespace(node=export),),
    )
    return graph, export, entry, point, contraction, bound


def test_exclusive_original_operand_closure():
    _, _, _, point, contraction, bound = _graph()
    assert _closed_users(bound, point, contraction)


@pytest.mark.parametrize(
    "site", ["export", "other_role", "nested_args", "nested_kwargs"]
)
def test_extra_original_users_never_authorize_omission(site):
    graph, export, entry, point, contraction, bound = _graph()
    if site == "export":
        graph.call_function(operator.neg, (export,))
    elif site == "other_role":
        contraction.args = (point, point)
    elif site == "nested_args":
        contraction.args = (point, entry, (point,))
    else:
        contraction.kwargs = {"explicit_accumulator": {"nested": point}}
    assert not _closed_users(bound, point, contraction)


def test_real_original_operand_can_be_the_common_published_multiuse_cut():
    graph, _, _, point, contraction, bound = _graph()
    graph.call_function(operator.neg, (point,))
    assert _closed_users(bound, point, contraction)


def test_internal_cast_descendant_cannot_hide_old_export_reader():
    graph, export, _, point, contraction, bound = _graph()
    inside = graph.call_function(operator.neg, (export,))
    bound.island.values += (SimpleNamespace(node=inside),)
    graph.call_function(operator.abs, (inside,))
    assert not _closed_users(bound, point, contraction)


def _prepared_graph():
    graph = Graph()
    operand = graph.placeholder("original_half_image")
    first = graph.call_function(torch.ops.aten.neg.default, (operand,))
    second = graph.call_function(torch.ops.aten.mul.Tensor, (operand, operand))
    frame = SimpleNamespace(
        cut=SimpleNamespace(preparation=(first, second)),
        buffers=(SimpleNamespace(node=first), SimpleNamespace(node=second)),
    )
    pipeline: Any = SimpleNamespace(
        frame=frame,
        prepared_leaves=(),
        prepared_operands=(),
        prepared_groups=(),
        operand_retention=None,
    )
    return graph, operand, first, second, pipeline


def test_unrelated_pure_multiusers_end_at_original_materializations():
    graph, operand, first, _, pipeline = _prepared_graph()
    graph.output(first)
    assert _prepared_users(pipeline, operand)


@pytest.mark.parametrize(
    "escape",
    [
        "output",
        "host_load",
        "store",
        "gather",
        "unknown",
        "late_collective",
        "native",
        "tma",
        "retained",
        "carry",
    ],
)
def test_unknown_native_async_and_carry_readers_fail_closed(escape):
    graph, operand, _, _, pipeline = _prepared_graph()
    if escape in ("output", "carry"):
        graph.output(operand)
    elif escape == "native":
        pipeline.prepared_groups = (
            SimpleNamespace(
                candidate=SimpleNamespace(
                    members=(SimpleNamespace(buffer=SimpleNamespace(node=operand)),)
                )
            ),
        )
    elif escape == "tma":
        pipeline.prepared_leaves = (SimpleNamespace(node=operand),)
    elif escape == "retained":
        pipeline.operand_retention = SimpleNamespace(
            candidates=(SimpleNamespace(node=operand),)
        )
    else:
        targets = {
            "host_load": memory_ops.load,
            "store": memory_ops.store,
            "gather": view_ops.subscript,
            "unknown": math.gcd,
            "late_collective": torch.ops.aten.sum.dim_IntList,
        }
        node = graph.call_function(targets[escape], (operand,))
        pipeline.frame.cut.preparation += (node,)
    assert not _prepared_users(pipeline, operand)
