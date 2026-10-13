from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import inspect
import sys
from types import FunctionType
from unittest.mock import patch

import pytest
import torch

from test import test_cute_chunk_recurrence as original

from helion import exc
from helion._compiler.cute import chained_body_program as common
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute import prepared_epoch_body as epoch
from helion._compiler.cute.prepared_epoch_issue import DescriptorIssue
from helion._compiler.cute.prepared_epoch_memory import EpochMemoryAction
from helion._compiler.cute.prepared_epoch_protocol import EpochEvent
from helion._compiler.cute.prepared_epoch_protocol import EpochResourceOperation
from helion._compiler.cute.prepared_epoch_protocol import EpochSynchronizationAction
from helion._compiler.cute.prepared_state_body import StateEffect
from helion.runtime.cute.launcher import _create_cute_wrapper


def _code():
    planner = chunk_recurrence._plan_chunk_recurrence

    def selected(*args, **kwargs):
        return planner(*args, **kwargs, prepared_epoch=True)

    with patch.object(chunk_recurrence, "_plan_chunk_recurrence", selected):
        return original._code(fp32_state=True, dv_partitions=2, pipeline="wide")


def test_actual_complete_epoch_walk_and_original_host_link(tmp_path):
    bodies = []
    accept = epoch.EpochSourceBody.accept

    def observe(body, lines):
        accept(body, lines)
        bodies.append(body)

    with patch.object(epoch.EpochSourceBody, "accept", observe):
        source = _code()
    assert len(bodies) == 1
    body = bodies[0]
    assert body.consumed and body.trace
    assert {
        EpochEvent,
        DescriptorIssue,
        StateEffect,
        EpochMemoryAction,
        EpochSynchronizationAction,
    } <= {type(step) for _, step in body.trace}
    selected = body.plan.prepared_epoch
    assert selected is not None
    issues = [step for _, step in body.trace if isinstance(step, DescriptorIssue)]
    assert [step.issue.spec for step in issues] == [
        selected.continuation.first,
        selected.continuation.second,
        selected.projected.spec,
        selected.projected.spec,
        selected.update,
        selected.update,
    ]
    assert [step.segment for step in issues[2:4]] == list(selected.projected.segments)
    assert all(
        epoch.EpochLoop.ALL in scope
        for scope, step in body.trace
        if isinstance(step, DescriptorIssue)
    )
    assert any(
        (epoch.EpochPredicate.PREVIOUS, True) in scope for scope, _ in body.trace
    )
    terminal = [
        (scope, step)
        for scope, step in body.trace
        if isinstance(step, EpochSynchronizationAction)
        and step.operation is EpochResourceOperation.RETIRE
    ]
    assert len(terminal) == 1 and not any(
        isinstance(item, epoch.EpochLoop) for item in terminal[0][0]
    )
    parsed = ast.parse(source)
    function = next(
        n
        for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name == "_helion_epoch_kernel"
    )
    calls = [ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)]
    assert calls.count("prepared_tcgen_edge.execute_prepared_issue") == 6
    assert calls.count("prepared_tcgen_edge.execute_prepared_read") == 7
    assert calls.count("prepared_residual_point") == 2
    assert "execute_prepared_continuation" not in calls
    assert "._helion_cute_epoch_kernel = _helion_epoch_kernel" in source
    assert "'epoch_kernel': '_helion_epoch_kernel'" in source
    (tmp_path / "selected.py").write_text(source)


@pytest.mark.parametrize("mutation", (None, "missing", "both"))
def test_actual_emitted_callable_reaches_original_default_wrapper(tmp_path, mutation):
    before_cuda = torch.cuda.is_initialized()
    source = _code()
    path = tmp_path / "epoch_source.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("selected_epoch_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    target = next(
        node.targets[0].value.id
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].attr == "_helion_cute_epoch_kernel"
    )
    kernel = vars(module)[target]
    assert kernel._helion_cute_epoch_kernel is module._helion_epoch_kernel
    plan = kernel._helion_cute_wrapper_plans[0]
    if mutation == "missing":
        kernel._helion_cute_epoch_kernel = None
    elif mutation == "both":
        plan["prepared_continuation"] = ((0, 0, False, False),)
    args = original._fake_inputs(fp32_state=True)
    schema = (
        *(
            ("tensor", arg.dtype, arg.ndim, tuple(arg.shape), tuple(arg.stride()))
            for arg in args
        ),
        ("scalar", "float"),
    )
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
    ):
        if mutation is not None:
            with pytest.raises(exc.BackendUnsupported):
                _create_cute_wrapper(kernel, schema, (512, 1, 1), 152)
        else:
            wrapper = _create_cute_wrapper(kernel, schema, (512, 1, 1), 152)
            assert isinstance(wrapper, FunctionType)
            text = inspect.getsource(wrapper)
            assert "EPOCH_KERNEL=_helion_epoch_kernel" in text
            assert "_helion_sm100_chain_host(" in text
    assert torch.cuda.is_initialized() == before_cuda


@pytest.mark.parametrize(
    "mutation",
    (
        "drop",
        "copy",
        "order",
        "scope",
        "native",
        "relation",
        "segment",
        "return",
        "accepted",
    ),
)
def test_actual_epoch_mutations_reject(mutation):
    interpreter = common.emit_body_program
    leaf = epoch.EpochSourceBody.step
    accept = epoch.EpochSourceBody.accept
    seen = []

    def interpret(*args, **kwargs):
        body = kwargs.get("epoch_body")
        if body is not None and mutation in ("drop", "copy", "order", "scope"):
            actions = body.program.actions
            if mutation == "scope":
                object.__setattr__(actions[0], "predicate", epoch.EpochPredicate.QUERY)
            else:
                object.__setattr__(
                    body.program,
                    "actions",
                    {
                        "drop": actions[1:],
                        "copy": (replace(actions[0]), *actions[1:]),
                        "order": tuple(reversed(actions)),
                    }[mutation],
                )
            seen.append(True)
        lines = interpreter(*args, **kwargs)
        if body is not None and mutation == "return":
            lines.append("foreign_write()")
            seen.append(True)
        return lines

    def step(body, action, scope):
        if (
            not seen
            and isinstance(action, DescriptorIssue)
            and (mutation != "segment" or action.segment is not None)
        ):
            if mutation == "native":
                object.__setattr__(
                    action.issue, "initialized", not action.issue.initialized
                )
            elif mutation == "relation":
                object.__setattr__(
                    action.issue, "spec", body.plan.prepared_epoch.update
                )
            elif mutation == "segment":
                object.__setattr__(
                    action, "segment", body.plan.prepared_epoch.projected.segments[1]
                )
            if mutation in ("native", "relation", "segment"):
                seen.append(True)
        return leaf(body, action, scope)

    def accepted(body, lines):
        accept(body, lines)
        if mutation == "accepted":
            lines.append("foreign_write()")
            seen.append(True)

    with (
        patch.object(common, "emit_body_program", interpret),
        patch.object(epoch.EpochSourceBody, "step", step),
        patch.object(epoch.EpochSourceBody, "accept", accepted),
        pytest.raises(exc.InternalError) as caught,
    ):
        _code()
    assert seen == [True]
    assert isinstance(caught.value.__cause__, chain._UnsupportedChain)


@pytest.mark.parametrize("cuda_initialized", [False, True])
@pytest.mark.parametrize("mutation", (None, "missing", "both"))
def test_default_wrapper_preserves_prior_cuda_state(
    tmp_path, mutation, cuda_initialized
):
    with (
        patch("torch.cuda.is_initialized", return_value=cuda_initialized),
        # FakeTensor normalizes an initialized CUDA device through this metadata.
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
    ):
        test_actual_emitted_callable_reaches_original_default_wrapper(
            tmp_path, mutation
        )
        assert torch.cuda.is_initialized() == cuda_initialized
