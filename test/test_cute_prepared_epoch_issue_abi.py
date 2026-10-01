from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chunk_recurrence as original
from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute import prepared_epoch_body
from helion._compiler.cute import prepared_tcgen_edge
from helion._compiler.cute.prepared_epoch_issue import DescriptorIssue

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture(scope="module")
def public_epoch_calls(
    tmp_path_factory,
) -> tuple[tuple[ast.Call, DescriptorIssue], ...]:
    bodies = []
    accept = prepared_epoch_body.EpochSourceBody.accept

    def observe(body, lines):
        accept(body, lines)
        bodies.append(body)

    config = helion.Config.from_dict(
        original._CONFIG.config
        | {
            "cute_chunk_recurrence_dv_partitions": 2,
            "cute_chunk_recurrence_pipeline": "wide",
        }
    )
    # Normal public selection: no planner suppression or prepared_* keyword.
    with (
        _cpu_codegen(),
        patch.object(original, "DEVICE", torch.device("cuda", 0)),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3)),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.runtime.get_num_sm", return_value=152),
        patch.object(prepared_epoch_body.EpochSourceBody, "accept", observe),
    ):
        bound = original._bt16_fp32_chain._bind_isolated(
            (*original._fake_inputs(fp32_state=True), 128**-0.5)
        )
        source = bound.to_code(config)
    assert len(bodies) == 1 and bodies[0].consumed
    issues = [step for _, step in bodies[0].trace if isinstance(step, DescriptorIssue)]
    calls = sorted(
        (
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "prepared_tcgen_edge.execute_prepared_issue"
        ),
        key=lambda node: node.lineno,
    )
    assert len(calls) == len(issues) == 6
    assert [issue.kind for issue in issues] == [
        "query",
        "output",
        "projection",
        "projection",
        "update",
        "update",
    ]
    (tmp_path_factory.mktemp("public_epoch") / "source.py").write_text(source)
    return tuple(zip(calls, issues, strict=True))


def _bind(call: ast.Call) -> Mapping[str, ast.expr]:
    assert all(keyword.arg is not None for keyword in call.keywords)
    keywords = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }
    signature = inspect.signature(prepared_tcgen_edge.execute_prepared_issue)
    return signature.bind(*call.args, **keywords).arguments


def _assert_controls(values: Mapping[str, ast.expr], action: DescriptorIssue) -> None:
    issue = action.issue
    expected = {
        "issuer": True,
        "completion_phase": 0,
        "input_ready": None,
        "input_phase": 0,
        "DESCRIPTOR": True,
        "TMEM_A": True,
        "K_BEGIN": issue.begin,
        "K_END": issue.end,
        "K_ATOM": issue.atom_k,
        "INITIALIZED": issue.initialized,
        "COMMIT": issue.commit,
        "WAIT_AFTER": issue.wait_after,
    }
    for name, value in expected.items():
        actual = ast.literal_eval(values[name])
        assert type(actual) is type(value) and actual == value, name
    completion = {
        "query": "stateq_done_mbar.subview(kr_stage)",
        "output": "qstate_acc_ready_mbar.subview(qstate_stage)",
        "projection": "shared_acc_ready_mbar.subview(acc_stage)",
        "update": "None",
    }[action.kind]
    assert ast.unparse(values["completion"]) == completion


def test_normal_public_six_issues_bind_actual_helper_abi(public_epoch_calls) -> None:
    parameters = tuple(
        inspect.signature(prepared_tcgen_edge.execute_prepared_issue).parameters
    )
    for call, action in public_epoch_calls:
        assert len(call.args) == 4
        assert tuple(keyword.arg for keyword in call.keywords) == parameters[4:]
        _assert_controls(_bind(call), action)


@pytest.mark.parametrize("site", range(6))
def test_previous_positional_controls_are_rejected(
    public_epoch_calls, site: int
) -> None:
    call, action = public_epoch_calls[site]
    values = _bind(call)
    parameters = tuple(
        inspect.signature(prepared_tcgen_edge.execute_prepared_issue).parameters
    )
    # Exact previous serializer, not a weakened surrogate mutation.
    legacy = ast.Call(
        func=call.func,
        args=[
            *call.args,
            ast.Constant(None),
            ast.Constant(0),
            values["completion"],
            ast.Constant(0),
            ast.Constant(True),
            ast.Constant(True),
            ast.Constant(True),
            *(values[name] for name in parameters[11:]),
        ],
        keywords=[],
    )
    with pytest.raises(AssertionError, match="issuer"):
        _assert_controls(_bind(legacy), action)
