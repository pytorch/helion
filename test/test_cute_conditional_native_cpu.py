"""Actual B3 source admission for the dedicated numerical fixture."""

from __future__ import annotations

import ast
import copy

import pytest
import torch

from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_conditional_native import validate_conditional_source
from helion._testing.cute_conditional_native import validate_page_lifetime
from helion._testing.cute_conditional_phase_binding import phase_fixture
from helion._testing.cute_fragment_shared_replay import begin_record
from helion._testing.cute_paged_conditional_semantics import (
    written_conditional_attention,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
@forbid_cuda_work()
def source(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("conditional-source")
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(written_conditional_attention, phase_fixture())
        source = bound.to_code(bound.config_spec.default_config())
    (tmp_path / "generated.py").write_text(source)
    return source


@forbid_cuda_work()
def test_bounded_conditional_source_and_role_refusals(source):
    validate_conditional_source(source)
    for before, after in (
        ("_conditional_attention_page(", "_other_page("),
        (" | ", " & "),
        ("block=(128, 1, 1)", "block=(256, 1, 1)"),
        ("(6,)", "(640,)"),
    ):
        assert before in source
        with pytest.raises(AssertionError):
            validate_conditional_source(source.replace(before, after))


@forbid_cuda_work()
def test_nonempty_lifetime_and_handle_ownership(source):
    # Exercise both actual first-root and seeded-continuation lifetimes. Each
    # mutation would make zero trips acquire, free a live page, or mix handles.
    for stage in stage_sources(source):
        kernel = next(
            node
            for node in ast.parse(stage).body
            if isinstance(node, ast.FunctionDef)
            and any(ast.unparse(x) == "cute.kernel" for x in node.decorator_list)
        )
        validate_page_lifetime(kernel)
        for mutation in (
            "empty",
            "unguarded",
            "early_retire",
            "wrong_handle",
            "break",
            "wrong_initial_output",
            "wrong_final_output",
            "wrong_final_width",
        ):
            changed = copy.deepcopy(kernel)
            owner = next(
                node
                for node in ast.walk(changed)
                if isinstance(node, ast.If)
                and len(node.body) == 3
                and isinstance(node.body[1], ast.For)
                and isinstance(node.body[0], ast.Assign)
                and isinstance(node.body[0].value, ast.Call)
                and ast.unparse(node.body[0].value.func)
                == "_conditional_attention_begin"
            )
            assign, loop, retire = owner.body
            if mutation == "empty":
                owner.test.ops[0] = ast.LtE()
            elif mutation == "unguarded":
                owner.test = ast.Constant(True)
            elif mutation == "early_retire":
                loop.body.insert(0, retire)
                owner.body.pop()
            elif mutation == "wrong_handle":
                retire.value.args[0] = ast.Name("unrelated_tmem", ast.Load())
            elif mutation == "break":
                loop.body.append(ast.Break())
            elif mutation == "wrong_initial_output":
                assign.value.args[0] = ast.Name("unrelated_output", ast.Load())
            elif mutation == "wrong_final_output":
                retire.value.args[2] = ast.Name("unrelated_output", ast.Load())
            else:
                retire.value.args[4] = ast.Constant(64)
            with pytest.raises(AssertionError):
                validate_page_lifetime(changed)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
@forbid_cuda_work()
def test_native_record_uses_current_device_and_gates_unsupported_hardware(
    capability, tmp_path, monkeypatch
):
    root = tmp_path / "nested" / "raw"
    monkeypatch.setenv("HELION_SHARED_STORAGE_RAW", str(root))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def device_capability(device):
        assert device == 3
        return capability

    def forbidden_device_count():
        raise AssertionError("visible device count does not restrict this fixture")

    monkeypatch.setattr(torch.cuda, "get_device_capability", device_capability)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden_device_count)
    if capability != (10, 0):
        with pytest.raises(pytest.skip.Exception, match="requires CUDA SM100"):
            begin_record("unsupported", tmp_path)
        assert not root.exists()
    else:
        output, record = begin_record("current", tmp_path)
        assert output.parent == root and output.is_dir()
        assert record["device"] == 3 and not record["complete"]
