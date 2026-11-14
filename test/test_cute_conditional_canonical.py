"""CPU ABI and actual complete-source export for the canonical N32 case."""

from __future__ import annotations

import ast
import json
from types import SimpleNamespace

import pytest
import torch

from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_conditional_canonical import canonical_arguments
from helion._testing.cute_conditional_canonical import canonical_cpu_fixture
from helion._testing.cute_conditional_canonical import run
from helion._testing.cute_conditional_compile import compile_conditional_entries
from helion._testing.cute_paged_conditional_semantics import (
    written_conditional_attention,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
@forbid_cuda_work()
def inputs():
    return canonical_cpu_fixture()


@forbid_cuda_work()
def test_metadata_and_public_destinations(inputs, monkeypatch):
    q, k, v, ks, vs, table, lengths, output = inputs
    calls = []

    def bind(args):
        def execute(*current):
            assert current == args
            calls.append(current)
            return current[7], current[8]

        return SimpleNamespace(
            config_spec=SimpleNamespace(default_config=lambda: "default"),
            compile_config=lambda config: execute if config == "default" else None,
        )

    monkeypatch.setattr(written_conditional_attention, "bind", bind)
    for return_lse in (False, True):
        result, lse = run(
            q,
            k,
            v,
            ks,
            vs,
            table,
            lengths,
            max_kv_len=28027,
            window_left=2047,
            return_lse=return_lse,
            out=output,
        )
        assert result is output
        assert (lse is calls[-1][8]) if return_lse else (lse is None)
    assert calls[0][8] is not calls[1][8]
    assert calls[0][-3:] == (148, 18, 2047)
    assert all(a is b for a, b in zip(calls[0][:8], inputs, strict=True))


@forbid_cuda_work()
def test_abi_refuses_other_metadata_and_aliases(inputs):
    original = list(inputs)
    unrelated_scale = torch.empty(7132 * 33792, dtype=torch.uint8).as_strided(
        inputs[3].shape, inputs[3].stride(), 32768
    )
    aliased_output = inputs[0].as_strided(inputs[7].shape, inputs[7].stride(), 0)
    for changes in (
        {0: inputs[0].contiguous()},
        {3: unrelated_scale},
        {2: inputs[1], 4: inputs[3]},
        {7: aliased_output},
    ):
        values = original.copy()
        for index, replacement in changes.items():
            values[index] = replacement
        with pytest.raises(ValueError):
            canonical_arguments(
                *values[:7], out=values[7], max_kv_len=28027, window_left=2047
            )
    with pytest.raises(ValueError, match="draft SWA"):
        canonical_arguments(
            *original[:7], out=original[7], max_kv_len=28028, window_left=2047
        )


@pytest.mark.slow
@forbid_cuda_work(forbid_native=False)
def test_actual_canonical_source_and_entries(inputs, tmp_path):
    args = canonical_arguments(
        *inputs[:7], out=inputs[7], max_kv_len=28027, window_left=2047
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(written_conditional_attention, args)
        assert len(bound.host_function.device_ir.graphs) == 16
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
    (tmp_path / "generated.py").write_text(source)
    (tmp_path / "config.json").write_text(json.dumps(config.config, indent=2) + "\n")
    stages = stage_sources(source)
    assert len(stages) == 2
    for index, stage in enumerate(stages):
        tree = ast.parse(stage)
        launches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launcher"
        ]
        assert len(launches) == 1
        assert ast.literal_eval(launches[0].args[1]) == (640,)
        assert ast.literal_eval(launches[0].keywords[0].value) == (128, 1, 1)
        assert "_conditional_attention_page(" in stage
        assert "fragment_arena" not in stage
        (tmp_path / f"stage-{index}.py").write_text(stage)
    modules = compile_conditional_entries(
        tmp_path / "generated.py", args, complete=True
    )
    assert len(modules) == 2
    for index, module in enumerate(modules):
        (tmp_path / f"stage-{index}.mlir").write_text(module)
