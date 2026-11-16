"""CPU search-space and emitted-source checks for conditional attention."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from helion._compiler.cute.analysis.conditional_continuation import (
    prove_conditional_continuation,
)
from helion._compiler.cute.conditional_attention_codegen import (
    conditional_attention_compiler,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_conditional_canonical import canonical_arguments
from helion._testing.cute_conditional_canonical import canonical_cpu_fixture
from helion._testing.cute_paged_conditional_tunable import tunable_conditional_attention

pytestmark = pytest.mark.cpu_only


@pytest.mark.slow
@forbid_cuda_work()
def test_partition_tunable_reaches_both_dedicated_stages(tmp_path):
    inputs = canonical_cpu_fixture()
    args = list(
        canonical_arguments(
            *inputs[:7], out=inputs[7], max_kv_len=28027, window_left=2047
        )
    )
    args[9] = 148
    checked = []

    def observe(cg, graphs):
        result = conditional_attention_compiler(cg, graphs)
        if checked:
            return result
        ir = copy.copy(cg.host_function.device_ir)
        ir.graphs = graphs
        proof = prove_conditional_continuation(ir, bound.env)
        if proof is None:
            return result
        assert result is not None
        (symbol,) = [
            expression
            for expression, value in bound.env.config_value_expressions.items()
            if value.operation == "config" and value.arguments == ("partition_buckets",)
        ]
        # The symbolic value must have both declared tunable provenance and
        # an emitted config expression. A debug label or size hint is insufficient.
        bound.env.tunable_symbols.remove(symbol)
        try:
            assert prove_conditional_continuation(ir, bound.env) is None
        finally:
            bound.env.tunable_symbols.add(symbol)
        expression = bound.env.config_value_expressions.pop(symbol)
        try:
            assert prove_conditional_continuation(ir, bound.env) is None
        finally:
            bound.env.config_value_expressions[symbol] = expression
        # An equivalent factored/expanded identity is accepted; changing the
        # actual entry predicate while retaining rescale metadata is refused.
        saved = proof.seed.args
        try:
            proof.seed.args = (saved[0], saved[0])
            assert prove_conditional_continuation(ir, bound.env) is None
        finally:
            proof.seed.args = saved
        assert prove_conditional_continuation(ir, bound.env) == proof
        checked.append(symbol)
        return result

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(tunable_conditional_attention, tuple(args))
        assert bound.config_spec.user_defined_tunables["partition_buckets"].choices == (
            148,
            74,
            320,
        )
        sources = []
        for buckets in (148, 74, 320):
            config = bound.config_spec.default_config()
            config.config["partition_buckets"] = buckets
            with patch(
                "helion._compiler.cute.conditional_attention_codegen.conditional_attention_compiler",
                side_effect=observe,
            ):
                source = bound.to_code(config)
            (tmp_path / f"partition-{buckets}.py").write_text(source)
            stages = stage_sources(source)
            assert len(stages) == 2
            assert all("_conditional_attention_page(" in stage for stage in stages)
            assert all(f"work_buckets = {buckets}" in stage for stage in stages)
            sources.append(source)
    assert len(checked) == 1
    assert len(set(sources)) == 3
