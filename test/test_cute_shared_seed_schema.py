"""CPU reproduction of the real Inductor native module and search-entry schema."""

from __future__ import annotations

import ast
import random
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler import autotuner_heuristics as heuristics
from helion._compiler.cute import shared_gemm_seeds
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.kernel import BoundKernel

# Byte-exact Inductor module from the naturally observed registration failure.
# It is a portable regression fixture, not a changed frontend or benchmark.
INDUCTOR_NATIVE = """import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

from torch._inductor.runtime.helion import HelionKernelWrapper
@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full", ref_mode=helion.RefMode.OFF)
def helion_op0_op1_op2(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 384]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6 + tmp7
        out0[tile_y, tile_x] = tmp8.to(torch.bfloat16)
        tmp9 = in1[tile_y.index.reshape([tile_y, 1]) // 512, (tile_y.index.reshape([tile_y, 1]) % 512), tile_x.index.reshape([1, tile_x])].to(torch.float32)
        tmp10 = tmp9 * tmp8
        out1[tile_y, tile_x] = tmp10.to(torch.bfloat16)
helion_op0_op1_op2 = HelionKernelWrapper(helion_op0_op1_op2)

def call_helion_op0_op1_op2(arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg0_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op0_op1_op2(arg1_1, arg2_1, arg0_1, arg3_1, buf1, reinterpret_tensor(buf2, (16384, 384), (384, 1), 0))
"""


def exact_native():
    return INDUCTOR_NATIVE


def bind_inductor(tmp_path):
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.helion import HelionKernelWrapper

    module = PyCodeCache.load(exact_native())
    wrapper = module.helion_op0_op1_op2
    assert isinstance(wrapper, HelionKernelWrapper)
    # PyCodeCache deliberately reuses the module; a fresh real Kernel around
    # its exact function/settings/policy avoids a previous binding in a test.
    original = wrapper.kernel
    assert isinstance(original, helion.Kernel)
    kernel = helion.kernel(
        original.fn,
        settings=original.settings,
        cute_structural_policy=original.cute_structural_policy,
    )
    wrapper = HelionKernelWrapper(kernel)
    args = (
        torch.empty((16384, 768), dtype=torch.bfloat16),
        torch.empty_strided((768, 384), (1, 768), dtype=torch.bfloat16),
        torch.empty(384, dtype=torch.bfloat16),
        torch.empty_strided((32, 512, 384), (196608, 1, 512), dtype=torch.bfloat16),
        torch.empty((16384, 384), dtype=torch.bfloat16),
        torch.empty((16384, 384), dtype=torch.bfloat16),
    )
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, args)
    return wrapper, bound, args


def fields(spec):
    return tuple((k, *v.fingerprint()) for k, v in spec._flat_fields().items())


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def test_exact_Inductor_module_search_entry_preserves_registered_fields_and_all48(
    tmp_path,
):
    records = []
    append = heuristics.append_shared_gemm_seed_configs

    def observe(env, ir):
        before = fields(env.config_spec)
        append(env, ir)
        after = fields(env.config_spec)
        assert before == after
        records.append(before)

    with patch.object(heuristics, "append_shared_gemm_seed_configs", observe):
        wrapper, bound, args = bind_inductor(tmp_path)
    with bound.env, bound.host_function:
        spec = bound.config_spec
        spec.validate_compiler_coverage_groups()
        generation = ConfigGeneration(spec)
        initial = generation.initial_population_view()
        assert fields(spec) == spec._compiler_coverage_fields
        canonical = generation.seed_flat_config_pairs()
        assert len(canonical) == len(spec.compiler_seed_configs) == 48
        assert len(initial.flat_spec) == len(generation.flat_spec)
    assert len(records) == 1

    # Exercise the actual wrapper->Kernel->managed BoundKernel call. Stop only
    # at the autotune boundary before any native compile/benchmark/GPU work.
    class ReachedSearch(Exception):
        pass

    def search(self, args, **kwargs):
        with self.env, self.host_function:
            self.config_spec.validate_compiler_coverage_groups()
            ConfigGeneration(self.config_spec).initial_population_view()
        raise ReachedSearch

    with (
        patch.object(BoundKernel, "autotune", search),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        pytest.raises(ReachedSearch),
    ):
        wrapper(*args)


def test_optional_failure_does_not_change_schema_or_default_or_RNG(tmp_path):
    def fail(env, ir):
        spec = env.config_spec
        assert fields(spec) == spec._compiler_coverage_fields
        raise RuntimeError("optional discovery fails after immutable schema declared")

    with patch.object(shared_gemm_seeds, "shared_gemm_seed_configs", fail):
        _, bound, _ = bind_inductor(tmp_path)
    assert bound.config_spec.compiler_seed_configs == []
    before_python = random.getstate()
    before_torch = torch.get_rng_state().clone()
    with bound.env, bound.host_function:
        spec = bound.config_spec
        spec.validate_compiler_coverage_groups()
        generation = ConfigGeneration(spec)
        generation.initial_population_view()
        pid = next(
            f
            for identity, f in zip(
                generation._flat_coordinate_identities(),
                generation.flat_spec,
                strict=True,
            )
            if identity[0] == "pid_type"
        )
        from helion.autotuner.config_fragment import EnumFragment

        assert isinstance(pid, EnumFragment)
        assert pid.search_values(limit=16) == ["flat"]
        assert pid.choices == ("flat", "persistent_interleaved")
        model = next(
            f
            for identity, f in zip(
                generation._flat_coordinate_identities(),
                generation.flat_spec,
                strict=True,
            )
            if identity[0] == "tcgen05_persistence_model"
        )
        assert model.search_values(limit=16) == ["non_persistent"]
        assert spec.default_config()["pid_type"] == "flat"
        assert spec.default_config()["tcgen05_persistence_model"] == "non_persistent"
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_actual_unrelated_field_change_still_fails_coverage_invariant(tmp_path):
    _, bound, _ = bind_inductor(tmp_path)
    with bound.env, bound.host_function:
        bound.config_spec.validate_compiler_coverage_groups()
        previous = bound.config_spec.allowed_pid_types
        bound.config_spec.allowed_pid_types = ("flat", "persistent_blocked")
        try:
            with pytest.raises(
                ValueError,
                match="Config fields changed after compiler coverage registration",
            ):
                ConfigGeneration(bound.config_spec).initial_population_view()
        finally:
            bound.config_spec.allowed_pid_types = previous


def test_all48_original_native_bodies_preserve_two_C_rings_unrounded_second_output(
    tmp_path,
):
    _, bound, _ = bind_inductor(tmp_path)
    with bound.env, bound.host_function:
        generation = ConfigGeneration(bound.config_spec)
        pairs = generation.seed_flat_config_pairs()
        ConfigGeneration(bound.config_spec).initial_population_view()
    for _, config in pairs:
        assert bound._normalized_config_copy(config) == config
        source = bound.to_code(config)
        assert source.count("PipelineTmaStore.create(") == 2
        assert source.count("producer_commit()") == source.count("producer_tail()") == 2
        assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
        assert "tcgen05_dag_step_17 * tcgen05_dag_step_8" in source
        assert "workspace" not in source.lower()


def test_encoding_requires_whole_existing_shared_and_all_ring_facts(tmp_path):
    _, bound, _ = bind_inductor(tmp_path)
    state = bound.config_spec._cute_tcgen05_config
    registry = bound.config_spec.compiler_seed_configs
    old = registry.copy()
    registry.clear()
    try:
        assert state._shared_gemm_seed_encoding_pid_types() == (
            "persistent_interleaved",
        )
        with bound.env, bound.host_function:
            with patch.object(state, "epilogue_fanout_plans", ()):
                shared_gemm_seeds.register_shared_gemm_seed_encoding(
                    bound.env, bound.host_function.device_ir
                )
                assert state._shared_gemm_seed_encoding_pid_types() == ()
            facts = state.rowvec_aux_facts
            assert facts is not None
            with patch.object(
                state, "rowvec_aux_facts", facts._replace(output_count=1)
            ):
                shared_gemm_seeds.register_shared_gemm_seed_encoding(
                    bound.env, bound.host_function.device_ir
                )
                assert state._shared_gemm_seed_encoding_pid_types() == ()
            shared_gemm_seeds.register_shared_gemm_seed_encoding(
                bound.env, bound.host_function.device_ir
            )
            assert state._shared_gemm_seed_encoding_pid_types() == (
                "persistent_interleaved",
            )
    finally:
        registry[:] = old
    assert state._shared_gemm_seed_encoding_pid_types() == ()


def test_fresh_JSON_config_bind_replay_and_search_schema_remain_identical(tmp_path):
    import json

    wrapper, bound, args = bind_inductor(tmp_path)
    with bound.env, bound.host_function:
        pairs = ConfigGeneration(bound.config_spec).seed_flat_config_pairs()
    config = pairs[-1][1]
    decoded = helion.Config.from_dict(json.loads(config.to_json()))
    assert isinstance(wrapper.kernel, helion.Kernel)
    kernel = helion.kernel(
        wrapper.kernel.fn,
        config=decoded,
        settings=wrapper.kernel.settings,
        cute_structural_policy=wrapper.kernel.cute_structural_policy,
    )
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        cold = cpu_bind(kernel, args)
    assert cold is not bound
    with cold.env, cold.host_function:
        cold.config_spec.validate_compiler_coverage_groups()
        ConfigGeneration(cold.config_spec).initial_population_view()
    assert cold._normalized_config_copy(decoded) == config
    assert ast.dump(
        ast.parse(cold.to_code(decoded)), include_attributes=False
    ) == ast.dump(ast.parse(bound.to_code(config)), include_attributes=False)


def test_seed_commit_changes_no_fields_defaults_or_random_stream(tmp_path):
    captured = []
    original = heuristics.append_shared_gemm_seed_configs

    def observe(env, ir):
        spec = env.config_spec
        before = fields(spec)
        default = spec.default_config()
        generation = ConfigGeneration(spec)
        previous = random.getstate()
        random.seed(122314)
        samples_before = [generation.random_flat() for _ in range(8)]
        rng_before = random.getstate()
        original(env, ir)
        random.seed(122314)
        generation_after = ConfigGeneration(spec)
        samples_after = [generation_after.random_flat() for _ in range(8)]
        rng_after = random.getstate()
        random.setstate(previous)
        assert rng_before == rng_after and samples_before == samples_after
        assert fields(spec) == before and spec.default_config() == default
        captured.append(len(spec.compiler_seed_configs))

    with patch.object(heuristics, "append_shared_gemm_seed_configs", observe):
        bind_inductor(tmp_path)
    assert captured == [48]


def test_early_optional_proof_failure_stays_noop_after_late_proof_recovers(tmp_path):
    original = shared_gemm_seeds.prove_shared_gemm_seed_facts
    with patch.object(
        shared_gemm_seeds,
        "prove_shared_gemm_seed_facts",
        side_effect=RuntimeError("early optional proof unavailable"),
    ):
        _, bound, _ = bind_inductor(tmp_path)
    state = bound.config_spec._cute_tcgen05_config
    assert not state._shared_gemm_seed_encoding_available
    assert bound.config_spec.compiler_seed_configs == []
    with bound.env, bound.host_function:
        before = fields(bound.config_spec)
        bound.config_spec.validate_compiler_coverage_groups()
        assert original(bound.env, bound.host_function.device_ir) is not None
        with patch.object(
            shared_gemm_seeds,
            "shared_gemm_seed_configs",
            side_effect=AssertionError(
                "late recovery must not alter the frozen encoding"
            ),
        ) as discovery:
            shared_gemm_seeds.append_shared_gemm_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            discovery.assert_not_called()
        assert fields(bound.config_spec) == before
        ConfigGeneration(bound.config_spec).initial_population_view()
    assert bound.config_spec.compiler_seed_configs == []


def test_legacy_shared_plan_without_full_flat_shared_field_gets_no_encoding():
    from test.test_cute_mapped_aux_codegen import _mapped_rows
    from test.test_cute_mapped_aux_codegen import _row_args

    bound = cpu_bind(_mapped_rows, _row_args(torch.bfloat16))
    state = bound.config_spec._cute_tcgen05_config
    assert state.epilogue_fanout_plans
    assert not state.epilogue_fanout_search_enabled
    assert not state._shared_gemm_seed_encoding_available
    assert state._shared_gemm_seed_encoding_pid_types() == ()
