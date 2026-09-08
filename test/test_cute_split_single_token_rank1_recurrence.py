from __future__ import annotations

from contextlib import ExitStack
import dataclasses
import importlib
import operator
from types import SimpleNamespace
from typing import Any
from typing import Literal
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language._tracing_ops import _get_symnode

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])
_split_single_token_rank1_recurrence = importlib.import_module(
    "helion._compiler.cute.split_single_token_rank1_recurrence"
)
_is_direct_scalar_source = _split_single_token_rank1_recurrence._is_direct_scalar_source
_linear_offsets_fit_i32 = _split_single_token_rank1_recurrence._linear_offsets_fit_i32
plan_split_single_token_rank1_recurrence = (
    _split_single_token_rank1_recurrence.plan_split_single_token_rank1_recurrence
)

_LOG2_E = 1.4426950408889634
_NORM_EPSILON = 1.0e-6


def _split_t1_body(
    first_vector: torch.Tensor,
    second_vector: torch.Tensor,
    values: torch.Tensor,
    gate_source: torch.Tensor,
    update_logits: torch.Tensor,
    log_decay_rate: torch.Tensor,
    gate_bias: torch.Tensor,
    checkpoint_pool: torch.Tensor,
    checkpoint_ids: torch.Tensor,
    accepted_counts: torch.Tensor,
    result: torch.Tensor,
    projection_scale: float,
    decay_floor: float,
    token_count: hl.constexpr,
    use_lower_bound: hl.constexpr,
    beta_is_logit: hl.constexpr,
    subtract_projection: hl.constexpr,
    softplus_threshold: hl.constexpr,
    shrink_sequence_axis: hl.constexpr,
) -> torch.Tensor:
    sequences = hl.specialize(checkpoint_ids.size(0))
    heads = hl.specialize(first_vector.size(1))
    width = hl.specialize(first_vector.size(2))
    value_width = hl.specialize(values.size(2))
    hl.specialize(
        (
            first_vector.stride(),
            second_vector.stride(),
            values.stride(),
            gate_source.stride(),
            update_logits.stride(),
            log_decay_rate.stride(),
            gate_bias.stride(),
            checkpoint_pool.stride(),
            checkpoint_ids.stride(),
            accepted_counts.stride(),
            result.stride(),
        )
    )
    task_sequences = sequences - 1 if shrink_sequence_axis else sequences
    for sequence_tile, head_tile, value_tile in hl.tile(
        [task_sequences, heads, value_width], block_size=[1, 1, None]
    ):
        sequence = sequence_tile.id
        head = head_tile.id
        offsets = hl.arange(width)
        accepted_index = torch.clamp(
            accepted_counts[sequence].long() - 1,
            min=0,
            max=token_count - 1,  # pyrefly: ignore[unsupported-operation]
        )
        initial_slot = checkpoint_ids[sequence, accepted_index].long()
        recurrent = checkpoint_pool[
            initial_slot, head, value_tile.index, offsets
        ].float()

        query = first_vector[sequence, head, offsets].float()
        key = second_vector[sequence, head, offsets].float()
        query = query * torch.rsqrt((query * query).sum() + _NORM_EPSILON)
        key = key * torch.rsqrt((key * key).sum() + _NORM_EPSILON)

        gate = gate_source[sequence, head, offsets].float()
        gate_input = gate + gate_bias[head * width + offsets].float()
        decay_parameter = torch.exp2(log_decay_rate[head].float() * _LOG2_E)
        if use_lower_bound:
            log_decay = decay_floor * torch.sigmoid(decay_parameter * gate_input)
        else:
            gate_exp = torch.exp2(gate_input * _LOG2_E)
            softplus = torch.where(
                gate_input <= softplus_threshold,  # pyrefly: ignore[unsupported-operation]
                torch.log(1.0 + gate_exp),
                gate_input,
            )
            log_decay = -decay_parameter * softplus
        recurrent = recurrent * torch.exp2(log_decay * _LOG2_E)[None, :]

        beta_value = update_logits[sequence, head].float()
        if beta_is_logit:
            beta_value = torch.sigmoid(beta_value)
        value = values[sequence, head, value_tile].float()
        prediction = (recurrent * key[None, :]).sum(-1)
        if subtract_projection:
            residual = value - prediction
        else:
            residual = value + prediction
        recurrent = recurrent + (beta_value * residual)[:, None] * key[None, :]

        projected = (recurrent * query[None, :]).sum(-1) * projection_scale
        result[sequence, head, value_tile] = projected.to(result.dtype)
        checkpoint_slot = checkpoint_ids[sequence, 0].long()
        checkpoint_pool[checkpoint_slot, head, value_tile.index, offsets] = recurrent
    return result


_fast_split_t1 = helion.kernel(
    _split_t1_body,
    backend="cute",
    static_shapes=False,
    fast_math=True,
)
_precise_split_t1 = helion.kernel(
    _split_t1_body,
    backend="cute",
    static_shapes=False,
    fast_math=False,
)


def _to_cuda_fake_tensor(_backend: object, tensor: torch.Tensor) -> torch.Tensor:
    assert isinstance(tensor, FakeTensor)
    with torch._C._DisableTorchDispatch():
        elem = torch.empty_strided(
            tensor.size(),
            tensor.stride(),
            dtype=tensor.dtype,
            device="meta",  # @ignore-device-lint
        )
    return FakeTensor(tensor.fake_mode, elem, torch.device("cuda"))


def _inputs(
    *,
    width: int = 128,
    state_dtype: torch.dtype = torch.bfloat16,
    unaligned_query: bool = False,
    alias_output: bool = False,
) -> tuple[object, ...]:
    sequences = 2
    heads = 4
    vector_shape = (sequences, heads, width)
    query = torch.empty(vector_shape, dtype=torch.bfloat16)
    if unaligned_query:
        query = torch.empty((sequences, heads, width + 1), dtype=torch.bfloat16)[
            :, :, 1:
        ]
    state = torch.empty((2 * sequences + 1, heads, width, width), dtype=state_dtype)
    output = torch.empty(vector_shape, dtype=torch.bfloat16)
    if alias_output:
        output = torch.as_strided(
            state,
            vector_shape,
            (heads * width, width, 1),
        )
    return (
        query,
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape[:2], dtype=torch.bfloat16),
        torch.empty((heads,), dtype=torch.float32),
        torch.empty((heads * width,), dtype=torch.float32),
        state,
        torch.empty((sequences, 1), dtype=torch.int32),
        torch.empty((sequences,), dtype=torch.int32),
        output,
    )


def _config(
    block_v: int = 16,
    pid_type: Literal[
        "flat", "persistent_blocked", "persistent_interleaved", "xyz"
    ] = "flat",
) -> helion.Config:
    return helion.Config(
        block_sizes=[block_v],
        num_warps=1,
        num_stages=1,
        indexing="pointer",
        pid_type=pid_type,
    )


def _compile_with_production_hooks(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config | None,
    capability: tuple[int, int] = (10, 3),
) -> str:
    """Compile through the production planner, codegen, and launcher hooks."""

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "helion.runtime.kernel._find_device",
                return_value=torch.device("cuda"),
            )
        )
        stack.enter_context(
            patch(
                "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
                new=_to_cuda_fake_tensor,
            )
        )
        stack.enter_context(
            patch(
                "helion.runtime.kernel.target_device_capability",
                return_value=capability,
            )
        )
        stack.enter_context(
            patch(
                "helion._compiler.compile_environment.target_device_capability",
                return_value=capability,
            )
        )
        stack.enter_context(
            patch("helion.language.loops.use_tileir_tunables", return_value=False)
        )
        stack.enter_context(
            patch(
                "helion.language.loops._supports_warp_specialize",
                return_value=capability >= (10, 0),
            )
        )
        stack.enter_context(
            patch(
                "helion._compat._supports_tensor_descriptor",
                return_value=capability >= (9, 0),
            )
        )
        stack.enter_context(
            patch("helion._compat._min_dot_size", return_value=(16, 16, 16))
        )
        stack.enter_context(patch("helion._compat._is_hip", return_value=False))
        bound = kernel._bind_isolated(args)
        if config is None:
            matching_seeds = [
                seed
                for seed in bound.config_spec.compiler_seed_configs
                if seed.config.get("block_sizes") == [16]
                and seed.config.get("pid_type") == "flat"
            ]
            if not matching_seeds:
                raise AssertionError("no flat 16-row compiler seed")
            config = matching_seeds[0]
        return bound.to_triton_code(config)


class TestCuteSplitSingleTokenRank1Recurrence(unittest.TestCase):
    def _code(
        self,
        *,
        scale: float = 128**-0.5,
        epsilon: float = 1.0e-6,
        token_count: int = 1,
        use_lower_bound: bool = False,
        beta_is_logit: bool = True,
        subtract_projection: bool = True,
        softplus_threshold: float = 20.0,
        shrink_sequence_axis: bool = False,
        width: int = 128,
        state_dtype: torch.dtype = torch.bfloat16,
        unaligned_query: bool = False,
        alias_output: bool = False,
        block_v: int = 16,
        pid_type: Literal[
            "flat", "persistent_blocked", "persistent_interleaved", "xyz"
        ] = "flat",
        capability: tuple[int, int] = (10, 3),
        fast_math: bool = True,
        use_compiler_seed: bool = False,
    ) -> str:
        kernel = _fast_split_t1 if fast_math else _precise_split_t1
        args = (
            *_inputs(
                width=width,
                state_dtype=state_dtype,
                unaligned_query=unaligned_query,
                alias_output=alias_output,
            ),
            scale,
            -5.0,
            token_count,
            use_lower_bound,
            beta_is_logit,
            subtract_projection,
            softplus_threshold,
            shrink_sequence_axis,
        )
        with patch(
            "test.test_cute_split_single_token_rank1_recurrence._NORM_EPSILON",
            epsilon,
        ):
            return _compile_with_production_hooks(
                kernel,
                args,
                None if use_compiler_seed else _config(block_v, pid_type),
                capability,
            )

    def test_structural_match_emits_packed_one_warp_kernel(self) -> None:
        code = self._code()
        self.assertIn("split_t1_codegen_abi_version = 1", code)
        self.assertIn("split_t1_rank1_helper_abi_version = 4", code)
        self.assertIn("_cute_rank1_pack_bf16x2", code)
        self.assertIn("_cute_rank1_mul_bf16x2", code)
        self.assertIn("_cute_rank1_add_bf16x2", code)
        self.assertIn("_cute_rank1_fma_bf16x2", code)
        self.assertIn("_cute_rank1_load_u32x2_nc", code)
        self.assertIn("_cute_rank1_load_u32x4_if_valid", code)
        self.assertIn("_cute_rank1_store_u32x4_if_valid", code)
        self.assertNotIn("_cute_rank1_store_u16_or_zero", code)
        self.assertIn("cutlass.Uint32(split_t1_output_element_offset)).store(", code)
        self.assertIn("split_t1_state_bits = cute.make_rmem_tensor((8, 4)", code)
        self.assertIn("block=(32, 1, 1)", code)
        self.assertIn("split_t1_work % cutlass.Int32(8)", code)
        self.assertIn("split_t1_gate_exp_low = cute.exp", code)
        self.assertIn("split_t1_beta = cutlass.Float32(1.0) /", code)

    def test_generated_locals_do_not_shadow_tensor_arguments(self) -> None:
        from helion._compiler.cute import single_token_rank1_recurrence

        original = single_token_rank1_recurrence._host_tensor_ref

        def colliding_name(node: object) -> object:
            ref = original(node)
            if ref is not None and ref.suggested_name == "first_vector":
                return dataclasses.replace(ref, suggested_name="split_t1_work")
            return ref

        with patch.object(
            single_token_rank1_recurrence,
            "_host_tensor_ref",
            side_effect=colliding_name,
        ):
            code = self._code()
        self.assertIn("split_t1_work,", code)
        self.assertIn(
            "split_t1_work_1 = cutlass.Int32(cute.arch.block_idx()[0])",
            code,
        )
        self.assertIn("split_t1_work.iterator", code)
        self.assertNotIn(
            "split_t1_work = cutlass.Int32(cute.arch.block_idx()[0])",
            code,
        )
        self.assertNotIn("split_t1_work_1.iterator", code)

    def test_exact_benchmark_body_matches(self) -> None:
        from benchmarks.cute.compare_kda_recurrent_backends import (
            _helion_recurrent_kda_body,
        )

        sequences = 15
        heads = 32
        width = 128
        vector_shape = (sequences, heads, width)
        kernel = helion.kernel(
            _helion_recurrent_kda_body,
            backend="cute",
            static_shapes=False,
            fast_math=True,
        )
        code = _compile_with_production_hooks(
            kernel,
            (
                torch.empty(vector_shape, dtype=torch.bfloat16),
                torch.empty(vector_shape, dtype=torch.bfloat16),
                torch.empty(vector_shape, dtype=torch.bfloat16),
                torch.empty(vector_shape, dtype=torch.bfloat16),
                torch.empty(vector_shape[:2], dtype=torch.bfloat16),
                torch.empty((heads,), dtype=torch.float32),
                torch.empty((heads * width,), dtype=torch.float32),
                torch.empty(
                    (2 * sequences + 1, heads, width, width),
                    dtype=torch.bfloat16,
                ),
                torch.empty((sequences, 1), dtype=torch.int32),
                torch.empty((sequences,), dtype=torch.int32),
                torch.empty(vector_shape, dtype=torch.bfloat16),
                width**-0.5,
                0.0,
                1,
                True,
                False,
                True,
            ),
            _config(),
        )
        self.assertIn("split_t1_codegen_abi_version = 1", code)
        self.assertIn("block=(32, 1, 1)", code)

    def test_generic_compiler_seed_activates_split_t1(self) -> None:
        code = self._code(use_compiler_seed=True)
        self.assertIn("split_t1_codegen_abi_version = 1", code)
        self.assertIn("block=(32, 1, 1)", code)

    def test_scale_and_epsilon_sources_are_captured_as_constexprs(self) -> None:
        code = self._code()
        self.assertIn("projection_scale: cutlass.Constexpr", code)
        self.assertIn("_source_module_attr__NORM_EPSILON: cutlass.Constexpr", code)
        self.assertIn("cutlass.Float32(projection_scale)", code)
        self.assertIn("cutlass.Float32(_source_module_attr__NORM_EPSILON)", code)

    def test_scale_source_must_be_direct(self) -> None:
        graph = torch.fx.Graph()
        source = graph.call_function(_get_symnode, ("scale",))
        expression = graph.call_function(operator.mul, (source, source))
        self.assertTrue(_is_direct_scalar_source(source))
        self.assertFalse(_is_direct_scalar_source(expression))

    def test_all_semantic_scalars_are_in_constexpr_cache_key(self) -> None:
        scalars = [object() for _ in range(4)]
        plan = SimpleNamespace(
            scale=scalars[0],
            log2_e=scalars[1],
            epsilon=scalars[2],
            softplus_threshold=scalars[3],
        )
        device_function = Mock()
        with (
            patch(
                "helion._compiler.cute.split_single_token_rank1_recurrence._plan_split_single_token_rank1_recurrence",
                return_value=plan,
            ),
            patch(
                "helion._compiler.device_function.DeviceFunction.current",
                return_value=device_function,
            ),
        ):
            result = plan_split_single_token_rank1_recurrence([], cast("Any", object()))
        self.assertIs(result, plan)
        self.assertEqual(
            device_function.promote_expr_arg_to_constexpr.call_args_list,
            [call(value) for value in scalars],
        )

    def test_changed_epsilon_fails_closed(self) -> None:
        code = self._code(epsilon=1.0e-5)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_pre_blackwell_target_fails_closed(self) -> None:
        code = self._code(capability=(9, 0))
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_precise_math_policy_fails_closed(self) -> None:
        code = self._code(fast_math=False)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_non_flat_config_fails_closed(self) -> None:
        code = self._code(pid_type="xyz")
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_non_16_value_tile_fails_closed(self) -> None:
        code = self._code(block_v=32)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_non_t1_graph_fails_closed(self) -> None:
        code = self._code(token_count=2)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_bounded_gate_fails_closed(self) -> None:
        code = self._code(use_lower_bound=True)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_non_logit_beta_fails_closed(self) -> None:
        code = self._code(beta_is_logit=False)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_changed_recurrence_sign_fails_closed(self) -> None:
        code = self._code(subtract_projection=False)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_changed_softplus_guard_fails_closed(self) -> None:
        code = self._code(softplus_threshold=19.0)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_non_128_width_fails_closed(self) -> None:
        code = self._code(width=64)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_non_bf16_state_fails_closed(self) -> None:
        code = self._code(state_dtype=torch.float32)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_unaligned_query_fails_closed(self) -> None:
        code = self._code(unaligned_query=True)
        self.assertNotIn("split_t1_codegen_abi_version", code)

    def test_aliasing_output_fails_closed(self) -> None:
        with self.assertRaises(helion.exc.BackendUnsupported):
            self._code(alias_output=True)

    def test_i32_offset_bounds_include_grid_and_every_tensor(self) -> None:
        max_i32 = torch.iinfo(torch.int32).max
        self.assertTrue(_linear_offsets_fit_i32(15 * 32 * 8, (1, 1024, max_i32 + 1)))
        self.assertFalse(_linear_offsets_fit_i32(15 * 32 * 8, (max_i32 + 2,)))
        self.assertFalse(_linear_offsets_fit_i32(max_i32 + 1, (1,)))

    def test_mismatched_logical_grid_extent_fails_closed(self) -> None:
        code = self._code(shrink_sequence_axis=True)
        self.assertNotIn("split_t1_codegen_abi_version", code)


if __name__ == "__main__":
    unittest.main()
