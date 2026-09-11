from __future__ import annotations

from contextlib import ExitStack
import dataclasses
import importlib
import inspect
import itertools
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])
_fixed_token_rank1_recurrence = importlib.import_module(
    "helion._compiler.cute.fixed_token_rank1_recurrence"
)
_MAX_TOKENS = _fixed_token_rank1_recurrence._MAX_TOKENS
_MIN_TOKENS = _fixed_token_rank1_recurrence._MIN_TOKENS

if TYPE_CHECKING:
    from collections.abc import Callable

_LOG2_E = 1.4426950408889634


def _fixed_rank1_body(
    first_vector: torch.Tensor,
    second_vector: torch.Tensor,
    values: torch.Tensor,
    gate_source: torch.Tensor,
    update_weight: torch.Tensor,
    log_decay_rate: torch.Tensor,
    gate_bias: torch.Tensor,
    checkpoint_pool: torch.Tensor,
    checkpoint_ids: torch.Tensor,
    accepted_counts: torch.Tensor,
    result: torch.Tensor,
    projection_scale: float,
    decay_floor: float,
    token_count: hl.constexpr,
    epsilon: hl.constexpr,
    subtract_projection: hl.constexpr,
    beta_is_logit: hl.constexpr,
    reverse_checkpoints: hl.constexpr,
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
            update_weight.stride(),
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

        for token_index in hl.static_range(token_count):  # pyrefly: ignore[bad-argument-type]
            token = sequence * token_count + token_index  # pyrefly: ignore[unsupported-operation]
            query = first_vector[token, head, offsets].float()
            key = second_vector[token, head, offsets].float()
            query = query * torch.rsqrt(
                (query * query).sum() + epsilon  # pyrefly: ignore[unsupported-operation]
            )
            key = key * torch.rsqrt(
                (key * key).sum() + epsilon  # pyrefly: ignore[unsupported-operation]
            )

            gate = gate_source[token, head, offsets].float()
            gate = gate + gate_bias[head * width + offsets].float()
            decay_parameter = torch.exp2(log_decay_rate[head].float() * _LOG2_E)
            decay = torch.exp2(
                decay_floor * torch.sigmoid(decay_parameter * gate) * _LOG2_E
            )
            recurrent = recurrent * decay[None, :]

            beta_value = update_weight[token, head].float()
            if beta_is_logit:
                beta_value = torch.sigmoid(beta_value)
            value = values[token, head, value_tile].float()
            prediction = (recurrent * key[None, :]).sum(-1)
            if subtract_projection:
                residual = value - prediction
            else:
                residual = value + prediction
            recurrent = recurrent + (beta_value * residual)[:, None] * key[None, :]

            projected = (recurrent * query[None, :]).sum(-1) * projection_scale
            result[token, head, value_tile] = projected.to(result.dtype)
            if reverse_checkpoints:
                checkpoint_index = token_count - 1 - token_index  # pyrefly: ignore[unsupported-operation]
            else:
                checkpoint_index = token_index
            checkpoint_slot = checkpoint_ids[sequence, checkpoint_index].long()
            checkpoint_pool[checkpoint_slot, head, value_tile.index, offsets] = (
                recurrent
            )
    return result


_fixed_rank1 = helion.kernel(
    _fixed_rank1_body,
    backend="cute",
    static_shapes=False,
    fast_math=True,
)
_precise_fixed_rank1 = helion.kernel(
    _fixed_rank1_body,
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
    token_count: int,
    *,
    width: int = 128,
    state_dtype: torch.dtype = torch.bfloat16,
) -> tuple[object, ...]:
    sequences = 2
    heads = 4
    vectors = sequences * token_count
    vector_shape = (vectors, heads, width)
    return (
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape[:2], dtype=torch.bfloat16),
        torch.empty((heads,), dtype=torch.float32),
        torch.empty((heads * width,), dtype=torch.float32),
        torch.empty((vectors + 2, heads, width, width), dtype=state_dtype),
        torch.empty((sequences, token_count), dtype=torch.int32),
        torch.empty((sequences,), dtype=torch.int32),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        width**-0.5,
        -5.0,
    )


def _config(
    block_v: int = 32,
    *,
    num_warps: int = 2,
    loop_order: list[int] | None = None,
) -> helion.Config:
    return helion.Config(
        block_sizes=[block_v],
        loop_orders=None if loop_order is None else [loop_order],
        num_warps=num_warps,
        num_stages=1,
        indexing="pointer",
        pid_type="flat",
    )


class TestCuteFixedTokenRank1Recurrence(unittest.TestCase):
    def _code(
        self,
        token_count: int = 3,
        *,
        epsilon: float = 1.0e-6,
        subtract_projection: bool = True,
        beta_is_logit: bool = False,
        reverse_checkpoints: bool = False,
        width: int = 128,
        state_dtype: torch.dtype = torch.bfloat16,
        block_v: int = 32,
        num_warps: int = 2,
        loop_order: list[int] | None = None,
        capability: tuple[int, int] = (10, 3),
        fast_math: bool = True,
        shrink_sequence_axis: bool = False,
    ) -> str:
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
            kernel = _fixed_rank1 if fast_math else _precise_fixed_rank1
            bound = kernel._bind_isolated(
                (
                    *_inputs(token_count, width=width, state_dtype=state_dtype),
                    token_count,
                    epsilon,
                    subtract_projection,
                    beta_is_logit,
                    reverse_checkpoints,
                    shrink_sequence_axis,
                )
            )
            return bound.to_triton_code(
                _config(
                    block_v,
                    num_warps=num_warps,
                    loop_order=loop_order,
                )
            )

    def test_structural_match_emits_grouped_two_phase_kernel(self) -> None:
        code = self._code()
        self.assertIn("fixed_rank1_codegen_abi_version = 3", code)
        self.assertIn("fixed_rank1_rank1_helper_abi_version = 4", code)
        self.assertIn("fixed_rank1_key_split = 2", code)
        self.assertIn("fixed_rank1_value_split = 4", code)
        self.assertIn("fixed_rank1_decay_smem", code)
        self.assertIn("fixed_rank1_checkpoint_slots", code)
        self.assertIn("fixed_rank1_prediction_left", code)
        self.assertIn("fixed_rank1_prediction_right", code)
        self.assertIn("fixed_rank1_projected_left", code)
        self.assertIn("fixed_rank1_projected_right", code)
        self.assertIn("for fixed_rank1_token in cutlass.range_constexpr(3)", code)
        self.assertIn("cute.arch.sync_threads()", code)
        self.assertIn("block=(64, 1, 1)", code)
        self.assertIn("_helion_cute_preferred_smem_carveout = 25", code)
        self.assertIn("alignment=16", code)
        self.assertIn(
            "fixed_rank1_value_head = cutlass.Int32(cute.arch.block_idx()[0])",
            code,
        )
        self.assertIn(
            "fixed_rank1_sequence = cutlass.Int32(cute.arch.block_idx()[1])",
            code,
        )
        self.assertIn(
            "fixed_rank1_value_split_id = cutlass.Int32(cute.arch.block_idx()[2])",
            code,
        )
        self.assertIn("(4, 2, (128 + _BLOCK_SIZE_2 - 1) // _BLOCK_SIZE_2)", code)
        self.assertNotIn("fixed_rank1_work =", code)

    def test_num_warps_selects_key_split_and_launcher_block(self) -> None:
        for num_warps, key_split, threads in ((2, 2, 64), (4, 4, 128)):
            with self.subTest(num_warps=num_warps):
                code = self._code(num_warps=num_warps)
                self.assertIn(f"fixed_rank1_key_split = {key_split}", code)
                self.assertIn("fixed_rank1_value_split = 4", code)
                self.assertIn(f"block=({threads}, 1, 1)", code)
                if key_split == 4:
                    self.assertIn(
                        "fixed_rank1_prediction, 2",
                        code,
                        msg="KS4 must emit its second cross-key butterfly",
                    )
                else:
                    self.assertNotIn("fixed_rank1_prediction, 2", code)

    def test_loop_order_controls_all_xyz_axis_permutations(self) -> None:
        extent = {
            0: "2",
            1: "4",
            2: "(128 + _BLOCK_SIZE_2 - 1) // _BLOCK_SIZE_2",
        }
        logical_name = {0: "sequence", 1: "value_head", 2: "value_split_id"}
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                code = self._code(num_warps=4, loop_order=list(order))
                for logical_axis, name in logical_name.items():
                    self.assertIn(
                        f"fixed_rank1_{name} = cutlass.Int32("
                        f"cute.arch.block_idx()[{order.index(logical_axis)}])",
                        code,
                    )
                grid = ", ".join(extent[logical_axis] for logical_axis in order)
                self.assertIn(f"({grid})", code)
                self.assertIn("block=(128, 1, 1)", code)

    def test_unsupported_warp_count_fails_closed(self) -> None:
        code = self._code(num_warps=8)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_compiler_seed_exposes_winning_ks4_schedule(self) -> None:
        capability = (10, 3)
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
                    "helion.language.loops._supports_warp_specialize", return_value=True
                )
            )
            stack.enter_context(
                patch("helion._compat._supports_tensor_descriptor", return_value=True)
            )
            stack.enter_context(
                patch("helion._compat._min_dot_size", return_value=(16, 16, 16))
            )
            stack.enter_context(patch("helion._compat._is_hip", return_value=False))
            bound = _fixed_rank1._bind_isolated(
                (*_inputs(3), 3, 1.0e-6, True, False, False, False)
            )
            self.assertIn(
                "cute_fixed_token_rank1",
                bound.config_spec.autotuner_heuristics,
            )
            matching = [
                seed
                for seed in bound.config_spec.compiler_seed_configs
                if seed.config.get("block_sizes") == [32]
                and seed.config.get("num_warps") == 4
                and seed.config.get("num_stages") == 1
                and seed.config.get("indexing") == "pointer"
                and seed.config.get("pid_type") == "flat"
                and "loop_orders" not in seed.config
            ]
            self.assertEqual(len(matching), 1)
            code = bound.to_triton_code(matching[0])
        self.assertIn("fixed_rank1_key_split = 4", code)
        self.assertIn("fixed_rank1_value_split = 4", code)
        self.assertIn("block=(128, 1, 1)", code)
        self.assertIn(
            "fixed_rank1_value_head = cutlass.Int32(cute.arch.block_idx()[0])",
            code,
        )
        self.assertIn(
            "fixed_rank1_sequence = cutlass.Int32(cute.arch.block_idx()[1])",
            code,
        )

    def test_runtime_wrapper_applies_preferred_smem_carveout(self) -> None:
        from helion.runtime.cute import launcher

        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_preferred_smem_carveout = 25
        wrapper = launcher._create_cute_wrapper(cute_kernel, (), (128, 1, 1))
        source = inspect.getsource(cast("Callable[..., object]", wrapper))
        self.assertIn("preferred_smem_carveout=25", source)

    def test_negative_state_slots_use_predicated_unsigned_bounds(self) -> None:
        code = self._code()
        self.assertIn(
            "_cute_rank1_load_u32x4_if_valid(",
            code,
        )
        self.assertIn(
            "fixed_rank1_initial_slot, cutlass.Int64(8)",
            code,
        )
        self.assertIn(
            "_cute_rank1_store_u32x4_if_valid(",
            code,
        )
        self.assertIn(
            "fixed_rank1_checkpoint_slot, cutlass.Int64(8)",
            code,
        )
        self.assertNotIn(
            "if cutlass.Uint32(fixed_rank1_initial_slot)",
            code,
        )
        self.assertNotIn(
            "if cutlass.Uint32(fixed_rank1_checkpoint_slot)",
            code,
        )

    def test_accepted_count_subtraction_preserves_int64_semantics(self) -> None:
        code = self._code()
        self.assertIn(
            "fixed_rank1_accepted_index = cutlass.Int64(",
            code,
        )
        self.assertIn(
            ") - cutlass.Int64(1)",
            code,
        )
        self.assertIn(
            "cutlass.Int32(fixed_rank1_accepted_index)",
            code,
        )

    def test_oversized_linear_offsets_fail_closed(self) -> None:
        with patch(
            "helion._compiler.cute.fixed_token_rank1_recurrence._linear_offsets_fit_i32",
            return_value=False,
        ):
            code = self._code()
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_unsafe_physical_xyz_grid_fails_closed(self) -> None:
        with patch(
            "helion._compiler.cute.fixed_token_rank1_recurrence._xyz_grid_fits",
            return_value=False,
        ):
            code = self._code()
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_generated_locals_do_not_shadow_tensor_arguments(self) -> None:
        from helion._compiler.cute import single_token_rank1_recurrence

        original = single_token_rank1_recurrence._host_tensor_ref

        def colliding_name(node: object) -> object:
            ref = original(node)
            if ref is not None and ref.suggested_name == "first_vector":
                return dataclasses.replace(ref, suggested_name="fixed_rank1_value_head")
            return ref

        with patch.object(
            single_token_rank1_recurrence,
            "_host_tensor_ref",
            side_effect=colliding_name,
        ):
            code = self._code()
        self.assertIn("fixed_rank1_value_head,", code)
        self.assertIn(
            "fixed_rank1_value_head_1 = cutlass.Int32(cute.arch.block_idx()[0])",
            code,
        )
        self.assertIn("fixed_rank1_value_head[", code)
        self.assertNotIn(
            "fixed_rank1_value_head = cutlass.Int32(cute.arch.block_idx()[0])",
            code,
        )
        self.assertNotIn("fixed_rank1_value_head_1[", code)

    def test_token_range_is_admitted(self) -> None:
        for token_count in (2, 6):
            with self.subTest(token_count=token_count):
                code = self._code(token_count)
                self.assertIn(f"cutlass.range_constexpr({token_count})", code)

    def test_outside_token_range_fails_closed(self) -> None:
        self.assertEqual((_MIN_TOKENS, _MAX_TOKENS), (2, 6))
        code = self._code(1)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_changed_recurrence_sign_fails_closed(self) -> None:
        code = self._code(subtract_projection=False)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_beta_logits_fail_closed(self) -> None:
        code = self._code(beta_is_logit=True)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_changed_norm_epsilon_is_preserved(self) -> None:
        code = self._code(epsilon=1.0e-5)
        self.assertIn("fixed_rank1_codegen_abi_version", code)
        self.assertIn("cutlass.Float32(1e-05)", code)

    def test_changed_checkpoint_order_fails_closed(self) -> None:
        code = self._code(reverse_checkpoints=True)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_non_128_width_fails_closed(self) -> None:
        code = self._code(width=64)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_non_bf16_state_fails_closed(self) -> None:
        code = self._code(state_dtype=torch.float32)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_wrong_value_tile_fails_closed(self) -> None:
        code = self._code(block_v=64)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_pre_blackwell_target_fails_closed(self) -> None:
        code = self._code(capability=(9, 0))
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_precise_math_policy_fails_closed(self) -> None:
        code = self._code(fast_math=False)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)

    def test_mismatched_logical_grid_extent_fails_closed(self) -> None:
        code = self._code(shrink_sequence_axis=True)
        self.assertNotIn("fixed_rank1_codegen_abi_version", code)


if __name__ == "__main__":
    unittest.main()
