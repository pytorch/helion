from __future__ import annotations

from contextlib import ExitStack
import dataclasses
import importlib
import math
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

import helion
from helion._compiler.cute.fx_matcher import _xyz_grid_fits
from helion._testing import skipUnlessBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])
_single_token_rank1_recurrence = importlib.import_module(
    "helion._compiler.cute.single_token_rank1_recurrence"
)
_DECAY_TANH_POLY_COEFFICIENTS = (
    _single_token_rank1_recurrence._DECAY_TANH_POLY_COEFFICIENTS
)
_is_standard_key_iota = _single_token_rank1_recurrence._is_standard_key_iota

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_ir import DeviceIR

_LOG2_E = 1.4426950408889634


def _rank1_body(
    packed: torch.Tensor,
    gate_input: torch.Tensor,
    beta_input: torch.Tensor,
    log_a: torch.Tensor,
    gate_bias: torch.Tensor,
    query_scale: float,
    decay_bound: float,
    state_pool: torch.Tensor,
    result: torch.Tensor,
    slots: torch.Tensor,
    epsilon: hl.constexpr,
    subtract_projection: hl.constexpr,
    norm_alpha: hl.constexpr,
    narrow_state_index: hl.constexpr,
) -> torch.Tensor:
    batch = packed.size(0)
    value_heads = hl.specialize(state_pool.size(-3))
    value_width = hl.specialize(state_pool.size(-2))
    key_width = hl.specialize(state_pool.size(-1))
    query_heads = hl.specialize(
        (packed.size(1) - value_heads * value_width) // (2 * key_width)
    )
    heads_per_query = value_heads // query_heads
    hl.specialize(
        (
            packed.stride(0),
            packed.stride(1),
            gate_input.stride(0),
            gate_input.stride(1),
            beta_input.stride(0),
            beta_input.stride(1),
            log_a.stride(0),
            gate_bias.stride(0),
            state_pool.stride(0),
            state_pool.stride(1),
            state_pool.stride(2),
            state_pool.stride(3),
            result.stride(0),
            result.stride(1),
            result.stride(2),
            result.stride(3),
            slots.stride(0),
        )
    )
    block_v = hl.register_block_size(1, value_width)
    for tile_batch, tile_head, tile_value in hl.tile(
        [batch, value_heads, value_width], block_size=[1, 1, block_v]
    ):
        key_offset = hl.arange(key_width)
        batch_id = tile_batch.id
        value_head = tile_head.id
        query_head = value_head // heads_per_query
        if narrow_state_index:
            state_index = slots[batch_id].int().long()
        else:
            state_index = slots[batch_id].long()
        if state_index < 0:
            result[batch_id, 0, value_head, tile_value] = 0.0
        else:
            query_offset = query_head * key_width + key_offset
            key_input_offset = (
                query_heads * key_width + query_head * key_width + key_offset
            )
            value_offset = (
                2 * query_heads * key_width
                + value_head * value_width
                + tile_value.index
            )
            gate = gate_input[batch_id, value_head * key_width + key_offset].float()
            gate = gate + gate_bias[value_head * key_width + key_offset]
            a_value = torch.exp2(log_a[value_head].float() * _LOG2_E)
            decay = torch.exp2(decay_bound * torch.sigmoid(a_value * gate) * _LOG2_E)
            beta = torch.sigmoid(beta_input[batch_id, value_head].float())
            state = state_pool[
                state_index, value_head, tile_value.index, key_offset
            ].float()
            state = state * decay[None, :]
            key = packed[batch_id, key_input_offset].float()
            # pyrefly: ignore [unsupported-operation]
            key_norm = (key * key).sum()
            if norm_alpha == 1:
                key_norm = key_norm + epsilon  # pyrefly: ignore [unsupported-operation]
            else:
                key_norm = torch.add(  # pyrefly: ignore [no-matching-overload]
                    key_norm, epsilon, alpha=norm_alpha
                )
            key = key * torch.rsqrt(key_norm)
            value = packed[batch_id, value_offset].float()
            projection = (state * key[None, :]).sum(-1)
            if subtract_projection:
                residual = value - projection
            else:
                residual = value + projection
            residual = residual * beta
            state = state + residual[:, None] * key[None, :]
            query = packed[batch_id, query_offset].float()
            # pyrefly: ignore [unsupported-operation]
            query_norm = (query * query).sum()
            if norm_alpha == 1:
                query_norm = query_norm + epsilon  # pyrefly: ignore [unsupported-operation]
            else:
                query_norm = torch.add(  # pyrefly: ignore [no-matching-overload]
                    query_norm, epsilon, alpha=norm_alpha
                )
            query = query * torch.rsqrt(query_norm)
            query = query * query_scale
            result[batch_id, 0, value_head, tile_value] = (
                (state * query[None, :]).sum(-1).to(result.dtype)
            )
            state_pool[state_index, value_head, tile_value.index, key_offset] = state
    return result


_fast_rank1 = helion.kernel(
    _rank1_body,
    backend="cute",
    static_shapes=False,
    fast_math=True,
)
_precise_rank1 = helion.kernel(
    _rank1_body,
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
    value_width: int = 128,
    slot_dtype: torch.dtype = torch.int32,
    batch_size: int = 1,
    value_heads: int = 12,
    query_heads: int | None = None,
) -> tuple[object, ...]:
    query_heads = value_heads if query_heads is None else query_heads
    packed_width = 2 * query_heads * 128 + value_heads * value_width
    packed_stride = max(6144, packed_width)
    state_slot_width = value_heads * value_width * 128
    return (
        torch.empty_strided(
            (batch_size, packed_width), (packed_stride, 1), dtype=torch.bfloat16
        ),
        torch.empty((batch_size, value_heads * 128), dtype=torch.bfloat16),
        torch.empty((batch_size, value_heads), dtype=torch.bfloat16),
        torch.empty((value_heads,), dtype=torch.float32),
        torch.empty((value_heads * 128,), dtype=torch.float32),
        128**-0.5,
        -5.0,
        torch.empty_strided(
            (2, value_heads, value_width, 128),
            (state_slot_width + 256, value_width * 128, 128, 1),
            dtype=torch.bfloat16,
        ),
        torch.empty((batch_size, 1, value_heads, value_width), dtype=torch.bfloat16),
        torch.empty((batch_size,), dtype=slot_dtype),
    )


def _config(
    block_v: int = 16,
    num_warps: int = 8,
    num_threads: list[int] | None = None,
) -> helion.Config:
    return helion.Config(
        block_sizes=[block_v],
        num_threads=num_threads,
        num_warps=num_warps,
        num_stages=1,
        indexing="pointer",
        pid_type="flat",
    )


def _permute_root_task_axes(device_ir: DeviceIR) -> None:
    axes = list(device_ir.task_families[0].axes)
    axes[0], axes[1] = axes[1], axes[0]
    device_ir.task_families[0] = dataclasses.replace(
        device_ir.task_families[0],
        axes=tuple(axes),
    )


def _make_root_task_origin_noncanonical(device_ir: DeviceIR) -> None:
    axes = list(device_ir.task_families[0].axes)
    axes[0] = dataclasses.replace(axes[0], canonical_origin=False)
    device_ir.task_families[0] = dataclasses.replace(
        device_ir.task_families[0],
        axes=tuple(axes),
    )


def _shorten_root_task_extent(device_ir: DeviceIR) -> None:
    axes = list(device_ir.task_families[0].axes)
    extent = axes[1].extent
    assert extent is not None and not isinstance(extent, str)
    axes[1] = dataclasses.replace(
        axes[1],
        extent=extent - 1,  # pyrefly: ignore [unsupported-operation]
    )
    device_ir.task_families[0] = dataclasses.replace(
        device_ir.task_families[0],
        axes=tuple(axes),
    )


class TestCuteSingleTokenRank1Recurrence(unittest.TestCase):
    def _code(
        self,
        *,
        fast_math: bool = True,
        epsilon: float = 1.0e-6,
        subtract_projection: bool = True,
        value_width: int = 128,
        block_v: int = 16,
        alias_output: bool = False,
        unaligned_mixed: bool = False,
        slot_dtype: torch.dtype = torch.int32,
        capability: tuple[int, int] = (10, 3),
        norm_alpha: int = 1,
        narrow_state_index: bool = False,
        batch_size: int = 1,
        value_heads: int = 12,
        query_heads: int | None = None,
        num_warps: int = 8,
        num_threads: list[int] | None = None,
        mutate: Callable[[DeviceIR], None] | None = None,
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
            kernel = _fast_rank1 if fast_math else _precise_rank1
            inputs = list(
                _inputs(
                    value_width,
                    slot_dtype,
                    batch_size,
                    value_heads,
                    query_heads,
                )
            )
            if alias_output:
                state_input = inputs[7]
                assert isinstance(state_input, torch.Tensor)
                inputs[8] = torch.as_strided(
                    state_input,
                    (1, 1, value_heads, value_width),
                    (
                        value_heads * value_width,
                        value_heads * value_width,
                        value_width,
                        1,
                    ),
                )
            if unaligned_mixed:
                active_query_heads = value_heads if query_heads is None else query_heads
                width = 2 * active_query_heads * 128 + value_heads * value_width
                inputs[0] = torch.empty((width + 1,), dtype=torch.bfloat16)[1:].view(
                    1, width
                )
            bound = kernel._bind_isolated(
                (
                    *inputs,
                    epsilon,
                    subtract_projection,
                    norm_alpha,
                    narrow_state_index,
                )
            )
            if mutate is not None:
                assert bound.host_function is not None
                mutate(bound.host_function.device_ir)
            return bound.to_triton_code(
                _config(block_v, num_warps, num_threads=num_threads)
            )

    def test_structural_match_emits_packed_rank1_kernel(self) -> None:
        code = self._code()
        self.assertIn("rank1_state_offset", code)
        self.assertIn("rank1_helper_abi_version = 4", code)
        self.assertIn("_cute_rank1_fma_bf16x2", code)
        self.assertIn("_cute_rank1_load_i32_nc", code)
        self.assertIn("_cute_rank1_load_u32x2_nc", code)
        self.assertIn("_cute_rank1_load_f32x4_nc", code)
        self.assertIn("_cute_rank1_load_f32_nc", code)
        self.assertIn("_cute_rank1_load_u16_nc", code)
        self.assertIn("_cute_rank1_load_u32x4_if_valid", code)
        self.assertIn("_cute_rank1_store_u32x4_if_valid", code)
        self.assertIn("_cute_rank1_store_u16_or_zero", code)
        self.assertIn("block=(256, 1, 1)", code)
        self.assertIn(repr(1.0e-6 / 32.0), code)
        self.assertIn("rank1_gate_tanh_low", code)
        self.assertIn("-0.012315480037717208", code)
        self.assertNotIn("0.0028952701973490747", code)
        self.assertIn("rank1_global_row", code)
        self.assertNotIn("rank1_query_heads", code)
        self.assertIn("rank1_state_index, cutlass.Int64(2)", code)
        self.assertNotIn("state_pool_size_0", code)
        self.assertIn("query_scale: cutlass.Constexpr", code)
        self.assertIn("decay_bound: cutlass.Constexpr", code)
        self.assertIn("_source_module_attr__LOG2_E: cutlass.Constexpr", code)
        self.assertIn("if decay_bound == -5.0", code)
        self.assertNotIn("cutlass.Float32(decay_bound) ==", code)
        self.assertIn("cutlass.Int64", code)

    def test_batched_structural_match_emits_one_warp_kernel(self) -> None:
        code = self._code(batch_size=32, num_warps=1)
        self.assertIn("packed_rank1_codegen_abi_version = 2", code)
        self.assertIn("packed_rank1_helper_abi_version = 4", code)
        self.assertIn("packed_rank1_state_bits", code)
        self.assertIn(".iterator + packed_rank1_batch", code)
        self.assertIn("packed_rank1_batch * cutlass.Int32(6144)", code)
        self.assertIn(
            "packed_rank1_work = cutlass.Int32(cute.arch.block_idx()[0])", code
        )
        self.assertIn(
            "packed_rank1_batch = cutlass.Int32(cute.arch.block_idx()[1])", code
        )
        self.assertIn("packed_rank1_head = packed_rank1_work // cutlass.Int32(8)", code)
        self.assertIn(
            "packed_rank1_value_tile = packed_rank1_work % cutlass.Int32(8)", code
        )
        self.assertIn("(96, batch, 1)", code)
        self.assertIn("block=(32, 1, 1)", code)
        self.assertNotIn("rank1_global_row", code)

    def test_batched_one_warp_supports_eight_row_tiles(self) -> None:
        code = self._code(batch_size=32, block_v=8, num_warps=1)
        self.assertIn("packed_rank1_codegen_abi_version = 2", code)
        self.assertIn("(4, 4), cutlass.Uint32", code)
        self.assertIn("(192, batch, 1)", code)
        self.assertIn("block=(32, 1, 1)", code)

    def test_batched_one_warp_is_not_specific_to_batch_32(self) -> None:
        code = self._code(batch_size=64, num_warps=1)
        self.assertIn("packed_rank1_codegen_abi_version = 2", code)

    def test_b1_codegen_uses_static_non_twelve_head_count(self) -> None:
        code = self._code(value_heads=8)
        self.assertIn("rank1_state_offset", code)
        self.assertIn("rank1_key_input_base = cutlass.Int32(1024)", code)
        self.assertIn("rank1_value_input = cutlass.Int32(2048)", code)

    def test_batched_codegen_uses_static_non_twelve_head_count(self) -> None:
        code = self._code(batch_size=32, value_heads=8, num_warps=1)
        self.assertIn("packed_rank1_codegen_abi_version = 2", code)
        self.assertIn("packed_rank1_batch * cutlass.Int32(8)", code)
        self.assertIn("(64, batch, 1)", code)

    def test_grouped_query_layout_fails_closed(self) -> None:
        code = self._code(value_heads=12, query_heads=4)
        self.assertNotIn("rank1_state_offset", code)

    def test_batched_non_one_warp_config_fails_closed(self) -> None:
        code = self._code(batch_size=32, num_warps=4)
        self.assertNotIn("packed_rank1_codegen_abi_version", code)
        self.assertNotIn("rank1_state_offset", code)

    def test_batched_explicit_one_warp_threads_emit_packed_kernel(self) -> None:
        code = self._code(batch_size=32, num_threads=[1, 32])
        self.assertIn("packed_rank1_codegen_abi_version = 2", code)
        self.assertIn("block=(32, 1, 1)", code)

    def test_batched_conflicting_explicit_threads_fail_closed(self) -> None:
        code = self._code(batch_size=32, num_warps=1, num_threads=[2, 32])
        self.assertNotIn("packed_rank1_codegen_abi_version", code)

    def test_batched_unsafe_physical_xyz_grid_fails_closed(self) -> None:
        with patch(
            "helion._compiler.cute.single_token_rank1_recurrence._xyz_grid_fits",
            return_value=False,
        ):
            code = self._code(batch_size=2, num_warps=1)
        self.assertNotIn("packed_rank1_codegen_abi_version", code)

    def test_physical_xyz_grid_checks_each_architectural_limit(self) -> None:
        max_i32 = torch.iinfo(torch.int32).max
        with patch(
            "torch._inductor.runtime.triton_heuristics.get_max_y_grid",
            return_value=8,
        ):
            self.assertTrue(_xyz_grid_fits((max_i32, 7, 7)))
            self.assertFalse(_xyz_grid_fits((max_i32 + 1, 1, 1)))
            self.assertFalse(_xyz_grid_fits((1, 8, 1)))
            self.assertFalse(_xyz_grid_fits((1, 1, 8)))

    def test_generated_locals_do_not_shadow_tensor_arguments(self) -> None:
        from helion._compiler.cute import single_token_rank1_recurrence

        original = single_token_rank1_recurrence._host_tensor_ref

        for batch_size, num_warps, colliding, renamed in (
            (1, 8, "rank1_work", "rank1_work_1"),
            (2, 1, "packed_rank1_work", "packed_rank1_work_1"),
        ):
            with self.subTest(batch_size=batch_size):

                def colliding_name(node: object, collision: str = colliding) -> object:
                    ref = original(node)
                    if ref is not None and ref.suggested_name == "packed":
                        return dataclasses.replace(ref, suggested_name=collision)
                    return ref

                with patch.object(
                    single_token_rank1_recurrence,
                    "_host_tensor_ref",
                    side_effect=colliding_name,
                ):
                    code = self._code(
                        batch_size=batch_size,
                        num_warps=num_warps,
                    )
                self.assertIn(f"{colliding},", code)
                self.assertIn(
                    f"{renamed} = cutlass.Int32(cute.arch.block_idx()[0])",
                    code,
                )
                self.assertIn(f"{colliding}.iterator", code)
                self.assertNotIn(
                    f"{colliding} = cutlass.Int32(cute.arch.block_idx()[0])",
                    code,
                )
                self.assertNotIn(f"{renamed}.iterator", code)

    def test_b1_one_warp_config_preserves_existing_wide_body(self) -> None:
        code = self._code(batch_size=1, num_warps=1)
        self.assertIn("rank1_state_offset", code)
        self.assertIn("block=(256, 1, 1)", code)
        self.assertNotIn("packed_rank1_codegen_abi_version", code)

    def test_b1_changed_task_domain_fails_closed(self) -> None:
        for mutate in (
            _permute_root_task_axes,
            _make_root_task_origin_noncanonical,
            _shorten_root_task_extent,
        ):
            with self.subTest(mutate=mutate.__name__):
                code = self._code(mutate=mutate)
                self.assertNotIn("rank1_helper_abi_version", code)

    def test_batched_one_warp_compiler_seeds_and_metadata_fact(self) -> None:
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
                    return_value=(10, 3),
                )
            )
            stack.enter_context(
                patch(
                    "helion._compiler.compile_environment.target_device_capability",
                    return_value=(10, 3),
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
            inputs = (*_inputs(batch_size=32), 1.0e-6, True, 1, False)
            bound = _fast_rank1._bind_isolated(inputs)
            config_generation = bound.config_spec.create_config_generation(
                process_group_name=bound.env.process_group_name
            )
            transferred_packed_seeds: dict[int, tuple[helion.Config, str]] = {}
            for seed in bound.config_spec.compiler_seed_configs:
                if (
                    seed.config.get("num_warps") != 1
                    or seed.config.get("loop_orders") != [[2, 1, 0]]
                    or seed.config.get("block_sizes") not in ([8], [16])
                ):
                    continue
                transferred = config_generation.unflatten(
                    config_generation.flatten(seed)
                )
                tile_rows = transferred.block_sizes[0]
                transferred_packed_seeds[tile_rows] = (
                    transferred,
                    bound.to_triton_code(transferred),
                )

        self.assertIn(
            "cute_packed_single_token_rank1",
            bound.config_spec.autotuner_heuristics,
        )
        self.assertIn(
            "input_tensor_metadata",
            bound.env.compiler_fact_specialization_facts,
        )
        rank1_seeds = {
            (
                tuple(seed.block_sizes),
                seed.num_warps,
                tuple(tuple(order) for order in seed.loop_orders),
            )
            for seed in bound.config_spec.compiler_seed_configs
        }
        self.assertTrue(
            {
                ((16,), 1, ((2, 1, 0),)),
                ((8,), 1, ((2, 1, 0),)),
                ((16,), 8, ((2, 1, 0),)),
            }.issubset(rank1_seeds)
        )
        self.assertEqual(set(transferred_packed_seeds), {8, 16})
        for transferred, code in transferred_packed_seeds.values():
            self.assertEqual(transferred.num_threads, [1, 32])
            self.assertIn("packed_rank1_codegen_abi_version = 2", code)

    def test_int64_state_index_uses_wide_immutable_load(self) -> None:
        code = self._code(slot_dtype=torch.int64)
        self.assertIn("_cute_rank1_load_i64_nc", code)
        self.assertNotIn("_cute_rank1_load_i32_nc(ssm_state_indices.iterator)", code)

    def test_pre_blackwell_target_fails_closed(self) -> None:
        code = self._code(capability=(9, 0))
        self.assertNotIn("rank1_state_offset", code)

    def test_match_requires_fast_math_policy(self) -> None:
        code = self._code(fast_math=False)
        self.assertNotIn("rank1_state_offset", code)

    def test_changed_recurrence_sign_fails_closed(self) -> None:
        code = self._code(subtract_projection=False)
        self.assertNotIn("rank1_state_offset", code)

    def test_changed_norm_epsilon_fails_closed(self) -> None:
        code = self._code(epsilon=1.0e-5)
        self.assertNotIn("rank1_state_offset", code)

    def test_changed_norm_alpha_fails_closed(self) -> None:
        code = self._code(norm_alpha=2)
        self.assertNotIn("rank1_state_offset", code)

    def test_narrowed_state_index_fails_closed(self) -> None:
        code = self._code(slot_dtype=torch.int64, narrow_state_index=True)
        self.assertNotIn("rank1_state_offset", code)

    def test_iota_contract_checks_stride(self) -> None:
        graph = torch.fx.Graph()
        kwargs = {
            "start": 0,
            "step": 1,
            "dtype": torch.int32,
            "device": torch.device("cuda"),
            "requires_grad": False,
        }
        valid = graph.call_function(
            torch.ops.prims.iota.default,
            args=(128,),
            kwargs=kwargs,  # pyrefly: ignore [bad-argument-type]
        )
        invalid = graph.call_function(
            torch.ops.prims.iota.default,
            args=(128,),
            kwargs={**kwargs, "step": 2},
        )
        self.assertTrue(_is_standard_key_iota(valid))
        self.assertFalse(_is_standard_key_iota(invalid))

    def test_global_decay_polynomial_accuracy_and_bounds(self) -> None:
        inputs = [index / 100.0 for index in range(-2000, 2001)]
        inputs.extend((-100.0, 100.0))
        max_error = 0.0
        for value in inputs:
            transformed = math.tanh(0.5 * value)
            approximate = _DECAY_TANH_POLY_COEFFICIENTS[-1]
            for coefficient in reversed(_DECAY_TANH_POLY_COEFFICIENTS[:-1]):
                approximate = approximate * transformed + coefficient
            expected = math.exp(-5.0 / (1.0 + math.exp(-value)))
            max_error = max(max_error, abs(approximate - expected))
            self.assertGreaterEqual(approximate, 0.0)
            self.assertLessEqual(approximate, 1.0)
        self.assertLess(max_error, 3.0e-5)

    def test_non_128_value_width_fails_closed(self) -> None:
        code = self._code(value_width=64)
        self.assertNotIn("rank1_state_offset", code)

    def test_non_16_value_tile_fails_closed(self) -> None:
        code = self._code(block_v=32)
        self.assertNotIn("rank1_state_offset", code)

    def test_aliasing_output_fails_closed(self) -> None:
        with self.assertRaises(helion.exc.BackendUnsupported):
            self._code(alias_output=True)

    def test_unaligned_vector_input_fails_closed(self) -> None:
        code = self._code(unaligned_mixed=True)
        self.assertNotIn("rank1_state_offset", code)


if __name__ == "__main__":
    unittest.main()
