from __future__ import annotations

import ast
import math
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_search import _loop_search
from .test_cute_chained_residency_integration import _loop_cached
from .test_cute_chained_residency_search import _loop_residency
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._testing import DEVICE
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_UNROLL_KEY
from helion.autotuner.config_spec import EnumFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module")
def _unroll_bound() -> BoundKernel:
    with _cpu_codegen():
        return _loop_search._bind_isolated(
            (
                torch.empty((3, 128, 32), dtype=torch.bfloat16),
                torch.empty((3, 32, 128), dtype=torch.bfloat16),
                torch.empty((3, 32, 128), dtype=torch.bfloat16),
                torch.empty((128, 128), dtype=torch.float32),
            )
        )


@pytest.fixture
def unroll_bound(_unroll_bound: BoundKernel) -> Iterator[BoundKernel]:
    with _cpu_codegen():
        yield _unroll_bound


def _config(factor: object = 1, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128],
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            CUTE_CHAINED_POINTWISE_UNROLL_KEY: factor,
            **overrides,
        }
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_producers(left, right, initial):
    steps, m, k = left.shape
    n = right.shape[-1]
    output = torch.empty_like(initial)
    history = torch.empty((steps, m, n), dtype=torch.float32, device=left.device)
    for rows, cols, kk in hl.tile([m, n, k], block_size=[128, 48, k]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            product = hl.dot(
                left[step.id, rows, kk],
                right[step.id, kk, cols],
                out_dtype=torch.float32,
            )
            state = state * 0.5 + hl.dot(
                left[step.id, rows, kk],
                right[step.id, kk, cols],
                acc=product,
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


@pytest.mark.parametrize("k,factor", [(48, 8), (16, 8)])
def test_scalar_producer_residual_groups_keep_original_tail_predicate(
    k: int, factor: int
) -> None:
    with _cpu_codegen():
        bound = _residual_producers._bind_isolated(
            (
                torch.empty((3, 128, k), dtype=torch.bfloat16),
                torch.empty((3, k, 48), dtype=torch.bfloat16),
                torch.empty((128, 48), dtype=torch.float32),
            )
        )
        base = _config(num_warps=16, block_sizes=[])
        unrolled = _config(factor, num_warps=16, block_sizes=[])
        source = bound.to_code(unrolled)
        if k == 48:
            assert "chain_0_a_0_step in cutlass.range(12, unroll=8)" in source
            assert "chain_0_b_0_step in cutlass.range(5, unroll=5)" in source
            assert "if chain_0_b_0_index < 2304:" in source
        else:
            assert "chain_0_a_0_step in cutlass.range(4, unroll=4)" in source
            assert "chain_0_b_0_step in cutlass.range(2, unroll=2)" in source
            assert "if chain_0_b_0_index < 768:" in source
        assert _without_unroll(source) == _without_unroll(bound.to_code(base))


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
@pytest.mark.parametrize("trips", [0, 1, 2, 3, 6, 8, 9, 17])
def test_bounded_producer_factor_and_exact_activation(factor: int, trips: int) -> None:
    tracker = BoundedProducerUnroll(factor)
    actual = tracker.loop_factor(trips)
    assert actual == min(factor, max(1, trips))
    assert tracker.activated is (actual > 1)
    if factor > 1 and trips < 2:
        with pytest.raises(exc.BackendUnsupported, match="multi-trip loop operand"):
            tracker.validate()
    else:
        tracker.validate()


def test_bounded_activation_is_local_and_survives_single_trip() -> None:
    first, second = BoundedProducerUnroll(8), BoundedProducerUnroll(8)
    assert first.loop_factor(3) == 3
    assert first.loop_factor(1) == 1
    first.validate()
    with pytest.raises(exc.BackendUnsupported, match="multi-trip loop operand"):
        second.validate()


def test_root_complete_group_contract_is_unchanged() -> None:
    for trips in (2, 3, 6, 9):
        with pytest.raises(exc.BackendUnsupported, match="whole number"):
            PointwiseUnroll(8).loop_factor(trips)
    assert PointwiseUnroll(8).loop_factor(16) == 8
    assert PointwiseUnroll(2).loop_factor(3) == 2


def test_loop_unroll_search_roundtrip_and_override(unroll_bound: BoundKernel) -> None:
    spec = unroll_bound.config_spec
    assert spec.cute_chained_pointwise_unroll_search_enabled
    field = spec._flat_fields()[CUTE_CHAINED_POINTWISE_UNROLL_KEY]
    assert isinstance(field, EnumFragment)
    assert field.search_values(10) == [1, 2, 4, 8]
    generation = spec.create_config_generation()
    for factor in (1, 2, 4, 8):
        config = spec.normalized_config(_config(factor))
        assert config.config.get(CUTE_CHAINED_POINTWISE_UNROLL_KEY, 1) == factor
        assert generation.unflatten(generation.flatten(config)) == config
    default = spec.normalized_config(_config())
    assert CUTE_CHAINED_POINTWISE_UNROLL_KEY not in default.config
    override = spec.create_config_generation(
        overrides={CUTE_CHAINED_POINTWISE_UNROLL_KEY: 8}
    )
    assert (
        override.unflatten(override.flatten(default))[CUTE_CHAINED_POINTWISE_UNROLL_KEY]
        == 8
    )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("factor", [True, False, 0, 3, 16, None, 2.0, "2"])
def test_loop_unroll_strict_types_before_repair(
    unroll_bound: BoundKernel, factor: object, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 1, 2, 4 or 8"):
        unroll_bound.config_spec.normalize(_config(factor), _fix_invalid=repair)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "cp_async", "k_major"])
def test_loop_unroll_rejects_inactive_schedule_before_repair(
    unroll_bound: BoundKernel, schedule: str, repair: bool
) -> None:
    with pytest.raises(exc.InvalidConfig, match="loop pointwise unroll requires"):
        unroll_bound.config_spec.normalize(
            _config(8, cute_chained_mma_schedule=schedule), _fix_invalid=repair
        )
    canonical = unroll_bound.config_spec.normalized_config(
        _config(1, cute_chained_mma_schedule=schedule)
    )
    assert CUTE_CHAINED_POINTWISE_UNROLL_KEY not in canonical.config


def test_loop_unroll_rejects_unavailable_discovery(unroll_bound: BoundKernel) -> None:
    with (
        patch.object(
            unroll_bound.config_spec,
            "cute_chained_pointwise_unroll_search_enabled",
            False,
        ),
        pytest.raises(exc.InvalidConfig, match="loop pointwise unroll requires"),
    ):
        unroll_bound.config_spec.normalize(_config(2), _fix_invalid=True)


def _without_unroll(source: str) -> str:
    class StripUnroll(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if ast.unparse(node.func) == "cutlass.range":
                node.keywords = [kw for kw in node.keywords if kw.arg != "unroll"]
            return node

    return ast.dump(StripUnroll().visit(ast.parse(source)))


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("threads", [128, 512, 1024])
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_loop_unroll_changes_only_producer_annotations(
    unroll_bound: BoundKernel, vectorize: bool, threads: int, factor: int
) -> None:
    overrides = {
        "num_warps": threads // 32,
        "cute_chained_pointwise_vectorize": vectorize,
    }
    default = unroll_bound.to_code(_config(**overrides))
    implicit = _config(**overrides).config.copy()
    implicit.pop(CUTE_CHAINED_POINTWISE_UNROLL_KEY)
    assert default == unroll_bound.to_code(helion.Config.from_dict(implicit))
    source = unroll_bound.to_code(_config(factor, **overrides))
    assert _without_unroll(source) == _without_unroll(default)
    activated = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        if ast.unparse(node.iter.func) != "cutlass.range":
            continue
        unroll = next(
            (kw.value for kw in node.iter.keywords if kw.arg == "unroll"), None
        )
        if unroll is None or ast.literal_eval(unroll) == 1:
            continue
        name = ast.unparse(node.target)
        assert re.fullmatch(r"chain_\d+_[ab]_\d+(?:_vector)?_step", name), name
        trips = ast.literal_eval(node.iter.args[0])
        assert ast.literal_eval(unroll) == min(factor, trips)
        activated.append(name)
    assert activated
    assert any("_vector_step" in name for name in activated) is (
        vectorize and threads == 128
    )


def test_all_single_trip_vector_producers_reject_requested_unroll() -> None:
    with _cpu_codegen():
        right = torch.empty((3, 32, 16), dtype=torch.bfloat16).transpose(1, 2)
        bound = _loop_search._bind_isolated(
            (
                torch.empty((3, 128, 16), dtype=torch.bfloat16),
                right,
                right.clone(),
                torch.empty((128, 32), dtype=torch.float32),
            )
        )
        config = {
            "block_sizes": [32],
            "num_warps": 16,
            "cute_chained_pointwise_vectorize": True,
        }
        source = bound.to_code(_config(**config))
        assert "_vector_step in cutlass.range(1, unroll=1)" in source
        with pytest.raises(exc.BackendUnsupported, match="multi-trip loop operand"):
            bound.to_code(_config(8, **config))


def test_loop_unroll_seed_suffix_preserves_complete_scan_cache_prefix() -> None:
    with _cpu_codegen():
        bound = _loop_residency._bind_isolated(
            (
                torch.empty((3, 128, 16), dtype=torch.bfloat16),
                torch.empty((3, 16, 32), dtype=torch.bfloat16),
                torch.empty((128, 32), dtype=torch.float32),
            )
        )
        captured: list[list[helion.Config]] = []
        emitted: list[list[helion.Config]] = []
        original = CuteChainedMatmulHeuristic._with_loop_producer_unroll_seeds

        def record(
            env: CompileEnvironment, seeds: list[helion.Config]
        ) -> list[helion.Config]:
            captured.append(seeds)
            result = original(env, seeds)
            emitted.append(result)
            return result

        assert bound.host_function is not None
        with (
            bound.env,
            bound.host_function,
            patch.object(
                CuteChainedMatmulHeuristic,
                "_with_loop_producer_unroll_seeds",
                side_effect=record,
            ),
        ):
            seeds = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        assert seeds is not None and len(captured) == 1
        prefix = captured[0]
        seeds = emitted[0]
        assert all(a is b for a, b in zip(prefix, seeds, strict=False))
        suffix = seeds[len(prefix) :]
        assert len(suffix) == 2
        assert len(set(suffix)) == 2
        assert not set(prefix) & set(suffix)
        assert {
            seed.config.get("cute_chained_group_contractions", False) for seed in suffix
        } == {False, True}
        for seed in suffix:
            assert seed[CUTE_CHAINED_POINTWISE_UNROLL_KEY] == 8
            assert seed.num_warps == 16
            assert seed["cute_chained_pointwise_vectorize"] is True
            assert seed["cute_chained_scan_schedule"] == "warp"
            assert seed["cute_chained_pointwise_cache_bytes"] == 4096
            eligible = [
                parent
                for parent in prefix
                if parent.num_warps == 16
                and parent.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
                and parent.config.get("cute_chained_group_contractions", False)
                == seed.config.get("cute_chained_group_contractions", False)
            ]
            assert math.prod(seed.block_sizes) == max(
                math.prod(parent.block_sizes) for parent in eligible
            )
            bound.config_spec.normalized_config(seed)
        with patch.object(bound.config_spec, "cute_chunk_prefill_task_order", object()):
            assert original(bound.env, prefix) is prefix
        with patch.object(bound.config_spec, "cute_chained_loop_search_enabled", False):
            assert original(bound.env, prefix) is prefix


def _cached_vector_inputs(
    device: str | torch.device, dtype: torch.dtype, steps: int
) -> tuple[object, ...]:
    torch.manual_seed(934)
    count = max(steps, 1)
    tensors = [
        torch.randn((count, m, n), device=device, dtype=dtype) * 0.125
        for m, n in ((13, 32), (32, 128), (128, 37), (128, 37))
    ]
    # Contiguous reduction coordinates on RHS leaves admit actual vector
    # outer loops; the first A retains its element-varying mask fallback.
    tensors[1:] = [
        tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
        for tensor in tensors[1:]
    ]
    return (*tensors, torch.randn((13, 37), device=device) * 0.125, steps)


def _cached_vector_config(factor: int) -> helion.Config:
    return _config(
        factor,
        block_sizes=[],
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_scan_schedule="warp",
        cute_chained_scratch_layout="xor",
    )


@pytest.mark.parametrize("steps", [0, 3])
def test_unroll_vector_cache_scan_ragged_source_cpu(steps: int) -> None:
    with _cpu_codegen():
        bound = _loop_cached._bind_isolated(
            _cached_vector_inputs("cpu", torch.bfloat16, steps)
        )
        source = bound.to_code(_cached_vector_config(8))
        assert "_vector_step in cutlass.range(4, unroll=4)" in source
        assert "chain_pointwise_cache_0" in source
        assert "shuffle_sync_up" in source
        assert _without_unroll(source) == _without_unroll(
            bound.to_code(_cached_vector_config(1))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_unroll_residual_producers_ragged_gpu(dtype: torch.dtype) -> None:
    torch.manual_seed(935)
    args = (
        torch.randn((3, 141, 48), device=DEVICE, dtype=dtype) * 0.125,
        torch.randn((3, 48, 61), device=DEVICE, dtype=dtype) * 0.125,
        torch.randn((141, 61), device=DEVICE) * 0.125,
    )
    saved = tuple(arg.clone() for arg in args)
    bound = _residual_producers._bind_isolated(args)
    config = _config(8, num_warps=16, block_sizes=[])
    source = bound.to_code(config)
    assert "chain_0_a_0_step in cutlass.range(12, unroll=8)" in source
    assert "chain_0_b_0_step in cutlass.range(5, unroll=5)" in source
    original = bound.compile_config(_config(1, num_warps=16, block_sizes=[]))
    unrolled = _residual_producers._bind_isolated(args).compile_config(config)
    expected, actual = original(*args), unrolled(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(unrolled(*args), actual, rtol=0, atol=0)
    reference = args[2].clone()
    for step in range(3):
        product = args[0][step].float() @ args[1][step].float()
        reference = 0.5 * reference + product + product
        torch.testing.assert_close(actual[0][step], reference, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual[1], reference, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(args, saved, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 3])
def test_unroll_vector_cache_scan_ragged_gpu(dtype: torch.dtype, steps: int) -> None:
    args = _cached_vector_inputs(DEVICE, dtype, steps)
    tensors = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    saved = tuple(arg.clone() for arg in tensors)
    original = _loop_cached._bind_isolated(args).compile_config(
        _cached_vector_config(1)
    )
    unrolled = _loop_cached._bind_isolated(args).compile_config(
        _cached_vector_config(8)
    )
    expected, actual = original(*args), unrolled(*args)
    repeated = unrolled(*args)
    if steps == 0:
        # The unused history slot has no initialized value in a zero-trip loop.
        expected, actual, repeated = expected[1], actual[1], repeated[1]
        torch.testing.assert_close(actual, args[4], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(tensors, saved, rtol=0, atol=0)
