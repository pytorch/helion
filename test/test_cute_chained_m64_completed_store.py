from __future__ import annotations

from dataclasses import replace
import operator
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from . import test_cute_chained_fragment_reuse as reuse_tests
from ._cute_aux import _cpu_codegen
from .test_cute_chained_fragment_reuse import _config
from .test_cute_chained_fragment_reuse import _source
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_completed_store
from helion._compiler.cute import chained_final_carry_store
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_store_ownership
from helion._compiler.cute.chained_completed_members import plan_completed_member_map
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_store_ownership import _checked
from helion._compiler.cute.chained_store_ownership import _Index
from helion._compiler.cute.chained_store_ownership import _Selectors
from helion._compiler.cute.chained_tmem_drains import FP32DrainPanels
from helion._compiler.cute.chained_tmem_snapshots import PackedSnapshotPanels
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language import tile_index


def _selected(enabled: bool = True) -> str:
    def config(reuse: bool = True) -> helion.Config:
        return _config(reuse, cute_chained_completed_member_store=enabled)

    with patch.object(reuse_tests, "_config", config):
        return _source()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "width,offset,total", [(32, 0, 160), (64, 0, 192), (128, 64, 192)]
)
@pytest.mark.parametrize("transpose", [False, True])
def test_m64_separate_unique_native_map(
    dtype: torch.dtype, width: int, offset: int, total: int, transpose: bool
) -> None:
    shape = (width, 64) if transpose else (64, width)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        assert (
            plan_completed_member_map(
                (64, total), shape, offset, width, transpose, dtype
            )
            is None
        )
        native = plan_completed_member_map(
            (64, total), shape, offset, width, transpose, dtype, reordered=True
        )
    assert native is not None and not native.same_store_order
    visited = [
        native.coordinates[t][slot]
        for t, slots in enumerate(native.selected_slots)
        for slot in slots
    ]
    assert len(visited) == len(set(visited)) == 64 * width
    assert set(visited) == {
        (row, column) for row in range(64) for column in range(offset, offset + width)
    }


@pytest.mark.parametrize(
    "shape,width", [((63, 64), 64), ((64, 48), 48), ((64, 192), 96)]
)
def test_unsupported_native_geometry_does_not_gain_permission(
    shape: tuple[int, int], width: int
) -> None:
    assert (
        plan_completed_member_map(
            shape, (64, width), 0, width, False, torch.bfloat16, reordered=True
        )
        is None
    )


def test_existing_m64_bounded_panel_records_are_valid() -> None:
    # These records describe bounds; native emission/ownership is a separate
    # proof. M64 is no longer an invalid-shape example for either record.
    assert FP32DrainPanels((64, 128)).count == 4
    assert PackedSnapshotPanels((64, 128), 0, 256, 512).count == 4


def test_terminal_and_final_store_share_exact_original_protocol() -> None:
    allocations = []
    original = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result is not None
        allocations.append(result)
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        control, source = _selected(False), _selected()
    assert "chain_3_c[" in control and "chain_3_c[" not in source
    assert (
        "chain_final_store_0_step" in control
        and "chain_final_store_0_step" not in source
    )
    assert "chain_resident_carry_0_values[chain_resident_carry_0_index]" in source
    assert allocations[1].charged_bytes <= allocations[0].charged_bytes
    for index in (0, 1, 2):
        name = f"chain_{index}_c"
        assert (
            allocations[0].recurrence.layout.region(name).byte_size
            == allocations[1].recurrence.layout.region(name).byte_size
        )
    for marker in (
        "cute.gemm(",
        "mbarrier_wait(",
        "fence_view_async_tmem_load()",
        "fence_view_async_tmem_store()",
        "arrive_and_wait()",
        "cute.arch.sync_threads()",
    ):
        assert control.count(marker) == source.count(marker), marker


def test_unproved_endpoint_retains_original_shared_export() -> None:
    with patch.object(
        chained_final_carry_store, "prepare_final_carry_store", return_value=None
    ):
        source = _selected()
    assert "chain_3_c[" not in source
    assert "chain_final_store_0_step" in source


def test_missing_terminal_proof_rejects_the_requested_route() -> None:
    # Candidate discovery imported its function explicitly; block that same
    # proof rather than merely a downstream duplicate validation.
    with (
        patch(
            "helion._compiler.cute.chained_completed_members.plan_store_ownership",
            return_value=None,
        ),
        pytest.raises(exc.InternalError, match="exclusive result"),
    ):
        _selected()


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_terminal_storage_failure_cannot_escape_finalizer(fault: str) -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        if fault == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _selected()


def test_final_endpoint_must_consume_its_original_store_once() -> None:
    original = chained_completed_store.CompletedCarryEndpoints.consume_final_store

    def twice(self: Any, plan: Any, store: Any) -> bool:
        result = original(self, plan, store)
        assert result
        return original(self, plan, store)

    with (
        patch.object(
            chained_completed_store.CompletedCarryEndpoints,
            "consume_final_store",
            twice,
        ),
        pytest.raises(exc.InternalError, match="consumed out of order"),
    ):
        _selected()


def test_final_endpoint_changed_receipt_is_rejected() -> None:
    original = chained_final_carry_store.prepare_final_carry_store

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result is not None
        return replace(result, lines=())

    with (
        patch.object(chained_final_carry_store, "prepare_final_carry_store", corrupt),
        pytest.raises(exc.InternalError, match="carry drain"),
    ):
        _selected()


def test_coordinate_overflow_rejects_before_cancellation() -> None:
    coordinate = _Index(0, 1, (0, 63))
    with pytest.raises(_UnsupportedChain, match="overflowing"):
        _Selectors.binary(
            operator.add,
            coordinate,
            _Index(None, 0, (2**31 - 1, 2**31 - 1)),
            torch.int32,
        )
    with pytest.raises(_UnsupportedChain, match="combined"):
        _Selectors.binary(operator.add, coordinate, coordinate, torch.int32)
    with pytest.raises(_UnsupportedChain, match="noninteger"):
        _checked(coordinate, torch.float32)


@pytest.mark.parametrize("divisor", [-32, -1, 1, 32])
def test_uniform_signed_scalar_bounds_match_python(divisor: int) -> None:
    left = _Index(None, 0, (-129, 129))
    right = _Index(None, 0, (divisor, divisor))
    for operation in (operator.floordiv, operator.mod):
        result = _Selectors.binary(operation, left, right, torch.int32)
        assert all(
            result.bounds[0] <= operation(value, divisor) <= result.bounds[1]
            for value in range(-129, 130)
        )
    with pytest.raises(_UnsupportedChain, match="zero"):
        _Selectors.binary(operator.mod, left, _Index(None, 0, (0, 0)), torch.int32)
    for operation in (operator.floordiv, operator.mod):
        with pytest.raises(_UnsupportedChain, match="overflowing"):
            _Selectors.binary(
                operation,
                _Index(None, 0, (-(2**31), -(2**31))),
                _Index(None, 0, (-1, -1)),
                torch.int32,
            )


def test_vector_tile_requires_an_active_origin() -> None:
    graph = Graph()
    tile = graph.placeholder("tile")
    selector = graph.call_function(tile_index, (tile,))
    selector.meta["val"] = torch.empty((64,), dtype=torch.int32)
    with _cpu_codegen():
        bound = reuse_tests._square_intermediate._bind_isolated(reuse_tests._args())
        with bound.env:
            plan = ChainedMatmulPlan(
                root_graph_id=0,
                dots=(),
                store=selector,
                axes=(),
                shapes=(),
                dtype=torch.bfloat16,
                threads=128,
            )
            proof = _Selectors(plan)
            with (
                patch.object(proof, "block", return_value=99),
                patch.object(
                    chained_store_ownership.chain, "_shape", return_value=(64,)
                ),
                pytest.raises(_UnsupportedChain, match="inactive"),
            ):
                proof.vector(selector, _Index(0, 1, (0, 63)))


@pytest.mark.parametrize(
    "kind,accepted",
    [
        ("plain", True),
        ("shift", True),
        ("reverse", True),
        ("data", False),
        ("zero", False),
        ("stride2", False),
        ("overflow", False),
        ("broadcast", False),
        ("rank2", False),
    ],
)
def test_reordered_destination_requires_exact_injective_typed_axis(
    kind: str, accepted: bool
) -> None:
    graph = Graph()
    target = graph.call_function(_tracing_ops._host_tensor, ("output",))
    target.meta["val"] = torch.empty((80, 64))
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty((64, 64))
    selector = graph.call_function(
        torch.ops.prims.iota.default,
        (64,),
        {
            "start": 63 if kind == "reverse" else 0,
            "step": -1 if kind == "reverse" else 0 if kind == "zero" else 1,
            "dtype": torch.int32,
            "device": torch.device("cpu"),
            "requires_grad": False,
        },
    )
    selector.meta["val"] = torch.empty((64,), dtype=torch.int32)
    if kind in ("shift", "overflow", "stride2"):
        selector = graph.call_function(
            torch.ops.aten.mul.Scalar
            if kind == "stride2"
            else torch.ops.aten.add.Scalar,
            (
                selector,
                2 if kind == "stride2" else 2**31 - 32 if kind == "overflow" else 4,
            ),
        )
        selector.meta["val"] = torch.empty((64,), dtype=torch.int32)
    if kind == "data":
        selector = graph.placeholder("unproved_indices")
        selector.meta["val"] = torch.empty((64,), dtype=torch.int32)
    if kind in ("broadcast", "rank2"):
        selector.meta["val"] = torch.empty(
            (1,) if kind == "broadcast" else (64, 1), dtype=torch.int32
        )
    store = graph.call_function(
        memory_ops.store, (target, [selector, slice(None)], value, None)
    )
    with _cpu_codegen():
        bound = reuse_tests._square_intermediate._bind_isolated(reuse_tests._args())
        with (
            bound.env,
            patch.object(
                chained_store_ownership.chain,
                "_shape",
                side_effect=lambda node: tuple(node.meta["val"].shape),
            ),
        ):
            result = chained_store_ownership.plan_store_ownership(
                ChainedMatmulPlan(
                    root_graph_id=0,
                    dots=(),
                    store=store,
                    axes=(),
                    shapes=(),
                    dtype=torch.bfloat16,
                    threads=128,
                ),
                store,
            )
    assert (result is not None) == accepted


def test_terminal_seed_preserves_old_pool_and_explicit_override() -> None:
    key = "cute_chained_completed_member_store"
    with _cpu_codegen():
        bound = reuse_tests._square_intermediate._bind_isolated(reuse_tests._args())
        with bound.env:
            parents = [_config(False), _config()]
            seeded = CuteChainedMatmulHeuristic._with_m64_completed_store_seed(
                bound.env, parents
            )
            assert seeded[:-1] == parents and len(seeded) == len(parents) + 1
            assert seeded[-1].config == parents[-1].config | {key: True}
            assert bound.config_spec.normalized_config(seeded[-1])[key] is True
            assert (
                CuteChainedMatmulHeuristic._with_m64_completed_store_seed(
                    bound.env, seeded
                )
                == seeded
            )
            with patch.dict(bound.env.settings.autotune_config_overrides, {key: False}):
                assert (
                    CuteChainedMatmulHeuristic._with_m64_completed_store_seed(
                        bound.env, parents
                    )
                    == parents
                )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_scaled_sequence(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    transform: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(value.size(1))
    for batch in hl.tile(initial.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for step in hl.grid(steps):
            rows = hl.arange(value.size(2))
            columns = hl.arange(value.size(3))
            raw = hl.load(
                value,
                [batch.id, step, rows, columns],
                extra_mask=(rows < step * 17 + 3)[:, None],
            ).float()
            entering = state.to(left.dtype)
            corrected = raw - hl.dot(right[batch.id, step, :, :], entering)
            cross = hl.dot(left[batch.id, step, :, :], entering)
            solved = hl.dot(
                transform[batch.id, step, :, :].to(torch.float16),
                corrected.to(torch.float16),
            ).to(left.dtype)
            cross = cross * gate[batch.id, step, :][:, None]
            result = hl.dot(transform[batch.id, step, :, :], solved, acc=cross)
            hl.store(
                output,
                [batch.id, step, rows, columns],
                (result * scale).to(output.dtype),
                extra_mask=(rows < step * 13 + 5)[:, None],
            )
            state = hl.dot(
                update[batch.id, step, :, :],
                solved,
                acc=state * gate[batch.id, step, -1],
            )
        final[batch.id, :, :] = state
    return output, final


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_scaled_runtime_fixture_selects_both_exports(dtype: torch.dtype) -> None:
    args = (*reuse_tests._args(dtype=dtype), 0.5)
    with _cpu_codegen():
        bound = _masked_scaled_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_masked_scaled_sequence, args)
        ):
            source = bound.to_code(_config(cute_chained_completed_member_store=True))
    assert "chain_3_c[" not in source and "chain_final_store_0_step" not in source
    assert "* scale" in source and "chain_store_0_register" in source


def _runtime_config(enabled: bool, rectangular: bool) -> helion.Config:
    config = _config(enabled, cute_chained_completed_member_store=enabled)
    if rectangular and enabled:
        config.config.update(
            cute_chained_pointwise_vectorize=True,
            cute_chained_async_vector_store=True,
            cute_chained_rectangular_async=True,
        )
    return config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_combined_fixture_keeps_packet_config_and_selects_rectangles(
    dtype: torch.dtype,
) -> None:
    for enabled in (False, True):
        original = _config(enabled, cute_chained_completed_member_store=enabled)
        assert _runtime_config(enabled, False).config == original.config
        if not enabled:
            assert _runtime_config(enabled, True).config == original.config
    args = (*reuse_tests._args(dtype=dtype), 0.5)
    with _cpu_codegen():
        bound = _masked_scaled_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_masked_scaled_sequence, args)
        ):
            source = bound.to_code(_runtime_config(True, True))
    assert "_rectangle_copied" in source and "cp_async_commit_group()" in source
    assert "chain_3_c[" not in source and "chain_final_store_0_step" not in source
    assert "chain_store_0_register" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("rectangular", [False, True], ids=["packet", "rectangle"])
def test_gpu_m64_original_recipe_state_and_guard(
    dtype: torch.dtype, transposed: bool, rectangular: bool
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    steps, scale = 3, 0.5
    tensors = list(reuse_tests._args("cuda", dtype=dtype, steps=steps))
    if transposed:
        final = tensors[-1]
        tensors[-1] = torch.empty(
            (*final.shape[:-2], final.shape[-1], final.shape[-2]),
            device=final.device,
            dtype=final.dtype,
        ).transpose(-1, -2)
    left, right, value, transform, update, gate, initial, output, final = tensors
    generator = torch.Generator(device="cuda").manual_seed(3317)
    for tensor in (*tensors[:5], initial):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator)
            * 0.015625
        )
    gate.copy_(
        torch.randint(1, 4, gate.shape, device="cuda", generator=generator) * 0.25
    )
    unchanged = [tensor.clone() for tensor in tensors[:7]]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(value.size(2), device="cuda")
    for step in range(steps):
        raw = torch.where(
            (rows < step * 17 + 3)[None, :, None], value[:, step].double(), 0.0
        )
        entering = state.to(dtype).double()
        corrected = (
            (raw - right[:, step].double() @ entering).to(torch.float16).double()
        )
        solved = (
            (transform[:, step].to(torch.float16).double() @ corrected)
            .to(dtype)
            .double()
        )
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        result = ((transform[:, step].double() @ solved + cross) * scale).to(dtype)
        expected[:, step] = torch.where(
            (rows < step * 13 + 5)[None, :, None], result, expected[:, step]
        )
        state = (
            update[:, step].double() @ solved
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    args = (*tensors, scale)
    for enabled in (False, True):
        output.fill_(19.0)
        final.fill_(float("nan"))
        actual = _masked_scaled_sequence._bind_isolated(args).compile_config(
            _runtime_config(enabled, rectangular)
        )(*args)
        results.append(tuple(tensor.clone() for tensor in actual))
        torch.testing.assert_close(actual[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(actual[1], state.float(), atol=0.002, rtol=0.01)
    for old, new in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    for old, new in zip(unchanged, tensors[:7], strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
