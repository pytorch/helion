from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_loop import _config
from .test_cute_chained_m64_loop import _source
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_frame import _capture
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_grouped_fragment
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_recurrence_body
from helion._compiler.cute.chained_grouped_fragment import grouped_fragment_candidates
from helion._compiler.cute.chained_pointwise_residency import PointwiseCachePlan
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_result_transport import store_operation
from helion._compiler.cute.chained_seed_tiles import plan_seed_panels
import helion.language as hl

_KEY = "cute_chained_grouped_fragment_bridge"


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
@pytest.mark.parametrize(
    "width,total,offset", [(32, 96, 32), (64, 192, 128), (128, 256, 0)]
)
def test_actual_fragment_member_a_and_seed_ownership(
    dtype_name: str, width: int, total: int, offset: int
) -> None:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = vars(cutlass)[dtype_name]
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            scope: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05": tcgen05,
            }
            for prefix, shape, base in (
                ("src", (64, 2 * width), 0),
                ("dst", (64, total), 256),
            ):
                mma = blackwell_helpers.make_trivial_tiled_mma(
                    dtype,
                    dtype,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                    cutlass.Float32,
                    tcgen05.CtaGroup.ONE,
                    shape,
                    tcgen05.OperandSource.SMEM,
                )
                scope[f"{prefix}_acc"] = cute.make_tensor(
                    cute.make_ptr(cutlass.Float32, base, cute.AddressSpace.tmem),
                    mma.make_fragment_C(mma.partition_shape_C(shape)).layout,
                )
                scope[f"{prefix}_slice"] = mma.get_slice(0)
                scope[f"{prefix}_identity"] = mma.get_slice(0).partition_C(
                    cute.make_identity_tensor(shape)
                )
            reader = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    eval(load_operation((64, 2 * width)), scope), cutlass.Float32
                ),
                scope["src_acc"],
            )
            panel = plan_seed_panels((64, total), offset, width)[0]
            exec("\n".join(panel.views("seed", "dst")), scope)
            writer = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    eval(store_operation((64, width)), scope), cutlass.Float32
                ),
                scope["seed_segment"],
            )
            atom = (
                tcgen05.SmemLayoutAtomKind.K_SW64
                if width == 32
                else tcgen05.SmemLayoutAtomKind.K_SW128
            )
            layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(atom, dtype), (64, width), order=(0, 1)
            )
            # Identity, native shared address and source/seed partitions use
            # exactly the same rank-three composition as emitted code.
            identity = cute.make_identity_tensor((64, width))
            logical = cute.make_tensor(
                identity.iterator,
                cute.composition(
                    identity.layout, cute.make_layout(((64, width), 1, 1))
                ),
            )
            addresses = cute.make_tensor(
                0, cute.composition(layout.outer, cute.make_layout(((64, width), 1, 1)))
            )
            seen = []
            for thread in range(128):
                source = reader.get_slice(thread).partition_D(scope["src_identity"])
                seed = writer.get_slice(thread).partition_S(scope["seed_identity"])
                target = writer.get_slice(thread).partition_S(logical)
                target_addresses = writer.get_slice(thread).partition_S(addresses)
                coords = [tuple(source[i]) for i in range(cute.size(source))]
                assert cute.size(target) == cute.size(seed) == width // 2
                for i in range(width // 2):
                    row, column = target[i]
                    assert coords[i] == (row, column)
                    seed_row, seed_column = seed[i]
                    assert coords[i + width // 2] == (
                        seed_row,
                        seed_column - offset + width,
                    )
                    assert target_addresses[i] == layout.outer((row, column))
                    seen.append((row, column))
            assert len(seen) == len(set(seen)) == 64 * width
            native = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        dtype, 128, cute.AddressSpace.smem, assumed_align=128
                    ),
                    layout.inner,
                    dtype=dtype,
                ),
                layout.outer,
            )
            native_view = cute.make_tensor(
                native.iterator,
                cute.composition(native.layout, cute.make_layout(((64, width), 1, 1))),
            )
            target = writer.get_slice(127).partition_S(native_view)
            values = cute.make_rmem_tensor(target.shape, dtype)
            values.fill(0.0)
            cute.autovec_copy(values, target)
        assert module.operation.verify()


@pytest.mark.parametrize("vector", [False, True])
def test_codegen_elides_only_exclusive_spills_and_fill(vector: bool) -> None:
    events = []
    original = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result is not None
        events.append(result.allocations)
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        before = _source(_config(cute_chained_pointwise_vectorize=vector))
        after = _source(
            _config(cute_chained_pointwise_vectorize=vector, **{_KEY: True})
        )
    assert events[0] == events[1]
    assert "chain_group_bridge_0_values_a" in after
    assert "chain_0_c[" not in after and "chain_1_c[" not in after
    assert "chain_2_a_2_step" not in after
    assert before.count("cute.gemm(") == after.count("cute.gemm(") == 2
    for marker in (
        "mbarrier_wait(",
        "fence_view_async_tmem_load()",
        "fence_view_async_tmem_store()",
        "chain_recurrence_barrier.arrive_and_wait()",
    ):
        assert before.count(marker) == after.count(marker)
    ast.parse(after)


@pytest.mark.parametrize(
    "fault", ["extra_user", "seed_mix", "source_dtype", "changed_graph", "cache"]
)
def test_exclusive_fragment_proof_rejects_changed_edges(fault: str) -> None:
    plan, _, _ = _capture(_load_only_recurrence, _args(), _config())
    groups = plan.contraction_groups
    assert groups is not None
    candidates = grouped_fragment_candidates(plan, groups)
    assert len(candidates) == 1
    candidate = candidates[0]
    source = plan.dots[candidate.operand_member]
    if fault in ("extra_user", "changed_graph"):
        graph = source.graph
        with graph.inserting_after(source):
            user = graph.call_function(torch.ops.aten.neg.default, (source,))
            user.meta["val"] = source.meta["val"]
        if fault == "extra_user":
            assert plan.region is not None and plan.loop is not None
            region = replace(plan.region, nodes=tuple(graph.nodes))
            plan = replace(plan, region=region, loop=replace(plan.loop, region=region))
    elif fault == "seed_mix":
        candidate.seed.args = (source, *candidate.seed.args[1:])
    elif fault == "source_dtype":
        source.meta["val"] = source.meta["val"].to(torch.float16)
    else:
        # Even a separately selected cache must retain its materialization
        # contract; this first transport does not compose that policy.
        plan = replace(plan, pointwise_cache=PointwiseCachePlan())
    assert grouped_fragment_candidates(plan, groups) == ()


@pytest.mark.parametrize(
    "options",
    [
        {"cute_chained_loop_mma_rows": 128},
        {"cute_chained_pipeline_consumer_warps": 8, "num_warps": 16},
        {"cute_chained_seed_tile_columns": 32},
        {"cute_chained_drain_tile_columns": 32},
        {"cute_chained_preparation_pipeline": False},
    ],
)
def test_unsupported_policy_is_not_silently_dropped(options: dict[str, object]) -> None:
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalized_config(dict(_config(**{_KEY: True} | options)))


def test_default_off_and_one_appended_seed() -> None:
    assert _source(_config(**{_KEY: False})) == _source()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        assert bound.config_spec.cute_chained_grouped_fragment_bridge_search_enabled
        fields = bound.config_spec._flat_fields()
        assert _KEY in fields
        with patch.object(
            bound.config_spec,
            "cute_chained_grouped_fragment_bridge_search_enabled",
            False,
        ):
            assert tuple(bound.config_spec._flat_fields()) == tuple(
                key for key in fields if key != _KEY
            )
        parents = [_config()]
        seeds = CuteChainedMatmulHeuristic._with_grouped_fragment_seed(
            bound.env, parents
        )
        assert seeds[:-1] == parents and len(seeds) == 2
        assert seeds[-1].config == parents[0].config | {_KEY: True}
        assert (
            CuteChainedMatmulHeuristic._with_grouped_fragment_seed(bound.env, seeds)
            == seeds
        )


def test_declined_late_fragment_proof_is_not_a_silent_fallback() -> None:
    with (
        patch.object(
            chained_grouped_fragment, "bind_grouped_fragment_bridges", return_value=()
        ),
        pytest.raises(
            exc.BackendUnsupported, match="no complete adjacent grouped fragment bridge"
        ),
    ):
        _source(_config(**{_KEY: True}))


def test_same_attempt_context_rejects_changed_fragment_binding() -> None:
    original = chained_recurrence_body.bind_recurrence_body

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result.grouped_fragments
        result.grouped_fragments = ()
        return result

    with (
        patch.object(chained_recurrence_body, "bind_recurrence_body", changed),
        pytest.raises(exc.InternalError, match="recurrence body context"),
    ):
        _source(_config(**{_KEY: True}))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_reverse(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    transform: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(value.size(1))
    for batch in hl.tile(initial.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for step in hl.grid(steps):
            entering = state.to(left.dtype)
            cross = hl.dot(left[batch.id, step, :, :], entering)
            cross = cross * gate[batch.id, step, :][:, None]
            corrected = value[batch.id, step, :, :].float() - hl.dot(
                right[batch.id, step, :, :], entering
            )
            corrected = torch.where(
                gate[batch.id, step, :][:, None] > 0.5, corrected, 0.0
            )
            narrowed = corrected.to(left.dtype)
            result = hl.dot(transform[batch.id, step, :, :], narrowed, acc=cross)
            output[batch.id, step, :, :] = result.to(output.dtype)
            state = hl.dot(
                update[batch.id, step, :, :],
                narrowed,
                acc=state * gate[batch.id, step, -1],
            )
        final[batch.id, :, :] = state
    return output, final


def test_reversed_members_and_mask_codegen() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _masked_reverse._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_masked_reverse, args)):
            source = bound.to_code(_config(**{_KEY: True}))
    assert "chain_0_values[chain_group_bridge_0_a_index + 32]" in source
    assert "chain_0_values[chain_group_bridge_0_seed_index + 0]" in source
    assert "chain_group_bridge_0_values_a" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_changing_iterations_typed_a_seed_and_guard_gpu(
    dtype: torch.dtype, masked: bool
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = _args("cuda", dtype=dtype, steps=3)
    torch.manual_seed(2405)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(torch.randint(-2, 3, tensor.shape, device="cuda") * 0.015625)
    args[5].copy_(torch.randint(1, 4, args[5].shape, device="cuda") * 0.25)
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [item.clone() for item in args[:7]]
    expected, state = output.clone(), initial.double()
    for step in range(3):
        entering = state.to(dtype).double()
        corrected = value[:, step].double() - right[:, step].double() @ entering
        if masked:
            corrected = torch.where(gate[:, step, :, None] > 0.5, corrected, 0.0)
        narrowed = corrected.to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ narrowed + cross).to(dtype)
        state = (
            update[:, step].double() @ narrowed
            + state * gate[:, step, -1, None, None].double()
        )
    kernel = _masked_reverse if masked else _load_only_recurrence
    outputs = []
    for enabled in (False, True):
        result = kernel._bind_isolated(args).compile_config(_config(**{_KEY: enabled}))(
            *args
        )
        outputs.append(tuple(item.clone() for item in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert torch.equal(final, outputs[1][1])
