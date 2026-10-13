from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_orientation import _config as _orientation_config
from .test_cute_chained_m64_orientation import _square_intermediate
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_frame import _capture
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_fragment_reuse
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_recurrence_body
from helion._compiler.cute.chained_fragment_reuse import exclusive_members
from helion._compiler.cute.chained_fragment_reuse import fragment_uses
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_result_transport import store_operation
from helion._compiler.cute.chained_seed_tiles import plan_seed_panels
import helion.language as hl

_KEY = "cute_chained_fragment_reuse"


def _config(enabled: bool = True, **options: object) -> helion.Config:
    return helion.Config.from_dict(
        _orientation_config().config | {_KEY: enabled} | options
    )


def _source(enabled: bool = True) -> str:
    args = _args()
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_square_intermediate, args)
        ):
            return bound.to_code(_config(enabled))


def test_separate_operand_and_delayed_seed_edges() -> None:
    plan, _, _ = _capture(_square_intermediate, _args(), _config(False))
    groups = plan.contraction_groups
    assert groups is not None
    assert tuple(group.stages for group in groups) == ((0, 1), (2,), (3, 4))
    uses = fragment_uses(plan, groups)
    assert [(u.member, u.kind, u.destinations) for u in uses] == [
        (0, "operand", (2,)),
        (1, "seed", (3,)),
        (2, "operand", (3, 4)),
    ]
    assert exclusive_members(plan, uses) == {0, 1, 2}


def test_codegen_preserves_allocation_and_protocol() -> None:
    allocations = []
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = finalize(*args, **kwargs)
        assert result is not None
        allocations.append(result.allocations)
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        original, selected = _source(False), _source()
    assert allocations[0] == allocations[1]
    for member in (0, 1, 2):
        assert f"chain_{member}_c[" in original
        assert f"chain_{member}_c[" not in selected
    assert "chain_fragment_0_operand_2_values_a" in selected
    assert "chain_fragment_2_operand_3_values_a" in selected
    assert "chain_fragment_1_seed_3_index + 32" in selected
    for marker in (
        "cute.gemm(",
        "mbarrier_wait(",
        "fence_view_async_tmem_load()",
        "fence_view_async_tmem_store()",
        "chain_recurrence_barrier.arrive_and_wait()",
        "chain_prep_barrier.arrive_and_wait()",
    ):
        assert original.count(marker) == selected.count(marker)
    ast.parse(selected)


def test_fanout_retains_original_member_publication() -> None:
    plan, _, _ = _capture(_square_intermediate, _args(), _config(False))
    groups = plan.contraction_groups
    assert groups is not None and plan.region is not None and plan.loop is not None
    source = plan.dots[0]
    with source.graph.inserting_after(source):
        extra = source.graph.call_function(torch.ops.aten.neg.default, (source,))
        extra.meta["val"] = source.meta["val"]
    region = replace(plan.region, nodes=tuple(source.graph.nodes))
    plan = replace(plan, region=region, loop=replace(plan.loop, region=region))
    uses = fragment_uses(plan, groups)
    assert len(uses) == 3
    assert exclusive_members(plan, uses) == {1, 2}


@pytest.mark.parametrize("value", [1, "reuse", None])
def test_option_requires_boolean(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be bool"):
        bound = _square_intermediate._bind_isolated(_args())
        bound.config_spec.normalized_config(_config().config | {_KEY: value})


@pytest.mark.parametrize(
    "options",
    [
        {"cute_chained_grouped_fragment_bridge": True},
        {"cute_chained_loop_mma_rows": 128},
        {"cute_chained_pipeline_consumer_warps": 8, "num_warps": 16},
        {"cute_chained_seed_tile_columns": 32},
        {"cute_chained_drain_tile_columns": 32},
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_pointwise_cache_bytes": 4096},
    ],
)
def test_unsupported_choices_reject(options: dict[str, object]) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig):
        bound = _square_intermediate._bind_isolated(_args())
        bound.config_spec.normalized_config(_config().config | options)


def test_one_seed_and_explicit_override_preserved() -> None:
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(_args())
        assert bound.config_spec.cute_chained_fragment_reuse_search_enabled
        assert _KEY not in bound.config_spec.normalized_config(_config(False)).config
        parents = [_config(False)]
        # False defaults normalize away, so use the actual parent representation.
        parents = [bound.config_spec.normalized_config(parent) for parent in parents]
        seeded = CuteChainedMatmulHeuristic._with_fragment_reuse_seed(
            bound.env, parents
        )
        assert seeded[:-1] == parents and len(seeded) == len(parents) + 1
        assert seeded[-1].config == parents[-1].config | {_KEY: True}
        with patch.dict(bound.env.settings.autotune_config_overrides, {_KEY: False}):
            assert (
                CuteChainedMatmulHeuristic._with_fragment_reuse_seed(bound.env, parents)
                == parents
            )


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_final_storage_remains_authority(fault: str) -> None:
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
        _source()


def test_failed_expression_proof_is_not_silently_ignored() -> None:
    with (
        patch.object(chained_fragment_reuse, "bind_fragment_reuse", return_value=None),
        pytest.raises(exc.BackendUnsupported, match="no complete fragment reuse path"),
    ):
        _source()


def test_changed_binding_rejected_by_original_completion_ledger() -> None:
    original = chained_recurrence_body.bind_recurrence_body

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result.fragment_reuse is not None
        result.fragment_reuse = replace(result.fragment_reuse, omitted=frozenset())
        return result

    with (
        patch.object(chained_recurrence_body, "bind_recurrence_body", corrupt),
        pytest.raises(exc.InternalError, match="fragment reuse graph or users changed"),
    ):
        _source()


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("count", [1, 2])
def test_actual_complete_fragment_order(
    dtype_name: str, width: int, count: int
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
            shape = (64, count * width)
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
            acc = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 128, cute.AddressSpace.tmem),
                mma.make_fragment_C(mma.partition_shape_C(shape)).layout,
            )
            identity = mma.get_slice(0).partition_C(cute.make_identity_tensor(shape))
            scope: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05": tcgen05,
                "src_acc": acc,
                "src_slice": mma.get_slice(0),
            }
            reader = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    eval(load_operation(shape), scope), cutlass.Float32
                ),
                acc,
            )
            destination_shape = (64, min(3 * width, 256))
            destination_mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                destination_shape,
                tcgen05.OperandSource.SMEM,
            )
            scope["dst_acc"] = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 256, cute.AddressSpace.tmem),
                destination_mma.make_fragment_C(
                    destination_mma.partition_shape_C(destination_shape)
                ).layout,
            )
            scope["dst_slice"] = destination_mma.get_slice(0)
            seed_panel = plan_seed_panels(destination_shape, width, width)[0]
            exec("\n".join(seed_panel.views("delayed", "dst")), scope)
            delayed_writer = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    eval(store_operation((64, width)), scope), cutlass.Float32
                ),
                scope["delayed_segment"],
            )
            atom = (
                tcgen05.SmemLayoutAtomKind.K_SW64
                if width == 32
                else tcgen05.SmemLayoutAtomKind.K_SW128
            )
            layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(atom, dtype), (64, width), order=(0, 1)
            )
            native = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        dtype, 256, cute.AddressSpace.smem, assumed_align=128
                    ),
                    layout.inner,
                    dtype=dtype,
                ),
                layout.outer,
            )
            target = cute.make_tensor(
                native.iterator,
                cute.composition(native.layout, cute.make_layout(((64, width), 1, 1))),
            )
            local = cute.make_identity_tensor((64, width))
            logical = cute.make_tensor(
                local.iterator,
                cute.composition(local.layout, cute.make_layout(((64, width), 1, 1))),
            )
            for position in range(count):
                offset = position * width
                panel = plan_seed_panels(shape, offset, width)[0]
                exec("\n".join(panel.views("seed", "src")), scope)
                writer = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(
                        eval(store_operation((64, width)), scope), cutlass.Float32
                    ),
                    scope["seed_segment"],
                )
                seen = set()
                for thread in range(128):
                    source = reader.get_slice(thread).partition_D(identity)
                    seed = writer.get_slice(thread).partition_S(scope["seed_identity"])
                    operand = writer.get_slice(thread).partition_S(logical)
                    delayed = delayed_writer.get_slice(thread).partition_S(
                        scope["delayed_identity"]
                    )
                    assert cute.size(seed) == cute.size(operand) == width // 2
                    for i in range(width // 2):
                        row, column = source[i + offset // 2]
                        assert tuple(seed[i]) == (row, column)
                        assert tuple(operand[i]) == (row, column - offset)
                        assert tuple(delayed[i]) == (row, column - offset + width)
                        seen.add((row, column - offset))
                assert len(seen) == 64 * width
                destination = writer.get_slice(127).partition_S(target)
                values = cute.make_rmem_tensor(destination.shape, dtype)
                values.fill(0.0)
                cute.autovec_copy(values, destination)
        assert module.operation.verify()


@pytest.mark.parametrize("fault", ["graph", "dtype", "order", "expression"])
def test_retained_facts_reject_stale_evidence(fault: str) -> None:
    plan, _, _ = _capture(_square_intermediate, _args(), _config(False))
    groups = plan.contraction_groups
    assert groups is not None
    uses = fragment_uses(plan, groups)
    assert uses
    if fault == "graph":
        with plan.dots[0].graph.inserting_after(plan.dots[0]):
            extra = plan.dots[0].graph.call_function(
                torch.ops.aten.neg.default, (plan.dots[0],)
            )
            extra.meta["val"] = plan.dots[0].meta["val"]
    elif fault == "dtype":
        plan.dots[0].meta["val"] = plan.dots[0].meta["val"].to(torch.float16)
    elif fault == "order":
        groups = tuple(reversed(groups))
    else:
        source = uses[0].targets[0]
        source.args = (plan.dots[1], *source.args[1:])
    assert uses[0] not in fragment_uses(plan, groups)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
def test_gpu_separate_typed_paths_and_delayed_seed(
    dtype: torch.dtype, steps: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = _args("cuda", dtype=dtype, steps=steps)
    generator = torch.Generator(device="cuda").manual_seed(3205)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator)
            * 0.015625
        )
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, device="cuda", generator=generator) * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [tensor.clone() for tensor in args[:7]]
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
        expected[:, step] = (transform[:, step].double() @ solved + cross).to(dtype)
        state = (
            update[:, step].double() @ solved
            + state * gate[:, step, -1, None, None].double()
        )
    outputs = []
    for enabled in (False, True):
        output.fill_(19.0)
        final.fill_(float("nan"))
        result = _mixed_intermediate._bind_isolated(args).compile_config(
            _config(enabled)
        )(*args)
        outputs.append(tuple(tensor.clone() for tensor in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_intermediate(
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
            output[batch.id, step, :, :] = result.to(output.dtype)
            state = hl.dot(
                update[batch.id, step, :, :],
                solved,
                acc=state * gate[batch.id, step, -1],
            )
        final[batch.id, :, :] = state
    return output, final


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_dtype_fixture_codegen(dtype: torch.dtype) -> None:
    args = _args(dtype=dtype)
    with _cpu_codegen():
        bound = _mixed_intermediate._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_mixed_intermediate, args)
        ):
            source = bound.to_code(_config())
    assert "chain_fragment_0_operand_2_values_a" in source
    assert "chain_fragment_2_operand_3_values_a" in source
    assert "chain_fragment_1_seed_3_index + 32" in source
