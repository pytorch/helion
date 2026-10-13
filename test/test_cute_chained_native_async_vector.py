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
from .test_cute_chained_prepared_group_multi_atom import _placed
from .test_cute_chained_prepared_groups import _layout_expression
from .test_cute_chained_prepared_groups import _lowered_store_addresses
from .test_cute_chained_prepared_operands import _candidate as _operand_candidate
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_prepared_values
from helion._compiler.cute.chained_native_vector_sink import NativeVectorSink
from helion._compiler.cute.chained_native_vector_sink import native_vector_sinks
from helion._compiler.cute.chained_prepared_groups import _group_layout
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands
import helion.language as hl

_KEY = "cute_chained_async_vector_store"


def _selected(enabled: bool = True, **options: object) -> helion.Config:
    return _config(
        cute_chained_prepared_group_multi_atom=True,
        cute_chained_pointwise_vectorize=True,
        **{_KEY: enabled} | options,
    )


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize(
    "k,multi_atom", [(16, False), (32, False), (64, False), (128, False), (128, True)]
)
def test_actual_native_packet_pointer_preserves_order_and_footprint(
    dtype_name: str, k: int, multi_atom: bool
) -> None:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = vars(cutlass)[dtype_name]
    dtype_text = f"cutlass.{dtype_name}"
    namespace = {"cutlass": cutlass, "cute": cute, "tcgen05": tcgen05}
    sink = NativeVectorSink("image", (32, k), dtype_text)
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        expected = []
        with ir.InsertionPoint(module.body):
            layout = eval(
                _layout_expression(
                    _group_layout("image", (32, k), dtype_text, multi_atom=multi_atom)[
                        0
                    ]
                ),
                namespace,
            )
            offsets = [int(layout.outer((r, c))) for r in range(32) for c in range(k)]
            assert sorted(offsets) == list(range(32 * k))
            bits = min(3, (k // 8).bit_length() - 1)
            for phase in range(8):
                base = 4096 + 128 * phase
                image = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            dtype, base, cute.AddressSpace.smem, assumed_align=128
                        ),
                        layout.inner,
                        dtype=dtype,
                    ),
                    layout.outer,
                )
                # Exhaust every physical packet. Lower boundary-row samples
                # through actual tensor/pointer operations, not just formulas.
                physical = [
                    (base + 2 * offset)
                    ^ (((base + 2 * offset) & (((1 << bits) - 1) << 7)) >> 3)
                    for offset in offsets
                ]
                assert sorted(physical) == list(range(base, base + 64 * k, 2))
                for start in range(0, len(physical), 8):
                    packet = physical[start : start + 8]
                    assert packet[0] % 16 == 0
                    assert packet == list(range(packet[0], packet[0] + 16, 2))
                for row in (0, 7, 8, 31):
                    for column in (0, k - 8):
                        pointer = eval(
                            sink.pointer("row", "column"),
                            {"image": image, "row": row, "column": column},
                        )
                        packet = cute.make_tensor(
                            eval(
                                sink.aligned_pointer("pointer"),
                                namespace
                                | {"pointer": pointer, "image_layout": layout},
                            ),
                            cute.make_layout((8,)),
                        )
                        for element in range(8):
                            image[row, column + element] = dtype(1)
                            packet[element] = dtype(1)
                            expected.extend([physical[row * k + column + element]] * 2)
        assert _lowered_store_addresses(module) == expected
        copy_module = ir.Module.create()
        with ir.InsertionPoint(copy_module.body):
            layout = eval(
                _layout_expression(
                    _group_layout("image", (32, k), dtype_text, multi_atom=multi_atom)[
                        0
                    ]
                ),
                namespace,
            )
            image = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        dtype, 4096, cute.AddressSpace.smem, assumed_align=128
                    ),
                    layout.inner,
                    dtype=dtype,
                ),
                layout.outer,
            )
            pointer = eval(sink.pointer("7", str(k - 8)), {"image": image})
            target = cute.make_tensor(
                eval(
                    sink.aligned_pointer("pointer"),
                    namespace | {"pointer": pointer, "image_layout": layout},
                ),
                cute.make_layout((8,)),
            )
            atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), dtype, num_bits_per_copy=128
            )
            source = cute.make_tensor(
                cute.make_ptr(
                    dtype, 0x100000, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((8,)),
            )
            cute.copy(atom, source, target)
        assert copy_module.operation.verify()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_only_exact_untransposed_native_members_get_packet_sinks(
    dtype: torch.dtype,
) -> None:
    _, _, _, placed = _placed((16, 48), dtype, (True, False))
    group = placed.groups[0]
    sinks = native_vector_sinks(placed.frame, (), placed.groups)
    members = group.candidate.members
    assert set(sinks) == {members[0].buffer.name}
    assert sinks[members[0].buffer.name].shape == (16, 128)
    assert members[1].logical_modes == (1, 0)
    for changed in (
        replace(group, byte_offset=group.byte_offset + 128),
        replace(group, candidate=replace(group.candidate, native_layout=())),
        replace(
            group,
            candidate=replace(
                group.candidate,
                members=(replace(members[0], logical_modes=(1, 0)), members[1]),
            ),
        ),
    ):
        with pytest.raises(ValueError, match="native packet group binding changed"):
            native_vector_sinks(placed.frame, (), (changed,))


@pytest.mark.parametrize("k", [16, 32, 48, 64, 128])
def test_singleton_duplicate_consumers_retain_one_exact_sink(k: int) -> None:
    plan, frame, recurrence = _operand_candidate(32, k, mode="repeat")
    operands = plan_prepared_operands(plan, frame, recurrence)
    assert operands is not None and len(operands) == 2
    sinks = native_vector_sinks(frame, operands, ())
    assert len(sinks) == (0 if k == 48 else 1)
    changed = replace(operands[0], native_layout=())
    with pytest.raises(ValueError, match="native packet operand binding changed"):
        native_vector_sinks(frame, (changed,), ())


def test_native_async_composes_without_changing_allocations_or_ready_protocol() -> None:
    allocations = []
    original = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result is not None
        allocations.append(result.allocations)
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        before = _source(_selected(False))
        after = _source(_selected(True))
    assert allocations[0] == allocations[1]
    assert "CopyG2SOp" not in before
    assert after.count("cpasync.CopyG2SOp") == 4
    assert after.count("_sink = cute.make_tensor(cute.recast_ptr(") == 4
    assert ".iterator + chain_prepared_" in after and ".layout((" in after
    for marker in (
        "cute.gemm(",
        "mbarrier_wait(",
        "arrive_mbarrier(",
        "arrive_and_wait()",
        "fence_view_async_shared()",
    ):
        assert before.count(marker) == after.count(marker)
    assert (
        after.count("cp_async_commit_group()")
        == after.count("cp_async_wait_group(0)")
        == 4
    )
    assert "_last_address ==" in after and "_shared_pointer.toint()) % 16 == 0" in after
    ast.parse(after)


@pytest.mark.parametrize("fault", ["shape", "dtype", "drop"])
def test_changed_native_sink_is_not_silently_activated(fault: str) -> None:
    original = chained_prepared_values.native_vector_sinks

    def changed(*args: Any, **kwargs: Any) -> Any:
        sinks = original(*args, **kwargs)
        if fault == "drop":
            return {}
        return {
            name: (
                replace(sink, shape=(1, 1))
                if fault == "shape"
                else replace(sink, dtype="cutlass.Float32")
            )
            for name, sink in sinks.items()
        }

    with (
        patch.object(chained_prepared_values, "native_vector_sinks", changed),
        pytest.raises(exc.BackendUnsupported, match="identity shared sink"),
    ):
        _source(_selected())


def test_nonunit_raw_source_strides_keep_original_scalar_publication() -> None:
    args = list(_args())
    for index in (0, 1):
        value = args[index]
        args[index] = torch.empty(
            (*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype
        )[..., ::2]
        assert args[index].stride(-1) == 2
    source = _source(_selected(), tuple(args))
    # The two other complete identity images still copy asynchronously. These
    # strided members retain their original scalar loads and native stores.
    assert source.count("cpasync.CopyG2SOp") == 2
    assert "chain_prepared_0_offset" in source or "chain_prepared_1_offset" in source


def test_native_async_still_requires_complete_final_capacity_proof() -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source(_selected())


def test_native_async_adds_one_seed_and_preserves_old_choices() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(args)
        assert bound.host_function is not None
        parent = _config(cute_chained_prepared_group_multi_atom=True)
        parents = [_config(), parent]
        with (
            bound.env,
            bound.env.use_runtime_arg_values(
                _runtime_values(_load_only_recurrence, args)
            ),
        ):
            seeds = CuteChainedMatmulHeuristic._with_native_async_vector_seed(
                bound.env, bound.host_function.device_ir, parents
            )
            assert seeds[:-1] == parents and len(seeds) == 3
            assert seeds[-1].config == parent.config | {
                "cute_chained_pointwise_vectorize": True,
                _KEY: True,
            }
            assert bound.config_spec.normalized_config(seeds[-1])[_KEY] is True
            assert (
                CuteChainedMatmulHeuristic._with_native_async_vector_seed(
                    bound.env, bound.host_function.device_ir, seeds
                )
                == seeds
            )
            with patch.dict(
                bound.env.settings.autotune_config_overrides, {_KEY: False}
            ):
                assert (
                    CuteChainedMatmulHeuristic._with_native_async_vector_seed(
                        bound.env, bound.host_function.device_ir, parents
                    )
                    is parents
                )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_raw_recurrence(
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
            rr, kk = hl.arange(left.size(2)), hl.arange(left.size(3))
            valid = step * 17 + 3
            lhs = hl.load(
                left, [batch.id, step, rr, kk], extra_mask=(rr < valid)[:, None]
            )
            rhs = hl.load(
                right, [batch.id, step, rr, kk], extra_mask=(rr < valid)[:, None]
            )
            entering = state.to(left.dtype)
            corrected = value[batch.id, step, :, :].float() - hl.dot(rhs, entering)
            narrowed = corrected.to(left.dtype)
            cross = hl.dot(lhs, entering) * gate[batch.id, step, :][:, None]
            result = hl.dot(transform[batch.id, step, :, :], narrowed, acc=cross)
            output[batch.id, step, :, :] = result.to(output.dtype)
            state = hl.dot(
                update[batch.id, step, :, :],
                narrowed,
                acc=state * gate[batch.id, step, -1],
            )
        final[batch.id, :, :] = state
    return output, final


def test_masked_raw_native_codegen_keeps_vector_uniform_mask_and_fallback() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _masked_raw_recurrence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_masked_raw_recurrence, args)
        ):
            source = bound.to_code(_selected())
    assert "cpasync.CopyG2SOp" in source
    assert "cutlass.Boolean(False)" in source and "if not chain_prepared_" in source
    assert "17" in source
    ast.parse(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("misaligned", [False, True])
def test_changing_masked_frames_exact_native_async_gpu(
    dtype: torch.dtype, misaligned: bool
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = list(_args("cuda", dtype=dtype, steps=3))
    if misaligned:
        for i in (0, 1):
            original = args[i]
            storage = torch.empty(original.numel() + 1, dtype=dtype, device="cuda")
            args[i] = storage[1:].view(original.shape)
            assert args[i].data_ptr() % 16 != 0
    torch.manual_seed(2505)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(torch.randint(-2, 3, tensor.shape, device="cuda") * 0.015625)
    args[5].copy_(torch.randint(1, 4, args[5].shape, device="cuda") * 0.25)
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [item.clone() for item in args[:7]]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(left.size(2), device="cuda")
    for step in range(3):
        mask = (rows < step * 17 + 3)[None, :, None]
        lhs = torch.where(mask, left[:, step].double(), 0.0)
        rhs = torch.where(mask, right[:, step].double(), 0.0)
        entering = state.to(dtype).double()
        narrowed = (value[:, step].double() - rhs @ entering).to(dtype).double()
        cross = (lhs @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ narrowed + cross).to(dtype)
        state = (
            update[:, step].double() @ narrowed
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    for enabled in (False, True):
        result = _masked_raw_recurrence._bind_isolated(tuple(args)).compile_config(
            _selected(enabled)
        )(*args)
        results.append(tuple(item.clone() for item in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert torch.equal(final, results[1][1])
