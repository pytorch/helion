from __future__ import annotations

from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _args as _one_atom_args
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_loop import _config
from .test_cute_chained_m64_loop import _source
from .test_cute_chained_preparation_frame import _safe
from .test_cute_chained_prepared_groups import _candidate
from .test_cute_chained_prepared_groups import _layout_expression
from .test_cute_chained_prepared_groups import _lowered_store_addresses
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute.chained_frontier_ownership import _group_matches
from helion._compiler.cute.chained_prepared_groups import _native_row_subview
from helion._compiler.cute.chained_prepared_groups import coallocate_prepared_groups
from helion._compiler.cute.chained_prepared_groups import prepared_group_candidates
from helion.autotuner.config_fragment import EnumFragment

_KEY = "cute_chained_prepared_group_multi_atom"


def _placed(
    widths: tuple[int, ...], dtype: torch.dtype, transpose: tuple[bool, ...]
) -> tuple[Any, Any, Any, Any]:
    plan, frame, recurrence = _candidate(widths, 128, dtype, transpose=transpose)
    assert prepared_group_candidates(plan, frame, recurrence) == ()
    plan = replace(plan, prepared_group_multi_atom=True)
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    placed = coallocate_prepared_groups(
        plan, frame, recurrence, candidates, frame_capacity_bytes=1 << 20
    )
    assert placed is not None and len(placed.groups) == 1
    assert _group_matches(placed.frame, placed.groups[0])
    _safe(placed.frame)
    return plan, frame, recurrence, placed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "widths,transpose",
    [((16, 48), (False, True)), ((64, 64), (True, True)), ((64, 128), (False, False))],
)
def test_actual_multi_atom_bytes_copy_packets_and_descriptor(
    dtype: torch.dtype, widths: tuple[int, ...], transpose: tuple[bool, ...]
) -> None:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    _, original, _, placed = _placed(widths, dtype, transpose)
    candidate = placed.groups[0].candidate
    assert candidate.multi_atom
    assert candidate.byte_size == sum(widths) * 128 * 2
    for before, after in zip(
        original.layout.regions, placed.frame.layout.regions, strict=True
    ):
        assert replace(after, byte_offset=before.byte_offset) == before
    ir = importlib.import_module("cutlass._mlir.ir")
    manager = importlib.import_module("cutlass._mlir.passmanager")
    element = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    namespace = {"cutlass": cutlass, "cute": cute, "tcgen05": tcgen05}
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            full = eval(_layout_expression(candidate.native_layout[0]), namespace)
            # The native pointer swizzle uses absolute BYTES. Exhaust all eight
            # 128-byte base phases and every element of each mixed-mode member.
            swizzle = cute.make_composed_layout(
                full.inner, 0, cute.make_layout(candidate.byte_size + 1024)
            )
            # Evaluate each actual CuTe coordinate once. Base phases affect
            # only the absolute-address swizzle, whose full period is checked
            # below; they do not require rebuilding the same static layout.
            offsets = []
            for member in candidate.members:
                native = eval(_layout_expression(member.native_layout[0]), namespace)
                logical = (
                    native.outer
                    if member.logical_modes == (0, 1)
                    else cute.select(native.outer, mode=[1, 0])
                )
                member_offsets = []
                for row in range(member.physical_shape[0]):
                    for column in range(128):
                        coordinate = (
                            (row, column)
                            if member.logical_modes == (0, 1)
                            else (column, row)
                        )
                        address = member.byte_offset + 2 * int(logical(coordinate))
                        assert address == 2 * int(
                            full.outer((member.row_offset + row, column))
                        )
                        member_offsets.append(address)
                offsets.append(member_offsets)
            for address in range(1024):
                assert int(swizzle(address)) == address ^ (((address >> 7) & 7) << 4)
            for base in range(0, 1024, 128):
                seen: set[int] = set()
                for member, member_offsets in zip(
                    candidate.members, offsets, strict=True
                ):
                    physical = [
                        (base + address) ^ ((((base + address) >> 7) & 7) << 4)
                        for address in member_offsets
                    ]
                    assert all(
                        base + member.byte_offset
                        <= address
                        < base + member.byte_offset + 256 * member.physical_shape[0]
                        for address in physical
                    )
                    assert not seen.intersection(physical)
                    assert len(set(physical)) == len(physical)
                    seen.update(physical)
                    for index in range(0, len(physical), 8):
                        packet = physical[index : index + 8]
                        assert packet[0] % 16 == 0
                        assert packet == list(range(packet[0], packet[0] + 16, 2))
                assert seen == set(range(base, base + candidate.byte_size, 2))
            ptr = cute.make_ptr(element, 0, cute.AddressSpace.smem, assumed_align=128)
            tensor = cute.make_tensor(
                cute.recast_ptr(ptr, full.inner, dtype=element), full.outer
            )
            for rows in (64, 128):
                mma = blackwell_helpers.make_trivial_tiled_mma(
                    element,
                    element,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                    cutlass.Float32,
                    tcgen05.CtaGroup.ONE,
                    (rows, sum(widths)),
                    tcgen05.OperandSource.SMEM,
                )
                fragment = mma.make_fragment_B(mma.get_slice(0).partition_B(tensor))
                assert cute.size(fragment, mode=[2]) == 8
                a_layout = cute.tile_to_shape(
                    tcgen05.make_smem_layout_atom(
                        tcgen05.SmemLayoutAtomKind.K_SW128, element
                    ),
                    (rows, 128),
                    order=(0, 1),
                )
                a_ptr = cute.make_ptr(
                    element, 65536, cute.AddressSpace.smem, assumed_align=128
                )
                a_tensor = cute.make_tensor(
                    cute.recast_ptr(a_ptr, a_layout.inner, dtype=element),
                    a_layout.outer,
                )
                a_fragment = mma.make_fragment_A(mma.get_slice(0).partition_A(a_tensor))
                accumulator = cute.make_tensor(
                    cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem),
                    mma.make_fragment_C(
                        mma.partition_shape_C((rows, sum(widths)))
                    ).layout,
                )
                for atom in range(8):
                    mma.set(tcgen05.Field.ACCUMULATE, atom != 0)
                    cute.gemm(
                        mma,
                        accumulator,
                        a_fragment[None, None, atom],
                        fragment[None, None, atom],
                        accumulator,
                    )
            source = cute.local_tile(tensor, (1, 8), (sum(widths) - 1, 15))
            values = cute.make_rmem_tensor(source.shape, element)
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), element, num_bits_per_copy=128
                ),
                source,
                values,
            )
        manager.PassManager.parse(
            "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        assert str(module).count("nvvm.tcgen05.mma ") == 16


def test_actual_pointer_stores_use_same_parent_and_member_address() -> None:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    _, _, _, placed = _placed((16, 48), torch.bfloat16, (False, True))
    candidate = placed.groups[0].candidate
    ir = importlib.import_module("cutlass._mlir.ir")
    namespace = {"cutlass": cutlass, "cute": cute, "tcgen05": tcgen05}
    expected = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            full = eval(_layout_expression(candidate.native_layout[0]), namespace)
            for base in range(0, 1024, 128):
                ptr = cute.make_ptr(
                    cutlass.BFloat16, base, cute.AddressSpace.smem, assumed_align=128
                )
                parent = cute.make_tensor(
                    cute.recast_ptr(ptr, full.inner, dtype=cutlass.BFloat16), full.outer
                )
                for member in candidate.members:
                    layout = eval(
                        _layout_expression(member.native_layout[0]), namespace
                    )
                    logical = (
                        layout.outer
                        if member.logical_modes == (0, 1)
                        else cute.select(layout.outer, mode=[1, 0])
                    )
                    ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        base + member.byte_offset,
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    image = cute.make_tensor(
                        cute.recast_ptr(ptr, layout.inner, dtype=cutlass.BFloat16),
                        logical,
                    )
                    for row, column in (
                        (0, 0),
                        (7, 63),
                        (8, 64),
                        (member.physical_shape[0] - 1, 127),
                    ):
                        coords = (
                            (row, column)
                            if member.logical_modes == (0, 1)
                            else (column, row)
                        )
                        image[coords] = cutlass.BFloat16(1)
                        parent[row + member.row_offset, column] = cutlass.BFloat16(1)
                        r = row + member.row_offset
                        address = base + 2 * (
                            (r // 8) * 1024
                            + (r % 8) * 64
                            + (column // 64) * 512
                            + column % 64
                        )
                        address ^= ((address >> 7) & 7) << 4
                        expected.extend((address, address))
        assert _lowered_store_addresses(module) == expected


@pytest.mark.parametrize("k", [48, 80, 96, 256])
def test_multi_atom_does_not_relax_unproved_k_layouts(k: int) -> None:
    plan, frame, recurrence = _candidate((32, 32), k, transpose=(True, True))
    assert (
        prepared_group_candidates(
            replace(plan, prepared_group_multi_atom=True), frame, recurrence
        )
        == ()
    )
    assert not _native_row_subview((64, k), (32, k), 0, multi_atom=True)


@pytest.mark.parametrize("rows,origin", [(8, 0), (16, 8), (16, -16), (32, 48)])
def test_multi_atom_rejects_partial_or_out_of_bounds_members(
    rows: int, origin: int
) -> None:
    assert not _native_row_subview((64, 128), (rows, 128), origin, multi_atom=True)


@pytest.mark.parametrize("fault", ["layout", "mode", "offset", "capacity"])
def test_multi_atom_stale_proof_or_capacity_rejects(fault: str) -> None:
    plan, frame, recurrence, placed = _placed((64, 64), torch.bfloat16, (True, True))
    binding = placed.groups[0]
    candidate = binding.candidate
    if fault == "capacity":
        assert (
            coallocate_prepared_groups(
                plan, frame, recurrence, (candidate,), frame_capacity_bytes=128
            )
            is None
        )
        return
    if fault == "layout":
        candidate = replace(
            candidate,
            native_layout=tuple(
                line.replace("order=(1, 0)", "order=(0, 1)")
                for line in candidate.native_layout
            ),
        )
    elif fault == "mode":
        candidate = replace(candidate, multi_atom=False)
    else:
        first = candidate.members[0]
        candidate = replace(
            candidate,
            members=(
                replace(first, byte_offset=first.byte_offset + 128),
                *candidate.members[1:],
            ),
        )
    assert not _group_matches(placed.frame, replace(binding, candidate=candidate))
    assert (
        coallocate_prepared_groups(
            plan, frame, recurrence, (candidate,), frame_capacity_bytes=1 << 20
        )
        is None
    )


@pytest.mark.parametrize("vector", [False, True])
def test_multi_atom_codegen_elides_only_the_grouped_b_fill(vector: bool) -> None:
    options = {_KEY: True, "cute_chained_pointwise_vectorize": vector}
    source = _source(_config(**options))
    assert "chain_0_b = chain_prepared_group_0" in source
    assert "chain_0_b_0_step" not in source and "chain_0_b_1_step" not in source
    assert "order=(1, 0)" in source
    assert "chain_resident_carry" in source
    before = _source(_config(cute_chained_pointwise_vectorize=vector))
    assert before.count("cute.gemm(") == source.count("cute.gemm(") == 2


def test_multi_atom_off_default_and_normalization() -> None:
    assert _source(_config(**{_KEY: False})) == _source()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        assert _KEY not in bound.config_spec.normalized_config(dict(_config()))
        fields = bound.config_spec._flat_fields()
        field = fields[_KEY]
        assert isinstance(field, EnumFragment) and field.default() is False
        with patch.object(
            bound.config_spec,
            "cute_chained_prepared_group_multi_atom_search_enabled",
            False,
        ):
            assert tuple(bound.config_spec._flat_fields()) == tuple(
                key for key in fields if key != _KEY
            )
        for override in (
            {_KEY: 1},
            {_KEY: True, "cute_chained_preparation_pipeline": False},
            {_KEY: True, "cute_chained_group_contractions": False},
        ):
            with pytest.raises(exc.InvalidConfig):
                bound.config_spec.normalized_config(dict(_config(**override)))
        assert bound.host_function is not None
        parents = [_config(), _config(cute_chained_preparation_pipeline=False)]
        with bound.env:
            result = CuteChainedMatmulHeuristic._with_multi_atom_prepared_group_seed(
                bound.env, bound.host_function.device_ir, parents
            )
            assert result[:-1] == parents and result[-1].config[_KEY] is True
            assert (
                CuteChainedMatmulHeuristic._with_multi_atom_prepared_group_seed(
                    bound.env, bound.host_function.device_ir, result
                )
                == result
            )


def test_multi_atom_requires_final_capacity_proof() -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source(_config(**{_KEY: True}))


def test_multi_atom_requires_an_effective_image_and_composes_native_async() -> None:
    with pytest.raises(exc.InvalidConfig, match="discovered grouped TCgen05"):
        _source(
            _config(**{_KEY: True, "cute_chained_loop_mma_rows": 128}), _one_atom_args()
        )
    source = _source(
        _config(
            **{
                _KEY: True,
                "cute_chained_pointwise_vectorize": True,
                "cute_chained_async_vector_store": True,
            }
        )
    )
    assert source.count("cpasync.CopyG2SOp") == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_multi_atom_changing_frames_matches_reference_and_original_gpu(
    dtype: torch.dtype, strided: bool
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    torch.manual_seed(7257)
    args = list(_args("cuda", dtype=dtype, steps=3))
    if strided:
        for index in (0, 1):
            shape = args[index].shape
            args[index] = torch.empty(
                (*shape[:-1], shape[-1] * 2), device="cuda", dtype=dtype
            )[..., ::2]
    for tensor in (*args[:5], args[6]):
        tensor.copy_(torch.randint(-2, 3, tensor.shape, device="cuda") * 0.03125)
    args[5].copy_(torch.randint(1, 4, args[5].shape, device="cuda") * 0.25)
    left, right, value, transform, update, gate, initial, output, final = args
    originals = [tensor.clone() for tensor in args[:7]]
    expected = output.clone()
    state = initial.double()
    for step in range(3):
        entering = state.to(dtype).double()
        narrowed = (
            (value[:, step].double() - right[:, step].double() @ entering)
            .to(dtype)
            .double()
        )
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ narrowed + cross).to(dtype)
        state = (
            update[:, step].double() @ narrowed
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    for enabled in (False, True):
        output.fill_(19)
        final.fill_(-17)
        config = _config(**{_KEY: enabled, "cute_chained_pointwise_vectorize": True})
        actual = _load_only_recurrence._bind_isolated(tuple(args)).compile_config(
            config
        )(*args)
        torch.testing.assert_close(actual[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(actual[1], state.float(), atol=0.002, rtol=0.01)
        torch.testing.assert_close(actual[0][:, -1], expected[:, -1], atol=0, rtol=0)
        results.append(tuple(tensor.clone() for tensor in actual))
    for off, on in zip(*results, strict=True):
        torch.testing.assert_close(off, on, atol=0, rtol=0)
    for tensor, saved in zip(args[:7], originals, strict=True):
        torch.testing.assert_close(tensor, saved, atol=0, rtol=0)
