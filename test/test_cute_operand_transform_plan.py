"""CPU layout, descriptor and cache proofs for the dormant native RNA plan."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandDescriptorPlan
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandTransformKind
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandTransformPlan
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandType
from helion._compiler.cute.tcgen05_operand_transform import prove_operand_stage_span
from helion.runtime.cute.launcher import _cute_disk_cache_key

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
ir = pytest.importorskip("cutlass._mlir.ir")
sm100 = pytest.importorskip("cutlass.utils.blackwell_helpers")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with _mock_cuda_unavailable():

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("GPU forbidden in the operand plan proof")

        monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
        yield


@pytest.fixture
def context() -> Iterator[None]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield


def _descriptor() -> Tcgen05OperandDescriptorPlan:
    return Tcgen05OperandDescriptorPlan(
        Tcgen05OperandType.FLOAT32,
        Tcgen05OperandType.FLOAT32,
        Tcgen05OperandType.TFLOAT32,
    )


def _plan(
    *, a_words: int = 512, b_words: int = 1536, stages: int = 3
) -> Tcgen05OperandTransformPlan:
    a = prove_operand_stage_span(
        cute.make_layout((a_words, stages), stride=(1, a_words + 128)),
        stages=stages,
        alignment_bytes=16,
    )
    b = prove_operand_stage_span(
        cute.make_layout((b_words, stages), stride=(1, b_words + 256)),
        stages=stages,
        alignment_bytes=16,
    )
    assert a is not None and b is not None
    return Tcgen05OperandTransformPlan(
        Tcgen05OperandTransformKind.TF32_RNA, _descriptor(), a, b
    )


def test_separate_operand_spans_and_barrier_units(context: None) -> None:
    plan = _plan()
    assert plan.a.words == 512 and plan.b.words == 1536
    assert plan.a.stride_words == 640 and plan.b.stride_words == 1792
    assert plan.a.allocation_bytes == (2 * 640 + 512) * 4
    assert plan.b.allocation_bytes == (2 * 1792 + 1536) * 4
    assert plan.converted_full_arrivals == 32
    assert plan.scheduler_consumer_warps == 1
    assert ("converter_warps", 1) in plan.cache_identity()
    assert ("converted_full_thread_arrivals", 32) in plan.cache_identity()
    assert ("scheduler_release_warps", 1) in plan.cache_identity()
    assert plan.converted_full_barrier_bytes == 24
    for unproved_count in (0, True, 3, 6, 16):
        with pytest.raises(ValueError, match="proved converter warp choices"):
            replace(plan, converter_warps=unproved_count)


@pytest.mark.parametrize("stages", (1, 2, 3, 6))
@pytest.mark.parametrize("rhs_k_major", (False, True))
def test_actual_sdk_has_separate_dense_stage_proofs(
    context: None, stages: int, rhs_k_major: bool
) -> None:
    mma = sm100.make_trivial_tiled_mma(
        cutlass.TFloat32,
        cutlass.TFloat32,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K
        if rhs_k_major
        else cute.nvgpu.OperandMajorMode.MN,
        cutlass.Float32,
        cute.nvgpu.tcgen05.CtaGroup.ONE,
        (128, 64),
    )
    layouts = (
        sm100.make_smem_layout_a(mma, (128, 64, 32), cutlass.Float32, stages),
        sm100.make_smem_layout_b(mma, (128, 64, 32), cutlass.Float32, stages),
    )
    spans = tuple(
        prove_operand_stage_span(layout, stages=stages, alignment_bytes=1024)
        for layout in layouts
    )
    assert spans[0] is not None and spans[1] is not None
    assert spans[0].words == 4096 and spans[1].words == 2048
    for span, layout in zip(spans, layouts, strict=True):
        assert span is not None
        assert span.stride_words == span.words
        assert span.allocation_bytes == int(cute.cosize(layout.outer)) * 4


@cute.jit
def _record_stage_addresses(
    operand: cutlass.Constexpr[str],
    rhs_k_major: cutlass.Constexpr[bool],
    stages: cutlass.Constexpr[int],
    base_words: cutlass.Constexpr[int],
    output_address: cutlass.Int64,
) -> None:
    if cutlass.const_expr(operand == "synthetic"):
        layout = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 6),
            0,
            cute.make_layout((512, stages), stride=(1, 512)),
        )
    else:
        mma = sm100.make_trivial_tiled_mma(
            cutlass.TFloat32,
            cutlass.TFloat32,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K
            if rhs_k_major
            else cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            (128, 64),
        )
        if cutlass.const_expr(operand == "A"):
            layout = sm100.make_smem_layout_a(
                mma, (128, 64, 32), cutlass.Float32, stages
            )
        else:
            layout = sm100.make_smem_layout_b(
                mma, (128, 64, 32), cutlass.Float32, stages
            )
    count = cute.size(layout)
    byte_swizzle = cute.recast_layout(8, cutlass.Float32.width, layout).inner
    addresses = cute.make_composed_layout(
        byte_swizzle, 0, cute.make_layout((base_words + count) * 4)
    )
    output = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, output_address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout(count),
    )
    for linear in cutlass.range(count):
        relative = cute.crd2idx(cute.idx2crd(linear, layout.shape), layout.outer)
        output[linear] = cute.crd2idx((base_words + relative) * 4, addresses)


@pytest.mark.parametrize("stages", (1, 3, 6))
@pytest.mark.parametrize(
    "operand,rhs_k_major",
    (("A", False), ("B", False), ("B", True), ("synthetic", False)),
)
def test_actual_sdk_byte_swizzle_keeps_nonzero_bases_inside_each_stage(
    operand: str, rhs_k_major: bool, stages: int
) -> None:
    words = {"A": 4096, "B": 2048, "synthetic": 512}[operand]
    # Actual SMEM uses 1024-byte alignment. Also exercise the proof's smaller
    # sufficient alignment with a swizzle whose source includes stage bits.
    bases = (64, 192, 320) if operand == "synthetic" else (256, 512, 768)
    for base_words in bases:
        storage = (ctypes.c_int32 * (words * stages))()
        address = ctypes.addressof(storage)
        compiled = cute.compile(
            _record_stage_addresses,
            operand,
            rhs_k_major,
            stages,
            base_words,
            cutlass.Int64(address),
            options="--gpu-arch sm_100a",
        )
        # CPU host JIT: ordinary CPU memory, no GPU kernel or launch.
        compiled(address)
        for stage in range(stages):
            actual = list(storage[stage * words : (stage + 1) * words])
            first = (base_words + stage * words) * 4
            assert len(set(actual)) == words
            assert set(actual) == set(range(first, first + words * 4, 4))


@pytest.mark.parametrize(
    "shape,stride,stages",
    (
        ((16, 32, 3), (1, 1, 512), 3),  # repeated logical addresses
        ((16, 32, 3), (64, 1, 1024), 3),  # hole inside a stage
        ((512, 3), (1, 256), 3),  # overlap between stages
        ((512, 3), (1, 514), 3),  # misaligned next packet
        ((512, 3), (1, 512), 2),  # stage-count disagreement
        ((64, 3), (1, 64), 3),  # not a complete 32-lane packet iteration
    ),
)
def test_non_bijective_or_incomplete_spans_decline(
    context: None, shape: tuple[int, ...], stride: tuple[int, ...], stages: int
) -> None:
    layout = cute.make_layout(shape, stride=stride)
    assert prove_operand_stage_span(layout, stages=stages, alignment_bytes=1024) is None


@pytest.mark.parametrize("alignment", (0, 4, 12, 16, 128))
def test_swizzle_requires_actual_base_alignment(context: None, alignment: int) -> None:
    layout = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 6),
        0,
        cute.make_layout((512, 3), stride=(1, 512)),
    )
    assert prove_operand_stage_span(layout, stages=3, alignment_bytes=alignment) is None
    # The XOR source includes stage bits, which is safe: its destination bits
    # remain inside each aligned stage and XOR is a bijection on those bits.
    assert prove_operand_stage_span(layout, stages=3, alignment_bytes=256) is not None


@pytest.mark.parametrize("offset", (1, 128))
def test_nonzero_composition_offsets_decline(context: None, offset: int) -> None:
    layout = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3),
        offset,
        cute.make_layout((512, 3), stride=(1, 512)),
    )
    assert prove_operand_stage_span(layout, stages=3, alignment_bytes=1024) is None


def test_swizzle_cannot_move_a_word_into_another_stage(context: None) -> None:
    layout = cute.make_composed_layout(
        cute.make_swizzle(3, 7, 3),
        0,
        cute.make_layout((512, 3), stride=(1, 512)),
    )
    assert prove_operand_stage_span(layout, stages=3, alignment_bytes=4096) is None


def test_dynamic_layout_facts_are_not_static_proof(context: None) -> None:
    layout = cute.make_layout((cutlass.Int32(512), 3), stride=(1, 512))
    assert prove_operand_stage_span(layout, stages=3, alignment_bytes=1024) is None
    offset_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3),
        cutlass.Int32(0),
        cute.make_layout((512, 3), stride=(1, 512)),
    )
    assert (
        prove_operand_stage_span(offset_layout, stages=3, alignment_bytes=1024) is None
    )


def test_descriptor_policy_cannot_substitute_tma_rn(context: None) -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="raw Float32 TMA"):
        replace(
            plan,
            descriptor=replace(
                plan.descriptor, tma_internal_type=Tcgen05OperandType.TFLOAT32
            ),
        )
    with pytest.raises(ValueError, match="pipeline depth"):
        replace(plan, b=replace(plan.b, stages=2))
    with pytest.raises(ValueError, match="typed enum"):
        Tcgen05OperandDescriptorPlan(
            cast("Tcgen05OperandType", "cutlass.Float32"),
            plan.descriptor.tma_internal_type,
            plan.descriptor.mma_type,
        )


@pytest.mark.parametrize("operand", ("A", "B"))
def test_actual_raw_tma_descriptor_preserves_tf32_compute_geometry(
    context: None, operand: str
) -> None:
    mma = sm100.make_trivial_tiled_mma(
        cutlass.TFloat32,
        cutlass.TFloat32,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        cutlass.Float32,
        cute.nvgpu.tcgen05.CtaGroup.ONE,
        (128, 64),
    )
    descriptor = _descriptor()
    assert descriptor.mma_type is Tcgen05OperandType.TFLOAT32
    shape, stride = (
        ((257, 96), (96, 1))
        if operand == "A"
        else (
            (64, 96, 5),
            (1, 64, 6144),
        )
    )
    tensor = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout(shape, stride=stride),
    )
    layout_fn = sm100.make_smem_layout_a if operand == "A" else sm100.make_smem_layout_b
    atom_fn = (
        cute.nvgpu.make_tiled_tma_atom_A
        if operand == "A"
        else cute.nvgpu.make_tiled_tma_atom_B
    )
    layout = layout_fn(mma, (128, 64, 32), cutlass.Float32, 6)
    common = (
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cute.nvgpu.tcgen05.CtaGroup.ONE),
        tensor,
        cute.slice_(layout, (None, None, None, 0)),
        (128, 64, 32),
        mma,
        (1, 1, 1, 1),
    )
    raw_atom, raw_coords = atom_fn(*common, internal_type=cutlass.Float32)
    rn_atom, rn_coords = atom_fn(*common, internal_type=cutlass.TFloat32)
    assert "tma_format = F32_RN" in str(raw_atom.type)
    assert "tma_format = TF32_RN" in str(rn_atom.type)
    assert str(raw_atom.type) == str(rn_atom.type).replace("TF32_RN", "F32_RN")
    assert str(raw_coords) == str(rn_coords)


def _cache_key(plan: Tcgen05OperandTransformPlan) -> str | None:
    wrapper = {"kind": "tcgen05_ab_tma", "operand_transform": plan.cache_identity()}
    kernel = SimpleNamespace(
        _helion_cute_source_hash="unchanged generated source",
        _helion_cute_wrapper_plans=[wrapper],
    )
    return _cute_disk_cache_key(
        kernel, (), (32, 8, 1), (repr(wrapper),), (1, 1, 1), "--enable-tvm-ffi", 148
    )


def test_existing_cache_key_consumes_distinct_operand_plan_fields(
    context: None,
) -> None:
    plans = (
        _plan(),
        _plan(stages=6),
        _plan(a_words=1024),
        _plan(b_words=2048),
        replace(_plan(), a=_plan().b, b=_plan().a),
        replace(_plan(), a=replace(_plan().a, alignment_bytes=32)),
    )
    keys = tuple(_cache_key(plan) for plan in plans)
    assert all(key is not None for key in keys)
    assert len(set(keys)) == len(plans)
    assert _cache_key(_plan()) == keys[0]
