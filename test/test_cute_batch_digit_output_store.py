"""CPU tests for batch-digit (mixed-radix permuted) batched-GEMM output stores.

A batched matmul whose output is stored through a complete pair of batch
digits (e.g. Inductor's head-split ``out[b // H, m, b % H, n]``) is admitted to
the native tcgen05 path. The store goes through the ordinary SIMT epilogue,
using a per-batch (M, N) plane view of the original output pointer. Every
unproved form keeps the existing refusal. No CUDA work and no native
compilation happen here.
"""

from __future__ import annotations

import itertools
import random
import re
from unittest.mock import patch

import pytest
import sympy
import torch

import helion
from helion import exc
from helion._compiler.cute import cute_mma
from helion._compiler.cute.mapped_output import BatchDigitOutputMap
from helion._compiler.cute.mapped_output import prove_batch_digit_output_dims
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

# Byte-exact native module Inductor generated for a batched matmul whose output
# is stored through a head-split permutation. It is a portable fixture, not a
# changed frontend or benchmark.
INDUCTOR_HEAD_SPLIT_BMM = """import torch
import helion
import helion.language as hl

from torch._inductor.runtime.helion import HelionKernelWrapper
@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full", ref_mode=helion.RefMode.OFF)
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([512, 512, 64]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = hl.full([], 0.3535533905932738, dtype=torch.float32)
        tmp5 = acc * tmp4
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 64, ((tile_y.index.reshape([1, tile_y, 1]) + 512*tile_z.index.reshape([tile_z, 1, 1])) % 512), (tile_z.index.reshape([tile_z, 1, 1]) % 64), ((tile_x.index.reshape([1, 1, tile_x]) + 64*tile_y.index.reshape([1, tile_y, 1]) + 32768*tile_z.index.reshape([tile_z, 1, 1])) % 64)] = tmp5.to(torch.bfloat16)
helion_op0_op1 = HelionKernelWrapper(helion_op0_op1)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, buf1)
"""

Z, M, N, K, H = 8, 256, 64, 128, 4


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def head_split(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    # out: [Z // H, M, H, N]
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, (yi + m * zi) % m, zi % heads, (xi + n * yi) % n] = (
            acc * 0.5
        ).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def head_major(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    # out: [H, Z // H, M, N]; the inner digit is the outer destination dim.
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(0)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi % heads, zi // heads, yi, xi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def column_plane(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    # out: [Z // H, N, H, M]; the (M, N) plane is M-contiguous.
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, xi, zi % heads, yi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def wrapped_row(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, (yi + 1) % m, zi % heads, xi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def duplicate_digit(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, yi, zi // heads, xi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def data_dependent(
    A: torch.Tensor, B: torch.Tensor, perm: torch.Tensor, out: torch.Tensor
) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = perm[tz].reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, yi, zi % heads, xi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def rank1_m_index(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    # ``ty.index`` stays rank-1, so torch broadcasting runs it along N.
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, (ty.index + m * zi) % m, zi % heads, xi] = acc.to(
            torch.bfloat16
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def rank1_n_index(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    # A rank-1 N index already broadcasts along its own (last) axis.
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        out[zi // heads, yi, zi % heads, (tx.index + n * zi) % n] = acc.to(
            torch.bfloat16
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def masked_store(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        hl.store(
            out,
            [zi // heads, yi, zi % heads, xi],
            acc.to(torch.bfloat16),
            extra_mask=xi < n - 1,
        )


def _a() -> torch.Tensor:
    return torch.empty((Z, M, K), dtype=torch.bfloat16)


def _b() -> torch.Tensor:
    return torch.empty((Z, K, N), dtype=torch.bfloat16)


def _bind(kernel: helion.Kernel, args: tuple[object, ...]):
    fresh = helion.kernel(
        kernel.fn,
        settings=kernel.settings,
        cute_structural_policy=kernel.cute_structural_policy,
    )
    with patch.object(fresh, "_bind_isolated", side_effect=fresh.bind):
        return cpu_bind(fresh, args)


def _mma_candidate(bound):
    device_ir = bound.host_function.device_ir
    (node,) = [
        node
        for graph_info in device_ir.graphs
        for node in graph_info.graph.nodes
        if node.op == "call_function"
        and cute_mma._decode_cute_mma_target(node, device_ir=device_ir) is not None
    ]
    return node, cute_mma.analyze_cute_mma_node(node, device_ir=device_ir)


def _digit_map(bound) -> BatchDigitOutputMap | None:
    with bound.env, bound.host_function:
        _node, candidate = _mma_candidate(bound)
        return cute_mma._candidate_batch_digit_output(candidate)


def _first_native_config(bound) -> helion.Config:
    with bound.env, bound.host_function:
        generation = ConfigGeneration(bound.config_spec).initial_population_view()
        return generation.unflatten(generation.default_flat())


def _native_store_facts(code: str) -> dict[str, object]:
    return {
        "native": "'kind': 'tcgen05_ab_tma'" in code
        and "tcgen05_tma_warp" in code
        and "cute.gemm(" in code,
        "d_tma": "'kind': 'tcgen05_d_tma'" in code,
        "tma_store_pipeline": "PipelineTmaStore.create(" in code,
        "python_mod_emitted": "_cute_python_mod(" in code,
    }


# ---------------------------------------------------------------------------
# Actual archived Inductor module, default settings, managed bind.
# ---------------------------------------------------------------------------


def test_actual_inductor_head_split_module_admits_native_path_with_plane_store(
    tmp_path,
) -> None:
    from torch._inductor.codecache import PyCodeCache

    module = PyCodeCache.load(INDUCTOR_HEAD_SPLIT_BMM)
    original = module.helion_op0_op1.kernel
    args = (
        torch.empty((512, 512, 512), dtype=torch.bfloat16),
        torch.empty_strided((512, 512, 64), (32768, 1, 512), dtype=torch.bfloat16),
        torch.empty((8, 512, 64, 64), dtype=torch.bfloat16),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(original, args)
        assert bound.config_spec.cute_tcgen05_search_enabled
        mapping = _digit_map(bound)
        assert mapping is not None
        assert (mapping.outer_dim, mapping.inner_dim, mapping.m_dim, mapping.n_dim) == (
            0,
            2,
            1,
            3,
        )
        assert mapping.period == 64
        assert mapping.extents == (512, 512, 64)
        assert not mapping.plane_column_major
        config = _first_native_config(bound)
        code = bound.to_code(config)
    facts = _native_store_facts(code)
    assert facts == {
        "native": True,
        "d_tma": False,
        "tma_store_pipeline": False,
        "python_mod_emitted": False,
    }
    # B is K-contiguous per batch and staged through its rank-3 TensorMap.
    assert "'b_k_major': True" in code
    assert "'rhs_tma_order': (2, 1, 0)" in code
    # Exact plane base and plane layout over the original output pointer.
    assert re.search(
        r"cute\.make_tensor\(out0\.iterator \+ cute\.crd2idx\(\("
        r"cutlass\.Int32\((\w+)\) // cutlass\.Int32\(64\), cutlass\.Int32\(0\), "
        r"cutlass\.Int32\(\1\) % cutlass\.Int32\(64\), cutlass\.Int32\(0\)\), "
        r"out0\.layout\), cute\.make_layout\(\(out0\.shape\[1\], out0\.shape\[3\]\), "
        r"stride=\(out0\.layout\.stride\[1\], out0\.layout\.stride\[3\]\)\)\)",
        code,
    )
    assert "cute.make_identity_tensor((512, 64))" in code


# ---------------------------------------------------------------------------
# General positives through the frontend.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "out_shape", "dims", "column"),
    [
        pytest.param(
            head_split, (Z // H, M, H, N), (0, 2, 1, 3), False, id="head_split"
        ),
        pytest.param(
            head_major, (H, Z // H, M, N), (1, 0, 2, 3), False, id="head_major"
        ),
        pytest.param(
            column_plane, (Z // H, N, H, M), (0, 2, 3, 1), True, id="column_plane"
        ),
    ],
)
def test_general_digit_orders_and_planes_are_proved_and_lowered(
    kernel, out_shape, dims, column
) -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(
            kernel, (_a(), _b(), torch.empty(out_shape, dtype=torch.bfloat16))
        )
        assert bound.config_spec.cute_tcgen05_search_enabled
        mapping = _digit_map(bound)
        assert mapping is not None
        assert (
            mapping.outer_dim,
            mapping.inner_dim,
            mapping.m_dim,
            mapping.n_dim,
        ) == dims
        assert mapping.plane_column_major is column
        assert mapping.owned_index_nodes
        code = bound.to_code(_first_native_config(bound))
    facts = _native_store_facts(code)
    assert facts["native"] and not facts["d_tma"] and not facts["tma_store_pipeline"]
    assert not facts["python_mod_emitted"]
    outer, inner, m, n = dims
    coords = ["cutlass.Int32(0)"] * 4
    coords[outer] = r"cutlass\.Int32\((\w+)\) // cutlass\.Int32\(4\)"
    coords[inner] = r"cutlass\.Int32\(\w+\) % cutlass\.Int32\(4\)"
    pattern = (
        r"cute\.make_tensor\(out\.iterator \+ cute\.crd2idx\(\("
        + ", ".join(c if "Int32\\(" in c else re.escape(c) for c in coords)
        + rf"\), out\.layout\), cute\.make_layout\(\(out\.shape\[{m}\], out\.shape\[{n}\]\), "
        rf"stride=\(out\.layout\.stride\[{m}\], out\.layout\.stride\[{n}\]\)\)\)"
    )
    assert re.search(pattern, code), pattern


# ---------------------------------------------------------------------------
# Refusals through the frontend: the existing refusal stays.
# ---------------------------------------------------------------------------


def _assert_refused(bound) -> None:
    assert _digit_map(bound) is None
    with bound.env, bound.host_function:
        _node, candidate = _mma_candidate(bound)
    assert candidate is None
    assert not bound.config_spec.cute_tcgen05_search_enabled


def test_wrapped_index_is_refused() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(
            _bind(
                wrapped_row,
                (_a(), _b(), torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)),
            )
        )


def test_duplicate_or_missing_digit_is_refused() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(
            _bind(
                duplicate_digit,
                (_a(), _b(), torch.empty((Z // H, M, Z // H, N), dtype=torch.bfloat16)),
            )
        )


def test_data_dependent_index_is_refused() -> None:
    perm = torch.empty((Z,), dtype=torch.int32)
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(
            _bind(
                data_dependent,
                (
                    _a(),
                    _b(),
                    perm,
                    torch.empty((Z // H, M, H, N), dtype=torch.bfloat16),
                ),
            )
        )


@pytest.mark.parametrize("square", [False, True], ids=["m256_n64", "m64_n64"])
def test_rank1_index_broadcast_along_another_axis_is_refused(square) -> None:
    rows = N if square else M
    args = (
        torch.empty((Z, rows, K), dtype=torch.bfloat16),
        _b(),
        torch.empty((Z // H, rows, H, N), dtype=torch.bfloat16),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        try:
            bound = _bind(rank1_m_index, args)
        except exc.ShapeMismatch:
            return  # the frontend itself rejects the mismatched broadcast
        assert _digit_map(bound) is None


def test_rank1_index_along_its_own_axis_is_proved() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(
            rank1_n_index,
            (_a(), _b(), torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)),
        )
        mapping = _digit_map(bound)
        assert mapping is not None
        assert (mapping.outer_dim, mapping.inner_dim, mapping.m_dim, mapping.n_dim) == (
            0,
            2,
            1,
            3,
        )


def test_masked_store_is_refused() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(
            _bind(
                masked_store,
                (_a(), _b(), torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)),
            )
        )


@pytest.mark.parametrize(
    "make_out",
    [
        pytest.param(
            lambda: torch.empty((Z // H, M, H, 2 * N), dtype=torch.bfloat16)[..., :N],
            id="non_dense_gap",
        ),
        pytest.param(
            lambda: torch.empty((Z // H, M, N, H), dtype=torch.bfloat16).permute(
                0, 1, 3, 2
            ),
            id="inner_digit_unit_stride",
        ),
    ],
)
def test_destination_layout_refusals(make_out) -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(_bind(head_split, (_a(), _b(), make_out())))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def reads_destination(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        previous = out[zi // heads, yi, zi % heads, xi]
        out[zi // heads, yi, zi % heads, xi] = (acc + previous).to(torch.bfloat16)


def test_another_access_to_the_destination_is_refused() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(
            _bind(
                reads_destination,
                (_a(), _b(), torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)),
            )
        )


def test_destination_shared_with_an_input_is_refused() -> None:
    from helion._compiler.cute.mapped_output import _destination_storage_proven_disjoint

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(
            head_split,
            (_a(), _b(), torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)),
        )
        mapping = _digit_map(bound)
        assert mapping is not None
        destination = mapping.store_node.args[0]
        assert isinstance(destination, torch.fx.Node)
        out = destination.meta["val"]
        with bound.env, bound.host_function:
            _node, candidate = _mma_candidate(bound)
            assert candidate is not None
            inputs = (
                candidate.operands.lhs.source_fake,
                candidate.operands.rhs.source_fake,
            )
            assert _destination_storage_proven_disjoint(
                bound.env, out, inputs, allow_unbound=True
            )
            # The same tensor object passed as an input and as the output
            # (Helion's alias specialization binds one fake for both).
            assert not _destination_storage_proven_disjoint(
                bound.env, out, (*inputs, out), allow_unbound=True
            )


def test_runtime_storage_offset_is_addressed_through_the_output_pointer() -> None:
    # The plane base is added to the runtime output pointer, so a storage
    # offset needs no new proof. Vector-store alignment follows the bound
    # kernel's pointer specialization: the 2-byte-offset output gets no
    # 16-byte chunk claim, while the aligned output keeps it.
    aligned = torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)
    offset = torch.empty(Z * M * N + 1, dtype=torch.bfloat16)[1:].view(Z // H, M, H, N)
    codes = {}
    with cpu_codegen_target(), forbid_cuda_work():
        for label, out in (("aligned", aligned), ("offset", offset)):
            bound = _bind(head_split, (_a(), _b(), out))
            assert _digit_map(bound) is not None
            codes[label] = bound.to_code(_first_native_config(bound))
    for code in codes.values():
        assert "cute.make_tensor(out.iterator + cute.crd2idx((" in code
        assert "cute.AddressSpace.gmem, assumed_align=16" not in code
    claim = "assumed_align=min(out.iterator.alignment, 16)"
    assert claim in codes["aligned"]
    assert claim not in codes["offset"]


def test_dynamic_shapes_are_refused() -> None:
    dynamic = helion.kernel(
        head_split.fn,
        settings=head_split.settings.copy(static_shapes=False),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(
            _bind(
                dynamic,
                (_a(), _b(), torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)),
            )
        )


def test_rebind_with_another_layout_reproves_or_refuses() -> None:
    contiguous = torch.empty((Z // H, M, H, N), dtype=torch.bfloat16)
    # Same logical shape, a different dense permutation of the storage.
    permuted = torch.empty((H, Z // H, M, N), dtype=torch.bfloat16).permute(1, 2, 0, 3)
    gapped = torch.empty((Z // H, M, H, 2 * N), dtype=torch.bfloat16)[..., :N]
    with cpu_codegen_target(), forbid_cuda_work():
        first = _digit_map(_bind(head_split, (_a(), _b(), contiguous)))
        second = _digit_map(_bind(head_split, (_a(), _b(), permuted)))
        third = _digit_map(_bind(head_split, (_a(), _b(), gapped)))
    assert first is not None and second is not None and third is None
    assert first.strides == tuple(contiguous.stride())
    assert second.strides == tuple(permuted.stride())
    assert first.strides != second.strides
    for mapping in (first, second):
        offsets = sorted(
            itertools.starmap(
                mapping.offset, itertools.product(range(Z), range(M), range(N))
            )
        )
        assert offsets == list(range(Z * M * N))


# ---------------------------------------------------------------------------
# The static proof itself (no tensors) and small enumerated bijections.
# ---------------------------------------------------------------------------

_b_sym, _m_sym, _n_sym = (
    sympy.Symbol(name, integer=True, nonnegative=True) for name in ("b", "m", "n")
)
_SYMS = (_b_sym, _m_sym, _n_sym)


def _rank4(values: tuple[int, ...]) -> tuple[int, int, int, int]:
    first, second, third, fourth = values
    return first, second, third, fourth


def _expr(value: object) -> sympy.Expr:
    assert isinstance(value, sympy.Expr)
    return value


def _integer_value(value: object) -> int:
    assert isinstance(value, sympy.Integer)
    return int(value)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    span = 1
    for size in reversed(shape):
        strides.append(span)
        span *= size
    return tuple(reversed(strides))


@pytest.mark.parametrize(
    ("indices", "extents", "shape", "strides", "expected"),
    [
        pytest.param(
            [sympy.floor(_b_sym / 3), _m_sym, sympy.Mod(_b_sym, 3), _n_sym],
            (6, 4, 8),
            (2, 4, 3, 8),
            _contiguous_strides((2, 4, 3, 8)),
            (0, 2, 1, 3),
            id="head_split",
        ),
        pytest.param(
            [
                sympy.Mod(sympy.Add(_b_sym, sympy.Mul(6, _m_sym)), 3),
                sympy.floor(_b_sym / 3),
                _m_sym,
                _n_sym,
            ],
            (6, 4, 8),
            (3, 2, 4, 8),
            _contiguous_strides((3, 2, 4, 8)),
            (1, 0, 2, 3),
            id="cancelled_multiple",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 3), _n_sym, sympy.Mod(_b_sym, 3), _m_sym],
            (6, 4, 8),
            (2, 8, 3, 4),
            _contiguous_strides((2, 8, 3, 4)),
            (0, 2, 3, 1),
            id="column_plane",
        ),
    ],
)
def test_static_proof_positives(indices, extents, shape, strides, expected) -> None:
    assert (
        prove_batch_digit_output_dims(indices, _SYMS, extents, shape, strides)
        == expected
    )


@pytest.mark.parametrize(
    ("indices", "extents", "shape", "strides"),
    [
        pytest.param(
            [sympy.floor(_b_sym / 4), _m_sym, sympy.Mod(_b_sym, 4), _n_sym],
            (6, 4, 8),
            (2, 4, 4, 8),
            _contiguous_strides((2, 4, 4, 8)),
            id="period_does_not_divide_batch",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 3), _m_sym, sympy.floor(_b_sym / 3), _n_sym],
            (6, 4, 8),
            (2, 4, 2, 8),
            _contiguous_strides((2, 4, 2, 8)),
            id="duplicate_digit",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 3), _m_sym, sympy.Integer(0), _n_sym],
            (6, 4, 8),
            (2, 4, 3, 8),
            _contiguous_strides((2, 4, 3, 8)),
            id="missing_remainder",
        ),
        pytest.param(
            [_b_sym, sympy.floor(_m_sym / 2), sympy.Mod(_m_sym, 2), _n_sym],
            (6, 4, 8),
            (6, 2, 2, 8),
            _contiguous_strides((6, 2, 2, 8)),
            id="m_axis_split",
        ),
        pytest.param(
            [
                sympy.floor(_b_sym / 3),
                sympy.Mod(sympy.Add(_m_sym, 1), 4),
                sympy.Mod(_b_sym, 3),
                _n_sym,
            ],
            (6, 4, 8),
            (2, 4, 3, 8),
            _contiguous_strides((2, 4, 3, 8)),
            id="wrapped_offset",
        ),
        pytest.param(
            [
                sympy.floor(_b_sym / 3),
                sympy.Add(_m_sym, 1),
                sympy.Mod(_b_sym, 3),
                _n_sym,
            ],
            (6, 4, 8),
            (2, 5, 3, 8),
            _contiguous_strides((2, 5, 3, 8)),
            id="constant_offset",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 3), _m_sym, sympy.Mod(_b_sym, 3), _n_sym],
            (6, 4, 8),
            (2, 4, 3, 8),
            (192, 48, 8, 1),
            id="non_dense",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 3), _m_sym, sympy.Mod(_b_sym, 3), _n_sym],
            (6, 4, 8),
            (2, 4, 3, 8),
            (96, 24, 0, 1),
            id="overlapping_zero_stride",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 3), _m_sym, _n_sym, sympy.Mod(_b_sym, 3)],
            (6, 4, 8),
            (2, 4, 8, 3),
            _contiguous_strides((2, 4, 8, 3)),
            id="inner_digit_unit_stride",
        ),
        pytest.param(
            [sympy.floor(_b_sym / 2048), _m_sym, sympy.Mod(_b_sym, 2048), _n_sym],
            (4096, 1024, 1024),
            (2, 1024, 2048, 1024),
            _contiguous_strides((2, 1024, 2048, 1024)),
            id="int32_overflow",
        ),
    ],
)
def test_static_proof_refusals(indices, extents, shape, strides) -> None:
    assert (
        prove_batch_digit_output_dims(indices, _SYMS, extents, shape, strides) is None
    )


@pytest.mark.parametrize(
    ("extents", "shape", "order"),
    [
        ((6, 4, 8), (2, 4, 3, 8), (0, 2, 1, 3)),
        ((6, 4, 8), (3, 2, 4, 8), (1, 0, 2, 3)),
        ((12, 2, 4), (4, 4, 3, 2), (2, 1, 3, 0)),
        ((10, 3, 5), (5, 2, 5, 3), (1, 0, 3, 2)),
    ],
)
def test_enumerated_bijection_matches_static_proof(extents, shape, order) -> None:
    outer, inner, m, n = order
    period = shape[inner]
    by_dim: dict[int, sympy.Expr] = {
        outer: _expr(sympy.floor(_b_sym / period)),
        inner: _expr(sympy.Mod(_b_sym, period)),
        m: _m_sym,
        n: _n_sym,
    }
    indices = [by_dim[dim] for dim in range(4)]
    strides = _contiguous_strides(shape)
    dims = prove_batch_digit_output_dims(indices, _SYMS, extents, shape, strides)
    if 1 not in (strides[m], strides[n]):
        assert dims is None
        return
    assert dims is not None and dims == order
    outer_dim, inner_dim, m_dim, n_dim = dims
    mapping = BatchDigitOutputMap(
        store_node=torch.fx.Graph().placeholder("store"),
        shape=_rank4(shape),
        strides=_rank4(strides),
        extents=extents,
        outer_dim=outer_dim,
        inner_dim=inner_dim,
        m_dim=m_dim,
        n_dim=n_dim,
    )
    offsets = list(
        itertools.starmap(
            mapping.offset, itertools.product(*(range(e) for e in extents))
        )
    )
    assert sorted(offsets) == list(range(len(offsets)))
    # Each logical element lands where the original subscripts point.
    for b, r, c in itertools.product(*(range(e) for e in extents)):
        coordinates = [
            _integer_value(index.subs({_b_sym: b, _m_sym: r, _n_sym: c}))
            for index in indices
        ]
        assert mapping.offset(b, r, c) == sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, strides, strict=True)
        )


def test_store_analysis_unchanged_for_exact_rank3_store() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
    def plain(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
        z, m, k = A.size()
        n = B.size(2)
        for tz, ty, tx in hl.tile([z, m, n]):
            acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
            for tr in hl.tile(k):
                acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
            out[tz, ty, tx] = acc.to(torch.bfloat16)

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(plain, (_a(), _b(), torch.empty((Z, M, N), dtype=torch.bfloat16)))
        assert bound.config_spec.cute_tcgen05_search_enabled
        assert _digit_map(bound) is None
        with bound.env, bound.host_function:
            _node, candidate = _mma_candidate(bound)
        assert candidate is not None
        assert candidate.output_store_analysis is not None
        assert candidate.output_store_analysis.batch_digit_output is None


# ---------------------------------------------------------------------------
# Runtime storage aliasing: a registered POSITIVE disjointness fact is
# required for the destination against every operand.
# ---------------------------------------------------------------------------

_OUT_NUMEL = Z * M * N
_A_NUMEL = Z * M * K


def _alias_args(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_shape = (Z // H, M, H, N)
    if case == "different_storage":
        return _a(), _b(), torch.empty(out_shape, dtype=torch.bfloat16)
    if case == "equal_views":
        # A and out start at the same element of one storage.
        storage = torch.empty(_A_NUMEL, dtype=torch.bfloat16)
        return storage.view(Z, M, K), _b(), storage[:_OUT_NUMEL].view(out_shape)
    if case == "partial_overlap_128_bytes":
        storage = torch.empty(_A_NUMEL + _OUT_NUMEL - 64, dtype=torch.bfloat16)
        a = storage[:_A_NUMEL].view(Z, M, K)
        out = storage[_A_NUMEL - 64 :].view(out_shape)
        return a, _b(), out
    if case == "disjoint_views_same_storage":
        storage = torch.empty(_A_NUMEL + _OUT_NUMEL, dtype=torch.bfloat16)
        a = storage[:_A_NUMEL].view(Z, M, K)
        out = storage[_A_NUMEL:].view(out_shape)
        return a, _b(), out
    if case == "rhs_disjoint_views_same_storage":
        b_numel = Z * K * N
        storage = torch.empty(b_numel + _OUT_NUMEL, dtype=torch.bfloat16)
        b = storage[:b_numel].view(Z, K, N)
        out = storage[b_numel:].view(out_shape)
        return _a(), b, out
    raise AssertionError(case)


@pytest.mark.parametrize(
    ("case", "admitted"),
    [
        pytest.param("different_storage", True, id="different_storage"),
        pytest.param("equal_views", False, id="equal_views"),
        pytest.param("partial_overlap_128_bytes", False, id="partial_overlap"),
        pytest.param(
            "disjoint_views_same_storage", False, id="disjoint_views_same_storage"
        ),
        pytest.param(
            "rhs_disjoint_views_same_storage",
            False,
            id="rhs_disjoint_views_same_storage",
        ),
    ],
)
def test_destination_requires_positive_storage_disjointness(case, admitted) -> None:
    from helion._compiler.runtime_tensor_guards import (
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
    )

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(head_split, _alias_args(case))
        assert (
            _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
            in bound.env.runtime_input_specializations
        )
        if admitted:
            assert _digit_map(bound) is not None
            assert bound.config_spec.cute_tcgen05_search_enabled
            code = bound.to_code(_first_native_config(bound))
            assert "cute.make_tensor(out.iterator + cute.crd2idx((" in code
        else:
            _assert_refused(bound)


def test_destination_equal_to_an_operand_object_is_refused() -> None:
    from helion._compiler.cute.mapped_output import _destination_storage_proven_disjoint

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(head_split, _alias_args("different_storage"))
        with bound.env, bound.host_function:
            _node, candidate = _mma_candidate(bound)
            assert candidate is not None
            mapping = cute_mma._candidate_batch_digit_output(candidate)
            assert mapping is not None
            destination = mapping.store_node.args[0]
            assert isinstance(destination, torch.fx.Node)
            out = destination.meta["val"]
            inputs = (
                candidate.operands.lhs.source_fake,
                candidate.operands.rhs.source_fake,
            )
            assert _destination_storage_proven_disjoint(
                bound.env, out, inputs, allow_unbound=False
            )
            assert not _destination_storage_proven_disjoint(
                bound.env, out, (*inputs, out), allow_unbound=False
            )


def test_cached_rebind_with_overlapping_views_selects_another_binding() -> None:
    kernel = helion.kernel(head_split.fn, settings=head_split.settings)
    disjoint = _alias_args("different_storage")
    overlapping = _alias_args("disjoint_views_same_storage")
    with cpu_codegen_target(), forbid_cuda_work():
        with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
            first = cpu_bind(kernel, disjoint)
            second = cpu_bind(kernel, overlapping)
            again = cpu_bind(kernel, _alias_args("different_storage"))
        assert first is not second
        assert again is first
        assert _digit_map(first) is not None
        _assert_refused(second)


# ---------------------------------------------------------------------------
# Explicit root thread counts must agree with the specialized-loop detection.
# ---------------------------------------------------------------------------

# The archived production config of the Inductor head-split module (explicit
# ``num_threads`` and the collective request). Its native launch would be
# 12 threads without the TMA/MMA role warps.
SAVED_HEAD_SPLIT_CONFIG = {
    "block_sizes": [1, 128, 64, 32],
    "num_threads": [0, 2, 64, 1],
    "loop_orders": [[2, 1, 0]],
    "load_eviction_policies": ["", "streaming"],
    "cute_store_policy": "default",
    "cute_proven_bounds": True,
    "cute_pdl": False,
    "cute_pdl_spread": True,
    "cute_vector_widths": [1, 1, 1, 1],
    "cute_lane_layouts": ["blocked", "blocked", "blocked", "blocked"],
    "cute_collective_mma": True,
    "cute_collective_compute": "tcgen05",
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_seed": False,
    "cute_collective_tmem_a": True,
    "cute_collective_stages": 3,
    "cute_collective_copy": "async_cached",
    "cute_collective_recipe": "vector_unrolled",
    "cute_collective_operand_packets": False,
    "cute_collective_epilogue": "vector_unrolled",
    "cute_gathered_mma_n": 256,
    "cute_gathered_mma_stages": 3,
    "cute_register_chain": False,
    "cute_min_blocks_per_mp": 0,
}


def _launch_block(code: str) -> tuple[int, ...]:
    match = re.search(r"_launcher\([^\n]*block=\(([^)]*)\)\)", code)
    assert match is not None
    return tuple(int(part) for part in match.group(1).split(","))


def _inductor_head_split_bound():
    from torch._inductor.codecache import PyCodeCache

    module = PyCodeCache.load(INDUCTOR_HEAD_SPLIT_BMM)
    args = (
        torch.empty((512, 512, 512), dtype=torch.bfloat16),
        torch.empty_strided((512, 512, 64), (32768, 1, 512), dtype=torch.bfloat16),
        torch.empty((8, 512, 64, 64), dtype=torch.bfloat16),
    )
    return _bind(module.helion_op0_op1.kernel, args)


def test_saved_explicit_thread_config_keeps_the_collective_lowering() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _inductor_head_split_bound()
        assert bound.config_spec.cute_tcgen05_search_enabled
        code = bound.to_code(helion.Config.from_dict(SAVED_HEAD_SPLIT_CONFIG))
    assert "'kind': 'tcgen05_ab_tma'" not in code
    assert "tcgen05_tma_warp" not in code
    assert "collective_tmem" in code
    assert _launch_block(code) == (64, 2, 1)


@pytest.mark.parametrize(
    "b_factory",
    [
        pytest.param(_b, id="row_major_b"),
        pytest.param(
            lambda: torch.empty_strided((Z, K, N), (K * N, 1, K), dtype=torch.bfloat16),
            id="k_major_b",
        ),
    ],
)
def test_plain_batched_store_with_explicit_threads_never_launches_without_roles(
    b_factory,
) -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
    def plain(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
        z, m, k = A.size()
        n = B.size(2)
        for tz, ty, tx in hl.tile([z, m, n]):
            acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
            for tr in hl.tile(k):
                acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
            out[tz, ty, tx] = acc.to(torch.bfloat16)

    config = dict(SAVED_HEAD_SPLIT_CONFIG, block_sizes=[1, 128, 64, 32])
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(
            plain, (_a(), b_factory(), torch.empty((Z, M, N), dtype=torch.bfloat16))
        )
        code = bound.to_code(helion.Config.from_dict(config))
    # The role-warp pipeline needs a 32-lane M axis; the explicit 2-lane
    # request keeps the collective lowering instead.
    assert "tcgen05_tma_warp" not in code
    assert "collective_tmem" in code
    assert _launch_block(code) == (64, 2, 1)


# ---------------------------------------------------------------------------
# Seeds for the native A/B-TMA + plane-store path.
# ---------------------------------------------------------------------------


def test_actual_module_gets_native_plane_store_seeds() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _inductor_head_split_bound()
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert tcgen05.batch_digit_plane_store_seeds_available
        assert not tcgen05.matmul_operands_tma_provable
        seeds = list(spec.compiler_seed_configs)
        assert seeds
        assert all(seed.config.get("tcgen05_cluster_m") == 1 for seed in seeds)
        assert {seed.config.get("pid_type") for seed in seeds} <= {
            "persistent_interleaved",
            "flat",
        }
        with bound.env, bound.host_function:
            generation = ConfigGeneration(spec).initial_population_view()
            pairs = generation.seed_flat_config_pairs()
            assert len(pairs) == len(seeds)
            canonical = []
            for (_flat, config), seed in zip(pairs, seeds, strict=True):
                assert all(
                    config.config.get(key) == value
                    for key, value in seed.config.items()
                )
                fixed = generation.unflatten(generation.flatten(config))
                assert dict(fixed.config) == dict(config.config)
                strict = helion.Config.from_dict(dict(config.config))
                spec.normalize(strict)
                assert dict(strict.config) == dict(config.config)
                canonical.append(config)
        for seed, config in zip(seeds, canonical, strict=True):
            code = bound.to_code(seed)
            assert bound.to_code(config) == code
            assert "'kind': 'tcgen05_ab_tma'" in code and "tcgen05_tma_warp" in code
            assert "'kind': 'tcgen05_d_tma'" not in code
            assert "PipelineTmaStore.create(" not in code
            assert "cute.make_tensor(out0.iterator + cute.crd2idx((" in code
            assert _launch_block(code)[0] == 32


def test_plane_store_seed_family_is_gated_loud_and_consumes_no_rng() -> None:
    import random

    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    def explode(self: CuteTcgen05Config) -> list[helion.Config]:
        raise RuntimeError("plane-store seed family failure")

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _inductor_head_split_bound()
        tcgen05 = bound.config_spec._cute_tcgen05_config
        state = random.getstate()
        seeds = tcgen05._optional_batch_digit_plane_store_seed_configs()
        assert random.getstate() == state
        assert seeds
        with patch.object(
            CuteTcgen05Config, "_batch_digit_plane_store_seed_configs", explode
        ):
            # A failure in the family is never swallowed into a smaller list.
            with pytest.raises(RuntimeError, match="plane-store seed family"):
                tcgen05._optional_batch_digit_plane_store_seed_configs()
            # Without the fact the family is not consulted at all.
            tcgen05.batch_digit_plane_store_seeds_available = False
            assert tcgen05._optional_batch_digit_plane_store_seed_configs() == []


def test_plane_store_seeds_stay_off_for_ordinary_batched_stores() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
    def plain(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
        z, m, k = A.size()
        n = B.size(2)
        for tz, ty, tx in hl.tile([z, m, n]):
            acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
            for tr in hl.tile(k):
                acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
            out[tz, ty, tx] = acc.to(torch.bfloat16)

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(plain, (_a(), _b(), torch.empty((Z, M, N), dtype=torch.bfloat16)))
        tcgen05 = bound.config_spec._cute_tcgen05_config
        assert tcgen05.matmul_operands_tma_provable
        assert not tcgen05.batch_digit_plane_store_seeds_available
        assert tcgen05._optional_batch_digit_plane_store_seed_configs() == []


# ---------------------------------------------------------------------------
# Whole-program effect inventory: the first version owns one output.
# ---------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def with_side_store(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    side: torch.Tensor,
) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, yi, zi % heads, xi] = acc.to(torch.bfloat16)
        side[tz, ty, tx] = C[tz, ty, tx]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def with_inplace_update(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, out: torch.Tensor
) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, yi, zi % heads, xi] = acc.to(torch.bfloat16)
        C[tz, ty, tx] = C[tz, ty, tx] + 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def with_atomic(
    A: torch.Tensor, B: torch.Tensor, counter: torch.Tensor, out: torch.Tensor
) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, yi, zi % heads, xi] = acc.to(torch.bfloat16)
        hl.atomic_add(counter, [tz], 1)


def _side_store_args(alias_out: bool) -> tuple[torch.Tensor, ...]:
    out_shape = (Z // H, M, H, N)
    if alias_out:
        storage = torch.empty(_OUT_NUMEL, dtype=torch.bfloat16)
        out = storage.view(out_shape)
        side = storage.view(Z, M, N)
    else:
        out = torch.empty(out_shape, dtype=torch.bfloat16)
        side = torch.empty((Z, M, N), dtype=torch.bfloat16)
    return _a(), _b(), torch.empty((Z, M, N), dtype=torch.bfloat16), out, side


@pytest.mark.parametrize("alias_out", [False, True], ids=["separate", "aliases_out"])
def test_second_output_is_refused_whatever_its_storage(alias_out) -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(_bind(with_side_store, _side_store_args(alias_out)))


def test_inplace_update_of_another_input_is_refused() -> None:
    args = (
        _a(),
        _b(),
        torch.empty((Z, M, N), dtype=torch.bfloat16),
        torch.empty((Z // H, M, H, N), dtype=torch.bfloat16),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(_bind(with_inplace_update, args))


def test_atomic_side_effect_is_refused() -> None:
    args = (
        _a(),
        _b(),
        torch.zeros((Z,), dtype=torch.int32),
        torch.empty((Z // H, M, H, N), dtype=torch.bfloat16),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        _assert_refused(_bind(with_atomic, args))


def test_effect_inventory_of_the_admitted_program() -> None:
    from helion._compiler.cute.mapped_output import _program_effects_owned
    from helion.language import _tracing_ops
    from helion.language import memory_ops

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(head_split, _alias_args("different_storage"))
        with bound.env, bound.host_function:
            _node, candidate = _mma_candidate(bound)
            assert candidate is not None
            mapping = cute_mma._candidate_batch_digit_output(candidate)
            assert mapping is not None
            graphs = list(bound.host_function.device_ir.graphs)
            nodes = [
                n for g in graphs for n in g.graph.nodes if n.op == "call_function"
            ]
            stores = [n for n in nodes if n.target is memory_ops.store]
            loads = [n for n in nodes if n.target is memory_ops.load]
            hosts = [n for n in nodes if n.target is _tracing_ops._host_tensor]
            assert stores == [mapping.store_node]
            assert len(loads) == 2
            assert {n.args[0].meta["val"] for n in loads} == {
                candidate.operands.lhs.source_fake,
                candidate.operands.rhs.source_fake,
            }
            assert len(hosts) == 3
            destination = mapping.store_node.args[0]
            assert isinstance(destination, torch.fx.Node)
            out = destination.meta["val"]
            inputs = (
                candidate.operands.lhs.source_fake,
                candidate.operands.rhs.source_fake,
            )
            assert _program_effects_owned(
                bound.env, graphs, mapping.store_node, out, inputs
            )
            # Without B among the owned operands, B's load is an unmatched read.
            assert not _program_effects_owned(
                bound.env, graphs, mapping.store_node, out, inputs[:1]
            )


# ---------------------------------------------------------------------------
# Explicit root thread counts: planning, detection and emitter agree.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_threads", "native"),
    [
        pytest.param([0, 0, 0, 0], True, id="automatic"),
        pytest.param([0, 32, 0, 0], True, id="m32_matches_role_width"),
        pytest.param([0, 64, 0, 0], False, id="m64_mismatch"),
        pytest.param([0, 2, 0, 0], False, id="m2_mismatch"),
        pytest.param([0, 0, 4, 0], False, id="n4_mismatch"),
    ],
)
def test_explicit_root_thread_counts_agree_across_sites(num_threads, native) -> None:
    config = helion.Config(
        block_sizes=[1, 128, 64, 64],
        num_threads=num_threads,
        pid_type="flat",
    )
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(head_split, _alias_args("different_storage"))
        assert bound.config_spec.cute_tcgen05_search_enabled
        code = bound.to_code(config)
    if native:
        assert "tcgen05_tma_warp" in code
        assert "'kind': 'tcgen05_ab_tma'" in code
        assert _launch_block(code)[0] == 32
        assert "cute.make_tensor(out.iterator + cute.crd2idx((" in code
    else:
        assert "tcgen05_tma_warp" not in code
        assert "'kind': 'tcgen05_ab_tma'" not in code


# ---------------------------------------------------------------------------
# Seed / validator agreement: seeds, canonical pairs and the population lower.
# ---------------------------------------------------------------------------

# A pre-existing codegen guard (``cute_mma.py``) that the search does not
# repair; the same draws fail identically on rank-2 K-major B GEMMs.
_K_MAJOR_NARROW_B_GUARD = (
    "tcgen05 K-major B TMA with block_n=8 needs block_k * itemsize >= 64"
)


def _k_major_b(n: int = N) -> torch.Tensor:
    return torch.empty((Z, n, K), dtype=torch.bfloat16).transpose(1, 2)


def _batch_digit_bound(case: str):
    if case == "actual_module":
        return _inductor_head_split_bound()
    n = 256 if case == "k_major_b_wide_n" else N
    b = _b() if case == "row_major_b" else _k_major_b(n)
    out = torch.empty((Z // H, M, H, n), dtype=torch.bfloat16)
    return _bind(head_split, (_a(), b, out))


def _render(bound, config: helion.Config) -> str | Exception:
    try:
        return bound.to_code(config)
    except exc.BackendUnsupported as error:
        return error


def _population(bound, draws: int) -> tuple[list[helion.Config], list[helion.Config]]:
    spec = bound.config_spec
    with bound.env, bound.host_function:
        generation = ConfigGeneration(spec).initial_population_view()
        pairs = [config for _flat, config in generation.seed_flat_config_pairs()]
        state = random.getstate()
        random.seed(20261104)
        try:
            population = [generation.unflatten(generation.default_flat())] + [
                generation.unflatten(generation.random_flat()) for _ in range(draws)
            ]
        finally:
            random.setstate(state)
    return pairs, population


def _assert_native_plane(code: str) -> None:
    assert "'kind': 'tcgen05_ab_tma'" in code and "tcgen05_tma_warp" in code
    assert "'kind': 'tcgen05_d_tma'" not in code
    assert "PipelineTmaStore.create(" not in code
    assert ".iterator + cute.crd2idx((" in code
    assert _launch_block(code)[0] == 32


@pytest.mark.parametrize(
    "case", ["row_major_b", "k_major_b", "k_major_b_wide_n", "actual_module"]
)
def test_seeds_pairs_and_population_of_a_batch_digit_kernel_lower(case) -> None:
    from helion._compiler.cute.cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _batch_digit_bound(case)
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert spec.cute_tcgen05_search_enabled
        # The register-MMA family's own store contract rejects the digit map.
        assert not tcgen05.warp_mma_admitted
        assert WARP_MMA_FAMILY_KEY not in tcgen05.optional_fragments(for_search=True)
        seeds = list(spec.compiler_seed_configs)
        pairs, population = _population(bound, draws=8)
        assert seeds and len(pairs) == len(seeds)
        for config in [*seeds, *pairs]:
            assert config.config.get(WARP_MMA_FAMILY_KEY) is None
            code = _render(bound, config)
            assert isinstance(code, str), code
            _assert_native_plane(code)
        for config in population:
            code = _render(bound, config)
            if isinstance(code, Exception):
                assert case != "row_major_b", code
                assert _K_MAJOR_NARROW_B_GUARD in str(code), code
            elif "tcgen05_tma_warp" in code:
                # A draw the native tile rule declines lowers generically.
                _assert_native_plane(code)


def test_explicit_register_mma_request_on_a_batch_digit_kernel_is_refused() -> None:
    from helion._compiler.cute.cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _batch_digit_bound("row_major_b")
        _pairs, population = _population(bound, draws=0)
        requested = helion.Config.from_dict(
            {**dict(population[0].config), WARP_MMA_FAMILY_KEY: "warp_mma"}
        )
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalize(requested)


def test_partial_tile_batch_digit_kernel_offers_no_native_configuration() -> None:
    rows = 200
    args = (
        torch.empty((Z, rows, K), dtype=torch.bfloat16),
        _b(),
        torch.empty((Z // H, rows, H, N), dtype=torch.bfloat16),
    )
    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(head_split, args)
        assert list(bound.config_spec.compiler_seed_configs) == []
        _pairs, population = _population(bound, draws=8)
        for config in population:
            code = _render(bound, config)
            assert isinstance(code, str), code
            assert "tcgen05_tma_warp" not in code


def test_overlapping_binding_default_lowers_generically() -> None:
    with cpu_codegen_target(), forbid_cuda_work():
        native = _bind(head_split, _alias_args("different_storage"))
        overlapping = _bind(head_split, _alias_args("disjoint_views_same_storage"))
        _pairs, native_population = _population(native, draws=0)
        _pairs, fallback_population = _population(overlapping, draws=0)
        native_default = helion.Config.from_dict(dict(native_population[0].config))
        assert "tcgen05_tma_warp" in native.to_code(native_default)
        with pytest.raises(exc.InvalidConfig):
            overlapping.to_code(native_default)
        code = overlapping.to_code(fallback_population[0])
        assert "tcgen05_tma_warp" not in code


def test_register_mma_admission_is_unchanged_for_an_exact_tile_batched_store() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
    def exact(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
        z, m, k = A.size()
        n = B.size(2)
        for tz, ty, tx in hl.tile([z, m, n]):
            acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
            for tr in hl.tile(k):
                acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
            out[tz, ty, tx] = acc.to(torch.bfloat16)

    with cpu_codegen_target(), forbid_cuda_work():
        bound = _bind(exact, (_a(), _b(), torch.empty((Z, M, N), dtype=torch.bfloat16)))
        assert bound.config_spec._cute_tcgen05_config.warp_mma_admitted
